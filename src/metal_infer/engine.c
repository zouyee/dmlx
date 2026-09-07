// Metal inference engine — per-token forward pass with flash-moe pipeline.
// Phase 1: MoE forward only (attention handled by MLX for now).
// Phase 2: Add attention MatVecs on Metal.
// Phase 3: Full flash-moe alignment.
//
// See docs/analysis/flash-moe-alignment-plan.md for architecture details.
#include "engine.h"
#include "dspark_engine.h"
#include "mla_attention.h"
#include "mhc.h"
#include "routing_trace.h"
#include <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dispatch/dispatch.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <inttypes.h>
#include <errno.h>
#include <stdatomic.h>
#include <sched.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>

#ifndef DMLX_DECODE_TIMELINE_PROBE
#define DMLX_DECODE_TIMELINE_PROBE 0
#endif
#if DMLX_DECODE_TIMELINE_PROBE != 0 && DMLX_DECODE_TIMELINE_PROBE != 1
#error "DMLX_DECODE_TIMELINE_PROBE must be 0 or 1"
#endif

#ifndef DMLX_GPU_TIMESTAMP_PROBE
#define DMLX_GPU_TIMESTAMP_PROBE 1
#endif
#if DMLX_GPU_TIMESTAMP_PROBE != 0 && DMLX_GPU_TIMESTAMP_PROBE != 1
#error "DMLX_GPU_TIMESTAMP_PROBE must be 0 or 1"
#endif

_Static_assert(sizeof(NativeLayerMetrics) == 168,
    "NativeLayerMetrics C ABI drift");
_Static_assert(offsetof(NativeLayerMetrics, cache_admissions) == 104,
    "NativeLayerMetrics cache ABI drift");
_Static_assert(offsetof(NativeLayerMetrics, moe_dispatches) == 120,
    "NativeLayerMetrics dispatch ABI drift");
_Static_assert(sizeof(NativeRequestMetrics) == 21696,
    "NativeRequestMetrics C ABI drift");

uint64_t moe_infer_monotonic_ns(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) return 0;
    return (uint64_t)timestamp.tv_sec * UINT64_C(1000000000) +
        (uint64_t)timestamp.tv_nsec;
}

#if DMLX_DECODE_TIMELINE_PROBE
static mach_timebase_info_data_t decode_timeline_timebase;
static pthread_once_t decode_timeline_timebase_once = PTHREAD_ONCE_INIT;
static void decode_timeline_init_timebase(void) {
    (void)mach_timebase_info(&decode_timeline_timebase);
}

uint64_t moe_infer_decode_timeline_now_ns(void) {
    pthread_once(&decode_timeline_timebase_once, decode_timeline_init_timebase);
    const uint64_t ticks = mach_continuous_time();
    if (decode_timeline_timebase.denom == 0u) return 0u;
    return (uint64_t)(((__uint128_t)ticks * decode_timeline_timebase.numer) /
        decode_timeline_timebase.denom);
}

typedef struct DmlxDecodeTimelineState {
    uint32_t capacity;
    _Atomic uint32_t count;
    _Atomic uint32_t dropped;
    _Atomic uint32_t active_publishers;
    _Atomic uint32_t incomplete;
    uint32_t token;
    uint32_t request_active;
    uint32_t token_active;
    uint32_t finalized;
    DmlxDecodeTimelineEvent *events;
} DmlxDecodeTimelineState;

static void decode_timeline_gate_lock(MoEInferEngine *eng) {
    uint32_t expected = 0u;
    while (!atomic_compare_exchange_weak_explicit(
            &eng->decode_timeline_gate, &expected, 1u,
            memory_order_acquire, memory_order_relaxed)) {
        expected = 0u;
        sched_yield();
    }
}

static void decode_timeline_gate_unlock(MoEInferEngine *eng) {
    atomic_store_explicit(&eng->decode_timeline_gate, 0u, memory_order_release);
}

static DmlxDecodeTimelineState *decode_timeline_hot_load(
    const MoEInferEngine *eng) {
    return eng ? atomic_load_explicit(&eng->decode_timeline, memory_order_acquire)
               : NULL;
}

static bool decode_timeline_active(const MoEInferEngine *eng) {
    DmlxDecodeTimelineState *state = decode_timeline_hot_load(eng);
    if (!state) return false;
    MoEInferEngine *mutable_eng = (MoEInferEngine *)eng;
    decode_timeline_gate_lock(mutable_eng);
    state = atomic_load_explicit(&eng->decode_timeline, memory_order_relaxed);
    const bool active = state && state->request_active && state->token_active &&
        !state->finalized;
    decode_timeline_gate_unlock(mutable_eng);
    return active;
}

int moe_infer_decode_timeline_configure(MoEInferEngine *eng, uint32_t capacity) {
    if (!eng || !eng->initialized || capacity == 0u ||
        capacity > DMLX_DECODE_TIMELINE_MAX_CAPACITY)
        return DMLX_DECODE_TIMELINE_INVALID_ARGUMENT;
    size_t bytes = sizeof(DmlxDecodeTimelineState) +
        (size_t)capacity * sizeof(DmlxDecodeTimelineEvent);
    DmlxDecodeTimelineState *state = calloc(1u, bytes);
    if (!state) return DMLX_DECODE_TIMELINE_CAPACITY_EXCEEDED;
    state->capacity = capacity;
    state->events = (DmlxDecodeTimelineEvent *)(state + 1);

    decode_timeline_gate_lock(eng);
    DmlxDecodeTimelineState *old = eng->decode_timeline_storage;
    int status = DMLX_DECODE_TIMELINE_OK;
    if (atomic_load_explicit(&eng->decode_timeline, memory_order_relaxed)) {
        status = DMLX_DECODE_TIMELINE_INVALID_STATE;
    } else if (old && atomic_load_explicit(
            &old->active_publishers, memory_order_relaxed)) {
        status = DMLX_DECODE_TIMELINE_BUSY;
    } else {
        eng->decode_timeline_storage = state;
    }
    decode_timeline_gate_unlock(eng);
    if (status != DMLX_DECODE_TIMELINE_OK) {
        free(state);
        return status;
    }
    free(old);
    return DMLX_DECODE_TIMELINE_OK;
}

int moe_infer_decode_timeline_enable(MoEInferEngine *eng, bool enabled) {
    if (!eng) return DMLX_DECODE_TIMELINE_INVALID_ARGUMENT;
    decode_timeline_gate_lock(eng);
    DmlxDecodeTimelineState *state = eng->decode_timeline_storage;
    if (!state) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_DISABLED;
    }
    if (atomic_load_explicit(&state->active_publishers, memory_order_relaxed)) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_BUSY;
    }
    atomic_store_explicit(
        &eng->decode_timeline, enabled ? state : NULL, memory_order_release);
    decode_timeline_gate_unlock(eng);
    return DMLX_DECODE_TIMELINE_OK;
}

bool moe_infer_decode_timeline_recording_enabled(const MoEInferEngine *eng) {
    return decode_timeline_hot_load(eng) != NULL;
}

int moe_infer_decode_timeline_begin_request(MoEInferEngine *eng) {
    if (!decode_timeline_hot_load(eng)) return DMLX_DECODE_TIMELINE_DISABLED;
    decode_timeline_gate_lock(eng);
    DmlxDecodeTimelineState *state = atomic_load_explicit(
        &eng->decode_timeline, memory_order_relaxed);
    if (!state) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_DISABLED;
    }
    if (state->request_active || atomic_load_explicit(
            &state->active_publishers, memory_order_relaxed)) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_INVALID_STATE;
    }
    memset(state->events, 0, (size_t)state->capacity * sizeof(*state->events));
    atomic_store(&state->count, 0u); atomic_store(&state->dropped, 0u);
    atomic_store(&state->incomplete, 0u);
    state->token = 0u; state->token_active = 0u;
    state->finalized = 0u; state->request_active = 1u;
    decode_timeline_gate_unlock(eng);
    return DMLX_DECODE_TIMELINE_OK;
}

int moe_infer_decode_timeline_begin_token(MoEInferEngine *eng, uint32_t token) {
    if (!decode_timeline_hot_load(eng)) return DMLX_DECODE_TIMELINE_DISABLED;
    decode_timeline_gate_lock(eng);
    DmlxDecodeTimelineState *state = atomic_load_explicit(
        &eng->decode_timeline, memory_order_relaxed);
    if (!state) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_DISABLED;
    }
    if (!state->request_active || state->finalized || atomic_load_explicit(
            &state->active_publishers, memory_order_relaxed)) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_INVALID_STATE;
    }
    state->token = token; state->token_active = 1u;
    decode_timeline_gate_unlock(eng);
    return DMLX_DECODE_TIMELINE_OK;
}

int moe_infer_decode_timeline_append(
    MoEInferEngine *eng, const DmlxDecodeTimelineEvent *event) {
    DmlxDecodeTimelineState *state = decode_timeline_hot_load(eng);
    if (!state) return DMLX_DECODE_TIMELINE_DISABLED;
    const bool stage_valid = event &&
        event->stage < DMLX_DECODE_TIMELINE_STAGE_COUNT;
    const bool layerless = stage_valid &&
        (event->stage == DMLX_DECODE_TIMELINE_TOKEN_FORWARD ||
         event->stage == DMLX_DECODE_TIMELINE_LOGITS_CB ||
         event->stage == DMLX_DECODE_TIMELINE_TOKEN_SAMPLE);
    const bool layer_valid = stage_valid &&
        (layerless ? event->layer == DMLX_DECODE_TIMELINE_NO_LAYER
                   : event->layer < N_LAYERS);
    const bool times_valid = event &&
        event->begin_ns <= event->encode_done_ns &&
        event->encode_done_ns <= event->pre_commit_ns &&
        event->pre_commit_ns <= event->post_commit_ns &&
        event->post_commit_ns <= event->post_wait_ns &&
        event->post_wait_ns <= event->end_ns;

    decode_timeline_gate_lock(eng);
    state = atomic_load_explicit(&eng->decode_timeline, memory_order_relaxed);
    if (!state) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_DISABLED;
    }
    if (!event || !state->request_active || !state->token_active ||
        state->finalized || !stage_valid || !layer_valid ||
        event->status != DMLX_DECODE_TIMELINE_EVENT_COMPLETE || !times_valid) {
        atomic_store_explicit(&state->incomplete, 1u, memory_order_release);
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_INVALID_ARGUMENT;
    }
    atomic_fetch_add_explicit(&state->active_publishers, 1u, memory_order_relaxed);
    const uint32_t token = state->token;
    DmlxDecodeTimelinePublisherHookFn hook =
        eng->decode_timeline_publisher_hook;
    void *hook_context = eng->decode_timeline_publisher_hook_context;
    decode_timeline_gate_unlock(eng);

    if (hook) hook(hook_context);
    const uint32_t index = atomic_fetch_add_explicit(&state->count, 1u, memory_order_acq_rel);
    if (index >= state->capacity) {
        atomic_fetch_add_explicit(&state->dropped, 1u, memory_order_relaxed);
        atomic_store_explicit(&state->incomplete, 1u, memory_order_release);
        atomic_fetch_sub_explicit(&state->active_publishers, 1u, memory_order_acq_rel);
        return DMLX_DECODE_TIMELINE_CAPACITY_EXCEEDED;
    }
    DmlxDecodeTimelineEvent record = *event;
    record.token = token;
    record.status = DMLX_DECODE_TIMELINE_EVENT_RESERVED;
    state->events[index] = record;
    atomic_thread_fence(memory_order_release);
    state->events[index].status = DMLX_DECODE_TIMELINE_EVENT_COMPLETE;
    atomic_fetch_sub_explicit(&state->active_publishers, 1u, memory_order_acq_rel);
    return DMLX_DECODE_TIMELINE_OK;
}

static bool decode_timeline_validate_complete_token(
    uint32_t token, uint32_t expected_token, const uint16_t stage_counts[10],
    const uint8_t layer_seen[N_LAYERS][7]) {
    /* Per-layer stages are the contiguous enum range ATTENTION_CB..CB3_CB
       (ids 1..7); layer_seen is indexed by stage - 1, so 7 slots suffice. */
    if (token != expected_token || stage_counts[DMLX_DECODE_TIMELINE_TOKEN_FORWARD] != 1u ||
        stage_counts[DMLX_DECODE_TIMELINE_LOGITS_CB] != 1u ||
        stage_counts[DMLX_DECODE_TIMELINE_TOKEN_SAMPLE] != 1u)
        return false;
    for (uint32_t stage = DMLX_DECODE_TIMELINE_ATTENTION_CB;
         stage <= DMLX_DECODE_TIMELINE_CB3_CB; ++stage) {
        if (stage_counts[stage] != N_LAYERS) return false;
    }
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer)
        for (uint32_t stage = DMLX_DECODE_TIMELINE_ATTENTION_CB;
             stage <= DMLX_DECODE_TIMELINE_CB3_CB; ++stage)
            if (!layer_seen[layer][stage - 1u]) return false;
    return true;
}

static bool decode_timeline_validate_schema(const DmlxDecodeTimelineState *state) {
    const uint32_t count = atomic_load_explicit(&state->count, memory_order_acquire);
    if (count == 0u || count > state->capacity) return false;
    uint16_t stage_counts[10] = {0};
    uint8_t layer_seen[N_LAYERS][7] = {{0}};
    uint32_t expected_token = 0u;
    uint32_t current_token = state->events[0].token;
    for (uint32_t i = 0u; i < count; ++i) {
        const DmlxDecodeTimelineEvent *event = &state->events[i];
        if (event->status != DMLX_DECODE_TIMELINE_EVENT_COMPLETE) return false;
        if (event->token != current_token) {
            if (!decode_timeline_validate_complete_token(
                    current_token, expected_token, stage_counts, layer_seen))
                return false;
            expected_token++;
            current_token = event->token;
            memset(stage_counts, 0, sizeof(stage_counts));
            memset(layer_seen, 0, sizeof(layer_seen));
        }
        if (event->stage >= DMLX_DECODE_TIMELINE_STAGE_COUNT ||
            stage_counts[event->stage] == UINT16_MAX)
            return false;
        stage_counts[event->stage]++;
        if (event->stage >= DMLX_DECODE_TIMELINE_ATTENTION_CB &&
            event->stage <= DMLX_DECODE_TIMELINE_CB3_CB) {
            if (event->layer >= N_LAYERS ||
                layer_seen[event->layer][event->stage - 1u]) return false;
            layer_seen[event->layer][event->stage - 1u] = 1u;
        }
    }
    return decode_timeline_validate_complete_token(
        current_token, expected_token, stage_counts, layer_seen);
}

int moe_infer_decode_timeline_finalize(MoEInferEngine *eng) {
    if (!decode_timeline_hot_load(eng)) return DMLX_DECODE_TIMELINE_DISABLED;
    decode_timeline_gate_lock(eng);
    DmlxDecodeTimelineState *state = atomic_load_explicit(
        &eng->decode_timeline, memory_order_relaxed);
    if (!state) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_DISABLED;
    }
    if (atomic_load_explicit(&state->active_publishers, memory_order_relaxed)) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_BUSY;
    }
    if (!state->request_active || state->finalized) {
        atomic_store_explicit(&state->incomplete, 1u, memory_order_release);
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_BUSY;
    }
    if (atomic_load_explicit(&state->incomplete, memory_order_acquire) ||
        atomic_load_explicit(&state->dropped, memory_order_acquire) ||
        !decode_timeline_validate_schema(state)) {
        atomic_store_explicit(&state->incomplete, 1u, memory_order_release);
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_INVALID_STATE;
    }
    state->request_active = 0u; state->token_active = 0u; state->finalized = 1u;
    decode_timeline_gate_unlock(eng);
    return DMLX_DECODE_TIMELINE_OK;
}

int moe_infer_decode_timeline_snapshot(
    const MoEInferEngine *eng, DmlxDecodeTimelineSnapshot *snapshot,
    DmlxDecodeTimelineEvent *events, uint32_t capacity) {
    if (!snapshot) return DMLX_DECODE_TIMELINE_INVALID_ARGUMENT;
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->abi_version = DMLX_DECODE_TIMELINE_ABI_VERSION;
    if (!decode_timeline_hot_load(eng)) return DMLX_DECODE_TIMELINE_DISABLED;
    MoEInferEngine *mutable_eng = (MoEInferEngine *)eng;
    decode_timeline_gate_lock(mutable_eng);
    DmlxDecodeTimelineState *state = atomic_load_explicit(
        &eng->decode_timeline, memory_order_relaxed);
    if (!state) {
        decode_timeline_gate_unlock(mutable_eng);
        return DMLX_DECODE_TIMELINE_DISABLED;
    }
    const uint32_t count = atomic_load_explicit(&state->count, memory_order_acquire);
    snapshot->enabled = 1u; snapshot->finalized = state->finalized;
    snapshot->capacity = state->capacity; snapshot->count = count;
    snapshot->dropped = atomic_load_explicit(&state->dropped, memory_order_acquire);
    snapshot->active_publishers = atomic_load_explicit(&state->active_publishers, memory_order_acquire);
    snapshot->incomplete = atomic_load_explicit(&state->incomplete, memory_order_acquire);
    if (count > state->capacity || capacity < count || (count && !events)) {
        decode_timeline_gate_unlock(mutable_eng);
        return DMLX_DECODE_TIMELINE_CAPACITY_EXCEEDED;
    }
    if (!snapshot->active_publishers) {
        for (uint32_t i = 0u; i < count; ++i) {
            if (state->events[i].status != DMLX_DECODE_TIMELINE_EVENT_COMPLETE) {
                snapshot->incomplete = 1u; break;
            }
        }
        if (count) memcpy(events, state->events, (size_t)count * sizeof(*events));
    }
    snapshot->events = events;
    snapshot->valid = snapshot->finalized && !snapshot->dropped &&
        !snapshot->incomplete && !snapshot->active_publishers;
    decode_timeline_gate_unlock(mutable_eng);
    return snapshot->valid ? DMLX_DECODE_TIMELINE_OK : DMLX_DECODE_TIMELINE_INVALID_STATE;
}

int moe_infer_decode_timeline_reset(MoEInferEngine *eng) {
    if (!eng) return DMLX_DECODE_TIMELINE_INVALID_ARGUMENT;
    decode_timeline_gate_lock(eng);
    DmlxDecodeTimelineState *state = eng->decode_timeline_storage;
    if (!state) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_DISABLED;
    }
    if (atomic_load_explicit(&state->active_publishers, memory_order_relaxed)) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_BUSY;
    }
    state->request_active = state->token_active = state->finalized = 0u;
    atomic_store(&state->count, 0u); atomic_store(&state->dropped, 0u);
    atomic_store(&state->incomplete, 0u);
    decode_timeline_gate_unlock(eng);
    return DMLX_DECODE_TIMELINE_OK;
}

int moe_infer_decode_timeline_deconfigure(MoEInferEngine *eng) {
    if (!eng) return DMLX_DECODE_TIMELINE_INVALID_ARGUMENT;
    decode_timeline_gate_lock(eng);
    DmlxDecodeTimelineState *state = eng->decode_timeline_storage;
    if (!state) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_OK;
    }
    if (state->request_active || state->token_active ||
        atomic_load_explicit(&state->active_publishers, memory_order_relaxed)) {
        decode_timeline_gate_unlock(eng);
        return DMLX_DECODE_TIMELINE_BUSY;
    }
    atomic_store_explicit(&eng->decode_timeline, NULL, memory_order_release);
    eng->decode_timeline_storage = NULL;
    decode_timeline_gate_unlock(eng);
    free(state);
    return DMLX_DECODE_TIMELINE_OK;
}

static void decode_timeline_record(
    MoEInferEngine *eng, uint16_t layer, uint8_t stage, uint64_t begin_ns,
    uint64_t encode_done_ns, uint64_t pre_commit_ns, uint64_t post_commit_ns,
    uint64_t post_wait_ns, uint64_t end_ns) {
    if (!decode_timeline_active(eng)) return;
    const DmlxDecodeTimelineEvent event = {
        .layer = layer, .stage = stage,
        .status = DMLX_DECODE_TIMELINE_EVENT_COMPLETE,
        .begin_ns = begin_ns, .encode_done_ns = encode_done_ns,
        .pre_commit_ns = pre_commit_ns, .post_commit_ns = post_commit_ns,
        .post_wait_ns = post_wait_ns, .end_ns = end_ns,
    };
    (void)moe_infer_decode_timeline_append(eng, &event);
}
#else
static bool decode_timeline_active(const MoEInferEngine *e) { (void)e; return false; }
static void decode_timeline_record(MoEInferEngine *e, uint16_t l, uint8_t s,
    uint64_t b, uint64_t d, uint64_t p, uint64_t c, uint64_t w, uint64_t n) {
    (void)e; (void)l; (void)s; (void)b; (void)d; (void)p; (void)c; (void)w; (void)n;
}
uint64_t moe_infer_decode_timeline_now_ns(void) { return 0u; }
int moe_infer_decode_timeline_configure(MoEInferEngine *e, uint32_t c) { (void)e; (void)c; return DMLX_DECODE_TIMELINE_DISABLED; }
int moe_infer_decode_timeline_enable(MoEInferEngine *e, bool v) { (void)e; (void)v; return DMLX_DECODE_TIMELINE_DISABLED; }
bool moe_infer_decode_timeline_recording_enabled(const MoEInferEngine *e) { (void)e; return false; }
int moe_infer_decode_timeline_begin_request(MoEInferEngine *e) { (void)e; return DMLX_DECODE_TIMELINE_DISABLED; }
int moe_infer_decode_timeline_begin_token(MoEInferEngine *e, uint32_t t) { (void)e; (void)t; return DMLX_DECODE_TIMELINE_DISABLED; }
int moe_infer_decode_timeline_append(MoEInferEngine *e, const DmlxDecodeTimelineEvent *v) { (void)e; (void)v; return DMLX_DECODE_TIMELINE_DISABLED; }
int moe_infer_decode_timeline_finalize(MoEInferEngine *e) { (void)e; return DMLX_DECODE_TIMELINE_DISABLED; }
int moe_infer_decode_timeline_snapshot(const MoEInferEngine *e, DmlxDecodeTimelineSnapshot *s, DmlxDecodeTimelineEvent *v, uint32_t c) { (void)e; (void)v; (void)c; if (s) memset(s, 0, sizeof(*s)); return DMLX_DECODE_TIMELINE_DISABLED; }
int moe_infer_decode_timeline_reset(MoEInferEngine *e) { (void)e; return DMLX_DECODE_TIMELINE_DISABLED; }
int moe_infer_decode_timeline_deconfigure(MoEInferEngine *e) { (void)e; return DMLX_DECODE_TIMELINE_OK; }
#endif

typedef struct DmlxExactQ4ProducerState {
    uint32_t stage_capacity;
    uint32_t read_capacity;
    uint32_t path_count;
    uint32_t next_path_ordinal;
    uint32_t dropped_paths;
    uint32_t stage_count;
    uint32_t read_count;
    uint32_t next_stage_ordinal;
    uint32_t next_read_ordinal;
    uint32_t dropped_stages;
    _Atomic uint32_t dropped_reads;
    _Atomic uint32_t active_read_publishers;
    _Atomic uint32_t read_owner_operations;
    _Atomic uint32_t aborting;
    _Atomic uint32_t incomplete;
    uint32_t request_ordinal;
    uint32_t token_ordinal;
    uint32_t request_active;
    uint32_t token_active;
    uint32_t finalized;
    uint32_t aborted;
    size_t allocation_bytes;
    DmlxExactQ4StageEvent *stages;
    DmlxMlaPathEvent *paths;
    DmlxExactQ4ExpertReadEvent *reads;
    _Atomic uint32_t *read_states;
#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
    DmlxExactQ4TestClockFn test_clock;
    void *test_clock_context;
#endif
} DmlxExactQ4ProducerState;

#if DMLX_GPU_TIMESTAMP_PROBE
typedef struct DmlxGpuTimestampProbeState {
    uint32_t count;
    uint32_t reserved;
    uint64_t total_seen;
    uint64_t overflow_count;
    pthread_t owner_thread;
    DmlxGpuTimestampSample samples[DMLX_GPU_TIMESTAMP_PROBE_CAPACITY];
} DmlxGpuTimestampProbeState;

static bool gpu_timestamp_probe_owned_by_current_thread(
    const DmlxGpuTimestampProbeState *state) {
    return state && pthread_equal(state->owner_thread, pthread_self()) != 0;
}
#endif

typedef struct DmlxAttentionStageProbeState {
    DmlxAttentionStageStatus status;
    uint32_t next_stage;
    bool layer_active;
    id<MTLCounterSampleBuffer> sample_buffer;
    MTLComputePassDescriptor *descriptors[DMLX_ATTENTION_STAGE_COUNT];
    uint64_t duration_ns_sum[DMLX_ATTENTION_STAGE_COUNT];
    uint64_t valid_count[DMLX_ATTENTION_STAGE_COUNT];
} DmlxAttentionStageProbeState;

static const char *const attention_stage_names[DMLX_ATTENTION_STAGE_COUNT] = {
    "mhc_pre", "wq_a", "q_rms", "wq_b", "q_head_rms", "q_rope",
    "wkv", "kv_rms", "kv_rope", "kv_update", "sdpa", "inv_q_rope",
    "wo_a", "wo_b",
};

typedef enum DmlxExactQ4ReadState {
    DMLX_EXACT_Q4_READ_RESERVED = 1,
    DMLX_EXACT_Q4_READ_STARTING = 2,
    DMLX_EXACT_Q4_READ_STARTED = 3,
    DMLX_EXACT_Q4_READ_PUBLISHING = 4,
    DMLX_EXACT_Q4_READ_COMPLETE = 5,
    DMLX_EXACT_Q4_READ_ABORTED = 6,
} DmlxExactQ4ReadState;

static bool exact_q4_size_add(size_t left, size_t right, size_t *sum) {
    if (!sum || right > SIZE_MAX - left) return false;
    *sum = left + right;
    return true;
}

static bool exact_q4_size_multiply(size_t left, size_t right, size_t *product) {
    if (!product || (left != 0u && right > SIZE_MAX / left)) return false;
    *product = left * right;
    return true;
}

static bool exact_q4_align_size(size_t value, size_t alignment, size_t *aligned) {
    const size_t mask = alignment - 1u;
    if (!aligned || alignment == 0u || (alignment & mask) != 0u ||
        value > SIZE_MAX - mask) return false;
    *aligned = (value + mask) & ~mask;
    return true;
}

static uint32_t exact_q4_saturating_increment(_Atomic uint32_t *value) {
    uint32_t current = atomic_load_explicit(value, memory_order_relaxed);
    while (current != UINT32_MAX &&
           !atomic_compare_exchange_weak_explicit(
               value, &current, current + 1u,
               memory_order_relaxed, memory_order_relaxed)) {}
    return current == UINT32_MAX ? UINT32_MAX : current + 1u;
}

static void exact_q4_mark_incomplete(DmlxExactQ4ProducerState *state) {
    atomic_store_explicit(&state->incomplete, 1u, memory_order_release);
}

static bool exact_q4_enter_read_owner_operation(
    DmlxExactQ4ProducerState *state) {
    if (atomic_load_explicit(&state->aborting, memory_order_acquire) != 0u)
        return false;
    atomic_fetch_add_explicit(
        &state->read_owner_operations, 1u, memory_order_acq_rel);
    if (atomic_load_explicit(&state->aborting, memory_order_acquire) == 0u)
        return true;
    atomic_fetch_sub_explicit(
        &state->read_owner_operations, 1u, memory_order_acq_rel);
    return false;
}

static void exact_q4_leave_read_owner_operation(
    DmlxExactQ4ProducerState *state) {
    atomic_fetch_sub_explicit(
        &state->read_owner_operations, 1u, memory_order_acq_rel);
}

static void exact_q4_clear_records(DmlxExactQ4ProducerState *state) {
    if (state->stage_count > 0u) {
        memset(state->stages, 0,
            (size_t)state->stage_count * sizeof(*state->stages));
    }
    if (state->read_count > 0u) {
        memset(state->reads, 0,
            (size_t)state->read_count * sizeof(*state->reads));
        for (uint32_t index = 0u; index < state->read_count; ++index) {
            atomic_store_explicit(
                &state->read_states[index], 0u, memory_order_relaxed);
        }
    }
    if (state->path_count > 0u) {
        memset(state->paths, 0,
            (size_t)state->path_count * sizeof(*state->paths));
    }
    state->stage_count = 0u;
    state->read_count = 0u;
    state->path_count = 0u;
    state->next_stage_ordinal = 0u;
    state->next_read_ordinal = 0u;
    state->next_path_ordinal = 0u;
    state->dropped_stages = 0u;
    state->dropped_paths = 0u;
    atomic_store_explicit(&state->dropped_reads, 0u, memory_order_relaxed);
    atomic_store_explicit(&state->read_owner_operations, 0u, memory_order_relaxed);
    atomic_store_explicit(&state->aborting, 0u, memory_order_relaxed);
    atomic_store_explicit(&state->incomplete, 0u, memory_order_relaxed);
    state->request_ordinal = 0u;
    state->token_ordinal = 0u;
    state->request_active = 0u;
    state->token_active = 0u;
    state->finalized = 0u;
    state->aborted = 0u;
}

static int exact_q4_reset_state(DmlxExactQ4ProducerState *state) {
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (state->request_active || state->token_active ||
        atomic_load_explicit(
            &state->active_read_publishers, memory_order_acquire) != 0u) {
        return DMLX_EXACT_Q4_PRODUCER_BUSY;
    }
    exact_q4_clear_records(state);
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

int moe_infer_exact_q4_configure(
    MoEInferEngine *eng, uint32_t stage_capacity, uint32_t read_capacity) {
    if (!eng || !eng->initialized || stage_capacity == 0u || read_capacity == 0u)
        return DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT;
    if (stage_capacity > DMLX_EXACT_Q4_MAX_STAGE_CAPACITY ||
        read_capacity > DMLX_EXACT_Q4_MAX_READ_CAPACITY)
        return DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED;
    DmlxExactQ4ProducerState *old = eng->exact_q4_storage;
    if (old && atomic_load_explicit(
            &old->active_read_publishers, memory_order_acquire) != 0u)
        return DMLX_EXACT_Q4_PRODUCER_BUSY;
    if (eng->exact_q4) return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;

    size_t stage_bytes = 0u, path_bytes = 0u, read_bytes = 0u, published_bytes = 0u;
    size_t stage_offset = 0u, path_offset = 0u, read_offset = 0u, published_offset = 0u;
    size_t total = sizeof(DmlxExactQ4ProducerState);
    if (!exact_q4_size_multiply(stage_capacity,
            sizeof(DmlxExactQ4StageEvent), &stage_bytes) ||
        !exact_q4_size_multiply(stage_capacity,
            sizeof(DmlxMlaPathEvent), &path_bytes) ||
        !exact_q4_size_multiply(read_capacity,
            sizeof(DmlxExactQ4ExpertReadEvent), &read_bytes) ||
        !exact_q4_size_multiply(read_capacity,
            sizeof(_Atomic uint32_t), &published_bytes) ||
        !exact_q4_align_size(total, _Alignof(DmlxExactQ4StageEvent), &stage_offset) ||
        !exact_q4_size_add(stage_offset, stage_bytes, &total) ||
        !exact_q4_align_size(total, _Alignof(DmlxMlaPathEvent), &path_offset) ||
        !exact_q4_size_add(path_offset, path_bytes, &total) ||
        !exact_q4_align_size(total, _Alignof(DmlxExactQ4ExpertReadEvent), &read_offset) ||
        !exact_q4_size_add(read_offset, read_bytes, &total) ||
        !exact_q4_align_size(total, _Alignof(_Atomic uint32_t), &published_offset) ||
        !exact_q4_size_add(published_offset, published_bytes, &total)) {
        return DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED;
    }
    DmlxExactQ4ProducerState *state = calloc(1u, total);
    if (!state) return DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED;
    state->stage_capacity = stage_capacity;
    state->read_capacity = read_capacity;
    state->allocation_bytes = total;
    state->stages = (DmlxExactQ4StageEvent *)((uint8_t *)state + stage_offset);
    state->paths = (DmlxMlaPathEvent *)((uint8_t *)state + path_offset);
    state->reads = (DmlxExactQ4ExpertReadEvent *)((uint8_t *)state + read_offset);
    state->read_states = (_Atomic uint32_t *)((uint8_t *)state + published_offset);
    free(old);
    eng->exact_q4_storage = state;
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

int moe_infer_exact_q4_enable(MoEInferEngine *eng, bool enabled) {
    if (!eng) return DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT;
    DmlxExactQ4ProducerState *state = eng->exact_q4_storage;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (!enabled) {
        if (atomic_load_explicit(
                &state->active_read_publishers, memory_order_acquire) != 0u)
            return DMLX_EXACT_Q4_PRODUCER_BUSY;
        eng->exact_q4 = NULL;
        return DMLX_EXACT_Q4_PRODUCER_OK;
    }
    eng->exact_q4 = state;
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

bool moe_infer_exact_q4_recording_enabled(const MoEInferEngine *eng) {
    return eng && eng->exact_q4 != NULL;
}

int moe_infer_exact_q4_begin_request(
    MoEInferEngine *eng, uint32_t request_ordinal) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (state->request_active || state->finalized ||
        atomic_load_explicit(
            &state->active_read_publishers, memory_order_acquire) != 0u)
        return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    exact_q4_clear_records(state);
    state->request_ordinal = request_ordinal;
    state->request_active = 1u;
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

int moe_infer_exact_q4_begin_token(
    MoEInferEngine *eng, uint32_t token_ordinal) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (!state->request_active || state->finalized)
        return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    state->token_ordinal = token_ordinal;
    state->token_active = 1u;
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

int moe_infer_exact_q4_append_stage(
    MoEInferEngine *eng, const DmlxExactQ4StageEvent *event,
    uint32_t *out_event_ordinal) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (!event || !out_event_ordinal || !state->request_active ||
        !state->token_active || state->finalized ||
        event->stage >= DMLX_EXACT_Q4_STAGE_COUNT ||
        (event->stage == DMLX_EXACT_Q4_STAGE_LOGITS_SAMPLE
            ? event->layer != UINT16_MAX
            : event->layer >= N_LAYERS) ||
        (event->flags & ~DMLX_EXACT_Q4_EVENT_VALID_FLAGS) != 0u ||
        event->end_ns < event->begin_ns)
        return DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT;
    if (state->stage_count >= state->stage_capacity) {
        if (state->dropped_stages != UINT32_MAX) state->dropped_stages++;
        exact_q4_mark_incomplete(state);
        return DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED;
    }
    const uint32_t ordinal = state->next_stage_ordinal++;
    DmlxExactQ4StageEvent record = *event;
    record.request_ordinal = state->request_ordinal;
    record.token_ordinal = state->token_ordinal;
    record.event_ordinal = ordinal;
    state->stages[state->stage_count++] = record;
    *out_event_ordinal = ordinal;
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

int moe_infer_exact_q4_commit_layer(
    MoEInferEngine *eng, const DmlxExactQ4StageEvent *stages,
    uint32_t stage_count, const DmlxMlaPathEvent *path) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    bool valid = stages && stage_count > 0u && path &&
        state->request_active && state->token_active && !state->finalized &&
        !state->aborted && path->layer < N_LAYERS &&
        path->path < DMLX_MLA_PATH_COUNT &&
        (path->flags & ~DMLX_MLA_PATH_VALID_FLAGS) == 0u;
    const uint8_t persistent = DMLX_MLA_PATH_PERSISTENT_KV;
    const uint8_t requested = DMLX_MLA_PATH_EXTERNAL_CB_REQUESTED;
    const uint8_t used = DMLX_MLA_PATH_EXTERNAL_CB_USED;
    const uint8_t fallback = DMLX_MLA_PATH_FALLBACK;
    const uint8_t split = DMLX_MLA_PATH_STAGE_TIME_SPLIT;
    if (valid) {
        switch (path->path) {
            case DMLX_MLA_PATH_PERSISTENT_EXTERNAL:
                valid = path->flags == (persistent | requested | used); break;
            case DMLX_MLA_PATH_PERSISTENT_INTERNAL:
                valid = path->flags == persistent; break;
            case DMLX_MLA_PATH_PERSISTENT_INTERNAL_SPLIT:
                valid = path->flags == (persistent | requested | split); break;
            case DMLX_MLA_PATH_FALLBACK_TEMP_KV:
                valid = path->flags == fallback; break;
            default: valid = false; break;
        }
    }
    uint16_t previous_stage = 0u;
    for (uint32_t index = 0u; valid && index < stage_count; ++index) {
        const DmlxExactQ4StageEvent *event = &stages[index];
        valid = event->layer == path->layer && event->stage < DMLX_EXACT_Q4_STAGE_COUNT &&
            event->stage != DMLX_EXACT_Q4_STAGE_LOGITS_SAMPLE &&
            (event->flags & ~DMLX_EXACT_Q4_EVENT_VALID_FLAGS) == 0u &&
            event->end_ns >= event->begin_ns &&
            (index == 0u || event->stage >= previous_stage);
        previous_stage = event->stage;
    }
    if (!valid) {
        exact_q4_mark_incomplete(state);
        return DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT;
    }
    if (stage_count > state->stage_capacity - state->stage_count ||
        state->path_count >= state->stage_capacity) {
        if (stage_count > state->stage_capacity - state->stage_count) {
            const uint32_t available = state->stage_capacity - state->stage_count;
            const uint32_t missing = stage_count - available;
            state->dropped_stages = missing > UINT32_MAX - state->dropped_stages
                ? UINT32_MAX : state->dropped_stages + missing;
        }
        if (state->path_count >= state->stage_capacity) {
            state->dropped_paths = state->dropped_paths == UINT32_MAX
                ? UINT32_MAX : state->dropped_paths + 1u;
        }
        exact_q4_mark_incomplete(state);
        return DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED;
    }
    const uint32_t first_stage_ordinal = state->next_stage_ordinal;
    DmlxMlaPathEvent path_record = *path;
    path_record.request_ordinal = state->request_ordinal;
    path_record.token_ordinal = state->token_ordinal;
    path_record.event_ordinal = state->next_path_ordinal++;
    for (uint32_t index = 0u; index < stage_count; ++index) {
        DmlxExactQ4StageEvent record = stages[index];
        record.request_ordinal = state->request_ordinal;
        record.token_ordinal = state->token_ordinal;
        record.event_ordinal = first_stage_ordinal + index;
        state->stages[state->stage_count + index] = record;
    }
    state->next_stage_ordinal += stage_count;
    state->stage_count += stage_count;
    state->paths[state->path_count++] = path_record;
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

int moe_infer_exact_q4_reserve_read(
    MoEInferEngine *eng, const DmlxExactQ4ExpertReadEvent *event,
    uint32_t *out_event_ordinal) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (!event || !out_event_ordinal || !state->request_active ||
        !state->token_active || state->finalized || event->layer >= N_LAYERS ||
        event->expert >= N_EXPERTS || event->route_rank >= N_ACTIVE ||
        event->source >= DMLX_EXACT_Q4_READ_SOURCE_COUNT || event->reserved != 0u ||
        event->flags != 0u)
        return DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT;
    if (!exact_q4_enter_read_owner_operation(state))
        return DMLX_EXACT_Q4_PRODUCER_BUSY;
    if (state->read_count >= state->read_capacity) {
        (void)exact_q4_saturating_increment(&state->dropped_reads);
        exact_q4_mark_incomplete(state);
        exact_q4_leave_read_owner_operation(state);
        return DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED;
    }
    const uint32_t index = state->read_count++;
    const uint32_t ordinal = state->next_read_ordinal++;
    DmlxExactQ4ExpertReadEvent record = *event;
    record.request_ordinal = state->request_ordinal;
    record.token_ordinal = state->token_ordinal;
    record.event_ordinal = ordinal;
    record.worker_start_ns = 0u;
    record.complete_ns = 0u;
    record.read_bytes = 0u;
    record.error_code = 0;
    record.flags = 0u;
    state->reads[index] = record;
    atomic_store_explicit(
        &state->read_states[index], DMLX_EXACT_Q4_READ_RESERVED,
        memory_order_release);
    atomic_fetch_add_explicit(
        &state->active_read_publishers, 1u, memory_order_acq_rel);
    *out_event_ordinal = ordinal;
    exact_q4_leave_read_owner_operation(state);
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

static DmlxExactQ4ExpertReadEvent *exact_q4_find_read(
    DmlxExactQ4ProducerState *state, uint32_t ordinal, uint32_t *out_index) {
    if (ordinal >= state->read_count) return NULL;
    if (out_index) *out_index = ordinal;
    return &state->reads[ordinal];
}

int moe_infer_exact_q4_publish_read_worker_start(
    MoEInferEngine *eng, uint32_t event_ordinal, uint64_t worker_start_ns) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (!exact_q4_enter_read_owner_operation(state))
        return DMLX_EXACT_Q4_PRODUCER_BUSY;
    uint32_t index = 0u;
    DmlxExactQ4ExpertReadEvent *record =
        exact_q4_find_read(state, event_ordinal, &index);
    if (!record || atomic_load_explicit(
            &state->read_states[index], memory_order_acquire) !=
            DMLX_EXACT_Q4_READ_RESERVED ||
        record->event_ordinal != event_ordinal ||
        worker_start_ns < record->submit_ns) {
        exact_q4_leave_read_owner_operation(state);
        return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    }
    uint32_t expected = DMLX_EXACT_Q4_READ_RESERVED;
    if (!atomic_compare_exchange_strong_explicit(
            &state->read_states[index], &expected,
            DMLX_EXACT_Q4_READ_STARTING,
            memory_order_acq_rel, memory_order_acquire)) {
        exact_q4_leave_read_owner_operation(state);
        return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    }
    record->worker_start_ns = worker_start_ns;
    atomic_store_explicit(
        &state->read_states[index], DMLX_EXACT_Q4_READ_STARTED,
        memory_order_release);
    exact_q4_leave_read_owner_operation(state);
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

int moe_infer_exact_q4_publish_read_completion(
    MoEInferEngine *eng, uint32_t event_ordinal, uint64_t complete_ns,
    uint64_t read_bytes, int32_t error_code) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (!exact_q4_enter_read_owner_operation(state))
        return DMLX_EXACT_Q4_PRODUCER_BUSY;
    uint32_t index = 0u;
    DmlxExactQ4ExpertReadEvent *record =
        exact_q4_find_read(state, event_ordinal, &index);
    if (!record || atomic_load_explicit(
            &state->read_states[index], memory_order_acquire) !=
            DMLX_EXACT_Q4_READ_STARTED ||
        record->event_ordinal != event_ordinal ||
        record->worker_start_ns < record->submit_ns ||
        complete_ns < record->worker_start_ns ||
        read_bytes > record->requested_bytes) {
        exact_q4_leave_read_owner_operation(state);
        return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    }
    uint32_t expected = DMLX_EXACT_Q4_READ_STARTED;
    if (!atomic_compare_exchange_strong_explicit(
            &state->read_states[index], &expected,
            DMLX_EXACT_Q4_READ_PUBLISHING,
            memory_order_acq_rel, memory_order_acquire)) {
        exact_q4_leave_read_owner_operation(state);
        return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    }
    record->complete_ns = complete_ns;
    record->read_bytes = read_bytes;
    record->error_code = error_code;
    record->flags = DMLX_EXACT_Q4_READ_COMPLETED;
    atomic_store_explicit(
        &state->read_states[index], DMLX_EXACT_Q4_READ_COMPLETE,
        memory_order_release);
    atomic_fetch_sub_explicit(
        &state->active_read_publishers, 1u, memory_order_acq_rel);
    exact_q4_leave_read_owner_operation(state);
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

int moe_infer_exact_q4_abort_read(
    MoEInferEngine *eng, uint32_t event_ordinal) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (!exact_q4_enter_read_owner_operation(state))
        return DMLX_EXACT_Q4_PRODUCER_BUSY;
    uint32_t index = 0u;
    if (!exact_q4_find_read(state, event_ordinal, &index)) {
        exact_q4_leave_read_owner_operation(state);
        return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    }
    uint32_t current = atomic_load_explicit(
        &state->read_states[index], memory_order_acquire);
    while (current == DMLX_EXACT_Q4_READ_RESERVED ||
           current == DMLX_EXACT_Q4_READ_STARTED) {
        if (atomic_compare_exchange_weak_explicit(
                &state->read_states[index], &current,
                DMLX_EXACT_Q4_READ_ABORTED,
                memory_order_acq_rel, memory_order_acquire)) {
            exact_q4_mark_incomplete(state);
            atomic_fetch_sub_explicit(
                &state->active_read_publishers, 1u, memory_order_acq_rel);
            exact_q4_leave_read_owner_operation(state);
            return DMLX_EXACT_Q4_PRODUCER_OK;
        }
    }
    exact_q4_leave_read_owner_operation(state);
    return current == DMLX_EXACT_Q4_READ_PUBLISHING
        ? DMLX_EXACT_Q4_PRODUCER_BUSY
        : DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
}

int moe_infer_exact_q4_abort_request(MoEInferEngine *eng) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (state->aborted) return DMLX_EXACT_Q4_PRODUCER_OK;
    if (!state->request_active || state->finalized)
        return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    uint32_t expected_aborting = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &state->aborting, &expected_aborting, 1u,
            memory_order_acq_rel, memory_order_acquire))
        return DMLX_EXACT_Q4_PRODUCER_BUSY;
    if (atomic_load_explicit(
            &state->read_owner_operations, memory_order_acquire) != 0u) {
        atomic_store_explicit(&state->aborting, 0u, memory_order_release);
        return DMLX_EXACT_Q4_PRODUCER_BUSY;
    }

    // The abort gate prevents new owner transitions while this preflight and
    // the following slot claims execute as one request-level transaction.
    for (uint32_t index = 0u; index < state->read_count; ++index) {
        const uint32_t current = atomic_load_explicit(
            &state->read_states[index], memory_order_acquire);
        if (current == DMLX_EXACT_Q4_READ_STARTING ||
            current == DMLX_EXACT_Q4_READ_PUBLISHING) {
            atomic_store_explicit(&state->aborting, 0u, memory_order_release);
            return DMLX_EXACT_Q4_PRODUCER_BUSY;
        }
    }
    for (uint32_t index = 0u; index < state->read_count; ++index) {
        uint32_t current = atomic_load_explicit(
            &state->read_states[index], memory_order_acquire);
        while (current == DMLX_EXACT_Q4_READ_RESERVED ||
               current == DMLX_EXACT_Q4_READ_STARTED) {
            if (atomic_compare_exchange_weak_explicit(
                    &state->read_states[index], &current,
                    DMLX_EXACT_Q4_READ_ABORTED,
                    memory_order_acq_rel, memory_order_acquire)) {
                atomic_fetch_sub_explicit(
                    &state->active_read_publishers, 1u, memory_order_acq_rel);
                break;
            }
            if (current == DMLX_EXACT_Q4_READ_STARTING ||
                current == DMLX_EXACT_Q4_READ_PUBLISHING) {
                atomic_store_explicit(
                    &state->aborting, 0u, memory_order_release);
                return DMLX_EXACT_Q4_PRODUCER_BUSY;
            }
        }
    }
    eng->deferred.exact_q4_pending.active = false;
    exact_q4_mark_incomplete(state);
    state->request_active = 0u;
    state->token_active = 0u;
    state->aborted = 1u;
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

int moe_infer_exact_q4_finalize(MoEInferEngine *eng) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (!state->request_active || state->finalized || state->aborted)
        return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    if (atomic_load_explicit(
            &state->active_read_publishers, memory_order_acquire) != 0u) {
        exact_q4_mark_incomplete(state);
        return DMLX_EXACT_Q4_PRODUCER_BUSY;
    }
    if (atomic_load_explicit(&state->incomplete, memory_order_acquire))
        return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    state->request_active = 0u;
    state->token_active = 0u;
    state->finalized = 1u;
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

int moe_infer_exact_q4_snapshot(
    const MoEInferEngine *eng, DmlxExactQ4LatencySnapshot *snapshot,
    DmlxExactQ4StageEvent *stages, uint32_t stage_capacity,
    DmlxExactQ4ExpertReadEvent *reads, uint32_t read_capacity) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!snapshot) return DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT;
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->abi_version = DMLX_EXACT_Q4_LATENCY_EVENT_ABI_VERSION;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    snapshot->enabled = 1u;
    snapshot->finalized = state->finalized;
    snapshot->request_ordinal = state->request_ordinal;
    snapshot->token_ordinal = state->token_ordinal;
    snapshot->stage_capacity = state->stage_capacity;
    snapshot->read_capacity = state->read_capacity;
    snapshot->stage_count = state->stage_count;
    snapshot->dropped_stages = state->dropped_stages;
    snapshot->dropped_reads = atomic_load_explicit(
        &state->dropped_reads, memory_order_acquire);
    snapshot->active_read_publishers = atomic_load_explicit(
        &state->active_read_publishers, memory_order_acquire);
    uint32_t published = 0u;
    for (uint32_t index = 0u; index < state->read_count; ++index) {
        if (atomic_load_explicit(
                &state->read_states[index], memory_order_acquire) ==
                    DMLX_EXACT_Q4_READ_COMPLETE) {
            published++;
        }
    }
    snapshot->read_count = published;
    snapshot->incomplete = atomic_load_explicit(
        &state->incomplete, memory_order_acquire) ||
        published != state->read_count || snapshot->active_read_publishers != 0u;
    snapshot->valid = state->finalized && !snapshot->incomplete &&
        snapshot->dropped_stages == 0u && snapshot->dropped_reads == 0u;
    if ((state->stage_count > 0u &&
         (!stages || stage_capacity < state->stage_count)) ||
        (published > 0u && (!reads || read_capacity < published))) {
        snapshot->valid = 0u;
        snapshot->incomplete = 1u;
        return DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED;
    }
    if (state->stage_count > 0u) {
        memcpy(stages, state->stages,
            (size_t)state->stage_count * sizeof(*stages));
        snapshot->stages = stages;
    }
    if (published > 0u) {
        uint32_t output = 0u;
        for (uint32_t index = 0u; index < state->read_count; ++index) {
            if (atomic_load_explicit(
                    &state->read_states[index], memory_order_acquire) ==
                    DMLX_EXACT_Q4_READ_COMPLETE) {
                reads[output++] = state->reads[index];
            }
        }
        snapshot->reads = reads;
    }
    return snapshot->valid
        ? DMLX_EXACT_Q4_PRODUCER_OK
        : DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
}

int moe_infer_exact_q4_snapshot_mla_paths(
    const MoEInferEngine *eng, DmlxExactQ4MlaPathSnapshot *snapshot,
    DmlxMlaPathEvent *paths, uint32_t path_capacity) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!snapshot) return DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT;
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->abi_version = DMLX_EXACT_Q4_MLA_PATH_ABI_VERSION;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    snapshot->enabled = 1u;
    snapshot->finalized = state->finalized;
    snapshot->request_ordinal = state->request_ordinal;
    snapshot->token_ordinal = state->token_ordinal;
    snapshot->capacity = state->stage_capacity;
    snapshot->count = state->path_count;
    snapshot->dropped = state->dropped_paths;
    snapshot->valid = state->finalized && !state->aborted &&
        !atomic_load_explicit(&state->incomplete, memory_order_acquire) &&
        state->dropped_paths == 0u;
    if (!state->finalized) return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    if (state->path_count > 0u && (!paths || path_capacity < state->path_count)) {
        snapshot->valid = 0u;
        return DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED;
    }
    if (state->path_count > 0u) {
        memcpy(paths, state->paths,
            (size_t)state->path_count * sizeof(*paths));
        snapshot->paths = paths;
    }
    return snapshot->valid ? DMLX_EXACT_Q4_PRODUCER_OK
        : DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
}

int moe_infer_exact_q4_reset(MoEInferEngine *eng) {
    if (!eng) return DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT;
    return exact_q4_reset_state(eng->exact_q4_storage);
}

int moe_infer_exact_q4_deconfigure(MoEInferEngine *eng) {
    if (!eng) return DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT;
    DmlxExactQ4ProducerState *state = eng->exact_q4_storage;
    if (!state) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (atomic_load_explicit(
            &state->active_read_publishers, memory_order_acquire) != 0u)
        return DMLX_EXACT_Q4_PRODUCER_BUSY;
    eng->exact_q4 = NULL;
    eng->exact_q4_storage = NULL;
    free(state);
    return DMLX_EXACT_Q4_PRODUCER_OK;
}

#if DMLX_GPU_TIMESTAMP_PROBE
static uint64_t gpu_timestamp_saturating_increment(uint64_t value) {
    return value == UINT64_MAX ? UINT64_MAX : value + 1u;
}

int moe_infer_gpu_timestamp_probe_set_enabled(
    MoEInferEngine *eng, bool enabled) {
    if (!eng || !eng->initialized)
        return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
    if (!enabled) {
        DmlxGpuTimestampProbeState *state =
            eng->gpu_timestamp_probe_storage;
        // Deactivation does not mutate or free owner-bound storage. Clear the
        // hot-path pointer first so every failure path reliably stops timestamp
        // reads even when reset was attempted from a non-owner thread.
        eng->gpu_timestamp_probe = NULL;
        if (state && !gpu_timestamp_probe_owned_by_current_thread(state))
            return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
        return DMLX_GPU_TIMESTAMP_PROBE_OK;
    }
    if (!eng->gpu_timestamp_probe_storage) {
        eng->gpu_timestamp_probe_storage = calloc(
            1u, sizeof(DmlxGpuTimestampProbeState));
        if (!eng->gpu_timestamp_probe_storage)
            return DMLX_GPU_TIMESTAMP_PROBE_ALLOCATION_FAILED;
        eng->gpu_timestamp_probe_storage->owner_thread = pthread_self();
    } else if (!gpu_timestamp_probe_owned_by_current_thread(
                   eng->gpu_timestamp_probe_storage)) {
        return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
    }
    eng->gpu_timestamp_probe = eng->gpu_timestamp_probe_storage;
    return DMLX_GPU_TIMESTAMP_PROBE_OK;
}

bool moe_infer_gpu_timestamp_probe_recording_enabled(
    const MoEInferEngine *eng) {
    return eng && eng->gpu_timestamp_probe != NULL;
}

int moe_infer_gpu_timestamp_probe_reset(MoEInferEngine *eng) {
    if (!eng) return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
    DmlxGpuTimestampProbeState *state = eng->gpu_timestamp_probe_storage;
    if (!state) return DMLX_GPU_TIMESTAMP_PROBE_DISABLED;
    if (!gpu_timestamp_probe_owned_by_current_thread(state))
        return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
    const pthread_t owner_thread = state->owner_thread;
    memset(state, 0, sizeof(*state));
    state->owner_thread = owner_thread;
    return DMLX_GPU_TIMESTAMP_PROBE_OK;
}

static bool gpu_timestamp_sample_matches_reason(
    const DmlxGpuTimestampSample *sample) {
    const bool host_ordered =
        sample->pre_commit_ns <= sample->post_commit_ns &&
        sample->post_commit_ns <= sample->post_wait_ns;
    const bool gpu_finite = isfinite(sample->gpu_start_s) &&
        isfinite(sample->gpu_end_s);
    if (sample->valid != 0u) {
        return sample->reason == DMLX_GPU_TIMESTAMP_VALID && host_ordered &&
            gpu_finite && sample->gpu_start_s > 0.0 &&
            sample->gpu_end_s > sample->gpu_start_s;
    }
    switch (sample->reason) {
        case DMLX_GPU_TIMESTAMP_SELECTOR_UNAVAILABLE:
            return host_ordered && sample->gpu_start_s == 0.0 &&
                sample->gpu_end_s == 0.0;
        case DMLX_GPU_TIMESTAMP_NONFINITE:
            return host_ordered && !gpu_finite;
        case DMLX_GPU_TIMESTAMP_ZERO:
            return host_ordered && gpu_finite &&
                (sample->gpu_start_s <= 0.0 || sample->gpu_end_s <= 0.0);
        case DMLX_GPU_TIMESTAMP_REVERSED:
            return host_ordered && gpu_finite &&
                sample->gpu_start_s > 0.0 &&
                sample->gpu_end_s <= sample->gpu_start_s;
        case DMLX_GPU_TIMESTAMP_HOST_ORDER:
            return !host_ordered;
        default:
            return false;
    }
}

static int gpu_timestamp_probe_append(
    MoEInferEngine *eng, const DmlxGpuTimestampSample *sample) {
    DmlxGpuTimestampProbeState *state =
        eng ? eng->gpu_timestamp_probe : NULL;
    if (!state) return DMLX_GPU_TIMESTAMP_PROBE_DISABLED;
    if (!gpu_timestamp_probe_owned_by_current_thread(state))
        return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
    if (!sample || sample->layer >= N_LAYERS ||
        (sample->kind != DMLX_GPU_CB_ATTENTION &&
         sample->kind != DMLX_GPU_CB_ROUTED_MOE) ||
        sample->valid > 1u || !gpu_timestamp_sample_matches_reason(sample)) {
        return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
    }

    const uint64_t sequence = state->total_seen;
    state->total_seen = gpu_timestamp_saturating_increment(state->total_seen);
    if (state->count >= DMLX_GPU_TIMESTAMP_PROBE_CAPACITY) {
        state->overflow_count =
            gpu_timestamp_saturating_increment(state->overflow_count);
        return DMLX_GPU_TIMESTAMP_PROBE_CAPACITY_EXCEEDED;
    }
    DmlxGpuTimestampSample record = *sample;
    record.sequence = sequence;
    state->samples[state->count++] = record;
    return DMLX_GPU_TIMESTAMP_PROBE_OK;
}

static void gpu_timestamp_probe_identity(
    const MoEInferEngine *eng, uint32_t fallback_token_index,
    uint64_t *request_id, uint32_t *token_index) {
    const DmlxExactQ4ProducerState *exact = eng ? eng->exact_q4 : NULL;
    if (exact && exact->request_active && exact->token_active) {
        *request_id = exact->request_ordinal;
        *token_index = exact->token_ordinal;
        return;
    }
    // Standalone probe collection is defined as one request (ID 0), with the
    // absolute model position as token index. Exact-Q4 capture supplies true
    // request-local ordinals when both diagnostics are enabled.
    *request_id = 0u;
    *token_index = fallback_token_index;
}

static int gpu_timestamp_probe_record_raw(
    MoEInferEngine *eng, uint16_t kind, uint16_t layer,
    uint32_t fallback_token_index, uint64_t pre_commit_ns,
    uint64_t post_commit_ns, uint64_t post_wait_ns,
    bool selectors_available, double gpu_start_s, double gpu_end_s) {
    DmlxGpuTimestampSample sample = {
        .layer = layer,
        .kind = kind,
        .pre_commit_ns = pre_commit_ns,
        .post_commit_ns = post_commit_ns,
        .post_wait_ns = post_wait_ns,
        .gpu_start_s = gpu_start_s,
        .gpu_end_s = gpu_end_s,
    };
    gpu_timestamp_probe_identity(
        eng, fallback_token_index, &sample.request_id, &sample.token_index);
    if (pre_commit_ns > post_commit_ns || post_commit_ns > post_wait_ns) {
        sample.reason = DMLX_GPU_TIMESTAMP_HOST_ORDER;
    } else if (!selectors_available) {
        sample.gpu_start_s = 0.0;
        sample.gpu_end_s = 0.0;
        sample.reason = DMLX_GPU_TIMESTAMP_SELECTOR_UNAVAILABLE;
    } else if (!isfinite(gpu_start_s) || !isfinite(gpu_end_s)) {
        sample.reason = DMLX_GPU_TIMESTAMP_NONFINITE;
    } else if (gpu_start_s <= 0.0 || gpu_end_s <= 0.0) {
        sample.reason = DMLX_GPU_TIMESTAMP_ZERO;
    } else if (gpu_end_s <= gpu_start_s) {
        sample.reason = DMLX_GPU_TIMESTAMP_REVERSED;
    } else {
        sample.valid = 1u;
        sample.reason = DMLX_GPU_TIMESTAMP_VALID;
    }
    return gpu_timestamp_probe_append(eng, &sample);
}

static void gpu_timestamp_probe_record_completed_cb(
    MoEInferEngine *eng, uint16_t kind, uint16_t layer,
    uint32_t fallback_token_index, uint64_t pre_commit_ns,
    uint64_t post_commit_ns, uint64_t post_wait_ns,
    id<MTLCommandBuffer> command_buffer) {
    const bool selectors_available = command_buffer &&
        [command_buffer respondsToSelector:@selector(GPUStartTime)] &&
        [command_buffer respondsToSelector:@selector(GPUEndTime)];
    const double gpu_start_s = selectors_available
        ? [command_buffer GPUStartTime] : 0.0;
    const double gpu_end_s = selectors_available
        ? [command_buffer GPUEndTime] : 0.0;
    (void)gpu_timestamp_probe_record_raw(
        eng, kind, layer, fallback_token_index, pre_commit_ns,
        post_commit_ns, post_wait_ns, selectors_available,
        gpu_start_s, gpu_end_s);
}

size_t moe_infer_gpu_timestamp_probe_snapshot(
    const MoEInferEngine *eng, DmlxGpuTimestampSample *out, size_t capacity,
    uint64_t *total_seen, uint64_t *overflow_count) {
    const DmlxGpuTimestampProbeState *state =
        eng ? eng->gpu_timestamp_probe_storage : NULL;
    if (state && !gpu_timestamp_probe_owned_by_current_thread(state)) {
        if (total_seen) *total_seen = 0u;
        if (overflow_count) *overflow_count = 0u;
        return 0u;
    }
    if (total_seen) *total_seen = state ? state->total_seen : 0u;
    if (overflow_count) *overflow_count = state ? state->overflow_count : 0u;
    if (!state) return 0u;
    if (!out || capacity == 0u) return state->count;
    const size_t copied = capacity < state->count ? capacity : state->count;
    if (copied > 0u) {
        memcpy(out, state->samples, copied * sizeof(*out));
    }
    return copied;
}

int moe_infer_gpu_timestamp_probe_deconfigure(MoEInferEngine *eng) {
    if (!eng) return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
    DmlxGpuTimestampProbeState *state = eng->gpu_timestamp_probe_storage;
    if (state && !gpu_timestamp_probe_owned_by_current_thread(state))
        return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
    eng->gpu_timestamp_probe = NULL;
    eng->gpu_timestamp_probe_storage = NULL;
    if (!state) return DMLX_GPU_TIMESTAMP_PROBE_DISABLED;
    free(state);
    return DMLX_GPU_TIMESTAMP_PROBE_OK;
}

#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
int moe_infer_gpu_timestamp_probe_test_append(
    MoEInferEngine *eng, const DmlxGpuTimestampSample *sample) {
    return gpu_timestamp_probe_append(eng, sample);
}

int moe_infer_gpu_timestamp_probe_test_record_raw(
    MoEInferEngine *eng, uint16_t kind, uint16_t layer,
    uint32_t fallback_token_index, uint64_t pre_commit_ns,
    uint64_t post_commit_ns, uint64_t post_wait_ns,
    bool selectors_available, double gpu_start_s, double gpu_end_s) {
    return gpu_timestamp_probe_record_raw(
        eng, kind, layer, fallback_token_index, pre_commit_ns,
        post_commit_ns, post_wait_ns, selectors_available,
        gpu_start_s, gpu_end_s);
}

void moe_infer_gpu_timestamp_probe_test_corrupt_accounting(
    MoEInferEngine *eng, uint64_t total_seen, uint64_t overflow_count) {
    DmlxGpuTimestampProbeState *state =
        eng ? eng->gpu_timestamp_probe_storage : NULL;
    if (!state) return;
    state->total_seen = total_seen;
    state->overflow_count = overflow_count;
}

void moe_infer_gpu_timestamp_probe_test_corrupt_sample(
    MoEInferEngine *eng, uint32_t index, uint16_t kind, uint32_t reason) {
    DmlxGpuTimestampProbeState *state =
        eng ? eng->gpu_timestamp_probe_storage : NULL;
    if (!state || index >= state->count) return;
    state->samples[index].kind = kind;
    state->samples[index].reason = reason;
}
#endif
#else
int moe_infer_gpu_timestamp_probe_set_enabled(
    MoEInferEngine *eng, bool enabled) {
    if (!eng || !eng->initialized)
        return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
    eng->gpu_timestamp_probe = NULL;
    eng->gpu_timestamp_probe_storage = NULL;
    return enabled ? DMLX_GPU_TIMESTAMP_PROBE_DISABLED
                   : DMLX_GPU_TIMESTAMP_PROBE_OK;
}

bool moe_infer_gpu_timestamp_probe_recording_enabled(
    const MoEInferEngine *eng) {
    (void)eng;
    return false;
}

int moe_infer_gpu_timestamp_probe_reset(MoEInferEngine *eng) {
    if (!eng) return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
    eng->gpu_timestamp_probe = NULL;
    eng->gpu_timestamp_probe_storage = NULL;
    return DMLX_GPU_TIMESTAMP_PROBE_DISABLED;
}

size_t moe_infer_gpu_timestamp_probe_snapshot(
    const MoEInferEngine *eng, DmlxGpuTimestampSample *out, size_t capacity,
    uint64_t *total_seen, uint64_t *overflow_count) {
    (void)eng;
    (void)out;
    (void)capacity;
    if (total_seen) *total_seen = 0u;
    if (overflow_count) *overflow_count = 0u;
    return 0u;
}

int moe_infer_gpu_timestamp_probe_deconfigure(MoEInferEngine *eng) {
    if (!eng) return DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT;
    eng->gpu_timestamp_probe = NULL;
    eng->gpu_timestamp_probe_storage = NULL;
    return DMLX_GPU_TIMESTAMP_PROBE_DISABLED;
}

#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
int moe_infer_gpu_timestamp_probe_test_append(
    MoEInferEngine *eng, const DmlxGpuTimestampSample *sample) {
    (void)eng;
    (void)sample;
    return DMLX_GPU_TIMESTAMP_PROBE_DISABLED;
}

int moe_infer_gpu_timestamp_probe_test_record_raw(
    MoEInferEngine *eng, uint16_t kind, uint16_t layer,
    uint32_t fallback_token_index, uint64_t pre_commit_ns,
    uint64_t post_commit_ns, uint64_t post_wait_ns,
    bool selectors_available, double gpu_start_s, double gpu_end_s) {
    (void)eng;
    (void)kind;
    (void)layer;
    (void)fallback_token_index;
    (void)pre_commit_ns;
    (void)post_commit_ns;
    (void)post_wait_ns;
    (void)selectors_available;
    (void)gpu_start_s;
    (void)gpu_end_s;
    return DMLX_GPU_TIMESTAMP_PROBE_DISABLED;
}
#endif
#endif

#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
void moe_infer_exact_q4_test_set_clock(
    MoEInferEngine *eng, DmlxExactQ4TestClockFn clock_fn, void *context) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4_storage : NULL;
    if (!state) return;
    state->test_clock = clock_fn;
    state->test_clock_context = context;
}

uint64_t moe_infer_exact_q4_test_clock_now(MoEInferEngine *eng) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (!state) return 0u;
    return state->test_clock
        ? state->test_clock(state->test_clock_context)
        : moe_infer_monotonic_ns();
}

void moe_infer_exact_q4_test_seed_drops(
    MoEInferEngine *eng, uint32_t dropped_stages, uint32_t dropped_reads) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4_storage : NULL;
    if (!state) return;
    state->dropped_stages = dropped_stages;
    atomic_store_explicit(
        &state->dropped_reads, dropped_reads, memory_order_relaxed);
}

void moe_infer_exact_q4_test_seed_read_state(
    MoEInferEngine *eng, uint32_t event_ordinal, uint32_t read_state) {
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4_storage : NULL;
    if (!state || event_ordinal >= state->read_count) return;
    atomic_store_explicit(
        &state->read_states[event_ordinal], read_state, memory_order_release);
}

const void *moe_infer_exact_q4_test_storage(const MoEInferEngine *eng) {
    return eng ? eng->exact_q4_storage : NULL;
}

size_t moe_infer_exact_q4_test_allocation_bytes(const MoEInferEngine *eng) {
    return eng && eng->exact_q4_storage
        ? eng->exact_q4_storage->allocation_bytes : 0u;
}
#endif

typedef enum NativeTelemetryGpuTimestampStatus {
    NATIVE_TELEMETRY_GPU_TIMESTAMP_DISABLED = 0,
    NATIVE_TELEMETRY_GPU_TIMESTAMP_AVAILABLE = 1,
    NATIVE_TELEMETRY_GPU_TIMESTAMP_ENABLE_FAILED = 2,
    NATIVE_TELEMETRY_GPU_TIMESTAMP_RESET_FAILED = 3,
    NATIVE_TELEMETRY_GPU_TIMESTAMP_INVALID_SNAPSHOT = 4,
} NativeTelemetryGpuTimestampStatus;

typedef struct NativeTelemetryGpuKindAggregate {
    uint64_t duration_ns_sum;
    uint64_t valid_count;
    uint64_t invalid_counts[5];
    bool duration_sum_overflow;
} NativeTelemetryGpuKindAggregate;

typedef struct NativeTelemetryGpuAggregate {
    NativeTelemetryGpuTimestampStatus status;
    uint64_t total_seen;
    uint64_t retained_count;
    uint64_t overflow_count;
    NativeTelemetryGpuKindAggregate kinds[2];
} NativeTelemetryGpuAggregate;

static const char *telemetry_phase_name(NativeTelemetryPhase phase) {
    switch (phase) {
        case NATIVE_TELEMETRY_PHASE_PREFILL: return "prefill";
        case NATIVE_TELEMETRY_PHASE_DECODE: return "decode";
        case NATIVE_TELEMETRY_PHASE_VERIFY: return "verify";
        default: return "invalid";
    }
}

static void attention_stage_fail(
    DmlxAttentionStageProbeState *state, DmlxAttentionStageStatus status) {
    if (!state) return;
    state->status = status;
    state->layer_active = false;
    state->next_stage = 0u;
    memset(state->duration_ns_sum, 0, sizeof(state->duration_ns_sum));
    memset(state->valid_count, 0, sizeof(state->valid_count));
}

int moe_infer_attention_stage_configure(MoEInferEngine *eng, bool enabled) {
    if (!eng || !eng->initialized) return -1;
    if (!enabled) {
        moe_infer_attention_stage_deconfigure(eng);
        return 0;
    }
    if (!eng->telemetry_enabled || getenv("MF_STAGE_TIME")) return -1;
    if (eng->attention_stage_probe_storage) {
        eng->attention_stage_probe = eng->attention_stage_probe_storage;
        return 0;
    }
    id<MTLDevice> device = (id<MTLDevice>)eng->device;
    if (!device || ![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        eng->telemetry_attention_stage_status = DMLX_ATTENTION_STAGE_UNSUPPORTED;
        return 1;
    }
    id<MTLCounterSet> selected = nil;
    NSUInteger timestamp_sets = 0;
    for (id<MTLCounterSet> set in [device counterSets]) {
        if ([[set name] caseInsensitiveCompare:@"timestamp"] != NSOrderedSame) continue;
        bool contains_timestamp = false;
        for (id<MTLCounter> counter in [set counters]) {
            if ([[counter name] isEqualToString:MTLCommonCounterTimestamp]) {
                contains_timestamp = true;
                break;
            }
        }
        if (contains_timestamp) {
            timestamp_sets++;
            selected = set;
        }
    }
    if (timestamp_sets != 1u || !selected) {
        eng->telemetry_attention_stage_status = DMLX_ATTENTION_STAGE_UNSUPPORTED;
        return 1;
    }
    DmlxAttentionStageProbeState *state = calloc(1, sizeof(*state));
    if (!state) {
        eng->telemetry_attention_stage_status = DMLX_ATTENTION_STAGE_CONFIGURE_FAILED;
        return -1;
    }
    MTLCounterSampleBufferDescriptor *sample_desc =
        [[MTLCounterSampleBufferDescriptor alloc] init];
    [sample_desc setCounterSet:selected];
    [sample_desc setSampleCount:DMLX_ATTENTION_STAGE_SAMPLE_COUNT];
    [sample_desc setStorageMode:MTLStorageModeShared];
    NSError *error = nil;
    state->sample_buffer = [device newCounterSampleBufferWithDescriptor:sample_desc
        error:&error];
    [sample_desc release];
    if (!state->sample_buffer) {
        free(state);
        eng->telemetry_attention_stage_status = DMLX_ATTENTION_STAGE_CONFIGURE_FAILED;
        return -1;
    }
    for (NSUInteger i = 0; i < DMLX_ATTENTION_STAGE_COUNT; ++i) {
        MTLComputePassDescriptor *descriptor =
            [[MTLComputePassDescriptor computePassDescriptor] retain];
        id attachment =
            [[descriptor sampleBufferAttachments] objectAtIndexedSubscript:0];
        [attachment setSampleBuffer:state->sample_buffer];
        [attachment setStartOfEncoderSampleIndex:2u * i];
        [attachment setEndOfEncoderSampleIndex:2u * i + 1u];
        state->descriptors[i] = descriptor;
    }
    state->status = DMLX_ATTENTION_STAGE_AVAILABLE;
    eng->telemetry_attention_stage_status = DMLX_ATTENTION_STAGE_AVAILABLE;
    eng->attention_stage_probe_storage = state;
    eng->attention_stage_probe = state;
    return 0;
}

void moe_infer_attention_stage_reset(MoEInferEngine *eng) {
    DmlxAttentionStageProbeState *state = eng ? eng->attention_stage_probe_storage : NULL;
    if (!state) return;
    memset(state->duration_ns_sum, 0, sizeof(state->duration_ns_sum));
    memset(state->valid_count, 0, sizeof(state->valid_count));
    state->next_stage = 0u;
    state->layer_active = false;
    state->status = DMLX_ATTENTION_STAGE_AVAILABLE;
}

bool moe_infer_attention_stage_recording_enabled(const MoEInferEngine *eng) {
    return eng && eng->attention_stage_probe != NULL;
}

void *moe_infer_attention_stage_encoder(
    MoEInferEngine *eng, void *command_buffer, DmlxAttentionStage stage) {
    DmlxAttentionStageProbeState *state = eng ? eng->attention_stage_probe : NULL;
    id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)command_buffer;
    if (!state || !cb) return NULL;
    if ((uint32_t)stage >= DMLX_ATTENTION_STAGE_COUNT ||
        (uint32_t)stage != state->next_stage) {
        attention_stage_fail(state, DMLX_ATTENTION_STAGE_TOPOLOGY_MISMATCH);
        eng->attention_stage_probe = NULL;
        return [cb computeCommandEncoder];
    }
    if (stage == DMLX_ATTENTION_STAGE_MHC_PRE) state->layer_active = true;
    id<MTLComputeCommandEncoder> encoder =
        [cb computeCommandEncoderWithDescriptor:state->descriptors[stage]];
    if (!encoder) {
        attention_stage_fail(state, DMLX_ATTENTION_STAGE_CONFIGURE_FAILED);
        eng->attention_stage_probe = NULL;
        return [cb computeCommandEncoder];
    }
    state->next_stage++;
    return encoder;
}

static bool attention_stage_aggregate(
    DmlxAttentionStageProbeState *state, const uint64_t *samples, size_t count) {
    if (!state || !samples || count != DMLX_ATTENTION_STAGE_SAMPLE_COUNT ||
        !state->layer_active || state->next_stage != DMLX_ATTENTION_STAGE_COUNT) return false;
    uint64_t durations[DMLX_ATTENTION_STAGE_COUNT];
    for (size_t i = 0; i < DMLX_ATTENTION_STAGE_COUNT; ++i) {
        uint64_t begin = samples[2u * i], end = samples[2u * i + 1u];
        if (begin == 0u || end == 0u || end <= begin) return false;
        durations[i] = end - begin;
        if (durations[i] > UINT64_MAX - state->duration_ns_sum[i] ||
            state->valid_count[i] == UINT64_MAX) return false;
    }
    for (size_t i = 0; i < DMLX_ATTENTION_STAGE_COUNT; ++i) {
        state->duration_ns_sum[i] += durations[i];
        state->valid_count[i]++;
    }
    return true;
}

void moe_infer_attention_stage_resolve(
    MoEInferEngine *eng, const DmlxMlaPathIdentity *identity) {
    DmlxAttentionStageProbeState *state = eng ? eng->attention_stage_probe : NULL;
    if (!state) return;
    const uint8_t required = DMLX_MLA_PATH_PERSISTENT_KV |
        DMLX_MLA_PATH_EXTERNAL_CB_REQUESTED | DMLX_MLA_PATH_EXTERNAL_CB_USED;
    if (!identity || identity->path != DMLX_MLA_PATH_PERSISTENT_EXTERNAL ||
        identity->flags != required || getenv("MF_STAGE_TIME") ||
        !state->layer_active || state->next_stage != DMLX_ATTENTION_STAGE_COUNT) {
        attention_stage_fail(state, DMLX_ATTENTION_STAGE_TOPOLOGY_MISMATCH);
        eng->attention_stage_probe = NULL;
        return;
    }
    NSData *data = [state->sample_buffer
        resolveCounterRange:NSMakeRange(0, DMLX_ATTENTION_STAGE_SAMPLE_COUNT)];
    if (!data || [data length] !=
            DMLX_ATTENTION_STAGE_SAMPLE_COUNT * sizeof(MTLCounterResultTimestamp)) {
        attention_stage_fail(state, DMLX_ATTENTION_STAGE_INVALID_SAMPLE);
        eng->attention_stage_probe = NULL;
        return;
    }
    const MTLCounterResultTimestamp *results = [data bytes];
    uint64_t samples[DMLX_ATTENTION_STAGE_SAMPLE_COUNT];
    for (size_t i = 0; i < DMLX_ATTENTION_STAGE_SAMPLE_COUNT; ++i)
        samples[i] = results[i].timestamp;
    if (!attention_stage_aggregate(state, samples, DMLX_ATTENTION_STAGE_SAMPLE_COUNT)) {
        attention_stage_fail(state, DMLX_ATTENTION_STAGE_INVALID_SAMPLE);
        eng->attention_stage_probe = NULL;
        return;
    }
    state->next_stage = 0u;
    state->layer_active = false;
}

void moe_infer_attention_stage_snapshot(
    const MoEInferEngine *eng, DmlxAttentionStageSnapshot *snapshot) {
    if (!snapshot) return;
    memset(snapshot, 0, sizeof(*snapshot));
    const DmlxAttentionStageProbeState *state =
        eng ? eng->attention_stage_probe_storage : NULL;
    if (!state) {
        snapshot->status = eng && eng->telemetry_attention_stage_status !=
            DMLX_ATTENTION_STAGE_AVAILABLE
                ? eng->telemetry_attention_stage_status
                : DMLX_ATTENTION_STAGE_UNSUPPORTED;
        return;
    }
    snapshot->status = state->status;
    snapshot->enabled = eng->attention_stage_probe != NULL;
    if (state->status == DMLX_ATTENTION_STAGE_AVAILABLE) {
        memcpy(snapshot->duration_ns_sum, state->duration_ns_sum,
            sizeof(snapshot->duration_ns_sum));
        memcpy(snapshot->valid_count, state->valid_count,
            sizeof(snapshot->valid_count));
        uint64_t expected = snapshot->valid_count[0];
        if (expected == 0u) snapshot->status = DMLX_ATTENTION_STAGE_INVALID_SAMPLE;
        for (size_t i = 1; i < DMLX_ATTENTION_STAGE_COUNT; ++i)
            if (snapshot->valid_count[i] != expected)
                snapshot->status = DMLX_ATTENTION_STAGE_INVALID_SAMPLE;
        if (snapshot->status != DMLX_ATTENTION_STAGE_AVAILABLE) {
            memset(snapshot->duration_ns_sum, 0, sizeof(snapshot->duration_ns_sum));
            memset(snapshot->valid_count, 0, sizeof(snapshot->valid_count));
        }
    }
}

void moe_infer_attention_stage_deconfigure(MoEInferEngine *eng) {
    DmlxAttentionStageProbeState *state = eng ? eng->attention_stage_probe_storage : NULL;
    if (!state) return;
    eng->attention_stage_probe = NULL;
    eng->attention_stage_probe_storage = NULL;
    for (size_t i = 0; i < DMLX_ATTENTION_STAGE_COUNT; ++i)
        [state->descriptors[i] release];
    [state->sample_buffer release];
    free(state);
}

#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
int moe_infer_attention_stage_test_enable(MoEInferEngine *eng) {
    if (!eng || eng->attention_stage_probe_storage) return -1;
    DmlxAttentionStageProbeState *state = calloc(1, sizeof(*state));
    if (!state) return -1;
    state->status = DMLX_ATTENTION_STAGE_AVAILABLE;
    eng->telemetry_attention_stage_status = DMLX_ATTENTION_STAGE_AVAILABLE;
    eng->attention_stage_probe_storage = state;
    eng->attention_stage_probe = state;
    return 0;
}

int moe_infer_attention_stage_test_aggregate(
    MoEInferEngine *eng, const uint64_t *samples, size_t count,
    bool topology_valid) {
    DmlxAttentionStageProbeState *state = eng ? eng->attention_stage_probe_storage : NULL;
    if (!state || !topology_valid) {
        if (state) attention_stage_fail(state, DMLX_ATTENTION_STAGE_TOPOLOGY_MISMATCH);
        return -1;
    }
    state->layer_active = true;
    state->next_stage = DMLX_ATTENTION_STAGE_COUNT;
    if (!attention_stage_aggregate(state, samples, count)) {
        attention_stage_fail(state, DMLX_ATTENTION_STAGE_INVALID_SAMPLE);
        return -1;
    }
    state->layer_active = false;
    state->next_stage = 0u;
    return 0;
}

void moe_infer_attention_stage_test_seed_sum(
    MoEInferEngine *eng, uint32_t stage, uint64_t sum, uint64_t count) {
    DmlxAttentionStageProbeState *state = eng ? eng->attention_stage_probe_storage : NULL;
    if (!state || stage >= DMLX_ATTENTION_STAGE_COUNT) return;
    state->duration_ns_sum[stage] = sum;
    state->valid_count[stage] = count;
}
#endif

void moe_infer_telemetry_reset(MoEInferEngine *eng) {
    if (!eng || !eng->telemetry_enabled) return;
    memset(&eng->telemetry, 0, sizeof(eng->telemetry));
    memset(eng->telemetry_seen, 0, sizeof(eng->telemetry_seen));
    eng->batch_scalar_layer_calls = 0u;
    eng->batch_batched_moe_layers = 0u;
    eng->batch_workspace_high_water_bytes = 0u;
    eng->telemetry.schema_version = NATIVE_TELEMETRY_SCHEMA_VERSION;
    eng->telemetry.enabled = 1;
    eng->telemetry.active_phase = NATIVE_TELEMETRY_PHASE_PREFILL;
    eng->telemetry.valid_fields = NATIVE_TELEMETRY_VALID_EXPERT_IO |
        NATIVE_TELEMETRY_VALID_MOE_GEOMETRY;
    eng->telemetry_phase = NATIVE_TELEMETRY_PHASE_PREFILL;
    moe_infer_attention_stage_reset(eng);
    for (int phase = 0; phase < NATIVE_TELEMETRY_PHASE_COUNT; ++phase) {
        for (int layer = 0; layer < N_LAYERS; ++layer) {
            eng->telemetry.layers[phase][layer].phase = (uint32_t)phase;
            eng->telemetry.layers[phase][layer].layer = (uint32_t)layer;
        }
    }

    if (eng->telemetry_gpu_timestamp_status !=
        NATIVE_TELEMETRY_GPU_TIMESTAMP_ENABLE_FAILED) {
        const int reset_status =
            moe_infer_gpu_timestamp_probe_reset(eng);
        const int enable_status = reset_status == DMLX_GPU_TIMESTAMP_PROBE_OK
            ? moe_infer_gpu_timestamp_probe_set_enabled(eng, true)
            : reset_status;
        if (reset_status == DMLX_GPU_TIMESTAMP_PROBE_OK &&
            enable_status == DMLX_GPU_TIMESTAMP_PROBE_OK) {
            eng->telemetry_gpu_timestamp_status =
                NATIVE_TELEMETRY_GPU_TIMESTAMP_AVAILABLE;
        } else {
            eng->telemetry_gpu_timestamp_status =
                NATIVE_TELEMETRY_GPU_TIMESTAMP_RESET_FAILED;
            (void)moe_infer_gpu_timestamp_probe_set_enabled(eng, false);
        }
    }
}

void moe_infer_telemetry_enable(MoEInferEngine *eng, bool enabled) {
    if (!eng) return;
    if (!enabled) {
        (void)moe_infer_gpu_timestamp_probe_set_enabled(eng, false);
        moe_infer_attention_stage_deconfigure(eng);
        eng->telemetry_enabled = false;
        eng->telemetry_gpu_timestamp_status =
            NATIVE_TELEMETRY_GPU_TIMESTAMP_DISABLED;
        memset(&eng->telemetry, 0, sizeof(eng->telemetry));
        memset(eng->telemetry_seen, 0, sizeof(eng->telemetry_seen));
        return;
    }

    eng->telemetry_enabled = true;
    eng->telemetry_gpu_timestamp_status =
        moe_infer_gpu_timestamp_probe_set_enabled(eng, true) ==
                DMLX_GPU_TIMESTAMP_PROBE_OK
            ? NATIVE_TELEMETRY_GPU_TIMESTAMP_AVAILABLE
            : NATIVE_TELEMETRY_GPU_TIMESTAMP_ENABLE_FAILED;
    moe_infer_telemetry_reset(eng);
}

void moe_infer_telemetry_set_phase(
    MoEInferEngine *eng,
    NativeTelemetryPhase phase) {
    if (!eng || !eng->telemetry_enabled) return;
    if (phase < 0 || phase >= NATIVE_TELEMETRY_PHASE_COUNT) return;
    eng->telemetry_phase = phase;
    eng->telemetry.active_phase = (uint32_t)phase;
}

int moe_infer_routing_trace_configure(
    MoEInferEngine *eng,
    const char *path,
    size_t record_capacity,
    uint64_t model_fingerprint,
    uint32_t cache_capacity) {
    if (!eng || !path || path[0] == '\0') return -1;
    // Schema v1 records I/O against the route that synchronously resolves an
    // expert. Predictive prefetch can issue unmatched speculative reads and
    // complete them after that route record is emitted, so enabling both would
    // silently under-report bytes, waits, and short reads. Fail explicitly
    // instead of writing a trace that cannot reconcile physical I/O.
    if (eng->prefetch_disabled == 0) {
        fprintf(stderr,
            "[routing-trace] NATIVE_PREFETCH=1 is incompatible with routing trace schema v1; disable prefetch to capture an exact trace\n");
        return -1;
    }
    routing_trace_writer_destroy(eng->routing_trace);
    eng->routing_trace = NULL;
    eng->routing_trace_phase = NATIVE_TELEMETRY_PHASE_PREFILL;
    return routing_trace_writer_create(
        &eng->routing_trace,
        path,
        record_capacity,
        model_fingerprint,
        N_LAYERS,
        N_EXPERTS,
        N_ACTIVE,
        cache_capacity,
        eng->active_expert_size);
}

int moe_infer_routing_trace_request_begin(MoEInferEngine *eng) {
    if (!eng || !eng->routing_trace) return 0;
    eng->routing_trace_phase = NATIVE_TELEMETRY_PHASE_PREFILL;
    return routing_trace_request_begin(
        eng->routing_trace,
        (uint32_t)eng->routing_trace_phase);
}

void moe_infer_routing_trace_set_phase(
    MoEInferEngine *eng,
    NativeTelemetryPhase phase) {
    if (!eng || !eng->routing_trace) return;
    if (phase < 0 || phase >= NATIVE_TELEMETRY_PHASE_COUNT) return;
    eng->routing_trace_phase = phase;
}

int moe_infer_routing_trace_request_end(MoEInferEngine *eng) {
    if (!eng || !eng->routing_trace) return 0;
    return routing_trace_request_end(eng->routing_trace);
}

void moe_infer_telemetry_snapshot(
    const MoEInferEngine *eng,
    NativeRequestMetrics *out_metrics) {
    if (!out_metrics) return;
    memset(out_metrics, 0, sizeof(*out_metrics));
    if (!eng || !eng->telemetry_enabled) return;
    memcpy(out_metrics, &eng->telemetry, sizeof(*out_metrics));
}

static inline NativeLayerMetrics *telemetry_current_layer(
    MoEInferEngine *eng,
    int layer) {
    if (!eng || !eng->telemetry_enabled) return NULL;
    if (layer < 0 || layer >= N_LAYERS) return NULL;
    if (eng->telemetry_phase < 0 ||
        eng->telemetry_phase >= NATIVE_TELEMETRY_PHASE_COUNT) return NULL;
    return &eng->telemetry.layers[eng->telemetry_phase][layer];
}

static inline void telemetry_record_routes(
    MoEInferEngine *eng,
    int layer,
    const int *expert_ids,
    int count) {
    NativeLayerMetrics *metrics = telemetry_current_layer(eng, layer);
    if (!metrics) return;
    metrics->calls++;
    uint64_t *seen = eng->telemetry_seen[eng->telemetry_phase][layer];
    for (int index = 0; index < count; ++index) {
        const int expert = expert_ids[index];
        if (expert < 0 || expert >= N_EXPERTS) continue;
        const int word = expert / 64;
        const uint64_t bit = UINT64_C(1) << (expert % 64);
        if ((seen[word] & bit) == 0) {
            seen[word] |= bit;
            metrics->unique_experts++;
        }
    }
}

static inline void telemetry_record_moe_dispatch(
    MoEInferEngine *eng,
    int layer,
    uint64_t threadgroups,
    uint32_t threads_per_threadgroup,
    uint64_t active_lanes) {
    NativeLayerMetrics *metrics = telemetry_current_layer(eng, layer);
    if (!metrics) return;
    const uint64_t scheduled = threadgroups * threads_per_threadgroup;
    metrics->moe_dispatches++;
    metrics->threadgroups += threadgroups;
    metrics->scheduled_lanes += scheduled;
    metrics->active_lanes += active_lanes <= scheduled ? active_lanes : scheduled;
    if (threads_per_threadgroup > metrics->max_threads_per_threadgroup) {
        metrics->max_threads_per_threadgroup = threads_per_threadgroup;
    }
}

static void telemetry_gpu_invalidate(NativeTelemetryGpuAggregate *aggregate) {
    memset(aggregate, 0, sizeof(*aggregate));
    aggregate->status = NATIVE_TELEMETRY_GPU_TIMESTAMP_INVALID_SNAPSHOT;
}

static bool telemetry_gpu_round_scaled_ns(
    long double scaled_ns, uint64_t *rounded_ns) {
    if (!rounded_ns || !isfinite(scaled_ns) || scaled_ns < 0.0L)
        return false;
    long double integral_ns = 0.0L;
    const long double fractional_ns = modfl(scaled_ns, &integral_ns);
    // 2^64 is exactly representable in every supported floating format. This
    // check occurs before the cast, while UINT64_MAX itself remains allowed.
    if (integral_ns >= 0x1p64L) return false;
    uint64_t value = (uint64_t)integral_ns;
    if (fractional_ns >= 0.5L) {
        if (value == UINT64_MAX) return false;
        value++;
    }
    *rounded_ns = value;
    return true;
}

#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
bool moe_infer_gpu_timestamp_test_round_scaled_ns(
    long double scaled_ns, uint64_t *rounded_ns) {
    return telemetry_gpu_round_scaled_ns(scaled_ns, rounded_ns);
}
#endif

static NativeTelemetryGpuAggregate telemetry_gpu_aggregate(
    const MoEInferEngine *eng) {
    NativeTelemetryGpuAggregate aggregate = {0};
    if (!eng) {
        aggregate.status = NATIVE_TELEMETRY_GPU_TIMESTAMP_INVALID_SNAPSHOT;
        return aggregate;
    }
    aggregate.status =
        (NativeTelemetryGpuTimestampStatus)eng->telemetry_gpu_timestamp_status;
    if (aggregate.status != NATIVE_TELEMETRY_GPU_TIMESTAMP_AVAILABLE) {
        if (aggregate.status != NATIVE_TELEMETRY_GPU_TIMESTAMP_ENABLE_FAILED &&
            aggregate.status != NATIVE_TELEMETRY_GPU_TIMESTAMP_RESET_FAILED) {
            telemetry_gpu_invalidate(&aggregate);
        }
        return aggregate;
    }
    if (!moe_infer_gpu_timestamp_probe_recording_enabled(eng)) {
        telemetry_gpu_invalidate(&aggregate);
        return aggregate;
    }

    DmlxGpuTimestampSample samples[DMLX_GPU_TIMESTAMP_PROBE_CAPACITY];
    aggregate.retained_count = moe_infer_gpu_timestamp_probe_snapshot(
        eng, samples, DMLX_GPU_TIMESTAMP_PROBE_CAPACITY,
        &aggregate.total_seen, &aggregate.overflow_count);
    if (aggregate.retained_count > DMLX_GPU_TIMESTAMP_PROBE_CAPACITY) {
        telemetry_gpu_invalidate(&aggregate);
        return aggregate;
    }

    uint64_t classified_count = 0u;
    for (uint64_t index = 0u; index < aggregate.retained_count; ++index) {
        const DmlxGpuTimestampSample *sample = &samples[index];
        size_t kind_index;
        if (sample->kind == DMLX_GPU_CB_ATTENTION) {
            kind_index = 0u;
        } else if (sample->kind == DMLX_GPU_CB_ROUTED_MOE) {
            kind_index = 1u;
        } else {
            telemetry_gpu_invalidate(&aggregate);
            return aggregate;
        }
        NativeTelemetryGpuKindAggregate *kind =
            &aggregate.kinds[kind_index];
        if (sample->valid == 0u) {
            if (sample->reason < DMLX_GPU_TIMESTAMP_SELECTOR_UNAVAILABLE ||
                sample->reason > DMLX_GPU_TIMESTAMP_HOST_ORDER) {
                telemetry_gpu_invalidate(&aggregate);
                return aggregate;
            }
            kind->invalid_counts[
                sample->reason - DMLX_GPU_TIMESTAMP_SELECTOR_UNAVAILABLE]++;
            classified_count++;
            continue;
        }
        if (sample->valid != 1u ||
            sample->reason != DMLX_GPU_TIMESTAMP_VALID) {
            telemetry_gpu_invalidate(&aggregate);
            return aggregate;
        }

        kind->valid_count++;
        classified_count++;
        const double delta_s = sample->gpu_end_s - sample->gpu_start_s;
        const long double scaled_ns = (long double)delta_s * 1000000000.0L;
        uint64_t duration_ns = 0u;
        if (!isfinite(delta_s) || delta_s <= 0.0 ||
            !telemetry_gpu_round_scaled_ns(scaled_ns, &duration_ns)) {
            kind->duration_sum_overflow = true;
            continue;
        }
        if (kind->duration_sum_overflow ||
            duration_ns > UINT64_MAX - kind->duration_ns_sum) {
            kind->duration_sum_overflow = true;
            continue;
        }
        kind->duration_ns_sum += duration_ns;
    }

    if (classified_count != aggregate.retained_count ||
        aggregate.overflow_count > UINT64_MAX - aggregate.retained_count ||
        aggregate.total_seen !=
            aggregate.retained_count + aggregate.overflow_count) {
        telemetry_gpu_invalidate(&aggregate);
    }
    return aggregate;
}

static const char *telemetry_gpu_status_name(
    NativeTelemetryGpuTimestampStatus status) {
    switch (status) {
        case NATIVE_TELEMETRY_GPU_TIMESTAMP_AVAILABLE: return "available";
        case NATIVE_TELEMETRY_GPU_TIMESTAMP_ENABLE_FAILED:
            return "enable_failed";
        case NATIVE_TELEMETRY_GPU_TIMESTAMP_RESET_FAILED:
            return "reset_failed";
        default: return "invalid_snapshot";
    }
}

static const char *attention_stage_status_name(uint32_t status) {
    switch ((DmlxAttentionStageStatus)status) {
        case DMLX_ATTENTION_STAGE_AVAILABLE: return "available";
        case DMLX_ATTENTION_STAGE_UNSUPPORTED: return "unsupported";
        case DMLX_ATTENTION_STAGE_CONFIGURE_FAILED: return "configure_failed";
        case DMLX_ATTENTION_STAGE_INVALID_SAMPLE: return "invalid_sample";
        case DMLX_ATTENTION_STAGE_TOPOLOGY_MISMATCH: return "topology_mismatch";
        default: return "configure_failed";
    }
}

static void telemetry_gpu_emit_kind_json(
    const char *name, const NativeTelemetryGpuKindAggregate *kind,
    bool available) {
    fprintf(stderr, "\"%s\":{\"duration_ns_sum\":", name);
    if (available && !kind->duration_sum_overflow) {
        fprintf(stderr, "%llu", (unsigned long long)kind->duration_ns_sum);
    } else {
        fputs("null", stderr);
    }
    fprintf(stderr,
        ",\"duration_sum_overflow\":%s,\"valid_count\":%llu,"
        "\"selector_unavailable_count\":%llu,\"nonfinite_count\":%llu,"
        "\"zero_count\":%llu,\"reversed_count\":%llu,"
        "\"host_order_count\":%llu}",
        available && kind->duration_sum_overflow ? "true" : "false",
        (unsigned long long)(available ? kind->valid_count : 0u),
        (unsigned long long)(available ? kind->invalid_counts[0] : 0u),
        (unsigned long long)(available ? kind->invalid_counts[1] : 0u),
        (unsigned long long)(available ? kind->invalid_counts[2] : 0u),
        (unsigned long long)(available ? kind->invalid_counts[3] : 0u),
        (unsigned long long)(available ? kind->invalid_counts[4] : 0u));
}

void moe_infer_telemetry_emit_json(
    const MoEInferEngine *eng,
    uint64_t request_start_ns,
    uint64_t first_token_ns,
    uint64_t decode_start_ns,
    uint64_t request_end_ns,
    uint32_t prompt_tokens,
    uint32_t generated_tokens,
    const uint64_t *token_timestamps_ns,
    size_t recorded_token_count,
    uint64_t dspark_candidate_count,
    uint64_t dspark_accepted_count,
    uint64_t dspark_verification_batches,
    uint64_t dspark_verification_ns) {
    if (!eng || !eng->telemetry_enabled) return;

    NativeRequestMetrics snapshot;
    moe_infer_telemetry_snapshot(eng, &snapshot);
    uint64_t hot_hits = 0, prefetch_hits = 0, batch_pool_hits = 0;
    uint64_t cold_misses = 0, pread_calls = 0, bytes_requested = 0;
    uint64_t bytes_read = 0, short_reads = 0, io_wait_ns = 0, bytes_avoided = 0;
    uint64_t cache_admissions = 0, cache_evictions = 0;
    uint64_t moe_dispatches = 0, moe_command_buffers = 0;
    uint64_t threadgroups = 0, scheduled_lanes = 0, active_lanes = 0;
    uint32_t max_threads_per_threadgroup = 0;
    for (int phase = 0; phase < NATIVE_TELEMETRY_PHASE_COUNT; ++phase) {
        for (int layer = 0; layer < N_LAYERS; ++layer) {
            const NativeLayerMetrics *metrics = &snapshot.layers[phase][layer];
            hot_hits += metrics->hot_hits;
            prefetch_hits += metrics->prefetch_hits;
            batch_pool_hits += metrics->batch_pool_hits;
            cold_misses += metrics->cold_misses;
            pread_calls += metrics->pread_calls;
            bytes_requested += metrics->bytes_requested;
            bytes_read += metrics->bytes_read;
            short_reads += metrics->short_reads;
            io_wait_ns += metrics->io_wait_ns;
            bytes_avoided += metrics->bytes_avoided;
            cache_admissions += metrics->cache_admissions;
            cache_evictions += metrics->cache_evictions;
            moe_dispatches += metrics->moe_dispatches;
            moe_command_buffers += metrics->moe_command_buffers;
            threadgroups += metrics->threadgroups;
            scheduled_lanes += metrics->scheduled_lanes;
            active_lanes += metrics->active_lanes;
            if (metrics->max_threads_per_threadgroup > max_threads_per_threadgroup) {
                max_threads_per_threadgroup = metrics->max_threads_per_threadgroup;
            }
        }
    }

    const bool has_end = request_end_ns >= request_start_ns;
    const bool has_first = generated_tokens > 0 && first_token_ns >= request_start_ns &&
        first_token_ns <= request_end_ns;
    const bool have_all_tokens = token_timestamps_ns &&
        recorded_token_count >= generated_tokens;
    const uint64_t last_token_ns =
        have_all_tokens && generated_tokens > 0
            ? token_timestamps_ns[generated_tokens - 1]
            : 0;
    const bool has_decode = generated_tokens > 1 && has_first &&
        decode_start_ns >= first_token_ns && decode_start_ns <= last_token_ns &&
        last_token_ns <= request_end_ns;
    bool has_itl = generated_tokens > 1 && has_first && have_all_tokens &&
        last_token_ns <= request_end_ns;
    if (has_itl) {
        for (uint32_t index = 1; index < generated_tokens; ++index) {
            if (token_timestamps_ns[index] < token_timestamps_ns[index - 1]) {
                has_itl = false;
                break;
            }
        }
    }

    const NativeTelemetryGpuAggregate gpu_aggregate =
        telemetry_gpu_aggregate(eng);
    const bool gpu_aggregate_available = gpu_aggregate.status ==
        NATIVE_TELEMETRY_GPU_TIMESTAMP_AVAILABLE;

    DmlxAttentionStageSnapshot attention_stages;
    moe_infer_attention_stage_snapshot(eng, &attention_stages);

    flockfile(stderr);
    fprintf(stderr,
        "[NATIVE_TELEMETRY] {\"schema_version\":%d,"
        "\"kind\":\"dmlx-native-request-telemetry\","
        "\"scope\":\"native_engine_after_tokenization\","
        "\"request\":{\"prompt_tokens\":%u,\"generated_tokens\":%u,"
        "\"request_start_ns\":%" PRIu64 ",",
        NATIVE_TELEMETRY_SCHEMA_VERSION,
        prompt_tokens,
        generated_tokens,
        request_start_ns);
    if (has_first) {
        fprintf(stderr,
            "\"first_token_at_ns\":%" PRIu64
            ",\"engine_first_token_latency_ns\":%" PRIu64
            ",\"ttft_ns\":null,\"prefill_ns\":null,",
            first_token_ns,
            first_token_ns - request_start_ns);
    } else {
        fprintf(stderr,
            "\"first_token_at_ns\":null,\"engine_first_token_latency_ns\":null,"
            "\"ttft_ns\":null,\"prefill_ns\":null,");
    }
    if (has_decode) {
        fprintf(stderr,
            "\"decode_start_ns\":%" PRIu64 ",\"decode_ns\":%" PRIu64 ",",
            decode_start_ns,
            last_token_ns - decode_start_ns);
    } else {
        fprintf(stderr, "\"decode_start_ns\":null,\"decode_ns\":null,");
    }
    if (has_end) {
        fprintf(stderr,
            "\"request_end_ns\":%" PRIu64 ",\"request_wall_ns\":%" PRIu64 ",",
            request_end_ns,
            request_end_ns - request_start_ns);
    } else {
        fprintf(stderr, "\"request_end_ns\":null,\"request_wall_ns\":null,");
    }
    fprintf(stderr, "\"itl_ns\":");
    if (has_itl) {
        fputc('[', stderr);
        for (uint32_t index = 1; index < generated_tokens; ++index) {
            if (index > 1) fputc(',', stderr);
            fprintf(stderr, "%" PRIu64,
                token_timestamps_ns[index] - token_timestamps_ns[index - 1]);
        }
        fputc(']', stderr);
    } else {
        fputs("null", stderr);
    }
    fprintf(stderr,
        "},\"batch_path\":{\"scalar_layer_calls\":%" PRIu64
        ",\"batched_moe_layers\":%" PRIu64 "},"
        "\"dspark_verification\":{\"candidate_count\":%" PRIu64
        ",\"accepted_count\":%" PRIu64
        ",\"verification_batches\":%" PRIu64
        ",\"verification_ns\":%" PRIu64 "},"
        "\"cache_policy\":{\"name\":\"%s\","
        "\"experimental\":%s,\"capacity_per_score_layer\":%d,"
        "\"resident_slots\":%" PRIu64 ",\"resident_bytes\":%" PRIu64
        ",\"metadata_bytes\":%zu,\"admissions\":%" PRIu64
        ",\"evictions\":%" PRIu64 "},"
        "\"expert_io\":{\"measurement_scope\":\"critical_path_routed_expert_resolution\","
        "\"pread_calls\":%" PRIu64
        ",\"bytes_requested\":%" PRIu64
        ",\"bytes_read\":%" PRIu64
        ",\"io_wait_ns\":%" PRIu64
        ",\"hot_hits\":%" PRIu64
        ",\"prefetch_hits\":%" PRIu64
        ",\"batch_pool_hits\":%" PRIu64
        ",\"cold_misses\":%" PRIu64
        ",\"bytes_avoided\":%" PRIu64
        ",\"short_reads\":%" PRIu64 "},",
        eng->batch_scalar_layer_calls,
        eng->batch_batched_moe_layers,
        dspark_candidate_count,
        dspark_accepted_count,
        dspark_verification_batches,
        dspark_verification_ns,
        eng->experimental_cache_enabled ? "lru_observed_hot_seeded" : "fixed_hot_observed",
        eng->experimental_cache_enabled ? "true" : "false",
        eng->smelt_n_per_layer,
        (uint64_t)(3 * N_EXPERTS + (N_LAYERS - 3) * eng->smelt_n_per_layer),
        (uint64_t)(3 * N_EXPERTS + (N_LAYERS - 3) * eng->smelt_n_per_layer) *
            eng->active_expert_size,
        sizeof(eng->cache_lru_recency) + sizeof(eng->cache_lru_clock) +
            sizeof(eng->cache_lru_admissions) + sizeof(eng->cache_lru_evictions),
        cache_admissions,
        cache_evictions,
        pread_calls,
        bytes_requested,
        bytes_read,
        io_wait_ns,
        hot_hits,
        prefetch_hits,
        batch_pool_hits,
        cold_misses,
        bytes_avoided,
        short_reads);
    fprintf(stderr,
        "\"dispatch\":{\"measurement_scope\":\"routed_moe_only\","
        "\"dispatches\":%" PRIu64
        ",\"command_buffers\":%" PRIu64
        ",\"threadgroups\":%" PRIu64
        ",\"scheduled_lanes\":%" PRIu64
        ",\"active_lanes\":%" PRIu64
        ",\"max_threads_per_threadgroup\":%u,"
        "\"active_lane_utilization_proxy\":",
        moe_dispatches,
        moe_command_buffers,
        threadgroups,
        scheduled_lanes,
        active_lanes,
        max_threads_per_threadgroup);
    if (scheduled_lanes > 0) {
        fprintf(stderr, "%.9g", (double)active_lanes / (double)scheduled_lanes);
    } else {
        fputs("null", stderr);
    }
    fputs(",\"workspace_high_water_bytes\":", stderr);
    if (eng->batch_workspace_high_water_bytes > 0u) {
        fprintf(stderr, "%" PRIu64, eng->batch_workspace_high_water_bytes);
    } else {
        fputs("null", stderr);
    }
    fputs(",\"attention_gpu_stage_ns\":null,\"peak_rss_bytes\":", stderr);
    struct rusage usage = {0};
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss > 0) {
        // macOS reports ru_maxrss in bytes (Linux reports KiB; dmlx native is
        // currently a macOS/Metal-only runtime).
        fprintf(stderr, "%" PRIu64, (uint64_t)usage.ru_maxrss);
    } else {
        fputs("null", stderr);
    }
    fputs("},\"sampled_gpu_cb_duration\":{"
          "\"measurement_scope\":\"sampled_command_buffer_duration_mass\","
          "\"status\":\"", stderr);
    fputs(telemetry_gpu_status_name(gpu_aggregate.status), stderr);
    fprintf(stderr,
        "\",\"total_seen\":%llu,\"retained_count\":%llu,"
        "\"overflow_count\":%llu,",
        (unsigned long long)(gpu_aggregate_available
            ? gpu_aggregate.total_seen : 0u),
        (unsigned long long)(gpu_aggregate_available
            ? gpu_aggregate.retained_count : 0u),
        (unsigned long long)(gpu_aggregate_available
            ? gpu_aggregate.overflow_count : 0u));
    telemetry_gpu_emit_kind_json(
        "attention", &gpu_aggregate.kinds[0], gpu_aggregate_available);
    fputc(',', stderr);
    telemetry_gpu_emit_kind_json(
        "routed_moe", &gpu_aggregate.kinds[1], gpu_aggregate_available);
    fputs("},\"attention_stage_timestamp\":{"
          "\"measurement_scope\":\"normal_scalar_merged_attention_compute_stages\","
          "\"status\":\"", stderr);
    fputs(attention_stage_status_name(attention_stages.status), stderr);
    fputs("\",\"stages\":[", stderr);
    const bool attention_available =
        attention_stages.status == DMLX_ATTENTION_STAGE_AVAILABLE;
    for (size_t i = 0; i < DMLX_ATTENTION_STAGE_COUNT; ++i) {
        if (i) fputc(',', stderr);
        fprintf(stderr, "{\"name\":\"%s\",\"duration_ns_sum\":",
            attention_stage_names[i]);
        if (attention_available)
            fprintf(stderr, "%" PRIu64, attention_stages.duration_ns_sum[i]);
        else
            fputs("null", stderr);
        fprintf(stderr, ",\"valid_count\":%" PRIu64 "}",
            attention_available ? attention_stages.valid_count[i] : 0u);
    }
    fputs("]},\"phases\":[", stderr);
    for (int phase = 0; phase < NATIVE_TELEMETRY_PHASE_COUNT; ++phase) {
        if (phase > 0) fputc(',', stderr);
        fprintf(stderr, "{\"name\":\"%s\",\"layers\":[",
            telemetry_phase_name((NativeTelemetryPhase)phase));
        bool first_layer = true;
        for (int layer = 0; layer < N_LAYERS; ++layer) {
            const NativeLayerMetrics *metrics = &snapshot.layers[phase][layer];
            if (metrics->calls == 0) continue;
            if (!first_layer) fputc(',', stderr);
            first_layer = false;
            fprintf(stderr,
                "{\"layer\":%u,\"calls\":%" PRIu64
                ",\"unique_experts\":%" PRIu64
                ",\"hot_hits\":%" PRIu64
                ",\"prefetch_hits\":%" PRIu64
                ",\"batch_pool_hits\":%" PRIu64
                ",\"cold_misses\":%" PRIu64
                ",\"pread_calls\":%" PRIu64
                ",\"bytes_requested\":%" PRIu64
                ",\"bytes_read\":%" PRIu64
                ",\"short_reads\":%" PRIu64
                ",\"io_wait_ns\":%" PRIu64
                ",\"bytes_avoided\":%" PRIu64
                ",\"cache_admissions\":%" PRIu64
                ",\"cache_evictions\":%" PRIu64
                ",\"moe_dispatches\":%" PRIu64
                ",\"moe_command_buffers\":%" PRIu64
                ",\"threadgroups\":%" PRIu64
                ",\"scheduled_lanes\":%" PRIu64
                ",\"active_lanes\":%" PRIu64
                ",\"max_threads_per_threadgroup\":%u}",
                metrics->layer,
                metrics->calls,
                metrics->unique_experts,
                metrics->hot_hits,
                metrics->prefetch_hits,
                metrics->batch_pool_hits,
                metrics->cold_misses,
                metrics->pread_calls,
                metrics->bytes_requested,
                metrics->bytes_read,
                metrics->short_reads,
                metrics->io_wait_ns,
                metrics->bytes_avoided,
                metrics->cache_admissions,
                metrics->cache_evictions,
                metrics->moe_dispatches,
                metrics->moe_command_buffers,
                metrics->threadgroups,
                metrics->scheduled_lanes,
                metrics->active_lanes,
                metrics->max_threads_per_threadgroup);
        }
        fputs("]}", stderr);
    }
    fputs("]}\n", stderr);
    funlockfile(stderr);
}

static inline uint64_t capture_hash_bytes(
    uint64_t hash,
    const void *data,
    size_t byte_count) {
    if (byte_count == 0) return hash;
    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t index = 0; index < byte_count; ++index) {
        hash ^= bytes ? bytes[index] : 0;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int moe_infer_capture_state_snapshot(
    const MoEInferEngine *eng,
    int layer,
    int logical_kv_length,
    MoeInferStateSnapshot *snapshot) {
    const uint64_t offset_basis = UINT64_C(1469598103934665603);
    if (!snapshot) return NATIVE_BATCH_INVALID_ARGUMENT;
    *snapshot = (MoeInferStateSnapshot){
        .kv_length = logical_kv_length,
        .kv_state_digest = offset_basis,
        .compressor_state_digest = offset_basis,
        .indexer_state_digest = offset_basis,
    };
    if (!eng || layer < 0 || layer >= N_LAYERS ||
        logical_kv_length < 0 || logical_kv_length > MAX_SEQ_LEN) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }

    const KVCache *kv = &eng->kv_cache[layer];
    if (logical_kv_length > 0) {
        if (!kv->kv) return NATIVE_BATCH_INVALID_STATE;
        snapshot->kv_state_digest = capture_hash_bytes(
            snapshot->kv_state_digest,
            kv->kv,
            (size_t)logical_kv_length * KV_LORA_RANK * sizeof(uint16_t));
    }

    const CompressorState *state = &eng->comp_state[layer];
    snapshot->compressor_length = state->n_comp;
    snapshot->indexer_length = state->n_idx_comp;
    const uint32_t ratio = eng->compress_ratio[layer];
    const uint32_t out_dim = ratio == 4 ? CSA_OUT_DIM :
        (ratio == 128 ? HCA_OUT_DIM : 0);
    if (ratio > 0 && out_dim > 0) {
        const size_t rolling_count = (size_t)2 * ratio * out_dim;
        snapshot->compressor_state_digest = capture_hash_bytes(
            snapshot->compressor_state_digest,
            state->state_kv, rolling_count * sizeof(float));
        snapshot->compressor_state_digest = capture_hash_bytes(
            snapshot->compressor_state_digest,
            state->state_score, rolling_count * sizeof(float));
        snapshot->compressor_state_digest = capture_hash_bytes(
            snapshot->compressor_state_digest,
            state->comp_kv,
            (size_t)state->n_comp * COMP_HEAD_DIM * sizeof(float));

        if (ratio == 4) {
            const size_t indexer_rolling_count = (size_t)8 * 256;
            snapshot->indexer_state_digest = capture_hash_bytes(
                snapshot->indexer_state_digest,
                state->idx_state_kv,
                indexer_rolling_count * sizeof(float));
            snapshot->indexer_state_digest = capture_hash_bytes(
                snapshot->indexer_state_digest,
                state->idx_state_score,
                indexer_rolling_count * sizeof(float));
            snapshot->indexer_state_digest = capture_hash_bytes(
                snapshot->indexer_state_digest,
                state->idx_comp_kv,
                (size_t)state->n_idx_comp * IDX_HEAD_DIM * sizeof(float));
        }
    }
    return NATIVE_BATCH_OK;
}

static inline void capture_state_digests(
    MoEInferEngine *eng,
    int layer,
    uint64_t *kv_digest,
    uint64_t *compressor_digest,
    uint64_t *indexer_digest) {
    MoeInferStateSnapshot snapshot;
    (void)moe_infer_capture_state_snapshot(
        eng,
        layer,
        (layer >= 0 && layer < N_LAYERS) ? eng->kv_cache[layer].len : 0,
        &snapshot);
    *kv_digest = snapshot.kv_state_digest;
    *compressor_digest = snapshot.compressor_state_digest;
    *indexer_digest = snapshot.indexer_state_digest;
}

static inline void emit_debug_capture_with_snapshot(
    MoEInferEngine *eng,
    MoeInferCaptureCheckpoint checkpoint,
    MoeInferCaptureDType dtype,
    int layer,
    int position,
    const float *values,
    size_t value_count,
    const int *expert_ids,
    size_t expert_count,
    const MoeInferStateSnapshot *snapshot) {
    MoeInferDebugCaptureFn callback = eng->debug_capture;
    if (!callback || !snapshot) return;

    const MoeInferDebugCaptureRecord record = {
        .path = eng->in_batch > 0 ? MOE_INFER_CAPTURE_BATCH : MOE_INFER_CAPTURE_SCALAR,
        .checkpoint = checkpoint,
        .dtype = dtype,
        .layer = layer,
        .position = position,
        .kv_length = snapshot->kv_length,
        .compressor_length = snapshot->compressor_length,
        .indexer_length = snapshot->indexer_length,
        .kv_state_digest = snapshot->kv_state_digest,
        .compressor_state_digest = snapshot->compressor_state_digest,
        .indexer_state_digest = snapshot->indexer_state_digest,
        .values = values,
        .value_count = value_count,
        .expert_ids = expert_ids,
        .expert_count = expert_count,
    };
    callback(eng->debug_capture_context, &record);
}

static inline void emit_debug_capture(
    MoEInferEngine *eng,
    MoeInferCaptureCheckpoint checkpoint,
    MoeInferCaptureDType dtype,
    int layer,
    int position,
    const float *values,
    size_t value_count,
    const int *expert_ids,
    size_t expert_count) {
    if (!eng->debug_capture) return;
    MoeInferStateSnapshot snapshot;
    if (moe_infer_capture_state_snapshot(
            eng, layer,
            (layer >= 0 && layer < N_LAYERS) ? eng->kv_cache[layer].len : 0,
            &snapshot) != NATIVE_BATCH_OK) {
        return;
    }
    emit_debug_capture_with_snapshot(
        eng, checkpoint, dtype, layer, position, values, value_count,
        expert_ids, expert_count, &snapshot);
}

typedef struct {
    MoeInferDebugCaptureRecord record;
    float *values;
    int *expert_ids;
} MoeInferCaptureJournalEntry;

typedef struct {
    MoeInferCaptureJournalEntry *entries;
    size_t count;
    size_t capacity;
    MoeInferDebugCaptureFn target;
    void *target_context;
    bool failed;
} MoeInferCaptureJournal;

static void capture_journal_record(
    void *context,
    const MoeInferDebugCaptureRecord *record) {
    MoeInferCaptureJournal *journal = (MoeInferCaptureJournal *)context;
    if (!journal || !record || journal->failed) return;
    if (journal->count >= journal->capacity ||
        (record->value_count > 0u && !record->values) ||
        (record->expert_count > 0u && !record->expert_ids) ||
        record->value_count > SIZE_MAX / sizeof(float) ||
        record->expert_count > SIZE_MAX / sizeof(int)) {
        journal->failed = true;
        return;
    }

    float *values = NULL;
    int *expert_ids = NULL;
    if (record->value_count > 0u) {
        values = (float *)malloc(record->value_count * sizeof(float));
        if (!values) {
            journal->failed = true;
            return;
        }
        memcpy(values, record->values, record->value_count * sizeof(float));
    }
    if (record->expert_count > 0u) {
        expert_ids = (int *)malloc(record->expert_count * sizeof(int));
        if (!expert_ids) {
            free(values);
            journal->failed = true;
            return;
        }
        memcpy(expert_ids, record->expert_ids,
               record->expert_count * sizeof(int));
    }

    MoeInferCaptureJournalEntry *entry = &journal->entries[journal->count++];
    *entry = (MoeInferCaptureJournalEntry){
        .record = *record,
        .values = values,
        .expert_ids = expert_ids,
    };
    entry->record.values = values;
    entry->record.expert_ids = expert_ids;
}

static int capture_journal_init(
    MoeInferCaptureJournal *journal,
    size_t capacity,
    MoeInferDebugCaptureFn target,
    void *target_context) {
    if (!journal || !target || capacity == 0u ||
        capacity > SIZE_MAX / sizeof(MoeInferCaptureJournalEntry)) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    memset(journal, 0, sizeof(*journal));
    journal->entries = (MoeInferCaptureJournalEntry *)calloc(
        capacity, sizeof(MoeInferCaptureJournalEntry));
    if (!journal->entries) return NATIVE_BATCH_CAPACITY_EXCEEDED;
    journal->capacity = capacity;
    journal->target = target;
    journal->target_context = target_context;
    return NATIVE_BATCH_OK;
}

static void capture_journal_publish(const MoeInferCaptureJournal *journal) {
    if (!journal || journal->failed || !journal->target) return;
    for (size_t index = 0u; index < journal->count; ++index) {
        journal->target(journal->target_context,
                        &journal->entries[index].record);
    }
}

static void capture_journal_deinit(MoeInferCaptureJournal *journal) {
    if (!journal) return;
    for (size_t index = 0u; index < journal->count; ++index) {
        free(journal->entries[index].values);
        free(journal->entries[index].expert_ids);
    }
    free(journal->entries);
    memset(journal, 0, sizeof(*journal));
}

// ============================================================================
// Metal setup
// ============================================================================

static void release_persistent_routing_buffers(MoEInferEngine *eng) {
    if (!eng) return;
#define RELEASE_ROUTING_BUFFER(field) do { \
    if (eng->field) { \
        [(id<MTLBuffer>)eng->field release]; \
        eng->field = NULL; \
    } \
} while (0)
    RELEASE_ROUTING_BUFFER(buf_gpu_route_selected);
    RELEASE_ROUTING_BUFFER(buf_gpu_route_weights);
    RELEASE_ROUTING_BUFFER(buf_batch_routing_scores);
    RELEASE_ROUTING_BUFFER(buf_batch_route_bias);
    RELEASE_ROUTING_BUFFER(buf_cached_flags);
#undef RELEASE_ROUTING_BUFFER
}

static void release_wave4_bucket_pipelines(MoEInferEngine *eng) {
    if (!eng) return;
#define RELEASE_BUCKET_PIPELINE(field) do { \
    if (eng->field) { \
        [(id<MTLComputePipelineState>)eng->field release]; \
        eng->field = NULL; \
    } \
} while (0)
    RELEASE_BUCKET_PIPELINE(pipe_bucket_gather_f32);
    RELEASE_BUCKET_PIPELINE(pipe_bucket_mxfp4_gate_up);
    RELEASE_BUCKET_PIPELINE(pipe_bucket_mxfp4_down);
    RELEASE_BUCKET_PIPELINE(pipe_bucket6_mxfp4_gate_up);
    RELEASE_BUCKET_PIPELINE(pipe_bucket6_mxfp4_down);
    RELEASE_BUCKET_PIPELINE(pipe_bucket_combine_f32);
#undef RELEASE_BUCKET_PIPELINE
}

static void release_shared_expert_weight_buffers(MoEInferEngine *eng) {
    if (!eng) return;
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        for (int index = 0; index < 9; ++index) {
            if (eng->buf_shared_w[layer][index]) {
                [(id<MTLBuffer>)eng->buf_shared_w[layer][index] release];
                eng->buf_shared_w[layer][index] = NULL;
            }
        }
    }
}

static int init_metal(MoEInferEngine *eng, const char *kernel_src, unsigned long kernel_src_len) {
    setvbuf(stderr, NULL, _IONBF, 0); // unbuffered so crash diagnostics are not lost
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "Metal: no device\n"); return -1; }
    eng->device = (void *)dev;
    eng->queue = (void *)[dev newCommandQueue];

    NSString *src_str = [[NSString alloc]
        initWithBytes:kernel_src
        length:kernel_src_len
        encoding:NSUTF8StringEncoding];

    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src_str options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "Metal compile: %s\n", [[err localizedDescription] UTF8String]);
        return -1;
    }

    id<MTLDevice> d = (id<MTLDevice>)dev;
    eng->pipe_gate_up_swiglu = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_gate_up_swiglu"] error:&err]);
    eng->pipe_gate_up_swiglu_v2 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_gate_up_swiglu_v2"] error:&err]);
    eng->pipe_gate_up_swiglu_v4 = NULL;  // not dispatched
    eng->pipe_gather_gate_up_v4  = NULL;  // not dispatched
    eng->pipe_gate_up_swiglu_v2_affine = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_gate_up_swiglu_v2_affine"] error:&err]);
    eng->pipe_dequant_matvec_4bit_affine = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_4bit_affine"] error:&err]);
    eng->pipe_dequant_matvec = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_4bit"] error:&err]);
    eng->pipe_moe_combine    = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_combine"] error:&err]);
    eng->pipe_fused_6expert_gate_up = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_6expert_gate_up_swiglu"] error:&err]);
    eng->pipe_fused_6expert_down    = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_6expert_down"] error:&err]);
    eng->pipe_gather_gate_up = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather_gate_up_swiglu"] error:&err]);
    eng->pipe_gather_down    = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather_down"] error:&err]);
    eng->pipe_gather6_gate_up = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_gate_up_swiglu"] error:&err]);
    eng->pipe_gather6_down    = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_down"] error:&err]);
    eng->pipe_gather6_gate_up_2b = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_gate_up_swiglu_2b"] error:&err]);
    eng->pipe_gather6_down_2b    = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_down_2b"] error:&err]);
    if (!eng->pipe_gather6_gate_up_2b || !eng->pipe_gather6_down_2b)
        fprintf(stderr, "Metal engine: WARNING int2_gs64 gather6 pipelines missing: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
    eng->pipe_int8_gate_up_swiglu = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_gate_up_swiglu_int8_e8m0"] error:&err]);
    eng->pipe_int8_dequant_matvec = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_int8_e8m0"] error:&err]);
    eng->pipe_rms_norm_sum_sq = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_sum_sq"] error:&err]);
    eng->pipe_rms_norm_apply = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_apply"] error:&err]);
    eng->pipe_matvec = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec_f32"] error:&err]);
    // S7: MLA attention pipelines
    eng->pipe_dequant_matvec_affine = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine"] error:&err]);
    eng->pipe_dequant_matvec_affine_v2 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_v2"] error:&err]);
    eng->pipe_rms_norm_rows = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows"] error:&err]);
    eng->pipe_rope_tail = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rope_tail_interleaved"] error:&err]);
    eng->pipe_mla_sdpa = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_decode"] error:&err]);
    eng->pipe_mla_sdpa_f16 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_decode_f16"] error:&err]);
    // F16 precision chain (validated correct, see docs/analysis/dsv4-first-class-support-plan.md)
    eng->pipe_dequant_matvec_affine_f16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_f16out"] error:&err]);
    eng->pipe_rms_norm_rows_f16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows_f16out"] error:&err]);
    eng->pipe_dequant_matvec_affine_f16in_f16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_f16in_f16out"] error:&err]);
    eng->pipe_rms_norm_rows_f16in_f16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows_f16in_f16out"] error:&err]);
    eng->pipe_rope_tail_f16 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rope_tail_interleaved_f16"] error:&err]);
    eng->pipe_matvec_f32_f16in = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec_f32_f16in"] error:&err]);
    eng->pipe_mla_sdpa_f16in_f16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_decode_f16in_f16out"] error:&err]);
    // BF16 precision chain — matches MLX training precision
    eng->pipe_dequant_matvec_affine_bf16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16out"] error:&err]);
    eng->pipe_rms_norm_rows_bf16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows_bf16out"] error:&err]);
    eng->pipe_dequant_matvec_affine_bf16in_bf16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_bf16out"] error:&err]);
    {
        NSError *ev2err = nil;
        id<MTLFunction> fv2 = [lib newFunctionWithName:@"dequant_matvec_affine_bf16in_bf16out_v2"];
        if (fv2) {
            id<MTLComputePipelineState> pv2 = [d newComputePipelineStateWithFunction:fv2 error:&ev2err];
            eng->pipe_dequant_matvec_affine_bf16in_bf16out_v2 = pv2 ? (void *)pv2 : NULL;
            if (!pv2) fprintf(stderr, "Metal: dequant_matvec_affine_bf16in_bf16out_v2 failed: %s\n",
                              [[ev2err localizedDescription] UTF8String]);
            else fprintf(stderr, "Metal: dequant_matvec_affine_bf16in_bf16out_v2 compiled OK\n");
        } else {
            fprintf(stderr, "Metal: dequant_matvec_affine_bf16in_bf16out_v2 not found\n");
            eng->pipe_dequant_matvec_affine_bf16in_bf16out_v2 = NULL;
        }
    }
    eng->pipe_rms_norm_rows_bf16in_bf16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows_bf16in_bf16out"] error:&err]);
    eng->pipe_rope_tail_bf16 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rope_tail_interleaved_bf16"] error:&err]);
    eng->pipe_matvec_f32_bf16in = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec_f32_bf16in"] error:&err]);
    {
        NSError *rerr2 = nil;
        id<MTLFunction> fn2 = [lib newFunctionWithName:@"matvec_f32_bf16in_simd"];
        if (fn2) {
            id<MTLComputePipelineState> ps2 = [d newComputePipelineStateWithFunction:fn2 error:&rerr2];
            eng->pipe_matvec_f32_bf16in_simd = ps2 ? (void *)ps2 : NULL;
            if (!ps2) fprintf(stderr, "Metal: matvec_f32_bf16in_simd compile failed: %s\n",
                             [[rerr2 localizedDescription] UTF8String]);
            else fprintf(stderr, "Metal: matvec_f32_bf16in_simd compiled OK\n");
        } else {
            fprintf(stderr, "Metal: matvec_f32_bf16in_simd not found in library — fallback to naive\n");
            eng->pipe_matvec_f32_bf16in_simd = NULL;
        }
    }
    eng->pipe_matvec_q8_0_f32    = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec_q8_0_f32"]    error:&err]);
    eng->pipe_mla_sdpa_bfloat = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_decode_bfloat"] error:&err]);
    eng->pipe_dequant_matvec_affine_bf16in_f32out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_f32out"] error:&err]);
    eng->pipe_dequant_matvec_affine_bf16in_f32out_grp8 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_f32out_grp8"] error:&err]);
    eng->pipe_mla_sdpa_prefill_bfloat = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_prefill_bfloat"] error:&err]);
    eng->pipe_mla_sdpa_batch_f16_kv = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_batch_f16_kv"] error:&err]);
    eng->pipe_bf16_to_f16_row = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bf16_to_f16_row"] error:&err]);
    eng->pipe_limited_swiglu  = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"limited_swiglu"] error:&err]);
    eng->pipe_f32_to_bf16_vec = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"f32_to_bf16_vec"] error:&err]);
    eng->pipe_bf16_to_f32_vec = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bf16_to_f32_vec"] error:&err]);
    eng->pipe_f32_to_bf16_trunc_vec = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"f32_to_bf16_trunc_vec"] error:&err]);
    eng->pipe_bucket_gather_f32 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bucket_gather_f32"] error:&err]);
    eng->pipe_bucket_mxfp4_gate_up = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bucket_mxfp4_gate_up_swiglu"] error:&err]);
    eng->pipe_bucket_mxfp4_down = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bucket_mxfp4_down"] error:&err]);
    eng->pipe_bucket6_mxfp4_gate_up = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bucket6_mxfp4_gate_up_swiglu"] error:&err]);
    eng->pipe_bucket6_mxfp4_down = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bucket6_mxfp4_down"] error:&err]);
    eng->pipe_bucket_combine_f32 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bucket_combine_f32"] error:&err]);
    if (!eng->pipe_bucket_gather_f32 || !eng->pipe_bucket_mxfp4_gate_up ||
        !eng->pipe_bucket_mxfp4_down || !eng->pipe_bucket6_mxfp4_gate_up ||
        !eng->pipe_bucket6_mxfp4_down || !eng->pipe_bucket_combine_f32) {
        fprintf(stderr, "Metal: Wave-4 bucket pipeline failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
        return -1;
    }
    {
        NSError *rerr = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:@"moe_route_gpu"];
        if (fn) {
            id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:fn error:&rerr];
            eng->pipe_moe_route_gpu = ps ? (void *)ps : NULL;
            if (!ps) fprintf(stderr, "Metal: moe_route_gpu compile failed: %s\n",
                             [[rerr localizedDescription] UTF8String]);
        } else {
            fprintf(stderr, "Metal: moe_route_gpu function not found in library\n");
            eng->pipe_moe_route_gpu = NULL;
        }
    }
    // mHC GPU kernels (f16/bfloat)
    eng->pipe_mhc_pre_f16   = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mhc_pre_f16"] error:&err]);
    eng->pipe_mhc_post_f16  = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mhc_post_f16"] error:&err]);
    eng->pipe_mhc_pre_bfloat  = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mhc_pre_gpu"] error:&err]);
    eng->pipe_mhc_post_bfloat = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mhc_post_bfloat"] error:&err]);
    eng->pipe_mhc_post_ffn_expand4 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mhc_post_ffn_expand4"] error:&err]);
    eng->pipe_mhc_pre_split_weighted_sum_norm = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mhc_pre_split_weighted_sum_norm"] error:&err]);
    if (!eng->pipe_gate_up_swiglu || !eng->pipe_dequant_matvec || !eng->pipe_moe_combine) {
        fprintf(stderr, "Metal: pipeline state failed\n");
        return -1;
    }

    // Allocate Metal buffers
    int buf_size = DIM * sizeof(float);
    int mid_size = INTERMEDIATE * sizeof(float);
    eng->buf_hidden    = (void *)[d newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    eng->buf_h_mid     = (void *)[d newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    eng->buf_normed    = (void *)[d newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    eng->buf_comp_normed = (void *)[d newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    eng->buf_attn_out  = (void *)[d newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    eng->buf_routing_scores = (void *)[d newBufferWithLength:N_EXPERTS * sizeof(float) options:MTLResourceStorageModeShared];
    eng->buf_norm_sum_sq    = (void *)[d newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    for (int k = 0; k < N_ACTIVE; k++) {
        eng->buf_expert_mid[k] = (void *)[d newBufferWithLength:mid_size options:MTLResourceStorageModeShared];
        eng->buf_expert_out[k] = (void *)[d newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    }
    eng->buf_expert_contiguous = (void *)[d newBufferWithLength:(size_t)N_ACTIVE*DIM*sizeof(float) options:MTLResourceStorageModeShared];

    // Gather mode output buffers
    eng->buf_gather_mid = (void *)[d newBufferWithLength:(size_t)N_ACTIVE*INTERMEDIATE*sizeof(float) options:MTLResourceStorageModeShared];
    eng->buf_gather_out = (void *)[d newBufferWithLength:(size_t)N_ACTIVE*DIM*sizeof(float) options:MTLResourceStorageModeShared];
    eng->buf_gather_expert_ids = (void *)[d newBufferWithLength:N_ACTIVE*sizeof(uint32_t) options:MTLResourceStorageModeShared];

    // Persistent scratch buffers for mhc_pre / mhc_post — eliminates per-call allocation
    const int MIX3_size = MHC_MULT * (MHC_MULT + 2);  // 24
    eng->buf_mhc_res_in        = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(float)         options:MTLResourceStorageModeShared];
    eng->buf_mhc_attn_in       = (void *)[d newBufferWithLength:DIM * sizeof(float)                    options:MTLResourceStorageModeShared];
    eng->buf_mhc_attn_norm_bf16= (void *)[d newBufferWithLength:DIM * sizeof(uint16_t)                 options:MTLResourceStorageModeShared];
    eng->buf_mhc_post_weights  = (void *)[d newBufferWithLength:MHC_MULT * sizeof(float)               options:MTLResourceStorageModeShared];
    eng->buf_mhc_comb_weights  = (void *)[d newBufferWithLength:MHC_MULT * MHC_MULT * sizeof(float)    options:MTLResourceStorageModeShared];
    eng->buf_mhc_res_out       = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(uint16_t)      options:MTLResourceStorageModeShared];
    eng->buf_mhc_ffn_in        = (void *)[d newBufferWithLength:DIM * sizeof(float)                    options:MTLResourceStorageModeShared];
    eng->buf_mhc_ffn_norm_bf16 = (void *)[d newBufferWithLength:DIM * sizeof(uint16_t)                 options:MTLResourceStorageModeShared];
    eng->buf_mhc_ffn_res_in    = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(float)         options:MTLResourceStorageModeShared];
    eng->buf_mhc_attn_out_bf16 = (void *)[d newBufferWithLength:DIM * sizeof(uint16_t)                 options:MTLResourceStorageModeShared];
    eng->buf_mhc_res_bf16_in   = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(uint16_t)      options:MTLResourceStorageModeShared];
    eng->buf_mhc_post_res_out  = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(uint16_t)      options:MTLResourceStorageModeShared];
    eng->buf_mhc_ffn_post_out  = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(uint16_t)      options:MTLResourceStorageModeShared];
    eng->buf_ffn_out_f32        = (void *)[d newBufferWithLength:DIM * sizeof(float)                        options:MTLResourceStorageModeShared];
    {   const float one = 1.0f;   // unit weight for the K=1 moe_combine GPU add
        eng->buf_combine_one      = (void *)[d newBufferWithBytes:&one length:sizeof(float)                 options:MTLResourceStorageModeShared];
    }
    (void)MIX3_size;

    // GPU-resident residual buffer (Path B)
    eng->buf_residual_gpu = (void *)[d newBufferWithLength:(size_t)MHC_MULT * DIM * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    // Persistent batch routing scratch. Scalar routing uses row zero; Task 3.2
    // dispatches up to MAX_SEQ_LEN rows without allocating in a layer loop.
    eng->buf_gpu_route_selected = (void *)[d newBufferWithLength:
        (size_t)MAX_SEQ_LEN * N_ACTIVE * sizeof(int32_t)
        options:MTLResourceStorageModeShared];
    eng->buf_gpu_route_weights = (void *)[d newBufferWithLength:
        (size_t)MAX_SEQ_LEN * N_ACTIVE * sizeof(float)
        options:MTLResourceStorageModeShared];
    eng->buf_batch_routing_scores = (void *)[d newBufferWithLength:
        (size_t)MAX_SEQ_LEN * N_EXPERTS * sizeof(float)
        options:MTLResourceStorageModeShared];
    eng->buf_batch_route_bias = (void *)[d newBufferWithLength:
        N_EXPERTS * sizeof(float) options:MTLResourceStorageModeShared];
    eng->buf_cached_flags = (void *)[d newBufferWithLength:
        N_EXPERTS * sizeof(uint8_t) options:MTLResourceStorageModeShared];
    if (!eng->buf_gpu_route_selected || !eng->buf_gpu_route_weights ||
        !eng->buf_batch_routing_scores || !eng->buf_batch_route_bias ||
        !eng->buf_cached_flags) {
        fprintf(stderr, "Metal: persistent batch routing buffer allocation failed\n");
        release_persistent_routing_buffers(eng);
        return -1;
    }
    // Zero shared routing controls before any scalar or batch dispatch.
    memset([(id<MTLBuffer>)eng->buf_batch_route_bias contents], 0,
        N_EXPERTS * sizeof(float));
    memset([(id<MTLBuffer>)eng->buf_cached_flags contents], 0,
        N_EXPERTS * sizeof(uint8_t));

    // 2MB-aligned expert I/O buffers
    for (int k = 0; k < N_ACTIVE; k++) {
        posix_memalign((void**)&eng->expert_buf[k], 2*1024*1024, AFFINE_EXPERT_SIZE);
        posix_memalign((void**)&eng->expert_buf_pred[k], 2*1024*1024, AFFINE_EXPERT_SIZE);
        posix_memalign((void**)&eng->expert_buf_pred2[k], 2*1024*1024, AFFINE_EXPERT_SIZE);
    }

    // Initialize expert GPU buffer cache to NULL (populated after SMELT warmup)
    memset(eng->expert_gpu_buf, 0, sizeof(eng->expert_gpu_buf));

    // Initialize gather mode buffers to NULL
    memset(eng->buf_gather_gate_W, 0, sizeof(eng->buf_gather_gate_W));
    memset(eng->buf_gather_gate_s, 0, sizeof(eng->buf_gather_gate_s));
    memset(eng->buf_gather_up_W,   0, sizeof(eng->buf_gather_up_W));
    memset(eng->buf_gather_up_s,   0, sizeof(eng->buf_gather_up_s));
    memset(eng->buf_gather_down_W, 0, sizeof(eng->buf_gather_down_W));
    memset(eng->buf_gather_down_s, 0, sizeof(eng->buf_gather_down_s));
    eng->gather_mode = false;
    eng->use_affine_experts = (getenv("DMLX_AFFINE_EXPERTS") != NULL);
    eng->active_expert_size = eng->use_affine_experts ? AFFINE_EXPERT_SIZE : EXPERT_SIZE;
    if (eng->use_affine_experts)
        fprintf(stderr, "Metal engine: affine expert format (gs=64, bf16 scale+bias)\n");
    // Initialize all pool_pos remapping slots to -1 (not in pool)
    for (int l = 0; l < N_LAYERS; l++)
        for (int e = 0; e < N_EXPERTS; e++)
            eng->smelt_pool_pos[l][e] = -1;

    fprintf(stderr, "Metal engine: initialized\n");
    return 0;
}

// ============================================================================
// Expert I/O — parallel pread with persistent threads (flash-moe pattern)
// ============================================================================

#define NUM_IO_THREADS 6

typedef struct {
#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
    DmlxExactQ4TestClockFn callback;
    void *context;
#else
    uint8_t reserved;
#endif
} ExactQ4DeferredClock;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t work_ready, work_done;
    int fd[NUM_IO_THREADS];       // original fd per task (set by dispatcher)
    int claimed_fd[NUM_IO_THREADS]; // fd saved when task is claimed (to preserve original)
    void *buf[NUM_IO_THREADS];
    size_t size;
    off_t offset[NUM_IO_THREADS];
    MoEInferEngine *exact_q4_engine[NUM_IO_THREADS];
    uint32_t exact_q4_ordinal[NUM_IO_THREADS];
    uint8_t exact_q4_owned[NUM_IO_THREADS];
    uint8_t exact_q4_deferred[NUM_IO_THREADS];
    ExactQ4DeferredClock exact_q4_deferred_clock[NUM_IO_THREADS];
    int task_generation[NUM_IO_THREADS];
    uint64_t submit_ns[NUM_IO_THREADS];
    uint64_t worker_start_ns[NUM_IO_THREADS];
    uint64_t complete_ns[NUM_IO_THREADS];
    int num_tasks;
    int tasks_done;
    int done[NUM_IO_THREADS];     // per-task completion flags (for wait_task)
    ssize_t result[NUM_IO_THREADS]; // exact pread result for short-read/error checks
    int error_code[NUM_IO_THREADS];
    int generation;
    int in_flight_generation;
    bool shutdown;
#ifdef DMLX_EXACT_Q4_IO_TESTING
    NativeExpertPreadFn test_reader;
    void *test_reader_context;
    bool dependency_wait_entered;
#endif
    pthread_t threads[NUM_IO_THREADS];
} IOPool;

static uint64_t exact_q4_clock_now(MoEInferEngine *eng);
static uint64_t exact_q4_deferred_clock_now(
    const ExactQ4DeferredClock *clock);

static void *io_worker(void *arg) {
    IOPool *pool = (IOPool *)arg;
    int last_gen = -1;
    while (1) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->generation == last_gen && !pool->shutdown) {
            pthread_cond_wait(&pool->work_ready, &pool->mutex);
        }
        if (pool->shutdown) { pthread_mutex_unlock(&pool->mutex); break; }
        last_gen = pool->generation;
        int tid = -1;
        // Find and claim an unclaimed task atomically while mutex is held
        for (int i = 0; i < pool->num_tasks; i++) {
            if (pool->fd[i] >= 0) {
                tid = i;
                pool->claimed_fd[i] = pool->fd[i]; // save fd before claiming
                pool->fd[i] = -1;  // mark as claimed immediately
                break;
            }
        }
        if (tid < 0) {
            pthread_mutex_unlock(&pool->mutex);
            continue;
        }
        MoEInferEngine *exact_q4_engine = pool->exact_q4_engine[tid];
        const uint32_t exact_q4_ordinal = pool->exact_q4_ordinal[tid];
        const bool exact_q4_owned = pool->exact_q4_owned[tid] != 0u;
        const bool exact_q4_deferred = pool->exact_q4_deferred[tid] != 0u;
        const ExactQ4DeferredClock exact_q4_deferred_clock =
            pool->exact_q4_deferred_clock[tid];
        pthread_mutex_unlock(&pool->mutex);

        bool exact_q4_started = false;
        uint64_t worker_start_ns = 0u;
        if (exact_q4_deferred) {
            worker_start_ns = exact_q4_deferred_clock_now(
                &exact_q4_deferred_clock);
        } else if (exact_q4_owned) {
            worker_start_ns = exact_q4_clock_now(exact_q4_engine);
        }
        if (exact_q4_owned) {
            exact_q4_started =
                moe_infer_exact_q4_publish_read_worker_start(
                    exact_q4_engine, exact_q4_ordinal, worker_start_ns) ==
                DMLX_EXACT_Q4_PRODUCER_OK;
            if (!exact_q4_started) {
                (void)moe_infer_exact_q4_abort_read(
                    exact_q4_engine, exact_q4_ordinal);
            }
        }
        errno = 0;
#ifdef DMLX_EXACT_Q4_IO_TESTING
        ssize_t n = pool->test_reader
            ? (ssize_t)pool->test_reader(
                pool->test_reader_context, pool->claimed_fd[tid],
                pool->buf[tid], pool->size, (uint64_t)pool->offset[tid])
            : pread(pool->claimed_fd[tid], pool->buf[tid], pool->size,
                    pool->offset[tid]);
#else
        ssize_t n = pread(pool->claimed_fd[tid], pool->buf[tid], pool->size,
                          pool->offset[tid]);
#endif
        const int read_error = n < 0 ? (errno != 0 ? errno : EIO) : 0;
        uint64_t complete_ns = 0u;
        if (exact_q4_deferred) {
            complete_ns = exact_q4_deferred_clock_now(
                &exact_q4_deferred_clock);
        } else if (exact_q4_started) {
            complete_ns = exact_q4_clock_now(exact_q4_engine);
        }
        if (exact_q4_started) {
            const uint64_t bytes = n > 0 ? (uint64_t)n : 0u;
            const int publish_status = moe_infer_exact_q4_publish_read_completion(
                exact_q4_engine, exact_q4_ordinal, complete_ns,
                bytes <= pool->size ? bytes : pool->size, read_error);
            if (publish_status != DMLX_EXACT_Q4_PRODUCER_OK) {
                (void)moe_infer_exact_q4_abort_read(
                    exact_q4_engine, exact_q4_ordinal);
            }
        }

        pthread_mutex_lock(&pool->mutex);
        pool->worker_start_ns[tid] = worker_start_ns;
        pool->complete_ns[tid] = complete_ns;
        pool->result[tid] = n;
        pool->error_code[tid] = read_error;
        pool->done[tid] = 1;
        pool->tasks_done++;
        if (pool->tasks_done == pool->num_tasks) {
            pool->in_flight_generation = 0;
        }
        // Signal on EVERY completion so io_pool_wait_task waiters wake promptly;
        // only the dispatcher thread ever waits on work_done.
        pthread_cond_broadcast(&pool->work_done);
        pthread_mutex_unlock(&pool->mutex);
    }
    return NULL;
}

static void io_pool_init(IOPool *pool) {
    memset(pool, 0, sizeof(*pool));
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->work_ready, NULL);
    pthread_cond_init(&pool->work_done, NULL);
    pool->num_tasks = 0;
    pool->tasks_done = 0;
    pool->generation = 0;
    pool->shutdown = false;
    for (int i = 0; i < NUM_IO_THREADS; i++) {
        pthread_create(&pool->threads[i], NULL, io_worker, pool);
    }
}

static void io_pool_wait(IOPool *pool);

// Forward declaration for io_pool_dispatch (defined below)
static int io_pool_dispatch(IOPool *pool, MoEInferEngine *engine,
                            int layer_fd, int *expert_ids,
                            const int *route_ranks, int layer, int K,
                            uint8_t *buffers[6], size_t expert_sz,
                            uint64_t *bytes_read, uint64_t *short_reads,
                            DmlxExactQ4PendingLayer *transaction);

// ============================================================================
// Expert memory cache — preload experts from SSD into RAM to eliminate I/O
// ============================================================================

int moe_infer_preload_experts(MoEInferEngine *eng, int expert_cache_mb) {
    // Calculate how many experts per layer fit in the budget
    // Each expert is EXPERT_SIZE bytes. Total budget across all layers:
    size_t total_bytes = (size_t)expert_cache_mb * 1024 * 1024;
    int n_per_layer;
    if (expert_cache_mb == 0) {
        // Preload all experts
        n_per_layer = N_EXPERTS;
        total_bytes = (size_t)N_LAYERS * N_EXPERTS * eng->active_expert_size;
    } else {
        n_per_layer = (int)(total_bytes / ((size_t)N_LAYERS * eng->active_expert_size));
        if (n_per_layer > N_EXPERTS) n_per_layer = N_EXPERTS;
        if (n_per_layer == 0) {
            fprintf(stderr, "[expert-cache] Budget %dMB too small for even 1 expert per layer (need %luMB)\n",
                expert_cache_mb, (unsigned long)((size_t)N_LAYERS * eng->active_expert_size / (1024*1024)));
            return 0;
        }
    }

    fprintf(stderr, "[expert-cache] Preloading %d/%d experts per layer (%lu MB total)...\n",
        n_per_layer, N_EXPERTS, (unsigned long)(((size_t)N_LAYERS * n_per_layer * eng->active_expert_size) / (1024*1024)));
    eng->expert_cache_n_experts = n_per_layer;

    for (int layer = 0; layer < N_LAYERS; layer++) {
        // Allocate lookup table
        eng->expert_mem_cache[layer] = (uint8_t **)calloc(N_EXPERTS, sizeof(uint8_t *));
        if (!eng->expert_mem_cache[layer]) {
            fprintf(stderr, "[expert-cache] OOM allocating lookup table for layer %d\n", layer);
            eng->expert_cache_n_experts = 0;
            return 0;
        }

        // Allocate flat pool for this layer
        size_t pool_bytes = (size_t)n_per_layer * eng->active_expert_size;
        if (posix_memalign((void**)&eng->expert_mem_pool[layer], 2*1024*1024, pool_bytes) != 0) {
            fprintf(stderr, "[expert-cache] OOM allocating %luMB pool for layer %d\n",
                (unsigned long)(pool_bytes / (1024*1024)), layer);
            eng->expert_cache_n_experts = 0;
            return 0;
        }

        // Read n_per_layer experts starting from 0 (most frequently used in practice)
        // For hash routing layers (0-2), these are exactly the needed experts
        for (int eid = 0; eid < n_per_layer; eid++) {
            uint8_t *dst = eng->expert_mem_pool[layer] + (size_t)eid * eng->active_expert_size;
            off_t offset = (off_t)eid * eng->active_expert_size;
            ssize_t n = pread(eng->packed_fd[layer], dst, eng->active_expert_size, offset);
            if (n != eng->active_expert_size) {
                fprintf(stderr, "[expert-cache] pread failed: layer=%d expert=%d got=%ld\n", layer, eid, (long)n);
                eng->expert_cache_n_experts = 0;
                return 0;
            }
            eng->expert_mem_cache[layer][eid] = dst;
            eng->smelt_pool_pos[layer][eid] = eid;  // slot == eid for sequential preload
        }
    }

    fprintf(stderr, "[expert-cache] Done. %d experts/layer cached (%lu MB)\n",
        n_per_layer, (unsigned long)(((size_t)N_LAYERS * n_per_layer * eng->active_expert_size) / (1024*1024)));
    return n_per_layer;
}

// ============================================================================
// SMELT: warmup-based hot expert preloading with routing bias
// ============================================================================

void moe_infer_smelt_init(MoEInferEngine *eng, int warmup_tokens, int n_per_layer, float penalty) {
    memset(eng->routing_counts, 0, sizeof(eng->routing_counts));
    eng->smelt_warmup_tokens = warmup_tokens;
    eng->smelt_n_per_layer   = n_per_layer;
    eng->smelt_tokens_seen   = 0;
    eng->smelt_warmup_done   = false;
    eng->smelt_enabled       = true;
    eng->smelt_penalty       = penalty;
    eng->smelt_in_decode_phase = false;
    fprintf(stderr, "[smelt] Init: warmup=%d tokens, cache=%d experts/layer, penalty=%.0f\n",
        warmup_tokens, n_per_layer, penalty);
}

int moe_infer_smelt_finish_warmup(MoEInferEngine *eng) {
    if (!eng->smelt_enabled || eng->smelt_warmup_done) return eng->expert_cache_n_experts;

    const int n = eng->smelt_n_per_layer;
    if (n <= 0 || n > N_EXPERTS) {
        eng->smelt_warmup_done = true;
        return 0;
    }

    fprintf(stderr, "[smelt] Warmup complete (%d tokens). Preloading experts per layer...\n",
        eng->smelt_tokens_seen);

    // Total memory: hash layers (0-2) cache ALL 256 experts, score layers cache top-N
    size_t total_bytes = 0;
    for (int layer = 0; layer < N_LAYERS; layer++) {
        int n_this_layer = (eng->tid2eid[layer] != NULL) ? N_EXPERTS : n;
        total_bytes += (size_t)n_this_layer * eng->active_expert_size;
    }
    fprintf(stderr, "[smelt] Total preload: %.1f GB (hash layers: all 256, score layers: top-%d)\n",
        (double)total_bytes / (1024.0*1024.0*1024.0), n);

    // Allocate lookup tables if not already allocated
    for (int layer = 0; layer < N_LAYERS; layer++) {
        if (!eng->expert_mem_cache[layer]) {
            eng->expert_mem_cache[layer] = (uint8_t **)calloc(N_EXPERTS, sizeof(uint8_t *));
            if (!eng->expert_mem_cache[layer]) {
                fprintf(stderr, "[smelt] OOM: lookup table for layer %d\n", layer);
                return 0;
            }
        } else {
            memset(eng->expert_mem_cache[layer], 0, N_EXPERTS * sizeof(uint8_t *));
        }
    }

    int sorted[N_EXPERTS];
    for (int layer = 0; layer < N_LAYERS; layer++) {
        // For hash routing layers (0-2): cache ALL experts (no bias can be applied,
        // so we must ensure every expert that hash routing might select is in RAM).
        // For score-based layers (3-42): cache top-N by routing frequency.
        const bool is_hash_layer = (eng->tid2eid[layer] != NULL);
        const int n_this_layer = is_hash_layer ? N_EXPERTS : n;

        // Build sorted list: for hash layers all experts, for score layers sort by frequency
        for (int i = 0; i < N_EXPERTS; i++) sorted[i] = i;
        if (!is_hash_layer) {
            // Sort by descending routing_counts (simple insertion sort, N=256 is fine)
            for (int i = 1; i < N_EXPERTS; i++) {
                int key = sorted[i];
                uint32_t kc = eng->routing_counts[layer][key];
                int j = i - 1;
                while (j >= 0 && eng->routing_counts[layer][sorted[j]] < kc) {
                    sorted[j+1] = sorted[j]; j--;
                }
                sorted[j+1] = key;
            }
        }

        // Allocate pool for this layer
        if (eng->expert_mem_pool[layer]) {
            free(eng->expert_mem_pool[layer]);
            eng->expert_mem_pool[layer] = NULL;
        }
        size_t pool_bytes = (size_t)n_this_layer * eng->active_expert_size;
        if (posix_memalign((void**)&eng->expert_mem_pool[layer], 2*1024*1024, pool_bytes) != 0) {
            fprintf(stderr, "[smelt] OOM: pool for layer %d (%lu MB)\n", layer,
                (unsigned long)(pool_bytes / (1024*1024)));
            return 0;
        }

        // Preload experts — also record pool_pos remapping table
        // Initialize all slots to -1 (uncached)
        for (int i = 0; i < N_EXPERTS; i++) eng->smelt_pool_pos[layer][i] = -1;

        int loaded = 0;
        for (int i = 0; i < n_this_layer && i < N_EXPERTS; i++) {
            int eid = sorted[i];
            uint8_t *dst = eng->expert_mem_pool[layer] + (size_t)loaded * eng->active_expert_size;
            off_t offset = (off_t)eid * eng->active_expert_size;
            ssize_t bytes = pread(eng->packed_fd[layer], dst, eng->active_expert_size, offset);
            if (bytes == eng->active_expert_size) {
                eng->expert_mem_cache[layer][eid] = dst;
                eng->smelt_pool_pos[layer][eid] = loaded;  // record slot position
                eng->cache_lru_recency[layer][eid] = ++eng->cache_lru_clock;
                loaded++;
            } else {
                fprintf(stderr, "[smelt] pread failed layer=%d eid=%d got=%ld\n", layer, eid, (long)bytes);
            }
        }

        if (layer == 0) {
            fprintf(stderr, "[smelt] L0 (hash): loaded %d/%d experts\n", loaded, n_this_layer);
        } else if (layer == 5) {
            fprintf(stderr, "[smelt] L5 (score) top-%d experts: [", loaded);
            for (int i = 0; i < (loaded < 8 ? loaded : 8); i++)
                fprintf(stderr, "%d(×%u)%s", sorted[i], eng->routing_counts[layer][sorted[i]], i+1<8?",":"");
            fprintf(stderr, "...]\n");
        }
    }

    eng->expert_cache_n_experts = n;  // used as guard in expert data resolution (cache > prefetch > pread)
    eng->smelt_warmup_done = true;
    fprintf(stderr, "[smelt] Done. %.1f GB preloaded, routing bias penalty=%.0f\n",
        (double)total_bytes / (1024.0*1024.0*1024.0), eng->smelt_penalty);

    // === Create persistent GPU MTLBuffer wrappers for all cached experts ===
    // Use MTLResourceStorageModeShared (zero-copy from RAM pool, no extra memory).
    // Private mode would require double the memory (35GB×2 = 70GB > 38GB limit).
    // Shared persistent buffers still eliminate per-call newBufferWithBytesNoCopy overhead.
    {
        id<MTLDevice> d = (id<MTLDevice>)eng->device;
        int n_created = 0;
        for (int layer = 0; layer < N_LAYERS; layer++) {
            if (!eng->expert_mem_cache[layer]) continue;
            for (int eid = 0; eid < N_EXPERTS; eid++) {
                uint8_t *base = eng->expert_mem_cache[layer][eid];
                if (!base) continue;
                if (eng->use_affine_experts) {
                    eng->expert_gpu_buf[layer][eid][0] = (void *)[d newBufferWithBytesNoCopy:base+AFFINE_GATE_W_OFF length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][1] = (void *)[d newBufferWithBytesNoCopy:base+AFFINE_GATE_S_OFF length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][2] = (void *)[d newBufferWithBytesNoCopy:base+AFFINE_GATE_B_OFF length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][3] = (void *)[d newBufferWithBytesNoCopy:base+AFFINE_UP_W_OFF   length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][4] = (void *)[d newBufferWithBytesNoCopy:base+AFFINE_UP_S_OFF   length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][5] = (void *)[d newBufferWithBytesNoCopy:base+AFFINE_UP_B_OFF   length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][6] = (void *)[d newBufferWithBytesNoCopy:base+AFFINE_DOWN_W_OFF length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][7] = (void *)[d newBufferWithBytesNoCopy:base+AFFINE_DOWN_S_OFF length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][8] = (void *)[d newBufferWithBytesNoCopy:base+AFFINE_DOWN_B_OFF length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                } else {
                    eng->expert_gpu_buf[layer][eid][0] = (void *)[d newBufferWithBytesNoCopy:base+GATE_W_OFF length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][1] = (void *)[d newBufferWithBytesNoCopy:base+GATE_S_OFF length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][2] = (void *)[d newBufferWithBytesNoCopy:base+UP_W_OFF   length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][3] = (void *)[d newBufferWithBytesNoCopy:base+UP_S_OFF   length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][4] = (void *)[d newBufferWithBytesNoCopy:base+DOWN_W_OFF length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
                    eng->expert_gpu_buf[layer][eid][5] = (void *)[d newBufferWithBytesNoCopy:base+DOWN_S_OFF length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                }
                n_created++;
            }
        }
        fprintf(stderr, "[smelt] Created %d persistent Shared GPU buffers for cached experts\n", n_created * 6);
    }

    // === Populate buf_cached_flags for GPU routing (moe_route_gpu Enc 7) ===
    // buf_cached_flags is per-expert (not per-layer), shared across all layers.
    // A flag is set if the expert is cached in ANY score-based layer (union across layers).
    // This ensures penalty is only applied to experts not cached in any layer.
    // Hash layers (0-2) don't use GPU routing, so they don't consume this buffer.
    if (eng->buf_cached_flags) {
        uint8_t *flags = (uint8_t *)[(id<MTLBuffer>)eng->buf_cached_flags contents];
        memset(flags, 0, N_EXPERTS * sizeof(uint8_t));
        // Union across ALL score layers (not just layer 5) — avoids forcing routing to
        // uncached experts in layers whose hot experts differ from layer 5.
        for (int layer = 3; layer < N_LAYERS; layer++) {
            if (!eng->expert_mem_cache[layer]) continue;
            for (int eid = 0; eid < N_EXPERTS; eid++) {
                if (eng->expert_mem_cache[layer][eid]) {
                    flags[eid] = 1;
                }
            }
        }
        int n_flagged = 0;
        for (int eid = 0; eid < N_EXPERTS; eid++) n_flagged += flags[eid];
        fprintf(stderr, "[smelt] GPU routing flags: %d/%d experts marked cached (union of all score layers)\n", n_flagged, N_EXPERTS);
    }

    return n;
}

void moe_infer_smelt_set_decode_phase(MoEInferEngine *eng) {
    if (eng) eng->decode_phase = true;  // unconditional: gates predictive prefetch
    if (eng && eng->smelt_enabled && !eng->smelt_in_decode_phase) {
        eng->smelt_in_decode_phase = true;
        fprintf(stderr, "[smelt] Decode phase started — SMELT token counting enabled\n");
    }
}

// Background preload thread state
typedef struct {
    MoEInferEngine *eng;
} SmeltPreloadArgs;

static void *smelt_preload_thread(void *arg) {
    SmeltPreloadArgs *a = (SmeltPreloadArgs *)arg;
    moe_infer_smelt_finish_warmup(a->eng);
    free(a);
    return NULL;
}

// Async version: spawn background thread for preloading, return immediately.
// The routing bias will only be active once smelt_warmup_done becomes true.
// Meanwhile, all SSD reads proceed normally (no penalty applied yet).
void moe_infer_smelt_preload_async(MoEInferEngine *eng) {
    if (!eng->smelt_enabled || eng->smelt_warmup_done) return;
    // Set warmup_done = true IMMEDIATELY to prevent duplicate thread launches.
    // The preload thread will finish loading experts; smelt_warmup_done stays true.
    eng->smelt_warmup_done = true;  // guard: prevent re-entry before thread finishes
    SmeltPreloadArgs *a = (SmeltPreloadArgs *)malloc(sizeof(SmeltPreloadArgs));
    if (!a) { moe_infer_smelt_finish_warmup(eng); return; }
    a->eng = eng;
    pthread_t t;
    if (pthread_create(&t, NULL, smelt_preload_thread, a) != 0) {
        free(a);
        moe_infer_smelt_finish_warmup(eng);
        return;
    }
    pthread_detach(t);
    fprintf(stderr, "[smelt] Async preload started (routing bias inactive until complete)\n");
}

// ============================================================================
// SMELT routing stats persistence: save/load routing_counts to/from disk.
// Format: magic(4) + version(4) + n_layers(4) + n_experts(4) + counts(N_LAYERS*N_EXPERTS*4)
// Total: ~44 KB. Saved once at server shutdown, loaded at next startup so that
// smelt_finish_warmup loads the ACTUAL hot experts rather than experts 0..N-1.
// ============================================================================

#define SMELT_STATS_MAGIC   0x534D454C  // "SMEL"
#define SMELT_STATS_VERSION 1

void moe_infer_smelt_save_stats(MoEInferEngine *eng, const char *path) {
    if (!eng || !path || !eng->smelt_enabled) return;
    // Create parent dirs if needed (best effort)
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = '\0';
        mkdir(dir, 0755);
    }
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[smelt] save_stats: cannot open %s: %s\n", path, strerror(errno));
        return;
    }
    uint32_t hdr[4] = { SMELT_STATS_MAGIC, SMELT_STATS_VERSION, N_LAYERS, N_EXPERTS };
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(eng->routing_counts, sizeof(eng->routing_counts), 1, f);
    fclose(f);
    // Compute total token count for logging
    uint64_t total = 0;
    for (int l = 0; l < N_LAYERS; l++)
        for (int e = 0; e < N_EXPERTS; e++)
            total += eng->routing_counts[l][e];
    fprintf(stderr, "[smelt] save_stats: saved to %s (%.1fK total selections)\n",
            path, (double)total / 1000.0);
}

int moe_infer_smelt_load_stats(MoEInferEngine *eng, const char *path) {
    if (!eng || !path) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[smelt] load_stats: no saved stats at %s (first run — will use default order)\n", path);
        return 0;
    }
    uint32_t hdr[4];
    if (fread(hdr, sizeof(hdr), 1, f) != 1 ||
        hdr[0] != SMELT_STATS_MAGIC || hdr[1] != SMELT_STATS_VERSION ||
        hdr[2] != N_LAYERS || hdr[3] != N_EXPERTS) {
        fprintf(stderr, "[smelt] load_stats: invalid or stale stats file (will regenerate)\n");
        fclose(f);
        return 0;
    }
    if (fread(eng->routing_counts, sizeof(eng->routing_counts), 1, f) != 1) {
        fprintf(stderr, "[smelt] load_stats: read error\n");
        fclose(f);
        return 0;
    }
    fclose(f);
    uint64_t total = 0;
    for (int l = 0; l < N_LAYERS; l++)
        for (int e = 0; e < N_EXPERTS; e++)
            total += eng->routing_counts[l][e];
    fprintf(stderr, "[smelt] load_stats: loaded from %s (%.1fK historical selections)\n",
            path, (double)total / 1000.0);
    return 1;
}

void moe_infer_smelt_set_penalty(MoEInferEngine *eng, float penalty) {
    if (!eng) return;
    eng->smelt_penalty = penalty;
    fprintf(stderr, "[smelt] routing bias penalty updated to %.0f\n", penalty);
}

void moe_infer_smelt_set_stats_path(MoEInferEngine *eng, const char *path) {
    if (!eng) return;
    eng->smelt_stats_path = path;  // caller owns the string
}

int moe_infer_set_experimental_cache(MoEInferEngine *eng, bool enabled) {
    if (!eng || !eng->initialized || eng->smelt_warmup_done ||
        (enabled && eng->prefetch_disabled == 0)) return -1;
    eng->experimental_cache_enabled = enabled;
    fprintf(stderr,
        "[cache-policy] bounded per-layer LRU %s (exact-Q4 residency only, fixed capacity)\n",
        enabled ? "enabled" : "disabled");
    return 0;
}

static void cache_lru_touch_hits(
    MoEInferEngine *eng, int layer, const int *expert_ids,
    uint8_t *const expert_data[N_ACTIVE]) {
    if (!eng->experimental_cache_enabled || eng->tid2eid[layer]) return;
    for (int k = 0; k < N_ACTIVE; ++k) {
        const int eid = expert_ids[k];
        if (eid >= 0 && eid < N_EXPERTS && expert_data[k] &&
            eng->expert_mem_cache[layer][eid] == expert_data[k]) {
            eng->cache_lru_recency[layer][eid] = ++eng->cache_lru_clock;
        }
    }
}

static int cache_lru_admit_misses(
    MoEInferEngine *eng, int layer, const int *route_ids,
    const int *miss_ids, const int *miss_map,
    uint8_t *const miss_bufs[N_ACTIVE], int n_miss,
    uint8_t *expert_data[N_ACTIVE], NativeLayerMetrics *telemetry) {
    if (!eng->experimental_cache_enabled || eng->tid2eid[layer] || n_miss == 0)
        return 0;
    if (!eng->expert_mem_cache[layer] || !eng->expert_mem_pool[layer] ||
        eng->smelt_n_per_layer < N_ACTIVE) return -1;

    for (int i = 0; i < n_miss; ++i) {
        const int admitted = miss_ids[i];
        int victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (int eid = 0; eid < N_EXPERTS; ++eid) {
            if (!eng->expert_mem_cache[layer][eid]) continue;
            bool pinned = false;
            for (int rank = 0; rank < N_ACTIVE; ++rank) {
                if (route_ids[rank] == eid) { pinned = true; break; }
            }
            if (pinned) continue;
            const uint64_t recency = eng->cache_lru_recency[layer][eid];
            if (victim < 0 || recency < oldest ||
                (recency == oldest && eid < victim)) {
                victim = eid;
                oldest = recency;
            }
        }
        if (victim < 0 || victim == admitted) return -1;

        uint8_t *slot = eng->expert_mem_cache[layer][victim];
        const int slot_index = eng->smelt_pool_pos[layer][victim];
        if (!slot || slot_index < 0 ||
            slot_index >= eng->smelt_n_per_layer) return -1;
        memcpy(slot, miss_bufs[i], eng->active_expert_size);

        eng->expert_mem_cache[layer][victim] = NULL;
        eng->smelt_pool_pos[layer][victim] = -1;
        eng->cache_lru_recency[layer][victim] = 0;
        for (int component = 0; component < 9; ++component) {
            eng->expert_gpu_buf[layer][admitted][component] =
                eng->expert_gpu_buf[layer][victim][component];
            eng->expert_gpu_buf[layer][victim][component] = NULL;
        }
        eng->expert_mem_cache[layer][admitted] = slot;
        eng->smelt_pool_pos[layer][admitted] = slot_index;
        eng->cache_lru_recency[layer][admitted] = ++eng->cache_lru_clock;
        expert_data[miss_map[i]] = slot;
        eng->cache_lru_admissions++;
        eng->cache_lru_evictions++;
        if (telemetry) {
            telemetry->cache_admissions++;
            telemetry->cache_evictions++;
        }
    }
    return 0;
}

// Set DSpark engine pointer (enables hidden state extraction during forward)
void moe_infer_set_dspark_engine(MoEInferEngine *eng, void *dspark) {
    if (!eng) return;
    eng->dspark_engine = dspark;
    eng->dspark_accumulate_enabled = (dspark != NULL);  // enabled by default when DSpark is active
}

void moe_infer_set_dspark_accumulate(MoEInferEngine *eng, bool enabled) {
    if (eng) eng->dspark_accumulate_enabled = enabled;
}

int moe_infer_dspark_commit(MoEInferEngine *eng, int n_accepted, int start_pos) {
    if (!eng || !eng->dspark_engine || !eng->dspark_hidden3 ||
        n_accepted < 0 || n_accepted >= eng->dspark_hidden3_n) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    const int status = dspark_commit_accepted(
        (DSparkEngine *)eng->dspark_engine,
        eng->dspark_hidden3,
        eng->dspark_hidden3_n,
        n_accepted,
        start_pos);
    switch (status) {
        case DSPARK_COMMIT_OK:
            return NATIVE_BATCH_OK;
        case DSPARK_COMMIT_INVALID_ARGUMENT:
            return NATIVE_BATCH_INVALID_ARGUMENT;
        case DSPARK_COMMIT_ALLOCATION_FAILED:
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        case DSPARK_COMMIT_INVALID_STATE:
        default:
            return NATIVE_BATCH_INVALID_STATE;
    }
}

// ============================================================================
// Gather mode: create per-layer full-expert buffers for gatherQmm
// ============================================================================

// Initialize gather mode using the SMELT preloaded expert pool.
// Works with any smelt_n (doesn't require all 256 experts — only K=6 must be cached).
// Creates per-layer NoCopy Metal buffer views over the SMELT RAM pool.
// The gather kernels use pool_pos (slot index) instead of raw expert_id.
// Caller translates: pool_pos = smelt_pool_pos[layer][expert_id] before dispatch.
//
// Returns 1 on success, 0 on failure (SMELT not ready or insufficient cache).
int moe_infer_init_gather_mode(MoEInferEngine *eng) {
    if (!eng->smelt_warmup_done) {
        fprintf(stderr, "[gather] Cannot init gather mode before SMELT warmup\n");
        return 0;
    }

    // Count layers with enough cached experts
    int layers_ok = 0;
    for (int layer = 0; layer < N_LAYERS; layer++) {
        if (!eng->expert_mem_pool[layer]) continue;
        int n_cached = 0;
        for (int eid = 0; eid < N_EXPERTS; eid++) {
            if (eng->smelt_pool_pos[layer][eid] >= 0) n_cached++;
        }
        if (n_cached >= N_ACTIVE) layers_ok++;
    }
    if (layers_ok == 0) {
        fprintf(stderr, "[gather] No layers have >= %d cached experts for gather mode\n", N_ACTIVE);
        return 0;
    }

    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    fprintf(stderr, "[gather] Initializing gather mode (%d/%d layers ready)...\n",
            layers_ok, N_LAYERS);

    // Each layer: create a single NoCopy Metal buffer over the SMELT pool.
    // The pool stores experts in slot order: [slot0][slot1]...[slot(n-1)]
    // Each slot follows packed_experts layout (EXPERT_SIZE bytes):
    //   [GATE_W: 4MB][GATE_S: 256KB][UP_W: 4MB][UP_S: 256KB][DOWN_W: 4MB][DOWN_S: 256KB]
    //
    // Gather kernels use: base = pool + pool_pos * EXPERT_SIZE_U32 + COMPONENT_OFFSET
    // pool_pos comes from smelt_pool_pos[layer][expert_id] (set during SMELT warmup).
    for (int layer = 0; layer < N_LAYERS; layer++) {
        uint8_t *pool = eng->expert_mem_pool[layer];
        if (!pool) continue;

        // Determine actual pool size (max slot + 1) * EXPERT_SIZE
        int n_slots = 0;
        for (int eid = 0; eid < N_EXPERTS; eid++) {
            int s = eng->smelt_pool_pos[layer][eid];
            if (s >= 0 && s + 1 > n_slots) n_slots = s + 1;
        }
        if (n_slots < N_ACTIVE) continue;

        size_t pool_size = (size_t)n_slots * eng->active_expert_size;
        id<MTLBuffer> pool_buf = [d newBufferWithBytesNoCopy:pool
                                                      length:pool_size
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
        // All 6 "component buffers" alias the same pool (kernel uses internal offsets)
        eng->buf_gather_gate_W[layer] = (void *)pool_buf;
        eng->buf_gather_gate_s[layer] = (void *)pool_buf;
        eng->buf_gather_up_W[layer]   = (void *)pool_buf;
        eng->buf_gather_up_s[layer]   = (void *)pool_buf;
        eng->buf_gather_down_W[layer] = (void *)pool_buf;
        eng->buf_gather_down_s[layer] = (void *)pool_buf;

        if (layer < 3 || layer == 42) {
            fprintf(stderr, "[gather] L%d: %d slots, %.1f MB NoCopy buffer\n",
                    layer, n_slots, (double)pool_size/(1024.0*1024.0));
        }
    }

    eng->gather_mode = true;
    fprintf(stderr, "[gather] Ready. Expert IDs remapped via smelt_pool_pos table.\n");
    return 1;
}

static uint64_t exact_q4_clock_now(MoEInferEngine *eng) {
#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
    return moe_infer_exact_q4_test_clock_now(eng);
#else
    (void)eng;
    return moe_infer_monotonic_ns();
#endif
}

static ExactQ4DeferredClock exact_q4_capture_deferred_clock(
    MoEInferEngine *eng) {
    ExactQ4DeferredClock clock = {0};
#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
    DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
    if (state) {
        clock.callback = state->test_clock;
        clock.context = state->test_clock_context;
    }
#else
    (void)eng;
#endif
    return clock;
}

static uint64_t exact_q4_deferred_clock_now(
    const ExactQ4DeferredClock *clock) {
#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
    return clock->callback
        ? clock->callback(clock->context)
        : moe_infer_monotonic_ns();
#else
    (void)clock;
    return moe_infer_monotonic_ns();
#endif
}

static bool exact_q4_recording_active(const MoEInferEngine *eng) {
    return eng && eng->exact_q4 && eng->exact_q4->request_active &&
        eng->exact_q4->token_active && !eng->exact_q4->finalized;
}

static void exact_q4_layer_discard(DmlxExactQ4PendingLayer *transaction) {
    if (!transaction) return;
    transaction->active = false;
    transaction->failed = false;
    transaction->path_set = false;
    transaction->stage_count = 0u;
}

static void exact_q4_layer_begin(
    MoEInferEngine *eng, DmlxExactQ4PendingLayer *transaction, int layer) {
    exact_q4_layer_discard(transaction);
    if (!exact_q4_recording_active(eng) || layer < 0 || layer >= N_LAYERS)
        return;
    transaction->active = true;
    transaction->layer = (uint16_t)layer;
    transaction->reserved = 0u;
    transaction->reserved2 = 0u;
    memset(&transaction->path, 0, sizeof(transaction->path));
    transaction->path.layer = (uint16_t)layer;
}

static bool exact_q4_layer_add_stage(
    DmlxExactQ4PendingLayer *transaction,
    const DmlxExactQ4StageEvent *event) {
    if (!transaction || !transaction->active) return true;
    if (!event || event->layer != transaction->layer ||
        event->stage >= DMLX_EXACT_Q4_STAGE_COUNT ||
        event->stage == DMLX_EXACT_Q4_STAGE_LOGITS_SAMPLE ||
        event->end_ns < event->begin_ns ||
        (event->flags & ~DMLX_EXACT_Q4_EVENT_VALID_FLAGS) != 0u ||
        transaction->stage_count >= DMLX_EXACT_Q4_LAYER_STAGE_CAPACITY ||
        (transaction->stage_count > 0u &&
         event->stage < transaction->stages[transaction->stage_count - 1u].stage)) {
        transaction->failed = true;
        return false;
    }
    transaction->stages[transaction->stage_count++] = *event;
    return true;
}

static void exact_q4_layer_set_path(
    DmlxExactQ4PendingLayer *transaction,
    const DmlxMlaPathIdentity *identity) {
    if (!transaction || !transaction->active) return;
    if (!identity) {
        transaction->failed = true;
        return;
    }
    transaction->path.path = identity->path;
    transaction->path.flags = identity->flags;
    transaction->path_set = true;
}

static int exact_q4_layer_commit(
    MoEInferEngine *eng, DmlxExactQ4PendingLayer *transaction) {
    if (!transaction || !transaction->active)
        return DMLX_EXACT_Q4_PRODUCER_OK;
    if (transaction->failed || !transaction->path_set ||
        transaction->stage_count == 0u) {
        DmlxExactQ4ProducerState *state = eng ? eng->exact_q4 : NULL;
        if (state) exact_q4_mark_incomplete(state);
        exact_q4_layer_discard(transaction);
        return DMLX_EXACT_Q4_PRODUCER_INVALID_STATE;
    }
    const int status = moe_infer_exact_q4_commit_layer(
        eng, transaction->stages, transaction->stage_count,
        &transaction->path);
    exact_q4_layer_discard(transaction);
    return status;
}

static bool exact_q4_reserve_physical_read(
    MoEInferEngine *eng, uint16_t layer, uint16_t expert, uint8_t route_rank,
    uint8_t source, uint64_t requested_bytes, uint32_t *ordinal) {
    if (!exact_q4_recording_active(eng)) return false;
    const DmlxExactQ4ExpertReadEvent event = {
        .layer = layer,
        .expert = expert,
        .route_rank = route_rank,
        .source = source,
        .submit_ns = exact_q4_clock_now(eng),
        .requested_bytes = requested_bytes,
    };
    return moe_infer_exact_q4_reserve_read(eng, &event, ordinal) ==
        DMLX_EXACT_Q4_PRODUCER_OK;
}

static void exact_q4_record_logical_read(
    MoEInferEngine *eng, uint16_t layer, uint16_t expert, uint8_t route_rank,
    uint8_t source) {
    if (!exact_q4_recording_active(eng)) return;
    const uint64_t now = exact_q4_clock_now(eng);
    const DmlxExactQ4ExpertReadEvent event = {
        .layer = layer,
        .expert = expert,
        .route_rank = route_rank,
        .source = source,
        .submit_ns = now,
        .requested_bytes = 0u,
    };
    uint32_t ordinal = 0u;
    if (moe_infer_exact_q4_reserve_read(eng, &event, &ordinal) !=
            DMLX_EXACT_Q4_PRODUCER_OK) return;
    if (moe_infer_exact_q4_publish_read_worker_start(eng, ordinal, now) !=
            DMLX_EXACT_Q4_PRODUCER_OK ||
        moe_infer_exact_q4_publish_read_completion(
            eng, ordinal, now, 0u, 0) != DMLX_EXACT_Q4_PRODUCER_OK) {
        (void)moe_infer_exact_q4_abort_read(eng, ordinal);
    }
}

// Post K parallel preads without blocking. Completion is observed via io_pool_wait.
static int io_pool_start(IOPool *pool, int layer_fd, int *expert_ids, int K,
                         uint8_t *buffers[6], size_t expert_sz,
                         MoEInferEngine *exact_q4_engine,
                         const uint32_t exact_q4_ordinals[6],
                         const uint8_t exact_q4_owned[6],
                         bool exact_q4_deferred) {
    io_pool_wait(pool);
    pthread_mutex_lock(&pool->mutex);
    pool->generation++;
    const int generation = pool->generation;
    ExactQ4DeferredClock deferred_clock = {0};
    uint64_t submit_ns = 0u;
    if (exact_q4_deferred) {
        deferred_clock = exact_q4_capture_deferred_clock(exact_q4_engine);
        submit_ns = exact_q4_deferred_clock_now(&deferred_clock);
    }
    for (int k = 0; k < K; k++) {
        pool->fd[k] = layer_fd;
        pool->buf[k] = buffers[k];
        pool->size = expert_sz;
        pool->offset[k] = (off_t)expert_ids[k] * (off_t)expert_sz;
        pool->exact_q4_engine[k] = exact_q4_engine;
        pool->exact_q4_ordinal[k] = exact_q4_ordinals
            ? exact_q4_ordinals[k] : 0u;
        pool->exact_q4_owned[k] = exact_q4_owned
            ? exact_q4_owned[k] : 0u;
        pool->exact_q4_deferred[k] = exact_q4_deferred ? 1u : 0u;
        pool->exact_q4_deferred_clock[k] = deferred_clock;
        pool->task_generation[k] = generation;
        pool->submit_ns[k] = submit_ns;
        pool->worker_start_ns[k] = 0u;
        pool->complete_ns[k] = 0u;
        pool->done[k] = 0;
        pool->result[k] = -1;
        pool->error_code[k] = 0;
    }
    pool->num_tasks = K;
    pool->tasks_done = 0;
    pool->in_flight_generation = K > 0 ? generation : 0;
#ifdef DMLX_EXACT_Q4_IO_TESTING
    pool->dependency_wait_entered = false;
#endif
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);
    return generation;
}

// Block until all tasks posted by the last io_pool_start complete.
static void io_pool_wait(IOPool *pool) {
    pthread_mutex_lock(&pool->mutex);
    while (pool->tasks_done < pool->num_tasks) {
        pthread_cond_wait(&pool->work_done, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
}

typedef struct {
    uint64_t submit_ns;
    uint64_t worker_start_ns;
    uint64_t complete_ns;
    uint64_t wait_begin_ns;
    uint64_t wait_end_ns;
    ssize_t result;
    int error_code;
} ExactQ4DeferredRead;

// Wait for one task only when its generation still owns the slot. Clock reads
// are restricted to a recording-active join that actually blocks.
static bool io_pool_wait_task_blocked(
    IOPool *pool, int task_idx, int expected_generation, MoEInferEngine *engine,
    ExactQ4DeferredRead *read, bool *blocked) {
    pthread_mutex_lock(&pool->mutex);
    if (task_idx < 0 || task_idx >= pool->num_tasks ||
        pool->generation != expected_generation ||
        pool->task_generation[task_idx] != expected_generation) {
        pthread_mutex_unlock(&pool->mutex);
        return false;
    }
    *blocked = false;
    const bool record_wait = !pool->done[task_idx] &&
        exact_q4_recording_active(engine);
    uint64_t begin_ns = 0u;
    if (!pool->done[task_idx]) {
        if (record_wait) begin_ns = exact_q4_clock_now(engine);
        do {
            pthread_cond_wait(&pool->work_done, &pool->mutex);
        } while (pool->generation == expected_generation &&
                 pool->task_generation[task_idx] == expected_generation &&
                 !pool->done[task_idx]);
        if (pool->generation != expected_generation ||
            pool->task_generation[task_idx] != expected_generation) {
            pthread_mutex_unlock(&pool->mutex);
            return false;
        }
        if (record_wait) {
            read->wait_begin_ns = begin_ns;
            read->wait_end_ns = exact_q4_clock_now(engine);
            *blocked = true;
        }
    }
    read->submit_ns = pool->submit_ns[task_idx];
    read->worker_start_ns = pool->worker_start_ns[task_idx];
    read->complete_ns = pool->complete_ns[task_idx];
    read->result = pool->result[task_idx];
    read->error_code = pool->error_code[task_idx];
    pthread_mutex_unlock(&pool->mutex);
    return true;
}

// Block until the submitted blocking-read generation completes. Only the
// current generation's dependency join can produce an exposed-wait span.
static void io_pool_wait_dependency(
    IOPool *pool, int expected_generation, MoEInferEngine *engine, int layer,
    DmlxExactQ4PendingLayer *transaction) {
    bool record_wait = false;
    bool blocked = false;
    const bool record_timeline = decode_timeline_active(engine);
    uint64_t begin_ns = record_timeline
        ? moe_infer_decode_timeline_now_ns() : 0u;
    uint64_t end_ns = begin_ns;
    pthread_mutex_lock(&pool->mutex);
    if (pool->generation == expected_generation &&
        pool->in_flight_generation == expected_generation &&
        pool->tasks_done < pool->num_tasks) {
        record_wait = exact_q4_recording_active(engine) || record_timeline;
        if (record_wait && !record_timeline) begin_ns = exact_q4_clock_now(engine);
        do {
            blocked = true;
#ifdef DMLX_EXACT_Q4_IO_TESTING
            pool->dependency_wait_entered = true;
            pthread_cond_broadcast(&pool->work_done);
#endif
            pthread_cond_wait(&pool->work_done, &pool->mutex);
        } while (pool->generation == expected_generation &&
                 pool->tasks_done < pool->num_tasks);
        if (record_wait) end_ns = record_timeline
            ? moe_infer_decode_timeline_now_ns() : exact_q4_clock_now(engine);
    }
    pthread_mutex_unlock(&pool->mutex);
    if (record_timeline) {
        decode_timeline_record(engine, (uint16_t)layer,
            DMLX_DECODE_TIMELINE_IO_EXPOSED_WAIT,
            begin_ns, begin_ns, begin_ns, begin_ns, end_ns, end_ns);
    }
    if (blocked && record_wait) {
        const DmlxExactQ4StageEvent event = {
            .layer = (uint16_t)layer,
            .stage = DMLX_EXACT_Q4_STAGE_IO_EXPOSED_WAIT,
            .flags = DMLX_EXACT_Q4_EVENT_JOIN_BLOCKED,
            .calls = 1u,
            .begin_ns = begin_ns,
            .end_ns = end_ns,
        };
        (void)exact_q4_layer_add_stage(transaction, &event);
    }
}

// Dispatch K parallel preads. Blocks until all complete and validates that
// every task returned a full expert blob.
static int io_pool_dispatch(IOPool *pool, MoEInferEngine *engine,
                            int layer_fd, int *expert_ids,
                            const int *route_ranks, int layer, int K,
                            uint8_t *buffers[6], size_t expert_sz,
                            uint64_t *bytes_read, uint64_t *short_reads,
                            DmlxExactQ4PendingLayer *transaction) {
    uint32_t ordinals[NUM_IO_THREADS] = {0};
    uint8_t owned[NUM_IO_THREADS] = {0};
    bool instrument = exact_q4_recording_active(engine);
    for (int index = 0; instrument && index < K; ++index) {
        if (!exact_q4_reserve_physical_read(
                engine, (uint16_t)layer, (uint16_t)expert_ids[index],
                (uint8_t)route_ranks[index], DMLX_EXACT_Q4_READ_PREAD,
                expert_sz, &ordinals[index])) {
            for (int prior = 0; prior < index; ++prior) {
                (void)moe_infer_exact_q4_abort_read(engine, ordinals[prior]);
            }
            memset(owned, 0, sizeof(owned));
            instrument = false;
            break;
        }
        owned[index] = 1u;
    }
    const int generation = io_pool_start(
        pool, layer_fd, expert_ids, K, buffers, expert_sz,
        instrument ? engine : NULL, ordinals, owned, false);
    io_pool_wait_dependency(pool, generation, engine, layer, transaction);
    uint64_t total = 0;
    uint64_t incomplete = 0;
    for (int index = 0; index < K; ++index) {
        const ssize_t result = pool->result[index];
        if (result > 0) total += (uint64_t)result;
        if (result != (ssize_t)expert_sz) incomplete++;
    }
    if (bytes_read) *bytes_read = total;
    if (short_reads) *short_reads = incomplete;
    return incomplete == 0 ? 0 : -1;
}

// ============================================================================
// Temporal expert prediction + predictive cross-layer prefetch
// ============================================================================

static void predictor_record(ExpertPredictor *pred, int layer, int *experts, int K) {
    for (int k = 0; k < K && k < N_ACTIVE; k++) {
        pred->experts[layer][k] = experts[k];
    }
    pred->valid = true;
}

// Issue an async prefetch for `layer`'s predicted experts (previous token's
// routing at that layer) into pred buffer set (layer % 2). Non-blocking:
// the reads proceed on the dedicated prefetch pool while the current layer's
// GPU work and the target layer's attention chain execute (~11ms window).
static void prefetch_issue(MoEInferEngine *eng, int layer) {
    if (eng->prefetch_disabled == 1 || !eng->predictor.valid) return;
    if (eng->in_batch) return;
    if (!eng->decode_phase) return;  // decode only: prefill hit rate ~16% and
                                     // prefetch contention costs +240ms/layer there
    if (layer < 0 || layer >= N_LAYERS) return;
    if (eng->tid2eid[layer] != NULL) return;   // hash-routing layer: all preloaded
    if (!eng->prefetch_pool) return;

    uint8_t **set = (layer % 2 == 0) ? eng->expert_buf_pred : eng->expert_buf_pred2;
    uint8_t *bufs[6];
    int ids[6], slots[6];
    int n = 0;
    for (int k = 0; k < N_ACTIVE; k++) {
        int eid = eng->predictor.experts[layer][k];
        if (eid < 0 || eid >= N_EXPERTS) continue;
        // Skip SMELT-cached experts — they are zero-copy hits at use time.
        if (eng->expert_cache_n_experts > 0 && eng->expert_mem_cache[layer] &&
            eng->expert_mem_cache[layer][eid]) continue;
        ids[n] = eid; bufs[n] = set[k]; slots[n] = k;
        n++;
    }
    if (n == 0) return;
    const bool instrument = exact_q4_recording_active(eng);
    IOPool *pool = (IOPool *)eng->prefetch_pool;
    const int generation = io_pool_start(
        pool, eng->packed_fd[layer], ids, n, bufs,
        eng->active_expert_size, instrument ? eng : NULL,
        NULL, NULL, instrument);
    pthread_mutex_lock(&pool->mutex);
    for (int k = 0; k < N_ACTIVE; k++) {
        eng->prefetch_ids[k] = -1;
        eng->prefetch_task[k] = -1;
    }
    for (int i = 0; i < n; i++) {
        eng->prefetch_ids[slots[i]] = ids[i];
        eng->prefetch_task[slots[i]] = i;
    }
    eng->prefetch_count = n;
    eng->prefetch_generation = generation;
    eng->prefetch_layer = layer;
    eng->prefetch_active = true;
    pthread_mutex_unlock(&pool->mutex);
}

// Consume a prefetch targeted at `layer`: for slots whose actual routing
// matches a predicted expert, wait ONLY for that expert's read task and point
// expert_data at its buffer. Missed slots fall back to the blocking pread
// path, and unmatched prefetch reads are left to finish in the background
// (their buffers are not reused until a later prefetch_issue safety-drains
// the pool) — the critical path never waits for useless experts.
static void prefetch_consume(
    MoEInferEngine *eng, int layer, const int *expert_ids,
    uint8_t *expert_data[6], DmlxExactQ4PendingLayer *transaction) {
    IOPool *pool = (IOPool *)eng->prefetch_pool;
    if (!pool) return;
    pthread_mutex_lock(&pool->mutex);
    if (!eng->prefetch_active || eng->prefetch_layer != layer ||
        eng->prefetch_generation == 0 ||
        pool->generation != eng->prefetch_generation) {
        pthread_mutex_unlock(&pool->mutex);
        return;
    }
    const int expected_generation = eng->prefetch_generation;
    eng->prefetch_active = false;
    pthread_mutex_unlock(&pool->mutex);

    uint8_t **set = (layer % 2 == 0) ? eng->expert_buf_pred : eng->expert_buf_pred2;
    for (int k = 0; k < N_ACTIVE; k++) {
        if (expert_data[k] != NULL) continue;
        if (expert_ids[k] < 0 || expert_ids[k] >= N_EXPERTS) continue;
        for (int s = 0; s < N_ACTIVE; s++) {
            if (eng->prefetch_ids[s] != expert_ids[k]) continue;
            const int task = eng->prefetch_task[s];
            ExactQ4DeferredRead read = {0};
            bool blocked = false;
            if (!io_pool_wait_task_blocked(
                    pool, task, expected_generation, eng, &read, &blocked)) {
                break;
            }
            if (blocked) {
                DmlxExactQ4StageEvent event = {
                    .layer = (uint16_t)layer,
                    .stage = DMLX_EXACT_Q4_STAGE_IO_EXPOSED_WAIT,
                    .flags = DMLX_EXACT_Q4_EVENT_JOIN_BLOCKED,
                    .calls = 1u,
                    .begin_ns = read.wait_begin_ns,
                    .end_ns = read.wait_end_ns,
                };
                (void)exact_q4_layer_add_stage(transaction, &event);
            }
            if (exact_q4_recording_active(eng) && read.submit_ns != 0u) {
                const DmlxExactQ4ExpertReadEvent event = {
                    .layer = (uint16_t)layer,
                    .expert = (uint16_t)expert_ids[k],
                    .route_rank = (uint8_t)k,
                    .source = DMLX_EXACT_Q4_READ_PREFETCH,
                    .submit_ns = read.submit_ns,
                    .requested_bytes = eng->active_expert_size,
                };
                uint32_t ordinal = 0u;
                if (moe_infer_exact_q4_reserve_read(eng, &event, &ordinal) ==
                        DMLX_EXACT_Q4_PRODUCER_OK) {
                    const int start_status =
                        moe_infer_exact_q4_publish_read_worker_start(
                            eng, ordinal, read.worker_start_ns);
                    if (start_status == DMLX_EXACT_Q4_PRODUCER_OK) {
                        const uint64_t bytes = read.result > 0
                            ? (uint64_t)read.result : 0u;
                        if (moe_infer_exact_q4_publish_read_completion(
                                eng, ordinal, read.complete_ns,
                                bytes <= eng->active_expert_size
                                    ? bytes : eng->active_expert_size,
                                read.error_code) != DMLX_EXACT_Q4_PRODUCER_OK) {
                            (void)moe_infer_exact_q4_abort_read(eng, ordinal);
                        }
                    } else {
                        (void)moe_infer_exact_q4_abort_read(eng, ordinal);
                    }
                }
            }
            if (read.result == (ssize_t)eng->active_expert_size) {
                expert_data[k] = set[s];
                eng->predictor.hits++;
            } else {
                NativeLayerMetrics *metrics = telemetry_current_layer(eng, layer);
                if (metrics) metrics->short_reads++;
            }
            break;
        }
        if (expert_data[k] == NULL) eng->predictor.misses++;
    }
}

// Adaptive kill-switch: with enough samples, stop prefetching when the
// temporal hit rate is too low — a wrong prediction wastes ~80MB of SSD
// bandwidth and pollutes the page cache. Only relevant when prefetch is
// enabled (NATIVE_PREFETCH=1 → prefetch_disabled == 0); default is off.
// Threshold is 5 tokens' worth of samples (1200) so the decision reflects
// steady-state routing, not the atypical first token(s) of a request
// (special chat-template tokens route near-disjointly and would otherwise
// trip the gate instantly — observed 0/240 on token 1 vs 39% steady-state).
static void prefetch_update_adaptive(MoEInferEngine *eng) {
    if (eng->prefetch_disabled != 0) return;
    int total = eng->predictor.hits + eng->predictor.misses;
    if (total >= 1200 && eng->predictor.hits * 4 < total) {  // hit rate < 25%
        eng->prefetch_disabled = 1;
        fprintf(stderr, "[prefetch] adaptive disable: hit rate %d/%d < 25%%\n",
                eng->predictor.hits, total);
    }
}

// Drain every posted prefetch generation, including unmatched reads left after
// logical consumption cleared prefetch_active.
static void prefetch_drain(MoEInferEngine *eng) {
    if (eng->prefetch_pool) {
        IOPool *pool = (IOPool *)eng->prefetch_pool;
        pthread_mutex_lock(&pool->mutex);
        const bool posted = eng->prefetch_generation != 0;
        while (posted && pool->tasks_done < pool->num_tasks) {
            pthread_cond_wait(&pool->work_done, &pool->mutex);
        }
        eng->prefetch_active = false;
        eng->prefetch_layer = -1;
        eng->prefetch_count = 0;
        eng->prefetch_generation = 0;
        for (int slot = 0; slot < N_ACTIVE; ++slot) {
            eng->prefetch_ids[slot] = -1;
            eng->prefetch_task[slot] = -1;
        }
        pthread_mutex_unlock(&pool->mutex);
        return;
    }
    eng->prefetch_active = false;
    eng->prefetch_layer = -1;
    eng->prefetch_count = 0;
    eng->prefetch_generation = 0;
}

#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
static bool exact_q4_test_pool_drained(IOPool *pool) {
    if (!pool) return true;
    pthread_mutex_lock(&pool->mutex);
    const bool drained = pool->tasks_done == pool->num_tasks &&
        pool->in_flight_generation == 0;
    pthread_mutex_unlock(&pool->mutex);
    return drained;
}

bool moe_infer_exact_q4_test_no_inflight_owners(MoEInferEngine *eng) {
    if (!eng) return false;
    DmlxExactQ4ProducerState *state = eng->exact_q4_storage;
    const bool producer_idle = !state ||
        (atomic_load_explicit(&state->active_read_publishers,
             memory_order_acquire) == 0u &&
         atomic_load_explicit(&state->read_owner_operations,
             memory_order_acquire) == 0u);
    return producer_idle && !eng->deferred.active &&
        eng->deferred.cmd_experts == NULL &&
        !eng->deferred.exact_q4_pending.active &&
        !eng->prefetch_active && eng->prefetch_generation == 0 &&
        eng->prefetch_count == 0 && eng->prefetch_layer == -1 &&
        exact_q4_test_pool_drained((IOPool *)eng->prefetch_pool) &&
        exact_q4_test_pool_drained((IOPool *)eng->io_pool);
}
#endif

// ============================================================================
// MoE forward for one layer (CMD3 — expert matmuls + combine)
// ============================================================================

static int moe_forward_layer(MoEInferEngine *eng, int layer_idx,
                              uint8_t *expert_bufs[6], int *expert_ids,
                              float *expert_weights, int K,
                              uint32_t token_index) {
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    const bool decode_timeline_recording = decode_timeline_active(eng);
    const uint64_t timeline_routed_begin = decode_timeline_recording
        ? moe_infer_decode_timeline_now_ns() : 0u;
    id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBuffer];
    const int gw_off = GATE_W_OFF, gs_off = GATE_S_OFF;
    const int uw_off = UP_W_OFF, us_off = UP_S_OFF;
    const int dw_off = DOWN_W_OFF, ds_off = DOWN_S_OFF;
    const int time_en = (getenv("NATIVE_TIME_LAYERS") != NULL);
    NativeLayerMetrics *telemetry = telemetry_current_layer(eng, layer_idx);
    if (telemetry) telemetry->moe_command_buffers++;
    double t0=0, t1=0, t2=0, t3=0;
    if (time_en) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec*1e9+ts.tv_nsec; }
    (void)t3;

    // Helper: get or create MTLBuffer for a given expert slot.
    // Uses persistent GPU buffer if available (post-SMELT), else creates on-the-fly.
    #define EXPERT_BUF(layer, eid, slot, ptr, len) \
        (eng->expert_gpu_buf[layer][eid][slot] ? \
            (id<MTLBuffer>)eng->expert_gpu_buf[layer][eid][slot] : \
            [d newBufferWithBytesNoCopy:(ptr) length:(len) options:MTLResourceStorageModeShared deallocator:nil])

    // Step 1 + 2: expert gate+up+SwiGLU and down_proj (6 separate per-expert kernels)
    // Note: fused_6expert_gate_up was tried but is slower due to reduced GPU parallelism
    // (serial expert loop in kernel vs GPU scheduling multiple dispatches in parallel).
    {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBufferWithUnretainedReferences];
        int used_gather6 = 0;  // set when gather6 path wrote directly to buf_expert_contiguous

        if (eng->gather_mode && eng->buf_gather_gate_W[layer_idx] && !eng->use_affine_experts) {
            // === GATHER MODE: gatherQmm-equivalent ===
            // Single dispatch covering all K experts simultaneously.
            // GPU reads only selected experts' rows from the full SMELT pool.
            // Translate expert_ids → pool_pos (slot indices in SMELT pool)
            uint32_t *eid_buf = (uint32_t *)[(id<MTLBuffer>)eng->buf_gather_expert_ids contents];
            int all_in_pool = 1;
            for (int k = 0; k < K; k++) {
                int pos = eng->smelt_pool_pos[layer_idx][expert_ids[k]];
                if (pos < 0) { all_in_pool = 0; break; }
                eid_buf[k] = (uint32_t)pos;  // write pool slot, not raw expert_id
            }

            if (!all_in_pool) goto separate_mode;  // fall back if any expert not in pool

            // Encoder 1: gather gate+up+SwiGLU (K experts in parallel via Y dimension)
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_gather_gate_up];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_gate_W[layer_idx] offset:0 atIndex:0];
                [enc setBuffer:eng->buf_normed offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_mid offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_expert_ids offset:0 atIndex:3];
                uint k_val = K;
                [enc setBytes:&k_val length:4 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(INTERMEDIATE/8, K, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                telemetry_record_moe_dispatch(
                    eng, layer_idx, (uint64_t)(INTERMEDIATE / 8) * (uint64_t)K,
                    256, (uint64_t)(INTERMEDIATE / 8) * (uint64_t)K * 256);
                [enc endEncoding];
            }

            // Encoder 2: gather down_proj (K experts in parallel)
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_gather_down];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_gate_W[layer_idx] offset:0 atIndex:0]; // pool
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_mid offset:0 atIndex:1];                // x_mid
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_out offset:0 atIndex:2];                // out [K×DIM]
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_expert_ids offset:0 atIndex:3];
                uint k_val = K;
                [enc setBytes:&k_val length:4 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(DIM/8, K, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                telemetry_record_moe_dispatch(
                    eng, layer_idx, (uint64_t)(DIM / 8) * (uint64_t)K,
                    256, (uint64_t)(DIM / 8) * (uint64_t)K * 256);
                [enc endEncoding];
            }
        } else {
            separate_mode:;
            if (!eng->use_affine_experts) {
                // === GATHER6 MODE: 6 independent expert blobs via pointer array ===
                // MXFP4 (gs=32). 2 dispatches total replace 12 per-expert encoders
                // and the 6 blits: down writes directly into buf_expert_contiguous.
                id<MTLBuffer> blob_bufs[6];
                for (int k = 0; k < 6; k++) {
                    int kk = (k < K) ? k : 0;
                    void *ptr = expert_bufs[kk] ? (void *)expert_bufs[kk]
                                                : (void *)eng->expert_buf[kk];
                    // Reuse persistent wrappers for the fixed pread/prefetch buffers
                    // so GPU page mappings stay stable across tokens; fresh NoCopy
                    // only for SMELT pool pointers (rare per-slot case).
                    void **slot = NULL;
                    void **slot_ptr = NULL;  // backing-pointer guard for stale-wrapper detection
                    if (ptr == (void *)eng->expert_buf[kk]) {
                        slot = &eng->blob_wrap_pread[kk];
                        slot_ptr = &eng->blob_wrap_pread_ptr[kk];
                    } else if (ptr == (void *)eng->expert_buf_pred[kk]) {
                        slot = &eng->blob_wrap_pred[kk];
                        slot_ptr = &eng->blob_wrap_pred_ptr[kk];
                    } else if (ptr == (void *)eng->expert_buf_pred2[kk]) {
                        slot = &eng->blob_wrap_pred2[kk];
                        slot_ptr = &eng->blob_wrap_pred2_ptr[kk];
                    } else if (eng->batch_pool && eng->batch_slot_size &&
                             (uint8_t *)ptr >= eng->batch_pool &&
                             (uint8_t *)ptr < eng->batch_pool + (size_t)eng->batch_capacity * eng->batch_slot_size) {
                        // Batch dedup pool slot → per-slot persistent wrapper
                        // (pool memory is stable, so the wrapper stays valid
                        // across layers even as slot contents change).
                        size_t sidx = (size_t)((uint8_t *)ptr - eng->batch_pool) / eng->batch_slot_size;
                        slot = &eng->batch_wrap[sidx];
                    }
                    if (slot) {
                        // Invalidate stale wrapper if the backing pointer changed
                        // (e.g. SMELT evicted the slot and reloaded a different expert).
                        if (*slot && slot_ptr && *slot_ptr != ptr) {
                            [(id<MTLBuffer>)*slot release];
                            *slot = NULL;
                            *slot_ptr = NULL;
                        }
                        if (!*slot) {
                            *slot = (void *)[d newBufferWithBytesNoCopy:ptr
                                                                 length:eng->active_expert_size
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
                            if (slot_ptr) *slot_ptr = ptr;
                        }
                        blob_bufs[k] = (id<MTLBuffer>)*slot;
                    } else {
                        blob_bufs[k] = [d newBufferWithBytesNoCopy:ptr
                                                            length:eng->active_expert_size
                                                           options:MTLResourceStorageModeShared
                                                       deallocator:nil];
                    }
                }
                // Encoder 1: fused gate+up+SwiGLU for all K experts
                if (eng->expert_format_2b) {
                    static int logged_2b = 0;
                    if (!logged_2b) { logged_2b = 1; fprintf(stderr, "[2b] gather6 int2_gs64 kernels dispatched\n"); }
                }
                {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)(eng->expert_format_2b ? eng->pipe_gather6_gate_up_2b : eng->pipe_gather6_gate_up)];
                    for (int k = 0; k < 6; k++) [enc setBuffer:blob_bufs[k] offset:0 atIndex:k];
                    [enc setBuffer:eng->buf_normed offset:0 atIndex:6];
                    [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_mid offset:0 atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(INTERMEDIATE/8, K, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                telemetry_record_moe_dispatch(
                    eng, layer_idx, (uint64_t)(INTERMEDIATE / 8) * (uint64_t)K,
                    256, (uint64_t)(INTERMEDIATE / 8) * (uint64_t)K * 256);
                    [enc endEncoding];
                }
                // Encoder 2: down_proj for all K experts → contiguous [K×DIM]
                {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)(eng->expert_format_2b ? eng->pipe_gather6_down_2b : eng->pipe_gather6_down)];
                    for (int k = 0; k < 6; k++) [enc setBuffer:blob_bufs[k] offset:0 atIndex:k];
                    [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_mid offset:0 atIndex:6];
                    [enc setBuffer:(id<MTLBuffer>)eng->buf_expert_contiguous offset:0 atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(DIM/8, K, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                telemetry_record_moe_dispatch(
                    eng, layer_idx, (uint64_t)(DIM / 8) * (uint64_t)K,
                    256, (uint64_t)(DIM / 8) * (uint64_t)K * 256);
                    [enc endEncoding];
                }
                used_gather6 = 1;
            } else {
            // === SEPARATE MODE: 6 per-expert dispatches (affine, gs=64) ===
            for (int k = 0; k < K; k++) {
                int eid = expert_ids[k];
                if (eid < 0 || eid >= N_EXPERTS) continue;
                char *base = (char *)expert_bufs[k];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                {
                    // affine v2: bf16 scale+bias, gs=64
                    [enc setComputePipelineState:eng->pipe_gate_up_swiglu_v2_affine];
                    [enc setBuffer:EXPERT_BUF(layer_idx,eid,0,base+AFFINE_GATE_W_OFF,4194304) offset:0 atIndex:0];
                    [enc setBuffer:EXPERT_BUF(layer_idx,eid,1,base+AFFINE_GATE_S_OFF,262144)  offset:0 atIndex:1];
                    [enc setBuffer:EXPERT_BUF(layer_idx,eid,2,base+AFFINE_GATE_B_OFF,262144)  offset:0 atIndex:2];
                    [enc setBuffer:EXPERT_BUF(layer_idx,eid,3,base+AFFINE_UP_W_OFF,4194304)   offset:0 atIndex:3];
                    [enc setBuffer:EXPERT_BUF(layer_idx,eid,4,base+AFFINE_UP_S_OFF,262144)    offset:0 atIndex:4];
                    [enc setBuffer:EXPERT_BUF(layer_idx,eid,5,base+AFFINE_UP_B_OFF,262144)    offset:0 atIndex:5];
                    [enc setBuffer:eng->buf_normed offset:0 atIndex:6];
                    [enc setBuffer:eng->buf_expert_mid[k] offset:0 atIndex:7];
                    uint od=INTERMEDIATE,id_=DIM,gs=64;
                    [enc setBytes:&od length:4 atIndex:8]; [enc setBytes:&id_ length:4 atIndex:9]; [enc setBytes:&gs length:4 atIndex:10];
                    [enc setThreadgroupMemoryLength:512 atIndex:0];
                    uint ntg = (INTERMEDIATE + 1) / 2;
                    [enc dispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
                    telemetry_record_moe_dispatch(
                        eng, layer_idx, ntg, 128, (uint64_t)ntg * 128);
                }
                [enc endEncoding];
            }
            for (int k = 0; k < K; k++) {
                int eid = expert_ids[k];
                if (eid < 0 || eid >= N_EXPERTS) continue;
                char *base = (char *)expert_bufs[k];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                {
                    // affine v2 down: bf16 scale+bias, gs=64
                    [enc setComputePipelineState:eng->pipe_dequant_matvec_4bit_affine];
                    [enc setBuffer:EXPERT_BUF(layer_idx,eid,6,base+AFFINE_DOWN_W_OFF,4194304) offset:0 atIndex:0];
                    [enc setBuffer:EXPERT_BUF(layer_idx,eid,7,base+AFFINE_DOWN_S_OFF,262144)  offset:0 atIndex:1];
                    [enc setBuffer:EXPERT_BUF(layer_idx,eid,8,base+AFFINE_DOWN_B_OFF,262144)  offset:0 atIndex:2];
                    [enc setBuffer:eng->buf_expert_mid[k] offset:0 atIndex:3];
                    [enc setBuffer:eng->buf_expert_out[k] offset:0 atIndex:4];
                    uint od=DIM,id_=INTERMEDIATE,gs=64;
                    [enc setBytes:&od length:4 atIndex:5]; [enc setBytes:&id_ length:4 atIndex:6]; [enc setBytes:&gs length:4 atIndex:7];
                    [enc setThreadgroupMemoryLength:256 atIndex:0];
                    uint d_ntg = (DIM + 1) / 2;
                    [enc dispatchThreadgroups:MTLSizeMake(d_ntg,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
                    telemetry_record_moe_dispatch(
                        eng, layer_idx, d_ntg, 128, (uint64_t)d_ntg * 128);
                }
                [enc endEncoding];
            }
            } // end if gather6 / else affine separate
        } // end if gather_mode / else separate mode

        // Step 3: blit outputs into contiguous buffer for combine
        // In gather mode: buf_gather_out already holds [K×DIM] contiguous
        // In gather6 mode: down kernel wrote directly into buf_expert_contiguous
        // In separate mode: blit from 6 separate buffers
        if (used_gather6) {
            // no copy needed
        } else if (eng->gather_mode && eng->buf_gather_gate_W[layer_idx]) {
            // Gather mode: buf_gather_out is already [K×DIM] contiguous
            // Copy it to buf_expert_contiguous so combine kernel works uniformly
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromBuffer:(id<MTLBuffer>)eng->buf_gather_out
                   sourceOffset:0
                       toBuffer:(id<MTLBuffer>)eng->buf_expert_contiguous
              destinationOffset:0
                           size:(size_t)K*DIM*sizeof(float)];
            [blit endEncoding];
        } else {
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            id<MTLBuffer> contiguous = (id<MTLBuffer>)eng->buf_expert_contiguous;
            for (int k = 0; k < K; k++) {
                [blit copyFromBuffer:(id<MTLBuffer>)eng->buf_expert_out[k]
                       sourceOffset:0
                           toBuffer:contiguous
                  destinationOffset:(size_t)k * DIM * sizeof(float)
                               size:(size_t)DIM * sizeof(float)];
            }
            [blit endEncoding];
        }

        // Step 4: moe_combine in the SAME CB
        // Lazily create per-layer engine-lifetime buffers for the combine kernel
        // (hoisted from per-token allocations; contents written below before encode).
        if (!eng->buf_weights[layer_idx])
            eng->buf_weights[layer_idx] = (void *)[d newBufferWithLength:(size_t)N_ACTIVE*sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
        if (!eng->buf_zero_resid[layer_idx])
            eng->buf_zero_resid[layer_idx] = (void *)[d newBufferWithLength:(size_t)DIM*sizeof(float)
                                                                     options:MTLResourceStorageModeShared];
        memcpy([(id<MTLBuffer>)eng->buf_weights[layer_idx] contents], expert_weights, (size_t)K*sizeof(float));
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:eng->pipe_moe_combine];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_expert_contiguous offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_weights[layer_idx]   offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_zero_resid[layer_idx] offset:0 atIndex:2];
            [enc setBuffer:eng->buf_hidden offset:0 atIndex:3];
            uint kv = K, hd = DIM;
            [enc setBytes:&kv length:4 atIndex:4];
            [enc setBytes:&hd length:4 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((DIM+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            telemetry_record_moe_dispatch(
                eng, layer_idx, (DIM + 255) / 256, 256, DIM);
            [enc endEncoding];
        }

        if (time_en) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); t1 = ts.tv_sec*1e9+ts.tv_nsec; }
#if DMLX_GPU_TIMESTAMP_PROBE
        const bool gpu_timestamp_recording =
            moe_infer_gpu_timestamp_probe_recording_enabled(eng);
        const uint64_t gpu_pre_commit_ns = gpu_timestamp_recording
            ? moe_infer_monotonic_ns() : 0u;
#endif
        const uint64_t timeline_routed_encode_done = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        const uint64_t timeline_routed_pre_commit = timeline_routed_encode_done;
        [cb commit];
        const uint64_t timeline_routed_post_commit = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
#if DMLX_GPU_TIMESTAMP_PROBE
        const uint64_t gpu_post_commit_ns = gpu_timestamp_recording
            ? moe_infer_monotonic_ns() : 0u;
#endif
        [cb waitUntilCompleted];
        const uint64_t timeline_routed_post_wait = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        if (decode_timeline_recording) decode_timeline_record(
            eng, (uint16_t)layer_idx, DMLX_DECODE_TIMELINE_ROUTED_MOE_CB,
            timeline_routed_begin, timeline_routed_encode_done,
            timeline_routed_pre_commit, timeline_routed_post_commit,
            timeline_routed_post_wait, timeline_routed_post_wait);
#if DMLX_GPU_TIMESTAMP_PROBE
        const uint64_t gpu_post_wait_ns = gpu_timestamp_recording
            ? moe_infer_monotonic_ns() : 0u;
        if (gpu_timestamp_recording) {
            gpu_timestamp_probe_record_completed_cb(
                eng, DMLX_GPU_CB_ROUTED_MOE, (uint16_t)layer_idx,
                token_index, gpu_pre_commit_ns, gpu_post_commit_ns,
                gpu_post_wait_ns, cb);
        }
#endif
        if (time_en) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); t2 = ts.tv_sec*1e9+ts.tv_nsec;
            fprintf(stderr, "[MOE-TIME] encode=%.2fms gpu=%.2fms\n", (t1-t0)/1e6, (t2-t1)/1e6); }
    } // end expert block

    #undef EXPERT_BUF
    return 0;
}

// ============================================================================
// Top-K routing on CPU (softmax + topK)
// ============================================================================

// MLX routing: sqrtsoftplus scoring, topK selection, L1-normalize, scale by route_scale.
// Matches DSV4Gate.forward (scoring_func=sqrtsoftplus, norm_topk_prob=true, route_scale=1.5).
// bias: optional [N_EXPERTS] e_score_correction_bias (NULL for hash layers 0-2).
// cache_mask: optional [N_EXPERTS] bool — if non-NULL and smelt_warmup_done, adds penalty
//             to uncached experts to steer routing toward cached ones (SMELT).
static void cpu_moe_route_flags(
    const float *logits,
    const float *bias,
    int n,
    int K,
    int *out_indices,
    float *out_weights,
    const uint8_t *cached,
    float smelt_penalty) {
    float *scores = (float *)alloca((size_t)n * sizeof(float));
    float *choice = (float *)alloca((size_t)n * sizeof(float));
    uint8_t *taken = (uint8_t *)alloca((size_t)n * sizeof(uint8_t));
    memset(taken, 0, (size_t)n * sizeof(uint8_t));

    for (int i = 0; i < n; i++) {
        const float l = logits[i] + (bias ? bias[i] : 0.0f);
        const float sp = l > 0.0f
            ? l + log1pf(expf(-l))
            : log1pf(expf(l));
        scores[i] = sqrtf(sp);
        choice[i] = scores[i] - (cached && !cached[i] ? smelt_penalty : 0.0f);
    }
    // Strict `>` preserves the first (lowest expert ID) on equal scores.
    for (int k = 0; k < K; k++) {
        int best = -1;
        float best_value = -INFINITY;
        for (int i = 0; i < n; i++) {
            if (!taken[i] && choice[i] > best_value) {
                best_value = choice[i];
                best = i;
            }
        }
        out_indices[k] = best;
        out_weights[k] = scores[best];
        taken[best] = 1u;
    }
    float weight_sum = 0.0f;
    for (int k = 0; k < K; k++) weight_sum += out_weights[k];
    weight_sum += 1e-20f;
    for (int k = 0; k < K; k++) {
        out_weights[k] = out_weights[k] / weight_sum * 1.5f;
    }
}

static void cpu_moe_route(const float *logits, const float *bias, int n, int K,
                          int *out_indices, float *out_weights,
                          const uint8_t *const *cache_ptr, float smelt_penalty) {
    uint8_t *cached = NULL;
    if (cache_ptr) {
        cached = (uint8_t *)alloca((size_t)n * sizeof(uint8_t));
        for (int i = 0; i < n; ++i) cached[i] = cache_ptr[i] ? 1u : 0u;
    }
    cpu_moe_route_flags(
        logits, bias, n, K, out_indices, out_weights, cached, smelt_penalty);
}

// ============================================================================
// CPU RoPE — DeepSeek V4 partial RoPE with YaRN (mode==2, neox-style pairing).
// Adapted from ds4/metal/dsv4_rope.metal algorithm.
// ============================================================================

#define ROPE_DIM 64         // qk_rope_head_dim
#define ROPE_THETA 10000000.0f
#define ROPE_N_CTX_ORIG 65536

static void apply_rope_tail(float *q_or_k, int dim, int n_nope, int pos, float freq_scale) {
    float corr_dims[2];
    {
        float beta_fast = 32.0f, beta_slow = 1.0f;
        float c0 = ROPE_DIM * logf(ROPE_N_CTX_ORIG / (beta_fast * 2.0f * M_PI)) / (2.0f * logf(ROPE_THETA));
        float c1 = ROPE_DIM * logf(ROPE_N_CTX_ORIG / (beta_slow * 2.0f * M_PI)) / (2.0f * logf(ROPE_THETA));
        corr_dims[0] = fmaxf(0.0f, floorf(c0));
        corr_dims[1] = fminf(ROPE_DIM - 1.0f, ceilf(c1));
    }

    float theta_base = (float)pos;
    float inv_ndims = -1.0f / ROPE_DIM;
    int n_half = ROPE_DIM / 2;

    for (int ic = 0; ic < n_half; ic++) {
        int rel_i0 = 2 * ic;
        float theta = theta_base * powf(ROPE_THETA, inv_ndims * rel_i0);

        // YaRN
        float ramp_mix = 0.0f;
        float low = corr_dims[0], high = corr_dims[1];
        float yarn_ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, ((float)rel_i0 / 2.0f - low) / fmaxf(0.001f, high - low)));
        theta = theta * (1.0f - yarn_ramp) + theta / freq_scale * yarn_ramp;

        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        int j0 = n_nope + ic;
        int j1 = n_nope + ic + n_half;
        float x0 = q_or_k[j0];
        float x1 = q_or_k[j1];
        q_or_k[j0] = x0 * cos_t - x1 * sin_t;
        q_or_k[j1] = x0 * sin_t + x1 * cos_t;
    }
}

// ============================================================================
// Main forward pass (1 token)
// ============================================================================

// Encode a float matvec: out = W @ x. W: [out_dim, in_dim], x: [in_dim]
static void encode_matvec(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
                          id<MTLBuffer> W_buf, id<MTLBuffer> x_buf, id<MTLBuffer> out_buf,
                          uint out_dim, uint in_dim) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:W_buf offset:0 atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBytes:&out_dim length:4 atIndex:3];
    [enc setBytes:&in_dim length:4 atIndex:4];
    [enc dispatchThreads:MTLSizeMake(out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc endEncoding];
}

// Encode RMSNorm: out = rms_norm(x, weight)
static void encode_rms_norm(id<MTLCommandBuffer> cb, MoEInferEngine *eng,
                            id<MTLBuffer> x_buf, id<MTLBuffer> weight_buf, id<MTLBuffer> out_buf) {
    // Step 1: sum of squares
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:eng->pipe_rms_norm_sum_sq];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:eng->buf_norm_sum_sq offset:0 atIndex:1];
        uint dim = DIM;
        [enc setBytes:&dim length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }
    // Step 2: apply normalization
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:eng->pipe_rms_norm_apply];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:weight_buf offset:0 atIndex:1];
        [enc setBuffer:eng->buf_norm_sum_sq offset:0 atIndex:2];
        [enc setBuffer:out_buf offset:0 atIndex:3];
        uint dim = DIM;
        float eps = 1e-6f;
        [enc setBytes:&dim length:4 atIndex:4];
        [enc setBytes:&eps length:4 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }
}

static MlaPipes moe_infer_mla_pipes(MoEInferEngine *eng) {
    return (MlaPipes){
        .dev = (id<MTLDevice>)eng->device,
        .queue = (id<MTLCommandQueue>)eng->queue,
        .dequant_matvec_affine = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine,
        .dequant_matvec_affine_v2 = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_v2,
        .rms_norm_rows = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows,
        .rope_tail_interleaved = (id<MTLComputePipelineState>)eng->pipe_rope_tail,
        .mla_sdpa_decode = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa,
        .mla_sdpa_decode_f16 = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_f16,
        .matvec_f32 = (id<MTLComputePipelineState>)eng->pipe_matvec,
        .dequant_matvec_affine_f16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_f16out,
        .rms_norm_rows_f16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_f16out,
        .dequant_matvec_affine_f16in_f16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_f16in_f16out,
        .rms_norm_rows_f16in_f16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_f16in_f16out,
        .rope_tail_interleaved_f16 = (id<MTLComputePipelineState>)eng->pipe_rope_tail_f16,
        .matvec_f32_f16in = (id<MTLComputePipelineState>)eng->pipe_matvec_f32_f16in,
        .mla_sdpa_decode_f16in_f16out = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_f16in_f16out,
        .dequant_matvec_affine_bf16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16out,
        .rms_norm_rows_bf16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_bf16out,
        .dequant_matvec_affine_bf16in_bf16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_bf16out,
        .dequant_matvec_affine_bf16in_bf16out_v2 = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_bf16out_v2,
        .rms_norm_rows_bf16in_bf16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_bf16in_bf16out,
        .rope_tail_interleaved_bf16 = (id<MTLComputePipelineState>)eng->pipe_rope_tail_bf16,
        .matvec_f32_bf16in = (id<MTLComputePipelineState>)eng->pipe_matvec_f32_bf16in,
        .matvec_q8_0_f32 = (id<MTLComputePipelineState>)eng->pipe_matvec_q8_0_f32,
        .mla_sdpa_decode_bfloat = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_bfloat,
        .dequant_matvec_affine_bf16in_f32out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_f32out,
        .dequant_matvec_affine_bf16in_f32out_grp8 = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_f32out_grp8,
        .mla_sdpa_prefill_bfloat = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_prefill_bfloat,
        .mla_sdpa_batch_f16_kv = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_batch_f16_kv,
        .bf16_to_f16_row = (id<MTLComputePipelineState>)eng->pipe_bf16_to_f16_row,
        .f32_to_bf16_vec = (id<MTLComputePipelineState>)eng->pipe_f32_to_bf16_vec,
        .mhc_post_bfloat = (id<MTLComputePipelineState>)eng->pipe_mhc_post_bfloat,
        .bf16_to_f32_vec = (id<MTLComputePipelineState>)eng->pipe_bf16_to_f32_vec,
    };
}

typedef struct NativePreparedMoEToken {
    float *normed;
    float *routing_scores;
    float *residual;
    float *post_ffn;
    float *comb_ffn;
    int token_id;
    int position;
    int kv_length;
    uint32_t compressor_length;
    uint32_t indexer_length;
    uint64_t kv_state_digest;
    uint64_t compressor_state_digest;
    uint64_t indexer_state_digest;
} NativePreparedMoEToken;

static inline void emit_prepared_debug_capture(
    MoEInferEngine *eng,
    const NativePreparedMoEToken *prepared,
    MoeInferCaptureCheckpoint checkpoint,
    MoeInferCaptureDType dtype,
    int layer,
    const float *values,
    size_t value_count,
    const int *expert_ids,
    size_t expert_count) {
    if (!eng || !prepared || !eng->debug_capture) return;
    const MoeInferDebugCaptureRecord record = {
        .path = MOE_INFER_CAPTURE_BATCH,
        .checkpoint = checkpoint,
        .dtype = dtype,
        .layer = layer,
        .position = prepared->position,
        .kv_length = prepared->kv_length,
        .compressor_length = prepared->compressor_length,
        .indexer_length = prepared->indexer_length,
        .kv_state_digest = prepared->kv_state_digest,
        .compressor_state_digest = prepared->compressor_state_digest,
        .indexer_state_digest = prepared->indexer_state_digest,
        .values = values,
        .value_count = value_count,
        .expert_ids = expert_ids,
        .expert_count = expert_count,
    };
    eng->debug_capture(eng->debug_capture_context, &record);
}

static int moe_infer_forward_layer_impl(
    MoEInferEngine *eng,
    int layer,
    float *hidden,
    int pos,
    NativePreparedMoEToken *prepared) {
    if (!eng->initialized) return -1;
    const bool exact_q4_recording = moe_infer_exact_q4_recording_enabled(eng);
    const bool decode_timeline_recording =
        moe_infer_decode_timeline_recording_enabled(eng);
    const uint64_t decode_timeline_layer_begin = decode_timeline_recording
        ? moe_infer_decode_timeline_now_ns() : 0u;
    uint64_t timeline_attention_encode_done = decode_timeline_layer_begin;
    uint64_t timeline_attention_pre_commit = decode_timeline_layer_begin;
    uint64_t timeline_attention_post_commit = decode_timeline_layer_begin;
    uint64_t timeline_attention_post_wait = decode_timeline_layer_begin;
    DmlxExactQ4PendingLayer exact_q4_layer = {0};
    exact_q4_layer_begin(eng, &exact_q4_layer, layer);
    const uint64_t exact_q4_layer_begin = exact_q4_recording
        ? moe_infer_monotonic_ns() : 0u;
    uint64_t exact_q4_attention_encode_done = exact_q4_layer_begin;
    uint64_t exact_q4_attention_done = exact_q4_layer_begin;
    uint64_t exact_q4_routing_done = exact_q4_layer_begin;
    uint64_t exact_q4_moe_begin = exact_q4_layer_begin;
    uint64_t exact_q4_moe_encode_begin = exact_q4_layer_begin;
    uint64_t exact_q4_moe_encode_done = exact_q4_layer_begin;
    DmlxExactQ4StageEvent exact_q4_cache_event = {0};
    DmlxExactQ4StageEvent exact_q4_host_events[2] = {0};
    uint32_t exact_q4_host_event_count = 0u;
    DmlxMlaPathIdentity mla_identity = {0};
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    const int phase_time = (getenv("NATIVE_PHASE_TIME") != NULL);
    double pt0=0, pt1=0, pt2=0, pt3=0, pt4=0, pt5=0;
    if (phase_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); pt0 = ts.tv_sec*1e9+ts.tv_nsec; }

    // `hidden` is the mHC-expanded residual: [MHC_MULT, DIM] contiguous, in place.
    float *residual = hidden;

    // Track whether buf_residual_gpu already holds the correct residual for
    // this layer (written by the previous layer's deferred cb3 on the GPU).
    // If deferred gpu_combined was true → GPU wrote buf_residual_gpu → no sync.
    // In all other cases (layer 0 of a token, batch path after clear, embed was
    // the last writer) → residual has data that is NOT yet in buf_residual_gpu
    // → must sync.
    bool gpu_residual_current = false;

    // === Drain deferred cb3 handle from previous layer (deferred mhc_post_ffn) ===
    // Production path: NO waitUntilCompleted, NO readback. cb3(N-1) wrote
    // buf_residual_gpu; the command queue is serial, so it completes before any
    // of this layer's CBs execute — the GPU buffer is already the correct
    // residual and CPU residual[] is intentionally left stale. Configurations
    // that need the residual on the CPU (debug_capture / MF_DBG / DSpark
    // accumulate on layers 40-42) drain synchronously at the cb3 commit site
    // and never arm the deferred slot. The batch scalar-fallback loop drains
    // the deferred slot itself (wait + readback into the previous token's
    // hidden) before calling into this function.
    if (eng->deferred.active && eng->deferred.cmd_experts) {
        [(id<MTLCommandBuffer>)eng->deferred.cmd_experts release];
        eng->deferred.cmd_experts = NULL;
        eng->deferred.active = false;
        if (eng->deferred.gpu_combined) {
            eng->deferred.gpu_combined = false;
            gpu_residual_current = true;  // buf_residual_gpu already has correct data
        }
    }

    emit_debug_capture(
        eng,
        MOE_INFER_CAPTURE_LAYER_INPUT,
        MOE_INFER_CAPTURE_DTYPE_F32,
        layer,
        pos,
        residual,
        (size_t)MHC_MULT * DIM,
        NULL,
        0);

    // Sync CPU residual → buf_residual_gpu only when needed (not already current).
    // Skip when the previous layer's deferred cb3 left the correct data in the
    // GPU buffer. Always sync for: layer 0 of a token (no deferred), the batch
    // path (deferred cleared without readback), or when embed was the last writer.
    if (!gpu_residual_current && eng->buf_residual_gpu) {
        memcpy([(id<MTLBuffer>)eng->buf_residual_gpu contents],
               residual, (size_t)MHC_MULT * DIM * sizeof(float));
    }
    double ptA=0, ptB=0, ptC=0, gpu_exec_ms=0;
    if (phase_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ptA = ts.tv_sec*1e9+ts.tv_nsec; }

    {    // Build MlaPipes view over the engine's pipelines.
    MlaPipes P;
    P.dev = d;
    P.queue = (id<MTLCommandQueue>)eng->queue;
    P.dequant_matvec_affine = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine;
    P.dequant_matvec_affine_v2 = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_v2;
    P.rms_norm_rows = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows;
    P.rope_tail_interleaved = (id<MTLComputePipelineState>)eng->pipe_rope_tail;
    P.mla_sdpa_decode = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa;
    P.mla_sdpa_decode_f16 = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_f16;
    P.matvec_f32 = (id<MTLComputePipelineState>)eng->pipe_matvec;
    // F16 precision chain (validated correct)
    P.dequant_matvec_affine_f16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_f16out;
    P.rms_norm_rows_f16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_f16out;
    P.dequant_matvec_affine_f16in_f16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_f16in_f16out;
    P.rms_norm_rows_f16in_f16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_f16in_f16out;
    P.rope_tail_interleaved_f16 = (id<MTLComputePipelineState>)eng->pipe_rope_tail_f16;
    P.matvec_f32_f16in = (id<MTLComputePipelineState>)eng->pipe_matvec_f32_f16in;
    P.mla_sdpa_decode_f16in_f16out = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_f16in_f16out;
    // BF16 precision chain — matches MLX training precision
    P.dequant_matvec_affine_bf16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16out;
    P.rms_norm_rows_bf16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_bf16out;
    P.dequant_matvec_affine_bf16in_bf16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_bf16out;
    P.dequant_matvec_affine_bf16in_bf16out_v2 = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_bf16out_v2;
    P.rms_norm_rows_bf16in_bf16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_bf16in_bf16out;
    P.rope_tail_interleaved_bf16 = (id<MTLComputePipelineState>)eng->pipe_rope_tail_bf16;
    P.matvec_f32_bf16in = (id<MTLComputePipelineState>)eng->pipe_matvec_f32_bf16in;
    P.matvec_q8_0_f32    = (id<MTLComputePipelineState>)eng->pipe_matvec_q8_0_f32;
    P.mla_sdpa_decode_bfloat = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_bfloat;
    P.dequant_matvec_affine_bf16in_f32out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_f32out;
    P.dequant_matvec_affine_bf16in_f32out_grp8 = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_f32out_grp8;
    P.mla_sdpa_prefill_bfloat = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_prefill_bfloat;
    P.mla_sdpa_batch_f16_kv = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_batch_f16_kv;
    P.bf16_to_f16_row = (id<MTLComputePipelineState>)eng->pipe_bf16_to_f16_row;
    P.f32_to_bf16_vec = (id<MTLComputePipelineState>)eng->pipe_f32_to_bf16_vec;
    P.mhc_post_bfloat = (id<MTLComputePipelineState>)eng->pipe_mhc_post_bfloat;
    P.bf16_to_f32_vec = (id<MTLComputePipelineState>)eng->pipe_bf16_to_f32_vec;

    // Large scratch buffers are static (forward_layer runs serially).
    static float attn_input[DIM], normed[DIM], attn_out[DIM];
    static float ffn_input[DIM], ffn_out[DIM];
    static uint16_t normed_bf16_direct[DIM];
    float post[MHC_MULT], comb[MHC_MULT * MHC_MULT];          // attn mHC post/comb (for cb3)
    float post_ffn[MHC_MULT], comb_ffn[MHC_MULT * MHC_MULT];  // ffn mHC post/comb (deferred cb3)
    bool shared_done_early = false;
    const bool use_hash_routing = (eng->tid2eid[layer] != NULL && eng->current_token_id >= 0);
    bool use_gpu_routing = false;  // set in CMD2 enc7 when SMELT warm + score-based layer
    const bool prepared_uses_gpu_routing = prepared && !use_hash_routing &&
        eng->smelt_warmup_done && eng->pipe_moe_route_gpu &&
        eng->gate_proj[layer];

    // === Attention sublayer — GPU mhc_pre bfloat computation ===
    if (layer == 0 && getenv("MF_DBG")) {
        double rn=0; for(int z=0;z<MHC_MULT*DIM;z++) rn+=(double)residual[z]*residual[z];
        fprintf(stderr, "[mf-dbg] L0 pos=%d in residual norm=%.4f res[0]=%.6f res[DIM]=%.6f\n", pos, sqrt(rn), residual[0], residual[DIM]);
        const char *dd = getenv("DSV4_DUMP_DIR");
        if (dd) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/L0_residual_metal.bin", dd);
            FILE *f = fopen(path, "wb"); if (f) { fwrite(residual, sizeof(float), MHC_MULT*DIM, f); fclose(f); }
        }
    }
    // === CB-A + CB1 merged: mhc_pre + Q/KV/SDPA in ONE command buffer (1 wait) ===
    // Encoder 1: mhc_pre_split_weighted_sum_norm → post/comb/normed_bf16/attn_input on GPU
    // Encoders 2..N: Q chain + KV chain + SDPA (reads normed_bf16 from GPU, no CPU roundtrip)
    // Eliminates CB-A wait (32ms/token) by merging two CB boundaries into one.
    KVCache *kvc = &eng->kv_cache[layer];
    if (!kvc->kv) {
        const size_t kvc_size =
            (size_t)MAX_SEQ_LEN * KV_LORA_RANK * sizeof(uint16_t);
        const uint64_t allocation_begin = exact_q4_recording
            ? moe_infer_monotonic_ns() : 0u;
        id<MTLBuffer> kvbuf = [(id<MTLDevice>)eng->device
            newBufferWithLength:kvc_size options:MTLResourceStorageModeShared];
        uint16_t *kv_contents = kvbuf ? (uint16_t *)[kvbuf contents] : NULL;
        if (!kvbuf || !kv_contents) return -1;
        kvc->kv_gpu_buf = (void *)kvbuf;
        kvc->kv = kv_contents;
        memset(kvc->kv, 0, kvc_size);
        kvc->len = 0;
        if (exact_q4_recording) {
            exact_q4_host_events[exact_q4_host_event_count++] =
                (DmlxExactQ4StageEvent){
                    .layer = (uint16_t)layer,
                    .stage = DMLX_EXACT_Q4_STAGE_HOST_ALLOC_WRAPPER,
                    .flags = DMLX_EXACT_Q4_EVENT_CPU_ACTIVE,
                    .calls = 1u,
                    .begin_ns = allocation_begin,
                    .end_ns = moe_infer_monotonic_ns(),
                    .bytes = kvc_size,
                };
        }
    }
    if (!kvc->kv_gpu_buf || kvc->len >= MAX_SEQ_LEN) return -1;
    const int previous_kvc_len = kvc->len;
    kvc->len += 1;
    // Scalar decode production path: NO waitUntilCompleted on merged_cb.
    // Its outputs stay GPU-resident — attn_out is truncated to bf16 by the
    // f32_to_bf16_trunc_vec encoder at the head of cb2cmd2 (same queue,
    // in-order), and the attn-normed vector is read back at the cb2cmd2
    // routing wait below (zero extra syncs) to feed the compressor/indexer.
    // Consumers that still need these values on the CPU at the merged_cb
    // point drain synchronously: the batch/prepared path (unchanged
    // behavior), MF_STAGE_TIME (staged CBs already synchronized),
    // debug_capture (ATTENTION_OUTPUT record), and MF_DBG dumps.
    const bool merged_cb_cpu_drain =
        prepared != NULL || getenv("MF_STAGE_TIME") ||
        eng->debug_capture || getenv("MF_DBG");
    {
        id<MTLCommandBuffer> merged_cb = [(id<MTLCommandQueue>)eng->queue commandBufferWithUnretainedReferences];
        if (!merged_cb) {
            kvc->len = previous_kvc_len;
            return -1;
        }

        // Encoder 1: mhc_pre (attn) — writes normed_bf16 to GPU, consumed by Q/KV chain below
        {
            id<MTLComputeCommandEncoder> enc =
                moe_infer_attention_stage_recording_enabled(eng)
                    ? (id<MTLComputeCommandEncoder>)moe_infer_attention_stage_encoder(
                        eng, (void *)merged_cb, DMLX_ATTENTION_STAGE_MHC_PRE)
                    : [merged_cb computeCommandEncoder];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_mhc_pre_split_weighted_sum_norm];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_attn_hc_fn[layer]       offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_attn_hc_base[layer]     offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_attn_hc_scale[layer]    offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu            offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_weights        offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_comb_weights        offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_attn_in             offset:0 atIndex:6];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_input_norm_gpu[layer]   offset:0 atIndex:7];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_attn_norm_bf16      offset:0 atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
        }

        // Encoders 2..N: Q/KV/SDPA via external CB1 (reads buf_mhc_attn_norm_bf16 from GPU)
        // Uses kv_cache GPU path to keep everything in the same CB (no CPU roundtrip).
        // This call encodes into merged_cb and does NOT commit or wait.

        // MF_STAGE_TIME: commit the mhc_pre-only CB now (mla_attention will use
        // its own staged CBs and must not run ahead of mhc_pre's output).
        const int stage_time = getenv("MF_STAGE_TIME") != NULL;
        if (stage_time) {
            double s0 = 0;
            { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); s0 = ts.tv_sec*1e9+ts.tv_nsec; }
            [merged_cb commit]; [merged_cb waitUntilCompleted];
            { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
              fprintf(stderr, "[STAGE] L%d mhc_pre=%.2f\n", layer, (ts.tv_sec*1e9+ts.tv_nsec-s0)/1e6); }
        }

        int attn_ret = mla_attention_decode_bf16(&P, &eng->attn[layer],
            NULL, kvc->kv, kvc->len, pos, attn_out,
            kvc->kv_gpu_buf,                                    // GPU KV path
            eng->buf_mhc_attn_norm_bf16,                        // x_gpu_buf (from encoder 1)
            (void *)merged_cb,                                  // external_cb1 (mla_attention stages internally under MF_STAGE_TIME)
            eng->buf_attn_out,                                  // out_gpu_buf — wo_b writes here
            &mla_identity,
            eng);
        if (attn_ret != 0) {
            kvc->len = previous_kvc_len;
            return -1;
        }
        if (exact_q4_recording)
            exact_q4_attention_encode_done = moe_infer_monotonic_ns();
        if (!stage_time) {
        // merged_cb now contains: mhc_pre + Q/KV/SDPA + GPU blit wo_a grouping + wo_a×8 + wo_b
#if DMLX_GPU_TIMESTAMP_PROBE
        const bool gpu_timestamp_recording =
            moe_infer_gpu_timestamp_probe_recording_enabled(eng);
        const uint64_t gpu_pre_commit_ns = gpu_timestamp_recording
            ? moe_infer_monotonic_ns() : 0u;
#endif
        timeline_attention_encode_done = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        timeline_attention_pre_commit = timeline_attention_encode_done;
        [merged_cb commit];
        timeline_attention_post_commit = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
#if DMLX_GPU_TIMESTAMP_PROBE
        const uint64_t gpu_post_commit_ns = gpu_timestamp_recording
            ? moe_infer_monotonic_ns() : 0u;
#endif
        if (phase_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ptB = ts.tv_sec*1e9+ts.tv_nsec; }
        // phase_time / gpu_timestamp_recording read GPUEndTime below and also
        // require completion; they are diagnostic-only modes.
#if DMLX_GPU_TIMESTAMP_PROBE
        if (merged_cb_cpu_drain || phase_time || gpu_timestamp_recording) {
#else
        if (merged_cb_cpu_drain || phase_time) {
#endif
            [merged_cb waitUntilCompleted];
        }
        timeline_attention_post_wait = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        if (moe_infer_attention_stage_recording_enabled(eng))
            moe_infer_attention_stage_resolve(eng, &mla_identity);
#if DMLX_GPU_TIMESTAMP_PROBE
        const uint64_t gpu_post_wait_ns = gpu_timestamp_recording
            ? moe_infer_monotonic_ns() : 0u;
        if (gpu_timestamp_recording) {
            gpu_timestamp_probe_record_completed_cb(
                eng, DMLX_GPU_CB_ATTENTION, (uint16_t)layer,
                pos >= 0 ? (uint32_t)pos : 0u, gpu_pre_commit_ns,
                gpu_post_commit_ns, gpu_post_wait_ns, merged_cb);
        }
#endif
        if (phase_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ptC = ts.tv_sec*1e9+ts.tv_nsec; }
        if (phase_time) gpu_exec_ms = ([merged_cb GPUEndTime] - [merged_cb GPUStartTime]) * 1e3;
        } else {
            if (phase_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ptB = ptC = ts.tv_sec*1e9+ts.tv_nsec; }
        }

        // CPU reads after merged CB completes (drain paths only — the scalar
        // production path keeps all three values GPU-resident):
        // - attn_out (debug_capture ATTENTION_OUTPUT; GPU trunc kernel writes
        //   buf_mhc_attn_out_bf16 on the production path)
        // - attn_input (diagnostics only; no CPU consumer)
        // - attn-normed → normed (compressor/indexer at the old site below)
        if (merged_cb_cpu_drain) {
            memcpy(attn_out, [(id<MTLBuffer>)eng->buf_attn_out contents], DIM * sizeof(float));
            if (phase_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); pt1 = ts.tv_sec*1e9+ts.tv_nsec; }
            memcpy(attn_input, [(id<MTLBuffer>)eng->buf_mhc_attn_in     contents], DIM * sizeof(float));
            uint16_t *bf16_out = (uint16_t *)[(id<MTLBuffer>)eng->buf_mhc_attn_norm_bf16 contents];
            memcpy(normed_bf16_direct, bf16_out, DIM * sizeof(uint16_t));
            for (int i = 0; i < DIM; i++) {
                uint32_t u = ((uint32_t)normed_bf16_direct[i]) << 16;
                memcpy(&normed[i], &u, 4);
            }
        }
        // post/comb NOT read here — CMD2 reads buf_mhc_post/comb_weights directly from GPU
        if (exact_q4_recording) {
            exact_q4_attention_done = moe_infer_monotonic_ns();
            // The production path no longer waits on merged_cb, so the two
            // timestamps can land on the same clock tick. The validator
            // requires positive stage durations — floor at +1ns.
            if (exact_q4_attention_done <= exact_q4_attention_encode_done)
                exact_q4_attention_done = exact_q4_attention_encode_done + 1u;
        }
        if (decode_timeline_recording && !stage_time) {
            const uint64_t timeline_attention_end = moe_infer_decode_timeline_now_ns();
            decode_timeline_record(eng, (uint16_t)layer,
                DMLX_DECODE_TIMELINE_ATTENTION_CB,
                decode_timeline_layer_begin, timeline_attention_encode_done,
                timeline_attention_pre_commit, timeline_attention_post_commit,
                timeline_attention_post_wait, timeline_attention_end);
        }
    }

    // DSA_CPU stage: CPU compressor/indexer bookkeeping plus the small
    // truncation/conversion loops below, up to the CB2+CMD2 command buffer.
    const uint64_t timeline_dsa_begin = decode_timeline_recording
        ? moe_infer_decode_timeline_now_ns() : 0u;

    // Compressor step (before attention, for layers with compress_ratio > 0)
    // Note: now runs AFTER merged CB-A+CB1. Compressor state updates are for future tokens.
    // Drain paths (batch/prepared, MF_STAGE_TIME, debug_capture, MF_DBG) keep the
    // original program point with the attn-normed readback above. The scalar
    // production path runs these steps at the cb2cmd2 routing wait instead —
    // buf_mhc_attn_norm_bf16 is read back there anyway-adjacent (zero extra
    // syncs) and both steps only mutate future-token state (comp_state), which
    // no consumer touches between the two program points within this token.
    if (merged_cb_cpu_drain) {
        if (eng->compress_ratio[layer] > 0) {
            moe_infer_compressor_step(eng, layer, pos, normed);
        }

        // Indexer step (ratio=4 layers only — select which comp blocks to attend)
        static bool comp_allowed[MAX_COMP_BLOCKS];
        bool has_comp_selection = false;
        if (eng->compress_ratio[layer] == 4 && eng->comp_state[layer].n_comp > 0) {
            has_comp_selection = moe_infer_indexer_step(eng, layer, pos, normed, NULL, comp_allowed);
        }
        (void)has_comp_selection;
    }

    // Truncate attn_out to bf16 for the debug capture below (matches MLX
    // precision). Only the capture consumes the CPU copy — the GPU trunc
    // kernel at the head of cb2cmd2 produces buf_mhc_attn_out_bf16
    // bit-identically (u >> 16) on every path.
    if (eng->debug_capture) {
        for (int i = 0; i < DIM; i++) {
            uint32_t u; memcpy(&u, &attn_out[i], 4); u &= 0xFFFF0000U; memcpy(&attn_out[i], &u, 4);
        }
    }
    emit_debug_capture(
        eng,
        MOE_INFER_CAPTURE_ATTENTION_OUTPUT,
        MOE_INFER_CAPTURE_DTYPE_BF16,
        layer,
        pos,
        attn_out,
        DIM,
        NULL,
        0);

    // === CB2+CMD2 merged: mhc_post(attn) + mhc_pre(ffn) + ffn_norm + routing in ONE CB ===
    // No CPU work between cb2 and CMD2 — buf_residual_gpu flows GPU→GPU.
    // CB2's post/comb outputs are NOT read by CPU (CMD2 overwrites them immediately).
    // Saves 1 GPU sync/layer = ~13ms × 43 = ~560ms/token.
    {
        // attn_out → bf16 now runs on the GPU: the f32_to_bf16_trunc_vec
        // encoder at the head of cb2cmd2 below reads buf_attn_out (wo_b
        // output, committed in merged_cb earlier on this serial queue) and
        // writes buf_mhc_attn_out_bf16 with bit-exact truncation (u >> 16),
        // replacing the former CPU readback + truncate + upload.
        // NOTE: post/comb NOT written here — CMD2 (mhc_pre_ffn) overwrites buf_mhc_post_weights/comb
        // so any CPU memcpy of post/comb before CMD2 is wasted work.

        const uint64_t timeline_cb2_begin = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        if (decode_timeline_recording) {
            decode_timeline_record(eng, (uint16_t)layer,
                DMLX_DECODE_TIMELINE_DSA_CPU,
                timeline_dsa_begin, timeline_dsa_begin, timeline_dsa_begin,
                timeline_dsa_begin, timeline_dsa_begin, timeline_cb2_begin);
        }
        id<MTLCommandBuffer> cb2cmd2 = [(id<MTLCommandQueue>)eng->queue commandBufferWithUnretainedReferences];

        // Enc 0: attn_out f32 → bf16 truncation (GPU-side, u >> 16 semantics)
        {
            id<MTLComputeCommandEncoder> e = [cb2cmd2 computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_f32_to_bf16_trunc_vec];
            [e setBuffer:(id<MTLBuffer>)eng->buf_attn_out          offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_attn_out_bf16 offset:0 atIndex:1];
            uint n = DIM;
            [e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }

        // --- CB2 section: mhc_post(attn) ---
        // Enc 1: f32→bf16 for residual
        {
            id<MTLComputeCommandEncoder> e = [cb2cmd2 computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_f32_to_bf16_vec];
            [e setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu    offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_res_bf16_in offset:0 atIndex:1];
            uint n = MHC_MULT * DIM;
            [e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(MHC_MULT*DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }
        // Enc 2: mhc_post(attn) — reads buf_mhc_post_weights/comb from CB-A output
        //        (these are the ATTN post/comb values, not yet overwritten by CMD2)
        {
            id<MTLComputeCommandEncoder> enc2 = [cb2cmd2 computeCommandEncoder];
            [enc2 setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_mhc_post_bfloat];
            [enc2 setBuffer:(id<MTLBuffer>)eng->buf_mhc_attn_out_bf16  offset:0 atIndex:0];
            [enc2 setBuffer:(id<MTLBuffer>)eng->buf_mhc_res_bf16_in    offset:0 atIndex:1];
            [enc2 setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_weights   offset:0 atIndex:2];
            [enc2 setBuffer:(id<MTLBuffer>)eng->buf_mhc_comb_weights   offset:0 atIndex:3];
            [enc2 setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_res_out   offset:0 atIndex:4];
            uint hc2 = MHC_MULT, dim2 = DIM;
            [enc2 setBytes:&hc2 length:4 atIndex:5]; [enc2 setBytes:&dim2 length:4 atIndex:6];
            [enc2 dispatchThreads:MTLSizeMake(DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc2 endEncoding];
        }
        // Enc 3: bf16→f32 writeback to buf_residual_gpu
        {
            id<MTLComputeCommandEncoder> e = [cb2cmd2 computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_bf16_to_f32_vec];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_res_out offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu     offset:0 atIndex:1];
            uint n = MHC_MULT * DIM;
            [e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(MHC_MULT*DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }

        // --- CMD2 section: mhc_pre(ffn) + ffn_norm + routing ---
        // Enc 4+5 fused: mhc_pre(ffn) + RMSNorm in ONE dispatch (same fused
        // kernel the merged_cb uses for the ATTN side). Writes ffn_in (f32),
        // post/comb weights, and normed bf16 — all former Enc4+Enc5 outputs.
        {
            id<MTLComputeCommandEncoder> enc = [cb2cmd2 computeCommandEncoder];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_mhc_pre_split_weighted_sum_norm];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_ffn_hc_fn[layer]    offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_ffn_hc_base[layer]  offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_ffn_hc_scale[layer] offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu        offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_weights    offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_comb_weights    offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_in          offset:0 atIndex:6];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_attn_norm_gpu[layer] offset:0 atIndex:7];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_norm_bf16   offset:0 atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
        }
        // Enc 6: routing_gate → buf_routing_scores
        if (eng->gate_proj[layer]) {
            id<MTLComputeCommandEncoder> enc = [cb2cmd2 computeCommandEncoder];
            // Routing gate matmul [N_EXPERTS=256, DIM=4096] × [DIM] bfloat → [N_EXPERTS] f32.
            // Use SIMD parallel dispatch if kernel available (ROWS_PER_TG=8, x_shared, simd_sum).
            // Fall back to naive 1-thread-per-row dispatch.
            if (eng->pipe_matvec_f32_bf16in_simd) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_matvec_f32_bf16in_simd];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gate_proj_gpu[layer] offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_norm_bf16    offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_routing_scores        offset:0 atIndex:2];
                uint od=N_EXPERTS, id_=DIM;
                [enc setBytes:&od length:4 atIndex:3]; [enc setBytes:&id_ length:4 atIndex:4];
                uint ntg_route = (N_EXPERTS + 7) / 8;  // 32 threadgroups
                [enc dispatchThreadgroups:MTLSizeMake(ntg_route,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_matvec_f32_bf16in];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gate_proj_gpu[layer] offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_norm_bf16    offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_routing_scores        offset:0 atIndex:2];
                uint od=N_EXPERTS, id_=DIM;
                [enc setBytes:&od length:4 atIndex:3]; [enc setBytes:&id_ length:4 atIndex:4];
                [enc dispatchThreads:MTLSizeMake(N_EXPERTS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            }
            [enc endEncoding];
        }

        // Enc 7: GPU routing — sqrtsoftplus + SMELT penalty + bitonic top-6 + L1-normalize.
        // Dispatched when SMELT warm + score-based routing + kernel compiled.
        // Reads buf_routing_scores (Enc 6 output), writes buf_gpu_route_selected/weights.
        // Update buf_cached_flags for THIS layer so routing penalty correctly reflects
        // which experts are cached in the current layer's SMELT pool.
        // Note: CPU write to a Shared MTLBuffer before commit is safe on Apple Silicon
        // (unified memory, no explicit cache flush needed).
        if (!use_hash_routing && eng->smelt_warmup_done && eng->buf_cached_flags &&
            eng->expert_mem_cache[layer] && eng->smelt_penalty > 0.0f) {
            uint8_t *flags = (uint8_t *)[(id<MTLBuffer>)eng->buf_cached_flags contents];
            for (int eid = 0; eid < N_EXPERTS; eid++) {
                flags[eid] = eng->expert_mem_cache[layer][eid] ? 1 : 0;
            }
        }
        if (!use_hash_routing && eng->smelt_warmup_done && eng->pipe_moe_route_gpu && eng->gate_proj[layer]) {
            id<MTLBuffer> bias_buf;
            if (eng->gate_bias[layer] && eng->gate_bias_gpu[layer]) {
                bias_buf = (id<MTLBuffer>)eng->gate_bias_gpu[layer];  // use persistent buffer
            } else if (eng->gate_bias[layer]) {
                // Fallback: should not happen after set_weights, but safe
                bias_buf = [(id<MTLDevice>)eng->device
                    newBufferWithBytes:eng->gate_bias[layer]
                    length:N_EXPERTS * sizeof(float)
                    options:MTLResourceStorageModeShared];
            } else {
                bias_buf = (id<MTLBuffer>)eng->buf_routing_scores; // dummy, unused
            }
            id<MTLComputeCommandEncoder> enc = [cb2cmd2 computeCommandEncoder];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_moe_route_gpu];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_routing_scores     offset:0 atIndex:0];
            [enc setBuffer:bias_buf                                    offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_cached_flags       offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_gpu_route_selected offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_gpu_route_weights  offset:0 atIndex:4];
            float penalty = eng->smelt_penalty;
            uint has_bias = (eng->gate_bias[layer] != NULL) ? 1u : 0u;
            uint has_smelt = 1u;
            [enc setBytes:&penalty   length:sizeof(float) atIndex:5];
            [enc setBytes:&has_bias  length:sizeof(uint)  atIndex:6];
            [enc setBytes:&has_smelt length:sizeof(uint)  atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
            use_gpu_routing = true;
        }

        // Enc 8+9: ffn-normed bf16 → f32 expansion into the two same-token
        // consumers, replacing the former CPU readback + expand + re-upload:
        //   buf_normed      — routed-expert input (expert CBs, committed later)
        //   buf_mhc_attn_in — shared-expert input (shared CB, committed later)
        // bf16_to_f32_vec (float(bfloat) == u<<16) is bit-identical to the CPU
        // expansion loop it replaces. buf_mhc_attn_in's previous contents
        // (mhc_pre attn_input) have no remaining consumer at this point.
        {
            id<MTLComputeCommandEncoder> e = [cb2cmd2 computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_bf16_to_f32_vec];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_norm_bf16 offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_normed            offset:0 atIndex:1];
            uint n = DIM;
            [e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }
        {
            id<MTLComputeCommandEncoder> e = [cb2cmd2 computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_bf16_to_f32_vec];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_norm_bf16 offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_attn_in       offset:0 atIndex:1];
            uint n = DIM;
            [e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }

        const uint64_t timeline_cb2_encode_done = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        const uint64_t timeline_cb2_pre_commit = timeline_cb2_encode_done;
        [cb2cmd2 commit];
        const uint64_t timeline_cb2_post_commit = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        [cb2cmd2 waitUntilCompleted];
        const uint64_t timeline_cb2_post_wait = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        if (phase_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); pt2 = ts.tv_sec*1e9+ts.tv_nsec; }

        // CPU readbacks — drain paths only (batch/prepared, MF_STAGE_TIME,
        // debug_capture, MF_DBG). On the scalar production path these values
        // stay GPU-resident: cb2cmd2's Enc3 already expanded
        // buf_mhc_post_res_out into buf_residual_gpu bit-identically, cb3
        // binds buf_mhc_post_weights/buf_mhc_comb_weights directly below, and
        // ffn_input has no CPU consumer at all. CPU residual[] is left stale
        // here; consumers that need it (debug_capture / MF_DBG / DSpark
        // layers 40-42) drain synchronously at the cb3 commit site below.
        if (merged_cb_cpu_drain) {
            uint16_t *res2_out = (uint16_t *)[(id<MTLBuffer>)eng->buf_mhc_post_res_out contents];
            for (int i = 0; i < MHC_MULT * DIM; i++) { uint32_t u = ((uint32_t)res2_out[i]) << 16; memcpy(&residual[i], &u, 4); }
            memcpy(ffn_input, [(id<MTLBuffer>)eng->buf_mhc_ffn_in      contents], DIM * sizeof(float));
            memcpy(post_ffn,  [(id<MTLBuffer>)eng->buf_mhc_post_weights contents], MHC_MULT * sizeof(float));
            memcpy(comb_ffn,  [(id<MTLBuffer>)eng->buf_mhc_comb_weights contents], MHC_MULT * MHC_MULT * sizeof(float));
        }
        // ffn-normed expansion: needed on the CPU only by the batch/prepared
        // path (prepared->normed) and MF_DBG dumps. The scalar path's
        // same-token consumers (buf_normed, buf_mhc_attn_in) were filled by
        // the Enc 8+9 GPU conversions above, bit-identically.
        if (prepared || getenv("MF_DBG")) {
            uint16_t *bf16_out = (uint16_t *)[(id<MTLBuffer>)eng->buf_mhc_ffn_norm_bf16 contents];
            memcpy(normed_bf16_direct, bf16_out, DIM * sizeof(uint16_t));
            for (int i = 0; i < DIM; i++) {
                uint32_t u = ((uint32_t)bf16_out[i]) << 16;
                memcpy(&normed[i], &u, 4);
            }
        }
        // Scalar production path: the attn-normed vector (merged_cb mhc_pre
        // output) was never read back — drain it here, at the cb2cmd2 wait
        // that stays for routing, adding zero syncs. Compressor/indexer run at
        // this point instead of their pre-cb2cmd2 site; both mutate only
        // future-token state (comp_state), and no consumer reads comp_state
        // between the old and new program points within this token (the
        // batch/prepared copies happen below, after this block).
        if (!merged_cb_cpu_drain) {
            static float attn_normed[DIM];
            const uint16_t *an = (const uint16_t *)[(id<MTLBuffer>)eng->buf_mhc_attn_norm_bf16 contents];
            for (int i = 0; i < DIM; i++) {
                uint32_t u = ((uint32_t)an[i]) << 16;
                memcpy(&attn_normed[i], &u, 4);
            }
            if (eng->compress_ratio[layer] > 0) {
                moe_infer_compressor_step(eng, layer, pos, attn_normed);
            }
            static bool comp_allowed[MAX_COMP_BLOCKS];
            bool has_comp_selection = false;
            if (eng->compress_ratio[layer] == 4 && eng->comp_state[layer].n_comp > 0) {
                has_comp_selection = moe_infer_indexer_step(eng, layer, pos, attn_normed, NULL, comp_allowed);
            }
            (void)has_comp_selection;
        }
        if (decode_timeline_recording) {
            decode_timeline_record(eng, (uint16_t)layer,
                DMLX_DECODE_TIMELINE_CB2CMD2_CB,
                timeline_cb2_begin, timeline_cb2_encode_done,
                timeline_cb2_pre_commit, timeline_cb2_post_commit,
                timeline_cb2_post_wait, moe_infer_decode_timeline_now_ns());
        }
    }

    if (layer == 0 && getenv("MF_DBG")) {
        double nn=0; for(int z=0;z<DIM;z++) nn+=(double)normed[z]*normed[z];
        fprintf(stderr, "[mf-dbg] L0 normed(ffn input) norm=%.4f\n", sqrt(nn));
        const char *dd = getenv("DSV4_DUMP_DIR");
        if (dd) {
            char path[1024]; snprintf(path, sizeof(path), "%s/L0_normed_ffn_in.bin", dd);
            FILE *f = fopen(path, "wb"); if (f) { fwrite(normed, sizeof(float), DIM, f); fclose(f); }
        }
    }

    // Routing scores ready in buf_routing_scores (written by CMD2 encoder 3)
    float *scores = (float *)[(id<MTLBuffer>)eng->buf_routing_scores contents];
    // Scalar GPU routing consumes the raw f32 gate output inside cb2cmd2,
    // before the CPU-side BF16 truncation below. Preserve that exact input for
    // the batch router; hash/CPU routing intentionally captures the truncated
    // values instead.
    if (prepared_uses_gpu_routing) {
        memcpy(prepared->routing_scores, scores,
               N_EXPERTS * sizeof(float));
    }
    if (!eng->gate_proj[layer]) {
        for (int i = 0; i < N_EXPERTS; i++) scores[i] = (float)(N_EXPERTS - i);
    } else {
        // Truncate to bf16 to match MLX's bf16 gate computation
        for (int i = 0; i < N_EXPERTS; i++) {
            uint32_t u; memcpy(&u, &scores[i], 4); u &= 0xFFFF0000U; memcpy(&scores[i], &u, 4);
        }
    }

    // The enabled forwardBatch path stops here: attention/KV, compressor,
    // mHC-attention post, FFN pre-norm, and router logits are complete. Copy
    // every token-owned value out of the engine's scalar scratch before the
    // next token reuses it. Routing and all expert work then run once for the
    // whole layer/chunk.
    if (prepared) {
        if (!prepared->normed || !prepared->routing_scores ||
            !prepared->residual || !prepared->post_ffn ||
            !prepared->comb_ffn) {
            return -1;
        }
        memcpy(prepared->normed, normed, DIM * sizeof(float));
        if (!prepared_uses_gpu_routing) {
            memcpy(prepared->routing_scores, scores,
                   N_EXPERTS * sizeof(float));
        }
        memcpy(prepared->residual, residual,
               MHC_MULT * DIM * sizeof(float));
        memcpy(prepared->post_ffn, post_ffn,
               MHC_MULT * sizeof(float));
        memcpy(prepared->comb_ffn, comb_ffn,
               MHC_MULT * MHC_MULT * sizeof(float));
        prepared->token_id = eng->current_token_id;
        prepared->position = pos;
        prepared->kv_length = eng->kv_cache[layer].len;
        prepared->compressor_length = eng->comp_state[layer].n_comp;
        prepared->indexer_length = eng->comp_state[layer].n_idx_comp;
        capture_state_digests(
            eng, layer, &prepared->kv_state_digest,
            &prepared->compressor_state_digest,
            &prepared->indexer_state_digest);
        return 0;
    }

    int expert_ids[N_ACTIVE];
    float expert_weights[N_ACTIVE];
    if (use_hash_routing) {
        const int64_t *row = eng->tid2eid[layer] + (size_t)eng->current_token_id * N_ACTIVE;
        int hash_valid = 1;
        for (int k = 0; k < N_ACTIVE; k++) {
            expert_ids[k] = (int)row[k];
            if (expert_ids[k] < 0 || expert_ids[k] >= N_EXPERTS) hash_valid = 0;
        }
        // Weights: gather sqrtsoftplus(logits) at hash-selected positions, L1-normalize, scale.
        // If any expert_id is invalid (token not in hash table), fall back to CPU score routing.
        if (!hash_valid) {
            cpu_moe_route(scores, eng->gate_bias[layer], N_EXPERTS, N_ACTIVE, expert_ids, expert_weights,
                          NULL, 0.0f);
        } else {
            float wsum = 0;
            for (int k = 0; k < N_ACTIVE; k++) {
                float l = scores[expert_ids[k]];
                float sp = l > 0 ? l + log1pf(expf(-l)) : log1pf(expf(l));
                expert_weights[k] = sqrtf(sp);
                wsum += expert_weights[k];
            }
            wsum += 1e-20f;
            for (int k = 0; k < N_ACTIVE; k++) expert_weights[k] = expert_weights[k] / wsum * 1.5f;
        }
    } else if (use_gpu_routing) {
        // GPU routing: read results from buf_gpu_route_selected / buf_gpu_route_weights
        // (written by CMD2 Enc 7, guaranteed complete after waitUntilCompleted)
        int32_t *sel = (int32_t *)[(id<MTLBuffer>)eng->buf_gpu_route_selected contents];
        float   *wts = (float   *)[(id<MTLBuffer>)eng->buf_gpu_route_weights  contents];
        for (int k = 0; k < N_ACTIVE; k++) {
            expert_ids[k]     = (int)sel[k];
            expert_weights[k] = wts[k];
        }
    } else {
        // CPU routing fallback (warmup phase or hash routing layers)
        const uint8_t *const *smelt_cache = NULL;
        if (eng->smelt_enabled && eng->smelt_warmup_done && eng->expert_mem_cache[layer]) {
            smelt_cache = (const uint8_t *const *)eng->expert_mem_cache[layer];
        }
        cpu_moe_route(scores, eng->gate_bias[layer], N_EXPERTS, N_ACTIVE, expert_ids, expert_weights,
                      smelt_cache, eng->smelt_penalty);
    }
    telemetry_record_routes(eng, layer, expert_ids, N_ACTIVE);
    emit_debug_capture(
        eng,
        MOE_INFER_CAPTURE_ROUTER,
        MOE_INFER_CAPTURE_DTYPE_F32,
        layer,
        pos,
        expert_weights,
        N_ACTIVE,
        expert_ids,
        N_ACTIVE);
    // SMELT: always accumulate routing statistics for score-based layers during decode.
    if (eng->smelt_enabled && eng->smelt_in_decode_phase && !use_hash_routing) {
        for (int k = 0; k < N_ACTIVE; k++) {
            int eid = expert_ids[k];
            if (eid >= 0 && eid < N_EXPERTS) {
                eng->routing_counts[layer][eid]++;
            }
        }
    }
    // Temporal prediction: record this layer's routing for next-token prefetch.
    predictor_record(&eng->predictor, layer, expert_ids, N_ACTIVE);
    prefetch_update_adaptive(eng);
    if (exact_q4_recording) {
        exact_q4_routing_done = moe_infer_monotonic_ns();
        exact_q4_moe_begin = exact_q4_routing_done;
        const DmlxExactQ4StageEvent prefix[3] = {
            { .layer = (uint16_t)layer,
              .stage = DMLX_EXACT_Q4_STAGE_ATTENTION_ENCODE,
              .flags = DMLX_EXACT_Q4_EVENT_CPU_ACTIVE,
              .calls = 1u, .begin_ns = exact_q4_layer_begin,
              .end_ns = exact_q4_attention_encode_done },
            { .layer = (uint16_t)layer,
              .stage = DMLX_EXACT_Q4_STAGE_ATTENTION_GPU_WAIT,
              .flags = DMLX_EXACT_Q4_EVENT_GPU_TIME_UNKNOWN,
              .calls = 1u, .begin_ns = exact_q4_attention_encode_done,
              .end_ns = exact_q4_attention_done },
            { .layer = (uint16_t)layer,
              .stage = DMLX_EXACT_Q4_STAGE_ROUTING,
              .flags = DMLX_EXACT_Q4_EVENT_CPU_ACTIVE,
              .calls = 1u, .begin_ns = exact_q4_attention_done,
              .end_ns = exact_q4_routing_done },
        };
        if (!exact_q4_layer_add_stage(&exact_q4_layer, &prefix[0]) ||
            !exact_q4_layer_add_stage(&exact_q4_layer, &prefix[1]) ||
            !exact_q4_layer_add_stage(&exact_q4_layer, &prefix[2])) return -1;
        exact_q4_layer_set_path(&exact_q4_layer, &mla_identity);
    }
    if (layer == 0 && getenv("MF_DBG")) {
        fprintf(stderr, "[mf-dbg] L0 expert_ids=[%d,%d,%d,%d,%d,%d] weights=[%.3f,%.3f,%.3f] hash=%s tok=%d\n",
            expert_ids[0],expert_ids[1],expert_ids[2],expert_ids[3],expert_ids[4],expert_ids[5],
            expert_weights[0],expert_weights[1],expert_weights[2],
            eng->tid2eid[layer] != NULL ? "yes" : "no",
            eng->current_token_id);
    }

    // Expert I/O + MoE. The expert kernel reads buf_normed as input — cb2cmd2's
    // Enc 8 (bf16_to_f32_vec from buf_mhc_ffn_norm_bf16) already wrote the
    // ffn-normed vector there, bit-identical to the former CPU re-upload.
    // MoE combine writes the pure routed-expert sum (zero residual) to buf_hidden.
    {
        IOPool *io = (IOPool *)eng->io_pool;
        const int tl = (getenv("NATIVE_TIME_LAYERS") != NULL);
        double ti0=0, ti1=0, ti2=0;
        if (tl) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ti0 = ts.tv_sec*1e9+ts.tv_nsec; }
        // Resolve expert data in priority order:
        //   1. SMELT RAM cache (zero-copy)
        //   2. predictive prefetch buffers (async pread issued one layer earlier)
        //   3. blocking parallel pread of the remaining misses
        uint8_t *expert_data[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
        NativeLayerMetrics *telemetry = telemetry_current_layer(eng, layer);
        const bool trace = eng->routing_trace != NULL;
        uint8_t route_cache_states[N_ACTIVE] = {
            ROUTING_TRACE_CACHE_COLD,
            ROUTING_TRACE_CACHE_COLD,
            ROUTING_TRACE_CACHE_COLD,
            ROUTING_TRACE_CACHE_COLD,
            ROUTING_TRACE_CACHE_COLD,
            ROUTING_TRACE_CACHE_COLD,
        };
        uint64_t route_bytes[N_ACTIVE] = {0, 0, 0, 0, 0, 0};
        uint64_t hot_hits = 0;
        uint64_t prefetch_hits = 0;
        uint64_t batch_pool_hits = 0;
        if (eng->expert_cache_n_experts > 0 && eng->expert_mem_cache[layer]) {
            for (int k = 0; k < N_ACTIVE; k++) {
                int eid = expert_ids[k];
                if (eid >= 0 && eid < N_EXPERTS && eng->expert_mem_cache[layer][eid]) {
                    expert_data[k] = eng->expert_mem_cache[layer][eid];
                    exact_q4_record_logical_read(
                        eng, (uint16_t)layer, (uint16_t)eid, (uint8_t)k,
                        DMLX_EXACT_Q4_READ_FIXED_HOT);
                    route_cache_states[k] = ROUTING_TRACE_CACHE_HOT;
                    eng->smelt_hits++;
                    if (telemetry || trace) hot_hits++;
                }
                eng->smelt_total++;
            }
            if (getenv("NATIVE_PHASE_TIME") && eng->smelt_total >= 6 * 43 * 20) {
                fprintf(stderr, "[smelt] hit-rate %.1f%% (%llu/%llu)\n",
                        100.0 * (double)eng->smelt_hits / (double)eng->smelt_total,
                        (unsigned long long)eng->smelt_hits, (unsigned long long)eng->smelt_total);
                eng->smelt_hits = 0; eng->smelt_total = 0;
            }
        }
        // The complete top-K was classified against one residency snapshot.
        // Only now may hits advance recency; admissions happen after all reads.
        cache_lru_touch_hits(eng, layer, expert_ids, expert_data);
        bool resolved_before_prefetch[N_ACTIVE] = {false, false, false, false, false, false};
        if (telemetry || trace) {
            for (int k = 0; k < N_ACTIVE; ++k) {
                resolved_before_prefetch[k] = expert_data[k] != NULL;
            }
        }
        prefetch_consume(eng, layer, expert_ids, expert_data, &exact_q4_layer);
        if (telemetry || trace) {
            for (int k = 0; k < N_ACTIVE; ++k) {
                if (!resolved_before_prefetch[k] && expert_data[k] != NULL) {
                    route_cache_states[k] = ROUTING_TRACE_CACHE_PREFETCH;
                    prefetch_hits++;
                }
            }
        }
        // Cross-token dedup (batch/prefill path only): resolve misses against
        // this layer's pool first; each unique expert is read once per layer.
        const int use_batch_pool = (eng->in_batch > 0 && eng->batch_pool != NULL);
        if (use_batch_pool) {
            for (int k = 0; k < N_ACTIVE; k++) {
                if (expert_data[k] != NULL) continue;
                for (int s = 0; s < eng->batch_count; s++) {
                    if (eng->batch_eids[s] == expert_ids[k]) {
                        expert_data[k] = eng->batch_pool + (size_t)s * eng->batch_slot_size;
                        exact_q4_record_logical_read(
                            eng, (uint16_t)layer, (uint16_t)expert_ids[k],
                            (uint8_t)k, DMLX_EXACT_Q4_READ_BATCH_POOL);
                        route_cache_states[k] = ROUTING_TRACE_CACHE_BATCH_POOL;
                        eng->batch_hits++;
                        if (telemetry || trace) batch_pool_hits++;
                        break;
                    }
                }
            }
        }
        int miss_ids[6], miss_map[6];
        uint8_t *miss_bufs[6];
        int n_miss = 0;
        for (int k = 0; k < N_ACTIVE; k++) {
            if (expert_data[k] == NULL) {
                route_cache_states[k] = ROUTING_TRACE_CACHE_COLD;
                miss_ids[n_miss] = expert_ids[k];
                if (use_batch_pool && eng->batch_count < eng->batch_capacity) {
                    int s = eng->batch_count++;
                    eng->batch_eids[s] = expert_ids[k];
                    miss_bufs[n_miss] = eng->batch_pool + (size_t)s * eng->batch_slot_size;
                } else {
                    miss_bufs[n_miss] = eng->expert_buf[k];
                }
                miss_map[n_miss] = k;
                n_miss++;
            }
        }
        uint64_t bytes_read = 0;
        uint64_t blocking_short_reads = 0;
        const bool observe_io = telemetry != NULL || trace;
        uint64_t io_started_ns = observe_io && n_miss > 0 ? moe_infer_monotonic_ns() : 0;
        int io_status = 0;
        uint32_t ordinals[NUM_IO_THREADS] = {0};
        uint8_t owned[NUM_IO_THREADS] = {0};
        bool instrument_io = false;
        int io_generation = 0;
        if (n_miss > 0) {
            // --- In-layer overlap (Task 2.2) ---
            // Submit cold preads asynchronously, then run the shared expert CB in
            // the read window, then join before the cold-dependent routed CB.
            // Safety: the shared CB commits before moe_forward_layer's CBs on the
            // same queue, and command buffers execute in commit order, so
            // buf_hidden (used as scratch by both CBs) is never written
            // concurrently. See docs/en/analysis/q2-layer-transaction-partition.md §5.
            //
            // ORDERING NOTE: moe_forward_layer writes buf_hidden (moe_combine
            // result). The shared expert down-output (buf_attn_out) is summed
            // with buf_hidden on the GPU after moe_forward_layer returns (see
            // the K=1 moe_combine dispatch below), so the shared CB only needs
            // to be committed here — no wait, no CPU readback.
            instrument_io = exact_q4_recording_active(eng);
            for (int index = 0; instrument_io && index < n_miss; ++index) {
                if (!exact_q4_reserve_physical_read(
                        eng, (uint16_t)layer, (uint16_t)miss_ids[index],
                        (uint8_t)miss_map[index], DMLX_EXACT_Q4_READ_PREAD,
                        eng->active_expert_size, &ordinals[index])) {
                    for (int prior = 0; prior < index; ++prior)
                        (void)moe_infer_exact_q4_abort_read(eng, ordinals[prior]);
                    memset(owned, 0, sizeof(owned));
                    instrument_io = false;
                    break;
                }
                owned[index] = 1u;
            }
            // Non-blocking submit: io_pool_start drains any prior generation on
            // the pool (always idle in scalar decode), then posts n_miss pread
            // tasks to worker threads and returns the generation token immediately.
            io_generation = io_pool_start(
                io, eng->packed_fd[layer], miss_ids, n_miss,
                miss_bufs, eng->active_expert_size,
                instrument_io ? eng : NULL, ordinals, owned, false);
        }

        // Shared expert is independent of routed-expert residency. With misses
        // it overlaps the read window; on an all-hot layer it still must run
        // exactly once before the routed result is published.
        // Its down-output (buf_attn_out) is summed with the routed result on
        // the GPU after moe_forward_layer below.
        if (eng->shared[layer].gate.packed != NULL) {
                const int SE_GS = 64;
                const int SE_NG_GU = DIM / SE_GS;
                const int SE_NG_D  = INTERMEDIATE / SE_GS;
                // buf_mhc_attn_in already holds the ffn-normed vector: cb2cmd2's
                // Enc 9 (bf16_to_f32_vec from buf_mhc_ffn_norm_bf16) wrote it,
                // bit-identical to the former CPU re-upload here.
                id<MTLBuffer> bx    = (id<MTLBuffer>)eng->buf_mhc_attn_in;
                id<MTLBuffer> bgate = (id<MTLBuffer>)eng->buf_hidden;
                id<MTLBuffer> bup   = (id<MTLBuffer>)eng->buf_h_mid;
                id<MTLBuffer> bdown = (id<MTLBuffer>)eng->buf_attn_out;
                const QuantWeight *qw_g = &eng->shared[layer].gate;
                const QuantWeight *qw_u = &eng->shared[layer].up;
                const QuantWeight *qw_d = &eng->shared[layer].down;
                // Lazy one-time weight buffer init (first token per layer).
                if (!eng->buf_shared_w[layer][0]) {
                    const uint64_t shared_allocation_begin = exact_q4_recording
                        ? moe_infer_monotonic_ns() : 0u;
                    id<MTLBuffer> *sb = (id<MTLBuffer> *)eng->buf_shared_w[layer];
                    const QuantWeight *qws[3] = {qw_g, qw_u, qw_d};
                    const int ngs[3] = {SE_NG_GU, SE_NG_GU, SE_NG_D};
                    uint64_t shared_allocation_bytes = 0u;
                    for (int i = 0; i < 3; i++) {
                        const size_t packed_bytes = (size_t)qws[i]->out_dim *
                            (qws[i]->in_dim / 8) * sizeof(uint32_t);
                        const size_t metadata_bytes = (size_t)qws[i]->out_dim *
                            ngs[i] * sizeof(float);
                        sb[i*3+0] = [d newBufferWithBytes:(void*)qws[i]->packed length:packed_bytes options:MTLResourceStorageModeShared];
                        sb[i*3+1] = [d newBufferWithBytes:(void*)qws[i]->scales length:metadata_bytes options:MTLResourceStorageModeShared];
                        sb[i*3+2] = [d newBufferWithBytes:(void*)qws[i]->biases length:metadata_bytes options:MTLResourceStorageModeShared];
                        if (!sb[i*3+0] || !sb[i*3+1] || !sb[i*3+2]) {
                            for (int component = 0; component < 9; ++component) {
                                if (sb[component]) [sb[component] release];
                                sb[component] = nil;
                            }
                            if (n_miss > 0) {
                                // io_pool_start published this generation before
                                // the lazy shared-weight allocation. Drain it
                                // before returning so workers no longer own the
                                // engine or miss destination buffers.
                                io_pool_wait_dependency(
                                    io, io_generation, eng, layer,
                                    &exact_q4_layer);
                            }
                            if (instrument_io) {
                                for (int index = 0; index < n_miss; ++index)
                                    if (owned[index]) (void)moe_infer_exact_q4_abort_read(eng, ordinals[index]);
                            }
                            return -1;
                        }
                        shared_allocation_bytes += packed_bytes + 2u * metadata_bytes;
                    }
                    if (exact_q4_recording && exact_q4_host_event_count < 2u) {
                        exact_q4_host_events[exact_q4_host_event_count++] =
                            (DmlxExactQ4StageEvent){
                                .layer = (uint16_t)layer,
                                .stage = DMLX_EXACT_Q4_STAGE_HOST_ALLOC_WRAPPER,
                                .flags = DMLX_EXACT_Q4_EVENT_CPU_ACTIVE,
                                .calls = 9u,
                                .begin_ns = shared_allocation_begin,
                                .end_ns = moe_infer_monotonic_ns(),
                                .bytes = shared_allocation_bytes,
                            };
                    }
                }
                id bw_g = (id)eng->buf_shared_w[layer][0], bs_g = (id)eng->buf_shared_w[layer][1], bb_g = (id)eng->buf_shared_w[layer][2];
                id bw_u = (id)eng->buf_shared_w[layer][3], bs_u = (id)eng->buf_shared_w[layer][4], bb_u = (id)eng->buf_shared_w[layer][5];
                id bw_d = (id)eng->buf_shared_w[layer][6], bs_d = (id)eng->buf_shared_w[layer][7], bb_d = (id)eng->buf_shared_w[layer][8];
                {
                    const uint64_t timeline_shared_begin = decode_timeline_recording
                        ? moe_infer_decode_timeline_now_ns() : 0u;
                    id<MTLCommandBuffer> cb = [P.queue commandBufferWithUnretainedReferences];
                    { id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                      [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
                      [e setBuffer:bw_g offset:0 atIndex:0]; [e setBuffer:bs_g offset:0 atIndex:1]; [e setBuffer:bb_g offset:0 atIndex:2];
                      [e setBuffer:bx offset:0 atIndex:3]; [e setBuffer:bgate offset:0 atIndex:4];
                      uint od=qw_g->out_dim, id_=qw_g->in_dim, gs=SE_GS;
                      [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
                      [e dispatchThreads:MTLSizeMake(qw_g->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                      [e endEncoding]; }
                    { id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                      [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
                      [e setBuffer:bw_u offset:0 atIndex:0]; [e setBuffer:bs_u offset:0 atIndex:1]; [e setBuffer:bb_u offset:0 atIndex:2];
                      [e setBuffer:bx offset:0 atIndex:3]; [e setBuffer:bup offset:0 atIndex:4];
                      uint od=qw_u->out_dim, id_=qw_u->in_dim, gs=SE_GS;
                      [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
                      [e dispatchThreads:MTLSizeMake(qw_u->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                      [e endEncoding]; }
                    { id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                      [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_limited_swiglu];
                      [e setBuffer:bgate offset:0 atIndex:0]; [e setBuffer:bup offset:0 atIndex:1];
                      uint n = INTERMEDIATE; [e setBytes:&n length:4 atIndex:2];
                      [e dispatchThreads:MTLSizeMake(INTERMEDIATE,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                      [e endEncoding]; }
                    { id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                      [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
                      [e setBuffer:bw_d offset:0 atIndex:0]; [e setBuffer:bs_d offset:0 atIndex:1]; [e setBuffer:bb_d offset:0 atIndex:2];
                      [e setBuffer:bgate offset:0 atIndex:3]; [e setBuffer:bdown offset:0 atIndex:4];
                      uint od=qw_d->out_dim, id_=qw_d->in_dim, gs=SE_GS;
                      [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
                      [e dispatchThreads:MTLSizeMake(qw_d->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                      [e endEncoding]; }
                    const uint64_t timeline_shared_encode_done = decode_timeline_recording
                        ? moe_infer_decode_timeline_now_ns() : 0u;
                    const uint64_t timeline_shared_pre_commit = timeline_shared_encode_done;
                    [cb commit];
                    const uint64_t timeline_shared_post_commit = decode_timeline_recording
                        ? moe_infer_decode_timeline_now_ns() : 0u;
                    // No waitUntilCompleted: the GPU add consuming buf_attn_out
                    // is committed later on the same queue, so commit order
                    // alone guarantees the shared CB finishes first.
                    if (decode_timeline_recording) decode_timeline_record(
                        eng, (uint16_t)layer, DMLX_DECODE_TIMELINE_SHARED_EXPERT_CB,
                        timeline_shared_begin, timeline_shared_encode_done,
                        timeline_shared_pre_commit, timeline_shared_post_commit,
                        timeline_shared_post_commit, timeline_shared_post_commit);
                }
            if (phase_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); pt5 = ts.tv_sec*1e9+ts.tv_nsec; }
        }

        if (n_miss > 0) {
            // Cold join: blocks only if preads are still in-flight.
            io_pool_wait_dependency(io, io_generation, eng, layer, &exact_q4_layer);
            // Collect results and detect short reads.
            {
                uint64_t total = 0, incomplete = 0;
                for (int index = 0; index < n_miss; ++index) {
                    const ssize_t result = io->result[index];
                    if (result > 0) total += (uint64_t)result;
                    if (result != (ssize_t)eng->active_expert_size) incomplete++;
                    route_bytes[miss_map[index]] = result > 0 ? (uint64_t)result : 0;
                }
                bytes_read = total;
                blocking_short_reads = incomplete;
                io_status = (incomplete == 0) ? 0 : -1;
            }
            if (telemetry) telemetry->short_reads += blocking_short_reads;
            if (io_status == 0) {
                for (int i = 0; i < n_miss; i++) expert_data[miss_map[i]] = miss_bufs[i];
            }
        } /* end if (n_miss > 0) */
        if (n_miss == 0 && decode_timeline_recording) {
            const uint64_t now = moe_infer_decode_timeline_now_ns();
            decode_timeline_record(
                eng, (uint16_t)layer, DMLX_DECODE_TIMELINE_IO_EXPOSED_WAIT,
                now, now, now, now, now, now);
        }
        const uint64_t io_finished_ns = observe_io && n_miss > 0
            ? moe_infer_monotonic_ns()
            : io_started_ns;
        if (telemetry) {
            telemetry->hot_hits += hot_hits;
            telemetry->prefetch_hits += prefetch_hits;
            telemetry->batch_pool_hits += batch_pool_hits;
            telemetry->cold_misses += (uint64_t)n_miss;
            telemetry->pread_calls += (uint64_t)n_miss;
            telemetry->bytes_requested += (uint64_t)n_miss * eng->active_expert_size;
            telemetry->bytes_read += bytes_read;
            telemetry->io_wait_ns += io_finished_ns - io_started_ns;
            telemetry->bytes_avoided +=
                (hot_hits + prefetch_hits + batch_pool_hits) * eng->active_expert_size;
        }
        if (trace) {
            RoutingTraceRecord trace_record = {0};
            trace_record.phase = (uint32_t)eng->routing_trace_phase;
            trace_record.token_position = pos >= 0 ? (uint32_t)pos : 0;
            trace_record.layer = (uint32_t)layer;
            for (int k = 0; k < N_ACTIVE; ++k) {
                trace_record.expert_ids[k] = (int32_t)expert_ids[k];
                trace_record.weights[k] = expert_weights[k];
                trace_record.cache_states[k] = route_cache_states[k];
                trace_record.route_bytes[k] = route_bytes[k];
            }
            trace_record.pread_bytes = bytes_read;
            trace_record.io_wait_ns = io_finished_ns - io_started_ns;
            trace_record.short_reads = (uint32_t)blocking_short_reads;
            trace_record.io_status = io_status;
            (void)routing_trace_append(eng->routing_trace, &trace_record);
        }
        if (io_status != 0) {
            fprintf(stderr,
                "[expert-io] short read: layer=%d expected=%zu reads=%d bytes=%" PRIu64 " incomplete=%" PRIu64 "\n",
                layer,
                eng->active_expert_size,
                n_miss,
                bytes_read,
                blocking_short_reads);
            return -1;
        }
        const bool exact_q4_records_admission = exact_q4_recording &&
            eng->experimental_cache_enabled && !eng->tid2eid[layer] &&
            n_miss > 0;
        const uint64_t exact_q4_admission_begin = exact_q4_records_admission
            ? moe_infer_monotonic_ns() : 0u;
        if (cache_lru_admit_misses(
                eng, layer, expert_ids, miss_ids, miss_map, miss_bufs, n_miss,
                expert_data, telemetry) != 0) {
            fprintf(stderr, "[cache-policy] LRU admission failed at layer=%d misses=%d\n",
                layer, n_miss);
            return -1;
        }
        if (exact_q4_records_admission) {
            exact_q4_cache_event = (DmlxExactQ4StageEvent){
                .layer = (uint16_t)layer,
                .stage = DMLX_EXACT_Q4_STAGE_CACHE_ADMISSION,
                .flags = DMLX_EXACT_Q4_EVENT_CPU_ACTIVE,
                .calls = (uint64_t)n_miss,
                .begin_ns = exact_q4_admission_begin,
                .end_ns = moe_infer_monotonic_ns(),
                .bytes = (uint64_t)n_miss * eng->active_expert_size,
            };
        }
        if (tl) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ti1 = ts.tv_sec*1e9+ts.tv_nsec; }
        if (phase_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); pt3 = ts.tv_sec*1e9+ts.tv_nsec; }
        // Kick off async prefetch for the next layer now that this layer's
        // critical-path I/O is done — the reads overlap the GPU MoE/attention
        // work between here and the next layer's expert dispatch.
        prefetch_issue(eng, layer + 1);
        moe_forward_layer(
            eng, layer, expert_data, expert_ids, expert_weights, N_ACTIVE,
            pos >= 0 ? (uint32_t)pos : 0u);
        if (tl) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ti2 = ts.tv_sec*1e9+ts.tv_nsec;
            fprintf(stderr, "[MOE-IO] L%d io=%.2fms moe=%.2fms\n", layer, (ti1-ti0)/1e6, (ti2-ti1)/1e6); }
        if (phase_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); pt4 = ts.tv_sec*1e9+ts.tv_nsec; }
        // GPU-resident sum: buf_ffn_out_f32 = buf_hidden (routed moe_combine
        // output) + buf_attn_out (shared expert down-output). Reuses
        // moe_combine with K=1 and a unit weight: out = residual + 1.0f*expert
        // is an f32 add of the same two operands in the same order as the
        // former CPU scalar loop, so the result is bit-identical. This CB is
        // committed after both the shared-expert CB and moe_forward_layer's
        // routed CB on the same queue, so commit order enforces the
        // producer->consumer ordering with no CPU sync or readback.
        {
            id<MTLCommandBuffer> cb_add = [(id<MTLCommandQueue>)eng->queue commandBufferWithUnretainedReferences];
            if (eng->shared[layer].gate.packed != NULL) {
                id<MTLComputeCommandEncoder> e = [cb_add computeCommandEncoder];
                [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_moe_combine];
                [e setBuffer:(id<MTLBuffer>)eng->buf_attn_out    offset:0 atIndex:0];
                [e setBuffer:(id<MTLBuffer>)eng->buf_combine_one offset:0 atIndex:1];
                [e setBuffer:(id<MTLBuffer>)eng->buf_hidden      offset:0 atIndex:2];
                [e setBuffer:(id<MTLBuffer>)eng->buf_ffn_out_f32 offset:0 atIndex:3];
                uint kv = 1, hd = DIM;
                [e setBytes:&kv length:4 atIndex:4];
                [e setBytes:&hd length:4 atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake((DIM+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            } else {
                // No shared expert: ffn_out is just the routed combine result.
                id<MTLBlitCommandEncoder> blit = [cb_add blitCommandEncoder];
                [blit copyFromBuffer:(id<MTLBuffer>)eng->buf_hidden
                        sourceOffset:0
                            toBuffer:(id<MTLBuffer>)eng->buf_ffn_out_f32
                     destinationOffset:0
                                size:(size_t)DIM * sizeof(float)];
                [blit endEncoding];
            }
            [cb_add commit];
            // Debug capture / MF_DBG still consume ffn_out on the CPU; pay the
            // readback only when one of them is active.
            if (eng->debug_capture || getenv("MF_DBG")) {
                [cb_add waitUntilCompleted];
                memcpy(ffn_out, [(id<MTLBuffer>)eng->buf_ffn_out_f32 contents],
                       DIM * sizeof(float));
            }
        }
        if (phase_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); pt5 = ts.tv_sec*1e9+ts.tv_nsec; }
    }
    emit_debug_capture(
        eng,
        MOE_INFER_CAPTURE_MOE_OUTPUT,
        MOE_INFER_CAPTURE_DTYPE_F32,
        layer,
        pos,
        ffn_out,
        DIM,
        NULL,
        0);

    if (getenv("MF_DBG")) {
        float *test_input = (float *)[(id<MTLBuffer>)eng->buf_normed contents];
        fprintf(stderr, "[moe-dbg] buf_normed[0..3]=[%.6f,%.6f,%.6f,%.6f]\n", test_input[0], test_input[1], test_input[2], test_input[3]);
        float *test_mid = (float *)[(id<MTLBuffer>)eng->buf_expert_mid[0] contents];
        fprintf(stderr, "[moe-dbg] expert_mid[0][0..3]=[%.6f,%.6f,%.6f,%.6f]\n", test_mid[0], test_mid[1], test_mid[2], test_mid[3]);
        float *test_out = (float *)[(id<MTLBuffer>)eng->buf_expert_out[0] contents];
        fprintf(stderr, "[moe-dbg] expert_out[0][0..3]=[%.6f,%.6f,%.6f,%.6f]\n", test_out[0], test_out[1], test_out[2], test_out[3]);
    }

    if (layer == 0 && getenv("MF_DBG")) {
        int first_nonzero = -1;
        float maxv = 0.0f, minv = 0.0f, sum = 0.0f;
        for (int i=0;i<DIM;i++) {
            float v = ffn_out[i];
            sum += v;
            if (v > maxv) maxv = v;
            if (v < minv) minv = v;
            if (first_nonzero < 0 && v != 0.0f) first_nonzero = i;
        }
        fprintf(stderr, "[mf-dbg] L%d moe_out[0..3]=[%.6f,%.6f,%.6f,%.6f] sum=%.6f min=%.6f max=%.6f nonzero@%d\n",
                layer, ffn_out[0], ffn_out[1], ffn_out[2], ffn_out[3], sum, minv, maxv, first_nonzero);
    }

    if (layer == 0 && getenv("MF_DBG")) {
        double fn=0; for(int z=0;z<DIM;z++) fn+=(double)ffn_out[z]*ffn_out[z];
        const char *dd = getenv("DSV4_DUMP_DIR");
        fprintf(stderr, "[mf-dbg] L0 ffn_out(moe only) norm=%.4f\n", sqrt(fn));
        // Dump pre-shared-expert MoE output for debugging
        if (dd) {
            char path[1024]; snprintf(path, sizeof(path), "%s/L0_moe_only_out.bin", dd);
            FILE *f = fopen(path, "wb"); if (f) { fwrite(ffn_out, sizeof(float), DIM, f); fclose(f); }
        }
    }

    // Shared expert CB is now executed in the read window above (Task 2.2).
    // Its down-output was summed with the routed result into buf_ffn_out_f32
    // by the GPU add committed after moe_forward_layer, so cb3's ffn input is
    // already GPU-resident — no CPU round-trip here.

    // mHC post (FFN) — ds4 kernel_dsv4_hc_expand4: single pure-f32 dispatch.
    // Replaces 3-encoder f32→bf16 + mhc_post_bfloat + bf16→f32 with 1 encoder.
    {
        if (exact_q4_recording)
            exact_q4_moe_encode_begin = moe_infer_monotonic_ns();
        // buf_ffn_out_f32 already holds the ffn output (written by the GPU add
        // above, committed earlier on this queue — ordering is by commit).
        // buf_residual_gpu already holds the post-attention residual: cb2cmd2's
        // Enc3 (bf16_to_f32_vec from buf_mhc_post_res_out) wrote it, and the
        // CPU residual[] read back from the same bf16 buffer converts
        // bit-identically (float(bfloat) == u<<16, exact), so the former CPU
        // re-upload here was a bit-exact no-op and is removed.
        // buf_mhc_post_weights/comb_weights already hold correct FFN values from cb2cmd2.
        // (The previous uninitialized post/comb upload was a latent bug masked by commandBuffer
        //  retaining the correct buffer state from cb2cmd2's kernel output.)
        const uint64_t timeline_cb3_begin = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        id<MTLCommandBuffer> cb3 = [(id<MTLCommandQueue>)eng->queue commandBufferWithUnretainedReferences];

        // Single encoder: mhc_post_ffn_expand4
        // block_out = buf_ffn_out_f32 (ffn output in f32)
        // residual = buf_residual_gpu (reads old residual)
        // post/comb = buf_mhc_post/comb_weights
        // dst = buf_residual_gpu (writes new residual, in-place OK because
        //      each output dim depends only on same-dim inputs)
        {
            id<MTLComputeCommandEncoder> e = [cb3 computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_mhc_post_ffn_expand4];
            [e setBuffer:(id<MTLBuffer>)eng->buf_ffn_out_f32       offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu       offset:0 atIndex:1];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_weights   offset:0 atIndex:2];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_comb_weights   offset:0 atIndex:3];
            [e setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu       offset:0 atIndex:4];
            uint dim3 = DIM;
            [e setBytes:&dim3 length:4 atIndex:5];
            [e dispatchThreads:MTLSizeMake(DIM, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
        }

        if (exact_q4_recording)
            exact_q4_moe_encode_done = moe_infer_monotonic_ns();
        const uint64_t timeline_cb3_encode_done = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        const uint64_t timeline_cb3_pre_commit = timeline_cb3_encode_done;
        [cb3 commit];
        const uint64_t timeline_cb3_post_commit = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        // Consumers that need the residual on the CPU right after this layer:
        // debug_capture (LAYER_OUTPUT record below), MF_DBG dumps, and the
        // DSpark target-hidden accumulate in moe_infer_forward (layers 40-42).
        const bool need_cpu_residual =
            eng->debug_capture || getenv("MF_DBG") ||
            (eng->dspark_engine && eng->dspark_accumulate_enabled &&
             (layer == 40 || layer == 41 || layer == 42));
        if (need_cpu_residual) {
            // Synchronous drain, matching the pre-deferred behavior: wait for
            // cb3 and read the new residual back into CPU residual[].
            [cb3 waitUntilCompleted];
            memcpy(residual, [(id<MTLBuffer>)eng->buf_residual_gpu contents],
                   (size_t)MHC_MULT * DIM * sizeof(float));
            if (eng->deferred.cmd_experts) {
                [(id<MTLCommandBuffer>)eng->deferred.cmd_experts waitUntilCompleted];
                [(id<MTLCommandBuffer>)eng->deferred.cmd_experts release];
            }
            eng->deferred.cmd_experts  = NULL;
            eng->deferred.active       = false;
            eng->deferred.gpu_combined = false;
        } else {
            // Production path: no wait, no readback. Arm the deferred slot so
            // the residual stays GPU-resident — the next layer's drain binds
            // buf_residual_gpu directly, and the moe_infer_forward /
            // forward_batch end-of-layer drains perform the single wait +
            // readback at the token/layer boundary where a CPU consumer
            // (logits, next token's hidden) actually needs the value.
            // mhc_post_ffn_expand4 rounds its output through bfloat
            // (float(bfloat(acc)), same rounding the old 3-encoder cb3 had),
            // so the GPU-resident residual is bit-identical to the value the
            // removed CPU round trip used to carry.
            if (eng->deferred.cmd_experts) {
                [(id<MTLCommandBuffer>)eng->deferred.cmd_experts waitUntilCompleted];
                [(id<MTLCommandBuffer>)eng->deferred.cmd_experts release];
            }
            eng->deferred.cmd_experts  = [cb3 retain];
            eng->deferred.active       = true;
            eng->deferred.gpu_combined = true;
        }
        const uint64_t timeline_cb3_post_wait = decode_timeline_recording
            ? moe_infer_decode_timeline_now_ns() : 0u;
        if (decode_timeline_recording) decode_timeline_record(
            eng, (uint16_t)layer, DMLX_DECODE_TIMELINE_CB3_CB,
            timeline_cb3_begin, timeline_cb3_encode_done,
            timeline_cb3_pre_commit, timeline_cb3_post_commit,
            timeline_cb3_post_wait, moe_infer_decode_timeline_now_ns());
    }
    if (layer == 0 && getenv("MF_DBG")) {
        double fn=0; for(int z=0;z<DIM;z++) fn+=(double)ffn_out[z]*ffn_out[z];
        double rn=0; for(int z=0;z<MHC_MULT*DIM;z++) rn+=(double)residual[z]*residual[z];
        fprintf(stderr, "[mf-dbg] L0 ffn_out norm=%.4f, residual after ffn-post norm=%.4f\n", sqrt(fn), sqrt(rn));        const char *dd = getenv("DSV4_DUMP_DIR");
        if (dd) {
            char path[1024]; snprintf(path, sizeof(path), "%s/L0_ffn_out_metal.bin", dd);
            FILE *ff = fopen(path, "wb");
            if (ff) { fwrite(ffn_out, sizeof(float), DIM, ff); fclose(ff); }
            // Also dump the full post-layer-0 residual [MHC_MULT, DIM] for pos=8
            if (pos == 8) {
                snprintf(path, sizeof(path), "%s/L0_residual_out_pos8.bin", dd);
                FILE *fp = fopen(path, "wb");
                if (fp) { fwrite(residual, sizeof(float), MHC_MULT * DIM, fp); fclose(fp); }
            }
        }
    }

    emit_debug_capture(
        eng,
        MOE_INFER_CAPTURE_LAYER_OUTPUT,
        MOE_INFER_CAPTURE_DTYPE_F32,
        layer,
        pos,
        residual,
        (size_t)MHC_MULT * DIM,
        NULL,
        0);

    } // end layer body
    if (exact_q4_recording) {
        uint64_t layer_end = moe_infer_monotonic_ns();
        if (exact_q4_attention_done <= exact_q4_layer_begin)
            exact_q4_attention_done = exact_q4_layer_begin + 1u;
        if (exact_q4_routing_done <= exact_q4_attention_done)
            exact_q4_routing_done = exact_q4_attention_done + 1u;
        if (exact_q4_moe_begin < exact_q4_routing_done)
            exact_q4_moe_begin = exact_q4_routing_done;
        if (exact_q4_moe_encode_begin < exact_q4_moe_begin)
            exact_q4_moe_encode_begin = exact_q4_moe_begin;
        if (exact_q4_moe_encode_done <= exact_q4_moe_encode_begin)
            exact_q4_moe_encode_done = exact_q4_moe_encode_begin + 1u;
        if (layer_end <= exact_q4_moe_begin) layer_end = exact_q4_moe_begin + 1u;
        if (layer_end < exact_q4_moe_encode_done)
            layer_end = exact_q4_moe_encode_done;
        const DmlxExactQ4StageEvent moe_encode = {
            .layer = (uint16_t)layer,
            .stage = DMLX_EXACT_Q4_STAGE_MOE_ENCODE,
            .flags = DMLX_EXACT_Q4_EVENT_CPU_ACTIVE,
            .calls = 1u,
            .begin_ns = exact_q4_moe_encode_begin,
            .end_ns = exact_q4_moe_encode_done,
        };
        const DmlxExactQ4StageEvent moe_wait = {
            .layer = (uint16_t)layer,
            .stage = DMLX_EXACT_Q4_STAGE_MOE_GPU_WAIT,
            .flags = DMLX_EXACT_Q4_EVENT_GPU_TIME_UNKNOWN,
            .calls = 1u,
            .begin_ns = exact_q4_moe_begin,
            .end_ns = layer_end,
        };
        if (!exact_q4_layer_add_stage(&exact_q4_layer, &moe_encode) ||
            !exact_q4_layer_add_stage(&exact_q4_layer, &moe_wait)) return -1;
        if (exact_q4_cache_event.calls > 0u &&
            !exact_q4_layer_add_stage(
                &exact_q4_layer, &exact_q4_cache_event)) return -1;
        for (uint32_t index = 0u; index < exact_q4_host_event_count; ++index) {
            if (!exact_q4_layer_add_stage(
                    &exact_q4_layer, &exact_q4_host_events[index])) return -1;
        }
        if (exact_q4_layer_commit(eng, &exact_q4_layer) !=
                DMLX_EXACT_Q4_PRODUCER_OK) return -1;
    }
    if (phase_time) { fprintf(stderr, "[PHASE] L%d mla=%.2fms cmd2=%.2fms io=%.2fms moe=%.2fms shared=%.2fms total=%.2fms\n",
        layer, (pt1-pt0)/1e6, (pt2-pt1)/1e6, (pt3-pt2)/1e6, (pt4-pt3)/1e6, (pt5-pt4)/1e6, (pt5-pt0)/1e6);
        fprintf(stderr, "[PHASE-MLA] L%d drain=%.2f enc=%.2f gpu=%.2f rb=%.2f gpuexec=%.2f\n",
        layer, (ptA-pt0)/1e6, (ptB-ptA)/1e6, (ptC-ptB)/1e6, (pt1-ptC)/1e6, gpu_exec_ms); }
    return 0;
}

int moe_infer_forward_layer(
    MoEInferEngine *eng, int layer, float *hidden, int pos) {
    if (eng && eng->in_batch > 0) {
        eng->batch_scalar_layer_calls++;
    }
    return moe_infer_forward_layer_impl(eng, layer, hidden, pos, NULL);
}

static int moe_infer_finish_prepared_token(
    MoEInferEngine *eng,
    int layer,
    NativePreparedMoEToken *prepared,
    const float *ffn_out) {
    if (!eng || !prepared || !prepared->residual || !prepared->post_ffn ||
        !prepared->comb_ffn || !ffn_out || !eng->pipe_mhc_post_ffn_expand4 ||
        !eng->buf_ffn_out_f32 || !eng->buf_residual_gpu ||
        !eng->buf_mhc_post_weights || !eng->buf_mhc_comb_weights) {
        return -1;
    }
    for (uint32_t column = 0u; column < DIM; ++column) {
        if (!isfinite(ffn_out[column])) return -1;
    }
    memcpy([(id<MTLBuffer>)eng->buf_ffn_out_f32 contents], ffn_out,
           DIM * sizeof(float));
    memcpy([(id<MTLBuffer>)eng->buf_residual_gpu contents], prepared->residual,
           MHC_MULT * DIM * sizeof(float));
    memcpy([(id<MTLBuffer>)eng->buf_mhc_post_weights contents],
           prepared->post_ffn, MHC_MULT * sizeof(float));
    memcpy([(id<MTLBuffer>)eng->buf_mhc_comb_weights contents],
           prepared->comb_ffn, MHC_MULT * MHC_MULT * sizeof(float));

    @autoreleasepool {
        id<MTLCommandBuffer> command =
            [(id<MTLCommandQueue>)eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = command
            ? [command computeCommandEncoder]
            : nil;
        if (!command || !encoder) return -1;
        [encoder setComputePipelineState:
            (id<MTLComputePipelineState>)eng->pipe_mhc_post_ffn_expand4];
        [encoder setBuffer:(id<MTLBuffer>)eng->buf_ffn_out_f32
                    offset:0 atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu
                    offset:0 atIndex:1];
        [encoder setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_weights
                    offset:0 atIndex:2];
        [encoder setBuffer:(id<MTLBuffer>)eng->buf_mhc_comb_weights
                    offset:0 atIndex:3];
        [encoder setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu
                    offset:0 atIndex:4];
        const uint32_t hidden_dim = DIM;
        [encoder setBytes:&hidden_dim length:sizeof(hidden_dim) atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(DIM, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return -1;
    }
    memcpy(prepared->residual,
           [(id<MTLBuffer>)eng->buf_residual_gpu contents],
           MHC_MULT * DIM * sizeof(float));
    emit_prepared_debug_capture(
        eng, prepared, MOE_INFER_CAPTURE_LAYER_OUTPUT,
        MOE_INFER_CAPTURE_DTYPE_F32, layer, prepared->residual,
        (size_t)MHC_MULT * DIM, NULL, 0);
    return 0;
}

static int moe_infer_forward_batch_layer_batched_moe(
    MoEInferEngine *eng,
    int layer,
    float *hidden_batch,
    int n_tokens,
    int start_pos,
    const int *token_ids,
    const MoeInferStateSnapshot *state_snapshots,
    bool attention_already_batched);

int moe_infer_forward(MoEInferEngine *eng, float *hidden, int pos) {
    int max_layers = N_LAYERS;
    const char *nl = getenv("NATIVE_MAX_LAYERS");
    if (nl) max_layers = atoi(nl);
    const char *time_layers = getenv("NATIVE_TIME_LAYERS");
    @autoreleasepool {  // Per-token pool: drain after all 43 layers complete (avoids per-layer race)
    for (int layer = 0; layer < max_layers && layer < N_LAYERS; layer++) {
        double t0 = 0;
        if (time_layers) {
            struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
            t0 = ts.tv_sec * 1e9 + ts.tv_nsec;
        }
        if (moe_infer_forward_layer(eng, layer, hidden, pos) != 0) return -1;
        if (time_layers) {
            struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
            double t1 = ts.tv_sec * 1e9 + ts.tv_nsec;
            fprintf(stderr, "[TIME] pos=%d layer=%d: %.1fms\n", pos, layer, (t1-t0)/1e6);
        }
        // DSpark: extract target hidden states from layers 40, 41, 42
        // for use as input to the DSpark draft backbone.
        if (eng->dspark_engine && eng->dspark_accumulate_enabled && (layer == 40 || layer == 41 || layer == 42)) {
            dspark_accumulate_target_hidden(
                (DSparkEngine *)eng->dspark_engine, hidden, layer);
        }
    }

    // DSpark main_kv: the window is now written inside dspark_forward from
    // main_x (kv_norm(wkv(main_x)) per draft layer, reference semantics).
    // The old path fed target layer-0 KV entries — a different vector space
    // the backbone's queries are not trained to read. Disabled.
    (void)0;

    // Wait for last layer's deferred cb3 + read final residual back to hidden.
    if (eng->deferred.active && eng->deferred.cmd_experts) {
        [(id<MTLCommandBuffer>)eng->deferred.cmd_experts waitUntilCompleted];
        [(id<MTLCommandBuffer>)eng->deferred.cmd_experts release];
        eng->deferred.cmd_experts = NULL;
        eng->deferred.active = false;
        if (eng->deferred.gpu_combined) {
            float *gpu_res = (float *)[(id<MTLBuffer>)eng->buf_residual_gpu contents];
            memcpy(hidden, gpu_res, MHC_MULT * DIM * sizeof(float));
            eng->deferred.gpu_combined = false;
        }
    }
    } // end per-token @autoreleasepool

    if ((time_layers || getenv("NATIVE_PREFETCH_DEBUG")) && eng->predictor.valid) {
        int h = eng->predictor.hits, m = eng->predictor.misses;
        fprintf(stderr, "[prefetch] pos=%d hits=%d misses=%d rate=%.2f%s\n", pos, h, m,
                (h + m) > 0 ? (double)h / (h + m) : 0.0,
                eng->prefetch_disabled == 1 ? " (disabled)" : "");
    }

    // SMELT: count this decode token, trigger warmup completion if threshold reached.
    // Only count tokens after prefill (pos > 0 is not sufficient since prefill also
    // calls forward token-by-token). We detect decode mode by checking the engine's
    // current_pos which is managed by the caller (native_engine.zig sets it after prefill).
    // Use a dedicated flag: smelt_in_decode_phase (set by moe_infer_set_decode_mode).
    if (eng->smelt_enabled && !eng->smelt_warmup_done && eng->smelt_in_decode_phase) {
        eng->smelt_tokens_seen++;
        if (eng->smelt_tokens_seen >= eng->smelt_warmup_tokens) {
            // Async: spawn background thread to pread top-N experts per layer.
            // Routing bias activates once smelt_warmup_done becomes true (after preload).
            // Decode continues at full (un-cached) speed until preload completes.
            moe_infer_smelt_preload_async(eng);
        }
    }
    return 0;
}

// Batch forward pass for N tokens (proper transformer order: all tokens per layer).
// Each token starts from its embed in hidden_batch[t * MHC_MULT * DIM].
// After this call, hidden_batch[t] contains the post-layer-42 residual for token t.
// KV caches are built up correctly for each layer: token 0 first, then token 1, etc.
// token_ids: optional [n_tokens] array of token IDs for per-token hash routing.
//            If NULL, uses eng->current_token_id (set by caller) for all tokens.
int moe_infer_forward_batch(MoEInferEngine *eng, float *hidden_batch, int n_tokens, int start_pos, const int *token_ids) {
    if (moe_infer_exact_q4_recording_enabled(eng)) return -1;
    if (!eng || !hidden_batch || n_tokens <= 0 ||
        eng->unique_resolver_owner != 0u) {
        if (eng && eng->unique_resolver_owner != 0u) {
            fprintf(stderr,
                "[expert-resolver] forward_batch rejected while slots are pinned\n");
        }
        return -1;
    }
    if (eng->batch_pins) {
        for (int slot = 0; slot < eng->batch_capacity; ++slot) {
            if (eng->batch_pins[slot] != 0u) {
                fprintf(stderr,
                    "[expert-resolver] forward_batch rejected stale slot pin=%d\n",
                    slot);
                return -1;
            }
        }
    }
    int max_layers = N_LAYERS;
    const char *nl = getenv("NATIVE_MAX_LAYERS");
    if (nl) max_layers = atoi(nl);
    eng->in_batch++;  // disable predictive prefetch for the whole batch pass
    const int phase_time_batch = (getenv("NATIVE_PHASE_TIME") != NULL);
    double batch_dwait_ms = 0;  // per-layer deferred-wait total (MoE GPU tail)

    // Lazy-allocate the cross-token expert dedup pool (gather6/MXFP4 mode only).
    // First batch forward pays one bounded aligned allocation; decode-only
    // processes never allocate. Explicit v2 resolution reuses the same pool.
    if (!eng->batch_pool && !eng->use_affine_experts &&
        eng->active_expert_size > 0 && getenv("NATIVE_BATCH_POOL") == NULL) {
        int cap = 48;
        const char *cs = getenv("NATIVE_BATCH_POOL_SLOTS");
        if (cs) cap = atoi(cs);
        if (cap < N_ACTIVE) cap = N_ACTIVE;
        if (cap > N_EXPERTS ||
            moe_infer_batch_pool_prepare(eng, (uint32_t)cap) !=
                NATIVE_BATCH_OK) {
            // Preserve the legacy per-token fallback when the optional pool is
            // unavailable; explicit bucket resolution will fail closed.
            eng->batch_capacity = 0;
        }
    }

    // Dump all token embed outputs (before any layer processing) for diagnostics
    if (getenv("DSV4_DUMP_DIR") && getenv("MF_DBG")) {
        const char *dd = getenv("DSV4_DUMP_DIR");
        char path[1024];
        snprintf(path, sizeof(path), "%s/embed_all_tokens.bin", dd);
        FILE *fe = fopen(path, "wb");
        if (fe) {
            // Write [n_tokens, MHC_MULT, DIM] f32
            fwrite(hidden_batch, sizeof(float), (size_t)n_tokens * MHC_MULT * DIM, fe);
            fclose(fe);
        }
        // Also print first token residual details
        fprintf(stderr, "[embed-dump] n_tokens=%d tok0[0..3]=[%.4f %.4f %.4f %.4f]\n",
            n_tokens, hidden_batch[0], hidden_batch[1], hidden_batch[2], hidden_batch[3]);
        fprintf(stderr, "[embed-dump] tok2[0..3]=[%.4f %.4f %.4f %.4f]\n",
            hidden_batch[2*MHC_MULT*DIM+0], hidden_batch[2*MHC_MULT*DIM+1],
            hidden_batch[2*MHC_MULT*DIM+2], hidden_batch[2*MHC_MULT*DIM+3]);
        fprintf(stderr, "[embed-dump] tok4[0..3]=[%.4f %.4f %.4f %.4f]\n",
            hidden_batch[4*MHC_MULT*DIM+0], hidden_batch[4*MHC_MULT*DIM+1],
            hidden_batch[4*MHC_MULT*DIM+2], hidden_batch[4*MHC_MULT*DIM+3]);
    }

    // DSpark prefill: per-position mean-pooled hidden for layers 40/41/42,
    // used to build the main_kv context window after the layer loop.
    // DSpark prefill: per-position mean-pooled hidden for layers 40/41/42.
    // Collected whenever a DSpark engine is attached (also during verify
    // batches — needed to backfill accepted positions afterward). The
    // main_kv window build itself happens only for real prefill (accumulate
    // enabled); verify callers commit accepted positions explicitly.
    if (eng->dspark_engine) {
        if (eng->dspark_hidden3_cap < n_tokens) {
            free(eng->dspark_hidden3);
            eng->dspark_hidden3 = (float *)malloc((size_t)3 * n_tokens * DIM * sizeof(float));
            eng->dspark_hidden3_cap = eng->dspark_hidden3 ? n_tokens : 0;
        }
        eng->dspark_hidden3_n = eng->dspark_hidden3 ? n_tokens : 0;
    }

    // Task 4.5 kernels are validated only through B=8. Keep this as a hard
    // engine boundary even if a direct C caller configures a larger planner
    // capacity; Task 5.1 owns chunking larger requests into supported pieces.
    const bool use_batched_moe = eng->batch_config_set == 1u &&
        eng->batch_enable_expert_buckets == 1u &&
        eng->batch_enable_batched_expert_kernels == 1u &&
        eng->forward_batch_workspace != NULL &&
        n_tokens <= 8 &&
        n_tokens <= (int)eng->batch_max_prefill_chunk;
    if (eng->batch_enable_expert_buckets == 1u &&
        eng->batch_enable_batched_expert_kernels == 1u &&
        !use_batched_moe) {
        fprintf(stderr,
            "[batch-moe] chunk=%d exceeds supported/configured capacity (hard=8 configured=%u); using scalar fallback\n",
            n_tokens, eng->batch_max_prefill_chunk);
    }

    for (int layer = 0; layer < max_layers && layer < N_LAYERS; layer++) {
        // Per-layer autorelease pool for the batch path. Unlike moe_infer_forward
        // (which wraps each token in a pool), forward_batch previously had NO pool:
        // the ~5 autoreleased MTLCommandBuffers created per (layer,token) leaked into
        // the void, and after ~64 outstanding CB objects Metal's queue semaphore
        // blocked commandBuffer creation forever (0% GPU deadlock on 9+ token prompts).
        // Draining here is safe: the per-layer deferred drain below waits + releases
        // the last in-flight CB before the pool exits, so no drain races live GPU work.
        @autoreleasepool {
        eng->batch_count = 0;  // per-layer reset: pool contents are layer-specific
        eng->batch_hits = 0;
        batch_dwait_ms = 0;
        // Per-token pool: the per-layer pool above alone only covers ≤12-token
        // prompts — ~5 unretained CBs accumulate per (layer,token) and Metal
        // caps outstanding unretained CB objects (~64), so CB creation
        // deadlocks mid-layer on 13+ token prompts (semaphore_wait in
        // commandBufferWithUnretainedReferences, 0% GPU). Drain cadence mirrors
        // moe_infer_forward's per-token pool; each drain happens right after
        // the deferred wait+release below, when no GPU work is in flight, so
        // no drain races live GPU work (ac6e291/13a433a pattern).
        if (use_batched_moe) {
            if (moe_infer_forward_batch_layer_batched_moe(
                    eng, layer, hidden_batch, n_tokens, start_pos,
                    token_ids, NULL, false) != NATIVE_BATCH_OK) {
                eng->in_batch--;
                return -1;
            }
        } else {
        NSAutoreleasePool *tok_pool = [[NSAutoreleasePool alloc] init];
        for (int t = 0; t < n_tokens; t++) {
            float *hidden_t = hidden_batch + (size_t)t * (MHC_MULT * DIM);
            if (token_ids != NULL) eng->current_token_id = (int)token_ids[t];
            // Batch prefill deferred state management:
            // The previous token (t-1) may have left deferred.active=true with its GPU output
            // in buf_residual_gpu. We must:
            //   1. Wait for GPU to finish (correctness barrier)
            //   2. Read GPU output back to hidden_batch[t-1] — the correct layer-N output for
            //      token t-1, which token t-1 needs as input for layer N+1
            //   3. Do NOT overwrite hidden_t (current token's input for this layer)
            // Then forward_layer will sync hidden_t → buf_residual_gpu for this token.
            if (eng->deferred.active && eng->deferred.cmd_experts) {
                double dw0 = 0;
                if (phase_time_batch) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); dw0 = ts.tv_sec*1e9+ts.tv_nsec; }
                [(id<MTLCommandBuffer>)eng->deferred.cmd_experts waitUntilCompleted];
                if (phase_time_batch) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); batch_dwait_ms += (ts.tv_sec*1e9+ts.tv_nsec - dw0)/1e6; }
                [(id<MTLCommandBuffer>)eng->deferred.cmd_experts release];
                eng->deferred.cmd_experts = NULL;
                eng->deferred.active = false;
                if (eng->deferred.gpu_combined && t > 0) {
                    float *gpu_res = (float *)[(id<MTLBuffer>)eng->buf_residual_gpu contents];
                    float *prev = hidden_batch + (size_t)(t - 1) * (MHC_MULT * DIM);
                    memcpy(prev, gpu_res, (size_t)MHC_MULT * DIM * sizeof(float));
                }
                eng->deferred.gpu_combined = false;
            }
            // Previous token's last in-flight CB was just wait+released above —
            // nothing is in flight now, so draining its autoreleased CBs is safe.
            [tok_pool drain];
            tok_pool = [[NSAutoreleasePool alloc] init];
            if (moe_infer_forward_layer(eng, layer, hidden_t, start_pos + t) != 0) {
                [tok_pool drain];
                eng->in_batch--;
                return -1;
            }
        }
        // After all tokens in this layer: drain deferred for the last token
        // and update its hidden with the GPU's final output.
        if (eng->deferred.active && eng->deferred.cmd_experts) {
            double dw0 = 0;
            if (phase_time_batch) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); dw0 = ts.tv_sec*1e9+ts.tv_nsec; }
            [(id<MTLCommandBuffer>)eng->deferred.cmd_experts waitUntilCompleted];
            if (phase_time_batch) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); batch_dwait_ms += (ts.tv_sec*1e9+ts.tv_nsec - dw0)/1e6; }
            [(id<MTLCommandBuffer>)eng->deferred.cmd_experts release];
            eng->deferred.cmd_experts = NULL;
            eng->deferred.active = false;
            if (eng->deferred.gpu_combined) {
                float *gpu_res = (float *)[(id<MTLBuffer>)eng->buf_residual_gpu contents];
                float *last = hidden_batch + (size_t)(n_tokens - 1) * (MHC_MULT * DIM);
                memcpy(last, gpu_res, (size_t)MHC_MULT * DIM * sizeof(float));
                eng->deferred.gpu_combined = false;
            }
        }
        // Last token's deferred CB was wait+released above — nothing in flight.
        [tok_pool drain];
        } // scalar fallback
        if (eng->batch_pool && getenv("NATIVE_PHASE_TIME")) {
            fprintf(stderr, "[BATCH-IO] L%d uniq=%d saved=%d dwait=%.2fms\n",
                    layer, eng->batch_count, eng->batch_hits, batch_dwait_ms);
        }
        // DSpark: collect mean-pooled hidden from layers 40/41/42 for EVERY
        // position (main_kv window backfill) whenever a DSpark engine is
        // attached — including verify batches. The last token's
        // buf_main_hidden accumulate stays gated by accumulate_enabled.
        if (eng->dspark_engine && (layer == 40 || layer == 41 || layer == 42)) {
            if (eng->dspark_accumulate_enabled) {
                float *last = hidden_batch + (size_t)(n_tokens - 1) * (MHC_MULT * DIM);
                dspark_accumulate_target_hidden(
                    (DSparkEngine *)eng->dspark_engine, last, layer);
            }
            if (eng->dspark_hidden3) {
                float *dst = eng->dspark_hidden3 + (size_t)(layer - 40) * n_tokens * DIM;
                for (int t = 0; t < n_tokens; t++) {
                    const float *src = hidden_batch + (size_t)t * (MHC_MULT * DIM);
                    float *d = dst + (size_t)t * DIM;
                    for (int i = 0; i < DIM; i++) {
                        float s = 0.0f;
                        for (int m = 0; m < MHC_MULT; m++) s += src[m * DIM + i];
                        d[i] = s / (float)MHC_MULT;
                    }
                }
            }
        }
        // Per-layer residual dump: write last token's residual as raw f32 bin
        if (getenv("DSV4_DUMP_DIR") && getenv("MF_DBG")) {
            const char *dd = getenv("DSV4_DUMP_DIR");
            char path[1024];
            snprintf(path, sizeof(path), "%s/L%02d_residual_last.bin", dd, layer);
            FILE *fl = fopen(path, "wb");
            if (fl) {
                float *last = hidden_batch + (size_t)(n_tokens-1) * (MHC_MULT * DIM);
                fwrite(last, sizeof(float), MHC_MULT * DIM, fl);
                fclose(fl);
            }
        }
        } // end per-layer @autoreleasepool
    }

    // DSpark main_kv prefill: build the context window for every prefill
    // position (reference DSparkAttention start_pos==0 branch semantics).
    // DSpark main_kv prefill: build the context window for every prefill
    // position (reference DSparkAttention start_pos==0 branch semantics).
    // Skipped during verify batches (accumulate disabled) — the verify caller
    // commits accepted positions via moe_infer_dspark_commit instead.
    if (eng->dspark_engine && eng->dspark_accumulate_enabled &&
        eng->dspark_hidden3 && eng->dspark_hidden3_n == n_tokens) {
        dspark_build_window((DSparkEngine *)eng->dspark_engine,
                            eng->dspark_hidden3, n_tokens, start_pos);
    }

    // SMELT: do NOT count prefill tokens for warmup statistics.
    // Prefill uses hash-routing for layers 0-2 (routing_counts all 0), so
    // triggering warmup here would preload the wrong experts.
    // Warmup is counted only in moe_infer_forward (decode path).
    (void)0; // prefill path intentionally skips smelt_tokens_seen

    // Dump last token's final residual for diagnostics
    if (getenv("DSV4_DUMP_DIR") && getenv("MF_DBG")) {
        const char *dd = getenv("DSV4_DUMP_DIR");
        char path[1024];
        snprintf(path, sizeof(path), "%s/L42_residual_last.bin", dd);
        FILE *f = fopen(path, "wb");
        if (f) {
            float *last = hidden_batch + (size_t)(n_tokens-1) * (MHC_MULT * DIM);
            fwrite(last, sizeof(float), MHC_MULT * DIM, f);
            fclose(f);
        }
        KVCache *kvc0 = &eng->kv_cache[0];
        if (kvc0->kv && kvc0->len > 0) {
            snprintf(path, sizeof(path), "%s/L0_kvcache_prefill.bin", dd);
            FILE *fkv = fopen(path, "wb");
            if (fkv) {
                fwrite(kvc0->kv, sizeof(uint16_t), (size_t)kvc0->len * KV_LORA_RANK, fkv);
                fclose(fkv);
            }
        }
    }
    eng->in_batch--;
    return 0;
}

// ============================================================================
// Init / Deinit
// ============================================================================

MoEInferEngine *moe_infer_init(const char *packed_dir,
                                const char *kernel_src, unsigned long kernel_src_len) {
    MoEInferEngine *eng = (MoEInferEngine *)calloc(1, sizeof(MoEInferEngine));
    if (!eng) return NULL;

    if (init_metal(eng, kernel_src, kernel_src_len) != 0) {
        release_wave4_bucket_pipelines(eng);
        free(eng);
        return NULL;
    }

    // Detect int2_gs64 (2-bit) expert format: packed_dir/manifest.json with
    // "expert_format": "int2_gs64". Minimal text scan, no JSON parser needed.
    if (!eng->use_affine_experts) {
        char mpath[4096];
        snprintf(mpath, sizeof(mpath), "%s/manifest.json", packed_dir);
        int mfd = open(mpath, O_RDONLY);
        if (mfd >= 0) {
            char mbuf[4097] = {0};
            ssize_t mn = read(mfd, mbuf, 4096);
            close(mfd);
            if (mn > 0 && strstr(mbuf, "int2_gs64")) {
                eng->expert_format_2b = 1;
                eng->active_expert_size = EXPERT_SIZE_2B;
                fprintf(stderr, "Metal engine: int2_gs64 expert format detected (2-bit, expert_size=%d)\n",
                        EXPERT_SIZE_2B);
            }
        }
    }

    char path[4096];
    for (int l = 0; l < N_LAYERS; l++) {
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", packed_dir, l);
        eng->packed_fd[l] = open(path, O_RDONLY);
        if (eng->packed_fd[l] < 0) {
            fprintf(stderr, "open %s: %s\n", path, strerror(errno));
            // The engine is not marked initialized yet, so deinit's public
            // guard cannot own these Task 3.2 buffers on this failure path.
            release_persistent_routing_buffers(eng);
            release_wave4_bucket_pipelines(eng);
            moe_infer_deinit(eng); return NULL;
        }
        fcntl(eng->packed_fd[l], F_RDAHEAD, 1);
    }

    IOPool *io = (IOPool *)calloc(1, sizeof(IOPool));
    io_pool_init(io);
    eng->io_pool = io;

    // Dedicated pool for async prediction prefetch (kept separate from the
    // critical-path pool so prefetches never steal bandwidth from blocking reads)
    IOPool *pf = (IOPool *)calloc(1, sizeof(IOPool));
    io_pool_init(pf);
    eng->prefetch_pool = pf;
    eng->prefetch_layer = -1;
    eng->prefetch_active = false;
    eng->in_batch = 0;
    // NATIVE_PREFETCH: "1"=on (opt-in), unset/other=off.
    // Default OFF: official benchmark A/B (2026-07-18, M4 Pro 48GB) showed
    // 0.851 tok/s with auto-prefetch vs 0.889 without (-4.3%). Mechanism: on
    // cold SSD the 80MB prefetch takes longer than the ~11ms GPU window, and
    // prefetch_consume waits for the whole batch (incl. ~61% useless experts),
    // blocking the critical path. Code kept for future refinement (wait only
    // on hit slots; re-evaluate on long-decode workloads where hit rate is
    // ~39% and io= dropped 6.8→3.84ms/layer).
    const char *pfenv = getenv("NATIVE_PREFETCH");
    eng->prefetch_disabled = (pfenv && pfenv[0] == '1') ? 0 : 1;

    eng->initialized = true;
    // Initialize SMELT fields (disabled by default)
    memset(eng->routing_counts, 0, sizeof(eng->routing_counts));
    eng->smelt_warmup_tokens = 0;
    eng->smelt_n_per_layer   = 0;
    eng->smelt_tokens_seen   = 0;
    eng->smelt_warmup_done   = false;
    eng->smelt_enabled       = false;
    eng->smelt_in_decode_phase = false;
    eng->smelt_penalty       = 1e9f;
    fprintf(stderr, "Metal engine: %d expert files opened\n", N_LAYERS);
    return eng;
}

void moe_infer_set_weights(MoEInferEngine *eng,
    const float *embed, int vocab_size, const float *lm_head, const float *final_norm,
    const float **input_norms, const float **attn_norms,
    const float **gate_proj_w, const float **gate_bias_w) {
    eng->embed = embed;
    eng->vocab_size = vocab_size;
    eng->lm_head = lm_head;
    eng->final_norm = final_norm;
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    for (int i = 0; i < N_LAYERS; i++) {
        eng->input_norms[i] = input_norms[i];
        eng->attn_norms[i] = attn_norms[i];
        eng->gate_proj[i]  = gate_proj_w[i];
        eng->gate_bias[i]  = gate_bias_w[i];
        // Pre-allocate persistent GPU buffer for gate_bias (avoids per-call newBufferWithBytes)
        if (gate_bias_w[i] && !eng->gate_bias_gpu[i]) {
            eng->gate_bias_gpu[i] = (void *)[d newBufferWithBytes:(void*)gate_bias_w[i]
                length:N_EXPERTS*sizeof(float) options:MTLResourceStorageModeShared];
        }
        // Upload per-layer norm weights to persistent GPU buffers
        if (input_norms[i] && !eng->buf_input_norm_gpu[i]) {
            eng->buf_input_norm_gpu[i] = (void *)[d newBufferWithBytes:(void*)input_norms[i] length:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        }
        if (attn_norms[i] && !eng->buf_attn_norm_gpu[i]) {
            eng->buf_attn_norm_gpu[i] = (void *)[d newBufferWithBytes:(void*)attn_norms[i] length:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        }
        if (gate_proj_w[i] && !eng->buf_gate_proj_gpu[i]) {
            eng->buf_gate_proj_gpu[i] = (void *)[d newBufferWithBytes:(void*)gate_proj_w[i] length:(size_t)N_EXPERTS*DIM*sizeof(float) options:MTLResourceStorageModeShared];
        }
    }
}

void moe_infer_set_layer_attn(MoEInferEngine *eng, int layer, AttnWeights attn) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->attn[layer] = attn;
}

void moe_infer_set_layer_shared(MoEInferEngine *eng, int layer, SharedExpert se) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->shared[layer] = se;
}

void moe_infer_set_layer_tid2eid(MoEInferEngine *eng, int layer, const int64_t *tid2eid) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->tid2eid[layer] = tid2eid;
}

void moe_infer_set_token_id(MoEInferEngine *eng, int token_id) {
    eng->current_token_id = token_id;
}

void moe_infer_set_debug_capture(
    MoEInferEngine *eng,
    MoeInferDebugCaptureFn callback,
    void *context) {
    if (!eng) return;
    eng->debug_capture_context = callback ? context : NULL;
    eng->debug_capture = callback;
}

int moe_infer_batch_config_validate(const NativeBatchConfig *config) {
    if (!config ||
        config->enable_batch_prefill > 1u ||
        config->enable_batch_mla > 1u ||
        config->enable_expert_buckets > 1u ||
        config->enable_batched_expert_kernels > 1u ||
        config->enable_dspark_batch_verify > 1u ||
        config->max_prefill_chunk == 0u ||
        config->max_prefill_chunk > MAX_SEQ_LEN ||
        config->max_verify_chunk == 0u ||
        config->max_verify_chunk > MAX_SEQ_LEN ||
        config->workspace_budget_bytes == 0u) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    return NATIVE_BATCH_OK;
}

int moe_infer_set_batch_config(
    MoEInferEngine *eng,
    const NativeBatchConfig *config) {
    if (!eng || !eng->initialized ||
        moe_infer_batch_config_validate(config) != NATIVE_BATCH_OK) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if (eng->batch_config_set != 0u || eng->forward_batch_workspace) {
        return NATIVE_BATCH_INVALID_STATE;
    }

    NativeBatchWorkspace *workspace = NULL;
    if (config->enable_expert_buckets == 1u &&
        config->enable_batched_expert_kernels == 1u) {
        if (eng->use_affine_experts || eng->expert_format_2b != 0) {
            return NATIVE_BATCH_INVALID_STATE;
        }
        const uint64_t fixed_bytes = UINT64_C(8) * 1024u * 1024u;
        const uint64_t per_token_bytes = UINT64_C(512) * 1024u;
        if (config->max_prefill_chunk >
            (UINT64_MAX - fixed_bytes) / per_token_bytes) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }
        const uint64_t allocated_bytes = fixed_bytes +
            (uint64_t)config->max_prefill_chunk * per_token_bytes;
        if (allocated_bytes > config->workspace_budget_bytes) {
            return NATIVE_BATCH_BUDGET_EXCEEDED;
        }
        workspace = (NativeBatchWorkspace *)calloc(1, sizeof(*workspace));
        if (!workspace) return NATIVE_BATCH_CAPACITY_EXCEEDED;
        int status = moe_infer_batch_workspace_init(
            workspace, config->max_prefill_chunk, allocated_bytes,
            config->workspace_budget_bytes);
        if (status != NATIVE_BATCH_OK) {
            free(workspace);
            return status;
        }
        uint32_t pool_capacity = config->max_prefill_chunk >
                N_EXPERTS / N_ACTIVE
            ? N_EXPERTS
            : config->max_prefill_chunk * N_ACTIVE;
        if (pool_capacity < N_ACTIVE) pool_capacity = N_ACTIVE;
        status = moe_infer_batch_pool_prepare(eng, pool_capacity);
        if (status != NATIVE_BATCH_OK) {
            moe_infer_batch_workspace_deinit(workspace);
            free(workspace);
            return status;
        }
    }

    eng->batch_enable_prefill = config->enable_batch_prefill;
    eng->batch_enable_mla = config->enable_batch_mla;
    eng->batch_enable_expert_buckets = config->enable_expert_buckets;
    eng->batch_enable_batched_expert_kernels =
        config->enable_batched_expert_kernels;
    eng->batch_enable_dspark_verify = config->enable_dspark_batch_verify;
    eng->batch_max_prefill_chunk = config->max_prefill_chunk;
    eng->batch_max_verify_chunk = config->max_verify_chunk;
    eng->batch_workspace_budget_bytes = config->workspace_budget_bytes;
    eng->forward_batch_workspace = workspace;
    eng->batch_config_set = 1u;
    return NATIVE_BATCH_OK;
}

void moe_infer_get_batch_path_counters(
    const MoEInferEngine *eng,
    uint64_t *out_scalar_layer_calls,
    uint64_t *out_batched_moe_layers) {
    if (out_scalar_layer_calls) {
        *out_scalar_layer_calls = eng ? eng->batch_scalar_layer_calls : 0u;
    }
    if (out_batched_moe_layers) {
        *out_batched_moe_layers = eng ? eng->batch_batched_moe_layers : 0u;
    }
}

static bool batch_workspace_is_owned(const NativeBatchWorkspace *workspace) {
    return workspace && workspace->self_cookie == (uintptr_t)workspace &&
        workspace->arena_epoch != 0u && workspace->storage &&
        workspace->allocated_bytes > 0u &&
        workspace->used_bytes <= workspace->allocated_bytes &&
        workspace->high_water_bytes <= workspace->allocated_bytes;
}

int moe_infer_batch_workspace_init(
    NativeBatchWorkspace *workspace,
    uint32_t capacity_tokens,
    uint64_t allocated_bytes,
    uint64_t budget_bytes) {
    if (!workspace || workspace->arena_epoch || workspace->storage ||
        workspace->self_cookie || workspace->metal_buffer ||
        workspace->metal_buffer_release ||
        workspace->metal_buffer_offset != 0u ||
        workspace->metal_buffer_creations != 0u ||
        capacity_tokens == 0u || capacity_tokens > MAX_SEQ_LEN ||
        budget_bytes == 0u || allocated_bytes == 0u ||
        allocated_bytes > SIZE_MAX) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if (allocated_bytes > budget_bytes) return NATIVE_BATCH_BUDGET_EXCEEDED;

    // Metal no-copy workspace views require page-aligned host storage. This is
    // a stronger internal guarantee than the public 256-byte reserve contract;
    // callers still cannot request alignment above 256 bytes.
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || ((size_t)page_size & ((size_t)page_size - 1u)) != 0u) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    const size_t storage_alignment = page_size > (long)NATIVE_BATCH_WORKSPACE_ALIGNMENT
        ? (size_t)page_size
        : (size_t)NATIVE_BATCH_WORKSPACE_ALIGNMENT;
    void *storage = NULL;
    if (posix_memalign(&storage, storage_alignment, (size_t)allocated_bytes) != 0 || !storage) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    memset(storage, 0, (size_t)allocated_bytes);
    *workspace = (NativeBatchWorkspace){
        .capacity_tokens = capacity_tokens,
        .arena_epoch = 1u,
        .budget_bytes = budget_bytes,
        .allocated_bytes = allocated_bytes,
        .storage = storage,
        .self_cookie = (uintptr_t)workspace,
    };
    return NATIVE_BATCH_OK;
}

void moe_infer_batch_workspace_reset(NativeBatchWorkspace *workspace) {
    if (!batch_workspace_is_owned(workspace)) return;
    workspace->used_bytes = 0u;
    if (workspace->arena_epoch == UINT32_MAX) {
        // Never wrap to an epoch that an arbitrarily old typed view may hold.
        // Deinit remains available, but no future reserve can reuse this arena.
        workspace->arena_epoch = 0u;
        return;
    }
    workspace->arena_epoch += 1u;
}

void *moe_infer_batch_workspace_reserve(
    NativeBatchWorkspace *workspace,
    uint64_t size_bytes,
    uint64_t alignment) {
    if (!batch_workspace_is_owned(workspace) || size_bytes == 0u ||
        alignment == 0u || alignment > NATIVE_BATCH_WORKSPACE_ALIGNMENT ||
        (alignment & (alignment - 1u)) != 0u) {
        return NULL;
    }
    const uint64_t mask = alignment - 1u;
    if (workspace->used_bytes > UINT64_MAX - mask) return NULL;
    const uint64_t offset = (workspace->used_bytes + mask) & ~mask;
    if (offset > workspace->allocated_bytes ||
        size_bytes > workspace->allocated_bytes - offset) {
        return NULL;
    }
    workspace->used_bytes = offset + size_bytes;
    if (workspace->high_water_bytes < workspace->used_bytes) {
        workspace->high_water_bytes = workspace->used_bytes;
    }
    return (uint8_t *)workspace->storage + offset;
}

void moe_infer_batch_workspace_deinit(NativeBatchWorkspace *workspace) {
    if (!workspace || workspace->self_cookie != (uintptr_t)workspace) return;
    if (workspace->metal_buffer && workspace->metal_buffer_release) {
        workspace->metal_buffer_release(workspace->metal_buffer);
    }
    free(workspace->storage);
    memset(workspace, 0, sizeof(*workspace));
}

_Static_assert(sizeof(NativeActivationPair) == 16,
    "NativeActivationPair ABI drift");
_Static_assert(sizeof(NativeExpertBucketWorkspace) == 104,
    "NativeExpertBucketWorkspace ABI drift");

typedef struct NativeExpertBucketWorkspacePlan {
    uint64_t activation_pairs_offset;
    uint64_t bucketed_pairs_offset;
    uint64_t counts_offset;
    uint64_t offsets_offset;
    uint64_t unique_expert_ids_offset;
    uint64_t inverse_mapping_offset;
    uint64_t total_bytes;
    uint32_t pair_capacity;
} NativeExpertBucketWorkspacePlan;

static bool batch_u64_multiply(
    uint64_t left, uint64_t right, uint64_t *product) {
    if (!product || (left != 0u && right > UINT64_MAX / left)) return false;
    *product = left * right;
    return true;
}

static bool batch_u64_add(uint64_t left, uint64_t right, uint64_t *sum) {
    if (!sum || right > UINT64_MAX - left) return false;
    *sum = left + right;
    return true;
}

static bool batch_workspace_align(uint64_t value, uint64_t *aligned) {
    const uint64_t mask = NATIVE_BATCH_WORKSPACE_ALIGNMENT - 1u;
    if (!aligned || value > UINT64_MAX - mask) return false;
    *aligned = (value + mask) & ~mask;
    return true;
}

static bool batch_workspace_plan_region(
    uint64_t *cursor, uint64_t bytes, uint64_t *offset) {
    uint64_t aligned = 0u;
    if (!cursor || !offset || bytes == 0u ||
        !batch_workspace_align(*cursor, &aligned) ||
        !batch_u64_add(aligned, bytes, cursor)) {
        return false;
    }
    *offset = aligned;
    return true;
}

static int expert_bucket_workspace_plan(
    uint32_t batch_tokens,
    NativeExpertBucketWorkspacePlan *plan) {
    if (!plan || batch_tokens == 0u) return NATIVE_BATCH_INVALID_ARGUMENT;
    memset(plan, 0, sizeof(*plan));

    uint64_t pair_capacity = 0u;
    uint64_t pair_bytes = 0u;
    uint64_t inverse_bytes = 0u;
    uint64_t counts_bytes = 0u;
    uint64_t offsets_bytes = 0u;
    if (!batch_u64_multiply(batch_tokens, N_ACTIVE, &pair_capacity) ||
        pair_capacity > UINT32_MAX ||
        !batch_u64_multiply(
            pair_capacity, sizeof(NativeActivationPair), &pair_bytes) ||
        !batch_u64_multiply(pair_capacity, sizeof(uint32_t), &inverse_bytes) ||
        !batch_u64_multiply(N_EXPERTS, sizeof(uint32_t), &counts_bytes) ||
        !batch_u64_multiply(
            (uint64_t)N_EXPERTS + 1u, sizeof(uint32_t), &offsets_bytes)) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }

    uint64_t cursor = 0u;
    if (!batch_workspace_plan_region(
            &cursor, pair_bytes, &plan->activation_pairs_offset) ||
        !batch_workspace_plan_region(
            &cursor, pair_bytes, &plan->bucketed_pairs_offset) ||
        !batch_workspace_plan_region(
            &cursor, counts_bytes, &plan->counts_offset) ||
        !batch_workspace_plan_region(
            &cursor, offsets_bytes, &plan->offsets_offset) ||
        !batch_workspace_plan_region(
            &cursor, counts_bytes, &plan->unique_expert_ids_offset) ||
        !batch_workspace_plan_region(
            &cursor, inverse_bytes, &plan->inverse_mapping_offset) ||
        !batch_workspace_align(cursor, &plan->total_bytes)) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    plan->pair_capacity = (uint32_t)pair_capacity;
    return NATIVE_BATCH_OK;
}

int moe_infer_expert_bucket_workspace_bytes(
    uint32_t batch_tokens,
    uint64_t *out_bytes) {
    if (!out_bytes) return NATIVE_BATCH_INVALID_ARGUMENT;
    NativeExpertBucketWorkspacePlan plan;
    const int status = expert_bucket_workspace_plan(batch_tokens, &plan);
    if (status != NATIVE_BATCH_OK) return status;
    *out_bytes = plan.total_bytes;
    return NATIVE_BATCH_OK;
}

int moe_infer_expert_bucket_workspace_bind(
    NativeBatchWorkspace *workspace,
    uint32_t batch_tokens,
    NativeExpertBucketWorkspace *out_view) {
    if (!workspace || !out_view || out_view->activation_pairs ||
        out_view->bucketed_pairs || out_view->counts || out_view->offsets ||
        out_view->unique_expert_ids || out_view->inverse_mapping ||
        out_view->owner_workspace || out_view->self_cookie ||
        out_view->arena_offset_bytes || out_view->arena_size_bytes ||
        out_view->batch_tokens || out_view->pair_capacity ||
        out_view->unique_capacity || out_view->unique_count ||
        out_view->arena_epoch) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if (!batch_workspace_is_owned(workspace)) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    if (batch_tokens == 0u || batch_tokens > workspace->capacity_tokens) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }

    NativeExpertBucketWorkspacePlan plan;
    const int plan_status = expert_bucket_workspace_plan(batch_tokens, &plan);
    if (plan_status != NATIVE_BATCH_OK) return plan_status;
    uint8_t *base = (uint8_t *)moe_infer_batch_workspace_reserve(
        workspace, plan.total_bytes, NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    if (!base) return NATIVE_BATCH_CAPACITY_EXCEEDED;

    memset(base, 0, (size_t)plan.total_bytes);
    *out_view = (NativeExpertBucketWorkspace){
        .activation_pairs = (NativeActivationPair *)(
            base + plan.activation_pairs_offset),
        .bucketed_pairs = (NativeActivationPair *)(
            base + plan.bucketed_pairs_offset),
        .counts = (uint32_t *)(base + plan.counts_offset),
        .offsets = (uint32_t *)(base + plan.offsets_offset),
        .unique_expert_ids = (uint32_t *)(
            base + plan.unique_expert_ids_offset),
        .inverse_mapping = (uint32_t *)(base + plan.inverse_mapping_offset),
        .owner_workspace = workspace,
        .arena_offset_bytes = (uint64_t)(
            base - (uint8_t *)workspace->storage),
        .arena_size_bytes = plan.total_bytes,
        .batch_tokens = batch_tokens,
        .pair_capacity = plan.pair_capacity,
        .unique_capacity = N_EXPERTS,
        .arena_epoch = workspace->arena_epoch,
    };
    out_view->self_cookie = (uintptr_t)out_view;
    return NATIVE_BATCH_OK;
}

static bool expert_bucket_workspace_is_owned(
    const NativeExpertBucketWorkspace *view,
    NativeExpertBucketWorkspacePlan *plan) {
    if (!view || view->self_cookie != (uintptr_t)view ||
        !view->owner_workspace || !batch_workspace_is_owned(view->owner_workspace) ||
        !view->activation_pairs || !view->bucketed_pairs || !view->counts ||
        !view->offsets || !view->unique_expert_ids || !view->inverse_mapping ||
        view->arena_epoch == 0u ||
        view->arena_epoch != view->owner_workspace->arena_epoch ||
        view->unique_capacity != N_EXPERTS ||
        view->unique_count > view->unique_capacity ||
        expert_bucket_workspace_plan(view->batch_tokens, plan) != NATIVE_BATCH_OK ||
        view->pair_capacity != plan->pair_capacity ||
        view->arena_size_bytes != plan->total_bytes ||
        view->arena_offset_bytes > view->owner_workspace->allocated_bytes ||
        view->arena_size_bytes >
            view->owner_workspace->allocated_bytes - view->arena_offset_bytes ||
        view->owner_workspace->used_bytes <
            view->arena_offset_bytes + view->arena_size_bytes) {
        return false;
    }
    uint8_t *base = (uint8_t *)view->owner_workspace->storage +
        view->arena_offset_bytes;
    return view->activation_pairs == (NativeActivationPair *)(
               base + plan->activation_pairs_offset) &&
        view->bucketed_pairs == (NativeActivationPair *)(
               base + plan->bucketed_pairs_offset) &&
        view->counts == (uint32_t *)(base + plan->counts_offset) &&
        view->offsets == (uint32_t *)(base + plan->offsets_offset) &&
        view->unique_expert_ids == (uint32_t *)(
               base + plan->unique_expert_ids_offset) &&
        view->inverse_mapping == (uint32_t *)(
               base + plan->inverse_mapping_offset);
}

int moe_infer_expert_bucket_workspace_reset(
    NativeExpertBucketWorkspace *view) {
    NativeExpertBucketWorkspacePlan plan;
    if (!expert_bucket_workspace_is_owned(view, &plan)) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    memset(view->counts, 0, (size_t)N_EXPERTS * sizeof(uint32_t));
    memset(view->offsets, 0, (size_t)(N_EXPERTS + 1u) * sizeof(uint32_t));
    memset(view->unique_expert_ids, 0,
        (size_t)N_EXPERTS * sizeof(uint32_t));
    memset(view->inverse_mapping, 0,
        (size_t)view->pair_capacity * sizeof(uint32_t));
    view->unique_count = 0u;
    return NATIVE_BATCH_OK;
}

int moe_infer_expert_bucket_build(
    NativeExpertBucketWorkspace *view) {
    NativeExpertBucketWorkspacePlan plan;
    if (!expert_bucket_workspace_is_owned(view, &plan)) {
        return NATIVE_BATCH_INVALID_STATE;
    }

    // Validate the complete token-major source before publishing any derived
    // bucket state. Task 3.2 guarantees this shape, but keeping the boundary
    // defensive prevents malformed routes from producing a partial mapping.
    for (uint32_t token = 0u; token < view->batch_tokens; ++token) {
        uint8_t seen[N_EXPERTS] = {0};
        for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
            const uint32_t source = token * N_ACTIVE + rank;
            const NativeActivationPair *pair = &view->activation_pairs[source];
            if (pair->token_id != token || pair->route_rank != rank ||
                pair->expert_id >= N_EXPERTS || seen[pair->expert_id] ||
                !isfinite(pair->weight)) {
                return NATIVE_BATCH_INVALID_ARGUMENT;
            }
            seen[pair->expert_id] = 1u;
        }
    }

    memset(view->counts, 0, (size_t)N_EXPERTS * sizeof(uint32_t));
    memset(view->offsets, 0, (size_t)(N_EXPERTS + 1u) * sizeof(uint32_t));
    memset(view->unique_expert_ids, 0,
        (size_t)N_EXPERTS * sizeof(uint32_t));
    memset(view->inverse_mapping, 0,
        (size_t)view->pair_capacity * sizeof(uint32_t));

    for (uint32_t source = 0u; source < view->pair_capacity; ++source) {
        ++view->counts[view->activation_pairs[source].expert_id];
    }

    uint32_t unique_count = 0u;
    for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
        view->offsets[expert + 1u] =
            view->offsets[expert] + view->counts[expert];
        if (view->counts[expert] != 0u) {
            view->unique_expert_ids[unique_count++] = expert;
        }
    }

    uint32_t cursors[N_EXPERTS];
    memcpy(cursors, view->offsets, sizeof(cursors));
    // Traversing source pairs in token-major order makes every expert bucket
    // stable. route_rank and weight are copied verbatim for deterministic
    // token-order reconstruction during combine.
    for (uint32_t source = 0u; source < view->pair_capacity; ++source) {
        const NativeActivationPair pair = view->activation_pairs[source];
        const uint32_t destination = cursors[pair.expert_id]++;
        view->bucketed_pairs[destination] = pair;
        view->inverse_mapping[source] = destination;
    }
    view->unique_count = unique_count;
    return NATIVE_BATCH_OK;
}

int moe_infer_batch_pool_prepare(
    MoEInferEngine *eng,
    uint32_t capacity) {
    if (!eng || capacity < N_ACTIVE || capacity > N_EXPERTS ||
        eng->active_expert_size == 0u ||
        capacity > SIZE_MAX / eng->active_expert_size) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if (eng->batch_pool || eng->batch_wrap || eng->batch_eids ||
        eng->batch_pins || eng->batch_capacity != 0 ||
        eng->batch_slot_size != 0u) {
        if (eng->batch_pool && eng->batch_wrap && eng->batch_eids &&
            eng->batch_pins && eng->batch_capacity >= (int)capacity &&
            eng->batch_slot_size == eng->active_expert_size) {
            return NATIVE_BATCH_OK;
        }
        return NATIVE_BATCH_INVALID_STATE;
    }

    uint8_t *pool = NULL;
    void **wrap = NULL;
    int *eids = NULL;
    uint8_t *pins = NULL;
    const size_t pool_bytes = (size_t)capacity * eng->active_expert_size;
    if (posix_memalign((void **)&pool, 2u * 1024u * 1024u, pool_bytes) == 0 &&
        pool) {
        wrap = (void **)calloc(capacity, sizeof(void *));
        eids = (int *)malloc((size_t)capacity * sizeof(int));
        pins = (uint8_t *)calloc(capacity, sizeof(uint8_t));
    }
    if (!pool || !wrap || !eids || !pins) {
        free(pool);
        free(wrap);
        free(eids);
        free(pins);
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    for (uint32_t slot = 0u; slot < capacity; ++slot) eids[slot] = -1;
    eng->batch_pool = pool;
    eng->batch_wrap = wrap;
    eng->batch_eids = eids;
    eng->batch_pins = pins;
    eng->batch_capacity = (int)capacity;
    eng->batch_slot_size = eng->active_expert_size;
    eng->batch_count = 0;
    eng->batch_hits = 0;
    eng->unique_resolver_owner = 0u;
    return NATIVE_BATCH_OK;
}

void moe_infer_unique_expert_resolution_init(
    NativeUniqueExpertResolution *resolution) {
    if (!resolution) return;
    memset(resolution, 0, sizeof(*resolution));
    resolution->self_cookie = (uintptr_t)resolution;
    resolution->metrics.error_expert = UINT32_MAX;
}

static bool expert_bucket_index_is_complete(
    const NativeExpertBucketWorkspace *buckets) {
    NativeExpertBucketWorkspacePlan plan;
    if (!expert_bucket_workspace_is_owned(buckets, &plan) ||
        buckets->unique_count == 0u ||
        buckets->offsets[0] != 0u ||
        buckets->offsets[N_EXPERTS] != buckets->pair_capacity) {
        return false;
    }
    uint32_t unique = 0u;
    uint32_t previous_unique = 0u;
    for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
        if (buckets->offsets[expert + 1u] < buckets->offsets[expert] ||
            buckets->offsets[expert + 1u] - buckets->offsets[expert] !=
                buckets->counts[expert]) {
            return false;
        }
        if (buckets->counts[expert] != 0u) {
            if (unique >= buckets->unique_count ||
                buckets->unique_expert_ids[unique] != expert ||
                (unique > 0u && expert <= previous_unique)) {
                return false;
            }
            previous_unique = expert;
            ++unique;
        }
    }
    return unique == buckets->unique_count;
}

static void bucket_workspace_release_metal_buffer(void *opaque) {
    if (opaque) [(id<MTLBuffer>)opaque release];
}

static id<MTLBuffer> bucket_workspace_metal_buffer(
    id<MTLDevice> device, NativeBatchWorkspace *workspace) {
    if (!device || !batch_workspace_is_owned(workspace) ||
        workspace->allocated_bytes > (uint64_t)NSUIntegerMax ||
        workspace->metal_buffer_offset > (uint64_t)NSUIntegerMax) {
        return nil;
    }
    if (workspace->metal_buffer) {
        id<MTLBuffer> existing = (id<MTLBuffer>)workspace->metal_buffer;
        const uint64_t length = (uint64_t)existing.length;
        if (existing.device != device ||
            workspace->metal_buffer_offset > length ||
            workspace->allocated_bytes >
                length - workspace->metal_buffer_offset) {
            return nil;
        }
        return existing;
    }
    // A nonzero offset denotes a borrowed parent view; only its parent may
    // create and own the no-copy MTLBuffer.
    if (workspace->metal_buffer_offset != 0u ||
        workspace->metal_buffer_release ||
        workspace->metal_buffer_creations == UINT64_MAX) {
        return nil;
    }
    id<MTLBuffer> created = [device
        newBufferWithBytesNoCopy:workspace->storage
        length:(NSUInteger)workspace->allocated_bytes
        options:MTLResourceStorageModeShared
        deallocator:nil];
    if (!created) return nil;
    workspace->metal_buffer = (void *)created;
    workspace->metal_buffer_release = bucket_workspace_release_metal_buffer;
    workspace->metal_buffer_creations++;
    return created;
}

int moe_infer_bucket_input_gather_f32(
    MoEInferEngine *eng,
    const float *hidden_rows,
    const NativeExpertBucketWorkspace *buckets,
    uint32_t expert_id,
    float **out_shared_rows,
    float **out_bucket_rows,
    uint32_t *out_bucket_count) {
    if (!eng || !eng->initialized || !eng->device || !eng->queue ||
        !eng->pipe_bucket_gather_f32 || !hidden_rows || !buckets ||
        !out_shared_rows || !out_bucket_rows || !out_bucket_count ||
        expert_id >= N_EXPERTS || !expert_bucket_index_is_complete(buckets)) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    NativeBatchWorkspace *workspace = buckets->owner_workspace;
    NativeExpertBucketWorkspacePlan plan;
    if (!workspace || !expert_bucket_workspace_is_owned(buckets, &plan) ||
        !batch_workspace_is_owned(workspace)) {
        return NATIVE_BATCH_INVALID_STATE;
    }

    const uint32_t bucket_start = buckets->offsets[expert_id];
    const uint32_t bucket_count = buckets->counts[expert_id];
    if (bucket_count == 0u || bucket_count > buckets->batch_tokens ||
        bucket_start > buckets->pair_capacity ||
        bucket_count > buckets->pair_capacity - bucket_start) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    // Validate all device indices before reserving or writing staging memory.
    for (uint32_t row = 0u; row < bucket_count; ++row) {
        const NativeActivationPair *pair =
            &buckets->bucketed_pairs[bucket_start + row];
        if (pair->expert_id != expert_id ||
            pair->token_id >= buckets->batch_tokens) {
            return NATIVE_BATCH_INVALID_ARGUMENT;
        }
    }

    uint64_t shared_values = 0u;
    uint64_t gathered_values = 0u;
    uint64_t shared_bytes = 0u;
    uint64_t gathered_bytes = 0u;
    uint64_t total_bytes = 0u;
    if (!batch_u64_multiply(buckets->batch_tokens, DIM, &shared_values) ||
        !batch_u64_multiply(bucket_count, DIM, &gathered_values) ||
        !batch_u64_multiply(shared_values, sizeof(float), &shared_bytes) ||
        !batch_u64_multiply(gathered_values, sizeof(float), &gathered_bytes) ||
        !batch_u64_add(shared_bytes, gathered_bytes, &total_bytes) ||
        total_bytes == 0u || total_bytes > SIZE_MAX) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    // Preflight the exact aligned reservation so capacity failure cannot create
    // a Metal view or advance the arena high-water mark.
    const uint64_t mask = NATIVE_BATCH_WORKSPACE_ALIGNMENT - 1u;
    if (workspace->used_bytes > UINT64_MAX - mask) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    const uint64_t staging_offset = (workspace->used_bytes + mask) & ~mask;
    if (staging_offset > workspace->allocated_bytes ||
        total_bytes > workspace->allocated_bytes - staging_offset) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }

    id<MTLDevice> device = (id<MTLDevice>)eng->device;
    id<MTLBuffer> arena_buffer =
        bucket_workspace_metal_buffer(device, workspace);
    if (!arena_buffer) return NATIVE_BATCH_INVALID_STATE;
    uint8_t *staging = (uint8_t *)moe_infer_batch_workspace_reserve(
        workspace, total_bytes, NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    if (!staging) return NATIVE_BATCH_CAPACITY_EXCEEDED;
    float *shared_rows = (float *)staging;
    float *gathered_rows = (float *)(staging + shared_bytes);
    memmove(shared_rows, hidden_rows, (size_t)shared_bytes);

    const uint64_t pairs_storage_offset = (uint64_t)(
        (const uint8_t *)buckets->bucketed_pairs -
        (const uint8_t *)workspace->storage);
    const uint64_t metal_base = workspace->metal_buffer_offset;
    uint64_t source_metal_offset = 0u;
    uint64_t pairs_metal_offset = 0u;
    uint64_t output_metal_offset = 0u;
    if (!batch_u64_add(
            metal_base, staging_offset, &source_metal_offset) ||
        !batch_u64_add(
            metal_base, pairs_storage_offset, &pairs_metal_offset) ||
        !batch_u64_add(
            source_metal_offset, shared_bytes, &output_metal_offset) ||
        source_metal_offset > (uint64_t)NSUIntegerMax ||
        pairs_metal_offset > (uint64_t)NSUIntegerMax ||
        output_metal_offset > (uint64_t)NSUIntegerMax) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> command =
            [(id<MTLCommandQueue>)eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder) return NATIVE_BATCH_INVALID_STATE;
        [encoder setComputePipelineState:
            (id<MTLComputePipelineState>)eng->pipe_bucket_gather_f32];
        [encoder setBuffer:arena_buffer
                    offset:(NSUInteger)source_metal_offset
                   atIndex:0];
        [encoder setBuffer:arena_buffer
                    offset:(NSUInteger)pairs_metal_offset
                   atIndex:1];
        [encoder setBuffer:arena_buffer
                    offset:(NSUInteger)output_metal_offset
                   atIndex:2];
        const uint32_t batch_tokens = buckets->batch_tokens;
        const uint32_t hidden_dim = DIM;
        [encoder setBytes:&bucket_start length:sizeof(bucket_start) atIndex:3];
        [encoder setBytes:&bucket_count length:sizeof(bucket_count) atIndex:4];
        [encoder setBytes:&batch_tokens length:sizeof(batch_tokens) atIndex:5];
        [encoder setBytes:&hidden_dim length:sizeof(hidden_dim) atIndex:6];
        [encoder dispatchThreadgroups:
            MTLSizeMake((DIM + 255u) / 256u, bucket_count, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "[bucket-gather] Metal dispatch failed: %s\n",
                    command.error
                        ? [[command.error localizedDescription] UTF8String]
                        : "unknown error");
            return NATIVE_BATCH_INVALID_STATE;
        }
    }

    *out_shared_rows = shared_rows;
    *out_bucket_rows = gathered_rows;
    *out_bucket_count = bucket_count;
    return NATIVE_BATCH_OK;
}

static bool expert_bucket_combine_mapping_is_complete(
    const NativeExpertBucketWorkspace *buckets) {
    if (!expert_bucket_index_is_complete(buckets)) return false;
    for (uint32_t token = 0u; token < buckets->batch_tokens; ++token) {
        for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
            const uint32_t source = token * N_ACTIVE + rank;
            const NativeActivationPair *pair =
                &buckets->activation_pairs[source];
            const uint32_t destination = buckets->inverse_mapping[source];
            if (pair->token_id != token || pair->route_rank != rank ||
                pair->expert_id >= N_EXPERTS || !isfinite(pair->weight) ||
                destination >= buckets->pair_capacity ||
                memcmp(pair, &buckets->bucketed_pairs[destination],
                       sizeof(*pair)) != 0) {
                return false;
            }
        }
    }
    return true;
}

static bool finite_f32_array(const float *values, uint64_t count) {
    if (!values || count > SIZE_MAX / sizeof(float)) return false;
    for (uint64_t index = 0u; index < count; ++index) {
        if (!isfinite(values[index])) return false;
    }
    return true;
}

int moe_infer_expert_bucket_combine_reference_f32(
    const NativeExpertBucketWorkspace *buckets,
    const float *bucket_outputs,
    const float *shared_outputs,
    uint32_t hidden_dim,
    float *combined_outputs) {
    if (!buckets || !bucket_outputs || !shared_outputs || !combined_outputs ||
        hidden_dim == 0u ||
        !expert_bucket_combine_mapping_is_complete(buckets)) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    uint64_t bucket_values = 0u;
    uint64_t shared_values = 0u;
    if (!batch_u64_multiply(
            buckets->pair_capacity, hidden_dim, &bucket_values) ||
        !batch_u64_multiply(
            buckets->batch_tokens, hidden_dim, &shared_values) ||
        !finite_f32_array(bucket_outputs, bucket_values) ||
        !finite_f32_array(shared_outputs, shared_values)) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }

    // Validate every derived value in the exact route-rank order used by the
    // publication pass. This keeps caller output byte-for-byte unchanged when
    // finite operands overflow during multiplication, accumulation, or shared
    // addition, without adding a per-call staging allocation.
    for (uint32_t token = 0u; token < buckets->batch_tokens; ++token) {
        for (uint32_t column = 0u; column < hidden_dim; ++column) {
            float routed = 0.0f;
            for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
                const uint32_t source = token * N_ACTIVE + rank;
                const uint32_t bucket_row = buckets->inverse_mapping[source];
                const float contribution = bucket_outputs[
                    (size_t)bucket_row * hidden_dim + column] *
                    buckets->activation_pairs[source].weight;
                if (!isfinite(contribution)) return NATIVE_BATCH_INVALID_STATE;
                routed += contribution;
                if (!isfinite(routed)) return NATIVE_BATCH_INVALID_STATE;
            }
            const float combined = routed +
                shared_outputs[(size_t)token * hidden_dim + column];
            if (!isfinite(combined)) return NATIVE_BATCH_INVALID_STATE;
        }
    }

    for (uint32_t token = 0u; token < buckets->batch_tokens; ++token) {
        for (uint32_t column = 0u; column < hidden_dim; ++column) {
            float routed = 0.0f;
            for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
                const uint32_t source = token * N_ACTIVE + rank;
                const uint32_t bucket_row = buckets->inverse_mapping[source];
                routed += bucket_outputs[
                    (size_t)bucket_row * hidden_dim + column] *
                    buckets->activation_pairs[source].weight;
            }
            combined_outputs[(size_t)token * hidden_dim + column] = routed +
                shared_outputs[(size_t)token * hidden_dim + column];
        }
    }
    return NATIVE_BATCH_OK;
}

static bool ensure_shared_expert_weight_buffers(
    MoEInferEngine *eng, int layer) {
    if (!eng || layer < 0 || layer >= N_LAYERS || !eng->device) return false;
    bool all_present = true;
    bool any_present = false;
    for (int index = 0; index < 9; ++index) {
        all_present = all_present && eng->buf_shared_w[layer][index] != NULL;
        any_present = any_present || eng->buf_shared_w[layer][index] != NULL;
    }
    if (all_present) return true;
    if (any_present) return false;

    const QuantWeight *weights[3] = {
        &eng->shared[layer].gate,
        &eng->shared[layer].up,
        &eng->shared[layer].down,
    };
    id<MTLBuffer> created[9] = {nil};
    id<MTLDevice> device = (id<MTLDevice>)eng->device;
    for (int weight_index = 0; weight_index < 3; ++weight_index) {
        const QuantWeight *weight = weights[weight_index];
        if (!weight->packed || !weight->scales || !weight->biases ||
            weight->out_dim <= 0 || weight->in_dim <= 0 ||
            weight->group_size <= 0 ||
            weight->in_dim % weight->group_size != 0) {
            for (int index = 0; index < 9; ++index) [created[index] release];
            return false;
        }
        const size_t packed_bytes = (size_t)weight->out_dim *
            ((size_t)weight->in_dim / 8u) * sizeof(uint32_t);
        const size_t parameter_bytes = (size_t)weight->out_dim *
            ((size_t)weight->in_dim / (size_t)weight->group_size) *
            sizeof(float);
        created[weight_index * 3 + 0] = [device
            newBufferWithBytes:(void *)weight->packed
            length:packed_bytes
            options:MTLResourceStorageModeShared];
        created[weight_index * 3 + 1] = [device
            newBufferWithBytes:(void *)weight->scales
            length:parameter_bytes
            options:MTLResourceStorageModeShared];
        created[weight_index * 3 + 2] = [device
            newBufferWithBytes:(void *)weight->biases
            length:parameter_bytes
            options:MTLResourceStorageModeShared];
        if (!created[weight_index * 3 + 0] ||
            !created[weight_index * 3 + 1] ||
            !created[weight_index * 3 + 2]) {
            for (int index = 0; index < 9; ++index) [created[index] release];
            return false;
        }
    }
    for (int index = 0; index < 9; ++index) {
        eng->buf_shared_w[layer][index] = (void *)created[index];
    }
    return true;
}

int moe_infer_shared_expert_batch_f32(
    MoEInferEngine *eng,
    int layer,
    const float *shared_inputs,
    uint32_t batch_tokens,
    NativeBatchWorkspace *workspace,
    float **out_shared_outputs) {
    if (!eng || layer < 0 || layer >= N_LAYERS || !shared_inputs ||
        batch_tokens == 0u || !workspace || !out_shared_outputs) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if (!eng->initialized || !eng->device || !eng->queue ||
        !eng->pipe_dequant_matvec_affine || !eng->pipe_limited_swiglu) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    if (!batch_workspace_is_owned(workspace)) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    if (batch_tokens > workspace->capacity_tokens) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    const QuantWeight *gate = &eng->shared[layer].gate;
    const QuantWeight *up = &eng->shared[layer].up;
    const QuantWeight *down = &eng->shared[layer].down;
    if (gate->in_dim != DIM || gate->out_dim != INTERMEDIATE ||
        up->in_dim != DIM || up->out_dim != INTERMEDIATE ||
        down->in_dim != INTERMEDIATE || down->out_dim != DIM ||
        gate->group_size != 64 || up->group_size != 64 ||
        down->group_size != 64) {
        return NATIVE_BATCH_INVALID_STATE;
    }

    uint64_t input_values = 0u;
    uint64_t mid_values = 0u;
    uint64_t input_bytes = 0u;
    uint64_t mid_bytes = 0u;
    uint64_t total_bytes = 0u;
    if (!batch_u64_multiply(batch_tokens, DIM, &input_values) ||
        !batch_u64_multiply(batch_tokens, INTERMEDIATE, &mid_values)) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    if (!finite_f32_array(shared_inputs, input_values)) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if (!batch_u64_multiply(input_values, sizeof(float), &input_bytes) ||
        !batch_u64_multiply(mid_values, sizeof(float), &mid_bytes) ||
        !batch_u64_add(input_bytes, mid_bytes, &total_bytes) ||
        !batch_u64_add(total_bytes, mid_bytes, &total_bytes) ||
        !batch_u64_add(total_bytes, input_bytes, &total_bytes) ||
        total_bytes > SIZE_MAX) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }

    const uint64_t mask = NATIVE_BATCH_WORKSPACE_ALIGNMENT - 1u;
    if (workspace->used_bytes > UINT64_MAX - mask) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    const uint64_t staging_offset = (workspace->used_bytes + mask) & ~mask;
    if (staging_offset > workspace->allocated_bytes ||
        total_bytes > workspace->allocated_bytes - staging_offset) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }

    const uint64_t metal_base = workspace->metal_buffer_offset;
    uint64_t input_offset = 0u;
    uint64_t gate_offset = 0u;
    uint64_t up_offset = 0u;
    uint64_t output_offset = 0u;
    uint64_t end_offset = 0u;
    if (!batch_u64_add(metal_base, staging_offset, &input_offset) ||
        !batch_u64_add(input_offset, input_bytes, &gate_offset) ||
        !batch_u64_add(gate_offset, mid_bytes, &up_offset) ||
        !batch_u64_add(up_offset, mid_bytes, &output_offset) ||
        !batch_u64_add(output_offset, input_bytes, &end_offset) ||
        end_offset > (uint64_t)NSUIntegerMax) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    if (!ensure_shared_expert_weight_buffers(eng, layer)) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    id<MTLBuffer> arena = bucket_workspace_metal_buffer(
        (id<MTLDevice>)eng->device, workspace);
    if (!arena) return NATIVE_BATCH_INVALID_STATE;

    const uint64_t used_before = workspace->used_bytes;
    uint8_t *staging = (uint8_t *)moe_infer_batch_workspace_reserve(
        workspace, total_bytes, NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    if (!staging) return NATIVE_BATCH_CAPACITY_EXCEEDED;
    float *input_rows = (float *)staging;
    float *gate_rows = (float *)(staging + input_bytes);
    float *up_rows = (float *)(staging + input_bytes + mid_bytes);
    float *output_rows = (float *)(
        staging + input_bytes + mid_bytes + mid_bytes);
    memmove(input_rows, shared_inputs, (size_t)input_bytes);

    int dispatch_status = NATIVE_BATCH_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> command =
            [(id<MTLCommandQueue>)eng->queue commandBuffer];
        if (!command) dispatch_status = NATIVE_BATCH_INVALID_STATE;
        const QuantWeight *weights[3] = {gate, up, down};
        const NSUInteger source_offsets[3] = {
            (NSUInteger)input_offset,
            (NSUInteger)input_offset,
            (NSUInteger)gate_offset,
        };
        const NSUInteger destination_offsets[3] = {
            (NSUInteger)gate_offset,
            (NSUInteger)up_offset,
            (NSUInteger)output_offset,
        };
        for (int projection = 0;
             projection < 3 && dispatch_status == NATIVE_BATCH_OK;
             ++projection) {
            if (projection == 2) {
                id<MTLComputeCommandEncoder> activation =
                    [command computeCommandEncoder];
                if (!activation) {
                    dispatch_status = NATIVE_BATCH_INVALID_STATE;
                    break;
                }
                [activation setComputePipelineState:
                    (id<MTLComputePipelineState>)eng->pipe_limited_swiglu];
                [activation setBuffer:arena
                               offset:(NSUInteger)gate_offset
                              atIndex:0];
                [activation setBuffer:arena
                               offset:(NSUInteger)up_offset
                              atIndex:1];
                const uint32_t count = batch_tokens * INTERMEDIATE;
                [activation setBytes:&count length:sizeof(count) atIndex:2];
                [activation dispatchThreads:MTLSizeMake(count, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
                [activation endEncoding];
            }
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (!encoder) {
                dispatch_status = NATIVE_BATCH_INVALID_STATE;
                break;
            }
            const QuantWeight *weight = weights[projection];
            [encoder setComputePipelineState:
                (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
            for (int part = 0; part < 3; ++part) {
                [encoder setBuffer:(id<MTLBuffer>)
                    eng->buf_shared_w[layer][projection * 3 + part]
                            offset:0
                           atIndex:part];
            }
            [encoder setBuffer:arena
                         offset:source_offsets[projection]
                        atIndex:3];
            [encoder setBuffer:arena
                         offset:destination_offsets[projection]
                        atIndex:4];
            const uint32_t out_dim = (uint32_t)weight->out_dim;
            const uint32_t in_dim = (uint32_t)weight->in_dim;
            const uint32_t group_size = (uint32_t)weight->group_size;
            [encoder setBytes:&out_dim length:sizeof(out_dim) atIndex:5];
            [encoder setBytes:&in_dim length:sizeof(in_dim) atIndex:6];
            [encoder setBytes:&group_size length:sizeof(group_size) atIndex:7];
            [encoder dispatchThreads:MTLSizeMake(out_dim, batch_tokens, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            [encoder endEncoding];
        }
        if (dispatch_status == NATIVE_BATCH_OK) {
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted ||
                !finite_f32_array(output_rows, input_values)) {
                dispatch_status = NATIVE_BATCH_INVALID_STATE;
            }
        }
    }
    if (dispatch_status != NATIVE_BATCH_OK) {
        workspace->used_bytes = used_before;
        return dispatch_status;
    }
    *out_shared_outputs = output_rows;
    return NATIVE_BATCH_OK;
}

int moe_infer_bucket_combine_f32(
    MoEInferEngine *eng,
    const NativeExpertBucketWorkspace *buckets,
    const float *bucket_outputs,
    const float *shared_outputs,
    uint32_t hidden_dim,
    float **out_combined_outputs) {
    if (!eng || !buckets || !bucket_outputs || !shared_outputs ||
        hidden_dim == 0u || !out_combined_outputs) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if (!eng->initialized || !eng->device || !eng->queue ||
        !eng->pipe_bucket_combine_f32) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    NativeExpertBucketWorkspacePlan plan;
    if (!expert_bucket_workspace_is_owned(buckets, &plan)) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    if (!expert_bucket_combine_mapping_is_complete(buckets)) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    NativeBatchWorkspace *workspace = buckets->owner_workspace;

    uint64_t bucket_values = 0u;
    uint64_t shared_values = 0u;
    uint64_t bucket_bytes = 0u;
    uint64_t shared_bytes = 0u;
    uint64_t total_bytes = 0u;
    if (!batch_u64_multiply(
            buckets->pair_capacity, hidden_dim, &bucket_values) ||
        !batch_u64_multiply(
            buckets->batch_tokens, hidden_dim, &shared_values) ||
        !batch_u64_multiply(bucket_values, sizeof(float), &bucket_bytes) ||
        !batch_u64_multiply(shared_values, sizeof(float), &shared_bytes) ||
        !batch_u64_add(bucket_bytes, shared_bytes, &total_bytes) ||
        !batch_u64_add(total_bytes, shared_bytes, &total_bytes) ||
        total_bytes > SIZE_MAX) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    if (!finite_f32_array(bucket_outputs, bucket_values) ||
        !finite_f32_array(shared_outputs, shared_values)) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }

    const uint64_t mask = NATIVE_BATCH_WORKSPACE_ALIGNMENT - 1u;
    if (workspace->used_bytes > UINT64_MAX - mask) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    const uint64_t staging_offset = (workspace->used_bytes + mask) & ~mask;
    if (staging_offset > workspace->allocated_bytes ||
        total_bytes > workspace->allocated_bytes - staging_offset) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }

    const uint64_t metal_base = workspace->metal_buffer_offset;
    const uint64_t activation_storage_offset = (uint64_t)(
        (const uint8_t *)buckets->activation_pairs -
        (const uint8_t *)workspace->storage);
    const uint64_t inverse_storage_offset = (uint64_t)(
        (const uint8_t *)buckets->inverse_mapping -
        (const uint8_t *)workspace->storage);
    uint64_t bucket_offset = 0u;
    uint64_t activation_offset = 0u;
    uint64_t inverse_offset = 0u;
    uint64_t shared_offset = 0u;
    uint64_t combined_offset = 0u;
    uint64_t end_offset = 0u;
    if (!batch_u64_add(metal_base, staging_offset, &bucket_offset) ||
        !batch_u64_add(
            metal_base, activation_storage_offset, &activation_offset) ||
        !batch_u64_add(metal_base, inverse_storage_offset, &inverse_offset) ||
        !batch_u64_add(bucket_offset, bucket_bytes, &shared_offset) ||
        !batch_u64_add(shared_offset, shared_bytes, &combined_offset) ||
        !batch_u64_add(combined_offset, shared_bytes, &end_offset) ||
        end_offset > (uint64_t)NSUIntegerMax) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    id<MTLBuffer> arena = bucket_workspace_metal_buffer(
        (id<MTLDevice>)eng->device, workspace);
    if (!arena) return NATIVE_BATCH_INVALID_STATE;

    const uint64_t used_before = workspace->used_bytes;
    uint8_t *staging = (uint8_t *)moe_infer_batch_workspace_reserve(
        workspace, total_bytes, NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    if (!staging) return NATIVE_BATCH_CAPACITY_EXCEEDED;
    float *bucket_rows = (float *)staging;
    float *shared_rows = (float *)(staging + bucket_bytes);
    float *combined_rows = (float *)(
        staging + bucket_bytes + shared_bytes);
    memmove(bucket_rows, bucket_outputs, (size_t)bucket_bytes);
    memmove(shared_rows, shared_outputs, (size_t)shared_bytes);

    int dispatch_status = NATIVE_BATCH_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> command =
            [(id<MTLCommandQueue>)eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = command
            ? [command computeCommandEncoder]
            : nil;
        if (!command || !encoder) {
            dispatch_status = NATIVE_BATCH_INVALID_STATE;
        } else {
            [encoder setComputePipelineState:
                (id<MTLComputePipelineState>)eng->pipe_bucket_combine_f32];
            [encoder setBuffer:arena offset:(NSUInteger)bucket_offset atIndex:0];
            [encoder setBuffer:arena offset:(NSUInteger)activation_offset atIndex:1];
            [encoder setBuffer:arena offset:(NSUInteger)inverse_offset atIndex:2];
            [encoder setBuffer:arena offset:(NSUInteger)shared_offset atIndex:3];
            [encoder setBuffer:arena offset:(NSUInteger)combined_offset atIndex:4];
            const uint32_t batch_tokens = buckets->batch_tokens;
            const uint32_t pair_count = buckets->pair_capacity;
            [encoder setBytes:&batch_tokens length:sizeof(batch_tokens) atIndex:5];
            [encoder setBytes:&pair_count length:sizeof(pair_count) atIndex:6];
            [encoder setBytes:&hidden_dim length:sizeof(hidden_dim) atIndex:7];
            [encoder dispatchThreadgroups:
                MTLSizeMake((hidden_dim + 255u) / 256u, batch_tokens, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted ||
                !finite_f32_array(combined_rows, shared_values)) {
                dispatch_status = NATIVE_BATCH_INVALID_STATE;
            }
        }
    }
    if (dispatch_status != NATIVE_BATCH_OK) {
        workspace->used_bytes = used_before;
        return dispatch_status;
    }
    *out_combined_outputs = combined_rows;
    return NATIVE_BATCH_OK;
}

static int64_t expert_resolver_system_pread(
    void *context,
    int fd,
    void *buffer,
    uint64_t size_bytes,
    uint64_t offset_bytes) {
    (void)context;
    if (size_bytes > SIZE_MAX || offset_bytes > INT64_MAX) return -1;
    return (int64_t)pread(
        fd, buffer, (size_t)size_bytes, (off_t)offset_bytes);
}

static void expert_resolution_record_metrics(
    MoEInferEngine *eng,
    const NativeExpertResolutionMetrics *metrics,
    const NativeExpertBucketWorkspace *buckets,
    const NativeResolvedExpert *entries) {
    NativeLayerMetrics *layer = telemetry_current_layer(eng, (int)metrics->layer);
    if (!layer) return;

    // Resolver metrics intentionally classify unique experts because physical
    // reads and pins happen once per unique expert. Request telemetry instead
    // classifies every activation pair so the cache classes partition the
    // layer's calls * N_ACTIVE routes. Expand each unique source by its stable
    // bucket count without changing the physical pread counters below.
    for (uint32_t index = 0u; index < buckets->unique_count; ++index) {
        const NativeResolvedExpert *entry = &entries[index];
        const uint64_t route_count = buckets->counts[entry->expert_id];
        switch (entry->source) {
            case NATIVE_EXPERT_RESOLUTION_HOT:
                layer->hot_hits += route_count;
                break;
            case NATIVE_EXPERT_RESOLUTION_BATCH_POOL:
                layer->batch_pool_hits += route_count;
                break;
            case NATIVE_EXPERT_RESOLUTION_COLD:
                layer->cold_misses += route_count;
                break;
            default:
                break;
        }
    }
    layer->pread_calls += metrics->pread_calls;
    layer->bytes_requested += metrics->bytes_requested;
    layer->bytes_read += metrics->bytes_read;
    layer->short_reads += metrics->short_reads;
    layer->io_wait_ns += metrics->io_wait_ns;
    layer->bytes_avoided += metrics->bytes_avoided;
}

int moe_infer_unique_expert_resolve(
    MoEInferEngine *eng,
    uint32_t layer,
    const NativeExpertBucketWorkspace *buckets,
    NativeUniqueExpertResolution *resolution,
    NativeExpertPreadFn reader,
    void *reader_context) {
    if (!eng || !resolution ||
        resolution->self_cookie != (uintptr_t)resolution ||
        resolution->owner_engine != 0u || resolution->count != 0u ||
        resolution->active != 0u || layer >= N_LAYERS ||
        !expert_bucket_index_is_complete(buckets)) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if (!eng->initialized || eng->active_expert_size == 0u ||
        !eng->batch_pool || !eng->batch_wrap || !eng->batch_eids ||
        !eng->batch_pins || eng->batch_capacity < N_ACTIVE ||
        eng->batch_slot_size != eng->active_expert_size ||
        eng->batch_count < 0 || eng->batch_count > eng->batch_capacity ||
        eng->unique_resolver_owner != 0u) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    uint8_t occupied[N_EXPERTS] = {0};
    for (int slot = 0; slot < eng->batch_count; ++slot) {
        const int expert = eng->batch_eids[slot];
        if (expert < 0 || expert >= N_EXPERTS || occupied[expert] ||
            eng->batch_pins[slot] != 0u) {
            return NATIVE_BATCH_INVALID_STATE;
        }
        occupied[expert] = 1u;
    }
    for (int slot = eng->batch_count; slot < eng->batch_capacity; ++slot) {
        if (eng->batch_pins[slot] != 0u) return NATIVE_BATCH_INVALID_STATE;
    }

    NativeResolvedExpert staged[N_EXPERTS] = {0};
    NativeExpertResolutionMetrics metrics = {0};
    metrics.layer = layer;
    metrics.activation_pairs = buckets->pair_capacity;
    metrics.unique_experts = buckets->unique_count;
    metrics.error_expert = UINT32_MAX;
    const int original_batch_count = eng->batch_count;
    uint32_t cold_count = 0u;

    // Classify every unique expert and reserve all required slots before the
    // first read. This makes slot pressure a pre-I/O, non-mutating failure.
    for (uint32_t index = 0u; index < buckets->unique_count; ++index) {
        const uint32_t expert = buckets->unique_expert_ids[index];
        NativeResolvedExpert *entry = &staged[index];
        entry->expert_id = expert;
        entry->slot_index = UINT32_MAX;
        uint8_t *hot = eng->expert_mem_cache[layer]
            ? eng->expert_mem_cache[layer][expert]
            : NULL;
        if (hot) {
            entry->source = NATIVE_EXPERT_RESOLUTION_HOT;
            entry->data = hot;
            ++metrics.hot_hits;
            continue;
        }
        int existing_slot = -1;
        for (int slot = 0; slot < original_batch_count; ++slot) {
            if (eng->batch_eids[slot] == (int)expert) {
                existing_slot = slot;
                break;
            }
        }
        if (existing_slot >= 0) {
            entry->source = NATIVE_EXPERT_RESOLUTION_BATCH_POOL;
            entry->slot_index = (uint32_t)existing_slot;
            entry->data = eng->batch_pool +
                (size_t)existing_slot * eng->batch_slot_size;
            ++metrics.batch_pool_hits;
            continue;
        }
        if ((uint32_t)original_batch_count + cold_count >=
            (uint32_t)eng->batch_capacity) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }
        const uint32_t slot = (uint32_t)original_batch_count + cold_count++;
        entry->source = NATIVE_EXPERT_RESOLUTION_COLD;
        entry->slot_index = slot;
        entry->data = eng->batch_pool + (size_t)slot * eng->batch_slot_size;
    }

    if (!reader) reader = expert_resolver_system_pread;
    for (uint32_t index = 0u; index < buckets->unique_count; ++index) {
        NativeResolvedExpert *entry = &staged[index];
        if (entry->source != NATIVE_EXPERT_RESOLUTION_COLD) continue;
        if (entry->expert_id > UINT64_MAX / eng->active_expert_size) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }
        const uint64_t offset =
            (uint64_t)entry->expert_id * eng->active_expert_size;
        const uint64_t started = moe_infer_monotonic_ns();
        const int64_t bytes = reader(
            reader_context,
            eng->packed_fd[layer],
            entry->data,
            eng->active_expert_size,
            offset);
        const uint64_t finished = moe_infer_monotonic_ns();
        ++metrics.cold_reads;
        ++metrics.pread_calls;
        metrics.bytes_requested += eng->active_expert_size;
        if (bytes > 0) metrics.bytes_read += (uint64_t)bytes;
        metrics.io_wait_ns += finished - started;
        if (bytes != (int64_t)eng->active_expert_size) {
            ++metrics.short_reads;
            metrics.error_expert = entry->expert_id;
            metrics.bytes_avoided =
                ((uint64_t)buckets->pair_capacity - metrics.cold_reads) *
                eng->active_expert_size;
            resolution->metrics = metrics;
            expert_resolution_record_metrics(eng, &metrics, buckets, staged);
            fprintf(stderr,
                "[expert-resolver] short read: layer=%u expert=%u expected=%zu got=%" PRId64 "\n",
                layer,
                entry->expert_id,
                eng->active_expert_size,
                bytes);
            return NATIVE_BATCH_INVALID_STATE;
        }
    }

    for (uint32_t index = 0u; index < buckets->unique_count; ++index) {
        const NativeResolvedExpert *entry = &staged[index];
        if (entry->source == NATIVE_EXPERT_RESOLUTION_COLD) {
            eng->batch_eids[entry->slot_index] = (int)entry->expert_id;
        }
        if (entry->source != NATIVE_EXPERT_RESOLUTION_HOT) {
            eng->batch_pins[entry->slot_index] = 1u;
        }
    }
    eng->batch_count = original_batch_count + (int)cold_count;
    metrics.bytes_avoided =
        ((uint64_t)buckets->pair_capacity - metrics.cold_reads) *
        eng->active_expert_size;
    memcpy(resolution->experts, staged,
        (size_t)buckets->unique_count * sizeof(NativeResolvedExpert));
    resolution->metrics = metrics;
    resolution->owner_engine = (uintptr_t)eng;
    resolution->count = buckets->unique_count;
    resolution->active = 1u;
    eng->unique_resolver_owner = (uintptr_t)resolution;
    expert_resolution_record_metrics(eng, &metrics, buckets, staged);
    return NATIVE_BATCH_OK;
}

int moe_infer_unique_expert_release(
    MoEInferEngine *eng,
    NativeUniqueExpertResolution *resolution) {
    if (!eng || !resolution ||
        resolution->self_cookie != (uintptr_t)resolution ||
        resolution->active != 1u ||
        resolution->owner_engine != (uintptr_t)eng ||
        eng->unique_resolver_owner != (uintptr_t)resolution ||
        !eng->batch_pins) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    for (uint32_t index = 0u; index < resolution->count; ++index) {
        const NativeResolvedExpert *entry = &resolution->experts[index];
        if (entry->source != NATIVE_EXPERT_RESOLUTION_HOT) {
            if (entry->slot_index >= (uint32_t)eng->batch_capacity ||
                eng->batch_pins[entry->slot_index] != 1u) {
                return NATIVE_BATCH_INVALID_STATE;
            }
        }
    }
    for (uint32_t index = 0u; index < resolution->count; ++index) {
        const NativeResolvedExpert *entry = &resolution->experts[index];
        if (entry->source != NATIVE_EXPERT_RESOLUTION_HOT) {
            eng->batch_pins[entry->slot_index] = 0u;
        }
    }
    eng->unique_resolver_owner = 0u;
    moe_infer_unique_expert_resolution_init(resolution);
    return NATIVE_BATCH_OK;
}

static int expert_bucket_route_validate(
    const float *logits,
    const float *bias,
    const uint8_t *cached,
    float smelt_penalty,
    uint32_t has_bias,
    uint32_t has_smelt,
    NativeExpertBucketWorkspace *view) {
    NativeExpertBucketWorkspacePlan plan;
    if (!logits || !expert_bucket_workspace_is_owned(view, &plan) ||
        view->batch_tokens > MAX_SEQ_LEN || has_bias > 1u || has_smelt > 1u ||
        (has_bias && !bias) || (has_smelt && !cached) ||
        !isfinite(smelt_penalty)) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    const size_t score_count = (size_t)view->batch_tokens * N_EXPERTS;
    for (size_t index = 0; index < score_count; ++index) {
        if (!isfinite(logits[index])) return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if (has_bias) {
        for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
            if (!isfinite(bias[expert])) return NATIVE_BATCH_INVALID_ARGUMENT;
        }
    }
    if (has_smelt) {
        for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
            if (cached[expert] > 1u) return NATIVE_BATCH_INVALID_ARGUMENT;
        }
    }
    return NATIVE_BATCH_OK;
}

static bool expert_bucket_routes_are_valid(
    const int32_t *selected,
    const float *weights,
    uint32_t batch_tokens) {
    for (uint32_t token = 0u; token < batch_tokens; ++token) {
        uint8_t seen[N_EXPERTS] = {0};
        for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
            const size_t index = (size_t)token * N_ACTIVE + rank;
            const int32_t expert = selected[index];
            if (expert < 0 || expert >= N_EXPERTS || seen[expert] ||
                !isfinite(weights[index])) {
                return false;
            }
            seen[expert] = 1u;
        }
    }
    return true;
}

static void expert_bucket_publish_routes(
    const int32_t *selected,
    const float *weights,
    NativeExpertBucketWorkspace *view) {
    for (uint32_t token = 0u; token < view->batch_tokens; ++token) {
        for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
            const size_t index = (size_t)token * N_ACTIVE + rank;
            view->activation_pairs[index] = (NativeActivationPair){
                .token_id = token,
                .expert_id = (uint32_t)selected[index],
                .route_rank = rank,
                .weight = weights[index],
            };
        }
    }
}

int moe_infer_expert_bucket_route_batch_reference(
    const float *logits,
    const float *bias,
    const uint8_t *cached,
    float smelt_penalty,
    uint32_t has_bias,
    uint32_t has_smelt,
    NativeExpertBucketWorkspace *view) {
    const int status = expert_bucket_route_validate(
        logits, bias, cached, smelt_penalty, has_bias, has_smelt, view);
    if (status != NATIVE_BATCH_OK) return status;

    // Stage the complete batch in the workspace's secondary pair region. A
    // derived non-finite value (for example finite logits+bias overflowing)
    // must not partially publish earlier tokens to activation_pairs.
    for (uint32_t token = 0u; token < view->batch_tokens; ++token) {
        int ids[N_ACTIVE];
        int32_t selected[N_ACTIVE];
        float weights[N_ACTIVE];
        cpu_moe_route_flags(
            logits + (size_t)token * N_EXPERTS,
            has_bias ? bias : NULL,
            N_EXPERTS,
            N_ACTIVE,
            ids,
            weights,
            has_smelt ? cached : NULL,
            smelt_penalty);
        for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
            selected[rank] = ids[rank];
        }
        if (!expert_bucket_routes_are_valid(selected, weights, 1u)) {
            return NATIVE_BATCH_INVALID_STATE;
        }
        for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
            const size_t index = (size_t)token * N_ACTIVE + rank;
            view->bucketed_pairs[index] = (NativeActivationPair){
                .token_id = token,
                .expert_id = (uint32_t)selected[rank],
                .route_rank = rank,
                .weight = weights[rank],
            };
        }
    }
    memcpy(view->activation_pairs, view->bucketed_pairs,
        (size_t)view->pair_capacity * sizeof(*view->activation_pairs));
    return NATIVE_BATCH_OK;
}

int moe_infer_expert_bucket_route_batch(
    MoEInferEngine *eng,
    const float *logits,
    const float *bias,
    const uint8_t *cached,
    float smelt_penalty,
    uint32_t has_bias,
    uint32_t has_smelt,
    NativeExpertBucketWorkspace *view) {
    const int status = expert_bucket_route_validate(
        logits, bias, cached, smelt_penalty, has_bias, has_smelt, view);
    if (status != NATIVE_BATCH_OK) return status;
    if (!eng || !eng->initialized || !eng->pipe_moe_route_gpu ||
        !eng->buf_batch_routing_scores || !eng->buf_batch_route_bias ||
        !eng->buf_cached_flags || !eng->buf_gpu_route_selected ||
        !eng->buf_gpu_route_weights) {
        return NATIVE_BATCH_INVALID_STATE;
    }

    const size_t score_bytes = (size_t)view->batch_tokens *
        N_EXPERTS * sizeof(float);
    memcpy([(id<MTLBuffer>)eng->buf_batch_routing_scores contents],
        logits, score_bytes);
    if (has_bias) {
        memcpy([(id<MTLBuffer>)eng->buf_batch_route_bias contents],
            bias, N_EXPERTS * sizeof(float));
    }
    if (has_smelt) {
        memcpy([(id<MTLBuffer>)eng->buf_cached_flags contents],
            cached, N_EXPERTS * sizeof(uint8_t));
    }

    id<MTLCommandBuffer> command =
        [(id<MTLCommandQueue>)eng->queue commandBuffer];
    if (!command) return NATIVE_BATCH_INVALID_STATE;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) return NATIVE_BATCH_INVALID_STATE;
    [encoder setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_moe_route_gpu];
    [encoder setBuffer:(id<MTLBuffer>)eng->buf_batch_routing_scores offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)eng->buf_batch_route_bias offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)eng->buf_cached_flags offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)eng->buf_gpu_route_selected offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)eng->buf_gpu_route_weights offset:0 atIndex:4];
    [encoder setBytes:&smelt_penalty length:sizeof(float) atIndex:5];
    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&has_smelt length:sizeof(uint32_t) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(view->batch_tokens, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(N_EXPERTS, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if ([command status] == MTLCommandBufferStatusError) {
        return NATIVE_BATCH_INVALID_STATE;
    }

    const int32_t *selected = (const int32_t *)
        [(id<MTLBuffer>)eng->buf_gpu_route_selected contents];
    const float *weights = (const float *)
        [(id<MTLBuffer>)eng->buf_gpu_route_weights contents];
    if (!expert_bucket_routes_are_valid(
            selected, weights, view->batch_tokens)) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    expert_bucket_publish_routes(selected, weights, view);
    return NATIVE_BATCH_OK;
}

static int batch_workspace_pointer_offset(
    const NativeBatchWorkspace *workspace,
    const void *pointer,
    uint64_t size_bytes,
    uint64_t *out_metal_offset) {
    if (!batch_workspace_is_owned(workspace) || !pointer ||
        !out_metal_offset) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    const uintptr_t base = (uintptr_t)workspace->storage;
    const uintptr_t address = (uintptr_t)pointer;
    if (address < base) return NATIVE_BATCH_INVALID_ARGUMENT;
    const uint64_t offset = (uint64_t)(address - base);
    if (offset > workspace->allocated_bytes ||
        size_bytes > workspace->allocated_bytes - offset ||
        workspace->metal_buffer_offset > UINT64_MAX - offset) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    const uint64_t metal_offset = workspace->metal_buffer_offset + offset;
    if (metal_offset > (uint64_t)NSUIntegerMax) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    *out_metal_offset = metal_offset;
    return NATIVE_BATCH_OK;
}

static id<MTLBuffer> batch_resolved_expert_blob(
    MoEInferEngine *eng,
    const NativeResolvedExpert *expert,
    bool *out_owned) {
    if (!eng || !expert || !expert->data || !out_owned ||
        expert->expert_id >= N_EXPERTS) {
        return nil;
    }
    *out_owned = false;
    if (expert->source != NATIVE_EXPERT_RESOLUTION_HOT) {
        if (expert->slot_index >= (uint32_t)eng->batch_capacity ||
            !eng->batch_wrap) {
            return nil;
        }
        void **slot = &eng->batch_wrap[expert->slot_index];
        if (!*slot) {
            *slot = (void *)[(id<MTLDevice>)eng->device
                newBufferWithBytesNoCopy:expert->data
                length:eng->active_expert_size
                options:MTLResourceStorageModeShared
                deallocator:nil];
        }
        return (id<MTLBuffer>)*slot;
    }
    id<MTLBuffer> blob = [(id<MTLDevice>)eng->device
        newBufferWithBytesNoCopy:expert->data
        length:eng->active_expert_size
        options:MTLResourceStorageModeShared
        deallocator:nil];
    *out_owned = blob != nil;
    return blob;
}

static const NativeResolvedExpert *batch_find_resolved_expert(
    const NativeUniqueExpertResolution *resolution,
    uint32_t expert_id);

static int batch_dispatch_resolved_expert_groups(
    MoEInferEngine *eng,
    int layer,
    const NativeUniqueExpertResolution *resolution,
    const NativeExpertBucketWorkspace *buckets,
    const float *bucket_inputs,
    float *bucket_mid,
    float *bucket_outputs,
    NativeBatchWorkspace *workspace) {
    if (!eng || !resolution || !buckets || !bucket_inputs || !bucket_mid ||
        !bucket_outputs || !workspace || buckets->unique_count == 0u ||
        buckets->pair_capacity == 0u || eng->use_affine_experts ||
        eng->expert_format_2b != 0 ||
        eng->active_expert_size != EXPERT_SIZE ||
        !eng->pipe_bucket6_mxfp4_gate_up ||
        !eng->pipe_bucket6_mxfp4_down) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    const uint64_t pair_count = buckets->pair_capacity;
    const uint64_t input_bytes = pair_count * DIM * sizeof(float);
    const uint64_t mid_bytes = pair_count * INTERMEDIATE * sizeof(float);
    const uint64_t output_bytes = input_bytes;
    uint64_t input_offset = 0u, mid_offset = 0u, output_offset = 0u;
    int status = batch_workspace_pointer_offset(
        workspace, bucket_inputs, input_bytes, &input_offset);
    if (status == NATIVE_BATCH_OK) {
        status = batch_workspace_pointer_offset(
            workspace, bucket_mid, mid_bytes, &mid_offset);
    }
    if (status == NATIVE_BATCH_OK) {
        status = batch_workspace_pointer_offset(
            workspace, bucket_outputs, output_bytes, &output_offset);
    }
    if (status != NATIVE_BATCH_OK) return status;

    id<MTLBuffer> arena = bucket_workspace_metal_buffer(
        (id<MTLDevice>)eng->device, workspace);
    if (!arena) return NATIVE_BATCH_INVALID_STATE;

    id<MTLBuffer> owned_blobs[N_EXPERTS] = {nil};
    uint32_t owned_count = 0u;
    @autoreleasepool {
        id<MTLCommandBuffer> command =
            [(id<MTLCommandQueue>)eng->queue commandBuffer];
        if (!command) {
            status = NATIVE_BATCH_INVALID_STATE;
        }
        for (uint32_t group_begin = 0u;
             status == NATIVE_BATCH_OK &&
                 group_begin < buckets->unique_count;
             group_begin += N_ACTIVE) {
            id<MTLBuffer> blobs[N_ACTIVE] = {nil};
            uint32_t group_counts[N_ACTIVE] = {0u};
            uint32_t group_rows = 0u;
            const uint32_t group_end =
                group_begin + N_ACTIVE < buckets->unique_count
                    ? group_begin + N_ACTIVE
                    : buckets->unique_count;
            const uint32_t first_expert =
                buckets->unique_expert_ids[group_begin];
            const uint32_t group_base = buckets->offsets[first_expert];
            for (uint32_t unique = group_begin;
                 unique < group_end;
                 ++unique) {
                const uint32_t slot = unique - group_begin;
                const uint32_t expert_id =
                    buckets->unique_expert_ids[unique];
                const NativeResolvedExpert *resolved =
                    batch_find_resolved_expert(resolution, expert_id);
                if (!resolved || buckets->counts[expert_id] == 0u ||
                    group_rows > UINT32_MAX - buckets->counts[expert_id]) {
                    status = NATIVE_BATCH_INVALID_STATE;
                    break;
                }
                bool owns_blob = false;
                blobs[slot] = batch_resolved_expert_blob(
                    eng, resolved, &owns_blob);
                if (!blobs[slot]) {
                    status = NATIVE_BATCH_INVALID_STATE;
                    break;
                }
                if (owns_blob) {
                    if (owned_count >= N_EXPERTS) {
                        [blobs[slot] release];
                        status = NATIVE_BATCH_CAPACITY_EXCEEDED;
                        break;
                    }
                    owned_blobs[owned_count++] = blobs[slot];
                }
                group_counts[slot] = buckets->counts[expert_id];
                group_rows += group_counts[slot];
            }
            if (status != NATIVE_BATCH_OK) break;
            if (group_rows == 0u ||
                group_base > buckets->pair_capacity ||
                group_rows > buckets->pair_capacity - group_base) {
                status = NATIVE_BATCH_INVALID_STATE;
                break;
            }
            for (uint32_t slot = group_end - group_begin;
                 slot < N_ACTIVE;
                 ++slot) {
                blobs[slot] = blobs[0];
            }

            id<MTLComputeCommandEncoder> gate =
                [command computeCommandEncoder];
            if (!gate) {
                status = NATIVE_BATCH_INVALID_STATE;
                break;
            }
            [gate setComputePipelineState:
                (id<MTLComputePipelineState>)eng->pipe_bucket6_mxfp4_gate_up];
            for (uint32_t slot = 0u; slot < N_ACTIVE; ++slot) {
                [gate setBuffer:blobs[slot] offset:0 atIndex:slot];
            }
            [gate setBuffer:arena offset:(NSUInteger)input_offset atIndex:6];
            [gate setBuffer:arena offset:(NSUInteger)mid_offset atIndex:7];
            [gate setBytes:&group_base length:sizeof(group_base) atIndex:8];
            [gate setBytes:group_counts length:sizeof(group_counts) atIndex:9];
            [gate dispatchThreadgroups:
                MTLSizeMake(INTERMEDIATE / 8u, group_rows, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            telemetry_record_moe_dispatch(
                eng, layer,
                (uint64_t)(INTERMEDIATE / 8u) * group_rows,
                256u,
                (uint64_t)(INTERMEDIATE / 8u) * group_rows * 256u);
            [gate endEncoding];

            id<MTLComputeCommandEncoder> down =
                [command computeCommandEncoder];
            if (!down) {
                status = NATIVE_BATCH_INVALID_STATE;
                break;
            }
            [down setComputePipelineState:
                (id<MTLComputePipelineState>)eng->pipe_bucket6_mxfp4_down];
            for (uint32_t slot = 0u; slot < N_ACTIVE; ++slot) {
                [down setBuffer:blobs[slot] offset:0 atIndex:slot];
            }
            [down setBuffer:arena offset:(NSUInteger)mid_offset atIndex:6];
            [down setBuffer:arena offset:(NSUInteger)output_offset atIndex:7];
            [down setBytes:&group_base length:sizeof(group_base) atIndex:8];
            [down setBytes:group_counts length:sizeof(group_counts) atIndex:9];
            [down dispatchThreadgroups:
                MTLSizeMake(DIM / 8u, group_rows, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            telemetry_record_moe_dispatch(
                eng, layer,
                (uint64_t)(DIM / 8u) * group_rows,
                256u,
                (uint64_t)(DIM / 8u) * group_rows * 256u);
            [down endEncoding];
        }
        if (status == NATIVE_BATCH_OK) {
            NativeLayerMetrics *metrics = telemetry_current_layer(eng, layer);
            if (metrics) metrics->moe_command_buffers++;
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted ||
                !finite_f32_array(bucket_outputs, pair_count * DIM)) {
                status = NATIVE_BATCH_INVALID_STATE;
            }
        }
    }
    for (uint32_t index = 0u; index < owned_count; ++index) {
        [owned_blobs[index] release];
    }
    return status;
}

static int batch_route_hash_layer(
    MoEInferEngine *eng,
    int layer,
    NativePreparedMoEToken *prepared,
    uint32_t batch_tokens,
    NativeExpertBucketWorkspace *buckets) {
    if (!eng || !prepared || !buckets || !eng->tid2eid[layer]) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    for (uint32_t token = 0u; token < batch_tokens; ++token) {
        const float *scores = prepared[token].routing_scores;
        int expert_ids[N_ACTIVE];
        float weights[N_ACTIVE];
        bool valid = prepared[token].token_id >= 0 &&
            prepared[token].token_id < eng->vocab_size;
        const int64_t *row = valid
            ? eng->tid2eid[layer] +
                (size_t)prepared[token].token_id * N_ACTIVE
            : NULL;
        for (int rank = 0; rank < N_ACTIVE && valid; ++rank) {
            expert_ids[rank] = (int)row[rank];
            valid = expert_ids[rank] >= 0 &&
                expert_ids[rank] < N_EXPERTS;
        }
        if (!valid) {
            cpu_moe_route(
                scores, eng->gate_bias[layer], N_EXPERTS, N_ACTIVE,
                expert_ids, weights, NULL, 0.0f);
        } else {
            float sum = 0.0f;
            for (int rank = 0; rank < N_ACTIVE; ++rank) {
                const float value = scores[expert_ids[rank]];
                const float softplus = value > 0.0f
                    ? value + log1pf(expf(-value))
                    : log1pf(expf(value));
                weights[rank] = sqrtf(softplus);
                sum += weights[rank];
            }
            sum += 1e-20f;
            for (int rank = 0; rank < N_ACTIVE; ++rank) {
                weights[rank] = weights[rank] / sum * 1.5f;
            }
        }
        for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
            buckets->activation_pairs[token * N_ACTIVE + rank] =
                (NativeActivationPair){
                    .token_id = token,
                    .expert_id = (uint32_t)expert_ids[rank],
                    .route_rank = rank,
                    .weight = weights[rank],
                };
        }
    }
    return NATIVE_BATCH_OK;
}

static const NativeResolvedExpert *batch_find_resolved_expert(
    const NativeUniqueExpertResolution *resolution,
    uint32_t expert_id) {
    if (!resolution) return NULL;
    for (uint32_t index = 0u; index < resolution->count; ++index) {
        if (resolution->experts[index].expert_id == expert_id) {
            return &resolution->experts[index];
        }
    }
    return NULL;
}

static void batch_record_routing_trace(
    MoEInferEngine *eng,
    const NativeExpertBucketWorkspace *buckets,
    const NativeUniqueExpertResolution *resolution,
    int start_pos) {
    if (!eng || !eng->routing_trace || !buckets || !resolution) return;
    uint8_t cold_recorded[N_EXPERTS] = {0};
    bool wait_recorded = false;
    for (uint32_t token = 0u; token < buckets->batch_tokens; ++token) {
        RoutingTraceRecord record = {0};
        record.phase = (uint32_t)eng->routing_trace_phase;
        record.token_position = start_pos >= 0
            ? (uint32_t)start_pos + token
            : token;
        record.layer = resolution->metrics.layer;
        for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
            const NativeActivationPair *pair =
                &buckets->activation_pairs[token * N_ACTIVE + rank];
            const NativeResolvedExpert *resolved =
                batch_find_resolved_expert(resolution, pair->expert_id);
            record.expert_ids[rank] = (int32_t)pair->expert_id;
            record.weights[rank] = pair->weight;
            if (!resolved) continue;
            if (resolved->source == NATIVE_EXPERT_RESOLUTION_HOT) {
                record.cache_states[rank] = ROUTING_TRACE_CACHE_HOT;
            } else if (resolved->source ==
                       NATIVE_EXPERT_RESOLUTION_BATCH_POOL) {
                record.cache_states[rank] = ROUTING_TRACE_CACHE_BATCH_POOL;
            } else {
                record.cache_states[rank] = ROUTING_TRACE_CACHE_COLD;
                if (!cold_recorded[pair->expert_id]) {
                    cold_recorded[pair->expert_id] = 1u;
                    record.route_bytes[rank] = eng->active_expert_size;
                    record.pread_bytes += eng->active_expert_size;
                }
            }
        }
        if (!wait_recorded && record.pread_bytes > 0u) {
            record.io_wait_ns = resolution->metrics.io_wait_ns;
            wait_recorded = true;
        }
        (void)routing_trace_append(eng->routing_trace, &record);
    }
}

static int moe_infer_mhc_pre_norm_row(
    MoEInferEngine *eng,
    int layer,
    bool attention,
    const float *residual,
    uint16_t *normed_bf16,
    float *post,
    float *comb) {
    if (!eng || layer < 0 || layer >= N_LAYERS || !residual ||
        !normed_bf16 || !post || !comb ||
        !eng->pipe_mhc_pre_split_weighted_sum_norm ||
        !eng->buf_residual_gpu || !eng->buf_mhc_post_weights ||
        !eng->buf_mhc_comb_weights) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    void *fn = attention ? eng->buf_attn_hc_fn[layer] : eng->buf_ffn_hc_fn[layer];
    void *base = attention ? eng->buf_attn_hc_base[layer] : eng->buf_ffn_hc_base[layer];
    void *scale = attention ? eng->buf_attn_hc_scale[layer] : eng->buf_ffn_hc_scale[layer];
    void *input = attention ? eng->buf_mhc_attn_in : eng->buf_mhc_ffn_in;
    void *norm_weight = attention ? eng->buf_input_norm_gpu[layer] : eng->buf_attn_norm_gpu[layer];
    void *norm_output = attention ? eng->buf_mhc_attn_norm_bf16 : eng->buf_mhc_ffn_norm_bf16;
    if (!fn || !base || !scale || !input || !norm_weight || !norm_output) {
        return NATIVE_BATCH_INVALID_STATE;
    }

    memcpy([(id<MTLBuffer>)eng->buf_residual_gpu contents], residual,
           (size_t)MHC_MULT * DIM * sizeof(float));
    @autoreleasepool {
        id<MTLCommandBuffer> command =
            [(id<MTLCommandQueue>)eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = command
            ? [command computeCommandEncoder] : nil;
        if (!command || !encoder) return NATIVE_BATCH_INVALID_STATE;
        [encoder setComputePipelineState:
            (id<MTLComputePipelineState>)eng->pipe_mhc_pre_split_weighted_sum_norm];
        [encoder setBuffer:(id<MTLBuffer>)fn offset:0 atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)base offset:0 atIndex:1];
        [encoder setBuffer:(id<MTLBuffer>)scale offset:0 atIndex:2];
        [encoder setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu offset:0 atIndex:3];
        [encoder setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_weights offset:0 atIndex:4];
        [encoder setBuffer:(id<MTLBuffer>)eng->buf_mhc_comb_weights offset:0 atIndex:5];
        [encoder setBuffer:(id<MTLBuffer>)input offset:0 atIndex:6];
        [encoder setBuffer:(id<MTLBuffer>)norm_weight offset:0 atIndex:7];
        [encoder setBuffer:(id<MTLBuffer>)norm_output offset:0 atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];

        if (!attention && eng->gate_proj[layer]) {
            id<MTLComputeCommandEncoder> route = [command computeCommandEncoder];
            if (!route || !eng->buf_gate_proj_gpu[layer] ||
                !eng->buf_routing_scores || !eng->pipe_matvec_f32_bf16in) {
                return NATIVE_BATCH_INVALID_STATE;
            }
            if (eng->pipe_matvec_f32_bf16in_simd) {
                [route setComputePipelineState:
                    (id<MTLComputePipelineState>)eng->pipe_matvec_f32_bf16in_simd];
                [route setBuffer:(id<MTLBuffer>)eng->buf_gate_proj_gpu[layer] offset:0 atIndex:0];
                [route setBuffer:(id<MTLBuffer>)norm_output offset:0 atIndex:1];
                [route setBuffer:(id<MTLBuffer>)eng->buf_routing_scores offset:0 atIndex:2];
                uint out_dim = N_EXPERTS, in_dim = DIM;
                [route setBytes:&out_dim length:sizeof(out_dim) atIndex:3];
                [route setBytes:&in_dim length:sizeof(in_dim) atIndex:4];
                [route dispatchThreadgroups:MTLSizeMake((N_EXPERTS + 7) / 8, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [route setComputePipelineState:
                    (id<MTLComputePipelineState>)eng->pipe_matvec_f32_bf16in];
                [route setBuffer:(id<MTLBuffer>)eng->buf_gate_proj_gpu[layer] offset:0 atIndex:0];
                [route setBuffer:(id<MTLBuffer>)norm_output offset:0 atIndex:1];
                [route setBuffer:(id<MTLBuffer>)eng->buf_routing_scores offset:0 atIndex:2];
                uint out_dim = N_EXPERTS, in_dim = DIM;
                [route setBytes:&out_dim length:sizeof(out_dim) atIndex:3];
                [route setBytes:&in_dim length:sizeof(in_dim) atIndex:4];
                [route dispatchThreads:MTLSizeMake(N_EXPERTS, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [route endEncoding];
        }
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return NATIVE_BATCH_INVALID_STATE;
        }
    }
    memcpy(normed_bf16, [(id<MTLBuffer>)norm_output contents],
           DIM * sizeof(uint16_t));
    memcpy(post, [(id<MTLBuffer>)eng->buf_mhc_post_weights contents],
           MHC_MULT * sizeof(float));
    memcpy(comb, [(id<MTLBuffer>)eng->buf_mhc_comb_weights contents],
           MHC_MULT * MHC_MULT * sizeof(float));
    return NATIVE_BATCH_OK;
}

static int moe_infer_prepare_ffn_after_batched_attention(
    MoEInferEngine *eng,
    int layer,
    float *hidden,
    const MoeInferStateSnapshot *state_snapshot,
    NativePreparedMoEToken *prepared) {
    if (!eng || !hidden || !prepared || !prepared->normed ||
        !prepared->routing_scores || !prepared->residual ||
        !prepared->post_ffn || !prepared->comb_ffn) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    uint16_t normed_bf16[DIM];
    int status = moe_infer_mhc_pre_norm_row(
        eng, layer, false, hidden, normed_bf16,
        prepared->post_ffn, prepared->comb_ffn);
    if (status != NATIVE_BATCH_OK) return status;

    for (uint32_t dim = 0u; dim < DIM; ++dim) {
        uint32_t bits = (uint32_t)normed_bf16[dim] << 16;
        memcpy(&prepared->normed[dim], &bits, sizeof(bits));
    }
    float *scores = (float *)[(id<MTLBuffer>)eng->buf_routing_scores contents];
    if (!eng->gate_proj[layer]) {
        for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
            prepared->routing_scores[expert] = (float)(N_EXPERTS - expert);
        }
    } else {
        const bool scalar_uses_raw_gpu_scores = !eng->tid2eid[layer] &&
            eng->smelt_warmup_done && eng->pipe_moe_route_gpu;
        memcpy(prepared->routing_scores, scores,
               N_EXPERTS * sizeof(float));
        if (!scalar_uses_raw_gpu_scores) {
            for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
                uint32_t bits;
                memcpy(&bits, &prepared->routing_scores[expert], sizeof(bits));
                bits &= UINT32_C(0xFFFF0000);
                memcpy(&prepared->routing_scores[expert], &bits, sizeof(bits));
            }
        }
    }
    memcpy(prepared->residual, hidden,
           (size_t)MHC_MULT * DIM * sizeof(float));
    if (state_snapshot) {
        prepared->kv_length = state_snapshot->kv_length;
        prepared->compressor_length = state_snapshot->compressor_length;
        prepared->indexer_length = state_snapshot->indexer_length;
        prepared->kv_state_digest = state_snapshot->kv_state_digest;
        prepared->compressor_state_digest =
            state_snapshot->compressor_state_digest;
        prepared->indexer_state_digest = state_snapshot->indexer_state_digest;
    }
    return NATIVE_BATCH_OK;
}

static int moe_infer_forward_batch_layer_batched_moe(
    MoEInferEngine *eng,
    int layer,
    float *hidden_batch,
    int n_tokens,
    int start_pos,
    const int *token_ids,
    const MoeInferStateSnapshot *state_snapshots,
    bool attention_already_batched) {
    if (!eng || !hidden_batch || n_tokens <= 0 ||
        n_tokens > (int)eng->batch_max_prefill_chunk ||
        (attention_already_batched && eng->debug_capture && !state_snapshots) ||
        !eng->forward_batch_workspace) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    NativeBatchWorkspace *workspace =
        (NativeBatchWorkspace *)eng->forward_batch_workspace;
    moe_infer_batch_workspace_reset(workspace);
    NativeExpertBucketWorkspace buckets = {0};
    int status = moe_infer_expert_bucket_workspace_bind(
        workspace, (uint32_t)n_tokens, &buckets);
    if (status != NATIVE_BATCH_OK) return status;

    const uint64_t batch_tokens = (uint64_t)n_tokens;
    float *normed_rows = (float *)moe_infer_batch_workspace_reserve(
        workspace, batch_tokens * DIM * sizeof(float),
        NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    float *routing_scores = (float *)moe_infer_batch_workspace_reserve(
        workspace, batch_tokens * N_EXPERTS * sizeof(float),
        NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    float *residual_rows = (float *)moe_infer_batch_workspace_reserve(
        workspace, batch_tokens * MHC_MULT * DIM * sizeof(float),
        NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    float *post_rows = (float *)moe_infer_batch_workspace_reserve(
        workspace, batch_tokens * MHC_MULT * sizeof(float),
        NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    float *comb_rows = (float *)moe_infer_batch_workspace_reserve(
        workspace, batch_tokens * MHC_MULT * MHC_MULT * sizeof(float),
        NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    float *bucket_outputs = (float *)moe_infer_batch_workspace_reserve(
        workspace, batch_tokens * N_ACTIVE * DIM * sizeof(float),
        NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    float *bucket_inputs = (float *)moe_infer_batch_workspace_reserve(
        workspace, batch_tokens * N_ACTIVE * DIM * sizeof(float),
        NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    float *bucket_mid = (float *)moe_infer_batch_workspace_reserve(
        workspace, batch_tokens * N_ACTIVE * INTERMEDIATE * sizeof(float),
        NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    float *shared_saved = (float *)moe_infer_batch_workspace_reserve(
        workspace, batch_tokens * DIM * sizeof(float),
        NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    if (!normed_rows || !routing_scores || !residual_rows || !post_rows ||
        !comb_rows || !bucket_outputs || !bucket_inputs || !bucket_mid ||
        !shared_saved) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    memset(bucket_outputs, 0,
           (size_t)batch_tokens * N_ACTIVE * DIM * sizeof(float));

    NativePreparedMoEToken *prepared = (NativePreparedMoEToken *)alloca(
        (size_t)n_tokens * sizeof(*prepared));
    NSAutoreleasePool *token_pool = [[NSAutoreleasePool alloc] init];
    for (int token = 0; token < n_tokens; ++token) {
        [token_pool drain];
        token_pool = [[NSAutoreleasePool alloc] init];
        prepared[token] = (NativePreparedMoEToken){
            .normed = normed_rows + (size_t)token * DIM,
            .routing_scores = routing_scores + (size_t)token * N_EXPERTS,
            .residual = residual_rows +
                (size_t)token * MHC_MULT * DIM,
            .post_ffn = post_rows + (size_t)token * MHC_MULT,
            .comb_ffn = comb_rows +
                (size_t)token * MHC_MULT * MHC_MULT,
            .token_id = token_ids ? token_ids[token] : eng->current_token_id,
            .position = start_pos + token,
        };
        eng->current_token_id = prepared[token].token_id;
        float *hidden = hidden_batch +
            (size_t)token * MHC_MULT * DIM;
        int prepare_status;
        if (attention_already_batched) {
            prepare_status = moe_infer_prepare_ffn_after_batched_attention(
                eng, layer, hidden,
                state_snapshots ? &state_snapshots[token] : NULL,
                &prepared[token]);
        } else {
            prepare_status = moe_infer_forward_layer_impl(
                eng, layer, hidden, start_pos + token,
                &prepared[token]) == 0
                ? NATIVE_BATCH_OK : NATIVE_BATCH_INVALID_STATE;
        }
        if (prepare_status != NATIVE_BATCH_OK) {
            [token_pool drain];
            return prepare_status;
        }
    }
    [token_pool drain];

    if (eng->tid2eid[layer]) {
        status = batch_route_hash_layer(
            eng, layer, prepared, (uint32_t)n_tokens, &buckets);
    } else {
        uint8_t cached[N_EXPERTS] = {0};
        const bool has_smelt = eng->smelt_enabled &&
            eng->smelt_warmup_done && eng->expert_mem_cache[layer];
        if (has_smelt) {
            for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
                cached[expert] = eng->expert_mem_cache[layer][expert]
                    ? 1u : 0u;
            }
        }
        const bool scalar_would_use_gpu = eng->smelt_warmup_done &&
            eng->pipe_moe_route_gpu && eng->gate_proj[layer];
        if (scalar_would_use_gpu) {
            status = moe_infer_expert_bucket_route_batch(
                eng, routing_scores, eng->gate_bias[layer], cached,
                eng->smelt_penalty,
                eng->gate_bias[layer] ? 1u : 0u,
                has_smelt ? 1u : 0u,
                &buckets);
        } else {
            status = moe_infer_expert_bucket_route_batch_reference(
                routing_scores, eng->gate_bias[layer], cached,
                eng->smelt_penalty,
                eng->gate_bias[layer] ? 1u : 0u,
                has_smelt ? 1u : 0u,
                &buckets);
        }
    }
    if (status != NATIVE_BATCH_OK) return status;

    for (int token = 0; token < n_tokens; ++token) {
        int ids[N_ACTIVE];
        float weights[N_ACTIVE];
        for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
            const NativeActivationPair *pair =
                &buckets.activation_pairs[token * N_ACTIVE + rank];
            ids[rank] = (int)pair->expert_id;
            weights[rank] = pair->weight;
        }
        telemetry_record_routes(eng, layer, ids, N_ACTIVE);
        emit_prepared_debug_capture(
            eng, &prepared[token], MOE_INFER_CAPTURE_ROUTER,
            MOE_INFER_CAPTURE_DTYPE_F32, layer, weights, N_ACTIVE,
            ids, N_ACTIVE);
        if (eng->smelt_enabled && eng->smelt_in_decode_phase &&
            !eng->tid2eid[layer]) {
            for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
                eng->routing_counts[layer][ids[rank]]++;
            }
        }
        predictor_record(&eng->predictor, layer, ids, N_ACTIVE);
    }
    prefetch_update_adaptive(eng);

    status = moe_infer_expert_bucket_build(&buckets);
    if (status != NATIVE_BATCH_OK) return status;
    NativeUniqueExpertResolution resolution;
    moe_infer_unique_expert_resolution_init(&resolution);
    status = moe_infer_unique_expert_resolve(
        eng, (uint32_t)layer, &buckets, &resolution, NULL, NULL);
    if (status != NATIVE_BATCH_OK) return status;
    batch_record_routing_trace(
        eng, &buckets, &resolution, start_pos);

    const uint64_t persistent_end = workspace->used_bytes;
    if (eng->shared[layer].gate.packed) {
        float *shared_outputs = NULL;
        status = moe_infer_shared_expert_batch_f32(
            eng, layer, normed_rows, (uint32_t)n_tokens,
            workspace, &shared_outputs);
        if (status == NATIVE_BATCH_OK) {
            memcpy(shared_saved, shared_outputs,
                   (size_t)batch_tokens * DIM * sizeof(float));
        }
        workspace->used_bytes = persistent_end;
    } else {
        memset(shared_saved, 0,
               (size_t)batch_tokens * DIM * sizeof(float));
    }
    if (status != NATIVE_BATCH_OK) {
        (void)moe_infer_unique_expert_release(eng, &resolution);
        return status;
    }

    for (uint32_t row = 0u; row < buckets.pair_capacity; ++row) {
        const NativeActivationPair *pair = &buckets.bucketed_pairs[row];
        if (pair->token_id >= (uint32_t)n_tokens) {
            (void)moe_infer_unique_expert_release(eng, &resolution);
            return NATIVE_BATCH_INVALID_STATE;
        }
        memcpy(bucket_inputs + (size_t)row * DIM,
               normed_rows + (size_t)pair->token_id * DIM,
               DIM * sizeof(float));
    }
    status = batch_dispatch_resolved_expert_groups(
        eng, layer, &resolution, &buckets, bucket_inputs, bucket_mid,
        bucket_outputs, workspace);
    if (status != NATIVE_BATCH_OK) {
        (void)moe_infer_unique_expert_release(eng, &resolution);
        return status;
    }
    if (eng->debug_capture) {
        float *routed = (float *)alloca(DIM * sizeof(float));
        for (int token = 0; token < n_tokens; ++token) {
            for (uint32_t column = 0u; column < DIM; ++column) {
                float sum = 0.0f;
                for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
                    const uint32_t source =
                        (uint32_t)token * N_ACTIVE + rank;
                    const uint32_t bucket_row =
                        buckets.inverse_mapping[source];
                    sum += bucket_outputs[(size_t)bucket_row * DIM + column] *
                        buckets.activation_pairs[source].weight;
                }
                routed[column] = sum;
            }
            emit_prepared_debug_capture(
                eng, &prepared[token], MOE_INFER_CAPTURE_MOE_OUTPUT,
                MOE_INFER_CAPTURE_DTYPE_F32, layer, routed, DIM, NULL, 0);
        }
    }
    workspace->used_bytes = persistent_end;
    const int release_status = moe_infer_unique_expert_release(
        eng, &resolution);
    if (status == NATIVE_BATCH_OK) status = release_status;
    if (status != NATIVE_BATCH_OK) return status;

    float *combined = NULL;
    status = moe_infer_bucket_combine_f32(
        eng, &buckets, bucket_outputs, shared_saved, DIM, &combined);
    if (status != NATIVE_BATCH_OK) return status;
    for (int token = 0; token < n_tokens; ++token) {
        const float *ffn_out = combined + (size_t)token * DIM;
        if (moe_infer_finish_prepared_token(
                eng, layer, &prepared[token], ffn_out) != 0) {
            return NATIVE_BATCH_INVALID_STATE;
        }
        memcpy(hidden_batch + (size_t)token * MHC_MULT * DIM,
               prepared[token].residual,
               MHC_MULT * DIM * sizeof(float));
    }
    eng->batch_batched_moe_layers++;
    return NATIVE_BATCH_OK;
}

static int moe_infer_prefill_v2_ensure_kv_storage(MoEInferEngine *eng) {
    if (!eng || !eng->device) return NATIVE_BATCH_INVALID_STATE;
    const size_t bytes = (size_t)MAX_SEQ_LEN * KV_LORA_RANK * sizeof(uint16_t);
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
        KVCache *cache = &eng->kv_cache[layer];
        if (!cache->kv) {
            id<MTLBuffer> buffer = [(id<MTLDevice>)eng->device
                newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!buffer) return NATIVE_BATCH_CAPACITY_EXCEEDED;
            cache->kv_gpu_buf = (void *)buffer;
            cache->kv = (uint16_t *)[buffer contents];
            if (!cache->kv) return NATIVE_BATCH_INVALID_STATE;
            memset(cache->kv, 0, bytes);
            cache->len = 0;
        }
        if (!cache->kv_gpu_buf) return NATIVE_BATCH_INVALID_STATE;
    }
    return NATIVE_BATCH_OK;
}

static bool moe_infer_prefill_v2_resources_ready(
    const MoEInferEngine *eng,
    const MlaPipes *pipes) {
    if (!eng || !pipes || !pipes->dev || !pipes->queue ||
        !pipes->dequant_matvec_affine_bf16out ||
        !pipes->rms_norm_rows_bf16in_bf16out ||
        !pipes->dequant_matvec_affine_bf16in_bf16out_v2 ||
        !pipes->mla_sdpa_batch_f16_kv ||
        !pipes->rope_tail_interleaved_bf16 ||
        !pipes->dequant_matvec_affine_bf16in_f32out_grp8 ||
        !pipes->dequant_matvec_affine_v2 ||
        !pipes->f32_to_bf16_vec || !pipes->mhc_post_bfloat ||
        !pipes->bf16_to_f32_vec ||
        !eng->pipe_mhc_pre_split_weighted_sum_norm ||
        !eng->forward_batch_workspace) {
        return false;
    }
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
        if (!eng->buf_attn_hc_fn[layer] || !eng->buf_attn_hc_base[layer] ||
            !eng->buf_attn_hc_scale[layer] || !eng->buf_ffn_hc_fn[layer] ||
            !eng->buf_ffn_hc_base[layer] || !eng->buf_ffn_hc_scale[layer] ||
            !eng->buf_input_norm_gpu[layer] || !eng->buf_attn_norm_gpu[layer] ||
            !eng->attn[layer].wq_a.packed || !eng->attn[layer].wkv.packed ||
            !eng->attn[layer].wo_a.packed || !eng->attn[layer].wo_b.packed) {
            return false;
        }
    }
    return true;
}

static int moe_infer_forward_batch_v2_internal(
    MoEInferEngine *eng,
    float *hidden_batch,
    int n_tokens,
    int start_pos,
    const int *token_ids,
    NativeBatchPhase phase,
    NativeKVTransaction *verify_transaction) {
    if (moe_infer_exact_q4_recording_enabled(eng))
        return NATIVE_BATCH_INVALID_STATE;
    const bool is_verify = phase == NATIVE_BATCH_PHASE_VERIFY;
    const int phase_limit = is_verify
        ? (eng ? (int)eng->batch_max_verify_chunk : 0)
        : (eng ? (int)eng->batch_max_prefill_chunk : 0);
    if (!eng || !eng->initialized || !hidden_batch || !token_ids ||
        (phase != NATIVE_BATCH_PHASE_PREFILL && !is_verify) ||
        (is_verify && !verify_transaction) ||
        n_tokens <= 0 || n_tokens > 8 || start_pos < 0 ||
        start_pos > MAX_SEQ_LEN - n_tokens || eng->in_batch != 0 ||
        eng->unique_resolver_owner != 0u || eng->batch_config_set != 1u ||
        (!is_verify && eng->batch_enable_prefill != 1u) ||
        (is_verify && eng->batch_enable_dspark_verify != 1u) ||
        eng->batch_enable_mla != 1u ||
        eng->batch_enable_expert_buckets != 1u ||
        eng->batch_enable_batched_expert_kernels != 1u ||
        n_tokens > phase_limit) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    const char *max_layers = getenv("NATIVE_MAX_LAYERS");
    if (max_layers && atoi(max_layers) != N_LAYERS) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    if (eng->batch_pins) {
        for (int slot = 0; slot < eng->batch_capacity; ++slot) {
            if (eng->batch_pins[slot] != 0u) return NATIVE_BATCH_INVALID_STATE;
        }
    }
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
        if (eng->kv_cache[layer].len != start_pos) {
            return NATIVE_BATCH_INVALID_POSITION;
        }
    }

    MlaPipes pipes = moe_infer_mla_pipes(eng);
    if (!moe_infer_prefill_v2_resources_ready(eng, &pipes)) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    int status = moe_infer_prefill_v2_ensure_kv_storage(eng);
    if (status != NATIVE_BATCH_OK) return status;

    const size_t rows = (size_t)n_tokens;
    MoeInferCaptureJournal capture_journal = {0};
    float *layer_input_capture = NULL;
    MoeInferStateSnapshot pre_state_snapshots[8] = {0};
    MoeInferStateSnapshot post_state_snapshots[8] = {0};
    uint16_t *attention_hidden = (uint16_t *)calloc(rows * DIM, sizeof(uint16_t));
    uint16_t *attention_output = (uint16_t *)calloc(rows * DIM, sizeof(uint16_t));
    float *attention_post = (float *)calloc(rows * MHC_MULT, sizeof(float));
    float *attention_comb = (float *)calloc(
        rows * MHC_MULT * MHC_MULT, sizeof(float));
    if (!attention_hidden || !attention_output || !attention_post ||
        !attention_comb) {
        status = NATIVE_BATCH_CAPACITY_EXCEEDED;
        goto prefill_v2_cleanup;
    }
    // Debug capture is externally visible and cannot publish speculative rows.
    // VERIFY relies on its explicit transaction and therefore leaves capture
    // disabled until an accepted-prefix-aware capture journal exists.
    if (!is_verify && eng->debug_capture) {
        layer_input_capture = (float *)malloc(
            rows * MHC_MULT * DIM * sizeof(float));
        if (!layer_input_capture) {
            status = NATIVE_BATCH_CAPACITY_EXCEEDED;
            goto prefill_v2_cleanup;
        }
        status = capture_journal_init(
            &capture_journal,
            (size_t)N_LAYERS * rows *
                (MOE_INFER_CAPTURE_LAYER_OUTPUT -
                 MOE_INFER_CAPTURE_LAYER_INPUT + 1u),
            eng->debug_capture,
            eng->debug_capture_context);
        if (status != NATIVE_BATCH_OK) goto prefill_v2_cleanup;
    }
    if (eng->dspark_engine && eng->dspark_hidden3_cap < n_tokens) {
        float *replacement = (float *)malloc(
            (size_t)3 * n_tokens * DIM * sizeof(float));
        if (!replacement) {
            status = NATIVE_BATCH_CAPACITY_EXCEEDED;
            goto prefill_v2_cleanup;
        }
        free(eng->dspark_hidden3);
        eng->dspark_hidden3 = replacement;
        eng->dspark_hidden3_cap = n_tokens;
    }
    eng->dspark_hidden3_n = eng->dspark_engine && eng->dspark_hidden3
        ? n_tokens : 0;

    uint32_t positions[8] = {0};
    for (int token = 0; token < n_tokens; ++token) {
        positions[token] = (uint32_t)(start_pos + token);
    }
    MLARecurrentBatchState recurrent;
    mla_recurrent_batch_state_init(
        &recurrent, hidden_batch, attention_post, attention_comb,
        NULL, (uint32_t)n_tokens);
    if (capture_journal.target) {
        recurrent.pre_state_snapshots = pre_state_snapshots;
        recurrent.post_state_snapshots = post_state_snapshots;
    }

    uint64_t mla_workspace_bytes = 0u;
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
        MLAPrefillBatchArgs estimate_args = {
            .hidden = attention_hidden,
            .attention_output = attention_output,
            .batch_tokens = (uint32_t)n_tokens,
            .layer = layer,
            .positions = positions,
            .kv = {
                .start_position = (uint32_t)start_pos,
                .row_count = (uint32_t)n_tokens,
            },
        };
        uint64_t layer_bytes = 0u;
        if (mla_attention_prefill_batch_workspace_bytes(
                eng, &estimate_args, &recurrent,
                &layer_bytes) != NATIVE_BATCH_OK ||
            layer_bytes > eng->batch_workspace_budget_bytes) {
            status = NATIVE_BATCH_BUDGET_EXCEEDED;
            goto prefill_v2_cleanup;
        }
        if (layer_bytes > mla_workspace_bytes) mla_workspace_bytes = layer_bytes;
    }
    if (mla_workspace_bytes == 0u) {
        status = NATIVE_BATCH_INVALID_STATE;
        goto prefill_v2_cleanup;
    }

    NativeBatchWorkspace mla_workspace = {0};
    status = moe_infer_batch_workspace_init(
        &mla_workspace, (uint32_t)n_tokens, mla_workspace_bytes,
        eng->batch_workspace_budget_bytes);
    if (status != NATIVE_BATCH_OK) goto prefill_v2_cleanup;

    const NativeBatchConfig config = {
        .enable_batch_prefill = eng->batch_enable_prefill,
        .enable_batch_mla = eng->batch_enable_mla,
        .enable_expert_buckets = eng->batch_enable_expert_buckets,
        .enable_batched_expert_kernels =
            eng->batch_enable_batched_expert_kernels,
        .enable_dspark_batch_verify = eng->batch_enable_dspark_verify,
        .max_prefill_chunk = eng->batch_max_prefill_chunk,
        .max_verify_chunk = eng->batch_max_verify_chunk,
        .workspace_budget_bytes = eng->batch_workspace_budget_bytes,
    };
    NativeKVTransaction local_transaction;
    NativeKVTransaction *transaction = is_verify
        ? verify_transaction
        : &local_transaction;
    if (!is_verify) moe_infer_kv_transaction_init(transaction);
    status = moe_infer_kv_transaction_begin(
        eng, &config, phase, positions,
        (uint32_t)n_tokens, &mla_workspace, transaction);
    if (status != NATIVE_BATCH_OK) {
        moe_infer_batch_workspace_deinit(&mla_workspace);
        goto prefill_v2_cleanup;
    }

    if (capture_journal.target) {
        eng->debug_capture = capture_journal_record;
        eng->debug_capture_context = &capture_journal;
    }
    eng->in_batch++;
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
        @autoreleasepool {
            if (layer_input_capture) {
                memcpy(layer_input_capture, hidden_batch,
                       rows * MHC_MULT * DIM * sizeof(float));
            }
            for (int token = 0; token < n_tokens; ++token) {
                float *residual = hidden_batch +
                    (size_t)token * MHC_MULT * DIM;
                status = moe_infer_mhc_pre_norm_row(
                    eng, (int)layer, true, residual,
                    attention_hidden + (size_t)token * DIM,
                    attention_post + (size_t)token * MHC_MULT,
                    attention_comb +
                        (size_t)token * MHC_MULT * MHC_MULT);
                if (status != NATIVE_BATCH_OK) break;
            }
            if (status != NATIVE_BATCH_OK) break;

            NativeKVStagingView view = {0};
            status = moe_infer_kv_transaction_layer_view(
                eng, transaction, layer, &view);
            if (status != NATIVE_BATCH_OK) break;
            moe_infer_batch_workspace_reset(&mla_workspace);
            recurrent.completed_tokens = 0u;
            MLAPrefillBatchArgs args = {
                .hidden = attention_hidden,
                .attention_output = attention_output,
                .batch_tokens = (uint32_t)n_tokens,
                .layer = layer,
                .positions = positions,
                .kv = view,
                .kv_transaction = transaction,
                .workspace = &mla_workspace,
            };
            MLAAttentionMetrics metrics = {0};
            status = mla_attention_prefill_batch_v2(
                eng, &pipes, &eng->attn[layer], &args,
                &recurrent, &metrics);
            if (status != NATIVE_BATCH_OK) break;
            if (capture_journal.target) {
                float attention_capture[DIM];
                for (int token = 0; token < n_tokens; ++token) {
                    emit_debug_capture_with_snapshot(
                        eng,
                        MOE_INFER_CAPTURE_LAYER_INPUT,
                        MOE_INFER_CAPTURE_DTYPE_F32,
                        (int)layer,
                        start_pos + token,
                        layer_input_capture +
                            (size_t)token * MHC_MULT * DIM,
                        (size_t)MHC_MULT * DIM,
                        NULL,
                        0,
                        &pre_state_snapshots[token]);
                    const uint16_t *source = attention_output +
                        (size_t)token * DIM;
                    for (uint32_t dim = 0u; dim < DIM; ++dim) {
                        const uint32_t bits = (uint32_t)source[dim] << 16;
                        memcpy(&attention_capture[dim], &bits, sizeof(bits));
                    }
                    emit_debug_capture_with_snapshot(
                        eng,
                        MOE_INFER_CAPTURE_ATTENTION_OUTPUT,
                        MOE_INFER_CAPTURE_DTYPE_BF16,
                        (int)layer,
                        start_pos + token,
                        attention_capture,
                        DIM,
                        NULL,
                        0,
                        &post_state_snapshots[token]);
                }
                if (capture_journal.failed) {
                    status = NATIVE_BATCH_CAPACITY_EXCEEDED;
                    break;
                }
            }
            if (eng->telemetry_enabled &&
                metrics.workspace_high_water_bytes >
                    eng->batch_workspace_high_water_bytes) {
                eng->batch_workspace_high_water_bytes =
                    metrics.workspace_high_water_bytes;
            }

            eng->batch_count = 0;
            eng->batch_hits = 0;
            status = moe_infer_forward_batch_layer_batched_moe(
                eng, (int)layer, hidden_batch, n_tokens, start_pos,
                token_ids,
                capture_journal.target ? post_state_snapshots : NULL,
                true);
            if (status != NATIVE_BATCH_OK) break;
            if (capture_journal.failed) {
                status = NATIVE_BATCH_CAPACITY_EXCEEDED;
                break;
            }

            if (eng->dspark_engine &&
                (layer == 40u || layer == 41u || layer == 42u)) {
                if (eng->dspark_accumulate_enabled) {
                    float *last = hidden_batch +
                        (size_t)(n_tokens - 1) * MHC_MULT * DIM;
                    dspark_accumulate_target_hidden(
                        (DSparkEngine *)eng->dspark_engine, last, (int)layer);
                }
                if (eng->dspark_hidden3) {
                    float *dst = eng->dspark_hidden3 +
                        (size_t)(layer - 40u) * n_tokens * DIM;
                    for (int token = 0; token < n_tokens; ++token) {
                        const float *source = hidden_batch +
                            (size_t)token * MHC_MULT * DIM;
                        float *row = dst + (size_t)token * DIM;
                        for (uint32_t dim = 0u; dim < DIM; ++dim) {
                            float sum = 0.0f;
                            for (uint32_t stream = 0u;
                                 stream < MHC_MULT; ++stream) {
                                sum += source[(size_t)stream * DIM + dim];
                            }
                            row[dim] = sum / (float)MHC_MULT;
                        }
                    }
                }
            }
        }
        if (status != NATIVE_BATCH_OK) break;
    }

    if (status == NATIVE_BATCH_OK && !is_verify) {
        status = moe_infer_kv_transaction_commit(
            eng, transaction, (uint32_t)n_tokens);
    }
    if (status != NATIVE_BATCH_OK && transaction->active == 1u) {
        (void)moe_infer_kv_transaction_abort(eng, transaction);
    }
    if (capture_journal.target) {
        eng->debug_capture = capture_journal.target;
        eng->debug_capture_context = capture_journal.target_context;
    }
    eng->in_batch--;
    if (status == NATIVE_BATCH_OK) {
        capture_journal_publish(&capture_journal);
    }

    if (status == NATIVE_BATCH_OK && !is_verify && eng->dspark_engine &&
        eng->dspark_accumulate_enabled && eng->dspark_hidden3 &&
        eng->dspark_hidden3_n == n_tokens) {
        dspark_build_window((DSparkEngine *)eng->dspark_engine,
                            eng->dspark_hidden3, n_tokens, start_pos);
    }
    moe_infer_batch_workspace_deinit(&mla_workspace);

prefill_v2_cleanup:
    if (eng->debug_capture == capture_journal_record &&
        eng->debug_capture_context == &capture_journal) {
        eng->debug_capture = capture_journal.target;
        eng->debug_capture_context = capture_journal.target_context;
    }
    capture_journal_deinit(&capture_journal);
    free(layer_input_capture);
    free(attention_hidden);
    free(attention_output);
    free(attention_post);
    free(attention_comb);
    return status;
}

int moe_infer_forward_prefill_batch_v2(
    MoEInferEngine *eng,
    float *hidden_batch,
    int n_tokens,
    int start_pos,
    const int *token_ids) {
    return moe_infer_forward_batch_v2_internal(
        eng, hidden_batch, n_tokens, start_pos, token_ids,
        NATIVE_BATCH_PHASE_PREFILL, NULL);
}

int moe_infer_forward_verify_batch_v2(
    MoEInferEngine *eng,
    float *hidden_batch,
    int n_tokens,
    int start_pos,
    const int *token_ids,
    NativeKVTransaction *transaction) {
    if (!transaction) return NATIVE_BATCH_INVALID_ARGUMENT;
    moe_infer_kv_transaction_init(transaction);
    return moe_infer_forward_batch_v2_internal(
        eng, hidden_batch, n_tokens, start_pos, token_ids,
        NATIVE_BATCH_PHASE_VERIFY, transaction);
}

typedef struct NativeVerifyLayerState {
    CompressorState original;
    float *state_kv;
    float *state_score;
    float *comp_append;
    float *idx_state_kv;
    float *idx_state_score;
    float *idx_append;
    float *main_row_kv;
    float *main_row_score;
    float *main_output;
    float *idx_row_kv;
    float *idx_row_score;
    float *idx_output;
    uint8_t *main_boundary;
    uint8_t *idx_row_valid;
    uint8_t *idx_boundary;
    uint32_t comp_append_rows;
    uint32_t idx_append_rows;
    uint32_t recorded_rows;
} NativeVerifyLayerState;

typedef struct NativeVerifyState {
    uint64_t allocated_bytes;
    uint64_t budget_bytes;
    uint32_t rows;
    NativeVerifyLayerState layers[N_LAYERS];
} NativeVerifyState;

static void *verify_state_alloc(
    NativeVerifyState *state, uint64_t count, uint64_t element_size) {
    if (!state || count == 0u || element_size == 0u ||
        count > UINT64_MAX / element_size) return NULL;
    const uint64_t bytes = count * element_size;
    if (bytes > state->budget_bytes - state->allocated_bytes ||
        bytes > SIZE_MAX) return NULL;
    void *result = calloc(1u, (size_t)bytes);
    if (!result) return NULL;
    state->allocated_bytes += bytes;
    return result;
}

static float *verify_state_copy_f32(
    NativeVerifyState *state, const float *source, uint64_t count) {
    if (!source || count == 0u) return NULL;
    float *copy = (float *)verify_state_alloc(state, count, sizeof(float));
    if (copy) memcpy(copy, source, (size_t)count * sizeof(float));
    return copy;
}

static void verify_layer_state_deinit(NativeVerifyLayerState *layer) {
    if (!layer) return;
    free(layer->state_kv);
    free(layer->state_score);
    free(layer->comp_append);
    free(layer->idx_state_kv);
    free(layer->idx_state_score);
    free(layer->idx_append);
    free(layer->main_row_kv);
    free(layer->main_row_score);
    free(layer->main_output);
    free(layer->idx_row_kv);
    free(layer->idx_row_score);
    free(layer->idx_output);
    free(layer->main_boundary);
    free(layer->idx_row_valid);
    free(layer->idx_boundary);
    memset(layer, 0, sizeof(*layer));
}

static void verify_state_deinit(NativeVerifyState *state) {
    if (!state) return;
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
        verify_layer_state_deinit(&state->layers[layer]);
    }
    free(state);
}

static bool verify_layer_state_init(
    NativeVerifyState *state,
    NativeVerifyLayerState *snapshot,
    const CompressorState *current) {
    if (!state || !snapshot || !current) return false;
    snapshot->original = *current;
    const uint64_t rows = state->rows;
    if (current->ratio == 0u) return true;
    if (!current->state_kv || !current->state_score || current->out_dim == 0u) {
        return false;
    }

    const uint64_t state_elements =
        (uint64_t)2u * current->ratio * current->out_dim;
    snapshot->state_kv = verify_state_copy_f32(
        state, current->state_kv, state_elements);
    snapshot->state_score = verify_state_copy_f32(
        state, current->state_score, state_elements);
    snapshot->main_row_kv = (float *)verify_state_alloc(
        state, rows * current->out_dim, sizeof(float));
    snapshot->main_row_score = (float *)verify_state_alloc(
        state, rows * current->out_dim, sizeof(float));
    snapshot->main_output = (float *)verify_state_alloc(
        state, rows * COMP_HEAD_DIM, sizeof(float));
    snapshot->main_boundary = (uint8_t *)verify_state_alloc(
        state, rows, sizeof(uint8_t));
    if (!snapshot->state_kv || !snapshot->state_score ||
        !snapshot->main_row_kv || !snapshot->main_row_score ||
        !snapshot->main_output || !snapshot->main_boundary) return false;

    if (current->comp_kv && current->n_comp < MAX_COMP_BLOCKS) {
        snapshot->comp_append_rows = (uint32_t)fmin(
            (double)rows, (double)(MAX_COMP_BLOCKS - current->n_comp));
        snapshot->comp_append = verify_state_copy_f32(
            state,
            current->comp_kv + (size_t)current->n_comp * COMP_HEAD_DIM,
            (uint64_t)snapshot->comp_append_rows * COMP_HEAD_DIM);
        if (snapshot->comp_append_rows != 0u && !snapshot->comp_append) return false;
    }

    if (current->idx_state_kv || current->idx_state_score ||
        current->idx_comp_kv) {
        if (!current->idx_state_kv || !current->idx_state_score ||
            !current->idx_comp_kv) return false;
        const uint64_t idx_state_elements = (uint64_t)8u * 256u;
        snapshot->idx_state_kv = verify_state_copy_f32(
            state, current->idx_state_kv, idx_state_elements);
        snapshot->idx_state_score = verify_state_copy_f32(
            state, current->idx_state_score, idx_state_elements);
        snapshot->idx_row_kv = (float *)verify_state_alloc(
            state, rows * 256u, sizeof(float));
        snapshot->idx_row_score = (float *)verify_state_alloc(
            state, rows * 256u, sizeof(float));
        snapshot->idx_output = (float *)verify_state_alloc(
            state, rows * IDX_HEAD_DIM, sizeof(float));
        snapshot->idx_row_valid = (uint8_t *)verify_state_alloc(
            state, rows, sizeof(uint8_t));
        snapshot->idx_boundary = (uint8_t *)verify_state_alloc(
            state, rows, sizeof(uint8_t));
        if (!snapshot->idx_state_kv || !snapshot->idx_state_score ||
            !snapshot->idx_row_kv || !snapshot->idx_row_score ||
            !snapshot->idx_output || !snapshot->idx_row_valid ||
            !snapshot->idx_boundary) return false;
        if (current->n_idx_comp < MAX_COMP_BLOCKS) {
            snapshot->idx_append_rows = (uint32_t)fmin(
                (double)rows,
                (double)(MAX_COMP_BLOCKS - current->n_idx_comp));
            snapshot->idx_append = verify_state_copy_f32(
                state,
                current->idx_comp_kv +
                    (size_t)current->n_idx_comp * IDX_HEAD_DIM,
                (uint64_t)snapshot->idx_append_rows * IDX_HEAD_DIM);
            if (snapshot->idx_append_rows != 0u && !snapshot->idx_append) {
                return false;
            }
        }
    }
    return true;
}

static NativeVerifyState *verify_state_create(
    const MoEInferEngine *eng, uint32_t rows, uint64_t budget_bytes) {
    if (!eng || rows == 0u || budget_bytes < sizeof(NativeVerifyState)) {
        return NULL;
    }
    NativeVerifyState *state = (NativeVerifyState *)calloc(1u, sizeof(*state));
    if (!state) return NULL;
    // The fixed journal object contains all 43 layer descriptors and is part
    // of the same hard budget as payload snapshots/deltas. Starting payload
    // accounting at sizeof(*state) makes budget_bytes an actual upper bound on
    // total journal heap memory rather than only its variable allocations.
    state->allocated_bytes = sizeof(*state);
    state->rows = rows;
    state->budget_bytes = budget_bytes;
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
        if (!verify_layer_state_init(
                state, &state->layers[layer], &eng->comp_state[layer])) {
            verify_state_deinit(state);
            return NULL;
        }
    }
    return state;
}

static void verify_layer_restore(
    CompressorState *current, const NativeVerifyLayerState *snapshot) {
    if (!current || !snapshot) return;
    const CompressorState original = snapshot->original;
    if (original.state_kv && snapshot->state_kv) {
        const size_t elements =
            (size_t)2u * original.ratio * original.out_dim;
        memcpy(original.state_kv, snapshot->state_kv,
            elements * sizeof(float));
        memcpy(original.state_score, snapshot->state_score,
            elements * sizeof(float));
    }
    if (original.comp_kv && snapshot->comp_append &&
        snapshot->comp_append_rows != 0u) {
        memcpy(original.comp_kv + (size_t)original.n_comp * COMP_HEAD_DIM,
            snapshot->comp_append,
            (size_t)snapshot->comp_append_rows * COMP_HEAD_DIM * sizeof(float));
    }
    if (original.idx_state_kv && snapshot->idx_state_kv) {
        memcpy(original.idx_state_kv, snapshot->idx_state_kv,
            (size_t)8u * 256u * sizeof(float));
        memcpy(original.idx_state_score, snapshot->idx_state_score,
            (size_t)8u * 256u * sizeof(float));
    }
    if (original.idx_comp_kv && snapshot->idx_append &&
        snapshot->idx_append_rows != 0u) {
        memcpy(original.idx_comp_kv +
                (size_t)original.n_idx_comp * IDX_HEAD_DIM,
            snapshot->idx_append,
            (size_t)snapshot->idx_append_rows * IDX_HEAD_DIM * sizeof(float));
    }
    *current = original;
}

static uint32_t verify_count_flags(const uint8_t *flags, uint32_t rows) {
    if (!flags) return 0u;
    uint32_t count = 0u;
    for (uint32_t row = 0u; row < rows; ++row) count += flags[row] != 0u;
    return count;
}

static bool verify_layer_apply_prefix(
    CompressorState *current,
    const NativeVerifyLayerState *snapshot,
    uint32_t accepted_rows,
    uint32_t start_position) {
    if (!current || !snapshot || accepted_rows > snapshot->recorded_rows) {
        return false;
    }
    verify_layer_restore(current, snapshot);
    if (current->ratio == 0u) return true;
    for (uint32_t row = 0u; row < accepted_rows; ++row) {
        const uint32_t position = start_position + row;
        const uint32_t slot = current->ratio == 4u
            ? current->ratio + position % current->ratio
            : position % current->ratio;
        memcpy(current->state_kv + (size_t)slot * current->out_dim,
            snapshot->main_row_kv + (size_t)row * current->out_dim,
            (size_t)current->out_dim * sizeof(float));
        memcpy(current->state_score + (size_t)slot * current->out_dim,
            snapshot->main_row_score + (size_t)row * current->out_dim,
            (size_t)current->out_dim * sizeof(float));
        if (snapshot->main_boundary[row]) {
            if (!current->comp_kv || current->n_comp >= MAX_COMP_BLOCKS) {
                return false;
            }
            memcpy(current->comp_kv +
                    (size_t)current->n_comp * COMP_HEAD_DIM,
                snapshot->main_output + (size_t)row * COMP_HEAD_DIM,
                (size_t)COMP_HEAD_DIM * sizeof(float));
            current->n_comp++;
            memcpy(current->state_kv,
                current->state_kv +
                    (size_t)current->ratio * current->out_dim,
                (size_t)current->ratio * current->out_dim * sizeof(float));
            memcpy(current->state_score,
                current->state_score +
                    (size_t)current->ratio * current->out_dim,
                (size_t)current->ratio * current->out_dim * sizeof(float));
        }
        if (snapshot->idx_row_valid && snapshot->idx_row_valid[row]) {
            const uint32_t idx_slot = 4u + position % 4u;
            memcpy(current->idx_state_kv + (size_t)idx_slot * 256u,
                snapshot->idx_row_kv + (size_t)row * 256u,
                (size_t)256u * sizeof(float));
            memcpy(current->idx_state_score + (size_t)idx_slot * 256u,
                snapshot->idx_row_score + (size_t)row * 256u,
                (size_t)256u * sizeof(float));
            if (snapshot->idx_boundary[row]) {
                if (!current->idx_comp_kv ||
                    current->n_idx_comp >= MAX_COMP_BLOCKS) return false;
                memcpy(current->idx_comp_kv +
                        (size_t)current->n_idx_comp * IDX_HEAD_DIM,
                    snapshot->idx_output + (size_t)row * IDX_HEAD_DIM,
                    (size_t)IDX_HEAD_DIM * sizeof(float));
                current->n_idx_comp++;
                memcpy(current->idx_state_kv,
                    current->idx_state_kv + (size_t)4u * 256u,
                    (size_t)4u * 256u * sizeof(float));
                memcpy(current->idx_state_score,
                    current->idx_state_score + (size_t)4u * 256u,
                    (size_t)4u * 256u * sizeof(float));
            }
        }
    }
    return true;
}

static void verify_state_restore_all(
    MoEInferEngine *eng, const NativeVerifyState *state) {
    if (!eng || !state) return;
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
        verify_layer_restore(&eng->comp_state[layer], &state->layers[layer]);
    }
}

static bool verify_state_commit_prefix(
    MoEInferEngine *eng,
    NativeVerifyState *state,
    uint32_t accepted_rows,
    uint32_t start_position) {
    if (!eng || !state || accepted_rows > state->rows) return false;
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
        const NativeVerifyLayerState *snapshot = &state->layers[layer];
        if (snapshot->original.ratio != 0u &&
            snapshot->recorded_rows != state->rows) return false;
    }
    if (accepted_rows == state->rows) return true;
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
        if (!verify_layer_apply_prefix(
                &eng->comp_state[layer], &state->layers[layer],
                accepted_rows, start_position)) {
            verify_state_restore_all(eng, state);
            return false;
        }
    }
    return true;
}

void moe_infer_kv_transaction_init(NativeKVTransaction *transaction) {
    if (!transaction) return;
    memset(transaction, 0, sizeof(*transaction));
    transaction->self_cookie = (uintptr_t)transaction;
}

static bool kv_transaction_is_owned(
    const MoEInferEngine *eng,
    const NativeKVTransaction *transaction) {
    return eng && transaction && transaction->active == 1u &&
        transaction->self_cookie == (uintptr_t)transaction &&
        transaction->owner_engine == (uintptr_t)eng &&
        transaction->generation != 0u &&
        eng->kv_active_transaction_generation == transaction->generation &&
        eng->kv_active_transaction_owner == (uintptr_t)transaction;
}

int moe_infer_kv_transaction_validate_layer_view(
    const MoEInferEngine *eng,
    const NativeKVTransaction *transaction,
    uint32_t layer,
    const NativeKVStagingView *view) {
    if (!kv_transaction_is_owned(eng, transaction) || !view ||
        layer >= N_LAYERS || !eng->kv_cache[layer].kv ||
        transaction->original_kv_lengths[layer] !=
            (int)transaction->start_position ||
        eng->kv_cache[layer].len != transaction->original_kv_lengths[layer] ||
        (transaction->completed_layer_mask & (UINT64_C(1) << layer)) != 0u) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    const void *expected_data = eng->kv_cache[layer].kv +
        (size_t)transaction->start_position * KV_LORA_RANK;
    if (view->kv_data != expected_data ||
        view->kv_gpu_buffer != eng->kv_cache[layer].kv_gpu_buf ||
        view->generation != transaction->generation ||
        view->layer != layer ||
        view->start_position != transaction->start_position ||
        view->row_count != transaction->staged_tokens ||
        view->row_stride_bytes != KV_LORA_RANK * sizeof(uint16_t) ||
        view->reserved != 0u) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    return NATIVE_BATCH_OK;
}

int moe_infer_kv_transaction_begin(
    MoEInferEngine *eng,
    const NativeBatchConfig *config,
    NativeBatchPhase phase,
    const uint32_t *positions,
    uint32_t batch_tokens,
    const NativeBatchWorkspace *workspace,
    NativeKVTransaction *transaction) {
    if (!eng || !positions || !transaction ||
        transaction->self_cookie != (uintptr_t)transaction || transaction->active ||
        moe_infer_batch_config_validate(config) != NATIVE_BATCH_OK ||
        phase < NATIVE_BATCH_PHASE_PREFILL || phase > NATIVE_BATCH_PHASE_DECODE ||
        batch_tokens == 0u) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if (eng->kv_active_transaction_generation != 0u ||
        eng->kv_active_transaction_owner != 0u) {
        return NATIVE_BATCH_INVALID_STATE;
    }

    const uint32_t phase_limit = phase == NATIVE_BATCH_PHASE_PREFILL
        ? config->max_prefill_chunk
        : (phase == NATIVE_BATCH_PHASE_VERIFY ? config->max_verify_chunk : 1u);
    if (batch_tokens > phase_limit || !batch_workspace_is_owned(workspace) ||
        batch_tokens > workspace->capacity_tokens) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    if (workspace->budget_bytes > config->workspace_budget_bytes) {
        return NATIVE_BATCH_BUDGET_EXCEEDED;
    }

    const int committed_length = eng->kv_cache[0].len;
    if (committed_length < 0 || committed_length > MAX_SEQ_LEN) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    for (int layer = 1; layer < N_LAYERS; ++layer) {
        if (eng->kv_cache[layer].len != committed_length) {
            return NATIVE_BATCH_INVALID_STATE;
        }
    }
    if (positions[0] != (uint32_t)committed_length ||
        positions[0] >= MAX_SEQ_LEN ||
        batch_tokens > (uint32_t)MAX_SEQ_LEN - positions[0]) {
        return NATIVE_BATCH_INVALID_POSITION;
    }
    for (uint32_t token = 1; token < batch_tokens; ++token) {
        if (positions[token] != positions[0] + token) {
            return NATIVE_BATCH_INVALID_POSITION;
        }
    }

    NativeVerifyState *verify_state = NULL;
    if (phase == NATIVE_BATCH_PHASE_VERIFY) {
        if (workspace->allocated_bytes >= config->workspace_budget_bytes) {
            return NATIVE_BATCH_BUDGET_EXCEEDED;
        }
        verify_state = verify_state_create(
            eng, batch_tokens,
            config->workspace_budget_bytes - workspace->allocated_bytes);
        if (!verify_state) return NATIVE_BATCH_BUDGET_EXCEEDED;
    }

    // All validation and recurrent snapshot allocation precede ownership
    // reservation and any KV/recurrent mutation.
    uint64_t generation = eng->kv_transaction_generation + 1u;
    if (generation == 0u) generation = 1u;
    const uintptr_t self_cookie = transaction->self_cookie;
    memset(transaction, 0, sizeof(*transaction));
    transaction->self_cookie = self_cookie;
    transaction->generation = generation;
    transaction->owner_engine = (uintptr_t)eng;
    transaction->phase = (uint32_t)phase;
    transaction->start_position = positions[0];
    transaction->staged_tokens = batch_tokens;
    transaction->verify_state = verify_state;
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        transaction->original_kv_lengths[layer] = eng->kv_cache[layer].len;
    }
    transaction->active = 1u;
    eng->kv_transaction_generation = generation;
    eng->kv_active_transaction_generation = generation;
    eng->kv_active_transaction_owner = (uintptr_t)transaction;
    return NATIVE_BATCH_OK;
}

int moe_infer_kv_transaction_layer_view(
    MoEInferEngine *eng,
    NativeKVTransaction *transaction,
    uint32_t layer,
    NativeKVStagingView *out_view) {
    if (!kv_transaction_is_owned(eng, transaction) || !out_view ||
        layer >= N_LAYERS || !eng->kv_cache[layer].kv) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    *out_view = (NativeKVStagingView){
        .kv_data = eng->kv_cache[layer].kv +
            (size_t)transaction->start_position * KV_LORA_RANK,
        .kv_gpu_buffer = eng->kv_cache[layer].kv_gpu_buf,
        .generation = transaction->generation,
        .layer = layer,
        .start_position = transaction->start_position,
        .row_count = transaction->staged_tokens,
        .row_stride_bytes = KV_LORA_RANK * sizeof(uint16_t),
    };
    return NATIVE_BATCH_OK;
}

int moe_infer_kv_transaction_mark_layer_staged(
    MoEInferEngine *eng,
    NativeKVTransaction *transaction,
    uint32_t layer,
    uint32_t staged_tokens) {
    if (!kv_transaction_is_owned(eng, transaction) || layer >= N_LAYERS ||
        staged_tokens != transaction->staged_tokens) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    const int staged_end = (int)transaction->start_position + (int)staged_tokens;
    if (eng->kv_cache[layer].len != staged_end) return NATIVE_BATCH_INVALID_STATE;
    if (transaction->phase == NATIVE_BATCH_PHASE_VERIFY) {
        NativeVerifyState *state = (NativeVerifyState *)transaction->verify_state;
        if (!state || state->layers[layer].recorded_rows != staged_tokens) {
            return NATIVE_BATCH_INVALID_STATE;
        }
    }
    transaction->completed_layer_mask |= UINT64_C(1) << layer;
    return NATIVE_BATCH_OK;
}

int moe_infer_kv_transaction_record_recurrent_row(
    MoEInferEngine *eng,
    NativeKVTransaction *transaction,
    uint32_t layer,
    uint32_t row) {
    if (!kv_transaction_is_owned(eng, transaction) ||
        transaction->phase != NATIVE_BATCH_PHASE_VERIFY ||
        !transaction->verify_state || layer >= N_LAYERS ||
        row >= transaction->staged_tokens ||
        (transaction->completed_layer_mask & (UINT64_C(1) << layer)) != 0u) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    NativeVerifyState *state = (NativeVerifyState *)transaction->verify_state;
    NativeVerifyLayerState *snapshot = &state->layers[layer];
    CompressorState *current = &eng->comp_state[layer];
    if (snapshot->recorded_rows != row || state->rows != transaction->staged_tokens) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    if (current->ratio == 0u) {
        snapshot->recorded_rows++;
        return NATIVE_BATCH_OK;
    }
    if (current->ratio != snapshot->original.ratio ||
        current->out_dim != snapshot->original.out_dim ||
        current->state_kv != snapshot->original.state_kv ||
        current->state_score != snapshot->original.state_score) {
        return NATIVE_BATCH_INVALID_STATE;
    }

    const uint32_t position = transaction->start_position + row;
    const uint32_t slot = current->ratio == 4u
            ? current->ratio + position % current->ratio
            : position % current->ratio;
    memcpy(snapshot->main_row_kv + (size_t)row * current->out_dim,
        current->state_kv + (size_t)slot * current->out_dim,
        (size_t)current->out_dim * sizeof(float));
    memcpy(snapshot->main_row_score + (size_t)row * current->out_dim,
        current->state_score + (size_t)slot * current->out_dim,
        (size_t)current->out_dim * sizeof(float));

    const uint32_t expected_comp = snapshot->original.n_comp +
        verify_count_flags(snapshot->main_boundary, row);
    if (current->n_comp < expected_comp || current->n_comp > expected_comp + 1u ||
        (current->n_comp == expected_comp + 1u &&
         (position + 1u) % current->ratio != 0u)) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    if (current->n_comp == expected_comp + 1u) {
        snapshot->main_boundary[row] = 1u;
        memcpy(snapshot->main_output + (size_t)row * COMP_HEAD_DIM,
            current->comp_kv + (size_t)expected_comp * COMP_HEAD_DIM,
            (size_t)COMP_HEAD_DIM * sizeof(float));
    }

    const uint32_t expected_idx = snapshot->original.n_idx_comp +
        verify_count_flags(snapshot->idx_boundary, row);
    const bool idx_active = current->n_comp > 512u &&
        snapshot->original.idx_state_kv && snapshot->original.idx_state_score &&
        snapshot->original.idx_comp_kv;
    if (current->n_idx_comp < expected_idx ||
        current->n_idx_comp > expected_idx + 1u ||
        (current->n_idx_comp == expected_idx + 1u &&
         ((position + 1u) % 4u != 0u || !idx_active))) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    if (idx_active) {
        snapshot->idx_row_valid[row] = 1u;
        const uint32_t idx_slot = 4u + position % 4u;
        memcpy(snapshot->idx_row_kv + (size_t)row * 256u,
            current->idx_state_kv + (size_t)idx_slot * 256u,
            (size_t)256u * sizeof(float));
        memcpy(snapshot->idx_row_score + (size_t)row * 256u,
            current->idx_state_score + (size_t)idx_slot * 256u,
            (size_t)256u * sizeof(float));
        if (current->n_idx_comp == expected_idx + 1u) {
            snapshot->idx_boundary[row] = 1u;
            memcpy(snapshot->idx_output + (size_t)row * IDX_HEAD_DIM,
                current->idx_comp_kv + (size_t)expected_idx * IDX_HEAD_DIM,
                (size_t)IDX_HEAD_DIM * sizeof(float));
        }
    } else if (current->n_idx_comp != expected_idx) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    snapshot->recorded_rows++;
    return NATIVE_BATCH_OK;
}

int moe_infer_kv_transaction_commit(
    MoEInferEngine *eng,
    NativeKVTransaction *transaction,
    uint32_t accepted_tokens) {
    if (!kv_transaction_is_owned(eng, transaction)) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    if (accepted_tokens > transaction->staged_tokens) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if ((transaction->phase == NATIVE_BATCH_PHASE_PREFILL &&
         accepted_tokens != transaction->staged_tokens) ||
        (transaction->phase == NATIVE_BATCH_PHASE_DECODE &&
         (transaction->staged_tokens != 1u || accepted_tokens != 1u))) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    const uint64_t all_layers_mask = (UINT64_C(1) << N_LAYERS) - 1u;
    if (transaction->completed_layer_mask != all_layers_mask) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    const int target_length = (int)transaction->start_position + (int)accepted_tokens;
    const int staged_end = (int)transaction->start_position + (int)transaction->staged_tokens;
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        if (transaction->original_kv_lengths[layer] != (int)transaction->start_position ||
            eng->kv_cache[layer].len != staged_end) {
            return NATIVE_BATCH_INVALID_STATE;
        }
    }
    if (transaction->phase == NATIVE_BATCH_PHASE_VERIFY) {
        NativeVerifyState *state = (NativeVerifyState *)transaction->verify_state;
        if (!state || !verify_state_commit_prefix(
                eng, state, accepted_tokens, transaction->start_position)) {
            return NATIVE_BATCH_INVALID_STATE;
        }
    }
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        eng->kv_cache[layer].len = target_length;
    }
    verify_state_deinit((NativeVerifyState *)transaction->verify_state);
    transaction->verify_state = NULL;
    transaction->accepted_tokens = accepted_tokens;
    transaction->active = 0u;
    eng->kv_active_transaction_generation = 0u;
    eng->kv_active_transaction_owner = 0u;
    return NATIVE_BATCH_OK;
}

int moe_infer_kv_transaction_abort(
    MoEInferEngine *eng,
    NativeKVTransaction *transaction) {
    if (!kv_transaction_is_owned(eng, transaction)) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    if (transaction->verify_state) {
        NativeVerifyState *state = (NativeVerifyState *)transaction->verify_state;
        verify_state_restore_all(eng, state);
        verify_state_deinit(state);
        transaction->verify_state = NULL;
    }
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        eng->kv_cache[layer].len = transaction->original_kv_lengths[layer];
    }
    transaction->accepted_tokens = 0u;
    transaction->active = 0u;
    eng->kv_active_transaction_generation = 0u;
    eng->kv_active_transaction_owner = 0u;
    return NATIVE_BATCH_OK;
}

int moe_infer_exact_q4_drain_and_abort_request(MoEInferEngine *eng) {
    if (!eng) return DMLX_EXACT_Q4_PRODUCER_DISABLED;
    if (eng->deferred.active && eng->deferred.cmd_experts) {
        [(id<MTLCommandBuffer>)eng->deferred.cmd_experts waitUntilCompleted];
        [(id<MTLCommandBuffer>)eng->deferred.cmd_experts release];
        eng->deferred.cmd_experts = NULL;
    }
    eng->deferred.active = false;
    exact_q4_layer_discard(&eng->deferred.exact_q4_pending);
    prefetch_drain(eng);
    if (eng->io_pool) io_pool_wait((IOPool *)eng->io_pool);
    return moe_infer_exact_q4_abort_request(eng);
}

void moe_infer_reset_kv(MoEInferEngine *eng) {
    // If a deferred CMD3 is in-flight from the previous request, wait and release it
    // before clearing state. This prevents use-after-free on GPU buffers.
    if (eng->deferred.active && eng->deferred.cmd_experts) {
        [(id<MTLCommandBuffer>)eng->deferred.cmd_experts waitUntilCompleted];
        [(id<MTLCommandBuffer>)eng->deferred.cmd_experts release];
        eng->deferred.cmd_experts = NULL;
    }
    eng->deferred.active = false;
    exact_q4_layer_discard(&eng->deferred.exact_q4_pending);
    // Reset invalidates any caller-held transaction. Release an owned VERIFY
    // journal before zeroing engine state; the caller's generation remains
    // stale and later commit/abort fails without touching the new request.
    if (eng->kv_active_transaction_owner != 0u) {
        NativeKVTransaction *transaction = (NativeKVTransaction *)
            eng->kv_active_transaction_owner;
        if (transaction->self_cookie == (uintptr_t)transaction &&
            transaction->owner_engine == (uintptr_t)eng &&
            transaction->verify_state) {
            verify_state_deinit((NativeVerifyState *)transaction->verify_state);
            transaction->verify_state = NULL;
            transaction->active = 0u;
        }
    }
    eng->kv_active_transaction_generation = 0u;
    eng->kv_active_transaction_owner = 0u;
    prefetch_drain(eng);
    // Invalidate cross-request predictions: routing from the previous request
    // is stale, and the first token(s) of a new request (special chat-template
    // tokens) route atypically. Re-recording starts fresh on the next token.
    eng->predictor.valid = false;
    eng->predictor.hits = 0;
    eng->predictor.misses = 0;
    eng->decode_phase = false;  // re-armed by moe_infer_smelt_set_decode_phase after prefill
    eng->deferred.gpu_combined = false;

    // Reset SMELT decode phase flag so next request's prefill doesn't count for warmup.
    // But preserve routing_counts (accumulate across requests for better hot-expert detection)
    // and smelt_tokens_seen (counts total decode tokens seen, not per-request).
    if (eng->smelt_enabled && !eng->smelt_warmup_done) {
        eng->smelt_in_decode_phase = false;
        // NOTE: smelt_tokens_seen and routing_counts are NOT reset — they accumulate across requests
    }

    for (int l = 0; l < N_LAYERS; l++) {
        eng->kv_cache[l].len = 0;
        // kv_cache[l].len = 0 is sufficient to prevent stale reads.
        // The memset below is defensive but writes 180MB per request (43 × 4096 × 512 × 2 bytes),
        // causing unnecessary memory pressure during long E2E sessions. Removed.
        // Correctness: forward_layer only reads entries at indices 0..len-1; with len=0
        // no stale KV data is ever accessed.
        //
        // kv_gpu_buf is NOT freed/released here — it persists for the engine lifetime
        // and is reused for every new request.
        CompressorState *cs = &eng->comp_state[l];
        cs->n_comp = 0;
        cs->n_idx_comp = 0;
        if (cs->state_kv && cs->ratio > 0 && cs->out_dim > 0)
            memset(cs->state_kv, 0, 2 * cs->ratio * cs->out_dim * sizeof(float));
        if (cs->state_score && cs->ratio > 0 && cs->out_dim > 0)
            memset(cs->state_score, 0, 2 * cs->ratio * cs->out_dim * sizeof(float));
        if (cs->comp_kv)
            memset(cs->comp_kv, 0, (size_t)MAX_COMP_BLOCKS * COMP_HEAD_DIM * sizeof(float));
        if (cs->idx_state_kv)  memset(cs->idx_state_kv, 0, 8 * 256 * sizeof(float));
        if (cs->idx_state_score) memset(cs->idx_state_score, 0, 8 * 256 * sizeof(float));
        if (cs->idx_comp_kv)
            memset(cs->idx_comp_kv, 0, (size_t)MAX_COMP_BLOCKS * IDX_HEAD_DIM * sizeof(float));
    }
}

void moe_infer_rollback_kv(MoEInferEngine *eng, int valid_len) {
    // Truncate KV cache for all layers to valid_len.
    // Used by speculative decoding when draft tokens are rejected.
    if (valid_len < 0) valid_len = 0;
    prefetch_drain(eng);
    for (int l = 0; l < N_LAYERS; l++) {
        if (eng->kv_cache[l].len > valid_len) {
            eng->kv_cache[l].len = valid_len;
        }
        // Rollback compressor state: n_comp = valid_len / ratio (rounded down)
        CompressorState *cs = &eng->comp_state[l];
        if (cs->ratio > 0) {
            uint32_t new_n_comp = (uint32_t)valid_len / cs->ratio;
            if (cs->n_comp > new_n_comp) {
                cs->n_comp = new_n_comp;
            }
            // Reset the rolling state buffer for partial blocks
            // (tokens that haven't formed a complete compressed block yet)
            uint32_t partial = (uint32_t)valid_len % cs->ratio;
            if (partial == 0 && cs->state_kv && cs->out_dim > 0) {
                memset(cs->state_kv, 0, 2 * cs->ratio * cs->out_dim * sizeof(float));
                memset(cs->state_score, 0, 2 * cs->ratio * cs->out_dim * sizeof(float));
            }
            // Indexer compressor (ratio=4 layers only)
            if (cs->n_idx_comp > new_n_comp) {
                cs->n_idx_comp = new_n_comp;
            }
        }
    }
}

void moe_infer_set_layer_hc(MoEInferEngine *eng, int layer,
    const float *attn_fn, const float *attn_base, const float *attn_scale,
    const float *ffn_fn, const float *ffn_base, const float *ffn_scale) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->attn_hc_fn[layer] = attn_fn;
    eng->attn_hc_base[layer] = attn_base;
    eng->attn_hc_scale[layer] = attn_scale;
    eng->ffn_hc_fn[layer] = ffn_fn;
    eng->ffn_hc_base[layer] = ffn_base;
    eng->ffn_hc_scale[layer] = ffn_scale;

    // Upload mHC weights to persistent GPU buffers immediately (avoids per-call allocation)
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    const int MIX3 = MHC_MULT * (MHC_MULT + 2);  // 24
    const size_t fn_bytes    = (size_t)MIX3 * MHC_MULT * DIM * sizeof(float);
    const size_t base_bytes  = (size_t)MIX3 * sizeof(float);
    const size_t scale_bytes = 3 * sizeof(float);

    if (attn_fn && !eng->buf_attn_hc_fn[layer]) {
        eng->buf_attn_hc_fn[layer]    = (void *)[d newBufferWithBytes:(void*)attn_fn    length:fn_bytes    options:MTLResourceStorageModeShared];
        eng->buf_attn_hc_base[layer]  = (void *)[d newBufferWithBytes:(void*)attn_base  length:base_bytes  options:MTLResourceStorageModeShared];
        eng->buf_attn_hc_scale[layer] = (void *)[d newBufferWithBytes:(void*)attn_scale length:scale_bytes options:MTLResourceStorageModeShared];
    }
    if (ffn_fn && !eng->buf_ffn_hc_fn[layer]) {
        eng->buf_ffn_hc_fn[layer]    = (void *)[d newBufferWithBytes:(void*)ffn_fn    length:fn_bytes    options:MTLResourceStorageModeShared];
        eng->buf_ffn_hc_base[layer]  = (void *)[d newBufferWithBytes:(void*)ffn_base  length:base_bytes  options:MTLResourceStorageModeShared];
        eng->buf_ffn_hc_scale[layer] = (void *)[d newBufferWithBytes:(void*)ffn_scale length:scale_bytes options:MTLResourceStorageModeShared];
    }
}


// ============================================================================
// Embed / compress / logits helpers (needed by native engine, no MLX)
// ============================================================================

void moe_infer_embed(MoEInferEngine *eng, int token_id, float *hidden_out) {
    // Dequantize one row from the packed 4-bit affine embed table when the
    // dense f32 table is absent (saves 1.6GB RAM). The row expression is
    // identical to the loader's dequantAffineF32 — same f32 bits.
    float rowbuf[DIM];
    const float *row;
    if (eng->embed != NULL) {
        row = eng->embed + (size_t)token_id * DIM;
    } else {
        const QuantWeight *q = &eng->embed_q;
        const uint32_t *pw_row = q->packed + (size_t)token_id * (q->in_dim / 8);
        const int ng = q->in_dim / q->group_size;
        const float *sc_row = q->scales + (size_t)token_id * ng;
        const float *bi_row = q->biases + (size_t)token_id * ng;
        const int ppg = q->group_size / 8;
        for (int g = 0; g < ng; g++) {
            const float scale = sc_row[g], bias = bi_row[g];
            for (int p = 0; p < ppg; p++) {
                uint32_t pw = pw_row[g * ppg + p];
                for (int i = 0; i < 8; i++) {
                    float nib = (float)((pw >> (i * 4)) & 0xF);
                    rowbuf[g * q->group_size + p * 8 + i] = scale * nib + bias;
                }
            }
        }
        row = rowbuf;
    }
    for (int m = 0; m < MHC_MULT; m++) {
        for (int d = 0; d < DIM; d++) {
            // Truncate embed to bf16 to match MLX embedding lookup precision
            uint32_t u; memcpy(&u, &row[d], 4); u &= 0xFFFF0000U;
            memcpy(&hidden_out[m * DIM + d], &u, 4);
        }
    }
    // Step 1 (Path B): keep buf_residual_gpu in sync with CPU hidden_out.
    // Step 2 will make GPU the source-of-truth and remove the CPU copy.
    if (eng->buf_residual_gpu) {
        memcpy([(id<MTLBuffer>)eng->buf_residual_gpu contents],
               hidden_out, (size_t)MHC_MULT * DIM * sizeof(float));
    }
}

// Register quantized embed/lm_head (native 4-bit affine). Creates the
// persistent GPU buffers for the logits matvec (one-time 530MB copy).
void moe_infer_set_head_quant(MoEInferEngine *eng, QuantWeight embed_q, QuantWeight lm_head_q) {
    eng->embed_q = embed_q;
    eng->lm_head_q = lm_head_q;
    if (!lm_head_q.packed || lm_head_q.out_dim <= 0) return;
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    size_t wbytes = (size_t)lm_head_q.out_dim * (lm_head_q.in_dim / 8) * sizeof(uint32_t);
    size_t sbytes = (size_t)lm_head_q.out_dim * (lm_head_q.in_dim / lm_head_q.group_size) * sizeof(float);
    eng->lmh_w  = (void *)[d newBufferWithBytes:(void *)lm_head_q.packed length:wbytes options:MTLResourceStorageModeShared];
    eng->lmh_sc = (void *)[d newBufferWithBytes:(void *)lm_head_q.scales length:sbytes options:MTLResourceStorageModeShared];
    eng->lmh_bi = (void *)[d newBufferWithBytes:(void *)lm_head_q.biases length:sbytes options:MTLResourceStorageModeShared];
    eng->lmh_x  = (void *)[d newBufferWithLength:DIM * sizeof(float) options:MTLResourceStorageModeShared];
    eng->lmh_out = (void *)[d newBufferWithLength:(size_t)lm_head_q.out_dim * sizeof(float) options:MTLResourceStorageModeShared];
    if (!eng->lmh_w || !eng->lmh_sc || !eng->lmh_bi || !eng->lmh_x || !eng->lmh_out) {
        fprintf(stderr, "[head-quant] GPU buffer alloc failed — logits path unavailable\n");
    }
}

static void cpu_rms_norm(const float *x, const float *w, float *out, int dim, float eps) {
    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / dim) + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv * w[i];
}

int moe_infer_get_logits(MoEInferEngine *eng, const float *hidden, float *logits_out) {
    const bool decode_timeline_recording = decode_timeline_active(eng);
    const uint64_t timeline_logits_begin = decode_timeline_recording
        ? moe_infer_decode_timeline_now_ns() : 0u;
    uint64_t timeline_logits_encode_done = timeline_logits_begin;
    uint64_t timeline_logits_pre_commit = timeline_logits_begin;
    uint64_t timeline_logits_post_commit = timeline_logits_begin;
    uint64_t timeline_logits_post_wait = timeline_logits_begin;
    if (!eng->final_norm || !logits_out) return -1;
    if (!eng->lm_head && !eng->lm_head_q.packed) return -1;
    static float normed[DIM];
    cpu_rms_norm(hidden, eng->final_norm, normed, DIM, 1e-6f);
    if (eng->lm_head) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    eng->vocab_size, DIM,
                    1.0f, eng->lm_head, DIM,
                    normed, 1,
                    0.0f, logits_out, 1);
    } else {
        // Quantized lm_head: GPU dequant matvec (530MB packed read instead of
        // 2.1GB f32). Same (scale*nib+bias)*x expression per row.
        memcpy([(id<MTLBuffer>)eng->lmh_x contents], normed, DIM * sizeof(float));
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
        [e setBuffer:(id<MTLBuffer>)eng->lmh_w  offset:0 atIndex:0];
        [e setBuffer:(id<MTLBuffer>)eng->lmh_sc offset:0 atIndex:1];
        [e setBuffer:(id<MTLBuffer>)eng->lmh_bi offset:0 atIndex:2];
        [e setBuffer:(id<MTLBuffer>)eng->lmh_x  offset:0 atIndex:3];
        [e setBuffer:(id<MTLBuffer>)eng->lmh_out offset:0 atIndex:4];
        uint od = (uint)eng->lm_head_q.out_dim, id_ = (uint)eng->lm_head_q.in_dim, gs = (uint)eng->lm_head_q.group_size;
        [e setBytes:&od length:4 atIndex:5];
        [e setBytes:&id_ length:4 atIndex:6];
        [e setBytes:&gs length:4 atIndex:7];
        [e dispatchThreads:MTLSizeMake(od, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        if (decode_timeline_recording) {
            timeline_logits_encode_done = moe_infer_decode_timeline_now_ns();
            timeline_logits_pre_commit = timeline_logits_encode_done;
        }
        [cb commit];
        if (decode_timeline_recording)
            timeline_logits_post_commit = moe_infer_decode_timeline_now_ns();
        [cb waitUntilCompleted];
        if (decode_timeline_recording)
            timeline_logits_post_wait = moe_infer_decode_timeline_now_ns();
        memcpy(logits_out, [(id<MTLBuffer>)eng->lmh_out contents], (size_t)eng->vocab_size * sizeof(float));
    }
    if (decode_timeline_recording) {
        const uint64_t end_ns = moe_infer_decode_timeline_now_ns();
        decode_timeline_record(eng, DMLX_DECODE_TIMELINE_NO_LAYER,
            DMLX_DECODE_TIMELINE_LOGITS_CB, timeline_logits_begin,
            timeline_logits_encode_done, timeline_logits_pre_commit,
            timeline_logits_post_commit, timeline_logits_post_wait, end_ns);
    }
    if (getenv("MF_DBG")) {
        int max_i = 0;
        for (int i=1;i<eng->vocab_size;i++) if (logits_out[i]>logits_out[max_i]) max_i=i;
        fprintf(stderr, "[mf-dbg] logits max=%d val=%.3f\n", max_i, logits_out[max_i]);
    }
    return 0;
}

void moe_infer_compress_hc(MoEInferEngine *eng, const float *residual, float *out) {
    (void)eng;
    for (int d = 0; d < DIM; d++) {
        float sum = 0.0f;
        for (int m = 0; m < MHC_MULT; m++) sum += residual[m * DIM + d];
        out[d] = sum / MHC_MULT;
    }
}

// ============================================================================
// Compressor/Indexer implementation (T2C.2/T2C.3)
// ============================================================================

void moe_infer_set_layer_compressor(MoEInferEngine *eng, int layer,
    uint32_t compress_ratio,
    QuantWeight comp_wkv, QuantWeight comp_wgate,
    const float *comp_ape, const float *comp_norm) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->compress_ratio[layer] = compress_ratio;
    eng->comp_wkv[layer] = comp_wkv;
    eng->comp_wgate[layer] = comp_wgate;
    eng->comp_ape[layer] = comp_ape;
    eng->comp_norm[layer] = comp_norm;
    // Pre-allocate GPU buffers for compressor weights (eliminates CPU matvec in hot path)
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    if (comp_wkv.packed && !eng->comp_wkv_gpu_w[layer]) {
        const QuantWeight *w = &eng->comp_wkv[layer];
        int ng = w->in_dim / w->group_size;
        eng->comp_wkv_gpu_w[layer] = (void *)[d newBufferWithBytesNoCopy:(void*)w->packed
            length:(size_t)w->out_dim*(w->in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
        eng->comp_wkv_gpu_s[layer] = (void *)[d newBufferWithBytes:(void*)w->scales
            length:(size_t)w->out_dim*ng*sizeof(float) options:MTLResourceStorageModeShared];
        eng->comp_wkv_gpu_b[layer] = (void *)[d newBufferWithBytes:(void*)w->biases
            length:(size_t)w->out_dim*ng*sizeof(float) options:MTLResourceStorageModeShared];
    }
    if (comp_wgate.packed && !eng->comp_wgate_gpu_w[layer]) {
        const QuantWeight *w = &eng->comp_wgate[layer];
        int ng = w->in_dim / w->group_size;
        eng->comp_wgate_gpu_w[layer] = (void *)[d newBufferWithBytesNoCopy:(void*)w->packed
            length:(size_t)w->out_dim*(w->in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
        eng->comp_wgate_gpu_s[layer] = (void *)[d newBufferWithBytes:(void*)w->scales
            length:(size_t)w->out_dim*ng*sizeof(float) options:MTLResourceStorageModeShared];
        eng->comp_wgate_gpu_b[layer] = (void *)[d newBufferWithBytes:(void*)w->biases
            length:(size_t)w->out_dim*ng*sizeof(float) options:MTLResourceStorageModeShared];
    }
    if (!eng->comp_out_gpu[layer]) {
        int out_dim = (compress_ratio == 4) ? CSA_OUT_DIM : HCA_OUT_DIM;
        eng->comp_out_gpu[layer] = (void *)[d newBufferWithLength:out_dim*sizeof(float) options:MTLResourceStorageModeShared];
        eng->comp_out2_gpu[layer] = (void *)[d newBufferWithLength:out_dim*sizeof(float) options:MTLResourceStorageModeShared];
    }
}

void moe_infer_set_layer_indexer(MoEInferEngine *eng, int layer,
    QuantWeight idx_wq_b, QuantWeight idx_weights_proj,
    QuantWeight idx_comp_wkv, QuantWeight idx_comp_wgate,
    const float *idx_comp_ape, const float *idx_comp_norm) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->idx_wq_b[layer] = idx_wq_b;
    eng->idx_weights_proj[layer] = idx_weights_proj;
    eng->idx_comp_wkv[layer] = idx_comp_wkv;
    eng->idx_comp_wgate[layer] = idx_comp_wgate;
    eng->idx_comp_ape[layer] = idx_comp_ape;
    eng->idx_comp_norm[layer] = idx_comp_norm;
    // Pre-allocate GPU buffers for indexer compressor weights (same pattern
    // as moe_infer_set_layer_compressor — eliminates CPU matvec in hot path)
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    if (idx_comp_wkv.packed && !eng->idx_comp_wkv_gpu_w[layer]) {
        const QuantWeight *w = &eng->idx_comp_wkv[layer];
        int ng = w->in_dim / w->group_size;
        eng->idx_comp_wkv_gpu_w[layer] = (void *)[d newBufferWithBytesNoCopy:(void*)w->packed
            length:(size_t)w->out_dim*(w->in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
        eng->idx_comp_wkv_gpu_s[layer] = (void *)[d newBufferWithBytes:(void*)w->scales
            length:(size_t)w->out_dim*ng*sizeof(float) options:MTLResourceStorageModeShared];
        eng->idx_comp_wkv_gpu_b[layer] = (void *)[d newBufferWithBytes:(void*)w->biases
            length:(size_t)w->out_dim*ng*sizeof(float) options:MTLResourceStorageModeShared];
    }
    if (idx_comp_wgate.packed && !eng->idx_comp_wgate_gpu_w[layer]) {
        const QuantWeight *w = &eng->idx_comp_wgate[layer];
        int ng = w->in_dim / w->group_size;
        eng->idx_comp_wgate_gpu_w[layer] = (void *)[d newBufferWithBytesNoCopy:(void*)w->packed
            length:(size_t)w->out_dim*(w->in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
        eng->idx_comp_wgate_gpu_s[layer] = (void *)[d newBufferWithBytes:(void*)w->scales
            length:(size_t)w->out_dim*ng*sizeof(float) options:MTLResourceStorageModeShared];
        eng->idx_comp_wgate_gpu_b[layer] = (void *)[d newBufferWithBytes:(void*)w->biases
            length:(size_t)w->out_dim*ng*sizeof(float) options:MTLResourceStorageModeShared];
    }
    if (!eng->idx_comp_out_gpu[layer]) {
        eng->idx_comp_out_gpu[layer]  = (void *)[d newBufferWithLength:256*sizeof(float) options:MTLResourceStorageModeShared];
        eng->idx_comp_out2_gpu[layer] = (void *)[d newBufferWithLength:256*sizeof(float) options:MTLResourceStorageModeShared];
    }
}

// CPU affine-4bit matvec (safe version): out[out_dim] = W[out_dim, in_dim] @ x[in_dim]
static void cpu_affine_matvec_safe(float *out, const QuantWeight *w, const float *x) {
    if (w->out_dim == 0 || w->in_dim == 0 || w->packed == NULL) return;
    const int num_groups = w->in_dim / w->group_size;
    const int ppg = w->group_size / 8;
    const int packed_cols = w->in_dim / 8;
    for (int row = 0; row < w->out_dim; row++) {
        const uint32_t *wr = w->packed + (size_t)row * packed_cols;
        const float *sc = w->scales + (size_t)row * num_groups;
        const float *bi = (w->biases) ? w->biases + (size_t)row * num_groups : NULL;
        float acc = 0.0f;
        for (int g = 0; g < num_groups; g++) {
            float scale = sc[g];
            float bias = bi ? bi[g] : 0.0f;
            for (int p = 0; p < ppg; p++) {
                uint32_t pw = wr[g * ppg + p];
                for (int k = 0; k < 8; k++) {
                    float nib = (float)((pw >> (k * 4)) & 0xF);
                    acc += (scale * nib + bias) * x[g * w->group_size + p * 8 + k];
                }
            }
        }
        out[row] = acc;
    }
}

// Softmax in-place over x[n]
static void cpu_softmax_inplace(float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void moe_infer_compressor_step_impl(
                                MoEInferEngine *eng, int layer, int pos,
                                const float *attn_normed,
                                float *bounded_scratch,
                                size_t bounded_scratch_floats) {
    const uint32_t ratio = eng->compress_ratio[layer];
    if (ratio == 0) return;

    CompressorState *cs = &eng->comp_state[layer];

    // Lazy initialize state buffers
    if (cs->state_kv == NULL) {
        const uint32_t out_dim = (ratio == 4) ? CSA_OUT_DIM : HCA_OUT_DIM;
        cs->ratio = ratio;
        cs->out_dim = out_dim;
        cs->state_kv    = (float*)calloc((size_t)2 * ratio * out_dim, sizeof(float));
        cs->state_score = (float*)calloc((size_t)2 * ratio * out_dim, sizeof(float));
        cs->comp_kv     = (float*)calloc((size_t)MAX_COMP_BLOCKS * COMP_HEAD_DIM, sizeof(float));
        cs->n_comp      = 0;
        if (ratio == 4) {
            cs->idx_state_kv    = (float*)calloc((size_t)8 * 256, sizeof(float));
            cs->idx_state_score = (float*)calloc((size_t)8 * 256, sizeof(float));
            cs->idx_comp_kv     = (float*)calloc((size_t)MAX_COMP_BLOCKS * IDX_HEAD_DIM, sizeof(float));
            cs->n_idx_comp      = 0;
        }
    }

    const uint32_t out_dim = cs->out_dim;
    const uint32_t pos_mod = (uint32_t)pos % ratio;
    // For CSA (ratio=4): store in upper half [ratio+pos_mod]; for HCA: at [pos_mod]
    const uint32_t row = (ratio == 4) ? (ratio + pos_mod) : pos_mod;

    const size_t recurrent_scratch_floats =
        (size_t)2u * out_dim + COMP_HEAD_DIM;
    float *kv_cur;
    float *gate_cur;
    float *pooled;
    if (bounded_scratch) {
        if (bounded_scratch_floats < recurrent_scratch_floats) return;
        kv_cur = bounded_scratch;
        gate_cur = kv_cur + out_dim;
        pooled = gate_cur + out_dim;
    } else {
        kv_cur = (float*)alloca(out_dim * sizeof(float));
        gate_cur = (float*)alloca(out_dim * sizeof(float));
        pooled = (float*)alloca(COMP_HEAD_DIM * sizeof(float));
    }

    // 1. Project — bounded batch recurrence stays on CPU so its dispatch count
    // and scratch contract remain independent of token count. Scalar callers
    // retain the existing GPU path.
    if (!bounded_scratch && eng->comp_wkv_gpu_w[layer]) {
        memcpy([(id<MTLBuffer>)eng->buf_comp_normed contents], attn_normed, DIM * sizeof(float));
        id<MTLBuffer> kv_buf = (id<MTLBuffer>)eng->comp_out_gpu[layer];
        id<MTLBuffer> gate_buf = (id<MTLBuffer>)eng->comp_out2_gpu[layer];
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBufferWithUnretainedReferences];
        // kv projection
        {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
            [e setBuffer:(id<MTLBuffer>)eng->comp_wkv_gpu_w[layer] offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->comp_wkv_gpu_s[layer] offset:0 atIndex:1];
            [e setBuffer:(id<MTLBuffer>)eng->comp_wkv_gpu_b[layer] offset:0 atIndex:2];
            [e setBuffer:(id<MTLBuffer>)eng->buf_comp_normed offset:0 atIndex:3];
            [e setBuffer:kv_buf offset:0 atIndex:4];
            uint od = eng->comp_wkv[layer].out_dim, id_ = eng->comp_wkv[layer].in_dim, gs = eng->comp_wkv[layer].group_size;
            [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
            [e dispatchThreads:MTLSizeMake(od,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }
        // gate projection (same CB, different output buffer)
        {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
            [e setBuffer:(id<MTLBuffer>)eng->comp_wgate_gpu_w[layer] offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->comp_wgate_gpu_s[layer] offset:0 atIndex:1];
            [e setBuffer:(id<MTLBuffer>)eng->comp_wgate_gpu_b[layer] offset:0 atIndex:2];
            [e setBuffer:(id<MTLBuffer>)eng->buf_comp_normed offset:0 atIndex:3];
            [e setBuffer:gate_buf offset:0 atIndex:4];
            uint od = eng->comp_wgate[layer].out_dim, id_ = eng->comp_wgate[layer].in_dim, gs = eng->comp_wgate[layer].group_size;
            [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
            [e dispatchThreads:MTLSizeMake(od,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }
        [cb commit]; [cb waitUntilCompleted];
        memcpy(kv_cur, [kv_buf contents], out_dim * sizeof(float));
        memcpy(gate_cur, [gate_buf contents], out_dim * sizeof(float));
    } else {
        // Fallback: CPU (when GPU buffers not allocated)
        cpu_affine_matvec_safe(kv_cur,   &eng->comp_wkv[layer],   attn_normed);
        cpu_affine_matvec_safe(gate_cur, &eng->comp_wgate[layer], attn_normed);
    }

    // 2. Add positional bias
    if (eng->comp_ape[layer]) {
        const float *ape_row = eng->comp_ape[layer] + (size_t)pos_mod * out_dim;
        for (uint32_t j = 0; j < out_dim; j++) gate_cur[j] += ape_row[j];
    }

    // 3. Store in rolling state
    memcpy(cs->state_kv    + (size_t)row * out_dim, kv_cur,   out_dim * sizeof(float));
    memcpy(cs->state_score + (size_t)row * out_dim, gate_cur, out_dim * sizeof(float));

    // 4. Check if we should emit a block
    if (((uint32_t)pos + 1) % ratio != 0) return;
    if (cs->n_comp >= MAX_COMP_BLOCKS) return;

    const uint32_t head_dim = COMP_HEAD_DIM; // 512

    // 5 & 6. Per-dimension softmax pooling (matches ds4 compressor_pool_decode_state).
    // For each dimension j: softmax across the ratio token scores, weighted sum of kv values.
    // For CSA (ratio=4): state is [2*ratio, out_dim] where front half [0..ratio) is the
    // overlap buffer and upper half [ratio..2*ratio) is current window.
    // The score gate for each token-slot r is state_score[r * out_dim + j] for HCA,
    // or state_score[(ratio+r) * out_dim + j] for the current CSA window.
    // width = out_dim for HCA; out_dim = 2*head_dim for CSA (front and back halves
    // per token: state_score[r*out_dim+j] is front-half score, +head_dim is back-half).
    for (uint32_t j = 0; j < head_dim; j++) {
        float max_score = -1e30f;
        if (ratio == 4) {
            // CSA: per-slot two half-scores (j and j+head_dim)
            for (uint32_t r = 0; r < ratio; r++) {
                float sp = cs->state_score[(size_t)r * out_dim + j];
                float sc = cs->state_score[(size_t)(ratio + r) * out_dim + head_dim + j];
                if (sp > max_score) max_score = sp;
                if (sc > max_score) max_score = sc;
            }
        } else {
            for (uint32_t r = 0; r < ratio; r++) {
                float s = cs->state_score[(size_t)r * out_dim + j];
                if (s > max_score) max_score = s;
            }
        }
        float denom = 0.0f, sum = 0.0f;
        if (ratio == 4) {
            for (uint32_t r = 0; r < ratio; r++) {
                float wp = expf(cs->state_score[(size_t)r * out_dim + j] - max_score);
                float wc = expf(cs->state_score[(size_t)(ratio + r) * out_dim + head_dim + j] - max_score);
                denom += wp + wc;
                sum += wp * cs->state_kv[(size_t)r * out_dim + j];
                sum += wc * cs->state_kv[(size_t)(ratio + r) * out_dim + head_dim + j];
            }
        } else {
            for (uint32_t r = 0; r < ratio; r++) {
                float w = expf(cs->state_score[(size_t)r * out_dim + j] - max_score);
                denom += w;
                sum += w * cs->state_kv[(size_t)r * out_dim + j];
            }
        }
        pooled[j] = denom > 0.0f ? sum / denom : 0.0f;
    }

    // 7. RMSNorm
    if (eng->comp_norm[layer]) {
        cpu_rms_norm(pooled, eng->comp_norm[layer], pooled, (int)head_dim, 1e-6f);
    }

    // 8. Apply YaRN tail RoPE at compressed position
    const int comp_pos = (int)(((uint32_t)pos + 1) - ratio);
    apply_rope_tail(pooled, (int)head_dim, QK_NOPE_DIM, comp_pos, 16.0f);

    // 9. Append to comp_kv
    memcpy(cs->comp_kv + (size_t)cs->n_comp * head_dim, pooled, head_dim * sizeof(float));
    cs->n_comp++;

    // 10. For CSA: shift overlap buffer
    if (ratio == 4) {
        for (uint32_t t = 0; t < ratio; t++) {
            memcpy(cs->state_kv    + (size_t)t * out_dim,
                   cs->state_kv    + (size_t)(ratio + t) * out_dim,
                   out_dim * sizeof(float));
            memcpy(cs->state_score + (size_t)t * out_dim,
                   cs->state_score + (size_t)(ratio + t) * out_dim,
                   out_dim * sizeof(float));
        }
    }
}

void moe_infer_compressor_step(MoEInferEngine *eng, int layer, int pos,
                                const float *attn_normed) {
    moe_infer_compressor_step_impl(
        eng, layer, pos, attn_normed, NULL, 0u);
}

int moe_infer_compressor_step_bounded(
    MoEInferEngine *eng, int layer, int pos, const float *attn_normed,
    float *scratch, size_t scratch_floats) {
    if (!eng || !attn_normed || !scratch || layer < 0 || layer >= N_LAYERS) {
        return -1;
    }
    const uint32_t ratio = eng->compress_ratio[layer];
    if (ratio == 0u) return 0;
    const uint32_t out_dim = eng->comp_state[layer].state_kv
        ? eng->comp_state[layer].out_dim
        : (ratio == 4u ? CSA_OUT_DIM : HCA_OUT_DIM);
    if (scratch_floats < (size_t)2u * out_dim + COMP_HEAD_DIM) return -1;
    moe_infer_compressor_step_impl(
        eng, layer, pos, attn_normed, scratch, scratch_floats);
    return 0;
}

static bool moe_infer_indexer_step_impl(
                              MoEInferEngine *eng, int layer, int pos,
                              const float *attn_normed, const float *q_a_out,
                              bool *allowed_out, float *bounded_scratch,
                              size_t bounded_scratch_floats) {
    CompressorState *cs = &eng->comp_state[layer];
    const uint32_t n_comp = cs->n_comp;

    if (n_comp == 0) return false;

    uint32_t top_k = 512;
    if (n_comp <= top_k) {
        for (uint32_t c = 0; c < n_comp; c++) allowed_out[c] = true;
        return true;
    }

    // Run indexer's own compressor step (ratio=4, idx_out_dim=256)
    if (eng->idx_comp_wkv[layer].packed != NULL && cs->idx_state_kv != NULL) {
        const uint32_t idx_out_dim = 256;
        const uint32_t idx_pos_mod = (uint32_t)pos % 4;
        const uint32_t idx_row = 4 + idx_pos_mod;

        const size_t idx_scratch_floats =
            (size_t)2u * idx_out_dim + 4u + IDX_HEAD_DIM;
        float *kv_cur;
        float *gate_cur;
        float *weights_arr;
        float *pooled;
        if (bounded_scratch) {
            if (bounded_scratch_floats < idx_scratch_floats) return false;
            kv_cur = bounded_scratch;
            gate_cur = kv_cur + idx_out_dim;
            weights_arr = gate_cur + idx_out_dim;
            pooled = weights_arr + 4u;
        } else {
            kv_cur = (float*)alloca(idx_out_dim * sizeof(float));
            gate_cur = (float*)alloca(idx_out_dim * sizeof(float));
            weights_arr = (float*)alloca(4u * sizeof(float));
            pooled = (float*)alloca(IDX_HEAD_DIM * sizeof(float));
        }

        // Bounded batch recurrence uses CPU projection and caller-owned scratch;
        // scalar callers retain the existing persistent GPU scratch path.
        if (!bounded_scratch && eng->idx_comp_wkv_gpu_w[layer]) {
            memcpy([(id<MTLBuffer>)eng->buf_comp_normed contents], attn_normed, DIM * sizeof(float));
            id<MTLBuffer> kv_buf = (id<MTLBuffer>)eng->idx_comp_out_gpu[layer];
            id<MTLBuffer> gate_buf = (id<MTLBuffer>)eng->idx_comp_out2_gpu[layer];
            id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBufferWithUnretainedReferences];
            // kv projection
            {
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
                [e setBuffer:(id<MTLBuffer>)eng->idx_comp_wkv_gpu_w[layer] offset:0 atIndex:0];
                [e setBuffer:(id<MTLBuffer>)eng->idx_comp_wkv_gpu_s[layer] offset:0 atIndex:1];
                [e setBuffer:(id<MTLBuffer>)eng->idx_comp_wkv_gpu_b[layer] offset:0 atIndex:2];
                [e setBuffer:(id<MTLBuffer>)eng->buf_comp_normed offset:0 atIndex:3];
                [e setBuffer:kv_buf offset:0 atIndex:4];
                uint od = eng->idx_comp_wkv[layer].out_dim, id_ = eng->idx_comp_wkv[layer].in_dim, gs = eng->idx_comp_wkv[layer].group_size;
                [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
                [e dispatchThreads:MTLSizeMake(od,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            }
            // gate projection (same CB, different output buffer)
            {
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
                [e setBuffer:(id<MTLBuffer>)eng->idx_comp_wgate_gpu_w[layer] offset:0 atIndex:0];
                [e setBuffer:(id<MTLBuffer>)eng->idx_comp_wgate_gpu_s[layer] offset:0 atIndex:1];
                [e setBuffer:(id<MTLBuffer>)eng->idx_comp_wgate_gpu_b[layer] offset:0 atIndex:2];
                [e setBuffer:(id<MTLBuffer>)eng->buf_comp_normed offset:0 atIndex:3];
                [e setBuffer:gate_buf offset:0 atIndex:4];
                uint od = eng->idx_comp_wgate[layer].out_dim, id_ = eng->idx_comp_wgate[layer].in_dim, gs = eng->idx_comp_wgate[layer].group_size;
                [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
                [e dispatchThreads:MTLSizeMake(od,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            }
            [cb commit]; [cb waitUntilCompleted];
            memcpy(kv_cur, [kv_buf contents], idx_out_dim * sizeof(float));
            memcpy(gate_cur, [gate_buf contents], idx_out_dim * sizeof(float));
        } else {
            cpu_affine_matvec_safe(kv_cur,   &eng->idx_comp_wkv[layer],   attn_normed);
            cpu_affine_matvec_safe(gate_cur, &eng->idx_comp_wgate[layer], attn_normed);
        }

        if (eng->idx_comp_ape[layer]) {
            const float *ape_row = eng->idx_comp_ape[layer] + (size_t)idx_pos_mod * idx_out_dim;
            for (uint32_t j = 0; j < idx_out_dim; j++) gate_cur[j] += ape_row[j];
        }

        memcpy(cs->idx_state_kv    + (size_t)idx_row * idx_out_dim, kv_cur,   idx_out_dim * sizeof(float));
        memcpy(cs->idx_state_score + (size_t)idx_row * idx_out_dim, gate_cur, idx_out_dim * sizeof(float));

        if (((uint32_t)pos + 1) % 4 == 0 && cs->n_idx_comp < MAX_COMP_BLOCKS) {
            for (int t = 0; t < 4; t++) {
                float s = 0.0f;
                for (uint32_t d = 0; d < IDX_HEAD_DIM; d++)
                    s += cs->idx_state_score[(size_t)t * idx_out_dim + d];
                weights_arr[t] = s;
            }
            cpu_softmax_inplace(weights_arr, 4);
            memset(pooled, 0, IDX_HEAD_DIM * sizeof(float));
            for (int t = 0; t < 4; t++) {
                for (uint32_t d = 0; d < IDX_HEAD_DIM; d++)
                    pooled[d] += weights_arr[t] * cs->idx_state_kv[(size_t)t * idx_out_dim + d];
            }
            if (eng->idx_comp_norm[layer])
                cpu_rms_norm(pooled, eng->idx_comp_norm[layer], pooled, IDX_HEAD_DIM, 1e-6f);
            memcpy(cs->idx_comp_kv + (size_t)cs->n_idx_comp * IDX_HEAD_DIM, pooled, IDX_HEAD_DIM * sizeof(float));
            cs->n_idx_comp++;
            // Shift CSA overlap buffer for indexer
            for (int t = 0; t < 4; t++) {
                memcpy(cs->idx_state_kv    + (size_t)t * idx_out_dim,
                       cs->idx_state_kv    + (size_t)(4+t) * idx_out_dim, idx_out_dim * sizeof(float));
                memcpy(cs->idx_state_score + (size_t)t * idx_out_dim,
                       cs->idx_state_score + (size_t)(4+t) * idx_out_dim, idx_out_dim * sizeof(float));
            }
        }
    }

    const uint32_t n_idx = cs->n_idx_comp;
    if (n_idx == 0 || q_a_out == NULL) {
        for (uint32_t c = 0; c < n_comp; c++) allowed_out[c] = true;
        return true;
    }

    // Score each idx_comp block using q
    const int n_idx_heads = 64;
    const int idx_head_dim = 128;
    float *q = (float*)alloca((size_t)n_idx_heads * idx_head_dim * sizeof(float));
    cpu_affine_matvec_safe(q, &eng->idx_wq_b[layer], q_a_out);

    float *head_w = (float*)alloca((size_t)n_idx_heads * sizeof(float));
    cpu_affine_matvec_safe(head_w, &eng->idx_weights_proj[layer], attn_normed);
    float scale = 1.0f / sqrtf((float)(n_idx_heads * idx_head_dim));
    for (int h = 0; h < n_idx_heads; h++) head_w[h] *= scale;

    float *scores = (float*)calloc(n_idx, sizeof(float));
    for (uint32_t c = 0; c < n_idx; c++) {
        const float *kv_c = cs->idx_comp_kv + (size_t)c * idx_head_dim;
        float s = 0.0f;
        for (int h = 0; h < n_idx_heads; h++) {
            const float *q_h = q + (size_t)h * idx_head_dim;
            float dot = 0.0f;
            for (int d = 0; d < idx_head_dim; d++) dot += q_h[d] * kv_c[d];
            if (dot > 0.0f) s += dot * head_w[h];
        }
        scores[c] = s;
    }

    memset(allowed_out, 0, n_comp * sizeof(bool));
    uint32_t actual_k = (top_k < n_idx) ? top_k : n_idx;
    for (uint32_t k = 0; k < actual_k; k++) {
        uint32_t best = 0;
        float best_score = -1e30f;
        for (uint32_t c = 0; c < n_idx; c++) {
            if (!allowed_out[c] && scores[c] > best_score) {
                best_score = scores[c];
                best = c;
            }
        }
        if (best < n_comp) allowed_out[best] = true;
    }
    free(scores);
    return true;
}

bool moe_infer_indexer_step(MoEInferEngine *eng, int layer, int pos,
                            const float *attn_normed, const float *q_a_out,
                            bool *allowed_out) {
    return moe_infer_indexer_step_impl(
        eng, layer, pos, attn_normed, q_a_out, allowed_out, NULL, 0u);
}

int moe_infer_indexer_step_bounded(
    MoEInferEngine *eng, int layer, int pos, const float *attn_normed,
    bool *allowed_out, float *scratch, size_t scratch_floats) {
    const size_t required = (size_t)2u * 256u + 4u + IDX_HEAD_DIM;
    if (!eng || !attn_normed || !allowed_out || !scratch ||
        layer < 0 || layer >= N_LAYERS || scratch_floats < required) {
        return -1;
    }
    (void)moe_infer_indexer_step_impl(
        eng, layer, pos, attn_normed, NULL, allowed_out,
        scratch, scratch_floats);
    return 0;
}

void moe_infer_deinit(MoEInferEngine *eng) {
    if (!eng || !eng->initialized) return;
    // No producer owner may outlive its I/O generation. Drain both pools before
    // aborting unclaimed records or freeing producer storage.
    prefetch_drain(eng);
    if (eng->io_pool) io_pool_wait((IOPool *)eng->io_pool);
    if (eng->exact_q4_storage) {
        while (atomic_load_explicit(
                   &eng->exact_q4_storage->active_read_publishers,
                   memory_order_acquire) != 0u) {
            for (uint32_t index = 0u;
                 index < eng->exact_q4_storage->read_count; ++index) {
                (void)moe_infer_exact_q4_abort_read(eng, index);
            }
            sched_yield();
        }
    }
    release_wave4_bucket_pipelines(eng);
    release_shared_expert_weight_buffers(eng);
    // Release engine-lifetime buffers hoisted from per-layer-per-token allocations.
    for (int l = 0; l < N_LAYERS; ++l) {
        if (eng->buf_weights[l])   { [(id<MTLBuffer>)eng->buf_weights[l]   release]; eng->buf_weights[l]   = NULL; }
        if (eng->buf_zero_resid[l]){ [(id<MTLBuffer>)eng->buf_zero_resid[l] release]; eng->buf_zero_resid[l] = NULL; }
    }
    if (eng->buf_combine_one) { [(id<MTLBuffer>)eng->buf_combine_one release]; eng->buf_combine_one = NULL; }
    // Release persistent NoCopy blob wrappers (were leaked before this fix).
    for (int k = 0; k < N_ACTIVE; ++k) {
        if (eng->blob_wrap_pread[k])  { [(id<MTLBuffer>)eng->blob_wrap_pread[k]  release]; eng->blob_wrap_pread[k]  = NULL; }
        if (eng->blob_wrap_pred[k])   { [(id<MTLBuffer>)eng->blob_wrap_pred[k]   release]; eng->blob_wrap_pred[k]   = NULL; }
        if (eng->blob_wrap_pred2[k])  { [(id<MTLBuffer>)eng->blob_wrap_pred2[k]  release]; eng->blob_wrap_pred2[k]  = NULL; }
        eng->blob_wrap_pread_ptr[k] = eng->blob_wrap_pred_ptr[k] = eng->blob_wrap_pred2_ptr[k] = NULL;
    }
    release_persistent_routing_buffers(eng);
    routing_trace_writer_destroy(eng->routing_trace);
    eng->routing_trace = NULL;
    IOPool *io = (IOPool *)eng->io_pool;
    if (io) {
        pthread_mutex_lock(&io->mutex);
        io->shutdown = true;
        pthread_cond_broadcast(&io->work_ready);
        pthread_mutex_unlock(&io->mutex);
        for (int i = 0; i < NUM_IO_THREADS; i++) {
            pthread_join(io->threads[i], NULL);
        }
        free(io);
    }
    IOPool *pfp = (IOPool *)eng->prefetch_pool;
    if (pfp) {
        pthread_mutex_lock(&pfp->mutex);
        pfp->shutdown = true;
        pthread_cond_broadcast(&pfp->work_ready);
        pthread_mutex_unlock(&pfp->mutex);
        for (int i = 0; i < NUM_IO_THREADS; i++) {
            pthread_join(pfp->threads[i], NULL);
        }
        free(pfp);
    }
    for (int l = 0; l < N_LAYERS; l++) {
        if (eng->packed_fd[l] >= 0) close(eng->packed_fd[l]);
    }
    for (int k = 0; k < N_ACTIVE; k++) {
        free(eng->expert_buf[k]);
        free(eng->expert_buf_pred[k]);
        free(eng->expert_buf_pred2[k]);
    }
    // Free cross-token batch dedup pool (wrappers are NoCopy views — release
    // the MTLBuffer objects but never the backing memory they point at).
    if (eng->batch_wrap) {
        for (int s = 0; s < eng->batch_capacity; s++) {
            if (eng->batch_wrap[s]) [(id<MTLBuffer>)eng->batch_wrap[s] release];
        }
        free(eng->batch_wrap);
    }
    free(eng->batch_eids);
    free(eng->batch_pins);
    free(eng->batch_pool);
    if (eng->forward_batch_workspace) {
        NativeBatchWorkspace *workspace =
            (NativeBatchWorkspace *)eng->forward_batch_workspace;
        moe_infer_batch_workspace_deinit(workspace);
        free(workspace);
        eng->forward_batch_workspace = NULL;
    }
    // Free expert memory cache
    if (eng->expert_cache_n_experts > 0) {
        for (int layer = 0; layer < N_LAYERS; layer++) {
            if (eng->expert_mem_cache[layer]) free(eng->expert_mem_cache[layer]);
            if (eng->expert_mem_pool[layer]) free(eng->expert_mem_pool[layer]);
        }
    }
    // Free compressor state
    for (int l = 0; l < N_LAYERS; l++) {
        CompressorState *cs = &eng->comp_state[l];
        free(cs->state_kv);
        free(cs->state_score);
        free(cs->comp_kv);
        free(cs->idx_state_kv);
        free(cs->idx_state_score);
        free(cs->idx_comp_kv);
    }
    moe_infer_attention_stage_deconfigure(eng);
    (void)moe_infer_gpu_timestamp_probe_deconfigure(eng);
    (void)moe_infer_exact_q4_deconfigure(eng);
    (void)moe_infer_decode_timeline_deconfigure(eng);
    free(eng);
}

// ============================================================================
// DSpark INT8 MoE GPU dispatch (called from dspark_engine.c via C linkage)
// ============================================================================
// Performs: gate_up_swiglu + down_proj for K=6 MXFP4 experts on GPU.
// Input: normed[DIM] in buf_normed, expert data pointers, expert weights.
// Output: combined MoE output in out_buf[DIM].
//
// Uses the SAME MXFP4 kernels as target model (pipe_gate_up_swiglu_v2 + pipe_dequant_matvec).

void dspark_moe_forward_gpu(
    MoEInferEngine *eng,
    const float *normed_input,       // [DIM] — will be copied to buf_normed
    uint8_t *expert_ptrs[6],         // [K] pointers to preloaded expert data
    int *expert_ids,                 // [K] (for logging only)
    float *expert_weights,           // [K] normalized routing weights
    int K,
    float *moe_out                   // [DIM] output
) {
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    id<MTLCommandQueue> q = (id<MTLCommandQueue>)eng->queue;

    // Copy input to buf_normed
    float *bn = (float *)[(id<MTLBuffer>)eng->buf_normed contents];
    memcpy(bn, normed_input, DIM * sizeof(float));

    // MXFP4 expert layout (same as target: dim=4096, inter=2048, gs=32)
    const uint out_dim_gu = INTERMEDIATE;  // 2048
    const uint in_dim_gu  = DIM;           // 4096
    const uint out_dim_d  = DIM;           // 4096
    const uint in_dim_d   = INTERMEDIATE;  // 2048
    const uint gs = MOE_GROUP_SIZE;        // 32

    // Byte sizes for MXFP4 packed weights (2 nibbles per byte)
    const size_t gu_w_bytes = (size_t)out_dim_gu * in_dim_gu / 2;  // 4194304
    const size_t gu_s_bytes = (size_t)out_dim_gu * (in_dim_gu / gs); // 262144
    const size_t dn_w_bytes = (size_t)out_dim_d * in_dim_d / 2;     // 4194304
    const size_t dn_s_bytes = (size_t)out_dim_d * (in_dim_d / gs);   // 262144
    // Offsets within each expert: w1.w, w1.s, w3.w, w3.s, w2.w, w2.s
    const size_t w1_w_off = 0;
    const size_t w1_s_off = gu_w_bytes;
    const size_t w3_w_off = gu_w_bytes + gu_s_bytes;
    const size_t w3_s_off = w3_w_off + gu_w_bytes;
    const size_t w2_w_off = w3_s_off + gu_s_bytes;
    const size_t w2_s_off = w2_w_off + dn_w_bytes;

    id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];

    // Step 1: gate_up_swiglu for each expert (MXFP4 kernel — same as target)
    for (int k = 0; k < K; k++) {
        uint8_t *base = expert_ptrs[k];
        if (!base) continue;

        id<MTLBuffer> gate_w = [d newBufferWithBytesNoCopy:base+w1_w_off length:gu_w_bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> gate_s = [d newBufferWithBytesNoCopy:base+w1_s_off length:gu_s_bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> up_w   = [d newBufferWithBytesNoCopy:base+w3_w_off length:gu_w_bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> up_s   = [d newBufferWithBytesNoCopy:base+w3_s_off length:gu_s_bytes options:MTLResourceStorageModeShared deallocator:nil];

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_gate_up_swiglu_v2];
        [enc setBuffer:gate_w offset:0 atIndex:0];
        [enc setBuffer:gate_s offset:0 atIndex:1];
        [enc setBuffer:up_w   offset:0 atIndex:2];
        [enc setBuffer:up_s   offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)eng->buf_normed offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)eng->buf_expert_mid[k] offset:0 atIndex:5];
        uint od=out_dim_gu, id_=in_dim_gu, g=gs;
        [enc setBytes:&od  length:4 atIndex:6];
        [enc setBytes:&id_ length:4 atIndex:7];
        [enc setBytes:&g   length:4 atIndex:8];
        [enc setThreadgroupMemoryLength:512 atIndex:0];
        uint ntg = (out_dim_gu + 1) / 2;
        [enc dispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
        [enc endEncoding];
    }

    // Step 2: down_proj for each expert (MXFP4 dequant matvec — same as target)
    for (int k = 0; k < K; k++) {
        uint8_t *base = expert_ptrs[k];
        if (!base) continue;

        id<MTLBuffer> dn_w = [d newBufferWithBytesNoCopy:base+w2_w_off length:dn_w_bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> dn_s = [d newBufferWithBytesNoCopy:base+w2_s_off length:dn_s_bytes options:MTLResourceStorageModeShared deallocator:nil];

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec];
        [enc setBuffer:dn_w offset:0 atIndex:0];
        [enc setBuffer:dn_s offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)eng->buf_expert_mid[k] offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)eng->buf_expert_out[k] offset:0 atIndex:3];
        uint od=out_dim_d, id_=in_dim_d, g=gs;
        [enc setBytes:&od  length:4 atIndex:4];
        [enc setBytes:&id_ length:4 atIndex:5];
        [enc setBytes:&g   length:4 atIndex:6];
        [enc setThreadgroupMemoryLength:256 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(DIM/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }

    // Step 3: combine (blit to contiguous + weighted sum)
    {
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        for (int k = 0; k < K; k++) {
            [blit copyFromBuffer:(id<MTLBuffer>)eng->buf_expert_out[k]
                   sourceOffset:0
                       toBuffer:(id<MTLBuffer>)eng->buf_expert_contiguous
              destinationOffset:(size_t)k * DIM * sizeof(float)
                           size:(size_t)DIM * sizeof(float)];
        }
        [blit endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_moe_combine];
        id<MTLBuffer> weights_buf = [d newBufferWithBytes:expert_weights length:K*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> zero_resid  = [d newBufferWithLength:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        [enc setBuffer:(id<MTLBuffer>)eng->buf_expert_contiguous offset:0 atIndex:0];
        [enc setBuffer:weights_buf offset:0 atIndex:1];
        [enc setBuffer:zero_resid  offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)eng->buf_hidden offset:0 atIndex:3];
        uint kv = K, hd = DIM;
        [enc setBytes:&kv length:4 atIndex:4];
        [enc setBytes:&hd length:4 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((DIM+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }

    [cb commit];
    [cb waitUntilCompleted];

    // Copy result from buf_hidden
    float *result = (float *)[(id<MTLBuffer>)eng->buf_hidden contents];
    memcpy(moe_out, result, DIM * sizeof(float));
}

// ============================================================================
// DSpark GPU-accelerated shared expert (f32 weights, matvec_f32 kernel)
// ============================================================================
// Computes: out = silu(w1 @ x) * (w3 @ x), then down = w2 @ mid
// All weights are pre-dequantized f32. Uses pipe_matvec for GPU dispatch.

void dspark_shared_expert_gpu(
    MoEInferEngine *eng,
    const float *input,         // [DIM=4096]
    const float *w1_f32,        // [INTERMEDIATE*DIM] = [2048*4096] gate
    const float *w3_f32,        // [INTERMEDIATE*DIM] up
    const float *w2_f32,        // [DIM*INTERMEDIATE] down
    float *output               // [DIM=4096]
) {
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    id<MTLCommandQueue> q = (id<MTLCommandQueue>)eng->queue;

    // Create GPU buffers (copy data — malloc'd f32 arrays aren't page-aligned)
    id<MTLBuffer> x_buf  = [d newBufferWithBytes:(void*)input length:DIM*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> w1_buf = [d newBufferWithBytes:(void*)w1_f32 length:(size_t)INTERMEDIATE*DIM*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> w3_buf = [d newBufferWithBytes:(void*)w3_f32 length:(size_t)INTERMEDIATE*DIM*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> w2_buf = [d newBufferWithBytes:(void*)w2_f32 length:(size_t)DIM*INTERMEDIATE*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate_buf = [d newBufferWithLength:INTERMEDIATE*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> up_buf   = [d newBufferWithLength:INTERMEDIATE*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> mid_buf  = [d newBufferWithLength:INTERMEDIATE*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf  = [d newBufferWithLength:DIM*4 options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
    uint od, id_;

    // gate = w1 @ x
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_matvec];
        [enc setBuffer:w1_buf offset:0 atIndex:0];
        [enc setBuffer:x_buf  offset:0 atIndex:1];
        [enc setBuffer:gate_buf offset:0 atIndex:2];
        od = INTERMEDIATE; id_ = DIM;
        [enc setBytes:&od  length:4 atIndex:3];
        [enc setBytes:&id_ length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((INTERMEDIATE+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }
    // up = w3 @ x
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_matvec];
        [enc setBuffer:w3_buf offset:0 atIndex:0];
        [enc setBuffer:x_buf  offset:0 atIndex:1];
        [enc setBuffer:up_buf offset:0 atIndex:2];
        od = INTERMEDIATE; id_ = DIM;
        [enc setBytes:&od  length:4 atIndex:3];
        [enc setBytes:&id_ length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((INTERMEDIATE+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }

    [cb commit];
    [cb waitUntilCompleted];

    // SwiGLU on CPU (small: 2048 elements)
    float *gate = (float*)[gate_buf contents];
    float *up = (float*)[up_buf contents];
    float *mid = (float*)[mid_buf contents];
    for (int i = 0; i < INTERMEDIATE; i++) {
        float g = gate[i];
        mid[i] = (g / (1.0f + expf(-g))) * up[i];
    }

    // down = w2 @ mid (GPU)
    id<MTLCommandBuffer> cb2 = [q commandBufferWithUnretainedReferences];
    {
        id<MTLComputeCommandEncoder> enc = [cb2 computeCommandEncoder];
        [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_matvec];
        [enc setBuffer:w2_buf offset:0 atIndex:0];
        [enc setBuffer:mid_buf offset:0 atIndex:1];
        [enc setBuffer:out_buf offset:0 atIndex:2];
        od = DIM; id_ = INTERMEDIATE;
        [enc setBytes:&od  length:4 atIndex:3];
        [enc setBytes:&id_ length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((DIM+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }
    [cb2 commit];
    [cb2 waitUntilCompleted];

    memcpy(output, [out_buf contents], DIM * sizeof(float));
}

// ============================================================================
// DSpark GPU f32 matvec (generic, for attention projections)
// ============================================================================
// out = W @ x, where W is [out_dim, in_dim] f32, x is [in_dim] f32.
// Uses pipe_matvec (1-thread-per-row). Fast for large out_dim (e.g. wq_b 32768).

void dspark_matvec_f32_gpu(
    MoEInferEngine *eng,
    const float *W,        // [out_dim * in_dim] f32
    const float *x,        // [in_dim] f32
    float *out,            // [out_dim] f32
    int out_dim,
    int in_dim
) {
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    id<MTLCommandQueue> q = (id<MTLCommandQueue>)eng->queue;

    id<MTLBuffer> w_buf = [d newBufferWithBytes:(void*)W length:(size_t)out_dim*in_dim*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> x_buf = [d newBufferWithBytes:(void*)x length:(size_t)in_dim*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> o_buf = [d newBufferWithLength:(size_t)out_dim*4 options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_matvec];
        [enc setBuffer:w_buf offset:0 atIndex:0];
        [enc setBuffer:x_buf offset:0 atIndex:1];
        [enc setBuffer:o_buf offset:0 atIndex:2];
        uint od = (uint)out_dim, id_ = (uint)in_dim;
        [enc setBytes:&od  length:4 atIndex:3];
        [enc setBytes:&id_ length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(((uint)out_dim+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];

    memcpy(out, [o_buf contents], (size_t)out_dim * sizeof(float));
}

// ============================================================================
// DSpark batched MoE: all positions in ONE command buffer
// ============================================================================
// Processes n_pos positions, each with K=6 experts. Single commit+wait.
// This eliminates the 15× GPU sync overhead (was: 1 commit per position per layer).

void dspark_moe_forward_batched_gpu(
    MoEInferEngine *eng,
    int n_pos,                        // number of positions (typically 5)
    const float *normed_inputs,       // [n_pos * DIM] — all normed inputs concatenated
    uint8_t **all_expert_ptrs,        // [n_pos * K] expert data pointers
    float *all_expert_weights,        // [n_pos * K] routing weights
    float *all_moe_out                // [n_pos * DIM] output
) {
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    id<MTLCommandQueue> q = (id<MTLCommandQueue>)eng->queue;
    const int K = N_ACTIVE;

    // MXFP4 layout constants
    const uint out_dim_gu = INTERMEDIATE;
    const uint in_dim_gu  = DIM;
    const uint out_dim_d  = DIM;
    const uint in_dim_d   = INTERMEDIATE;
    const uint gs = MOE_GROUP_SIZE;
    const size_t gu_w_bytes = (size_t)out_dim_gu * in_dim_gu / 2;
    const size_t gu_s_bytes = (size_t)out_dim_gu * (in_dim_gu / gs);
    const size_t dn_w_bytes = (size_t)out_dim_d * in_dim_d / 2;
    const size_t dn_s_bytes = (size_t)out_dim_d * (in_dim_d / gs);
    const size_t w1_w_off = 0;
    const size_t w1_s_off = gu_w_bytes;
    const size_t w3_w_off = gu_w_bytes + gu_s_bytes;
    const size_t w3_s_off = w3_w_off + gu_w_bytes;
    const size_t w2_w_off = w3_s_off + gu_s_bytes;

    // Allocate per-position GPU buffers for input/mid/out
    id<MTLBuffer> pos_normed[5];
    id<MTLBuffer> pos_mid[5][6];
    id<MTLBuffer> pos_out[5][6];
    for (int p = 0; p < n_pos; p++) {
        pos_normed[p] = [d newBufferWithBytes:(void*)(normed_inputs + p * DIM) length:DIM*4 options:MTLResourceStorageModeShared];
        for (int k = 0; k < K; k++) {
            pos_mid[p][k] = [d newBufferWithLength:INTERMEDIATE*4 options:MTLResourceStorageModeShared];
            pos_out[p][k] = [d newBufferWithLength:DIM*4 options:MTLResourceStorageModeShared];
        }
    }

    id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];

    // Encode ALL gate_up_swiglu dispatches
    for (int p = 0; p < n_pos; p++) {
        for (int k = 0; k < K; k++) {
            uint8_t *base = all_expert_ptrs[p * K + k];
            if (!base) continue;
            id<MTLBuffer> gw = [d newBufferWithBytesNoCopy:base+w1_w_off length:gu_w_bytes options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLBuffer> gs_b = [d newBufferWithBytesNoCopy:base+w1_s_off length:gu_s_bytes options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLBuffer> uw = [d newBufferWithBytesNoCopy:base+w3_w_off length:gu_w_bytes options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLBuffer> us = [d newBufferWithBytesNoCopy:base+w3_s_off length:gu_s_bytes options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_gate_up_swiglu_v2];
            [enc setBuffer:gw offset:0 atIndex:0];
            [enc setBuffer:gs_b offset:0 atIndex:1];
            [enc setBuffer:uw offset:0 atIndex:2];
            [enc setBuffer:us offset:0 atIndex:3];
            [enc setBuffer:pos_normed[p] offset:0 atIndex:4];
            [enc setBuffer:pos_mid[p][k] offset:0 atIndex:5];
            uint od=out_dim_gu, id_v=in_dim_gu, g=gs;
            [enc setBytes:&od length:4 atIndex:6];
            [enc setBytes:&id_v length:4 atIndex:7];
            [enc setBytes:&g length:4 atIndex:8];
            [enc setThreadgroupMemoryLength:512 atIndex:0];
            uint ntg = (out_dim_gu + 1) / 2;
            [enc dispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
            [enc endEncoding];
        }
    }

    // Encode ALL down_proj dispatches
    for (int p = 0; p < n_pos; p++) {
        for (int k = 0; k < K; k++) {
            uint8_t *base = all_expert_ptrs[p * K + k];
            if (!base) continue;
            id<MTLBuffer> dw = [d newBufferWithBytesNoCopy:base+w2_w_off length:dn_w_bytes options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLBuffer> ds = [d newBufferWithBytesNoCopy:base+w2_w_off+dn_w_bytes length:dn_s_bytes options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec];
            [enc setBuffer:dw offset:0 atIndex:0];
            [enc setBuffer:ds offset:0 atIndex:1];
            [enc setBuffer:pos_mid[p][k] offset:0 atIndex:2];
            [enc setBuffer:pos_out[p][k] offset:0 atIndex:3];
            uint od=out_dim_d, id_v=in_dim_d, g=gs;
            [enc setBytes:&od length:4 atIndex:4];
            [enc setBytes:&id_v length:4 atIndex:5];
            [enc setBytes:&g length:4 atIndex:6];
            [enc setThreadgroupMemoryLength:256 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(DIM/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
        }
    }

    // Single commit + wait for ALL positions
    [cb commit];
    [cb waitUntilCompleted];

    // Combine on CPU (weighted sum of expert outputs per position)
    for (int p = 0; p < n_pos; p++) {
        float *moe_out = all_moe_out + p * DIM;
        memset(moe_out, 0, DIM * sizeof(float));
        for (int k = 0; k < K; k++) {
            float w = all_expert_weights[p * K + k];
            float *expert_out = (float *)[pos_out[p][k] contents];
            for (int d = 0; d < DIM; d++) moe_out[d] += w * expert_out[d];
        }
    }
}

#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
int moe_infer_exact_q4_test_run_layer_transaction_suite(void) {
    MoEInferEngine engine = {0};
    engine.initialized = true;
    if (moe_infer_exact_q4_configure(&engine, 8u, 1u) != 0 ||
        moe_infer_exact_q4_enable(&engine, true) != 0 ||
        moe_infer_exact_q4_begin_request(&engine, 4u) != 0 ||
        moe_infer_exact_q4_begin_token(&engine, 2u) != 0)
        return __LINE__;
    const DmlxMlaPathIdentity identity = {
        .path = DMLX_MLA_PATH_PERSISTENT_INTERNAL,
        .flags = DMLX_MLA_PATH_PERSISTENT_KV,
    };
    const DmlxExactQ4StageEvent blocked = {
        .layer = 7u,
        .stage = DMLX_EXACT_Q4_STAGE_IO_EXPOSED_WAIT,
        .flags = DMLX_EXACT_Q4_EVENT_JOIN_BLOCKED,
        .calls = 1u,
        .begin_ns = 10u,
        .end_ns = 20u,
    };
    DmlxExactQ4PendingLayer transaction = {0};
    exact_q4_layer_begin(&engine, &transaction, 7);
    exact_q4_layer_set_path(&transaction, &identity);
    if (!exact_q4_layer_add_stage(&transaction, &blocked)) return __LINE__;
    exact_q4_layer_discard(&transaction);
    DmlxExactQ4LatencySnapshot snapshot = {0};
    DmlxExactQ4StageEvent stages[8] = {0};
    DmlxExactQ4ExpertReadEvent reads[1] = {0};
    if (moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 8u, reads, 1u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE || snapshot.stage_count != 0u)
        return __LINE__;

    exact_q4_layer_begin(&engine, &transaction, 7);
    exact_q4_layer_set_path(&transaction, &identity);
    if (!exact_q4_layer_add_stage(&transaction, &blocked) ||
        exact_q4_layer_commit(&engine, &transaction) != 0)
        return __LINE__;
    if (moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 8u, reads, 1u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE || snapshot.stage_count != 1u ||
        stages[0].stage != DMLX_EXACT_Q4_STAGE_IO_EXPOSED_WAIT)
        return __LINE__;

    const DmlxExactQ4StageEvent counters[2] = {
        { .layer = 8u, .stage = DMLX_EXACT_Q4_STAGE_CACHE_ADMISSION,
          .flags = DMLX_EXACT_Q4_EVENT_CPU_ACTIVE, .calls = 2u,
          .begin_ns = 21u, .end_ns = 24u, .bytes = 4096u },
        { .layer = 8u, .stage = DMLX_EXACT_Q4_STAGE_HOST_ALLOC_WRAPPER,
          .flags = DMLX_EXACT_Q4_EVENT_CPU_ACTIVE, .calls = 3u,
          .begin_ns = 24u, .end_ns = 28u, .bytes = 8192u },
    };
    exact_q4_layer_begin(&engine, &transaction, 8);
    exact_q4_layer_set_path(&transaction, &identity);
    if (!exact_q4_layer_add_stage(&transaction, &counters[0]) ||
        !exact_q4_layer_add_stage(&transaction, &counters[1]) ||
        exact_q4_layer_commit(&engine, &transaction) != 0 ||
        moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 8u, reads, 1u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE ||
        snapshot.stage_count != 3u || stages[1].calls != 2u ||
        stages[1].bytes != 4096u || stages[2].calls != 3u ||
        stages[2].bytes != 8192u)
        return __LINE__;

    DmlxExactQ4StageEvent overflow_stage = blocked;
    overflow_stage.layer = 8u;
    exact_q4_layer_begin(&engine, &transaction, 8);
    exact_q4_layer_set_path(&transaction, &identity);
    for (uint32_t index = 0u;
         index < DMLX_EXACT_Q4_LAYER_STAGE_CAPACITY; ++index) {
        if (!exact_q4_layer_add_stage(&transaction, &overflow_stage))
            return __LINE__;
    }
    if (exact_q4_layer_add_stage(&transaction, &overflow_stage) ||
        !transaction.active || !transaction.failed ||
        exact_q4_layer_commit(&engine, &transaction) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE ||
        transaction.active)
        return __LINE__;
    if (moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 8u, reads, 1u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE ||
        snapshot.stage_count != 3u || !snapshot.incomplete)
        return __LINE__;
    if (moe_infer_exact_q4_abort_request(&engine) != 0 ||
        moe_infer_exact_q4_reset(&engine) != 0 ||
        moe_infer_exact_q4_begin_request(&engine, 5u) != 0 ||
        moe_infer_exact_q4_begin_token(&engine, 0u) != 0)
        return __LINE__;

    DmlxExactQ4StageEvent pending_stage = blocked;
    pending_stage.layer = 8u;
    exact_q4_layer_begin(&engine, &engine.deferred.exact_q4_pending, 8);
    exact_q4_layer_set_path(&engine.deferred.exact_q4_pending, &identity);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (!device || !queue || !command || !exact_q4_layer_add_stage(
            &engine.deferred.exact_q4_pending, &pending_stage))
        return __LINE__;
    [command retain];
    [command commit];
    engine.deferred.active = true;
    engine.deferred.cmd_experts = command;
    if (moe_infer_exact_q4_drain_and_abort_request(&engine) != 0 ||
        !moe_infer_exact_q4_test_no_inflight_owners(&engine))
        return __LINE__;

    if (moe_infer_exact_q4_reset(&engine) != 0 ||
        moe_infer_exact_q4_begin_request(&engine, 6u) != 0 ||
        moe_infer_exact_q4_begin_token(&engine, 0u) != 0)
        return __LINE__;
    exact_q4_layer_begin(&engine, &engine.deferred.exact_q4_pending, 9);
    engine.deferred.exact_q4_pending.failed = true;
    command = [queue commandBuffer];
    if (!command) return __LINE__;
    [command retain];
    [command commit];
    engine.deferred.active = true;
    engine.deferred.cmd_experts = command;
    if (moe_infer_exact_q4_drain_and_abort_request(&engine) != 0 ||
        !moe_infer_exact_q4_test_no_inflight_owners(&engine))
        return __LINE__;
    [queue release];
    if (moe_infer_exact_q4_reset(&engine) != 0 ||
        moe_infer_exact_q4_deconfigure(&engine) != 0)
        return __LINE__;
    return 0;
}
#endif

#ifdef DMLX_EXACT_Q4_IO_TESTING

typedef struct ExactQ4IoTestReader {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int release_order[NUM_IO_THREADS];
    int completion_order[NUM_IO_THREADS];
    ssize_t results[N_EXPERTS];
    int release_count;
    int next_release;
    int completion_count;
    bool hold;
    bool entered;
} ExactQ4IoTestReader;

typedef struct ExactQ4IoTestThread {
    MoEInferEngine *engine;
    ExactQ4IoTestReader *reader;
    IOPool *pool;
    int *ids;
    int *ranks;
    uint8_t **buffers;
    size_t expert_size;
    int layer;
    int task_count;
    int dispatch_status;
    uint64_t bytes_read;
    uint64_t short_reads;
    DmlxExactQ4PendingLayer *transaction;
    _Atomic bool done;
} ExactQ4IoTestThread;

static uint64_t exact_q4_io_test_clock(void *context) {
    _Atomic uint64_t *clock = (_Atomic uint64_t *)context;
    return atomic_fetch_add_explicit(clock, 10u, memory_order_relaxed);
}

static int64_t exact_q4_io_test_reader(
    void *context, int fd, void *buffer, uint64_t size, uint64_t offset) {
    (void)fd;
    ExactQ4IoTestReader *reader = (ExactQ4IoTestReader *)context;
    const int expert = size == 0u ? 0 : (int)(offset / size);
    pthread_mutex_lock(&reader->mutex);
    reader->entered = true;
    pthread_cond_broadcast(&reader->condition);
    while (reader->hold ||
           (reader->next_release < reader->release_count &&
            reader->release_order[reader->next_release] != expert)) {
        pthread_cond_wait(&reader->condition, &reader->mutex);
    }
    if (reader->next_release < reader->release_count) {
        reader->next_release++;
        pthread_cond_broadcast(&reader->condition);
    }
    reader->completion_order[reader->completion_count++] = expert;
    const ssize_t result = reader->results[expert];
    pthread_mutex_unlock(&reader->mutex);
    if (result < 0) errno = EIO;
    if (result > 0 && buffer) memset(buffer, expert, (size_t)result);
    return result;
}

static void exact_q4_io_test_reader_init(
    ExactQ4IoTestReader *reader, size_t full_size) {
    memset(reader, 0, sizeof(*reader));
    pthread_mutex_init(&reader->mutex, NULL);
    pthread_cond_init(&reader->condition, NULL);
    for (int expert = 0; expert < N_EXPERTS; ++expert) {
        reader->results[expert] = (ssize_t)full_size;
    }
}

static void exact_q4_io_test_reader_deinit(ExactQ4IoTestReader *reader) {
    pthread_cond_destroy(&reader->condition);
    pthread_mutex_destroy(&reader->mutex);
}

static void exact_q4_io_test_pool_deinit(IOPool *pool) {
    io_pool_wait(pool);
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);
    for (int index = 0; index < NUM_IO_THREADS; ++index) {
        pthread_join(pool->threads[index], NULL);
    }
    pthread_cond_destroy(&pool->work_done);
    pthread_cond_destroy(&pool->work_ready);
    pthread_mutex_destroy(&pool->mutex);
}

static int exact_q4_io_test_begin(
    MoEInferEngine *engine, _Atomic uint64_t *clock,
    uint32_t stage_capacity, uint32_t read_capacity) {
    engine->initialized = true;
    engine->prefetch_layer = -1;
    if (moe_infer_exact_q4_configure(
            engine, stage_capacity, read_capacity) != DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;
    if (moe_infer_exact_q4_enable(engine, true) != DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;
    moe_infer_exact_q4_test_set_clock(engine, exact_q4_io_test_clock, clock);
    if (moe_infer_exact_q4_begin_request(engine, 7u) !=
            DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;
    if (moe_infer_exact_q4_begin_token(engine, 11u) !=
            DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;
    return 0;
}

static int exact_q4_io_test_finish(MoEInferEngine *engine) {
    if (moe_infer_exact_q4_finalize(engine) != DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;
    if (moe_infer_exact_q4_enable(engine, false) != DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;
    if (moe_infer_exact_q4_deconfigure(engine) != DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;
    return 0;
}

static void *exact_q4_io_test_dispatch(void *context) {
    ExactQ4IoTestThread *thread = (ExactQ4IoTestThread *)context;
    thread->dispatch_status = io_pool_dispatch(
        thread->pool, thread->engine, 0, thread->ids, thread->ranks,
        thread->layer, thread->task_count, thread->buffers,
        thread->expert_size, &thread->bytes_read, &thread->short_reads,
        thread->transaction);
    atomic_store_explicit(&thread->done, true, memory_order_release);
    return NULL;
}

static void *exact_q4_io_test_release(void *context) {
    ExactQ4IoTestThread *thread = (ExactQ4IoTestThread *)context;
    usleep(20000);
    pthread_mutex_lock(&thread->reader->mutex);
    thread->reader->hold = false;
    pthread_cond_broadcast(&thread->reader->condition);
    pthread_mutex_unlock(&thread->reader->mutex);
    return NULL;
}

static void *exact_q4_io_test_drain(void *context) {
    ExactQ4IoTestThread *thread = (ExactQ4IoTestThread *)context;
    moe_infer_reset_kv(thread->engine);
    atomic_store_explicit(&thread->done, true, memory_order_release);
    return NULL;
}

static int exact_q4_io_test_source_and_results(void) {
    enum { TEST_SIZE = 16 };
    MoEInferEngine engine = {0};
    _Atomic uint64_t clock = 100u;
    int status = exact_q4_io_test_begin(&engine, &clock, 8u, 32u);
    if (status != 0) return status;

    IOPool pool;
    io_pool_init(&pool);
    engine.io_pool = &pool;
    pool.test_reader = exact_q4_io_test_reader;
    ExactQ4IoTestReader reader;
    exact_q4_io_test_reader_init(&reader, TEST_SIZE);
    pool.test_reader_context = &reader;

    int ids[NUM_IO_THREADS] = {10, 11, 12, 13, 14, 15};
    int ranks[NUM_IO_THREADS] = {0, 1, 2, 3, 4, 5};
    uint8_t storage[NUM_IO_THREADS][TEST_SIZE];
    uint8_t *buffers[NUM_IO_THREADS];
    for (int index = 0; index < NUM_IO_THREADS; ++index) {
        buffers[index] = storage[index];
    }
    reader.release_count = 0;

    uint64_t bytes = 99u;
    uint64_t short_reads = 99u;
    if (io_pool_dispatch(
            &pool, &engine, 0, ids, ranks, 3, 0, buffers, TEST_SIZE,
            &bytes, &short_reads, NULL) != 0 || bytes != 0u || short_reads != 0u)
        return __LINE__;
    if (io_pool_dispatch(
            &pool, &engine, 0, ids, ranks, 3, 1, buffers, TEST_SIZE,
            &bytes, &short_reads, NULL) != 0 || bytes != TEST_SIZE || short_reads != 0u)
        return __LINE__;
    reader.next_release = 0;
    reader.completion_count = 0;
    reader.release_count = NUM_IO_THREADS;
    const int completion_order[NUM_IO_THREADS] = {15, 11, 14, 10, 13, 12};
    memcpy(reader.release_order, completion_order, sizeof(completion_order));
    if (io_pool_dispatch(
            &pool, &engine, 0, ids, ranks, 3, NUM_IO_THREADS, buffers,
            TEST_SIZE, &bytes, &short_reads, NULL) != 0 ||
        bytes != NUM_IO_THREADS * TEST_SIZE || short_reads != 0u)
        return __LINE__;

    exact_q4_record_logical_read(
        &engine, 3u, 20u, 0u, DMLX_EXACT_Q4_READ_FIXED_HOT);
    exact_q4_record_logical_read(
        &engine, 3u, 21u, 1u, DMLX_EXACT_Q4_READ_BATCH_POOL);

    DmlxExactQ4ExpertReadEvent reads[32];
    DmlxExactQ4StageEvent stages[8];
    DmlxExactQ4LatencySnapshot snapshot;
    const int snapshot_status = moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 8u, reads, 32u);
    if (snapshot_status != DMLX_EXACT_Q4_PRODUCER_INVALID_STATE ||
        snapshot.read_count != 9u || snapshot.active_read_publishers != 0u)
        return __LINE__;
    for (int index = 0; index < NUM_IO_THREADS; ++index) {
        const DmlxExactQ4ExpertReadEvent *event = &reads[index + 1];
        if (event->expert != (uint16_t)ids[index] ||
            event->route_rank != (uint8_t)ranks[index] ||
            event->source != DMLX_EXACT_Q4_READ_PREAD ||
            event->read_bytes != TEST_SIZE ||
            event->flags != DMLX_EXACT_Q4_READ_COMPLETED)
            return __LINE__;
    }
    if (memcmp(reader.completion_order, completion_order,
               sizeof(completion_order)) != 0)
        return __LINE__;
    if (reads[7].source != DMLX_EXACT_Q4_READ_FIXED_HOT ||
        reads[8].source != DMLX_EXACT_Q4_READ_BATCH_POOL ||
        reads[7].requested_bytes != 0u || reads[7].read_bytes != 0u ||
        reads[7].submit_ns != reads[7].worker_start_ns ||
        reads[7].submit_ns != reads[7].complete_ns ||
        !moe_infer_exact_q4_test_no_inflight_owners(&engine))
        return __LINE__;

    exact_q4_io_test_pool_deinit(&pool);
    exact_q4_io_test_reader_deinit(&reader);
    status = exact_q4_io_test_finish(&engine);
    if (status != 0) return status;

    memset(&engine, 0, sizeof(engine));
    atomic_store_explicit(&clock, 1000u, memory_order_relaxed);
    status = exact_q4_io_test_begin(&engine, &clock, 4u, 12u);
    if (status != 0) return status;
    io_pool_init(&pool);
    engine.io_pool = &pool;
    pool.test_reader = exact_q4_io_test_reader;
    exact_q4_io_test_reader_init(&reader, TEST_SIZE);
    pool.test_reader_context = &reader;
    reader.results[10] = TEST_SIZE;
    reader.results[11] = TEST_SIZE / 2;
    reader.results[12] = 0;
    reader.results[13] = -1;
    reader.results[14] = TEST_SIZE;
    reader.results[15] = TEST_SIZE / 4;
    if (io_pool_dispatch(
            &pool, &engine, 0, ids, ranks, 4, NUM_IO_THREADS, buffers,
            TEST_SIZE, &bytes, &short_reads, NULL) == 0 || short_reads != 4u)
        return __LINE__;
    if (moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 4u, reads, 12u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE ||
        snapshot.read_count != NUM_IO_THREADS ||
        snapshot.active_read_publishers != 0u || reads[1].read_bytes != 8u ||
        reads[2].read_bytes != 0u || reads[2].error_code != 0 ||
        reads[3].read_bytes != 0u || reads[3].error_code != EIO ||
        reads[5].read_bytes != 4u ||
        !moe_infer_exact_q4_test_no_inflight_owners(&engine))
        return __LINE__;
    exact_q4_io_test_pool_deinit(&pool);
    exact_q4_io_test_reader_deinit(&reader);
    status = exact_q4_io_test_finish(&engine);
    if (status != 0) return status;
    return 0;
}

static int exact_q4_io_test_blocking_dependency_wait(void) {
    enum { TEST_SIZE = 16 };
    MoEInferEngine engine = {0};
    _Atomic uint64_t clock = 6000u;
    int status = exact_q4_io_test_begin(&engine, &clock, 8u, 16u);
    if (status != 0) return status;
    IOPool pool;
    io_pool_init(&pool);
    engine.io_pool = &pool;
    pool.test_reader = exact_q4_io_test_reader;
    ExactQ4IoTestReader reader;
    exact_q4_io_test_reader_init(&reader, TEST_SIZE);
    pool.test_reader_context = &reader;
    uint8_t buffer[TEST_SIZE];
    uint8_t *buffers[NUM_IO_THREADS] = {buffer};
    int ids[NUM_IO_THREADS] = {80};
    int ranks[NUM_IO_THREADS] = {0};
    DmlxExactQ4ProducerState *state = engine.exact_q4;
    DmlxExactQ4PendingLayer transaction = {0};
    exact_q4_layer_begin(&engine, &transaction, 12);

    for (int mode = 0; mode < 3; ++mode) {
        reader.hold = true;
        reader.entered = false;
        if (mode == 1) state->token_active = 0u;
        if (mode == 2) engine.exact_q4 = NULL;
        const uint64_t before = atomic_load_explicit(
            &clock, memory_order_relaxed);
        ExactQ4IoTestThread dispatch = {
            .engine = &engine,
            .reader = &reader,
            .pool = &pool,
            .ids = ids,
            .ranks = ranks,
            .buffers = buffers,
            .expert_size = TEST_SIZE,
            .layer = 12 + mode,
            .task_count = 1,
            .transaction = &transaction,
        };
        atomic_init(&dispatch.done, false);
        pthread_t dispatch_thread;
        pthread_create(
            &dispatch_thread, NULL, exact_q4_io_test_dispatch, &dispatch);
        pthread_mutex_lock(&pool.mutex);
        while (!pool.dependency_wait_entered)
            pthread_cond_wait(&pool.work_done, &pool.mutex);
        pthread_mutex_unlock(&pool.mutex);
        if (atomic_load_explicit(&dispatch.done, memory_order_acquire))
            return __LINE__;
        pthread_mutex_lock(&reader.mutex);
        reader.hold = false;
        pthread_cond_broadcast(&reader.condition);
        pthread_mutex_unlock(&reader.mutex);
        pthread_join(dispatch_thread, NULL);
        if (!atomic_load_explicit(&dispatch.done, memory_order_acquire) ||
            dispatch.dispatch_status != 0 || dispatch.bytes_read != TEST_SIZE ||
            dispatch.short_reads != 0u)
            return __LINE__;
        const uint64_t after = atomic_load_explicit(
            &clock, memory_order_relaxed);
        if ((mode == 0 && after != before + 50u) ||
            (mode != 0 && after != before))
            return __LINE__;
        if (mode == 1) state->token_active = 1u;
        if (mode == 2) engine.exact_q4 = state;
    }

    DmlxExactQ4LatencySnapshot snapshot;
    DmlxExactQ4StageEvent stages[8];
    DmlxExactQ4ExpertReadEvent reads[16];
    if (moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 8u, reads, 16u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE || snapshot.stage_count != 0u)
        return __LINE__;
    const DmlxMlaPathIdentity identity = {
        .path = DMLX_MLA_PATH_PERSISTENT_INTERNAL,
        .flags = DMLX_MLA_PATH_PERSISTENT_KV,
    };
    exact_q4_layer_set_path(&transaction, &identity);
    if (exact_q4_layer_commit(&engine, &transaction) !=
            DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;
    if (moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 8u, reads, 16u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE || snapshot.stage_count != 1u ||
        stages[0].layer != 12u ||
        stages[0].stage != DMLX_EXACT_Q4_STAGE_IO_EXPOSED_WAIT ||
        stages[0].flags != DMLX_EXACT_Q4_EVENT_JOIN_BLOCKED ||
        stages[0].calls != 1u || stages[0].begin_ns >= stages[0].end_ns)
        return __LINE__;

    const int generation = io_pool_start(
        &pool, 0, ids, 1, buffers, TEST_SIZE, NULL, NULL, NULL, false);
    io_pool_wait(&pool);
    const uint64_t before_complete = atomic_load_explicit(
        &clock, memory_order_relaxed);
    io_pool_wait_dependency(&pool, generation, &engine, 15, NULL);
    if (atomic_load_explicit(&clock, memory_order_relaxed) != before_complete ||
        moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 8u, reads, 16u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE || snapshot.stage_count != 1u)
        return __LINE__;

    exact_q4_io_test_pool_deinit(&pool);
    exact_q4_io_test_reader_deinit(&reader);
    status = exact_q4_io_test_finish(&engine);
    return status;
}

static int exact_q4_io_test_prefetch_wait_and_drain(void) {
    enum { TEST_SIZE = 16 };
    MoEInferEngine engine = {0};
    _Atomic uint64_t clock = 2000u;
    int status = exact_q4_io_test_begin(&engine, &clock, 8u, 16u);
    if (status != 0) return status;
    IOPool pool;
    io_pool_init(&pool);
    engine.prefetch_pool = &pool;
    engine.active_expert_size = TEST_SIZE;
    pool.test_reader = exact_q4_io_test_reader;
    ExactQ4IoTestReader reader;
    exact_q4_io_test_reader_init(&reader, TEST_SIZE);
    pool.test_reader_context = &reader;
    uint8_t buffer[TEST_SIZE];
    engine.expert_buf_pred[0] = buffer;
    engine.expert_buf_pred[1] = buffer;
    engine.expert_buf_pred2[2] = buffer;
    uint8_t *buffers[NUM_IO_THREADS] = {buffer};
    int ids[NUM_IO_THREADS] = {30};
    const int generation = io_pool_start(
        &pool, 0, ids, 1, buffers, TEST_SIZE, &engine,
        NULL, NULL, true);
    io_pool_wait(&pool);
    engine.prefetch_active = true;
    engine.prefetch_layer = 5;
    engine.prefetch_generation = generation;
    engine.prefetch_ids[2] = 30;
    engine.prefetch_task[2] = 0;
    int actual_ids[NUM_IO_THREADS] = {30, 31, 32, 33, 34, 35};
    uint8_t *expert_data[NUM_IO_THREADS] = {0};
    prefetch_consume(&engine, 5, actual_ids, expert_data, NULL);
    if (expert_data[0] != buffer) return __LINE__;

    DmlxExactQ4LatencySnapshot snapshot;
    DmlxExactQ4StageEvent stages[8];
    DmlxExactQ4ExpertReadEvent reads[16];
    if (moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 8u, reads, 16u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE || snapshot.stage_count != 0u ||
        reads[0].source != DMLX_EXACT_Q4_READ_PREFETCH ||
        reads[0].route_rank != 0u)
        return __LINE__;
    prefetch_drain(&engine);
    if (!moe_infer_exact_q4_test_no_inflight_owners(&engine))
        return __LINE__;

    reader.hold = true;
    reader.entered = false;
    ids[0] = 31;
    const int blocked_generation = io_pool_start(
        &pool, 0, ids, 1, buffers, TEST_SIZE, &engine,
        NULL, NULL, true);
    engine.prefetch_active = true;
    engine.prefetch_layer = 6;
    engine.prefetch_generation = blocked_generation;
    engine.prefetch_ids[0] = 31;
    engine.prefetch_task[0] = 0;
    pthread_mutex_lock(&reader.mutex);
    while (!reader.entered) pthread_cond_wait(&reader.condition, &reader.mutex);
    pthread_mutex_unlock(&reader.mutex);
    ExactQ4IoTestThread release = {
        .engine = &engine,
        .reader = &reader,
    };
    pthread_t release_thread;
    pthread_create(&release_thread, NULL, exact_q4_io_test_release, &release);
    memset(expert_data, 0, sizeof(expert_data));
    actual_ids[0] = 31;
    actual_ids[1] = 60;
    DmlxExactQ4PendingLayer blocked_transaction = {0};
    exact_q4_layer_begin(&engine, &blocked_transaction, 6);
    prefetch_consume(
        &engine, 6, actual_ids, expert_data, &blocked_transaction);
    pthread_join(release_thread, NULL);
    if (moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 8u, reads, 16u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE || snapshot.stage_count != 0u)
        return __LINE__;
    const DmlxMlaPathIdentity identity = {
        .path = DMLX_MLA_PATH_PERSISTENT_INTERNAL,
        .flags = DMLX_MLA_PATH_PERSISTENT_KV,
    };
    exact_q4_layer_set_path(&blocked_transaction, &identity);
    if (exact_q4_layer_commit(&engine, &blocked_transaction) !=
            DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;
    if (moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 8u, reads, 16u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE || snapshot.stage_count != 1u ||
        stages[0].stage != DMLX_EXACT_Q4_STAGE_IO_EXPOSED_WAIT ||
        stages[0].flags != DMLX_EXACT_Q4_EVENT_JOIN_BLOCKED ||
        stages[0].calls != 1u)
        return __LINE__;
    prefetch_drain(&engine);

    reader.hold = true;
    reader.entered = false;
    ids[0] = 32;
    const int unmatched_generation = io_pool_start(
        &pool, 0, ids, 1, buffers, TEST_SIZE, &engine,
        NULL, NULL, true);
    engine.prefetch_active = false;
    engine.prefetch_layer = 7;
    engine.prefetch_count = 1;
    engine.prefetch_generation = unmatched_generation;
    engine.prefetch_ids[1] = 32;
    engine.prefetch_task[1] = 0;
    pthread_mutex_lock(&reader.mutex);
    while (!reader.entered) pthread_cond_wait(&reader.condition, &reader.mutex);
    pthread_mutex_unlock(&reader.mutex);
    ExactQ4IoTestThread drain = {
        .engine = &engine,
        .reader = &reader,
    };
    atomic_init(&drain.done, false);
    pthread_t drain_thread;
    pthread_create(&drain_thread, NULL, exact_q4_io_test_drain, &drain);
    usleep(10000);
    if (atomic_load_explicit(&drain.done, memory_order_acquire)) return __LINE__;
    pthread_mutex_lock(&reader.mutex);
    reader.hold = false;
    pthread_cond_broadcast(&reader.condition);
    pthread_mutex_unlock(&reader.mutex);
    pthread_join(drain_thread, NULL);
    if (!atomic_load_explicit(&drain.done, memory_order_acquire) ||
        engine.prefetch_generation != 0 || engine.prefetch_count != 0 ||
        engine.prefetch_layer != -1 ||
        !moe_infer_exact_q4_test_no_inflight_owners(&engine))
        return __LINE__;
    const int unmatched_snapshot_status = moe_infer_exact_q4_snapshot(
        &engine, &snapshot, stages, 8u, reads, 16u);
    if (unmatched_snapshot_status != DMLX_EXACT_Q4_PRODUCER_INVALID_STATE ||
        snapshot.read_count != 2u)
        return __LINE__;

    exact_q4_io_test_pool_deinit(&pool);
    exact_q4_io_test_reader_deinit(&reader);
    status = exact_q4_io_test_finish(&engine);
    if (status != 0) return status;
    return 0;
}

static int exact_q4_io_test_generation_and_clock_gates(void) {
    enum { TEST_SIZE = 16 };
    MoEInferEngine engine = {0};
    _Atomic uint64_t clock = 4000u;
    int status = exact_q4_io_test_begin(&engine, &clock, 4u, 8u);
    if (status != 0) return status;
    IOPool pool;
    io_pool_init(&pool);
    engine.prefetch_pool = &pool;
    engine.active_expert_size = TEST_SIZE;
    pool.test_reader = exact_q4_io_test_reader;
    ExactQ4IoTestReader reader;
    exact_q4_io_test_reader_init(&reader, TEST_SIZE);
    pool.test_reader_context = &reader;
    uint8_t buffer[TEST_SIZE];
    uint8_t *buffers[NUM_IO_THREADS] = {buffer};
    int ids[NUM_IO_THREADS] = {50};
    ExactQ4IoTestThread release = {.engine = &engine, .reader = &reader};

    for (int mode = 0; mode < 2; ++mode) {
        reader.hold = true;
        reader.entered = false;
        const int generation = io_pool_start(
            &pool, 0, ids, 1, buffers, TEST_SIZE, NULL, NULL, NULL, false);
        pthread_mutex_lock(&reader.mutex);
        while (!reader.entered)
            pthread_cond_wait(&reader.condition, &reader.mutex);
        pthread_mutex_unlock(&reader.mutex);
        DmlxExactQ4ProducerState *state = engine.exact_q4;
        if (mode == 0) state->token_active = 0u;
        else engine.exact_q4 = NULL;
        const uint64_t before = atomic_load_explicit(&clock, memory_order_relaxed);
        pthread_t release_thread;
        pthread_create(&release_thread, NULL, exact_q4_io_test_release, &release);
        ExactQ4DeferredRead read = {0};
        bool blocked = false;
        if (!io_pool_wait_task_blocked(
                &pool, 0, generation, &engine, &read, &blocked) || blocked)
            return __LINE__;
        pthread_join(release_thread, NULL);
        if (atomic_load_explicit(&clock, memory_order_relaxed) != before)
            return __LINE__;
        if (mode == 0) state->token_active = 1u;
        else engine.exact_q4 = state;
    }

    const int generation = io_pool_start(
        &pool, 0, ids, 1, buffers, TEST_SIZE, &engine, NULL, NULL, true);
    io_pool_wait(&pool);
    engine.prefetch_active = true;
    engine.prefetch_layer = 9;
    engine.prefetch_generation = generation + 1;
    engine.prefetch_ids[0] = ids[0];
    engine.prefetch_task[0] = 0;
    int actual_ids[NUM_IO_THREADS] = {50, 51, 52, 53, 54, 55};
    uint8_t *expert_data[NUM_IO_THREADS] = {0};
    prefetch_consume(&engine, 9, actual_ids, expert_data, NULL);
    if (expert_data[0] != NULL) return __LINE__;
    DmlxExactQ4LatencySnapshot snapshot;
    DmlxExactQ4StageEvent stages[4];
    DmlxExactQ4ExpertReadEvent reads[8];
    if (moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 4u, reads, 8u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE || snapshot.read_count != 0u ||
        snapshot.stage_count != 0u)
        return __LINE__;
    prefetch_drain(&engine);
    exact_q4_io_test_pool_deinit(&pool);
    exact_q4_io_test_reader_deinit(&reader);
    status = exact_q4_io_test_finish(&engine);
    return status;
}

static int exact_q4_io_test_reconfigure_during_deferred_read(void) {
    enum { TEST_SIZE = 16 };
    MoEInferEngine engine = {0};
    _Atomic uint64_t old_clock = 5000u;
    int status = exact_q4_io_test_begin(&engine, &old_clock, 4u, 8u);
    if (status != 0) return status;

    IOPool pool;
    io_pool_init(&pool);
    engine.prefetch_pool = &pool;
    engine.active_expert_size = TEST_SIZE;
    pool.test_reader = exact_q4_io_test_reader;
    ExactQ4IoTestReader reader;
    exact_q4_io_test_reader_init(&reader, TEST_SIZE);
    reader.hold = true;
    pool.test_reader_context = &reader;
    uint8_t buffer[TEST_SIZE];
    uint8_t *buffers[NUM_IO_THREADS] = {buffer};
    int ids[NUM_IO_THREADS] = {70};
    const int generation = io_pool_start(
        &pool, 0, ids, 1, buffers, TEST_SIZE, &engine,
        NULL, NULL, true);
    engine.prefetch_generation = generation;

    pthread_mutex_lock(&reader.mutex);
    while (!reader.entered)
        pthread_cond_wait(&reader.condition, &reader.mutex);
    pthread_mutex_unlock(&reader.mutex);
    const uint64_t old_before_release = atomic_load_explicit(
        &old_clock, memory_order_relaxed);

    if (moe_infer_exact_q4_finalize(&engine) != DMLX_EXACT_Q4_PRODUCER_OK ||
        moe_infer_exact_q4_enable(&engine, false) !=
            DMLX_EXACT_Q4_PRODUCER_OK ||
        moe_infer_exact_q4_deconfigure(&engine) !=
            DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;

    _Atomic uint64_t new_clock = 9000u;
    if (moe_infer_exact_q4_configure(&engine, 4u, 8u) !=
            DMLX_EXACT_Q4_PRODUCER_OK ||
        moe_infer_exact_q4_enable(&engine, true) !=
            DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;
    moe_infer_exact_q4_test_set_clock(
        &engine, exact_q4_io_test_clock, &new_clock);
    if (moe_infer_exact_q4_begin_request(&engine, 8u) !=
            DMLX_EXACT_Q4_PRODUCER_OK ||
        moe_infer_exact_q4_begin_token(&engine, 12u) !=
            DMLX_EXACT_Q4_PRODUCER_OK)
        return __LINE__;

    pthread_mutex_lock(&reader.mutex);
    reader.hold = false;
    pthread_cond_broadcast(&reader.condition);
    pthread_mutex_unlock(&reader.mutex);
    prefetch_drain(&engine);
    if (!moe_infer_exact_q4_test_no_inflight_owners(&engine))
        return __LINE__;

    if (pool.submit_ns[0] != 5000u || pool.worker_start_ns[0] != 5010u ||
        pool.complete_ns[0] != 5020u ||
        atomic_load_explicit(&old_clock, memory_order_relaxed) !=
            old_before_release + 10u ||
        atomic_load_explicit(&new_clock, memory_order_relaxed) != 9000u)
        return __LINE__;

    DmlxExactQ4LatencySnapshot snapshot;
    DmlxExactQ4StageEvent stages[4];
    DmlxExactQ4ExpertReadEvent reads[8];
    if (moe_infer_exact_q4_snapshot(
            &engine, &snapshot, stages, 4u, reads, 8u) !=
            DMLX_EXACT_Q4_PRODUCER_INVALID_STATE ||
        snapshot.read_count != 0u || snapshot.stage_count != 0u)
        return __LINE__;

    exact_q4_io_test_pool_deinit(&pool);
    exact_q4_io_test_reader_deinit(&reader);
    return exact_q4_io_test_finish(&engine);
}

static int exact_q4_io_test_deinit_drain(void) {
    enum { TEST_SIZE = 16 };
    MoEInferEngine *engine = calloc(1u, sizeof(*engine));
    if (!engine) return __LINE__;
    for (int layer = 0; layer < N_LAYERS; ++layer) engine->packed_fd[layer] = -1;
    engine->initialized = true;
    engine->active_expert_size = TEST_SIZE;
    _Atomic uint64_t clock = 3000u;
    int status = exact_q4_io_test_begin(engine, &clock, 4u, 4u);
    if (status != 0) return status;
    IOPool *pool = calloc(1u, sizeof(*pool));
    if (!pool) return __LINE__;
    io_pool_init(pool);
    engine->prefetch_pool = pool;
    pool->test_reader = exact_q4_io_test_reader;
    ExactQ4IoTestReader reader;
    exact_q4_io_test_reader_init(&reader, TEST_SIZE);
    reader.hold = true;
    pool->test_reader_context = &reader;
    uint8_t buffer[TEST_SIZE];
    uint8_t *buffers[NUM_IO_THREADS] = {buffer};
    int ids[NUM_IO_THREADS] = {40};
    const int generation = io_pool_start(
        pool, 0, ids, 1, buffers, TEST_SIZE, engine,
        NULL, NULL, true);
    engine->prefetch_generation = generation;
    pthread_mutex_lock(&reader.mutex);
    while (!reader.entered) pthread_cond_wait(&reader.condition, &reader.mutex);
    pthread_mutex_unlock(&reader.mutex);
    ExactQ4IoTestThread release = {
        .engine = engine,
        .reader = &reader,
    };
    pthread_t release_thread;
    pthread_create(&release_thread, NULL, exact_q4_io_test_release, &release);
    moe_infer_deinit(engine);
    pthread_join(release_thread, NULL);
    exact_q4_io_test_reader_deinit(&reader);
    return 0;
}

int moe_infer_exact_q4_test_run_io_suite(void) {
    int status = exact_q4_io_test_source_and_results();
    if (status != 0) return status;
    status = exact_q4_io_test_blocking_dependency_wait();
    if (status != 0) return status;
    status = exact_q4_io_test_prefetch_wait_and_drain();
    if (status != 0) return status;
    status = exact_q4_io_test_generation_and_clock_gates();
    if (status != 0) return status;
    status = exact_q4_io_test_reconfigure_during_deferred_read();
    if (status != 0) return status;
    return exact_q4_io_test_deinit_drain();
}

#endif
