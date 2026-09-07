#include "engine.h"

#include <pthread.h>
#include <math.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

_Static_assert(sizeof(DmlxExactQ4StageEvent) == 48, "stage ABI");
_Static_assert(sizeof(DmlxExactQ4ExpertReadEvent) == 72, "read ABI");
_Static_assert(sizeof(DmlxExactQ4LatencySnapshot) == 72, "snapshot ABI");
_Static_assert(offsetof(DmlxExactQ4LatencySnapshot, stages) == 56,
               "snapshot stage pointer ABI");
_Static_assert(sizeof(DmlxGpuTimestampSample) == 72, "GPU timestamp sample ABI");
_Static_assert(offsetof(DmlxGpuTimestampSample, gpu_start_s) == 48,
               "GPU timestamp timing ABI");

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "exact-Q4 producer self-test failed: %s\n", message); \
        return 1; \
    } \
} while (0)

static uint64_t test_clock(void *context) {
    uint32_t *calls = (uint32_t *)context;
    (*calls)++;
    return UINT64_C(123456789);
}

static DmlxExactQ4StageEvent stage_event(uint16_t stage, uint64_t begin) {
    return (DmlxExactQ4StageEvent){
        .stage = stage,
        .flags = DMLX_EXACT_Q4_EVENT_CPU_ACTIVE,
        .calls = 1u,
        .begin_ns = begin,
        .end_ns = begin + 5u,
        .bytes = 64u,
    };
}

static DmlxExactQ4ExpertReadEvent read_event(
    uint16_t expert, uint8_t route_rank, uint64_t submit) {
    return (DmlxExactQ4ExpertReadEvent){
        .layer = 7u,
        .expert = expert,
        .route_rank = route_rank,
        .source = DMLX_EXACT_Q4_READ_PREAD,
        .submit_ns = submit,
        .requested_bytes = 4096u,
    };
}

typedef struct PublishThread {
    MoEInferEngine *engine;
    uint32_t ordinal;
    int status;
} PublishThread;

static void *publish_thread(void *opaque) {
    PublishThread *thread = (PublishThread *)opaque;
    thread->status = moe_infer_exact_q4_publish_read_worker_start(
        thread->engine, thread->ordinal, 210u);
    return NULL;
}

enum {
    SCALAR_ROUTE_CANDIDATES = 256,
    SCALAR_ACTIVE_EXPERTS = 6,
    SCALAR_LAYER_ENTRY_PROBES = 2,
    SCALAR_SOURCE_PROBES = SCALAR_ACTIVE_EXPERTS,
    SCALAR_PREFETCH_PROBES = 2,
    SCALAR_PROBES_PER_LAYER =
        SCALAR_LAYER_ENTRY_PROBES + SCALAR_SOURCE_PROBES +
        SCALAR_PREFETCH_PROBES,
};

static uint64_t representative_scalar_token(
    MoEInferEngine *engine, uint32_t iterations, bool probe_disabled) {
    uint64_t value = UINT64_C(0x9e3779b97f4a7c15);
    for (uint32_t iteration = 0u; iteration < iterations; ++iteration) {
        for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
            if (probe_disabled && moe_infer_exact_q4_recording_enabled(engine))
                value ^= UINT64_MAX;
            if (probe_disabled && moe_infer_exact_q4_recording_enabled(engine))
                value ^= UINT64_MAX;

            for (uint32_t expert = 0u; expert < SCALAR_ROUTE_CANDIDATES;
                 ++expert) {
                value ^= value << 7;
                value ^= value >> 9;
                value += (uint64_t)iteration + layer + expert +
                    UINT64_C(0x100000001b3);
            }
            for (uint32_t rank = 0u; rank < SCALAR_ACTIVE_EXPERTS; ++rank) {
                value += (value >> (rank + 1u)) ^ rank;
                if (probe_disabled &&
                    moe_infer_exact_q4_recording_enabled(engine))
                    value ^= UINT64_MAX;
            }
            if (probe_disabled && moe_infer_exact_q4_recording_enabled(engine))
                value ^= UINT64_MAX;
            if (probe_disabled && moe_infer_exact_q4_recording_enabled(engine))
                value ^= UINT64_MAX;
        }
    }
    return value;
}

static uint64_t elapsed_ns(
    MoEInferEngine *engine, uint32_t iterations, bool probe_disabled,
    volatile uint64_t *sink) {
    struct timespec begin;
    struct timespec end;
    CHECK(clock_gettime(CLOCK_MONOTONIC_RAW, &begin) == 0, "clock begin");
    *sink += representative_scalar_token(engine, iterations, probe_disabled);
    CHECK(clock_gettime(CLOCK_MONOTONIC_RAW, &end) == 0, "clock end");
    const uint64_t begin_ns = (uint64_t)begin.tv_sec * UINT64_C(1000000000) +
        (uint64_t)begin.tv_nsec;
    const uint64_t end_ns = (uint64_t)end.tv_sec * UINT64_C(1000000000) +
        (uint64_t)end.tv_nsec;
    return end_ns - begin_ns;
}

static int compare_u64(const void *left, const void *right) {
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static int test_disabled_micro_overhead(void) {
    enum { ROUNDS = 11, ITERATIONS = 5000, WARMUP = 3 };
    MoEInferEngine engine;
    memset(&engine, 0, sizeof(engine));
    engine.initialized = true;
    volatile uint64_t sink = 0u;
    for (uint32_t round = 0u; round < WARMUP; ++round) {
        (void)elapsed_ns(&engine, ITERATIONS / 4u, false, &sink);
        (void)elapsed_ns(&engine, ITERATIONS / 4u, true, &sink);
    }
    uint64_t baseline[ROUNDS];
    uint64_t disabled[ROUNDS];
    double paired_ratios[ROUNDS];
    for (uint32_t round = 0u; round < ROUNDS; ++round) {
        if ((round & 1u) == 0u) {
            baseline[round] = elapsed_ns(&engine, ITERATIONS, false, &sink);
            disabled[round] = elapsed_ns(&engine, ITERATIONS, true, &sink);
        } else {
            disabled[round] = elapsed_ns(&engine, ITERATIONS, true, &sink);
            baseline[round] = elapsed_ns(&engine, ITERATIONS, false, &sink);
        }
        paired_ratios[round] =
            ((double)disabled[round] - (double)baseline[round]) /
            (double)baseline[round];
    }
    qsort(baseline, ROUNDS, sizeof(baseline[0]), compare_u64);
    qsort(disabled, ROUNDS, sizeof(disabled[0]), compare_u64);
    qsort(paired_ratios, ROUNDS, sizeof(paired_ratios[0]), compare_double);
    const double baseline_ns = (double)baseline[ROUNDS / 2] / ITERATIONS;
    const double disabled_ns = (double)disabled[ROUNDS / 2] / ITERATIONS;
    const double ratio = paired_ratios[ROUNDS / 2];
    const double delta_ns = ratio * baseline_ns;
    fprintf(stderr,
        "exact-Q4 disabled overhead: baseline_ns/token=%.3f disabled_ns/token=%.3f delta_ns/token=%.3f ratio=%.6f probes/token=%u (%u layers x %u probes)\n",
        baseline_ns, disabled_ns, delta_ns, ratio,
        N_LAYERS * SCALAR_PROBES_PER_LAYER, N_LAYERS,
        SCALAR_PROBES_PER_LAYER);
    CHECK(ratio <= 0.01, "disabled median overhead exceeds 1%");

    uint32_t ordinal = UINT32_MAX;
    DmlxExactQ4StageEvent stage = stage_event(DMLX_EXACT_Q4_STAGE_ROUTING, 1u);
    DmlxExactQ4ExpertReadEvent read = read_event(1u, 0u, 1u);
    CHECK(engine.exact_q4 == NULL && engine.exact_q4_storage == NULL,
          "disabled microbenchmark allocated producer storage");
    CHECK(moe_infer_exact_q4_append_stage(&engine, &stage, &ordinal) ==
              DMLX_EXACT_Q4_PRODUCER_DISABLED && ordinal == UINT32_MAX,
          "disabled microbenchmark published a stage event");
    CHECK(moe_infer_exact_q4_reserve_read(&engine, &read, &ordinal) ==
              DMLX_EXACT_Q4_PRODUCER_DISABLED && ordinal == UINT32_MAX,
          "disabled microbenchmark published a read event");
    CHECK(moe_infer_exact_q4_test_clock_now(&engine) == 0u,
          "disabled microbenchmark read the producer clock");
    CHECK(sink != 0u, "representative bookkeeping was optimized away");
    return 0;
}

static int test_disabled_and_configuration(MoEInferEngine *engine) {
    uint32_t clock_calls = 0u;
    uint32_t ordinal = UINT32_MAX;
    DmlxExactQ4StageEvent stage = stage_event(
        DMLX_EXACT_Q4_STAGE_ROUTING, 10u);
    DmlxExactQ4ExpertReadEvent read = read_event(3u, 0u, 20u);
    NativeRequestMetrics telemetry_before;
    memset(&telemetry_before, 0x5a, sizeof(telemetry_before));
    engine->telemetry = telemetry_before;

    CHECK(engine->exact_q4 == NULL && engine->exact_q4_storage == NULL,
          "default state is not null");
    CHECK(!moe_infer_exact_q4_recording_enabled(engine),
          "disabled recording query returned true");
    CHECK(moe_infer_exact_q4_append_stage(engine, &stage, &ordinal) ==
              DMLX_EXACT_Q4_PRODUCER_DISABLED,
          "disabled stage helper did work");
    CHECK(moe_infer_exact_q4_reserve_read(engine, &read, &ordinal) ==
              DMLX_EXACT_Q4_PRODUCER_DISABLED,
          "disabled read helper did work");
    CHECK(moe_infer_exact_q4_test_clock_now(engine) == 0u,
          "disabled clock helper returned time");
    CHECK(clock_calls == 0u, "disabled helper called clock");
    CHECK(memcmp(&engine->telemetry, &telemetry_before,
                 sizeof(telemetry_before)) == 0,
          "disabled producer changed telemetry v1");

    CHECK(moe_infer_exact_q4_configure(engine, 0u, 1u) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT,
          "zero stage capacity accepted");
    CHECK(moe_infer_exact_q4_configure(
              engine, DMLX_EXACT_Q4_MAX_STAGE_CAPACITY + 1u, 1u) ==
              DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED,
          "stage maximum not enforced");
    CHECK(moe_infer_exact_q4_configure(
              engine, 1u, DMLX_EXACT_Q4_MAX_READ_CAPACITY + 1u) ==
              DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED,
          "read maximum not enforced");
    CHECK(moe_infer_exact_q4_configure(engine, 3u, 2u) == 0,
          "configure failed");
    CHECK(engine->exact_q4 == NULL && engine->exact_q4_storage != NULL,
          "configure enabled producer");
    CHECK(moe_infer_exact_q4_test_allocation_bytes(engine) >=
              3u * 48u + 2u * 72u + 2u * sizeof(uint32_t),
          "bounded allocation does not contain configured records");
    moe_infer_exact_q4_test_set_clock(engine, test_clock, &clock_calls);
    CHECK(moe_infer_exact_q4_test_clock_now(engine) == 0u,
          "configured-disabled clock helper returned time");
    CHECK(clock_calls == 0u, "configured-disabled helper called clock");
    CHECK(moe_infer_exact_q4_enable(engine, true) == 0, "enable failed");
    CHECK(moe_infer_exact_q4_test_clock_now(engine) == UINT64_C(123456789),
          "test clock result");
    CHECK(clock_calls == 1u, "enabled clock count");
    CHECK(memcmp(&engine->telemetry, &telemetry_before,
                 sizeof(telemetry_before)) == 0,
          "configure changed telemetry v1");
    return 0;
}

static int test_order_publication_and_reuse(MoEInferEngine *engine) {
    CHECK(moe_infer_exact_q4_begin_request(engine, 9u) == 0,
          "begin request");
    CHECK(moe_infer_exact_q4_begin_token(engine, 4u) == 0,
          "begin token");
    for (uint32_t index = 0u; index < 3u; ++index) {
        DmlxExactQ4StageEvent stage = stage_event(
            (uint16_t)(DMLX_EXACT_Q4_STAGE_ROUTING + index), 30u + index);
        uint32_t ordinal = UINT32_MAX;
        CHECK(moe_infer_exact_q4_append_stage(engine, &stage, &ordinal) == 0,
              "append stage");
        CHECK(ordinal == index, "stage ordinal is not reservation order");
    }

    DmlxExactQ4ExpertReadEvent first = read_event(11u, 0u, 100u);
    DmlxExactQ4ExpertReadEvent second = read_event(12u, 1u, 200u);
    uint32_t first_ordinal = UINT32_MAX, second_ordinal = UINT32_MAX;
    CHECK(moe_infer_exact_q4_reserve_read(
              engine, &first, &first_ordinal) == 0,
          "reserve first read");
    CHECK(moe_infer_exact_q4_reserve_read(
              engine, &second, &second_ordinal) == 0,
          "reserve second read");
    CHECK(first_ordinal == 0u && second_ordinal == 1u,
          "read ordinal is not reservation order");

    DmlxExactQ4StageEvent stages[3];
    DmlxExactQ4ExpertReadEvent reads[2];
    DmlxExactQ4LatencySnapshot snapshot;
    CHECK(moe_infer_exact_q4_snapshot(
              engine, &snapshot, stages, 3u, reads, 2u) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE,
          "live snapshot appeared valid");
    CHECK(snapshot.read_count == 0u && snapshot.incomplete == 1u &&
              snapshot.active_read_publishers == 2u,
          "snapshot exposed unpublished reads");

    PublishThread thread_state = {
        .engine = engine,
        .ordinal = second_ordinal,
        .status = -99,
    };
    pthread_t thread;
    CHECK(pthread_create(&thread, NULL, publish_thread, &thread_state) == 0,
          "create publication thread");
    CHECK(pthread_join(thread, NULL) == 0, "join publication thread");
    CHECK(thread_state.status == 0, "thread start publication failed");
    CHECK(moe_infer_exact_q4_publish_read_completion(
              engine, second_ordinal, 220u, 4096u, 0) == 0,
          "cross-thread completion publication failed");
    CHECK(moe_infer_exact_q4_publish_read_worker_start(
              engine, second_ordinal, 211u) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE,
          "duplicate worker start accepted");
    CHECK(moe_infer_exact_q4_publish_read_completion(
              engine, second_ordinal, 221u, 4096u, 0) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE,
          "duplicate completion accepted");
    CHECK(moe_infer_exact_q4_snapshot(
              engine, &snapshot, stages, 3u, reads, 2u) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE,
          "partial publication appeared valid");
    CHECK(snapshot.read_count == 1u && reads[0].event_ordinal == 1u &&
              reads[0].expert == 12u && reads[0].complete_ns == 220u &&
              reads[0].flags == DMLX_EXACT_Q4_READ_COMPLETED,
          "acquire snapshot did not see complete payload");
    CHECK(moe_infer_exact_q4_reset(engine) ==
              DMLX_EXACT_Q4_PRODUCER_BUSY,
          "reset accepted active publisher");
    CHECK(moe_infer_exact_q4_enable(engine, false) ==
              DMLX_EXACT_Q4_PRODUCER_BUSY,
          "disable accepted active publisher");
    CHECK(moe_infer_exact_q4_configure(engine, 4u, 4u) ==
              DMLX_EXACT_Q4_PRODUCER_BUSY,
          "reconfigure accepted active publisher");
    CHECK(moe_infer_exact_q4_deconfigure(engine) ==
              DMLX_EXACT_Q4_PRODUCER_BUSY,
          "deconfigure accepted active publisher");

    CHECK(moe_infer_exact_q4_publish_read_worker_start(
              engine, first_ordinal, 110u) == 0,
          "first worker start");
    CHECK(moe_infer_exact_q4_publish_read_completion(
              engine, first_ordinal, 120u, 3072u, 0) == 0,
          "first completion");
    CHECK(moe_infer_exact_q4_finalize(engine) == 0, "finalize");
    CHECK(moe_infer_exact_q4_snapshot(
              engine, &snapshot, stages, 3u, reads, 2u) == 0,
          "valid snapshot failed");
    CHECK(snapshot.valid == 1u && snapshot.stage_count == 3u &&
              snapshot.read_count == 2u && snapshot.request_ordinal == 9u &&
              snapshot.token_ordinal == 4u,
          "valid snapshot metadata");
    for (uint32_t index = 0u; index < 3u; ++index) {
        CHECK(stages[index].event_ordinal == index &&
                  stages[index].request_ordinal == 9u &&
                  stages[index].token_ordinal == 4u,
              "stage snapshot ordering/identity");
    }
    CHECK(reads[0].event_ordinal == 0u && reads[1].event_ordinal == 1u,
          "completion order changed read order");

    const void *storage = moe_infer_exact_q4_test_storage(engine);
    const size_t allocation_bytes =
        moe_infer_exact_q4_test_allocation_bytes(engine);
    CHECK(moe_infer_exact_q4_reset(engine) == 0, "reset failed");
    CHECK(moe_infer_exact_q4_test_storage(engine) == storage &&
              moe_infer_exact_q4_test_allocation_bytes(engine) == allocation_bytes,
          "reset reallocated storage");
    CHECK(moe_infer_exact_q4_begin_request(engine, 10u) == 0,
          "begin request after reset");
    CHECK(moe_infer_exact_q4_begin_token(engine, 0u) == 0,
          "begin token after reset");
    DmlxExactQ4ExpertReadEvent cancelled = read_event(13u, 0u, 300u);
    uint32_t cancelled_ordinal = UINT32_MAX;
    CHECK(moe_infer_exact_q4_reserve_read(
              engine, &cancelled, &cancelled_ordinal) == 0,
          "reserve cancelled read");
    moe_infer_exact_q4_test_seed_read_state(engine, cancelled_ordinal, 2u);
    DmlxExactQ4LatencySnapshot busy_snapshot = {0};
    DmlxExactQ4StageEvent busy_stages[3] = {0};
    DmlxExactQ4ExpertReadEvent busy_reads[2] = {0};
    CHECK(moe_infer_exact_q4_abort_request(engine) ==
              DMLX_EXACT_Q4_PRODUCER_BUSY,
          "abort accepted STARTING publisher");
    CHECK(moe_infer_exact_q4_snapshot(
              engine, &busy_snapshot, busy_stages, 3u, busy_reads, 2u) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE &&
              busy_snapshot.active_read_publishers == 1u &&
              busy_snapshot.incomplete == 1u,
          "BUSY abort changed publisher ownership");
    moe_infer_exact_q4_test_seed_read_state(engine, cancelled_ordinal, 1u);
    CHECK(moe_infer_exact_q4_abort_read(engine, cancelled_ordinal) == 0,
          "abort reserved read");
    CHECK(moe_infer_exact_q4_abort_read(engine, cancelled_ordinal) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE,
          "duplicate abort accepted");
    CHECK(moe_infer_exact_q4_reset(engine) ==
              DMLX_EXACT_Q4_PRODUCER_BUSY,
          "reset accepted active request after read abort");
    CHECK(moe_infer_exact_q4_abort_request(engine) == 0,
          "request abort failed");
    CHECK(moe_infer_exact_q4_abort_request(engine) == 0,
          "request abort is not idempotent");
    CHECK(moe_infer_exact_q4_finalize(engine) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE,
          "finalize accepted aborted request");
    CHECK(moe_infer_exact_q4_reset(engine) == 0,
          "request abort did not permit reset");
    return 0;
}

static int test_overflow_and_invalid_lifecycle(MoEInferEngine *engine) {
    uint32_t ordinal = UINT32_MAX;
    DmlxExactQ4StageEvent overflow_stage = stage_event(
        DMLX_EXACT_Q4_STAGE_ROUTING, 500u);
    DmlxExactQ4ExpertReadEvent overflow_read = read_event(20u, 2u, 600u);
    CHECK(moe_infer_exact_q4_begin_request(engine, 11u) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE,
          "nested request accepted");
    moe_infer_exact_q4_test_seed_drops(
        engine, UINT32_MAX - 1u, UINT32_MAX - 1u);
    for (uint32_t index = 0u; index < 4u; ++index) {
        const int status = moe_infer_exact_q4_append_stage(
            engine, &overflow_stage, &ordinal);
        CHECK(status == (index < 3u ? 0 :
              DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED),
              "stage overflow status");
    }
    for (uint32_t index = 0u; index < 3u; ++index) {
        overflow_read.expert = (uint16_t)(20u + index);
        const int status = moe_infer_exact_q4_reserve_read(
            engine, &overflow_read, &ordinal);
        CHECK(status == (index < 2u ? 0 :
              DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED),
              "read overflow status");
    }
    CHECK(moe_infer_exact_q4_publish_read_worker_start(engine, 1u, 610u) == 0,
          "overflow read1 start");
    CHECK(moe_infer_exact_q4_publish_read_completion(
              engine, 1u, 620u, 4096u, 0) == 0,
          "overflow read1 completion");
    CHECK(moe_infer_exact_q4_publish_read_worker_start(engine, 0u, 611u) == 0,
          "overflow read0 start");
    CHECK(moe_infer_exact_q4_publish_read_completion(
              engine, 0u, 621u, 0u, -5) == 0,
          "overflow read0 completion");
    CHECK(moe_infer_exact_q4_finalize(engine) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE,
          "incomplete finalize appeared valid");
    CHECK(moe_infer_exact_q4_drain_and_abort_request(engine) == 0,
          "incomplete request drain and abort");
    CHECK(moe_infer_exact_q4_drain_and_abort_request(engine) == 0,
          "incomplete request repeated drain and abort");
    CHECK(moe_infer_exact_q4_test_no_inflight_owners(engine),
          "drain/abort retained an in-flight owner");

    DmlxExactQ4StageEvent stages[3];
    DmlxExactQ4ExpertReadEvent reads[2];
    DmlxExactQ4LatencySnapshot snapshot;
    CHECK(moe_infer_exact_q4_snapshot(
              engine, &snapshot, stages, 3u, reads, 2u) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE,
          "overflow snapshot appeared valid");
    CHECK(snapshot.valid == 0u && snapshot.incomplete == 1u &&
              snapshot.dropped_stages == UINT32_MAX &&
              snapshot.dropped_reads == UINT32_MAX &&
              snapshot.stage_count == 3u && snapshot.read_count == 2u,
          "overflow/saturation snapshot");
    CHECK(moe_infer_exact_q4_snapshot(
              engine, &snapshot, stages, 2u, reads, 2u) ==
              DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED,
          "small output capacity accepted");
    CHECK(moe_infer_exact_q4_finalize(engine) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE,
          "duplicate finalize accepted");
    CHECK(moe_infer_exact_q4_reset(engine) == 0, "overflow reset");
    CHECK(moe_infer_exact_q4_enable(engine, false) == 0, "disable");
    CHECK(engine->exact_q4 == NULL, "disable left fast-path state");
    CHECK(moe_infer_exact_q4_begin_request(engine, 12u) ==
              DMLX_EXACT_Q4_PRODUCER_DISABLED,
          "disabled request begin did work");
    CHECK(moe_infer_exact_q4_configure(engine, 2u, 1u) == 0,
          "safe reconfigure failed");
    CHECK(moe_infer_exact_q4_enable(engine, true) == 0,
          "enable after reconfigure");
    CHECK(moe_infer_exact_q4_deconfigure(engine) == 0,
          "deconfigure failed");
    CHECK(engine->exact_q4 == NULL && engine->exact_q4_storage == NULL,
          "deconfigure retained state");
    return 0;
}

static int test_deinit_aborts_unclaimed_reservation(void) {
    MoEInferEngine *engine = calloc(1u, sizeof(*engine));
    CHECK(engine != NULL, "allocate deinit engine");
    engine->initialized = true;
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer) {
        engine->packed_fd[layer] = -1;
    }
    CHECK(moe_infer_exact_q4_configure(engine, 1u, 1u) == 0,
          "configure deinit engine");
    CHECK(moe_infer_exact_q4_enable(engine, true) == 0,
          "enable deinit engine");
    CHECK(moe_infer_exact_q4_begin_request(engine, 1u) == 0,
          "begin deinit request");
    CHECK(moe_infer_exact_q4_begin_token(engine, 0u) == 0,
          "begin deinit token");
    DmlxExactQ4ExpertReadEvent read = read_event(1u, 0u, 1u);
    uint32_t ordinal = UINT32_MAX;
    CHECK(moe_infer_exact_q4_reserve_read(engine, &read, &ordinal) == 0,
          "reserve deinit read");
    CHECK(moe_infer_exact_q4_drain_and_abort_request(engine) == 0,
          "deinit request drain and abort");
    CHECK(moe_infer_exact_q4_test_no_inflight_owners(engine),
          "deinit drain retained an in-flight owner");
    moe_infer_deinit(engine);
    return 0;
}

static char *capture_telemetry_json(const MoEInferEngine *engine) {
    fflush(stderr);
    const int saved_stderr = dup(STDERR_FILENO);
    FILE *capture = tmpfile();
    if (saved_stderr < 0 || !capture ||
        dup2(fileno(capture), STDERR_FILENO) < 0) {
        if (saved_stderr >= 0) close(saved_stderr);
        if (capture) fclose(capture);
        return NULL;
    }
    moe_infer_telemetry_emit_json(
        engine, 100u, 120u, 120u, 150u, 1u, 1u, NULL, 0u,
        0u, 0u, 0u, 0u);
    fflush(stderr);
    (void)dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
    if (fseek(capture, 0, SEEK_END) != 0) {
        fclose(capture);
        return NULL;
    }
    const long length = ftell(capture);
    if (length < 0 || fseek(capture, 0, SEEK_SET) != 0) {
        fclose(capture);
        return NULL;
    }
    char *text = calloc((size_t)length + 1u, 1u);
    if (!text || fread(text, 1u, (size_t)length, capture) != (size_t)length) {
        free(text);
        fclose(capture);
        return NULL;
    }
    fclose(capture);
    return text;
}

typedef struct TelemetryResetThread {
    MoEInferEngine *engine;
} TelemetryResetThread;

static void *telemetry_reset_foreign_thread(void *opaque) {
    TelemetryResetThread *thread = (TelemetryResetThread *)opaque;
    moe_infer_telemetry_reset(thread->engine);
    return NULL;
}

static int test_gpu_timestamp_telemetry(void) {
    uint64_t rounded_ns = 0u;
    CHECK(moe_infer_gpu_timestamp_test_round_scaled_ns(
              1.5L, &rounded_ns) && rounded_ns == 2u,
          "GPU timestamp half-up rounding mismatch");
    const long double largest_representable_below_u64_limit =
        nextafterl(0x1p64L, 0.0L);
    CHECK(moe_infer_gpu_timestamp_test_round_scaled_ns(
              largest_representable_below_u64_limit, &rounded_ns) &&
              rounded_ns ==
                  (uint64_t)largest_representable_below_u64_limit,
          "largest representable in-range GPU duration rejected");
#if LDBL_MANT_DIG >= 64
    CHECK(moe_infer_gpu_timestamp_test_round_scaled_ns(
              (long double)UINT64_MAX, &rounded_ns) &&
              rounded_ns == UINT64_MAX,
          "exact UINT64_MAX GPU duration rejected");
#endif
    CHECK(!moe_infer_gpu_timestamp_test_round_scaled_ns(
              0x1p64L, &rounded_ns),
          "first representable out-of-range GPU duration accepted");

    MoEInferEngine disabled;
    memset(&disabled, 0, sizeof(disabled));
    disabled.initialized = true;
    moe_infer_telemetry_enable(&disabled, false);
    CHECK(disabled.gpu_timestamp_probe_storage == NULL,
          "disabled telemetry allocated GPU timestamp storage");

    MoEInferEngine enable_failed;
    memset(&enable_failed, 0, sizeof(enable_failed));
    moe_infer_telemetry_enable(&enable_failed, true);
    char *json = capture_telemetry_json(&enable_failed);
    CHECK(json && strstr(json,
              "\"status\":\"enable_failed\",\"total_seen\":0") != NULL,
          "telemetry enable failure did not fail closed");
    free(json);

    MoEInferEngine engine;
    memset(&engine, 0, sizeof(engine));
    engine.initialized = true;
    moe_infer_telemetry_enable(&engine, true);
    CHECK(engine.gpu_timestamp_probe_storage != NULL,
          "telemetry did not enable GPU timestamp probe");

    DmlxGpuTimestampSample sample = {
        .request_id = 1u,
        .layer = 1u,
        .kind = DMLX_GPU_CB_ATTENTION,
        .pre_commit_ns = 10u,
        .post_commit_ns = 20u,
        .post_wait_ns = 30u,
        .gpu_start_s = 1.0,
        .gpu_end_s = 1.25,
        .valid = 1u,
        .reason = DMLX_GPU_TIMESTAMP_VALID,
    };
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &sample) == 0,
          "append telemetry attention duration");
    sample.kind = DMLX_GPU_CB_ROUTED_MOE;
    sample.gpu_start_s = 1.0e-9;
    sample.gpu_end_s = 2.5e-9;
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &sample) == 0,
          "append telemetry routed-MoE duration");
    const uint32_t reasons[] = {
        DMLX_GPU_TIMESTAMP_SELECTOR_UNAVAILABLE,
        DMLX_GPU_TIMESTAMP_NONFINITE,
        DMLX_GPU_TIMESTAMP_ZERO,
        DMLX_GPU_TIMESTAMP_REVERSED,
        DMLX_GPU_TIMESTAMP_HOST_ORDER,
    };
    for (uint32_t index = 0u; index < 5u; ++index) {
        sample.kind = index & 1u ? DMLX_GPU_CB_ROUTED_MOE
                                 : DMLX_GPU_CB_ATTENTION;
        sample.valid = 0u;
        sample.reason = reasons[index];
        sample.pre_commit_ns = 10u;
        sample.post_commit_ns = 20u;
        sample.post_wait_ns = 30u;
        sample.gpu_start_s = 1.0;
        sample.gpu_end_s = 2.0;
        if (sample.reason == DMLX_GPU_TIMESTAMP_SELECTOR_UNAVAILABLE) {
            sample.gpu_start_s = 0.0;
            sample.gpu_end_s = 0.0;
        } else if (sample.reason == DMLX_GPU_TIMESTAMP_NONFINITE) {
            sample.gpu_start_s = NAN;
        } else if (sample.reason == DMLX_GPU_TIMESTAMP_ZERO) {
            sample.gpu_start_s = 0.0;
        } else if (sample.reason == DMLX_GPU_TIMESTAMP_REVERSED) {
            sample.gpu_end_s = sample.gpu_start_s;
        } else {
            sample.post_commit_ns = 9u;
        }
        CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &sample) == 0,
              "append telemetry invalid reason");
    }
    json = capture_telemetry_json(&engine);
    CHECK(json && strstr(json,
              "\"status\":\"available\",\"total_seen\":7,"
              "\"retained_count\":7,\"overflow_count\":0") != NULL,
          "available telemetry request accounting mismatch");
    CHECK(strstr(json,
              "\"attention\":{\"duration_ns_sum\":250000000,"
              "\"duration_sum_overflow\":false,\"valid_count\":1,"
              "\"selector_unavailable_count\":1,\"nonfinite_count\":0,"
              "\"zero_count\":1,\"reversed_count\":0,"
              "\"host_order_count\":1}") != NULL,
          "attention duration/reason aggregation mismatch");
    CHECK(strstr(json,
              "\"routed_moe\":{\"duration_ns_sum\":2,"
              "\"duration_sum_overflow\":false,\"valid_count\":1,"
              "\"selector_unavailable_count\":0,\"nonfinite_count\":1,"
              "\"zero_count\":0,\"reversed_count\":1,"
              "\"host_order_count\":0}") != NULL,
          "routed-MoE duration/reason aggregation mismatch");
    free(json);

    moe_infer_telemetry_reset(&engine);
    sample.valid = 1u;
    sample.reason = DMLX_GPU_TIMESTAMP_VALID;
    sample.pre_commit_ns = 10u;
    sample.post_commit_ns = 20u;
    sample.post_wait_ns = 30u;
    sample.kind = DMLX_GPU_CB_ATTENTION;
    sample.gpu_start_s = 1.0;
    sample.gpu_end_s = 2.0e10;
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &sample) == 0,
          "append duration conversion overflow");
    sample.kind = DMLX_GPU_CB_ROUTED_MOE;
    sample.gpu_end_s = 9.5e9;
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &sample) == 0 &&
          moe_infer_gpu_timestamp_probe_test_append(&engine, &sample) == 0,
          "append duration accumulation overflow");
    json = capture_telemetry_json(&engine);
    CHECK(json && strstr(json,
              "\"attention\":{\"duration_ns_sum\":null,"
              "\"duration_sum_overflow\":true,\"valid_count\":1") != NULL &&
          strstr(json,
              "\"routed_moe\":{\"duration_ns_sum\":null,"
              "\"duration_sum_overflow\":true,\"valid_count\":2") != NULL,
          "duration overflow did not emit null plus flag");
    free(json);

    moe_infer_telemetry_reset(&engine);
    sample.kind = DMLX_GPU_CB_ATTENTION;
    sample.gpu_end_s = 2.0;
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &sample) == 0,
          "append accounting corruption seed");
    moe_infer_gpu_timestamp_probe_test_corrupt_accounting(&engine, 2u, 0u);
    json = capture_telemetry_json(&engine);
    CHECK(json && strstr(json,
              "\"status\":\"invalid_snapshot\",\"total_seen\":0,"
              "\"retained_count\":0,\"overflow_count\":0") != NULL &&
          strstr(json,
              "\"attention\":{\"duration_ns_sum\":null,"
              "\"duration_sum_overflow\":false,\"valid_count\":0") != NULL,
          "invalid accounting snapshot did not fail closed");
    free(json);

    moe_infer_telemetry_reset(&engine);
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &sample) == 0,
          "append unknown-kind corruption seed");
    moe_infer_gpu_timestamp_probe_test_corrupt_sample(
        &engine, 0u, UINT16_MAX, DMLX_GPU_TIMESTAMP_VALID);
    json = capture_telemetry_json(&engine);
    CHECK(json && strstr(json, "\"status\":\"invalid_snapshot\"") != NULL,
          "unknown sample kind did not invalidate snapshot");
    free(json);

    moe_infer_telemetry_reset(&engine);
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &sample) == 0,
          "append unknown-reason corruption seed");
    moe_infer_gpu_timestamp_probe_test_corrupt_sample(
        &engine, 0u, DMLX_GPU_CB_ATTENTION, UINT32_MAX);
    json = capture_telemetry_json(&engine);
    CHECK(json && strstr(json, "\"status\":\"invalid_snapshot\"") != NULL,
          "unknown sample reason did not invalidate snapshot");
    free(json);

    moe_infer_telemetry_reset(&engine);
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &sample) == 0,
          "append consecutive-reset seed");
    moe_infer_telemetry_reset(&engine);
    json = capture_telemetry_json(&engine);
    CHECK(json && strstr(json,
              "\"status\":\"available\",\"total_seen\":0,"
              "\"retained_count\":0,\"overflow_count\":0") != NULL,
          "consecutive telemetry reset retained stale sample");
    free(json);

    TelemetryResetThread reset_thread = { .engine = &engine };
    pthread_t thread;
    CHECK(pthread_create(
              &thread, NULL, telemetry_reset_foreign_thread, &reset_thread) == 0 &&
          pthread_join(thread, NULL) == 0,
          "foreign telemetry reset thread failed");
    json = capture_telemetry_json(&engine);
    CHECK(json && strstr(json,
              "\"status\":\"reset_failed\",\"total_seen\":0") != NULL,
          "telemetry reset failure did not fail closed");
    CHECK(!moe_infer_gpu_timestamp_probe_recording_enabled(&engine) &&
          moe_infer_gpu_timestamp_probe_test_append(&engine, &sample) ==
              DMLX_GPU_TIMESTAMP_PROBE_DISABLED,
          "telemetry reset failure left timestamp instrumentation active");
    free(json);

    CHECK(moe_infer_gpu_timestamp_probe_deconfigure(&engine) == 0,
          "deconfigure telemetry test probe");
    return 0;
}

typedef struct GpuProbeForeignThread {
    MoEInferEngine *engine;
    DmlxGpuTimestampSample sample;
    int append_status;
    size_t snapshot_count;
    uint64_t total_seen;
    uint64_t overflow_count;
} GpuProbeForeignThread;

static void *gpu_probe_foreign_thread(void *opaque) {
    GpuProbeForeignThread *thread = (GpuProbeForeignThread *)opaque;
    thread->append_status = moe_infer_gpu_timestamp_probe_test_append(
        thread->engine, &thread->sample);
    thread->snapshot_count = moe_infer_gpu_timestamp_probe_snapshot(
        thread->engine, NULL, 0u, &thread->total_seen,
        &thread->overflow_count);
    return NULL;
}

static int test_gpu_timestamp_probe_storage(void) {
    MoEInferEngine engine;
    memset(&engine, 0, sizeof(engine));
    engine.initialized = true;

    DmlxGpuTimestampSample valid = {
        .sequence = UINT64_MAX,
        .request_id = 17u,
        .token_index = 3u,
        .layer = 5u,
        .kind = DMLX_GPU_CB_ATTENTION,
        .pre_commit_ns = 100u,
        .post_commit_ns = 110u,
        .post_wait_ns = 200u,
        .gpu_start_s = 1.25,
        .gpu_end_s = 1.50,
        .valid = 1u,
        .reason = DMLX_GPU_TIMESTAMP_VALID,
    };
    CHECK(!moe_infer_gpu_timestamp_probe_recording_enabled(&engine),
          "GPU timestamp probe enabled by default");
    CHECK(engine.gpu_timestamp_probe_storage == NULL,
          "GPU timestamp probe allocated by default");
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &valid) ==
              DMLX_GPU_TIMESTAMP_PROBE_DISABLED,
          "disabled GPU timestamp append did work");
    CHECK(moe_infer_gpu_timestamp_probe_set_enabled(&engine, true) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "GPU timestamp probe enable failed");
    CHECK(moe_infer_gpu_timestamp_probe_recording_enabled(&engine) &&
              engine.gpu_timestamp_probe_storage != NULL,
          "GPU timestamp probe storage not retained");
    void *const storage = engine.gpu_timestamp_probe_storage;
    CHECK(moe_infer_gpu_timestamp_probe_set_enabled(&engine, true) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK &&
              engine.gpu_timestamp_probe_storage == storage,
          "idempotent GPU timestamp enable replaced storage");
    GpuProbeForeignThread foreign = {
        .engine = &engine,
        .sample = valid,
        .append_status = 0,
        .snapshot_count = SIZE_MAX,
        .total_seen = UINT64_MAX,
        .overflow_count = UINT64_MAX,
    };
    pthread_t foreign_thread;
    CHECK(pthread_create(
              &foreign_thread, NULL, gpu_probe_foreign_thread, &foreign) == 0,
          "create foreign GPU timestamp owner test");
    CHECK(pthread_join(foreign_thread, NULL) == 0,
          "join foreign GPU timestamp owner test");
    CHECK(foreign.append_status == DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT &&
              foreign.snapshot_count == 0u && foreign.total_seen == 0u &&
              foreign.overflow_count == 0u,
          "foreign GPU timestamp owner accessed storage");

    DmlxGpuTimestampSample reason_case = valid;
    reason_case.valid = 0u;
    reason_case.reason = DMLX_GPU_TIMESTAMP_NONFINITE;
    reason_case.gpu_start_s = NAN;
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &reason_case) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "nonfinite GPU timestamp reason rejected");
    reason_case = valid;
    reason_case.valid = 0u;
    reason_case.reason = DMLX_GPU_TIMESTAMP_ZERO;
    reason_case.gpu_start_s = 0.0;
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &reason_case) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "zero GPU timestamp reason rejected");
    reason_case = valid;
    reason_case.valid = 0u;
    reason_case.reason = DMLX_GPU_TIMESTAMP_REVERSED;
    reason_case.gpu_end_s = reason_case.gpu_start_s;
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &reason_case) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "reversed GPU timestamp reason rejected");
    reason_case = valid;
    reason_case.valid = 0u;
    reason_case.reason = DMLX_GPU_TIMESTAMP_HOST_ORDER;
    reason_case.post_commit_ns = reason_case.pre_commit_ns - 1u;
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &reason_case) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "host-order GPU timestamp reason rejected");
    reason_case.reason = UINT32_MAX;
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &reason_case) ==
              DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT,
          "unknown GPU timestamp reason accepted");
    CHECK(moe_infer_gpu_timestamp_probe_reset(&engine) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "GPU timestamp reason-matrix reset failed");

    const struct {
        bool selectors;
        uint64_t pre_commit;
        uint64_t post_commit;
        uint64_t post_wait;
        double gpu_start;
        double gpu_end;
        uint32_t expected_valid;
        uint32_t expected_reason;
    } raw_cases[] = {
        { true, 100u, 110u, 200u, 1.0, 2.0, 1u,
          DMLX_GPU_TIMESTAMP_VALID },
        { false, 100u, 110u, 200u, NAN, NAN, 0u,
          DMLX_GPU_TIMESTAMP_SELECTOR_UNAVAILABLE },
        { true, 100u, 110u, 200u, NAN, 2.0, 0u,
          DMLX_GPU_TIMESTAMP_NONFINITE },
        { true, 100u, 110u, 200u, 0.0, 2.0, 0u,
          DMLX_GPU_TIMESTAMP_ZERO },
        { true, 100u, 110u, 200u, 2.0, 1.0, 0u,
          DMLX_GPU_TIMESTAMP_REVERSED },
        { false, 120u, 110u, 200u, 0.0, 0.0, 0u,
          DMLX_GPU_TIMESTAMP_HOST_ORDER },
    };
    for (uint32_t index = 0u;
         index < sizeof(raw_cases) / sizeof(raw_cases[0]); ++index) {
        CHECK(moe_infer_gpu_timestamp_probe_test_record_raw(
                  &engine, index & 1u ? DMLX_GPU_CB_ROUTED_MOE
                                     : DMLX_GPU_CB_ATTENTION,
                  (uint16_t)index, 99u,
                  raw_cases[index].pre_commit,
                  raw_cases[index].post_commit,
                  raw_cases[index].post_wait,
                  raw_cases[index].selectors,
                  raw_cases[index].gpu_start,
                  raw_cases[index].gpu_end) == DMLX_GPU_TIMESTAMP_PROBE_OK,
              "raw GPU timestamp classification failed");
    }
    DmlxGpuTimestampSample classified[
        sizeof(raw_cases) / sizeof(raw_cases[0])] = {0};
    uint64_t classified_total = 0u;
    uint64_t classified_overflow = 0u;
    CHECK(moe_infer_gpu_timestamp_probe_snapshot(
              &engine, classified,
              sizeof(classified) / sizeof(classified[0]),
              &classified_total, &classified_overflow) ==
              sizeof(classified) / sizeof(classified[0]) &&
              classified_total == sizeof(classified) / sizeof(classified[0]) &&
              classified_overflow == 0u,
          "classified GPU timestamp snapshot failed");
    for (uint32_t index = 0u;
         index < sizeof(raw_cases) / sizeof(raw_cases[0]); ++index) {
        CHECK(classified[index].sequence == index &&
                  classified[index].request_id == 0u &&
                  classified[index].token_index == 99u &&
                  classified[index].valid == raw_cases[index].expected_valid &&
                  classified[index].reason == raw_cases[index].expected_reason,
              "raw GPU timestamp classification mismatch");
    }
    CHECK(moe_infer_gpu_timestamp_probe_reset(&engine) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "GPU timestamp classifier reset failed");

    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &valid) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "valid GPU timestamp append failed");
    DmlxGpuTimestampSample invalid = valid;
    invalid.kind = DMLX_GPU_CB_ROUTED_MOE;
    invalid.layer = 6u;
    invalid.valid = 0u;
    invalid.reason = DMLX_GPU_TIMESTAMP_SELECTOR_UNAVAILABLE;
    invalid.gpu_start_s = 0.0;
    invalid.gpu_end_s = 0.0;
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &invalid) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "invalid-source GPU timestamp append failed");

    uint64_t total_seen = UINT64_MAX;
    uint64_t overflow = UINT64_MAX;
    CHECK(moe_infer_gpu_timestamp_probe_snapshot(
              &engine, NULL, 0u, &total_seen, &overflow) == 2u &&
              total_seen == 2u && overflow == 0u,
          "GPU timestamp snapshot query mismatch");
    DmlxGpuTimestampSample copied = {0};
    CHECK(moe_infer_gpu_timestamp_probe_snapshot(
              &engine, &copied, 1u, &total_seen, &overflow) == 1u &&
              copied.sequence == 0u && copied.request_id == 17u &&
              copied.kind == DMLX_GPU_CB_ATTENTION,
          "GPU timestamp ordered snapshot mismatch");
    DmlxGpuTimestampSample copied_again = {0};
    CHECK(moe_infer_gpu_timestamp_probe_snapshot(
              &engine, &copied_again, 1u, &total_seen, &overflow) == 1u &&
              memcmp(&copied, &copied_again, sizeof(copied)) == 0 &&
              total_seen == 2u && overflow == 0u,
          "GPU timestamp snapshot was destructive");

    DmlxGpuTimestampSample malformed = valid;
    malformed.valid = 1u;
    malformed.reason = DMLX_GPU_TIMESTAMP_REVERSED;
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &malformed) ==
              DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT,
          "malformed valid GPU timestamp accepted");
    CHECK(moe_infer_gpu_timestamp_probe_snapshot(
              &engine, NULL, 0u, &total_seen, &overflow) == 2u &&
              total_seen == 2u,
          "rejected GPU timestamp changed counters");

    for (uint32_t index = 2u;
         index < DMLX_GPU_TIMESTAMP_PROBE_CAPACITY; ++index) {
        valid.token_index = index;
        valid.kind = (index & 1u) ? DMLX_GPU_CB_ROUTED_MOE
                                  : DMLX_GPU_CB_ATTENTION;
        CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &valid) ==
                  DMLX_GPU_TIMESTAMP_PROBE_OK,
              "GPU timestamp fill failed");
    }
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &valid) ==
              DMLX_GPU_TIMESTAMP_PROBE_CAPACITY_EXCEEDED &&
              moe_infer_gpu_timestamp_probe_test_append(&engine, &valid) ==
              DMLX_GPU_TIMESTAMP_PROBE_CAPACITY_EXCEEDED,
          "GPU timestamp repeated overflow did not fail");
    CHECK(moe_infer_gpu_timestamp_probe_snapshot(
              &engine, NULL, 0u, &total_seen, &overflow) ==
              DMLX_GPU_TIMESTAMP_PROBE_CAPACITY &&
              total_seen == DMLX_GPU_TIMESTAMP_PROBE_CAPACITY + 2u &&
              overflow == 2u,
          "GPU timestamp overflow accounting mismatch");
    DmlxGpuTimestampSample *all_samples = calloc(
        DMLX_GPU_TIMESTAMP_PROBE_CAPACITY, sizeof(*all_samples));
    CHECK(all_samples != NULL, "allocate full GPU timestamp snapshot");
    CHECK(moe_infer_gpu_timestamp_probe_snapshot(
              &engine, all_samples, DMLX_GPU_TIMESTAMP_PROBE_CAPACITY,
              &total_seen, &overflow) == DMLX_GPU_TIMESTAMP_PROBE_CAPACITY,
          "full GPU timestamp snapshot failed");
    for (uint32_t index = 0u;
         index < DMLX_GPU_TIMESTAMP_PROBE_CAPACITY; ++index) {
        CHECK(all_samples[index].sequence == index,
              "GPU timestamp full snapshot sequence mismatch");
    }
    CHECK(all_samples[0].kind == DMLX_GPU_CB_ATTENTION &&
              all_samples[1].kind == DMLX_GPU_CB_ROUTED_MOE &&
              all_samples[DMLX_GPU_TIMESTAMP_PROBE_CAPACITY - 1u].kind ==
                  DMLX_GPU_CB_ROUTED_MOE,
          "GPU timestamp retained payload order mismatch");
    free(all_samples);

    CHECK(moe_infer_gpu_timestamp_probe_reset(&engine) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "GPU timestamp reset failed");
    CHECK(engine.gpu_timestamp_probe_storage == storage,
          "GPU timestamp reset replaced bounded storage");
    CHECK(moe_infer_gpu_timestamp_probe_snapshot(
              &engine, NULL, 0u, &total_seen, &overflow) == 0u &&
              total_seen == 0u && overflow == 0u,
          "GPU timestamp reset retained counters");
    CHECK(moe_infer_gpu_timestamp_probe_set_enabled(&engine, false) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK &&
              !moe_infer_gpu_timestamp_probe_recording_enabled(&engine),
          "GPU timestamp disable failed");
    CHECK(moe_infer_gpu_timestamp_probe_test_append(&engine, &valid) ==
              DMLX_GPU_TIMESTAMP_PROBE_DISABLED,
          "disabled retained GPU timestamp storage accepted append");
    CHECK(moe_infer_gpu_timestamp_probe_reset(&engine) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK &&
              engine.gpu_timestamp_probe_storage == storage,
          "disabled GPU timestamp reset failed");
    CHECK(moe_infer_gpu_timestamp_probe_set_enabled(&engine, true) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK &&
              engine.gpu_timestamp_probe_storage == storage,
          "GPU timestamp re-enable replaced storage");
    CHECK(moe_infer_gpu_timestamp_probe_deconfigure(&engine) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK &&
              engine.gpu_timestamp_probe == NULL &&
              engine.gpu_timestamp_probe_storage == NULL,
          "GPU timestamp deconfigure failed");
    CHECK(moe_infer_gpu_timestamp_probe_deconfigure(&engine) ==
              DMLX_GPU_TIMESTAMP_PROBE_DISABLED,
          "repeated GPU timestamp deconfigure status mismatch");
    CHECK(moe_infer_gpu_timestamp_probe_set_enabled(&engine, true) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK &&
              engine.gpu_timestamp_probe_storage != NULL,
          "GPU timestamp enable after deconfigure failed");
    CHECK(moe_infer_gpu_timestamp_probe_deconfigure(&engine) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "GPU timestamp final deconfigure failed");

    MoEInferEngine *heap_engine = calloc(1u, sizeof(*heap_engine));
    CHECK(heap_engine != NULL, "allocate GPU timestamp deinit engine");
    heap_engine->initialized = true;
    for (uint32_t layer = 0u; layer < N_LAYERS; ++layer)
        heap_engine->packed_fd[layer] = -1;
    CHECK(moe_infer_gpu_timestamp_probe_set_enabled(heap_engine, true) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK &&
              moe_infer_gpu_timestamp_probe_test_append(heap_engine, &valid) ==
                  DMLX_GPU_TIMESTAMP_PROBE_OK,
          "GPU timestamp deinit setup failed");
    moe_infer_deinit(heap_engine);
    return 0;
}

int main(void) {
    CHECK(test_gpu_timestamp_telemetry() == 0,
          "GPU timestamp telemetry suite");
    CHECK(test_gpu_timestamp_probe_storage() == 0,
          "GPU timestamp bounded storage suite");
    CHECK(test_disabled_micro_overhead() == 0,
          "disabled micro-overhead suite");
    MoEInferEngine engine;
    memset(&engine, 0, sizeof(engine));
    engine.initialized = true;
    CHECK(test_disabled_and_configuration(&engine) == 0,
          "disabled/configuration suite");
    CHECK(test_order_publication_and_reuse(&engine) == 0,
          "ordering/publication suite");
    CHECK(moe_infer_exact_q4_reset(&engine) == 0,
          "reset before overflow suite");
    CHECK(moe_infer_exact_q4_begin_request(&engine, 10u) == 0,
          "overflow request begin");
    CHECK(moe_infer_exact_q4_begin_token(&engine, 0u) == 0,
          "overflow token begin");
    CHECK(test_overflow_and_invalid_lifecycle(&engine) == 0,
          "overflow/lifecycle suite");
    CHECK(test_deinit_aborts_unclaimed_reservation() == 0,
          "deinit ownership suite");
    fprintf(stderr, "exact-Q4 producer self-test: PASS\n");
    return 0;
}
