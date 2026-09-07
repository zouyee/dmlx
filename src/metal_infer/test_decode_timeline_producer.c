#include "engine.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#undef assert
#define assert(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "assertion failed: %s (%s:%d)\n", #condition, __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)

static DmlxDecodeTimelineEvent event_for(
    uint8_t stage, uint16_t layer, uint64_t begin, uint64_t end) {
    DmlxDecodeTimelineEvent event = {0};
    event.layer = layer;
    event.stage = stage;
    event.status = DMLX_DECODE_TIMELINE_EVENT_COMPLETE;
    event.begin_ns = begin;
    event.encode_done_ns = begin;
    event.pre_commit_ns = begin;
    event.post_commit_ns = begin;
    event.post_wait_ns = end;
    event.end_ns = end;
    return event;
}

static void append_complete_token(MoEInferEngine *engine, uint32_t token) {
    assert(moe_infer_decode_timeline_begin_token(engine, token) ==
        DMLX_DECODE_TIMELINE_OK);
    uint64_t clock = 100u + (uint64_t)token * 10000u;
    for (uint16_t layer = 0; layer < 43; ++layer) {
        for (uint8_t stage = DMLX_DECODE_TIMELINE_ATTENTION_CB;
             stage <= DMLX_DECODE_TIMELINE_CB3_CB; ++stage) {
            const uint64_t end = stage == DMLX_DECODE_TIMELINE_IO_EXPOSED_WAIT
                ? clock : clock + 1u;
            DmlxDecodeTimelineEvent event = event_for(stage, layer, clock, end);
            assert(moe_infer_decode_timeline_append(engine, &event) ==
                DMLX_DECODE_TIMELINE_OK);
            clock += 2u;
        }
    }
    DmlxDecodeTimelineEvent logits = event_for(
        DMLX_DECODE_TIMELINE_LOGITS_CB, DMLX_DECODE_TIMELINE_NO_LAYER,
        clock, clock + 1u);
    assert(moe_infer_decode_timeline_append(engine, &logits) ==
        DMLX_DECODE_TIMELINE_OK);
    clock += 2u;
    DmlxDecodeTimelineEvent sample = event_for(
        DMLX_DECODE_TIMELINE_TOKEN_SAMPLE, DMLX_DECODE_TIMELINE_NO_LAYER,
        clock, clock + 1u);
    assert(moe_infer_decode_timeline_append(engine, &sample) ==
        DMLX_DECODE_TIMELINE_OK);
    DmlxDecodeTimelineEvent envelope = event_for(
        DMLX_DECODE_TIMELINE_TOKEN_FORWARD, DMLX_DECODE_TIMELINE_NO_LAYER,
        100u + (uint64_t)token * 10000u, clock + 2u);
    assert(moe_infer_decode_timeline_append(engine, &envelope) ==
        DMLX_DECODE_TIMELINE_OK);
}

static void test_lifecycle_and_reuse(void) {
    MoEInferEngine engine = {0};
    engine.initialized = 1;
    assert(moe_infer_decode_timeline_configure(&engine, 640u) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_enable(&engine, true) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_recording_enabled(&engine));
    assert(moe_infer_decode_timeline_begin_request(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_deconfigure(&engine) ==
        DMLX_DECODE_TIMELINE_BUSY);
    append_complete_token(&engine, 0u);
    append_complete_token(&engine, 1u);

    DmlxDecodeTimelineEvent copied[640];
    DmlxDecodeTimelineSnapshot snapshot;
    assert(moe_infer_decode_timeline_snapshot(
        &engine, &snapshot, copied, 640u) ==
        DMLX_DECODE_TIMELINE_INVALID_STATE);
    assert(snapshot.finalized == 0u && snapshot.valid == 0u);
    assert(moe_infer_decode_timeline_finalize(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_snapshot(
        &engine, &snapshot, copied, 640u) == DMLX_DECODE_TIMELINE_OK);
    assert(snapshot.valid == 1u && snapshot.count == 608u);
    for (uint32_t i = 0; i < snapshot.count; ++i)
        assert(copied[i].status == DMLX_DECODE_TIMELINE_EVENT_COMPLETE);

    assert(moe_infer_decode_timeline_reset(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_begin_request(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    append_complete_token(&engine, 0u);
    assert(moe_infer_decode_timeline_finalize(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_reset(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_enable(&engine, false) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(!moe_infer_decode_timeline_recording_enabled(&engine));
    assert(moe_infer_decode_timeline_deconfigure(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
}

static void test_invalid_schema_and_overflow_fail_closed(void) {
    MoEInferEngine engine = {0};
    engine.initialized = 1;
    assert(moe_infer_decode_timeline_configure(&engine, 1u) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_enable(&engine, true) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_begin_request(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_begin_token(&engine, 0u) ==
        DMLX_DECODE_TIMELINE_OK);
    DmlxDecodeTimelineEvent event = event_for(
        DMLX_DECODE_TIMELINE_ATTENTION_CB, 0u, 10u, 11u);
    assert(moe_infer_decode_timeline_append(&engine, &event) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_append(&engine, &event) ==
        DMLX_DECODE_TIMELINE_CAPACITY_EXCEEDED);
    assert(moe_infer_decode_timeline_finalize(&engine) ==
        DMLX_DECODE_TIMELINE_INVALID_STATE);
    assert(moe_infer_decode_timeline_reset(&engine) ==
        DMLX_DECODE_TIMELINE_OK);

    assert(moe_infer_decode_timeline_begin_request(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_begin_token(&engine, 0u) ==
        DMLX_DECODE_TIMELINE_OK);
    event = event_for(DMLX_DECODE_TIMELINE_LOGITS_CB, 0u, 20u, 21u);
    assert(moe_infer_decode_timeline_append(&engine, &event) ==
        DMLX_DECODE_TIMELINE_INVALID_ARGUMENT);
    assert(moe_infer_decode_timeline_finalize(&engine) ==
        DMLX_DECODE_TIMELINE_INVALID_STATE);
    assert(moe_infer_decode_timeline_reset(&engine) ==
        DMLX_DECODE_TIMELINE_OK);

    assert(moe_infer_decode_timeline_begin_request(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_begin_token(&engine, 0u) ==
        DMLX_DECODE_TIMELINE_OK);
    event = event_for(DMLX_DECODE_TIMELINE_TOKEN_SAMPLE,
        DMLX_DECODE_TIMELINE_NO_LAYER, 30u, 31u);
    event.encode_done_ns = 32u;
    assert(moe_infer_decode_timeline_append(&engine, &event) ==
        DMLX_DECODE_TIMELINE_INVALID_ARGUMENT);
    assert(moe_infer_decode_timeline_reset(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_deconfigure(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
}

typedef struct PublisherBarrier {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    bool acquired;
    bool release;
} PublisherBarrier;

typedef struct AppendThreadArgs {
    MoEInferEngine *engine;
    DmlxDecodeTimelineEvent event;
    int status;
} AppendThreadArgs;

static void block_after_publisher_acquire(void *context) {
    PublisherBarrier *barrier = context;
    assert(pthread_mutex_lock(&barrier->mutex) == 0);
    barrier->acquired = true;
    assert(pthread_cond_broadcast(&barrier->cond) == 0);
    while (!barrier->release)
        assert(pthread_cond_wait(&barrier->cond, &barrier->mutex) == 0);
    assert(pthread_mutex_unlock(&barrier->mutex) == 0);
}

static void *append_thread_main(void *context) {
    AppendThreadArgs *args = context;
    args->status = moe_infer_decode_timeline_append(args->engine, &args->event);
    return NULL;
}

static void test_publisher_lifetime_gate(void) {
    MoEInferEngine engine = {0};
    engine.initialized = 1;
    assert(moe_infer_decode_timeline_configure(&engine, 4u) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_enable(&engine, true) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_begin_request(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_begin_token(&engine, 0u) ==
        DMLX_DECODE_TIMELINE_OK);

    PublisherBarrier barrier = {
        .mutex = PTHREAD_MUTEX_INITIALIZER,
        .cond = PTHREAD_COND_INITIALIZER,
    };
    engine.decode_timeline_publisher_hook = block_after_publisher_acquire;
    engine.decode_timeline_publisher_hook_context = &barrier;
    AppendThreadArgs args = {
        .engine = &engine,
        .event = event_for(DMLX_DECODE_TIMELINE_ATTENTION_CB, 0u, 10u, 11u),
        .status = DMLX_DECODE_TIMELINE_INVALID_STATE,
    };
    pthread_t publisher;
    assert(pthread_create(&publisher, NULL, append_thread_main, &args) == 0);

    assert(pthread_mutex_lock(&barrier.mutex) == 0);
    while (!barrier.acquired)
        assert(pthread_cond_wait(&barrier.cond, &barrier.mutex) == 0);
    assert(pthread_mutex_unlock(&barrier.mutex) == 0);

    assert(moe_infer_decode_timeline_enable(&engine, false) ==
        DMLX_DECODE_TIMELINE_BUSY);
    assert(moe_infer_decode_timeline_finalize(&engine) ==
        DMLX_DECODE_TIMELINE_BUSY);
    assert(moe_infer_decode_timeline_reset(&engine) ==
        DMLX_DECODE_TIMELINE_BUSY);
    assert(moe_infer_decode_timeline_deconfigure(&engine) ==
        DMLX_DECODE_TIMELINE_BUSY);
    assert(moe_infer_decode_timeline_configure(&engine, 8u) ==
        DMLX_DECODE_TIMELINE_INVALID_STATE);
    assert(moe_infer_decode_timeline_recording_enabled(&engine));

    DmlxDecodeTimelineSnapshot snapshot;
    DmlxDecodeTimelineEvent copied[4];
    assert(moe_infer_decode_timeline_snapshot(
        &engine, &snapshot, copied, 4u) == DMLX_DECODE_TIMELINE_INVALID_STATE);
    assert(snapshot.active_publishers == 1u);
    assert(snapshot.count == 0u);
    assert(snapshot.incomplete == 0u);
    assert(snapshot.finalized == 0u);

    assert(pthread_mutex_lock(&barrier.mutex) == 0);
    barrier.release = true;
    assert(pthread_cond_broadcast(&barrier.cond) == 0);
    assert(pthread_mutex_unlock(&barrier.mutex) == 0);
    assert(pthread_join(publisher, NULL) == 0);
    assert(args.status == DMLX_DECODE_TIMELINE_OK);
    engine.decode_timeline_publisher_hook = NULL;
    engine.decode_timeline_publisher_hook_context = NULL;

    assert(moe_infer_decode_timeline_snapshot(
        &engine, &snapshot, copied, 4u) == DMLX_DECODE_TIMELINE_INVALID_STATE);
    assert(snapshot.active_publishers == 0u);
    assert(snapshot.count == 1u);
    assert(snapshot.incomplete == 0u);
    assert(copied[0].status == DMLX_DECODE_TIMELINE_EVENT_COMPLETE);
    assert(moe_infer_decode_timeline_reset(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_enable(&engine, false) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_deconfigure(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(moe_infer_decode_timeline_deconfigure(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    assert(pthread_cond_destroy(&barrier.cond) == 0);
    assert(pthread_mutex_destroy(&barrier.mutex) == 0);
}

static void test_configuration_bounds(void) {
    MoEInferEngine engine = {0};
    assert(moe_infer_decode_timeline_configure(&engine, 1u) ==
        DMLX_DECODE_TIMELINE_INVALID_ARGUMENT);
    engine.initialized = 1;
    assert(moe_infer_decode_timeline_configure(&engine, 0u) ==
        DMLX_DECODE_TIMELINE_INVALID_ARGUMENT);
    assert(moe_infer_decode_timeline_configure(
        &engine, DMLX_DECODE_TIMELINE_MAX_CAPACITY + 1u) ==
        DMLX_DECODE_TIMELINE_INVALID_ARGUMENT);
}

int main(void) {
    test_configuration_bounds();
    test_lifecycle_and_reuse();
    test_invalid_schema_and_overflow_fail_closed();
    test_publisher_lifetime_gate();
    puts("decode timeline producer tests passed");
    return 0;
}
