#include "engine.h"

#include <assert.h>
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

int main(void) {
    MoEInferEngine engine = {0};
    engine.initialized = 1;
    DmlxDecodeTimelineEvent event = {0};
    DmlxDecodeTimelineSnapshot snapshot;
    memset(&snapshot, 0xff, sizeof(snapshot));

    assert(moe_infer_decode_timeline_now_ns() == 0u);
    assert(!moe_infer_decode_timeline_recording_enabled(&engine));
    assert(moe_infer_decode_timeline_configure(&engine, 16u) ==
        DMLX_DECODE_TIMELINE_DISABLED);
    assert(moe_infer_decode_timeline_enable(&engine, true) ==
        DMLX_DECODE_TIMELINE_DISABLED);
    assert(moe_infer_decode_timeline_begin_request(&engine) ==
        DMLX_DECODE_TIMELINE_DISABLED);
    assert(moe_infer_decode_timeline_begin_token(&engine, 0u) ==
        DMLX_DECODE_TIMELINE_DISABLED);
    assert(moe_infer_decode_timeline_append(&engine, &event) ==
        DMLX_DECODE_TIMELINE_DISABLED);
    assert(moe_infer_decode_timeline_finalize(&engine) ==
        DMLX_DECODE_TIMELINE_DISABLED);
    assert(moe_infer_decode_timeline_snapshot(
        &engine, &snapshot, &event, 1u) == DMLX_DECODE_TIMELINE_DISABLED);
    assert(snapshot.abi_version == 0u && snapshot.valid == 0u &&
        snapshot.events == NULL);
    assert(moe_infer_decode_timeline_reset(&engine) ==
        DMLX_DECODE_TIMELINE_DISABLED);
    assert(moe_infer_decode_timeline_deconfigure(&engine) ==
        DMLX_DECODE_TIMELINE_OK);
    puts("compiled-out decode timeline probe tests passed");
    return 0;
}
