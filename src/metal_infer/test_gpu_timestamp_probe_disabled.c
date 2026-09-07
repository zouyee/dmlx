#include "engine.h"
#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

int main(void) {
    MoEInferEngine engine;
    memset(&engine, 0, sizeof(engine));
    engine.initialized = true;
    engine.gpu_timestamp_probe = (void *)1;
    engine.gpu_timestamp_probe_storage = (void *)1;

    CHECK(moe_infer_gpu_timestamp_probe_set_enabled(&engine, true) ==
              DMLX_GPU_TIMESTAMP_PROBE_DISABLED,
          "compiled-out enable did not report disabled");
    CHECK(!moe_infer_gpu_timestamp_probe_recording_enabled(&engine),
          "compiled-out probe reported recording enabled");
    CHECK(engine.gpu_timestamp_probe == NULL &&
              engine.gpu_timestamp_probe_storage == NULL,
          "compiled-out enable did not clear stale state");

    uint64_t total = UINT64_MAX;
    uint64_t overflow = UINT64_MAX;
    DmlxGpuTimestampSample sample;
    memset(&sample, 0xa5, sizeof(sample));
    CHECK(moe_infer_gpu_timestamp_probe_snapshot(
              &engine, &sample, 1u, &total, &overflow) == 0u &&
              total == 0u && overflow == 0u,
          "compiled-out snapshot was not empty");
    CHECK(moe_infer_gpu_timestamp_probe_reset(&engine) ==
              DMLX_GPU_TIMESTAMP_PROBE_DISABLED,
          "compiled-out reset status mismatch");
    CHECK(moe_infer_gpu_timestamp_probe_set_enabled(&engine, false) ==
              DMLX_GPU_TIMESTAMP_PROBE_OK,
          "compiled-out disable status mismatch");
    CHECK(moe_infer_gpu_timestamp_probe_deconfigure(&engine) ==
              DMLX_GPU_TIMESTAMP_PROBE_DISABLED,
          "compiled-out deconfigure status mismatch");
    CHECK(moe_infer_gpu_timestamp_probe_set_enabled(NULL, true) ==
              DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT &&
              moe_infer_gpu_timestamp_probe_reset(NULL) ==
              DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT &&
              moe_infer_gpu_timestamp_probe_deconfigure(NULL) ==
              DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT,
          "compiled-out invalid argument status mismatch");

    fprintf(stderr, "GPU timestamp compiled-out self-test: PASS\n");
    return 0;
}
