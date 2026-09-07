#include "engine.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", m); return 1; } } while (0)

static void valid_samples(uint64_t samples[DMLX_ATTENTION_STAGE_SAMPLE_COUNT]) {
    for (uint32_t i = 0; i < DMLX_ATTENTION_STAGE_COUNT; ++i) {
        samples[2u * i] = 100u + 20u * i;
        samples[2u * i + 1u] = samples[2u * i] + 10u;
    }
}

int main(void) {
    MoEInferEngine engine;
    DmlxAttentionStageSnapshot snapshot;
    uint64_t samples[DMLX_ATTENTION_STAGE_SAMPLE_COUNT];
    memset(&engine, 0, sizeof(engine));

    moe_infer_attention_stage_snapshot(&engine, &snapshot);
    CHECK(snapshot.status == DMLX_ATTENTION_STAGE_UNSUPPORTED, "default-off status");
    CHECK(engine.attention_stage_probe_storage == NULL, "default-off allocated storage");

    CHECK(moe_infer_attention_stage_test_enable(&engine) == 0, "test enable");
    valid_samples(samples);
    CHECK(moe_infer_attention_stage_test_aggregate(&engine, samples,
        DMLX_ATTENTION_STAGE_SAMPLE_COUNT, true) == 0, "valid aggregate");
    moe_infer_attention_stage_snapshot(&engine, &snapshot);
    CHECK(snapshot.status == DMLX_ATTENTION_STAGE_AVAILABLE, "valid status");
    for (uint32_t i = 0; i < DMLX_ATTENTION_STAGE_COUNT; ++i) {
        CHECK(snapshot.duration_ns_sum[i] == 10u, "stage mapping/sum");
        CHECK(snapshot.valid_count[i] == 1u, "stage count");
    }
    CHECK(moe_infer_attention_stage_test_aggregate(&engine, samples,
        DMLX_ATTENTION_STAGE_SAMPLE_COUNT, true) == 0, "second aggregate");
    moe_infer_attention_stage_snapshot(&engine, &snapshot);
    CHECK(snapshot.valid_count[0] == 2u && snapshot.duration_ns_sum[13] == 20u,
        "consecutive aggregation");

    moe_infer_attention_stage_reset(&engine);
    moe_infer_attention_stage_snapshot(&engine, &snapshot);
    CHECK(snapshot.status == DMLX_ATTENTION_STAGE_INVALID_SAMPLE,
        "empty reset request fails closed at publication");

    valid_samples(samples); samples[0] = 0u;
    CHECK(moe_infer_attention_stage_test_aggregate(&engine, samples, 28u, true) != 0,
        "zero sample accepted");
    moe_infer_attention_stage_snapshot(&engine, &snapshot);
    CHECK(snapshot.status == DMLX_ATTENTION_STAGE_INVALID_SAMPLE, "zero status");

    moe_infer_attention_stage_reset(&engine);
    valid_samples(samples); samples[3] = samples[2];
    CHECK(moe_infer_attention_stage_test_aggregate(&engine, samples, 28u, true) != 0,
        "reversed pair accepted");

    moe_infer_attention_stage_reset(&engine);
    valid_samples(samples); samples[2] = samples[1] - 1u; samples[3] = samples[1] + 1u;
    CHECK(moe_infer_attention_stage_test_aggregate(&engine, samples, 28u, true) == 0,
        "cross-stage overlap rejected");
    moe_infer_attention_stage_snapshot(&engine, &snapshot);
    CHECK(snapshot.status == DMLX_ATTENTION_STAGE_AVAILABLE,
        "cross-stage overlap status");

    moe_infer_attention_stage_reset(&engine);
    valid_samples(samples);
    CHECK(moe_infer_attention_stage_test_aggregate(&engine, samples, 27u, true) != 0,
        "wrong result count accepted");

    moe_infer_attention_stage_reset(&engine);
    CHECK(moe_infer_attention_stage_test_aggregate(&engine, samples, 28u, false) != 0,
        "topology mismatch accepted");
    moe_infer_attention_stage_snapshot(&engine, &snapshot);
    CHECK(snapshot.status == DMLX_ATTENTION_STAGE_TOPOLOGY_MISMATCH,
        "topology status");

    moe_infer_attention_stage_reset(&engine);
    moe_infer_attention_stage_test_seed_sum(&engine, 0u, UINT64_MAX, 0u);
    CHECK(moe_infer_attention_stage_test_aggregate(&engine, samples, 28u, true) != 0,
        "sum overflow accepted");

    moe_infer_attention_stage_deconfigure(&engine);
    CHECK(engine.attention_stage_probe_storage == NULL &&
        engine.attention_stage_probe == NULL, "deconfigure reset");

    engine.initialized = true;
    CHECK(moe_infer_attention_stage_test_enable(&engine) == 0,
        "test re-enable before configure-off");
    CHECK(moe_infer_attention_stage_configure(&engine, false) == 0,
        "configure-off");
    CHECK(engine.attention_stage_probe_storage == NULL &&
        engine.attention_stage_probe == NULL,
        "configure-off retained counter resources");
    puts("attention stage timestamp tests PASS");
    return 0;
}
