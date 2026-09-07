#include "engine.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(DmlxMlaPathIdentity) == 4, "identity ABI");
_Static_assert(sizeof(DmlxMlaPathEvent) == 16, "path event ABI");
_Static_assert(offsetof(DmlxMlaPathEvent, layer) == 12, "path layer ABI");
_Static_assert(sizeof(DmlxExactQ4MlaPathSnapshot) == 48, "path snapshot ABI");
_Static_assert(offsetof(DmlxExactQ4MlaPathSnapshot, paths) == 40, "path pointer ABI");

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "exact-Q4 stage self-test failed: %s\n", message); \
        return 1; \
    } \
} while (0)

static DmlxExactQ4StageEvent stage(uint16_t layer, uint16_t kind, uint64_t begin) {
    return (DmlxExactQ4StageEvent){
        .layer = layer,
        .stage = kind,
        .flags = kind == DMLX_EXACT_Q4_STAGE_ATTENTION_GPU_WAIT
            ? DMLX_EXACT_Q4_EVENT_GPU_TIME_UNKNOWN
            : DMLX_EXACT_Q4_EVENT_CPU_ACTIVE,
        .calls = 1,
        .begin_ns = begin,
        .end_ns = begin + 1,
    };
}

static int run_success(void) {
    MoEInferEngine engine = {0};
    engine.initialized = true;
    CHECK(moe_infer_exact_q4_configure(&engine, 4, 1) == 0, "configure");
    CHECK(moe_infer_exact_q4_enable(&engine, true) == 0, "enable");
    CHECK(moe_infer_exact_q4_begin_request(&engine, 7) == 0, "begin request");
    CHECK(moe_infer_exact_q4_begin_token(&engine, 3) == 0, "begin token");
    DmlxExactQ4StageEvent stages[2] = {
        stage(2, DMLX_EXACT_Q4_STAGE_ATTENTION_ENCODE, 10),
        stage(2, DMLX_EXACT_Q4_STAGE_ATTENTION_GPU_WAIT, 12),
    };
    DmlxMlaPathEvent path = {
        .layer = 2,
        .path = DMLX_MLA_PATH_PERSISTENT_EXTERNAL,
        .flags = DMLX_MLA_PATH_PERSISTENT_KV |
            DMLX_MLA_PATH_EXTERNAL_CB_REQUESTED |
            DMLX_MLA_PATH_EXTERNAL_CB_USED,
    };
    CHECK(moe_infer_exact_q4_commit_layer(&engine, stages, 2, &path) == 0,
          "commit layer");
    DmlxExactQ4StageEvent invalid_logits = stage(
        2, DMLX_EXACT_Q4_STAGE_LOGITS_SAMPLE, 14);
    uint32_t logits_ordinal = UINT32_MAX;
    CHECK(moe_infer_exact_q4_append_stage(
              &engine, &invalid_logits, &logits_ordinal) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT,
          "accepted layer-scoped logits/sample");
    DmlxExactQ4StageEvent invalid_layer = stage(
        UINT16_MAX, DMLX_EXACT_Q4_STAGE_ROUTING, 14);
    CHECK(moe_infer_exact_q4_append_stage(
              &engine, &invalid_layer, &logits_ordinal) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT,
          "accepted sentinel for layer-scoped stage");
    DmlxExactQ4StageEvent logits = stage(
        UINT16_MAX, DMLX_EXACT_Q4_STAGE_LOGITS_SAMPLE, 14);
    CHECK(moe_infer_exact_q4_append_stage(
              &engine, &logits, &logits_ordinal) == 0 && logits_ordinal == 2,
          "append token logits/sample");
    DmlxExactQ4MlaPathSnapshot live = {0};
    DmlxMlaPathEvent copied[1] = {0};
    CHECK(moe_infer_exact_q4_snapshot_mla_paths(&engine, &live, copied, 1) ==
              DMLX_EXACT_Q4_PRODUCER_INVALID_STATE,
          "live path snapshot accepted");
    CHECK(copied[0].flags == 0, "live path snapshot copied data");
    CHECK(moe_infer_exact_q4_finalize(&engine) == 0, "finalize");
    DmlxExactQ4StageEvent copied_stages[3] = {0};
    DmlxExactQ4ExpertReadEvent copied_reads[1] = {0};
    DmlxExactQ4LatencySnapshot latency = {0};
    CHECK(moe_infer_exact_q4_snapshot(
              &engine, &latency, copied_stages, 3, copied_reads, 1) == 0,
          "latency snapshot");
    CHECK(latency.valid == 1 && latency.stage_count == 3 &&
              copied_stages[2].request_ordinal == 7 &&
              copied_stages[2].token_ordinal == 3 &&
              copied_stages[2].layer == UINT16_MAX &&
              copied_stages[2].stage == DMLX_EXACT_Q4_STAGE_LOGITS_SAMPLE &&
              copied_stages[2].end_ns > copied_stages[2].begin_ns,
          "token logits/sample identity and duration");
    CHECK(moe_infer_exact_q4_snapshot_mla_paths(&engine, &live, copied, 1) == 0,
          "path snapshot");
    CHECK(live.valid == 1 && live.count == 1 && copied[0].request_ordinal == 7 &&
              copied[0].token_ordinal == 3 && copied[0].event_ordinal == 0 &&
              copied[0].layer == 2,
          "path identity/order");
    CHECK(moe_infer_exact_q4_reset(&engine) == 0, "reset");
    CHECK(moe_infer_exact_q4_deconfigure(&engine) == 0, "deconfigure");
    return 0;
}

static int run_atomic_failure(void) {
    MoEInferEngine engine = {0};
    engine.initialized = true;
    CHECK(moe_infer_exact_q4_configure(&engine, 1, 1) == 0, "configure small");
    CHECK(moe_infer_exact_q4_enable(&engine, true) == 0, "enable small");
    CHECK(moe_infer_exact_q4_begin_request(&engine, 1) == 0, "begin small request");
    CHECK(moe_infer_exact_q4_begin_token(&engine, 0) == 0, "begin small token");
    DmlxExactQ4StageEvent stages[2] = {
        stage(0, DMLX_EXACT_Q4_STAGE_ATTENTION_ENCODE, 1),
        stage(0, DMLX_EXACT_Q4_STAGE_ROUTING, 3),
    };
    DmlxMlaPathEvent path = {
        .layer = 0,
        .path = DMLX_MLA_PATH_FALLBACK_TEMP_KV,
        .flags = DMLX_MLA_PATH_FALLBACK,
    };
    CHECK(moe_infer_exact_q4_commit_layer(&engine, stages, 2, &path) ==
              DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED,
          "capacity preflight");
    CHECK(moe_infer_exact_q4_abort_request(&engine) == 0, "abort failed request");
    CHECK(moe_infer_exact_q4_reset(&engine) == 0, "reset failed request");
    CHECK(moe_infer_exact_q4_deconfigure(&engine) == 0, "deconfigure small");
    return 0;
}

static int run_identity_matrix(void) {
    const struct {
        bool kv;
        bool external;
        bool stage_time;
        int status;
        uint8_t path;
        uint8_t flags;
    } cases[] = {
        { true, true, false, 0, DMLX_MLA_PATH_PERSISTENT_EXTERNAL,
          DMLX_MLA_PATH_PERSISTENT_KV | DMLX_MLA_PATH_EXTERNAL_CB_REQUESTED |
              DMLX_MLA_PATH_EXTERNAL_CB_USED },
        { true, false, false, 0, DMLX_MLA_PATH_PERSISTENT_INTERNAL,
          DMLX_MLA_PATH_PERSISTENT_KV },
        { true, true, true, 0, DMLX_MLA_PATH_PERSISTENT_INTERNAL_SPLIT,
          DMLX_MLA_PATH_PERSISTENT_KV | DMLX_MLA_PATH_EXTERNAL_CB_REQUESTED |
              DMLX_MLA_PATH_STAGE_TIME_SPLIT },
        { false, false, false, 0, DMLX_MLA_PATH_FALLBACK_TEMP_KV,
          DMLX_MLA_PATH_FALLBACK },
        { false, true, false, -1, 0, 0 },
    };
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        DmlxMlaPathIdentity identity = { .path = 0xff, .flags = 0xff, .reserved = 0xffff };
        CHECK(mla_attention_select_bf16_identity(
                  cases[index].kv, cases[index].external,
                  cases[index].stage_time, &identity) == cases[index].status,
              "identity status");
        CHECK(identity.path == cases[index].path &&
                  identity.flags == cases[index].flags && identity.reserved == 0,
              "identity value/zero-on-failure");
    }
    return 0;
}

static int run_43_layer_order(void) {
    MoEInferEngine engine = {0};
    engine.initialized = true;
    CHECK(moe_infer_exact_q4_configure(&engine, N_LAYERS, 1) == 0,
          "configure 43 layers");
    CHECK(moe_infer_exact_q4_enable(&engine, true) == 0, "enable 43 layers");
    CHECK(moe_infer_exact_q4_begin_request(&engine, 9) == 0, "request 43 layers");
    CHECK(moe_infer_exact_q4_begin_token(&engine, 2) == 0, "token 43 layers");
    for (uint16_t layer = 0; layer < N_LAYERS; ++layer) {
        DmlxExactQ4StageEvent event = stage(
            layer, DMLX_EXACT_Q4_STAGE_MOE_GPU_WAIT, 100u + layer * 2u);
        DmlxMlaPathEvent path = {
            .layer = layer,
            .path = DMLX_MLA_PATH_PERSISTENT_INTERNAL,
            .flags = DMLX_MLA_PATH_PERSISTENT_KV,
        };
        CHECK(moe_infer_exact_q4_commit_layer(&engine, &event, 1, &path) == 0,
              "commit ordered layer");
    }
    CHECK(moe_infer_exact_q4_finalize(&engine) == 0, "finalize 43 layers");
    DmlxMlaPathEvent paths[N_LAYERS] = {0};
    DmlxExactQ4MlaPathSnapshot snapshot = {0};
    CHECK(moe_infer_exact_q4_snapshot_mla_paths(
              &engine, &snapshot, paths, N_LAYERS) == 0,
          "snapshot 43 layers");
    for (uint16_t layer = 0; layer < N_LAYERS; ++layer) {
        CHECK(paths[layer].layer == layer && paths[layer].event_ordinal == layer,
              "43-layer path ordering");
    }
    CHECK(moe_infer_exact_q4_reset(&engine) == 0, "reset 43 layers");
    CHECK(moe_infer_exact_q4_deconfigure(&engine) == 0, "deconfigure 43 layers");
    return 0;
}

int main(void) {
    CHECK(moe_infer_exact_q4_test_run_layer_transaction_suite() == 0,
          "production layer transaction suite");
    CHECK(run_identity_matrix() == 0, "identity matrix suite");
    CHECK(run_success() == 0, "success suite");
    CHECK(run_atomic_failure() == 0, "atomic failure suite");
    CHECK(run_43_layer_order() == 0, "43-layer order suite");
    puts("exact-Q4 stage self-test: PASS");
    return 0;
}
