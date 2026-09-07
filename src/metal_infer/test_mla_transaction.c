#include "engine.h"
#include "dspark_engine.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(NativeBatchConfig) == 40, "NativeBatchConfig ABI");
_Static_assert(sizeof(NativeBatchWorkspace) == 88, "NativeBatchWorkspace ABI");
_Static_assert(offsetof(NativeBatchWorkspace, metal_buffer) == 56,
               "NativeBatchWorkspace Metal buffer offset");
_Static_assert(offsetof(NativeBatchWorkspace, metal_buffer_release) == 80,
               "NativeBatchWorkspace Metal release offset");
_Static_assert(sizeof(NativeKVStagingView) == 48, "NativeKVStagingView ABI");
_Static_assert(sizeof(NativeKVTransaction) == 240, "NativeKVTransaction ABI");
_Static_assert(offsetof(NativeKVTransaction, completed_layer_mask) == 24,
               "NativeKVTransaction mask offset");
_Static_assert(offsetof(NativeKVTransaction, original_kv_lengths) == 32,
               "NativeKVTransaction lengths offset");

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "MLA transaction self-test failed: %s\n", message); \
        return 1; \
    } \
} while (0)

static int workspace_metal_release_calls = 0;

static void test_workspace_metal_release(void *metal_buffer) {
    if (metal_buffer == (void *)(uintptr_t)1u) {
        workspace_metal_release_calls++;
    }
}

static void set_lengths(MoEInferEngine *engine, int length) {
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        engine->kv_cache[layer].len = length;
    }
}

static int lengths_equal(const MoEInferEngine *engine, int length) {
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        if (engine->kv_cache[layer].len != length) return 0;
    }
    return 1;
}

static int stage_all_layers(
    MoEInferEngine *engine,
    NativeKVTransaction *transaction) {
    const int staged_end = (int)transaction->start_position +
        (int)transaction->staged_tokens;
    set_lengths(engine, staged_end);
    for (uint32_t layer = 0; layer < N_LAYERS; ++layer) {
        if (transaction->phase == NATIVE_BATCH_PHASE_VERIFY) {
            for (uint32_t row = 0; row < transaction->staged_tokens; ++row) {
                if (moe_infer_kv_transaction_record_recurrent_row(
                        engine, transaction, layer, row) != NATIVE_BATCH_OK) {
                    return 0;
                }
            }
        }
        if (moe_infer_kv_transaction_mark_layer_staged(
                engine, transaction, layer, transaction->staged_tokens) !=
            NATIVE_BATCH_OK) {
            return 0;
        }
    }
    return 1;
}

typedef struct RecurrentFixture {
    float *state_kv;
    float *state_score;
    float *comp_kv;
    float *idx_state_kv;
    float *idx_state_score;
    float *idx_comp_kv;
} RecurrentFixture;

static int recurrent_fixture_init(
    MoEInferEngine *engine, RecurrentFixture *fixture) {
    const size_t comp_rows = MAX_COMP_BLOCKS;
    const size_t idx_rows = MAX_COMP_BLOCKS;
    *fixture = (RecurrentFixture){
        .state_kv = (float *)calloc(8u * 4u, sizeof(float)),
        .state_score = (float *)calloc(8u * 4u, sizeof(float)),
        .comp_kv = (float *)calloc(comp_rows * COMP_HEAD_DIM, sizeof(float)),
        .idx_state_kv = (float *)calloc(8u * 256u, sizeof(float)),
        .idx_state_score = (float *)calloc(8u * 256u, sizeof(float)),
        .idx_comp_kv = (float *)calloc(idx_rows * IDX_HEAD_DIM, sizeof(float)),
    };
    if (!fixture->state_kv || !fixture->state_score || !fixture->comp_kv ||
        !fixture->idx_state_kv || !fixture->idx_state_score ||
        !fixture->idx_comp_kv) return 0;
    engine->comp_state[0] = (CompressorState){
        .state_kv = fixture->state_kv,
        .state_score = fixture->state_score,
        .ratio = 4u,
        .out_dim = 4u,
        .comp_kv = fixture->comp_kv,
        .n_comp = 513u,
        .idx_state_kv = fixture->idx_state_kv,
        .idx_state_score = fixture->idx_state_score,
        .idx_comp_kv = fixture->idx_comp_kv,
        .n_idx_comp = 7u,
    };
    return 1;
}

static void recurrent_fixture_reset(
    MoEInferEngine *engine, RecurrentFixture *fixture) {
    for (size_t index = 0; index < 8u * 4u; ++index) {
        fixture->state_kv[index] = 1000.0f + (float)index;
        fixture->state_score[index] = 2000.0f + (float)index;
    }
    for (size_t index = 0; index < 520u * COMP_HEAD_DIM; ++index) {
        fixture->comp_kv[index] = 3000.0f + (float)index;
    }
    for (size_t index = 0; index < 8u * 256u; ++index) {
        fixture->idx_state_kv[index] = 4000.0f + (float)index;
        fixture->idx_state_score[index] = 7000.0f + (float)index;
    }
    for (size_t index = 0; index < 16u * IDX_HEAD_DIM; ++index) {
        fixture->idx_comp_kv[index] = 9000.0f + (float)index;
    }
    engine->comp_state[0].n_comp = 513u;
    engine->comp_state[0].n_idx_comp = 7u;
}

static void recurrent_fixture_deinit(
    MoEInferEngine *engine, RecurrentFixture *fixture) {
    free(fixture->state_kv);
    free(fixture->state_score);
    free(fixture->comp_kv);
    free(fixture->idx_state_kv);
    free(fixture->idx_state_score);
    free(fixture->idx_comp_kv);
    memset(fixture, 0, sizeof(*fixture));
    memset(&engine->comp_state[0], 0, sizeof(engine->comp_state[0]));
}

static int simulate_recurrent_row(
    MoEInferEngine *engine,
    NativeKVTransaction *transaction,
    uint32_t row) {
    CompressorState *state = &engine->comp_state[0];
    const uint32_t position = transaction->start_position + row;
    const uint32_t slot = state->ratio + position % state->ratio;
    for (uint32_t dim = 0; dim < state->out_dim; ++dim) {
        state->state_kv[(size_t)slot * state->out_dim + dim] =
            100.0f + 10.0f * (float)row + (float)dim;
        state->state_score[(size_t)slot * state->out_dim + dim] =
            200.0f + 10.0f * (float)row + (float)dim;
    }
    if ((position + 1u) % state->ratio == 0u) {
        for (uint32_t dim = 0; dim < COMP_HEAD_DIM; ++dim) {
            state->comp_kv[(size_t)state->n_comp * COMP_HEAD_DIM + dim] =
                500.0f + (float)dim;
        }
        state->n_comp++;
        memcpy(state->state_kv,
            state->state_kv + (size_t)state->ratio * state->out_dim,
            (size_t)state->ratio * state->out_dim * sizeof(float));
        memcpy(state->state_score,
            state->state_score + (size_t)state->ratio * state->out_dim,
            (size_t)state->ratio * state->out_dim * sizeof(float));
    }

    const uint32_t idx_slot = 4u + position % 4u;
    for (uint32_t dim = 0; dim < 256u; ++dim) {
        state->idx_state_kv[(size_t)idx_slot * 256u + dim] =
            600.0f + 10.0f * (float)row + (float)dim;
        state->idx_state_score[(size_t)idx_slot * 256u + dim] =
            900.0f + 10.0f * (float)row + (float)dim;
    }
    if ((position + 1u) % 4u == 0u) {
        for (uint32_t dim = 0; dim < IDX_HEAD_DIM; ++dim) {
            state->idx_comp_kv[(size_t)state->n_idx_comp * IDX_HEAD_DIM + dim] =
                1200.0f + (float)dim;
        }
        state->n_idx_comp++;
        memcpy(state->idx_state_kv,
            state->idx_state_kv + (size_t)4u * 256u,
            (size_t)4u * 256u * sizeof(float));
        memcpy(state->idx_state_score,
            state->idx_state_score + (size_t)4u * 256u,
            (size_t)4u * 256u * sizeof(float));
    }
    return moe_infer_kv_transaction_record_recurrent_row(
        engine, transaction, 0u, row) == NATIVE_BATCH_OK;
}

static int run_recurrent_prefix_case(
    MoEInferEngine *engine,
    RecurrentFixture *fixture,
    const NativeBatchConfig *config,
    NativeBatchWorkspace *workspace,
    uint32_t accepted) {
    recurrent_fixture_reset(engine, fixture);
    set_lengths(engine, 3);
    const uint32_t positions[] = {3, 4, 5};
    NativeKVTransaction transaction;
    moe_infer_kv_transaction_init(&transaction);
    CHECK(moe_infer_kv_transaction_begin(
              engine, config, NATIVE_BATCH_PHASE_VERIFY,
              positions, 3u, workspace, &transaction) == NATIVE_BATCH_OK,
          "begin recurrent verify transaction");
    for (uint32_t row = 0u; row < 3u; ++row) {
        CHECK(simulate_recurrent_row(engine, &transaction, row),
              "record recurrent verify row");
    }
    set_lengths(engine, 6);
    CHECK(moe_infer_kv_transaction_mark_layer_staged(
              engine, &transaction, 0u, 3u) == NATIVE_BATCH_OK,
          "mark recurrent layer staged");
    for (uint32_t layer = 1u; layer < N_LAYERS; ++layer) {
        for (uint32_t row = 0u; row < 3u; ++row) {
            CHECK(moe_infer_kv_transaction_record_recurrent_row(
                      engine, &transaction, layer, row) == NATIVE_BATCH_OK,
                  "record empty recurrent layer row");
        }
        CHECK(moe_infer_kv_transaction_mark_layer_staged(
                  engine, &transaction, layer, 3u) == NATIVE_BATCH_OK,
              "mark empty recurrent layer staged");
    }
    CHECK(moe_infer_kv_transaction_commit(
              engine, &transaction, accepted) == NATIVE_BATCH_OK,
          "commit recurrent prefix");
    CHECK(lengths_equal(engine, 3 + (int)accepted),
          "recurrent prefix KV visibility");
    CHECK(engine->comp_state[0].n_comp == 513u + (accepted > 0u),
          "recurrent main count");
    CHECK(engine->comp_state[0].n_idx_comp == 7u + (accepted > 0u),
          "recurrent indexer count");
    CHECK(fixture->state_kv[4u * 4u] ==
              (accepted >= 2u ? 110.0f : 1016.0f),
          "middle/rejected main row visibility");
    CHECK(fixture->state_kv[5u * 4u] ==
              (accepted >= 3u ? 120.0f : 1020.0f),
          "tail/rejected main row visibility");
    CHECK(fixture->comp_kv[513u * COMP_HEAD_DIM] ==
              (accepted > 0u ? 500.0f : 3000.0f + 513.0f * COMP_HEAD_DIM),
          "main compressed output prefix");
    CHECK(fixture->idx_comp_kv[7u * IDX_HEAD_DIM] ==
              (accepted > 0u ? 1200.0f : 9000.0f + 7.0f * IDX_HEAD_DIM),
          "indexer compressed output prefix");
    CHECK(transaction.verify_state == NULL && transaction.active == 0u,
          "recurrent journal release");
    return 0;
}

static int begin_recurrent_with_journal_budget(
    MoEInferEngine *engine,
    const NativeBatchConfig *base_config,
    NativeBatchWorkspace *workspace,
    uint64_t journal_budget_bytes) {
    if (journal_budget_bytes > UINT64_MAX - workspace->allocated_bytes) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    NativeBatchConfig config = *base_config;
    config.workspace_budget_bytes =
        workspace->allocated_bytes + journal_budget_bytes;
    const uint32_t positions[] = {3u, 4u, 5u};
    NativeKVTransaction transaction;
    moe_infer_kv_transaction_init(&transaction);
    const int status = moe_infer_kv_transaction_begin(
        engine, &config, NATIVE_BATCH_PHASE_VERIFY,
        positions, 3u, workspace, &transaction);
    if (status == NATIVE_BATCH_OK &&
        moe_infer_kv_transaction_abort(engine, &transaction) != NATIVE_BATCH_OK) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    return status;
}

static int test_hca_verify_prefix(
    MoEInferEngine *engine,
    const NativeBatchConfig *config,
    NativeBatchWorkspace *workspace) {
    const uint32_t ratio = 128u;
    const uint32_t out_dim = 4u;
    const size_t state_elements = (size_t)2u * ratio * out_dim;
    float *state_kv = (float *)malloc(state_elements * sizeof(float));
    float *state_score = (float *)malloc(state_elements * sizeof(float));
    float *comp_kv = (float *)malloc(
        (size_t)3u * COMP_HEAD_DIM * sizeof(float));
    CHECK(state_kv && state_score && comp_kv,
          "HCA verify fixture allocation");
    for (size_t index = 0u; index < state_elements; ++index) {
        state_kv[index] = 1000.0f + (float)index;
        state_score[index] = 2000.0f + (float)index;
    }
    for (size_t index = 0u; index < (size_t)3u * COMP_HEAD_DIM; ++index) {
        comp_kv[index] = 3000.0f + (float)index;
    }
    engine->comp_state[0] = (CompressorState){
        .state_kv = state_kv,
        .state_score = state_score,
        .ratio = ratio,
        .out_dim = out_dim,
        .comp_kv = comp_kv,
    };

    set_lengths(engine, 5);
    const uint32_t positions[] = {5u, 6u, 7u};
    NativeKVTransaction transaction;
    moe_infer_kv_transaction_init(&transaction);
    CHECK(moe_infer_kv_transaction_begin(
              engine, config, NATIVE_BATCH_PHASE_VERIFY,
              positions, 3u, workspace, &transaction) == NATIVE_BATCH_OK,
          "begin HCA verify transaction");
    for (uint32_t row = 0u; row < 3u; ++row) {
        const uint32_t slot = positions[row] % ratio;
        for (uint32_t dim = 0u; dim < out_dim; ++dim) {
            state_kv[(size_t)slot * out_dim + dim] =
                100.0f + 10.0f * (float)row + (float)dim;
            state_score[(size_t)slot * out_dim + dim] =
                200.0f + 10.0f * (float)row + (float)dim;
        }
        CHECK(moe_infer_kv_transaction_record_recurrent_row(
                  engine, &transaction, 0u, row) == NATIVE_BATCH_OK,
              "record HCA verify row without indexer journal");
    }
    set_lengths(engine, 8);
    CHECK(moe_infer_kv_transaction_mark_layer_staged(
              engine, &transaction, 0u, 3u) == NATIVE_BATCH_OK,
          "mark HCA layer staged");
    for (uint32_t layer = 1u; layer < N_LAYERS; ++layer) {
        for (uint32_t row = 0u; row < 3u; ++row) {
            CHECK(moe_infer_kv_transaction_record_recurrent_row(
                      engine, &transaction, layer, row) == NATIVE_BATCH_OK,
                  "record empty HCA peer layer row");
        }
        CHECK(moe_infer_kv_transaction_mark_layer_staged(
                  engine, &transaction, layer, 3u) == NATIVE_BATCH_OK,
              "mark empty HCA peer layer staged");
    }
    CHECK(moe_infer_kv_transaction_commit(
              engine, &transaction, 2u) == NATIVE_BATCH_OK,
          "commit HCA verify prefix");
    CHECK(lengths_equal(engine, 7), "HCA verify prefix KV visibility");
    CHECK(state_kv[(size_t)5u * out_dim] == 100.0f &&
              state_kv[(size_t)6u * out_dim] == 110.0f &&
              state_kv[(size_t)7u * out_dim] == 1000.0f + 7.0f * out_dim,
          "HCA verify prefix rolling-slot reconciliation");
    CHECK(transaction.verify_state == NULL && transaction.active == 0u,
          "HCA verify journal release");

    memset(&engine->comp_state[0], 0, sizeof(engine->comp_state[0]));
    free(comp_kv);
    free(state_score);
    free(state_kv);
    return 0;
}

static int test_dspark_commit_atomicity(void) {
    DSparkEngine *dspark = (DSparkEngine *)calloc(1u, sizeof(*dspark));
    MoEInferEngine *target = (MoEInferEngine *)calloc(1u, sizeof(*target));
    float *hidden3 = (float *)calloc((size_t)3u * 2u * DIM, sizeof(float));
    float *main_hidden = (float *)malloc((size_t)3u * DIM * sizeof(float));
    uint16_t *kv_storage[DSPARK_N_LAYERS] = {0};
    CHECK(dspark && target && hidden3 && main_hidden,
          "DSpark transaction fixture allocation");

    for (size_t index = 0u; index < (size_t)3u * DIM; ++index) {
        main_hidden[index] = 31.25f;
    }
    for (int layer = 0; layer < DSPARK_N_LAYERS; ++layer) {
        kv_storage[layer] = (uint16_t *)malloc(
            (size_t)5u * KV_LORA_RANK * sizeof(uint16_t));
        CHECK(kv_storage[layer], "DSpark KV fixture allocation");
        for (size_t index = 0u; index < (size_t)5u * KV_LORA_RANK; ++index) {
            kv_storage[layer][index] = (uint16_t)0x5a5au;
        }
        dspark->kv_cache[layer].main_pos = 4;
        dspark->kv_cache[layer].main_len = 4;
    }
    dspark->initialized = true;
    dspark->buf_main_hidden = main_hidden;

    CHECK(dspark_commit_accepted(dspark, hidden3, 2, 1, 4) ==
              DSPARK_COMMIT_INVALID_STATE,
          "partial DSpark head state accepted");
    CHECK(dspark->kv_cache[0].main_pos == 4 &&
              dspark->kv_cache[0].main_len == 4 &&
              main_hidden[0] == 31.25f,
          "partial DSpark head validation mutated state");

    float dependency_sentinel = 1.0f;
    dspark->head.main_proj_f32 = &dependency_sentinel;
    dspark->head.main_norm = &dependency_sentinel;
    for (int layer = 0; layer < DSPARK_N_LAYERS; ++layer) {
        dspark->layers[layer].attn.wkv_f32 = &dependency_sentinel;
        dspark->layers[layer].attn.kv_norm = &dependency_sentinel;
        dspark->kv_cache[layer].main_kv = kv_storage[layer];
    }
    dspark->kv_cache[1].main_kv = NULL;
    CHECK(dspark_commit_accepted(dspark, hidden3, 2, 1, 4) ==
              DSPARK_COMMIT_INVALID_STATE,
          "partial DSpark KV state accepted");
    CHECK(dspark->kv_cache[0].main_pos == 4 &&
              dspark->kv_cache[1].main_pos == 4 &&
              main_hidden[0] == 31.25f &&
              kv_storage[0][0] == (uint16_t)0x5a5au,
          "partial DSpark KV validation mutated state");
    dspark->kv_cache[1].main_kv = kv_storage[1];

    target->dspark_engine = dspark;
    target->dspark_hidden3 = hidden3;
    target->dspark_hidden3_n = 2;
    dspark_test_fail_next_commit_allocation();
    CHECK(moe_infer_dspark_commit(target, 1, 4) ==
              NATIVE_BATCH_CAPACITY_EXCEEDED,
          "DSpark allocation failure status mapping");
    CHECK(main_hidden[0] == 31.25f,
          "DSpark allocation failure mutated hidden state");
    for (int layer = 0; layer < DSPARK_N_LAYERS; ++layer) {
        CHECK(dspark->kv_cache[layer].main_pos == 4 &&
                  dspark->kv_cache[layer].main_len == 4,
              "DSpark allocation failure mutated KV cursor");
        for (size_t index = 0u; index < (size_t)5u * KV_LORA_RANK; ++index) {
            CHECK(kv_storage[layer][index] == (uint16_t)0x5a5au,
                  "DSpark allocation failure mutated KV data");
        }
        free(kv_storage[layer]);
    }
    free(main_hidden);
    free(hidden3);
    free(target);
    free(dspark);
    return 0;
}

int main(void) {
    MoEInferEngine *engine = (MoEInferEngine *)calloc(1, sizeof(*engine));
    MoEInferEngine *other_engine = (MoEInferEngine *)calloc(1, sizeof(*other_engine));
    CHECK(engine && other_engine, "engine allocation");
    CHECK(test_dspark_commit_atomicity() == 0,
          "DSpark commit atomicity tests");

    NativeBatchConfig config = {
        .max_prefill_chunk = 8,
        .max_verify_chunk = 5,
        .workspace_budget_bytes = 1024 * 1024,
    };
    CHECK(moe_infer_batch_config_validate(&config) == NATIVE_BATCH_OK, "valid config");
    config.enable_batch_mla = 2;
    CHECK(moe_infer_batch_config_validate(&config) == NATIVE_BATCH_INVALID_ARGUMENT,
          "non-boolean flag accepted");
    config.enable_batch_mla = 0;

    NativeBatchWorkspace workspace = {0};
    CHECK(moe_infer_batch_workspace_init(&workspace, 4, 512, 1024 * 1024) == NATIVE_BATCH_OK,
          "valid workspace");
    CHECK(workspace.capacity_tokens == 4 && workspace.storage &&
              workspace.high_water_bytes == 0,
          "owned workspace metadata");
    void *first = moe_infer_batch_workspace_reserve(&workspace, 128, 64);
    CHECK(first && ((uintptr_t)first & 63u) == 0u, "aligned workspace reserve");
    CHECK(moe_infer_batch_workspace_reserve(&workspace, 300, 16) != NULL,
          "second workspace reserve");
    CHECK(moe_infer_batch_workspace_reserve(&workspace, 100, 16) == NULL,
          "workspace overflow accepted");
    CHECK(moe_infer_batch_workspace_reserve(&workspace, 1, 512) == NULL,
          "unsupported base alignment accepted");
    CHECK(workspace.high_water_bytes == 428, "workspace high-water accounting");
    NativeBatchWorkspace copied_workspace = workspace;
    CHECK(moe_infer_batch_workspace_reserve(&copied_workspace, 8, 8) == NULL,
          "copied workspace retained ownership");
    moe_infer_batch_workspace_reset(&workspace);
    CHECK(workspace.used_bytes == 0 && workspace.high_water_bytes == 428,
          "workspace reset/high-water semantics");

    NativeBatchWorkspace over_budget = {0};
    CHECK(moe_infer_batch_workspace_init(&over_budget, 4, 1024 * 1024 + 1, 1024 * 1024) ==
              NATIVE_BATCH_BUDGET_EXCEEDED,
          "over-budget workspace accepted");
    NativeBatchWorkspace too_small = {0};
    CHECK(moe_infer_batch_workspace_init(&too_small, 2, 256, 1024 * 1024) == NATIVE_BATCH_OK,
          "small workspace init");
    NativeBatchWorkspace external_budget = {0};
    CHECK(moe_infer_batch_workspace_init(&external_budget, 4, 512, 1024 * 1024 + 1) == NATIVE_BATCH_OK,
          "external workspace init");

    uint16_t *kv_rows = (uint16_t *)calloc(8u * KV_LORA_RANK, sizeof(uint16_t));
    CHECK(kv_rows, "KV fixture allocation");
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        engine->kv_cache[layer].kv = kv_rows;
        other_engine->kv_cache[layer].kv = kv_rows;
    }

    NativeKVTransaction transaction;
    NativeKVTransaction concurrent;
    moe_infer_kv_transaction_init(&transaction);
    moe_infer_kv_transaction_init(&concurrent);
    const uint32_t positions[] = {4, 5, 6};
    set_lengths(engine, 4);
    set_lengths(other_engine, 4);
    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_PREFILL,
              positions, 3, &workspace, &transaction) == NATIVE_BATCH_OK,
          "begin valid prefill transaction");
    CHECK(transaction.active == 1 && transaction.owner_engine == (uintptr_t)engine &&
              lengths_equal(engine, 4),
          "begin ownership or mutation");
    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_PREFILL,
              positions, 3, &workspace, &concurrent) == NATIVE_BATCH_INVALID_STATE,
          "concurrent transaction accepted");

    NativeKVTransaction copied_transaction = transaction;
    CHECK(moe_infer_kv_transaction_abort(engine, &copied_transaction) ==
              NATIVE_BATCH_INVALID_STATE,
          "copied transaction retained ownership");
    CHECK(moe_infer_kv_transaction_abort(other_engine, &transaction) ==
              NATIVE_BATCH_INVALID_STATE,
          "cross-engine abort accepted");
    CHECK(moe_infer_kv_transaction_abort(engine, &transaction) == NATIVE_BATCH_OK,
          "owner abort");
    CHECK(lengths_equal(engine, 4), "abort changed committed lengths");

    const uint32_t bad_positions[] = {4, 6, 7};
    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_PREFILL,
              bad_positions, 3, &workspace, &transaction) == NATIVE_BATCH_INVALID_POSITION,
          "non-contiguous positions accepted");
    CHECK(transaction.active == 0 && lengths_equal(engine, 4),
          "invalid positions mutated state");
    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_PREFILL,
              positions, 3, &too_small, &transaction) == NATIVE_BATCH_CAPACITY_EXCEEDED,
          "workspace capacity overflow accepted");
    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_PREFILL,
              positions, 3, &external_budget, &transaction) == NATIVE_BATCH_BUDGET_EXCEEDED,
          "config budget overflow accepted");

    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_PREFILL,
              positions, 3, &workspace, &transaction) == NATIVE_BATCH_OK,
          "begin incomplete-stage transaction");
    set_lengths(engine, 7);
    for (uint32_t layer = 0; layer + 1 < N_LAYERS; ++layer) {
        CHECK(moe_infer_kv_transaction_mark_layer_staged(
                  engine, &transaction, layer, 3) == NATIVE_BATCH_OK,
              "mark partial layer set");
    }
    CHECK(moe_infer_kv_transaction_commit(engine, &transaction, 3) ==
              NATIVE_BATCH_INVALID_STATE,
          "incomplete layer set committed");
    CHECK(transaction.active == 1, "failed commit released ownership");
    CHECK(moe_infer_kv_transaction_abort(engine, &transaction) == NATIVE_BATCH_OK,
          "abort after failed commit");
    CHECK(lengths_equal(engine, 4), "failed commit/abort did not restore lengths");

    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_PREFILL,
              positions, 3, &workspace, &transaction) == NATIVE_BATCH_OK,
          "begin full prefill transaction");
    NativeKVStagingView view;
    CHECK(moe_infer_kv_transaction_layer_view(engine, &transaction, 7, &view) ==
              NATIVE_BATCH_OK,
          "layer staging view");
    CHECK(view.kv_data == kv_rows + 4u * KV_LORA_RANK && view.layer == 7 &&
              view.row_count == 3 && view.generation == transaction.generation,
          "staging view metadata");
    CHECK(stage_all_layers(engine, &transaction), "stage all prefill layers");
    CHECK(moe_infer_kv_transaction_commit(engine, &transaction, 2) ==
              NATIVE_BATCH_INVALID_ARGUMENT,
          "partial prefill commit accepted");
    CHECK(moe_infer_kv_transaction_commit(engine, &transaction, 3) == NATIVE_BATCH_OK,
          "full prefill commit");
    CHECK(lengths_equal(engine, 7), "full prefill length");

    const uint32_t verify_positions[] = {7, 8, 9};
    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_VERIFY,
              verify_positions, 3, &workspace, &transaction) == NATIVE_BATCH_OK,
          "begin verify transaction");
    NativeKVTransaction stale_copy = transaction;
    CHECK(stage_all_layers(engine, &transaction), "stage all verify layers");
    CHECK(moe_infer_kv_transaction_commit(engine, &transaction, 1) == NATIVE_BATCH_OK,
          "verification prefix commit");
    CHECK(lengths_equal(engine, 8), "verification prefix length");
    CHECK(moe_infer_kv_transaction_abort(engine, &stale_copy) == NATIVE_BATCH_INVALID_STATE,
          "stale copied transaction mutated engine");

    const uint32_t next_positions[] = {8, 9};
    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_VERIFY,
              next_positions, 2, &workspace, &transaction) == NATIVE_BATCH_OK,
          "transaction reuse");
    set_lengths(engine, 10);
    CHECK(moe_infer_kv_transaction_abort(engine, &transaction) == NATIVE_BATCH_OK,
          "reused transaction abort");
    CHECK(lengths_equal(engine, 8), "reused abort length");

    RecurrentFixture recurrent_fixture;
    CHECK(recurrent_fixture_init(engine, &recurrent_fixture),
          "recurrent fixture allocation");
    const uint32_t accepted_cases[] = {0u, 1u, 2u, 3u};
    for (size_t index = 0; index <
            sizeof(accepted_cases) / sizeof(accepted_cases[0]); ++index) {
        CHECK(run_recurrent_prefix_case(
                  engine, &recurrent_fixture, &config, &workspace,
                  accepted_cases[index]) == 0,
              "recurrent prefix matrix");
    }

    // Injected outer failure after one recurrent row must restore the complete
    // original state, including rolling buffers and append destinations.
    recurrent_fixture_reset(engine, &recurrent_fixture);
    set_lengths(engine, 3);
    const uint32_t failure_positions[] = {3, 4, 5};
    moe_infer_kv_transaction_init(&transaction);
    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_VERIFY,
              failure_positions, 3u, &workspace, &transaction) == NATIVE_BATCH_OK,
          "begin recurrent failure transaction");
    CHECK(simulate_recurrent_row(engine, &transaction, 0u),
          "record recurrent row before failure");
    CHECK(moe_infer_kv_transaction_mark_layer_staged(
              engine, &transaction, 0u, 3u) == NATIVE_BATCH_INVALID_STATE,
          "partial recurrent layer accepted");
    CHECK(moe_infer_kv_transaction_abort(engine, &transaction) == NATIVE_BATCH_OK,
          "abort recurrent injected failure");
    CHECK(lengths_equal(engine, 3) &&
              engine->comp_state[0].n_comp == 513u &&
              engine->comp_state[0].n_idx_comp == 7u &&
              recurrent_fixture.state_kv[7u * 4u] == 1028.0f &&
              recurrent_fixture.comp_kv[513u * COMP_HEAD_DIM] ==
                  3000.0f + 513.0f * COMP_HEAD_DIM,
          "recurrent injected failure leaked state");

    // Prove the fixed 43-layer descriptor and all variable snapshots/deltas
    // share one exact hard ceiling. The first successful available-byte value
    // is discovered rather than duplicating private journal layout constants.
    NativeBatchWorkspace journal_workspace = {0};
    CHECK(moe_infer_batch_workspace_init(
              &journal_workspace, 3u, 1u, 1u) == NATIVE_BATCH_OK,
          "journal boundary workspace init");
    recurrent_fixture_reset(engine, &recurrent_fixture);
    set_lengths(engine, 3);
    uint64_t journal_upper = 1u;
    int journal_status = begin_recurrent_with_journal_budget(
        engine, &config, &journal_workspace, journal_upper);
    while (journal_status == NATIVE_BATCH_BUDGET_EXCEEDED &&
           journal_upper < 64u * 1024u * 1024u) {
        journal_upper *= 2u;
        journal_status = begin_recurrent_with_journal_budget(
            engine, &config, &journal_workspace, journal_upper);
    }
    CHECK(journal_status == NATIVE_BATCH_OK,
          "find successful recurrent journal budget");
    uint64_t journal_lower = 0u;
    while (journal_lower < journal_upper) {
        const uint64_t midpoint =
            journal_lower + (journal_upper - journal_lower) / 2u;
        journal_status = begin_recurrent_with_journal_budget(
            engine, &config, &journal_workspace, midpoint);
        CHECK(journal_status == NATIVE_BATCH_OK ||
                  journal_status == NATIVE_BATCH_BUDGET_EXCEEDED,
              "unexpected journal boundary status");
        if (journal_status == NATIVE_BATCH_OK) {
            journal_upper = midpoint;
        } else {
            journal_lower = midpoint + 1u;
        }
    }
    const uint64_t exact_journal_bytes = journal_lower;
    CHECK(exact_journal_bytes > 0u,
          "journal fixed descriptor escaped budget");
    CHECK(begin_recurrent_with_journal_budget(
              engine, &config, &journal_workspace,
              exact_journal_bytes - 1u) == NATIVE_BATCH_BUDGET_EXCEEDED,
          "journal accepted one byte below exact budget");
    CHECK(begin_recurrent_with_journal_budget(
              engine, &config, &journal_workspace,
              exact_journal_bytes) == NATIVE_BATCH_OK,
          "journal rejected exact budget");
    moe_infer_batch_workspace_deinit(&journal_workspace);

    // Reset must release an active recurrent journal and invalidate both the
    // caller-owned handle and stale copies without dereferencing freed state.
    recurrent_fixture_reset(engine, &recurrent_fixture);
    set_lengths(engine, 3);
    moe_infer_kv_transaction_init(&transaction);
    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_VERIFY,
              failure_positions, 3u, &workspace, &transaction) ==
              NATIVE_BATCH_OK,
          "begin recurrent reset transaction");
    CHECK(transaction.active == 1u && transaction.verify_state != NULL,
          "active recurrent reset journal missing");
    NativeKVTransaction reset_stale = transaction;
    moe_infer_reset_kv(engine);
    CHECK(transaction.active == 0u && transaction.verify_state == NULL &&
              lengths_equal(engine, 0),
          "reset did not invalidate active recurrent journal");
    CHECK(moe_infer_kv_transaction_commit(engine, &transaction, 0u) ==
              NATIVE_BATCH_INVALID_STATE,
          "reset transaction committed after invalidation");
    CHECK(moe_infer_kv_transaction_abort(engine, &reset_stale) ==
              NATIVE_BATCH_INVALID_STATE,
          "stale reset transaction touched freed journal");

    recurrent_fixture_deinit(engine, &recurrent_fixture);
    CHECK(test_hca_verify_prefix(engine, &config, &workspace) == 0,
          "HCA multi-row verify prefix regression");
    set_lengths(engine, 8);

    engine->kv_cache[17].len = 5;
    CHECK(moe_infer_kv_transaction_begin(
              engine, &config, NATIVE_BATCH_PHASE_VERIFY,
              next_positions, 2, &workspace, &transaction) == NATIVE_BATCH_INVALID_STATE,
          "inconsistent layer lengths accepted");
    CHECK(transaction.active == 0, "invalid layer state activated transaction");

    workspace.metal_buffer = (void *)(uintptr_t)1u;
    workspace.metal_buffer_release = test_workspace_metal_release;
    workspace.metal_buffer_creations = 1u;
    moe_infer_batch_workspace_deinit(&workspace);
    CHECK(workspace_metal_release_calls == 1 && !workspace.storage &&
              !workspace.metal_buffer && !workspace.metal_buffer_release,
          "workspace Metal view release/deinit lifecycle");
    moe_infer_batch_workspace_deinit(&too_small);
    moe_infer_batch_workspace_deinit(&external_budget);
    free(kv_rows);
    free(other_engine);
    free(engine);
    fprintf(stderr, "MLA transaction self-test: PASS\n");
    return 0;
}
