// Standalone test for mla_attention_decode against the Python golden.
// Loads /tmp/attn_golden/* (gen_attn_golden.py) and checks the host MLA
// attention output matches golden.bin.
//
// Build & run:
//   clang -framework Metal -framework Foundation -fobjc-arc \
//     -I src/metal_infer scripts/mla_attention_test.m src/metal_infer/mla_attention.m \
//     -o /tmp/mat && /tmp/mat
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <math.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <sys/mman.h>
#import <unistd.h>
#import "mla_attention.h"
#import "batch_oracle.h"

#define GD "/tmp/attn_golden/"

static float *readf(const char *name, size_t *n_out) {
    char path[512]; snprintf(path, sizeof(path), "%s%s", GD, name);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz); fread(buf, 1, sz, f); fclose(f);
    if (n_out) *n_out = sz / sizeof(float);
    return (float *)buf;
}
static uint32_t *readu(const char *name) {
    char path[512]; snprintf(path, sizeof(path), "%s%s", GD, name);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz); fread(buf, 1, sz, f); fclose(f);
    return (uint32_t *)buf;
}

static id<MTLComputePipelineState> mkpipe(id<MTLDevice> d, id<MTLLibrary> lib, const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { printf("FAIL pipeline %s: %s\n", name, [[err localizedDescription] UTF8String]); exit(1); }
    return p;
}

static QuantWeight load_qw(const char *base, int out_dim, int in_dim) {
    char nm[256];
    QuantWeight q; q.out_dim = out_dim; q.in_dim = in_dim; q.group_size = ATTN_GROUP_SIZE;
    snprintf(nm, sizeof(nm), "%s.packed", base); q.packed = readu(nm);
    snprintf(nm, sizeof(nm), "%s.sc", base); q.scales = readf(nm, NULL);
    snprintf(nm, sizeof(nm), "%s.bi", base); q.biases = readf(nm, NULL);
    return q;
}

static uint16_t f32_to_bf16_test(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static uint16_t f32_to_bf16_rne_test(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = UINT32_C(0x7fff) +
        ((bits >> 16u) & UINT32_C(1));
    bits += rounding_bias;
    return (uint16_t)(bits >> 16u);
}

static float bf16_to_f32_test(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_f16_test(float value) {
    const _Float16 half_value = (_Float16)value;
    uint16_t bits;
    memcpy(&bits, &half_value, sizeof(bits));
    return bits;
}

static float f16_to_f32_test(uint16_t value) {
    _Float16 half_value;
    memcpy(&half_value, &value, sizeof(value));
    return (float)half_value;
}

static int run_batched_q_tests(MlaPipes *P, AttnWeights *aw, const float *x) {
    enum { CAPACITY = 5, Q_WIDTH = N_HEADS * HEAD_DIM };
    const size_t hidden_count = (size_t)CAPACITY * DIM;
    const size_t q_count = (size_t)CAPACITY * Q_WIDTH;
    uint16_t *hidden = malloc(hidden_count * sizeof(uint16_t));
    uint16_t *batch_q = malloc(q_count * sizeof(uint16_t));
    uint16_t *scalar_q = malloc(q_count * sizeof(uint16_t));
    float *batch_f32 = malloc(q_count * sizeof(float));
    float *scalar_f32 = malloc(q_count * sizeof(float));
    if (!hidden || !batch_q || !scalar_q || !batch_f32 || !scalar_f32) {
        printf("FAIL: batched Q test allocation\n");
        return 0;
    }
    for (int row = 0; row < CAPACITY; ++row) {
        for (int col = 0; col < DIM; ++col) {
            const float scale = 1.0f + 0.03125f * (float)row;
            hidden[(size_t)row * DIM + col] =
                f32_to_bf16_test(x[(col + row * 17) % DIM] * scale);
        }
    }

    NativeBatchWorkspace workspace = {0};
    const uint64_t workspace_bytes = 1024u * 1024u;
    if (moe_infer_batch_workspace_init(
            &workspace, CAPACITY, workspace_bytes, workspace_bytes) != NATIVE_BATCH_OK) {
        printf("FAIL: batched Q workspace init\n");
        return 0;
    }

    // The Q-only test does not need the legacy 128MB wo_a dense fallback.
    // Clearing it before the first persistent-cache lookup avoids unrelated
    // output-stage allocation; the Q weights/norm remain unchanged.
    const float *saved_wo_a_dense = aw->wo_a_dense;
    aw->wo_a_dense = NULL;
    const int cases[] = {1, 2, 3, 5};
    int ok = 1;
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        const uint32_t rows = (uint32_t)cases[ci];
        MLAAttentionMetrics batch_metrics = {0};
        moe_infer_batch_workspace_reset(&workspace);
        const int batch_rc = mla_attention_q_batch(
            P, aw, hidden, batch_q, rows, &workspace, &batch_metrics);
        if (batch_rc != NATIVE_BATCH_OK || batch_metrics.dispatches != 4u ||
            batch_metrics.q_ns == 0u ||
            batch_metrics.workspace_high_water_bytes !=
                (uint64_t)rows *
                    (DIM + 2u * Q_LORA_RANK + 2u * Q_WIDTH) * sizeof(uint16_t)) {
            printf("FAIL: Q batch B=%u rc=%d dispatches=%u q_ns=%llu high_water=%llu\n",
                   rows, batch_rc, batch_metrics.dispatches,
                   (unsigned long long)batch_metrics.q_ns,
                   (unsigned long long)batch_metrics.workspace_high_water_bytes);
            ok = 0;
            continue;
        }

        uint32_t scalar_dispatches = 0;
        for (uint32_t row = 0; row < rows; ++row) {
            MLAAttentionMetrics scalar_metrics = {0};
            moe_infer_batch_workspace_reset(&workspace);
            const int scalar_rc = mla_attention_q_batch(
                P, aw, hidden + (size_t)row * DIM,
                scalar_q + (size_t)row * Q_WIDTH,
                1u, &workspace, &scalar_metrics);
            if (scalar_rc != NATIVE_BATCH_OK) {
                printf("FAIL: scalar Q reference row=%u rc=%d\n", row, scalar_rc);
                ok = 0;
                break;
            }
            scalar_dispatches += scalar_metrics.dispatches;
        }
        if (!ok) continue;

        const size_t compared = (size_t)rows * Q_WIDTH;
        for (size_t i = 0; i < compared; ++i) {
            batch_f32[i] = bf16_to_f32_test(batch_q[i]);
            scalar_f32[i] = bf16_to_f32_test(scalar_q[i]);
        }
        BatchOracleReport report;
        const BatchOracleStatus status = batch_oracle_compare_f32(
            scalar_f32, batch_f32, compared,
            BATCH_ORACLE_DTYPE_BF16,
            BATCH_ORACLE_CHECKPOINT_ATTENTION_Q,
            0, -1, &report);
        if (status != BATCH_ORACLE_OK || scalar_dispatches != rows * 4u) {
            printf("FAIL: Q batch B=%u status=%s first=%zu max_abs=%.3e "
                   "batch_dispatch=%u scalar_dispatch=%u\n",
                   rows, batch_oracle_status_name(status), report.first_index,
                   report.max_absolute_error, batch_metrics.dispatches,
                   scalar_dispatches);
            ok = 0;
        } else {
            printf("Q batch B=%u PASS max_abs=%.3e dispatches=%u vs %u q_ms=%.3f%s\n",
                   rows, report.max_absolute_error, batch_metrics.dispatches,
                   scalar_dispatches, (double)batch_metrics.q_ns / 1.0e6,
                   rows == 3u ? " (tail case for capacity 5)" : "");
        }
    }

    // Warm both paths, then compare the best of three B=3 stage timings.
    // The scalar reference performs the same four stages three times.
    const uint32_t perf_rows = 3u;
    uint64_t batch_best = UINT64_MAX;
    uint64_t scalar_best = UINT64_MAX;
    for (int iteration = -1; iteration < 3; ++iteration) {
        MLAAttentionMetrics bm = {0};
        moe_infer_batch_workspace_reset(&workspace);
        if (mla_attention_q_batch(P, aw, hidden, batch_q, perf_rows,
                                  &workspace, &bm) != NATIVE_BATCH_OK) {
            ok = 0;
            break;
        }
        uint64_t scalar_total = 0u;
        for (uint32_t row = 0; row < perf_rows; ++row) {
            MLAAttentionMetrics sm = {0};
            moe_infer_batch_workspace_reset(&workspace);
            if (mla_attention_q_batch(
                    P, aw, hidden + (size_t)row * DIM,
                    scalar_q + (size_t)row * Q_WIDTH,
                    1u, &workspace, &sm) != NATIVE_BATCH_OK) {
                ok = 0;
                break;
            }
            scalar_total += sm.q_ns;
        }
        if (iteration >= 0) {
            if (bm.q_ns < batch_best) batch_best = bm.q_ns;
            if (scalar_total < scalar_best) scalar_best = scalar_total;
        }
    }
    if (ok && !(batch_best < scalar_best)) {
        printf("FAIL: warm Q batch B=3 not faster: batch=%.3fms scalar=%.3fms\n",
               (double)batch_best / 1.0e6, (double)scalar_best / 1.0e6);
        ok = 0;
    } else if (ok) {
        printf("Q batch warm performance PASS: B=3 %.3fms < repeated scalar %.3fms\n",
               (double)batch_best / 1.0e6, (double)scalar_best / 1.0e6);
    }

    // The arena contract is exact: all five GPU-visible row matrices are
    // charged, exact capacity succeeds, and one byte less fails before encode.
    const uint64_t exact_bytes = (uint64_t)CAPACITY *
        (DIM + 2u * Q_LORA_RANK + 2u * Q_WIDTH) * sizeof(uint16_t);
    moe_infer_batch_workspace_deinit(&workspace);
    if (moe_infer_batch_workspace_init(
            &workspace, CAPACITY, exact_bytes, exact_bytes) != NATIVE_BATCH_OK) {
        printf("FAIL: exact-capacity Q workspace init\n");
        ok = 0;
    } else {
        MLAAttentionMetrics exact_metrics = {0};
        const int exact_rc = mla_attention_q_batch(
            P, aw, hidden, batch_q, CAPACITY, &workspace, &exact_metrics);
        if (exact_rc != NATIVE_BATCH_OK ||
            exact_metrics.workspace_high_water_bytes != exact_bytes) {
            printf("FAIL: exact-capacity Q workspace rc=%d high_water=%llu expected=%llu\n",
                   exact_rc,
                   (unsigned long long)exact_metrics.workspace_high_water_bytes,
                   (unsigned long long)exact_bytes);
            ok = 0;
        }
    }
    moe_infer_batch_workspace_deinit(&workspace);
    if (moe_infer_batch_workspace_init(
            &workspace, CAPACITY, exact_bytes - 1u, exact_bytes - 1u) !=
            NATIVE_BATCH_OK) {
        printf("FAIL: one-byte-short Q workspace init\n");
        ok = 0;
    } else {
        MLAAttentionMetrics short_metrics = {0};
        const int short_rc = mla_attention_q_batch(
            P, aw, hidden, batch_q, CAPACITY, &workspace, &short_metrics);
        if (short_rc != NATIVE_BATCH_CAPACITY_EXCEEDED || workspace.used_bytes != 0u) {
            printf("FAIL: one-byte-short Q workspace rc=%d used=%llu\n",
                   short_rc, (unsigned long long)workspace.used_bytes);
            ok = 0;
        } else {
            printf("Q batch workspace bounds PASS: exact=%llu one-byte-short rejected\n",
                   (unsigned long long)exact_bytes);
        }
    }

    aw->wo_a_dense = saved_wo_a_dense;
    moe_infer_batch_workspace_deinit(&workspace);
    free(hidden);
    free(batch_q);
    free(scalar_q);
    free(batch_f32);
    free(scalar_f32);
    return ok;
}

enum {
    KV_TEST_CAPACITY = 5,
    KV_TEST_START = 11,
    KV_TEST_CACHE_ROWS = KV_TEST_START + KV_TEST_CAPACITY,
};

typedef struct TestKVFixture {
    MoEInferEngine *engine;
    NativeKVTransaction transaction;
    NativeKVStagingView view;
} TestKVFixture;

static int begin_test_kv_fixture(
    TestKVFixture *fixture,
    uint16_t *cache,
    void *gpu_buffer,
    uint32_t committed_length,
    const uint32_t *positions,
    uint32_t rows,
    NativeBatchWorkspace *workspace) {
    memset(fixture, 0, sizeof(*fixture));
    fixture->engine = calloc(1, sizeof(*fixture->engine));
    if (!fixture->engine) return NATIVE_BATCH_INVALID_STATE;
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        fixture->engine->kv_cache[layer].len = (int)committed_length;
    }
    fixture->engine->kv_cache[0].kv = cache;
    fixture->engine->kv_cache[0].kv_gpu_buf = gpu_buffer;
    const NativeBatchConfig config = {
        .max_prefill_chunk = KV_TEST_CAPACITY,
        .max_verify_chunk = KV_TEST_CAPACITY,
        .workspace_budget_bytes = workspace->budget_bytes,
    };
    moe_infer_kv_transaction_init(&fixture->transaction);
    int rc = moe_infer_kv_transaction_begin(
        fixture->engine, &config, NATIVE_BATCH_PHASE_PREFILL,
        positions, rows, workspace, &fixture->transaction);
    if (rc == NATIVE_BATCH_OK) {
        rc = moe_infer_kv_transaction_layer_view(
            fixture->engine, &fixture->transaction, 0u, &fixture->view);
    }
    if (rc != NATIVE_BATCH_OK) {
        free(fixture->engine);
        fixture->engine = NULL;
    }
    return rc;
}

static void end_test_kv_fixture(TestKVFixture *fixture) {
    if (!fixture || !fixture->engine) return;
    if (fixture->transaction.active == 1u) {
        (void)moe_infer_kv_transaction_abort(
            fixture->engine, &fixture->transaction);
    }
    free(fixture->engine);
    fixture->engine = NULL;
}

static MLAPrefillBatchArgs make_kv_args(
    const uint16_t *hidden,
    uint32_t rows,
    const uint32_t *positions,
    NativeKVStagingView view,
    NativeKVTransaction *transaction,
    NativeBatchWorkspace *workspace) {
    return (MLAPrefillBatchArgs){
        .hidden = hidden,
        .batch_tokens = rows,
        .layer = 0u,
        .positions = positions,
        .kv = view,
        .kv_transaction = transaction,
        .workspace = workspace,
    };
}

static int reject_kv_atomically(
    const char *name,
    int expected_status,
    MoEInferEngine *engine,
    MlaPipes *P,
    AttnWeights *aw,
    MLAPrefillBatchArgs *args,
    NativeBatchWorkspace *workspace,
    const void *watch_a,
    size_t watch_a_bytes,
    const void *watch_b,
    size_t watch_b_bytes) {
    void *before_a = watch_a_bytes ? malloc(watch_a_bytes) : NULL;
    void *before_b = watch_b_bytes ? malloc(watch_b_bytes) : NULL;
    if ((watch_a_bytes && !before_a) || (watch_b_bytes && !before_b)) {
        free(before_a);
        free(before_b);
        printf("FAIL: %s snapshot allocation\n", name);
        return 0;
    }
    if (watch_a_bytes) memcpy(before_a, watch_a, watch_a_bytes);
    if (watch_b_bytes) memcpy(before_b, watch_b, watch_b_bytes);
    moe_infer_batch_workspace_reset(workspace);
    const int rc = mla_attention_kv_rope_batch(engine, P, aw, args, NULL);
    const int unchanged =
        (!watch_a_bytes || memcmp(before_a, watch_a, watch_a_bytes) == 0) &&
        (!watch_b_bytes || memcmp(before_b, watch_b, watch_b_bytes) == 0);
    const int passed =
        rc == expected_status && workspace->used_bytes == 0u && unchanged;
    if (!passed) {
        printf("FAIL: %s rc=%d expected=%d used=%llu unchanged=%d\n",
               name, rc, expected_status,
               (unsigned long long)workspace->used_bytes, unchanged);
    }
    free(before_a);
    free(before_b);
    return passed;
}

static uint64_t align_workspace_test(uint64_t value) {
    return (value + NATIVE_BATCH_WORKSPACE_ALIGNMENT - 1u) &
        ~(uint64_t)(NATIVE_BATCH_WORKSPACE_ALIGNMENT - 1u);
}

static int run_batched_kv_tests(MlaPipes *P, AttnWeights *aw, const float *x) {
    const uint32_t all_positions[KV_TEST_CAPACITY] = {11, 12, 13, 14, 15};
    const size_t kv_row_bytes = KV_LORA_RANK * sizeof(uint16_t);
    const size_t cache_bytes = (size_t)KV_TEST_CACHE_ROWS * kv_row_bytes;
    const size_t staged_bytes = (size_t)KV_TEST_CAPACITY * kv_row_bytes;
    uint16_t *hidden = malloc((size_t)KV_TEST_CAPACITY * DIM * sizeof(uint16_t));
    uint16_t *kv_storage = malloc(cache_bytes);
    uint16_t *scalar_kv = calloc(1, cache_bytes);
    uint16_t *full_kv = malloc(staged_bytes);
    uint16_t *chunk_kv = calloc(1, cache_bytes);
    float *batch_f32 = malloc((size_t)KV_TEST_CAPACITY * KV_LORA_RANK * sizeof(float));
    float *scalar_f32 = malloc((size_t)KV_TEST_CAPACITY * KV_LORA_RANK * sizeof(float));
    if (!hidden || !kv_storage || !scalar_kv || !full_kv || !chunk_kv ||
        !batch_f32 || !scalar_f32) {
        printf("FAIL: batched KV test allocation\n");
        return 0;
    }
    for (int row = 0; row < KV_TEST_CAPACITY; ++row) {
        for (int col = 0; col < DIM; ++col) {
            hidden[(size_t)row * DIM + col] = f32_to_bf16_test(
                x[(col + row * 29) % DIM] * (1.0f + 0.015625f * row));
        }
    }

    NativeBatchWorkspace workspace = {0};
    const uint64_t generous_bytes = 1024u * 1024u;
    if (moe_infer_batch_workspace_init(
            &workspace, KV_TEST_CAPACITY, generous_bytes, generous_bytes) !=
            NATIVE_BATCH_OK) {
        printf("FAIL: batched KV workspace init\n");
        return 0;
    }

    const int cases[] = {1, 2, 3, 5};
    int ok = 1;
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        const uint32_t rows = (uint32_t)cases[ci];
        for (size_t i = 0; i < (size_t)KV_TEST_START * KV_LORA_RANK; ++i) {
            kv_storage[i] = 0x3c00u;
        }
        memset(kv_storage + (size_t)KV_TEST_START * KV_LORA_RANK, 0,
               staged_bytes);
        memset(scalar_kv, 0, cache_bytes);

        TestKVFixture batch;
        int batch_rc = begin_test_kv_fixture(
            &batch, kv_storage, NULL, KV_TEST_START,
            all_positions, rows, &workspace);
        MLAAttentionMetrics batch_metrics = {0};
        if (batch_rc == NATIVE_BATCH_OK) {
            MLAPrefillBatchArgs args = make_kv_args(
                hidden, rows, all_positions, batch.view,
                &batch.transaction, &workspace);
            moe_infer_batch_workspace_reset(&workspace);
            batch_rc = mla_attention_kv_rope_batch(
                batch.engine, P, aw, &args, &batch_metrics);
        }
        end_test_kv_fixture(&batch);
        if (batch_rc != NATIVE_BATCH_OK || batch_metrics.dispatches != 3u ||
            batch_metrics.kv_rope_ns == 0u) {
            printf("FAIL: KV batch B=%u rc=%d dispatches=%u kv_ns=%llu\n",
                   rows, batch_rc, batch_metrics.dispatches,
                   (unsigned long long)batch_metrics.kv_rope_ns);
            ok = 0;
            continue;
        }
        for (size_t i = 0; i < (size_t)KV_TEST_START * KV_LORA_RANK; ++i) {
            if (kv_storage[i] != 0x3c00u) {
                printf("FAIL: KV batch B=%u changed committed prefix at %zu\n", rows, i);
                ok = 0;
                break;
            }
        }

        for (uint32_t row = 0; row < rows && ok; ++row) {
            TestKVFixture scalar;
            int scalar_rc = begin_test_kv_fixture(
                &scalar, scalar_kv, NULL, all_positions[row],
                all_positions + row, 1u, &workspace);
            if (scalar_rc == NATIVE_BATCH_OK) {
                MLAPrefillBatchArgs scalar_args = make_kv_args(
                    hidden + (size_t)row * DIM, 1u, all_positions + row,
                    scalar.view, &scalar.transaction, &workspace);
                moe_infer_batch_workspace_reset(&workspace);
                scalar_rc = mla_attention_kv_rope_batch(
                    scalar.engine, P, aw, &scalar_args, NULL);
            }
            end_test_kv_fixture(&scalar);
            if (scalar_rc != NATIVE_BATCH_OK) {
                printf("FAIL: scalar KV reference row=%u rc=%d\n", row, scalar_rc);
                ok = 0;
            }
        }
        if (!ok) continue;
        const size_t count = (size_t)rows * KV_LORA_RANK;
        const uint16_t *batch_kv =
            kv_storage + (size_t)KV_TEST_START * KV_LORA_RANK;
        const uint16_t *scalar_rows =
            scalar_kv + (size_t)KV_TEST_START * KV_LORA_RANK;
        for (size_t i = 0; i < count; ++i) {
            batch_f32[i] = f16_to_f32_test(batch_kv[i]);
            scalar_f32[i] = f16_to_f32_test(scalar_rows[i]);
        }
        BatchOracleReport report;
        const BatchOracleStatus status = batch_oracle_compare_f32(
            scalar_f32, batch_f32, count, BATCH_ORACLE_DTYPE_F16,
            BATCH_ORACLE_CHECKPOINT_ATTENTION_KV, 0, -1, &report);
        if (status != BATCH_ORACLE_OK) {
            printf("FAIL: KV batch B=%u status=%s first=%zu max_abs=%.3e\n",
                   rows, batch_oracle_status_name(status), report.first_index,
                   report.max_absolute_error);
            ok = 0;
        } else {
            printf("KV/RoPE batch B=%u PASS pos=%u..%u max_abs=%.3e "
                   "dispatches=3 vs %u kv_ms=%.3f%s\n",
                   rows, all_positions[0], all_positions[rows - 1u],
                   report.max_absolute_error, rows * 3u,
                   (double)batch_metrics.kv_rope_ns / 1.0e6,
                   rows == 3u ? " (tail case)" : "");
        }
        if (rows == KV_TEST_CAPACITY) {
            memcpy(full_kv, batch_kv, staged_bytes);
        }
    }

    // A full nonzero-position chunk must be invariant under [2,2,1]
    // partitioning. Every partition uses a newly begun authentic transaction.
    if (ok) {
        const uint32_t partitions[] = {2u, 2u, 1u};
        uint32_t offset = 0u;
        uint32_t dispatches = 0u;
        memset(chunk_kv, 0, cache_bytes);
        for (size_t part = 0; part < sizeof(partitions) / sizeof(partitions[0]); ++part) {
            const uint32_t rows = partitions[part];
            TestKVFixture fixture;
            int rc = begin_test_kv_fixture(
                &fixture, chunk_kv, NULL, all_positions[offset],
                all_positions + offset, rows, &workspace);
            MLAAttentionMetrics metrics = {0};
            if (rc == NATIVE_BATCH_OK) {
                MLAPrefillBatchArgs args = make_kv_args(
                    hidden + (size_t)offset * DIM, rows,
                    all_positions + offset, fixture.view,
                    &fixture.transaction, &workspace);
                moe_infer_batch_workspace_reset(&workspace);
                rc = mla_attention_kv_rope_batch(
                    fixture.engine, P, aw, &args, &metrics);
            }
            end_test_kv_fixture(&fixture);
            if (rc != NATIVE_BATCH_OK) {
                ok = 0;
                break;
            }
            dispatches += metrics.dispatches;
            offset += rows;
        }
        const uint16_t *chunk_rows =
            chunk_kv + (size_t)KV_TEST_START * KV_LORA_RANK;
        for (size_t i = 0; i < (size_t)KV_TEST_CAPACITY * KV_LORA_RANK; ++i) {
            batch_f32[i] = f16_to_f32_test(chunk_rows[i]);
            scalar_f32[i] = f16_to_f32_test(full_kv[i]);
        }
        BatchOracleReport report;
        if (!ok || offset != KV_TEST_CAPACITY || dispatches != 9u ||
            batch_oracle_compare_f32(
                scalar_f32, batch_f32,
                (size_t)KV_TEST_CAPACITY * KV_LORA_RANK,
                BATCH_ORACLE_DTYPE_F16,
                BATCH_ORACLE_CHECKPOINT_ATTENTION_KV,
                0, -1, &report) != BATCH_ORACLE_OK) {
            printf("FAIL: KV partition [2,2,1] mismatch dispatches=%u\n", dispatches);
            ok = 0;
        } else {
            printf("KV/RoPE partition PASS: [5] == [2,2,1], pos=11..15, dispatches=%u\n",
                   dispatches);
        }
    }

    // Non-contiguous arguments are rejected even when ownership and the view
    // were obtained from a valid contiguous transaction.
    {
        const uint32_t bad_positions[] = {11, 13};
        TestKVFixture fixture;
        int rc = begin_test_kv_fixture(
            &fixture, kv_storage, NULL, KV_TEST_START,
            all_positions, 2u, &workspace);
        if (rc == NATIVE_BATCH_OK) {
            MLAPrefillBatchArgs args = make_kv_args(
                hidden, 2u, bad_positions, fixture.view,
                &fixture.transaction, &workspace);
            rc = reject_kv_atomically(
                "non-contiguous KV positions", NATIVE_BATCH_INVALID_POSITION,
                fixture.engine, P, aw, &args, &workspace,
                kv_storage, cache_bytes, NULL, 0u)
                ? NATIVE_BATCH_OK : NATIVE_BATCH_INVALID_STATE;
        }
        end_test_kv_fixture(&fixture);
        if (rc != NATIVE_BATCH_OK) ok = 0;
    }

    // Account every GPU-visible matrix plus alignment padding. Exact capacity
    // is 52608 bytes; one byte less fails without consuming the arena.
    const uint64_t hidden_bytes =
        (uint64_t)KV_TEST_CAPACITY * DIM * sizeof(uint16_t);
    const uint64_t kv_bytes =
        (uint64_t)KV_TEST_CAPACITY * kv_row_bytes;
    const uint64_t rope_bytes =
        (uint64_t)KV_TEST_CAPACITY * (QK_ROPE_DIM / 2) * sizeof(float);
    const uint64_t kv_rel = align_workspace_test(hidden_bytes);
    const uint64_t kv_norm_rel = align_workspace_test(kv_rel + kv_bytes);
    const uint64_t cos_rel = align_workspace_test(kv_norm_rel + kv_bytes);
    const uint64_t sin_rel = align_workspace_test(cos_rel + rope_bytes);
    const uint64_t exact_bytes = sin_rel + rope_bytes;
    if (exact_bytes != 52608u) {
        printf("FAIL: KV exact workspace changed: %llu\n",
               (unsigned long long)exact_bytes);
        ok = 0;
    }
    moe_infer_batch_workspace_deinit(&workspace);
    if (moe_infer_batch_workspace_init(
            &workspace, KV_TEST_CAPACITY, exact_bytes, exact_bytes) !=
            NATIVE_BATCH_OK) {
        printf("FAIL: exact-capacity KV workspace init\n");
        ok = 0;
    } else {
        TestKVFixture fixture;
        int rc = begin_test_kv_fixture(
            &fixture, chunk_kv, NULL, KV_TEST_START,
            all_positions, KV_TEST_CAPACITY, &workspace);
        MLAAttentionMetrics metrics = {0};
        if (rc == NATIVE_BATCH_OK) {
            MLAPrefillBatchArgs args = make_kv_args(
                hidden, KV_TEST_CAPACITY, all_positions, fixture.view,
                &fixture.transaction, &workspace);
            rc = mla_attention_kv_rope_batch(
                fixture.engine, P, aw, &args, &metrics);
        }
        end_test_kv_fixture(&fixture);
        if (rc != NATIVE_BATCH_OK || workspace.used_bytes != exact_bytes ||
            metrics.workspace_high_water_bytes != exact_bytes) {
            printf("FAIL: exact-capacity KV workspace rc=%d used=%llu expected=%llu\n",
                   rc, (unsigned long long)workspace.used_bytes,
                   (unsigned long long)exact_bytes);
            ok = 0;
        }
    }
    moe_infer_batch_workspace_deinit(&workspace);
    if (moe_infer_batch_workspace_init(
            &workspace, KV_TEST_CAPACITY, exact_bytes - 1u, exact_bytes - 1u) !=
            NATIVE_BATCH_OK) {
        printf("FAIL: one-byte-short KV workspace init\n");
        ok = 0;
    } else {
        TestKVFixture fixture;
        int rc = begin_test_kv_fixture(
            &fixture, chunk_kv, NULL, KV_TEST_START,
            all_positions, KV_TEST_CAPACITY, &workspace);
        if (rc == NATIVE_BATCH_OK) {
            MLAPrefillBatchArgs args = make_kv_args(
                hidden, KV_TEST_CAPACITY, all_positions, fixture.view,
                &fixture.transaction, &workspace);
            rc = reject_kv_atomically(
                "one-byte-short KV workspace", NATIVE_BATCH_CAPACITY_EXCEEDED,
                fixture.engine, P, aw, &args, &workspace,
                chunk_kv, cache_bytes, NULL, 0u)
                ? NATIVE_BATCH_OK : NATIVE_BATCH_INVALID_STATE;
        }
        end_test_kv_fixture(&fixture);
        if (rc != NATIVE_BATCH_OK) {
            ok = 0;
        } else {
            printf("KV/RoPE workspace bounds PASS: exact=%llu one-byte-short rejected\n",
                   (unsigned long long)exact_bytes);
        }
    }

    // Use a full-cache shared Metal buffer. The CPU f16 conversion must land
    // at absolute row 11; abort restores lengths and leaves the prefix untouched.
    moe_infer_batch_workspace_deinit(&workspace);
    if (moe_infer_batch_workspace_init(
            &workspace, KV_TEST_CAPACITY, generous_bytes, generous_bytes) !=
            NATIVE_BATCH_OK) {
        printf("FAIL: transaction KV workspace init\n");
        ok = 0;
    } else {
        id<MTLBuffer> gpu_cache = [P->dev
            newBufferWithLength:cache_bytes options:MTLResourceStorageModeShared];
        uint16_t *gpu_words = gpu_cache ? (uint16_t *)[gpu_cache contents] : NULL;
        if (!gpu_cache) {
            printf("FAIL: transaction GPU fixture allocation\n");
            ok = 0;
        } else {
            for (size_t i = 0; i < (size_t)KV_TEST_START * KV_LORA_RANK; ++i) {
                gpu_words[i] = 0x3c00u;
            }
            TestKVFixture fixture;
            int rc = begin_test_kv_fixture(
                &fixture, gpu_words, (void *)gpu_cache, KV_TEST_START,
                all_positions, KV_TEST_CAPACITY, &workspace);
            if (rc == NATIVE_BATCH_OK) {
                MLAPrefillBatchArgs args = make_kv_args(
                    hidden, KV_TEST_CAPACITY, all_positions, fixture.view,
                    &fixture.transaction, &workspace);
                moe_infer_batch_workspace_reset(&workspace);
                rc = mla_attention_kv_rope_batch(
                    fixture.engine, P, aw, &args, NULL);
            }
            if (rc != NATIVE_BATCH_OK ||
                memcmp(gpu_words + (size_t)KV_TEST_START * KV_LORA_RANK,
                       full_kv, staged_bytes) != 0) {
                printf("FAIL: absolute GPU staged KV write rc=%d\n", rc);
                ok = 0;
            }
            for (size_t i = 0; i < (size_t)KV_TEST_START * KV_LORA_RANK; ++i) {
                if (gpu_words[i] != 0x3c00u) {
                    printf("FAIL: GPU staging changed committed prefix at %zu\n", i);
                    ok = 0;
                    break;
                }
            }
            if (fixture.engine) {
                for (int layer = 0; layer < N_LAYERS; ++layer) {
                    fixture.engine->kv_cache[layer].len =
                        KV_TEST_START + KV_TEST_CAPACITY;
                }
                if (moe_infer_kv_transaction_abort(
                        fixture.engine, &fixture.transaction) != NATIVE_BATCH_OK) {
                    printf("FAIL: KV transaction abort\n");
                    ok = 0;
                }
                for (int layer = 0; layer < N_LAYERS; ++layer) {
                    if (fixture.engine->kv_cache[layer].len != KV_TEST_START) {
                        printf("FAIL: abort did not restore KV length for layer %d\n", layer);
                        ok = 0;
                        break;
                    }
                }
            }
            end_test_kv_fixture(&fixture);
            for (size_t i = 0; i < (size_t)KV_TEST_START * KV_LORA_RANK; ++i) {
                if (gpu_words[i] != 0x3c00u) {
                    printf("FAIL: abort changed committed KV content at %zu\n", i);
                    ok = 0;
                    break;
                }
            }
            if (ok) {
                printf("KV/RoPE transaction PASS: absolute row 11 staged, abort restored len=11/prefix\n");
            }
        }
    }

    // Ownership and destination failures must reject before workspace use and
    // before changing either the CPU cache or any relevant GPU buffer.
    {
        int rejection_ok = 1;
        uint16_t *cache = malloc(cache_bytes);
        uint16_t *alternate = malloc(cache_bytes);
        id<MTLBuffer> gpu_a = [P->dev
            newBufferWithLength:cache_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> gpu_b = [P->dev
            newBufferWithLength:cache_bytes options:MTLResourceStorageModeShared];
        if (!cache || !alternate || !gpu_a || !gpu_b) {
            printf("FAIL: ownership rejection fixture allocation\n");
            rejection_ok = 0;
        } else {
            memset(cache, 0x5a, cache_bytes);
            memset(alternate, 0xa5, cache_bytes);
            memset([gpu_a contents], 0x3c, cache_bytes);
            memset([gpu_b contents], 0xc3, cache_bytes);

            // A copied transaction remains invalid even after its self-cookie
            // is repaired: the engine owns the original transaction address.
            TestKVFixture fixture;
            int rc = begin_test_kv_fixture(
                &fixture, cache, (void *)gpu_a, KV_TEST_START,
                all_positions, KV_TEST_CAPACITY, &workspace);
            if (rc == NATIVE_BATCH_OK) {
                NativeKVTransaction copied = fixture.transaction;
                copied.self_cookie = (uintptr_t)&copied;
                MLAPrefillBatchArgs args = make_kv_args(
                    hidden, KV_TEST_CAPACITY, all_positions, fixture.view,
                    &copied, &workspace);
                rejection_ok &= reject_kv_atomically(
                    "copied transaction with repaired self-cookie",
                    NATIVE_BATCH_INVALID_STATE, fixture.engine, P, aw,
                    &args, &workspace, cache, cache_bytes,
                    [gpu_a contents], cache_bytes);
            } else rejection_ok = 0;
            end_test_kv_fixture(&fixture);

            // The engine's live generation is authoritative; a stale/mismatched
            // transaction generation cannot stage.
            rc = begin_test_kv_fixture(
                &fixture, cache, (void *)gpu_a, KV_TEST_START,
                all_positions, KV_TEST_CAPACITY, &workspace);
            if (rc == NATIVE_BATCH_OK) {
                fixture.engine->kv_active_transaction_generation++;
                MLAPrefillBatchArgs args = make_kv_args(
                    hidden, KV_TEST_CAPACITY, all_positions, fixture.view,
                    &fixture.transaction, &workspace);
                rejection_ok &= reject_kv_atomically(
                    "active-generation mismatch/stale generation",
                    NATIVE_BATCH_INVALID_STATE, fixture.engine, P, aw,
                    &args, &workspace, cache, cache_bytes,
                    [gpu_a contents], cache_bytes);
                fixture.engine->kv_active_transaction_generation =
                    fixture.transaction.generation;
            } else rejection_ok = 0;
            end_test_kv_fixture(&fixture);

            // Reset invalidates the transaction at the engine boundary.
            rc = begin_test_kv_fixture(
                &fixture, cache, (void *)gpu_a, KV_TEST_START,
                all_positions, KV_TEST_CAPACITY, &workspace);
            if (rc == NATIVE_BATCH_OK) {
                moe_infer_reset_kv(fixture.engine);
                MLAPrefillBatchArgs args = make_kv_args(
                    hidden, KV_TEST_CAPACITY, all_positions, fixture.view,
                    &fixture.transaction, &workspace);
                rejection_ok &= reject_kv_atomically(
                    "moe_infer_reset_kv-invalidated transaction",
                    NATIVE_BATCH_INVALID_STATE, fixture.engine, P, aw,
                    &args, &workspace, cache, cache_bytes,
                    [gpu_a contents], cache_bytes);
            } else rejection_ok = 0;
            end_test_kv_fixture(&fixture);

            // Fully self-consistent caller fields are not authority without a
            // transaction begun and registered by the engine.
            MoEInferEngine *forged_engine = calloc(1, sizeof(*forged_engine));
            NativeKVTransaction forged;
            NativeKVStagingView forged_view = {0};
            if (!forged_engine) {
                rejection_ok = 0;
            } else {
                for (int layer = 0; layer < N_LAYERS; ++layer) {
                    forged_engine->kv_cache[layer].len = KV_TEST_START;
                }
                forged_engine->kv_cache[0].kv = cache;
                forged_engine->kv_cache[0].kv_gpu_buf = (void *)gpu_a;
                moe_infer_kv_transaction_init(&forged);
                forged.generation = 777u;
                forged.owner_engine = (uintptr_t)forged_engine;
                forged.phase = NATIVE_BATCH_PHASE_PREFILL;
                forged.start_position = KV_TEST_START;
                forged.staged_tokens = KV_TEST_CAPACITY;
                forged.active = 1u;
                for (int layer = 0; layer < N_LAYERS; ++layer) {
                    forged.original_kv_lengths[layer] = KV_TEST_START;
                }
                forged_view = (NativeKVStagingView){
                    .kv_data = cache + (size_t)KV_TEST_START * KV_LORA_RANK,
                    .kv_gpu_buffer = (void *)gpu_a,
                    .generation = forged.generation,
                    .layer = 0u,
                    .start_position = KV_TEST_START,
                    .row_count = KV_TEST_CAPACITY,
                    .row_stride_bytes = (uint32_t)kv_row_bytes,
                };
                MLAPrefillBatchArgs args = make_kv_args(
                    hidden, KV_TEST_CAPACITY, all_positions, forged_view,
                    &forged, &workspace);
                rejection_ok &= reject_kv_atomically(
                    "wholly forged self-consistent transaction/view",
                    NATIVE_BATCH_INVALID_STATE, forged_engine, P, aw,
                    &args, &workspace, cache, cache_bytes,
                    [gpu_a contents], cache_bytes);
                free(forged_engine);
            }

            // CPU and GPU destinations must exactly match the engine-derived view.
            // The CPU-only fixture lets the rejection snapshot cover both the
            // genuine cache and the substituted destination completely.
            rc = begin_test_kv_fixture(
                &fixture, cache, NULL, KV_TEST_START,
                all_positions, KV_TEST_CAPACITY, &workspace);
            if (rc == NATIVE_BATCH_OK) {
                NativeKVStagingView swapped = fixture.view;
                swapped.kv_data = alternate + (size_t)KV_TEST_START * KV_LORA_RANK;
                MLAPrefillBatchArgs args = make_kv_args(
                    hidden, KV_TEST_CAPACITY, all_positions, swapped,
                    &fixture.transaction, &workspace);
                rejection_ok &= reject_kv_atomically(
                    "swapped CPU kv_data", NATIVE_BATCH_INVALID_ARGUMENT,
                    fixture.engine, P, aw, &args, &workspace,
                    cache, cache_bytes, alternate, cache_bytes);
            } else rejection_ok = 0;
            end_test_kv_fixture(&fixture);

            // Use the genuine shared GPU buffer as both the CPU cache backing
            // and GPU destination so two snapshots cover every reachable byte.
            rc = begin_test_kv_fixture(
                &fixture, (uint16_t *)[gpu_a contents], (void *)gpu_a,
                KV_TEST_START, all_positions, KV_TEST_CAPACITY, &workspace);
            if (rc == NATIVE_BATCH_OK) {
                NativeKVStagingView swapped = fixture.view;
                swapped.kv_gpu_buffer = (void *)gpu_b;
                MLAPrefillBatchArgs args = make_kv_args(
                    hidden, KV_TEST_CAPACITY, all_positions, swapped,
                    &fixture.transaction, &workspace);
                rejection_ok &= reject_kv_atomically(
                    "swapped GPU kv_gpu_buffer", NATIVE_BATCH_INVALID_ARGUMENT,
                    fixture.engine, P, aw, &args, &workspace,
                    [gpu_a contents], cache_bytes, [gpu_b contents], cache_bytes);
            } else rejection_ok = 0;
            end_test_kv_fixture(&fixture);

            // The undersized buffer is genuine: it is installed on the engine
            // before begin and returned unchanged by layer_view.
            const size_t short_bytes = cache_bytes - 1u;
            id<MTLBuffer> short_gpu = [P->dev
                newBufferWithLength:short_bytes options:MTLResourceStorageModeShared];
            if (!short_gpu) {
                rejection_ok = 0;
            } else {
                memset([short_gpu contents], 0x69, short_bytes);
                rc = begin_test_kv_fixture(
                    &fixture, cache, (void *)short_gpu, KV_TEST_START,
                    all_positions, KV_TEST_CAPACITY, &workspace);
                if (rc == NATIVE_BATCH_OK) {
                    MLAPrefillBatchArgs args = make_kv_args(
                        hidden, KV_TEST_CAPACITY, all_positions, fixture.view,
                        &fixture.transaction, &workspace);
                    rejection_ok &= reject_kv_atomically(
                        "undersized genuine GPU buffer",
                        NATIVE_BATCH_CAPACITY_EXCEEDED,
                        fixture.engine, P, aw, &args, &workspace,
                        cache, cache_bytes, [short_gpu contents], short_bytes);
                } else rejection_ok = 0;
                end_test_kv_fixture(&fixture);
            }
        }
        free(cache);
        free(alternate);
        if (!rejection_ok) {
            ok = 0;
        } else {
            printf("KV/RoPE ownership rejection PASS: copied/stale/reset/forged/swapped/short all atomic\n");
        }
    }

    moe_infer_batch_workspace_deinit(&workspace);
    free(hidden);
    free(kv_storage);
    free(scalar_kv);
    free(full_kv);
    free(chunk_kv);
    free(batch_f32);
    free(scalar_f32);
    return ok;
}

static int run_sdpa_stage_once(
    MlaPipes *P,
    AttnWeights *aw,
    uint16_t *cache,
    uint32_t committed,
    const uint32_t *positions,
    uint32_t rows,
    const uint16_t *q_input,
    uint16_t *output,
    NativeBatchWorkspace *workspace,
    MLAAttentionMetrics *metrics) {
    TestKVFixture fixture;
    int rc = begin_test_kv_fixture(
        &fixture, cache, NULL, committed, positions, rows, workspace);
    if (rc == NATIVE_BATCH_OK) {
        MLAPrefillBatchArgs args = make_kv_args(
            NULL, rows, positions, fixture.view,
            &fixture.transaction, workspace);
        moe_infer_batch_workspace_reset(workspace);
        rc = mla_attention_sdpa_batch(
            fixture.engine, P, aw, &args, q_input, output, metrics);
    }
    end_test_kv_fixture(&fixture);
    return rc;
}

static uint64_t sdpa_workspace_bytes_test(uint32_t rows, uint32_t committed) {
    const uint64_t q_bytes =
        (uint64_t)rows * N_HEADS * HEAD_DIM * sizeof(uint16_t);
    const uint64_t kv_bytes =
        ((uint64_t)committed + (uint64_t)rows) *
        KV_LORA_RANK * sizeof(uint16_t);
    const uint64_t rope_bytes =
        (uint64_t)rows * (QK_ROPE_DIM / 2) * sizeof(float);
    const uint64_t kv_rel = align_workspace_test(q_bytes);
    const uint64_t cos_rel = align_workspace_test(kv_rel + kv_bytes);
    const uint64_t sin_rel = align_workspace_test(cos_rel + rope_bytes);
    const uint64_t output_rel = align_workspace_test(sin_rel + rope_bytes);
    return output_rel + q_bytes;
}

static uint64_t kv_workspace_bytes_test(uint32_t rows) {
    const uint64_t hidden = (uint64_t)rows * DIM * sizeof(uint16_t);
    const uint64_t kv = (uint64_t)rows * KV_LORA_RANK * sizeof(uint16_t);
    const uint64_t rope =
        (uint64_t)rows * (QK_ROPE_DIM / 2) * sizeof(float);
    const uint64_t kv_rel = align_workspace_test(hidden);
    const uint64_t kv_norm_rel = align_workspace_test(kv_rel + kv);
    const uint64_t cos_rel = align_workspace_test(kv_norm_rel + kv);
    const uint64_t sin_rel = align_workspace_test(cos_rel + rope);
    return sin_rel + rope;
}

static int run_workspace_arithmetic_tests(void) {
    uint64_t kv_actual = 0u, sdpa_actual = 0u, end = 0u;
    const uint64_t kv_expected = kv_workspace_bytes_test(UINT32_MAX);
    const uint64_t sdpa_expected =
        sdpa_workspace_bytes_test(UINT32_MAX, UINT32_MAX);
    const int kv_rc = mla_attention_kv_stage_workspace_bytes(
        UINT32_MAX, &kv_actual);
    const int sdpa_rc = mla_attention_sdpa_stage_workspace_bytes(
        UINT32_MAX, UINT32_MAX, &sdpa_actual);
    const int near_max_ok = mla_attention_workspace_region_end(
        UINT64_MAX - 511u, 1u, &end);
    const uint64_t expected_end = UINT64_MAX - 255u;
    const int add_overflow = mla_attention_workspace_region_end(
        UINT64_MAX - 127u, 256u, &end);
    const int align_overflow = mla_attention_workspace_region_end(
        UINT64_MAX - 255u, 255u, &end);
    if (kv_rc != NATIVE_BATCH_OK || kv_actual != kv_expected ||
        sdpa_rc != NATIVE_BATCH_OK || sdpa_actual != sdpa_expected ||
        near_max_ok != NATIVE_BATCH_OK ||
        add_overflow != NATIVE_BATCH_CAPACITY_EXCEEDED ||
        align_overflow != NATIVE_BATCH_CAPACITY_EXCEEDED) {
        printf("FAIL: workspace arithmetic kv=%d/%llu/%llu sdpa=%d/%llu/%llu near=%d overflow=%d/%d\n",
            kv_rc, (unsigned long long)kv_actual,
            (unsigned long long)kv_expected, sdpa_rc,
            (unsigned long long)sdpa_actual,
            (unsigned long long)sdpa_expected, near_max_ok,
            add_overflow, align_overflow);
        return 0;
    }
    // Re-run the successful near-max case because the two rejection probes do
    // not promise to preserve the output parameter.
    end = 0u;
    if (mla_attention_workspace_region_end(
            UINT64_MAX - 511u, 1u, &end) != NATIVE_BATCH_OK ||
        end != expected_end) {
        printf("FAIL: near-UINT64 workspace alignment end=%llu expected=%llu\n",
            (unsigned long long)end, (unsigned long long)expected_end);
        return 0;
    }
    MoEInferEngine *hostile_engine = calloc(1, sizeof(*hostile_engine));
    if (!hostile_engine) {
        printf("FAIL: hostile workspace planner allocation\n");
        return 0;
    }
    hostile_engine->comp_state[0].state_kv = (float *)(uintptr_t)1u;
    hostile_engine->comp_state[0].ratio = UINT32_MAX;
    hostile_engine->comp_state[0].out_dim = UINT32_MAX;
    const MLAPrefillBatchArgs hostile_args = {
        .batch_tokens = 1u,
        .layer = 0u,
    };
    const MLARecurrentBatchState hostile_state = {0};
    uint64_t hostile_bytes = 0u;
    const int planner_overflow = mla_attention_prefill_batch_workspace_bytes(
        hostile_engine, &hostile_args, &hostile_state, &hostile_bytes);
    free(hostile_engine);
    if (planner_overflow != NATIVE_BATCH_INVALID_ARGUMENT) {
        printf("FAIL: hostile recurrent dimensions wrapped in workspace planner rc=%d bytes=%llu\n",
            planner_overflow, (unsigned long long)hostile_bytes);
        return 0;
    }
    printf("Workspace arithmetic PASS: UINT32_MAX estimators exact, UINT64 add/alignment and recurrent-dimension overflow rejected\n");
    return 1;
}

static void cpu_yarn_cos_sin_test(
    uint32_t position, float *cos_values, float *sin_values) {
    const float base = 10000.0f;
    const float factor = 16.0f;
    const float beta_fast = 32.0f;
    const float beta_slow = 1.0f;
    const float original_max = 65536.0f;
    const float pi = 3.14159265358979323846f;
    const float correction_fast = QK_ROPE_DIM *
        logf(original_max / (beta_fast * 2.0f * pi)) /
        (2.0f * logf(base));
    const float correction_slow = QK_ROPE_DIM *
        logf(original_max / (beta_slow * 2.0f * pi)) /
        (2.0f * logf(base));
    const int low = (int)fmaxf(0.0f, floorf(correction_fast));
    const int high = (int)fminf(
        (float)(QK_ROPE_DIM - 1), ceilf(correction_slow));

    for (uint32_t index = 0; index < QK_ROPE_DIM / 2; ++index) {
        float frequency = 1.0f /
            powf(base, (2.0f * (float)index) / (float)QK_ROPE_DIM);
        const float ramp = low == high
            ? (index <= (uint32_t)low ? 0.0f : 1.0f)
            : fminf(1.0f, fmaxf(
                0.0f, ((float)index - (float)low) / (float)(high - low)));
        const float smooth = 1.0f - ramp;
        frequency = frequency / factor * (1.0f - smooth) +
            frequency * smooth;
        cos_values[index] = cosf((float)position * frequency);
        sin_values[index] = sinf((float)position * frequency);
    }
}

// Independent CPU reference for Task 2.4. It intentionally does not call the
// batch Metal stage: Q RoPE, f16 KV decoding, causal visibility, sink-only
// denominator semantics, and stable softmax are recomputed directly.
static int cpu_sdpa_reference(
    const AttnWeights *aw,
    const uint16_t *q_input,
    const uint16_t *cache,
    uint32_t committed,
    const uint32_t *positions,
    uint32_t rows,
    uint16_t *output) {
    const size_t q_elements = (size_t)rows * N_HEADS * HEAD_DIM;
    uint16_t *q_rope = malloc(q_elements * sizeof(uint16_t));
    if (!aw || !aw->attn_sink || !q_input || !cache || !positions ||
        !output || !q_rope) {
        free(q_rope);
        return 0;
    }
    memcpy(q_rope, q_input, q_elements * sizeof(uint16_t));

    for (uint32_t row = 0; row < rows; ++row) {
        float cos_values[QK_ROPE_DIM / 2];
        float sin_values[QK_ROPE_DIM / 2];
        cpu_yarn_cos_sin_test(positions[row], cos_values, sin_values);
        for (uint32_t head = 0; head < N_HEADS; ++head) {
            uint16_t *q_head = q_rope +
                ((size_t)row * N_HEADS + head) * HEAD_DIM;
            for (uint32_t pair = 0; pair < QK_ROPE_DIM / 2; ++pair) {
                const uint32_t first = QK_NOPE_DIM + 2u * pair;
                const uint32_t second = first + 1u;
                const float x0 = bf16_to_f32_test(q_head[first]);
                const float x1 = bf16_to_f32_test(q_head[second]);
                q_head[first] = f32_to_bf16_test(
                    x0 * cos_values[pair] - x1 * sin_values[pair]);
                q_head[second] = f32_to_bf16_test(
                    x0 * sin_values[pair] + x1 * cos_values[pair]);
            }
        }
    }

    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t visible_rows = committed + row + 1u;
        for (uint32_t head = 0; head < N_HEADS; ++head) {
            const uint16_t *q_head = q_rope +
                ((size_t)row * N_HEADS + head) * HEAD_DIM;
            float accumulator[HEAD_DIM];
            memset(accumulator, 0, sizeof(accumulator));
            float maximum = -INFINITY;
            float denominator = 0.0f;

            for (uint32_t key = 0; key < visible_rows; ++key) {
                const uint16_t *kv = cache + (size_t)key * KV_LORA_RANK;
                float dot = 0.0f;
                for (uint32_t dim = 0; dim < HEAD_DIM; ++dim) {
                    dot += bf16_to_f32_test(q_head[dim]) *
                        f16_to_f32_test(kv[dim]);
                }
                const float score = dot * scale;
                const float next_maximum = fmaxf(maximum, score);
                const float correction = isinf(maximum)
                    ? 0.0f : expf(maximum - next_maximum);
                const float probability = expf(score - next_maximum);
                denominator = denominator * correction + probability;
                for (uint32_t dim = 0; dim < HEAD_DIM; ++dim) {
                    accumulator[dim] = accumulator[dim] * correction +
                        probability * f16_to_f32_test(kv[dim]);
                }
                maximum = next_maximum;
            }

            const float sink = aw->attn_sink[head];
            const float next_maximum = fmaxf(maximum, sink);
            const float correction = isinf(maximum)
                ? 0.0f : expf(maximum - next_maximum);
            denominator = denominator * correction + expf(sink - next_maximum);
            const float inverse = denominator > 0.0f
                ? 1.0f / denominator : 0.0f;
            uint16_t *out_head = output +
                ((size_t)row * N_HEADS + head) * HEAD_DIM;
            for (uint32_t dim = 0; dim < HEAD_DIM; ++dim) {
                out_head[dim] = f32_to_bf16_test(
                    accumulator[dim] * correction * inverse);
            }
        }
    }

    free(q_rope);
    return 1;
}

static int run_batched_sdpa_tests(MlaPipes *P, AttnWeights *aw) {
    enum { SDPA_WIDTH = N_HEADS * HEAD_DIM };
    const uint32_t positions[KV_TEST_CAPACITY] = {11, 12, 13, 14, 15};
    const size_t q_elements = (size_t)KV_TEST_CAPACITY * SDPA_WIDTH;
    const size_t q_bytes = q_elements * sizeof(uint16_t);
    const size_t cache_elements = (size_t)KV_TEST_CACHE_ROWS * KV_LORA_RANK;
    const size_t cache_bytes = cache_elements * sizeof(uint16_t);
    uint16_t *q_input = malloc(q_bytes);
    uint16_t *q_snapshot = malloc(q_bytes);
    uint16_t *cache = malloc(cache_bytes);
    uint16_t *cache_snapshot = malloc(cache_bytes);
    uint16_t *batch_output = malloc(q_bytes);
    uint16_t *reference_output = malloc(q_bytes);
    uint16_t *cpu_output = malloc(q_bytes);
    uint16_t *full_output = malloc(q_bytes);
    uint16_t *partition_output = malloc(q_bytes);
    uint16_t *mutated_output = malloc(q_bytes);
    uint16_t *mutated_cache = malloc(cache_bytes);
    float *batch_f32 = malloc(q_elements * sizeof(float));
    float *reference_f32 = malloc(q_elements * sizeof(float));
    if (!q_input || !q_snapshot || !cache || !cache_snapshot ||
        !batch_output || !reference_output || !cpu_output || !full_output ||
        !partition_output || !mutated_output || !mutated_cache ||
        !batch_f32 || !reference_f32) {
        printf("FAIL: batched SDPA test allocation\n");
        free(q_input); free(q_snapshot); free(cache); free(cache_snapshot);
        free(batch_output); free(reference_output); free(cpu_output);
        free(full_output); free(partition_output); free(mutated_output);
        free(mutated_cache); free(batch_f32); free(reference_f32);
        return 0;
    }

    // Positive pre-RoPE BF16 queries make the future-row mutation test
    // deterministic while retaining distinct token/head/dimension values.
    for (uint32_t row = 0; row < KV_TEST_CAPACITY; ++row) {
        for (uint32_t head = 0; head < N_HEADS; ++head) {
            for (uint32_t dim = 0; dim < HEAD_DIM; ++dim) {
                const float value = 0.25f + 0.00390625f * (float)row +
                    0.000244140625f * (float)(head % 11u) +
                    0.0001220703125f * (float)(dim % 23u);
                q_input[((size_t)row * N_HEADS + head) * HEAD_DIM + dim] =
                    f32_to_bf16_test(value);
            }
        }
    }
    for (uint32_t row = 0; row < KV_TEST_CACHE_ROWS; ++row) {
        for (uint32_t dim = 0; dim < KV_LORA_RANK; ++dim) {
            const int pattern = (int)((row * 13u + dim * 7u) % 29u) - 14;
            const float value = 0.015625f * (float)pattern +
                0.0009765625f * (float)row;
            cache[(size_t)row * KV_LORA_RANK + dim] =
                f32_to_f16_test(value);
        }
    }
    memcpy(q_snapshot, q_input, q_bytes);
    memcpy(cache_snapshot, cache, cache_bytes);

    NativeBatchWorkspace workspace = {0};
    const uint64_t generous_bytes = 1024u * 1024u;
    if (moe_infer_batch_workspace_init(
            &workspace, KV_TEST_CAPACITY, generous_bytes, generous_bytes) !=
            NATIVE_BATCH_OK) {
        printf("FAIL: batched SDPA workspace init\n");
        return 0;
    }

    const uint32_t cases[] = {1u, 2u, 3u, 5u};
    int ok = 1;
    for (size_t case_index = 0;
         case_index < sizeof(cases) / sizeof(cases[0]); ++case_index) {
        const uint32_t rows = cases[case_index];
        MLAAttentionMetrics metrics = {0};
        memset(batch_output, 0, q_bytes);
        const int rc = run_sdpa_stage_once(
            P, aw, cache, KV_TEST_START, positions, rows,
            q_input, batch_output, &workspace, &metrics);
        if (rc != NATIVE_BATCH_OK || metrics.dispatches != 2u ||
            metrics.sdpa_ns == 0u ||
            memcmp(q_snapshot, q_input, q_bytes) != 0 ||
            memcmp(cache_snapshot, cache, cache_bytes) != 0) {
            printf("FAIL: SDPA batch B=%u rc=%d dispatches=%u sdpa_ns=%llu immutable=%d\n",
                   rows, rc, metrics.dispatches,
                   (unsigned long long)metrics.sdpa_ns,
                   memcmp(q_snapshot, q_input, q_bytes) == 0 &&
                       memcmp(cache_snapshot, cache, cache_bytes) == 0);
            ok = 0;
            continue;
        }

        uint32_t reference_dispatches = 0u;
        for (uint32_t row = 0; row < rows; ++row) {
            MLAAttentionMetrics reference_metrics = {0};
            const int reference_rc = run_sdpa_stage_once(
                P, aw, cache, KV_TEST_START + row, positions + row, 1u,
                q_input + (size_t)row * SDPA_WIDTH,
                reference_output + (size_t)row * SDPA_WIDTH,
                &workspace, &reference_metrics);
            if (reference_rc != NATIVE_BATCH_OK ||
                reference_metrics.dispatches != 2u) {
                printf("FAIL: SDPA repeated B=1 row=%u rc=%d dispatches=%u\n",
                       row, reference_rc, reference_metrics.dispatches);
                ok = 0;
                break;
            }
            reference_dispatches += reference_metrics.dispatches;
        }
        if (!ok) continue;
        if (!cpu_sdpa_reference(
                aw, q_input, cache, KV_TEST_START, positions,
                rows, cpu_output)) {
            printf("FAIL: SDPA CPU reference B=%u\n", rows);
            ok = 0;
            continue;
        }

        const size_t compared = (size_t)rows * SDPA_WIDTH;
        for (size_t index = 0; index < compared; ++index) {
            batch_f32[index] = bf16_to_f32_test(batch_output[index]);
            reference_f32[index] = bf16_to_f32_test(reference_output[index]);
        }
        BatchOracleReport repeated_report;
        const BatchOracleStatus repeated_status = batch_oracle_compare_f32(
            reference_f32, batch_f32, compared,
            BATCH_ORACLE_DTYPE_BF16,
            BATCH_ORACLE_CHECKPOINT_ATTENTION_SDPA,
            0, -1, &repeated_report);

        for (size_t index = 0; index < compared; ++index) {
            reference_f32[index] = bf16_to_f32_test(cpu_output[index]);
        }
        BatchOracleReport cpu_report;
        const BatchOracleStatus cpu_status = batch_oracle_compare_f32(
            reference_f32, batch_f32, compared,
            BATCH_ORACLE_DTYPE_BF16,
            BATCH_ORACLE_CHECKPOINT_ATTENTION_SDPA,
            0, -1, &cpu_report);
        if (repeated_status != BATCH_ORACLE_OK ||
            cpu_status != BATCH_ORACLE_OK ||
            reference_dispatches != rows * 2u) {
            printf("FAIL: SDPA batch B=%u repeated=%s cpu=%s "
                   "repeated_max=%.3e cpu_max=%.3e batch_dispatch=%u "
                   "reference_dispatch=%u\n",
                   rows, batch_oracle_status_name(repeated_status),
                   batch_oracle_status_name(cpu_status),
                   repeated_report.max_absolute_error,
                   cpu_report.max_absolute_error, metrics.dispatches,
                   reference_dispatches);
            ok = 0;
        } else {
            printf("SDPA batch B=%u PASS pos=11..%u repeated_max=%.3e "
                   "cpu_max=%.3e dispatches=2 vs %u sdpa_ms=%.3f%s\n",
                   rows, positions[rows - 1u],
                   repeated_report.max_absolute_error,
                   cpu_report.max_absolute_error, reference_dispatches,
                   (double)metrics.sdpa_ns / 1.0e6,
                   rows == 3u ? " (odd/tail)" : "");
        }
        if (rows == KV_TEST_CAPACITY) memcpy(full_output, batch_output, q_bytes);
    }

    // The same absolute positions and full f16 cache must make chunking
    // irrelevant: a single [5] stage equals [2,2,1].
    if (ok) {
        const uint32_t partitions[] = {2u, 2u, 1u};
        uint32_t offset = 0u;
        uint32_t dispatches = 0u;
        for (size_t part = 0;
             part < sizeof(partitions) / sizeof(partitions[0]); ++part) {
            const uint32_t rows = partitions[part];
            MLAAttentionMetrics metrics = {0};
            const int rc = run_sdpa_stage_once(
                P, aw, cache, KV_TEST_START + offset, positions + offset, rows,
                q_input + (size_t)offset * SDPA_WIDTH,
                partition_output + (size_t)offset * SDPA_WIDTH,
                &workspace, &metrics);
            if (rc != NATIVE_BATCH_OK || metrics.dispatches != 2u) {
                ok = 0;
                break;
            }
            dispatches += metrics.dispatches;
            offset += rows;
        }
        for (size_t index = 0; index < q_elements; ++index) {
            batch_f32[index] = bf16_to_f32_test(partition_output[index]);
            reference_f32[index] = bf16_to_f32_test(full_output[index]);
        }
        BatchOracleReport report;
        const BatchOracleStatus status = batch_oracle_compare_f32(
            reference_f32, batch_f32, q_elements,
            BATCH_ORACLE_DTYPE_BF16,
            BATCH_ORACLE_CHECKPOINT_ATTENTION_SDPA,
            0, -1, &report);
        if (!ok || offset != KV_TEST_CAPACITY || dispatches != 6u ||
            status != BATCH_ORACLE_OK) {
            printf("FAIL: SDPA partition [2,2,1] status=%s dispatches=%u\n",
                   batch_oracle_status_name(status), dispatches);
            ok = 0;
        } else {
            printf("SDPA partition PASS: [5] == [2,2,1], max_abs=%.3e, dispatches=%u\n",
                   report.max_absolute_error, dispatches);
        }
    }

    // Mutating staged row 4 cannot affect rows 0..3, while row 4 may see it.
    if (ok) {
        memcpy(mutated_cache, cache, cache_bytes);
        for (uint32_t dim = 0; dim < KV_LORA_RANK; ++dim) {
            mutated_cache[(size_t)(KV_TEST_START + 4u) * KV_LORA_RANK + dim] =
                f32_to_f16_test(4.0f);
        }
        memcpy(cache_snapshot, mutated_cache, cache_bytes);
        MLAAttentionMetrics metrics = {0};
        const int rc = run_sdpa_stage_once(
            P, aw, mutated_cache, KV_TEST_START, positions,
            KV_TEST_CAPACITY, q_input, mutated_output, &workspace, &metrics);
        const size_t earlier_bytes =
            (size_t)(KV_TEST_CAPACITY - 1u) * SDPA_WIDTH * sizeof(uint16_t);
        const size_t last_offset =
            (size_t)(KV_TEST_CAPACITY - 1u) * SDPA_WIDTH;
        const int earlier_equal =
            memcmp(full_output, mutated_output, earlier_bytes) == 0;
        const int visible_row_changed = memcmp(
            full_output + last_offset, mutated_output + last_offset,
            SDPA_WIDTH * sizeof(uint16_t)) != 0;
        if (rc != NATIVE_BATCH_OK || metrics.dispatches != 2u ||
            !earlier_equal || !visible_row_changed ||
            memcmp(cache_snapshot, mutated_cache, cache_bytes) != 0 ||
            memcmp(q_snapshot, q_input, q_bytes) != 0) {
            printf("FAIL: SDPA future visibility rc=%d earlier_equal=%d "
                   "visible_changed=%d immutable=%d\n",
                   rc, earlier_equal, visible_row_changed,
                   memcmp(cache_snapshot, mutated_cache, cache_bytes) == 0 &&
                       memcmp(q_snapshot, q_input, q_bytes) == 0);
            ok = 0;
        } else {
            printf("SDPA causal visibility PASS: future row 15 leaves rows 11..14 bitwise unchanged and changes row 15\n");
        }
    }

    // Exact high-water for B=5,start=11:
    // q 327680, visible KV 16384, cos/sin 640 each, aligned output at
    // 345600, total 673280. One byte less rejects before arena consumption.
    const uint64_t exact_bytes =
        sdpa_workspace_bytes_test(KV_TEST_CAPACITY, KV_TEST_START);
    if (exact_bytes != 673280u) {
        printf("FAIL: SDPA exact workspace changed: %llu\n",
               (unsigned long long)exact_bytes);
        ok = 0;
    }
    moe_infer_batch_workspace_deinit(&workspace);
    if (moe_infer_batch_workspace_init(
            &workspace, KV_TEST_CAPACITY, exact_bytes, exact_bytes) !=
            NATIVE_BATCH_OK) {
        printf("FAIL: exact-capacity SDPA workspace init\n");
        ok = 0;
    } else {
        MLAAttentionMetrics metrics = {0};
        const int rc = run_sdpa_stage_once(
            P, aw, cache, KV_TEST_START, positions, KV_TEST_CAPACITY,
            q_input, batch_output, &workspace, &metrics);
        if (rc != NATIVE_BATCH_OK || workspace.used_bytes != exact_bytes ||
            metrics.workspace_high_water_bytes != exact_bytes ||
            metrics.dispatches != 2u) {
            printf("FAIL: exact-capacity SDPA rc=%d used=%llu high_water=%llu dispatches=%u\n",
                   rc, (unsigned long long)workspace.used_bytes,
                   (unsigned long long)metrics.workspace_high_water_bytes,
                   metrics.dispatches);
            ok = 0;
        }
    }
    moe_infer_batch_workspace_deinit(&workspace);
    if (moe_infer_batch_workspace_init(
            &workspace, KV_TEST_CAPACITY, exact_bytes - 1u, exact_bytes - 1u) !=
            NATIVE_BATCH_OK) {
        printf("FAIL: one-byte-short SDPA workspace init\n");
        ok = 0;
    } else {
        memset(batch_output, 0x5a, q_bytes);
        memcpy(reference_output, batch_output, q_bytes);
        memcpy(cache_snapshot, cache, cache_bytes);
        MLAAttentionMetrics metrics = {0};
        const int rc = run_sdpa_stage_once(
            P, aw, cache, KV_TEST_START, positions, KV_TEST_CAPACITY,
            q_input, batch_output, &workspace, &metrics);
        if (rc != NATIVE_BATCH_CAPACITY_EXCEEDED ||
            workspace.used_bytes != 0u || metrics.dispatches != 0u ||
            memcmp(reference_output, batch_output, q_bytes) != 0 ||
            memcmp(cache_snapshot, cache, cache_bytes) != 0 ||
            memcmp(q_snapshot, q_input, q_bytes) != 0) {
            printf("FAIL: one-byte-short SDPA rc=%d used=%llu immutable=%d\n",
                   rc, (unsigned long long)workspace.used_bytes,
                   memcmp(reference_output, batch_output, q_bytes) == 0 &&
                       memcmp(cache_snapshot, cache, cache_bytes) == 0 &&
                       memcmp(q_snapshot, q_input, q_bytes) == 0);
            ok = 0;
        } else {
            printf("SDPA workspace bounds PASS: exact=%llu one-byte-short rejected with used=0\n",
                   (unsigned long long)exact_bytes);
        }
    }

    moe_infer_batch_workspace_deinit(&workspace);
    free(q_input); free(q_snapshot); free(cache); free(cache_snapshot);
    free(batch_output); free(reference_output); free(cpu_output);
    free(full_output); free(partition_output); free(mutated_output);
    free(mutated_cache); free(batch_f32); free(reference_f32);
    return ok;
}

typedef struct V2CaptureLog {
    uint32_t count;
    uint32_t positions[KV_TEST_CAPACITY];
} V2CaptureLog;

static uint32_t expected_v2_order_signature(uint32_t rows) {
    uint32_t signature = UINT32_C(2166136261);
    for (uint32_t row = 0u; row < rows; ++row) {
        for (uint32_t stage = 1u; stage <= 4u; ++stage) {
            signature ^= (row << 8u) | stage;
            signature *= UINT32_C(16777619);
        }
    }
    return signature;
}

static void v2_capture_callback(
    void *context, uint32_t layer, uint32_t position,
    const uint16_t *attention_bf16, uint32_t element_count) {
    V2CaptureLog *log = context;
    if (!log || layer != 0u || !attention_bf16 || element_count != DIM ||
        log->count >= KV_TEST_CAPACITY) return;
    log->positions[log->count++] = position;
}

static QuantWeight make_zero_weight(int out_dim, int in_dim) {
    const size_t packed_count = (size_t)out_dim * (in_dim / 8);
    const size_t group_count = (size_t)out_dim * (in_dim / ATTN_GROUP_SIZE);
    return (QuantWeight){
        .packed = calloc(packed_count, sizeof(uint32_t)),
        .scales = calloc(group_count, sizeof(float)),
        .biases = calloc(group_count, sizeof(float)),
        .out_dim = out_dim,
        .in_dim = in_dim,
        .group_size = ATTN_GROUP_SIZE,
    };
}

static void free_zero_weight(QuantWeight *weight) {
    if (!weight) return;
    free((void *)weight->packed);
    free((void *)weight->scales);
    free((void *)weight->biases);
    memset(weight, 0, sizeof(*weight));
}

static int init_deterministic_recurrent_engine(
    MoEInferEngine *engine, QuantWeight *comp, QuantWeight *idx) {
    if (!engine || !comp || !idx) return 0;
    *comp = make_zero_weight(CSA_OUT_DIM, DIM);
    *idx = make_zero_weight(256, DIM);
    if (!comp->packed || !comp->scales || !comp->biases ||
        !idx->packed || !idx->scales || !idx->biases) return 0;
    engine->compress_ratio[0] = 4u;
    engine->comp_wkv[0] = *comp;
    engine->comp_wgate[0] = *comp;
    engine->idx_comp_wkv[0] = *idx;
    engine->idx_comp_wgate[0] = *idx;
    const size_t comp_groups =
        (size_t)comp->out_dim * (comp->in_dim / comp->group_size);
    const size_t idx_groups =
        (size_t)idx->out_dim * (idx->in_dim / idx->group_size);
    for (size_t index = 0; index < comp_groups; ++index) {
        ((float *)comp->biases)[index] =
            ((float)((int)(index % 11u) - 5)) / 65536.0f;
    }
    for (size_t index = 0; index < idx_groups; ++index) {
        ((float *)idx->biases)[index] =
            ((float)((int)(index % 7u) - 3)) / 32768.0f;
    }
    CompressorState *cs = &engine->comp_state[0];
    cs->ratio = 4u;
    cs->out_dim = CSA_OUT_DIM;
    cs->state_kv = calloc((size_t)8u * CSA_OUT_DIM, sizeof(float));
    cs->state_score = calloc((size_t)8u * CSA_OUT_DIM, sizeof(float));
    cs->comp_kv = calloc((size_t)MAX_COMP_BLOCKS * COMP_HEAD_DIM, sizeof(float));
    cs->idx_state_kv = calloc((size_t)8u * 256u, sizeof(float));
    cs->idx_state_score = calloc((size_t)8u * 256u, sizeof(float));
    cs->idx_comp_kv = calloc((size_t)MAX_COMP_BLOCKS * IDX_HEAD_DIM, sizeof(float));
    cs->n_comp = 513u;
    cs->n_idx_comp = 513u;
    if (cs->state_kv && cs->state_score && cs->idx_state_kv &&
        cs->idx_state_score) {
        for (size_t index = 0; index < (size_t)8u * CSA_OUT_DIM; ++index) {
            cs->state_kv[index] =
                ((float)((int)(index % 17u) - 8)) / 512.0f;
            cs->state_score[index] =
                ((float)((int)(index % 13u) - 6)) / 256.0f;
        }
        for (size_t index = 0; index < (size_t)8u * 256u; ++index) {
            cs->idx_state_kv[index] =
                ((float)((int)(index % 19u) - 9)) / 512.0f;
            cs->idx_state_score[index] =
                ((float)((int)(index % 23u) - 11)) / 256.0f;
        }
    }
    return cs->state_kv && cs->state_score && cs->comp_kv &&
        cs->idx_state_kv && cs->idx_state_score && cs->idx_comp_kv;
}

static void free_deterministic_recurrent_engine(
    MoEInferEngine *engine, QuantWeight *comp, QuantWeight *idx) {
    if (engine) {
        CompressorState *cs = &engine->comp_state[0];
        free(cs->state_kv); free(cs->state_score); free(cs->comp_kv);
        free(cs->idx_state_kv); free(cs->idx_state_score); free(cs->idx_comp_kv);
        memset(cs, 0, sizeof(*cs));
    }
    free_zero_weight(comp);
    free_zero_weight(idx);
}

static void set_test_engine_lengths(MoEInferEngine *engine, uint32_t length) {
    for (uint32_t layer = 0; layer < N_LAYERS; ++layer) {
        engine->kv_cache[layer].len = (int)length;
    }
}

static int clone_recurrent_state(
    const CompressorState *source, CompressorState *clone) {
    if (!source || !clone) return 0;
    memset(clone, 0, sizeof(*clone));
    *clone = *source;
    clone->state_kv = malloc((size_t)8u * CSA_OUT_DIM * sizeof(float));
    clone->state_score = malloc((size_t)8u * CSA_OUT_DIM * sizeof(float));
    clone->comp_kv = calloc((size_t)MAX_COMP_BLOCKS * COMP_HEAD_DIM, sizeof(float));
    clone->idx_state_kv = malloc((size_t)8u * 256u * sizeof(float));
    clone->idx_state_score = malloc((size_t)8u * 256u * sizeof(float));
    clone->idx_comp_kv = calloc((size_t)MAX_COMP_BLOCKS * IDX_HEAD_DIM, sizeof(float));
    if (!clone->state_kv || !clone->state_score || !clone->comp_kv ||
        !clone->idx_state_kv || !clone->idx_state_score || !clone->idx_comp_kv) {
        return 0;
    }
    memcpy(clone->state_kv, source->state_kv,
        (size_t)8u * CSA_OUT_DIM * sizeof(float));
    memcpy(clone->state_score, source->state_score,
        (size_t)8u * CSA_OUT_DIM * sizeof(float));
    memcpy(clone->comp_kv, source->comp_kv,
        (size_t)source->n_comp * COMP_HEAD_DIM * sizeof(float));
    memcpy(clone->idx_state_kv, source->idx_state_kv,
        (size_t)8u * 256u * sizeof(float));
    memcpy(clone->idx_state_score, source->idx_state_score,
        (size_t)8u * 256u * sizeof(float));
    memcpy(clone->idx_comp_kv, source->idx_comp_kv,
        (size_t)source->n_idx_comp * IDX_HEAD_DIM * sizeof(float));
    return 1;
}

static void free_recurrent_clone(CompressorState *state) {
    if (!state) return;
    free(state->state_kv); free(state->state_score); free(state->comp_kv);
    free(state->idx_state_kv); free(state->idx_state_score);
    free(state->idx_comp_kv);
    memset(state, 0, sizeof(*state));
}

static void oracle_affine_matvec(
    float *output, const QuantWeight *weight, const float *input) {
    const int groups = weight->in_dim / weight->group_size;
    const int packed_per_group = weight->group_size / 8;
    const int packed_columns = weight->in_dim / 8;
    for (int row = 0; row < weight->out_dim; ++row) {
        float sum = 0.0f;
        for (int group = 0; group < groups; ++group) {
            const float scale = weight->scales[(size_t)row * groups + group];
            const float bias = weight->biases[(size_t)row * groups + group];
            for (int packed = 0; packed < packed_per_group; ++packed) {
                const uint32_t values = weight->packed[
                    (size_t)row * packed_columns +
                    group * packed_per_group + packed];
                for (int lane = 0; lane < 8; ++lane) {
                    const float nibble =
                        (float)((values >> (lane * 4)) & 0x0fu);
                    const size_t column = (size_t)group * weight->group_size +
                        packed * 8u + lane;
                    sum += (scale * nibble + bias) * input[column];
                }
            }
        }
        output[row] = sum;
    }
}

static void oracle_apply_compressed_rope(float *values, int position) {
    const float base = 10000000.0f;
    const float factor = 16.0f;
    const float original_max = 65536.0f;
    const float pi = 3.14159265358979323846f;
    const float low = fmaxf(0.0f, floorf(QK_ROPE_DIM *
        logf(original_max / (32.0f * 2.0f * pi)) /
        (2.0f * logf(base))));
    const float high = fminf((float)QK_ROPE_DIM - 1.0f,
        ceilf(QK_ROPE_DIM * logf(original_max / (1.0f * 2.0f * pi)) /
            (2.0f * logf(base))));
    for (uint32_t pair = 0; pair < QK_ROPE_DIM / 2; ++pair) {
        float theta = (float)position *
            powf(base, -(2.0f * (float)pair) / QK_ROPE_DIM);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f,
            ((float)pair - low) / fmaxf(0.001f, high - low)));
        theta = theta * (1.0f - ramp) + theta / factor * ramp;
        const float cosine = cosf(theta);
        const float sine = sinf(theta);
        const uint32_t first = QK_NOPE_DIM + pair;
        const uint32_t second = first + QK_ROPE_DIM / 2;
        const float x0 = values[first];
        const float x1 = values[second];
        values[first] = x0 * cosine - x1 * sine;
        values[second] = x0 * sine + x1 * cosine;
    }
}

static void oracle_recurrent_prefix(
    CompressorState *state, const QuantWeight *comp,
    const QuantWeight *idx, const uint16_t *hidden,
    const uint32_t *positions, uint32_t rows) {
    float normed[DIM];
    float comp_kv[CSA_OUT_DIM], comp_score[CSA_OUT_DIM];
    float idx_kv[256], idx_score[256];
    for (uint32_t batch_row = 0; batch_row < rows; ++batch_row) {
        const uint32_t position = positions[batch_row];
        for (uint32_t dim = 0; dim < DIM; ++dim) {
            normed[dim] = bf16_to_f32_test(
                hidden[(size_t)batch_row * DIM + dim]);
        }
        oracle_affine_matvec(comp_kv, comp, normed);
        oracle_affine_matvec(comp_score, comp, normed);
        const uint32_t slot = 4u + position % 4u;
        memcpy(state->state_kv + (size_t)slot * CSA_OUT_DIM,
            comp_kv, sizeof(comp_kv));
        memcpy(state->state_score + (size_t)slot * CSA_OUT_DIM,
            comp_score, sizeof(comp_score));
        if ((position + 1u) % 4u == 0u && state->n_comp < MAX_COMP_BLOCKS) {
            float *pooled = state->comp_kv +
                (size_t)state->n_comp * COMP_HEAD_DIM;
            for (uint32_t dim = 0; dim < COMP_HEAD_DIM; ++dim) {
                float maximum = -1e30f;
                for (uint32_t row = 0; row < 4u; ++row) {
                    const float previous =
                        state->state_score[(size_t)row * CSA_OUT_DIM + dim];
                    const float current = state->state_score[
                        (size_t)(4u + row) * CSA_OUT_DIM + COMP_HEAD_DIM + dim];
                    if (previous > maximum) maximum = previous;
                    if (current > maximum) maximum = current;
                }
                float denominator = 0.0f, sum = 0.0f;
                for (uint32_t row = 0; row < 4u; ++row) {
                    const float previous_weight = expf(
                        state->state_score[(size_t)row * CSA_OUT_DIM + dim] - maximum);
                    const float current_weight = expf(state->state_score[
                        (size_t)(4u + row) * CSA_OUT_DIM + COMP_HEAD_DIM + dim] - maximum);
                    denominator += previous_weight + current_weight;
                    sum += previous_weight *
                        state->state_kv[(size_t)row * CSA_OUT_DIM + dim];
                    sum += current_weight * state->state_kv[
                        (size_t)(4u + row) * CSA_OUT_DIM + COMP_HEAD_DIM + dim];
                }
                pooled[dim] = sum / denominator;
            }
            oracle_apply_compressed_rope(
                pooled, (int)(position + 1u - 4u));
            state->n_comp++;
            for (uint32_t row = 0; row < 4u; ++row) {
                memcpy(state->state_kv + (size_t)row * CSA_OUT_DIM,
                    state->state_kv + (size_t)(4u + row) * CSA_OUT_DIM,
                    CSA_OUT_DIM * sizeof(float));
                memcpy(state->state_score + (size_t)row * CSA_OUT_DIM,
                    state->state_score + (size_t)(4u + row) * CSA_OUT_DIM,
                    CSA_OUT_DIM * sizeof(float));
            }
        }

        if (state->n_comp > 512u) {
            oracle_affine_matvec(idx_kv, idx, normed);
            oracle_affine_matvec(idx_score, idx, normed);
            const uint32_t idx_slot = 4u + position % 4u;
            memcpy(state->idx_state_kv + (size_t)idx_slot * 256u,
                idx_kv, sizeof(idx_kv));
            memcpy(state->idx_state_score + (size_t)idx_slot * 256u,
                idx_score, sizeof(idx_score));
            if ((position + 1u) % 4u == 0u &&
                state->n_idx_comp < MAX_COMP_BLOCKS) {
                float weights[4];
                float maximum = -1e30f;
                for (uint32_t row = 0; row < 4u; ++row) {
                    float sum = 0.0f;
                    for (uint32_t dim = 0; dim < IDX_HEAD_DIM; ++dim) {
                        sum += state->idx_state_score[(size_t)row * 256u + dim];
                    }
                    weights[row] = sum;
                    if (sum > maximum) maximum = sum;
                }
                float denominator = 0.0f;
                for (uint32_t row = 0; row < 4u; ++row) {
                    weights[row] = expf(weights[row] - maximum);
                    denominator += weights[row];
                }
                float *pooled = state->idx_comp_kv +
                    (size_t)state->n_idx_comp * IDX_HEAD_DIM;
                memset(pooled, 0, IDX_HEAD_DIM * sizeof(float));
                for (uint32_t row = 0; row < 4u; ++row) {
                    weights[row] /= denominator;
                    for (uint32_t dim = 0; dim < IDX_HEAD_DIM; ++dim) {
                        pooled[dim] += weights[row] *
                            state->idx_state_kv[(size_t)row * 256u + dim];
                    }
                }
                state->n_idx_comp++;
                for (uint32_t row = 0; row < 4u; ++row) {
                    memcpy(state->idx_state_kv + (size_t)row * 256u,
                        state->idx_state_kv + (size_t)(4u + row) * 256u,
                        256u * sizeof(float));
                    memcpy(state->idx_state_score + (size_t)row * 256u,
                        state->idx_state_score + (size_t)(4u + row) * 256u,
                        256u * sizeof(float));
                }
            }
        }
    }
}

static float recurrent_max_difference(
    const float *actual, const float *expected, size_t count) {
    float maximum = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        const float difference = fabsf(actual[index] - expected[index]);
        if (difference > maximum) maximum = difference;
    }
    return maximum;
}

static int recurrent_states_match(
    const CompressorState *actual, const CompressorState *expected) {
    if (!actual || !expected || actual->n_comp != expected->n_comp ||
        actual->n_idx_comp != expected->n_idx_comp) return 0;
    const float state_kv = recurrent_max_difference(
        actual->state_kv, expected->state_kv, (size_t)8u * CSA_OUT_DIM);
    const float state_score = recurrent_max_difference(
        actual->state_score, expected->state_score, (size_t)8u * CSA_OUT_DIM);
    const float comp = recurrent_max_difference(
        actual->comp_kv, expected->comp_kv,
        (size_t)actual->n_comp * COMP_HEAD_DIM);
    const float idx_state_kv = recurrent_max_difference(
        actual->idx_state_kv, expected->idx_state_kv, (size_t)8u * 256u);
    const float idx_state_score = recurrent_max_difference(
        actual->idx_state_score, expected->idx_state_score, (size_t)8u * 256u);
    const float idx = recurrent_max_difference(
        actual->idx_comp_kv, expected->idx_comp_kv,
        (size_t)actual->n_idx_comp * IDX_HEAD_DIM);
    const float maximum = fmaxf(fmaxf(fmaxf(state_kv, state_score), comp),
        fmaxf(fmaxf(idx_state_kv, idx_state_score), idx));
    if (maximum > 2e-5f) {
        printf("recurrent oracle max_abs state=%.3e/%.3e comp=%.3e idx_state=%.3e/%.3e idx=%.3e\n",
            state_kv, state_score, comp,
            idx_state_kv, idx_state_score, idx);
    }
    return maximum <= 2e-5f;
}

static void fill_v2_inputs(
    const float *x, uint16_t *hidden, float *residual,
    float *post, float *comb, uint32_t rows) {
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t dim = 0; dim < DIM; ++dim) {
            hidden[(size_t)row * DIM + dim] = f32_to_bf16_test(
                x[(dim + row * 31u) % DIM] * (1.0f + row / 64.0f));
            for (uint32_t stream = 0; stream < MHC_MULT; ++stream) {
                const float value = ((float)((dim + stream * 5u + row * 7u) % 29u) - 14.0f) / 64.0f;
                residual[((size_t)row * MHC_MULT + stream) * DIM + dim] =
                    bf16_to_f32_test(f32_to_bf16_test(value));
            }
        }
        for (uint32_t stream = 0; stream < MHC_MULT; ++stream) {
            post[(size_t)row * MHC_MULT + stream] =
                0.125f * (float)(stream + 1u);
            for (uint32_t source = 0; source < MHC_MULT; ++source) {
                comb[((size_t)row * MHC_MULT + source) * MHC_MULT + stream] =
                    source == stream ? 0.75f : 0.03125f;
            }
        }
    }
}

static void cpu_mhc_post_reference(
    const uint16_t *attention, const float *residual_before,
    const float *post, const float *comb, uint32_t rows, float *expected) {
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t dim = 0; dim < DIM; ++dim) {
            for (uint32_t stream = 0; stream < MHC_MULT; ++stream) {
                float value = post[(size_t)row * MHC_MULT + stream] *
                    bf16_to_f32_test(attention[(size_t)row * DIM + dim]);
                for (uint32_t source = 0; source < MHC_MULT; ++source) {
                    value += comb[((size_t)row * MHC_MULT + source) *
                                      MHC_MULT + stream] *
                        bf16_to_f32_test(f32_to_bf16_rne_test(
                            residual_before[((size_t)row * MHC_MULT + source) *
                                                DIM + dim]));
                }
                expected[((size_t)row * MHC_MULT + stream) * DIM + dim] =
                    bf16_to_f32_test(f32_to_bf16_rne_test(value));
            }
        }
    }
}

static int begin_v2_transaction(
    MoEInferEngine *engine, NativeBatchWorkspace *workspace,
    NativeKVTransaction *transaction, NativeKVStagingView *view,
    const uint32_t *positions, uint32_t rows) {
    const NativeBatchConfig config = {
        .max_prefill_chunk = KV_TEST_CAPACITY,
        .max_verify_chunk = KV_TEST_CAPACITY,
        .workspace_budget_bytes = workspace->budget_bytes,
    };
    moe_infer_kv_transaction_init(transaction);
    int rc = moe_infer_kv_transaction_begin(engine, &config,
        NATIVE_BATCH_PHASE_PREFILL, positions, rows, workspace, transaction);
    if (rc == NATIVE_BATCH_OK) {
        rc = moe_infer_kv_transaction_layer_view(
            engine, transaction, 0u, view);
    }
    return rc;
}

static int run_complete_v2_tests(
    MlaPipes *P, AttnWeights *aw, const float *x) {
    const uint32_t positions[KV_TEST_CAPACITY] = {11, 12, 13, 14, 15};
    const size_t cache_bytes =
        (size_t)KV_TEST_CACHE_ROWS * KV_LORA_RANK * sizeof(uint16_t);
    const size_t hidden_bytes =
        (size_t)KV_TEST_CAPACITY * DIM * sizeof(uint16_t);
    const size_t residual_bytes =
        (size_t)KV_TEST_CAPACITY * MHC_MULT * DIM * sizeof(float);
    const size_t output_bytes =
        (size_t)KV_TEST_CAPACITY * DIM * sizeof(uint16_t);
    uint16_t *cache = calloc(1, cache_bytes);
    uint16_t *cache_before = malloc(cache_bytes);
    uint16_t *cache_full = malloc(cache_bytes);
    uint16_t *hidden = malloc(hidden_bytes);
    float *residual = malloc(residual_bytes);
    float *residual_before = malloc(residual_bytes);
    float *expected_residual = malloc(residual_bytes);
    float *post = malloc((size_t)KV_TEST_CAPACITY * MHC_MULT * sizeof(float));
    float *comb = malloc((size_t)KV_TEST_CAPACITY * MHC_MULT * MHC_MULT * sizeof(float));
    uint16_t *output = malloc(output_bytes);
    uint16_t *capture = malloc(output_bytes);
    uint16_t *output_before = malloc(output_bytes);
    if (!cache || !cache_before || !cache_full || !hidden || !residual || !residual_before ||
        !expected_residual || !post || !comb || !output || !capture ||
        !output_before) {
        printf("FAIL: complete v2 allocation\n");
        return 0;
    }
    for (uint32_t row = 0; row < KV_TEST_CACHE_ROWS; ++row) {
        for (uint32_t dim = 0; dim < KV_LORA_RANK; ++dim) {
            cache[(size_t)row * KV_LORA_RANK + dim] = f32_to_f16_test(
                ((float)((row * 11u + dim * 3u) % 31u) - 15.0f) / 64.0f);
        }
    }
    fill_v2_inputs(x, hidden, residual, post, comb, KV_TEST_CAPACITY);
    memcpy(residual_before, residual, residual_bytes);
    memset(output, 0x5a, output_bytes);
    memset(capture, 0xa5, output_bytes);
    memcpy(output_before, output, output_bytes);

    MoEInferEngine *engine = calloc(1, sizeof(*engine));
    QuantWeight comp = {0}, idx = {0};
    NativeBatchWorkspace sizing_workspace = {0};
    int ok = engine && init_deterministic_recurrent_engine(engine, &comp, &idx);
    CompressorState initial_recurrent = {0};
    CompressorState expected_recurrent = {0};
    if (ok) {
        ok &= clone_recurrent_state(
            &engine->comp_state[0], &initial_recurrent);
        ok &= clone_recurrent_state(
            &engine->comp_state[0], &expected_recurrent);
        if (ok) {
            oracle_recurrent_prefix(&expected_recurrent, &comp, &idx,
                hidden, positions, KV_TEST_CAPACITY);
        }
    }
    if (!ok) {
        printf("FAIL: complete v2 recurrent fixture\n");
        return 0;
    }
    set_test_engine_lengths(engine, KV_TEST_START);
    engine->kv_cache[0].kv = cache;

    // First use a generous arena only to authenticate the transaction and ask
    // the implementation for its exact current-state byte requirement.
    const uint64_t generous = 4u * 1024u * 1024u;
    ok &= moe_infer_batch_workspace_init(&sizing_workspace,
        KV_TEST_CAPACITY, generous, generous) == NATIVE_BATCH_OK;
    NativeKVTransaction transaction;
    NativeKVStagingView view;
    ok &= begin_v2_transaction(engine, &sizing_workspace, &transaction, &view,
        positions, KV_TEST_CAPACITY) == NATIVE_BATCH_OK;
    MLAPrefillBatchArgs args = make_kv_args(hidden, KV_TEST_CAPACITY,
        positions, view, &transaction, &sizing_workspace);
    args.attention_output = output;
    MLARecurrentBatchState state;
    mla_recurrent_batch_state_init(&state, residual, post, comb, capture,
        KV_TEST_CAPACITY);
    V2CaptureLog log = {0};
    state.capture = v2_capture_callback;
    state.capture_context = &log;
    uint64_t exact = 0u;
    ok &= mla_attention_prefill_batch_workspace_bytes(
        engine, &args, &state, &exact) == NATIVE_BATCH_OK;

    // Authentication must fail before touching positions. PROT_NONE makes any
    // accidental pre-authentication read deterministic; invalid, stale and
    // forged views must all reject without reserving or mutating state.
    const long poison_page_size = sysconf(_SC_PAGESIZE);
    void *poison_positions = poison_page_size > 0
        ? mmap(NULL, (size_t)poison_page_size, PROT_NONE,
            MAP_PRIVATE | MAP_ANON, -1, 0)
        : MAP_FAILED;
    if (poison_positions == MAP_FAILED) {
        printf("FAIL: complete v2 poisoned positions mmap\n");
        ok = 0;
    } else {
        MLAPrefillBatchArgs poisoned = args;
        poisoned.positions = (const uint32_t *)poison_positions;
        poisoned.kv.kv_data = (void *)(uintptr_t)1u;
        const int forged_rc = mla_attention_prefill_batch_v2(
            engine, P, aw, &poisoned, &state, NULL);
        poisoned = args;
        poisoned.positions = (const uint32_t *)poison_positions;
        poisoned.kv.generation--;
        const int stale_rc = mla_attention_prefill_batch_v2(
            engine, P, aw, &poisoned, &state, NULL);
        poisoned = args;
        poisoned.positions = (const uint32_t *)poison_positions;
        poisoned.kv.reserved = 1u;
        const int invalid_rc = mla_attention_prefill_batch_v2(
            engine, P, aw, &poisoned, &state, NULL);
        if (forged_rc != NATIVE_BATCH_INVALID_ARGUMENT ||
            stale_rc != NATIVE_BATCH_INVALID_ARGUMENT ||
            invalid_rc != NATIVE_BATCH_INVALID_ARGUMENT ||
            sizing_workspace.used_bytes != 0u ||
            transaction.completed_layer_mask != 0u ||
            memcmp(residual, residual_before, residual_bytes) != 0 ||
            memcmp(output, output_before, output_bytes) != 0 ||
            !recurrent_states_match(
                &engine->comp_state[0], &initial_recurrent)) {
            printf("FAIL: poisoned positions auth ordering forged/stale/invalid=%d/%d/%d used=%llu\n",
                forged_rc, stale_rc, invalid_rc,
                (unsigned long long)sizing_workspace.used_bytes);
            ok = 0;
        } else {
            printf("Complete v2 authentication PASS: poisoned positions untouched for forged/stale/invalid views\n");
        }
        munmap(poison_positions, (size_t)poison_page_size);
    }
    if (transaction.active) {
        ok &= moe_infer_kv_transaction_abort(engine, &transaction) == NATIVE_BATCH_OK;
    }
    moe_infer_batch_workspace_deinit(&sizing_workspace);

    // One byte short rejects before KV, output, mHC, compressor or indexer
    // mutation. The live transaction remains abortable and unmarked.
    NativeBatchWorkspace short_workspace = {0};
    ok &= moe_infer_batch_workspace_init(&short_workspace,
        KV_TEST_CAPACITY, exact - 1u, exact - 1u) == NATIVE_BATCH_OK;
    set_test_engine_lengths(engine, KV_TEST_START);
    ok &= begin_v2_transaction(engine, &short_workspace, &transaction, &view,
        positions, KV_TEST_CAPACITY) == NATIVE_BATCH_OK;
    args = make_kv_args(hidden, KV_TEST_CAPACITY, positions, view,
        &transaction, &short_workspace);
    args.attention_output = output;
    memcpy(cache_before, cache, cache_bytes);
    const int short_rc = mla_attention_prefill_batch_v2(
        engine, P, aw, &args, &state, NULL);
    if (short_rc != NATIVE_BATCH_CAPACITY_EXCEEDED ||
        short_workspace.used_bytes != 0u ||
        memcmp(cache, cache_before, cache_bytes) != 0 ||
        memcmp(residual, residual_before, residual_bytes) != 0 ||
        memcmp(output, output_before, output_bytes) != 0 ||
        engine->comp_state[0].n_comp != 513u ||
        engine->comp_state[0].n_idx_comp != 513u ||
        transaction.completed_layer_mask != 0u) {
        printf("FAIL: complete v2 one-byte-short atomic rc=%d used=%llu\n",
            short_rc, (unsigned long long)short_workspace.used_bytes);
        ok = 0;
    }
    if (transaction.active) {
        ok &= moe_infer_kv_transaction_abort(engine, &transaction) == NATIVE_BATCH_OK;
    }
    moe_infer_batch_workspace_deinit(&short_workspace);

    // Exact arena success, 12 fixed dispatches and independent mHC/state oracle.
    NativeBatchWorkspace workspace = {0};
    ok &= moe_infer_batch_workspace_init(&workspace,
        KV_TEST_CAPACITY, exact, exact) == NATIVE_BATCH_OK;
    set_test_engine_lengths(engine, KV_TEST_START);
    ok &= begin_v2_transaction(engine, &workspace, &transaction, &view,
        positions, KV_TEST_CAPACITY) == NATIVE_BATCH_OK;
    args = make_kv_args(hidden, KV_TEST_CAPACITY, positions, view,
        &transaction, &workspace);
    args.attention_output = output;
    MLAAttentionMetrics metrics = {0};
    const int rc = mla_attention_prefill_batch_v2(
        engine, P, aw, &args, &state, &metrics);
    cpu_mhc_post_reference(output, residual_before, post, comb,
        KV_TEST_CAPACITY, expected_residual);
    if (rc != NATIVE_BATCH_OK || metrics.dispatches != 12u ||
        metrics.operator_order_signature !=
            expected_v2_order_signature(KV_TEST_CAPACITY) ||
        workspace.metal_buffer_creations != 1u ||
        workspace.used_bytes != exact || metrics.workspace_high_water_bytes != exact ||
        memcmp(output, capture, output_bytes) != 0 ||
        memcmp(residual, expected_residual, residual_bytes) != 0 ||
        state.completed_tokens != KV_TEST_CAPACITY ||
        engine->comp_state[0].n_comp != 515u ||
        engine->comp_state[0].n_idx_comp != 515u ||
        !recurrent_states_match(
            &engine->comp_state[0], &expected_recurrent) ||
        transaction.completed_layer_mask != 1u ||
        engine->kv_cache[0].len != KV_TEST_CACHE_ROWS ||
        log.count != KV_TEST_CAPACITY) {
        printf("FAIL: complete v2 rc=%d dispatch=%u order=%08x buffers=%llu used=%llu exact=%llu counts=%u/%u capture=%u\n",
            rc, metrics.dispatches, metrics.operator_order_signature,
            (unsigned long long)workspace.metal_buffer_creations,
            (unsigned long long)workspace.used_bytes,
            (unsigned long long)exact, engine->comp_state[0].n_comp,
            engine->comp_state[0].n_idx_comp, log.count);
        ok = 0;
    }
    for (uint32_t row = 0; row < log.count; ++row) {
        if (log.positions[row] != positions[row]) ok = 0;
    }
    if (transaction.active) {
        ok &= moe_infer_kv_transaction_abort(engine, &transaction) == NATIVE_BATCH_OK;
    }
    moe_infer_batch_workspace_deinit(&workspace);
    if (ok) {
        printf("Complete v2 PASS: exact=%llu dispatches=12 strict order=%08x, one parent MTLBuffer, compressor/indexer 513->515, BF16 capture+mHC oracle\n",
            (unsigned long long)exact,
            metrics.operator_order_signature);
    }

    // Re-run the same complete boundary as [2,2,1]. Recurrent state is carried
    // between chunks while each KV transaction is closed before the next one.
    memcpy(output_before, output, output_bytes);
    memcpy(cache_full, cache, cache_bytes);
    free_deterministic_recurrent_engine(engine, &comp, &idx);
    memset(engine, 0, sizeof(*engine));
    ok &= init_deterministic_recurrent_engine(engine, &comp, &idx);
    engine->kv_cache[0].kv = cache;
    memcpy(cache, cache_before, cache_bytes);
    memcpy(residual, residual_before, residual_bytes);
    memset(capture, 0, output_bytes);
    ok &= moe_infer_batch_workspace_init(&workspace,
        KV_TEST_CAPACITY, exact, exact) == NATIVE_BATCH_OK;
    const uint32_t partitions[] = {2u, 2u, 1u};
    uint32_t offset = 0u;
    uint32_t partition_dispatches = 0u;
    for (size_t part = 0; ok && part < 3u; ++part) {
        const uint32_t rows = partitions[part];
        set_test_engine_lengths(engine, KV_TEST_START + offset);
        moe_infer_batch_workspace_reset(&workspace);
        ok &= begin_v2_transaction(engine, &workspace, &transaction, &view,
            positions + offset, rows) == NATIVE_BATCH_OK;
        args = make_kv_args(hidden + (size_t)offset * DIM, rows,
            positions + offset, view, &transaction, &workspace);
        args.attention_output = capture + (size_t)offset * DIM;
        MLARecurrentBatchState partition_state;
        mla_recurrent_batch_state_init(&partition_state,
            residual + (size_t)offset * MHC_MULT * DIM,
            post + (size_t)offset * MHC_MULT,
            comb + (size_t)offset * MHC_MULT * MHC_MULT,
            NULL, rows);
        MLAAttentionMetrics partition_metrics = {0};
        const int partition_rc = mla_attention_prefill_batch_v2(
            engine, P, aw, &args, &partition_state, &partition_metrics);
        if (partition_rc != NATIVE_BATCH_OK ||
            partition_metrics.dispatches != 12u ||
            partition_state.completed_tokens != rows) {
            printf("FAIL: complete v2 partition part=%zu rc=%d dispatch=%u\n",
                part, partition_rc, partition_metrics.dispatches);
            ok = 0;
        }
        partition_dispatches += partition_metrics.dispatches;
        if (transaction.active) {
            ok &= moe_infer_kv_transaction_abort(engine, &transaction) ==
                NATIVE_BATCH_OK;
        }
        offset += rows;
    }
    if (ok && (offset != KV_TEST_CAPACITY || partition_dispatches != 36u ||
        workspace.metal_buffer_creations != 1u ||
        memcmp(capture, output_before, output_bytes) != 0 ||
        memcmp(residual, expected_residual, residual_bytes) != 0 ||
        memcmp(cache, cache_full, cache_bytes) != 0 ||
        engine->comp_state[0].n_comp != 515u ||
        engine->comp_state[0].n_idx_comp != 515u ||
        !recurrent_states_match(
            &engine->comp_state[0], &expected_recurrent))) {
        printf("FAIL: complete v2 [5] != [2,2,1] dispatch=%u counts=%u/%u\n",
            partition_dispatches, engine->comp_state[0].n_comp,
            engine->comp_state[0].n_idx_comp);
        ok = 0;
    } else if (ok) {
        printf("Complete v2 partition PASS: [5] == [2,2,1] output/KV/mHC/recurrent state, dispatches=%u\n",
            partition_dispatches);
    }
    moe_infer_batch_workspace_deinit(&workspace);

    // Complete odd/tail coverage at B=3, including exact capacity, immutable
    // inputs, independent recurrent state, mHC, output, and staged KV.
    free_deterministic_recurrent_engine(engine, &comp, &idx);
    memset(engine, 0, sizeof(*engine));
    ok &= init_deterministic_recurrent_engine(engine, &comp, &idx);
    set_test_engine_lengths(engine, KV_TEST_START);
    engine->kv_cache[0].kv = cache;
    memcpy(cache, cache_before, cache_bytes);
    memcpy(residual, residual_before, residual_bytes);
    memset(output, 0x5a, output_bytes);
    memset(capture, 0xa5, output_bytes);
    CompressorState expected_b3 = {0};
    ok &= clone_recurrent_state(&initial_recurrent, &expected_b3);
    if (ok) {
        oracle_recurrent_prefix(
            &expected_b3, &comp, &idx, hidden, positions, 3u);
    }
    uint16_t *hidden_snapshot = malloc((size_t)3u * DIM * sizeof(uint16_t));
    float *post_snapshot = malloc((size_t)3u * MHC_MULT * sizeof(float));
    float *comb_snapshot = malloc(
        (size_t)3u * MHC_MULT * MHC_MULT * sizeof(float));
    if (!hidden_snapshot || !post_snapshot || !comb_snapshot) ok = 0;
    if (ok) {
        memcpy(hidden_snapshot, hidden,
            (size_t)3u * DIM * sizeof(uint16_t));
        memcpy(post_snapshot, post,
            (size_t)3u * MHC_MULT * sizeof(float));
        memcpy(comb_snapshot, comb,
            (size_t)3u * MHC_MULT * MHC_MULT * sizeof(float));
    }
    uint64_t exact_b3 = 0u;
    ok &= moe_infer_batch_workspace_init(
        &workspace, KV_TEST_CAPACITY, generous, generous) == NATIVE_BATCH_OK;
    ok &= begin_v2_transaction(
        engine, &workspace, &transaction, &view, positions, 3u) == NATIVE_BATCH_OK;
    args = make_kv_args(hidden, 3u, positions, view, &transaction, &workspace);
    args.attention_output = output;
    MLARecurrentBatchState state_b3;
    mla_recurrent_batch_state_init(
        &state_b3, residual, post, comb, capture, 3u);
    ok &= mla_attention_prefill_batch_workspace_bytes(
        engine, &args, &state_b3, &exact_b3) == NATIVE_BATCH_OK;
    if (transaction.active) {
        ok &= moe_infer_kv_transaction_abort(engine, &transaction) == NATIVE_BATCH_OK;
    }
    moe_infer_batch_workspace_deinit(&workspace);
    ok &= moe_infer_batch_workspace_init(
        &workspace, 3u, exact_b3, exact_b3) == NATIVE_BATCH_OK;
    set_test_engine_lengths(engine, KV_TEST_START);
    ok &= begin_v2_transaction(
        engine, &workspace, &transaction, &view, positions, 3u) == NATIVE_BATCH_OK;
    args = make_kv_args(hidden, 3u, positions, view, &transaction, &workspace);
    args.attention_output = output;
    MLAAttentionMetrics metrics_b3 = {0};
    const int rc_b3 = mla_attention_prefill_batch_v2(
        engine, P, aw, &args, &state_b3, &metrics_b3);
    const size_t b3_output_bytes = (size_t)3u * DIM * sizeof(uint16_t);
    const size_t b3_residual_bytes =
        (size_t)3u * MHC_MULT * DIM * sizeof(float);
    const size_t b3_cache_prefix =
        (size_t)(KV_TEST_START + 3u) * KV_LORA_RANK * sizeof(uint16_t);
    if (rc_b3 != NATIVE_BATCH_OK || metrics_b3.dispatches != 12u ||
        metrics_b3.operator_order_signature != expected_v2_order_signature(3u) ||
        workspace.metal_buffer_creations != 1u ||
        workspace.used_bytes != exact_b3 ||
        metrics_b3.workspace_high_water_bytes != exact_b3 ||
        state_b3.completed_tokens != 3u ||
        memcmp(output, output_before, b3_output_bytes) != 0 ||
        memcmp(output, capture, b3_output_bytes) != 0 ||
        memcmp(residual, expected_residual, b3_residual_bytes) != 0 ||
        !recurrent_states_match(&engine->comp_state[0], &expected_b3) ||
        memcmp(cache, cache_full, b3_cache_prefix) != 0 ||
        memcmp((uint8_t *)cache + b3_cache_prefix,
            (uint8_t *)cache_before + b3_cache_prefix,
            cache_bytes - b3_cache_prefix) != 0 ||
        memcmp(hidden, hidden_snapshot,
            (size_t)3u * DIM * sizeof(uint16_t)) != 0 ||
        memcmp(post, post_snapshot,
            (size_t)3u * MHC_MULT * sizeof(float)) != 0 ||
        memcmp(comb, comb_snapshot,
            (size_t)3u * MHC_MULT * MHC_MULT * sizeof(float)) != 0 ||
        transaction.completed_layer_mask != 1u ||
        engine->kv_cache[0].len != KV_TEST_START + 3u) {
        printf("FAIL: complete v2 B=3 rc=%d dispatch=%u used=%llu exact=%llu\n",
            rc_b3, metrics_b3.dispatches,
            (unsigned long long)workspace.used_bytes,
            (unsigned long long)exact_b3);
        ok = 0;
    } else {
        printf("Complete v2 B=3 PASS: exact=%llu dispatches=12 odd/tail immutable oracle\n",
            (unsigned long long)exact_b3);
    }
    if (transaction.active) {
        ok &= moe_infer_kv_transaction_abort(engine, &transaction) == NATIVE_BATCH_OK;
    }
    moe_infer_batch_workspace_deinit(&workspace);
    free(hidden_snapshot); free(post_snapshot); free(comb_snapshot);
    free_recurrent_clone(&expected_b3);

    // Inject after recurrent row 2. Deep snapshots must restore staged KV,
    // every recurrent array/count and all mHC rows; output/capture stay hidden.
    free_deterministic_recurrent_engine(engine, &comp, &idx);
    memset(engine, 0, sizeof(*engine));
    ok &= init_deterministic_recurrent_engine(engine, &comp, &idx);
    set_test_engine_lengths(engine, KV_TEST_START);
    engine->kv_cache[0].kv = cache;
    memcpy(cache, cache_before, cache_bytes);
    memcpy(residual, residual_before, residual_bytes);
    memset(output, 0x5a, output_bytes);
    memcpy(output_before, output, output_bytes);
    memset(capture, 0xa5, output_bytes);
    mla_recurrent_batch_state_init(&state, residual, post, comb, capture,
        KV_TEST_CAPACITY);
    state.inject_failure_after_row = 2u;
    ok &= moe_infer_batch_workspace_init(&workspace,
        KV_TEST_CAPACITY, exact, exact) == NATIVE_BATCH_OK;
    ok &= begin_v2_transaction(engine, &workspace, &transaction, &view,
        positions, KV_TEST_CAPACITY) == NATIVE_BATCH_OK;
    args = make_kv_args(hidden, KV_TEST_CAPACITY, positions, view,
        &transaction, &workspace);
    args.attention_output = output;
    const int injected_rc = mla_attention_prefill_batch_v2(
        engine, P, aw, &args, &state, NULL);
    int capture_unchanged = 1;
    for (size_t index = 0; index < output_bytes; ++index) {
        if (((uint8_t *)capture)[index] != 0xa5u) { capture_unchanged = 0; break; }
    }
    if (injected_rc != NATIVE_BATCH_INVALID_STATE ||
        memcmp(cache, cache_before, cache_bytes) != 0 ||
        memcmp(residual, residual_before, residual_bytes) != 0 ||
        memcmp(output, output_before, output_bytes) != 0 || !capture_unchanged ||
        state.completed_tokens != 0u ||
        engine->comp_state[0].n_comp != 513u ||
        engine->comp_state[0].n_idx_comp != 513u ||
        !recurrent_states_match(
            &engine->comp_state[0], &initial_recurrent) ||
        transaction.completed_layer_mask != 0u ||
        engine->kv_cache[0].len != KV_TEST_START) {
        printf("FAIL: complete v2 injected rollback rc=%d counts=%u/%u capture_unchanged=%d\n",
            injected_rc, engine->comp_state[0].n_comp,
            engine->comp_state[0].n_idx_comp, capture_unchanged);
        ok = 0;
    } else {
        printf("Complete v2 rollback PASS: injected row 2 restored KV+compressor+indexer+mHC atomically\n");
    }
    if (transaction.active) {
        ok &= moe_infer_kv_transaction_abort(engine, &transaction) == NATIVE_BATCH_OK;
    }
    moe_infer_batch_workspace_deinit(&workspace);

    free_deterministic_recurrent_engine(engine, &comp, &idx);
    free_recurrent_clone(&initial_recurrent);
    free_recurrent_clone(&expected_recurrent);
    free(engine);
    free(cache); free(cache_before); free(hidden); free(residual);
    free(residual_before); free(expected_residual); free(post); free(comb);
    free(output); free(capture); free(output_before);
    return ok;
}

static uint64_t output_workspace_bytes_test(uint32_t rows) {
    const uint64_t attn =
        (uint64_t)rows * N_HEADS * HEAD_DIM * sizeof(uint16_t);
    const uint64_t rope =
        (uint64_t)rows * (QK_ROPE_DIM / 2) * sizeof(float);
    const uint64_t concat =
        (uint64_t)rows * O_GROUPS * O_LORA_RANK * sizeof(float);
    const uint64_t projected = (uint64_t)rows * DIM * sizeof(float);
    const uint64_t cos_rel = align_workspace_test(attn);
    const uint64_t sin_rel = align_workspace_test(cos_rel + rope);
    const uint64_t concat_rel = align_workspace_test(sin_rel + rope);
    const uint64_t projected_rel = align_workspace_test(concat_rel + concat);
    return projected_rel + projected;
}

static int cpu_output_reference(
    const float *wo_a_dense,
    const float *wo_b_dense,
    const uint32_t *positions,
    const uint16_t *sdpa,
    uint32_t rows,
    uint16_t *output) {
    const size_t attn_elements = (size_t)rows * N_HEADS * HEAD_DIM;
    uint16_t *inverse = malloc(attn_elements * sizeof(uint16_t));
    float *group = malloc((size_t)(N_HEADS / O_GROUPS) * HEAD_DIM * sizeof(float));
    float *concat = malloc((size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    float *projected = malloc(DIM * sizeof(float));
    if (!inverse || !group || !concat || !projected) {
        free(inverse); free(group); free(concat); free(projected);
        return 0;
    }
    memcpy(inverse, sdpa, attn_elements * sizeof(uint16_t));
    const int group_feat = (N_HEADS / O_GROUPS) * HEAD_DIM;
    for (uint32_t row = 0; row < rows; ++row) {
        float cos_values[QK_ROPE_DIM / 2];
        float sin_values[QK_ROPE_DIM / 2];
        cpu_yarn_cos_sin_test(positions[row], cos_values, sin_values);
        uint16_t *row_values = inverse + (size_t)row * N_HEADS * HEAD_DIM;
        for (uint32_t head = 0; head < N_HEADS; ++head) {
            uint16_t *head_values = row_values + (size_t)head * HEAD_DIM;
            for (uint32_t pair = 0; pair < QK_ROPE_DIM / 2; ++pair) {
                const uint32_t first = QK_NOPE_DIM + 2u * pair;
                const uint32_t second = first + 1u;
                const float x0 = bf16_to_f32_test(head_values[first]);
                const float x1 = bf16_to_f32_test(head_values[second]);
                head_values[first] = f32_to_bf16_test(
                    x0 * cos_values[pair] + x1 * sin_values[pair]);
                head_values[second] = f32_to_bf16_test(
                    -x0 * sin_values[pair] + x1 * cos_values[pair]);
            }
        }
        for (uint32_t group_index = 0; group_index < O_GROUPS; ++group_index) {
            const uint16_t *source = row_values +
                (size_t)group_index * group_feat;
            for (int index = 0; index < group_feat; ++index) {
                group[index] = bf16_to_f32_test(source[index]);
            }
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                O_LORA_RANK, group_feat, 1.0f,
                wo_a_dense + (size_t)group_index * O_LORA_RANK * group_feat,
                group_feat, group, 1, 0.0f,
                concat + (size_t)group_index * O_LORA_RANK, 1);
        }
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
            DIM, O_GROUPS * O_LORA_RANK, 1.0f,
            wo_b_dense, O_GROUPS * O_LORA_RANK,
            concat, 1, 0.0f, projected, 1);
        for (uint32_t dim = 0; dim < DIM; ++dim) {
            output[(size_t)row * DIM + dim] =
                f32_to_bf16_test(projected[dim]);
        }
    }
    free(inverse); free(group); free(concat); free(projected);
    return 1;
}

static int run_batched_output_tests(
    MlaPipes *P,
    AttnWeights *aw,
    const float *wo_a_dense,
    const float *wo_b_dense) {
    enum { OUTPUT_CAPACITY = 5, SDPA_WIDTH = N_HEADS * HEAD_DIM };
    const uint32_t positions[OUTPUT_CAPACITY] = {11, 12, 13, 14, 15};
    const size_t sdpa_elements = (size_t)OUTPUT_CAPACITY * SDPA_WIDTH;
    const size_t output_elements = (size_t)OUTPUT_CAPACITY * DIM;
    uint16_t *sdpa = malloc(sdpa_elements * sizeof(uint16_t));
    uint16_t *snapshot = malloc(sdpa_elements * sizeof(uint16_t));
    uint16_t *batch = malloc(output_elements * sizeof(uint16_t));
    uint16_t *reference = malloc(output_elements * sizeof(uint16_t));
    uint16_t *scalar = malloc(output_elements * sizeof(uint16_t));
    uint16_t *partition = malloc(output_elements * sizeof(uint16_t));
    float *batch_f32 = malloc(output_elements * sizeof(float));
    float *reference_f32 = malloc(output_elements * sizeof(float));
    if (!sdpa || !snapshot || !batch || !reference || !scalar || !partition ||
        !batch_f32 || !reference_f32) {
        printf("FAIL: output-stage allocation\n");
        return 0;
    }
    for (uint32_t row = 0; row < OUTPUT_CAPACITY; ++row) {
        for (uint32_t head = 0; head < N_HEADS; ++head) {
            for (uint32_t dim = 0; dim < HEAD_DIM; ++dim) {
                const int pattern = (int)((row * 19u + head * 7u + dim * 3u) % 41u) - 20;
                sdpa[((size_t)row * N_HEADS + head) * HEAD_DIM + dim] =
                    f32_to_bf16_test((float)pattern / 64.0f);
            }
        }
    }
    memcpy(snapshot, sdpa, sdpa_elements * sizeof(uint16_t));

    NativeBatchWorkspace workspace = {0};
    const uint64_t generous = 2u * 1024u * 1024u;
    if (moe_infer_batch_workspace_init(
            &workspace, OUTPUT_CAPACITY, generous, generous) != NATIVE_BATCH_OK) {
        printf("FAIL: output-stage workspace init\n");
        return 0;
    }
    int ok = cpu_output_reference(
        wo_a_dense, wo_b_dense, positions, sdpa, OUTPUT_CAPACITY, reference);
    const uint32_t cases[] = {1u, 2u, 3u, 5u};
    for (size_t case_index = 0; ok && case_index < 4u; ++case_index) {
        const uint32_t rows = cases[case_index];
        MLAAttentionMetrics metrics = {0};
        moe_infer_batch_workspace_reset(&workspace);
        const int rc = mla_attention_output_batch(P, aw, positions, sdpa,
            batch, rows, &workspace, &metrics);
        uint32_t scalar_dispatches = 0u;
        for (uint32_t row = 0; row < rows; ++row) {
            MLAAttentionMetrics one = {0};
            moe_infer_batch_workspace_reset(&workspace);
            if (mla_attention_output_batch(P, aw, positions + row,
                    sdpa + (size_t)row * SDPA_WIDTH,
                    scalar + (size_t)row * DIM, 1u,
                    &workspace, &one) != NATIVE_BATCH_OK) {
                ok = 0;
                break;
            }
            scalar_dispatches += one.dispatches;
        }
        for (size_t index = 0; index < (size_t)rows * DIM; ++index) {
            batch_f32[index] = bf16_to_f32_test(batch[index]);
            reference_f32[index] = bf16_to_f32_test(reference[index]);
        }
        BatchOracleReport cpu_report;
        const BatchOracleStatus cpu_status = batch_oracle_compare_f32(
            reference_f32, batch_f32, (size_t)rows * DIM,
            BATCH_ORACLE_DTYPE_BF16,
            BATCH_ORACLE_CHECKPOINT_ATTENTION_OUTPUT,
            0, -1, &cpu_report);
        BatchOracleReport scalar_report;
        for (size_t index = 0; index < (size_t)rows * DIM; ++index) {
            reference_f32[index] = bf16_to_f32_test(scalar[index]);
        }
        const BatchOracleStatus scalar_status = batch_oracle_compare_f32(
            reference_f32, batch_f32, (size_t)rows * DIM,
            BATCH_ORACLE_DTYPE_BF16,
            BATCH_ORACLE_CHECKPOINT_ATTENTION_OUTPUT,
            0, -1, &scalar_report);
        if (rc != NATIVE_BATCH_OK || metrics.dispatches != 3u ||
            metrics.output_ns == 0u || scalar_dispatches != rows * 3u ||
            cpu_status != BATCH_ORACLE_OK || scalar_status != BATCH_ORACLE_OK ||
            memcmp(sdpa, snapshot, sdpa_elements * sizeof(uint16_t)) != 0) {
            printf("FAIL: output batch B=%u rc=%d dispatch=%u cpu=%s scalar=%s immutable=%d\n",
                rows, rc, metrics.dispatches,
                batch_oracle_status_name(cpu_status),
                batch_oracle_status_name(scalar_status),
                memcmp(sdpa, snapshot, sdpa_elements * sizeof(uint16_t)) == 0);
            ok = 0;
        } else {
            printf("Output batch B=%u PASS cpu_max=%.3e scalar_max=%.3e dispatches=3 vs %u%s\n",
                rows, cpu_report.max_absolute_error,
                scalar_report.max_absolute_error, scalar_dispatches,
                rows == 3u ? " (odd/tail)" : "");
        }
    }

    if (ok) {
        const uint32_t parts[] = {2u, 2u, 1u};
        uint32_t offset = 0u;
        for (size_t part = 0; part < 3u; ++part) {
            const uint32_t rows = parts[part];
            moe_infer_batch_workspace_reset(&workspace);
            if (mla_attention_output_batch(P, aw, positions + offset,
                    sdpa + (size_t)offset * SDPA_WIDTH,
                    partition + (size_t)offset * DIM, rows,
                    &workspace, NULL) != NATIVE_BATCH_OK) {
                ok = 0;
                break;
            }
            offset += rows;
        }
        if (!ok || memcmp(batch, partition,
                output_elements * sizeof(uint16_t)) != 0) {
            printf("FAIL: output partition [5] != [2,2,1]\n");
            ok = 0;
        } else {
            printf("Output partition PASS: [5] == [2,2,1] bitwise\n");
        }
    }

    const uint64_t warm_buffer_creations = workspace.metal_buffer_creations;
    uint64_t batch_best = UINT64_MAX, scalar_best = UINT64_MAX;
    for (int iteration = -1; ok && iteration < 3; ++iteration) {
        MLAAttentionMetrics many = {0};
        moe_infer_batch_workspace_reset(&workspace);
        if (mla_attention_output_batch(P, aw, positions, sdpa,
                batch, 3u, &workspace, &many) != NATIVE_BATCH_OK) {
            ok = 0; break;
        }
        uint64_t one_total = 0u;
        for (uint32_t row = 0; row < 3u; ++row) {
            MLAAttentionMetrics one = {0};
            moe_infer_batch_workspace_reset(&workspace);
            if (mla_attention_output_batch(P, aw, positions + row,
                    sdpa + (size_t)row * SDPA_WIDTH,
                    scalar + (size_t)row * DIM, 1u,
                    &workspace, &one) != NATIVE_BATCH_OK) {
                ok = 0; break;
            }
            one_total += one.output_ns;
        }
        if (iteration >= 0) {
            if (many.output_ns < batch_best) batch_best = many.output_ns;
            if (one_total < scalar_best) scalar_best = one_total;
        }
    }
    if (ok && workspace.metal_buffer_creations != warm_buffer_creations) {
        printf("FAIL: warm output created MTLBuffer wrappers: before=%llu after=%llu\n",
            (unsigned long long)warm_buffer_creations,
            (unsigned long long)workspace.metal_buffer_creations);
        ok = 0;
    } else if (ok && !(batch_best < scalar_best)) {
        printf("FAIL: warm output B=3 not faster: %.3fms vs %.3fms\n",
            (double)batch_best / 1e6, (double)scalar_best / 1e6);
        ok = 0;
    } else if (ok) {
        printf("Output warm performance PASS: B=3 %.3fms < repeated B=1 %.3fms, MTLBuffer creations unchanged at %llu\n",
            (double)batch_best / 1e6, (double)scalar_best / 1e6,
            (unsigned long long)workspace.metal_buffer_creations);
    }

    const uint64_t exact = output_workspace_bytes_test(OUTPUT_CAPACITY);
    moe_infer_batch_workspace_deinit(&workspace);
    if (moe_infer_batch_workspace_init(
            &workspace, OUTPUT_CAPACITY, exact, exact) != NATIVE_BATCH_OK) {
        ok = 0;
    } else {
        MLAAttentionMetrics metrics = {0};
        const int rc = mla_attention_output_batch(P, aw, positions, sdpa,
            batch, OUTPUT_CAPACITY, &workspace, &metrics);
        if (rc != NATIVE_BATCH_OK || workspace.used_bytes != exact ||
            metrics.workspace_high_water_bytes != exact) ok = 0;
    }
    moe_infer_batch_workspace_deinit(&workspace);
    if (moe_infer_batch_workspace_init(
            &workspace, OUTPUT_CAPACITY, exact - 1u, exact - 1u) != NATIVE_BATCH_OK) {
        ok = 0;
    } else {
        memset(batch, 0x5a, output_elements * sizeof(uint16_t));
        memcpy(reference, batch, output_elements * sizeof(uint16_t));
        const int rc = mla_attention_output_batch(P, aw, positions, sdpa,
            batch, OUTPUT_CAPACITY, &workspace, NULL);
        if (rc != NATIVE_BATCH_CAPACITY_EXCEEDED || workspace.used_bytes != 0u ||
            memcmp(batch, reference, output_elements * sizeof(uint16_t)) != 0 ||
            memcmp(sdpa, snapshot, sdpa_elements * sizeof(uint16_t)) != 0) {
            ok = 0;
        } else {
            printf("Output workspace bounds PASS: exact=%llu one-byte-short atomic\n",
                (unsigned long long)exact);
        }
    }

    moe_infer_batch_workspace_deinit(&workspace);
    free(sdpa); free(snapshot); free(batch); free(reference); free(scalar);
    free(partition); free(batch_f32); free(reference_f32);
    return ok;
}

int main(void) {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [d newCommandQueue];
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@"src/models/moe_kernel.metal"
                                              encoding:NSUTF8StringEncoding error:&err];
    MTLCompileOptions *opts = [MTLCompileOptions new]; opts.languageVersion = MTLLanguageVersion3_1;
    id<MTLLibrary> lib = [d newLibraryWithSource:src options:opts error:&err];
    if (!lib) { printf("COMPILE FAIL: %s\n", [[err localizedDescription] UTF8String]); return 1; }

    MlaPipes P = {0};
    P.dev = d; P.queue = q;
    P.dequant_matvec_affine = mkpipe(d, lib, "dequant_matvec_affine");
    P.rms_norm_rows = mkpipe(d, lib, "rms_norm_rows");
    P.rope_tail_interleaved = mkpipe(d, lib, "rope_tail_interleaved");
    P.mla_sdpa_decode = mkpipe(d, lib, "mla_sdpa_decode");
    P.mla_sdpa_decode_f16 = mkpipe(d, lib, "mla_sdpa_decode_f16");
    P.matvec_f32 = mkpipe(d, lib, "matvec_f32");
    // F16 precision chain
    P.dequant_matvec_affine_f16out = mkpipe(d, lib, "dequant_matvec_affine_f16out");
    P.rms_norm_rows_f16out = mkpipe(d, lib, "rms_norm_rows_f16out");
    P.dequant_matvec_affine_f16in_f16out = mkpipe(d, lib, "dequant_matvec_affine_f16in_f16out");
    P.rms_norm_rows_f16in_f16out = mkpipe(d, lib, "rms_norm_rows_f16in_f16out");
    P.rope_tail_interleaved_f16 = mkpipe(d, lib, "rope_tail_interleaved_f16");
    P.matvec_f32_f16in = mkpipe(d, lib, "matvec_f32_f16in");
    P.mla_sdpa_decode_f16in_f16out = mkpipe(d, lib, "mla_sdpa_decode_f16in_f16out");
    // BF16 Q chain (Task 2.2)
    P.dequant_matvec_affine_bf16in_bf16out =
        mkpipe(d, lib, "dequant_matvec_affine_bf16in_bf16out");
    P.rms_norm_rows_bf16in_bf16out =
        mkpipe(d, lib, "rms_norm_rows_bf16in_bf16out");
    P.rope_tail_interleaved_bf16 =
        mkpipe(d, lib, "rope_tail_interleaved_bf16");
    P.mla_sdpa_batch_f16_kv =
        mkpipe(d, lib, "mla_sdpa_batch_f16_kv");
    P.f32_to_bf16_vec = mkpipe(d, lib, "f32_to_bf16_vec");
    P.mhc_post_bfloat = mkpipe(d, lib, "mhc_post_bfloat");
    P.bf16_to_f32_vec = mkpipe(d, lib, "bf16_to_f32_vec");
    // Task 2.5 packed output chain.
    P.dequant_matvec_affine_v2 =
        mkpipe(d, lib, "dequant_matvec_affine_v2");
    P.dequant_matvec_affine_bf16in_f32out_grp8 =
        mkpipe(d, lib, "dequant_matvec_affine_bf16in_f32out_grp8");

    AttnWeights aw = {0};
    aw.wq_a = load_qw("wq_a", Q_LORA_RANK, DIM);            // [1024,4096]
    aw.wq_b = load_qw("wq_b", N_HEADS * HEAD_DIM, Q_LORA_RANK); // [32768,1024]
    aw.wkv  = load_qw("wkv", KV_LORA_RANK, DIM);            // [512,4096]
    aw.wo_a = load_qw("wo_a", O_GROUPS * O_LORA_RANK,
                      (N_HEADS / O_GROUPS) * HEAD_DIM);         // [8192,4096]
    aw.wo_b = load_qw("wo_b", DIM, O_GROUPS * O_LORA_RANK); // [4096,8192]
    aw.q_norm = readf("q_norm.bin", NULL);
    aw.kv_norm = readf("kv_norm.bin", NULL);
    aw.attn_sink = readf("attn_sink.bin", NULL);

    // wo_a is DENSE f32 (loader already dequantized it).
    // Layout: [O_GROUPS, O_LORA_RANK, group_feat] flattened = [8*1024, 4096] f32.
    // gen_attn_golden.py writes wo_a_dense.bin = f32 array of that shape.
    size_t woa_n;
    float *woa_dense = readf("wo_a_dense.bin", &woa_n);
    float *wob_dense = readf("wo_b_dense.bin", NULL);
    aw.wo_a_dense = woa_dense;
    (void)woa_n;

    float *x = readf("hidden.bin", NULL);
    size_t gn; float *golden = readf("golden.bin", &gn);

    // pos from meta.txt
    int pos = 7;
    FILE *mf = fopen(GD "meta.txt", "r"); if (mf) { fscanf(mf, "pos=%d", &pos); fclose(mf); }

    uint16_t *kv_cache = calloc((size_t)MAX_SEQ_LEN * KV_LORA_RANK, sizeof(uint16_t));
    float *out = malloc(DIM * sizeof(float));
    // single token at position pos: cache_len = 1 (only current token)
    mla_attention_decode(&P, &aw, x, kv_cache, 1, pos, out);

    float maxd = 0; double ss = 0, gss = 0;
    for (int i = 0; i < DIM; i++) { float dd = fabsf(out[i] - golden[i]); if (dd > maxd) maxd = dd; ss += dd*dd; gss += (double)golden[i]*golden[i]; }
    float rel = (float)(sqrt(ss) / sqrt(gss));
    printf("out[:4]   = [%.4f %.4f %.4f %.4f]\n", out[0], out[1], out[2], out[3]);
    printf("golden[:4]= [%.4f %.4f %.4f %.4f]\n", golden[0], golden[1], golden[2], golden[3]);
    printf("max_abs_diff=%.3e  rel_L2=%.3e\n", maxd, rel);
    int ok = rel < 1e-3;
    printf("%s\n", ok ? "RESULT: GO — mla_attention_decode matches golden" : "RESULT: NO-GO");
    const int arithmetic_ok = run_workspace_arithmetic_tests();
    printf("%s\n", arithmetic_ok ?
        "RESULT: GO — MLA workspace overflow arithmetic" :
        "RESULT: NO-GO — MLA workspace overflow arithmetic");
    const int q_ok = run_batched_q_tests(&P, &aw, x);
    printf("%s\n", q_ok ? "RESULT: GO — batched MLA Q chain" :
                            "RESULT: NO-GO — batched MLA Q chain");
    const int kv_ok = run_batched_kv_tests(&P, &aw, x);
    printf("%s\n", kv_ok ? "RESULT: GO — batched MLA KV/RoPE chain" :
                             "RESULT: NO-GO — batched MLA KV/RoPE chain");
    const int sdpa_ok = run_batched_sdpa_tests(&P, &aw);
    printf("%s\n", sdpa_ok ? "RESULT: GO — batched MLA causal SDPA" :
                               "RESULT: NO-GO — batched MLA causal SDPA");
    const int output_ok = run_batched_output_tests(
        &P, &aw, woa_dense, wob_dense);
    printf("%s\n", output_ok ?
        "RESULT: GO — batched MLA packed output projection" :
        "RESULT: NO-GO — batched MLA packed output projection");
    const int v2_ok = run_complete_v2_tests(&P, &aw, x);
    printf("%s\n", v2_ok ?
        "RESULT: GO — complete transactional MLA v2 chunk boundary" :
        "RESULT: NO-GO — complete transactional MLA v2 chunk boundary");
    return (ok && arithmetic_ok && q_ok && kv_ok && sdpa_ok &&
        output_ok && v2_ok) ? 0 : 1;
}
