#include "batch_oracle.h"
#include "engine.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define EXPECT(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__); \
        ++failures; \
    } \
} while (0)

// Test-only capacity override linked exclusively by the batch-oracle target.
// It lets V4 prove that the production engine's hard B=8 gate is independent
// of a larger valid NativeBatchConfig capacity, then restore immutable state.
uint32_t dmlx_task45_test_override_max_prefill_chunk(
    MoEInferEngine *engine,
    uint32_t capacity) {
    if (!engine || capacity == 0 || capacity > MAX_SEQ_LEN) return 0;
    const uint32_t previous = engine->batch_max_prefill_chunk;
    engine->batch_max_prefill_chunk = capacity;
    return previous;
}

typedef struct {
    MoeInferCapturePath path;
    MoeInferCaptureCheckpoint checkpoint;
    MoeInferCaptureDType dtype;
    int layer;
    int position;
    int kv_length;
    uint32_t compressor_length;
    uint32_t indexer_length;
    uint64_t kv_state_digest;
    uint64_t compressor_state_digest;
    uint64_t indexer_state_digest;
    float *values;
    size_t value_count;
    int *expert_ids;
    size_t expert_count;
} OwnedCapture;

typedef struct {
    OwnedCapture *records;
    size_t count;
    size_t capacity;
    bool overflow;
} CaptureSet;

typedef struct {
    bool equal;
    bool expert_id_mismatch;
    bool state_mismatch;
    int layer;
    int position;
    MoeInferCaptureCheckpoint checkpoint;
    size_t index;
    BatchOracleReport numeric;
} CaptureComparison;

static void clear_capture_set(CaptureSet *set) {
    if (!set) return;
    for (size_t index = 0; index < set->count; ++index) {
        free(set->records[index].values);
        free(set->records[index].expert_ids);
    }
    set->count = 0;
    set->overflow = false;
}

void *dmlx_batch_capture_create(void) {
    return calloc(1, sizeof(CaptureSet));
}

void dmlx_batch_capture_clear(void *context) {
    clear_capture_set((CaptureSet *)context);
}

void dmlx_batch_capture_destroy(void *context) {
    CaptureSet *set = (CaptureSet *)context;
    if (!set) return;
    clear_capture_set(set);
    free(set->records);
    free(set);
}

void dmlx_batch_capture_record(void *context, const MoeInferDebugCaptureRecord *record) {
    CaptureSet *set = (CaptureSet *)context;
    if (!set || !record ||
        (record->value_count > 0 && !record->values) ||
        (record->expert_count > 0 && !record->expert_ids)) {
        if (set) set->overflow = true;
        return;
    }

    if (set->count == set->capacity) {
        size_t new_capacity = set->capacity == 0 ? 256 : set->capacity * 2;
        OwnedCapture *records = (OwnedCapture *)realloc(
            set->records, new_capacity * sizeof(OwnedCapture));
        if (!records) {
            set->overflow = true;
            return;
        }
        set->records = records;
        set->capacity = new_capacity;
    }

    OwnedCapture owned;
    memset(&owned, 0, sizeof(owned));
    owned.path = record->path;
    owned.checkpoint = record->checkpoint;
    owned.dtype = record->dtype;
    owned.layer = record->layer;
    owned.position = record->position;
    owned.kv_length = record->kv_length;
    owned.compressor_length = record->compressor_length;
    owned.indexer_length = record->indexer_length;
    owned.kv_state_digest = record->kv_state_digest;
    owned.compressor_state_digest = record->compressor_state_digest;
    owned.indexer_state_digest = record->indexer_state_digest;
    owned.value_count = record->value_count;
    owned.expert_count = record->expert_count;

    if (owned.value_count > 0) {
        owned.values = (float *)malloc(owned.value_count * sizeof(float));
        if (!owned.values) {
            set->overflow = true;
            return;
        }
        memcpy(owned.values, record->values, owned.value_count * sizeof(float));
    }
    if (owned.expert_count > 0) {
        owned.expert_ids = (int *)malloc(owned.expert_count * sizeof(int));
        if (!owned.expert_ids) {
            free(owned.values);
            set->overflow = true;
            return;
        }
        memcpy(owned.expert_ids, record->expert_ids, owned.expert_count * sizeof(int));
    }

    set->records[set->count++] = owned;
}

static OwnedCapture *find_capture(
    CaptureSet *set,
    int layer,
    int position,
    MoeInferCaptureCheckpoint checkpoint) {
    OwnedCapture *found = NULL;
    for (size_t index = 0; index < set->count; ++index) {
        OwnedCapture *candidate = &set->records[index];
        if (candidate->layer == layer && candidate->position == position &&
            candidate->checkpoint == checkpoint) {
            if (found != NULL) return NULL;
            found = candidate;
        }
    }
    return found;
}

static BatchOracleCheckpoint oracle_checkpoint(MoeInferCaptureCheckpoint checkpoint) {
    switch (checkpoint) {
        case MOE_INFER_CAPTURE_LAYER_INPUT:
        case MOE_INFER_CAPTURE_LAYER_OUTPUT:
            return BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE;
        case MOE_INFER_CAPTURE_ATTENTION_OUTPUT:
            return BATCH_ORACLE_CHECKPOINT_ATTENTION_OUTPUT;
        case MOE_INFER_CAPTURE_ROUTER:
            return BATCH_ORACLE_CHECKPOINT_ROUTER_WEIGHTS;
        case MOE_INFER_CAPTURE_MOE_OUTPUT:
            return BATCH_ORACLE_CHECKPOINT_MOE_OUTPUT;
    }
    return BATCH_ORACLE_CHECKPOINT_COUNT;
}

static BatchOracleDType oracle_dtype(MoeInferCaptureDType dtype) {
    return dtype == MOE_INFER_CAPTURE_DTYPE_BF16
        ? BATCH_ORACLE_DTYPE_BF16
        : BATCH_ORACLE_DTYPE_F32;
}

static CaptureComparison compare_captures(
    CaptureSet *scalar,
    CaptureSet *batch,
    int start_pos,
    int n_tokens) {
    CaptureComparison result;
    memset(&result, 0, sizeof(result));
    result.layer = -1;
    result.position = -1;
    result.index = SIZE_MAX;

    const size_t expected = (size_t)N_LAYERS * (size_t)n_tokens *
        (MOE_INFER_CAPTURE_LAYER_OUTPUT - MOE_INFER_CAPTURE_LAYER_INPUT + 1);
    if (!scalar || !batch || scalar->overflow || batch->overflow ||
        scalar->count != expected || batch->count != expected) {
        return result;
    }

    // Compare in semantic forward order, independent of callback arrival order.
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        for (int token = 0; token < n_tokens; ++token) {
            const int position = start_pos + token;
            for (int checkpoint = MOE_INFER_CAPTURE_LAYER_INPUT;
                 checkpoint <= MOE_INFER_CAPTURE_LAYER_OUTPUT;
                 ++checkpoint) {
                OwnedCapture *reference = find_capture(
                    scalar, layer, position, (MoeInferCaptureCheckpoint)checkpoint);
                OwnedCapture *actual = find_capture(
                    batch, layer, position, (MoeInferCaptureCheckpoint)checkpoint);
                if (!reference || !actual ||
                    reference->path != MOE_INFER_CAPTURE_SCALAR ||
                    actual->path != MOE_INFER_CAPTURE_BATCH ||
                    reference->dtype != actual->dtype ||
                    reference->value_count != actual->value_count ||
                    reference->expert_count != actual->expert_count) {
                    result.layer = layer;
                    result.position = position;
                    result.checkpoint = (MoeInferCaptureCheckpoint)checkpoint;
                    return result;
                }

                if (reference->kv_length != actual->kv_length ||
                    reference->compressor_length != actual->compressor_length ||
                    reference->indexer_length != actual->indexer_length ||
                    reference->kv_state_digest != actual->kv_state_digest ||
                    reference->compressor_state_digest != actual->compressor_state_digest ||
                    reference->indexer_state_digest != actual->indexer_state_digest) {
                    result.state_mismatch = true;
                    result.layer = layer;
                    result.position = position;
                    result.checkpoint = (MoeInferCaptureCheckpoint)checkpoint;
                    return result;
                }

                for (size_t index = 0; index < reference->expert_count; ++index) {
                    if (reference->expert_ids[index] != actual->expert_ids[index]) {
                        result.expert_id_mismatch = true;
                        result.layer = layer;
                        result.position = position;
                        result.checkpoint = (MoeInferCaptureCheckpoint)checkpoint;
                        result.index = index;
                        return result;
                    }
                }

                const BatchOracleStatus status = batch_oracle_compare_f32(
                    reference->values,
                    actual->values,
                    reference->value_count,
                    oracle_dtype(reference->dtype),
                    oracle_checkpoint(reference->checkpoint),
                    layer,
                    position,
                    &result.numeric);
                if (getenv("DMLX_BATCH_ORACLE_VERBOSE") &&
                    (status != BATCH_ORACLE_OK ||
                     result.numeric.max_absolute_error > 0.0)) {
                    size_t max_index = 0u;
                    double max_absolute = 0.0;
                    for (size_t value_index = 0u;
                         value_index < reference->value_count;
                         ++value_index) {
                        const double absolute = fabs(
                            (double)reference->values[value_index] -
                            (double)actual->values[value_index]);
                        if (absolute > max_absolute) {
                            max_absolute = absolute;
                            max_index = value_index;
                        }
                    }
                    fprintf(stderr,
                        "batch oracle checkpoint: layer=%d position=%d checkpoint=%d status=%s index=%zu reference=%.9g actual=%.9g max_abs=%.9g max_rel=%.9g\n",
                        layer,
                        position,
                        checkpoint,
                        batch_oracle_status_name(status),
                        max_index,
                        reference->value_count > 0u
                            ? reference->values[max_index] : 0.0f,
                        actual->value_count > 0u
                            ? actual->values[max_index] : 0.0f,
                        result.numeric.max_absolute_error,
                        result.numeric.max_relative_error);
                }
                if (status != BATCH_ORACLE_OK) {
                    result.layer = layer;
                    result.position = position;
                    result.checkpoint = (MoeInferCaptureCheckpoint)checkpoint;
                    result.index = result.numeric.first_index;
                    return result;
                }
            }
        }
    }

    result.equal = true;
    return result;
}

static const char *capture_checkpoint_name(MoeInferCaptureCheckpoint checkpoint) {
    switch (checkpoint) {
        case MOE_INFER_CAPTURE_LAYER_INPUT: return "layer_input";
        case MOE_INFER_CAPTURE_ATTENTION_OUTPUT: return "attention_output";
        case MOE_INFER_CAPTURE_ROUTER: return "router";
        case MOE_INFER_CAPTURE_MOE_OUTPUT: return "moe_output";
        case MOE_INFER_CAPTURE_LAYER_OUTPUT: return "layer_output";
    }
    return "unknown";
}

int dmlx_batch_capture_compare(
    void *scalar_context,
    void *batch_context,
    int start_pos,
    int n_tokens) {
    CaptureSet *scalar = (CaptureSet *)scalar_context;
    CaptureSet *batch = (CaptureSet *)batch_context;
    const CaptureComparison result = compare_captures(
        scalar, batch, start_pos, n_tokens);
    if (result.equal) return 0;

    fprintf(stderr,
        "batch oracle divergence: layer=%d position=%d checkpoint=%s index=%zu",
        result.layer,
        result.position,
        capture_checkpoint_name(result.checkpoint),
        result.index);
    if (result.expert_id_mismatch) {
        fprintf(stderr, " reason=expert-id-mismatch\n");
    } else if (result.state_mismatch) {
        fprintf(stderr, " reason=kv/compressor/indexer-state-mismatch\n");
    } else if (result.numeric.status != BATCH_ORACLE_OK) {
        fprintf(stderr,
            " reason=%s max_abs=%.9g max_rel=%.9g\n",
            batch_oracle_status_name(result.numeric.status),
            result.numeric.max_absolute_error,
            result.numeric.max_relative_error);
    } else {
        fprintf(stderr,
            " reason=capture-shape/count expected=%zu scalar=%zu batch=%zu overflow=%d/%d\n",
            (size_t)N_LAYERS * (size_t)n_tokens * 5,
            scalar ? scalar->count : 0,
            batch ? batch->count : 0,
            scalar ? scalar->overflow : 1,
            batch ? batch->overflow : 1);
    }
    return -1;
}

typedef struct {
    int status;
    int has_first_divergence;
    size_t first_index;
    double first_absolute_error;
    double first_relative_error;
    double max_absolute_error;
    double max_relative_error;
    double tolerance_absolute;
    double tolerance_relative;
    int layer;
    int token;
} DmlxBatchLogitsReport;

/*
 * Test-only ABI used by the Zig model runner. Keeping this wrapper here makes
 * batch_oracle.c's F32/final-logits tolerance table the sole authority; Zig
 * and Python callers receive the selected tolerance instead of duplicating it.
 */
int dmlx_batch_logits_compare(
    const float *reference,
    const float *actual,
    size_t count,
    int token,
    DmlxBatchLogitsReport *out_report) {
    if (!out_report) return -1;
    memset(out_report, 0, sizeof(*out_report));
    out_report->first_index = SIZE_MAX;

    BatchOracleReport report;
    const BatchOracleStatus status = batch_oracle_compare_f32(
        reference,
        actual,
        count,
        BATCH_ORACLE_DTYPE_F32,
        BATCH_ORACLE_CHECKPOINT_FINAL_LOGITS,
        N_LAYERS,
        token,
        &report);
    out_report->status = (int)status;
    out_report->has_first_divergence = report.has_first_divergence ? 1 : 0;
    out_report->first_index = report.first_index;
    out_report->first_absolute_error = report.first_absolute_error;
    out_report->first_relative_error = report.first_relative_error;
    out_report->max_absolute_error = report.max_absolute_error;
    out_report->max_relative_error = report.max_relative_error;
    out_report->tolerance_absolute = report.tolerance.absolute;
    out_report->tolerance_relative = report.tolerance.relative;
    out_report->layer = report.layer;
    out_report->token = report.token;

    if (status != BATCH_ORACLE_OK) {
        fprintf(stderr,
            "final logits divergence: checkpoint=%s layer=%d token=%d status=%s index=%zu max_abs=%.9g max_rel=%.9g atol=%.9g rtol=%.9g\n",
            batch_oracle_checkpoint_name(report.checkpoint),
            report.layer,
            report.token,
            batch_oracle_status_name(status),
            report.first_index,
            report.max_absolute_error,
            report.max_relative_error,
            report.tolerance.absolute,
            report.tolerance.relative);
        return -1;
    }
    return 0;
}

static uint64_t task44_align_workspace(uint64_t value) {
    const uint64_t mask = NATIVE_BATCH_WORKSPACE_ALIGNMENT - 1u;
    return (value + mask) & ~mask;
}

static bool task44_shared_buffers_all(
    const MoEInferEngine *engine, int layer, bool expected_present) {
    if (!engine || layer < 0 || layer >= N_LAYERS) return false;
    for (int index = 0; index < 9; ++index) {
        if ((engine->buf_shared_w[layer][index] != NULL) != expected_present) {
            return false;
        }
    }
    return true;
}

static void task44_fill_bucket_fixture(
    NativeExpertBucketWorkspace *view,
    uint32_t hidden_dim,
    float *logical_rows,
    float *shared_rows) {
    for (uint32_t token = 0u; token < view->batch_tokens; ++token) {
        for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
            const uint32_t source = token * N_ACTIVE + rank;
            view->activation_pairs[source] = (NativeActivationPair){
                .token_id = token,
                .expert_id = rank * 31u + token,
                .route_rank = rank,
                .weight = (float)(rank + 1u) * 0.0625f,
            };
            for (uint32_t column = 0u; column < hidden_dim; ++column) {
                logical_rows[(size_t)source * hidden_dim + column] =
                    (float)((int)((source * 7u + column * 5u) % 23u) - 11) *
                    0.015625f;
            }
        }
        for (uint32_t column = 0u; column < hidden_dim; ++column) {
            shared_rows[(size_t)token * hidden_dim + column] =
                (float)((int)((token * 3u + column) % 13u) - 6) *
                0.0078125f;
        }
    }
}

/*
 * Task 4.4 host-API test. The Zig model runner invokes this only after every
 * scalar/batch inference oracle, so deliberate failure injection cannot alter
 * production-path observations. The real initialized engine supplies the
 * pipelines and affine shared-expert weights under test.
 */
int dmlx_task44_host_wrapper_test(MoEInferEngine *engine) {
    int local_failures = 0;
#define TASK44_EXPECT(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "Task 4.4 host wrapper FAIL: %s (line %d)\n", \
                (message), __LINE__); \
        ++local_failures; \
    } \
} while (0)

    if (!engine) return -1;
    const uint32_t batch_sizes[] = {1u, 2u, 3u, 5u};
    enum { COMBINE_HIDDEN = 17 };

    for (size_t case_index = 0u;
         case_index < sizeof(batch_sizes) / sizeof(batch_sizes[0]);
         ++case_index) {
        const uint32_t tokens = batch_sizes[case_index];
        const uint32_t pairs = tokens * N_ACTIVE;
        const size_t bucket_value_count = (size_t)pairs * COMBINE_HIDDEN;
        const size_t shared_value_count = (size_t)tokens * COMBINE_HIDDEN;
        const uint64_t bucket_bytes =
            (uint64_t)bucket_value_count * sizeof(float);
        const uint64_t shared_bytes =
            (uint64_t)shared_value_count * sizeof(float);
        const uint64_t combine_bytes = bucket_bytes + 2u * shared_bytes;
        uint64_t layout_bytes = 0u;
        TASK44_EXPECT(moe_infer_expert_bucket_workspace_bytes(
                          tokens, &layout_bytes) == NATIVE_BATCH_OK,
                      "combine workspace estimator");
        const uint64_t arena_bytes = layout_bytes +
            4u * (combine_bytes + NATIVE_BATCH_WORKSPACE_ALIGNMENT);

        float *logical = (float *)calloc(bucket_value_count, sizeof(float));
        float *bucket_a = (float *)calloc(bucket_value_count, sizeof(float));
        float *bucket_b = (float *)calloc(bucket_value_count, sizeof(float));
        float *shared = (float *)calloc(shared_value_count, sizeof(float));
        float *expected = (float *)calloc(shared_value_count, sizeof(float));
        TASK44_EXPECT(logical && bucket_a && bucket_b && shared && expected,
                      "combine fixture allocations");
        if (!logical || !bucket_a || !bucket_b || !shared || !expected) {
            free(logical);
            free(bucket_a);
            free(bucket_b);
            free(shared);
            free(expected);
            ++local_failures;
            continue;
        }

        NativeBatchWorkspace workspace = {0};
        NativeExpertBucketWorkspace buckets = {0};
        TASK44_EXPECT(moe_infer_batch_workspace_init(
                          &workspace, tokens, arena_bytes, arena_bytes) ==
                          NATIVE_BATCH_OK,
                      "combine arena init");
        TASK44_EXPECT(moe_infer_expert_bucket_workspace_bind(
                          &workspace, tokens, &buckets) == NATIVE_BATCH_OK,
                      "combine bucket bind");
        task44_fill_bucket_fixture(
            &buckets, COMBINE_HIDDEN, logical, shared);
        TASK44_EXPECT(moe_infer_expert_bucket_build(&buckets) == NATIVE_BATCH_OK,
                      "combine bucket build");
        for (uint32_t source = 0u; source < pairs; ++source) {
            memcpy(bucket_a +
                       (size_t)buckets.inverse_mapping[source] * COMBINE_HIDDEN,
                   logical + (size_t)source * COMBINE_HIDDEN,
                   COMBINE_HIDDEN * sizeof(float));
        }
        TASK44_EXPECT(moe_infer_expert_bucket_combine_reference_f32(
                          &buckets, bucket_a, shared, COMBINE_HIDDEN,
                          expected) == NATIVE_BATCH_OK,
                      "combine CPU reference");

        const uint64_t canonical_used = workspace.used_bytes;
        const uint64_t canonical_start = task44_align_workspace(canonical_used);
        float *actual_a = NULL;
        TASK44_EXPECT(moe_infer_bucket_combine_f32(
                          engine, &buckets, bucket_a, shared, COMBINE_HIDDEN,
                          &actual_a) == NATIVE_BATCH_OK,
                      "combine host wrapper canonical dispatch");
        TASK44_EXPECT(actual_a == (float *)((uint8_t *)workspace.storage +
                          canonical_start + bucket_bytes + shared_bytes),
                      "combine output offset must match arena layout");
        TASK44_EXPECT(actual_a && memcmp(actual_a, expected,
                          shared_value_count * sizeof(float)) == 0,
                      "combine GPU output must exactly match CPU reference");

        for (uint32_t source = 0u; source < pairs; ++source) {
            const uint32_t destination = pairs - 1u - source;
            buckets.bucketed_pairs[destination] =
                buckets.activation_pairs[source];
            buckets.inverse_mapping[source] = destination;
            memcpy(bucket_b + (size_t)destination * COMBINE_HIDDEN,
                   logical + (size_t)source * COMBINE_HIDDEN,
                   COMBINE_HIDDEN * sizeof(float));
        }
        float *actual_b = NULL;
        TASK44_EXPECT(moe_infer_bucket_combine_f32(
                          engine, &buckets, bucket_b, shared, COMBINE_HIDDEN,
                          &actual_b) == NATIVE_BATCH_OK,
                      "combine host wrapper permuted dispatch");
        TASK44_EXPECT(actual_b && memcmp(actual_b, expected,
                          shared_value_count * sizeof(float)) == 0,
                      "combine wrapper must be bucket-permutation invariant");

        const uint64_t used_before_state_failure = workspace.used_bytes;
        float *sentinel_output = (float *)(uintptr_t)1u;
        void *saved_combine_pipeline = engine->pipe_bucket_combine_f32;
        engine->pipe_bucket_combine_f32 = NULL;
        TASK44_EXPECT(moe_infer_bucket_combine_f32(
                          engine, &buckets, bucket_b, shared, COMBINE_HIDDEN,
                          &sentinel_output) == NATIVE_BATCH_INVALID_STATE,
                      "missing combine pipeline is engine state failure");
        TASK44_EXPECT(workspace.used_bytes == used_before_state_failure &&
                          sentinel_output == (float *)(uintptr_t)1u,
                      "combine state failure must not consume or publish");
        engine->pipe_bucket_combine_f32 = saved_combine_pipeline;

        NativeExpertBucketWorkspace stale_buckets = buckets;
        TASK44_EXPECT(moe_infer_bucket_combine_f32(
                          engine, &stale_buckets, bucket_b, shared,
                          COMBINE_HIDDEN, &sentinel_output) ==
                          NATIVE_BATCH_INVALID_STATE,
                      "moved combine bucket view is invalid state");
        TASK44_EXPECT(workspace.used_bytes == used_before_state_failure,
                      "stale combine view must not consume workspace");

        const uint64_t saved_allocated = workspace.allocated_bytes;
        workspace.allocated_bytes = workspace.used_bytes;
        TASK44_EXPECT(moe_infer_bucket_combine_f32(
                          engine, &buckets, bucket_b, shared, COMBINE_HIDDEN,
                          &sentinel_output) == NATIVE_BATCH_CAPACITY_EXCEEDED,
                      "combine arena exhaustion is capacity failure");
        TASK44_EXPECT(workspace.used_bytes == used_before_state_failure,
                      "combine capacity failure must not consume workspace");
        workspace.allocated_bytes = saved_allocated;

        const uint32_t overflow_destination = buckets.inverse_mapping[0];
        const NativeActivationPair saved_activation =
            buckets.activation_pairs[0];
        const NativeActivationPair saved_bucket_pair =
            buckets.bucketed_pairs[overflow_destination];
        const float saved_bucket_value =
            bucket_b[(size_t)overflow_destination * COMBINE_HIDDEN];
        buckets.activation_pairs[0].weight = FLT_MAX;
        buckets.bucketed_pairs[overflow_destination].weight = FLT_MAX;
        bucket_b[(size_t)overflow_destination * COMBINE_HIDDEN] = FLT_MAX;
        const uint64_t used_before_overflow = workspace.used_bytes;
        sentinel_output = (float *)(uintptr_t)1u;
        TASK44_EXPECT(moe_infer_bucket_combine_f32(
                          engine, &buckets, bucket_b, shared, COMBINE_HIDDEN,
                          &sentinel_output) == NATIVE_BATCH_INVALID_STATE,
                      "finite combine operands producing Inf must fail");
        TASK44_EXPECT(workspace.used_bytes == used_before_overflow &&
                          sentinel_output == (float *)(uintptr_t)1u,
                      "combine post-dispatch failure must roll back cursor");
        buckets.activation_pairs[0] = saved_activation;
        buckets.bucketed_pairs[overflow_destination] = saved_bucket_pair;
        bucket_b[(size_t)overflow_destination * COMBINE_HIDDEN] =
            saved_bucket_value;
        TASK44_EXPECT(moe_infer_bucket_combine_f32(
                          engine, &buckets, bucket_b, shared, COMBINE_HIDDEN,
                          &sentinel_output) == NATIVE_BATCH_OK,
                      "combine retry after rollback must succeed");

        moe_infer_batch_workspace_deinit(&workspace);
        free(logical);
        free(bucket_a);
        free(bucket_b);
        free(shared);
        free(expected);
        printf("Task 4.4 combine host wrapper: PASS B=%u\n", tokens);
    }

    const size_t max_shared_values = 5u * DIM;
    float *shared_inputs = (float *)malloc(max_shared_values * sizeof(float));
    float *batch_copy = (float *)malloc(max_shared_values * sizeof(float));
    float *scalar_copy = (float *)malloc(max_shared_values * sizeof(float));
    TASK44_EXPECT(shared_inputs && batch_copy && scalar_copy,
                  "shared wrapper fixture allocations");
    if (shared_inputs && batch_copy && scalar_copy) {
        for (size_t index = 0u; index < max_shared_values; ++index) {
            shared_inputs[index] =
                (float)((int)(index % 29u) - 14) * 0.0009765625f;
        }

        TASK44_EXPECT(task44_shared_buffers_all(engine, 0, true),
                      "real engine must expose a complete shared-buffer set");
        void *persistent_buffers[9] = {0};
        memcpy(persistent_buffers, engine->buf_shared_w[0],
               sizeof(persistent_buffers));
        for (size_t case_index = 0u;
             case_index < sizeof(batch_sizes) / sizeof(batch_sizes[0]);
             ++case_index) {
            const uint32_t tokens = batch_sizes[case_index];
            const uint64_t input_bytes =
                (uint64_t)tokens * DIM * sizeof(float);
            const uint64_t mid_bytes =
                (uint64_t)tokens * INTERMEDIATE * sizeof(float);
            const uint64_t workspace_bytes = 2u * input_bytes + 2u * mid_bytes;
            NativeBatchWorkspace workspace = {0};
            TASK44_EXPECT(moe_infer_batch_workspace_init(
                              &workspace, tokens, workspace_bytes,
                              workspace_bytes) == NATIVE_BATCH_OK,
                          "shared batch arena init");

            float *sentinel_output = (float *)(uintptr_t)1u;
            const uint64_t initial_used = workspace.used_bytes;
            if (case_index == 0u) {
                void *saved_activation_pipeline = engine->pipe_limited_swiglu;
                engine->pipe_limited_swiglu = NULL;
                TASK44_EXPECT(moe_infer_shared_expert_batch_f32(
                                  engine, 0, shared_inputs, tokens, &workspace,
                                  &sentinel_output) == NATIVE_BATCH_INVALID_STATE,
                              "missing shared pipeline is engine state failure");
                TASK44_EXPECT(workspace.used_bytes == initial_used &&
                                  sentinel_output == (float *)(uintptr_t)1u,
                              "shared state failure must not consume or publish");
                engine->pipe_limited_swiglu = saved_activation_pipeline;

                NativeBatchWorkspace moved_workspace = workspace;
                TASK44_EXPECT(moe_infer_shared_expert_batch_f32(
                                  engine, 0, shared_inputs, tokens,
                                  &moved_workspace, &sentinel_output) ==
                                  NATIVE_BATCH_INVALID_STATE,
                              "moved shared workspace is invalid state");
                TASK44_EXPECT(workspace.used_bytes == initial_used,
                              "moved shared workspace must not consume owner");

                float invalid_input[DIM];
                memcpy(invalid_input, shared_inputs, sizeof(invalid_input));
                invalid_input[DIM / 2] = NAN;
                TASK44_EXPECT(moe_infer_shared_expert_batch_f32(
                                  engine, 0, invalid_input, 1u, &workspace,
                                  &sentinel_output) ==
                                  NATIVE_BATCH_INVALID_ARGUMENT,
                              "nonfinite shared input is invalid argument");
                TASK44_EXPECT(workspace.used_bytes == initial_used,
                              "invalid shared input must not consume workspace");
            }
            if (tokens == 2u) {
                NativeBatchWorkspace short_workspace = {0};
                TASK44_EXPECT(moe_infer_batch_workspace_init(
                                  &short_workspace, 1u, workspace_bytes,
                                  workspace_bytes) == NATIVE_BATCH_OK,
                              "shared token-capacity arena init");
                TASK44_EXPECT(moe_infer_shared_expert_batch_f32(
                                  engine, 0, shared_inputs, tokens,
                                  &short_workspace, &sentinel_output) ==
                                  NATIVE_BATCH_CAPACITY_EXCEEDED,
                              "shared token overflow is capacity failure");
                TASK44_EXPECT(short_workspace.used_bytes == 0u,
                              "shared token-capacity failure is atomic");
                moe_infer_batch_workspace_deinit(&short_workspace);
            }

            float *batch_output = NULL;
            TASK44_EXPECT(moe_infer_shared_expert_batch_f32(
                              engine, 0, shared_inputs, tokens, &workspace,
                              &batch_output) == NATIVE_BATCH_OK,
                          "shared batch wrapper dispatch");
            TASK44_EXPECT(batch_output == (float *)((uint8_t *)workspace.storage +
                              input_bytes + 2u * mid_bytes),
                          "shared output offset must match arena layout");
            if (batch_output) {
                memcpy(batch_copy, batch_output,
                    (size_t)tokens * DIM * sizeof(float));
            }
            TASK44_EXPECT(task44_shared_buffers_all(engine, 0, true),
                          "shared buffers must remain all-or-none");
            TASK44_EXPECT(memcmp(persistent_buffers,
                                  engine->buf_shared_w[0],
                                  sizeof(persistent_buffers)) == 0,
                          "shared buffers must be reused across batches");

            for (uint32_t token = 0u; token < tokens; ++token) {
                const uint64_t scalar_input_bytes =
                    (uint64_t)DIM * sizeof(float);
                const uint64_t scalar_mid_bytes =
                    (uint64_t)INTERMEDIATE * sizeof(float);
                const uint64_t scalar_workspace_bytes =
                    2u * scalar_input_bytes + 2u * scalar_mid_bytes;
                NativeBatchWorkspace scalar_workspace = {0};
                TASK44_EXPECT(moe_infer_batch_workspace_init(
                                  &scalar_workspace, 1u,
                                  scalar_workspace_bytes,
                                  scalar_workspace_bytes) == NATIVE_BATCH_OK,
                              "shared scalar arena init");
                float *scalar_output = NULL;
                TASK44_EXPECT(moe_infer_shared_expert_batch_f32(
                                  engine, 0,
                                  shared_inputs + (size_t)token * DIM,
                                  1u, &scalar_workspace, &scalar_output) ==
                                  NATIVE_BATCH_OK,
                              "shared scalar wrapper dispatch");
                if (scalar_output) {
                    memcpy(scalar_copy + (size_t)token * DIM, scalar_output,
                           DIM * sizeof(float));
                }
                moe_infer_batch_workspace_deinit(&scalar_workspace);
            }
            float max_abs = 0.0f;
            for (size_t index = 0u; index < (size_t)tokens * DIM; ++index) {
                const float difference =
                    fabsf(batch_copy[index] - scalar_copy[index]);
                if (difference > max_abs) max_abs = difference;
            }
            TASK44_EXPECT(max_abs <= 1e-6f,
                          "shared batched rows must match repeated B=1");
            printf("Task 4.4 shared host wrapper: PASS B=%u max_abs=%.3e\n",
                   tokens, max_abs);
            moe_infer_batch_workspace_deinit(&workspace);
        }

        const uint64_t scalar_input_bytes = (uint64_t)DIM * sizeof(float);
        const uint64_t scalar_mid_bytes =
            (uint64_t)INTERMEDIATE * sizeof(float);
        const uint64_t scalar_workspace_bytes =
            2u * scalar_input_bytes + 2u * scalar_mid_bytes;
        NativeBatchWorkspace overflow_workspace = {0};
        TASK44_EXPECT(moe_infer_batch_workspace_init(
                          &overflow_workspace, 1u, scalar_workspace_bytes,
                          scalar_workspace_bytes) == NATIVE_BATCH_OK,
                      "shared overflow arena init");
        float *huge_input = (float *)malloc(DIM * sizeof(float));
        TASK44_EXPECT(huge_input != NULL, "shared overflow input allocation");
        if (huge_input) {
            for (size_t index = 0u; index < DIM; ++index) {
                huge_input[index] = FLT_MAX;
            }
            float *sentinel_output = (float *)(uintptr_t)1u;
            const uint64_t used_before = overflow_workspace.used_bytes;
            TASK44_EXPECT(moe_infer_shared_expert_batch_f32(
                              engine, 0, huge_input, 1u, &overflow_workspace,
                              &sentinel_output) == NATIVE_BATCH_INVALID_STATE,
                          "finite shared input producing nonfinite output fails");
            TASK44_EXPECT(overflow_workspace.used_bytes == used_before &&
                              sentinel_output == (float *)(uintptr_t)1u,
                          "shared post-dispatch failure must roll back cursor");
            TASK44_EXPECT(moe_infer_shared_expert_batch_f32(
                              engine, 0, shared_inputs, 1u,
                              &overflow_workspace, &sentinel_output) ==
                              NATIVE_BATCH_OK,
                          "shared retry after rollback must succeed");
            free(huge_input);
        }
        moe_infer_batch_workspace_deinit(&overflow_workspace);

        const int cleanup_layer = N_LAYERS - 1;
        MoEInferEngine *partial_engine =
            (MoEInferEngine *)malloc(sizeof(*partial_engine));
        TASK44_EXPECT(partial_engine != NULL,
                      "partial shared engine allocation");
        if (partial_engine) {
            memcpy(partial_engine, engine, sizeof(*partial_engine));
            memset(partial_engine->buf_shared_w[cleanup_layer], 0,
                   sizeof(partial_engine->buf_shared_w[cleanup_layer]));
            NativeBatchWorkspace cleanup_workspace = {0};
            TASK44_EXPECT(moe_infer_batch_workspace_init(
                              &cleanup_workspace, 1u, scalar_workspace_bytes,
                              scalar_workspace_bytes) == NATIVE_BATCH_OK,
                          "shared partial-allocation cleanup arena init");
            float *cleanup_output = (float *)(uintptr_t)1u;

            partial_engine->shared[cleanup_layer].gate.in_dim = DIM - 1;
            TASK44_EXPECT(moe_infer_shared_expert_batch_f32(
                              partial_engine, cleanup_layer, shared_inputs, 1u,
                              &cleanup_workspace, &cleanup_output) ==
                              NATIVE_BATCH_INVALID_STATE,
                          "malformed shared geometry is engine state failure");
            TASK44_EXPECT(cleanup_workspace.used_bytes == 0u &&
                              cleanup_output == (float *)(uintptr_t)1u,
                          "shared geometry failure must not consume or publish");

            partial_engine->shared[cleanup_layer] =
                engine->shared[cleanup_layer];
            partial_engine->shared[cleanup_layer].down.packed = NULL;
            TASK44_EXPECT(moe_infer_shared_expert_batch_f32(
                              partial_engine, cleanup_layer, shared_inputs, 1u,
                              &cleanup_workspace, &cleanup_output) ==
                              NATIVE_BATCH_INVALID_STATE,
                          "missing shared weight resource is engine state failure");
            TASK44_EXPECT(task44_shared_buffers_all(
                              partial_engine, cleanup_layer, false) &&
                              cleanup_workspace.used_bytes == 0u &&
                              cleanup_output == (float *)(uintptr_t)1u,
                          "partial shared buffer creation must clean up atomically");
            TASK44_EXPECT(task44_shared_buffers_all(engine, cleanup_layer, true),
                          "isolated cleanup test must preserve real engine buffers");
            moe_infer_batch_workspace_deinit(&cleanup_workspace);
            free(partial_engine);
        }
    }
    free(shared_inputs);
    free(batch_copy);
    free(scalar_copy);

    if (local_failures == 0) {
        printf("Task 4.4 host wrappers: PASS (B=1,2,3,5,rollback,retry,lifecycle)\n");
    }
#undef TASK44_EXPECT
    return local_failures == 0 ? 0 : -1;
}

#ifdef DMLX_BATCH_CAPTURE_SELF_TEST
static int pointer_is_workspace_aligned(const void *pointer) {
    return pointer &&
        ((uintptr_t)pointer & (NATIVE_BATCH_WORKSPACE_ALIGNMENT - 1u)) == 0u;
}

static void test_expert_bucket_workspace_layout(void) {
    enum { TOKENS = 5, PAIRS = TOKENS * N_ACTIVE };
    uint64_t exact = 0u;
    EXPECT(moe_infer_expert_bucket_workspace_bytes(TOKENS, &exact) ==
               NATIVE_BATCH_OK,
           "expert bucket workspace estimator must succeed");
    EXPECT(exact == 4608u,
           "B=5 expert bucket workspace layout must remain exact");

    uint64_t overflow_sentinel = UINT64_C(0xfeedface);
    EXPECT(moe_infer_expert_bucket_workspace_bytes(
               UINT32_MAX, &overflow_sentinel) ==
               NATIVE_BATCH_CAPACITY_EXCEEDED,
           "pair-capacity overflow must be rejected");
    EXPECT(overflow_sentinel == UINT64_C(0xfeedface),
           "failed estimator must not publish a partial size");

    NativeBatchWorkspace workspace = {0};
    EXPECT(moe_infer_batch_workspace_init(
               &workspace, TOKENS, exact, exact) == NATIVE_BATCH_OK,
           "exact expert bucket arena init");
    NativeExpertBucketWorkspace view = {0};
    EXPECT(moe_infer_expert_bucket_workspace_bind(
               &workspace, TOKENS, &view) == NATIVE_BATCH_OK,
           "exact expert bucket bind");
    EXPECT(view.self_cookie == (uintptr_t)&view &&
               view.owner_workspace == &workspace,
           "expert bucket view ownership");
    EXPECT(view.batch_tokens == TOKENS && view.pair_capacity == PAIRS &&
               view.unique_capacity == N_EXPERTS,
           "expert bucket view capacities");
    EXPECT(view.arena_offset_bytes == 0u &&
               view.arena_size_bytes == exact &&
               workspace.used_bytes == exact &&
               workspace.high_water_bytes == exact,
           "expert bucket exact bytes and high-water");
    EXPECT(pointer_is_workspace_aligned(view.activation_pairs) &&
               pointer_is_workspace_aligned(view.bucketed_pairs) &&
               pointer_is_workspace_aligned(view.counts) &&
               pointer_is_workspace_aligned(view.offsets) &&
               pointer_is_workspace_aligned(view.unique_expert_ids) &&
               pointer_is_workspace_aligned(view.inverse_mapping),
           "every expert bucket region must be 256-byte aligned");

    view.activation_pairs[0] = (NativeActivationPair){
        .token_id = 4u,
        .expert_id = 255u,
        .route_rank = 5u,
        .weight = 1.5f,
    };
    memset(view.counts, 0xff, (size_t)N_EXPERTS * sizeof(uint32_t));
    memset(view.offsets, 0xff,
        (size_t)(N_EXPERTS + 1u) * sizeof(uint32_t));
    memset(view.unique_expert_ids, 0xff,
        (size_t)N_EXPERTS * sizeof(uint32_t));
    memset(view.inverse_mapping, 0xff,
        (size_t)PAIRS * sizeof(uint32_t));
    view.unique_count = 17u;
    const uint64_t used_before_reset = workspace.used_bytes;
    EXPECT(moe_infer_expert_bucket_workspace_reset(&view) == NATIVE_BATCH_OK,
           "expert bucket per-layer reset");
    EXPECT(moe_infer_expert_bucket_workspace_reset(&view) == NATIVE_BATCH_OK,
           "expert bucket view must be reusable without another reserve");
    EXPECT(workspace.used_bytes == used_before_reset &&
               workspace.high_water_bytes == exact && view.unique_count == 0u,
           "per-layer reset must not allocate and must clear unique count");
    EXPECT(view.activation_pairs[0].token_id == 4u &&
               view.activation_pairs[0].expert_id == 255u &&
               view.activation_pairs[0].route_rank == 5u &&
               view.activation_pairs[0].weight == 1.5f,
           "index reset must not rewrite activation-pair payloads");
    for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
        EXPECT(view.counts[expert] == 0u,
               "counts reset must clear every expert");
        EXPECT(view.offsets[expert] == 0u,
               "offset reset must clear every expert");
        EXPECT(view.unique_expert_ids[expert] == 0u,
               "unique-id reset must clear every slot");
    }
    EXPECT(view.offsets[N_EXPERTS] == 0u,
           "offset sentinel must be reset");
    for (uint32_t pair = 0u; pair < PAIRS; ++pair) {
        EXPECT(view.inverse_mapping[pair] == 0u,
               "inverse mapping reset must clear every pair");
    }

    const uint32_t stale_epoch = view.arena_epoch;
    moe_infer_batch_workspace_reset(&workspace);
    EXPECT(workspace.used_bytes == 0u &&
               workspace.arena_epoch == stale_epoch + 1u,
           "parent reset must advance the arena lifetime epoch");
    NativeExpertBucketWorkspace rebound_view = {0};
    EXPECT(moe_infer_expert_bucket_workspace_bind(
               &workspace, TOKENS, &rebound_view) == NATIVE_BATCH_OK,
           "expert bucket arena must bind again after parent reset");
    EXPECT(rebound_view.activation_pairs == view.activation_pairs &&
               rebound_view.arena_epoch == workspace.arena_epoch,
           "rebind must reuse storage under the new epoch");
    rebound_view.counts[0] = UINT32_C(0x5a5aa5a5);
    EXPECT(moe_infer_expert_bucket_workspace_reset(&view) ==
               NATIVE_BATCH_INVALID_STATE,
           "stale view must stay invalid after same-address arena reuse");
    EXPECT(rebound_view.counts[0] == UINT32_C(0x5a5aa5a5),
           "stale reset must not clear a replacement allocation");

    NativeExpertBucketWorkspace moved_view = rebound_view;
    EXPECT(moe_infer_expert_bucket_workspace_reset(&moved_view) ==
               NATIVE_BATCH_INVALID_STATE,
           "moved expert bucket view must fail self-cookie validation");
    EXPECT(rebound_view.counts[0] == UINT32_C(0x5a5aa5a5),
           "moved-view rejection must not mutate live storage");

    const uint32_t live_epoch = rebound_view.arena_epoch;
    rebound_view.arena_epoch = live_epoch + 1u;
    EXPECT(moe_infer_expert_bucket_workspace_reset(&rebound_view) ==
               NATIVE_BATCH_INVALID_STATE,
           "poisoned expert bucket epoch must be rejected");
    EXPECT(rebound_view.counts[0] == UINT32_C(0x5a5aa5a5),
           "poisoned-view rejection must not mutate live storage");
    rebound_view.arena_epoch = live_epoch;
    EXPECT(moe_infer_expert_bucket_workspace_reset(&rebound_view) ==
               NATIVE_BATCH_OK,
           "restored live epoch must remain reusable");

    workspace.arena_epoch = UINT32_MAX;
    moe_infer_batch_workspace_reset(&workspace);
    EXPECT(workspace.arena_epoch == 0u && workspace.used_bytes == 0u &&
               moe_infer_batch_workspace_reserve(&workspace, 1u, 1u) == NULL,
           "arena epoch exhaustion must poison rather than wrap");
    moe_infer_batch_workspace_deinit(&workspace);

    NativeBatchWorkspace short_workspace = {0};
    NativeExpertBucketWorkspace short_view = {0};
    EXPECT(moe_infer_batch_workspace_init(
               &short_workspace, TOKENS, exact - 1u, exact - 1u) ==
               NATIVE_BATCH_OK,
           "one-byte-short expert bucket arena init");
    EXPECT(moe_infer_expert_bucket_workspace_bind(
               &short_workspace, TOKENS, &short_view) ==
               NATIVE_BATCH_CAPACITY_EXCEEDED,
           "one-byte-short expert bucket bind must fail");
    EXPECT(short_workspace.used_bytes == 0u &&
               short_workspace.high_water_bytes == 0u &&
               short_view.self_cookie == 0u && !short_view.activation_pairs,
           "failed expert bucket bind must be atomic");
    moe_infer_batch_workspace_deinit(&short_workspace);

    NativeBatchWorkspace padded_workspace = {0};
    NativeExpertBucketWorkspace padded_view = {0};
    EXPECT(moe_infer_batch_workspace_init(
               &padded_workspace, TOKENS, exact + 256u, exact + 256u) ==
               NATIVE_BATCH_OK,
           "padded expert bucket arena init");
    EXPECT(moe_infer_batch_workspace_reserve(
               &padded_workspace, 1u, 1u) != NULL,
           "prefix reservation before expert bucket layout");
    EXPECT(moe_infer_expert_bucket_workspace_bind(
               &padded_workspace, TOKENS, &padded_view) == NATIVE_BATCH_OK,
           "expert bucket bind after unaligned prefix");
    EXPECT(padded_view.arena_offset_bytes == 256u &&
               padded_workspace.used_bytes == exact + 256u &&
               padded_workspace.high_water_bytes == exact + 256u,
           "high-water must include actual alignment padding");
    moe_infer_batch_workspace_deinit(&padded_workspace);

    NativeBatchWorkspace capacity_workspace = {0};
    NativeExpertBucketWorkspace capacity_view = {0};
    EXPECT(moe_infer_batch_workspace_init(
               &capacity_workspace, TOKENS - 1u, exact, exact) ==
               NATIVE_BATCH_OK,
           "token-capacity expert bucket arena init");
    EXPECT(moe_infer_expert_bucket_workspace_bind(
               &capacity_workspace, TOKENS, &capacity_view) ==
               NATIVE_BATCH_CAPACITY_EXCEEDED,
           "token-capacity overflow must fail before reserve");
    EXPECT(capacity_workspace.used_bytes == 0u &&
               capacity_workspace.high_water_bytes == 0u,
           "token-capacity failure must not mutate workspace");
    moe_infer_batch_workspace_deinit(&capacity_workspace);
}

static void test_stable_expert_bucket_build(void) {
    enum { MAX_TOKENS = 5, MAX_PAIRS = MAX_TOKENS * N_ACTIVE };
    const uint32_t batch_sizes[] = {1u, 2u, 3u, 5u};

    for (size_t case_index = 0u;
         case_index < sizeof(batch_sizes) / sizeof(batch_sizes[0]);
         ++case_index) {
        const uint32_t tokens = batch_sizes[case_index];
        uint64_t bytes = 0u;
        EXPECT(moe_infer_expert_bucket_workspace_bytes(tokens, &bytes) ==
                   NATIVE_BATCH_OK,
               "bucket build workspace sizing");
        NativeBatchWorkspace arena = {0};
        NativeExpertBucketWorkspace view = {0};
        EXPECT(moe_infer_batch_workspace_init(
                   &arena, tokens, bytes, bytes) == NATIVE_BATCH_OK &&
                   moe_infer_expert_bucket_workspace_bind(
                   &arena, tokens, &view) == NATIVE_BATCH_OK,
               "bucket build workspace bind");

        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
                const uint32_t source = token * N_ACTIVE + rank;
                view.activation_pairs[source] = (NativeActivationPair){
                    .token_id = token,
                    .expert_id = rank * 11u + (token % 2u) * 3u,
                    .route_rank = rank,
                    .weight = (float)(source + 1u) / 100.0f,
                };
            }
        }

        EXPECT(moe_infer_expert_bucket_build(&view) == NATIVE_BATCH_OK,
               "stable counting-sort bucket build");
        uint32_t count_sum = 0u;
        uint32_t expected_unique = 0u;
        uint32_t previous_unique = 0u;
        EXPECT(view.offsets[0] == 0u,
               "bucket offsets must start at zero");
        for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
            count_sum += view.counts[expert];
            EXPECT(view.offsets[expert + 1u] == count_sum,
                   "bucket offsets must be an exclusive count scan");
            if (view.counts[expert] != 0u) {
                EXPECT(expected_unique == 0u || expert > previous_unique,
                       "unique expert IDs must be strictly ascending");
                EXPECT(view.unique_expert_ids[expected_unique] == expert,
                       "unique expert IDs must match non-empty buckets");
                previous_unique = expert;
                ++expected_unique;
            }

            uint32_t previous_source = 0u;
            bool have_previous = false;
            for (uint32_t destination = view.offsets[expert];
                 destination < view.offsets[expert + 1u]; ++destination) {
                const NativeActivationPair *pair =
                    &view.bucketed_pairs[destination];
                const uint32_t source =
                    pair->token_id * N_ACTIVE + pair->route_rank;
                EXPECT(pair->expert_id == expert,
                       "every bucket row must belong to its expert");
                EXPECT(!have_previous || source > previous_source,
                       "stable scatter must preserve token-major source order");
                previous_source = source;
                have_previous = true;
            }
        }
        EXPECT(count_sum == tokens * N_ACTIVE &&
                   view.offsets[N_EXPERTS] == view.pair_capacity,
               "bucket counts and sentinel must conserve every pair");
        EXPECT(view.unique_count == expected_unique,
               "unique expert count must match non-empty buckets");

        for (uint32_t source = 0u; source < view.pair_capacity; ++source) {
            const uint32_t destination = view.inverse_mapping[source];
            EXPECT(destination < view.pair_capacity &&
                       memcmp(&view.activation_pairs[source],
                              &view.bucketed_pairs[destination],
                              sizeof(NativeActivationPair)) == 0,
                   "inverse mapping must reconstruct token-major pairs exactly");
        }

        NativeActivationPair bucket_snapshot[MAX_PAIRS];
        uint32_t counts_snapshot[N_EXPERTS];
        uint32_t offsets_snapshot[N_EXPERTS + 1u];
        uint32_t unique_snapshot[N_EXPERTS];
        uint32_t inverse_snapshot[MAX_PAIRS];
        memcpy(bucket_snapshot, view.bucketed_pairs,
            (size_t)view.pair_capacity * sizeof(NativeActivationPair));
        memcpy(counts_snapshot, view.counts, sizeof(counts_snapshot));
        memcpy(offsets_snapshot, view.offsets, sizeof(offsets_snapshot));
        memcpy(unique_snapshot, view.unique_expert_ids, sizeof(unique_snapshot));
        memcpy(inverse_snapshot, view.inverse_mapping,
            (size_t)view.pair_capacity * sizeof(uint32_t));
        const uint32_t unique_snapshot_count = view.unique_count;

        EXPECT(moe_infer_expert_bucket_build(&view) == NATIVE_BATCH_OK,
               "repeated bucket build");
        EXPECT(memcmp(bucket_snapshot, view.bucketed_pairs,
                      (size_t)view.pair_capacity * sizeof(NativeActivationPair)) == 0 &&
                   memcmp(counts_snapshot, view.counts, sizeof(counts_snapshot)) == 0 &&
                   memcmp(offsets_snapshot, view.offsets, sizeof(offsets_snapshot)) == 0 &&
                   memcmp(unique_snapshot, view.unique_expert_ids,
                          sizeof(unique_snapshot)) == 0 &&
                   memcmp(inverse_snapshot, view.inverse_mapping,
                          (size_t)view.pair_capacity * sizeof(uint32_t)) == 0 &&
                   view.unique_count == unique_snapshot_count,
               "bucket build must be deterministic across runs");

        const uint32_t malformed = view.pair_capacity - 1u;
        const uint32_t original_rank = view.activation_pairs[malformed].route_rank;
        view.activation_pairs[malformed].route_rank = 0u;
        EXPECT(moe_infer_expert_bucket_build(&view) ==
                   NATIVE_BATCH_INVALID_ARGUMENT,
               "malformed token-major route rank must fail");
        EXPECT(memcmp(bucket_snapshot, view.bucketed_pairs,
                      (size_t)view.pair_capacity * sizeof(NativeActivationPair)) == 0 &&
                   memcmp(counts_snapshot, view.counts, sizeof(counts_snapshot)) == 0 &&
                   memcmp(offsets_snapshot, view.offsets, sizeof(offsets_snapshot)) == 0 &&
                   memcmp(unique_snapshot, view.unique_expert_ids,
                          sizeof(unique_snapshot)) == 0 &&
                   memcmp(inverse_snapshot, view.inverse_mapping,
                          (size_t)view.pair_capacity * sizeof(uint32_t)) == 0 &&
                   view.unique_count == unique_snapshot_count,
               "malformed pairs must fail before bucket publication");
        view.activation_pairs[malformed].route_rank = original_rank;
        moe_infer_batch_workspace_deinit(&arena);
    }
}

static void test_deterministic_bucket_combine_reference(void) {
    enum { HIDDEN = 17, MAX_TOKENS = 5, MAX_PAIRS = MAX_TOKENS * N_ACTIVE };
    const uint32_t batch_sizes[] = {1u, 2u, 3u, 5u};
    for (size_t case_index = 0u;
         case_index < sizeof(batch_sizes) / sizeof(batch_sizes[0]);
         ++case_index) {
        const uint32_t tokens = batch_sizes[case_index];
        const uint32_t pairs = tokens * N_ACTIVE;
        uint64_t bytes = 0u;
        EXPECT(moe_infer_expert_bucket_workspace_bytes(tokens, &bytes) ==
                   NATIVE_BATCH_OK,
               "combine reference workspace sizing");
        NativeBatchWorkspace arena = {0};
        NativeExpertBucketWorkspace view = {0};
        EXPECT(moe_infer_batch_workspace_init(
                   &arena, tokens, bytes, bytes) == NATIVE_BATCH_OK &&
                   moe_infer_expert_bucket_workspace_bind(
                   &arena, tokens, &view) == NATIVE_BATCH_OK,
               "combine reference workspace bind");
        float logical[MAX_PAIRS * HIDDEN];
        float bucket_a[MAX_PAIRS * HIDDEN];
        float bucket_b[MAX_PAIRS * HIDDEN];
        float shared[MAX_TOKENS * HIDDEN];
        float expected[MAX_TOKENS * HIDDEN];
        float actual_a[MAX_TOKENS * HIDDEN];
        float actual_b[MAX_TOKENS * HIDDEN];
        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
                const uint32_t source = token * N_ACTIVE + rank;
                view.activation_pairs[source] = (NativeActivationPair){
                    .token_id = token,
                    .expert_id = rank * 31u + token,
                    .route_rank = rank,
                    .weight = (float)(rank + 1u) * 0.0625f +
                        (float)token * 0.001f,
                };
                for (uint32_t column = 0u; column < HIDDEN; ++column) {
                    logical[source * HIDDEN + column] =
                        (float)((int)((source * 7u + column * 5u) % 23u) - 11) *
                        0.015625f;
                }
            }
            for (uint32_t column = 0u; column < HIDDEN; ++column) {
                shared[token * HIDDEN + column] =
                    (float)((int)((token * 3u + column) % 13u) - 6) *
                    0.0078125f;
                float routed = 0.0f;
                for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
                    const uint32_t source = token * N_ACTIVE + rank;
                    routed += logical[source * HIDDEN + column] *
                        view.activation_pairs[source].weight;
                }
                expected[token * HIDDEN + column] = routed +
                    shared[token * HIDDEN + column];
            }
        }
        EXPECT(moe_infer_expert_bucket_build(&view) == NATIVE_BATCH_OK,
               "combine reference bucket build");
        for (uint32_t source = 0u; source < pairs; ++source) {
            memcpy(bucket_a + (size_t)view.inverse_mapping[source] * HIDDEN,
                   logical + (size_t)source * HIDDEN,
                   HIDDEN * sizeof(float));
        }
        EXPECT(moe_infer_expert_bucket_combine_reference_f32(
                   &view, bucket_a, shared, HIDDEN, actual_a) ==
                   NATIVE_BATCH_OK,
               "combine reference canonical mapping");

        // Apply a complete reverse permutation to bucket payloads and metadata;
        // token-major route order remains authoritative through inverse_mapping.
        for (uint32_t source = 0u; source < pairs; ++source) {
            const uint32_t destination = pairs - 1u - source;
            view.bucketed_pairs[destination] = view.activation_pairs[source];
            view.inverse_mapping[source] = destination;
            memcpy(bucket_b + (size_t)destination * HIDDEN,
                   logical + (size_t)source * HIDDEN,
                   HIDDEN * sizeof(float));
        }
        EXPECT(moe_infer_expert_bucket_combine_reference_f32(
                   &view, bucket_b, shared, HIDDEN, actual_b) ==
                   NATIVE_BATCH_OK,
               "combine reference permuted mapping");
        EXPECT(memcmp(actual_a, actual_b,
                      (size_t)tokens * HIDDEN * sizeof(float)) == 0 &&
                   memcmp(actual_a, expected,
                          (size_t)tokens * HIDDEN * sizeof(float)) == 0,
               "bucket permutation must preserve exact route-rank accumulation");

        float unchanged[MAX_TOKENS * HIDDEN];
        for (size_t index = 0u; index < MAX_TOKENS * HIDDEN; ++index) {
            unchanged[index] = -9876.0f;
        }
        const uint32_t saved_destination = view.inverse_mapping[0];
        view.inverse_mapping[0] = pairs;
        EXPECT(moe_infer_expert_bucket_combine_reference_f32(
                   &view, bucket_b, shared, HIDDEN, unchanged) ==
                   NATIVE_BATCH_INVALID_ARGUMENT,
               "combine reference rejects out-of-range inverse mapping");
        bool untouched = true;
        for (size_t index = 0u; index < (size_t)tokens * HIDDEN; ++index) {
            untouched = untouched && unchanged[index] == -9876.0f;
        }
        EXPECT(untouched,
               "invalid combine metadata must fail before output publication");
        view.inverse_mapping[0] = saved_destination;
        bucket_b[0] = NAN;
        const int nonfinite_status =
            moe_infer_expert_bucket_combine_reference_f32(
                &view, bucket_b, shared, HIDDEN, unchanged);
        untouched = true;
        for (size_t index = 0u; index < (size_t)tokens * HIDDEN; ++index) {
            untouched = untouched && unchanged[index] == -9876.0f;
        }
        EXPECT(nonfinite_status == NATIVE_BATCH_INVALID_ARGUMENT && untouched,
               "nonfinite bucket output must fail atomically");

        memset(bucket_b, 0, sizeof(bucket_b));
        memset(shared, 0, sizeof(shared));
        for (size_t index = 0u; index < MAX_TOKENS * HIDDEN; ++index) {
            unchanged[index] = -9876.0f;
        }
        const uint32_t overflow_destination = view.inverse_mapping[0];
        view.activation_pairs[0].weight = FLT_MAX;
        view.bucketed_pairs[overflow_destination].weight = FLT_MAX;
        bucket_b[(size_t)overflow_destination * HIDDEN] = FLT_MAX;
        const int overflow_status =
            moe_infer_expert_bucket_combine_reference_f32(
                &view, bucket_b, shared, HIDDEN, unchanged);
        untouched = true;
        for (size_t index = 0u; index < (size_t)tokens * HIDDEN; ++index) {
            untouched = untouched && unchanged[index] == -9876.0f;
        }
        EXPECT(overflow_status == NATIVE_BATCH_INVALID_STATE && untouched,
               "finite derived combine overflow must fail before publication");
        moe_infer_batch_workspace_deinit(&arena);
    }
}

typedef struct {
    uint32_t calls;
    uint32_t calls_by_expert[N_EXPERTS];
    int short_expert;
} ResolverReaderState;

static int64_t resolver_test_reader(
    void *context,
    int fd,
    void *buffer,
    uint64_t size_bytes,
    uint64_t offset_bytes) {
    ResolverReaderState *state = (ResolverReaderState *)context;
    (void)fd;
    const uint32_t expert = size_bytes > 0u
        ? (uint32_t)(offset_bytes / size_bytes)
        : UINT32_MAX;
    ++state->calls;
    if (expert < N_EXPERTS) ++state->calls_by_expert[expert];
    memset(buffer, (int)(expert & 0xffu), (size_t)size_bytes);
    return state->short_expert == (int)expert
        ? (int64_t)size_bytes - 1
        : (int64_t)size_bytes;
}

static void test_unique_expert_resolver(void) {
    enum { TOKENS = 3, SLOT_SIZE = 32, CAPACITY = 6 };
    const uint32_t routed_experts[N_ACTIVE] = {2u, 5u, 9u, 11u, 13u, 17u};
    uint64_t bytes = 0u;
    EXPECT(moe_infer_expert_bucket_workspace_bytes(TOKENS, &bytes) ==
               NATIVE_BATCH_OK,
           "resolver bucket workspace sizing");
    NativeBatchWorkspace arena = {0};
    NativeExpertBucketWorkspace buckets = {0};
    EXPECT(moe_infer_batch_workspace_init(
               &arena, TOKENS, bytes, bytes) == NATIVE_BATCH_OK &&
               moe_infer_expert_bucket_workspace_bind(
               &arena, TOKENS, &buckets) == NATIVE_BATCH_OK,
           "resolver bucket workspace bind");
    for (uint32_t token = 0u; token < TOKENS; ++token) {
        for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
            buckets.activation_pairs[token * N_ACTIVE + rank] =
                (NativeActivationPair){
                    .token_id = token,
                    .expert_id = routed_experts[rank],
                    .route_rank = rank,
                    .weight = 0.25f,
                };
        }
    }
    EXPECT(moe_infer_expert_bucket_build(&buckets) == NATIVE_BATCH_OK &&
               buckets.unique_count == N_ACTIVE,
           "resolver consumes deduplicated ascending expert IDs");

    MoEInferEngine engine = {0};
    engine.initialized = true;
    engine.active_expert_size = SLOT_SIZE;
    engine.packed_fd[0] = 37;
    engine.packed_fd[1] = 38;
    EXPECT(moe_infer_batch_pool_prepare(&engine, CAPACITY) ==
               NATIVE_BATCH_OK,
           "explicit resolver pool prepare");
    EXPECT(moe_infer_batch_pool_prepare(&engine, CAPACITY) ==
               NATIVE_BATCH_OK,
           "resolver pool prepare must reuse compatible storage");
    engine.batch_count = 1;
    engine.batch_eids[0] = 5;
    memset(engine.batch_pool, 0x55, SLOT_SIZE);

    uint8_t hot_data[SLOT_SIZE] = {0};
    uint8_t **hot_layer0 = (uint8_t **)calloc(
        N_EXPERTS, sizeof(uint8_t *));
    uint8_t **hot_layer1 = (uint8_t **)calloc(
        N_EXPERTS, sizeof(uint8_t *));
    EXPECT(hot_layer0 && hot_layer1,
           "resolver test hot-cache tables");
    engine.expert_mem_cache[0] = hot_layer0;
    engine.expert_mem_cache[1] = hot_layer1;
    if (hot_layer0) hot_layer0[2] = hot_data;

    NativeUniqueExpertResolution resolution;
    moe_infer_unique_expert_resolution_init(&resolution);
    ResolverReaderState reader = {.short_expert = -1};
    EXPECT(moe_infer_unique_expert_resolve(
               &engine, 0u, &buckets, &resolution,
               resolver_test_reader, &reader) == NATIVE_BATCH_OK,
           "unique expert resolver success");
    EXPECT(resolution.active == 1u &&
               resolution.owner_engine == (uintptr_t)&engine &&
               resolution.count == N_ACTIVE,
           "successful resolution must publish one pinned entry per unique expert");
    EXPECT(resolution.metrics.activation_pairs == TOKENS * N_ACTIVE &&
               resolution.metrics.unique_experts == N_ACTIVE &&
               resolution.metrics.hot_hits == 1u &&
               resolution.metrics.batch_pool_hits == 1u &&
               resolution.metrics.cold_reads == 4u &&
               resolution.metrics.pread_calls == 4u &&
               resolution.metrics.short_reads == 0u,
           "hot, batch and cold resolver counters must reconcile");
    EXPECT(resolution.metrics.bytes_requested == 4u * SLOT_SIZE &&
               resolution.metrics.bytes_read == 4u * SLOT_SIZE &&
               resolution.metrics.bytes_avoided ==
                   (TOKENS * N_ACTIVE - 4u) * SLOT_SIZE,
           "resolver byte counters must reconcile against pair-wise reads");
    EXPECT(reader.calls == 4u && reader.calls_by_expert[9] == 1u &&
               reader.calls_by_expert[11] == 1u &&
               reader.calls_by_expert[13] == 1u &&
               reader.calls_by_expert[17] == 1u,
           "repeated activation pairs must issue one read per unique cold expert");
    EXPECT(engine.batch_count == 5 && engine.unique_resolver_owner ==
               (uintptr_t)&resolution,
           "cold slots publish only after all reads complete");
    for (int slot = 0; slot < engine.batch_count; ++slot) {
        EXPECT(engine.batch_pins[slot] == 1u,
               "resolved batch slots must stay pinned through use");
    }
    float hidden_guard = 0.0f;
    EXPECT(moe_infer_forward_batch(
               &engine, &hidden_guard, 1, 0, NULL) != 0,
           "legacy forward must reject an active pinned resolution");
    for (int slot = 0; slot < engine.batch_count; ++slot) {
        EXPECT(engine.batch_pins[slot] == 1u,
               "rejected legacy forward must not clear or overwrite pins");
    }
    EXPECT(moe_infer_unique_expert_release(&engine, &resolution) ==
               NATIVE_BATCH_OK,
           "resolver release");
    for (int slot = 0; slot < engine.batch_count; ++slot) {
        EXPECT(engine.batch_pins[slot] == 0u,
               "resolver release must clear every slot pin");
    }

    memset(&reader, 0, sizeof(reader));
    reader.short_expert = -1;
    EXPECT(moe_infer_unique_expert_resolve(
               &engine, 0u, &buckets, &resolution,
               resolver_test_reader, &reader) == NATIVE_BATCH_OK,
           "second resolution reuses per-layer batch pool");
    EXPECT(reader.calls == 0u && resolution.metrics.hot_hits == 1u &&
               resolution.metrics.batch_pool_hits == 5u &&
               resolution.metrics.cold_reads == 0u,
           "repeated resolution must classify prior cold slots as batch hits");
    EXPECT(moe_infer_unique_expert_release(&engine, &resolution) ==
               NATIVE_BATCH_OK,
           "second resolver release");

    engine.batch_count = 1;
    for (int slot = 0; slot < engine.batch_capacity; ++slot) {
        engine.batch_eids[slot] = -1;
        engine.batch_pins[slot] = 0u;
    }
    engine.batch_eids[0] = 5;
    if (hot_layer1) hot_layer1[2] = hot_data;
    memset(&reader, 0, sizeof(reader));
    reader.short_expert = 11;
    EXPECT(moe_infer_unique_expert_resolve(
               &engine, 1u, &buckets, &resolution,
               resolver_test_reader, &reader) ==
               NATIVE_BATCH_INVALID_STATE,
           "short expert read after a successful cold read must abort resolution");
    EXPECT(reader.calls == 2u && reader.calls_by_expert[9] == 1u &&
               reader.calls_by_expert[11] == 1u &&
               resolution.active == 0u && resolution.count == 0u &&
               resolution.owner_engine == 0u &&
               engine.unique_resolver_owner == 0u && engine.batch_count == 1,
           "partial successful I/O must not publish resolution or new batch slots");
    EXPECT(resolution.metrics.layer == 1u &&
               resolution.metrics.error_expert == 11u &&
               resolution.metrics.hot_hits == 1u &&
               resolution.metrics.batch_pool_hits == 1u &&
               resolution.metrics.cold_reads == 2u &&
               resolution.metrics.short_reads == 1u &&
               resolution.metrics.pread_calls == 2u &&
               resolution.metrics.bytes_requested == 2u * SLOT_SIZE &&
               resolution.metrics.bytes_read == 2u * SLOT_SIZE - 1u,
           "partial short-read diagnostics must include prior successful I/O");
    EXPECT(engine.batch_eids[0] == 5 && engine.batch_pins[0] == 0u,
           "failed resolution must preserve the pre-existing batch slot");
    for (int slot = 1; slot < engine.batch_capacity; ++slot) {
        EXPECT(engine.batch_eids[slot] == -1 &&
                   engine.batch_pins[slot] == 0u,
               "failed resolution must leave every new slot uncommitted");
    }

    free(engine.batch_wrap);
    free(engine.batch_eids);
    free(engine.batch_pins);
    free(engine.batch_pool);
    free(hot_layer0);
    free(hot_layer1);
    moe_infer_batch_workspace_deinit(&arena);
}

static void test_batch_route_reference(void) {
    enum { MAX_TOKENS = 5 };
    float logits[MAX_TOKENS * N_EXPERTS];
    float bias[N_EXPERTS];
    uint8_t cached[N_EXPERTS];
    for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
        bias[expert] = (float)((int)(expert % 7u) - 3) * 0.015625f;
        cached[expert] = (expert % 3u) != 0u;
    }
    for (uint32_t token = 0u; token < MAX_TOKENS; ++token) {
        for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
            logits[(size_t)token * N_EXPERTS + expert] =
                (float)((expert * 17u + token * 13u) % 41u) * 0.125f - 2.0f;
        }
    }

    const uint32_t batch_sizes[] = {1u, 2u, 3u, 5u};
    for (size_t case_index = 0u;
         case_index < sizeof(batch_sizes) / sizeof(batch_sizes[0]);
         ++case_index) {
        const uint32_t tokens = batch_sizes[case_index];
        uint64_t batch_bytes = 0u;
        uint64_t scalar_bytes = 0u;
        EXPECT(moe_infer_expert_bucket_workspace_bytes(tokens, &batch_bytes) ==
                   NATIVE_BATCH_OK &&
                   moe_infer_expert_bucket_workspace_bytes(1u, &scalar_bytes) ==
                   NATIVE_BATCH_OK,
               "batch route workspace sizing");
        NativeBatchWorkspace batch_arena = {0};
        NativeBatchWorkspace scalar_arena = {0};
        NativeExpertBucketWorkspace batch_view = {0};
        NativeExpertBucketWorkspace scalar_view = {0};
        EXPECT(moe_infer_batch_workspace_init(
                   &batch_arena, tokens, batch_bytes, batch_bytes) ==
                   NATIVE_BATCH_OK &&
                   moe_infer_batch_workspace_init(
                   &scalar_arena, 1u, scalar_bytes, scalar_bytes) ==
                   NATIVE_BATCH_OK,
               "batch route arena init");
        EXPECT(moe_infer_expert_bucket_workspace_bind(
                   &batch_arena, tokens, &batch_view) == NATIVE_BATCH_OK &&
                   moe_infer_expert_bucket_workspace_bind(
                   &scalar_arena, 1u, &scalar_view) == NATIVE_BATCH_OK,
               "batch route view bind");
        EXPECT(moe_infer_expert_bucket_route_batch_reference(
                   logits, bias, cached, 0.25f, 1u, 1u, &batch_view) ==
                   NATIVE_BATCH_OK,
               "deterministic batch route reference");

        for (uint32_t token = 0u; token < tokens; ++token) {
            EXPECT(moe_infer_expert_bucket_route_batch_reference(
                       logits + (size_t)token * N_EXPERTS,
                       bias, cached, 0.25f, 1u, 1u, &scalar_view) ==
                       NATIVE_BATCH_OK,
                   "repeated scalar route reference");
            uint8_t seen[N_EXPERTS] = {0};
            float weight_sum = 0.0f;
            for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
                const NativeActivationPair *batch_pair =
                    &batch_view.activation_pairs[(size_t)token * N_ACTIVE + rank];
                const NativeActivationPair *scalar_pair =
                    &scalar_view.activation_pairs[rank];
                EXPECT(batch_pair->token_id == token &&
                           batch_pair->route_rank == rank,
                       "activation pairs must be token-major with explicit rank");
                EXPECT(batch_pair->expert_id == scalar_pair->expert_id &&
                           batch_pair->weight == scalar_pair->weight,
                       "batch route must exactly match repeated scalar route");
                EXPECT(batch_pair->expert_id < N_EXPERTS &&
                           !seen[batch_pair->expert_id] &&
                           isfinite(batch_pair->weight),
                       "every token must contain six unique finite routes");
                seen[batch_pair->expert_id] = 1u;
                weight_sum += batch_pair->weight;
            }
            EXPECT(fabsf(weight_sum - 1.5f) <= 2e-6f,
                   "route weights must L1-normalize to route scale");
        }
        moe_infer_batch_workspace_deinit(&scalar_arena);
        moe_infer_batch_workspace_deinit(&batch_arena);
    }

    float tie_logits[N_EXPERTS] = {0};
    uint64_t tie_bytes = 0u;
    EXPECT(moe_infer_expert_bucket_workspace_bytes(1u, &tie_bytes) ==
               NATIVE_BATCH_OK,
           "tie route workspace sizing");
    NativeBatchWorkspace tie_arena = {0};
    NativeExpertBucketWorkspace tie_view = {0};
    EXPECT(moe_infer_batch_workspace_init(
               &tie_arena, 1u, tie_bytes, tie_bytes) == NATIVE_BATCH_OK &&
               moe_infer_expert_bucket_workspace_bind(
               &tie_arena, 1u, &tie_view) == NATIVE_BATCH_OK,
           "tie route workspace bind");
    EXPECT(moe_infer_expert_bucket_route_batch_reference(
               tie_logits, NULL, NULL, 0.0f, 0u, 0u, &tie_view) ==
               NATIVE_BATCH_OK,
           "tie route reference");
    for (uint32_t rank = 0u; rank < N_ACTIVE; ++rank) {
        EXPECT(tie_view.activation_pairs[rank].expert_id == rank,
               "equal route scores must prefer lower expert IDs");
        EXPECT(fabsf(tie_view.activation_pairs[rank].weight - 0.25f) <= 1e-6f,
               "equal route scores must receive equal weights");
    }

    NativeActivationPair published_before_overflow[N_ACTIVE];
    memcpy(published_before_overflow, tie_view.activation_pairs,
           sizeof(published_before_overflow));
    float overflow_bias[N_EXPERTS];
    for (uint32_t expert = 0u; expert < N_EXPERTS; ++expert) {
        tie_logits[expert] = FLT_MAX;
        overflow_bias[expert] = FLT_MAX;
    }
    EXPECT(moe_infer_expert_bucket_route_batch_reference(
               tie_logits, overflow_bias, NULL, 0.0f, 1u, 0u, &tie_view) ==
               NATIVE_BATCH_INVALID_STATE,
           "finite route inputs with nonfinite derived weights must fail");
    EXPECT(memcmp(published_before_overflow, tie_view.activation_pairs,
                  sizeof(published_before_overflow)) == 0,
           "derived route overflow must not publish activation pairs");

    memset(tie_logits, 0, sizeof(tie_logits));
    tie_view.activation_pairs[0].expert_id = 123u;
    tie_logits[17] = NAN;
    EXPECT(moe_infer_expert_bucket_route_batch_reference(
               tie_logits, NULL, NULL, 0.0f, 0u, 0u, &tie_view) ==
               NATIVE_BATCH_INVALID_ARGUMENT,
           "nonfinite route input must fail before publication");
    EXPECT(tie_view.activation_pairs[0].expert_id == 123u,
           "failed route generation must not mutate activation pairs");
    moe_infer_batch_workspace_deinit(&tie_arena);
}

static void emit_synthetic_record(
    CaptureSet *set,
    MoeInferCapturePath path,
    int layer,
    int position,
    MoeInferCaptureCheckpoint checkpoint) {
    float values[8] = {0};
    int expert_ids[N_ACTIVE] = {0};
    const size_t value_count = checkpoint == MOE_INFER_CAPTURE_ROUTER ? N_ACTIVE : 4;
    const size_t expert_count = checkpoint == MOE_INFER_CAPTURE_ROUTER ? N_ACTIVE : 0;

    for (size_t index = 0; index < value_count; ++index) {
        values[index] = (float)(layer * 1000 + position * 10 + checkpoint) +
            (float)index * 0.03125f;
    }
    for (size_t index = 0; index < expert_count; ++index) {
        expert_ids[index] = (layer * 17 + position * 7 + (int)index) % N_EXPERTS;
    }

    const MoeInferDebugCaptureRecord record = {
        .path = path,
        .checkpoint = checkpoint,
        .dtype = checkpoint == MOE_INFER_CAPTURE_ATTENTION_OUTPUT
            ? MOE_INFER_CAPTURE_DTYPE_BF16
            : MOE_INFER_CAPTURE_DTYPE_F32,
        .layer = layer,
        .position = position,
        .kv_length = position + (checkpoint != MOE_INFER_CAPTURE_LAYER_INPUT),
        .compressor_length = (uint32_t)(position / 4),
        .indexer_length = (uint32_t)(position / 4),
        .values = values,
        .value_count = value_count,
        .expert_ids = expert_count ? expert_ids : NULL,
        .expert_count = expert_count,
    };
    dmlx_batch_capture_record(set, &record);
}

static void capture_synthetic(
    CaptureSet *set,
    MoeInferCapturePath path,
    int start_pos,
    int n_tokens) {
    if (path == MOE_INFER_CAPTURE_SCALAR) {
        for (int token = 0; token < n_tokens; ++token) {
            for (int layer = 0; layer < N_LAYERS; ++layer) {
                for (int checkpoint = MOE_INFER_CAPTURE_LAYER_INPUT;
                     checkpoint <= MOE_INFER_CAPTURE_LAYER_OUTPUT;
                     ++checkpoint) {
                    emit_synthetic_record(set, path, layer, start_pos + token,
                        (MoeInferCaptureCheckpoint)checkpoint);
                }
            }
        }
    } else {
        for (int layer = 0; layer < N_LAYERS; ++layer) {
            for (int token = 0; token < n_tokens; ++token) {
                for (int checkpoint = MOE_INFER_CAPTURE_LAYER_INPUT;
                     checkpoint <= MOE_INFER_CAPTURE_LAYER_OUTPUT;
                     ++checkpoint) {
                    emit_synthetic_record(set, path, layer, start_pos + token,
                        (MoeInferCaptureCheckpoint)checkpoint);
                }
            }
        }
    }
}

static void test_batch_size(int n_tokens) {
    CaptureSet scalar = {0};
    CaptureSet batch = {0};
    capture_synthetic(&scalar, MOE_INFER_CAPTURE_SCALAR, 0, n_tokens);
    capture_synthetic(&batch, MOE_INFER_CAPTURE_BATCH, 0, n_tokens);
    EXPECT(compare_captures(&scalar, &batch, 0, n_tokens).equal,
           "scalar and batch captures must match");
    clear_capture_set(&scalar);
    clear_capture_set(&batch);
    free(scalar.records);
    free(batch.records);
}

static void test_earliest_divergence(void) {
    CaptureSet scalar = {0};
    CaptureSet batch = {0};
    capture_synthetic(&scalar, MOE_INFER_CAPTURE_SCALAR, 0, 5);
    capture_synthetic(&batch, MOE_INFER_CAPTURE_BATCH, 0, 5);
    OwnedCapture *later = find_capture(&batch, 1, 4, MOE_INFER_CAPTURE_LAYER_OUTPUT);
    OwnedCapture *earlier = find_capture(&batch, 0, 2, MOE_INFER_CAPTURE_MOE_OUTPUT);
    EXPECT(later && earlier, "injected checkpoints must exist");
    if (later && earlier) {
        later->values[0] += 10.0f;
        earlier->values[3] += 1.0f;
    }
    const CaptureComparison comparison = compare_captures(&scalar, &batch, 0, 5);
    EXPECT(!comparison.equal && comparison.layer == 0 && comparison.position == 2,
           "comparison must report earliest layer/token");
    EXPECT(comparison.checkpoint == MOE_INFER_CAPTURE_MOE_OUTPUT && comparison.index == 3,
           "comparison must report earliest checkpoint/index");
    clear_capture_set(&scalar);
    clear_capture_set(&batch);
    free(scalar.records);
    free(batch.records);
}

static void test_state_digest_divergence(void) {
    CaptureSet scalar = {0};
    CaptureSet batch = {0};
    capture_synthetic(&scalar, MOE_INFER_CAPTURE_SCALAR, 0, 2);
    capture_synthetic(&batch, MOE_INFER_CAPTURE_BATCH, 0, 2);
    OwnedCapture *checkpoint = find_capture(
        &batch, 0, 1, MOE_INFER_CAPTURE_ATTENTION_OUTPUT);
    EXPECT(checkpoint != NULL, "state digest checkpoint must exist");
    if (checkpoint) checkpoint->kv_state_digest ^= UINT64_C(1);
    const CaptureComparison comparison = compare_captures(&scalar, &batch, 0, 2);
    EXPECT(!comparison.equal && comparison.state_mismatch,
           "state payload digest mismatch must fail exactly");
    EXPECT(comparison.layer == 0 && comparison.position == 1,
           "state mismatch must identify earliest layer/token");
    clear_capture_set(&scalar);
    clear_capture_set(&batch);
    free(scalar.records);
    free(batch.records);
}

static void test_router_id_divergence(void) {
    CaptureSet scalar = {0};
    CaptureSet batch = {0};
    capture_synthetic(&scalar, MOE_INFER_CAPTURE_SCALAR, 0, 3);
    capture_synthetic(&batch, MOE_INFER_CAPTURE_BATCH, 0, 3);
    OwnedCapture *router = find_capture(&batch, 0, 1, MOE_INFER_CAPTURE_ROUTER);
    EXPECT(router != NULL, "router checkpoint must exist");
    if (router) router->expert_ids[2] = (router->expert_ids[2] + 1) % N_EXPERTS;
    const CaptureComparison comparison = compare_captures(&scalar, &batch, 0, 3);
    EXPECT(!comparison.equal && comparison.expert_id_mismatch,
           "router expert set mismatch must fail exactly");
    EXPECT(comparison.layer == 0 && comparison.position == 1 && comparison.index == 2,
           "router mismatch must identify earliest layer/token/index");
    clear_capture_set(&scalar);
    clear_capture_set(&batch);
    free(scalar.records);
    free(batch.records);
}

static void test_final_logits_wrapper(void) {
    const float reference[] = {1.0f, -2.0f, 3.0f};
    float actual[] = {1.0f, -2.0f, 3.0f};
    DmlxBatchLogitsReport report;

    EXPECT(dmlx_batch_logits_compare(reference, actual, 3, 2, &report) == 0,
           "equal final logits must pass centralized wrapper");
    EXPECT(report.status == BATCH_ORACLE_OK,
           "equal final logits report OK");
    EXPECT(report.tolerance_absolute > 0.0 && report.tolerance_relative > 0.0,
           "centralized final-logits tolerance must be exported");

    actual[1] += (float)(0.25 *
        (report.tolerance_absolute + report.tolerance_relative * 2.0));
    EXPECT(dmlx_batch_logits_compare(reference, actual, 3, 2, &report) == 0,
           "within-tolerance final logits must pass wrapper");

    actual[1] = reference[1] + 1.0f;
    EXPECT(dmlx_batch_logits_compare(reference, actual, 3, 2, &report) != 0,
           "outside-tolerance final logits must fail wrapper");
    EXPECT(report.status == BATCH_ORACLE_OUTSIDE_TOLERANCE &&
               report.has_first_divergence && report.first_index == 1,
           "final-logits wrapper must expose first divergence");
    EXPECT(report.layer == N_LAYERS && report.token == 2,
           "final-logits wrapper must preserve terminal layer/token location");

    actual[1] = NAN;
    EXPECT(dmlx_batch_logits_compare(reference, actual, 3, 2, &report) != 0,
           "nonfinite final logits must fail wrapper");
    EXPECT(report.status == BATCH_ORACLE_NONFINITE_ACTUAL,
           "final-logits wrapper must report nonfinite actual");
}

int main(void) {
    test_expert_bucket_workspace_layout();
    test_stable_expert_bucket_build();
    test_deterministic_bucket_combine_reference();
    test_unique_expert_resolver();
    test_batch_route_reference();
    test_batch_size(1);
    test_batch_size(2);
    test_batch_size(3);
    test_batch_size(5);
    test_earliest_divergence();
    test_state_digest_divergence();
    test_router_id_divergence();
    test_final_logits_wrapper();

    if (failures != 0) {
        fprintf(stderr, "batch capture self-test: %d failure(s)\n", failures);
        return 1;
    }
    printf("batch capture self-test: PASS (B=1,2,3,5,ordering)\n");
    return 0;
}
#endif
