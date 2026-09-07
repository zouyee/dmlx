#ifndef DMLX_METAL_INFER_BATCH_ORACLE_H
#define DMLX_METAL_INFER_BATCH_ORACLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tensor values are decoded to float before comparison. The dtype records the
 * source storage/compute precision and selects the only authoritative tolerance
 * table. Callers must not add checkpoint-specific literals of their own.
 */
typedef enum BatchOracleDType {
    BATCH_ORACLE_DTYPE_F32 = 0,
    BATCH_ORACLE_DTYPE_F16,
    BATCH_ORACLE_DTYPE_BF16,
    BATCH_ORACLE_DTYPE_COUNT,
} BatchOracleDType;

typedef enum BatchOracleCheckpoint {
    BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE = 0,
    BATCH_ORACLE_CHECKPOINT_ROUTER_WEIGHTS,
    BATCH_ORACLE_CHECKPOINT_MOE_OUTPUT,
    BATCH_ORACLE_CHECKPOINT_ATTENTION_OUTPUT,
    BATCH_ORACLE_CHECKPOINT_FINAL_LOGITS,
    BATCH_ORACLE_CHECKPOINT_ATTENTION_Q,
    BATCH_ORACLE_CHECKPOINT_ATTENTION_KV,
    BATCH_ORACLE_CHECKPOINT_ATTENTION_SDPA,
    BATCH_ORACLE_CHECKPOINT_COUNT,
} BatchOracleCheckpoint;

typedef struct BatchOracleTolerance {
    double absolute;
    double relative;
} BatchOracleTolerance;

typedef enum BatchOracleStatus {
    BATCH_ORACLE_OK = 0,
    BATCH_ORACLE_INVALID_ARGUMENT,
    BATCH_ORACLE_NONFINITE_REFERENCE,
    BATCH_ORACLE_NONFINITE_ACTUAL,
    BATCH_ORACLE_OUTSIDE_TOLERANCE,
} BatchOracleStatus;

typedef struct BatchOracleReport {
    BatchOracleStatus status;
    BatchOracleDType dtype;
    BatchOracleCheckpoint checkpoint;
    BatchOracleTolerance tolerance;
    int32_t layer;
    int32_t token;
    size_t compared_count;

    bool has_first_divergence;
    size_t first_index;
    float first_reference_value;
    float first_actual_value;
    double first_absolute_error;
    double first_relative_error;

    double max_absolute_error;
    size_t max_absolute_index;
    double max_relative_error;
    size_t max_relative_index;
} BatchOracleReport;

/* Returns false for an invalid dtype/checkpoint or null output pointer. */
bool batch_oracle_get_tolerance(
    BatchOracleDType dtype,
    BatchOracleCheckpoint checkpoint,
    BatchOracleTolerance *out_tolerance);

/*
 * Compare one ordered checkpoint tensor. A caller comparing multiple
 * checkpoints should invoke this function in forward order and retain the first
 * non-OK report, which identifies the earliest layer/token/index divergence.
 *
 * Elements pass when:
 *   abs(ref - actual) <= absolute + relative * max(abs(ref), abs(actual))
 *
 * Any NaN or infinity fails regardless of tolerance. Empty tensors are valid;
 * null data pointers are valid only when count is zero.
 */
BatchOracleStatus batch_oracle_compare_f32(
    const float *reference,
    const float *actual,
    size_t count,
    BatchOracleDType dtype,
    BatchOracleCheckpoint checkpoint,
    int32_t layer,
    int32_t token,
    BatchOracleReport *out_report);

const char *batch_oracle_status_name(BatchOracleStatus status);
const char *batch_oracle_dtype_name(BatchOracleDType dtype);
const char *batch_oracle_checkpoint_name(BatchOracleCheckpoint checkpoint);

#ifdef __cplusplus
}
#endif

#endif
