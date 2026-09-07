#include "batch_oracle.h"

#include <float.h>
#include <math.h>
#include <string.h>

/*
 * Central scalar-versus-batch tolerance table.
 *
 * These thresholds compare checkpoint values after conversion to float32; they
 * do not redefine model quantization semantics. Changes require oracle fixture
 * evidence and must update the self-test expectations rather than introducing
 * local caller tolerances.
 */
static const BatchOracleTolerance k_tolerances[BATCH_ORACLE_DTYPE_COUNT]
                                                   [BATCH_ORACLE_CHECKPOINT_COUNT] = {
    [BATCH_ORACLE_DTYPE_F32] = {
        [BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE] = {1.0e-5, 1.0e-5},
        [BATCH_ORACLE_CHECKPOINT_ROUTER_WEIGHTS] = {1.0e-6, 1.0e-5},
        [BATCH_ORACLE_CHECKPOINT_MOE_OUTPUT] = {2.0e-4, 1.0e-4},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_OUTPUT] = {2.0e-4, 1.0e-4},
        [BATCH_ORACLE_CHECKPOINT_FINAL_LOGITS] = {1.0e-4, 1.0e-4},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_Q] = {1.0e-5, 1.0e-5},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_KV] = {1.0e-5, 1.0e-5},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_SDPA] = {2.0e-4, 1.0e-4},
    },
    [BATCH_ORACLE_DTYPE_F16] = {
        [BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE] = {2.0e-3, 2.0e-3},
        [BATCH_ORACLE_CHECKPOINT_ROUTER_WEIGHTS] = {1.0e-3, 2.0e-3},
        [BATCH_ORACLE_CHECKPOINT_MOE_OUTPUT] = {5.0e-3, 5.0e-3},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_OUTPUT] = {5.0e-3, 5.0e-3},
        [BATCH_ORACLE_CHECKPOINT_FINAL_LOGITS] = {3.0e-3, 3.0e-3},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_Q] = {2.0e-3, 2.0e-3},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_KV] = {2.0e-3, 2.0e-3},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_SDPA] = {5.0e-3, 5.0e-3},
    },
    [BATCH_ORACLE_DTYPE_BF16] = {
        [BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE] = {2.0e-2, 2.0e-2},
        [BATCH_ORACLE_CHECKPOINT_ROUTER_WEIGHTS] = {5.0e-3, 1.0e-2},
        [BATCH_ORACLE_CHECKPOINT_MOE_OUTPUT] = {3.0e-2, 3.0e-2},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_OUTPUT] = {3.0e-2, 3.0e-2},
        [BATCH_ORACLE_CHECKPOINT_FINAL_LOGITS] = {2.0e-2, 2.0e-2},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_Q] = {2.0e-2, 2.0e-2},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_KV] = {2.0e-2, 2.0e-2},
        [BATCH_ORACLE_CHECKPOINT_ATTENTION_SDPA] = {3.0e-2, 3.0e-2},
    },
};

static bool valid_dtype(BatchOracleDType dtype) {
    return dtype >= BATCH_ORACLE_DTYPE_F32 && dtype < BATCH_ORACLE_DTYPE_COUNT;
}

static bool valid_checkpoint(BatchOracleCheckpoint checkpoint) {
    return checkpoint >= BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE &&
           checkpoint < BATCH_ORACLE_CHECKPOINT_COUNT;
}

bool batch_oracle_get_tolerance(
    BatchOracleDType dtype,
    BatchOracleCheckpoint checkpoint,
    BatchOracleTolerance *out_tolerance) {
    if (!out_tolerance || !valid_dtype(dtype) || !valid_checkpoint(checkpoint)) {
        return false;
    }
    *out_tolerance = k_tolerances[dtype][checkpoint];
    return true;
}

static double symmetric_relative_error(double reference, double actual, double absolute_error) {
    const double denominator = fmax(fabs(reference), fabs(actual));
    return denominator > 0.0 ? absolute_error / denominator : 0.0;
}

static void set_first_divergence(
    BatchOracleReport *report,
    BatchOracleStatus status,
    size_t index,
    float reference,
    float actual,
    double absolute_error,
    double relative_error) {
    if (report->has_first_divergence) {
        return;
    }
    report->status = status;
    report->has_first_divergence = true;
    report->first_index = index;
    report->first_reference_value = reference;
    report->first_actual_value = actual;
    report->first_absolute_error = absolute_error;
    report->first_relative_error = relative_error;
}

BatchOracleStatus batch_oracle_compare_f32(
    const float *reference,
    const float *actual,
    size_t count,
    BatchOracleDType dtype,
    BatchOracleCheckpoint checkpoint,
    int32_t layer,
    int32_t token,
    BatchOracleReport *out_report) {
    if (!out_report) {
        return BATCH_ORACLE_INVALID_ARGUMENT;
    }

    memset(out_report, 0, sizeof(*out_report));
    out_report->status = BATCH_ORACLE_INVALID_ARGUMENT;
    out_report->dtype = dtype;
    out_report->checkpoint = checkpoint;
    out_report->layer = layer;
    out_report->token = token;
    out_report->compared_count = count;
    out_report->first_index = SIZE_MAX;
    out_report->max_absolute_index = SIZE_MAX;
    out_report->max_relative_index = SIZE_MAX;

    if ((!reference && count != 0) || (!actual && count != 0) ||
        !batch_oracle_get_tolerance(dtype, checkpoint, &out_report->tolerance)) {
        return out_report->status;
    }

    out_report->status = BATCH_ORACLE_OK;
    for (size_t index = 0; index < count; ++index) {
        const float reference_value = reference[index];
        const float actual_value = actual[index];
        const bool reference_finite = isfinite(reference_value);
        const bool actual_finite = isfinite(actual_value);

        if (!reference_finite || !actual_finite) {
            const BatchOracleStatus status = !reference_finite
                ? BATCH_ORACLE_NONFINITE_REFERENCE
                : BATCH_ORACLE_NONFINITE_ACTUAL;
            set_first_divergence(
                out_report,
                status,
                index,
                reference_value,
                actual_value,
                INFINITY,
                INFINITY);
            out_report->max_absolute_error = INFINITY;
            out_report->max_absolute_index = index;
            out_report->max_relative_error = INFINITY;
            out_report->max_relative_index = index;
            continue;
        }

        const double reference_double = reference_value;
        const double actual_double = actual_value;
        const double absolute_error = fabs(reference_double - actual_double);
        const double relative_error = symmetric_relative_error(
            reference_double, actual_double, absolute_error);

        if (out_report->max_absolute_index == SIZE_MAX ||
            absolute_error > out_report->max_absolute_error) {
            out_report->max_absolute_error = absolute_error;
            out_report->max_absolute_index = index;
        }
        if (out_report->max_relative_index == SIZE_MAX ||
            relative_error > out_report->max_relative_error) {
            out_report->max_relative_error = relative_error;
            out_report->max_relative_index = index;
        }

        const double scale = fmax(fabs(reference_double), fabs(actual_double));
        const double allowed_error = out_report->tolerance.absolute +
            out_report->tolerance.relative * scale;
        if (absolute_error > allowed_error) {
            set_first_divergence(
                out_report,
                BATCH_ORACLE_OUTSIDE_TOLERANCE,
                index,
                reference_value,
                actual_value,
                absolute_error,
                relative_error);
        }
    }

    return out_report->status;
}

const char *batch_oracle_status_name(BatchOracleStatus status) {
    switch (status) {
        case BATCH_ORACLE_OK: return "ok";
        case BATCH_ORACLE_INVALID_ARGUMENT: return "invalid_argument";
        case BATCH_ORACLE_NONFINITE_REFERENCE: return "nonfinite_reference";
        case BATCH_ORACLE_NONFINITE_ACTUAL: return "nonfinite_actual";
        case BATCH_ORACLE_OUTSIDE_TOLERANCE: return "outside_tolerance";
    }
    return "unknown";
}

const char *batch_oracle_dtype_name(BatchOracleDType dtype) {
    switch (dtype) {
        case BATCH_ORACLE_DTYPE_F32: return "f32";
        case BATCH_ORACLE_DTYPE_F16: return "f16";
        case BATCH_ORACLE_DTYPE_BF16: return "bf16";
        case BATCH_ORACLE_DTYPE_COUNT: break;
    }
    return "unknown";
}

const char *batch_oracle_checkpoint_name(BatchOracleCheckpoint checkpoint) {
    switch (checkpoint) {
        case BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE: return "hidden_state";
        case BATCH_ORACLE_CHECKPOINT_ROUTER_WEIGHTS: return "router_weights";
        case BATCH_ORACLE_CHECKPOINT_MOE_OUTPUT: return "moe_output";
        case BATCH_ORACLE_CHECKPOINT_ATTENTION_OUTPUT: return "attention_output";
        case BATCH_ORACLE_CHECKPOINT_FINAL_LOGITS: return "final_logits";
        case BATCH_ORACLE_CHECKPOINT_ATTENTION_Q: return "attention_q";
        case BATCH_ORACLE_CHECKPOINT_ATTENTION_KV: return "attention_kv";
        case BATCH_ORACLE_CHECKPOINT_ATTENTION_SDPA: return "attention_sdpa";
        case BATCH_ORACLE_CHECKPOINT_COUNT: break;
    }
    return "unknown";
}

#ifdef DMLX_BATCH_ORACLE_SELF_TEST

#include <stdio.h>

static int failures = 0;

#define EXPECT(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__); \
        ++failures; \
    } \
} while (0)

static void test_tolerance_table(void) {
    for (int dtype = 0; dtype < BATCH_ORACLE_DTYPE_COUNT; ++dtype) {
        for (int checkpoint = 0; checkpoint < BATCH_ORACLE_CHECKPOINT_COUNT; ++checkpoint) {
            BatchOracleTolerance tolerance;
            EXPECT(
                batch_oracle_get_tolerance(
                    (BatchOracleDType)dtype,
                    (BatchOracleCheckpoint)checkpoint,
                    &tolerance),
                "every dtype/checkpoint must have a tolerance");
            EXPECT(tolerance.absolute > 0.0, "absolute tolerance must be positive");
            EXPECT(tolerance.relative > 0.0, "relative tolerance must be positive");
        }
    }
}

static void test_equal(void) {
    const float values[] = {-3.0f, 0.0f, 1.5f};
    BatchOracleReport report;
    const BatchOracleStatus status = batch_oracle_compare_f32(
        values,
        values,
        3,
        BATCH_ORACLE_DTYPE_F32,
        BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE,
        4,
        7,
        &report);
    EXPECT(status == BATCH_ORACLE_OK, "equal tensors must pass");
    EXPECT(!report.has_first_divergence, "equal tensors have no divergence");
    EXPECT(report.max_absolute_error == 0.0, "equal max absolute error is zero");
    EXPECT(report.max_relative_error == 0.0, "equal max relative error is zero");
}

static void test_within_tolerance(void) {
    const float reference[] = {1.0f};
    BatchOracleTolerance tolerance;
    BatchOracleReport report;
    EXPECT(
        batch_oracle_get_tolerance(
            BATCH_ORACLE_DTYPE_F32,
            BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE,
            &tolerance),
        "lookup f32 hidden tolerance");
    const float actual[] = {
        1.0f + (float)(0.25 * (tolerance.absolute + tolerance.relative)),
    };
    EXPECT(
        batch_oracle_compare_f32(
            reference,
            actual,
            1,
            BATCH_ORACLE_DTYPE_F32,
            BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE,
            0,
            0,
            &report) == BATCH_ORACLE_OK,
        "value within combined tolerance must pass");
}

static void test_outside_tolerance_and_location(void) {
    const float reference[] = {1.0f, 2.0f, 3.0f};
    const float actual[] = {1.0f, 2.1f, 3.5f};
    BatchOracleReport report;
    const BatchOracleStatus status = batch_oracle_compare_f32(
        reference,
        actual,
        3,
        BATCH_ORACLE_DTYPE_F32,
        BATCH_ORACLE_CHECKPOINT_FINAL_LOGITS,
        11,
        13,
        &report);
    EXPECT(status == BATCH_ORACLE_OUTSIDE_TOLERANCE, "outside tolerance must fail");
    EXPECT(report.has_first_divergence, "mismatch must record first divergence");
    EXPECT(report.layer == 11 && report.token == 13, "location must preserve layer/token");
    EXPECT(report.first_index == 1, "first mismatch index must be reported");
    EXPECT(report.max_absolute_index == 2, "max absolute error index must be reported");
    EXPECT(report.max_relative_index == 2, "max relative error index must be reported");
}

static void test_nonfinite_reference(void) {
    const float reference[] = {0.0f, NAN};
    const float actual[] = {0.0f, 1.0f};
    BatchOracleReport report;
    const BatchOracleStatus status = batch_oracle_compare_f32(
        reference,
        actual,
        2,
        BATCH_ORACLE_DTYPE_BF16,
        BATCH_ORACLE_CHECKPOINT_MOE_OUTPUT,
        2,
        3,
        &report);
    EXPECT(status == BATCH_ORACLE_NONFINITE_REFERENCE, "reference NaN must fail");
    EXPECT(report.first_index == 1, "reference NaN index must be reported");
    EXPECT(isinf(report.max_absolute_error), "nonfinite max absolute error is infinite");
}

static void test_nonfinite_actual(void) {
    const float reference[] = {1.0f};
    const float actual[] = {INFINITY};
    BatchOracleReport report;
    const BatchOracleStatus status = batch_oracle_compare_f32(
        reference,
        actual,
        1,
        BATCH_ORACLE_DTYPE_F16,
        BATCH_ORACLE_CHECKPOINT_ATTENTION_OUTPUT,
        5,
        8,
        &report);
    EXPECT(status == BATCH_ORACLE_NONFINITE_ACTUAL, "actual infinity must fail");
    EXPECT(report.first_index == 0, "actual infinity index must be reported");
}

static void test_invalid_arguments_and_empty(void) {
    BatchOracleReport report;
    EXPECT(
        batch_oracle_compare_f32(
            NULL,
            NULL,
            0,
            BATCH_ORACLE_DTYPE_F32,
            BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE,
            0,
            0,
            &report) == BATCH_ORACLE_OK,
        "empty tensors may use null data pointers");
    EXPECT(
        batch_oracle_compare_f32(
            NULL,
            NULL,
            1,
            BATCH_ORACLE_DTYPE_F32,
            BATCH_ORACLE_CHECKPOINT_HIDDEN_STATE,
            0,
            0,
            &report) == BATCH_ORACLE_INVALID_ARGUMENT,
        "non-empty tensors require data pointers");
}

int main(void) {
    test_tolerance_table();
    test_equal();
    test_within_tolerance();
    test_outside_tolerance_and_location();
    test_nonfinite_reference();
    test_nonfinite_actual();
    test_invalid_arguments_and_empty();

    if (failures != 0) {
        fprintf(stderr, "batch_oracle self-test: %d failure(s)\n", failures);
        return 1;
    }
    printf("batch_oracle self-test: PASS\n");
    return 0;
}

#endif
