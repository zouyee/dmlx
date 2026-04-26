#!/usr/bin/env python3
"""
Golden tests / behavior exploration for mlx.core.fast.scaled_dot_product_attention.
Goal: understand the C API contract for mask_mode, mask array, and shapes.
"""

import sys
import traceback
import mlx.core as mx

OUT_PATH = "/Users/zouyee/work/code/mlx-zig/tests/golden/test_sdpa_out.txt"


def log(msg: str):
    print(msg)
    sys.stdout.flush()


def run_case(name: str, fn):
    log(f"\n=== {name} ===")
    try:
        result = fn()
        if isinstance(result, mx.array):
            log(f"  result shape: {result.shape}, dtype: {result.dtype}")
            # evaluate to force any lazy errors
            _ = result.tolist()
            log("  OK (evaluated)")
        else:
            log(f"  OK: {result}")
    except Exception as e:
        log(f"  EXCEPTION: {type(e).__name__}: {e}")
        log("  " + traceback.format_exc().replace("\n", "\n  "))


def test_no_mask():
    B, N_q, T_q, D = 2, 4, 8, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_q, T_q, D))
    v = mx.random.normal(shape=(B, N_q, T_q, D))
    scale = D ** -0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
    return out


def test_causal_mask():
    B, N_q, T_q, D = 2, 4, 8, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_q, T_q, D))
    v = mx.random.normal(shape=(B, N_q, T_q, D))
    scale = D ** -0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    return out


def test_invalid_mask_mode():
    B, N_q, T_q, D = 2, 4, 8, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_q, T_q, D))
    v = mx.random.normal(shape=(B, N_q, T_q, D))
    scale = D ** -0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="invalid_mode_xyz")
    return out


def test_empty_string_mask_mode():
    B, N_q, T_q, D = 2, 4, 8, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_q, T_q, D))
    v = mx.random.normal(shape=(B, N_q, T_q, D))
    scale = D ** -0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="")
    return out


def test_bool_mask_array():
    B, N_q, T_q, T_kv, D = 2, 4, 8, 8, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_q, T_kv, D))
    v = mx.random.normal(shape=(B, N_q, T_kv, D))
    scale = D ** -0.5
    mask = mx.random.uniform(shape=(B, N_q, T_q, T_kv)) > 0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return out


def test_additive_mask_array():
    B, N_q, T_q, T_kv, D = 2, 4, 8, 8, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_q, T_kv, D))
    v = mx.random.normal(shape=(B, N_q, T_kv, D))
    scale = D ** -0.5
    mask = mx.random.normal(shape=(B, N_q, T_q, T_kv))
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return out


def test_gqa():
    B, N_q, N_kv, T_q, T_kv, D = 2, 8, 2, 8, 8, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_kv, T_kv, D))
    v = mx.random.normal(shape=(B, N_kv, T_kv, D))
    scale = D ** -0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    return out


def test_mqa():
    B, N_q, T_q, T_kv, D = 2, 8, 8, 8, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, 1, T_kv, D))
    v = mx.random.normal(shape=(B, 1, T_kv, D))
    scale = D ** -0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    return out


def test_mismatched_qk_lengths():
    B, N_q, T_q, T_kv, D = 2, 4, 8, 12, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_q, T_kv, D))
    v = mx.random.normal(shape=(B, N_q, T_kv, D))
    scale = D ** -0.5
    mask = mx.random.normal(shape=(B, N_q, T_q, T_kv))
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return out


def test_incompatible_mask_shape():
    B, N_q, T_q, T_kv, D = 2, 4, 8, 8, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_q, T_kv, D))
    v = mx.random.normal(shape=(B, N_q, T_kv, D))
    scale = D ** -0.5
    mask = mx.random.normal(shape=(B, N_q, T_q, T_q + 1))  # incompatible T_kv
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return out


def test_sinks_none():
    B, N_q, T_q, D = 2, 4, 8, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_q, T_q, D))
    v = mx.random.normal(shape=(B, N_q, T_q, D))
    scale = D ** -0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None, sinks=None)
    return out


def test_sinks_array():
    B, N_q, T_q, D = 2, 4, 8, 16
    q = mx.random.normal(shape=(B, N_q, T_q, D))
    k = mx.random.normal(shape=(B, N_q, T_q, D))
    v = mx.random.normal(shape=(B, N_q, T_q, D))
    scale = D ** -0.5
    sinks = mx.array([0, 2])
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal", sinks=sinks)
    return out


def main():
    log("mlx version: " + mx.__version__)
    log("Running scaled_dot_product_attention behavior tests...")

    run_case("no_mask", test_no_mask)
    run_case("causal_mask", test_causal_mask)
    run_case("invalid_mask_mode", test_invalid_mask_mode)
    run_case("empty_string_mask_mode", test_empty_string_mask_mode)
    run_case("bool_mask_array", test_bool_mask_array)
    run_case("additive_mask_array", test_additive_mask_array)
    run_case("grouped_query_attention", test_gqa)
    run_case("multi_query_attention", test_mqa)
    run_case("mismatched_qk_lengths", test_mismatched_qk_lengths)
    run_case("incompatible_mask_shape", test_incompatible_mask_shape)
    run_case("sinks_none", test_sinks_none)
    run_case("sinks_array", test_sinks_array)

    log("\n=== All tests completed ===")


if __name__ == "__main__":
    main()
