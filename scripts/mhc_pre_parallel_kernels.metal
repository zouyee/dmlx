#include <metal_stdlib>
using namespace metal;

constant uint kRows = 24;
constant uint kWidth = 16384;

kernel void mhc_dot_serial_one_lane_per_row(
    device const float *weights [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    constant float &norm [[buffer(2)]],
    device float *out [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    for (uint row = lid; row < kRows; row += threads) {
        float acc = 0.0f;
        device const float *w = weights + row * kWidth;
        for (uint i = 0; i < kWidth; ++i) acc += w[i] * residual[i];
        out[row] = acc * norm;
    }
}

kernel void mhc_dot_simdgroup_per_row_wave8(
    device const float *weights [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    constant float &norm [[buffer(2)]],
    device float *out [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    for (uint wave = 0; wave < 3; ++wave) {
        uint row = wave * 8 + simdgroup;
        device const float *w = weights + row * kWidth;
        float partial = 0.0f;
        for (uint i = lane; i < kWidth; i += 32) partial += w[i] * residual[i];
        float total = simd_sum(partial);
        if (lane == 0) out[row] = total * norm;
    }
}

kernel void mhc_dot_eight_lane_per_row(
    device const float *weights [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    constant float &norm [[buffer(2)]],
    device float *out [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float partials[24 * 8];
    if (lid < 24 * 8) {
        uint row = lid / 8;
        uint part = lid % 8;
        device const float *w = weights + row * kWidth;
        float acc = 0.0f;
        for (uint i = part; i < kWidth; i += 8) acc += w[i] * residual[i];
        partials[lid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 24) {
        float total = 0.0f;
        for (uint part = 0; part < 8; ++part) total += partials[lid * 8 + part];
        out[lid] = total * norm;
    }
}
