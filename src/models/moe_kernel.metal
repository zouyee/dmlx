// DeepSeek V4 MoE Metal kernels — MXFP4 format: group_size=32, uint8 E8M0 scales, no biases.
// Formula (verified against MLX dequantize): w = NIBBLE_TO_FLOAT[nibble] * exp2(scale - 128)
// gate/up: [2048, 4096], down: [4096, 2048]
#include <metal_stdlib>
using namespace metal;

constant float NIBBLE_TO_FLOAT[16] = {
     0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  6.0f,  8.0f, 12.0f,
    -0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
};

// fused_gate_up_swiglu: gate+up dequant matvec + SwiGLU. SIMD reduction across 256 threads.
kernel void fused_gate_up_swiglu(
    device const uint32_t* gate_W   [[buffer(0)]],
    device const uint8_t*  gate_s   [[buffer(1)]],
    device const uint32_t* up_W     [[buffer(2)]],
    device const uint8_t*  up_s     [[buffer(3)]],
    device const float*    x        [[buffer(4)]],
    device float*          out      [[buffer(5)]],
    constant uint&         out_dim  [[buffer(6)]],
    constant uint&         in_dim   [[buffer(7)]],
    constant uint&         group_size [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* gr = gate_W + tgid * packed_cols;
    device const uint8_t*  gs = gate_s + tgid * num_groups;
    device const uint32_t* ur = up_W   + tgid * packed_cols;
    device const uint8_t*  us = up_s   + tgid * num_groups;

    float ga = 0.0f, ua = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float gsf = exp2((float)gs[g] - 128.0f);
        float usf = exp2((float)us[g] - 128.0f);
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gp = gr[bp + p], up = ur[bp + p];
            for (uint i = 0; i < 8; i++) {
                float xv = x[bx + p * 8 + i];
                ga += NIBBLE_TO_FLOAT[(gp >> (i * 4)) & 0xF] * gsf * xv;
                ua += NIBBLE_TO_FLOAT[(up >> (i * 4)) & 0xF] * usf * xv;
            }
        }
    }

    threadgroup float sg[32], su[32];
    float rg = simd_sum(ga), ru = simd_sum(ua);
    uint sl = lid % 32, si = lid / 32, ns = (tg_size + 31) / 32;
    if (sl == 0) { sg[si] = rg; su[si] = ru; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (si == 0 && sl < ns) {
        float vg = simd_sum(sg[sl]), vu = simd_sum(su[sl]);
        if (sl == 0) out[tgid] = (vg / (1.0f + exp(-vg))) * vu;
    }
}

// dequant_matvec_4bit: SIMD-optimized with threadgroup tiling (8 rows/TG, 256 threads).
kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed [[buffer(0)]],
    device const uint8_t*  scales   [[buffer(1)]],
    device const float*    x        [[buffer(2)]],
    device float*          out      [[buffer(3)]],
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         in_dim   [[buffer(5)]],
    constant uint&         group_size [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint ROWS = 8;
    uint start_row = tgid * ROWS;
    uint row = start_row + tid / 32;
    uint lane = tid % 32;
    if (row >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* wr = W_packed + row * packed_cols;
    device const uint8_t*  sc = scales   + row * num_groups;

    float acc = 0.0f;
    uint gi = lane;
    while (gi < num_groups) {
        float sf = exp2((float)sc[gi] - 128.0f);
        uint bp = gi * packed_per_group;
        uint bx = gi * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            float sx0 = sf * x[bx + p * 8 + 0], sx1 = sf * x[bx + p * 8 + 1];
            float sx2 = sf * x[bx + p * 8 + 2], sx3 = sf * x[bx + p * 8 + 3];
            float sx4 = sf * x[bx + p * 8 + 4], sx5 = sf * x[bx + p * 8 + 5];
            float sx6 = sf * x[bx + p * 8 + 6], sx7 = sf * x[bx + p * 8 + 7];
            acc += fma(NIBBLE_TO_FLOAT[(pw >>  0) & 0xF], sx0, 0.0f);
            acc += fma(NIBBLE_TO_FLOAT[(pw >>  4) & 0xF], sx1, 0.0f);
            acc += fma(NIBBLE_TO_FLOAT[(pw >>  8) & 0xF], sx2, 0.0f);
            acc += fma(NIBBLE_TO_FLOAT[(pw >> 12) & 0xF], sx3, 0.0f);
            acc += fma(NIBBLE_TO_FLOAT[(pw >> 16) & 0xF], sx4, 0.0f);
            acc += fma(NIBBLE_TO_FLOAT[(pw >> 20) & 0xF], sx5, 0.0f);
            acc += fma(NIBBLE_TO_FLOAT[(pw >> 24) & 0xF], sx6, 0.0f);
            acc += fma(NIBBLE_TO_FLOAT[(pw >> 28) & 0xF], sx7, 0.0f);
        }
        gi += 32;
    }
    float rg = simd_sum(acc);
    if (lane == 0) out[row] = rg;
}

// moe_combine: weighted sum of K expert outputs + residual.
kernel void moe_combine(
    device const float* expert_outs [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device const float* residual    [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant uint&      K           [[buffer(4)]],
    constant uint&      hidden_dim  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= hidden_dim) return;
    float sum = residual[tid];
    for (uint k = 0; k < K; k++) {
        sum += expert_outs[k * hidden_dim + tid] * weights[k];
    }
    output[tid] = sum;
}

// rms_norm_sum_sq: parallel reduction of sum(x_i^2)
kernel void rms_norm_sum_sq(
    device const float* x       [[buffer(0)]],
    device float*       sum_sq  [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];
    float acc = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) { float val = x[i]; acc += val * val; }
    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32, simd_group = lid / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) sum_sq[0] = val;
    }
}

// rms_norm_apply: out = x * rsqrt(sum_sq/dim + eps) * weight
kernel void rms_norm_apply(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* sum_sq  [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      dim     [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    out[tid] = x[tid] * rms * weight[tid];
}

// matvec_f32: out = W @ x. W: [out_dim, in_dim], x: [in_dim]
kernel void matvec_f32(
    device const float*    W      [[buffer(0)]],
    device const float*    x      [[buffer(1)]],
    device float*          out    [[buffer(2)]],
    constant uint&         out_dim [[buffer(3)]],
    constant uint&         in_dim  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    float acc = 0.0f;
    for (uint j = 0; j < in_dim; j++) acc += W[tid * in_dim + j] * x[j];
    out[tid] = acc;
}
