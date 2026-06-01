// DeepSeek V4 MoE Metal kernels — MXFP4 format: group_size=32, uint8 E8M0 scales, no biases.
// Formula (verified against MLX dequantize): w = NIBBLE_TO_FLOAT[nibble] * exp2(scale - 128)
// gate/up: [2048, 4096], down: [4096, 2048]
#include <metal_stdlib>
using namespace metal;

constant float NIBBLE_TO_FLOAT[16] = {
     0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  6.0f,  8.0f, 12.0f,
    -0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
};

// fused_gate_up_swiglu NAIVE: one thread per row, no SIMD reduction. For debugging.
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
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* g_row = gate_W + tid * packed_cols;
    device const uint8_t*  g_s   = gate_s + tid * num_groups;
    device const uint32_t* u_row = up_W   + tid * packed_cols;
    device const uint8_t*  u_s   = up_s   + tid * num_groups;

    float gate_val = 0.0f, up_val = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float gsf = exp2((float)g_s[g] - 128.0f);
        float usf = exp2((float)u_s[g] - 128.0f);
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gpw = g_row[bp + p], upw = u_row[bp + p];
            uint x_base = bx + p * 8;
            for (uint i = 0; i < 8; i++) {
                float g_w = NIBBLE_TO_FLOAT[(gpw >> (i * 4)) & 0xF] * gsf;
                float u_w = NIBBLE_TO_FLOAT[(upw >> (i * 4)) & 0xF] * usf;
                float xv = x[x_base + i];
                gate_val += g_w * xv;
                up_val   += u_w * xv;
            }
        }
    }
    // Limited SwiGLU (matches MLX DSV4SwitchGLU.limitedSwiGLU, swiglu_limit=10):
    //   gate_clipped = min(gate, +limit)
    //   up_clipped   = min(max(up, -limit), +limit)
    //   out = silu(gate_clipped) * up_clipped
    const float limit = 10.0f;
    float g_c = min(gate_val, limit);
    float u_c = min(max(up_val, -limit), limit);
    float act = g_c / (1.0f + exp(-g_c));
    out[tid] = act * u_c;
}

// dequant_matvec_4bit NAIVE: one thread per row. For correctness baseline.
kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed [[buffer(0)]],
    device const uint8_t*  scales   [[buffer(1)]],
    device const float*    x        [[buffer(2)]],
    device float*          out      [[buffer(3)]],
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         in_dim   [[buffer(5)]],
    constant uint&         group_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const uint8_t*  sc = scales   + tid * num_groups;

    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float sf = exp2((float)sc[g] - 128.0f);
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            uint x_base = bx + p * 8;
            for (uint i = 0; i < 8; i++) {
                float w_val = NIBBLE_TO_FLOAT[(pw >> (i * 4)) & 0xF] * sf;
                acc += w_val * x[x_base + i];
            }
        }
    }
    out[tid] = acc;
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
