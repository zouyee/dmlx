// DeepSeek V4 MoE fused Metal kernel (adapted from flash-moe FMA pattern).
// Shapes: gate/up_proj = [2048, 4096], down_proj = [4096, 2048].
// 4-bit affine quantization: weight = scale * nibble + bias (uint8 scales, zero bias).

#include <metal_stdlib>
using namespace metal;

// Scales are uint8 (no bf16). No conversion needed.
// ============================================================================
// fused_gate_up_swiglu: gate_proj + up_proj dequant+matvec + SwiGLU
// One threadgroup per output row [out_dim=2048], 256 threads/group.
// ============================================================================
kernel void fused_gate_up_swiglu(
    device const uint32_t* gate_W   [[buffer(0)]], // [2048, 4096/8] packed uint32
    device const uint8_t*  gate_s   [[buffer(1)]], // [2048, num_groups] uint8 scales
    device const uint32_t* up_W     [[buffer(2)]], // [2048, 4096/8] packed uint32
    device const uint8_t*  up_s     [[buffer(3)]], // [2048, num_groups] uint8 scales
    device const float*    x        [[buffer(4)]], // [4096] hidden state
    device float*          out      [[buffer(5)]], // [2048] gate_out + SwiGLU
    constant uint&         out_dim  [[buffer(6)]], // 2048
    constant uint&         in_dim   [[buffer(7)]], // 4096
    constant uint&         group_size [[buffer(8)]], // 32 (4096/128)
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;
    uint num_groups = in_dim / group_size;        // 128
    uint packed_per_group = group_size / 8;        // 4 uint32 per group
    uint packed_cols = in_dim / 8;                 // 512 uint32 per row

    device const uint32_t* gr = gate_W + tgid * packed_cols;
    device const uint8_t*  gs = gate_s + tgid * num_groups;
    device const uint32_t* ur = up_W   + tgid * packed_cols;
    device const uint8_t*  us = up_s   + tgid * num_groups;

    float ga = 0.0f, ua = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float gsc = (float)gs[g];  // uint8 → f32
        float usc = (float)us[g];
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gp = gr[bp+p], up = ur[bp+p];
            for (uint i = 0; i < 8; i++) {
                float xv = x[bx + p*8 + i];
                ga += (float((gp>>(i*4))&0xF) * gsc) * xv;
                ua += (float((up>>(i*4))&0xF) * usc) * xv;
            }
        }
    }

    // Parallel reduction via simd_sum
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

// ============================================================================
// dequant_matvec_4bit: FMA-optimized 4-bit dequant matvec.
// Used for down_proj [out_dim=4096, in_dim=2048]. 256 threads/group, 8 rows/group.
// ============================================================================
kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed [[buffer(0)]], // [out_dim, in_dim/8]
    device const uint8_t*  scales   [[buffer(1)]], // [out_dim, num_groups]
    device const float*    x        [[buffer(2)]], // [in_dim]
    device float*          out      [[buffer(3)]], // [out_dim]
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         in_dim   [[buffer(5)]],
    constant uint&         group_size [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint ROWS = 8;  // 256/32 = 8 rows per threadgroup
    uint start_row = tgid * ROWS;
    uint row = start_row + tid / 32;  // SIMD group per row
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
        float s = (float)sc[gi];
        uint bp = gi * packed_per_group;
        uint bx = gi * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            float sx0 = s * x[bx + p*8 + 0], sx1 = s * x[bx + p*8 + 1];
            float sx2 = s * x[bx + p*8 + 2], sx3 = s * x[bx + p*8 + 3];
            float sx4 = s * x[bx + p*8 + 4], sx5 = s * x[bx + p*8 + 5];
            float sx6 = s * x[bx + p*8 + 6], sx7 = s * x[bx + p*8 + 7];
            // FMA: dequant+multiply in one instruction
            acc += fma(float((pw >>  0) & 0xF), sx0, 0.f);
            acc += fma(float((pw >>  4) & 0xF), sx1, 0.f);
            acc += fma(float((pw >>  8) & 0xF), sx2, 0.f);
            acc += fma(float((pw >> 12) & 0xF), sx3, 0.f);
            acc += fma(float((pw >> 16) & 0xF), sx4, 0.f);
            acc += fma(float((pw >> 20) & 0xF), sx5, 0.f);
            acc += fma(float((pw >> 24) & 0xF), sx6, 0.f);
            acc += fma(float((pw >> 28) & 0xF), sx7, 0.f);
        }
        gi += 32; // stride by SIMD width
    }
    // Reduce across SIMD group
    float rg = simd_sum(acc);
    if (lane == 0) out[row] = rg;
}

// ============================================================================
// moe_combine: weighted sum of K expert outputs + residual.
// One thread per output element [hidden_dim=4096].
// ============================================================================
kernel void moe_combine(
    device const float* expert_outs [[buffer(0)]], // [K, hidden_dim]
    device const float* weights     [[buffer(1)]], // [K] router scores
    device const float* residual    [[buffer(2)]], // [hidden_dim] input
    device float*       output      [[buffer(3)]], // [hidden_dim]
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
