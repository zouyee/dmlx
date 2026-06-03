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

// dequant_matvec_affine: out = W @ x, W affine-4bit quantized (gs=64, bf16 scales+biases).
// Matches MLX affine dequant: w = scale_g * nibble + bias_g  (nibble in [0,15], no LUT).
// W_packed: [out_dim, in_dim/8] uint32 (8 nibbles per uint32).
// scales/biases: [out_dim, in_dim/group_size] — passed as float (caller converts bf16->f32).
kernel void dequant_matvec_affine(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const float*    x        [[buffer(3)]],
    device float*          out      [[buffer(4)]],
    constant uint&         out_dim  [[buffer(5)]],
    constant uint&         in_dim   [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const float*    sc = scales   + tid * num_groups;
    device const float*    bi = biases   + tid * num_groups;

    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = sc[g];
        float bias  = bi[g];
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            uint x_base = bx + p * 8;
            for (uint i = 0; i < 8; i++) {
                float nib = (float)((pw >> (i * 4)) & 0xF);
                acc += (scale * nib + bias) * x[x_base + i];
            }
        }
    }
    out[tid] = acc;
}

// dequant_matvec_affine_bf16out: same as dequant_matvec_affine but output is bfloat.
// This matches MLX's behavior: affine matmul result is stored as bfloat16, which is
// the critical precision alignment needed for exact expert selection match.
kernel void dequant_matvec_affine_bf16out(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const float*    x        [[buffer(3)]],
    device bfloat*         out      [[buffer(4)]],
    constant uint&         out_dim  [[buffer(5)]],
    constant uint&         in_dim   [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const float*    sc = scales   + tid * num_groups;
    device const float*    bi = biases   + tid * num_groups;

    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = sc[g];
        float bias  = bi[g];
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            uint x_base = bx + p * 8;
            for (uint i = 0; i < 8; i++) {
                float nib = (float)((pw >> (i * 4)) & 0xF);
                acc += (scale * nib + bias) * x[x_base + i];
            }
        }
    }
    out[tid] = (bfloat)acc;
}

// rms_norm_rows_bf16out: rms_norm_rows but output is bfloat16 (matching MLX's bf16 intermediate).
// Used for Q chain (wq_a output, q_norm output, wq_b output) to match MLX bf16 attention.
kernel void rms_norm_rows_bf16out(
    device const float*  x          [[buffer(0)]],
    device const float*  weight     [[buffer(1)]],
    device bfloat*       out        [[buffer(2)]],
    constant uint&       row_dim    [[buffer(3)]],
    constant float&      eps        [[buffer(4)]],
    constant uint&       has_weight [[buffer(5)]],
    uint row     [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    float ss = 0.0f;
    for (uint i = lid; i < row_dim; i += tg_size) {
        float v = x[row * row_dim + i]; ss += v * v;
    }
    shared_sum[lid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared_sum[lid] += shared_sum[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_inv = rsqrt(shared_sum[0] / float(row_dim) + eps);
    for (uint i = lid; i < row_dim; i += tg_size) {
        float v = x[row * row_dim + i] * rms_inv;
        out[row * row_dim + i] = (bfloat)(has_weight ? v * weight[i] : v);
    }
}

// bf16_to_f32: convert a bfloat buffer to float (element-wise copy with widening).
kernel void bf16_to_f32(
    device const bfloat* src [[buffer(0)]],
    device float*        dst [[buffer(1)]],
    constant uint&       n   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    dst[tid] = float(src[tid]);
}


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

// ===========================================================================
// S2: MLA Q-chain kernels
// ===========================================================================

// rms_norm_rows: per-row RMSNorm over `row_dim`, for `n_rows` rows.
// out[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * (weight ? weight[:] : 1)
// One threadgroup per row; 256 threads cooperatively reduce.
// weight may be null (pass has_weight=0) for the weightless per-head norm.
kernel void rms_norm_rows(
    device const float* x        [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant uint&      row_dim  [[buffer(3)]],
    constant float&     eps      [[buffer(4)]],
    constant uint&      has_weight [[buffer(5)]],
    uint  row [[threadgroup_position_in_grid]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  tg  [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];
    device const float* xr = x + (uint64_t)row * row_dim;
    device float*       orow = out + (uint64_t)row * row_dim;

    float acc = 0.0f;
    for (uint i = lid; i < row_dim; i += tg) { float v = xr[i]; acc += v * v; }
    float s = simd_sum(acc);
    uint lane = lid % 32, sg = lid / 32;
    if (lane == 0) shared[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    uint n_sg = (tg + 31) / 32;
    if (lid < n_sg) total = shared[lid];
    total = simd_sum(total);
    // broadcast via shared[0]
    if (lid == 0) shared[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = rsqrt(shared[0] / float(row_dim) + eps);
    for (uint i = lid; i < row_dim; i += tg) {
        float w = (has_weight != 0u) ? weight[i] : 1.0f;
        orow[i] = xr[i] * rms * w;
    }
}

// rope_tail_interleaved: apply YaRN tail RoPE to the last rope_dim of each head.
// q: [n_heads, head_dim]; rotates dims [nope_dim .. head_dim) in interleaved
// pairs (2i, 2i+1) using precomputed cos/sin (length rope_dim/2) for this pos.
// Matches DSV4YarnRoPE (interleaved, NOT split-half). inverse negates sin.
kernel void rope_tail_interleaved(
    device float*       q        [[buffer(0)]],
    device const float* cos_t    [[buffer(1)]],
    device const float* sin_t    [[buffer(2)]],
    constant uint&      n_heads  [[buffer(3)]],
    constant uint&      head_dim [[buffer(4)]],
    constant uint&      nope_dim [[buffer(5)]],
    constant uint&      rope_dim [[buffer(6)]],
    constant uint&      inverse  [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint half_rope = rope_dim / 2;
    uint total = n_heads * half_rope;
    if (tid >= total) return;
    uint h = tid / half_rope;
    uint i = tid % half_rope;
    float c = cos_t[i];
    float s = sin_t[i];
    if (inverse != 0u) s = -s;
    device float* row = q + (uint64_t)h * head_dim;
    uint j0 = nope_dim + 2 * i;
    uint j1 = nope_dim + 2 * i + 1;
    float x0 = row[j0];
    float x1 = row[j1];
    row[j0] = x0 * c - x1 * s;
    row[j1] = x0 * s + x1 * c;
}

// ===========================================================================
// S4: MLA SDPA (decode, single query token) + attention sink
// ===========================================================================

// mla_sdpa_decode: one query token attends to `n_kv` cached KV rows (MQA: a
// single KV head shared by all N_HEADS query heads), with per-head attention
// sink folded into the softmax denominator (matches MLX fast SDPA sinks).
//
//   score_k = (q_h . kv[k]) * scale         for k in [0, n_kv)
//   denom   = sum_k exp(score_k - M) + exp(sink_h - M)   (M = running max)
//   out_h   = sum_k softmax_k * kv[k]
//
// One threadgroup per head; 256 threads cooperatively reduce over head_dim.
// q:   [n_heads, head_dim]
// kv:  [n_kv, head_dim]   (shared single KV head)
// out: [n_heads, head_dim]
// sinks: [n_heads]
kernel void mla_sdpa_decode(
    device const float* q        [[buffer(0)]],
    device const float* kv       [[buffer(1)]],
    device const float* sinks    [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant uint&      n_heads  [[buffer(4)]],
    constant uint&      head_dim [[buffer(5)]],
    constant uint&      n_kv     [[buffer(6)]],
    constant float&     scale    [[buffer(7)]],
    uint  head [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tg   [[threads_per_threadgroup]]
) {
    if (head >= n_heads) return;
    threadgroup float red[32];
    device const float* qh = q + (uint64_t)head * head_dim;

    // Online softmax accumulation over keys. Each thread owns a partial output
    // slice acc[d] for d = lid, lid+tg, ...  Running max M and denom S are
    // threadgroup-wide scalars kept in red[0]/red[1] via reduction per key.
    // For simplicity (correctness-first), recompute the dot product cooperatively.
    threadgroup float t_m;     // running max
    threadgroup float t_s;     // running denom
    threadgroup float t_score; // current key score
    if (lid == 0) { t_m = -INFINITY; t_s = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // per-thread output accumulator over its strided dims
    float acc[8];
    uint n_slots = (head_dim + tg - 1) / tg;
    for (uint i = 0; i < n_slots; i++) acc[i] = 0.0f;

    for (uint k = 0; k < n_kv; k++) {
        device const float* kvk = kv + (uint64_t)k * head_dim;
        // cooperative dot product q_h . kv[k]
        float partial = 0.0f;
        for (uint d = lid; d < head_dim; d += tg) partial += qh[d] * kvk[d];
        float dot = simd_sum(partial);
        uint lane = lid % 32, sg = lid / 32;
        if (lane == 0) red[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tot = 0.0f;
            uint n_sg = (tg + 31) / 32;
            for (uint g = 0; g < n_sg; g++) tot += red[g];
            t_score = tot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = t_score;
        float m_old = t_m;
        float m_new = max(m_old, score);
        float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
        float p = exp(score - m_new);
        // rescale acc and add p * kv[k]
        for (uint i = 0; i < n_slots; i++) {
            uint d = lid + i * tg;
            if (d < head_dim) acc[i] = acc[i] * corr + p * kvk[d];
        }
        if (lid == 0) { t_s = t_s * corr + p; t_m = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Fold per-head sink into the denominator (no output contribution).
    if (lid == 0) {
        float sink = sinks[head];
        float m_old = t_m;
        float m_new = max(m_old, sink);
        float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
        t_s = t_s * corr + exp(sink - m_new);
        t_m = m_new;
        red[0] = corr; // broadcast correction for acc rescale
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float corr_final = red[0];
    float inv_s = (t_s == 0.0f) ? 0.0f : 1.0f / t_s;
    device float* oh = out + (uint64_t)head * head_dim;
    for (uint i = 0; i < n_slots; i++) {
        uint d = lid + i * tg;
        if (d < head_dim) oh[d] = (acc[i] * corr_final) * inv_s;
    }
}
