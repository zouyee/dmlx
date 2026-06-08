
// ============================================================================
// F16-precision kernels (ds4-style). Generated from bf16 variants above.
// ============================================================================
kernel void dequant_matvec_affine_f16out(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const float*    x        [[buffer(3)]],
    device half*         out      [[buffer(4)]],
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
    out[tid] = (half)acc;
}

kernel void rms_norm_rows_f16out(
    device const float*  x          [[buffer(0)]],
    device const float*  weight     [[buffer(1)]],
    device half*       out        [[buffer(2)]],
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
        out[row * row_dim + i] = (half)(has_weight ? v * weight[i] : v);
    }
}

kernel void dequant_matvec_affine_f16in_f16out(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const half*   x        [[buffer(3)]],  // half input
    device half*         out      [[buffer(4)]],  // half output
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
        float scale = sc[g], bias = bi[g];
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            for (uint i = 0; i < 8; i++) {
                float nib = (float)((pw >> (i * 4)) & 0xF);
                acc += (scale * nib + bias) * float(x[bx + p * 8 + i]);
            }
        }
    }
    out[tid] = (half)acc;
}

kernel void rms_norm_rows_f16in_f16out(
    device const half* x          [[buffer(0)]],
    device const float*  weight     [[buffer(1)]],
    device half*       out        [[buffer(2)]],
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
        float v = float(x[row * row_dim + i]); ss += v * v;
    }
    shared_sum[lid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared_sum[lid] += shared_sum[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_inv = rsqrt(shared_sum[0] / float(row_dim) + eps);
    for (uint i = lid; i < row_dim; i += tg_size) {
        float v = float(x[row * row_dim + i]) * rms_inv;
        out[row * row_dim + i] = (half)(has_weight ? v * weight[i] : v);
    }
}

kernel void rope_tail_interleaved_f16(
    device half*         q          [[buffer(0)]],  // in-place [n_heads, head_dim]
    device const float*    cos_vals   [[buffer(1)]],  // [half_rope]
    device const float*    sin_vals   [[buffer(2)]],  // [half_rope]
    constant uint&         n_heads    [[buffer(3)]],
    constant uint&         head_dim   [[buffer(4)]],
    constant uint&         n_nope     [[buffer(5)]],
    constant uint&         n_rope     [[buffer(6)]],
    constant uint&         inverse    [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint half_rope = n_rope / 2;
    if (tid >= n_heads * half_rope) return;
    uint head = tid / half_rope, ic = tid % half_rope;
    uint j0 = head * head_dim + n_nope + 2 * ic;
    uint j1 = j0 + 1;
    float cos_v = cos_vals[ic];
    float sin_v = inverse ? -sin_vals[ic] : sin_vals[ic];
    float x0 = float(q[j0]);
    float x1 = float(q[j1]);
    q[j0] = (half)(x0 * cos_v - x1 * sin_v);
    q[j1] = (half)(x0 * sin_v + x1 * cos_v);
}

kernel void matvec_f32_f16in(
    device const float*  W   [[buffer(0)]],
    device const half* x   [[buffer(1)]],
    device float*        out [[buffer(2)]],
    constant uint&       out_dim [[buffer(3)]],
    constant uint&       in_dim  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    float acc = 0.0f;
    for (uint i = 0; i < in_dim; i++) acc += W[tid * in_dim + i] * float(x[i]);
    out[tid] = acc;
}

kernel void mhc_pre_gpu_f16(
    device const float* fn_weight [[buffer(0)]],  // [24, 16384]
    device const float* base      [[buffer(1)]],  // [24]
    device const float* scale_v   [[buffer(2)]],  // [3]
    device const float* residual  [[buffer(3)]],  // [4, 4096]
    device float*       out_input [[buffer(4)]],  // [4096] half-truncated f32
    device float*       out_post  [[buffer(5)]],  // [4]
    device float*       out_comb  [[buffer(6)]],  // [16]
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint HC = 4, DIM = 4096, MHC_H = HC * DIM, MIX3 = 24;
    const float EPS = 1e-6f, POST_MULT = 2.0f;

    // Threadgroup scratch: mixes[24], pre_mix[4], comb[16]
    threadgroup float mixes[24];
    threadgroup float pre_mix[4];
    threadgroup float comb_mat[16];  // [4][4]
    threadgroup float sum_sq;
    threadgroup float rms_norm_factor;

    // Step 1: compute sum(residual^2) / MHC_H
    float local_ss = 0.0f;
    for (uint i = lid; i < MHC_H; i += tg_size) local_ss += residual[i] * residual[i];
    threadgroup float ss_buf[256];
    ss_buf[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) ss_buf[lid] += ss_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        sum_sq = ss_buf[0];
        rms_norm_factor = rsqrt(sum_sq / float(MHC_H) + EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm = rms_norm_factor;

    // Step 2: compute mixes[r] = (fn[r,:] @ residual) * norm, for r in 0..23
    for (uint r = lid; r < MIX3; r += tg_size) {
        device const float* fn_r = fn_weight + r * MHC_H;
        float acc = 0.0f;
        for (uint i = 0; i < MHC_H; i++) acc += fn_r[i] * residual[i];
        mixes[r] = acc * norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: compute pre_mix, post, comb from mixes
    if (lid == 0) {
        float s0 = scale_v[0], s1 = scale_v[1], s2 = scale_v[2];
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[m] * s0 + base[m];
            pre_mix[m] = 1.0f / (1.0f + exp(-biased)) + EPS;
        }
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[HC + m] * s1 + base[HC + m];
            out_post[m] = (1.0f / (1.0f + exp(-biased))) * POST_MULT;
        }
        for (uint c = 0; c < HC * HC; c++) {
            comb_mat[c] = mixes[2 * HC + c] * s2 + base[2 * HC + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Sinkhorn normalization on comb_mat[4][4]
    // (Single-threaded, tiny compute — 4x4 matrix, 20 iterations)
    if (lid == 0) {
        // Initial softmax per row + eps
        for (uint i = 0; i < HC; i++) {
            float m_val = comb_mat[i * HC];
            for (uint j = 1; j < HC; j++) if (comb_mat[i*HC+j] > m_val) m_val = comb_mat[i*HC+j];
            float s_val = 0.0f;
            for (uint j = 0; j < HC; j++) { comb_mat[i*HC+j] = exp(comb_mat[i*HC+j] - m_val); s_val += comb_mat[i*HC+j]; }
            for (uint j = 0; j < HC; j++) comb_mat[i*HC+j] = comb_mat[i*HC+j] / s_val + EPS;
        }
        // Initial col-norm
        for (uint j = 0; j < HC; j++) {
            float cs = 0.0f; for (uint i = 0; i < HC; i++) cs += comb_mat[i*HC+j];
            cs += EPS; for (uint i = 0; i < HC; i++) comb_mat[i*HC+j] /= cs;
        }
        // 19 more row/col-norm iterations
        for (uint it = 0; it < 19; it++) {
            for (uint i = 0; i < HC; i++) {
                float rs = 0.0f; for (uint j = 0; j < HC; j++) rs += comb_mat[i*HC+j];
                rs += EPS; for (uint j = 0; j < HC; j++) comb_mat[i*HC+j] /= rs;
            }
            for (uint j = 0; j < HC; j++) {
                float cs = 0.0f; for (uint i = 0; i < HC; i++) cs += comb_mat[i*HC+j];
                cs += EPS; for (uint i = 0; i < HC; i++) comb_mat[i*HC+j] /= cs;
            }
        }
        // Write sinkhorn result to out_comb
        for (uint c = 0; c < HC * HC; c++) out_comb[c] = comb_mat[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: out_input[d] = sum_m pre_mix[m] * residual[m,d], truncated to half
    for (uint d = lid; d < DIM; d += tg_size) {
        float acc = 0.0f;
        for (uint m = 0; m < HC; m++) acc += pre_mix[m] * residual[m * DIM + d];
        // Truncate to half and back to f32 (matches MLX's .astype(x.dtype) = half16)
        out_input[d] = float((half)acc);
    }
}

kernel void mhc_pre_f16(
    device const float*  fn_weight [[buffer(0)]],  // [24, 16384] f32
    device const float*  base      [[buffer(1)]],  // [24] f32
    device const float*  scale_v   [[buffer(2)]],  // [3] f32
    device const half* residual  [[buffer(3)]],  // [HC, DIM] half
    device half*       out_input [[buffer(4)]],  // [DIM] half
    device float*        out_post  [[buffer(5)]],  // [HC] f32
    device float*        out_comb  [[buffer(6)]],  // [HC*HC] f32
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint HC = 4, DIM = 4096, MHC_H = HC * DIM, MIX3 = 24;
    const float EPS = 1e-6f, POST_MULT = 2.0f;

    threadgroup float mixes[24];
    threadgroup float pre_mix[4];
    threadgroup float comb_mat[16];
    threadgroup float ss_buf[256];

    // Step 1: compute mean(residual^2) using half values (cast to float for accumulation)
    float local_ss = 0.0f;
    for (uint i = lid; i < MHC_H; i += tg_size) {
        float v = float(residual[i]); local_ss += v * v;
    }
    ss_buf[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) ss_buf[lid] += ss_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = rsqrt(ss_buf[0] / float(MHC_H) + EPS);

    // Step 2: mixes = (fn @ float(residual)) * norm
    for (uint r = lid; r < MIX3; r += tg_size) {
        device const float* fn_r = fn_weight + r * MHC_H;
        float acc = 0.0f;
        for (uint i = 0; i < MHC_H; i++) acc += fn_r[i] * float(residual[i]);
        mixes[r] = acc * norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: pre_mix, post, comb (single-threaded, same as mhc_pre_gpu)
    if (lid == 0) {
        float s0 = scale_v[0], s1 = scale_v[1], s2 = scale_v[2];
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[m] * s0 + base[m];
            pre_mix[m] = 1.0f / (1.0f + exp(-biased)) + EPS;
        }
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[HC + m] * s1 + base[HC + m];
            out_post[m] = (1.0f / (1.0f + exp(-biased))) * POST_MULT;
        }
        for (uint c = 0; c < HC * HC; c++)
            comb_mat[c] = mixes[2 * HC + c] * s2 + base[2 * HC + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Sinkhorn (single-threaded)
    if (lid == 0) {
        for (uint i = 0; i < HC; i++) {
            float m_val = comb_mat[i*HC]; for (uint j=1;j<HC;j++) if (comb_mat[i*HC+j]>m_val) m_val=comb_mat[i*HC+j];
            float s_val = 0.0f; for (uint j=0;j<HC;j++) { comb_mat[i*HC+j]=exp(comb_mat[i*HC+j]-m_val); s_val+=comb_mat[i*HC+j]; }
            for (uint j=0;j<HC;j++) comb_mat[i*HC+j]=comb_mat[i*HC+j]/s_val+EPS;
        }
        for (uint j=0;j<HC;j++) { float cs=0.0f; for (uint i=0;i<HC;i++) cs+=comb_mat[i*HC+j]; cs+=EPS; for (uint i=0;i<HC;i++) comb_mat[i*HC+j]/=cs; }
        for (uint it=0;it<19;it++) {
            for (uint i=0;i<HC;i++) { float rs=0.0f; for(uint j=0;j<HC;j++) rs+=comb_mat[i*HC+j]; rs+=EPS; for(uint j=0;j<HC;j++) comb_mat[i*HC+j]/=rs; }
            for (uint j=0;j<HC;j++) { float cs=0.0f; for(uint i=0;i<HC;i++) cs+=comb_mat[i*HC+j]; cs+=EPS; for(uint i=0;i<HC;i++) comb_mat[i*HC+j]/=cs; }
        }
        for (uint c=0;c<HC*HC;c++) out_comb[c]=comb_mat[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: out_input[d] = half( sum_m pre_mix[m] * float(residual[m,d]) )
    for (uint d = lid; d < DIM; d += tg_size) {
        float acc = 0.0f;
        for (uint m = 0; m < HC; m++) acc += pre_mix[m] * float(residual[m * DIM + d]);
        out_input[d] = (half)acc;
    }
}

kernel void mhc_post_f16(
    device const half* x            [[buffer(0)]],  // [DIM] half
    device const half* residual     [[buffer(1)]],  // [HC, DIM] half
    device const float*  post         [[buffer(2)]],  // [HC] f32
    device const float*  comb         [[buffer(3)]],  // [HC*HC] f32 (row-major [k][m])
    device half*       out_residual [[buffer(4)]],  // [HC, DIM] half
    constant uint&       hc           [[buffer(5)]],
    constant uint&       dim          [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    uint d = tid;
    float xv = float(x[d]);
    for (uint m = 0; m < hc; m++) {
        float acc = post[m] * xv;
        for (uint k = 0; k < hc; k++) {
            // comb[k][m] = comb[k*hc + m]
            acc += comb[k * hc + m] * float(residual[k * dim + d]);
        }
        out_residual[m * dim + d] = (half)acc;
    }
}

kernel void mla_sdpa_decode_f16in_f16out(
    device const half* q        [[buffer(0)]],
    device const half* kv       [[buffer(1)]],
    device const float*  sinks    [[buffer(2)]],
    device half*       out      [[buffer(3)]],
    constant uint&       n_heads  [[buffer(4)]],
    constant uint&       head_dim [[buffer(5)]],
    constant uint&       n_kv     [[buffer(6)]],
    constant float&      scale    [[buffer(7)]],
    uint  head [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tg   [[threads_per_threadgroup]]
) {
    if (head >= n_heads) return;
    threadgroup float red[32];
    device const half* qh = q + (uint64_t)head * head_dim;

    threadgroup float t_m;
    threadgroup float t_s;
    threadgroup float t_score;
    if (lid == 0) { t_m = -INFINITY; t_s = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[8];
    uint n_slots = (head_dim + tg - 1) / tg;
    for (uint i = 0; i < n_slots; i++) acc[i] = 0.0f;

    for (uint k = 0; k < n_kv; k++) {
        device const half* kvk = kv + (uint64_t)k * head_dim;
        float partial = 0.0f;
        for (uint d = lid; d < head_dim; d += tg) partial += float(qh[d]) * float(kvk[d]);
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
        for (uint i = 0; i < n_slots; i++) {
            uint d = lid + i * tg;
            if (d < head_dim) acc[i] = acc[i] * corr + p * float(kvk[d]);
        }
        if (lid == 0) { t_s = t_s * corr + p; t_m = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Sink
    if (lid == 0) {
        float sink = sinks[head];
        float m_old = t_m;
        float m_new = max(m_old, sink);
        float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
        t_s = t_s * corr + exp(sink - m_new);
        t_m = m_new;
        red[0] = corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float corr_final = red[0];
    float inv_s = (t_s == 0.0f) ? 0.0f : 1.0f / t_s;
    device half* oh = out + (uint64_t)head * head_dim;
    for (uint i = 0; i < n_slots; i++) {
        uint d = lid + i * tg;
        if (d < head_dim) oh[d] = (half)((acc[i] * corr_final) * inv_s);
    }
}

