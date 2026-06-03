// MLA attention host orchestration — full bfloat16 data flow.
// All Q/KV chain intermediates are bfloat16, matching MLX's bf16 computation.
// SDPA uses f32 accumulation internally; battn output is truncated to bfloat16.
// attn_out returned as float (bfloat16 bit-pattern in a float shell).
#import "mla_attention.h"
#import <Foundation/Foundation.h>
#import <math.h>
#import <stdlib.h>
#import <string.h>

// --- YaRN tail RoPE cos/sin ---
static void yarn_cos_sin(int pos, float *cos_out, float *sin_out) {
    const float base = 10000.0f, factor = 16.0f, beta_fast = 32.0f, beta_slow = 1.0f;
    const float orig_max = 65536.0f, PI = 3.14159265358979323846f;
    int half = QK_ROPE_DIM / 2;
    float cd_fast = QK_ROPE_DIM * logf(orig_max / (beta_fast * 2.0f * PI)) / (2.0f * logf(base));
    float cd_slow = QK_ROPE_DIM * logf(orig_max / (beta_slow * 2.0f * PI)) / (2.0f * logf(base));
    int low = (int)fmaxf(0.0f, floorf(cd_fast));
    int high = (int)fminf((float)(QK_ROPE_DIM - 1), ceilf(cd_slow));
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(base, (2.0f * i) / QK_ROPE_DIM);
        float ramp;
        if (low == high) ramp = (i <= low) ? 0.0f : 1.0f;
        else ramp = fminf(1.0f, fmaxf(0.0f, ((float)i - low) / (float)(high - low)));
        float smooth = 1.0f - ramp;
        freq = freq / factor * (1.0f - smooth) + freq * smooth;
        cos_out[i] = cosf(pos * freq);
        sin_out[i] = sinf(pos * freq);
    }
}

static id<MTLBuffer> mkbuf(id<MTLDevice> d, const void *p, size_t n) {
    return p ? [d newBufferWithBytes:p length:n options:MTLResourceStorageModeShared]
             : [d newBufferWithLength:n options:MTLResourceStorageModeShared];
}

// ---- dispatch helpers (all bfloat16 in/out where indicated) ----

// dequant_affine: bfloat in → bfloat out
static void enc_dq_bf16_bf16(MlaPipes *P, id<MTLCommandBuffer> cb,
                              const QuantWeight *qw, id<MTLBuffer> x, id<MTLBuffer> out) {
    id<MTLDevice> d = P->dev;
    int ng = qw->in_dim / qw->group_size;
    id<MTLBuffer> bw = mkbuf(d, qw->packed, (size_t)qw->out_dim*(qw->in_dim/8)*sizeof(uint32_t));
    id<MTLBuffer> bs = mkbuf(d, qw->scales,  (size_t)qw->out_dim*ng*sizeof(float));
    id<MTLBuffer> bb = mkbuf(d, qw->biases,  (size_t)qw->out_dim*ng*sizeof(float));
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->dequant_matvec_affine_bf16in_bf16out];
    [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bs offset:0 atIndex:1];
    [e setBuffer:bb offset:0 atIndex:2]; [e setBuffer:x  offset:0 atIndex:3];
    [e setBuffer:out offset:0 atIndex:4];
    uint od=qw->out_dim, id_=qw->in_dim, gs=qw->group_size;
    [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6];
    [e setBytes:&gs length:4 atIndex:7];
    [e dispatchThreads:MTLSizeMake(qw->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
}

// rms_norm: bfloat in → bfloat out
static void enc_rms_bf16_bf16(MlaPipes *P, id<MTLCommandBuffer> cb,
                               id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> out,
                               uint n_rows, uint row_dim, int has_w) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rms_norm_rows_bf16in_bf16out];
    [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:(w ? w : x) offset:0 atIndex:1];
    [e setBuffer:out offset:0 atIndex:2];
    uint rd=row_dim; float eps=1e-6f; uint hw=has_w?1u:0u;
    [e setBytes:&rd length:4 atIndex:3]; [e setBytes:&eps length:4 atIndex:4];
    [e setBytes:&hw length:4 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(n_rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
}

// bfloat RoPE (in-place)
static void enc_rope_bf16(MlaPipes *P, id<MTLCommandBuffer> cb, id<MTLBuffer> q,
                           id<MTLBuffer> cosb, id<MTLBuffer> sinb, uint n_heads, uint inverse) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rope_tail_bf16];
    [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:cosb offset:0 atIndex:1];
    [e setBuffer:sinb offset:0 atIndex:2];
    uint nh=n_heads, hd=HEAD_DIM, nd=QK_NOPE_DIM, rd=QK_ROPE_DIM, inv=inverse;
    [e setBytes:&nh length:4 atIndex:3]; [e setBytes:&hd length:4 atIndex:4];
    [e setBytes:&nd length:4 atIndex:5]; [e setBytes:&rd length:4 atIndex:6];
    [e setBytes:&inv length:4 atIndex:7];
    uint half=QK_ROPE_DIM/2;
    [e dispatchThreads:MTLSizeMake(n_heads*half,1,1) threadsPerThreadgroup:MTLSizeMake(half,1,1)];
    [e endEncoding];
}

// bfloat → float widening
static void enc_bf16_to_f32(MlaPipes *P, id<MTLCommandBuffer> cb,
                              id<MTLBuffer> src, id<MTLBuffer> dst, uint n) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->bf16_to_f32];
    [e setBuffer:src offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:1];
    [e setBytes:&n length:4 atIndex:2];
    [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
}

// float → bfloat truncation
static void enc_f32_to_bf16(MlaPipes *P, id<MTLCommandBuffer> cb,
                              id<MTLBuffer> src, id<MTLBuffer> dst, uint n) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->f32_to_bf16];
    [e setBuffer:src offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:1];
    [e setBytes:&n length:4 atIndex:2];
    [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
}

// f32 RoPE (in-place, used for battn inverse RoPE after SDPA)
static void enc_rope_f32(MlaPipes *P, id<MTLCommandBuffer> cb, id<MTLBuffer> q,
                          id<MTLBuffer> cosb, id<MTLBuffer> sinb, uint n_heads, uint inverse) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rope_tail_interleaved];
    [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:cosb offset:0 atIndex:1];
    [e setBuffer:sinb offset:0 atIndex:2];
    uint nh=n_heads, hd=HEAD_DIM, nd=QK_NOPE_DIM, rd=QK_ROPE_DIM, inv=inverse;
    [e setBytes:&nh length:4 atIndex:3]; [e setBytes:&hd length:4 atIndex:4];
    [e setBytes:&nd length:4 atIndex:5]; [e setBytes:&rd length:4 atIndex:6];
    [e setBytes:&inv length:4 atIndex:7];
    uint half=QK_ROPE_DIM/2;
    [e dispatchThreads:MTLSizeMake(n_heads*half,1,1) threadsPerThreadgroup:MTLSizeMake(half,1,1)];
    [e endEncoding];
}

// dequant_affine: f32 in → f32 out (used for KV chain to keep f32 for KV cache)
static void enc_dq_f32_f32(MlaPipes *P, id<MTLCommandBuffer> cb,
                            const QuantWeight *qw, id<MTLBuffer> x, id<MTLBuffer> out) {
    id<MTLDevice> d = P->dev;
    int ng = qw->in_dim / qw->group_size;
    id<MTLBuffer> bw = mkbuf(d, qw->packed, (size_t)qw->out_dim*(qw->in_dim/8)*sizeof(uint32_t));
    id<MTLBuffer> bs = mkbuf(d, qw->scales,  (size_t)qw->out_dim*ng*sizeof(float));
    id<MTLBuffer> bb = mkbuf(d, qw->biases,  (size_t)qw->out_dim*ng*sizeof(float));
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->dequant_matvec_affine];
    [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bs offset:0 atIndex:1];
    [e setBuffer:bb offset:0 atIndex:2]; [e setBuffer:x  offset:0 atIndex:3];
    [e setBuffer:out offset:0 atIndex:4];
    uint od=qw->out_dim, id_=qw->in_dim, gs=qw->group_size;
    [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6];
    [e setBytes:&gs length:4 atIndex:7];
    [e dispatchThreads:MTLSizeMake(qw->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
}

// rms_norm: f32 in → f32 out
static void enc_rms_f32_f32(MlaPipes *P, id<MTLCommandBuffer> cb,
                              id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> out,
                              uint n_rows, uint row_dim, int has_w) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rms_norm_rows];
    [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:(w ? w : x) offset:0 atIndex:1];
    [e setBuffer:out offset:0 atIndex:2];
    uint rd=row_dim; float eps=1e-6f; uint hw=has_w?1u:0u;
    [e setBytes:&rd length:4 atIndex:3]; [e setBytes:&eps length:4 atIndex:4];
    [e setBytes:&hw length:4 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(n_rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
}

// ---- main decode function ----
// x: [DIM] bfloat16 (attn_input, from mhc_pre_bfloat)
// out: [DIM] float (bfloat16 values in float32 shell, for mhc_post_bfloat)
int mla_attention_decode(MlaPipes *P, const AttnWeights *aw,
                         const float *x_f32_shell, float *kv_cache, int cache_len,
                         int pos, float *out_f32_shell) {
    id<MTLDevice> d = P->dev;
    int half_rope = QK_ROPE_DIM / 2;
    float cosv[QK_ROPE_DIM / 2], sinv[QK_ROPE_DIM / 2];
    yarn_cos_sin(pos, cosv, sinv);

    // x is bfloat16 values stored in float32 memory (bfloat16 bit-pattern in upper 16 bits)
    // Extract actual bfloat16 (uint16_t) values: upper 16 bits of each float32
    uint16_t x_bf16[DIM];
    for (int i = 0; i < DIM; i++) {
        uint32_t bits; memcpy(&bits, &x_f32_shell[i], 4);
        x_bf16[i] = (uint16_t)(bits >> 16);
    }
    id<MTLBuffer> bx_bf16 = mkbuf(d, x_bf16, DIM * sizeof(uint16_t));
    id<MTLBuffer> bcos    = mkbuf(d, cosv, half_rope * sizeof(float));
    id<MTLBuffer> bsin    = mkbuf(d, sinv, half_rope * sizeof(float));
    id<MTLBuffer> bqn_w   = mkbuf(d, aw->q_norm,  Q_LORA_RANK  * sizeof(float));
    id<MTLBuffer> bkvn_w  = mkbuf(d, aw->kv_norm, KV_LORA_RANK * sizeof(float));

    // --- Q chain: bfloat16 throughout ---
    // wq_a: [Q_LORA_RANK=1024, DIM=4096], bf16_in → bf16_out
    id<MTLBuffer> bq_a_b16   = mkbuf(d, NULL, Q_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bq_res_b16 = mkbuf(d, NULL, Q_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bq_b16     = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    id<MTLBuffer> bq_n_b16   = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    // Convert to f32 only for SDPA
    id<MTLBuffer> bq_n_f32   = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(float));

    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dq_bf16_bf16(P, cb, &aw->wq_a, bx_bf16, bq_a_b16);
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_bf16_bf16(P, cb, bq_a_b16, bqn_w, bq_res_b16, 1, Q_LORA_RANK, 1); // q_norm
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dq_bf16_bf16(P, cb, &aw->wq_b, bq_res_b16, bq_b16);
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_bf16_bf16(P, cb, bq_b16, NULL, bq_n_b16, N_HEADS, HEAD_DIM, 0);   // per-head norm
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope_bf16(P, cb, bq_n_b16, bcos, bsin, N_HEADS, 0);                   // tail RoPE
      [cb commit]; [cb waitUntilCompleted]; }
    // widen bfloat → float for SDPA (SDPA kernel uses f32)
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_bf16_to_f32(P, cb, bq_n_b16, bq_n_f32, (uint)N_HEADS * HEAD_DIM);
      [cb commit]; [cb waitUntilCompleted]; }

    // --- KV chain: f32 (KV cache is f32) ---
    // wkv uses bfloat input but f32 intermediate — use the bfloat→f32 input variant
    id<MTLBuffer> bx_f32 = mkbuf(d, NULL, DIM * sizeof(float));
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_bf16_to_f32(P, cb, bx_bf16, bx_f32, DIM);
      [cb commit]; [cb waitUntilCompleted]; }
    id<MTLBuffer> bkv    = mkbuf(d, NULL, KV_LORA_RANK * sizeof(float));
    id<MTLBuffer> bkv_n  = mkbuf(d, NULL, KV_LORA_RANK * sizeof(float));
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dq_f32_f32(P, cb, &aw->wkv, bx_f32, bkv);
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_f32_f32(P, cb, bkv, bkvn_w, bkv_n, 1, KV_LORA_RANK, 1);
      [cb commit]; [cb waitUntilCompleted]; }
    id<MTLBuffer> bcos_kv = mkbuf(d, cosv, half_rope * sizeof(float));
    id<MTLBuffer> bsin_kv = mkbuf(d, sinv, half_rope * sizeof(float));
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope_f32(P, cb, bkv_n, bcos_kv, bsin_kv, 1, 0);
      [cb commit]; [cb waitUntilCompleted]; }

    // Write KV into cache
    {
        float *kvn = (float *)[bkv_n contents];
        memcpy(kv_cache + (size_t)(cache_len - 1) * KV_LORA_RANK, kvn, KV_LORA_RANK * sizeof(float));
    }

    // --- SDPA (f32) ---
    id<MTLBuffer> bkvcache = mkbuf(d, kv_cache, (size_t)cache_len * KV_LORA_RANK * sizeof(float));
    id<MTLBuffer> bsink    = mkbuf(d, aw->attn_sink, N_HEADS * sizeof(float));
    id<MTLBuffer> battn_f32= mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(float));
    {
        id<MTLCommandBuffer> cb=[P->queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:P->mla_sdpa_decode];
        [e setBuffer:bq_n_f32 offset:0 atIndex:0];
        [e setBuffer:bkvcache  offset:0 atIndex:1];
        [e setBuffer:bsink     offset:0 atIndex:2];
        [e setBuffer:battn_f32 offset:0 atIndex:3];
        uint nh=N_HEADS, hd=HEAD_DIM, nk=cache_len;
        float scale=1.0f/sqrtf((float)HEAD_DIM);
        [e setBytes:&nh    length:4 atIndex:4];
        [e setBytes:&hd    length:4 atIndex:5];
        [e setBytes:&nk    length:4 atIndex:6];
        [e setBytes:&scale length:4 atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(N_HEADS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }
    // Inverse RoPE on f32 battn
    id<MTLBuffer> bcos_inv = mkbuf(d, cosv, half_rope * sizeof(float));
    id<MTLBuffer> bsin_inv = mkbuf(d, sinv, half_rope * sizeof(float));
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope_f32(P, cb, battn_f32, bcos_inv, bsin_inv, N_HEADS, 1);
      [cb commit]; [cb waitUntilCompleted]; }

    // --- grouped wo_a (dense f32) → concat → wo_b ---
    float *attn_ptr = (float *)[battn_f32 contents];
    int heads_per_group = N_HEADS / O_GROUPS;  // 8
    int group_feat = heads_per_group * HEAD_DIM; // 4096
    float *concat = malloc((size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    for (int g = 0; g < O_GROUPS; g++) {
        float *gv = malloc((size_t)group_feat * sizeof(float));
        for (int hh = 0; hh < heads_per_group; hh++)
            memcpy(gv + hh * HEAD_DIM,
                   attn_ptr + (g * heads_per_group + hh) * HEAD_DIM,
                   HEAD_DIM * sizeof(float));
        id<MTLBuffer> bgv = mkbuf(d, gv, group_feat * sizeof(float));
        id<MTLBuffer> bog = mkbuf(d, NULL, O_LORA_RANK * sizeof(float));
        const float *wg = aw->wo_a_dense + (size_t)g * O_LORA_RANK * group_feat;
        id<MTLBuffer> bwg = mkbuf(d, wg, (size_t)O_LORA_RANK * group_feat * sizeof(float));
        id<MTLCommandBuffer> cb=[P->queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:P->matvec_f32];
        [e setBuffer:bwg offset:0 atIndex:0];
        [e setBuffer:bgv offset:0 atIndex:1];
        [e setBuffer:bog offset:0 atIndex:2];
        uint od=O_LORA_RANK, idd=group_feat;
        [e setBytes:&od  length:4 atIndex:3];
        [e setBytes:&idd length:4 atIndex:4];
        [e dispatchThreads:MTLSizeMake(O_LORA_RANK,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        memcpy(concat + (size_t)g * O_LORA_RANK, [bog contents], O_LORA_RANK * sizeof(float));
        free(gv);
    }
    // wo_b: f32 in → f32 out, then truncate to bfloat16
    id<MTLBuffer> bconcat = mkbuf(d, concat, (size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    id<MTLBuffer> bout_f32= mkbuf(d, NULL, DIM * sizeof(float));
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dq_f32_f32(P, cb, &aw->wo_b, bconcat, bout_f32);
      [cb commit]; [cb waitUntilCompleted]; }

    // Truncate attn_out to bfloat16 → store back as "float shell" (bfloat16 bits in upper 16 bits,
    // lower 16 zero). This matches MLX's .astype(x.dtype) = bfloat16 output.
    // The caller (engine.c) uses this as bfloat16 input to mhc_post_bfloat.
    {
        float *src = (float *)[bout_f32 contents];
        for (int i = 0; i < DIM; i++) {
            uint32_t bits; memcpy(&bits, &src[i], 4);
            uint32_t rb = (bits >> 16) & 1;
            bits = (bits + 0x7FFF + rb) & 0xFFFF0000u;
            memcpy(&out_f32_shell[i], &bits, 4);
        }
    }
    free(concat);
    return 0;
}
