// MLA attention host orchestration (S7b) — see mla_attention.h.
#import "mla_attention.h"
#import <Foundation/Foundation.h>
#import <math.h>
#import <stdlib.h>
#import <string.h>

// --- YaRN tail RoPE cos/sin for a given position (matches DSV4YarnRoPE) ---
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

// --- small Metal dispatch helpers ---
static id<MTLBuffer> mkbuf(id<MTLDevice> d, const void *p, size_t n) {
    return p ? [d newBufferWithBytes:p length:n options:MTLResourceStorageModeShared]
             : [d newBufferWithLength:n options:MTLResourceStorageModeShared];
}

// out[out_dim] = dequant_affine(W) @ x[in_dim]
static void enc_dequant_matvec(MlaPipes *P, id<MTLCommandBuffer> cb,
                               const QuantWeight *qw, id<MTLBuffer> x, id<MTLBuffer> out) {
    id<MTLDevice> d = P->dev;
    id<MTLBuffer> bw = mkbuf(d, qw->packed, (size_t)qw->out_dim * (qw->in_dim / 8) * sizeof(uint32_t));
    int ng = qw->in_dim / qw->group_size;
    id<MTLBuffer> bs = mkbuf(d, qw->scales, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLBuffer> bb = mkbuf(d, qw->biases, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->dequant_matvec_affine];
    [e setBuffer:bw offset:0 atIndex:0];
    [e setBuffer:bs offset:0 atIndex:1];
    [e setBuffer:bb offset:0 atIndex:2];
    [e setBuffer:x offset:0 atIndex:3];
    [e setBuffer:out offset:0 atIndex:4];
    uint od = qw->out_dim, id_ = qw->in_dim, gs = qw->group_size;
    [e setBytes:&od length:4 atIndex:5];
    [e setBytes:&id_ length:4 atIndex:6];
    [e setBytes:&gs length:4 atIndex:7];
    [e dispatchThreads:MTLSizeMake(qw->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
}

// out[n_rows*row_dim] = rms_norm_rows(x, weight?)
static void enc_rms_norm_rows(MlaPipes *P, id<MTLCommandBuffer> cb,
                              id<MTLBuffer> x, id<MTLBuffer> weight, id<MTLBuffer> out,
                              uint n_rows, uint row_dim, int has_weight) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rms_norm_rows];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBuffer:(weight ? weight : x) offset:0 atIndex:1];
    [e setBuffer:out offset:0 atIndex:2];
    uint rd = row_dim; float eps = 1e-6f; uint hw = has_weight ? 1u : 0u;
    [e setBytes:&rd length:4 atIndex:3];
    [e setBytes:&eps length:4 atIndex:4];
    [e setBytes:&hw length:4 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(n_rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
}

static void enc_rope(MlaPipes *P, id<MTLCommandBuffer> cb, id<MTLBuffer> q,
                     id<MTLBuffer> cosb, id<MTLBuffer> sinb, uint n_heads, uint inverse) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rope_tail_interleaved];
    [e setBuffer:q offset:0 atIndex:0];
    [e setBuffer:cosb offset:0 atIndex:1];
    [e setBuffer:sinb offset:0 atIndex:2];
    uint nh = n_heads, hd = HEAD_DIM, nd = QK_NOPE_DIM, rd = QK_ROPE_DIM, inv = inverse;
    [e setBytes:&nh length:4 atIndex:3];
    [e setBytes:&hd length:4 atIndex:4];
    [e setBytes:&nd length:4 atIndex:5];
    [e setBytes:&rd length:4 atIndex:6];
    [e setBytes:&inv length:4 atIndex:7];
    uint half = QK_ROPE_DIM / 2;
    [e dispatchThreads:MTLSizeMake(n_heads * half,1,1) threadsPerThreadgroup:MTLSizeMake(half,1,1)];
    [e endEncoding];
}

int mla_attention_decode(MlaPipes *P, const AttnWeights *aw,
                         const float *x, float *kv_cache, int cache_len,
                         int pos, float *out) {
    id<MTLDevice> d = P->dev;
    int half = QK_ROPE_DIM / 2;
    float cosv[QK_ROPE_DIM / 2], sinv[QK_ROPE_DIM / 2];
    yarn_cos_sin(pos, cosv, sinv);

    id<MTLBuffer> bx     = mkbuf(d, x, DIM * sizeof(float));
    id<MTLBuffer> bcos   = mkbuf(d, cosv, half * sizeof(float));
    id<MTLBuffer> bsin   = mkbuf(d, sinv, half * sizeof(float));
    id<MTLBuffer> bq_a   = mkbuf(d, NULL, Q_LORA_RANK * sizeof(float));
    id<MTLBuffer> bq_res = mkbuf(d, NULL, Q_LORA_RANK * sizeof(float));
    id<MTLBuffer> bq     = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(float));
    id<MTLBuffer> bq_n   = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(float));
    id<MTLBuffer> bkv    = mkbuf(d, NULL, KV_LORA_RANK * sizeof(float));
    id<MTLBuffer> bkv_n  = mkbuf(d, NULL, KV_LORA_RANK * sizeof(float));
    id<MTLBuffer> bqn_w  = mkbuf(d, aw->q_norm, Q_LORA_RANK * sizeof(float));
    id<MTLBuffer> bkvn_w = mkbuf(d, aw->kv_norm, KV_LORA_RANK * sizeof(float));

    // --- Q chain ---
    {
        id<MTLCommandBuffer> cb = [P->queue commandBuffer];
        enc_dequant_matvec(P, cb, &aw->wq_a, bx, bq_a);                          // [1024]
        [cb commit]; [cb waitUntilCompleted];
    }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows(P, cb, bq_a, bqn_w, bq_res, 1, Q_LORA_RANK, 1);          // q_norm
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec(P, cb, &aw->wq_b, bq_res, bq);                          // [32768]
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows(P, cb, bq, NULL, bq_n, N_HEADS, HEAD_DIM, 0);            // per-head
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope(P, cb, bq_n, bcos, bsin, N_HEADS, 0);                            // tail RoPE
      [cb commit]; [cb waitUntilCompleted]; }

    // --- KV chain (single head) ---
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec(P, cb, &aw->wkv, bx, bkv);                              // [512]
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows(P, cb, bkv, bkvn_w, bkv_n, 1, KV_LORA_RANK, 1);          // kv_norm
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope(P, cb, bkv_n, bcos, bsin, 1, 0);                                  // tail RoPE
      [cb commit]; [cb waitUntilCompleted]; }

    // Write current KV into cache at row (cache_len-1).
    {
        float *kvn = (float *)[bkv_n contents];
        memcpy(kv_cache + (size_t)(cache_len - 1) * KV_LORA_RANK, kvn, KV_LORA_RANK * sizeof(float));
    }

    // --- SDPA + sink over cache_len cached KV rows (MQA broadcast) ---
    id<MTLBuffer> bkvcache = mkbuf(d, kv_cache, (size_t)cache_len * KV_LORA_RANK * sizeof(float));
    id<MTLBuffer> bsink    = mkbuf(d, aw->attn_sink, N_HEADS * sizeof(float));
    id<MTLBuffer> battn    = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(float));
    {
        id<MTLCommandBuffer> cb=[P->queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:P->mla_sdpa_decode];
        [e setBuffer:bq_n offset:0 atIndex:0];
        [e setBuffer:bkvcache offset:0 atIndex:1];
        [e setBuffer:bsink offset:0 atIndex:2];
        [e setBuffer:battn offset:0 atIndex:3];
        uint nh=N_HEADS, hd=HEAD_DIM, nk=cache_len; float scale=1.0f/sqrtf((float)HEAD_DIM);
        [e setBytes:&nh length:4 atIndex:4];
        [e setBytes:&hd length:4 atIndex:5];
        [e setBytes:&nk length:4 atIndex:6];
        [e setBytes:&scale length:4 atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(N_HEADS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }

    // --- inverse RoPE on attn output ---
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope(P, cb, battn, bcos, bsin, N_HEADS, 1);
      [cb commit]; [cb waitUntilCompleted]; }

    // --- grouped wo_a -> concat -> wo_b (host assembles group vecs) ---
    float *attn = (float *)[battn contents];   // [N_HEADS, HEAD_DIM]
    int heads_per_group = N_HEADS / O_GROUPS;  // 8
    int group_feat = heads_per_group * HEAD_DIM; // 4096
    float *concat = malloc((size_t)O_GROUPS * O_LORA_RANK * sizeof(float)); // 8192
    for (int g = 0; g < O_GROUPS; g++) {
        // group vector: heads [g*hpg .. g*hpg+hpg) flattened head-major
        float *gv = malloc((size_t)group_feat * sizeof(float));
        for (int hh = 0; hh < heads_per_group; hh++)
            memcpy(gv + hh * HEAD_DIM, attn + (g * heads_per_group + hh) * HEAD_DIM, HEAD_DIM * sizeof(float));
        id<MTLBuffer> bgv = mkbuf(d, gv, group_feat * sizeof(float));
        id<MTLBuffer> bog = mkbuf(d, NULL, O_LORA_RANK * sizeof(float));
        id<MTLCommandBuffer> cb=[P->queue commandBuffer];
        enc_dequant_matvec(P, cb, &aw->wo_a[g], bgv, bog);
        [cb commit]; [cb waitUntilCompleted];
        memcpy(concat + (size_t)g * O_LORA_RANK, [bog contents], O_LORA_RANK * sizeof(float));
        free(gv);
    }
    id<MTLBuffer> bconcat = mkbuf(d, concat, (size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    id<MTLBuffer> bout    = mkbuf(d, NULL, DIM * sizeof(float));
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec(P, cb, &aw->wo_b, bconcat, bout);                       // [4096]
      [cb commit]; [cb waitUntilCompleted]; }
    memcpy(out, [bout contents], DIM * sizeof(float));
    free(concat);
    return 0;
}
