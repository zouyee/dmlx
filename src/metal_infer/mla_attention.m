// MLA attention host orchestration (S7b) — see mla_attention.h.
#import "mla_attention.h"
#import <Foundation/Foundation.h>
#import <math.h>
#import <stdlib.h>
#import <string.h>

// ============================================================================
// Persistent attention weight GPU buffers — eliminates 120+ mkbuf calls/layer
// ============================================================================
//
// Each call to mla_attention_decode_bf16 previously created ~120 MTLBuffer objects
// for the attention weights (wq_a, wq_b, wkv, wo_a×8, wo_b, q_norm, kv_norm, attn_sink).
// These weights are fixed per layer — we cache them once after first use.
//
// Key sizes (per layer):
//   wq_b packed:   32768×(1024/8)×4B = 16MB
//   wq_a packed:   1024×(4096/8)×4B  = 2MB
//   wkv packed:    512×(4096/8)×4B   = 1MB
//   wo_b packed:   4096×(8192/8)×4B  = 16MB
//   wo_a_dense:    8×1024×4096×4B    = 128MB  ← largest!
//   q_norm, kv_norm, attn_sink: < 10KB each

#define ATTN_BUF_CACHE_SIZE 64  // max layers

typedef struct {
    // Quantized weight buffers (packed, scales, biases)
    id<MTLBuffer> wq_a_pack, wq_a_sc, wq_a_bi;
    id<MTLBuffer> wq_b_pack, wq_b_sc, wq_b_bi;
    id<MTLBuffer> wkv_pack,  wkv_sc,  wkv_bi;
    id<MTLBuffer> wo_b_pack, wo_b_sc, wo_b_bi;
    // Norm/sink (small, < 10KB each)
    id<MTLBuffer> q_norm_buf, kv_norm_buf, attn_sink_buf;
    // wo_a_dense: 8 groups × [O_LORA_RANK, group_feat] f32
    id<MTLBuffer> wo_a_grp[8];   // 8 × 16MB = 128MB total per layer
    // Validated: points to the AttnWeights that seeded this entry (pointer identity)
    const void *owner;
} AttnBufCache;

static AttnBufCache g_attn_cache[ATTN_BUF_CACHE_SIZE];
static int g_attn_cache_n = 0;

// Look up or create the persistent GPU buffers for a given AttnWeights pointer.
static AttnBufCache *attn_buf_cache_get(id<MTLDevice> d, const AttnWeights *aw) {
    // Fast path: check existing entries
    for (int i = 0; i < g_attn_cache_n; i++) {
        if (g_attn_cache[i].owner == (const void *)aw) return &g_attn_cache[i];
    }
    // New entry
    if (g_attn_cache_n >= ATTN_BUF_CACHE_SIZE) return NULL;  // overflow — fall back to mkbuf
    AttnBufCache *c = &g_attn_cache[g_attn_cache_n++];
    memset(c, 0, sizeof(*c));
    c->owner = (const void *)aw;

    // Helper: create Shared buffer from pointer+size
    #define MKGPU(ptr, sz) [d newBufferWithBytes:(ptr) length:(sz) options:MTLResourceStorageModeShared]
    #define MKGPU_QW(qw) do { \
        size_t pw_sz = (size_t)(qw).out_dim * ((qw).in_dim / 8) * sizeof(uint32_t); \
        int ng = (qw).in_dim / (qw).group_size; \
        size_t sc_sz = (size_t)(qw).out_dim * ng * sizeof(float); \
        c->qw##_pack = MKGPU((qw).packed, pw_sz); \
        c->qw##_sc   = MKGPU((qw).scales,  sc_sz); \
        c->qw##_bi   = MKGPU((qw).biases,  sc_sz); \
    } while(0)

    // wq_a, wq_b, wkv, wo_b: cache only when memory allows (SMELT=disabled).
    // With SMELT 35GB + these buffers 1.5GB = too much for 38GB M4 Pro.
    // Without SMELT: backbone (~5GB) + these buffers (~1.5GB per layer × 43 = 64GB total?) 
    // Actually: wq_b(16MB)+wo_b(16MB)+wkv(1MB)+wq_a(2MB) = 35MB/layer × 43 = 1.5GB — safe!
    // wo_a_dense: 128MB/layer × 43 = 5.5GB — too large.
    const char *smelt_n_env = getenv("NATIVE_SMELT_N");
    const int smelt_disabled = (smelt_n_env && atoi(smelt_n_env) == 0);
    if (smelt_disabled) {
        // SSD mode: memory available, cache medium-sized weight buffers
        size_t pw; int ng; size_t sc;
        pw = (size_t)aw->wq_a.out_dim * (aw->wq_a.in_dim / 8) * sizeof(uint32_t);
        ng = aw->wq_a.in_dim / aw->wq_a.group_size; sc = (size_t)aw->wq_a.out_dim * ng * sizeof(float);
        c->wq_a_pack = MKGPU(aw->wq_a.packed, pw);
        c->wq_a_sc   = MKGPU(aw->wq_a.scales, sc);
        c->wq_a_bi   = MKGPU(aw->wq_a.biases, sc);

        pw = (size_t)aw->wq_b.out_dim * (aw->wq_b.in_dim / 8) * sizeof(uint32_t);
        ng = aw->wq_b.in_dim / aw->wq_b.group_size; sc = (size_t)aw->wq_b.out_dim * ng * sizeof(float);
        c->wq_b_pack = MKGPU(aw->wq_b.packed, pw);
        c->wq_b_sc   = MKGPU(aw->wq_b.scales, sc);
        c->wq_b_bi   = MKGPU(aw->wq_b.biases, sc);

        pw = (size_t)aw->wkv.out_dim * (aw->wkv.in_dim / 8) * sizeof(uint32_t);
        ng = aw->wkv.in_dim / aw->wkv.group_size; sc = (size_t)aw->wkv.out_dim * ng * sizeof(float);
        c->wkv_pack = MKGPU(aw->wkv.packed, pw);
        c->wkv_sc   = MKGPU(aw->wkv.scales, sc);
        c->wkv_bi   = MKGPU(aw->wkv.biases, sc);

        pw = (size_t)aw->wo_b.out_dim * (aw->wo_b.in_dim / 8) * sizeof(uint32_t);
        ng = aw->wo_b.in_dim / aw->wo_b.group_size; sc = (size_t)aw->wo_b.out_dim * ng * sizeof(float);
        c->wo_b_pack = MKGPU(aw->wo_b.packed, pw);
        c->wo_b_sc   = MKGPU(aw->wo_b.scales, sc);
        c->wo_b_bi   = MKGPU(aw->wo_b.biases, sc);

        // wo_a_dense: 128MB/layer × 43 = 5.5GB — too large to copy (newBufferWithBytes)!
        // Use newBufferWithBytesNoCopy (zero-copy) and CACHE it.
        // Total: 8 × 16MB × 43 = 5.5GB of Metal buffer objects, but zero extra RAM.
        // (wo_a_dense is already in process memory from NativeLoader)
        {
            int heads_per_group = N_HEADS / O_GROUPS;
            int group_feat = heads_per_group * HEAD_DIM;  // 4096
            size_t grp_sz = (size_t)O_LORA_RANK * group_feat * sizeof(float);  // 16MB
            for (int g = 0; g < O_GROUPS; g++) {
                const float *wg = aw->wo_a_dense + (size_t)g * O_LORA_RANK * group_feat;
                // NoCopy: zero-copy wrapper, data stays in wo_a_dense memory
                c->wo_a_grp[g] = [d newBufferWithBytesNoCopy:(void*)wg
                                     length:grp_sz
                                     options:MTLResourceStorageModeShared
                                     deallocator:nil];
            }
        }
    } else {
        // SMELT mode: memory tight (35GB + backbone ~3GB = ~38GB, near M4 Pro limit)
        // Only cache tiny norms/sink. Skip wo_a_dense NoCopy to avoid OOM from Metal metadata.
        c->wq_a_pack = nil; c->wq_a_sc = nil; c->wq_a_bi = nil;
        c->wq_b_pack = nil; c->wq_b_sc = nil; c->wq_b_bi = nil;
        c->wkv_pack  = nil; c->wkv_sc  = nil; c->wkv_bi  = nil;
        c->wo_b_pack = nil; c->wo_b_sc = nil; c->wo_b_bi = nil;
        for (int g = 0; g < O_GROUPS; g++) c->wo_a_grp[g] = nil;
    }
    // q_norm, kv_norm, attn_sink (tiny — always cached)
    c->q_norm_buf    = MKGPU(aw->q_norm,    Q_LORA_RANK * sizeof(float));
    c->kv_norm_buf   = MKGPU(aw->kv_norm,   KV_LORA_RANK * sizeof(float));
    c->attn_sink_buf = MKGPU(aw->attn_sink, N_HEADS * sizeof(float));
    // wo_a_dense: 8 groups × 16MB = 128MB per layer × 43 layers = 5.5GB total
    // DISABLED: too large with SMELT (35GB + 5.5GB > 38GB M4 Pro limit → causes swap)
    // Instead cache only the small norms/sink (< 4KB total per layer).
    for (int g = 0; g < O_GROUPS; g++) c->wo_a_grp[g] = nil;
    #undef MKGPU
    return c;
}

// Thin wrapper for enc_dequant_matvec_* using pre-cached buffers.
// These shadow the helpers below so the bf16 function can use cached buffers.
static void enc_dq_bf16_cached(MlaPipes *P, id<MTLCommandBuffer> cb,
                               id<MTLBuffer> bw, id<MTLBuffer> bs, id<MTLBuffer> bb,
                               int out_dim, int in_dim, int group_size,
                               id<MTLBuffer> x, id<MTLBuffer> out,
                               id<MTLComputePipelineState> pipe) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pipe];
    [e setBuffer:bw offset:0 atIndex:0];
    [e setBuffer:bs offset:0 atIndex:1];
    [e setBuffer:bb offset:0 atIndex:2];
    [e setBuffer:x  offset:0 atIndex:3];
    [e setBuffer:out offset:0 atIndex:4];
    uint od = out_dim, id_ = in_dim, gs = group_size;
    [e setBytes:&od length:4 atIndex:5];
    [e setBytes:&id_ length:4 atIndex:6];
    [e setBytes:&gs  length:4 atIndex:7];
    [e dispatchThreads:MTLSizeMake(out_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
}

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

static void enc_dequant_matvec_f16out(MlaPipes *P, id<MTLCommandBuffer> cb,
                                      const QuantWeight *qw, id<MTLBuffer> x, id<MTLBuffer> out) {
    id<MTLDevice> d = P->dev;
    id<MTLBuffer> bw = mkbuf(d, qw->packed, (size_t)qw->out_dim * (qw->in_dim / 8) * sizeof(uint32_t));
    int ng = qw->in_dim / qw->group_size;
    id<MTLBuffer> bs = mkbuf(d, qw->scales, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLBuffer> bb = mkbuf(d, qw->biases, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->dequant_matvec_affine_f16out];
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

static void enc_dequant_matvec_f16in_f16out(MlaPipes *P, id<MTLCommandBuffer> cb,
                                            const QuantWeight *qw, id<MTLBuffer> x, id<MTLBuffer> out) {
    id<MTLDevice> d = P->dev;
    id<MTLBuffer> bw = mkbuf(d, qw->packed, (size_t)qw->out_dim * (qw->in_dim / 8) * sizeof(uint32_t));
    int ng = qw->in_dim / qw->group_size;
    id<MTLBuffer> bs = mkbuf(d, qw->scales, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLBuffer> bb = mkbuf(d, qw->biases, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->dequant_matvec_affine_f16in_f16out];
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

// --- BF16 dispatch helpers ---
static void enc_dequant_matvec_bf16out(MlaPipes *P, id<MTLCommandBuffer> cb,
                                       const QuantWeight *qw, id<MTLBuffer> x, id<MTLBuffer> out) {
    id<MTLDevice> d = P->dev;
    id<MTLBuffer> bw = mkbuf(d, qw->packed, (size_t)qw->out_dim * (qw->in_dim / 8) * sizeof(uint32_t));
    int ng = qw->in_dim / qw->group_size;
    id<MTLBuffer> bs = mkbuf(d, qw->scales, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLBuffer> bb = mkbuf(d, qw->biases, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->dequant_matvec_affine_bf16out];
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

static void enc_dequant_matvec_bf16in_bf16out(MlaPipes *P, id<MTLCommandBuffer> cb,
                                              const QuantWeight *qw, id<MTLBuffer> x, id<MTLBuffer> out) {
    id<MTLDevice> d = P->dev;
    id<MTLBuffer> bw = mkbuf(d, qw->packed, (size_t)qw->out_dim * (qw->in_dim / 8) * sizeof(uint32_t));
    int ng = qw->in_dim / qw->group_size;
    id<MTLBuffer> bs = mkbuf(d, qw->scales, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLBuffer> bb = mkbuf(d, qw->biases, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->dequant_matvec_affine_bf16in_bf16out];
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

static void enc_rms_norm_rows_bf16in_bf16out(MlaPipes *P, id<MTLCommandBuffer> cb,
                                             id<MTLBuffer> x, id<MTLBuffer> weight, id<MTLBuffer> out,
                                             uint n_rows, uint row_dim, int has_weight) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rms_norm_rows_bf16in_bf16out];
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

static void enc_rope_bf16(MlaPipes *P, id<MTLCommandBuffer> cb, id<MTLBuffer> q,
                          id<MTLBuffer> cosb, id<MTLBuffer> sinb, uint n_heads, uint inverse) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rope_tail_interleaved_bf16];
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

static void enc_matvec_f32_bf16in(MlaPipes *P, id<MTLCommandBuffer> cb,
                                  id<MTLBuffer> W, id<MTLBuffer> x, id<MTLBuffer> out,
                                  uint out_dim, uint in_dim) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->matvec_f32_bf16in];
    [e setBuffer:W offset:0 atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:out offset:0 atIndex:2];
    [e setBytes:&out_dim length:4 atIndex:3];
    [e setBytes:&in_dim length:4 atIndex:4];
    [e dispatchThreads:MTLSizeMake(out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
}

static void enc_dequant_matvec_bf16in_f32out(MlaPipes *P, id<MTLCommandBuffer> cb,
                                             const QuantWeight *qw, id<MTLBuffer> x, id<MTLBuffer> out) {
    id<MTLDevice> d = P->dev;
    id<MTLBuffer> bw = mkbuf(d, qw->packed, (size_t)qw->out_dim * (qw->in_dim / 8) * sizeof(uint32_t));
    int ng = qw->in_dim / qw->group_size;
    id<MTLBuffer> bs = mkbuf(d, qw->scales, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLBuffer> bb = mkbuf(d, qw->biases, (size_t)qw->out_dim * ng * sizeof(float));
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->dequant_matvec_affine_bf16in_f32out];
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

static void enc_rms_norm_rows_f16out(MlaPipes *P, id<MTLCommandBuffer> cb,
                                     id<MTLBuffer> x, id<MTLBuffer> weight, id<MTLBuffer> out,
                                     uint n_rows, uint row_dim, int has_weight) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rms_norm_rows_f16out];
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

static void enc_rms_norm_rows_f16in_f16out(MlaPipes *P, id<MTLCommandBuffer> cb,
                                           id<MTLBuffer> x, id<MTLBuffer> weight, id<MTLBuffer> out,
                                           uint n_rows, uint row_dim, int has_weight) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rms_norm_rows_f16in_f16out];
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

static void enc_rope_f16(MlaPipes *P, id<MTLCommandBuffer> cb, id<MTLBuffer> q,
                         id<MTLBuffer> cosb, id<MTLBuffer> sinb, uint n_heads, uint inverse) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rope_tail_interleaved_f16];
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

static void enc_matvec_f32_f16in(MlaPipes *P, id<MTLCommandBuffer> cb,
                                 id<MTLBuffer> W, id<MTLBuffer> x, id<MTLBuffer> out,
                                 uint out_dim, uint in_dim) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->matvec_f32_f16in];
    [e setBuffer:W offset:0 atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:out offset:0 atIndex:2];
    [e setBytes:&out_dim length:4 atIndex:3];
    [e setBytes:&in_dim length:4 atIndex:4];
    [e dispatchThreads:MTLSizeMake(out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
}

static void enc_matvec_f32(MlaPipes *P, id<MTLCommandBuffer> cb,
                             id<MTLBuffer> W, id<MTLBuffer> x, id<MTLBuffer> out,
                             uint out_dim, uint in_dim) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->matvec_f32];
    [e setBuffer:W offset:0 atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:out offset:0 atIndex:2];
    [e setBytes:&out_dim length:4 atIndex:3];
    [e setBytes:&in_dim length:4 atIndex:4];
    [e dispatchThreads:MTLSizeMake(out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
}

// f32 -> f16 round-trip (matches ds4's KV cache precision)
static inline uint16_t f32_to_f16(float f) {
    _Float16 h = (_Float16)f;
    return *(uint16_t *)&h;
}

int mla_attention_decode(MlaPipes *P, const AttnWeights *aw,
                         const float *x, uint16_t *kv_cache, int cache_len,
                         int pos, float *out) {
    id<MTLDevice> d = P->dev;
    int half = QK_ROPE_DIM / 2;
    float cosv[QK_ROPE_DIM / 2], sinv[QK_ROPE_DIM / 2];
    yarn_cos_sin(pos, cosv, sinv);

    id<MTLBuffer> bx     = mkbuf(d, x, DIM * sizeof(float));
    id<MTLBuffer> bcos   = mkbuf(d, cosv, half * sizeof(float));
    id<MTLBuffer> bsin   = mkbuf(d, sinv, half * sizeof(float));
    id<MTLBuffer> bq_a   = mkbuf(d, NULL, Q_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bq_res = mkbuf(d, NULL, Q_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bq     = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    id<MTLBuffer> bq_n   = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    id<MTLBuffer> bkv    = mkbuf(d, NULL, KV_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bkv_n  = mkbuf(d, NULL, KV_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bqn_w  = mkbuf(d, aw->q_norm, Q_LORA_RANK * sizeof(float));
    id<MTLBuffer> bkvn_w = mkbuf(d, aw->kv_norm, KV_LORA_RANK * sizeof(float));

    // --- Q chain (f16 precision) ---
    {
        id<MTLCommandBuffer> cb = [P->queue commandBuffer];
        enc_dequant_matvec_f16out(P, cb, &aw->wq_a, bx, bq_a);                   // [1024] f16
        [cb commit]; [cb waitUntilCompleted];
    }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows_f16in_f16out(P, cb, bq_a, bqn_w, bq_res, 1, Q_LORA_RANK, 1); // q_norm f16
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec_f16in_f16out(P, cb, &aw->wq_b, bq_res, bq);             // [32768] f16
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows_f16in_f16out(P, cb, bq, NULL, bq_n, N_HEADS, HEAD_DIM, 0); // per-head f16
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope_f16(P, cb, bq_n, bcos, bsin, N_HEADS, 0);                         // tail RoPE f16
      [cb commit]; [cb waitUntilCompleted]; }

    // --- KV chain (single head, f16 precision) ---
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec_f16out(P, cb, &aw->wkv, bx, bkv);                       // [512] f16
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows_f16in_f16out(P, cb, bkv, bkvn_w, bkv_n, 1, KV_LORA_RANK, 1); // kv_norm f16
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope_f16(P, cb, bkv_n, bcos, bsin, 1, 0);                              // tail RoPE f16
      [cb commit]; [cb waitUntilCompleted]; }

    // Write current KV into cache at row (cache_len-1) (already f16).
    {
        uint16_t *kvn = (uint16_t *)[bkv_n contents];
        uint16_t *dst = kv_cache + (size_t)(cache_len - 1) * KV_LORA_RANK;
        memcpy(dst, kvn, KV_LORA_RANK * sizeof(uint16_t));
    }

    // --- SDPA + sink over cache_len cached KV rows (MQA broadcast, all f16) ---
    id<MTLBuffer> bkvcache = mkbuf(d, kv_cache, (size_t)cache_len * KV_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bsink    = mkbuf(d, aw->attn_sink, N_HEADS * sizeof(float));
    id<MTLBuffer> battn    = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    {
        id<MTLCommandBuffer> cb=[P->queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:P->mla_sdpa_decode_f16in_f16out];
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
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope_f16(P, cb, battn, bcos, bsin, N_HEADS, 1);
      [cb commit]; [cb waitUntilCompleted]; }

    // --- grouped wo_a -> concat -> wo_b (host assembles group vecs from f16) ---
    uint16_t *attn_f16 = (uint16_t *)[battn contents]; // [N_HEADS, HEAD_DIM] f16
    int heads_per_group = N_HEADS / O_GROUPS;  // 8
    int group_feat = heads_per_group * HEAD_DIM; // 4096
    float *concat = malloc((size_t)O_GROUPS * O_LORA_RANK * sizeof(float)); // 8192 f32
    if (!concat) return -1;
    for (int g = 0; g < O_GROUPS; g++) {
        uint16_t *gv_f16 = malloc((size_t)group_feat * sizeof(uint16_t));
        if (!gv_f16) { free(concat); return -1; }
        for (int hh = 0; hh < heads_per_group; hh++)
            memcpy(gv_f16 + hh * HEAD_DIM, attn_f16 + (g * heads_per_group + hh) * HEAD_DIM, HEAD_DIM * sizeof(uint16_t));
        id<MTLBuffer> bgv = mkbuf(d, gv_f16, group_feat * sizeof(uint16_t));
        id<MTLBuffer> bog = mkbuf(d, NULL, O_LORA_RANK * sizeof(float));
        const float *wg = aw->wo_a_dense + (size_t)g * O_LORA_RANK * group_feat;
        id<MTLBuffer> bwg = mkbuf(d, wg, (size_t)O_LORA_RANK * group_feat * sizeof(float));
        id<MTLCommandBuffer> cb=[P->queue commandBuffer];
        enc_matvec_f32_f16in(P, cb, bwg, bgv, bog, O_LORA_RANK, group_feat);
        [cb commit]; [cb waitUntilCompleted];
        memcpy(concat + (size_t)g * O_LORA_RANK, [bog contents], O_LORA_RANK * sizeof(float));
        free(gv_f16);
    }
    id<MTLBuffer> bconcat = mkbuf(d, concat, (size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    id<MTLBuffer> bout    = mkbuf(d, NULL, DIM * sizeof(float));
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec(P, cb, &aw->wo_b, bconcat, bout);                       // [4096] f32
      [cb commit]; [cb waitUntilCompleted]; }
    memcpy(out, [bout contents], DIM * sizeof(float));
    free(concat);
    return 0;
}

// --- BF16 variant of mla_attention_decode ---
int mla_attention_decode_bf16(MlaPipes *P, const AttnWeights *aw,
                              const uint16_t *x, uint16_t *kv_cache, int cache_len,
                              int pos, float *out) {
    @autoreleasepool {
    id<MTLDevice> d = P->dev;
    int half = QK_ROPE_DIM / 2;
    float cosv[QK_ROPE_DIM / 2], sinv[QK_ROPE_DIM / 2];
    yarn_cos_sin(pos, cosv, sinv);

    // === Get (or create) persistent GPU buffers for fixed attention weights ===
    AttnBufCache *abc = attn_buf_cache_get(d, aw);

    id<MTLBuffer> bx     = mkbuf(d, x, DIM * sizeof(uint16_t));
    id<MTLBuffer> bcos   = mkbuf(d, cosv, half * sizeof(float));
    id<MTLBuffer> bsin   = mkbuf(d, sinv, half * sizeof(float));
    id<MTLBuffer> bq_a   = mkbuf(d, NULL, Q_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bq_res = mkbuf(d, NULL, Q_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bq     = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    id<MTLBuffer> bq_n   = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    id<MTLBuffer> bkv    = mkbuf(d, NULL, KV_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bkv_n  = mkbuf(d, NULL, KV_LORA_RANK * sizeof(uint16_t));

    // Use persistent norm/sink buffers if available, else fall back to mkbuf
    id<MTLBuffer> bqn_w  = abc ? abc->q_norm_buf    : mkbuf(d, aw->q_norm,    Q_LORA_RANK * sizeof(float));
    id<MTLBuffer> bkvn_w = abc ? abc->kv_norm_buf   : mkbuf(d, aw->kv_norm,   KV_LORA_RANK * sizeof(float));
    id<MTLBuffer> bsink  = abc ? abc->attn_sink_buf : mkbuf(d, aw->attn_sink, N_HEADS * sizeof(float));

    // === CB1: Q chain + KV chain merged into ONE command buffer (8 encoders, 1 wait) ===
    {
        id<MTLCommandBuffer> cb1 = [P->queue commandBuffer];
        // Q chain — use cached weight buffers when available (wq_a, wq_b, wkv may be nil if memory-limited)
        if (abc && abc->wq_a_pack) {
            enc_dq_bf16_cached(P, cb1, abc->wq_a_pack, abc->wq_a_sc, abc->wq_a_bi,
                               aw->wq_a.out_dim, aw->wq_a.in_dim, aw->wq_a.group_size, bx, bq_a,
                               P->dequant_matvec_affine_bf16in_bf16out);
        } else {
            enc_dequant_matvec_bf16in_bf16out(P, cb1, &aw->wq_a, bx, bq_a);
        }
        enc_rms_norm_rows_bf16in_bf16out(P, cb1, bq_a, bqn_w, bq_res, 1, Q_LORA_RANK, 1);
        if (abc && abc->wq_b_pack) {
            enc_dq_bf16_cached(P, cb1, abc->wq_b_pack, abc->wq_b_sc, abc->wq_b_bi,
                               aw->wq_b.out_dim, aw->wq_b.in_dim, aw->wq_b.group_size, bq_res, bq,
                               P->dequant_matvec_affine_bf16in_bf16out);
        } else {
            enc_dequant_matvec_bf16in_bf16out(P, cb1, &aw->wq_b, bq_res, bq);
        }
        enc_rms_norm_rows_bf16in_bf16out(P, cb1, bq, NULL, bq_n, N_HEADS, HEAD_DIM, 0);
        enc_rope_bf16(P, cb1, bq_n, bcos, bsin, N_HEADS, 0);
        // KV chain
        if (abc && abc->wkv_pack) {
            enc_dq_bf16_cached(P, cb1, abc->wkv_pack, abc->wkv_sc, abc->wkv_bi,
                               aw->wkv.out_dim, aw->wkv.in_dim, aw->wkv.group_size, bx, bkv,
                               P->dequant_matvec_affine_bf16in_bf16out);
        } else {
            enc_dequant_matvec_bf16in_bf16out(P, cb1, &aw->wkv, bx, bkv);
        }
        enc_rms_norm_rows_bf16in_bf16out(P, cb1, bkv, bkvn_w, bkv_n, 1, KV_LORA_RANK, 1);
        enc_rope_bf16(P, cb1, bkv_n, bcos, bsin, 1, 0);
        [cb1 commit]; [cb1 waitUntilCompleted];
    }

    // CPU: write KV to cache (must happen before SDPA reads cache)
    {
        uint16_t *kvn_bf16 = (uint16_t *)[bkv_n contents];
        uint16_t *dst = kv_cache + (size_t)(cache_len - 1) * KV_LORA_RANK;
        for (int i = 0; i < KV_LORA_RANK; i++) {
            uint32_t u32 = ((uint32_t)kvn_bf16[i]) << 16;
            float fval; memcpy(&fval, &u32, 4);
            _Float16 f16val = (_Float16)fval;
            memcpy(&dst[i], &f16val, 2);
        }
    }

    // === CB2: SDPA + inverse RoPE (2 encoders, 1 wait) ===
    id<MTLBuffer> bkvcache = mkbuf(d, kv_cache, (size_t)cache_len * KV_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> battn    = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    {
        id<MTLCommandBuffer> cb2 = [P->queue commandBuffer];
        {
            id<MTLComputeCommandEncoder> e = [cb2 computeCommandEncoder];
            [e setComputePipelineState:P->mla_sdpa_decode_bfloat];
            [e setBuffer:bq_n offset:0 atIndex:0];
            [e setBuffer:bkvcache offset:0 atIndex:1];
            [e setBuffer:bsink offset:0 atIndex:2];
            [e setBuffer:battn offset:0 atIndex:3];
            uint nh=N_HEADS, hd=HEAD_DIM, nk=cache_len; float scale=1.0f/sqrtf((float)HEAD_DIM);
            [e setBytes:&nh length:4 atIndex:4];
            [e setBytes:&hd length:4 atIndex:5];
            [e setBytes:&nk length:4 atIndex:6];
            [e setBytes:&scale length:4 atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake(N_HEADS,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [e endEncoding];
        }
        enc_rope_bf16(P, cb2, battn, bcos, bsin, N_HEADS, 1);
        [cb2 commit]; [cb2 waitUntilCompleted];
    }

    // === CB3: wo_a (8 groups) — use cached wo_a_dense buffers ===
    uint16_t *attn_bf16 = (uint16_t *)[battn contents];
    int heads_per_group = N_HEADS / O_GROUPS;
    int group_feat = heads_per_group * HEAD_DIM;
    float *concat = malloc((size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    if (!concat) return -1;

    id<MTLBuffer> bog_arr[O_GROUPS];
    {
        id<MTLCommandBuffer> cb3 = [P->queue commandBuffer];
        for (int g = 0; g < O_GROUPS; g++) {
            uint16_t *gv_bf16 = malloc((size_t)group_feat * sizeof(uint16_t));
            if (!gv_bf16) { free(concat); return -1; }
            for (int hh = 0; hh < heads_per_group; hh++)
                memcpy(gv_bf16 + hh * HEAD_DIM, attn_bf16 + (g * heads_per_group + hh) * HEAD_DIM, HEAD_DIM * sizeof(uint16_t));
            id<MTLBuffer> bgv = mkbuf(d, gv_bf16, group_feat * sizeof(uint16_t));
            bog_arr[g] = mkbuf(d, NULL, O_LORA_RANK * sizeof(float));
            // Use persistent wo_a group buffer if available, else NoCopy zero-copy
            id<MTLBuffer> bwg = (abc && abc->wo_a_grp[g])
                                    ? abc->wo_a_grp[g]
                                    : [d newBufferWithBytesNoCopy:(void*)(aw->wo_a_dense + (size_t)g * O_LORA_RANK * group_feat)
                                          length:(size_t)O_LORA_RANK * group_feat * sizeof(float)
                                          options:MTLResourceStorageModeShared
                                          deallocator:nil];
            enc_matvec_f32_bf16in(P, cb3, bwg, bgv, bog_arr[g], O_LORA_RANK, group_feat);
            free(gv_bf16);
        }
        [cb3 commit]; [cb3 waitUntilCompleted];
    }
    for (int g = 0; g < O_GROUPS; g++)
        memcpy(concat + (size_t)g * O_LORA_RANK, [bog_arr[g] contents], O_LORA_RANK * sizeof(float));

    // === CB4: wo_b (1 encoder, 1 wait) ===
    id<MTLBuffer> bconcat = mkbuf(d, concat, (size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    id<MTLBuffer> bout    = mkbuf(d, NULL, DIM * sizeof(float));
    {
        id<MTLCommandBuffer> cb4 = [P->queue commandBuffer];
        if (abc && abc->wo_b_pack) {
            id<MTLComputeCommandEncoder> e = [cb4 computeCommandEncoder];
            [e setComputePipelineState:P->dequant_matvec_affine];
            [e setBuffer:abc->wo_b_pack offset:0 atIndex:0];
            [e setBuffer:abc->wo_b_sc   offset:0 atIndex:1];
            [e setBuffer:abc->wo_b_bi   offset:0 atIndex:2];
            [e setBuffer:bconcat offset:0 atIndex:3];
            [e setBuffer:bout    offset:0 atIndex:4];
            uint od=aw->wo_b.out_dim, id_=aw->wo_b.in_dim, gs=aw->wo_b.group_size;
            [e setBytes:&od length:4 atIndex:5];
            [e setBytes:&id_ length:4 atIndex:6];
            [e setBytes:&gs  length:4 atIndex:7];
            [e dispatchThreads:MTLSizeMake(aw->wo_b.out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        } else {
            enc_dequant_matvec(P, cb4, &aw->wo_b, bconcat, bout);
        }
        [cb4 commit]; [cb4 waitUntilCompleted];
    }
    memcpy(out, [bout contents], DIM * sizeof(float));
    free(concat);
    } // end @autoreleasepool
    return 0;
}

// --- DS4-style attention: Q chain f32, KV cache f32 (full precision), SDPA f32×f32.
int mla_attention_decode_f16kv(MlaPipes *P, const AttnWeights *aw,
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

    // --- Q chain (f32 precision, ds4-style) ---
    {
        id<MTLCommandBuffer> cb = [P->queue commandBuffer];
        enc_dequant_matvec(P, cb, &aw->wq_a, bx, bq_a);                   // [1024] f32
        [cb commit]; [cb waitUntilCompleted];
    }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows(P, cb, bq_a, bqn_w, bq_res, 1, Q_LORA_RANK, 1);      // q_norm f32
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec(P, cb, &aw->wq_b, bq_res, bq);                     // [32768] f32
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows(P, cb, bq, NULL, bq_n, N_HEADS, HEAD_DIM, 0);        // per-head f32
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope(P, cb, bq_n, bcos, bsin, N_HEADS, 0);                         // tail RoPE f32
      [cb commit]; [cb waitUntilCompleted]; }

    // --- KV chain (f32 compute, f16 storage) ---
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec(P, cb, &aw->wkv, bx, bkv);                         // [512] f32
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows(P, cb, bkv, bkvn_w, bkv_n, 1, KV_LORA_RANK, 1);      // kv_norm f32
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope(P, cb, bkv_n, bcos, bsin, 1, 0);                              // tail RoPE f32
      [cb commit]; [cb waitUntilCompleted]; }

    // Write current KV into cache at row (cache_len-1) directly as f32.
    {
        float *kvn = (float *)[bkv_n contents];
        float *dst = kv_cache + (size_t)(cache_len - 1) * KV_LORA_RANK;
        memcpy(dst, kvn, KV_LORA_RANK * sizeof(float));
    }

    // --- SDPA + sink over cache_len cached KV rows (MQA, Q=f32, KV=f32, out=f32) ---
    id<MTLBuffer> bkvcache = mkbuf(d, kv_cache, (size_t)cache_len * KV_LORA_RANK * sizeof(float));
    id<MTLBuffer> bsink    = mkbuf(d, aw->attn_sink, N_HEADS * sizeof(float));
    id<MTLBuffer> battn    = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(float));
    {
        id<MTLCommandBuffer> cb=[P->queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:P->mla_sdpa_decode];   // f32 × f32
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
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope(P, cb, battn, bcos, bsin, N_HEADS, 1);                        // inverse RoPE f32
      [cb commit]; [cb waitUntilCompleted]; }

    // --- grouped wo_a -> concat -> wo_b (all f32) ---
    float *attn_f32 = (float *)[battn contents]; // [N_HEADS, HEAD_DIM] f32
    int heads_per_group = N_HEADS / O_GROUPS;  // 8
    int group_feat = heads_per_group * HEAD_DIM; // 4096
    float *concat = malloc((size_t)O_GROUPS * O_LORA_RANK * sizeof(float)); // 8192 f32
    if (!concat) return -1;
    for (int g = 0; g < O_GROUPS; g++) {
        id<MTLBuffer> bgv = mkbuf(d, attn_f32 + (size_t)g * heads_per_group * HEAD_DIM,
                                  group_feat * sizeof(float));
        id<MTLBuffer> bog = mkbuf(d, NULL, O_LORA_RANK * sizeof(float));
        const float *wg = aw->wo_a_dense + (size_t)g * O_LORA_RANK * group_feat;
        id<MTLBuffer> bwg = mkbuf(d, wg, (size_t)O_LORA_RANK * group_feat * sizeof(float));
        id<MTLCommandBuffer> cb=[P->queue commandBuffer];
        enc_matvec_f32(P, cb, bwg, bgv, bog, O_LORA_RANK, group_feat);
        [cb commit]; [cb waitUntilCompleted];
        memcpy(concat + (size_t)g * O_LORA_RANK, [bog contents], O_LORA_RANK * sizeof(float));
    }
    id<MTLBuffer> bconcat = mkbuf(d, concat, (size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    id<MTLBuffer> bout    = mkbuf(d, NULL, DIM * sizeof(float));
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec(P, cb, &aw->wo_b, bconcat, bout);                   // [4096] f32
      [cb commit]; [cb waitUntilCompleted]; }
    memcpy(out, [bout contents], DIM * sizeof(float));
    free(concat);
    return 0;
}

// ============================================================================
// Mixed attention: SWA raw KV (f16) + compressed KV (f32), CPU SDPA.
//
// Algorithm (matches ds4 layer_attention_mixed_one):
//   1. Q chain (same as f16kv): wq_a → q_norm → wq_b → per-head-norm → RoPE
//   2. KV chain (same as f16kv): wkv → kv_norm → RoPE → write to raw_kv_cache
//   3. SWA window: use last min(SWA_WINDOW, raw_cache_len) rows of raw_kv_cache
//   4. CPU SDPA over [swa_raw_rows | selected_comp_rows]:
//        - raw KV: f16→f32 on-the-fly
//        - comp KV: already f32
//        - sink: per-head scalar from attn_sink
//   5. Inverse RoPE on attn output
//   6. wo_a (grouped dense f32) + wo_b (affine quant)
// ============================================================================

// f16 → f32
static inline float f16_to_f32(uint16_t h) {
    float f;
    _Float16 hf = *(_Float16 *)&h;
    f = (float)hf;
    return f;
}

int mla_attention_decode_mixed(MlaPipes *P, const AttnWeights *aw,
                               const float *x, uint16_t *raw_kv_cache, int raw_cache_len,
                               int pos, const float *comp_kv, int n_comp,
                               const bool *comp_allowed, float *out) {
    id<MTLDevice> d = P->dev;
    int half = QK_ROPE_DIM / 2;
    float cosv[QK_ROPE_DIM / 2], sinv[QK_ROPE_DIM / 2];
    yarn_cos_sin(pos, cosv, sinv);

    // -----------------------------------------------------------------------
    // Q chain (identical to mla_attention_decode_f16kv): wq_a→q_norm→wq_b→norm→RoPE
    // -----------------------------------------------------------------------
    id<MTLBuffer> bx     = mkbuf(d, x, DIM * sizeof(float));
    id<MTLBuffer> bcos   = mkbuf(d, cosv, half * sizeof(float));
    id<MTLBuffer> bsin   = mkbuf(d, sinv, half * sizeof(float));
    id<MTLBuffer> bq_a   = mkbuf(d, NULL, Q_LORA_RANK * sizeof(float));
    id<MTLBuffer> bq_res = mkbuf(d, NULL, Q_LORA_RANK * sizeof(float));
    id<MTLBuffer> bq     = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(float));
    id<MTLBuffer> bq_n   = mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(float));
    id<MTLBuffer> bqn_w  = mkbuf(d, aw->q_norm, Q_LORA_RANK * sizeof(float));

    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec(P, cb, &aw->wq_a, bx, bq_a);
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows(P, cb, bq_a, bqn_w, bq_res, 1, Q_LORA_RANK, 1);
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec(P, cb, &aw->wq_b, bq_res, bq);
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows(P, cb, bq, NULL, bq_n, N_HEADS, HEAD_DIM, 0);
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope(P, cb, bq_n, bcos, bsin, N_HEADS, 0);
      [cb commit]; [cb waitUntilCompleted]; }

    // -----------------------------------------------------------------------
    // KV chain: wkv → kv_norm → RoPE → write f32 to raw_kv_cache
    // -----------------------------------------------------------------------
    id<MTLBuffer> bkv   = mkbuf(d, NULL, KV_LORA_RANK * sizeof(float));
    id<MTLBuffer> bkv_n = mkbuf(d, NULL, KV_LORA_RANK * sizeof(float));
    id<MTLBuffer> bkvn_w = mkbuf(d, aw->kv_norm, KV_LORA_RANK * sizeof(float));

    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec(P, cb, &aw->wkv, bx, bkv);
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rms_norm_rows(P, cb, bkv, bkvn_w, bkv_n, 1, KV_LORA_RANK, 1);
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_rope(P, cb, bkv_n, bcos, bsin, 1, 0);
      [cb commit]; [cb waitUntilCompleted]; }

    // Write f32 KV into cache as f16 at position (raw_cache_len - 1)
    {
        float *kvn_f32 = (float *)[bkv_n contents];
        uint16_t *dst = raw_kv_cache + (size_t)(raw_cache_len - 1) * KV_LORA_RANK;
        for (int d_ = 0; d_ < KV_LORA_RANK; d_++) {
            dst[d_] = f32_to_f16(kvn_f32[d_]);
        }
    }

    // -----------------------------------------------------------------------
    // CPU mixed SDPA: Q[N_HEADS, HEAD_DIM] × (raw_swa_KV f16 + comp_kv f32)
    //
    // SWA window: last min(SWA_WINDOW, raw_cache_len) rows of raw_kv_cache.
    // comp_kv: all rows where comp_allowed[r] == true (or all if null).
    // Sink: per-head scalar from aw->attn_sink.
    // -----------------------------------------------------------------------
    float *q_f32 = (float *)[bq_n contents];  // [N_HEADS, HEAD_DIM] f32

    // Count SWA rows
    const int swa_start = (raw_cache_len > SWA_WINDOW) ? (raw_cache_len - SWA_WINDOW) : 0;
    const int n_raw = raw_cache_len - swa_start;

    // Count allowed comp rows
    int n_comp_allowed = 0;
    if (n_comp > 0) {
        if (comp_allowed == NULL) {
            n_comp_allowed = n_comp;
        } else {
            for (int c = 0; c < n_comp; c++) if (comp_allowed[c]) n_comp_allowed++;
        }
    }

    const int n_total = n_raw + n_comp_allowed;
    const float kq_scale = 1.0f / sqrtf((float)HEAD_DIM);
    const float NEG_INF = -1e30f;

    // Output: [N_HEADS, HEAD_DIM]
    float *attn_out = (float *)malloc((size_t)N_HEADS * HEAD_DIM * sizeof(float));
    if (!attn_out) return -1;

    float *scores = (float *)malloc((size_t)(n_total + 1) * sizeof(float));
    if (!scores) { free(attn_out); return -1; }

    for (int h = 0; h < N_HEADS; h++) {
        const float *qh = q_f32 + (size_t)h * HEAD_DIM;
        float sink_score = (aw->attn_sink && h < N_HEADS) ? aw->attn_sink[h] : 0.0f;
        float max_score = sink_score;

        int idx = 0;

        // Score raw SWA rows (f16 → f32 on-the-fly, now uint16_t cache)
        for (int r = swa_start; r < raw_cache_len; r++, idx++) {
            const uint16_t *kv_f16 = raw_kv_cache + (size_t)r * KV_LORA_RANK;
            float dot = 0.0f;
            for (int d_ = 0; d_ < HEAD_DIM; d_++) dot += qh[d_] * f16_to_f32(kv_f16[d_]);
            scores[idx] = dot * kq_scale;
            if (scores[idx] > max_score) max_score = scores[idx];
        }

        // Score comp rows (f32)
        for (int c = 0; c < n_comp; c++) {
            if (comp_allowed != NULL && !comp_allowed[c]) continue;
            const float *kv = comp_kv + (size_t)c * COMP_HEAD_DIM;
            float dot = 0.0f;
            for (int d_ = 0; d_ < HEAD_DIM; d_++) dot += qh[d_] * kv[d_];
            scores[idx] = dot * kq_scale;
            if (scores[idx] > max_score) max_score = scores[idx];
            idx++;
        }

        float *oh = attn_out + (size_t)h * HEAD_DIM;
        memset(oh, 0, HEAD_DIM * sizeof(float));

        // Softmax denominator starts with sink
        float denom = expf(sink_score - max_score);

        idx = 0;
        // Accumulate raw SWA rows (f16 → f32 on-the-fly)
        for (int r = swa_start; r < raw_cache_len; r++, idx++) {
            const uint16_t *kv_f16 = raw_kv_cache + (size_t)r * KV_LORA_RANK;
            float w = expf(scores[idx] - max_score);
            denom += w;
            for (int d_ = 0; d_ < HEAD_DIM; d_++) oh[d_] += w * f16_to_f32(kv_f16[d_]);
        }

        // Accumulate comp rows
        for (int c = 0; c < n_comp; c++) {
            if (comp_allowed != NULL && !comp_allowed[c]) continue;
            const float *kv = comp_kv + (size_t)c * COMP_HEAD_DIM;
            float w = expf(scores[idx] - max_score);
            denom += w;
            for (int d_ = 0; d_ < HEAD_DIM; d_++) oh[d_] += w * kv[d_];
            idx++;
        }

        float inv = 1.0f / (denom + 1e-30f);
        for (int d_ = 0; d_ < HEAD_DIM; d_++) oh[d_] *= inv;
    }
    free(scores);

    // -----------------------------------------------------------------------
    // Inverse RoPE on attn_out [N_HEADS, HEAD_DIM]
    // -----------------------------------------------------------------------
    for (int h = 0; h < N_HEADS; h++) {
        float *ah = attn_out + (size_t)h * HEAD_DIM;
        int nope = QK_NOPE_DIM;
        for (int i = 0; i < half; i++) {
            int j0 = nope + i;
            int j1 = nope + i + half;
            float x0 = ah[j0], x1 = ah[j1];
            // Inverse: apply conjugate rotation (cos, -sin)
            ah[j0] = x0 * cosv[i] + x1 * sinv[i];
            ah[j1] = -x0 * sinv[i] + x1 * cosv[i];
        }
    }

    // -----------------------------------------------------------------------
    // wo_a (grouped dense f32 @ f32 attn_out) → concat → wo_b → out
    // -----------------------------------------------------------------------
    int heads_per_group = N_HEADS / O_GROUPS;
    int group_feat = heads_per_group * HEAD_DIM;
    float *concat = (float *)malloc((size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    if (!concat) { free(attn_out); return -1; }

    for (int g = 0; g < O_GROUPS; g++) {
        float *gv = (float *)malloc((size_t)group_feat * sizeof(float));
        if (!gv) { free(concat); free(attn_out); return -1; }
        for (int hh = 0; hh < heads_per_group; hh++)
            memcpy(gv + hh * HEAD_DIM, attn_out + (g * heads_per_group + hh) * HEAD_DIM,
                   HEAD_DIM * sizeof(float));
        id<MTLBuffer> bgv = mkbuf(d, gv, group_feat * sizeof(float));
        id<MTLBuffer> bog = mkbuf(d, NULL, O_LORA_RANK * sizeof(float));
        const float *wg = aw->wo_a_dense + (size_t)g * O_LORA_RANK * group_feat;
        id<MTLBuffer> bwg = mkbuf(d, wg, (size_t)O_LORA_RANK * group_feat * sizeof(float));
        id<MTLCommandBuffer> cb=[P->queue commandBuffer];
        enc_matvec_f32(P, cb, bwg, bgv, bog, O_LORA_RANK, group_feat);
        [cb commit]; [cb waitUntilCompleted];
        memcpy(concat + (size_t)g * O_LORA_RANK, [bog contents], O_LORA_RANK * sizeof(float));
        free(gv);
    }
    free(attn_out);

    id<MTLBuffer> bconcat = mkbuf(d, concat, (size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    id<MTLBuffer> bout    = mkbuf(d, NULL, DIM * sizeof(float));
    { id<MTLCommandBuffer> cb=[P->queue commandBuffer];
      enc_dequant_matvec(P, cb, &aw->wo_b, bconcat, bout);
      [cb commit]; [cb waitUntilCompleted]; }
    memcpy(out, [bout contents], DIM * sizeof(float));
    free(concat);
    return 0;
}

// ============================================================================
// mla_attention_prefill_bfloat — batch prefill for N tokens, bf16 end-to-end.
//
// Processes all n_tokens through the Q and KV chains (bf16), fills kv_cache,
// then dispatches mla_sdpa_prefill_bfloat for the full batch SDPA, then applies
// inverse RoPE + wo_a + wo_b to produce out_batch[n_tokens, DIM] in f32.
// ============================================================================
int mla_attention_prefill_bfloat(MlaPipes *P, const AttnWeights *aw,
                                  const uint16_t *x_batch, int n_tokens,
                                  uint16_t *kv_cache, int start_pos,
                                  float *out_batch) {
    id<MTLDevice> d = P->dev;
    const int half_rope = QK_ROPE_DIM / 2;

    // --- Allocate per-token cos/sin tables ---
    // cosv_all[t * half_rope .. (t+1)*half_rope)
    float *cosv_all = (float *)malloc((size_t)n_tokens * half_rope * sizeof(float));
    float *sinv_all = (float *)malloc((size_t)n_tokens * half_rope * sizeof(float));
    if (!cosv_all || !sinv_all) { free(cosv_all); free(sinv_all); return -1; }
    for (int t = 0; t < n_tokens; t++) {
        yarn_cos_sin(start_pos + t, cosv_all + t * half_rope, sinv_all + t * half_rope);
    }

    // --- Q chain: process ALL n_tokens, output q_all[n_tokens, N_HEADS, HEAD_DIM] bf16 ---
    // Layout: [n_tokens, N_HEADS, HEAD_DIM] bfloat16
    const size_t q_all_bytes = (size_t)n_tokens * N_HEADS * HEAD_DIM * sizeof(uint16_t);
    id<MTLBuffer> bq_all = [d newBufferWithLength:q_all_bytes options:MTLResourceStorageModeShared];

    for (int t = 0; t < n_tokens; t++) {
        const uint16_t *xt = x_batch + (size_t)t * DIM;
        id<MTLBuffer> bx     = mkbuf(d, xt, DIM * sizeof(uint16_t));
        id<MTLBuffer> bq_a   = [d newBufferWithLength:Q_LORA_RANK * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq_res = [d newBufferWithLength:Q_LORA_RANK * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq     = [d newBufferWithLength:(size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq_n   = [d newBufferWithLength:(size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bqn_w  = mkbuf(d, aw->q_norm, Q_LORA_RANK * sizeof(float));
        id<MTLBuffer> bcos   = mkbuf(d, cosv_all + t * half_rope, half_rope * sizeof(float));
        id<MTLBuffer> bsin   = mkbuf(d, sinv_all + t * half_rope, half_rope * sizeof(float));

        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_dequant_matvec_bf16in_bf16out(P, cb, &aw->wq_a, bx, bq_a);
          [cb commit]; [cb waitUntilCompleted]; }
        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_rms_norm_rows_bf16in_bf16out(P, cb, bq_a, bqn_w, bq_res, 1, Q_LORA_RANK, 1);
          [cb commit]; [cb waitUntilCompleted]; }
        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_dequant_matvec_bf16in_bf16out(P, cb, &aw->wq_b, bq_res, bq);
          [cb commit]; [cb waitUntilCompleted]; }
        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_rms_norm_rows_bf16in_bf16out(P, cb, bq, NULL, bq_n, N_HEADS, HEAD_DIM, 0);
          [cb commit]; [cb waitUntilCompleted]; }
        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_rope_bf16(P, cb, bq_n, bcos, bsin, N_HEADS, 0);
          [cb commit]; [cb waitUntilCompleted]; }

        // Copy q_n[t] into bq_all at offset t * N_HEADS * HEAD_DIM
        uint16_t *dst_q = (uint16_t *)[bq_all contents] + (size_t)t * N_HEADS * HEAD_DIM;
        memcpy(dst_q, [bq_n contents], (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    }

    // --- KV chain: process ALL n_tokens, fill kv_cache[start_pos..start_pos+n_tokens) ---
    // kv_all_buf: [n_tokens, KV_LORA_RANK] bfloat16 (for SDPA input)
    const size_t kv_all_bytes = (size_t)n_tokens * KV_LORA_RANK * sizeof(uint16_t);
    id<MTLBuffer> bkv_all = [d newBufferWithLength:kv_all_bytes options:MTLResourceStorageModeShared];

    for (int t = 0; t < n_tokens; t++) {
        const uint16_t *xt = x_batch + (size_t)t * DIM;
        id<MTLBuffer> bx     = mkbuf(d, xt, DIM * sizeof(uint16_t));
        id<MTLBuffer> bkv    = [d newBufferWithLength:KV_LORA_RANK * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bkv_n  = [d newBufferWithLength:KV_LORA_RANK * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bkvn_w = mkbuf(d, aw->kv_norm, KV_LORA_RANK * sizeof(float));
        id<MTLBuffer> bcos   = mkbuf(d, cosv_all + t * half_rope, half_rope * sizeof(float));
        id<MTLBuffer> bsin   = mkbuf(d, sinv_all + t * half_rope, half_rope * sizeof(float));

        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_dequant_matvec_bf16in_bf16out(P, cb, &aw->wkv, bx, bkv);
          [cb commit]; [cb waitUntilCompleted]; }
        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_rms_norm_rows_bf16in_bf16out(P, cb, bkv, bkvn_w, bkv_n, 1, KV_LORA_RANK, 1);
          [cb commit]; [cb waitUntilCompleted]; }
        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_rope_bf16(P, cb, bkv_n, bcos, bsin, 1, 0);
          [cb commit]; [cb waitUntilCompleted]; }

        // Write into kv_cache at row (start_pos + t)
        uint16_t *kv_n_ptr = (uint16_t *)[bkv_n contents];
        uint16_t *kv_dst = kv_cache + (size_t)(start_pos + t) * KV_LORA_RANK;
        memcpy(kv_dst, kv_n_ptr, KV_LORA_RANK * sizeof(uint16_t));

        // Also write into bkv_all for SDPA input
        uint16_t *kv_all_dst = (uint16_t *)[bkv_all contents] + (size_t)t * KV_LORA_RANK;
        memcpy(kv_all_dst, kv_n_ptr, KV_LORA_RANK * sizeof(uint16_t));
    }

    // --- SDPA: mla_sdpa_prefill_bfloat ---
    // Output: attn_out_all[n_tokens, N_HEADS, HEAD_DIM] bfloat16
    const size_t attn_all_bytes = (size_t)n_tokens * N_HEADS * HEAD_DIM * sizeof(uint16_t);
    id<MTLBuffer> battn_all = [d newBufferWithLength:attn_all_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bsink = mkbuf(d, aw->attn_sink, N_HEADS * sizeof(float));

    {
        const uint n_tok_u   = (uint)n_tokens;
        const uint n_heads_u = (uint)N_HEADS;
        const float sdpa_scale = 1.0f / sqrtf((float)HEAD_DIM);
        // Total tokens in kv = start_pos + n_tokens; but the kernel reads from bkv_all
        // which only has the prefill tokens [0..n_tokens). We pass the full kv_cache
        // buffer (which includes positions 0..start_pos+n_tokens-1), but for a fresh
        // sequence start_pos == 0 and kv_cache == bkv_all content anyway.
        // For simplicity, pass bkv_all (only the n_tokens just computed) and set n_tok
        // so causal masking works correctly relative to q_base.
        // NOTE: if start_pos > 0 (continuation), we need to pass the full kv_cache.
        // Use the full kv_cache buf for correctness.
        const size_t full_kv_bytes = (size_t)(start_pos + n_tokens) * KV_LORA_RANK * sizeof(uint16_t);
        id<MTLBuffer> bfull_kv = mkbuf(d, kv_cache, full_kv_bytes);

        // Reinterpret q layout: bq_all is [n_tokens, N_HEADS, HEAD_DIM]
        // kv layout: bfull_kv is [start_pos+n_tokens, HEAD_DIM] (KV_LORA_RANK == HEAD_DIM)
        // n_tok for SDPA = start_pos + n_tokens; q_base offset already in q layout
        // HOWEVER: the kernel assumes q[i] can attend to kv[0..i], so for continuation
        // we must pass n_tok = start_pos + n_tokens and adjust q buffer offset.
        // For now (start_pos == 0 typical prefill case), this is straightforward.
        const uint sdpa_n_tok = (uint)(start_pos + n_tokens);
        const uint q_offset_tok = (uint)start_pos; // first token of this batch in seq

        // We need to pass q starting at q_base offset; but q is [n_tokens, N_HEADS, HD]
        // and SDPA accesses q[q_base + qi, head, :]. So we need to place q at the right
        // position. Create a full-seq q buffer if start_pos > 0.
        id<MTLBuffer> bq_sdpa;
        if (start_pos == 0) {
            bq_sdpa = bq_all; // q is already indexed from 0
        } else {
            // Allocate a full [start_pos+n_tokens, N_HEADS, HD] buffer, place our q at offset
            const size_t full_q_bytes = (size_t)(start_pos + n_tokens) * N_HEADS * HEAD_DIM * sizeof(uint16_t);
            bq_sdpa = [d newBufferWithLength:full_q_bytes options:MTLResourceStorageModeShared];
            // Zero the pre-offset portion (causal mask will prevent those from being read)
            memset([bq_sdpa contents], 0, (size_t)start_pos * N_HEADS * HEAD_DIM * sizeof(uint16_t));
            memcpy((uint8_t *)[bq_sdpa contents] + (size_t)start_pos * N_HEADS * HEAD_DIM * sizeof(uint16_t),
                   [bq_all contents], (size_t)n_tokens * N_HEADS * HEAD_DIM * sizeof(uint16_t));
        }
        (void)q_offset_tok;

        id<MTLCommandBuffer> cb = [P->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:P->mla_sdpa_prefill_bfloat];
        [e setBuffer:bq_sdpa   offset:0 atIndex:0];
        [e setBuffer:bfull_kv  offset:0 atIndex:1];
        [e setBuffer:bsink     offset:0 atIndex:2];
        [e setBuffer:battn_all offset:0 atIndex:3];
        [e setBytes:&sdpa_n_tok   length:4 atIndex:4];
        [e setBytes:&n_heads_u    length:4 atIndex:5];
        [e setBytes:&sdpa_scale   length:4 atIndex:6];

        // Grid: [ceil(sdpa_n_tok/8) * N_HEADS, 1, 1] threadgroups, 32 threads each (1D flat)
        const uint n_blocks = (sdpa_n_tok + 7) / 8;
        const uint total_tg = n_blocks * N_HEADS;
        [e dispatchThreadgroups:MTLSizeMake(total_tg, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }

    // --- Inverse RoPE + wo_a + wo_b per token ---
    for (int t = 0; t < n_tokens; t++) {
        int pos = start_pos + t;
        float cosv[QK_ROPE_DIM / 2], sinv[QK_ROPE_DIM / 2];
        memcpy(cosv, cosv_all + t * half_rope, half_rope * sizeof(float));
        memcpy(sinv, sinv_all + t * half_rope, half_rope * sizeof(float));
        id<MTLBuffer> bcos = mkbuf(d, cosv, half_rope * sizeof(float));
        id<MTLBuffer> bsin = mkbuf(d, sinv, half_rope * sizeof(float));
        (void)pos;

        // attn[t] is at offset t*N_HEADS*HEAD_DIM within the full battn_all buffer
        // For start_pos > 0 prefill, SDPA wrote to offset (start_pos+t)*N_HEADS*HEAD_DIM.
        // Compute the source offset in battn_all:
        const size_t attn_t_offset_bf16 = (size_t)(start_pos + t) * N_HEADS * HEAD_DIM * sizeof(uint16_t);
        uint16_t *src_ptr = (uint16_t *)((uint8_t *)[battn_all contents] + attn_t_offset_bf16);

        // We need a separate buffer for enc_rope_bf16 (it modifies in place)
        id<MTLBuffer> battn_t = mkbuf(d, src_ptr, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));

        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_rope_bf16(P, cb, battn_t, bcos, bsin, N_HEADS, 1); // inverse RoPE
          [cb commit]; [cb waitUntilCompleted]; }

        // wo_a (grouped dense f32 @ bf16 attn) → concat (f32) → wo_b → out_t
        uint16_t *attn_bf16 = (uint16_t *)[battn_t contents];
        int heads_per_group = N_HEADS / O_GROUPS;
        int group_feat = heads_per_group * HEAD_DIM;
        float *concat = (float *)malloc((size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
        if (!concat) { free(cosv_all); free(sinv_all); return -1; }

        for (int g = 0; g < O_GROUPS; g++) {
            uint16_t *gv_bf16 = (uint16_t *)malloc((size_t)group_feat * sizeof(uint16_t));
            if (!gv_bf16) { free(concat); free(cosv_all); free(sinv_all); return -1; }
            for (int hh = 0; hh < heads_per_group; hh++)
                memcpy(gv_bf16 + hh * HEAD_DIM,
                       attn_bf16 + (g * heads_per_group + hh) * HEAD_DIM,
                       HEAD_DIM * sizeof(uint16_t));
            id<MTLBuffer> bgv = mkbuf(d, gv_bf16, group_feat * sizeof(uint16_t));
            id<MTLBuffer> bog = mkbuf(d, NULL, O_LORA_RANK * sizeof(float));
            const float *wg = aw->wo_a_dense + (size_t)g * O_LORA_RANK * group_feat;
            id<MTLBuffer> bwg = mkbuf(d, wg, (size_t)O_LORA_RANK * group_feat * sizeof(float));
            id<MTLCommandBuffer> cb = [P->queue commandBuffer];
            enc_matvec_f32_bf16in(P, cb, bwg, bgv, bog, O_LORA_RANK, group_feat);
            [cb commit]; [cb waitUntilCompleted];
            memcpy(concat + (size_t)g * O_LORA_RANK, [bog contents], O_LORA_RANK * sizeof(float));
            free(gv_bf16);
        }

        id<MTLBuffer> bconcat = mkbuf(d, concat, (size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
        id<MTLBuffer> bout_t  = mkbuf(d, NULL, DIM * sizeof(float));
        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_dequant_matvec(P, cb, &aw->wo_b, bconcat, bout_t);
          [cb commit]; [cb waitUntilCompleted]; }

        memcpy(out_batch + (size_t)t * DIM, [bout_t contents], DIM * sizeof(float));
        free(concat);
    }

    free(cosv_all);
    free(sinv_all);
    return 0;
}
