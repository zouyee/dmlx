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
    id<MTLBuffer> wo_a_grp[8];   // 8 × 16MB = 128MB total per layer (f32 dense, fallback)
    id<MTLBuffer> wo_a_q8_gpu[8]; // 8 × 4.5MB = 36MB total per layer (Q8_0 quantized)
    // wo_a packed affine 4-bit (native format): 8 × (2MB + 2×256KB) per layer.
    // Preferred path — 6.4x less per-token weight traffic than f32 dense.
    id<MTLBuffer> wo_a_pack[8], wo_a_sc[8], wo_a_bi[8];
    // Contiguous whole-weight views of the same data for the fused grp8
    // kernel (one dispatch instead of 8 + blits + concat).
    id<MTLBuffer> wo_a_pack_all, wo_a_sc_all, wo_a_bi_all;
    // Persistent scratch buffers for decode pass (reused every forward call)
    // Eliminates ~18 MTLBuffer allocations per layer per token
    id<MTLBuffer> scr_q_a;       // [Q_LORA_RANK] bf16   — wq_a output
    id<MTLBuffer> scr_q_res;     // [Q_LORA_RANK] bf16   — q_norm output
    id<MTLBuffer> scr_q;         // [N_HEADS*HEAD_DIM] bf16 — wq_b output
    id<MTLBuffer> scr_q_n;       // [N_HEADS*HEAD_DIM] bf16 — per-head-norm + RoPE
    id<MTLBuffer> scr_kv;        // [KV_LORA_RANK] bf16  — wkv output
    id<MTLBuffer> scr_kv_n;      // [KV_LORA_RANK] bf16  — kv_norm + RoPE
    id<MTLBuffer> scr_attn;      // [N_HEADS*HEAD_DIM] bf16 — SDPA output + RoPE
    id<MTLBuffer> scr_concat;    // [O_GROUPS*O_LORA_RANK] f32 — wo_a concat
    id<MTLBuffer> scr_out;       // [DIM] f32             — wo_b output
    id<MTLBuffer> scr_bog[8];    // [O_LORA_RANK] f32 ×8 — wo_a group outputs
    id<MTLBuffer> scr_bgv[8];    // [group_feat] bf16 ×8  — wo_a group inputs
    id<MTLBuffer> scr_cos;       // [QK_ROPE_DIM/2] f32  — RoPE cos (persistent, memcpy per pos)
    id<MTLBuffer> scr_sin;       // [QK_ROPE_DIM/2] f32  — RoPE sin (persistent, memcpy per pos)
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
    // Zero-initialize without memset (struct contains ObjC id fields — use bzero on raw bytes)
    bzero((void *)c, sizeof(*c));
    c->owner = (const void *)aw;

    // Use newBufferWithBytes (copy) for attention weights. NativeLoader data is not
    // guaranteed to be page-aligned, which newBufferWithBytesNoCopy requires on macOS.
    // Copying guarantees correct GPU reads even if the source pointer is unaligned.
    // Future optimization: page-align weight allocations in native_loader and revert to NoCopy.
    #define MKNC(ptr, sz) [d newBufferWithBytes:(void*)(ptr) length:(sz) \
                             options:MTLResourceStorageModeShared]
    {
        size_t pw; int ng; size_t sc;
        // wq_a: [1024, 4096] affine 4-bit → packed=2MB, scales/biases=2×256KB
        pw = (size_t)aw->wq_a.out_dim * (aw->wq_a.in_dim / 8) * sizeof(uint32_t);
        ng = aw->wq_a.in_dim / aw->wq_a.group_size; sc = (size_t)aw->wq_a.out_dim * ng * sizeof(float);
        c->wq_a_pack = MKNC(aw->wq_a.packed, pw);
        c->wq_a_sc   = MKNC(aw->wq_a.scales, sc);
        c->wq_a_bi   = MKNC(aw->wq_a.biases, sc);
        // wq_b: [32768, 1024] affine 4-bit → packed=16MB, scales/biases=2×2MB
        pw = (size_t)aw->wq_b.out_dim * (aw->wq_b.in_dim / 8) * sizeof(uint32_t);
        ng = aw->wq_b.in_dim / aw->wq_b.group_size; sc = (size_t)aw->wq_b.out_dim * ng * sizeof(float);
        c->wq_b_pack = MKNC(aw->wq_b.packed, pw);
        c->wq_b_sc   = MKNC(aw->wq_b.scales, sc);
        c->wq_b_bi   = MKNC(aw->wq_b.biases, sc);
        // wkv: [512, 4096] affine 4-bit → packed=1MB, scales/biases=2×128KB
        pw = (size_t)aw->wkv.out_dim * (aw->wkv.in_dim / 8) * sizeof(uint32_t);
        ng = aw->wkv.in_dim / aw->wkv.group_size; sc = (size_t)aw->wkv.out_dim * ng * sizeof(float);
        c->wkv_pack = MKNC(aw->wkv.packed, pw);
        c->wkv_sc   = MKNC(aw->wkv.scales,  sc);
        c->wkv_bi   = MKNC(aw->wkv.biases,  sc);
        // wo_b: [4096, 8192] affine 4-bit → packed=16MB, scales/biases=2×2MB
        pw = (size_t)aw->wo_b.out_dim * (aw->wo_b.in_dim / 8) * sizeof(uint32_t);
        ng = aw->wo_b.in_dim / aw->wo_b.group_size; sc = (size_t)aw->wo_b.out_dim * ng * sizeof(float);
        c->wo_b_pack = MKNC(aw->wo_b.packed, pw);
        c->wo_b_sc   = MKNC(aw->wo_b.scales, sc);
        c->wo_b_bi   = MKNC(aw->wo_b.biases, sc);
    }
    // wo_a: native packed affine 4-bit per group (preferred) — created whenever
    // the loader kept wo_a packed. The f32 dense (128MB/layer) and Q8_0 fallback
    // buffers are only created when packed is unavailable or explicitly forced.
    {
        int heads_per_group = N_HEADS / O_GROUPS;
        int group_feat = heads_per_group * HEAD_DIM;  // 8×512=4096
        const int have_woa_packed = (aw->wo_a.packed != NULL && aw->wo_a.out_dim > 0);
        if (have_woa_packed) {
            int ng = group_feat / aw->wo_a.group_size;  // 4096/64 = 64
            size_t grp_pack = (size_t)O_LORA_RANK * (group_feat / 8) * sizeof(uint32_t);
            size_t grp_sb   = (size_t)O_LORA_RANK * ng * sizeof(float);
            for (int g = 0; g < O_GROUPS; g++) {
                c->wo_a_pack[g] = MKNC(aw->wo_a.packed + (size_t)g * O_LORA_RANK * (group_feat / 8), grp_pack);
                c->wo_a_sc[g]   = MKNC(aw->wo_a.scales + (size_t)g * O_LORA_RANK * ng, grp_sb);
                c->wo_a_bi[g]   = MKNC(aw->wo_a.biases + (size_t)g * O_LORA_RANK * ng, grp_sb);
            }
            // Contiguous whole-weight views for the fused grp8 kernel
            c->wo_a_pack_all = MKNC(aw->wo_a.packed, (size_t)O_GROUPS * grp_pack);
            c->wo_a_sc_all   = MKNC(aw->wo_a.scales, (size_t)O_GROUPS * grp_sb);
            c->wo_a_bi_all   = MKNC(aw->wo_a.biases, (size_t)O_GROUPS * grp_sb);
        }
        const int want_f32 = getenv("DMLX_WOA_F32") != NULL;
        const int want_q8  = getenv("DMLX_USE_Q8_WOA") != NULL;
        if (aw->wo_a_dense && (!have_woa_packed || want_f32 || want_q8)) {
        size_t grp_sz = (size_t)O_LORA_RANK * group_feat * sizeof(float);  // 16MB each (f32)
        // Q8_0: 32 elements per block, 36 bytes per block (4B scale + 32B int8)
        int nb = group_feat / 32;  // 128 blocks per row
        size_t q8_grp_sz = (size_t)O_LORA_RANK * nb * 36;  // ~4.5MB per group
        for (int g = 0; g < O_GROUPS; g++) {
            const float *wg = aw->wo_a_dense + (size_t)g * O_LORA_RANK * group_feat;
            c->wo_a_grp[g] = MKNC(wg, grp_sz);  // f32 fallback

            // Quantize to Q8_0 format
            id<MTLBuffer> q8_buf = [d newBufferWithLength:q8_grp_sz options:MTLResourceStorageModeShared];
            uint8_t *q8_raw = (uint8_t *)[q8_buf contents];
            for (int r = 0; r < O_LORA_RANK; r++) {
                const float *row = wg + (size_t)r * group_feat;
                for (int b = 0; b < nb; b++) {
                    const float *x = row + b * 32;
                    float amax = 0.0f;
                    for (int i = 0; i < 32; i++) { float ax = fabsf(x[i]); if (ax > amax) amax = ax; }
                    float d = amax / 127.0f;
                    float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
                    // Write scale (4 bytes)
                    memcpy(q8_raw, &d, 4); q8_raw += 4;
                    // Write quantized values (32 bytes)
                    for (int i = 0; i < 32; i++) {
                        float v = x[i] * id;
                        int q = (int)roundf(v);        // round to nearest integer
                        if (q > 127) q = 127;
                        if (q < -127) q = -127;
                        *q8_raw++ = (uint8_t)((int8_t)q);
                    }
                }
            }
            c->wo_a_q8_gpu[g] = q8_buf;
        }
        } // end f32/Q8 fallback block
    }
    #undef MKNC
    // q_norm, kv_norm, attn_sink (tiny, always copied for alignment safety)
    #define MKNC(ptr, sz) [d newBufferWithBytes:(void*)(ptr) length:(sz) \
                             options:MTLResourceStorageModeShared]
    c->q_norm_buf    = MKNC(aw->q_norm,    Q_LORA_RANK * sizeof(float));
    c->kv_norm_buf   = MKNC(aw->kv_norm,   KV_LORA_RANK * sizeof(float));
    c->attn_sink_buf = MKNC(aw->attn_sink, N_HEADS * sizeof(float));
    #undef MKNC

    // Persistent scratch buffers: allocated once, reused every decode call.
    // Total: ~18 + 8*2 = ~34 allocations eliminated per layer per token.
    #define MKSCR(sz) [d newBufferWithLength:(sz) options:MTLResourceStorageModeShared]
    c->scr_q_a    = MKSCR((size_t)Q_LORA_RANK * sizeof(uint16_t));
    c->scr_q_res  = MKSCR((size_t)Q_LORA_RANK * sizeof(uint16_t));
    c->scr_q      = MKSCR((size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    c->scr_q_n    = MKSCR((size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    c->scr_kv     = MKSCR((size_t)KV_LORA_RANK * sizeof(uint16_t));
    c->scr_kv_n   = MKSCR((size_t)KV_LORA_RANK * sizeof(uint16_t));
    c->scr_attn   = MKSCR((size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    c->scr_concat = MKSCR((size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    c->scr_out    = MKSCR((size_t)DIM * sizeof(float));
    {
        int group_feat = (N_HEADS / O_GROUPS) * HEAD_DIM;  // 8*512=4096
        for (int g = 0; g < O_GROUPS; g++) {
            c->scr_bog[g] = MKSCR((size_t)O_LORA_RANK * sizeof(float));
            c->scr_bgv[g] = MKSCR((size_t)group_feat * sizeof(uint16_t));
        }
    }
    c->scr_cos = MKSCR((size_t)(QK_ROPE_DIM / 2) * sizeof(float));
    c->scr_sin = MKSCR((size_t)(QK_ROPE_DIM / 2) * sizeof(float));
    #undef MKSCR

    return c;
}

// Thin wrapper for enc_dequant_matvec_* using pre-cached buffers.
// These shadow the helpers below so the bf16 function can use cached buffers.
static id<MTLComputeCommandEncoder> attention_stage_encoder(
    MoEInferEngine *engine, id<MTLCommandBuffer> cb, DmlxAttentionStage stage) {
    if (!moe_infer_attention_stage_recording_enabled(engine))
        return [cb computeCommandEncoder];
    return (__bridge id<MTLComputeCommandEncoder>)
        moe_infer_attention_stage_encoder(engine, (__bridge void *)cb, stage);
}

static void enc_dq_bf16_cached(MlaPipes *P, id<MTLCommandBuffer> cb,
                               MoEInferEngine *stage_engine, DmlxAttentionStage stage,
                               id<MTLBuffer> bw, id<MTLBuffer> bs, id<MTLBuffer> bb,
                               int out_dim, int in_dim, int group_size,
                               id<MTLBuffer> x, id<MTLBuffer> out,
                               id<MTLComputePipelineState> pipe) {
    // NOTE: v2 (SIMD-parallel) dispatch was tried for attention matmuls but regresses
    // performance due to high TG-scheduling overhead (e.g. wq_b needs 16384 TGs vs 128
    // for naive). Apple GPU L2 cache (32-64MB) absorbs non-coalesced reads for these
    // relatively small matrices. Always use naive 1-thread-per-row dispatch here.
    id<MTLComputeCommandEncoder> e = attention_stage_encoder(stage_engine, cb, stage);
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

// Batch rows share one weight matrix and one encoder dispatch. The underlying
// kernel uses grid y as the input/output row; scalar callers continue to use
// the same pipeline with a y dimension of one.
static void enc_dq_bf16_cached_batch(MlaPipes *P, id<MTLCommandBuffer> cb,
                                     id<MTLBuffer> bw, id<MTLBuffer> bs, id<MTLBuffer> bb,
                                     int out_dim, int in_dim, int group_size,
                                     id<MTLBuffer> workspace,
                                     NSUInteger x_offset, NSUInteger out_offset,
                                     uint batch_rows) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->dequant_matvec_affine_bf16in_bf16out];
    [e setBuffer:bw offset:0 atIndex:0];
    [e setBuffer:bs offset:0 atIndex:1];
    [e setBuffer:bb offset:0 atIndex:2];
    [e setBuffer:workspace offset:x_offset atIndex:3];
    [e setBuffer:workspace offset:out_offset atIndex:4];
    uint od = (uint)out_dim, id_ = (uint)in_dim, gs = (uint)group_size;
    [e setBytes:&od length:4 atIndex:5];
    [e setBytes:&id_ length:4 atIndex:6];
    [e setBytes:&gs length:4 atIndex:7];
    [e dispatchThreads:MTLSizeMake((NSUInteger)out_dim, batch_rows, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
}

static void enc_rms_norm_rows_bf16_batch_offset(
    MlaPipes *P, id<MTLCommandBuffer> cb,
    id<MTLBuffer> workspace, NSUInteger input_offset,
    id<MTLBuffer> weight, NSUInteger output_offset,
    uint n_rows, uint row_dim, int has_weight) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rms_norm_rows_bf16in_bf16out];
    [e setBuffer:workspace offset:input_offset atIndex:0];
    [e setBuffer:(weight ? weight : workspace) offset:0 atIndex:1];
    [e setBuffer:workspace offset:output_offset atIndex:2];
    uint rd = row_dim;
    float eps = 1e-6f;
    uint hw = has_weight ? 1u : 0u;
    [e setBytes:&rd length:4 atIndex:3];
    [e setBytes:&eps length:4 atIndex:4];
    [e setBytes:&hw length:4 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
}

static void enc_rope_bf16_batch_offset(
    MlaPipes *P, id<MTLCommandBuffer> cb, id<MTLBuffer> workspace,
    NSUInteger values_offset, NSUInteger cos_offset, NSUInteger sin_offset,
    uint batch_rows, uint n_heads, uint inverse) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->rope_tail_interleaved_bf16];
    [e setBuffer:workspace offset:values_offset atIndex:0];
    [e setBuffer:workspace offset:cos_offset atIndex:1];
    [e setBuffer:workspace offset:sin_offset atIndex:2];
    uint nh = n_heads, hd = HEAD_DIM, nd = QK_NOPE_DIM, rd = QK_ROPE_DIM;
    [e setBytes:&nh length:4 atIndex:3];
    [e setBytes:&hd length:4 atIndex:4];
    [e setBytes:&nd length:4 atIndex:5];
    [e setBytes:&rd length:4 atIndex:6];
    [e setBytes:&inverse length:4 atIndex:7];
    const uint half = QK_ROPE_DIM / 2;
    [e dispatchThreads:MTLSizeMake(n_heads * half, batch_rows, 1)
        threadsPerThreadgroup:MTLSizeMake(half, 1, 1)];
    [e endEncoding];
}

// --- YaRN tail RoPE cos/sin for a given position (matches DSV4YarnRoPE) ---
static void yarn_cos_sin(int pos, float *cos_out, float *sin_out) {
    const float base = 10000.0f, factor = 16.0f, beta_fast = 32.0f, beta_slow = 1.0f;
    const float orig_max = 65536.0f, PI = 3.14159265358979323846f;
    int half = QK_ROPE_DIM / 2;
    // Adjusted frequencies are position-independent; compute once. Concurrent
    // first-touch would write identical values, so a plain flag is safe here.
    static float freq_cache[QK_ROPE_DIM / 2];
    static int freq_ready = 0;
    if (!freq_ready) {
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
            freq_cache[i] = freq / factor * (1.0f - smooth) + freq * smooth;
        }
        freq_ready = 1;
    }
    for (int i = 0; i < half; i++) {
        cos_out[i] = cosf(pos * freq_cache[i]);
        sin_out[i] = sinf(pos * freq_cache[i]);
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
                                             MoEInferEngine *stage_engine, DmlxAttentionStage stage,
                                             id<MTLBuffer> x, id<MTLBuffer> weight, id<MTLBuffer> out,
                                             uint n_rows, uint row_dim, int has_weight) {
    id<MTLComputeCommandEncoder> e = attention_stage_encoder(stage_engine, cb, stage);
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

static void enc_rope_bf16(MlaPipes *P, id<MTLCommandBuffer> cb,
                          MoEInferEngine *stage_engine, DmlxAttentionStage stage,
                          id<MTLBuffer> q,
                          id<MTLBuffer> cosb, id<MTLBuffer> sinb, uint n_heads, uint inverse) {
    id<MTLComputeCommandEncoder> e = attention_stage_encoder(stage_engine, cb, stage);
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

// ds4 kernel_mul_mv_q8_0_f32 dispatch for Q8_0-quantized wo_a.
// NR0=2 rows per TG, NSG=4 simdgroups, 256B threadgroup memory.
static void enc_matvec_q8_0(MlaPipes *P, id<MTLCommandBuffer> cb,
                            id<MTLBuffer> W, id<MTLBuffer> x, id<MTLBuffer> out,
                            uint out_dim, uint in_dim) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:P->matvec_q8_0_f32];
    [e setBuffer:W offset:0 atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:out offset:0 atIndex:2];
    [e setBytes:&out_dim length:4 atIndex:3];
    [e setBytes:&in_dim length:4 atIndex:4];
    [e setThreadgroupMemoryLength:256 atIndex:0];  // 32*2*4 bytes
    uint num_tgs = (out_dim + 1) / 2;  // NR0=2 rows per threadgroup
    [e dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];  // (SIMD width=32, NSG=4, 1)
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
    // wo_a_dense is not loaded by default anymore (packed path);
    // this legacy f32-dense path requires it (DMLX_WOA_F32=1).
    if (!aw->wo_a_dense) return -2;
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

int mla_attention_select_bf16_identity(
    bool has_persistent_kv, bool has_external_cb, bool stage_time,
    DmlxMlaPathIdentity *out_identity) {
    if (out_identity) memset(out_identity, 0, sizeof(*out_identity));
    if (!out_identity || (!has_persistent_kv && has_external_cb)) return -1;
    DmlxMlaPathIdentity candidate = {0};
    if (has_persistent_kv) {
        candidate.flags = DMLX_MLA_PATH_PERSISTENT_KV;
        if (has_external_cb) {
            candidate.flags |= DMLX_MLA_PATH_EXTERNAL_CB_REQUESTED;
            if (stage_time) {
                candidate.path = DMLX_MLA_PATH_PERSISTENT_INTERNAL_SPLIT;
                candidate.flags |= DMLX_MLA_PATH_STAGE_TIME_SPLIT;
            } else {
                candidate.path = DMLX_MLA_PATH_PERSISTENT_EXTERNAL;
                candidate.flags |= DMLX_MLA_PATH_EXTERNAL_CB_USED;
            }
        } else {
            candidate.path = DMLX_MLA_PATH_PERSISTENT_INTERNAL;
        }
    } else {
        candidate.path = DMLX_MLA_PATH_FALLBACK_TEMP_KV;
        candidate.flags = DMLX_MLA_PATH_FALLBACK;
    }
    *out_identity = candidate;
    return 0;
}

// --- BF16 variant of mla_attention_decode ---
int mla_attention_decode_bf16(MlaPipes *P, const AttnWeights *aw,
                              const uint16_t *x, uint16_t *kv_cache, int cache_len,
                              int pos, float *out, void *kv_cache_gpu_buf, void *x_gpu_buf,
                              void *external_cb1, void *out_gpu_buf,
                              DmlxMlaPathIdentity *out_identity,
                              MoEInferEngine *stage_engine) {
    if (out_identity) memset(out_identity, 0, sizeof(*out_identity));
    if (!out_identity || (!kv_cache_gpu_buf && external_cb1)) return -1;
    const int stage_time = getenv("MF_STAGE_TIME") != NULL;
    DmlxMlaPathIdentity identity_candidate;
    if (mla_attention_select_bf16_identity(
            kv_cache_gpu_buf != NULL, external_cb1 != NULL, stage_time != 0,
            &identity_candidate) != 0) return -1;
    @autoreleasepool {
    id<MTLDevice> d = P->dev;
    int half = QK_ROPE_DIM / 2;
    float cosv[QK_ROPE_DIM / 2], sinv[QK_ROPE_DIM / 2];
    yarn_cos_sin(pos, cosv, sinv);
    // MF_STAGE_TIME: split the merged CB into per-stage CBs (commit+wait each)
    // and print per-stage GPU wait times. Same kernels, same order — numerics
    // unchanged, only extra syncs. Caller (engine.c) must gate its own commit.

    // === Get (or create) persistent GPU buffers for fixed attention weights ===
    AttnBufCache *abc = attn_buf_cache_get(d, aw);

    id<MTLBuffer> bx     = (x_gpu_buf && !x) ? (__bridge id<MTLBuffer>)x_gpu_buf
                               : mkbuf(d, x, DIM * sizeof(uint16_t));
    id<MTLBuffer> bcos, bsin;
    if (abc && abc->scr_cos) {
        bcos = abc->scr_cos; bsin = abc->scr_sin;
        memcpy([bcos contents], cosv, half * sizeof(float));
        memcpy([bsin contents], sinv, half * sizeof(float));
    } else {
        bcos = mkbuf(d, cosv, half * sizeof(float));
        bsin = mkbuf(d, sinv, half * sizeof(float));
    }
    // Use persistent scratch buffers from AttnBufCache when available
    id<MTLBuffer> bq_a   = (abc && abc->scr_q_a)   ? abc->scr_q_a   : mkbuf(d, NULL, Q_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bq_res = (abc && abc->scr_q_res)  ? abc->scr_q_res : mkbuf(d, NULL, Q_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bq     = (abc && abc->scr_q)      ? abc->scr_q     : mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    id<MTLBuffer> bq_n   = (abc && abc->scr_q_n)    ? abc->scr_q_n   : mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));
    id<MTLBuffer> bkv    = (abc && abc->scr_kv)     ? abc->scr_kv    : mkbuf(d, NULL, KV_LORA_RANK * sizeof(uint16_t));
    id<MTLBuffer> bkv_n  = (abc && abc->scr_kv_n)   ? abc->scr_kv_n  : mkbuf(d, NULL, KV_LORA_RANK * sizeof(uint16_t));

    // Use persistent norm/sink buffers if available, else fall back to mkbuf
    id<MTLBuffer> bqn_w  = abc ? abc->q_norm_buf    : mkbuf(d, aw->q_norm,    Q_LORA_RANK * sizeof(float));
    id<MTLBuffer> bkvn_w = abc ? abc->kv_norm_buf   : mkbuf(d, aw->kv_norm,   KV_LORA_RANK * sizeof(float));
    id<MTLBuffer> bsink  = abc ? abc->attn_sink_buf : mkbuf(d, aw->attn_sink, N_HEADS * sizeof(float));

    // === CB1: Q chain + KV chain + KV blit + SDPA merged into ONE command buffer ===
    // (GPU path: 11 encoders, 1 wait; CPU fallback: 8 encoders + separate CB2)
    id<MTLCommandBuffer> cb1 = (external_cb1 && !stage_time) ? (__bridge id<MTLCommandBuffer>)external_cb1
                                             : [P->queue commandBuffer];
        // Q chain — use cached weight buffers when available (wq_a, wq_b, wkv may be nil if memory-limited)
        if (abc && abc->wq_a_pack) {
            enc_dq_bf16_cached(P, cb1, stage_engine, DMLX_ATTENTION_STAGE_WQ_A, abc->wq_a_pack, abc->wq_a_sc, abc->wq_a_bi,
                               aw->wq_a.out_dim, aw->wq_a.in_dim, aw->wq_a.group_size, bx, bq_a,
                               P->dequant_matvec_affine_bf16in_bf16out);
        } else {
            enc_dequant_matvec_bf16in_bf16out(P, cb1, &aw->wq_a, bx, bq_a);
        }
        enc_rms_norm_rows_bf16in_bf16out(P, cb1, stage_engine, DMLX_ATTENTION_STAGE_Q_RMS, bq_a, bqn_w, bq_res, 1, Q_LORA_RANK, 1);
        if (abc && abc->wq_b_pack) {
            enc_dq_bf16_cached(P, cb1, stage_engine, DMLX_ATTENTION_STAGE_WQ_B, abc->wq_b_pack, abc->wq_b_sc, abc->wq_b_bi,
                               aw->wq_b.out_dim, aw->wq_b.in_dim, aw->wq_b.group_size, bq_res, bq,
                               P->dequant_matvec_affine_bf16in_bf16out);
        } else {
            enc_dequant_matvec_bf16in_bf16out(P, cb1, &aw->wq_b, bq_res, bq);
        }
        enc_rms_norm_rows_bf16in_bf16out(P, cb1, stage_engine, DMLX_ATTENTION_STAGE_Q_HEAD_RMS, bq, NULL, bq_n, N_HEADS, HEAD_DIM, 0);
        enc_rope_bf16(P, cb1, stage_engine, DMLX_ATTENTION_STAGE_Q_ROPE, bq_n, bcos, bsin, N_HEADS, 0);
        // KV chain
        if (abc && abc->wkv_pack) {
            enc_dq_bf16_cached(P, cb1, stage_engine, DMLX_ATTENTION_STAGE_WKV, abc->wkv_pack, abc->wkv_sc, abc->wkv_bi,
                               aw->wkv.out_dim, aw->wkv.in_dim, aw->wkv.group_size, bx, bkv,
                               P->dequant_matvec_affine_bf16in_bf16out);
        } else {
            enc_dequant_matvec_bf16in_bf16out(P, cb1, &aw->wkv, bx, bkv);
        }
        enc_rms_norm_rows_bf16in_bf16out(P, cb1, stage_engine, DMLX_ATTENTION_STAGE_KV_RMS, bkv, bkvn_w, bkv_n, 1, KV_LORA_RANK, 1);
        enc_rope_bf16(P, cb1, stage_engine, DMLX_ATTENTION_STAGE_KV_ROPE, bkv_n, bcos, bsin, 1, 0);

        // === KV cache update + SDPA: merged into CB1 when kv_cache_gpu_buf is available ===
        // This eliminates the CPU round-trip (GPU→CPU→GPU) between CB1 and CB2,
        // saving ~1 GPU sync overhead (~8ms/layer).
        id<MTLBuffer> bkvcache_persistent = (__bridge id<MTLBuffer>)kv_cache_gpu_buf;
        id<MTLBuffer> battn_scr = (abc && abc->scr_attn) ? abc->scr_attn : mkbuf(d, NULL, (size_t)N_HEADS * HEAD_DIM * sizeof(uint16_t));

        if (bkvcache_persistent) {
            // GPU bf16→f16 conversion: write bkv_n into kvcache[cache_len-1] as f16.
            // Avoids CPU round-trip (GPU→CPU→GPU) saving ~1 waitUntilCompleted per layer.
            {
                id<MTLComputeCommandEncoder> e = attention_stage_encoder(
                    stage_engine, cb1, DMLX_ATTENTION_STAGE_KV_UPDATE);
                [e setComputePipelineState:P->bf16_to_f16_row];
                [e setBuffer:bkv_n                 offset:0 atIndex:0];
                [e setBuffer:bkvcache_persistent   offset:0 atIndex:1];
                uint row = (uint)(cache_len - 1);
                uint rank = KV_LORA_RANK;
                [e setBytes:&row  length:4 atIndex:2];
                [e setBytes:&rank length:4 atIndex:3];
                [e dispatchThreads:MTLSizeMake(KV_LORA_RANK,1,1)
                   threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            }
            // SDPA reads the full kvcache (including the new row just written)
            {
                id<MTLComputeCommandEncoder> e = attention_stage_encoder(
                    stage_engine, cb1, DMLX_ATTENTION_STAGE_SDPA);
                [e setComputePipelineState:P->mla_sdpa_decode_bfloat];
                [e setBuffer:bq_n                 offset:0 atIndex:0];
                [e setBuffer:bkvcache_persistent  offset:0 atIndex:1];
                [e setBuffer:bsink                offset:0 atIndex:2];
                [e setBuffer:battn_scr            offset:0 atIndex:3];
                uint nh=N_HEADS, hd=HEAD_DIM, nk=(uint)cache_len;
                float scale=1.0f/sqrtf((float)HEAD_DIM);
                [e setBytes:&nh    length:4 atIndex:4];
                [e setBytes:&hd    length:4 atIndex:5];
                [e setBytes:&nk    length:4 atIndex:6];
                [e setBytes:&scale length:4 atIndex:7];
                [e dispatchThreadgroups:MTLSizeMake(N_HEADS,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                [e endEncoding];
            }
            enc_rope_bf16(P, cb1, stage_engine, DMLX_ATTENTION_STAGE_INV_Q_ROPE, battn_scr, bcos, bsin, N_HEADS, 1);
            if (!external_cb1 || stage_time) {
                double s0 = 0;
                if (stage_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); s0 = ts.tv_sec*1e9+ts.tv_nsec; }
                [cb1 commit]; [cb1 waitUntilCompleted];
                if (stage_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
                    fprintf(stderr, "[STAGE] pos=%d qkv_sdpa=%.2f\n", pos, (ts.tv_sec*1e9+ts.tv_nsec-s0)/1e6); }
            }
            // No CPU KV copy needed — bf16_to_f16_row kernel already wrote to cache
        } else {
            // Fallback: CPU KV copy between CB1 and CB2
            [cb1 commit]; [cb1 waitUntilCompleted];
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
            id<MTLBuffer> bkvcache_tmp = mkbuf(d, kv_cache, (size_t)cache_len * KV_LORA_RANK * sizeof(uint16_t));
            id<MTLCommandBuffer> cb2 = [P->queue commandBuffer];
            {
                id<MTLComputeCommandEncoder> e = [cb2 computeCommandEncoder];
                [e setComputePipelineState:P->mla_sdpa_decode_bfloat];
                [e setBuffer:bq_n           offset:0 atIndex:0];
                [e setBuffer:bkvcache_tmp   offset:0 atIndex:1];
                [e setBuffer:bsink          offset:0 atIndex:2];
                [e setBuffer:battn_scr      offset:0 atIndex:3];
                uint nh=N_HEADS, hd=HEAD_DIM, nk=(uint)cache_len;
                float scale=1.0f/sqrtf((float)HEAD_DIM);
                [e setBytes:&nh    length:4 atIndex:4];
                [e setBytes:&hd    length:4 atIndex:5];
                [e setBytes:&nk    length:4 atIndex:6];
                [e setBytes:&scale length:4 atIndex:7];
                [e dispatchThreadgroups:MTLSizeMake(N_HEADS,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                [e endEncoding];
            }
            enc_rope_bf16(P, cb2, NULL, 0, battn_scr, bcos, bsin, N_HEADS, 1);
            [cb2 commit]; [cb2 waitUntilCompleted];
        }
        id<MTLBuffer> battn = battn_scr;  // alias for code below

    // === CB3+CB4 merged: wo_a (8 groups) + GPU blit concat + wo_b in ONE CB (1 wait) ===
    // When external_cb1 is set: uses it for CB3 too (GPU blit grouping, no CPU readback).
    // When external_cb1 is NULL: legacy path with CPU memcpy grouping (backward compat).
    uint16_t *attn_bf16 = external_cb1 ? NULL : (uint16_t *)[battn contents];
    int heads_per_group = N_HEADS / O_GROUPS;
    int group_feat = heads_per_group * HEAD_DIM;
    float *concat = (abc && abc->scr_concat) ? NULL : malloc((size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    if (!(abc && abc->scr_concat) && !concat) return -1;

    id<MTLBuffer> bog_arr[O_GROUPS];
    id<MTLBuffer> bconcat = (abc && abc->scr_concat) ? abc->scr_concat
                              : mkbuf(d, NULL, (size_t)O_GROUPS * O_LORA_RANK * sizeof(float));
    id<MTLBuffer> bout    = out_gpu_buf ? (__bridge id<MTLBuffer>)out_gpu_buf
                              : ((abc && abc->scr_out) ? abc->scr_out
                                  : mkbuf(d, NULL, DIM * sizeof(float)));
    {
        id<MTLCommandBuffer> cb3 = (external_cb1 && !stage_time) ? (__bridge id<MTLCommandBuffer>)external_cb1
                                                 : [P->queue commandBuffer];

        static int use_q8_woa = -1;
        if (use_q8_woa < 0) use_q8_woa = getenv("DMLX_USE_Q8_WOA") ? 1 : 0;
        static int use_f32_woa = -1;
        if (use_f32_woa < 0) use_f32_woa = getenv("DMLX_WOA_F32") ? 1 : 0;

        if (!use_f32_woa && !use_q8_woa && abc && abc->wo_a_pack_all && external_cb1) {
            // --- Fused wo_a: ONE grp8 dispatch for all 8 groups ---
            // Reads battn directly with per-group x offsets and writes the
            // concat layout straight into bconcat — replaces 8 blits + 8
            // matvec dispatches + the concat blit. Per-row accumulation is
            // identical to the per-group path (bit-exact).
            id<MTLComputeCommandEncoder> e = attention_stage_encoder(
                stage_engine, cb3, DMLX_ATTENTION_STAGE_WO_A);
            [e setComputePipelineState:P->dequant_matvec_affine_bf16in_f32out_grp8];
            [e setBuffer:abc->wo_a_pack_all offset:0 atIndex:0];
            [e setBuffer:abc->wo_a_sc_all   offset:0 atIndex:1];
            [e setBuffer:abc->wo_a_bi_all   offset:0 atIndex:2];
            [e setBuffer:battn              offset:0 atIndex:3];
            [e setBuffer:bconcat            offset:0 atIndex:4];
            uint og = O_LORA_RANK, id_ = (uint)group_feat, gs = 64;
            [e setBytes:&og  length:4 atIndex:5];
            [e setBytes:&id_ length:4 atIndex:6];
            [e setBytes:&gs  length:4 atIndex:7];
            [e dispatchThreads:MTLSizeMake(O_GROUPS * O_LORA_RANK, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
        } else {
        // --- Part 1: wo_a × 8 group matmuls ---
        for (int g = 0; g < O_GROUPS; g++) {
            id<MTLBuffer> bgv;
            if (abc && abc->scr_bgv[g]) {
                bgv = abc->scr_bgv[g];
            } else {
                bgv = mkbuf(d, NULL, group_feat * sizeof(uint16_t));
            }

            if (external_cb1) {
                // GPU blit: copy group g's heads from battn → bgv (no CPU readback)
                id<MTLBlitCommandEncoder> blit = [cb3 blitCommandEncoder];
                [blit copyFromBuffer:battn
                       sourceOffset:(size_t)g * group_feat * sizeof(uint16_t)
                           toBuffer:bgv
                  destinationOffset:0
                               size:(size_t)group_feat * sizeof(uint16_t)];
                [blit endEncoding];
            } else {
                // Legacy: CPU memcpy grouping
                uint16_t *gv_bf16_data = (abc && abc->scr_bgv[g])
                    ? (uint16_t *)[bgv contents]
                    : malloc((size_t)group_feat * sizeof(uint16_t));
                if (!(abc && abc->scr_bgv[g]) && !gv_bf16_data) { if (concat) free(concat); return -1; }
                for (int hh = 0; hh < heads_per_group; hh++)
                    memcpy(gv_bf16_data + hh * HEAD_DIM,
                           attn_bf16 + (g * heads_per_group + hh) * HEAD_DIM,
                           HEAD_DIM * sizeof(uint16_t));
                if (!(abc && abc->scr_bgv[g])) {
                    bgv = mkbuf(d, gv_bf16_data, group_feat * sizeof(uint16_t));
                    free(gv_bf16_data);
                }
            }

            bog_arr[g] = (abc && abc->scr_bog[g]) ? abc->scr_bog[g]
                           : mkbuf(d, NULL, O_LORA_RANK * sizeof(float));
            // wo_a path priority: native packed 4-bit (default; model's own
            // quantization, 2MB/group vs 16MB f32) → Q8_0 (DMLX_USE_Q8_WOA=1,
            // known correctness issues on some prompts) → f32 dense
            // (DMLX_WOA_F32=1 or packed unavailable).
            if (!use_f32_woa && abc && abc->wo_a_pack[g]) {
                enc_dq_bf16_cached(P, cb3, NULL, 0, abc->wo_a_pack[g], abc->wo_a_sc[g], abc->wo_a_bi[g],
                                   O_LORA_RANK, group_feat, 64, bgv, bog_arr[g],
                                   P->dequant_matvec_affine_bf16in_f32out);
            } else if (use_q8_woa && abc && abc->wo_a_q8_gpu[g]) {
                enc_matvec_q8_0(P, cb3, abc->wo_a_q8_gpu[g], bgv, bog_arr[g], O_LORA_RANK, group_feat);
            } else {
                id<MTLBuffer> bwg = (abc && abc->wo_a_grp[g])
                                        ? abc->wo_a_grp[g]
                                        : [d newBufferWithBytes:(void*)(aw->wo_a_dense + (size_t)g * O_LORA_RANK * group_feat)
                                              length:(size_t)O_LORA_RANK * group_feat * sizeof(float)
                                              options:MTLResourceStorageModeShared];
                enc_matvec_f32_bf16in(P, cb3, bwg, bgv, bog_arr[g], O_LORA_RANK, group_feat);
            }
        }

        // --- Part 2: GPU blit concat (bog_arr[g] → bconcat[g * O_LORA_RANK]) ---
        if (stage_time) {
            double s0 = 0;
            { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); s0 = ts.tv_sec*1e9+ts.tv_nsec; }
            [cb3 commit]; [cb3 waitUntilCompleted];
            { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
              fprintf(stderr, "[STAGE] pos=%d wo_a8=%.2f\n", pos, (ts.tv_sec*1e9+ts.tv_nsec-s0)/1e6); }
            cb3 = [P->queue commandBuffer];
        }
        {
            id<MTLBlitCommandEncoder> blit = [cb3 blitCommandEncoder];
            size_t chunk = (size_t)O_LORA_RANK * sizeof(float);
            for (int g = 0; g < O_GROUPS; g++) {
                [blit copyFromBuffer:bog_arr[g]
                       sourceOffset:0
                           toBuffer:bconcat
                  destinationOffset:(size_t)g * chunk
                               size:chunk];
            }
            [blit endEncoding];
        }
        } // end else (legacy per-group wo_a path)

        // --- Part 3: wo_b output projection (coalesced v2 kernel) ---
        if (abc && abc->wo_b_pack) {
            id<MTLComputeCommandEncoder> e = attention_stage_encoder(
                stage_engine, cb3, DMLX_ATTENTION_STAGE_WO_B);
            [e setComputePipelineState:P->dequant_matvec_affine_v2];
            [e setBuffer:abc->wo_b_pack offset:0 atIndex:0];
            [e setBuffer:abc->wo_b_sc   offset:0 atIndex:1];
            [e setBuffer:abc->wo_b_bi   offset:0 atIndex:2];
            [e setBuffer:bconcat        offset:0 atIndex:3];
            [e setBuffer:bout           offset:0 atIndex:4];
            uint od=aw->wo_b.out_dim, id_=aw->wo_b.in_dim, gs=aw->wo_b.group_size;
            [e setBytes:&od  length:4 atIndex:5];
            [e setBytes:&id_ length:4 atIndex:6];
            [e setBytes:&gs  length:4 atIndex:7];
            [e setThreadgroupMemoryLength:256 atIndex:0];  // 32*2*4 bytes
            uint num_tgs = (aw->wo_b.out_dim + 1) / 2;     // NR0=2
            [e dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];  // (32, NSG=4, 1)
            [e endEncoding];
        } else {
            enc_dequant_matvec(P, cb3, &aw->wo_b, bconcat, bout);
        }

        if (!external_cb1 || stage_time) {
            double s0 = 0;
            if (stage_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); s0 = ts.tv_sec*1e9+ts.tv_nsec; }
            [cb3 commit]; [cb3 waitUntilCompleted];
            if (stage_time) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
                fprintf(stderr, "[STAGE] pos=%d wo_b=%.2f\n", pos, (ts.tv_sec*1e9+ts.tv_nsec-s0)/1e6); }
        }
    }
    // When external_cb1: bout hasn't been committed yet — caller reads after commit+wait.
    if (!external_cb1) {
        memcpy(out, [bout contents], DIM * sizeof(float));
    }
    if (!abc || !abc->scr_concat) free(concat);
    } // end @autoreleasepool
    *out_identity = identity_candidate;
    return 0;
}

// --- DS4-style attention: Q chain f32, KV cache f32 (full precision), SDPA f32×f32.
int mla_attention_decode_f16kv(MlaPipes *P, const AttnWeights *aw,
                               const float *x, float *kv_cache, int cache_len,
                               int pos, float *out) {
    // wo_a_dense is not loaded by default anymore (packed path);
    // this legacy f32-dense path requires it (DMLX_WOA_F32=1).
    if (!aw->wo_a_dense) return -2;
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
    // wo_a_dense is not loaded by default anymore (packed path);
    // this legacy f32-dense path requires it (DMLX_WOA_F32=1).
    if (!aw->wo_a_dense) return -2;
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
// Persistent no-copy Metal view for NativeBatchWorkspace. The parent owns one
// retained MTLBuffer for its full lifetime; complete-v2 subworkspaces borrow it
// and add metal_buffer_offset instead of creating per-stage wrappers.
// ============================================================================
static void mla_workspace_release_metal_buffer(void *opaque) {
    if (!opaque) return;
#if __has_feature(objc_arc)
    CFBridgingRelease(opaque);
#else
    [(id<MTLBuffer>)opaque release];
#endif
}

static id<MTLBuffer> mla_workspace_metal_buffer(
    id<MTLDevice> device, NativeBatchWorkspace *workspace) {
    if (!device || !workspace || !workspace->storage ||
        workspace->allocated_bytes == 0u ||
        workspace->allocated_bytes > (uint64_t)NSUIntegerMax ||
        workspace->metal_buffer_offset > (uint64_t)NSUIntegerMax) {
        return nil;
    }
    if (workspace->metal_buffer) {
        id<MTLBuffer> existing =
            (__bridge id<MTLBuffer>)workspace->metal_buffer;
        const uint64_t length = (uint64_t)existing.length;
        if (existing.device != device ||
            workspace->metal_buffer_offset > length ||
            workspace->allocated_bytes >
                length - workspace->metal_buffer_offset) {
            return nil;
        }
        return existing;
    }
    if (workspace->metal_buffer_offset != 0u ||
        workspace->metal_buffer_release ||
        workspace->metal_buffer_creations == UINT64_MAX) {
        return nil;
    }
    id<MTLBuffer> created = [device
        newBufferWithBytesNoCopy:workspace->storage
        length:(NSUInteger)workspace->allocated_bytes
        options:MTLResourceStorageModeShared
        deallocator:nil];
    if (!created) return nil;
#if __has_feature(objc_arc)
    workspace->metal_buffer = (__bridge_retained void *)created;
#else
    workspace->metal_buffer = (void *)created;
#endif
    workspace->metal_buffer_release = mla_workspace_release_metal_buffer;
    workspace->metal_buffer_creations++;
    return created;
}

// ============================================================================
// mla_attention_q_batch — Task 2.2 batched pre-RoPE Q chain.
//
// One command buffer contains exactly four dispatches for every B:
//   [B,DIM] -> wq_a -> [B,Q_LORA_RANK]
//            -> q_norm
//            -> wq_b -> [B,N_HEADS,HEAD_DIM]
//            -> per-head norm
// The Q RoPE belongs to Task 2.3 because it consumes explicit positions.
// ============================================================================
int mla_attention_q_batch(MlaPipes *P, const AttnWeights *aw,
                          const uint16_t *hidden, uint16_t *q_output,
                          uint32_t batch_tokens, NativeBatchWorkspace *workspace,
                          MLAAttentionMetrics *metrics) {
    @autoreleasepool {
        if (metrics) memset(metrics, 0, sizeof(*metrics));
        if (!P || !P->dev || !P->queue ||
            !P->dequant_matvec_affine_bf16in_bf16out ||
            !P->rms_norm_rows_bf16in_bf16out ||
            !aw || !hidden || !q_output || !workspace ||
            batch_tokens == 0u || batch_tokens > workspace->capacity_tokens ||
            workspace->self_cookie != (uintptr_t)workspace || !workspace->storage ||
            aw->wq_a.out_dim != Q_LORA_RANK || aw->wq_a.in_dim != DIM ||
            aw->wq_b.out_dim != N_HEADS * HEAD_DIM || aw->wq_b.in_dim != Q_LORA_RANK) {
            return NATIVE_BATCH_INVALID_ARGUMENT;
        }

        const uint64_t hidden_bytes_u64 =
            (uint64_t)batch_tokens * DIM * sizeof(uint16_t);
        const uint64_t q_a_bytes =
            (uint64_t)batch_tokens * Q_LORA_RANK * sizeof(uint16_t);
        const uint64_t q_bytes =
            (uint64_t)batch_tokens * N_HEADS * HEAD_DIM * sizeof(uint16_t);
        const uint64_t scratch_bytes = hidden_bytes_u64 + q_a_bytes * 2u + q_bytes * 2u;
        void *scratch = moe_infer_batch_workspace_reserve(
            workspace, scratch_bytes, NATIVE_BATCH_WORKSPACE_ALIGNMENT);
        if (!scratch) return NATIVE_BATCH_CAPACITY_EXCEEDED;

        AttnBufCache *abc = attn_buf_cache_get(P->dev, aw);
        if (!abc || !abc->wq_a_pack || !abc->wq_a_sc || !abc->wq_a_bi ||
            !abc->wq_b_pack || !abc->wq_b_sc || !abc->wq_b_bi ||
            !abc->q_norm_buf) {
            return NATIVE_BATCH_INVALID_STATE;
        }

        const size_t hidden_bytes = (size_t)hidden_bytes_u64;
        const size_t q_size = (size_t)q_bytes;
        const NSUInteger storage_hidden_offset =
            (NSUInteger)((uint8_t *)scratch - (uint8_t *)workspace->storage);
        const NSUInteger metal_base = (NSUInteger)workspace->metal_buffer_offset;
        const NSUInteger hidden_offset = metal_base + storage_hidden_offset;
        const NSUInteger q_a_offset = hidden_offset + hidden_bytes;
        const NSUInteger q_res_offset = q_a_offset + (NSUInteger)q_a_bytes;
        const NSUInteger q_offset = q_res_offset + (NSUInteger)q_a_bytes;
        const NSUInteger q_norm_offset = q_offset + (NSUInteger)q_bytes;
        memcpy((uint8_t *)workspace->storage + storage_hidden_offset,
               hidden, hidden_bytes);

        // The bounded arena is page-aligned by workspace_init. Standalone
        // calls lazily create one persistent view; complete-v2 subworkspaces
        // borrow the parent view and use offsets into it.
        id<MTLBuffer> batch_buffer =
            mla_workspace_metal_buffer(P->dev, workspace);
        if (!batch_buffer) return NATIVE_BATCH_INVALID_STATE;

        id<MTLCommandBuffer> cb = [P->queue commandBuffer];
        if (!cb) return NATIVE_BATCH_INVALID_STATE;
        enc_dq_bf16_cached_batch(
            P, cb, abc->wq_a_pack, abc->wq_a_sc, abc->wq_a_bi,
            aw->wq_a.out_dim, aw->wq_a.in_dim, aw->wq_a.group_size,
            batch_buffer, hidden_offset, q_a_offset, batch_tokens);
        enc_rms_norm_rows_bf16_batch_offset(
            P, cb, batch_buffer, q_a_offset, abc->q_norm_buf, q_res_offset,
            batch_tokens, Q_LORA_RANK, 1);
        enc_dq_bf16_cached_batch(
            P, cb, abc->wq_b_pack, abc->wq_b_sc, abc->wq_b_bi,
            aw->wq_b.out_dim, aw->wq_b.in_dim, aw->wq_b.group_size,
            batch_buffer, q_res_offset, q_offset, batch_tokens);
        enc_rms_norm_rows_bf16_batch_offset(
            P, cb, batch_buffer, q_offset, nil, q_norm_offset,
            batch_tokens * N_HEADS, HEAD_DIM, 0);

        struct timespec start, finish;
        clock_gettime(CLOCK_MONOTONIC, &start);
        [cb commit];
        [cb waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC, &finish);
        if (cb.status == MTLCommandBufferStatusError) {
            return NATIVE_BATCH_INVALID_STATE;
        }
        memcpy(q_output,
               (uint8_t *)workspace->storage +
                   (q_norm_offset - metal_base),
               q_size);

        if (metrics) {
            const int64_t elapsed_ns =
                (int64_t)(finish.tv_sec - start.tv_sec) * 1000000000ll +
                (int64_t)(finish.tv_nsec - start.tv_nsec);
            metrics->q_ns = elapsed_ns > 0 ? (uint64_t)elapsed_ns : 0u;
            metrics->dispatches = 4u;
            metrics->workspace_high_water_bytes = workspace->high_water_bytes;
        }
        return NATIVE_BATCH_OK;
    }
}

static bool mla_align_workspace(uint64_t value, uint64_t *aligned) {
    const uint64_t mask = NATIVE_BATCH_WORKSPACE_ALIGNMENT - 1u;
    if (!aligned || value > UINT64_MAX - mask) return false;
    *aligned = (value + mask) & ~mask;
    return true;
}

static bool mla_u64_add(uint64_t left, uint64_t right, uint64_t *sum) {
    if (!sum || right > UINT64_MAX - left) return false;
    *sum = left + right;
    return true;
}

static bool mla_u64_multiply(uint64_t left, uint64_t right, uint64_t *product) {
    if (!product || (left != 0u && right > UINT64_MAX / left)) return false;
    *product = left * right;
    return true;
}

static bool mla_workspace_next(
    uint64_t offset, uint64_t size_bytes, uint64_t *next_offset) {
    uint64_t end = 0u;
    return mla_u64_add(offset, size_bytes, &end) &&
        mla_align_workspace(end, next_offset);
}

static bool mla_ranges_overlap(
    const void *left, uint64_t left_bytes,
    const void *right, uint64_t right_bytes) {
    if (!left || !right || left_bytes == 0u || right_bytes == 0u) return false;
    const uintptr_t left_start = (uintptr_t)left;
    const uintptr_t right_start = (uintptr_t)right;
    if (left_bytes > UINTPTR_MAX - left_start ||
        right_bytes > UINTPTR_MAX - right_start) {
        return true;
    }
    const uintptr_t left_end = left_start + (uintptr_t)left_bytes;
    const uintptr_t right_end = right_start + (uintptr_t)right_bytes;
    return left_start < right_end && right_start < left_end;
}

int mla_attention_kv_rope_batch(MoEInferEngine *engine,
                                MlaPipes *P, const AttnWeights *aw,
                                const MLAPrefillBatchArgs *args,
                                MLAAttentionMetrics *metrics) {
    @autoreleasepool {
        if (!engine || !P || !P->dev || !P->queue ||
            !P->dequant_matvec_affine_bf16in_bf16out ||
            !P->rms_norm_rows_bf16in_bf16out || !P->rope_tail_interleaved_bf16 ||
            !aw || !args || !args->hidden || !args->positions ||
            !args->workspace || !args->kv_transaction || !args->kv.kv_data ||
            args->batch_tokens == 0u || args->layer >= N_LAYERS ||
            args->batch_tokens > args->workspace->capacity_tokens ||
            args->workspace->self_cookie != (uintptr_t)args->workspace ||
            !args->workspace->storage ||
            args->kv_transaction->self_cookie != (uintptr_t)args->kv_transaction ||
            args->kv_transaction->owner_engine == 0u ||
            args->kv_transaction->active != 1u ||
            args->kv_transaction->generation == 0u ||
            args->kv_transaction->phase < NATIVE_BATCH_PHASE_PREFILL ||
            args->kv_transaction->phase > NATIVE_BATCH_PHASE_DECODE ||
            args->kv.generation != args->kv_transaction->generation ||
            args->kv.layer != args->layer ||
            args->kv.start_position != args->kv_transaction->start_position ||
            args->kv.row_count != args->batch_tokens ||
            args->kv.row_stride_bytes != KV_LORA_RANK * sizeof(uint16_t) ||
            args->kv_transaction->staged_tokens != args->batch_tokens ||
            aw->wkv.out_dim != KV_LORA_RANK || aw->wkv.in_dim != DIM) {
            return NATIVE_BATCH_INVALID_ARGUMENT;
        }
        const int view_status = moe_infer_kv_transaction_validate_layer_view(
            engine, args->kv_transaction, args->layer, &args->kv);
        if (view_status != NATIVE_BATCH_OK) return view_status;
        if (args->positions[0] != args->kv.start_position ||
            args->positions[0] >= MAX_SEQ_LEN ||
            args->batch_tokens > (uint32_t)MAX_SEQ_LEN - args->positions[0]) {
            return NATIVE_BATCH_INVALID_POSITION;
        }
        for (uint32_t row = 1; row < args->batch_tokens; ++row) {
            if (args->positions[row] != args->positions[0] + row) {
                return NATIVE_BATCH_INVALID_POSITION;
            }
        }

        uint64_t hidden_bytes = 0u, kv_bytes = 0u, rope_bytes = 0u;
        if (!mla_u64_multiply(args->batch_tokens,
                (uint64_t)DIM * sizeof(uint16_t), &hidden_bytes) ||
            !mla_u64_multiply(args->batch_tokens,
                (uint64_t)KV_LORA_RANK * sizeof(uint16_t), &kv_bytes) ||
            !mla_u64_multiply(args->batch_tokens,
                (uint64_t)(QK_ROPE_DIM / 2) * sizeof(float), &rope_bytes)) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }

        // A GPU staging view addresses the full layer cache, so validate its
        // absolute destination before reserving workspace or encoding commands.
        id<MTLBuffer> staged_gpu = (__bridge id<MTLBuffer>)args->kv.kv_gpu_buffer;
        uint64_t destination_u64 = 0u;
        if (!mla_u64_multiply(args->kv.start_position,
                args->kv.row_stride_bytes, &destination_u64) ||
            destination_u64 > (uint64_t)NSUIntegerMax) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }
        const NSUInteger destination = (NSUInteger)destination_u64;
        if (staged_gpu &&
            (destination > staged_gpu.length ||
             (NSUInteger)kv_bytes > staged_gpu.length - destination)) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }

        const uint64_t hidden_rel = 0u;
        uint64_t kv_rel = 0u;
        uint64_t kv_norm_rel = 0u;
        uint64_t cos_rel = 0u;
        uint64_t sin_rel = 0u;
        uint64_t scratch_bytes = 0u;
        if (!mla_workspace_next(hidden_rel, hidden_bytes, &kv_rel) ||
            !mla_workspace_next(kv_rel, kv_bytes, &kv_norm_rel) ||
            !mla_workspace_next(kv_norm_rel, kv_bytes, &cos_rel) ||
            !mla_workspace_next(cos_rel, rope_bytes, &sin_rel) ||
            !mla_u64_add(sin_rel, rope_bytes, &scratch_bytes)) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }
        void *scratch = moe_infer_batch_workspace_reserve(
            args->workspace, scratch_bytes, NATIVE_BATCH_WORKSPACE_ALIGNMENT);
        if (!scratch) return NATIVE_BATCH_CAPACITY_EXCEEDED;

        AttnBufCache *abc = attn_buf_cache_get(P->dev, aw);
        if (!abc || !abc->wkv_pack || !abc->wkv_sc || !abc->wkv_bi ||
            !abc->kv_norm_buf) {
            return NATIVE_BATCH_INVALID_STATE;
        }

        const NSUInteger storage_base =
            (NSUInteger)((uint8_t *)scratch -
                (uint8_t *)args->workspace->storage);
        const NSUInteger metal_base =
            (NSUInteger)args->workspace->metal_buffer_offset;
        const NSUInteger hidden_offset = metal_base + storage_base +
            (NSUInteger)hidden_rel;
        const NSUInteger kv_offset = metal_base + storage_base +
            (NSUInteger)kv_rel;
        const NSUInteger kv_norm_offset = metal_base + storage_base +
            (NSUInteger)kv_norm_rel;
        const NSUInteger cos_offset = metal_base + storage_base +
            (NSUInteger)cos_rel;
        const NSUInteger sin_offset = metal_base + storage_base +
            (NSUInteger)sin_rel;
        uint8_t *workspace_storage = (uint8_t *)args->workspace->storage;
        memcpy(workspace_storage + storage_base + (NSUInteger)hidden_rel,
               args->hidden, (size_t)hidden_bytes);
        float *cos_values = (float *)(workspace_storage + storage_base +
            (NSUInteger)cos_rel);
        float *sin_values = (float *)(workspace_storage + storage_base +
            (NSUInteger)sin_rel);
        for (uint32_t row = 0; row < args->batch_tokens; ++row) {
            yarn_cos_sin((int)args->positions[row],
                         cos_values + (size_t)row * (QK_ROPE_DIM / 2),
                         sin_values + (size_t)row * (QK_ROPE_DIM / 2));
        }

        id<MTLBuffer> batch_buffer =
            mla_workspace_metal_buffer(P->dev, args->workspace);
        if (!batch_buffer) return NATIVE_BATCH_INVALID_STATE;
        id<MTLCommandBuffer> cb = [P->queue commandBuffer];
        if (!cb) return NATIVE_BATCH_INVALID_STATE;
        enc_dq_bf16_cached_batch(
            P, cb, abc->wkv_pack, abc->wkv_sc, abc->wkv_bi,
            aw->wkv.out_dim, aw->wkv.in_dim, aw->wkv.group_size,
            batch_buffer, hidden_offset, kv_offset, args->batch_tokens);
        enc_rms_norm_rows_bf16_batch_offset(
            P, cb, batch_buffer, kv_offset, abc->kv_norm_buf, kv_norm_offset,
            args->batch_tokens, KV_LORA_RANK, 1);
        enc_rope_bf16_batch_offset(
            P, cb, batch_buffer, kv_norm_offset, cos_offset, sin_offset,
            args->batch_tokens, 1u, 0u);

        struct timespec start, finish;
        clock_gettime(CLOCK_MONOTONIC, &start);
        [cb commit];
        [cb waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC, &finish);
        if (cb.status == MTLCommandBufferStatusError) {
            return NATIVE_BATCH_INVALID_STATE;
        }

        // The projection/RoPE kernels produce BF16, while the native cache and
        // scalar decode path consume IEEE f16. Convert only after GPU completion
        // and write into the authenticated engine-derived destination. When a
        // shared GPU cache exists kv_data aliases its contents, so this one CPU
        // write updates the exact physical cache without a BF16 blit.
        const uint16_t *kv_bf16 = (const uint16_t *)(
            workspace_storage + storage_base + (NSUInteger)kv_norm_rel);
        uint16_t *kv_f16 = (uint16_t *)args->kv.kv_data;
        const size_t kv_elements =
            (size_t)args->batch_tokens * KV_LORA_RANK;
        for (size_t index = 0; index < kv_elements; ++index) {
            const uint32_t bits = (uint32_t)kv_bf16[index] << 16;
            float value;
            memcpy(&value, &bits, sizeof(value));
            kv_f16[index] = f32_to_f16(value);
        }
        if (metrics) {
            const int64_t elapsed_ns =
                (int64_t)(finish.tv_sec - start.tv_sec) * 1000000000ll +
                (int64_t)(finish.tv_nsec - start.tv_nsec);
            metrics->kv_rope_ns = elapsed_ns > 0 ? (uint64_t)elapsed_ns : 0u;
            metrics->dispatches += 3u;
            metrics->workspace_high_water_bytes = args->workspace->high_water_bytes;
        }
        return NATIVE_BATCH_OK;
    }
}

// ============================================================================
// mla_attention_sdpa_batch — Task 2.4 chunk-local causal SDPA stage.
//
// Owned workspace layout (every start offset is 256-byte aligned):
//   q_work[B,N_HEADS,HEAD_DIM] BF16
//   kv_work[start+B,HEAD_DIM]   f16
//   cos[B,QK_ROPE_DIM/2]       f32
//   sin[B,QK_ROPE_DIM/2]       f32
//   output[B,N_HEADS,HEAD_DIM] BF16
// Exactly two compute dispatches share one command buffer: Q RoPE in-place,
// then causal SDPA. No stage matrix is allocated outside this workspace.
// ============================================================================
int mla_attention_sdpa_batch(MoEInferEngine *engine,
                             MlaPipes *P, const AttnWeights *aw,
                             const MLAPrefillBatchArgs *args,
                             const uint16_t *q_input,
                             uint16_t *sdpa_output,
                             MLAAttentionMetrics *metrics) {
    @autoreleasepool {
        if (!engine || !P || !P->dev || !P->queue ||
            !P->rope_tail_interleaved_bf16 || !P->mla_sdpa_batch_f16_kv ||
            !aw || !aw->attn_sink || !args || !args->positions ||
            !args->workspace || !args->kv_transaction || !args->kv.kv_data ||
            !q_input || !sdpa_output ||
            args->batch_tokens == 0u || args->layer >= N_LAYERS ||
            args->batch_tokens > args->workspace->capacity_tokens ||
            args->workspace->self_cookie != (uintptr_t)args->workspace ||
            !args->workspace->storage ||
            args->kv_transaction->self_cookie != (uintptr_t)args->kv_transaction ||
            args->kv_transaction->owner_engine == 0u ||
            args->kv_transaction->active != 1u ||
            args->kv_transaction->generation == 0u ||
            args->kv.generation != args->kv_transaction->generation ||
            args->kv.layer != args->layer ||
            args->kv.start_position != args->kv_transaction->start_position ||
            args->kv.row_count != args->batch_tokens ||
            args->kv.row_stride_bytes != KV_LORA_RANK * sizeof(uint16_t) ||
            args->kv_transaction->staged_tokens != args->batch_tokens) {
            return NATIVE_BATCH_INVALID_ARGUMENT;
        }

        const int view_status = moe_infer_kv_transaction_validate_layer_view(
            engine, args->kv_transaction, args->layer, &args->kv);
        if (view_status != NATIVE_BATCH_OK) return view_status;

        const uint32_t committed = args->kv.start_position;
        if (args->positions[0] != committed || committed >= MAX_SEQ_LEN ||
            args->batch_tokens > (uint32_t)MAX_SEQ_LEN - committed) {
            return NATIVE_BATCH_INVALID_POSITION;
        }
        for (uint32_t row = 1u; row < args->batch_tokens; ++row) {
            if (args->positions[row] != committed + row) {
                return NATIVE_BATCH_INVALID_POSITION;
            }
        }
        uint64_t visible_rows = 0u;
        if (!mla_u64_add((uint64_t)committed,
                (uint64_t)args->batch_tokens, &visible_rows)) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }

        uint64_t q_bytes = 0u;
        uint64_t kv_bytes = 0u;
        uint64_t rope_bytes = 0u;
        if (!mla_u64_multiply(
                args->batch_tokens,
                (uint64_t)N_HEADS * HEAD_DIM * sizeof(uint16_t), &q_bytes) ||
            !mla_u64_multiply(
                visible_rows,
                (uint64_t)KV_LORA_RANK * sizeof(uint16_t), &kv_bytes) ||
            !mla_u64_multiply(
                args->batch_tokens,
                (uint64_t)(QK_ROPE_DIM / 2) * sizeof(float), &rope_bytes)) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }

        const uint16_t *full_kv = engine->kv_cache[args->layer].kv;
        if (!full_kv ||
            mla_ranges_overlap(q_input, q_bytes, sdpa_output, q_bytes) ||
            mla_ranges_overlap(full_kv, kv_bytes, sdpa_output, q_bytes) ||
            mla_ranges_overlap(
                args->workspace->storage, args->workspace->allocated_bytes,
                q_input, q_bytes) ||
            mla_ranges_overlap(
                args->workspace->storage, args->workspace->allocated_bytes,
                sdpa_output, q_bytes) ||
            mla_ranges_overlap(
                args->workspace->storage, args->workspace->allocated_bytes,
                full_kv, kv_bytes)) {
            return NATIVE_BATCH_INVALID_ARGUMENT;
        }

        // If the authoritative cache is shared with Metal, validate the entire
        // visible prefix before reserving workspace or encoding either dispatch.
        id<MTLBuffer> cache_gpu =
            (__bridge id<MTLBuffer>)engine->kv_cache[args->layer].kv_gpu_buf;
        if (cache_gpu && (NSUInteger)kv_bytes > cache_gpu.length) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }

        const uint64_t q_rel = 0u;
        uint64_t kv_rel = 0u;
        uint64_t cos_rel = 0u;
        uint64_t sin_rel = 0u;
        uint64_t output_rel = 0u;
        uint64_t scratch_bytes = 0u;
        if (!mla_workspace_next(q_rel, q_bytes, &kv_rel) ||
            !mla_workspace_next(kv_rel, kv_bytes, &cos_rel) ||
            !mla_workspace_next(cos_rel, rope_bytes, &sin_rel) ||
            !mla_workspace_next(sin_rel, rope_bytes, &output_rel) ||
            !mla_u64_add(output_rel, q_bytes, &scratch_bytes)) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }
        void *scratch = moe_infer_batch_workspace_reserve(
            args->workspace, scratch_bytes, NATIVE_BATCH_WORKSPACE_ALIGNMENT);
        if (!scratch) return NATIVE_BATCH_CAPACITY_EXCEEDED;

        // Sink weights are immutable per layer and live in the persistent
        // attention-weight cache. The hot SDPA stage must not allocate an
        // unaccounted buffer on every chunk.
        AttnBufCache *abc = attn_buf_cache_get(P->dev, aw);
        if (!abc || !abc->attn_sink_buf) return NATIVE_BATCH_INVALID_STATE;

        const NSUInteger storage_base = (NSUInteger)(
            (uint8_t *)scratch - (uint8_t *)args->workspace->storage);
        const NSUInteger metal_base =
            (NSUInteger)args->workspace->metal_buffer_offset;
        const NSUInteger q_offset = metal_base + storage_base +
            (NSUInteger)q_rel;
        const NSUInteger kv_offset = metal_base + storage_base +
            (NSUInteger)kv_rel;
        const NSUInteger cos_offset = metal_base + storage_base +
            (NSUInteger)cos_rel;
        const NSUInteger sin_offset = metal_base + storage_base +
            (NSUInteger)sin_rel;
        const NSUInteger output_offset = metal_base + storage_base +
            (NSUInteger)output_rel;

        uint8_t *workspace_bytes = (uint8_t *)args->workspace->storage;
        memcpy(workspace_bytes + storage_base + (NSUInteger)q_rel,
               q_input, (size_t)q_bytes);
        memcpy(workspace_bytes + storage_base + (NSUInteger)kv_rel,
               full_kv, (size_t)kv_bytes);
        float *cos_values = (float *)(workspace_bytes + storage_base +
            (NSUInteger)cos_rel);
        float *sin_values = (float *)(workspace_bytes + storage_base +
            (NSUInteger)sin_rel);
        for (uint32_t row = 0u; row < args->batch_tokens; ++row) {
            yarn_cos_sin(
                (int)args->positions[row],
                cos_values + (size_t)row * (QK_ROPE_DIM / 2),
                sin_values + (size_t)row * (QK_ROPE_DIM / 2));
        }

        id<MTLBuffer> batch_buffer =
            mla_workspace_metal_buffer(P->dev, args->workspace);
        if (!batch_buffer) return NATIVE_BATCH_INVALID_STATE;

        id<MTLCommandBuffer> cb = [P->queue commandBuffer];
        if (!cb) return NATIVE_BATCH_INVALID_STATE;
        enc_rope_bf16_batch_offset(
            P, cb, batch_buffer, q_offset, cos_offset, sin_offset,
            args->batch_tokens, N_HEADS, 0u);
        {
            id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
            [encoder setComputePipelineState:P->mla_sdpa_batch_f16_kv];
            [encoder setBuffer:batch_buffer offset:q_offset atIndex:0];
            [encoder setBuffer:batch_buffer offset:kv_offset atIndex:1];
            [encoder setBuffer:abc->attn_sink_buf offset:0 atIndex:2];
            [encoder setBuffer:batch_buffer offset:output_offset atIndex:3];
            const uint32_t batch_tokens = args->batch_tokens;
            const uint32_t n_heads = N_HEADS;
            const float scale = 1.0f / sqrtf((float)HEAD_DIM);
            [encoder setBytes:&batch_tokens length:sizeof(batch_tokens) atIndex:4];
            [encoder setBytes:&committed length:sizeof(committed) atIndex:5];
            [encoder setBytes:&n_heads length:sizeof(n_heads) atIndex:6];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:7];
            const uint32_t query_rows = args->batch_tokens;
            [encoder dispatchThreadgroups:MTLSizeMake(
                    (NSUInteger)query_rows * N_HEADS, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder endEncoding];
        }

        struct timespec start, finish;
        clock_gettime(CLOCK_MONOTONIC, &start);
        [cb commit];
        [cb waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC, &finish);
        if (cb.status == MTLCommandBufferStatusError) {
            return NATIVE_BATCH_INVALID_STATE;
        }
        memcpy(sdpa_output,
               workspace_bytes + storage_base + (NSUInteger)output_rel,
               (size_t)q_bytes);

        if (metrics) {
            const int64_t elapsed_ns =
                (int64_t)(finish.tv_sec - start.tv_sec) * 1000000000ll +
                (int64_t)(finish.tv_nsec - start.tv_nsec);
            metrics->sdpa_ns += elapsed_ns > 0 ? (uint64_t)elapsed_ns : 0u;
            metrics->dispatches += 2u;
            metrics->workspace_high_water_bytes =
                args->workspace->high_water_bytes;
        }
        return NATIVE_BATCH_OK;
    }
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
    // wo_a_dense is not loaded by default anymore (packed path);
    // this legacy f32-dense path requires it (DMLX_WOA_F32=1).
    if (!aw->wo_a_dense) return -2;
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
          enc_rms_norm_rows_bf16in_bf16out(P, cb, NULL, 0, bq_a, bqn_w, bq_res, 1, Q_LORA_RANK, 1);
          [cb commit]; [cb waitUntilCompleted]; }
        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_dequant_matvec_bf16in_bf16out(P, cb, &aw->wq_b, bq_res, bq);
          [cb commit]; [cb waitUntilCompleted]; }
        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_rms_norm_rows_bf16in_bf16out(P, cb, NULL, 0, bq, NULL, bq_n, N_HEADS, HEAD_DIM, 0);
          [cb commit]; [cb waitUntilCompleted]; }
        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_rope_bf16(P, cb, NULL, 0, bq_n, bcos, bsin, N_HEADS, 0);
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
          enc_rms_norm_rows_bf16in_bf16out(P, cb, NULL, 0, bkv, bkvn_w, bkv_n, 1, KV_LORA_RANK, 1);
          [cb commit]; [cb waitUntilCompleted]; }
        { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
          enc_rope_bf16(P, cb, NULL, 0, bkv_n, bcos, bsin, 1, 0);
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
          enc_rope_bf16(P, cb, NULL, 0, battn_t, bcos, bsin, N_HEADS, 1); // inverse RoPE
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

// ============================================================================
// Wave-2 Task 2.5 — bounded packed output projection and transactional v2
// layer boundary. Kept separate from the legacy prefill function above so the
// scalar path and production routing remain byte-for-byte unchanged.
// ============================================================================

typedef struct {
    uint64_t q_stage;
    uint64_t q_handoff;
    uint64_t kv_stage;
    uint64_t sdpa_stage;
    uint64_t sdpa_handoff;
    uint64_t output_stage;
    uint64_t mhc_post_stage;
    uint64_t kv_backup;
    uint64_t residual_backup;
    uint64_t output_backup;
    uint64_t capture_backup;
    uint64_t comp_state_kv_backup;
    uint64_t comp_state_score_backup;
    uint64_t comp_append_backup;
    uint64_t idx_state_kv_backup;
    uint64_t idx_state_score_backup;
    uint64_t idx_append_backup;
    uint64_t allowed_scratch;
    uint64_t normed_scratch;
    uint64_t recurrent_scratch;
    uint64_t total;
} MLAV2WorkspacePlan;

typedef struct {
    CompressorState original;
    float *state_kv;
    float *state_score;
    float *comp_append;
    float *idx_state_kv;
    float *idx_state_score;
    float *idx_append;
    uint32_t comp_append_rows;
    uint32_t idx_append_rows;
} MLACompressorSnapshot;

static uint16_t mla_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float mla_bf16_to_f32(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint32_t mla_order_signature_append(
    uint32_t signature, uint32_t row, uint32_t stage) {
    const uint32_t event = (row << 8u) | stage;
    signature ^= event;
    return signature * UINT32_C(16777619);
}

static bool mla_plan_region(uint64_t *cursor, uint64_t bytes) {
    uint64_t aligned = 0u;
    if (!cursor || !mla_align_workspace(*cursor, &aligned) ||
        bytes > UINT64_MAX - aligned) {
        return false;
    }
    *cursor = aligned + bytes;
    return true;
}

static bool mla_kv_stage_bytes(uint32_t rows, uint64_t *out) {
    uint64_t hidden = 0u, kv = 0u, rope = 0u;
    uint64_t kv_rel = 0u, kv_norm_rel = 0u, cos_rel = 0u, sin_rel = 0u;
    return out &&
        mla_u64_multiply(rows,
            (uint64_t)DIM * sizeof(uint16_t), &hidden) &&
        mla_u64_multiply(rows,
            (uint64_t)KV_LORA_RANK * sizeof(uint16_t), &kv) &&
        mla_u64_multiply(rows,
            (uint64_t)(QK_ROPE_DIM / 2) * sizeof(float), &rope) &&
        mla_workspace_next(0u, hidden, &kv_rel) &&
        mla_workspace_next(kv_rel, kv, &kv_norm_rel) &&
        mla_workspace_next(kv_norm_rel, kv, &cos_rel) &&
        mla_workspace_next(cos_rel, rope, &sin_rel) &&
        mla_u64_add(sin_rel, rope, out);
}

static bool mla_sdpa_stage_bytes(uint32_t rows, uint32_t committed, uint64_t *out) {
    uint64_t visible_rows = 0u, q = 0u, kv = 0u, rope = 0u;
    uint64_t kv_rel = 0u, cos_rel = 0u, sin_rel = 0u, output_rel = 0u;
    return out &&
        mla_u64_add((uint64_t)committed, (uint64_t)rows, &visible_rows) &&
        mla_u64_multiply(rows,
            (uint64_t)N_HEADS * HEAD_DIM * sizeof(uint16_t), &q) &&
        mla_u64_multiply(visible_rows,
            (uint64_t)KV_LORA_RANK * sizeof(uint16_t), &kv) &&
        mla_u64_multiply(rows,
            (uint64_t)(QK_ROPE_DIM / 2) * sizeof(float), &rope) &&
        mla_workspace_next(0u, q, &kv_rel) &&
        mla_workspace_next(kv_rel, kv, &cos_rel) &&
        mla_workspace_next(cos_rel, rope, &sin_rel) &&
        mla_workspace_next(sin_rel, rope, &output_rel) &&
        mla_u64_add(output_rel, q, out);
}

static bool mla_output_stage_bytes(uint32_t rows, uint64_t *out) {
    uint64_t attn = 0u, rope = 0u, concat = 0u, projected = 0u;
    uint64_t cos_rel = 0u, sin_rel = 0u, concat_rel = 0u, projected_rel = 0u;
    return out &&
        mla_u64_multiply(rows,
            (uint64_t)N_HEADS * HEAD_DIM * sizeof(uint16_t), &attn) &&
        mla_u64_multiply(rows,
            (uint64_t)(QK_ROPE_DIM / 2) * sizeof(float), &rope) &&
        mla_u64_multiply(rows,
            (uint64_t)O_GROUPS * O_LORA_RANK * sizeof(float), &concat) &&
        mla_u64_multiply(rows,
            (uint64_t)DIM * sizeof(float), &projected) &&
        mla_workspace_next(0u, attn, &cos_rel) &&
        mla_workspace_next(cos_rel, rope, &sin_rel) &&
        mla_workspace_next(sin_rel, rope, &concat_rel) &&
        mla_workspace_next(concat_rel, concat, &projected_rel) &&
        mla_u64_add(projected_rel, projected, out);
}

int mla_attention_kv_stage_workspace_bytes(uint32_t rows, uint64_t *out_bytes) {
    return mla_kv_stage_bytes(rows, out_bytes)
        ? NATIVE_BATCH_OK : NATIVE_BATCH_CAPACITY_EXCEEDED;
}

int mla_attention_sdpa_stage_workspace_bytes(
    uint32_t rows, uint32_t committed, uint64_t *out_bytes) {
    return mla_sdpa_stage_bytes(rows, committed, out_bytes)
        ? NATIVE_BATCH_OK : NATIVE_BATCH_CAPACITY_EXCEEDED;
}

int mla_attention_workspace_region_end(
    uint64_t offset, uint64_t size_bytes, uint64_t *out_end) {
    return out_end && mla_workspace_next(offset, size_bytes, out_end)
        ? NATIVE_BATCH_OK : NATIVE_BATCH_CAPACITY_EXCEEDED;
}

static bool mla_mhc_post_stage_bytes(uint32_t rows, uint64_t *out) {
    uint64_t attention = 0u, residual_f32 = 0u, post = 0u, comb = 0u;
    uint64_t residual_bf16 = 0u;
    uint64_t residual_rel = 0u, post_rel = 0u, comb_rel = 0u;
    uint64_t residual_bf16_rel = 0u, output_bf16_rel = 0u;
    return out &&
        mla_u64_multiply(rows,
            (uint64_t)DIM * sizeof(uint16_t), &attention) &&
        mla_u64_multiply(rows,
            (uint64_t)MHC_MULT * DIM * sizeof(float), &residual_f32) &&
        mla_u64_multiply(rows,
            (uint64_t)MHC_MULT * sizeof(float), &post) &&
        mla_u64_multiply(rows,
            (uint64_t)MHC_MULT * MHC_MULT * sizeof(float), &comb) &&
        mla_u64_multiply(rows,
            (uint64_t)MHC_MULT * DIM * sizeof(uint16_t), &residual_bf16) &&
        mla_workspace_next(0u, attention, &residual_rel) &&
        mla_workspace_next(residual_rel, residual_f32, &post_rel) &&
        mla_workspace_next(post_rel, post, &comb_rel) &&
        mla_workspace_next(comb_rel, comb, &residual_bf16_rel) &&
        mla_workspace_next(
            residual_bf16_rel, residual_bf16, &output_bf16_rel) &&
        mla_u64_add(output_bf16_rel, residual_bf16, out);
}

static bool mla_build_v2_plan(
    const MoEInferEngine *engine,
    const MLAPrefillBatchArgs *args,
    const MLARecurrentBatchState *state,
    MLAV2WorkspacePlan *plan) {
    if (!engine || !args || !state || !plan || args->layer >= N_LAYERS) return false;
    memset(plan, 0, sizeof(*plan));
    const uint32_t rows = args->batch_tokens;
    const CompressorState *cs = &engine->comp_state[args->layer];
    if (!mla_u64_multiply(rows,
            (uint64_t)(DIM + 2u * Q_LORA_RANK +
                       2u * N_HEADS * HEAD_DIM) * sizeof(uint16_t),
            &plan->q_stage) ||
        !mla_u64_multiply(rows,
            (uint64_t)N_HEADS * HEAD_DIM * sizeof(uint16_t),
            &plan->q_handoff) ||
        !mla_kv_stage_bytes(rows, &plan->kv_stage) ||
        !mla_sdpa_stage_bytes(rows, args->kv.start_position, &plan->sdpa_stage) ||
        !mla_u64_multiply(rows,
            (uint64_t)N_HEADS * HEAD_DIM * sizeof(uint16_t),
            &plan->sdpa_handoff) ||
        !mla_output_stage_bytes(rows, &plan->output_stage) ||
        !mla_mhc_post_stage_bytes(rows, &plan->mhc_post_stage) ||
        !mla_u64_multiply(rows,
            (uint64_t)KV_LORA_RANK * sizeof(uint16_t), &plan->kv_backup) ||
        !mla_u64_multiply(rows,
            (uint64_t)MHC_MULT * DIM * sizeof(float), &plan->residual_backup) ||
        !mla_u64_multiply(rows,
            (uint64_t)DIM * sizeof(uint16_t), &plan->output_backup)) {
        return false;
    }
    plan->capture_backup = state->attention_capture
        ? plan->output_backup : 0u;

    if (cs->state_kv) {
        uint64_t state_elements = 0u;
        if (!mla_u64_multiply(2u, (uint64_t)cs->ratio, &state_elements) ||
            !mla_u64_multiply(
                state_elements, (uint64_t)cs->out_dim, &state_elements) ||
            !mla_u64_multiply(state_elements, sizeof(float),
                              &plan->comp_state_kv_backup)) {
            return false;
        }
        plan->comp_state_score_backup = plan->comp_state_kv_backup;
        const uint32_t append_rows = cs->n_comp < MAX_COMP_BLOCKS
            ? (uint32_t)fmin((double)rows, (double)(MAX_COMP_BLOCKS - cs->n_comp)) : 0u;
        if (!mla_u64_multiply(append_rows,
                (uint64_t)COMP_HEAD_DIM * sizeof(float),
                &plan->comp_append_backup)) return false;
        if (cs->idx_state_kv) {
            plan->idx_state_kv_backup = (uint64_t)8u * 256u * sizeof(float);
            plan->idx_state_score_backup = plan->idx_state_kv_backup;
            const uint32_t idx_rows = cs->n_idx_comp < MAX_COMP_BLOCKS
                ? (uint32_t)fmin((double)rows,
                    (double)(MAX_COMP_BLOCKS - cs->n_idx_comp)) : 0u;
            if (!mla_u64_multiply(idx_rows,
                    (uint64_t)IDX_HEAD_DIM * sizeof(float),
                    &plan->idx_append_backup)) return false;
        }
    }
    plan->allowed_scratch = MAX_COMP_BLOCKS * sizeof(bool);
    plan->normed_scratch = (uint64_t)DIM * sizeof(float);
    plan->recurrent_scratch =
        (uint64_t)(2u * CSA_OUT_DIM + COMP_HEAD_DIM) * sizeof(float);

    uint64_t cursor = 0u;
#define MLA_ADD_PLAN(field) do { if (!mla_plan_region(&cursor, plan->field)) return false; } while (0)
    MLA_ADD_PLAN(q_stage);
    MLA_ADD_PLAN(q_handoff);
    MLA_ADD_PLAN(kv_stage);
    MLA_ADD_PLAN(sdpa_stage);
    MLA_ADD_PLAN(sdpa_handoff);
    MLA_ADD_PLAN(output_stage);
    MLA_ADD_PLAN(mhc_post_stage);
    MLA_ADD_PLAN(kv_backup);
    MLA_ADD_PLAN(residual_backup);
    MLA_ADD_PLAN(output_backup);
    MLA_ADD_PLAN(capture_backup);
    MLA_ADD_PLAN(comp_state_kv_backup);
    MLA_ADD_PLAN(comp_state_score_backup);
    MLA_ADD_PLAN(comp_append_backup);
    MLA_ADD_PLAN(idx_state_kv_backup);
    MLA_ADD_PLAN(idx_state_score_backup);
    MLA_ADD_PLAN(idx_append_backup);
    MLA_ADD_PLAN(allowed_scratch);
    MLA_ADD_PLAN(normed_scratch);
    MLA_ADD_PLAN(recurrent_scratch);
#undef MLA_ADD_PLAN
    plan->total = cursor;
    return true;
}

void mla_recurrent_batch_state_init(
    MLARecurrentBatchState *state,
    float *residual_rows,
    const float *post_rows,
    const float *comb_rows,
    uint16_t *attention_capture,
    uint32_t capacity_tokens) {
    if (!state) return;
    *state = (MLARecurrentBatchState){
        .residual_rows = residual_rows,
        .post_rows = post_rows,
        .comb_rows = comb_rows,
        .attention_capture = attention_capture,
        .capacity_tokens = capacity_tokens,
        .inject_failure_after_row = UINT32_MAX,
        .self_cookie = (uintptr_t)state,
    };
}

int mla_attention_prefill_batch_workspace_bytes(
    const MoEInferEngine *engine,
    const MLAPrefillBatchArgs *args,
    const MLARecurrentBatchState *state,
    uint64_t *out_bytes) {
    MLAV2WorkspacePlan plan;
    if (!out_bytes || !mla_build_v2_plan(engine, args, state, &plan)) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    *out_bytes = plan.total;
    return NATIVE_BATCH_OK;
}

static int mla_attention_output_batch_internal(
    MlaPipes *P,
    const AttnWeights *aw,
    const uint32_t *positions,
    const uint16_t *sdpa_input,
    uint16_t *attention_output,
    float *projected_output,
    uint32_t batch_tokens,
    NativeBatchWorkspace *workspace,
    MLAAttentionMetrics *metrics) {
    @autoreleasepool {
        if (!P || !P->dev || !P->queue ||
            !P->rope_tail_interleaved_bf16 ||
            !P->dequant_matvec_affine_bf16in_f32out_grp8 ||
            !P->dequant_matvec_affine_v2 || !aw || !positions ||
            !sdpa_input || (!attention_output && !projected_output) || !workspace ||
            batch_tokens == 0u || batch_tokens > workspace->capacity_tokens ||
            workspace->self_cookie != (uintptr_t)workspace || !workspace->storage ||
            !aw->wo_a.packed || !aw->wo_a.scales || !aw->wo_a.biases ||
            aw->wo_a.out_dim != O_GROUPS * O_LORA_RANK ||
            aw->wo_a.in_dim != (N_HEADS / O_GROUPS) * HEAD_DIM ||
            !aw->wo_b.packed || !aw->wo_b.scales || !aw->wo_b.biases ||
            aw->wo_b.out_dim != DIM ||
            aw->wo_b.in_dim != O_GROUPS * O_LORA_RANK) {
            return NATIVE_BATCH_INVALID_ARGUMENT;
        }
        for (uint32_t row = 1u; row < batch_tokens; ++row) {
            if (positions[row] != positions[0] + row) {
                return NATIVE_BATCH_INVALID_POSITION;
            }
        }

        uint64_t scratch_bytes = 0u;
        if (!mla_output_stage_bytes(batch_tokens, &scratch_bytes)) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }
        void *scratch = moe_infer_batch_workspace_reserve(
            workspace, scratch_bytes, NATIVE_BATCH_WORKSPACE_ALIGNMENT);
        if (!scratch) return NATIVE_BATCH_CAPACITY_EXCEEDED;

        const uint64_t attn_bytes =
            (uint64_t)batch_tokens * N_HEADS * HEAD_DIM * sizeof(uint16_t);
        const uint64_t rope_bytes =
            (uint64_t)batch_tokens * (QK_ROPE_DIM / 2) * sizeof(float);
        const uint64_t concat_bytes =
            (uint64_t)batch_tokens * O_GROUPS * O_LORA_RANK * sizeof(float);
        uint64_t cos_rel = 0u, sin_rel = 0u, concat_rel = 0u, projected_rel = 0u;
        if (!mla_workspace_next(0u, attn_bytes, &cos_rel) ||
            !mla_workspace_next(cos_rel, rope_bytes, &sin_rel) ||
            !mla_workspace_next(sin_rel, rope_bytes, &concat_rel) ||
            !mla_workspace_next(concat_rel, concat_bytes, &projected_rel)) {
            return NATIVE_BATCH_CAPACITY_EXCEEDED;
        }

        AttnBufCache *abc = attn_buf_cache_get(P->dev, aw);
        if (!abc || !abc->wo_a_pack_all || !abc->wo_a_sc_all ||
            !abc->wo_a_bi_all || !abc->wo_b_pack || !abc->wo_b_sc ||
            !abc->wo_b_bi) {
            return NATIVE_BATCH_INVALID_STATE;
        }

        const NSUInteger storage_base = (NSUInteger)(
            (uint8_t *)scratch - (uint8_t *)workspace->storage);
        const NSUInteger metal_base =
            (NSUInteger)workspace->metal_buffer_offset;
        const NSUInteger attn_offset = metal_base + storage_base;
        const NSUInteger cos_offset = metal_base + storage_base +
            (NSUInteger)cos_rel;
        const NSUInteger sin_offset = metal_base + storage_base +
            (NSUInteger)sin_rel;
        const NSUInteger concat_offset = metal_base + storage_base +
            (NSUInteger)concat_rel;
        const NSUInteger projected_offset = metal_base + storage_base +
            (NSUInteger)projected_rel;
        uint8_t *bytes = (uint8_t *)workspace->storage;
        memcpy(bytes + storage_base, sdpa_input, (size_t)attn_bytes);
        float *cos_values = (float *)(bytes + storage_base +
            (NSUInteger)cos_rel);
        float *sin_values = (float *)(bytes + storage_base +
            (NSUInteger)sin_rel);
        for (uint32_t row = 0u; row < batch_tokens; ++row) {
            yarn_cos_sin((int)positions[row],
                cos_values + (size_t)row * (QK_ROPE_DIM / 2),
                sin_values + (size_t)row * (QK_ROPE_DIM / 2));
        }

        id<MTLBuffer> batch_buffer =
            mla_workspace_metal_buffer(P->dev, workspace);
        if (!batch_buffer) return NATIVE_BATCH_INVALID_STATE;
        id<MTLCommandBuffer> cb = [P->queue commandBuffer];
        if (!cb) return NATIVE_BATCH_INVALID_STATE;

        enc_rope_bf16_batch_offset(P, cb, batch_buffer,
            attn_offset, cos_offset, sin_offset,
            batch_tokens, N_HEADS, 1u);
        {
            id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
            [encoder setComputePipelineState:
                P->dequant_matvec_affine_bf16in_f32out_grp8];
            [encoder setBuffer:abc->wo_a_pack_all offset:0 atIndex:0];
            [encoder setBuffer:abc->wo_a_sc_all offset:0 atIndex:1];
            [encoder setBuffer:abc->wo_a_bi_all offset:0 atIndex:2];
            [encoder setBuffer:batch_buffer offset:attn_offset atIndex:3];
            [encoder setBuffer:batch_buffer offset:concat_offset atIndex:4];
            const uint32_t out_g = O_LORA_RANK;
            const uint32_t in_dim = (N_HEADS / O_GROUPS) * HEAD_DIM;
            const uint32_t group_size = aw->wo_a.group_size;
            [encoder setBytes:&out_g length:sizeof(out_g) atIndex:5];
            [encoder setBytes:&in_dim length:sizeof(in_dim) atIndex:6];
            [encoder setBytes:&group_size length:sizeof(group_size) atIndex:7];
            [encoder dispatchThreads:MTLSizeMake(
                    O_GROUPS * O_LORA_RANK, batch_tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
        }
        {
            id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
            [encoder setComputePipelineState:P->dequant_matvec_affine_v2];
            [encoder setBuffer:abc->wo_b_pack offset:0 atIndex:0];
            [encoder setBuffer:abc->wo_b_sc offset:0 atIndex:1];
            [encoder setBuffer:abc->wo_b_bi offset:0 atIndex:2];
            [encoder setBuffer:batch_buffer offset:concat_offset atIndex:3];
            [encoder setBuffer:batch_buffer offset:projected_offset atIndex:4];
            const uint32_t out_dim = aw->wo_b.out_dim;
            const uint32_t in_dim = aw->wo_b.in_dim;
            const uint32_t group_size = aw->wo_b.group_size;
            [encoder setBytes:&out_dim length:sizeof(out_dim) atIndex:5];
            [encoder setBytes:&in_dim length:sizeof(in_dim) atIndex:6];
            [encoder setBytes:&group_size length:sizeof(group_size) atIndex:7];
            [encoder setThreadgroupMemoryLength:256 atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(
                    (out_dim + 1u) / 2u, batch_tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            [encoder endEncoding];
        }

        struct timespec start, finish;
        clock_gettime(CLOCK_MONOTONIC, &start);
        [cb commit];
        [cb waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC, &finish);
        if (cb.status == MTLCommandBufferStatusError) {
            return NATIVE_BATCH_INVALID_STATE;
        }
        const float *projected = (const float *)(bytes + storage_base +
            (NSUInteger)projected_rel);
        const size_t projected_elements = (size_t)batch_tokens * DIM;
        if (projected_output) {
            memcpy(projected_output, projected,
                projected_elements * sizeof(float));
        }
        if (attention_output) {
            for (size_t index = 0u; index < projected_elements; ++index) {
                attention_output[index] = mla_f32_to_bf16(projected[index]);
            }
        }
        if (metrics) {
            const int64_t elapsed_ns =
                (int64_t)(finish.tv_sec - start.tv_sec) * 1000000000ll +
                (int64_t)(finish.tv_nsec - start.tv_nsec);
            metrics->output_ns += elapsed_ns > 0 ? (uint64_t)elapsed_ns : 0u;
            metrics->dispatches += 3u;
            metrics->workspace_high_water_bytes = workspace->high_water_bytes;
        }
        return NATIVE_BATCH_OK;
    }
}

int mla_attention_output_batch(
    MlaPipes *P,
    const AttnWeights *aw,
    const uint32_t *positions,
    const uint16_t *sdpa_input,
    uint16_t *attention_output,
    uint32_t batch_tokens,
    NativeBatchWorkspace *workspace,
    MLAAttentionMetrics *metrics) {
    return mla_attention_output_batch_internal(P, aw, positions, sdpa_input,
        attention_output, NULL, batch_tokens, workspace, metrics);
}

static uint8_t *mla_take_region(uint8_t *base, uint64_t *cursor, uint64_t bytes) {
    uint64_t aligned = 0u;
    if (!base || !cursor || !mla_align_workspace(*cursor, &aligned) ||
        bytes > UINT64_MAX - aligned) {
        return NULL;
    }
    *cursor = aligned + bytes;
    return base + aligned;
}

static NativeBatchWorkspace mla_subworkspace(
    const NativeBatchWorkspace *parent,
    void *storage,
    uint64_t bytes,
    uint32_t capacity) {
    const uint64_t local_offset = (uint64_t)(
        (uint8_t *)storage - (uint8_t *)parent->storage);
    NativeBatchWorkspace result = {
        .capacity_tokens = capacity,
        .arena_epoch = parent->arena_epoch,
        .budget_bytes = bytes,
        .allocated_bytes = bytes,
        .storage = storage,
        .metal_buffer = parent->metal_buffer,
        .metal_buffer_offset = parent->metal_buffer_offset + local_offset,
    };
    result.self_cookie = (uintptr_t)&result;
    return result;
}

static void mla_snapshot_compressor(
    const CompressorState *cs,
    const MLAV2WorkspacePlan *plan,
    uint8_t *base,
    uint64_t *cursor,
    MLACompressorSnapshot *snapshot) {
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->original = *cs;
#define MLA_SNAPSHOT(member, source, bytes) do { \
    if ((bytes) != 0u) { \
        snapshot->member = (float *)mla_take_region(base, cursor, (bytes)); \
        memcpy(snapshot->member, (source), (size_t)(bytes)); \
    } \
} while (0)
    MLA_SNAPSHOT(state_kv, cs->state_kv, plan->comp_state_kv_backup);
    MLA_SNAPSHOT(state_score, cs->state_score, plan->comp_state_score_backup);
    snapshot->comp_append_rows = plan->comp_append_backup /
        ((uint64_t)COMP_HEAD_DIM * sizeof(float));
    MLA_SNAPSHOT(comp_append,
        cs->comp_kv ? cs->comp_kv + (size_t)cs->n_comp * COMP_HEAD_DIM : NULL,
        plan->comp_append_backup);
    MLA_SNAPSHOT(idx_state_kv, cs->idx_state_kv, plan->idx_state_kv_backup);
    MLA_SNAPSHOT(idx_state_score, cs->idx_state_score, plan->idx_state_score_backup);
    snapshot->idx_append_rows = plan->idx_append_backup /
        ((uint64_t)IDX_HEAD_DIM * sizeof(float));
    MLA_SNAPSHOT(idx_append,
        cs->idx_comp_kv ? cs->idx_comp_kv + (size_t)cs->n_idx_comp * IDX_HEAD_DIM : NULL,
        plan->idx_append_backup);
#undef MLA_SNAPSHOT
}

static void mla_restore_compressor(
    CompressorState *cs, const MLACompressorSnapshot *snapshot) {
    if (!snapshot->original.state_kv) {
        free(cs->state_kv);
        free(cs->state_score);
        free(cs->comp_kv);
        free(cs->idx_state_kv);
        free(cs->idx_state_score);
        free(cs->idx_comp_kv);
        *cs = snapshot->original;
        return;
    }
    if (snapshot->state_kv) {
        memcpy(snapshot->original.state_kv, snapshot->state_kv,
            (size_t)2u * snapshot->original.ratio *
            snapshot->original.out_dim * sizeof(float));
    }
    if (snapshot->state_score) {
        memcpy(snapshot->original.state_score, snapshot->state_score,
            (size_t)2u * snapshot->original.ratio *
            snapshot->original.out_dim * sizeof(float));
    }
    if (snapshot->comp_append) {
        memcpy(snapshot->original.comp_kv +
                (size_t)snapshot->original.n_comp * COMP_HEAD_DIM,
            snapshot->comp_append,
            (size_t)snapshot->comp_append_rows * COMP_HEAD_DIM * sizeof(float));
    }
    if (snapshot->idx_state_kv) {
        memcpy(snapshot->original.idx_state_kv, snapshot->idx_state_kv,
            (size_t)8u * 256u * sizeof(float));
    }
    if (snapshot->idx_state_score) {
        memcpy(snapshot->original.idx_state_score, snapshot->idx_state_score,
            (size_t)8u * 256u * sizeof(float));
    }
    if (snapshot->idx_append) {
        memcpy(snapshot->original.idx_comp_kv +
                (size_t)snapshot->original.n_idx_comp * IDX_HEAD_DIM,
            snapshot->idx_append,
            (size_t)snapshot->idx_append_rows * IDX_HEAD_DIM * sizeof(float));
    }
    *cs = snapshot->original;
}

static int mla_mhc_post_batch_exact(
    MoEInferEngine *engine,
    MlaPipes *P,
    const uint16_t *attention_output,
    MLARecurrentBatchState *state,
    uint32_t batch_tokens,
    NativeBatchWorkspace *workspace) {
    if (!engine || !P || !P->dev || !P->queue || !attention_output ||
        !state || !state->residual_rows ||
        !state->post_rows || !state->comb_rows || !workspace ||
        batch_tokens == 0u || batch_tokens > workspace->capacity_tokens ||
        workspace->self_cookie != (uintptr_t)workspace || !workspace->storage ||
        !P->f32_to_bf16_vec || !P->mhc_post_bfloat ||
        !P->bf16_to_f32_vec) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }

    uint64_t scratch_bytes = 0u;
    if (!mla_mhc_post_stage_bytes(batch_tokens, &scratch_bytes)) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    void *scratch = moe_infer_batch_workspace_reserve(
        workspace, scratch_bytes, NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    if (!scratch) return NATIVE_BATCH_CAPACITY_EXCEEDED;

    const uint64_t attention_bytes =
        (uint64_t)batch_tokens * DIM * sizeof(uint16_t);
    const uint64_t residual_f32_bytes =
        (uint64_t)batch_tokens * MHC_MULT * DIM * sizeof(float);
    const uint64_t post_bytes =
        (uint64_t)batch_tokens * MHC_MULT * sizeof(float);
    const uint64_t comb_bytes =
        (uint64_t)batch_tokens * MHC_MULT * MHC_MULT * sizeof(float);
    const uint64_t residual_bf16_bytes =
        (uint64_t)batch_tokens * MHC_MULT * DIM * sizeof(uint16_t);
    uint64_t residual_rel = 0u, post_rel = 0u, comb_rel = 0u;
    uint64_t residual_bf16_rel = 0u, output_bf16_rel = 0u;
    if (!mla_workspace_next(0u, attention_bytes, &residual_rel) ||
        !mla_workspace_next(residual_rel, residual_f32_bytes, &post_rel) ||
        !mla_workspace_next(post_rel, post_bytes, &comb_rel) ||
        !mla_workspace_next(comb_rel, comb_bytes, &residual_bf16_rel) ||
        !mla_workspace_next(residual_bf16_rel, residual_bf16_bytes,
            &output_bf16_rel)) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }

    const NSUInteger storage_base = (NSUInteger)(
        (uint8_t *)scratch - (uint8_t *)workspace->storage);
    const NSUInteger metal_base =
        (NSUInteger)workspace->metal_buffer_offset + storage_base;
    uint8_t *bytes = (uint8_t *)workspace->storage + storage_base;
    memcpy(bytes, attention_output, (size_t)attention_bytes);
    memcpy(bytes + (NSUInteger)residual_rel, state->residual_rows,
        (size_t)residual_f32_bytes);
    memcpy(bytes + (NSUInteger)post_rel, state->post_rows,
        (size_t)post_bytes);
    memcpy(bytes + (NSUInteger)comb_rel, state->comb_rows,
        (size_t)comb_bytes);

    id<MTLBuffer> batch_buffer =
        mla_workspace_metal_buffer(P->dev, workspace);
    if (!batch_buffer) return NATIVE_BATCH_INVALID_STATE;
    id<MTLCommandBuffer> command =
        [P->queue commandBuffer];
    if (!command) return NATIVE_BATCH_INVALID_STATE;

    {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) return NATIVE_BATCH_INVALID_STATE;
        [encoder setComputePipelineState:
            P->f32_to_bf16_vec];
        [encoder setBuffer:batch_buffer
                    offset:metal_base + (NSUInteger)residual_rel atIndex:0];
        [encoder setBuffer:batch_buffer
                    offset:metal_base + (NSUInteger)residual_bf16_rel atIndex:1];
        const uint32_t count = batch_tokens * MHC_MULT * DIM;
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) return NATIVE_BATCH_INVALID_STATE;
        [encoder setComputePipelineState:
            P->mhc_post_bfloat];
        [encoder setBuffer:batch_buffer offset:metal_base atIndex:0];
        [encoder setBuffer:batch_buffer
                    offset:metal_base + (NSUInteger)residual_bf16_rel atIndex:1];
        [encoder setBuffer:batch_buffer
                    offset:metal_base + (NSUInteger)post_rel atIndex:2];
        [encoder setBuffer:batch_buffer
                    offset:metal_base + (NSUInteger)comb_rel atIndex:3];
        [encoder setBuffer:batch_buffer
                    offset:metal_base + (NSUInteger)output_bf16_rel atIndex:4];
        const uint32_t hc = MHC_MULT;
        const uint32_t dim = DIM;
        [encoder setBytes:&hc length:sizeof(hc) atIndex:5];
        [encoder setBytes:&dim length:sizeof(dim) atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(DIM, batch_tokens, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) return NATIVE_BATCH_INVALID_STATE;
        [encoder setComputePipelineState:
            P->bf16_to_f32_vec];
        [encoder setBuffer:batch_buffer
                    offset:metal_base + (NSUInteger)output_bf16_rel atIndex:0];
        [encoder setBuffer:batch_buffer
                    offset:metal_base + (NSUInteger)residual_rel atIndex:1];
        const uint32_t count = batch_tokens * MHC_MULT * DIM;
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
    }

    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    memcpy(state->residual_rows, bytes + (NSUInteger)residual_rel,
        (size_t)residual_f32_bytes);
    return NATIVE_BATCH_OK;
}

int mla_attention_prefill_batch_v2(
    MoEInferEngine *engine,
    MlaPipes *P,
    const AttnWeights *aw,
    const MLAPrefillBatchArgs *args,
    MLARecurrentBatchState *state,
    MLAAttentionMetrics *metrics) {
    if (metrics) memset(metrics, 0, sizeof(*metrics));
    if (!engine || !P || !aw || !args || !args->hidden ||
        !args->attention_output || !args->positions || !args->workspace ||
        !args->kv_transaction || !state ||
        state->self_cookie != (uintptr_t)state || !state->residual_rows ||
        !state->post_rows || !state->comb_rows || state->reserved != 0u ||
        ((state->pre_state_snapshots == NULL) !=
         (state->post_state_snapshots == NULL)) ||
        args->batch_tokens == 0u || args->batch_tokens > state->capacity_tokens ||
        state->completed_tokens != 0u || args->layer >= N_LAYERS) {
        return NATIVE_BATCH_INVALID_ARGUMENT;
    }
    if (args->batch_tokens > args->workspace->capacity_tokens) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    if (args->workspace->self_cookie != (uintptr_t)args->workspace ||
        !args->workspace->storage || args->workspace->used_bytes != 0u) {
        return NATIVE_BATCH_INVALID_STATE;
    }
    const int view_rc = moe_infer_kv_transaction_validate_layer_view(
        engine, args->kv_transaction, args->layer, &args->kv);
    if (view_rc != NATIVE_BATCH_OK) return view_rc;
    // Only an authenticated engine-owned view permits touching caller memory.
    // This ordering protects poisoned positions pointers on stale/forged views.
    for (uint32_t row = 0u; row < args->batch_tokens; ++row) {
        if (args->positions[row] != args->kv.start_position + row) {
            return NATIVE_BATCH_INVALID_POSITION;
        }
    }

    // The plan sizes and the recurrent loop below read/snapshot comp_state —
    // materialize any deferred async tails from scalar decode first.
    moe_infer_comp_async_flush(engine);
    MLAV2WorkspacePlan plan;
    if (!mla_build_v2_plan(engine, args, state, &plan)) {
        return NATIVE_BATCH_CAPACITY_EXCEEDED;
    }
    void *all = moe_infer_batch_workspace_reserve(
        args->workspace, plan.total, NATIVE_BATCH_WORKSPACE_ALIGNMENT);
    if (!all) return NATIVE_BATCH_CAPACITY_EXCEEDED;
    // One parent Metal view spans every stage arena; subworkspaces below only
    // borrow it with offsets and never create or release wrappers themselves.
    if (!mla_workspace_metal_buffer(P->dev, args->workspace)) {
        return NATIVE_BATCH_INVALID_STATE;
    }

    uint8_t *base = (uint8_t *)all;
    uint64_t cursor = 0u;
#define MLA_REGION(name) uint8_t *name = mla_take_region(base, &cursor, plan.name)
    MLA_REGION(q_stage);
    MLA_REGION(q_handoff);
    MLA_REGION(kv_stage);
    MLA_REGION(sdpa_stage);
    MLA_REGION(sdpa_handoff);
    MLA_REGION(output_stage);
    MLA_REGION(mhc_post_stage);
    MLA_REGION(kv_backup);
    MLA_REGION(residual_backup);
    MLA_REGION(output_backup);
    MLA_REGION(capture_backup);
#undef MLA_REGION
    memcpy(kv_backup, args->kv.kv_data, (size_t)plan.kv_backup);
    memcpy(residual_backup, state->residual_rows, (size_t)plan.residual_backup);
    memcpy(output_backup, args->attention_output, (size_t)plan.output_backup);
    if (plan.capture_backup != 0u) {
        memcpy(capture_backup, state->attention_capture,
               (size_t)plan.capture_backup);
    }

    MLACompressorSnapshot compressor_snapshot;
    mla_snapshot_compressor(&engine->comp_state[args->layer], &plan,
                            base, &cursor, &compressor_snapshot);
    bool *allowed = (bool *)mla_take_region(
        base, &cursor, plan.allowed_scratch);
    float *normed = (float *)mla_take_region(
        base, &cursor, plan.normed_scratch);
    float *recurrent_scratch = (float *)mla_take_region(
        base, &cursor, plan.recurrent_scratch);
    const size_t recurrent_scratch_floats =
        (size_t)(plan.recurrent_scratch / sizeof(float));

    NativeBatchWorkspace q_ws = mla_subworkspace(
        args->workspace, q_stage, plan.q_stage, args->batch_tokens);
    q_ws.self_cookie = (uintptr_t)&q_ws;
    NativeBatchWorkspace kv_ws = mla_subworkspace(
        args->workspace, kv_stage, plan.kv_stage, args->batch_tokens);
    kv_ws.self_cookie = (uintptr_t)&kv_ws;
    NativeBatchWorkspace sdpa_ws = mla_subworkspace(
        args->workspace, sdpa_stage, plan.sdpa_stage, args->batch_tokens);
    sdpa_ws.self_cookie = (uintptr_t)&sdpa_ws;
    NativeBatchWorkspace output_ws = mla_subworkspace(
        args->workspace, output_stage, plan.output_stage, args->batch_tokens);
    output_ws.self_cookie = (uintptr_t)&output_ws;
    NativeBatchWorkspace mhc_post_ws = mla_subworkspace(
        args->workspace, mhc_post_stage, plan.mhc_post_stage,
        args->batch_tokens);
    mhc_post_ws.self_cookie = (uintptr_t)&mhc_post_ws;

    MLAAttentionMetrics stage = {0};
    int rc = mla_attention_q_batch(P, aw, (const uint16_t *)args->hidden,
        (uint16_t *)q_handoff, args->batch_tokens, &q_ws, &stage);
    if (rc == NATIVE_BATCH_OK) {
        MLAPrefillBatchArgs local = *args;
        local.workspace = &kv_ws;
        MLAAttentionMetrics kv_metrics = {0};
        rc = mla_attention_kv_rope_batch(engine, P, aw, &local, &kv_metrics);
        stage.kv_rope_ns += kv_metrics.kv_rope_ns;
        stage.dispatches += kv_metrics.dispatches;
    }
    if (rc == NATIVE_BATCH_OK) {
        MLAPrefillBatchArgs local = *args;
        local.workspace = &sdpa_ws;
        MLAAttentionMetrics sdpa_metrics = {0};
        rc = mla_attention_sdpa_batch(engine, P, aw, &local,
            (const uint16_t *)q_handoff, (uint16_t *)sdpa_handoff,
            &sdpa_metrics);
        stage.sdpa_ns += sdpa_metrics.sdpa_ns;
        stage.dispatches += sdpa_metrics.dispatches;
    }
    if (rc == NATIVE_BATCH_OK) {
        MLAAttentionMetrics output_metrics = {0};
        rc = mla_attention_output_batch_internal(P, aw, args->positions,
            (const uint16_t *)sdpa_handoff, NULL, (float *)q_handoff,
            args->batch_tokens, &output_ws, &output_metrics);
        stage.output_ns += output_metrics.output_ns;
        stage.dispatches += output_metrics.dispatches;
    }

    if (rc == NATIVE_BATCH_OK) {
        const uint16_t *hidden = (const uint16_t *)args->hidden;
        const float *projected = (const float *)q_handoff;
        uint32_t order_signature = UINT32_C(2166136261);
        for (uint32_t row = 0u; row < args->batch_tokens; ++row) {
            if (state->pre_state_snapshots &&
                moe_infer_capture_state_snapshot(
                    engine, (int)args->layer,
                    (int)args->positions[row],
                    &state->pre_state_snapshots[row]) != NATIVE_BATCH_OK) {
                rc = NATIVE_BATCH_INVALID_STATE;
                break;
            }
            for (uint32_t dim = 0u; dim < DIM; ++dim) {
                normed[dim] = mla_bf16_to_f32(
                    hidden[(size_t)row * DIM + dim]);
            }
            // This loop is the required recurrent prefix scan. Its order is
            // deliberately scalar and matches moe_infer_forward_layer.
            if (engine->compress_ratio[args->layer] > 0u) {
                if (moe_infer_compressor_step_bounded(
                        engine, (int)args->layer,
                        (int)args->positions[row], normed,
                        recurrent_scratch, recurrent_scratch_floats) != 0) {
                    rc = NATIVE_BATCH_INVALID_STATE;
                    break;
                }
            }
            order_signature = mla_order_signature_append(
                order_signature, row, 1u);
            if (engine->compress_ratio[args->layer] == 4u &&
                engine->comp_state[args->layer].n_comp > 0u) {
                if (moe_infer_indexer_step_bounded(
                        engine, (int)args->layer,
                        (int)args->positions[row], normed, allowed,
                        recurrent_scratch, recurrent_scratch_floats) != 0) {
                    rc = NATIVE_BATCH_INVALID_STATE;
                    break;
                }
            }
            order_signature = mla_order_signature_append(
                order_signature, row, 2u);
            if (args->kv_transaction->phase == NATIVE_BATCH_PHASE_VERIFY &&
                moe_infer_kv_transaction_record_recurrent_row(
                    engine, args->kv_transaction, args->layer, row) !=
                    NATIVE_BATCH_OK) {
                rc = NATIVE_BATCH_INVALID_STATE;
                break;
            }
            if (state->post_state_snapshots &&
                moe_infer_capture_state_snapshot(
                    engine, (int)args->layer,
                    (int)args->positions[row] + 1,
                    &state->post_state_snapshots[row]) != NATIVE_BATCH_OK) {
                rc = NATIVE_BATCH_INVALID_STATE;
                break;
            }

            // wo_b left this row unrounded in projected. Only after both
            // recurrent operators do we materialize the rollback-covered BF16
            // output/capture checkpoint consumed by mHC-post.
            const float *projected_row = projected + (size_t)row * DIM;
            uint16_t *attention_row = (uint16_t *)args->attention_output +
                (size_t)row * DIM;
            for (uint32_t dim = 0u; dim < DIM; ++dim) {
                attention_row[dim] = mla_f32_to_bf16(projected_row[dim]);
            }
            if (state->attention_capture) {
                memcpy(state->attention_capture + (size_t)row * DIM,
                    attention_row, (size_t)DIM * sizeof(uint16_t));
            }
            order_signature = mla_order_signature_append(
                order_signature, row, 3u);

            // The attention mHC-post is row-independent once the recurrent
            // prefix scan above has consumed this row's pre-attention input.
            // Record the semantic row order here; all rows are then executed
            // by the same Metal BF16 kernel as scalar decode in one dispatch.
            order_signature = mla_order_signature_append(
                order_signature, row, 4u);
            if (state->inject_failure_after_row == row) {
                rc = NATIVE_BATCH_INVALID_STATE;
                break;
            }
        }
        if (rc == NATIVE_BATCH_OK) {
            rc = mla_mhc_post_batch_exact(
                engine,
                P,
                (const uint16_t *)args->attention_output,
                state,
                args->batch_tokens,
                &mhc_post_ws);
        }
        stage.operator_order_signature = order_signature;
    }

    if (rc == NATIVE_BATCH_OK) {
        const int original_length = engine->kv_cache[args->layer].len;
        engine->kv_cache[args->layer].len =
            (int)args->kv.start_position + (int)args->batch_tokens;
        rc = moe_infer_kv_transaction_mark_layer_staged(
            engine, args->kv_transaction, args->layer, args->batch_tokens);
        if (rc != NATIVE_BATCH_OK) {
            engine->kv_cache[args->layer].len = original_length;
        }
    }

    if (rc != NATIVE_BATCH_OK) {
        memcpy(args->kv.kv_data, kv_backup, (size_t)plan.kv_backup);
        memcpy(state->residual_rows, residual_backup,
               (size_t)plan.residual_backup);
        memcpy(args->attention_output, output_backup,
               (size_t)plan.output_backup);
        if (plan.capture_backup != 0u) {
            memcpy(state->attention_capture, capture_backup,
                   (size_t)plan.capture_backup);
        }
        mla_restore_compressor(&engine->comp_state[args->layer],
                               &compressor_snapshot);
        state->completed_tokens = 0u;
    } else {
        if (state->capture) {
            for (uint32_t row = 0u; row < args->batch_tokens; ++row) {
                state->capture(state->capture_context, args->layer,
                    args->positions[row],
                    (const uint16_t *)args->attention_output +
                        (size_t)row * DIM,
                    DIM);
            }
        }
        state->completed_tokens = args->batch_tokens;
    }

    if (metrics) {
        *metrics = stage;
        metrics->workspace_high_water_bytes =
            args->workspace->high_water_bytes;
    }
    return rc;
}

int mla_attention_prefill_batch(
    MlaPipes *P,
    const AttnWeights *aw,
    const MLAPrefillBatchArgs *args,
    MLAAttentionMetrics *metrics) {
    (void)P;
    (void)aw;
    (void)args;
    if (metrics) memset(metrics, 0, sizeof(*metrics));
    // Never hide a token loop or silently omit recurrent state. Callers must
    // opt into the explicit v2 boundary above.
    return NATIVE_BATCH_INVALID_STATE;
}
