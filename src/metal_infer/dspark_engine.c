// DSpark Speculative Decoding Engine — Forward Orchestration
//
// Implements the DSpark 3-layer draft backbone forward pass:
//   forward_spec(main_hidden, anchor_token, start_pos):
//     1. forward_embed: main_proj(main_hidden) → main_x; embed([anchor, noise×4]) → draft_h
//     2. For each DSpark layer (0,1,2):
//        Block.forward(draft_h, start_pos, input_ids, main_x):
//          hc_pre(attn) → attn_norm → DSparkAttention → hc_post(attn)
//          hc_pre(ffn)  → ffn_norm  → MoE(256 INT8 experts) → hc_post(ffn)
//     3. forward_head: hc_head → norm → lm_head → logits [block_size, vocab]
//     4. Markov Head: sequential correction → corrected logits → sample tokens
//     5. Confidence Head: per-position acceptance estimate
//
// Integration points with target engine (MoEInferEngine):
//   - Borrows: device, queue, embed weights, lm_head, expert I/O pool
//   - Receives: main_hidden (accumulated from target layers 40/41/42 during target forward)
//   - Returns: draft_tokens[5], confidence[5]
//
// Key differences from target forward (engine.c moe_infer_forward_layer):
//   - Attention is DSparkAttention (no compressor/indexer, dense sliding window)
//   - Expert format is INT8 + E8M0 (not MXFP4)
//   - No hash routing (all layers use score-based routing)
//   - KV cache is ephemeral (reset each draft step, only main_kv persists across steps)
//   - Input to layer 0 comes from main_proj(target_hidden), not embed(token)
//
#include "dspark_engine.h"
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

// Forward declarations for dspark_attention.c
extern void dspark_attention_forward(
    DSparkEngine *eng, int layer_idx,
    const float *normed_input,   // [DSPARK_BLOCK_SIZE, DIM]
    const float *main_x,         // [DIM] (the single main_x token)
    float *attn_out,             // [DSPARK_BLOCK_SIZE, DIM] output
    int start_pos
);

// ============================================================================
// Helpers: RMSNorm, mHC, embedding, lm_head
// ============================================================================

static void rms_norm_cpu(const float *x, const float *weight, float *out, int dim) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / (float)dim + 1e-6f);
    if (weight) {
        for (int i = 0; i < dim; i++) out[i] = x[i] * rms * weight[i];
    } else {
        for (int i = 0; i < dim; i++) out[i] = x[i] * rms;
    }
}

// FP8 E4M3 dequant matvec on CPU (for attention projections).
// w: [out_dim, in_dim] uint8 (FP8 E4M3), scale: [ceil(out/bs), ceil(in/bs)] uint8 (E8M0).
// out[r] = sum_c (dequant(w[r,c]) * x[c])
// For simplicity, uses per-block scale: w_float = fp8_to_float(w[r,c]) * e8m0_to_float(scale[r/bs, c/bs])
static void fp8_matvec_cpu(
    const uint8_t *w, const uint8_t *scale,
    const float *x, float *out,
    int out_dim, int in_dim, int block_size
) {
    int scale_cols = (in_dim + block_size - 1) / block_size;
    for (int r = 0; r < out_dim; r++) {
        float acc = 0.0f;
        int sr = r / block_size;
        for (int c = 0; c < in_dim; c++) {
            int sc = c / block_size;
            // E8M0 scale decode via bit-manipulation
            uint8_t sb = scale[sr * scale_cols + sc];
            uint32_t sf_bits = (uint32_t)sb << 23;
            float sf = *(float*)&sf_bits;  // = 2^(sb - 127)
            // FP8 E4M3 decode
            uint8_t byte = w[r * in_dim + c];
            int sign = (byte >> 7) & 1;
            int exp = (byte >> 3) & 0xF;
            int man = byte & 0x7;
            float val;
            if (exp == 0) {
                val = (man == 0) ? 0.0f : (float)man / 8.0f * (1.0f / 64.0f);  // subnormal
            } else {
                val = (1.0f + (float)man / 8.0f) * exp2f((float)exp - 7.0f);
            }
            if (sign) val = -val;
            acc += val * sf * x[c];
        }
        out[r] = acc;
    }
}

// mHC pre: reduces [MHC_MULT, DIM] → [DIM] via learned weighted sum + sinkhorn.
// Reuses the same algorithm as engine.c's mhc_pre (CPU version from mhc.c).
// For DSpark we keep this simple — delegate to the existing mhc.h functions.
extern void mhc_pre(const float *residual, const float *fn, const float *base,
                    const float *scale, float *out_input, float *out_post,
                    float *out_comb, int dim, int hc_mult, int sinkhorn_iters, float eps);
extern void mhc_post(const float *x, const float *residual,
                     const float *post, const float *comb,
                     float *out, int dim, int hc_mult);
// hc_head (simplified mhc_pre without sinkhorn, used for final output compression)
// Uses hyper_head_compress from mhc.c
extern void hyper_head_compress(const float *fn, const float *base, const float *scale,
                                const float *residual, float *out);

// ============================================================================
// dspark_init
// ============================================================================

DSparkEngine *dspark_init(
    const char *dspark_weight_dir,
    const char *packed_expert_dir,
    void *target_engine
) {
    DSparkEngine *eng = calloc(1, sizeof(DSparkEngine));
    if (!eng) return NULL;

    MoEInferEngine *target = (MoEInferEngine *)target_engine;
    eng->target_engine = target_engine;
    eng->device = target->device;
    eng->queue = target->queue;
    eng->vocab_size = target->vocab_size;
    eng->block_size = DSPARK_BLOCK_SIZE;
    eng->markov_rank = DSPARK_MARKOV_RANK;
    eng->noise_token_id = DSPARK_NOISE_TOKEN_ID;

    // Allocate scratch buffers
    eng->buf_main_hidden = calloc(3 * DIM, sizeof(float));
    eng->buf_main_x = calloc(DIM, sizeof(float));
    eng->buf_draft_hidden = calloc(DSPARK_BLOCK_SIZE * MHC_MULT * DIM, sizeof(float));
    eng->buf_draft_logits = calloc(DSPARK_BLOCK_SIZE * eng->vocab_size, sizeof(float));
    eng->buf_confidence = calloc(DSPARK_BLOCK_SIZE, sizeof(float));

    if (!eng->buf_main_hidden || !eng->buf_main_x || !eng->buf_draft_hidden ||
        !eng->buf_draft_logits || !eng->buf_confidence) {
        fprintf(stderr, "[dspark] allocation failed\n");
        dspark_deinit(eng);
        return NULL;
    }

    // Allocate KV caches (tiny: window×KV_LORA_RANK per layer, f16)
    for (int l = 0; l < DSPARK_N_LAYERS; l++) {
        eng->kv_cache[l].main_kv = calloc(DSPARK_WINDOW_SIZE * KV_LORA_RANK, sizeof(uint16_t));
        eng->kv_cache[l].draft_kv = calloc(DSPARK_BLOCK_SIZE * KV_LORA_RANK, sizeof(uint16_t));
        eng->kv_cache[l].main_len = 0;
        eng->kv_cache[l].draft_len = 0;
    }

    // Open packed expert files (INT8 format)
    for (int l = 0; l < DSPARK_N_LAYERS; l++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/mtp_layer_%02d.bin", packed_expert_dir, l);
        FILE *f = fopen(path, "r");
        if (f) {
            eng->expert_fd[l] = fileno(f);
            // Don't fclose — keep fd open for pread
        } else {
            fprintf(stderr, "[dspark] WARNING: cannot open %s\n", path);
            eng->expert_fd[l] = -1;
        }
    }

    // Share expert buffers with target engine (same 2MB-aligned pool)
    for (int k = 0; k < 6; k++) {
        eng->expert_buf[k] = target->expert_buf[k];
    }

    // =========================================================
    // Load DSpark weights from dspark_weight_dir
    // =========================================================
    if (dspark_weight_dir) {
        char path[1024];
        // Helper: read a binary file into malloc'd buffer, return size or -1
        #define LOAD_BIN(dst, fpath, expected_bytes) do { \
            FILE *_f = fopen(fpath, "rb"); \
            if (!_f) { fprintf(stderr, "[dspark] WARNING: cannot open %s\n", fpath); } \
            else { \
                dst = (typeof(dst))malloc(expected_bytes); \
                if (dst && fread((void*)dst, 1, expected_bytes, _f) != (size_t)(expected_bytes)) { \
                    fprintf(stderr, "[dspark] WARNING: short read %s\n", fpath); \
                } \
                fclose(_f); \
            } \
        } while(0)

        // --- Markov Head ---
        size_t markov_bytes = (size_t)eng->vocab_size * DSPARK_MARKOV_RANK * sizeof(float);
        snprintf(path, sizeof(path), "%s/markov_w1.bin", dspark_weight_dir);
        LOAD_BIN(eng->head.markov_w1, path, markov_bytes);
        snprintf(path, sizeof(path), "%s/markov_w2.bin", dspark_weight_dir);
        LOAD_BIN(eng->head.markov_w2, path, markov_bytes);

        // --- Confidence Head ---
        size_t conf_bytes = (DIM + DSPARK_MARKOV_RANK) * sizeof(float);
        snprintf(path, sizeof(path), "%s/confidence_head.bin", dspark_weight_dir);
        LOAD_BIN(eng->head.confidence_proj, path, conf_bytes);

        // --- Per-layer weights ---
        for (int l = 0; l < DSPARK_N_LAYERS; l++) {
            DSparkLayerWeights *lw = &eng->layers[l];
            char ldir[1024];
            snprintf(ldir, sizeof(ldir), "%s/layer_%02d", dspark_weight_dir, l);

            // Attention weights (FP8 raw bytes)
            #define LOAD_FP8(field, fname, od, id, bs) do { \
                snprintf(path, sizeof(path), "%s/%s_weight.bin", ldir, fname); \
                LOAD_BIN(field.weight, path, (size_t)(od)*(id)); \
                field.out_dim = od; field.in_dim = id; field.block_size = bs; \
                snprintf(path, sizeof(path), "%s/%s_scale.bin", ldir, fname); \
                size_t sc_sz = (size_t)(((od)+bs-1)/bs) * (((id)+bs-1)/bs); \
                LOAD_BIN(field.scale, path, sc_sz); \
            } while(0)

            LOAD_FP8(lw->attn.wq_a, "attn_wq_a", Q_LORA_RANK, DIM, 128);
            LOAD_FP8(lw->attn.wq_b, "attn_wq_b", N_HEADS*HEAD_DIM, Q_LORA_RANK, 128);
            LOAD_FP8(lw->attn.wkv,  "attn_wkv",  KV_LORA_RANK, DIM, 128);
            LOAD_FP8(lw->attn.wo_a, "attn_wo_a", O_GROUPS*O_LORA_RANK, DIM, 128);
            LOAD_FP8(lw->attn.wo_b, "attn_wo_b", DIM, O_GROUPS*O_LORA_RANK, 128);

            // Norms (f32)
            snprintf(path, sizeof(path), "%s/attn_q_norm_weight.bin", ldir);
            LOAD_BIN(lw->attn.q_norm, path, Q_LORA_RANK * sizeof(float));
            snprintf(path, sizeof(path), "%s/attn_kv_norm_weight.bin", ldir);
            LOAD_BIN(lw->attn.kv_norm, path, KV_LORA_RANK * sizeof(float));
            snprintf(path, sizeof(path), "%s/attn_attn_sink.bin", ldir);
            LOAD_BIN(lw->attn.attn_sink, path, N_HEADS * sizeof(float));
            snprintf(path, sizeof(path), "%s/attn_norm_weight.bin", ldir);
            LOAD_BIN(lw->attn_norm, path, DIM * sizeof(float));
            snprintf(path, sizeof(path), "%s/ffn_norm_weight.bin", ldir);
            LOAD_BIN(lw->ffn_norm, path, DIM * sizeof(float));

            // Gate (f32)
            snprintf(path, sizeof(path), "%s/gate_weight.bin", ldir);
            LOAD_BIN(lw->gate_weight, path, N_EXPERTS * DIM * sizeof(float));
            snprintf(path, sizeof(path), "%s/gate_bias.bin", ldir);
            LOAD_BIN(lw->gate_bias, path, N_EXPERTS * sizeof(float));

            // HC weights (f32)
            snprintf(path, sizeof(path), "%s/hc_attn_fn.bin", ldir);
            LOAD_BIN(lw->hc_attn_fn, path, 24 * MHC_MULT * DIM * sizeof(float));
            snprintf(path, sizeof(path), "%s/hc_attn_base.bin", ldir);
            LOAD_BIN(lw->hc_attn_base, path, 24 * sizeof(float));
            snprintf(path, sizeof(path), "%s/hc_attn_scale.bin", ldir);
            LOAD_BIN(lw->hc_attn_scale, path, 3 * sizeof(float));
            snprintf(path, sizeof(path), "%s/hc_ffn_fn.bin", ldir);
            LOAD_BIN(lw->hc_ffn_fn, path, 24 * MHC_MULT * DIM * sizeof(float));
            snprintf(path, sizeof(path), "%s/hc_ffn_base.bin", ldir);
            LOAD_BIN(lw->hc_ffn_base, path, 24 * sizeof(float));
            snprintf(path, sizeof(path), "%s/hc_ffn_scale.bin", ldir);
            LOAD_BIN(lw->hc_ffn_scale, path, 3 * sizeof(float));

            // Layer 0 special: main_proj
            if (l == 0) {
                LOAD_FP8(eng->head.main_proj, "main_proj", DIM, 3*DIM, 128);
                snprintf(path, sizeof(path), "%s/main_norm_weight.bin", ldir);
                LOAD_BIN(eng->head.main_norm, path, DIM * sizeof(float));
            }
            // Layer 2 special: hc_head + final norm
            if (l == 2) {
                snprintf(path, sizeof(path), "%s/hc_head_fn.bin", ldir);
                LOAD_BIN(eng->head.hc_head_fn, path, MHC_MULT * MHC_MULT * DIM * sizeof(float));
                snprintf(path, sizeof(path), "%s/hc_head_base.bin", ldir);
                LOAD_BIN(eng->head.hc_head_base, path, MHC_MULT * sizeof(float));
                snprintf(path, sizeof(path), "%s/hc_head_scale.bin", ldir);
                LOAD_BIN(eng->head.hc_head_scale, path, 1 * sizeof(float));
                snprintf(path, sizeof(path), "%s/norm_weight.bin", ldir);
                LOAD_BIN(eng->head.final_norm, path, DIM * sizeof(float));
            }
        }
        #undef LOAD_BIN
        #undef LOAD_FP8

        // Verify critical weights loaded
        if (eng->head.markov_w1 && eng->head.markov_w2) {
            printf("[dspark] Markov Head loaded: W1+W2 [%d × %d] (%.1f MB)\n",
                   eng->vocab_size, eng->markov_rank,
                   (float)(markov_bytes * 2) / (1024.0f * 1024.0f));
        } else {
            fprintf(stderr, "[dspark] WARNING: Markov Head weights not loaded!\n");
        }
        if (eng->head.main_proj.weight) {
            printf("[dspark] main_proj loaded: [%d × %d] FP8\n", DIM, 3*DIM);
        }
    }

    printf("[dspark] engine initialized (block_size=%d, layers=%d, expert_format=INT8+E8M0)\n",
           eng->block_size, DSPARK_N_LAYERS);
    eng->initialized = true;
    return eng;
}

// ============================================================================
// dspark_deinit
// ============================================================================

void dspark_deinit(DSparkEngine *eng) {
    if (!eng) return;
    free(eng->buf_main_hidden);
    free(eng->buf_main_x);
    free(eng->buf_draft_hidden);
    free(eng->buf_draft_logits);
    free(eng->buf_confidence);
    for (int l = 0; l < DSPARK_N_LAYERS; l++) {
        free(eng->kv_cache[l].main_kv);
        free(eng->kv_cache[l].draft_kv);
        // Note: expert_fd is borrowed from target, don't close here
    }
    free(eng);
}

// ============================================================================
// dspark_reset
// ============================================================================

void dspark_reset(DSparkEngine *eng) {
    for (int l = 0; l < DSPARK_N_LAYERS; l++) {
        eng->kv_cache[l].main_len = 0;
        eng->kv_cache[l].draft_len = 0;
        memset(eng->kv_cache[l].main_kv, 0, DSPARK_WINDOW_SIZE * KV_LORA_RANK * sizeof(uint16_t));
    }
    memset(eng->buf_main_hidden, 0, 3 * DIM * sizeof(float));
}

// ============================================================================
// dspark_accumulate_target_hidden
// ============================================================================

void dspark_accumulate_target_hidden(DSparkEngine *eng, const float *hidden, int layer_idx) {
    // hidden: [MHC_MULT, DIM] f32 — current layer's full residual
    // We need mean over MHC_MULT dimension: out[d] = mean(hidden[m*DIM + d] for m in 0..MHC_MULT)
    // Store at offset (layer_idx - 40) * DIM in buf_main_hidden
    int slot = -1;
    for (int i = 0; i < DSPARK_TARGET_LAYER_IDS_COUNT; i++) {
        if (DSPARK_TARGET_LAYER_IDS[i] == layer_idx) { slot = i; break; }
    }
    if (slot < 0) return;

    float *dst = eng->buf_main_hidden + slot * DIM;
    const float inv_mult = 1.0f / (float)MHC_MULT;
    for (int d = 0; d < DIM; d++) {
        float sum = 0.0f;
        for (int m = 0; m < MHC_MULT; m++) {
            sum += hidden[m * DIM + d];
        }
        dst[d] = sum * inv_mult;
    }
}

// ============================================================================
// dspark_update_main_kv
// ============================================================================

void dspark_update_main_kv(DSparkEngine *eng, const uint16_t *target_kv_entry, int pos) {
    // Write target's latest KV entry into DSpark's sliding window for all layers.
    // DSpark layers share the same main_kv (they all attend to the same target sequence).
    int win_pos = pos % DSPARK_WINDOW_SIZE;
    for (int l = 0; l < DSPARK_N_LAYERS; l++) {
        memcpy(eng->kv_cache[l].main_kv + win_pos * KV_LORA_RANK,
               target_kv_entry, KV_LORA_RANK * sizeof(uint16_t));
        if (eng->kv_cache[l].main_len < DSPARK_WINDOW_SIZE) {
            eng->kv_cache[l].main_len = pos + 1;
            if (eng->kv_cache[l].main_len > DSPARK_WINDOW_SIZE)
                eng->kv_cache[l].main_len = DSPARK_WINDOW_SIZE;
        }
    }
}

// ============================================================================
// dspark_forward — Main DSpark 3-layer forward pass
// ============================================================================

int dspark_forward(
    DSparkEngine *eng,
    const float *main_hidden,   // [3 * DIM] or NULL (use internal buf_main_hidden)
    int anchor_token_id,
    int start_pos,
    float *draft_logits,        // [block_size * vocab_size] output
    float *confidence           // [block_size] output, or NULL
) {
    if (!eng || !eng->initialized) return 0;
    MoEInferEngine *target = (MoEInferEngine *)eng->target_engine;

    // Safety: skip if critical weights not loaded
    if (!eng->head.main_proj.weight || !eng->head.markov_w1 || !eng->layers[0].hc_attn_fn) {
        fprintf(stderr, "[dspark] forward skipped: weights not loaded (main_proj=%p, markov=%p, hc=%p)\n",
                (void*)eng->head.main_proj.weight, (void*)eng->head.markov_w1, (void*)eng->layers[0].hc_attn_fn);
        return 0;
    }

    const int bs = eng->block_size;
    const int vocab = eng->vocab_size;

    // =====================================================================
    // Phase 3 simplified forward: skip 3-layer backbone (placeholder),
    // just produce base logits from embedding + lm_head for each draft position.
    // The Markov Head correction will still provide useful draft tokens.
    // Full backbone will be enabled once stack overflow / mhc_pre issues are fixed.
    // =====================================================================

    // For each draft position, compute logits from the embedding of the noise token
    // (or anchor for pos 0). This is a simplified "embedding-only" draft.
    for (int k = 0; k < bs; k++) {
        int tok = (k == 0) ? anchor_token_id : (int)eng->noise_token_id;
        const float *emb_row = target->embed + tok * DIM;

        // norm + lm_head → logits
        float normed[DIM];
        rms_norm_cpu(emb_row, eng->head.final_norm, normed, DIM);

        float *logits_k = draft_logits + k * vocab;
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    vocab, DIM, 1.0f, target->lm_head, DIM,
                    normed, 1, 0.0f, logits_k, 1);
    }

    // Confidence: just return 0.5 for all positions (neutral)
    if (confidence) {
        for (int k = 0; k < bs; k++) confidence[k] = 0.5f;
    }

    return bs;
}

// ============================================================================
// dspark_markov_sample — Sequential Markov Head correction + greedy sampling
// ============================================================================

int dspark_markov_sample(
    DSparkEngine *eng,
    float *draft_logits,       // [block_size, vocab_size] — modified in-place
    int anchor_token_id,
    float *corrected_logits,   // may alias draft_logits
    uint32_t *draft_tokens
) {
    const int bs = eng->block_size;
    const int vocab = eng->vocab_size;
    const int rank = eng->markov_rank;
    const float *w1 = eng->head.markov_w1;
    const float *w2 = eng->head.markov_w2;

    if (!w1 || !w2) {
        // No Markov Head loaded — just do argmax on base logits
        for (int k = 0; k < bs; k++) {
            float *logits_k = draft_logits + k * vocab;
            int best = 0;
            float best_val = logits_k[0];
            for (int v = 1; v < vocab; v++) {
                if (logits_k[v] > best_val) { best_val = logits_k[v]; best = v; }
            }
            draft_tokens[k] = (uint32_t)best;
        }
        if (corrected_logits != draft_logits) {
            memcpy(corrected_logits, draft_logits, bs * vocab * sizeof(float));
        }
        return bs;
    }

    // Sequential Markov correction:
    // For each position k:
    //   logits[k] += B(prev_token) = W1[prev_token] · W2^T
    //   token[k] = argmax(logits[k])
    //   prev_token = token[k]  (for next position)
    int prev_token = anchor_token_id;
    for (int k = 0; k < bs; k++) {
        float *logits_k = draft_logits + k * vocab;

        // Add Markov bias: logits[v] += dot(W1[prev_token], W2[v])
        const float *w1_row = w1 + prev_token * rank;
        for (int v = 0; v < vocab; v++) {
            const float *w2_row = w2 + v * rank;
            float acc = 0.0f;
            // Unrolled dot product (rank=256, unroll by 8)
            int r = 0;
            for (; r + 7 < rank; r += 8) {
                acc += w1_row[r+0]*w2_row[r+0] + w1_row[r+1]*w2_row[r+1]
                     + w1_row[r+2]*w2_row[r+2] + w1_row[r+3]*w2_row[r+3]
                     + w1_row[r+4]*w2_row[r+4] + w1_row[r+5]*w2_row[r+5]
                     + w1_row[r+6]*w2_row[r+6] + w1_row[r+7]*w2_row[r+7];
            }
            for (; r < rank; r++) acc += w1_row[r] * w2_row[r];
            logits_k[v] += acc;
        }

        // Greedy argmax
        int best = 0;
        float best_val = logits_k[0];
        for (int v = 1; v < vocab; v++) {
            if (logits_k[v] > best_val) { best_val = logits_k[v]; best = v; }
        }
        draft_tokens[k] = (uint32_t)best;
        prev_token = best;
    }

    if (corrected_logits != draft_logits) {
        memcpy(corrected_logits, draft_logits, bs * vocab * sizeof(float));
    }
    return bs;
}
