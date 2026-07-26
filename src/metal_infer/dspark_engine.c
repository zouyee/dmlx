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
#include "mhc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
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

// mHC pre/post: now included via mhc.h

// ============================================================================
// FP8 E4M3 weight dequantization to f32 (done once at init for attention weights)
// ============================================================================

static void dequant_fp8_to_f32(const DSparkFP8Weight *w, float *out) {
    if (!w->weight || !w->scale) return;
    int out_dim = w->out_dim, in_dim = w->in_dim, bs = w->block_size;
    int scale_cols = (in_dim + bs - 1) / bs;
    for (int r = 0; r < out_dim; r++) {
        int sr = r / bs;
        for (int c = 0; c < in_dim; c++) {
            int sc = c / bs;
            uint8_t sb = w->scale[sr * scale_cols + sc];
            uint32_t sf_bits = (uint32_t)sb << 23;
            float sf = *(float*)&sf_bits;
            uint8_t byte = w->weight[r * in_dim + c];
            int sign = (byte >> 7) & 1;
            int exp = (byte >> 3) & 0xF;
            int man = byte & 0x7;
            float val;
            if (exp == 0) {
                val = (man == 0) ? 0.0f : (float)man / 8.0f * (1.0f / 64.0f);
            } else {
                val = (1.0f + (float)man / 8.0f) * exp2f((float)exp - 7.0f);
            }
            if (sign) val = -val;
            out[r * in_dim + c] = val * sf;
        }
    }
}

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

    // Open packed expert files (INT8 format) and mmap for GPU dispatch
    eng->expert_mmap_size = (size_t)N_EXPERTS * DSPARK_EXPERT_SIZE;
    for (int l = 0; l < DSPARK_N_LAYERS; l++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/mtp_layer_%02d.bin", packed_expert_dir, l);
        int fd = open(path, O_RDONLY);
        if (fd >= 0) {
            eng->expert_fd[l] = fd;
            // mmap the entire file for zero-copy GPU buffer creation
            void *mapped = mmap(NULL, eng->expert_mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapped == MAP_FAILED) {
                fprintf(stderr, "[dspark] WARNING: mmap failed for %s, falling back to pread\n", path);
                eng->expert_mmap[l] = NULL;
            } else {
                eng->expert_mmap[l] = (uint8_t *)mapped;
                // Advise sequential access for prefetching
                madvise(mapped, eng->expert_mmap_size, MADV_SEQUENTIAL);
                printf("[dspark] mmap'd layer %d experts: %.1f GB\n", l,
                       (float)eng->expert_mmap_size / (1024.0f * 1024.0f * 1024.0f));
            }
        } else {
            fprintf(stderr, "[dspark] WARNING: cannot open %s\n", path);
            eng->expert_fd[l] = -1;
            eng->expert_mmap[l] = NULL;
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

            // Shared expert (FP8 E4M3 + E8M0, block_size=128)
            {
                DSparkFP8Weight se_w1, se_w3, se_w2;
                LOAD_FP8(se_w1, "shared_w1", INTERMEDIATE, DIM, 128);
                LOAD_FP8(se_w3, "shared_w3", INTERMEDIATE, DIM, 128);
                LOAD_FP8(se_w2, "shared_w2", DIM, INTERMEDIATE, 128);
                if (se_w1.weight) {
                    lw->shared_w1_f32 = (float *)malloc((size_t)INTERMEDIATE * DIM * sizeof(float));
                    if (lw->shared_w1_f32) dequant_fp8_to_f32(&se_w1, lw->shared_w1_f32);
                }
                if (se_w3.weight) {
                    lw->shared_w3_f32 = (float *)malloc((size_t)INTERMEDIATE * DIM * sizeof(float));
                    if (lw->shared_w3_f32) dequant_fp8_to_f32(&se_w3, lw->shared_w3_f32);
                }
                if (se_w2.weight) {
                    lw->shared_w2_f32 = (float *)malloc((size_t)DIM * INTERMEDIATE * sizeof(float));
                    if (lw->shared_w2_f32) dequant_fp8_to_f32(&se_w2, lw->shared_w2_f32);
                }
                // Free raw FP8 data (only need f32)
                free((void*)se_w1.weight); free((void*)se_w1.scale);
                free((void*)se_w3.weight); free((void*)se_w3.scale);
                free((void*)se_w2.weight); free((void*)se_w2.scale);
            }

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

        // Pre-dequantize FP8 attention weights to f32 for fast cblas_sgemv
        for (int l = 0; l < DSPARK_N_LAYERS; l++) {
            DSparkAttnWeights *aw = &eng->layers[l].attn;
            if (aw->wq_a.weight) {
                aw->wq_a_f32 = (float *)malloc((size_t)aw->wq_a.out_dim * aw->wq_a.in_dim * sizeof(float));
                if (aw->wq_a_f32) dequant_fp8_to_f32(&aw->wq_a, aw->wq_a_f32);
            }
            if (aw->wq_b.weight) {
                aw->wq_b_f32 = (float *)malloc((size_t)aw->wq_b.out_dim * aw->wq_b.in_dim * sizeof(float));
                if (aw->wq_b_f32) dequant_fp8_to_f32(&aw->wq_b, aw->wq_b_f32);
            }
            if (aw->wkv.weight) {
                aw->wkv_f32 = (float *)malloc((size_t)aw->wkv.out_dim * aw->wkv.in_dim * sizeof(float));
                if (aw->wkv_f32) dequant_fp8_to_f32(&aw->wkv, aw->wkv_f32);
            }
            if (aw->wo_a.weight) {
                aw->wo_a_f32 = (float *)malloc((size_t)aw->wo_a.out_dim * aw->wo_a.in_dim * sizeof(float));
                if (aw->wo_a_f32) dequant_fp8_to_f32(&aw->wo_a, aw->wo_a_f32);
            }
            if (aw->wo_b.weight) {
                aw->wo_b_f32 = (float *)malloc((size_t)aw->wo_b.out_dim * aw->wo_b.in_dim * sizeof(float));
                if (aw->wo_b_f32) dequant_fp8_to_f32(&aw->wo_b, aw->wo_b_f32);
            }
        }
        {
            // Report memory usage
            size_t attn_mem = 0;
            for (int l = 0; l < DSPARK_N_LAYERS; l++) {
                DSparkAttnWeights *aw = &eng->layers[l].attn;
                if (aw->wq_a_f32) attn_mem += (size_t)aw->wq_a.out_dim * aw->wq_a.in_dim * 4;
                if (aw->wq_b_f32) attn_mem += (size_t)aw->wq_b.out_dim * aw->wq_b.in_dim * 4;
                if (aw->wkv_f32)  attn_mem += (size_t)aw->wkv.out_dim * aw->wkv.in_dim * 4;
                if (aw->wo_a_f32) attn_mem += (size_t)aw->wo_a.out_dim * aw->wo_a.in_dim * 4;
                if (aw->wo_b_f32) attn_mem += (size_t)aw->wo_b.out_dim * aw->wo_b.in_dim * 4;
            }
            printf("[dspark] attention weights dequantized to f32: %.1f MB\n",
                   (float)attn_mem / (1024.0f * 1024.0f));
        }

        // Pre-dequantize main_proj to f32
        if (eng->head.main_proj.weight) {
            size_t mp_size = (size_t)eng->head.main_proj.out_dim * eng->head.main_proj.in_dim;
            eng->head.main_proj_f32 = (float *)malloc(mp_size * sizeof(float));
            if (eng->head.main_proj_f32) {
                dequant_fp8_to_f32(&eng->head.main_proj, eng->head.main_proj_f32);
                printf("[dspark] main_proj dequantized to f32: %.1f MB\n",
                       (float)(mp_size * 4) / (1024.0f * 1024.0f));
            }
        }

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
        if (eng->expert_mmap[l]) {
            munmap(eng->expert_mmap[l], eng->expert_mmap_size);
        }
        if (eng->expert_fd[l] >= 0) close(eng->expert_fd[l]);
    }
    free(eng);
}

// ============================================================================
// dspark_reset
// ============================================================================

void dspark_reset(DSparkEngine *eng) {
    for (int l = 0; l < DSPARK_N_LAYERS; l++) {
        eng->kv_cache[l].main_len = 0;
        eng->kv_cache[l].main_pos = 0;
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
// main_x computation + main_kv window writes (shared by dspark_forward and
// the prefill window builder)
// ============================================================================

// main_x = main_norm(main_proj(src_hidden [3*DIM]))
static void dspark_compute_main_x(DSparkEngine *eng, const float *src_hidden, float *main_x) {
    float proj_raw[DIM];
    if (eng->head.main_proj_f32) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    DIM, 3 * DIM, 1.0f, eng->head.main_proj_f32, 3 * DIM,
                    src_hidden, 1, 0.0f, proj_raw, 1);
    } else {
        fp8_matvec_cpu(eng->head.main_proj.weight, eng->head.main_proj.scale,
                       src_hidden, proj_raw,
                       DIM, 3 * DIM, eng->head.main_proj.block_size);
    }
    rms_norm_cpu(proj_raw, eng->head.main_norm, main_x, DIM);
}

// Per draft layer: main_kv = kv_norm(wkv(main_x)) with RoPE at absolute
// position pos, appended to the layer's ring window. The attention reads the
// window as a contiguous [0, main_len) array, so the write slot follows the
// append cursor (RoPE uses the absolute position — slot is storage only).
static void dspark_write_main_kv(DSparkEngine *eng, const float *main_x, int pos) {
    const int win = DSPARK_WINDOW_SIZE;
    for (int l = 0; l < DSPARK_N_LAYERS; l++) {
        DSparkLayerWeights *lw = &eng->layers[l];
        DSparkAttnWeights *aw = &lw->attn;
        DSparkKVCache *kvc = &eng->kv_cache[l];
        if (!aw->wkv_f32 || !aw->kv_norm) continue;
        const int slot = kvc->main_pos % win;
        kvc->main_pos++;
        float kv_n[KV_LORA_RANK];
        {
            float kv_raw[KV_LORA_RANK];
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        KV_LORA_RANK, DIM, 1.0f, aw->wkv_f32, DIM,
                        main_x, 1, 0.0f, kv_raw, 1);
            float ss = 0.0f;
            for (int i = 0; i < KV_LORA_RANK; i++) ss += kv_raw[i] * kv_raw[i];
            float rms = 1.0f / sqrtf(ss / (float)KV_LORA_RANK + 1e-6f);
            for (int i = 0; i < KV_LORA_RANK; i++) kv_n[i] = kv_raw[i] * rms * aw->kv_norm[i];
            dspark_rope_yarn(kv_n, pos, 0);
        }
        uint16_t *dst = kvc->main_kv + (size_t)slot * KV_LORA_RANK;
        for (int i = 0; i < KV_LORA_RANK; i++) {
            union { _Float16 f; uint16_t u; } cv; cv.f = (_Float16)kv_n[i]; dst[i] = cv.u;
        }
        if (kvc->main_len < win) kvc->main_len++;
    }
}

// Build the main_kv window for all prefill positions (reference:
// DSparkAttention start_pos==0 branch). hidden3: [3, n_tokens, DIM]
// mean-pooled target hidden from layers 40/41/42 for every prefill position.
void dspark_build_window(DSparkEngine *eng, const float *hidden3, int n_tokens, int start_pos) {
    if (!eng || !eng->initialized || !hidden3 || n_tokens <= 0) return;
    float *mh = (float *)malloc(3 * DIM * sizeof(float));
    float *main_x = (float *)malloc(DIM * sizeof(float));
    if (!mh || !main_x) { free(mh); free(main_x); return; }
    for (int p = 0; p < n_tokens; p++) {
        for (int s = 0; s < 3; s++)
            memcpy(mh + s * DIM, hidden3 + ((size_t)s * n_tokens + p) * DIM, DIM * sizeof(float));
        dspark_compute_main_x(eng, mh, main_x);
        dspark_write_main_kv(eng, main_x, start_pos + p);
    }
    free(mh);
    free(main_x);
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
// CPU MoE forward for DSpark INT8+E8M0 experts
// ============================================================================

// INT8 dequant matvec: out[r] = sum_c( int8_to_float(w[r,c]) * e8m0_scale(scale[r*cols+c)/16]) * x[c] )
// Layout: weight is row-major [out_dim, in_dim] int8; scale is [out_dim*in_dim/16] E8M0
// (one scale byte per 16 consecutive weight bytes in row-major order).
static void int8_e8m0_matvec(
    const uint8_t *weight, const uint8_t *scale,
    const float *x, float *out,
    int out_dim, int in_dim
) {
    const int bs = DSPARK_EXPERT_BLOCK_SIZE; // 16
    for (int r = 0; r < out_dim; r++) {
        float acc = 0.0f;
        const uint8_t *w_row = weight + (size_t)r * in_dim;
        const uint8_t *s_row = scale + (size_t)r * (in_dim / bs);
        for (int c = 0; c < in_dim; c += bs) {
            // E8M0 scale for this block of 16
            uint8_t sb = s_row[c / bs];
            uint32_t sf_bits = (uint32_t)sb << 23;
            float sf = *(float*)&sf_bits;  // 2^(sb-127)
            for (int j = 0; j < bs && (c + j) < in_dim; j++) {
                int8_t wval = (int8_t)w_row[c + j];
                acc += (float)wval * sf * x[c + j];
            }
        }
        out[r] = acc;
    }
}

// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
static void swiglu(const float *gate, const float *up, float *out, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

// CPU MoE forward for a single token position in DSpark.
// Performs: gate matmul → routing → pread experts → gate_up_swiglu → down → weighted combine.
// Falls back to CPU if mmap not available; otherwise uses GPU dispatch.
static void dspark_moe_forward_cpu(
    DSparkEngine *eng, int layer_idx,
    const float *normed_input,  // [DIM] — ffn_norm output
    float *moe_out              // [DIM] — output (weighted sum of K=6 experts)
) {
    DSparkLayerWeights *lw = &eng->layers[layer_idx];
    MoEInferEngine *target = (MoEInferEngine *)eng->target_engine;

    // 1. Gate matmul: scores[e] = dot(gate_weight[e], normed_input) for e in 0..N_EXPERTS
    float scores[N_EXPERTS];
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                N_EXPERTS, DIM, 1.0f, lw->gate_weight, DIM,
                normed_input, 1, 0.0f, scores, 1);

    // 2. CPU routing: sqrtsoftplus + bias + topK + L1-normalize
    int expert_ids[N_ACTIVE];
    float expert_weights[N_ACTIVE];
    {
        float processed[N_EXPERTS];
        for (int i = 0; i < N_EXPERTS; i++) {
            float l = scores[i] + (lw->gate_bias ? lw->gate_bias[i] : 0.0f);
            float sp = l > 0 ? l + log1pf(expf(-l)) : log1pf(expf(l));
            processed[i] = sqrtf(sp);
        }
        uint8_t taken[N_EXPERTS];
        memset(taken, 0, sizeof(taken));
        for (int k = 0; k < N_ACTIVE; k++) {
            int best = -1; float bv = -1e30f;
            for (int i = 0; i < N_EXPERTS; i++) {
                if (!taken[i] && processed[i] > bv) { bv = processed[i]; best = i; }
            }
            if (best < 0) { memset(moe_out, 0, DIM * sizeof(float)); goto moe_cleanup; }
            expert_ids[k] = best;
            expert_weights[k] = processed[best];
            taken[best] = 1;
        }
        float wsum = 0;
        for (int k = 0; k < N_ACTIVE; k++) wsum += expert_weights[k];
        wsum += 1e-20f;
        for (int k = 0; k < N_ACTIVE; k++) expert_weights[k] = expert_weights[k] / wsum * 1.5f;
    }

    // 3. GPU dispatch if mmap is available
    if (eng->expert_mmap[layer_idx]) {
        uint8_t *expert_ptrs[6];
        for (int k = 0; k < N_ACTIVE; k++) {
            int eid = expert_ids[k];
            expert_ptrs[k] = eng->expert_mmap[layer_idx] + (size_t)eid * DSPARK_EXPERT_SIZE;
        }
        dspark_moe_forward_gpu(target, normed_input, expert_ptrs, expert_ids, expert_weights, N_ACTIVE, moe_out);

        // Add shared expert output: GPU-accelerated SwiGLU
        if (lw->shared_w1_f32 && lw->shared_w3_f32 && lw->shared_w2_f32) {
            float down_se[DIM];
            dspark_shared_expert_gpu(target, normed_input,
                                     lw->shared_w1_f32, lw->shared_w3_f32, lw->shared_w2_f32,
                                     down_se);
            for (int d = 0; d < DIM; d++) moe_out[d] += down_se[d];
        }
        return;
    }

    // 4. CPU fallback (no mmap — should not happen in normal operation)
    memset(moe_out, 0, DIM * sizeof(float));
    fprintf(stderr, "[dspark] WARNING: CPU MoE fallback (no mmap). Output will be zero.\n");

moe_cleanup:
    (void)0;  // no heap allocations in GPU path
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
    const float *src_hidden = main_hidden ? main_hidden : eng->buf_main_hidden;

    // DSPARK_DEBUG: verify the main_hidden injection chain is alive — print
    // L2 norms of the accumulated target hidden (3 slots) and main_x after
    // projection. norm≈0 means the accumulate path never fed real hidden
    // states (implementation bug upstream of the backbone).
    static int dbg = -1;
    if (dbg < 0) dbg = getenv("DSPARK_DEBUG") ? 1 : 0;
    if (dbg) {
        double n0=0,n1=0,n2=0;
        for (int d = 0; d < DIM; d++) {
            n0 += (double)src_hidden[d]*src_hidden[d];
            n1 += (double)src_hidden[DIM+d]*src_hidden[DIM+d];
            n2 += (double)src_hidden[2*DIM+d]*src_hidden[2*DIM+d];
        }
        fprintf(stderr, "[dspark-dbg] pos=%d anchor=%d |main_hidden| slots=[%.3f %.3f %.3f]\n",
                start_pos, anchor_token_id, sqrt(n0), sqrt(n1), sqrt(n2));
    }

    // =====================================================================
    // Step 1: main_proj(concat(target_hidden[40,41,42])) → main_x [DIM]
    // =====================================================================
    float *main_x = eng->buf_main_x;
    dspark_compute_main_x(eng, src_hidden, main_x);

    // =====================================================================
    // Step 1b: main_kv entry for this position (per draft layer).
    // Reference (DSpark inference/model.py DSparkAttention.forward):
    //   main_kv = kv_norm(wkv(main_x)) with RoPE at the anchor position,
    //   written to the layer's window at start_pos % window_size.
    // The context channel: the backbone attends to target hidden states
    // (layers 40/41/42) projected through main_proj + ITS OWN wkv — NOT the
    // target layer-0 KV cache (which lives in a different vector space).
    // =====================================================================
    dspark_write_main_kv(eng, main_x, start_pos);

    // DSPARK_DUMP_DIR: dump intermediates for the incremental alignment
    // oracle (numpy recompute). First call only — files are overwritten per
    // process: main_hidden.bin [3*DIM f32], main_x.bin [DIM f32],
    // meta.txt (anchor, start_pos).
    const char *dump_dir = getenv("DSPARK_DUMP_DIR");
    static int dumped = 0;
    if (dump_dir && !dumped) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/main_hidden.bin", dump_dir);
        FILE *f = fopen(path, "wb"); if (f) { fwrite(src_hidden, sizeof(float), 3 * DIM, f); fclose(f); }
        snprintf(path, sizeof(path), "%s/main_x.bin", dump_dir);
        f = fopen(path, "wb"); if (f) { fwrite(main_x, sizeof(float), DIM, f); fclose(f); }
        snprintf(path, sizeof(path), "%s/meta.txt", dump_dir);
        f = fopen(path, "w"); if (f) { fprintf(f, "anchor=%d start_pos=%d\n", anchor_token_id, start_pos); fclose(f); }
    }

    // =====================================================================
    // Step 2: Initialize draft_hidden [DSPARK_BLOCK_SIZE, MHC_MULT, DIM]
    //   Position 0: embed(anchor_token)
    //   Position 1-4: embed(noise_token)
    //   Each position's residual is [MHC_MULT, DIM] with the EMBEDDING ONLY
    //   replicated — main_x must NOT be added here. Reference (DSpark
    //   inference/model.py forward_embed + Block.forward): residual = embed;
    //   main_x enters exclusively as attention context (main_kv), never into
    //   the residual stream. (Previously added main_x here — polluted the
    //   residual from step 0 and destroyed backbone accuracy.)
    // =====================================================================
    float *draft_hidden = eng->buf_draft_hidden; // [bs * MHC_MULT * DIM]
    for (int k = 0; k < bs; k++) {
        int tok = (k == 0) ? anchor_token_id : (int)eng->noise_token_id;
        const float *emb_row = target->embed + (size_t)tok * DIM;
        float *res_k = draft_hidden + (size_t)k * MHC_MULT * DIM;
        // Initialize all MHC_MULT streams with the embedding only
        for (int m = 0; m < MHC_MULT; m++) {
            float *dst = res_k + m * DIM;
            memcpy(dst, emb_row, DIM * sizeof(float));
        }
    }

    // =====================================================================
    // Step 3: 3-layer backbone loop
    //   For each layer: mhc_pre(attn) → attn_norm → attention → mhc_post(attn)
    //                   mhc_pre(ffn)  → ffn_norm  → MoE       → mhc_post(ffn)
    // =====================================================================

    // Scratch per-position
    float sublayer_input[DIM];
    float post_mix[MHC_MULT];
    float comb_mix[MHC_MULT * MHC_MULT];
    float normed[DIM];
    float sublayer_out[DIM];
    float new_residual[MHC_MULT * DIM];

    // Batch buffers for attention (all positions at once)
    float *attn_normed_batch = (float *)malloc((size_t)bs * DIM * sizeof(float));
    float *attn_out_batch = (float *)malloc((size_t)bs * DIM * sizeof(float));

    if (!attn_normed_batch || !attn_out_batch) {
        fprintf(stderr, "[dspark] forward scratch alloc failed\n");
        free(attn_normed_batch);
        free(attn_out_batch);
        return 0;
    }

    for (int layer = 0; layer < DSPARK_N_LAYERS; layer++) {
        DSparkLayerWeights *lw = &eng->layers[layer];
        MhcWeights hc_attn = { .fn = lw->hc_attn_fn, .base = lw->hc_attn_base, .scale = lw->hc_attn_scale };
        MhcWeights hc_ffn  = { .fn = lw->hc_ffn_fn,  .base = lw->hc_ffn_base,  .scale = lw->hc_ffn_scale  };

        // --- Attention sublayer ---
        // mhc_pre(attn) for each position → gather normed inputs for batch attention
        float post_mix_attn[DSPARK_BLOCK_SIZE][MHC_MULT];
        float comb_mix_attn[DSPARK_BLOCK_SIZE][MHC_MULT * MHC_MULT];

        for (int k = 0; k < bs; k++) {
            float *res_k = draft_hidden + (size_t)k * MHC_MULT * DIM;
            mhc_pre(&hc_attn, res_k, sublayer_input, post_mix_attn[k], comb_mix_attn[k]);
            rms_norm_cpu(sublayer_input, lw->attn_norm, attn_normed_batch + k * DIM, DIM);
        }

        // Batch attention forward (placeholder: identity passthrough)
        dspark_attention_forward(eng, layer, attn_normed_batch, main_x, attn_out_batch, start_pos);

        if (dump_dir && !dumped && layer == 0) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/attn_normed_l0.bin", dump_dir);
            FILE *f = fopen(path, "wb"); if (f) { fwrite(attn_normed_batch, sizeof(float), (size_t)bs * DIM, f); fclose(f); }
            snprintf(path, sizeof(path), "%s/attn_out_l0.bin", dump_dir);
            f = fopen(path, "wb"); if (f) { fwrite(attn_out_batch, sizeof(float), (size_t)bs * DIM, f); fclose(f); }
            snprintf(path, sizeof(path), "%s/draft_hidden_init.bin", dump_dir);
            f = fopen(path, "wb"); if (f) { fwrite(draft_hidden, sizeof(float), (size_t)bs * MHC_MULT * DIM, f); fclose(f); }
        }

        // mhc_post(attn) for each position
        for (int k = 0; k < bs; k++) {
            float *res_k = draft_hidden + (size_t)k * MHC_MULT * DIM;
            mhc_post(attn_out_batch + k * DIM, res_k, post_mix_attn[k], comb_mix_attn[k], new_residual);
            memcpy(res_k, new_residual, MHC_MULT * DIM * sizeof(float));
        }

        // --- FFN/MoE sublayer (batched: all positions in one GPU dispatch) ---
        // Step 1: CPU pre-processing for all positions (mhc_pre + ffn_norm + routing)
        float normed_all[DSPARK_BLOCK_SIZE * DIM];
        float post_mix_ffn[DSPARK_BLOCK_SIZE][MHC_MULT];
        float comb_mix_ffn[DSPARK_BLOCK_SIZE][MHC_MULT * MHC_MULT];
        uint8_t *batch_expert_ptrs[DSPARK_BLOCK_SIZE * N_ACTIVE];
        float batch_expert_weights[DSPARK_BLOCK_SIZE * N_ACTIVE];
        int batch_valid = 1;

        for (int k = 0; k < bs; k++) {
            float *res_k = draft_hidden + (size_t)k * MHC_MULT * DIM;
            mhc_pre(&hc_ffn, res_k, sublayer_input, post_mix_ffn[k], comb_mix_ffn[k]);
            rms_norm_cpu(sublayer_input, lw->ffn_norm, normed_all + k * DIM, DIM);

            // CPU routing
            float scores[N_EXPERTS];
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        N_EXPERTS, DIM, 1.0f, lw->gate_weight, DIM,
                        normed_all + k * DIM, 1, 0.0f, scores, 1);
            float processed[N_EXPERTS];
            for (int i = 0; i < N_EXPERTS; i++) {
                float l = scores[i] + (lw->gate_bias ? lw->gate_bias[i] : 0.0f);
                float sp = l > 0 ? l + log1pf(expf(-l)) : log1pf(expf(l));
                processed[i] = sqrtf(sp);
            }
            uint8_t taken[N_EXPERTS];
            memset(taken, 0, sizeof(taken));
            for (int e = 0; e < N_ACTIVE; e++) {
                int best = -1; float bv = -1e30f;
                for (int i = 0; i < N_EXPERTS; i++) {
                    if (!taken[i] && processed[i] > bv) { bv = processed[i]; best = i; }
                }
                if (best < 0) { batch_valid = 0; break; }
                batch_expert_ptrs[k * N_ACTIVE + e] = eng->expert_mmap[layer]
                    ? eng->expert_mmap[layer] + (size_t)best * DSPARK_EXPERT_SIZE : NULL;
                batch_expert_weights[k * N_ACTIVE + e] = processed[best];
                taken[best] = 1;
            }
            if (!batch_valid) break;
            // L1-normalize weights
            float wsum = 0;
            for (int e = 0; e < N_ACTIVE; e++) wsum += batch_expert_weights[k * N_ACTIVE + e];
            wsum += 1e-20f;
            for (int e = 0; e < N_ACTIVE; e++) batch_expert_weights[k * N_ACTIVE + e] = batch_expert_weights[k * N_ACTIVE + e] / wsum * 1.5f;
        }

        // Step 2: Single batched GPU dispatch for all positions
        float moe_out_all[DSPARK_BLOCK_SIZE * DIM];
        if (batch_valid && eng->expert_mmap[layer]) {
            dspark_moe_forward_batched_gpu(target, bs, normed_all,
                                           batch_expert_ptrs, batch_expert_weights, moe_out_all);
        } else {
            memset(moe_out_all, 0, sizeof(moe_out_all));
        }

        if (dump_dir && !dumped && layer == 0) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/moe_normed_l0.bin", dump_dir);
            FILE *f = fopen(path, "wb"); if (f) { fwrite(normed_all, sizeof(float), (size_t)bs * DIM, f); fclose(f); }
            snprintf(path, sizeof(path), "%s/moe_out_l0.bin", dump_dir);
            f = fopen(path, "wb"); if (f) { fwrite(moe_out_all, sizeof(float), (size_t)bs * DIM, f); fclose(f); }
            // expert routing for the oracle (ids inferred from ptr offsets)
            snprintf(path, sizeof(path), "%s/moe_routing_l0.bin", dump_dir);
            f = fopen(path, "wb");
            if (f) {
                for (int kk = 0; kk < bs * N_ACTIVE; kk++) {
                    int32_t id = -1;
                    if (batch_expert_ptrs[kk] && eng->expert_mmap[layer])
                        id = (int32_t)((batch_expert_ptrs[kk] - eng->expert_mmap[layer]) / DSPARK_EXPERT_SIZE);
                    fwrite(&id, 4, 1, f);
                }
                fwrite(batch_expert_weights, sizeof(float), (size_t)bs * N_ACTIVE, f);
                fclose(f);
            }
        }

        // Step 2b: Add shared expert for each position (CPU cblas — fast, avoids GPU sync)
        for (int k = 0; k < bs; k++) {
            if (lw->shared_w1_f32 && lw->shared_w3_f32 && lw->shared_w2_f32) {
                float gate_se[INTERMEDIATE], up_se[INTERMEDIATE], mid_se[INTERMEDIATE], se_out[DIM];
                cblas_sgemv(CblasRowMajor, CblasNoTrans, INTERMEDIATE, DIM, 1.0f,
                            lw->shared_w1_f32, DIM, normed_all + k * DIM, 1, 0.0f, gate_se, 1);
                cblas_sgemv(CblasRowMajor, CblasNoTrans, INTERMEDIATE, DIM, 1.0f,
                            lw->shared_w3_f32, DIM, normed_all + k * DIM, 1, 0.0f, up_se, 1);
                for (int i = 0; i < INTERMEDIATE; i++) {
                    float g = gate_se[i];
                    mid_se[i] = (g / (1.0f + expf(-g))) * up_se[i];
                }
                cblas_sgemv(CblasRowMajor, CblasNoTrans, DIM, INTERMEDIATE, 1.0f,
                            lw->shared_w2_f32, INTERMEDIATE, mid_se, 1, 0.0f, se_out, 1);
                for (int d = 0; d < DIM; d++) moe_out_all[k * DIM + d] += se_out[d];
            }
        }

        // Step 3: CPU post-processing (mhc_post)
        for (int k = 0; k < bs; k++) {
            float *res_k = draft_hidden + (size_t)k * MHC_MULT * DIM;
            mhc_post(moe_out_all + k * DIM, res_k, post_mix_ffn[k], comb_mix_ffn[k], new_residual);
            memcpy(res_k, new_residual, MHC_MULT * DIM * sizeof(float));
        }
    }

    free(attn_normed_batch);
    free(attn_out_batch);

    // =====================================================================
    // Step 4: hc_head compress → final_norm → lm_head → logits
    // =====================================================================
    for (int k = 0; k < bs; k++) {
        float *res_k = draft_hidden + (size_t)k * MHC_MULT * DIM;
        float compressed[DIM];

        // hc_head: compress [MHC_MULT, DIM] → [DIM]
        hyper_head_compress(eng->head.hc_head_fn, eng->head.hc_head_base,
                           eng->head.hc_head_scale, res_k, compressed);

        // final RMSNorm
        float final_normed[DIM];
        rms_norm_cpu(compressed, eng->head.final_norm, final_normed, DIM);

        // lm_head matmul → logits
        float *logits_k = draft_logits + (size_t)k * vocab;
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    vocab, DIM, 1.0f, target->lm_head, DIM,
                    final_normed, 1, 0.0f, logits_k, 1);

        if (dump_dir && !dumped && k == 0) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/final_residual.bin", dump_dir);
            FILE *f = fopen(path, "wb"); if (f) { fwrite(res_k, sizeof(float), (size_t)MHC_MULT * DIM, f); fclose(f); }
            snprintf(path, sizeof(path), "%s/final_compressed.bin", dump_dir);
            f = fopen(path, "wb"); if (f) { fwrite(compressed, sizeof(float), DIM, f); fclose(f); }
            snprintf(path, sizeof(path), "%s/final_normed.bin", dump_dir);
            f = fopen(path, "wb"); if (f) { fwrite(final_normed, sizeof(float), DIM, f); fclose(f); }
        }
    }

    // Confidence: placeholder 0.5 for now (real confidence head needs hidden + markov embed)
    if (confidence) {
        for (int k = 0; k < bs; k++) confidence[k] = 0.5f;
    }

    if (dump_dir && !dumped) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/draft_logits.bin", dump_dir);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(draft_logits, sizeof(float), (size_t)bs * vocab, f); fclose(f); }
        dumped = 1;
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
