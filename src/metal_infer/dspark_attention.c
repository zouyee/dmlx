// DSparkAttention — Fast CPU MLA using pre-dequantized f32 weights + cblas_sgemv.
//
// Architecture (DeepSeek-V4 MLA):
//   Q: wq_a[1024,4096] → q_norm[1024] → wq_b[32768,1024] → [64 heads × 512]
//   KV: wkv[512,4096] → kv_norm[512] → latent KV [512]
//   RoPE on last 64 dims of both Q and KV
//   SDPA: score[h][t] = dot(q[h], kv[t]) / sqrt(512)
//   O-proj: grouped wo_a[8192,4096] → wo_b[4096,8192]
//
#include "dspark_engine.h"
#include "engine.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>

// RMSNorm
static void attn_rms_norm(const float *x, const float *w, float *out, int dim) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / (float)dim + 1e-6f);
    for (int i = 0; i < dim; i++) out[i] = x[i] * rms * w[i];
}

// RoPE (partial YaRN on last 64 dims of HEAD_DIM=512 vector)
#define ROPE_DIM_A 64
#define NOPE_DIM_A 448
#define ROPE_THETA_A 10000000.0f

static void attn_apply_rope(float *vec, int pos) {
    float *rp = vec + NOPE_DIM_A;
    int nh = ROPE_DIM_A / 2;
    for (int i = 0; i < nh; i++) {
        float theta = (float)pos * powf(ROPE_THETA_A, -2.0f * i / ROPE_DIM_A);
        float c = cosf(theta), s = sinf(theta);
        float x0 = rp[i], x1 = rp[i + nh];
        rp[i]      = x0 * c - x1 * s;
        rp[i + nh] = x0 * s + x1 * c;
    }
}

static void attn_inverse_rope(float *vec, int pos) {
    float *rp = vec + NOPE_DIM_A;
    int nh = ROPE_DIM_A / 2;
    for (int i = 0; i < nh; i++) {
        float theta = (float)pos * powf(ROPE_THETA_A, -2.0f * i / ROPE_DIM_A);
        float c = cosf(theta), s = sinf(theta);
        float x0 = rp[i], x1 = rp[i + nh];
        rp[i]      =  x0 * c + x1 * s;
        rp[i + nh] = -x0 * s + x1 * c;
    }
}

void dspark_attention_forward(
    DSparkEngine *eng, int layer_idx,
    const float *normed_input,   // [DSPARK_BLOCK_SIZE, DIM]
    const float *main_x,         // [DIM]
    float *attn_out,             // [DSPARK_BLOCK_SIZE, DIM]
    int start_pos
) {
    DSparkLayerWeights *lw = &eng->layers[layer_idx];
    DSparkAttnWeights *aw = &lw->attn;
    DSparkKVCache *kvc = &eng->kv_cache[layer_idx];

    // Fall back to identity if f32 weights not available
    if (!aw->wq_a_f32 || !aw->wkv_f32 || !aw->wo_b_f32) {
        memcpy(attn_out, normed_input, DSPARK_BLOCK_SIZE * DIM * sizeof(float));
        return;
    }

    const int bs = DSPARK_BLOCK_SIZE;
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    const int heads_per_group = N_HEADS / O_GROUPS; // 8

    // Scratch buffers (stack for small, heap for large)
    float q_lora[Q_LORA_RANK];
    float q_normed_buf[Q_LORA_RANK];
    float kv_raw[KV_LORA_RANK];
    float kv_normed_buf[KV_LORA_RANK];

    // Heap for large buffers
    float *q_full = (float *)malloc((size_t)N_HEADS * HEAD_DIM * sizeof(float));
    float *draft_kvs = (float *)calloc((size_t)bs * KV_LORA_RANK, sizeof(float));
    float *o_concat = (float *)malloc((size_t)N_HEADS * HEAD_DIM * sizeof(float));
    float *o_grouped = (float *)malloc((size_t)O_GROUPS * O_LORA_RANK * sizeof(float));

    if (!q_full || !draft_kvs || !o_concat || !o_grouped) {
        memcpy(attn_out, normed_input, bs * DIM * sizeof(float));
        goto cleanup;
    }

    // Step 1: Compute draft KV for all positions
    for (int k = 0; k < bs; k++) {
        const float *x_k = normed_input + k * DIM;
        float *kv_k = draft_kvs + k * KV_LORA_RANK;

        // wkv: [KV_LORA_RANK, DIM] @ x → [KV_LORA_RANK]
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    KV_LORA_RANK, DIM, 1.0f, aw->wkv_f32, DIM,
                    x_k, 1, 0.0f, kv_raw, 1);
        attn_rms_norm(kv_raw, aw->kv_norm, kv_k, KV_LORA_RANK);
        attn_apply_rope(kv_k, start_pos + k);
    }

    // Step 2: Convert main_kv from f16 to f32
    int main_len = kvc->main_len;
    float *main_kvs_f32 = NULL;
    if (main_len > 0) {
        main_kvs_f32 = (float *)malloc((size_t)main_len * KV_LORA_RANK * sizeof(float));
        if (main_kvs_f32) {
            for (int t = 0; t < main_len * KV_LORA_RANK; t++) {
                uint16_t h = kvc->main_kv[t];
                union { uint16_t u; _Float16 f; } conv;
                conv.u = h;
                main_kvs_f32[t] = (float)conv.f;
            }
        }
    }

    // Step 3: For each draft position — Q, SDPA, O-proj
    for (int k = 0; k < bs; k++) {
        const float *x_k = normed_input + k * DIM;

        // Q chain: wq_a → q_norm → wq_b
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    Q_LORA_RANK, DIM, 1.0f, aw->wq_a_f32, DIM,
                    x_k, 1, 0.0f, q_lora, 1);
        attn_rms_norm(q_lora, aw->q_norm, q_normed_buf, Q_LORA_RANK);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    N_HEADS * HEAD_DIM, Q_LORA_RANK, 1.0f, aw->wq_b_f32, Q_LORA_RANK,
                    q_normed_buf, 1, 0.0f, q_full, 1);

        // Per-head RoPE
        for (int h = 0; h < N_HEADS; h++) {
            attn_apply_rope(q_full + h * HEAD_DIM, start_pos + k);
        }

        // SDPA per head
        int total_kv = main_len + k + 1;
        float *scores = (float *)alloca(total_kv * sizeof(float));

        for (int h = 0; h < N_HEADS; h++) {
            float *q_h = q_full + h * HEAD_DIM;
            float *o_h = o_concat + h * HEAD_DIM;

            // Compute scores
            float max_s = -1e30f;

            // Against main_kv
            for (int t = 0; t < main_len; t++) {
                float dot = 0.0f;
                float *kv_t = main_kvs_f32 + t * KV_LORA_RANK;
                // cblas_sdot for HEAD_DIM=512
                dot = cblas_sdot(HEAD_DIM, q_h, 1, kv_t, 1);
                scores[t] = dot * scale;
                if (scores[t] > max_s) max_s = scores[t];
            }
            // Against draft_kv (causal)
            for (int t = 0; t <= k; t++) {
                float *kv_t = draft_kvs + t * KV_LORA_RANK;
                float dot = cblas_sdot(HEAD_DIM, q_h, 1, kv_t, 1);
                scores[main_len + t] = dot * scale;
                if (scores[main_len + t] > max_s) max_s = scores[main_len + t];
            }

            // Softmax
            float sum_exp = 0.0f;
            for (int t = 0; t < total_kv; t++) {
                scores[t] = expf(scores[t] - max_s);
                sum_exp += scores[t];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-20f);
            for (int t = 0; t < total_kv; t++) scores[t] *= inv_sum;

            // Weighted sum of values (in MLA: value == KV latent)
            memset(o_h, 0, HEAD_DIM * sizeof(float));
            for (int t = 0; t < main_len; t++) {
                cblas_saxpy(HEAD_DIM, scores[t], main_kvs_f32 + t * KV_LORA_RANK, 1, o_h, 1);
            }
            for (int t = 0; t <= k; t++) {
                cblas_saxpy(HEAD_DIM, scores[main_len + t], draft_kvs + t * KV_LORA_RANK, 1, o_h, 1);
            }

            // Inverse RoPE
            attn_inverse_rope(o_h, start_pos + k);
        }

        // O-proj: grouped wo_a → wo_b
        // wo_a: [O_GROUPS*O_LORA_RANK=8192, group_in_dim=4096]
        // Each group g processes heads [g*8..(g+1)*8] concatenated → [8*512=4096]
        int group_in = heads_per_group * HEAD_DIM;  // 4096
        for (int g = 0; g < O_GROUPS; g++) {
            float *group_input = o_concat + g * group_in;
            float *group_out = o_grouped + g * O_LORA_RANK;
            // wo_a rows for this group: [g*O_LORA_RANK .. (g+1)*O_LORA_RANK]
            const float *wa_g = aw->wo_a_f32 + (size_t)g * O_LORA_RANK * aw->wo_a.in_dim;
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        O_LORA_RANK, aw->wo_a.in_dim, 1.0f, wa_g, aw->wo_a.in_dim,
                        group_input, 1, 0.0f, group_out, 1);
        }

        // wo_b: [DIM, O_GROUPS*O_LORA_RANK=8192]
        float *out_k = attn_out + k * DIM;
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    DIM, O_GROUPS * O_LORA_RANK, 1.0f, aw->wo_b_f32, O_GROUPS * O_LORA_RANK,
                    o_grouped, 1, 0.0f, out_k, 1);
    }

cleanup:
    free(q_full);
    free(draft_kvs);
    free(o_concat);
    free(o_grouped);
    free(main_kvs_f32);
}
