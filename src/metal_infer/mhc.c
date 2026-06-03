// mHC CPU implementation — see mhc.h. Mirrors deepseek_v4.zig mhcPreNormFn /
// mhcPreSplitMixes / sinkhornNormalize / mhcPost (all validated in verify_mhc.py).
#include "mhc.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define HC      MHC_MULT          // 4
#define MIX3    (HC * (HC + 2))   // 24
#define MHC_H   (HC * DIM)        // 16384
#define HC_EPS  1e-6f
#define SINKHORN_ITERS 20
#define POST_MULT 2.0f

// mixes[MIX3] = preNormFn(residual): mix = (fn @ res_flat) * rsqrt(mean(res^2)+eps)
// residual is [HC, DIM] interpreted as res_flat[MHC_H].
static void pre_norm_fn(const MhcWeights *w, const float *residual, float *mixes) {
    const float *res = residual; // [MHC_H] contiguous
    double ss = 0.0;
    for (int i = 0; i < MHC_H; i++) ss += (double)res[i] * res[i];
    float norm = 1.0f / sqrtf((float)(ss / MHC_H) + HC_EPS);
    for (int r = 0; r < MIX3; r++) {
        const float *fr = w->fn + (size_t)r * MHC_H;
        double acc = 0.0;
        for (int i = 0; i < MHC_H; i++) acc += (double)fr[i] * res[i];
        mixes[r] = (float)acc * norm;
    }
}

// sinkhorn on comb[HC*HC] (row-major [i][j]): softmax over j, +eps,
// col-norm, then (iters-1) x (row-norm, col-norm).
static void sinkhorn(float *comb) {
    // softmax over last dim (j)
    for (int i = 0; i < HC; i++) {
        float *row = comb + i * HC;
        float m = row[0];
        for (int j = 1; j < HC; j++) if (row[j] > m) m = row[j];
        float s = 0;
        for (int j = 0; j < HC; j++) { row[j] = expf(row[j] - m); s += row[j]; }
        for (int j = 0; j < HC; j++) row[j] = row[j] / s + HC_EPS;
    }
    // initial col-norm (sum over i)
    for (int j = 0; j < HC; j++) {
        float cs = 0; for (int i = 0; i < HC; i++) cs += comb[i*HC+j];
        cs += HC_EPS;
        for (int i = 0; i < HC; i++) comb[i*HC+j] /= cs;
    }
    for (int it = 0; it < SINKHORN_ITERS - 1; it++) {
        for (int i = 0; i < HC; i++) {                 // row-norm
            float rs = 0; for (int j = 0; j < HC; j++) rs += comb[i*HC+j];
            rs += HC_EPS;
            for (int j = 0; j < HC; j++) comb[i*HC+j] /= rs;
        }
        for (int j = 0; j < HC; j++) {                 // col-norm
            float cs = 0; for (int i = 0; i < HC; i++) cs += comb[i*HC+j];
            cs += HC_EPS;
            for (int i = 0; i < HC; i++) comb[i*HC+j] /= cs;
        }
    }
}

void mhc_pre(const MhcWeights *w, const float *residual,
             float *out_input, float *out_post, float *out_comb) {
    float mixes[MIX3];
    pre_norm_fn(w, residual, mixes);
    // scale_expanded: [HC * scale[0], HC * scale[1], HC*HC * scale[2]] then +base
    float pre_mix[HC], comb[HC*HC];
    for (int m = 0; m < HC; m++) {
        float biased = mixes[m] * w->scale[0] + w->base[m];
        pre_mix[m] = 1.0f / (1.0f + expf(-biased)) + HC_EPS;   // sigmoid + pre_eps
    }
    for (int m = 0; m < HC; m++) {
        float biased = mixes[HC + m] * w->scale[1] + w->base[HC + m];
        out_post[m] = (1.0f / (1.0f + expf(-biased))) * POST_MULT;
    }
    for (int c = 0; c < HC*HC; c++) {
        comb[c] = mixes[2*HC + c] * w->scale[2] + w->base[2*HC + c];
    }
    sinkhorn(comb);
    memcpy(out_comb, comb, sizeof(comb));

    // sublayer input = sum_m pre_mix[m] * residual[m, :]
    for (int d = 0; d < DIM; d++) {
        float acc = 0;
        for (int m = 0; m < HC; m++) acc += pre_mix[m] * residual[m*DIM + d];
        out_input[d] = acc;
    }
}

// mhc_pre_with_premix: same as mhc_pre but also returns the pre_mix weights.
// Used by engine.c to perform the final blend step on GPU (bf16 precision).
void mhc_pre_with_premix(const MhcWeights *w, const float *residual,
                          float *out_input, float *out_post, float *out_comb,
                          float *out_premix) {
    float mixes[MIX3];
    pre_norm_fn(w, residual, mixes);
    float pre_mix[HC], comb[HC*HC];
    for (int m = 0; m < HC; m++) {
        float biased = mixes[m] * w->scale[0] + w->base[m];
        pre_mix[m] = 1.0f / (1.0f + expf(-biased)) + HC_EPS;
    }
    for (int m = 0; m < HC; m++) {
        float biased = mixes[HC + m] * w->scale[1] + w->base[HC + m];
        out_post[m] = (1.0f / (1.0f + expf(-biased))) * POST_MULT;
    }
    for (int c = 0; c < HC*HC; c++) {
        comb[c] = mixes[2*HC + c] * w->scale[2] + w->base[2*HC + c];
    }
    sinkhorn(comb);
    memcpy(out_comb, comb, sizeof(comb));
    memcpy(out_premix, pre_mix, HC * sizeof(float));

    // out_input (f32 version) — will be overridden by GPU bf16 blend in engine.c
    for (int d = 0; d < DIM; d++) {
        float acc = 0;
        for (int m = 0; m < HC; m++) acc += pre_mix[m] * residual[m*DIM + d];
        out_input[d] = acc;
    }
}

void mhc_post(const float *x, const float *residual,
              const float *post, const float *comb, float *out_residual) {
    // Use a temp buffer to avoid aliasing when out_residual == residual.
    static float tmp[MHC_MULT * DIM];
    for (int m = 0; m < HC; m++) {
        float *om = tmp + m*DIM;
        for (int d = 0; d < DIM; d++) om[d] = post[m] * x[d];
        for (int k = 0; k < HC; k++) {
            float ckm = comb[k*HC + m];
            const float *rk = residual + k*DIM;
            for (int d = 0; d < DIM; d++) om[d] += ckm * rk[d];
        }
    }
    memcpy(out_residual, tmp, (size_t)HC * DIM * sizeof(float));
}

void mhc_head_compress(const MhcWeights *w, const float *residual, float *out) {
    // mixes via preNormFn (fn rows beyond available are zero in practice);
    // compression mix = sigmoid(mix_slice * scale[0] + base[0..HC]) + eps,
    // out = sum_m mix[m] * residual[m,:]
    float mixes[MIX3];
    pre_norm_fn(w, residual, mixes);
    float mix[HC];
    for (int m = 0; m < HC; m++) {
        float biased = mixes[m] * w->scale[0] + w->base[m];
        mix[m] = 1.0f / (1.0f + expf(-biased)) + HC_EPS;
    }
    for (int d = 0; d < DIM; d++) {
        float acc = 0;
        for (int m = 0; m < HC; m++) acc += mix[m] * residual[m*DIM + d];
        out[d] = acc;
    }
}
