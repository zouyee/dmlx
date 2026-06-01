// Standalone test for mhc.c (CPU mHC) against the Python golden.
// Build & run:
//   clang -I src/metal_infer scripts/mhc_test.c src/metal_infer/mhc.c -o /tmp/mhct -lm && /tmp/mhct
#include "mhc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GD "/tmp/mhc_golden/"

static float *rd(const char *name) {
    char p[512]; snprintf(p, sizeof(p), "%s%s", GD, name);
    FILE *f = fopen(p, "rb");
    if (!f) { printf("FAIL open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    float *b = malloc(sz); fread(b, 1, sz, f); fclose(f); return b;
}
static float maxdiff(const float *a, const float *b, int n) {
    float m = 0; for (int i = 0; i < n; i++) { float d = fabsf(a[i]-b[i]); if (d>m) m=d; } return m;
}

int main(void) {
    MhcWeights w; w.fn = rd("fn.bin"); w.base = rd("base.bin"); w.scale = rd("scale.bin");
    float *residual = rd("residual.bin");  // [HC,DIM]
    float *x = rd("x.bin");
    float *g_sub = rd("sub_in.bin");
    float *g_post = rd("post.bin");
    float *g_comb = rd("comb.bin");
    float *g_out = rd("out_res.bin");

    float sub[DIM], post[MHC_MULT], comb[MHC_MULT*MHC_MULT];
    mhc_pre(&w, residual, sub, post, comb);
    float *out = malloc((size_t)MHC_MULT * DIM * sizeof(float));
    mhc_post(x, residual, post, comb, out);

    int fail = 0;
    float d;
    d = maxdiff(sub, g_sub, DIM);            printf("  mhc_pre sub_input  max_abs=%.3e %s\n", d, d<1e-4?"OK":"FAIL"); if(d>=1e-4)fail=1;
    d = maxdiff(post, g_post, MHC_MULT);     printf("  mhc_pre post_mix   max_abs=%.3e %s\n", d, d<1e-4?"OK":"FAIL"); if(d>=1e-4)fail=1;
    d = maxdiff(comb, g_comb, MHC_MULT*MHC_MULT); printf("  mhc_pre comb      max_abs=%.3e %s\n", d, d<1e-4?"OK":"FAIL"); if(d>=1e-4)fail=1;
    d = maxdiff(out, g_out, MHC_MULT*DIM);   printf("  mhc_post out_res   max_abs=%.3e %s\n", d, d<1e-4?"OK":"FAIL"); if(d>=1e-4)fail=1;

    printf("%s\n", fail ? "RESULT: MHC TEST FAILED" : "RESULT: MHC GO");
    return fail;
}

#define MHC_H_UNUSED 0
