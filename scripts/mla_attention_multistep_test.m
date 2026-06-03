// Multi-step MLA attention KV-cache correctness test.
// Loads /tmp/attn_ms_golden/* (gen_multistep_golden.py) and drives
// mla_attention_decode through N_PREFILL steps then one decode step,
// checking the final output against golden.bin.
//
// Build & run:
//   clang -framework Metal -framework Foundation -fobjc-arc \
//     -I src/metal_infer scripts/mla_attention_multistep_test.m \
//     src/metal_infer/mla_attention.m \
//     -o /tmp/mat_ms && /tmp/mat_ms
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import "mla_attention.h"

#define GD "/tmp/attn_ms_golden/"
#define GD_SINGLE "/tmp/attn_golden/"

static float *readf(const char *dir, const char *name, size_t *n_out) {
    char path[512]; snprintf(path, sizeof(path), "%s%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz); fread(buf, 1, sz, f); fclose(f);
    if (n_out) *n_out = sz / sizeof(float);
    return (float *)buf;
}
static uint32_t *readu(const char *dir, const char *name) {
    char path[512]; snprintf(path, sizeof(path), "%s%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz); fread(buf, 1, sz, f); fclose(f);
    return (uint32_t *)buf;
}

static id<MTLComputePipelineState> mkpipe(id<MTLDevice> d, id<MTLLibrary> lib, const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { printf("FAIL pipeline %s: %s\n", name, [[err localizedDescription] UTF8String]); exit(1); }
    return p;
}

static QuantWeight load_qw_dir(const char *dir, const char *base, int out_dim, int in_dim) {
    char nm[256]; QuantWeight q;
    q.out_dim = out_dim; q.in_dim = in_dim; q.group_size = ATTN_GROUP_SIZE;
    snprintf(nm, sizeof(nm), "%s.packed", base); q.packed = readu(dir, nm);
    snprintf(nm, sizeof(nm), "%s.sc", base); q.scales = readf(dir, nm, NULL);
    snprintf(nm, sizeof(nm), "%s.bi", base); q.biases = readf(dir, nm, NULL);
    return q;
}

int main(void) {
    // Read meta
    int N_PREFILL = 8, decode_pos = 8;
    FILE *mf = fopen(GD "meta.txt", "r");
    if (mf) { fscanf(mf, "N_PREFILL=%d\ndecode_pos=%d\n", &N_PREFILL, &decode_pos); fclose(mf); }
    printf("N_PREFILL=%d decode_pos=%d\n", N_PREFILL, decode_pos);

    // Metal setup
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [d newCommandQueue];
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@"src/models/moe_kernel.metal"
                                              encoding:NSUTF8StringEncoding error:&err];
    MTLCompileOptions *opts = [MTLCompileOptions new]; opts.languageVersion = MTLLanguageVersion3_1;
    id<MTLLibrary> lib = [d newLibraryWithSource:src options:opts error:&err];
    if (!lib) { printf("COMPILE FAIL: %s\n", [[err localizedDescription] UTF8String]); return 1; }

    MlaPipes P;
    P.dev = d; P.queue = q;
    P.dequant_matvec_affine = mkpipe(d, lib, "dequant_matvec_affine");
    P.rms_norm_rows = mkpipe(d, lib, "rms_norm_rows");
    P.rope_tail_interleaved = mkpipe(d, lib, "rope_tail_interleaved");
    P.mla_sdpa_decode = mkpipe(d, lib, "mla_sdpa_decode");
    P.matvec_f32 = mkpipe(d, lib, "matvec_f32");
    P.dequant_matvec_affine_bf16 = mkpipe(d, lib, "dequant_matvec_affine_bf16out");
    P.rms_norm_rows_bf16 = mkpipe(d, lib, "rms_norm_rows_bf16out");
    P.bf16_to_f32 = mkpipe(d, lib, "bf16_to_f32");

    // Load weights (from single-step golden dir, same layer-0 weights)
    AttnWeights aw;
    aw.wq_a = load_qw_dir(GD_SINGLE, "wq_a", Q_LORA_RANK, DIM);
    aw.wq_b = load_qw_dir(GD_SINGLE, "wq_b", N_HEADS * HEAD_DIM, Q_LORA_RANK);
    aw.wkv  = load_qw_dir(GD_SINGLE, "wkv", KV_LORA_RANK, DIM);
    aw.wo_b = load_qw_dir(GD_SINGLE, "wo_b", DIM, O_GROUPS * O_LORA_RANK);
    aw.q_norm = readf(GD_SINGLE, "q_norm.bin", NULL);
    aw.kv_norm = readf(GD_SINGLE, "kv_norm.bin", NULL);
    aw.attn_sink = readf(GD_SINGLE, "attn_sink.bin", NULL);
    aw.wo_a_dense = readf(GD_SINGLE, "wo_a_dense.bin", NULL);

    // KV cache (engine-style: [MAX_SEQ_LEN, KV_LORA_RANK])
    float *kv_cache = calloc((size_t)MAX_SEQ_LEN * KV_LORA_RANK, sizeof(float));
    float *out = malloc(DIM * sizeof(float));

    // Prefill: N_PREFILL steps
    for (int i = 0; i < N_PREFILL; i++) {
        char nm[64]; snprintf(nm, sizeof(nm), "hidden_%02d.bin", i);
        float *hidden = readf(GD, nm, NULL);
        // cache_len = i+1 (current token is i, written at row i)
        mla_attention_decode(&P, &aw, hidden, kv_cache, i + 1, i, out);
        free(hidden);
    }

    // Decode step at decode_pos
    char nm[64]; snprintf(nm, sizeof(nm), "hidden_%02d.bin", decode_pos);
    float *hidden_decode = readf(GD, nm, NULL);
    mla_attention_decode(&P, &aw, hidden_decode, kv_cache, decode_pos + 1, decode_pos, out);
    free(hidden_decode);

    // Compare to golden
    size_t gn; float *golden = readf(GD, "golden.bin", &gn);
    float maxd = 0; double ss = 0, gss = 0;
    for (int i = 0; i < DIM; i++) {
        float dd = fabsf(out[i] - golden[i]);
        if (dd > maxd) maxd = dd;
        ss += (double)dd * dd;
        gss += (double)golden[i] * golden[i];
    }
    float rel = (float)(sqrt(ss) / sqrt(gss));
    printf("out[:4]   = [%.4f %.4f %.4f %.4f]\n", out[0], out[1], out[2], out[3]);
    printf("golden[:4]= [%.4f %.4f %.4f %.4f]\n", golden[0], golden[1], golden[2], golden[3]);
    printf("max_abs_diff=%.3e  rel_L2=%.3e\n", maxd, rel);

    // Also compare KV cache at each position vs golden KV cache
    float *ref_kv = readf(GD, "kv_cache.bin", NULL);
    float kv_maxd = 0; double kv_ss = 0, kv_ref_ss = 0;
    for (int i = 0; i <= decode_pos; i++) {
        for (int j = 0; j < KV_LORA_RANK; j++) {
            float ref = ref_kv[(size_t)i * KV_LORA_RANK + j];
            float cmp = kv_cache[(size_t)i * KV_LORA_RANK + j];
            float dd = fabsf(ref - cmp);
            if (dd > kv_maxd) kv_maxd = dd;
            kv_ss += (double)dd * dd;
            kv_ref_ss += (double)ref * ref;
        }
    }
    float kv_rel = (float)(sqrt(kv_ss) / sqrt(kv_ref_ss));
    printf("KV cache comparison (all %d rows): max_abs=%.3e  rel_L2=%.3e\n",
           decode_pos + 1, kv_maxd, kv_rel);

    int ok = rel < 1e-3 && kv_rel < 1e-3;
    printf("%s\n", ok ? "RESULT: GO — multi-step KV cache correct"
                      : "RESULT: NO-GO — multi-step mismatch");
    return ok ? 0 : 1;
}
