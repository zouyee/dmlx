// Standalone test for mla_attention_decode against the Python golden.
// Loads /tmp/attn_golden/* (gen_attn_golden.py) and checks the host MLA
// attention output matches golden.bin.
//
// Build & run:
//   clang -framework Metal -framework Foundation -fobjc-arc \
//     -I src/metal_infer scripts/mla_attention_test.m src/metal_infer/mla_attention.m \
//     -o /tmp/mat && /tmp/mat
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import <stdlib.h>
#import "mla_attention.h"

#define GD "/tmp/attn_golden/"

static float *readf(const char *name, size_t *n_out) {
    char path[512]; snprintf(path, sizeof(path), "%s%s", GD, name);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz); fread(buf, 1, sz, f); fclose(f);
    if (n_out) *n_out = sz / sizeof(float);
    return (float *)buf;
}
static uint32_t *readu(const char *name) {
    char path[512]; snprintf(path, sizeof(path), "%s%s", GD, name);
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

static QuantWeight load_qw(const char *base, int out_dim, int in_dim) {
    char nm[256];
    QuantWeight q; q.out_dim = out_dim; q.in_dim = in_dim; q.group_size = ATTN_GROUP_SIZE;
    snprintf(nm, sizeof(nm), "%s.packed", base); q.packed = readu(nm);
    snprintf(nm, sizeof(nm), "%s.sc", base); q.scales = readf(nm, NULL);
    snprintf(nm, sizeof(nm), "%s.bi", base); q.biases = readf(nm, NULL);
    return q;
}

int main(void) {
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

    AttnWeights aw;
    aw.wq_a = load_qw("wq_a", Q_LORA_RANK, DIM);            // [1024,4096]
    aw.wq_b = load_qw("wq_b", N_HEADS * HEAD_DIM, Q_LORA_RANK); // [32768,1024]
    aw.wkv  = load_qw("wkv", KV_LORA_RANK, DIM);            // [512,4096]
    aw.wo_b = load_qw("wo_b", DIM, O_GROUPS * O_LORA_RANK); // [4096,8192]
    aw.q_norm = readf("q_norm.bin", NULL);
    aw.kv_norm = readf("kv_norm.bin", NULL);
    aw.attn_sink = readf("attn_sink.bin", NULL);

    // wo_a is grouped: packed file is [8192,512] = [8*1024, 512]; slice per group.
    uint32_t *woa_packed = readu("wo_a.packed");   // [8192, 512]
    float *woa_sc = readf("wo_a.sc", NULL);        // [8192, 64]
    float *woa_bi = readf("wo_a.bi", NULL);        // [8192, 64]
    int pcols = DIM / 8;        // 512
    int ng = DIM / ATTN_GROUP_SIZE; // 64
    for (int g = 0; g < O_GROUPS; g++) {
        aw.wo_a[g].out_dim = O_LORA_RANK; aw.wo_a[g].in_dim = DIM; aw.wo_a[g].group_size = ATTN_GROUP_SIZE;
        aw.wo_a[g].packed = woa_packed + (size_t)g * O_LORA_RANK * pcols;
        aw.wo_a[g].scales = woa_sc + (size_t)g * O_LORA_RANK * ng;
        aw.wo_a[g].biases = woa_bi + (size_t)g * O_LORA_RANK * ng;
    }

    float *x = readf("hidden.bin", NULL);
    size_t gn; float *golden = readf("golden.bin", &gn);

    // pos from meta.txt
    int pos = 7;
    FILE *mf = fopen(GD "meta.txt", "r"); if (mf) { fscanf(mf, "pos=%d", &pos); fclose(mf); }

    float *kv_cache = calloc((size_t)MAX_SEQ_LEN * KV_LORA_RANK, sizeof(float));
    float *out = malloc(DIM * sizeof(float));
    // single token at position pos: cache_len = 1 (only current token)
    mla_attention_decode(&P, &aw, x, kv_cache, 1, pos, out);

    float maxd = 0; double ss = 0, gss = 0;
    for (int i = 0; i < DIM; i++) { float dd = fabsf(out[i] - golden[i]); if (dd > maxd) maxd = dd; ss += dd*dd; gss += (double)golden[i]*golden[i]; }
    float rel = (float)(sqrt(ss) / sqrt(gss));
    printf("out[:4]   = [%.4f %.4f %.4f %.4f]\n", out[0], out[1], out[2], out[3]);
    printf("golden[:4]= [%.4f %.4f %.4f %.4f]\n", golden[0], golden[1], golden[2], golden[3]);
    printf("max_abs_diff=%.3e  rel_L2=%.3e\n", maxd, rel);
    int ok = rel < 1e-3;
    printf("%s\n", ok ? "RESULT: GO — mla_attention_decode matches golden" : "RESULT: NO-GO");
    return ok ? 0 : 1;
}
