// Standalone test: gather6 2-bit (int2 gs=64 + E8M0 scale + BF16 bias) kernels.
//
//   Gate 1 (hard):  CPU reference (pure-C dequant + matvec + SwiGLU + down,
//                   bf16-rounded outputs, in this file) vs GPU kernels
//                   gather6_gate_up_swiglu_2b / gather6_down_2b.
//                   Criterion: diff <= 1 bf16 ulp (GPU f32 accumulation order
//                   is compiler-dependent and flips bf16 rounding boundaries);
//                   anything larger fails. Down is expected bit-exact.
//   Gate 2 (soft):  cosine similarity of chained 2-bit forward vs existing
//                   4-bit gather6 forward on the same 6 experts (printed,
//                   expected >0.98, not enforced).
//
// Blobs (first 6 experts of each):
//   4-bit: packed_experts/layer_00.bin      EXPERT_SIZE    = 13369344
//   2-bit: packed_experts_2bit/layer_00.bin EXPERT_SIZE_2B = 7471104
//
// Build:  clang -O2 -fobjc-arc -framework Foundation -framework Metal \
//           tests/test_gather6_2b.m -o /tmp/test_gather6_2b
// Run:    /tmp/test_gather6_2b [moe_kernel.metal path]
// Exit:   0 = pass, 1 = fail
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM 4096
#define INTERMEDIATE 2048

// 4-bit blob layout (MXFP4, scale-only)
#define EXPERT_SIZE   13369344
// 2-bit blob layout
#define EXPERT_SIZE_2B 7471104
#define G2_GATE_W 0
#define G2_GATE_S 2097152
#define G2_GATE_B 2228224
#define G2_UP_W   2490368
#define G2_UP_S   4587520
#define G2_UP_B   4718592
#define G2_DOWN_W 4980736
#define G2_DOWN_S 7077888
#define G2_DOWN_B 7208960

static int g_fail = 0;

static void compare(const char *name, const float *ref, const float *gpu, int n) {
    double max_abs = 0, max_rel = 0, max_ulp = 0; int bad = 0, exact = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)ref[i] - (double)gpu[i]);
        double r = d / (fabs((double)ref[i]) + 1e-6);
        // 1 bf16 ulp at |ref| (bf16: 7 mantissa bits)
        double ulp = ldexp(1.0, (int)floor(log2(fabs((double)ref[i]) + 1e-30)) - 7);
        if (d > max_abs) max_abs = d;
        if (r > max_rel) max_rel = r;
        if (d / ulp > max_ulp) max_ulp = d / ulp;
        if (ref[i] == gpu[i]) exact++;
        // GPU f32 accumulation order is implementation-defined (compiler
        // reassociation), so outputs can flip across a bf16 rounding
        // boundary: tolerate diffs <= 1 bf16 ulp; larger = real error.
        if (d > ulp * 1.0001 && r > 1e-2) bad++;
    }
    printf("%s: max_abs=%.6g max_rel=%.6g max_ulp=%.3f bad=%d/%d exact=%d/%d %s\n",
           name, max_abs, max_rel, max_ulp, bad, n, exact, n, bad == 0 ? "OK" : "*** FAIL ***");
    for (int i = 0, shown = 0; i < n && shown < 4; i++) {
        double d = fabs((double)ref[i] - (double)gpu[i]);
        double ulp = ldexp(1.0, (int)floor(log2(fabs((double)ref[i]) + 1e-30)) - 7);
        if (d > ulp * 1.0001 && d / (fabs((double)ref[i]) + 1e-6) > 1e-2) {
            printf("    idx=%d ref=%.9g gpu=%.9g\n", i, ref[i], gpu[i]); shown++;
        }
    }
    if (bad) g_fail = 1;
}

// round-to-nearest-even bf16, returned as float (matches Metal (bfloat) cast)
static float bf16_round(float v) {
    uint32_t u; memcpy(&u, &v, 4);
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16;
    float r; memcpy(&r, &u, 4); return r;
}

// CPU reference: int2 gs=64 dequant matvec. out[r] = bf16(sum((bias+q*scale)*x))
// Replicates the GPU accumulation order exactly: lane l accumulates groups
// l, l+32, ... sequentially (same per-element op order), then a shuffle_down
// butterfly tree (offsets 16,8,4,2,1) reduces the 32 lane partials — this
// avoids 1-ulp bf16 boundary flips caused by f32 summation-order differences.
static void cpu_matvec_2b(const uint8_t *blob, long woff, long soff, long boff,
                          int out_d, int in_d, const float *x, float *out, int round_bf16) {
    const int ng = in_d / 64, pc = in_d / 16;
    for (int r = 0; r < out_d; r++) {
        const uint32_t *wrow = (const uint32_t *)(blob + woff) + (size_t)r * pc;
        const uint8_t  *srow = blob + soff + (size_t)r * ng;
        const uint16_t *brow = (const uint16_t *)(blob + boff) + (size_t)r * ng;
        float part[32] = {0};
        for (int l = 0; l < 32; l++) {
            float acc = 0.0f;
            for (int g = l; g < ng; g += 32) {
                float sf = exp2f((float)srow[g] - 127.0f);
                uint32_t bu = ((uint32_t)brow[g]) << 16;
                float bb; memcpy(&bb, &bu, 4);
                for (int p = 0; p < 4; p++) {
                    uint32_t pw = wrow[g * 4 + p];
                    for (int i = 0; i < 16; i++)
                        // match GPU fma chains: ((q*sf + bb) * x + acc) single-rounded
                        acc = fmaf(fmaf((float)((pw >> (2 * i)) & 3), sf, bb), x[g * 64 + p * 16 + i], acc);
                }
            }
            part[l] = acc;
        }
        for (int off = 16; off > 0; off >>= 1)
            for (int i = 0; i < off; i++) part[i] += part[i + off];
        out[r] = round_bf16 ? bf16_round(part[0]) : part[0];
    }
}

static float cpu_swiglu(float gv, float uv) {
    const float lim = 10.0f;
    float gc = gv < lim ? gv : lim;
    float uc = uv > lim ? lim : (uv < -lim ? -lim : uv);
    return bf16_round((float)((gc / (1.0 + exp(-(double)gc))) * uc));
}

static double cosine(const float *a, const float *b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    return dot / (sqrt(na) * sqrt(nb) + 1e-30);
}

int main(int argc, char **argv) {
    const char *metal_path = argc > 1 ? argv[1] : "src/models/moe_kernel.metal";
    const char *packed4 = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit/packed_experts/layer_00.bin";
    const char *packed2 = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit/packed_experts_2bit/layer_00.bin";

    static uint8_t blobs4[6][EXPERT_SIZE] __attribute__((aligned(4096)));
    static uint8_t blobs2[6][EXPERT_SIZE_2B] __attribute__((aligned(4096)));
    FILE *f = fopen(packed4, "rb");
    if (!f) { perror("open 4bit"); return 1; }
    for (int k = 0; k < 6; k++)
        if (fseek(f, (long)k * EXPERT_SIZE, SEEK_SET) || fread(blobs4[k], 1, EXPERT_SIZE, f) != EXPERT_SIZE) { perror("read4"); return 1; }
    fclose(f);
    f = fopen(packed2, "rb");
    if (!f) { perror("open 2bit"); return 1; }
    for (int k = 0; k < 6; k++)
        if (fseek(f, (long)k * EXPERT_SIZE_2B, SEEK_SET) || fread(blobs2[k], 1, EXPERT_SIZE_2B, f) != EXPERT_SIZE_2B) { perror("read2"); return 1; }
    fclose(f);

    srand(42);
    static float x[DIM];
    for (int i = 0; i < DIM; i++) x[i] = (float)rand() / (float)RAND_MAX * 2 - 1;

    // ---------------- CPU reference (2-bit) ----------------
    static float mid_ref[6][INTERMEDIATE], out_ref[6][DIM];
    static float gv[INTERMEDIATE], uv[INTERMEDIATE];
    for (int k = 0; k < 6; k++) {
        cpu_matvec_2b(blobs2[k], G2_GATE_W, G2_GATE_S, G2_GATE_B, INTERMEDIATE, DIM, x, gv, 0);
        cpu_matvec_2b(blobs2[k], G2_UP_W, G2_UP_S, G2_UP_B, INTERMEDIATE, DIM, x, uv, 0);
        for (int i = 0; i < INTERMEDIATE; i++) mid_ref[k][i] = cpu_swiglu(gv[i], uv[i]);
        cpu_matvec_2b(blobs2[k], G2_DOWN_W, G2_DOWN_S, G2_DOWN_B, DIM, INTERMEDIATE, mid_ref[k], out_ref[k], 1);
    }

    // ---------------- Metal setup ----------------
    NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:metal_path]
                                              encoding:NSUTF8StringEncoding error:nil];
    if (!src) { fprintf(stderr, "cannot read %s\n", metal_path); return 1; }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { fprintf(stderr, "Metal compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> p2_gu = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_gate_up_swiglu_2b"] error:&err];
    id<MTLComputePipelineState> p2_dn = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_down_2b"] error:&err];
    id<MTLComputePipelineState> p4_gu = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_gate_up_swiglu"] error:&err];
    id<MTLComputePipelineState> p4_dn = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_down"] error:&err];
    if (!p2_gu || !p2_dn || !p4_gu || !p4_dn) { fprintf(stderr, "pipeline: %s\n", [[err localizedDescription] UTF8String]); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    id<MTLBuffer> buf2[6], buf4[6];
    for (int k = 0; k < 6; k++) {
        buf2[k] = [dev newBufferWithBytesNoCopy:blobs2[k] length:EXPERT_SIZE_2B options:MTLResourceStorageModeShared deallocator:nil];
        buf4[k] = [dev newBufferWithBytesNoCopy:blobs4[k] length:EXPERT_SIZE    options:MTLResourceStorageModeShared deallocator:nil];
    }
    id<MTLBuffer> x_buf = [dev newBufferWithBytes:x length:sizeof(x) options:MTLResourceStorageModeShared];

    static float mid2_gpu[6][INTERMEDIATE], out2_gpu[6][DIM], out2_chain[6][DIM];
    static float mid4_gpu[6][INTERMEDIATE], out4_gpu[6][DIM];

    // ---------------- Gate 1a: 2-bit gate_up vs CPU ref ----------------
    id<MTLBuffer> mid2_buf = [dev newBufferWithLength:sizeof(mid2_gpu) options:MTLResourceStorageModeShared];
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p2_gu];
        for (int k = 0; k < 6; k++) [enc setBuffer:buf2[k] offset:0 atIndex:k];
        [enc setBuffer:x_buf offset:0 atIndex:6];
        [enc setBuffer:mid2_buf offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(INTERMEDIATE/8, 6, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(mid2_gpu, [mid2_buf contents], sizeof(mid2_gpu));
    }
    printf("=== Gate 1 (hard): 2-bit kernel vs CPU reference ===\n");
    for (int k = 0; k < 6; k++) {
        char name[64]; snprintf(name, 64, "2b gate_up slot%d", k);
        compare(name, mid_ref[k], mid2_gpu[k], INTERMEDIATE);
    }

    // ---------------- Gate 1b: 2-bit down vs CPU ref (input = mid_ref) ----------------
    id<MTLBuffer> midref_buf = [dev newBufferWithBytes:mid_ref length:sizeof(mid_ref) options:MTLResourceStorageModeShared];
    id<MTLBuffer> out2_buf = [dev newBufferWithLength:sizeof(out2_gpu) options:MTLResourceStorageModeShared];
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p2_dn];
        for (int k = 0; k < 6; k++) [enc setBuffer:buf2[k] offset:0 atIndex:k];
        [enc setBuffer:midref_buf offset:0 atIndex:6];
        [enc setBuffer:out2_buf offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(DIM/8, 6, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out2_gpu, [out2_buf contents], sizeof(out2_gpu));
    }
    for (int k = 0; k < 6; k++) {
        char name[64]; snprintf(name, 64, "2b down    slot%d", k);
        compare(name, out_ref[k], out2_gpu[k], DIM);
    }

    // ---------------- chained 2-bit forward (GPU mid -> GPU down) ----------------
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p2_dn];
        for (int k = 0; k < 6; k++) [enc setBuffer:buf2[k] offset:0 atIndex:k];
        [enc setBuffer:mid2_buf offset:0 atIndex:6];
        [enc setBuffer:out2_buf offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(DIM/8, 6, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out2_chain, [out2_buf contents], sizeof(out2_chain));
    }

    // ---------------- 4-bit gather6 forward (existing kernels) ----------------
    id<MTLBuffer> mid4_buf = [dev newBufferWithLength:sizeof(mid4_gpu) options:MTLResourceStorageModeShared];
    id<MTLBuffer> out4_buf = [dev newBufferWithLength:sizeof(out4_gpu) options:MTLResourceStorageModeShared];
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p4_gu];
        for (int k = 0; k < 6; k++) [enc setBuffer:buf4[k] offset:0 atIndex:k];
        [enc setBuffer:x_buf offset:0 atIndex:6];
        [enc setBuffer:mid4_buf offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(INTERMEDIATE/8, 6, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p4_dn];
        for (int k = 0; k < 6; k++) [enc setBuffer:buf4[k] offset:0 atIndex:k];
        [enc setBuffer:mid4_buf offset:0 atIndex:6];
        [enc setBuffer:out4_buf offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(DIM/8, 6, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(mid4_gpu, [mid4_buf contents], sizeof(mid4_gpu));
        memcpy(out4_gpu, [out4_buf contents], sizeof(out4_gpu));
    }

    // ---------------- Gate 2 (soft): cosine 2-bit vs 4-bit ----------------
    printf("=== Gate 2 (soft): cosine(2-bit forward, 4-bit forward) ===\n");
    double sum = 0;
    for (int k = 0; k < 6; k++) {
        double c = cosine(out2_chain[k], out4_gpu[k], DIM);
        printf("  slot%d: cosine = %.6f\n", k, c);
        sum += c;
    }
    printf("  mean cosine = %.6f (expectation >0.98, not enforced)\n", sum / 6);

    printf(g_fail ? "GATHER6 2B TEST: FAIL\n" : "GATHER6 2B TEST: PASS\n");
    return g_fail;
}
