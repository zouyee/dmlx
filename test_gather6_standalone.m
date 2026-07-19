// Standalone regression test: gather6 kernels vs old per-expert kernels.
//
// Loads 6 real expert blobs from packed_experts/layer_00.bin and compares
//   gather6_gate_up_swiglu vs fused_gate_up_swiglu_v2   (gate+up+SwiGLU)
//   gather6_down          vs dequant_matvec_4bit         (down projection)
// on identical inputs. Down must be bit-exact; gate_up may differ by ~1 ulp
// on a few elements due to reduction order (tolerance: 0 mismatches at
// abs>1e-3 && rel>1e-2).
//
// Build:  clang -O2 -fobjc-arc -framework Foundation -framework Metal \
//           test_gather6_standalone.m -o /tmp/test_gather6
// Run:    /tmp/test_gather6 [moe_kernel.metal path] [packed layer bin]
// Exit:   0 = pass, 1 = fail
//
// History: caught a latent UB bug in the original gather_down (simd_sum
// called inside a divergent `if (simd_lane == 0)` branch), inherited by
// gather6_down — outputs were garbage despite correct addressing.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EXPERT_SIZE 13369344
#define GATE_W_OFF  0
#define GATE_S_OFF  4194304
#define UP_W_OFF    4456448
#define UP_S_OFF    8650752
#define DOWN_W_OFF  8912896
#define DOWN_S_OFF  13107200
#define DIM 4096
#define INTERMEDIATE 2048

static int g_fail = 0;

static void compare(const char *name, const float *a, const float *b, int n) {
    double max_abs = 0, max_rel = 0; int bad = 0, exact = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        double r = d / (fabs((double)a[i]) + 1e-6);
        if (d > max_abs) max_abs = d;
        if (r > max_rel) max_rel = r;
        if (a[i] == b[i]) exact++;
        if (d > 1e-3 && r > 1e-2) bad++;
    }
    printf("%s: max_abs=%.6g max_rel=%.6g mismatched=%d/%d exact=%d/%d %s\n",
           name, max_abs, max_rel, bad, n, exact, n, bad == 0 ? "OK" : "*** FAIL ***");
    if (bad) g_fail = 1;
}

int main(int argc, char **argv) {
    const char *metal_path = argc > 1 ? argv[1] : "src/models/moe_kernel.metal";
    const char *packed = argc > 2 ? argv[2]
        : "packed_experts/layer_00.bin";

    // Load 6 expert blobs (eids 0..5)
    static uint8_t blobs[6][EXPERT_SIZE] __attribute__((aligned(4096)));
    FILE *f = fopen(packed, "rb");
    if (!f) { perror("open packed"); return 1; }
    for (int k = 0; k < 6; k++) {
        if (fseek(f, (long)k * EXPERT_SIZE, SEEK_SET) != 0 ||
            fread(blobs[k], 1, EXPERT_SIZE, f) != EXPERT_SIZE) { perror("read"); return 1; }
    }
    fclose(f);

    // Deterministic random inputs
    srand(42);
    static float x[DIM], mid_in[6][INTERMEDIATE];
    for (int i = 0; i < DIM; i++) x[i] = (float)rand() / (float)RAND_MAX * 2 - 1;
    for (int k = 0; k < 6; k++)
        for (int i = 0; i < INTERMEDIATE; i++) mid_in[k][i] = (float)rand() / (float)RAND_MAX * 2 - 1;

    NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:metal_path]
                                              encoding:NSUTF8StringEncoding error:nil];
    if (!src) { fprintf(stderr, "cannot read %s\n", metal_path); return 1; }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { fprintf(stderr, "Metal compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }

    id<MTLComputePipelineState> p_old_gu = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_gate_up_swiglu_v2"] error:&err];
    id<MTLComputePipelineState> p_old_dn = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_4bit"] error:&err];
    id<MTLComputePipelineState> p_g6_gu  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_gate_up_swiglu"] error:&err];
    id<MTLComputePipelineState> p_g6_dn  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_down"] error:&err];
    if (!p_old_gu || !p_old_dn || !p_g6_gu || !p_g6_dn) { fprintf(stderr, "pipeline: %s\n", [[err localizedDescription] UTF8String]); return 1; }

    id<MTLCommandQueue> q = [dev newCommandQueue];

    id<MTLBuffer> blob_buf[6];
    for (int k = 0; k < 6; k++)
        blob_buf[k] = [dev newBufferWithBytesNoCopy:blobs[k] length:EXPERT_SIZE
                                            options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> x_buf = [dev newBufferWithBytes:x length:sizeof(x) options:MTLResourceStorageModeShared];

    // ---- gate+up: old per-expert reference ----
    static float ref_mid[6][INTERMEDIATE];
    for (int k = 0; k < 6; k++) {
        id<MTLBuffer> gw = [dev newBufferWithBytesNoCopy:blobs[k]+GATE_W_OFF length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> gs = [dev newBufferWithBytesNoCopy:blobs[k]+GATE_S_OFF length:262144  options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> uw = [dev newBufferWithBytesNoCopy:blobs[k]+UP_W_OFF   length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> us = [dev newBufferWithBytesNoCopy:blobs[k]+UP_S_OFF   length:262144  options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> ob = [dev newBufferWithBytes:ref_mid[k] length:sizeof(ref_mid[k]) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_old_gu];
        [enc setBuffer:gw offset:0 atIndex:0]; [enc setBuffer:gs offset:0 atIndex:1];
        [enc setBuffer:uw offset:0 atIndex:2]; [enc setBuffer:us offset:0 atIndex:3];
        [enc setBuffer:x_buf offset:0 atIndex:4]; [enc setBuffer:ob offset:0 atIndex:5];
        uint od=INTERMEDIATE, id_=DIM, gsz=32;
        [enc setBytes:&od length:4 atIndex:6]; [enc setBytes:&id_ length:4 atIndex:7]; [enc setBytes:&gsz length:4 atIndex:8];
        [enc setThreadgroupMemoryLength:512 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((INTERMEDIATE+1)/2,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(ref_mid[k], [ob contents], sizeof(ref_mid[k]));
    }

    // ---- gate+up: gather6 ----
    static float new_mid[6][INTERMEDIATE];
    id<MTLBuffer> g6_mid_buf = [dev newBufferWithLength:sizeof(new_mid) options:MTLResourceStorageModeShared];
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_g6_gu];
        for (int k = 0; k < 6; k++) [enc setBuffer:blob_buf[k] offset:0 atIndex:k];
        [enc setBuffer:x_buf offset:0 atIndex:6];
        [enc setBuffer:g6_mid_buf offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(INTERMEDIATE/8, 6, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(new_mid, [g6_mid_buf contents], sizeof(new_mid));
    }
    for (int k = 0; k < 6; k++) {
        char name[64]; snprintf(name, 64, "gate_up slot%d", k);
        compare(name, ref_mid[k], new_mid[k], INTERMEDIATE);
    }

    // ---- down: old per-expert reference (input = mid_in) ----
    static float ref_out[6][DIM];
    for (int k = 0; k < 6; k++) {
        id<MTLBuffer> dw = [dev newBufferWithBytesNoCopy:blobs[k]+DOWN_W_OFF length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> ds = [dev newBufferWithBytesNoCopy:blobs[k]+DOWN_S_OFF length:262144  options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> mb = [dev newBufferWithBytes:mid_in[k] length:sizeof(mid_in[k]) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [dev newBufferWithBytes:ref_out[k] length:sizeof(ref_out[k]) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_old_dn];
        [enc setBuffer:dw offset:0 atIndex:0]; [enc setBuffer:ds offset:0 atIndex:1];
        [enc setBuffer:mb offset:0 atIndex:2]; [enc setBuffer:ob offset:0 atIndex:3];
        uint od=DIM, id_=INTERMEDIATE, gsz=32;
        [enc setBytes:&od length:4 atIndex:4]; [enc setBytes:&id_ length:4 atIndex:5]; [enc setBytes:&gsz length:4 atIndex:6];
        [enc setThreadgroupMemoryLength:256 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(DIM/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(ref_out[k], [ob contents], sizeof(ref_out[k]));
    }

    // ---- down: gather6 (input = same mid_in, contiguous [6×2048]) ----
    static float new_out[6][DIM];
    id<MTLBuffer> mid6 = [dev newBufferWithBytes:mid_in length:sizeof(mid_in) options:MTLResourceStorageModeShared];
    id<MTLBuffer> g6_out_buf = [dev newBufferWithLength:sizeof(new_out) options:MTLResourceStorageModeShared];
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_g6_dn];
        for (int k = 0; k < 6; k++) [enc setBuffer:blob_buf[k] offset:0 atIndex:k];
        [enc setBuffer:mid6 offset:0 atIndex:6];
        [enc setBuffer:g6_out_buf offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(DIM/8, 6, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(new_out, [g6_out_buf contents], sizeof(new_out));
    }
    for (int k = 0; k < 6; k++) {
        char name[64]; snprintf(name, 64, "down slot%d", k);
        compare(name, ref_out[k], new_out[k], DIM);
    }

    printf(g_fail ? "GATHER6 TEST: FAIL\n" : "GATHER6 TEST: PASS\n");
    return g_fail;
}
