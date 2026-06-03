// Test mhc_pre_gpu kernel against Python reference.
// Build & run:
//   clang -framework Metal -framework Foundation -fobjc-arc \
//     scripts/mhc_pre_gpu_test.m -o /tmp/mhcpre_test_bin && /tmp/mhcpre_test_bin
//
// Requires: python3 scripts/gen_mhcpre_golden.py to generate /tmp/mhcpre_test/ data
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import <stdlib.h>

#define GD "/tmp/mhcpre_test/"

static float *readf(const char *name, size_t *n_out) {
    char path[512]; snprintf(path, sizeof(path), "%s%s", GD, name);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    float *buf = (float*)malloc(sz); fread(buf, 1, sz, f); fclose(f);
    if (n_out) *n_out = sz/sizeof(float);
    return buf;
}

int main(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@"src/models/moe_kernel.metal"
                                              encoding:NSUTF8StringEncoding error:&err];
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        printf("COMPILE FAIL: %s\n", [[err localizedDescription] UTF8String]);
        return 1;
    }

    id<MTLFunction> fn_gpu = [lib newFunctionWithName:@"mhc_pre_gpu"];
    if (!fn_gpu) { printf("FAIL: mhc_pre_gpu not found in kernel\n"); return 1; }
    id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn_gpu error:&err];
    if (!pipe) {
        printf("FAIL pipeline: %s\n", [[err localizedDescription] UTF8String]);
        return 1;
    }

    float *fn_w    = readf("fn.bin", NULL);      // [24, 16384]
    float *base    = readf("base.bin", NULL);     // [24]
    float *scale   = readf("scale.bin", NULL);    // [3]
    float *residual= readf("residual.bin", NULL); // [4, 4096]
    size_t ref_n;
    float *ref     = readf("attn_input_ref.bin", &ref_n); // [4096] bfloat-truncated

    printf("ref_n=%zu (should be 4096)\n", (size_t)ref_n);

    id<MTLBuffer> bfn  = [dev newBufferWithBytes:fn_w     length:24*16384*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bbase= [dev newBufferWithBytes:base     length:24*4       options:MTLResourceStorageModeShared];
    id<MTLBuffer> bscl = [dev newBufferWithBytes:scale    length:3*4        options:MTLResourceStorageModeShared];
    id<MTLBuffer> bres = [dev newBufferWithBytes:residual length:4*4096*4   options:MTLResourceStorageModeShared];
    id<MTLBuffer> bout = [dev newBufferWithLength:4096*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bpost= [dev newBufferWithLength:4*4    options:MTLResourceStorageModeShared];
    id<MTLBuffer> bcomb= [dev newBufferWithLength:16*4   options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pipe];
    [e setBuffer:bfn   offset:0 atIndex:0];
    [e setBuffer:bbase offset:0 atIndex:1];
    [e setBuffer:bscl  offset:0 atIndex:2];
    [e setBuffer:bres  offset:0 atIndex:3];
    [e setBuffer:bout  offset:0 atIndex:4];
    [e setBuffer:bpost offset:0 atIndex:5];
    [e setBuffer:bcomb offset:0 atIndex:6];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];

    float *out = (float*)[bout contents];

    // Compare output against Python reference
    float maxd = 0.0f, ss = 0.0f, gss = 0.0f;
    int diff_bf16 = 0;
    for (int i = 0; i < 4096; i++) {
        float d = fabsf(out[i] - ref[i]);
        if (d > maxd) maxd = d;
        ss  += d * d;
        gss += ref[i] * ref[i];
        // Count bfloat-bin differences (compare upper 16 bits)
        unsigned int ua, ub;
        memcpy(&ua, &out[i], 4); memcpy(&ub, &ref[i], 4);
        if ((ua & 0xFFFF0000u) != (ub & 0xFFFF0000u)) diff_bf16++;
    }
    float rel = sqrtf(ss) / sqrtf(gss);

    printf("out[:4] = [%.5f %.5f %.5f %.5f]\n", out[0], out[1], out[2], out[3]);
    printf("ref[:4] = [%.5f %.5f %.5f %.5f]\n", ref[0], ref[1], ref[2], ref[3]);
    printf("max_abs_diff=%.3e  rel_L2=%.3e\n", maxd, rel);
    printf("bfloat-bin differences: %d/4096\n", diff_bf16);

    // Also show post and comb
    float *post = (float*)[bpost contents];
    float *comb = (float*)[bcomb contents];
    printf("post=[%.4f %.4f %.4f %.4f]\n", post[0],post[1],post[2],post[3]);
    printf("comb[0][0]=%.4f comb[0][1]=%.4f\n", comb[0],comb[1]);

    int ok = (diff_bf16 == 0);
    printf("%s — mhc_pre_gpu bfloat output %s Python reference\n",
           ok ? "RESULT: PASS" : "RESULT: FAIL",
           ok ? "matches" : "DOES NOT MATCH");
    return ok ? 0 : 1;
}
