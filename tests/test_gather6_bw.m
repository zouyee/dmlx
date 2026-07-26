// gather6 bandwidth microbenchmark: gate_up + down on 6 real expert blobs,
// 20 iterations in one CB, GPU timestamps. Answers: is the engine's ~53GB/s
// kernel-limited or environment-limited (page-in/swap)?
//
// Build:  clang -O2 -fobjc-arc -framework Foundation -framework Metal \
//           test_gather6_bw.m -o /tmp/test_gather6_bw
// Run:    /tmp/test_gather6_bw [moe_kernel.metal] [packed layer bin]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXPERT_SIZE 13369344
#define DIM 4096
#define INTERMEDIATE 2048

int main(int argc, char **argv) {
    const char *metal_path = argc > 1 ? argv[1] : "src/models/moe_kernel.metal";
    const char *packed = argc > 2 ? argv[2] : "packed_experts/layer_00.bin";

    static uint8_t blobs[6][EXPERT_SIZE] __attribute__((aligned(4096)));
    FILE *f = fopen(packed, "rb");
    if (!f) { perror("open packed"); return 1; }
    for (int k = 0; k < 6; k++)
        if (fseek(f, (long)k * EXPERT_SIZE, SEEK_SET) != 0 ||
            fread(blobs[k], 1, EXPERT_SIZE, f) != EXPERT_SIZE) { perror("read"); return 1; }
    fclose(f);

    NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:metal_path]
                                              encoding:NSUTF8StringEncoding error:nil];
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> p_gu = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_gate_up_swiglu"] error:&err];
    id<MTLComputePipelineState> p_dn = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather6_down"] error:&err];
    id<MTLCommandQueue> q = [dev newCommandQueue];

    id<MTLBuffer> blob_buf[6];
    for (int k = 0; k < 6; k++)
        blob_buf[k] = [dev newBufferWithBytesNoCopy:blobs[k] length:EXPERT_SIZE
                                            options:MTLResourceStorageModeShared deallocator:nil];
    static float x[DIM];
    srand(42);
    for (int i = 0; i < DIM; i++) x[i] = (float)rand() / RAND_MAX * 2 - 1;
    id<MTLBuffer> x_buf = [dev newBufferWithBytes:x length:sizeof(x) options:MTLResourceStorageModeShared];
    id<MTLBuffer> mid_buf = [dev newBufferWithLength:6 * INTERMEDIATE * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf = [dev newBufferWithLength:6 * DIM * sizeof(float) options:MTLResourceStorageModeShared];

    const int REPS = 20;
    // warmup
    for (int w = 0; w < 3; w++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p_gu];
        for (int k = 0; k < 6; k++) [e setBuffer:blob_buf[k] offset:0 atIndex:k];
        [e setBuffer:x_buf offset:0 atIndex:6];
        [e setBuffer:mid_buf offset:0 atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(INTERMEDIATE/8, 6, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }
    double t_gu = 0, t_dn = 0;
    for (int trial = 0; trial < 3; trial++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        for (int r = 0; r < REPS; r++) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p_gu];
            for (int k = 0; k < 6; k++) [e setBuffer:blob_buf[k] offset:0 atIndex:k];
            [e setBuffer:x_buf offset:0 atIndex:6];
            [e setBuffer:mid_buf offset:0 atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake(INTERMEDIATE/8, 6, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }
        [cb commit]; [cb waitUntilCompleted];
        double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3 / REPS;
        if (t_gu == 0 || ms < t_gu) t_gu = ms;

        id<MTLCommandBuffer> cb2 = [q commandBuffer];
        for (int r = 0; r < REPS; r++) {
            id<MTLComputeCommandEncoder> e = [cb2 computeCommandEncoder];
            [e setComputePipelineState:p_dn];
            for (int k = 0; k < 6; k++) [e setBuffer:blob_buf[k] offset:0 atIndex:k];
            [e setBuffer:mid_buf offset:0 atIndex:6];
            [e setBuffer:out_buf offset:0 atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake(DIM/8, 6, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }
        [cb2 commit]; [cb2 waitUntilCompleted];
        double ms2 = ([cb2 GPUEndTime] - [cb2 GPUStartTime]) * 1e3 / REPS;
        if (t_dn == 0 || ms2 < t_dn) t_dn = ms2;
    }
    // gate_up reads full 13.4MB blobs (gate+up weights+scales ~ 2/3 = 8.9MB effective, but kernel streams whole blob regions for gate+up = 8.9MB)
    // down reads the remaining down region = 4.5MB
    double gu_bytes = 6.0 * (4194304 + 262144) * 2;   // gate_w+gate_s + up_w+up_s per expert
    double dn_bytes = 6.0 * (4194304 + 262144);       // down_w+down_s per expert
    printf("gate_up: %.3fms/dispatch  %.1f GB/s (%.1f MB)\n", t_gu, gu_bytes / t_gu / 1e6, gu_bytes / 1e6);
    printf("down:    %.3fms/dispatch  %.1f GB/s (%.1f MB)\n", t_dn, dn_bytes / t_dn / 1e6, dn_bytes / 1e6);
    printf("combined per-layer cost: %.3fms for 80MB blob -> engine moe phase reference 1.5ms\n", t_gu + t_dn);
    return 0;
}
