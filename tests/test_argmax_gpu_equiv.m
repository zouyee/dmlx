// Standalone numerical equivalence test:
//   GPU two-stage argmax (argmax_partials + argmax_final in
//   src/models/moe_kernel.metal) vs a CPU linear scan with the exact
//   sampleFromLogits tie-breaking rule (strict `>`, first maximum wins).
// Covers non-power-of-2 vocab sizes, grid-stride multi-element threads,
// tied maxima, negative values, and all-equal vectors.
// Build: clang -fobjc-arc -framework Metal -framework Foundation tests/test_argmax_gpu_equiv.m -o /tmp/test_argmax_gpu_equiv
// Run from repo root: /tmp/test_argmax_gpu_equiv
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct ArgmaxCandidate { uint32_t idx; float val; };

static float frand(void) { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

static uint32_t cpu_argmax(const float *x, uint32_t n) {
    uint32_t max_i = 0;
    float max_v = x[0];
    for (uint32_t i = 1; i < n; i++) {
        if (x[i] > max_v) { max_v = x[i]; max_i = i; }
    }
    return max_i;
}

int main(void) {
    @autoreleasepool {
        NSString *path = @"src/models/moe_kernel.metal";
        NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
        if (!src) { fprintf(stderr, "cannot read %s (run from repo root)\n", [path UTF8String]); return 1; }
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "metal compile failed: %s\n", [[err description] UTF8String]); return 1; }
        id<MTLComputePipelineState> pipe_part =
            [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"argmax_partials"] error:&err];
        id<MTLComputePipelineState> pipe_final =
            [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"argmax_final"] error:&err];
        if (!pipe_part || !pipe_final) { fprintf(stderr, "pipeline failed: %s\n", [[err description] UTF8String]); return 1; }
        id<MTLCommandQueue> q = [d newCommandQueue];

        const uint32_t G = 256, T = 256, TOTAL = G * T;
        id<MTLBuffer> bpart = [d newBufferWithLength:G * sizeof(struct ArgmaxCandidate) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bout  = [d newBufferWithLength:sizeof(struct ArgmaxCandidate) options:MTLResourceStorageModeShared];

        int fails = 0;
        const uint32_t sizes[] = {1, 2, 255, 256, 257, 1000, 65536, 65537, 129023, 129024};
        for (int si = 0; si < 10; si++) {
            uint32_t n = sizes[si];
            float *x = malloc(n * sizeof(float));
            for (int variant = 0; variant < 4; variant++) {
                srand(1234 + si * 10 + variant);
                for (uint32_t i = 0; i < n; i++) x[i] = frand() * 100.0f;
                const char *name = "random";
                if (variant == 1 && n >= 3) {
                    // Tied global maximum at several indices; CPU keeps the first.
                    name = "tied-max";
                    float m = 12345.0f;
                    x[1] = m; x[n / 2] = m; x[n - 1] = m;
                } else if (variant == 2) {
                    name = "all-equal";
                    for (uint32_t i = 0; i < n; i++) x[i] = -7.25f;
                } else if (variant == 3 && n >= 2) {
                    name = "max-at-end";
                    for (uint32_t i = 0; i < n; i++) x[i] = -fabsf(frand()) - 1.0f;
                    x[n - 1] = -0.5f;
                }
                uint32_t expect = cpu_argmax(x, n);

                id<MTLBuffer> bx = [d newBufferWithBytes:x length:n * sizeof(float) options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:pipe_part];
                [e setBuffer:bx offset:0 atIndex:0];
                [e setBuffer:bpart offset:0 atIndex:1];
                [e setBytes:&n length:4 atIndex:2];
                [e setBytes:&TOTAL length:4 atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake(G, 1, 1) threadsPerThreadgroup:MTLSizeMake(T, 1, 1)];
                [e setComputePipelineState:pipe_final];
                [e setBuffer:bpart offset:0 atIndex:0];
                [e setBuffer:bout offset:0 atIndex:1];
                [e setBytes:&G length:4 atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(T, 1, 1)];
                [e endEncoding];
                [cb commit]; [cb waitUntilCompleted];

                const struct ArgmaxCandidate *res = (const struct ArgmaxCandidate *)[bout contents];
                int ok = (res->idx == expect) && (res->val == x[expect]);
                if (!ok) {
                    printf("FAIL n=%u %-10s expect=(%u,%.6f) got=(%u,%.6f)\n",
                           n, name, expect, x[expect], res->idx, res->val);
                    fails++;
                }
            }
            free(x);
        }
        printf(fails ? "FAIL (%d cases)\n" : "PASS (40 cases)\n", fails);
        return fails ? 1 : 0;
    }
}
