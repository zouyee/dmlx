// Standalone numerical equivalence test:
//   GPU dequant_matvec_affine (src/models/moe_kernel.metal)
//   vs CPU cpu_affine_matvec_safe (verbatim copy from src/metal_infer/engine.c)
// Uses indexer compressor dims: out_dim=256, in_dim=DIM=4096, several group sizes.
// Build: clang -fobjc-arc -framework Metal -framework Foundation test_idx_gpu_equiv.m -o /tmp/test_idx_gpu_equiv
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint32_t *packed;  // [out_dim, in_dim/8]
    float *scales;     // [out_dim, num_groups]
    float *biases;     // [out_dim, num_groups]
    int out_dim, in_dim, group_size;
} QuantWeight;

// Verbatim copy of cpu_affine_matvec_safe from src/metal_infer/engine.c:2803
static void cpu_affine_matvec_safe(float *out, const QuantWeight *w, const float *x) {
    if (w->out_dim == 0 || w->in_dim == 0 || w->packed == NULL) return;
    const int num_groups = w->in_dim / w->group_size;
    const int ppg = w->group_size / 8;
    const int packed_cols = w->in_dim / 8;
    for (int row = 0; row < w->out_dim; row++) {
        const uint32_t *wr = w->packed + (size_t)row * packed_cols;
        const float *sc = w->scales + (size_t)row * num_groups;
        const float *bi = (w->biases) ? w->biases + (size_t)row * num_groups : NULL;
        float acc = 0.0f;
        for (int g = 0; g < num_groups; g++) {
            float scale = sc[g];
            float bias = bi ? bi[g] : 0.0f;
            for (int p = 0; p < ppg; p++) {
                uint32_t pw = wr[g * ppg + p];
                for (int k = 0; k < 8; k++) {
                    float nib = (float)((pw >> (k * 4)) & 0xF);
                    acc += (scale * nib + bias) * x[g * w->group_size + p * 8 + k];
                }
            }
        }
        out[row] = acc;
    }
}

static float frand(void) { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

int main(void) {
    @autoreleasepool {
        // Load the project's actual kernel source
        NSString *path = @"src/models/moe_kernel.metal";
        NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
        if (!src) { fprintf(stderr, "cannot read %s (run from repo root)\n", [path UTF8String]); return 1; }
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "metal compile failed: %s\n", [[err description] UTF8String]); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"dequant_matvec_affine"];
        id<MTLComputePipelineState> pipe = [d newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) { fprintf(stderr, "pipeline failed: %s\n", [[err description] UTF8String]); return 1; }
        id<MTLCommandQueue> q = [d newCommandQueue];

        int gs_list[] = {64, 128};
        int fails = 0;
        for (int gi = 0; gi < 2; gi++) {
            QuantWeight w;
            w.out_dim = 256; w.in_dim = 4096; w.group_size = gs_list[gi];
            int ng = w.in_dim / w.group_size;
            w.packed = malloc((size_t)w.out_dim * (w.in_dim/8) * sizeof(uint32_t));
            w.scales = malloc((size_t)w.out_dim * ng * sizeof(float));
            w.biases = malloc((size_t)w.out_dim * ng * sizeof(float));
            float *x = malloc(w.in_dim * sizeof(float));
            float *out_cpu = malloc(w.out_dim * sizeof(float));
            srand(42 + gi);
            for (size_t i = 0; i < (size_t)w.out_dim * (w.in_dim/8); i++) w.packed[i] = (uint32_t)rand();
            for (size_t i = 0; i < (size_t)w.out_dim * ng; i++) { w.scales[i] = frand() * 0.1f; w.biases[i] = frand() * 0.5f; }
            for (int i = 0; i < w.in_dim; i++) x[i] = frand();

            // CPU reference
            cpu_affine_matvec_safe(out_cpu, &w, x);

            // GPU — same buffer/dispatch pattern as the indexer change
            id<MTLBuffer> bw = [d newBufferWithBytesNoCopy:w.packed length:(size_t)w.out_dim*(w.in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLBuffer> bs = [d newBufferWithBytes:w.scales length:(size_t)w.out_dim*ng*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bb = [d newBufferWithBytes:w.biases length:(size_t)w.out_dim*ng*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bx = [d newBufferWithBytes:x length:w.in_dim*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bo = [d newBufferWithLength:w.out_dim*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pipe];
            [e setBuffer:bw offset:0 atIndex:0];
            [e setBuffer:bs offset:0 atIndex:1];
            [e setBuffer:bb offset:0 atIndex:2];
            [e setBuffer:bx offset:0 atIndex:3];
            [e setBuffer:bo offset:0 atIndex:4];
            uint od = w.out_dim, id_ = w.in_dim, gs = w.group_size;
            [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
            [e dispatchThreads:MTLSizeMake(od,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            float *out_gpu = (float *)[bo contents];

            double max_abs = 0, max_rel = 0;
            for (int i = 0; i < w.out_dim; i++) {
                double diff = fabs((double)out_gpu[i] - out_cpu[i]);
                double rel = diff / (fabs((double)out_cpu[i]) + 1e-6);
                if (diff > max_abs) max_abs = diff;
                if (rel > max_rel) max_rel = rel;
            }
            printf("gs=%3d  out_dim=%d in_dim=%d  max_abs=%.3e  max_rel=%.3e  cpu[0]=%.6f gpu[0]=%.6f\n",
                   w.group_size, w.out_dim, w.in_dim, max_abs, max_rel, out_cpu[0], out_gpu[0]);
            if (max_rel > 1e-3 && max_abs > 1e-2) fails++;
            free(w.packed); free(w.scales); free(w.biases); free(x); free(out_cpu);
        }
        printf(fails ? "FAIL\n" : "PASS\n");
        return fails;
    }
}
