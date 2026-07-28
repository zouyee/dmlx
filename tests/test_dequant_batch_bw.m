// Standalone bit-exact oracle for the N-token batched affine dequant kernels:
// verifies dequant_matvec_affine_*_batch produce bit-identical output to N
// independent dequant_matvec_affine_* (per-token) dispatches.
//
// Builds synthetic packed-affine weights (4-bit nibbles + f32 scales/biases,
// group_size=64 — same layout as wq_a/wq_b/wkv/wo_b/wo_a), runs both the
// per-token kernel N times and the batched kernel once over N=16 random bf16
// inputs, and asserts maxd==0 (bit-identical). maxd!=0 means the batched
// indexing/accumulation diverged — DO NOT wire into the engine.
//
// Build:  clang -O2 -fobjc-arc -framework Foundation -framework Metal \
//           tests/test_dequant_batch_bw.m -o /tmp/test_dequant_batch_bw
// Run:    /tmp/test_dequant_batch_bw [moe_kernel.metal]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint16_t f32_to_bf16_trunc(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);                 // round-to-zero, matches engine.c:1521
}
static float bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;           // matches engine.c:1488
    float f; memcpy(&f, &u, 4);
    return f;
}

// Build synthetic packed-affine weight + scales/biases + N random bf16 inputs.
// nibbles uniform in [0,15], scales/biases in [-1,1], x in [-1,1].
typedef struct { uint32_t *W; float *sc, *bi; uint16_t *x; } AffW;

static AffW make_aff(int out_dim, int in_dim, int group_size, int n_tok) {
    int num_groups = in_dim / group_size;
    int packed_cols = in_dim / 8;
    uint32_t *W = (uint32_t *)malloc((size_t)out_dim * packed_cols * sizeof(uint32_t));
    float *sc = (float *)malloc((size_t)out_dim * num_groups * sizeof(float));
    float *bi = (float *)malloc((size_t)out_dim * num_groups * sizeof(float));
    uint16_t *x = (uint16_t *)malloc((size_t)n_tok * in_dim * sizeof(uint16_t));
    for (int i = 0; i < out_dim * packed_cols; i++) W[i] = (uint32_t)rand();
    for (int i = 0; i < out_dim * num_groups; i++) {
        sc[i] = (float)rand() / RAND_MAX * 2 - 1;
        bi[i] = (float)rand() / RAND_MAX * 2 - 1;
    }
    for (int i = 0; i < n_tok * in_dim; i++) x[i] = f32_to_bf16_trunc((float)rand() / RAND_MAX * 2 - 1);
    return (AffW){ W, sc, bi, x };
}

// max abs diff between two f32 arrays (length cnt).
static float maxd_f32(const float *a, const float *b, size_t cnt) {
    float m = 0;
    for (size_t i = 0; i < cnt; i++) {
        float d = a[i] - b[i]; if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char **argv) {
    const char *metal_path = argc > 1 ? argv[1] : "src/models/moe_kernel.metal";
    NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:metal_path]
                                              encoding:NSUTF8StringEncoding error:nil];
    if (!src) { fprintf(stderr, "cannot read %s\n", metal_path); return 1; }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }

    id<MTLComputePipelineState> p_bf16   = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_bf16out"]        error:&err];
    id<MTLComputePipelineState> p_bf16_b = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_bf16out_batch"]     error:&err];
    id<MTLComputePipelineState> p_f32    = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_f32out"]             error:&err];
    id<MTLComputePipelineState> p_f32_b = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_f32out_batch"]      error:&err];
    id<MTLComputePipelineState> p_grp8   = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_f32out_grp8"]        error:&err];
    id<MTLComputePipelineState> p_grp8_b = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_f32out_grp8_batch"]  error:&err];
    id<MTLComputePipelineState> p_f32in_b = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_f32in_f32out_batch"]  error:&err];
    if (!p_bf16||!p_bf16_b||!p_f32||!p_f32_b||!p_grp8||!p_grp8_b||!p_f32in_b) { fprintf(stderr,"pipeline missing\n"); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    srand(42);

    int fail = 0;
    const int N = 16, GS = 64;

    // ---- bf16out batch (wq_a/wq_b/wkv pattern) ----
    {
        int out_dim = 1024, in_dim = 4096;
        AffW w = make_aff(out_dim, in_dim, GS, N);
        id<MTLBuffer> bW  = [dev newBufferWithBytesNoCopy:w.W  length:(size_t)out_dim*(in_dim/8)*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bsc = [dev newBufferWithBytesNoCopy:w.sc length:(size_t)out_dim*(in_dim/GS)*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bbi = [dev newBufferWithBytesNoCopy:w.bi length:(size_t)out_dim*(in_dim/GS)*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bx  = [dev newBufferWithBytesNoCopy:w.x  length:(size_t)N*in_dim*2 options:MTLResourceStorageModeShared deallocator:nil];
        // per-token reference: N dispatches, each out_dim threads
        id<MTLBuffer> bref = [dev newBufferWithLength:(size_t)N*out_dim*2 options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            for (int t = 0; t < N; t++) {
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p_bf16];
                [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bsc offset:0 atIndex:1];
                [e setBuffer:bbi offset:0 atIndex:2]; [e setBuffer:bx offset:(NSUInteger)t*in_dim*2 atIndex:3];
                [e setBuffer:bref offset:(NSUInteger)t*out_dim*2 atIndex:4];
                uint od=out_dim, id_=in_dim, gs=GS;
                [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
                [e dispatchThreads:MTLSizeMake(out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            }
            [cb commit]; [cb waitUntilCompleted];
        }
        // batched: one dispatch, N*out_dim threads
        id<MTLBuffer> bbat = [dev newBufferWithLength:(size_t)N*out_dim*2 options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p_bf16_b];
            [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bsc offset:0 atIndex:1];
            [e setBuffer:bbi offset:0 atIndex:2]; [e setBuffer:bx offset:0 atIndex:3]; [e setBuffer:bbat offset:0 atIndex:4];
            uint od=out_dim, id_=in_dim, gs=GS, n=N;
            [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7]; [e setBytes:&n length:4 atIndex:8];
            [e dispatchThreads:MTLSizeMake(N*out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }
        uint16_t *ref = (uint16_t*)[bref contents], *bat = (uint16_t*)[bbat contents];
        float m = 0; for (size_t i=0;i<(size_t)N*out_dim;i++){ float d=bf16_to_f32(ref[i])-bf16_to_f32(bat[i]); if(d<0)d=-d; if(d>m)m=d; }
        printf("bf16out_batch  out=%d in=%d N=%d  maxd=%g  %s\n", out_dim, in_dim, N, m, m==0.0f?"GO":"NO-GO");
        if (m != 0.0f) fail = 1;
        free(w.W); free(w.sc); free(w.bi); free(w.x);
    }

    // ---- f32out batch (wo_b pattern) ----
    {
        int out_dim = 4096, in_dim = 8192;
        AffW w = make_aff(out_dim, in_dim, GS, N);
        id<MTLBuffer> bW  = [dev newBufferWithBytesNoCopy:w.W  length:(size_t)out_dim*(in_dim/8)*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bsc = [dev newBufferWithBytesNoCopy:w.sc length:(size_t)out_dim*(in_dim/GS)*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bbi = [dev newBufferWithBytesNoCopy:w.bi length:(size_t)out_dim*(in_dim/GS)*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bx  = [dev newBufferWithBytesNoCopy:w.x  length:(size_t)N*in_dim*2 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bref = [dev newBufferWithLength:(size_t)N*out_dim*4 options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            for (int t = 0; t < N; t++) {
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p_f32];
                [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bsc offset:0 atIndex:1];
                [e setBuffer:bbi offset:0 atIndex:2]; [e setBuffer:bx offset:(NSUInteger)t*in_dim*2 atIndex:3];
                [e setBuffer:bref offset:(NSUInteger)t*out_dim*4 atIndex:4];
                uint od=out_dim, id_=in_dim, gs=GS;
                [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
                [e dispatchThreads:MTLSizeMake(out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            }
            [cb commit]; [cb waitUntilCompleted];
        }
        id<MTLBuffer> bbat = [dev newBufferWithLength:(size_t)N*out_dim*4 options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p_f32_b];
            [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bsc offset:0 atIndex:1];
            [e setBuffer:bbi offset:0 atIndex:2]; [e setBuffer:bx offset:0 atIndex:3]; [e setBuffer:bbat offset:0 atIndex:4];
            uint od=out_dim, id_=in_dim, gs=GS, n=N;
            [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7]; [e setBytes:&n length:4 atIndex:8];
            [e dispatchThreads:MTLSizeMake(N*out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }
        float m = maxd_f32((float*)[bref contents], (float*)[bbat contents], (size_t)N*out_dim);
        printf("f32out_batch   out=%d in=%d N=%d  maxd=%g  %s\n", out_dim, in_dim, N, m, m==0.0f?"GO":"NO-GO");
        if (m != 0.0f) fail = 1;
        free(w.W); free(w.sc); free(w.bi); free(w.x);
    }

    // ---- grp8 batch (wo_a pattern: 8 groups, out_g per group) ----
    {
        int out_g = 1024, in_dim = 4096;        // wo_a: 8 groups * 1024 rows, in 4096
        int out_dim = 8 * out_g;
        // grp8 weight is [8*out_g, in_dim] → same layout as make_aff(out_dim=8*out_g,...)
        AffW w = make_aff(out_dim, in_dim, GS, N);
        // grp8 input is grouped [N, 8, in_dim]: replicate each token's in_dim vector 8 times
        // (the per-token grp8 kernel reads x + group*in_dim per group). For the test we just
        // need consistent inputs — build [N, 8, in_dim] where each group slice is the same
        // random vector (so the 8 groups differ only by weight row, as in real wo_a).
        uint16_t *xg = (uint16_t *)malloc((size_t)N * 8 * in_dim * sizeof(uint16_t));
        for (int t = 0; t < N; t++)
            for (int g = 0; g < 8; g++)
                memcpy(xg + ((size_t)t*8 + g)*in_dim, w.x + (size_t)t*in_dim, in_dim * sizeof(uint16_t));
        id<MTLBuffer> bW  = [dev newBufferWithBytesNoCopy:w.W  length:(size_t)out_dim*(in_dim/8)*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bsc = [dev newBufferWithBytesNoCopy:w.sc length:(size_t)out_dim*(in_dim/GS)*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bbi = [dev newBufferWithBytesNoCopy:w.bi length:(size_t)out_dim*(in_dim/GS)*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bx  = [dev newBufferWithBytesNoCopy:xg  length:(size_t)N*8*in_dim*2 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bref = [dev newBufferWithLength:(size_t)N*out_dim*4 options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            for (int t = 0; t < N; t++) {
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p_grp8];
                [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bsc offset:0 atIndex:1];
                [e setBuffer:bbi offset:0 atIndex:2]; [e setBuffer:bx offset:(NSUInteger)t*8*in_dim*2 atIndex:3];
                [e setBuffer:bref offset:(NSUInteger)t*out_dim*4 atIndex:4];
                uint og=out_g, id_=in_dim, gs=GS;
                [e setBytes:&og length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
                [e dispatchThreads:MTLSizeMake(out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            }
            [cb commit]; [cb waitUntilCompleted];
        }
        id<MTLBuffer> bbat = [dev newBufferWithLength:(size_t)N*out_dim*4 options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p_grp8_b];
            [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bsc offset:0 atIndex:1];
            [e setBuffer:bbi offset:0 atIndex:2]; [e setBuffer:bx offset:0 atIndex:3]; [e setBuffer:bbat offset:0 atIndex:4];
            uint og=out_g, id_=in_dim, gs=GS, n=N;
            [e setBytes:&og length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7]; [e setBytes:&n length:4 atIndex:8];
            [e dispatchThreads:MTLSizeMake(N*out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }
        float m = maxd_f32((float*)[bref contents], (float*)[bbat contents], (size_t)N*out_dim);
        printf("grp8_batch     8*%d in=%d N=%d  maxd=%g  %s\n", out_g, in_dim, N, m, m==0.0f?"GO":"NO-GO");
        if (m != 0.0f) fail = 1;
        free(w.W); free(w.sc); free(w.bi); free(w.x); free(xg);
    }

    // ---- f32in batch (wo_b pattern: f32 in, f32 out) ----
    // No per-token naive f32-in kernel exists (only tiled v2, different accum),
    // so self-reference: run batched n_tok=1 N times (offset per token) vs one
    // n_tok=N dispatch. Validates token indexing is correct.
    {
        int out_dim = 4096, in_dim = 8192;
        // build f32 weight+scales+biases + f32 x
        int num_groups = in_dim / GS, packed_cols = in_dim / 8;
        uint32_t *W = (uint32_t *)malloc((size_t)out_dim*packed_cols*4);
        float *sc = (float *)malloc((size_t)out_dim*num_groups*4);
        float *bi = (float *)malloc((size_t)out_dim*num_groups*4);
        float *x = (float *)malloc((size_t)N*in_dim*4);
        for (int i=0;i<out_dim*packed_cols;i++) W[i]=(uint32_t)rand();
        for (int i=0;i<out_dim*num_groups;i++){sc[i]=(float)rand()/RAND_MAX*2-1;bi[i]=(float)rand()/RAND_MAX*2-1;}
        for (int i=0;i<N*in_dim;i++) x[i]=(float)rand()/RAND_MAX*2-1;
        id<MTLBuffer> bW=[dev newBufferWithBytesNoCopy:W length:(size_t)out_dim*packed_cols*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bsc=[dev newBufferWithBytesNoCopy:sc length:(size_t)out_dim*num_groups*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bbi=[dev newBufferWithBytesNoCopy:bi length:(size_t)out_dim*num_groups*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bx=[dev newBufferWithBytesNoCopy:x length:(size_t)N*in_dim*4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bref=[dev newBufferWithLength:(size_t)N*out_dim*4 options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cb=[q commandBuffer];
            for (int t=0;t<N;t++){
                id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
                [e setComputePipelineState:p_f32in_b];
                [e setBuffer:bW offset:0 atIndex:0];[e setBuffer:bsc offset:0 atIndex:1];
                [e setBuffer:bbi offset:0 atIndex:2];[e setBuffer:bx offset:(NSUInteger)t*in_dim*4 atIndex:3];
                [e setBuffer:bref offset:(NSUInteger)t*out_dim*4 atIndex:4];
                uint od=out_dim,id_=in_dim,gs=GS,n=1;
                [e setBytes:&od length:4 atIndex:5];[e setBytes:&id_ length:4 atIndex:6];[e setBytes:&gs length:4 atIndex:7];[e setBytes:&n length:4 atIndex:8];
                [e dispatchThreads:MTLSizeMake(out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            }
            [cb commit];[cb waitUntilCompleted];
        }
        id<MTLBuffer> bbat=[dev newBufferWithLength:(size_t)N*out_dim*4 options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cb=[q commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:p_f32in_b];
            [e setBuffer:bW offset:0 atIndex:0];[e setBuffer:bsc offset:0 atIndex:1];
            [e setBuffer:bbi offset:0 atIndex:2];[e setBuffer:bx offset:0 atIndex:3];[e setBuffer:bbat offset:0 atIndex:4];
            uint od=out_dim,id_=in_dim,gs=GS,n=N;
            [e setBytes:&od length:4 atIndex:5];[e setBytes:&id_ length:4 atIndex:6];[e setBytes:&gs length:4 atIndex:7];[e setBytes:&n length:4 atIndex:8];
            [e dispatchThreads:MTLSizeMake(N*out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
            [cb commit];[cb waitUntilCompleted];
        }
        float m = maxd_f32((float*)[bref contents], (float*)[bbat contents], (size_t)N*out_dim);
        printf("f32in_batch    out=%d in=%d N=%d  maxd=%g  %s (self-ref n=1 vs n=N)\n", out_dim, in_dim, N, m, m==0.0f?"GO":"NO-GO");
        if (m != 0.0f) fail = 1;
        free(W);free(sc);free(bi);free(x);
    }

    printf(fail ? "RESULT: NO-GO (batched != per-token)\n" : "RESULT: GO (bit-exact)\n");
    return fail;
}
