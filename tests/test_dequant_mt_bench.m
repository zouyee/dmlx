// Standalone microbenchmark: single-token dequant matvec (v2) vs
// multi-token dequant GEMM prototype (v2_mt) on real MLA attention shapes.
//
// Question: does batching N tokens through one weight read amortize the
// attention projection cost? (prefill mla phase = 4.0ms/layer/token; if the
// matvecs are weight-bandwidth-bound, MT gives ~Nx on the matvec portion.)
//
// Build:  clang -O2 -fobjc-arc -framework Foundation -framework Metal \
//           test_dequant_mt_bench.m -o /tmp/test_dequant_mt
// Run:    /tmp/test_dequant_mt [moe_kernel.metal path]
// Exit:   0 = pass (correctness), 1 = fail
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_N 16

static int g_fail = 0;

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = ((uint32_t)b) << 16; float f; memcpy(&f, &u, 4); return f; }

// Append the MT prototype kernel source (kept here so the production metal
// file stays clean until the win is proven).
static NSString *mtKernelSrc(void) {
    return @"\n"
    "// Multi-token variant of dequant_matvec_affine_bf16in_bf16out_v2:\n"
    "// identical per-token accumulation + reduction order (bit-exact vs v2),\n"
    "// packed weight word read once per (gg,row) and applied to all N tokens.\n"
    "kernel void dequant_gemm_affine_bf16_mt(\n"
    "    device const uint32_t* W_packed   [[buffer(0)]],\n"
    "    device const float*    scales     [[buffer(1)]],\n"
    "    device const float*    biases     [[buffer(2)]],\n"
    "    device const bfloat*   x_batch    [[buffer(3)]],  // [N, in_dim]\n"
    "    device bfloat*         out        [[buffer(4)]],  // [N, out_dim]\n"
    "    constant uint&         out_dim    [[buffer(5)]],\n"
    "    constant uint&         in_dim     [[buffer(6)]],\n"
    "    constant uint&         group_size [[buffer(7)]],\n"
    "    constant uint&         ntok       [[buffer(8)]],\n"
    "    threadgroup float*     shmem      [[threadgroup(0)]],\n"
    "    uint3  tgpig  [[threadgroup_position_in_grid]],\n"
    "    ushort tiisg  [[thread_index_in_simdgroup]],\n"
    "    ushort sgitg  [[simdgroup_index_in_threadgroup]]\n"
    ") {\n"
    "    const short NR0 = 2, NSG = 4, NW = 32, NQ = 4, TPG = 8;\n"
    "    const uint num_groups = in_dim / group_size;\n"
    "    const uint packed_per_group = group_size / 8;\n"
    "    const uint packed_cols = in_dim / 8;\n"
    "    const int row0 = (int)tgpig.x * NR0;\n"
    "    const short ix = tiisg / TPG;\n"
    "    const short il = tiisg % TPG;\n"
    "    const int g0 = (int)sgitg * NQ + (int)ix;\n"
    "    device const uint32_t *wr[NR0];\n"
    "    device const float *sr[NR0], *br[NR0];\n"
    "    for (short row = 0; row < NR0; row++) {\n"
    "        int r = row0 + row;\n"
    "        if (r < (int)out_dim) {\n"
    "            wr[row] = W_packed + r * packed_cols;\n"
    "            sr[row] = scales + r * num_groups;\n"
    "            br[row] = biases + r * num_groups;\n"
    "        }\n"
    "    }\n"
    "    float acc[16][NR0];\n"
    "    for (uint t = 0; t < 16; t++) { acc[t][0] = 0.0f; acc[t][1] = 0.0f; }\n"
    "    for (int gg = g0; gg < (int)num_groups; gg += NSG * NQ) {\n"
    "        uint32_t pw[NR0];\n"
    "        float sc_r[NR0], bi_r[NR0];\n"
    "        for (short row = 0; row < NR0; row++) {\n"
    "            int r = row0 + row;\n"
    "            if (r >= (int)out_dim) continue;\n"
    "            sc_r[row] = sr[row][gg];\n"
    "            bi_r[row] = br[row][gg];\n"
    "            pw[row] = wr[row][gg * packed_per_group + (uint)il];\n"
    "        }\n"
    "        uint xb = (uint)gg * group_size + (uint)il * 8;\n"
    "        for (uint t = 0; t < ntok; t++) {\n"
    "            device const bfloat *x = x_batch + t * in_dim + xb;\n"
    "            float xv0 = float(x[0]), xv1 = float(x[1]), xv2 = float(x[2]), xv3 = float(x[3]);\n"
    "            float xv4 = float(x[4]), xv5 = float(x[5]), xv6 = float(x[6]), xv7 = float(x[7]);\n"
    "            for (short row = 0; row < NR0; row++) {\n"
    "                int r = row0 + row;\n"
    "                if (r >= (int)out_dim) continue;\n"
    "                float s = sc_r[row], b = bi_r[row];\n"
    "                uint32_t w = pw[row];\n"
    "                acc[t][row] += fma(float((w>> 0)&0xF), s*xv0, b*xv0);\n"
    "                acc[t][row] += fma(float((w>> 4)&0xF), s*xv1, b*xv1);\n"
    "                acc[t][row] += fma(float((w>> 8)&0xF), s*xv2, b*xv2);\n"
    "                acc[t][row] += fma(float((w>>12)&0xF), s*xv3, b*xv3);\n"
    "                acc[t][row] += fma(float((w>>16)&0xF), s*xv4, b*xv4);\n"
    "                acc[t][row] += fma(float((w>>20)&0xF), s*xv5, b*xv5);\n"
    "                acc[t][row] += fma(float((w>>24)&0xF), s*xv6, b*xv6);\n"
    "                acc[t][row] += fma(float((w>>28)&0xF), s*xv7, b*xv7);\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    // Reduce: mirror v2 exactly (zero-padded simd_sum), per (t,row).\n"
    "    // shmem layout: [t][row][NW] floats.\n"
    "    for (uint t = 0; t < ntok; t++) {\n"
    "        for (short row = 0; row < NR0; row++) {\n"
    "            threadgroup float *sf = shmem + (t * NR0 + row) * NW;\n"
    "            if (sgitg == 0) sf[tiisg] = 0.0f;\n"
    "            acc[t][row] = simd_sum(acc[t][row]);\n"
    "        }\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint t = 0; t < ntok; t++) {\n"
    "        for (short row = 0; row < NR0; row++) {\n"
    "            threadgroup float *sf = shmem + (t * NR0 + row) * NW;\n"
    "            if (tiisg == 0) sf[sgitg] = acc[t][row];\n"
    "        }\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint t = 0; t < ntok; t++) {\n"
    "        for (short row = 0; row < NR0; row++) {\n"
    "            const int d = row0 + row;\n"
    "            if (d >= (int)out_dim) continue;\n"
    "            threadgroup float *sf = shmem + (t * NR0 + row) * NW;\n"
    "            float tot = simd_sum(sf[tiisg]);\n"
    "            if (tiisg == 0 && sgitg == 0) out[t * out_dim + d] = (bfloat)tot;\n"
    "        }\n"
    "    }\n"
    "}\n";
}

typedef struct {
    uint out_dim, in_dim, gs;
    const char *name;
} Shape;

int main(int argc, char **argv) {
    const char *metal_path = argc > 1 ? argv[1] : "src/models/moe_kernel.metal";
    Shape shapes[] = {
        {32768, 1024, 64, "wq_b"},   // 16MB packed — biggest attention weight
        { 1024, 4096, 64, "wq_a"},   // 2MB packed
        {  576, 4096, 64, "wkv"},    // 1.2MB packed
    };

    NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:metal_path]
                                              encoding:NSUTF8StringEncoding error:nil];
    if (!src) { fprintf(stderr, "cannot read %s\n", metal_path); return 1; }
    src = [src stringByAppendingString:mtKernelSrc()];
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { fprintf(stderr, "Metal compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }
    id<MTLFunction> f_v2 = [lib newFunctionWithName:@"dequant_matvec_affine_bf16in_bf16out_v2"];
    id<MTLFunction> f_mt = [lib newFunctionWithName:@"dequant_gemm_affine_bf16_mt"];
    if (!f_v2 || !f_mt) { fprintf(stderr, "kernel lookup failed\n"); return 1; }
    id<MTLComputePipelineState> p_v2 = [dev newComputePipelineStateWithFunction:f_v2 error:&err];
    id<MTLComputePipelineState> p_mt = [dev newComputePipelineStateWithFunction:f_mt error:&err];
    id<MTLCommandQueue> q = [dev newCommandQueue];

    srand(42);
    for (int si = 0; si < 3; si++) {
        Shape S = shapes[si];
        uint num_groups = S.in_dim / S.gs;
        size_t w_bytes   = (size_t)S.out_dim * (S.in_dim / 8) * 4;
        size_t sb_bytes  = (size_t)S.out_dim * num_groups * 4;
        uint32_t *W  = malloc(w_bytes);
        float *sc = malloc(sb_bytes), *bi = malloc(sb_bytes);
        for (size_t i = 0; i < w_bytes / 4; i++) W[i] = (uint32_t)rand();
        for (size_t i = 0; i < (size_t)S.out_dim * num_groups; i++) {
            sc[i] = (float)rand() / RAND_MAX * 0.02f + 0.001f;
            bi[i] = (float)rand() / RAND_MAX * 0.1f - 0.05f;
        }
        uint16_t *xb = malloc((size_t)MAX_N * S.in_dim * 2);
        for (size_t i = 0; i < (size_t)MAX_N * S.in_dim; i++)
            xb[i] = f2bf((float)rand() / RAND_MAX * 2 - 1);

        id<MTLBuffer> bW  = [dev newBufferWithBytes:W length:w_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsc = [dev newBufferWithBytes:sc length:sb_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bbi = [dev newBufferWithBytes:bi length:sb_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bxb = [dev newBufferWithBytes:xb length:(size_t)MAX_N * S.in_dim * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo_v2 = [dev newBufferWithLength:(size_t)MAX_N * S.out_dim * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo_mt = [dev newBufferWithLength:(size_t)MAX_N * S.out_dim * 2 options:MTLResourceStorageModeShared];

        for (uint N = 1; N <= MAX_N; N *= 4) {
            // ---- correctness: v2 per token vs MT ----
            for (uint t = 0; t < N; t++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p_v2];
                [e setBuffer:bW offset:0 atIndex:0];
                [e setBuffer:bsc offset:0 atIndex:1];
                [e setBuffer:bbi offset:0 atIndex:2];
                [e setBuffer:bxb offset:(size_t)t * S.in_dim * 2 atIndex:3];
                [e setBuffer:bo_v2 offset:(size_t)t * S.out_dim * 2 atIndex:4];
                [e setBytes:&S.out_dim length:4 atIndex:5];
                [e setBytes:&S.in_dim length:4 atIndex:6];
                [e setBytes:&S.gs length:4 atIndex:7];
                [e setThreadgroupMemoryLength:256 atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake((S.out_dim + 1) / 2, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                [e endEncoding];
                [cb commit]; [cb waitUntilCompleted];
            }
            {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p_mt];
                [e setBuffer:bW offset:0 atIndex:0];
                [e setBuffer:bsc offset:0 atIndex:1];
                [e setBuffer:bbi offset:0 atIndex:2];
                [e setBuffer:bxb offset:0 atIndex:3];
                [e setBuffer:bo_mt offset:0 atIndex:4];
                [e setBytes:&S.out_dim length:4 atIndex:5];
                [e setBytes:&S.in_dim length:4 atIndex:6];
                [e setBytes:&S.gs length:4 atIndex:7];
                [e setBytes:&N length:4 atIndex:8];
                [e setThreadgroupMemoryLength:4096 atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake((S.out_dim + 1) / 2, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                [e endEncoding];
                [cb commit]; [cb waitUntilCompleted];
            }
            uint16_t *o_v2 = (uint16_t *)[bo_v2 contents];
            uint16_t *o_mt = (uint16_t *)[bo_mt contents];
            int exact = 0, bad = 0; double max_abs = 0;
            for (size_t i = 0; i < (size_t)N * S.out_dim; i++) {
                double d = fabs((double)bf2f(o_v2[i]) - (double)bf2f(o_mt[i]));
                if (d > max_abs) max_abs = d;
                if (o_v2[i] == o_mt[i]) exact++;
                if (d > 1e-2) bad++;
            }
            // ---- timing: 20 iterations in one CB, GPU timestamps ----
            const int REPS = 20;
            double t_v2, t_mt;
            {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                for (int r = 0; r < REPS; r++) for (uint t = 0; t < N; t++) {
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    [e setComputePipelineState:p_v2];
                    [e setBuffer:bW offset:0 atIndex:0];
                    [e setBuffer:bsc offset:0 atIndex:1];
                    [e setBuffer:bbi offset:0 atIndex:2];
                    [e setBuffer:bxb offset:(size_t)t * S.in_dim * 2 atIndex:3];
                    [e setBuffer:bo_v2 offset:(size_t)t * S.out_dim * 2 atIndex:4];
                    [e setBytes:&S.out_dim length:4 atIndex:5];
                    [e setBytes:&S.in_dim length:4 atIndex:6];
                    [e setBytes:&S.gs length:4 atIndex:7];
                    [e setThreadgroupMemoryLength:256 atIndex:0];
                    [e dispatchThreadgroups:MTLSizeMake((S.out_dim + 1) / 2, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                    [e endEncoding];
                }
                [cb commit]; [cb waitUntilCompleted];
                t_v2 = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3 / REPS;  // ms per N tokens
            }
            {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                for (int r = 0; r < REPS; r++) {
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    [e setComputePipelineState:p_mt];
                    [e setBuffer:bW offset:0 atIndex:0];
                    [e setBuffer:bsc offset:0 atIndex:1];
                    [e setBuffer:bbi offset:0 atIndex:2];
                    [e setBuffer:bxb offset:0 atIndex:3];
                    [e setBuffer:bo_mt offset:0 atIndex:4];
                    [e setBytes:&S.out_dim length:4 atIndex:5];
                    [e setBytes:&S.in_dim length:4 atIndex:6];
                    [e setBytes:&S.gs length:4 atIndex:7];
                    [e setBytes:&N length:4 atIndex:8];
                    [e setThreadgroupMemoryLength:4096 atIndex:0];
                    [e dispatchThreadgroups:MTLSizeMake((S.out_dim + 1) / 2, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                    [e endEncoding];
                }
                [cb commit]; [cb waitUntilCompleted];
                t_mt = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3 / REPS;  // ms per N tokens
            }
            double gbs_v2 = (double)w_bytes / (t_v2 / N) / 1e6;   // GB/s per single-token pass
            double gbs_mt = (double)w_bytes / t_mt / 1e6;
            printf("%s N=%2d: v2 %7.3fms (%6.3fms/tok, %5.1f GB/s) | mt %7.3fms (%6.3fms/tok, %5.1f GB/s) | speedup %5.2fx | exact %d/%zu bad %d %s\n",
                   S.name, N, t_v2, t_v2 / N, gbs_v2, t_mt, t_mt / N, gbs_mt,
                   (t_v2 / N) / (t_mt / N), exact, (size_t)N * S.out_dim, bad,
                   bad ? "*** FAIL ***" : "OK");
            if (bad) g_fail = 1;
        }
        free(W); free(sc); free(bi); free(xb);
    }
    return g_fail;
}
