// Minimal C Metal wrapper — avoids Zig ObjC interop.
// Compiles Metal shader at runtime, dispatches MoE kernels.
#include <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_gate_up_swiglu;
static id<MTLComputePipelineState> g_dequant_matvec;
static id<MTLComputePipelineState> g_moe_combine;
static int g_initialized = 0;

int moe_metal_init_from_source(const char *source, unsigned long source_len) {
    if (g_initialized) return 0;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) { fprintf(stderr, "Metal: no device\n"); return -1; }
    g_queue = [g_dev newCommandQueue];
    if (!g_queue) { fprintf(stderr, "Metal: no queue\n"); return -1; }

    NSString *src = [[NSString alloc] initWithBytes:source length:source_len encoding:NSUTF8StringEncoding];
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    NSError *err = nil;
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { fprintf(stderr, "Metal: compile: %s\n", [[err localizedDescription] UTF8String]); return -1; }

    g_gate_up_swiglu = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_gate_up_swiglu"] error:&err];
    g_dequant_matvec = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_4bit"] error:&err];
    g_moe_combine = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_combine"] error:&err];
    if (!g_gate_up_swiglu || !g_dequant_matvec || !g_moe_combine) { fprintf(stderr, "Metal: pipeline failed\n"); return -1; }
    g_initialized = 1;
    fprintf(stderr, "Metal MoE: initialized\n");
    return 0;
}

int moe_metal_forward(
    const void *const *expert_ptrs, const float *hidden, const float *scores,
    float *output, int K, int hidden_dim, int intermediate_dim, int group_size
) {
    if (!g_initialized) return -1;
    const int gate_w_off = 0, gate_s_off = 4194304;
    const int up_w_off = 4456448, up_s_off = 8650752;
    const int down_w_off = 8912896, down_s_off = 13107200;

    // Intermediate: [K * intermediate_dim] float for gate+up+SwiGLU output
    int mid_bytes = K * intermediate_dim * sizeof(float);
    id<MTLBuffer> mid_buf = [g_dev newBufferWithLength:mid_bytes options:MTLResourceStorageModeShared];
    // Expert outputs: [K * hidden_dim]
    id<MTLBuffer> out_buf = [g_dev newBufferWithLength:K * hidden_dim * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hidden_buf = [g_dev newBufferWithBytesNoCopy:(void*)hidden length:hidden_dim*sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> scores_buf = [g_dev newBufferWithBytesNoCopy:(void*)scores length:K*sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> output_buf = [g_dev newBufferWithLength:hidden_dim*sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    uint od, id_, gs = group_size;

    // Step 1: fused_gate_up_swiglu per expert
    for (int k = 0; k < K; k++) {
        const char *base = (const char *)expert_ptrs[k];
        id gw = [g_dev newBufferWithBytesNoCopy:(void*)(base+gate_w_off) length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
        id gs_b = [g_dev newBufferWithBytesNoCopy:(void*)(base+gate_s_off) length:262144 options:MTLResourceStorageModeShared deallocator:nil];
        id uw = [g_dev newBufferWithBytesNoCopy:(void*)(base+up_w_off) length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
        id us_b = [g_dev newBufferWithBytesNoCopy:(void*)(base+up_s_off) length:262144 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_gate_up_swiglu];
        [enc setBuffer:gw offset:0 atIndex:0];  [enc setBuffer:gs_b offset:0 atIndex:1];
        [enc setBuffer:uw offset:0 atIndex:2];  [enc setBuffer:us_b offset:0 atIndex:3];
        [enc setBuffer:hidden_buf offset:0 atIndex:4];
        [enc setBuffer:mid_buf offset:k*intermediate_dim*sizeof(float) atIndex:5];
        od=intermediate_dim; id_=hidden_dim;
        [enc setBytes:&od length:4 atIndex:6]; [enc setBytes:&id_ length:4 atIndex:7]; [enc setBytes:&gs length:4 atIndex:8];
        [enc dispatchThreads:MTLSizeMake(intermediate_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }

    // Step 2: dequant_matvec_4bit (down_proj) per expert
    for (int k = 0; k < K; k++) {
        const char *base = (const char *)expert_ptrs[k];
        id dw = [g_dev newBufferWithBytesNoCopy:(void*)(base+down_w_off) length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
        id ds_b = [g_dev newBufferWithBytesNoCopy:(void*)(base+down_s_off) length:262144 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_dequant_matvec];
        [enc setBuffer:dw offset:0 atIndex:0]; [enc setBuffer:ds_b offset:0 atIndex:1];
        [enc setBuffer:mid_buf offset:k*intermediate_dim*sizeof(float) atIndex:2];
        [enc setBuffer:out_buf offset:k*hidden_dim*sizeof(float) atIndex:3];
        od=hidden_dim; id_=intermediate_dim;
        [enc setBytes:&od length:4 atIndex:4]; [enc setBytes:&id_ length:4 atIndex:5]; [enc setBytes:&gs length:4 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(hidden_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }

    // Step 3: moe_combine
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_moe_combine];
        [enc setBuffer:out_buf offset:0 atIndex:0]; [enc setBuffer:scores_buf offset:0 atIndex:1];
        [enc setBuffer:hidden_buf offset:0 atIndex:2]; [enc setBuffer:output_buf offset:0 atIndex:3];
        uint kv=K, hd=hidden_dim;
        [enc setBytes:&kv length:4 atIndex:4]; [enc setBytes:&hd length:4 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((hidden_dim+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }

    [cb commit];
    [cb waitUntilCompleted];
    memcpy(output, [output_buf contents], hidden_dim * sizeof(float));
    return 0;
}
