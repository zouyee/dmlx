// test_affine_kernel.m — Standalone test for affine 4-bit MoE kernels
#import <Metal/Metal.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>

// Read kernel source from file
static char* read_file(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); return NULL; }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(*len);
    fread(buf, 1, *len, f);
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <packed_experts_dir>\n", argv[0]);
        return 1;
    }

    const char* packed_dir = argv[1];

    // Read kernel source
    size_t kernel_len;
    char* kernel_src = read_file("src/models/moe_kernel.metal", &kernel_len);
    if (!kernel_src) return 1;

    // Set up Metal
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    NSString* src = [[NSString alloc] initWithBytes:kernel_src length:kernel_len encoding:NSUTF8StringEncoding];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { fprintf(stderr, "Metal compile error: %s\n", [[err localizedDescription] UTF8String]); return 1; }

    id<MTLComputePipelineState> gate_up = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_gate_up_swiglu_v2_affine"] error:&err];
    id<MTLComputePipelineState> down = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_4bit_affine"] error:&err];
    if (!gate_up || !down) { fprintf(stderr, "Pipeline creation failed\n"); return 1; }

    printf("Metal initialized, kernels compiled.\n");

    // Load packed file for layer 0
    char path[1024];
    snprintf(path, sizeof(path), "%s/layer_00.bin", packed_dir);

    const int EXPERT_SIZE = 14155776;
    const int DIM = 4096;
    const int INTERMEDIATE = 2048;

    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); return 1; }
    char* expert_data = malloc(EXPERT_SIZE);
    fread(expert_data, 1, EXPERT_SIZE, f);
    fclose(f);

    // Read first expert (expert 0)
    char* base = expert_data;

    // Create Metal buffers
    const int GATE_W_OFF = 0, GATE_S_OFF = 4194304, GATE_B_OFF = 4456448;
    const int UP_W_OFF = 4718592, UP_S_OFF = 8912896, UP_B_OFF = 9175040;
    const int DOWN_W_OFF = 9437184, DOWN_S_OFF = 13631488, DOWN_B_OFF = 13893632;

    id<MTLBuffer> gw_buf = [dev newBufferWithBytesNoCopy:base+GATE_W_OFF length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> gs_buf = [dev newBufferWithBytesNoCopy:base+GATE_S_OFF length:262144 options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> gb_buf = [dev newBufferWithBytesNoCopy:base+GATE_B_OFF length:262144 options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> uw_buf = [dev newBufferWithBytesNoCopy:base+UP_W_OFF length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> us_buf = [dev newBufferWithBytesNoCopy:base+UP_S_OFF length:262144 options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> ub_buf = [dev newBufferWithBytesNoCopy:base+UP_B_OFF length:262144 options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> x_buf = [dev newBufferWithLength:DIM*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> mid_buf = [dev newBufferWithLength:INTERMEDIATE*sizeof(float) options:MTLResourceStorageModeShared];

    // Fill x with ones
    float* x_data = (float*)[x_buf contents];
    for(int i=0;i<DIM;i++) x_data[i]=1.0f;

    // Run gate-up kernel
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:gate_up];
    [enc setBuffer:gw_buf offset:0 atIndex:0];
    [enc setBuffer:gs_buf offset:0 atIndex:1];
    [enc setBuffer:gb_buf offset:0 atIndex:2];
    [enc setBuffer:uw_buf offset:0 atIndex:3];
    [enc setBuffer:us_buf offset:0 atIndex:4];
    [enc setBuffer:ub_buf offset:0 atIndex:5];
    [enc setBuffer:x_buf offset:0 atIndex:6];
    [enc setBuffer:mid_buf offset:0 atIndex:7];
    uint od=INTERMEDIATE, id_=DIM, gs=64;
    [enc setBytes:&od length:4 atIndex:8];
    [enc setBytes:&id_ length:4 atIndex:9];
    [enc setBytes:&gs length:4 atIndex:10];
    [enc setThreadgroupMemoryLength:512 atIndex:0];
    uint ntg = (INTERMEDIATE+1)/2;
    [enc dispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
    [enc endEncoding];

    // Run down-proj kernel
    id<MTLBuffer> dw_buf = [dev newBufferWithBytesNoCopy:base+DOWN_W_OFF length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> ds_buf = [dev newBufferWithBytesNoCopy:base+DOWN_S_OFF length:262144 options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> db_buf = [dev newBufferWithBytesNoCopy:base+DOWN_B_OFF length:262144 options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> out_buf = [dev newBufferWithLength:DIM*sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLComputeCommandEncoder> enc2 = [cb computeCommandEncoder];
    [enc2 setComputePipelineState:down];
    [enc2 setBuffer:dw_buf offset:0 atIndex:0];
    [enc2 setBuffer:ds_buf offset:0 atIndex:1];
    [enc2 setBuffer:db_buf offset:0 atIndex:2];
    [enc2 setBuffer:mid_buf offset:0 atIndex:3];
    [enc2 setBuffer:out_buf offset:0 atIndex:4];
    od=DIM; id_=INTERMEDIATE; gs=64;
    [enc2 setBytes:&od length:4 atIndex:5];
    [enc2 setBytes:&id_ length:4 atIndex:6];
    [enc2 setBytes:&gs length:4 atIndex:7];
    [enc2 setThreadgroupMemoryLength:256 atIndex:0];
    uint d_ntg = (DIM+1)/2;
    [enc2 dispatchThreadgroups:MTLSizeMake(d_ntg,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
    [enc2 endEncoding];

    [cb commit];
    [cb waitUntilCompleted];

    float* result = (float*)[out_buf contents];
    printf("Gate-up output (first 5 values): %f %f %f %f %f\n",
           result[0], result[1], result[2], result[3], result[4]);

    // Compare with CPU: dequantize the same data and compute manually
    // (would need original MXFP4 weights for comparison — skip for now)

    free(expert_data);
    return 0;
}