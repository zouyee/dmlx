// Standalone-only screen for test-owned mhc_pre dense-dot mappings.
// This harness does not compile or concatenate production Metal source.
#import <CommonCrypto/CommonDigest.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <math.h>
#import <stdint.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <unistd.h>

#define ROWS 24u
#define WIDTH 16384u
#define THREADS 256u
#define ARMS 3
#define WARMUPS 10
#define REPS 21
#define EVENTS ((WARMUPS + REPS) * ARMS)
#define ABS_TOL 0.05f

typedef struct { bool measured; int iteration; int step; int arm; uint64_t duration_ns; } Event;
static const char *arm_names[ARMS] = {"serial_one_lane_per_row", "simdgroup_per_row_wave8", "eight_lane_per_row"};
static uint32_t rng_state = 0x12345678u;
static float nextf(void) { rng_state = rng_state * 1664525u + 1013904223u; return ((rng_state >> 8) / 16777216.0f - 0.5f) * 0.1f; }

static bool file_sha256(NSString *path, char hex[CC_SHA256_DIGEST_LENGTH * 2 + 1]) {
    NSData *data = [NSData dataWithContentsOfFile:path]; if (!data) return false;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) sprintf(hex + i * 2, "%02x", digest[i]);
    hex[CC_SHA256_DIGEST_LENGTH * 2] = 0; return true;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> d, id<MTLLibrary> l, NSString *name) {
    NSError *err = nil; id<MTLFunction> f = [l newFunctionWithName:name];
    id<MTLComputePipelineState> p = f ? [d newComputePipelineStateWithFunction:f error:&err] : nil;
    if (!p) { fprintf(stderr, "pipeline %s failed: %s\n", name.UTF8String, err.localizedDescription.UTF8String); exit(1); }
    return p;
}

static uint64_t run_one(id<MTLDevice> d, id<MTLCommandQueue> q, id<MTLComputePipelineState> p,
                        id<MTLCounterSet> cs, id<MTLBuffer> weights, id<MTLBuffer> residual,
                        id<MTLBuffer> output, float norm) {
    MTLCounterSampleBufferDescriptor *sd = [MTLCounterSampleBufferDescriptor new];
    sd.counterSet = cs; sd.sampleCount = 2; sd.storageMode = MTLStorageModeShared;
    NSError *err = nil; id<MTLCounterSampleBuffer> sb = [d newCounterSampleBufferWithDescriptor:sd error:&err];
    if (!sb) { fprintf(stderr, "sample buffer failed: %s\n", err.localizedDescription.UTF8String); exit(1); }
    MTLComputePassDescriptor *pd = [MTLComputePassDescriptor computePassDescriptor]; id a = pd.sampleBufferAttachments[0];
    [a setSampleBuffer:sb]; [a setStartOfEncoderSampleIndex:0]; [a setEndOfEncoderSampleIndex:1];
    id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDescriptor:pd];
    [e setComputePipelineState:p]; [e setBuffer:weights offset:0 atIndex:0]; [e setBuffer:residual offset:0 atIndex:1];
    [e setBytes:&norm length:sizeof(norm) atIndex:2]; [e setBuffer:output offset:0 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(THREADS,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) { fprintf(stderr, "command failed: %s\n", cb.error.localizedDescription.UTF8String); exit(1); }
    NSData *data = [sb resolveCounterRange:NSMakeRange(0,2)];
    if (data.length != 2 * sizeof(MTLCounterResultTimestamp)) { fputs("invalid timestamp resolve\n", stderr); exit(1); }
    const MTLCounterResultTimestamp *r = data.bytes;
    if (!r[0].timestamp || r[1].timestamp <= r[0].timestamp) { fputs("invalid timestamp pair\n", stderr); exit(1); }
    return r[1].timestamp - r[0].timestamp;
}

static bool check_output(const float *actual, const float *reference) {
    for (uint32_t i = 0; i < ROWS; ++i)
        if (!isfinite(actual[i]) || fabsf(actual[i] - reference[i]) > ABS_TOL) return false;
    return true;
}

int main(int argc, const char **argv) { @autoreleasepool {
    if (argc != 2) { fputs("usage: mhc_pre_parallel_gpu_test PROCESS_ORDINAL(1..3)\n", stderr); return 2; }
    char *end = NULL; long ordinal = strtol(argv[1], &end, 10);
    if (!end || *end || ordinal < 1 || ordinal > 3) { fputs("invalid process ordinal\n", stderr); return 2; }
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d || ![d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) { fputs("unsupported stage-boundary timestamps\n", stderr); return 2; }
    id<MTLCounterSet> cs = nil; for (id<MTLCounterSet> s in d.counterSets) if ([s.name caseInsensitiveCompare:@"timestamp"] == NSOrderedSame) cs = s;
    if (!cs) { fputs("missing timestamp counter set\n", stderr); return 2; }
    NSError *err = nil; NSString *source = [NSString stringWithContentsOfFile:@"scripts/mhc_pre_parallel_kernels.metal" encoding:NSUTF8StringEncoding error:&err];
    if (!source) { fprintf(stderr, "read harness Metal failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    MTLCompileOptions *opts = [MTLCompileOptions new]; opts.languageVersion = MTLLanguageVersion3_1;
    id<MTLLibrary> lib = [d newLibraryWithSource:source options:opts error:&err];
    if (!lib) { fprintf(stderr, "compile harness Metal failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    NSString *names[ARMS] = {@"mhc_dot_serial_one_lane_per_row", @"mhc_dot_simdgroup_per_row_wave8", @"mhc_dot_eight_lane_per_row"};
    id<MTLComputePipelineState> p[ARMS]; for (int i = 0; i < ARMS; ++i) p[i] = make_pipeline(d, lib, names[i]);
    for (int i = 0; i < ARMS; ++i) if (p[i].maxTotalThreadsPerThreadgroup < THREADS) { fputs("pipeline lacks 256-thread capacity\n", stderr); return 2; }
    if (p[1].threadExecutionWidth != 32) { fprintf(stderr, "wave8 requires SIMD width 32, got %lu\n", (unsigned long)p[1].threadExecutionWidth); return 2; }
    char harness_sha[65], metal_sha[65], production_sha[65];
    if (!file_sha256(@"scripts/mhc_pre_parallel_gpu_test.m", harness_sha) ||
        !file_sha256(@"scripts/mhc_pre_parallel_kernels.metal", metal_sha) ||
        !file_sha256(@"src/models/moe_kernel.metal", production_sha)) { fputs("source hash failed\n", stderr); return 1; }
    id<MTLCommandQueue> q = [d newCommandQueue];
    id<MTLBuffer> weights = [d newBufferWithLength:(size_t)ROWS * WIDTH * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> residual = [d newBufferWithLength:(size_t)WIDTH * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> output[ARMS]; for (int i = 0; i < ARMS; ++i) output[i] = [d newBufferWithLength:ROWS * sizeof(float) options:MTLResourceStorageModeShared];
    float *w = weights.contents, *r = residual.contents, reference[ROWS];
    for (size_t i = 0; i < (size_t)ROWS * WIDTH; ++i) w[i] = nextf();
    double ss = 0.0; for (uint32_t i = 0; i < WIDTH; ++i) { r[i] = nextf(); ss += (double)r[i] * r[i]; }
    float norm = 1.0f / sqrtf((float)(ss / WIDTH) + 1e-6f);
    for (uint32_t row = 0; row < ROWS; ++row) { float acc = 0.0f; for (uint32_t i = 0; i < WIDTH; ++i) acc += w[(size_t)row * WIDTH + i] * r[i]; reference[row] = acc * norm; }
    Event events[EVENTS]; int event_count = 0;
    for (int iteration = 0; iteration < WARMUPS; ++iteration) for (int step = 0; step < ARMS; ++step) {
        int arm = (int)((ordinal - 1 + iteration + step) % ARMS); (void)run_one(d,q,p[arm],cs,weights,residual,output[arm],norm);
        events[event_count++] = (Event){false,iteration,step,arm,0};
    }
    float first[ARMS][ROWS]; bool saved[ARMS] = {false,false,false};
    for (int iteration = 0; iteration < REPS; ++iteration) for (int step = 0; step < ARMS; ++step) {
        int arm = (int)((ordinal - 1 + iteration + step) % ARMS); uint64_t duration = run_one(d,q,p[arm],cs,weights,residual,output[arm],norm);
        float *actual = output[arm].contents;
        if (!check_output(actual, reference)) { fprintf(stderr, "oracle mismatch arm=%s iteration=%d\n", arm_names[arm], iteration); return 1; }
        if (!saved[arm]) { memcpy(first[arm], actual, sizeof(first[arm])); saved[arm] = true; }
        else if (memcmp(first[arm], actual, sizeof(first[arm])) != 0) { fprintf(stderr, "nonrepeatable arm=%s iteration=%d\n", arm_names[arm], iteration); return 1; }
        events[event_count++] = (Event){true,iteration,step,arm,duration};
    }
    printf("{\"schema_version\":1,\"process_ordinal\":%ld,\"process_id\":%d,\"oracle_status\":\"PASS\",\"harness_source_sha256\":\"%s\",\"harness_metal_sha256\":\"%s\",\"production_kernel_sha256\":\"%s\",\"topology\":\"1cb/1encoder/1dispatch/1commit/1wait/1x256\",\"events\":[", ordinal, getpid(), harness_sha, metal_sha, production_sha);
    for (int i = 0; i < event_count; ++i) { Event e = events[i];
        printf("%s{\"phase\":\"%s\",\"process_ordinal\":%ld,\"iteration\":%d,\"step\":%d,\"arm\":\"%s\",\"duration_ns\":", i ? "," : "", e.measured ? "measured" : "warmup", ordinal, e.iteration, e.step, arm_names[e.arm]);
        if (e.measured) printf("%llu}", (unsigned long long)e.duration_ns); else printf("null}");
    }
    puts("]}"); return 0;
}}
