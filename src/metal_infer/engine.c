// Metal inference engine — per-token forward pass with flash-moe pipeline.
// Phase 1: MoE forward only (attention handled by MLX for now).
// Phase 2: Add attention MatVecs on Metal.
// Phase 3: Full flash-moe alignment.
//
// See docs/analysis/flash-moe-alignment-plan.md for architecture details.
#include "engine.h"
#include "mla_attention.h"
#include "mhc.h"
#include <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dispatch/dispatch.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>

// ============================================================================
// Metal setup
// ============================================================================

static int init_metal(MoEInferEngine *eng, const char *kernel_src, unsigned long kernel_src_len) {
    setvbuf(stderr, NULL, _IONBF, 0); // unbuffered so crash diagnostics are not lost
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "Metal: no device\n"); return -1; }
    eng->device = (void *)dev;
    eng->queue = (void *)[dev newCommandQueue];

    NSString *src_str = [[NSString alloc]
        initWithBytes:kernel_src
        length:kernel_src_len
        encoding:NSUTF8StringEncoding];

    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src_str options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "Metal compile: %s\n", [[err localizedDescription] UTF8String]);
        return -1;
    }

    id<MTLDevice> d = (id<MTLDevice>)dev;
    eng->pipe_gate_up_swiglu = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_gate_up_swiglu"] error:&err]);
    eng->pipe_dequant_matvec = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_4bit"] error:&err]);
    eng->pipe_moe_combine    = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_combine"] error:&err]);
    eng->pipe_rms_norm_sum_sq = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_sum_sq"] error:&err]);
    eng->pipe_rms_norm_apply = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_apply"] error:&err]);
    eng->pipe_matvec = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec_f32"] error:&err]);
    // S7: MLA attention pipelines
    eng->pipe_dequant_matvec_affine = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine"] error:&err]);
    eng->pipe_rms_norm_rows = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows"] error:&err]);
    eng->pipe_rope_tail = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rope_tail_interleaved"] error:&err]);
    eng->pipe_mla_sdpa = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_decode"] error:&err]);
    // bf16-output variants for Q chain (matches MLX bf16 intermediate precision)
    eng->pipe_dequant_matvec_affine_bf16 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16out"] error:&err]);
    eng->pipe_rms_norm_rows_bf16 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows_bf16out"] error:&err]);
    eng->pipe_bf16_to_f32 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bf16_to_f32"] error:&err]);
    eng->pipe_mhc_pre_gpu = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mhc_pre_gpu"] error:&err]);
    eng->pipe_f32_to_bf16 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"f32_to_bf16"] error:&err]);
    eng->pipe_dequant_matvec_affine_bf16in_f32out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_f32out"] error:&err]);
    eng->pipe_dequant_matvec_affine_bf16in_bf16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_bf16out"] error:&err]);
    eng->pipe_rms_norm_rows_bf16in_bf16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows_bf16in_bf16out"] error:&err]);
    eng->pipe_rope_tail_bf16 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rope_tail_interleaved_bf16"] error:&err]);
    eng->pipe_matvec_f32_bf16in = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec_f32_bf16in"] error:&err]);
    if (!eng->pipe_gate_up_swiglu || !eng->pipe_dequant_matvec || !eng->pipe_moe_combine) {
        fprintf(stderr, "Metal: pipeline state failed\n");
        return -1;
    }

    // Allocate Metal buffers
    int buf_size = DIM * sizeof(float);
    int mid_size = INTERMEDIATE * sizeof(float);
    eng->buf_hidden    = (void *)[d newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    eng->buf_h_mid     = (void *)[d newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    eng->buf_normed    = (void *)[d newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    eng->buf_attn_out  = (void *)[d newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    eng->buf_routing_scores = (void *)[d newBufferWithLength:N_EXPERTS * sizeof(float) options:MTLResourceStorageModeShared];
    eng->buf_norm_sum_sq    = (void *)[d newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    for (int k = 0; k < N_ACTIVE; k++) {
        eng->buf_expert_mid[k] = (void *)[d newBufferWithLength:mid_size options:MTLResourceStorageModeShared];
        eng->buf_expert_out[k] = (void *)[d newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    }

    // 2MB-aligned expert I/O buffers
    for (int k = 0; k < N_ACTIVE; k++) {
        posix_memalign((void**)&eng->expert_buf[k], 2*1024*1024, EXPERT_SIZE);
        posix_memalign((void**)&eng->expert_buf_pred[k], 2*1024*1024, EXPERT_SIZE);
    }

    fprintf(stderr, "Metal engine: initialized\n");
    return 0;
}

// ============================================================================
// Expert I/O — parallel pread with persistent threads (flash-moe pattern)
// ============================================================================

#define NUM_IO_THREADS 6

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t work_ready, work_done;
    int fd[NUM_IO_THREADS];
    void *buf[NUM_IO_THREADS];
    size_t size;
    off_t offset[NUM_IO_THREADS];
    int num_tasks;
    int tasks_done;
    int generation;
    bool shutdown;
    pthread_t threads[NUM_IO_THREADS];
} IOPool;

static void *io_worker(void *arg) {
    IOPool *pool = (IOPool *)arg;
    int last_gen = -1;
    while (1) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->generation == last_gen && !pool->shutdown) {
            pthread_cond_wait(&pool->work_ready, &pool->mutex);
        }
        if (pool->shutdown) { pthread_mutex_unlock(&pool->mutex); break; }
        last_gen = pool->generation;
        int tid = -1;
        // Find an unclaimed task
        for (int i = 0; i < pool->num_tasks; i++) {
            if (pool->fd[i] >= 0) { tid = i; break; }
        }
        pthread_mutex_unlock(&pool->mutex);
        if (tid < 0) continue;

        ssize_t n = pread(pool->fd[tid], pool->buf[tid], pool->size, pool->offset[tid]);

        pthread_mutex_lock(&pool->mutex);
        pool->fd[tid] = -1;  // mark done
        pool->tasks_done++;
        if (pool->tasks_done == pool->num_tasks) {
            pthread_cond_signal(&pool->work_done);
        }
        pthread_mutex_unlock(&pool->mutex);
    }
    return NULL;
}

static void io_pool_init(IOPool *pool) {
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->work_ready, NULL);
    pthread_cond_init(&pool->work_done, NULL);
    pool->num_tasks = 0;
    pool->tasks_done = 0;
    pool->generation = 0;
    pool->shutdown = false;
    for (int i = 0; i < NUM_IO_THREADS; i++) {
        pthread_create(&pool->threads[i], NULL, io_worker, pool);
    }
}

// Dispatch K parallel preads. Blocks until all complete.
static void io_pool_dispatch(IOPool *pool, int layer_fd, int *expert_ids, int K,
                              uint8_t *buffers[6]) {
    pthread_mutex_lock(&pool->mutex);
    for (int k = 0; k < K; k++) {
        pool->fd[k] = layer_fd;
        pool->buf[k] = buffers[k];
        pool->size = EXPERT_SIZE;
        pool->offset[k] = (off_t)expert_ids[k] * EXPERT_SIZE;
    }
    pool->num_tasks = K;
    pool->tasks_done = 0;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_ready);

    while (pool->tasks_done < K) {
        pthread_cond_wait(&pool->work_done, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
}

// ============================================================================
// Temporal expert prediction
// ============================================================================

static void predictor_record(ExpertPredictor *pred, int layer, int *experts, int K) {
    for (int k = 0; k < K && k < N_ACTIVE; k++) {
        pred->experts[layer][k] = experts[k];
    }
}

static void predictor_prefetch_start(ExpertPredictor *pred, int layer_fd,
                                      int layer, uint8_t *buffers[6]) {
    // TODO: async prefetch into prediction buffers
    (void)pred; (void)layer_fd; (void)layer; (void)buffers;
    // Phase 2: implement GCD async pread for prediction
}

// ============================================================================
// MoE forward for one layer (CMD3 — expert matmuls + combine)
// ============================================================================

static int moe_forward_layer(MoEInferEngine *eng, int layer_idx,
                              uint8_t *expert_bufs[6], int *expert_ids,
                              float *expert_weights, int K) {
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBuffer];
    uint od, id_, gs = 32;
    const int gw_off = GATE_W_OFF, gs_off = GATE_S_OFF;
    const int uw_off = UP_W_OFF, us_off = UP_S_OFF;
    const int dw_off = DOWN_W_OFF, ds_off = DOWN_S_OFF;

    // Step 1: fused_gate_up_swiglu per expert [2048, 4096], group_size=32
    for (int k = 0; k < K; k++) {
        char *base = (char *)expert_bufs[k];
        id gw   = [d newBufferWithBytesNoCopy:base+gw_off length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
        id gs_b = [d newBufferWithBytesNoCopy:base+gs_off length:262144 options:MTLResourceStorageModeShared deallocator:nil];
        id uw   = [d newBufferWithBytesNoCopy:base+uw_off length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
        id us_b = [d newBufferWithBytesNoCopy:base+us_off length:262144 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:eng->pipe_gate_up_swiglu];
        [enc setBuffer:gw offset:0 atIndex:0];   [enc setBuffer:gs_b offset:0 atIndex:1];
        [enc setBuffer:uw offset:0 atIndex:2];   [enc setBuffer:us_b offset:0 atIndex:3];
        [enc setBuffer:eng->buf_normed offset:0 atIndex:4];
        [enc setBuffer:eng->buf_expert_mid[k] offset:0 atIndex:5];
        od = INTERMEDIATE; id_ = DIM;
        [enc setBytes:&od length:4 atIndex:6];
        [enc setBytes:&id_ length:4 atIndex:7];
        [enc setBytes:&gs length:4 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(INTERMEDIATE,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }

    // Step 2: dequant_matvec_4bit (down_proj) per expert [4096, 2048], group_size=32
    for (int k = 0; k < K; k++) {
        char *base = (char *)expert_bufs[k];
        id dw   = [d newBufferWithBytesNoCopy:base+dw_off length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
        id ds_b = [d newBufferWithBytesNoCopy:base+ds_off length:262144 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:eng->pipe_dequant_matvec];
        [enc setBuffer:dw offset:0 atIndex:0];    [enc setBuffer:ds_b offset:0 atIndex:1];
        [enc setBuffer:eng->buf_expert_mid[k] offset:0 atIndex:2];
        [enc setBuffer:eng->buf_expert_out[k] offset:0 atIndex:3];
        od = DIM; id_ = INTERMEDIATE;
        [enc setBytes:&od length:4 atIndex:4];
        [enc setBytes:&id_ length:4 atIndex:5];
        [enc setBytes:&gs length:4 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(DIM/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }

    // Step 3: moe_combine — weighted sum of K routed-expert outputs ONLY.
    // Commit the gate+up+down work first, then copy into a contiguous buffer.
    [cb commit]; [cb waitUntilCompleted];
    // Copy each expert's output (now ready on CPU-accessible shared memory) into
    // a contiguous [K*DIM] buffer for the combine kernel.
    id<MTLBuffer> contiguous_out = [d newBufferWithLength:(size_t)K*DIM*sizeof(float) options:MTLResourceStorageModeShared];
    {
        float *dst_all = (float *)[contiguous_out contents];
        for (int k = 0; k < K; k++) {
            float *src = (float *)[(id<MTLBuffer>)eng->buf_expert_out[k] contents];
            memcpy(dst_all + (size_t)k * DIM, src, DIM * sizeof(float));
        }
    }
    {
        id<MTLCommandBuffer> cb2 = [(id<MTLCommandQueue>)eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb2 computeCommandEncoder];
        [enc setComputePipelineState:eng->pipe_moe_combine];
        id weights_buf = [d newBufferWithBytes:expert_weights length:K*sizeof(float) options:MTLResourceStorageModeShared];
        id zero_resid = [d newBufferWithLength:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        [enc setBuffer:contiguous_out offset:0 atIndex:0];
        [enc setBuffer:weights_buf offset:0 atIndex:1];
        [enc setBuffer:zero_resid offset:0 atIndex:2];
        [enc setBuffer:eng->buf_hidden offset:0 atIndex:3];
        uint kv = K, hd = DIM;
        [enc setBytes:&kv length:4 atIndex:4];
        [enc setBytes:&hd length:4 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((DIM+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        [cb2 commit]; [cb2 waitUntilCompleted];
    }
    return 0;
}

// ============================================================================
// Top-K routing on CPU (softmax + topK)
// ============================================================================

// MLX routing: sqrtsoftplus scoring, topK selection, L1-normalize, scale by route_scale.
// Matches DSV4Gate.forward (scoring_func=sqrtsoftplus, norm_topk_prob=true, route_scale=1.5).
// bias: optional [N_EXPERTS] e_score_correction_bias (NULL for hash layers 0-2).
static void cpu_moe_route(const float *logits, const float *bias, int n, int K,
                          int *out_indices, float *out_weights) {
    // 1. sqrtsoftplus: scores[i] = sqrt(log(1 + exp(logits[i])))
    float *scores = (float *)alloca(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float l = logits[i];
        float sp = l > 0 ? l + log1pf(expf(-l)) : log1pf(expf(l));
        scores[i] = sqrtf(sp);
    }
    // 2. Add e_score_correction_bias for topK selection (not for weight computation)
    float *scores_for_choice = scores;
    float *biased = NULL;
    if (bias != NULL) {
        biased = (float *)alloca(n * sizeof(float));
        for (int i = 0; i < n; i++) biased[i] = scores[i] + bias[i];
        scores_for_choice = biased;
    }
    // 3. topK selection by biased scores
    int *taken = (int *)calloc(n, sizeof(int));
    for (int k = 0; k < K; k++) {
        int best = -1; float bv = -1e30f;
        for (int i = 0; i < n; i++) {
            if (!taken[i] && scores_for_choice[i] > bv) { bv = scores_for_choice[i]; best = i; }
        }
        out_indices[k] = best;
        out_weights[k] = scores[best]; // gather ORIGINAL scores (not biased)
        taken[best] = 1;
    }
    free(taken);
    // 4. L1-normalize + scale by 1.5
    float wsum = 0; for (int k = 0; k < K; k++) wsum += out_weights[k];
    wsum += 1e-20f;
    for (int k = 0; k < K; k++) out_weights[k] = out_weights[k] / wsum * 1.5f;
}

// ============================================================================
// CPU RoPE — DeepSeek V4 partial RoPE with YaRN (mode==2, neox-style pairing).
// Adapted from ds4/metal/dsv4_rope.metal algorithm.
// ============================================================================

#define ROPE_DIM 64         // qk_rope_head_dim
#define ROPE_THETA 10000000.0f
#define ROPE_N_CTX_ORIG 65536

static void apply_rope_tail(float *q_or_k, int dim, int n_nope, int pos, float freq_scale) {
    float corr_dims[2];
    {
        float beta_fast = 32.0f, beta_slow = 1.0f;
        float c0 = ROPE_DIM * logf(ROPE_N_CTX_ORIG / (beta_fast * 2.0f * M_PI)) / (2.0f * logf(ROPE_THETA));
        float c1 = ROPE_DIM * logf(ROPE_N_CTX_ORIG / (beta_slow * 2.0f * M_PI)) / (2.0f * logf(ROPE_THETA));
        corr_dims[0] = fmaxf(0.0f, floorf(c0));
        corr_dims[1] = fminf(ROPE_DIM - 1.0f, ceilf(c1));
    }

    float theta_base = (float)pos;
    float inv_ndims = -1.0f / ROPE_DIM;
    int n_half = ROPE_DIM / 2;

    for (int ic = 0; ic < n_half; ic++) {
        int rel_i0 = 2 * ic;
        float theta = theta_base * powf(ROPE_THETA, inv_ndims * rel_i0);

        // YaRN
        float ramp_mix = 0.0f;
        float low = corr_dims[0], high = corr_dims[1];
        float yarn_ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, ((float)rel_i0 / 2.0f - low) / fmaxf(0.001f, high - low)));
        theta = theta * (1.0f - yarn_ramp) + theta / freq_scale * yarn_ramp;

        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        int j0 = n_nope + ic;
        int j1 = n_nope + ic + n_half;
        float x0 = q_or_k[j0];
        float x1 = q_or_k[j1];
        q_or_k[j0] = x0 * cos_t - x1 * sin_t;
        q_or_k[j1] = x0 * sin_t + x1 * cos_t;
    }
}

// ============================================================================
// Main forward pass (1 token)
// ============================================================================

// Encode a float matvec: out = W @ x. W: [out_dim, in_dim], x: [in_dim]
static void encode_matvec(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
                          id<MTLBuffer> W_buf, id<MTLBuffer> x_buf, id<MTLBuffer> out_buf,
                          uint out_dim, uint in_dim) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:W_buf offset:0 atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBytes:&out_dim length:4 atIndex:3];
    [enc setBytes:&in_dim length:4 atIndex:4];
    [enc dispatchThreads:MTLSizeMake(out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc endEncoding];
}

// Encode RMSNorm: out = rms_norm(x, weight)
static void encode_rms_norm(id<MTLCommandBuffer> cb, MoEInferEngine *eng,
                            id<MTLBuffer> x_buf, id<MTLBuffer> weight_buf, id<MTLBuffer> out_buf) {
    // Step 1: sum of squares
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:eng->pipe_rms_norm_sum_sq];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:eng->buf_norm_sum_sq offset:0 atIndex:1];
        uint dim = DIM;
        [enc setBytes:&dim length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }
    // Step 2: apply normalization
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:eng->pipe_rms_norm_apply];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:weight_buf offset:0 atIndex:1];
        [enc setBuffer:eng->buf_norm_sum_sq offset:0 atIndex:2];
        [enc setBuffer:out_buf offset:0 atIndex:3];
        uint dim = DIM;
        float eps = 1e-6f;
        [enc setBytes:&dim length:4 atIndex:4];
        [enc setBytes:&eps length:4 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }
}

int moe_infer_forward_layer(MoEInferEngine *eng, int layer, float *hidden, int pos) {
    if (!eng->initialized) return -1;
    id<MTLDevice> d = (id<MTLDevice>)eng->device;

    // `hidden` is the mHC-expanded residual: [MHC_MULT, DIM] contiguous, in place.
    float *residual = hidden;

    // Build MlaPipes view over the engine's pipelines.
    MlaPipes P;
    P.dev = d;
    P.queue = (id<MTLCommandQueue>)eng->queue;
    P.dequant_matvec_affine = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine;
    P.rms_norm_rows = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows;
    P.rope_tail_interleaved = (id<MTLComputePipelineState>)eng->pipe_rope_tail;
    P.mla_sdpa_decode = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa;
    P.matvec_f32 = (id<MTLComputePipelineState>)eng->pipe_matvec;
    P.dequant_matvec_affine_bf16 = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16;
    P.rms_norm_rows_bf16 = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_bf16;
    P.bf16_to_f32 = (id<MTLComputePipelineState>)eng->pipe_bf16_to_f32;
    P.f32_to_bf16 = (id<MTLComputePipelineState>)eng->pipe_f32_to_bf16;
    P.dequant_matvec_affine_bf16in_f32out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_f32out;
    P.dequant_matvec_affine_bf16in_bf16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_bf16out;
    P.rms_norm_rows_bf16in_bf16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_bf16in_bf16out;
    P.rope_tail_bf16 = (id<MTLComputePipelineState>)eng->pipe_rope_tail_bf16;
    P.matvec_f32_bf16in = (id<MTLComputePipelineState>)eng->pipe_matvec_f32_bf16in;

    // Large scratch buffers are static (forward_layer runs serially) to avoid
    // overflowing the warmup/engine fiber's limited stack.
    static float attn_input[DIM], normed[DIM], attn_out[DIM];
    static float ffn_input[DIM], ffn_out[DIM];
    float post[MHC_MULT], comb[MHC_MULT * MHC_MULT];

    // === Attention sublayer (mHC-wrapped) ===
    MhcWeights ahc = { eng->attn_hc_fn[layer], eng->attn_hc_base[layer], eng->attn_hc_scale[layer] };
    if (layer == 0 && getenv("MF_DBG")) {
        double rn=0; for(int z=0;z<MHC_MULT*DIM;z++) rn+=(double)residual[z]*residual[z];
        fprintf(stderr, "[mf-dbg] L0 in residual norm=%.4f\n", sqrt(rn));
    }
    mhc_pre(&ahc, residual, attn_input, post, comb);
    if (layer == 0 && getenv("MF_DBG")) {
        double an=0; for(int z=0;z<DIM;z++) an+=(double)attn_input[z]*attn_input[z];
        fprintf(stderr, "[mf-dbg] L0 attn_input norm=%.4f post=[%.3f %.3f %.3f %.3f] comb00=%.3f\n",
            sqrt(an), post[0],post[1],post[2],post[3], comb[0]);
    }

    // input RMSNorm (attn_norm) on attn_input -> normed (validated rms_norm_rows)
    {
        id<MTLBuffer> bx = [d newBufferWithBytes:attn_input length:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw = [d newBufferWithBytes:(void *)eng->input_norms[layer] length:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [d newBufferWithLength:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_rms_norm_rows];
        [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bw offset:0 atIndex:1]; [e setBuffer:bo offset:0 atIndex:2];
        uint rd = DIM; float eps = 1e-6f; uint hw = 1;
        [e setBytes:&rd length:4 atIndex:3]; [e setBytes:&eps length:4 atIndex:4]; [e setBytes:&hw length:4 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(normed, [bo contents], DIM * sizeof(float));
    }

    // MLA attention (decode, single token -> cache_len from kv_cache)
    {
        KVCache *kvc = &eng->kv_cache[layer];
        if (!kvc->kv) { kvc->kv = (float *)calloc((size_t)MAX_SEQ_LEN * KV_LORA_RANK, sizeof(float)); kvc->len = 0; }
        kvc->len += 1;
        mla_attention_decode(&P, &eng->attn[layer], normed, kvc->kv, kvc->len, pos, attn_out);
    }

    // mHC post -> residual' (in place)
    mhc_post(attn_out, residual, post, comb, residual);
    if (layer == 0 && getenv("MF_DBG")) {
        double an=0; for(int z=0;z<DIM;z++) an+=(double)attn_out[z]*attn_out[z];
        double rn=0; for(int z=0;z<MHC_MULT*DIM;z++) rn+=(double)residual[z]*residual[z];
        fprintf(stderr, "[mf-dbg] L0 attn_out norm=%.4f, residual after attn-post norm=%.4f\n", sqrt(an), sqrt(rn));
        // Dump attn_out for comparison
        const char *dd = getenv("DSV4_DUMP_DIR");
        if (dd) {
            char path[1024]; snprintf(path, sizeof(path), "%s/L0_attn_out_metal.bin", dd);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(attn_out, sizeof(float), DIM, f); fclose(f); }
        }
    }

    // === MoE sublayer (mHC-wrapped) ===
    MhcWeights fhc = { eng->ffn_hc_fn[layer], eng->ffn_hc_base[layer], eng->ffn_hc_scale[layer] };
    mhc_pre(&fhc, residual, ffn_input, post, comb);

    // ffn RMSNorm (attn_norms[] holds ffn_norm) on ffn_input -> normed
    {
        id<MTLBuffer> bx = [d newBufferWithBytes:ffn_input length:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw = [d newBufferWithBytes:(void *)eng->attn_norms[layer] length:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [d newBufferWithLength:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_rms_norm_rows];
        [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bw offset:0 atIndex:1]; [e setBuffer:bo offset:0 atIndex:2];
        uint rd = DIM; float eps = 1e-6f; uint hw = 1;
        [e setBytes:&rd length:4 atIndex:3]; [e setBytes:&eps length:4 atIndex:4]; [e setBytes:&hw length:4 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(normed, [bo contents], DIM * sizeof(float));
    }

    if (layer == 0 && getenv("MF_DBG")) {
        double nn=0; for(int z=0;z<DIM;z++) nn+=(double)normed[z]*normed[z];
        fprintf(stderr, "[mf-dbg] L0 normed(ffn input) norm=%.4f\n", sqrt(nn));
        const char *dd = getenv("DSV4_DUMP_DIR");
        if (dd) {
            char path[1024]; snprintf(path, sizeof(path), "%s/L0_normed_ffn_in.bin", dd);
            FILE *f = fopen(path, "wb"); if (f) { fwrite(normed, sizeof(float), DIM, f); fclose(f); }
        }
    }

    // === Routing gate ===
    float *scores = (float *)[(id<MTLBuffer>)eng->buf_routing_scores contents];
    if (eng->gate_proj[layer]) {
        float *buf_h = (float *)[(id<MTLBuffer>)eng->buf_hidden contents];
        memcpy(buf_h, normed, DIM * sizeof(float));
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBuffer];
        id gate_w = [d newBufferWithBytes:(void *)eng->gate_proj[layer] length:N_EXPERTS*DIM*sizeof(float) options:MTLResourceStorageModeShared];
        encode_matvec(cb, eng->pipe_matvec, gate_w, eng->buf_hidden, eng->buf_routing_scores, N_EXPERTS, DIM);
        [cb commit]; [cb waitUntilCompleted];
    } else {
        for (int i = 0; i < N_EXPERTS; i++) scores[i] = (float)(N_EXPERTS - i);
    }

    int expert_ids[N_ACTIVE];
    float expert_weights[N_ACTIVE];
    // Hash routing for layers 0-2: look up experts by token ID.
    // NOTE: Hash routing is correct data-wise but does not improve E2E output
    // because the f32 vs bf16 precision difference causes chaos divergence
    // in score-based layers (3+) regardless. Left disabled pending bf16 alignment.
    const bool use_hash_routing = false;
    if (use_hash_routing && eng->tid2eid[layer] != NULL && eng->current_token_id >= 0) {
        const int64_t *row = eng->tid2eid[layer] + (size_t)eng->current_token_id * N_ACTIVE;
        for (int k = 0; k < N_ACTIVE; k++) expert_ids[k] = (int)row[k];
        // Weights: gather sqrtsoftplus(logits) at hash-selected positions, L1-normalize, scale.
        // First compute sqrtsoftplus on the current gate scores.
        float wsum = 0;
        for (int k = 0; k < N_ACTIVE; k++) {
            float l = scores[expert_ids[k]];
            float sp = l > 0 ? l + log1pf(expf(-l)) : log1pf(expf(l));
            expert_weights[k] = sqrtf(sp);
            wsum += expert_weights[k];
        }
        wsum += 1e-20f;
        for (int k = 0; k < N_ACTIVE; k++) expert_weights[k] = expert_weights[k] / wsum * 1.5f;
    } else {
        cpu_moe_route(scores, eng->gate_bias[layer], N_EXPERTS, N_ACTIVE, expert_ids, expert_weights);
    }
    if (layer == 0 && getenv("MF_DBG")) {
        fprintf(stderr, "[mf-dbg] L0 expert_ids=[%d,%d,%d,%d,%d,%d] weights=[%.3f,%.3f,%.3f] hash=%s tok=%d\n",
            expert_ids[0],expert_ids[1],expert_ids[2],expert_ids[3],expert_ids[4],expert_ids[5],
            expert_weights[0],expert_weights[1],expert_weights[2],
            eng->tid2eid[layer] != NULL ? "yes" : "no",
            eng->current_token_id);
    }

    // Expert I/O + MoE. The expert kernel reads buf_normed as input, so load
    // the ffn-normed vector into it. MoE combine writes the pure routed-expert
    // sum (zero residual) to buf_hidden.
    {
        float *bn = (float *)[(id<MTLBuffer>)eng->buf_normed contents];
        memcpy(bn, normed, DIM * sizeof(float));
        IOPool *io = (IOPool *)eng->io_pool;
        io_pool_dispatch(io, eng->packed_fd[layer], expert_ids, N_ACTIVE, eng->expert_buf);
        moe_forward_layer(eng, layer, eng->expert_buf, expert_ids, expert_weights, N_ACTIVE);
        memcpy(ffn_out, [(id<MTLBuffer>)eng->buf_hidden contents], DIM * sizeof(float));
    }

    if (layer == 0 && getenv("MF_DBG")) {
        double fn=0; for(int z=0;z<DIM;z++) fn+=(double)ffn_out[z]*ffn_out[z];
        const char *dd = getenv("DSV4_DUMP_DIR");
        fprintf(stderr, "[mf-dbg] L0 ffn_out(moe only) norm=%.4f\n", sqrt(fn));
        // Dump pre-shared-expert MoE output for debugging
        if (dd) {
            char path[1024]; snprintf(path, sizeof(path), "%s/L0_moe_only_out.bin", dd);
            FILE *f = fopen(path, "wb"); if (f) { fwrite(ffn_out, sizeof(float), DIM, f); fclose(f); }
        }
    }

    // Shared expert: runs on the same normed input, output added to ffn_out.
    if (eng->shared[layer].gate.packed != NULL) {
        const int SE_GS = 64;
        const int SE_NG_GU = DIM / SE_GS;   // 64 groups for gate/up [2048,4096]
        const int SE_NG_D  = INTERMEDIATE / SE_GS; // 32 groups for down [4096,2048]
        id<MTLBuffer> bx = [d newBufferWithBytes:normed length:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bgate = [d newBufferWithLength:INTERMEDIATE*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bup   = [d newBufferWithLength:INTERMEDIATE*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bdown = [d newBufferWithLength:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        // gate and up projections
        {
            id<MTLCommandBuffer> cb = [P.queue commandBuffer];
            for (int proj = 0; proj < 2; proj++) {
                const QuantWeight *qw = (proj==0) ? &eng->shared[layer].gate : &eng->shared[layer].up;
                id<MTLBuffer> bout = (proj==0) ? bgate : bup;
                id bw = [d newBufferWithBytesNoCopy:(void*)qw->packed length:(size_t)qw->out_dim*(qw->in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
                id bs = [d newBufferWithBytes:(void*)qw->scales length:(size_t)qw->out_dim*SE_NG_GU*sizeof(float) options:MTLResourceStorageModeShared];
                id bb = [d newBufferWithBytes:(void*)qw->biases length:(size_t)qw->out_dim*SE_NG_GU*sizeof(float) options:MTLResourceStorageModeShared];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
                [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bs offset:0 atIndex:1]; [e setBuffer:bb offset:0 atIndex:2];
                [e setBuffer:bx offset:0 atIndex:3]; [e setBuffer:bout offset:0 atIndex:4];
                uint od=qw->out_dim, id_=qw->in_dim, gs=SE_GS;
                [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
                [e dispatchThreads:MTLSizeMake(qw->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            }
            [cb commit]; [cb waitUntilCompleted];
        }
        // Limited SwiGLU on CPU
        float *gv = (float *)[bgate contents]; float *uv = (float *)[bup contents];
        const float limit = 10.0f;
        for (int j = 0; j < INTERMEDIATE; j++) {
            float g = fminf(gv[j], limit);
            float u = fminf(fmaxf(uv[j], -limit), limit);
            gv[j] = g / (1.0f + expf(-g)) * u;
        }
        // down projection
        {
            id<MTLCommandBuffer> cb = [P.queue commandBuffer];
            const QuantWeight *qw = &eng->shared[layer].down;
            id bw = [d newBufferWithBytesNoCopy:(void*)qw->packed length:(size_t)qw->out_dim*(qw->in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
            id bs = [d newBufferWithBytes:(void*)qw->scales length:(size_t)qw->out_dim*SE_NG_D*sizeof(float) options:MTLResourceStorageModeShared];
            id bb = [d newBufferWithBytes:(void*)qw->biases length:(size_t)qw->out_dim*SE_NG_D*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
            [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bs offset:0 atIndex:1]; [e setBuffer:bb offset:0 atIndex:2];
            [e setBuffer:bgate offset:0 atIndex:3]; [e setBuffer:bdown offset:0 atIndex:4];
            uint od=qw->out_dim, id_=qw->in_dim, gs=SE_GS;
            [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
            [e dispatchThreads:MTLSizeMake(qw->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        float *sv = (float *)[bdown contents];
        for (int j = 0; j < DIM; j++) ffn_out[j] += sv[j];
        if (layer == 0 && getenv("MF_DBG")) {
            double sn=0; for(int z=0;z<DIM;z++) sn+=(double)sv[z]*sv[z];
            fprintf(stderr, "[mf-dbg] L0 shared_out norm=%.4f\n", sqrt(sn));
        }
    }

    // mHC post -> residual'' (in place)
    mhc_post(ffn_out, residual, post, comb, residual);
    if (layer == 0 && getenv("MF_DBG")) {
        double fn=0; for(int z=0;z<DIM;z++) fn+=(double)ffn_out[z]*ffn_out[z];
        double rn=0; for(int z=0;z<MHC_MULT*DIM;z++) rn+=(double)residual[z]*residual[z];
        fprintf(stderr, "[mf-dbg] L0 ffn_out norm=%.4f, residual after ffn-post norm=%.4f\n", sqrt(fn), sqrt(rn));
        const char *dd = getenv("DSV4_DUMP_DIR");
        if (dd) {
            char path[1024]; snprintf(path, sizeof(path), "%s/L0_ffn_out_metal.bin", dd);
            FILE *ff = fopen(path, "wb");
            if (ff) { fwrite(ffn_out, sizeof(float), DIM, ff); fclose(ff); }
        }
    }

    return 0;
}

int moe_infer_forward(MoEInferEngine *eng, float *hidden, int pos) {
    for (int layer = 0; layer < N_LAYERS; layer++) {
        if (moe_infer_forward_layer(eng, layer, hidden, pos) != 0) return -1;
    }
    return 0;
}

// ============================================================================
// Init / Deinit
// ============================================================================

MoEInferEngine *moe_infer_init(const char *packed_dir,
                                const char *kernel_src, unsigned long kernel_src_len) {
    MoEInferEngine *eng = (MoEInferEngine *)calloc(1, sizeof(MoEInferEngine));
    if (!eng) return NULL;

    if (init_metal(eng, kernel_src, kernel_src_len) != 0) { free(eng); return NULL; }

    char path[4096];
    for (int l = 0; l < N_LAYERS; l++) {
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", packed_dir, l);
        eng->packed_fd[l] = open(path, O_RDONLY);
        if (eng->packed_fd[l] < 0) {
            fprintf(stderr, "open %s: %s\n", path, strerror(errno));
            moe_infer_deinit(eng); return NULL;
        }
        fcntl(eng->packed_fd[l], F_RDAHEAD, 1);
    }

    IOPool *io = (IOPool *)calloc(1, sizeof(IOPool));
    io_pool_init(io);
    eng->io_pool = io;

    eng->initialized = true;
    fprintf(stderr, "Metal engine: %d expert files opened\n", N_LAYERS);
    return eng;
}

void moe_infer_set_weights(MoEInferEngine *eng,
    const float *embed, int vocab_size, const float *lm_head, const float *final_norm,
    const float **input_norms, const float **attn_norms,
    const float **gate_proj_w, const float **gate_bias_w) {
    eng->embed = embed;
    eng->vocab_size = vocab_size;
    eng->lm_head = lm_head;
    eng->final_norm = final_norm;
    for (int i = 0; i < N_LAYERS; i++) {
        eng->input_norms[i] = input_norms[i];
        eng->attn_norms[i] = attn_norms[i];
        eng->gate_proj[i]  = gate_proj_w[i];
        eng->gate_bias[i]  = gate_bias_w[i];
    }
}

void moe_infer_set_layer_attn(MoEInferEngine *eng, int layer, AttnWeights attn) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->attn[layer] = attn;
}

void moe_infer_set_layer_shared(MoEInferEngine *eng, int layer, SharedExpert se) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->shared[layer] = se;
}

void moe_infer_set_layer_tid2eid(MoEInferEngine *eng, int layer, const int64_t *tid2eid) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->tid2eid[layer] = tid2eid;
}

void moe_infer_set_token_id(MoEInferEngine *eng, int token_id) {
    eng->current_token_id = token_id;
}

void moe_infer_reset_kv(MoEInferEngine *eng) {
    for (int l = 0; l < N_LAYERS; l++) {
        eng->kv_cache[l].len = 0;
        // Clear KV buffer so stale entries from the previous request
        // cannot bleed into the new sequence when cache_len is small.
        if (eng->kv_cache[l].kv) {
            memset(eng->kv_cache[l].kv, 0,
                   (size_t)MAX_SEQ_LEN * KV_LORA_RANK * sizeof(float));
        }
    }
}

void moe_infer_set_layer_hc(MoEInferEngine *eng, int layer,
    const float *attn_fn, const float *attn_base, const float *attn_scale,
    const float *ffn_fn, const float *ffn_base, const float *ffn_scale) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->attn_hc_fn[layer] = attn_fn;
    eng->attn_hc_base[layer] = attn_base;
    eng->attn_hc_scale[layer] = attn_scale;
    eng->ffn_hc_fn[layer] = ffn_fn;
    eng->ffn_hc_base[layer] = ffn_base;
    eng->ffn_hc_scale[layer] = ffn_scale;
}

void moe_infer_deinit(MoEInferEngine *eng) {
    if (!eng || !eng->initialized) return;
    IOPool *io = (IOPool *)eng->io_pool;
    if (io) {
        pthread_mutex_lock(&io->mutex);
        io->shutdown = true;
        pthread_cond_broadcast(&io->work_ready);
        pthread_mutex_unlock(&io->mutex);
        for (int i = 0; i < NUM_IO_THREADS; i++) {
            pthread_join(io->threads[i], NULL);
        }
        free(io);
    }
    for (int l = 0; l < N_LAYERS; l++) {
        if (eng->packed_fd[l] >= 0) close(eng->packed_fd[l]);
    }
    for (int k = 0; k < N_ACTIVE; k++) {
        free(eng->expert_buf[k]);
        free(eng->expert_buf_pred[k]);
    }
    free(eng);
}
