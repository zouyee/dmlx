// Metal inference engine — per-token forward pass with flash-moe pipeline.
// Phase 1: MoE forward only (attention handled by MLX for now).
// Phase 2: Add attention MatVecs on Metal.
// Phase 3: Full flash-moe alignment.
//
// See docs/analysis/flash-moe-alignment-plan.md for architecture details.
#include "engine.h"
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

    // Step 3: moe_combine — weighted sum of K expert outputs + residual
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:eng->pipe_moe_combine];
        // Pass expert output buffers as a flat array: all K outputs concatenated
        // For simplicity, copy expert outputs into a temp buffer
        // (optimization: use buffer offset trick in kernel)
        id weights_buf = [d newBufferWithBytes:expert_weights length:K*sizeof(float) options:MTLResourceStorageModeShared];
        [enc setBuffer:eng->buf_expert_out[0] offset:0 atIndex:0]; // kernel reads all K from contiguous?
        [enc setBuffer:weights_buf offset:0 atIndex:1];
        [enc setBuffer:eng->buf_h_mid offset:0 atIndex:2]; // residual (input before MoE)
        [enc setBuffer:eng->buf_hidden offset:0 atIndex:3]; // output
        uint kv = K, hd = DIM;
        [enc setBytes:&kv length:4 atIndex:4];
        [enc setBytes:&hd length:4 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((DIM+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
    }

    [cb commit];
    [cb waitUntilCompleted];
    return 0;
}

// ============================================================================
// Top-K routing on CPU (softmax + topK)
// ============================================================================

static void cpu_softmax_topk(const float *scores, int n, int K,
                              int *out_indices, float *out_weights) {
    // Find max for numerical stability
    float max_val = scores[0];
    for (int i = 1; i < n; i++) if (scores[i] > max_val) max_val = scores[i];

    // Compute exp and sum
    float sum = 0.0f;
    float *probs = (float *)alloca(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        probs[i] = expf(scores[i] - max_val);
        sum += probs[i];
    }

    // Select top-K
    int *taken = (int *)calloc(n, sizeof(int));
    for (int k = 0; k < K; k++) {
        int best = -1;
        float best_val = -1.0f;
        for (int i = 0; i < n; i++) {
            if (!taken[i] && probs[i] > best_val) {
                best_val = probs[i];
                best = i;
            }
        }
        out_indices[k] = best;
        out_weights[k] = probs[best] / sum;
        taken[best] = 1;
    }
    free(taken);
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
    (void)pos;

    // Copy hidden to buf_hidden
    float *buf_h = (float *)[(id<MTLBuffer>)eng->buf_hidden contents];
    memcpy(buf_h, hidden, DIM * sizeof(float));

    // === RMSNorm (input norm) ===
    {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBuffer];
        id w_buf = [d newBufferWithBytesNoCopy:(void *)eng->input_norms[layer]
                                       length:DIM * sizeof(float)
                                       options:MTLResourceStorageModeShared deallocator:nil];
        encode_rms_norm(cb, eng, eng->buf_hidden, w_buf, eng->buf_normed);
        [cb commit];
        [cb waitUntilCompleted];
    }
    float *normed = (float *)[(id<MTLBuffer>)eng->buf_normed contents];

    // === Attention: Q/K/V projections + RoPE + simplified SDPA ===
    float *h_mid = (float *)[(id<MTLBuffer>)eng->buf_h_mid contents];
    float *attn_out = (float *)[(id<MTLBuffer>)eng->buf_attn_out contents];

    // Q projection
    if (eng->q_proj[layer]) {
        id<MTLCommandBuffer> cb_a = [(id<MTLCommandQueue>)eng->queue commandBuffer];
        // Q, K, V buffers
        id q_buf = [d newBufferWithLength:DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id k_buf = [d newBufferWithLength:DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id v_buf = [d newBufferWithLength:DIM * sizeof(float) options:MTLResourceStorageModeShared];

        // Q, K, V projections
        id q_w = [d newBufferWithBytesNoCopy:(void *)eng->q_proj[layer] length:DIM*DIM*sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        encode_matvec(cb_a, eng->pipe_matvec, q_w, eng->buf_normed, q_buf, DIM, DIM);

        if (eng->k_proj[layer]) {
            id k_w = [d newBufferWithBytesNoCopy:(void *)eng->k_proj[layer] length:DIM*KV_LORA_RANK*sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
            encode_matvec(cb_a, eng->pipe_matvec, k_w, eng->buf_normed, k_buf, KV_LORA_RANK, DIM);
        }
        if (eng->v_proj[layer]) {
            id v_w = [d newBufferWithBytesNoCopy:(void *)eng->v_proj[layer] length:DIM*KV_LORA_RANK*sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
            encode_matvec(cb_a, eng->pipe_matvec, v_w, eng->buf_normed, v_buf, KV_LORA_RANK, DIM);
        }

        [cb_a commit];
        [cb_a waitUntilCompleted];

        // CPU: Apply RoPE to Q and K (V4 partial RoPE, tail only)
        float *q = (float *)[q_buf contents];
        float *k = (float *)[k_buf contents];
        int n_nope = DIM - ROPE_DIM;
        apply_rope_tail(q, DIM, n_nope, pos, 1.0f);
        apply_rope_tail(k, KV_LORA_RANK, KV_LORA_RANK - ROPE_DIM, pos, 1.0f);

        // Simplified SDPA: single token self-attention = V (attention over one position is identity)
        // O = o_proj @ V → but V is [KV_LORA_RANK], not [DIM]
        // For now: use Q as attention output (simplified)
        memcpy(attn_out, q, DIM * sizeof(float));
    } else {
        // No Q weights: pass normed through
        memcpy(attn_out, normed, DIM * sizeof(float));
    }

    // Residual: h_mid = hidden + attn_out
    for (int i = 0; i < DIM; i++) h_mid[i] = buf_h[i] + attn_out[i];

    // === Post-attn RMSNorm ===
    {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBuffer];
        id w_buf = [d newBufferWithBytesNoCopy:(void *)eng->attn_norms[layer]
                                       length:DIM * sizeof(float)
                                       options:MTLResourceStorageModeShared deallocator:nil];
        memcpy(buf_h, h_mid, DIM * sizeof(float)); // buf_hidden = h_mid
        encode_rms_norm(cb, eng, eng->buf_hidden, w_buf, eng->buf_normed);
        [cb commit];
        [cb waitUntilCompleted];
    }
    memcpy(normed, [(id<MTLBuffer>)eng->buf_normed contents], DIM * sizeof(float));

    // === Routing gate ===
    float *scores = (float *)[(id<MTLBuffer>)eng->buf_routing_scores contents];
    if (eng->gate_proj[layer]) {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBuffer];
        id gate_w = [d newBufferWithBytesNoCopy:(void *)eng->gate_proj[layer]
                                       length:N_EXPERTS * DIM * sizeof(float)
                                       options:MTLResourceStorageModeShared deallocator:nil];
        // Copy normed to buf_hidden for matvec input
        memcpy(buf_h, normed, DIM * sizeof(float));
        encode_matvec(cb, eng->pipe_matvec, gate_w, eng->buf_hidden, eng->buf_routing_scores, N_EXPERTS, DIM);
        [cb commit];
        [cb waitUntilCompleted];
    } else {
        for (int i = 0; i < N_EXPERTS; i++) scores[i] = (float)(N_EXPERTS - i);
    }

    // CPU softmax + topK
    int expert_ids[N_ACTIVE];
    float expert_weights[N_ACTIVE];
    cpu_softmax_topk(scores, N_EXPERTS, N_ACTIVE, expert_ids, expert_weights);

    // === Expert I/O ===
    IOPool *io = (IOPool *)eng->io_pool;
    io_pool_dispatch(io, eng->packed_fd[layer], expert_ids, N_ACTIVE, eng->expert_buf);

    // === MoE forward (Metal) ===
    moe_forward_layer(eng, layer, eng->expert_buf, expert_ids, expert_weights, N_ACTIVE);

    // Read result back to hidden
    memcpy(hidden, [(id<MTLBuffer>)eng->buf_hidden contents], DIM * sizeof(float));
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
    const float **q_proj_w, const float **k_proj_w, const float **v_proj_w, const float **o_proj_w,
    const float **q_norms, const float **k_norms,
    const float **gate_proj_w) {
    eng->embed = embed;
    eng->vocab_size = vocab_size;
    eng->lm_head = lm_head;
    eng->final_norm = final_norm;
    for (int i = 0; i < N_LAYERS; i++) {
        eng->input_norms[i] = input_norms[i];
        eng->attn_norms[i] = attn_norms[i];
        eng->q_proj[i]     = q_proj_w[i];
        eng->k_proj[i]     = k_proj_w[i];
        eng->v_proj[i]     = v_proj_w[i];
        eng->o_proj[i]     = o_proj_w[i];
        eng->q_norms[i]    = q_norms[i];
        eng->k_norms[i]    = k_norms[i];
        eng->gate_proj[i]  = gate_proj_w[i];
    }
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
