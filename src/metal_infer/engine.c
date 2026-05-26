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
// Main forward pass (1 token)
// ============================================================================

int moe_infer_forward(MoEInferEngine *eng, int token, float *logits) {
    if (!eng->initialized) return -1;
    int pos = eng->current_pos;

    // Embedding lookup
    float *embed = eng->wf.embed; // [vocab, DIM]
    float *hidden = (float *)[(id<MTLBuffer>)eng->buf_hidden contents];
    for (int i = 0; i < DIM; i++) {
        hidden[i] = embed[token * DIM + i];
    }

    // Per-layer forward
    for (int layer = 0; layer < N_LAYERS; layer++) {
        // === Attention ===
        // Phase 2: Move attention to Metal/CPU
        // For now: skip attention, just RMSNorm input and pass through
        // (This means we're running MoE-only on raw embeddings — WRONG, but
        //  needed for integration testing)

        // Copy hidden to normed buffer (skip attention for now)
        float *normed = (float *)[(id<MTLBuffer>)eng->buf_normed contents];
        memcpy(normed, hidden, DIM * sizeof(float));

        // === Routing ===
        // Read routing scores from gate projection
        // TODO: compute gate projection for routing
        // For now: use dummy routing (experts 0..K-1)

        int expert_ids[N_ACTIVE];
        float expert_weights[N_ACTIVE];
        for (int k = 0; k < N_ACTIVE; k++) {
            expert_ids[k] = k;
            expert_weights[k] = 1.0f / N_ACTIVE;
        }

        // === Predictor check ===
        if (eng->predictor.valid) {
            // Check prediction hits
            for (int k = 0; k < N_ACTIVE; k++) {
                bool hit = false;
                for (int p = 0; p < N_ACTIVE; p++) {
                    if (expert_ids[k] == eng->predictor.experts[layer][p]) {
                        eng->predictor.hits++;
                        hit = true;
                        break;
                    }
                }
                if (!hit) eng->predictor.misses++;
            }
        }

        // Record for next token prediction
        predictor_record(&eng->predictor, layer, expert_ids, N_ACTIVE);

        // === I/O: Read experts ===
        IOPool *io = (IOPool *)eng->io_pool;
        io_pool_dispatch(io, eng->packed_fd[layer], expert_ids, N_ACTIVE,
                         eng->expert_buf);

        // === Store h_mid (residual) ===
        float *h_mid = (float *)[(id<MTLBuffer>)eng->buf_h_mid contents];
        memcpy(h_mid, hidden, DIM * sizeof(float));

        // === MoE forward (Metal) ===
        moe_forward_layer(eng, layer, eng->expert_buf, expert_ids,
                          expert_weights, N_ACTIVE);

        // Read result back to hidden
        memcpy(hidden, [(id<MTLBuffer>)eng->buf_hidden contents], DIM * sizeof(float));
    }

    // Final RMSNorm
    // TODO

    // LM head
    // TODO: compute logits from hidden @ lm_head^T

    eng->current_pos++;
    return 0;
}

// ============================================================================
// Init / Deinit
// ============================================================================

int moe_infer_init(MoEInferEngine *eng, const char *model_path,
                   const char *packed_dir,
                   const char *kernel_src, unsigned long kernel_src_len) {
    memset(eng, 0, sizeof(*eng));

    // Init Metal
    if (init_metal(eng, kernel_src, kernel_src_len) != 0) return -1;

    // Open packed expert files
    char path[4096];
    for (int l = 0; l < N_LAYERS; l++) {
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", packed_dir, l);
        eng->packed_fd[l] = open(path, O_RDONLY);
        if (eng->packed_fd[l] < 0) {
            fprintf(stderr, "open %s: %s\n", path, strerror(errno));
            return -1;
        }
        fcntl(eng->packed_fd[l], F_RDAHEAD, 1);
    }

    // Init I/O pool
    IOPool *io = (IOPool *)calloc(1, sizeof(IOPool));
    io_pool_init(io);
    eng->io_pool = io;

    // TODO: Load model weights (embed, attention, norms, lm_head) from
    // safetensors via MLX. For now, stubbed out.

    eng->initialized = true;
    fprintf(stderr, "Metal engine: %d expert files opened, I/O pool ready\n", N_LAYERS);
    return 0;
}

void moe_infer_deinit(MoEInferEngine *eng) {
    if (!eng->initialized) return;
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
    // Metal objects released by ARC
    memset(eng, 0, sizeof(*eng));
}
