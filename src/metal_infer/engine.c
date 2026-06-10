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
#include <Accelerate/Accelerate.h>

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
    eng->pipe_fused_6expert_gate_up = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_6expert_gate_up_swiglu"] error:&err]);
    eng->pipe_fused_6expert_down    = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_6expert_down"] error:&err]);
    eng->pipe_gather_gate_up = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather_gate_up_swiglu"] error:&err]);
    eng->pipe_gather_down    = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather_down"] error:&err]);
    eng->pipe_rms_norm_sum_sq = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_sum_sq"] error:&err]);
    eng->pipe_rms_norm_apply = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_apply"] error:&err]);
    eng->pipe_matvec = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec_f32"] error:&err]);
    // S7: MLA attention pipelines
    eng->pipe_dequant_matvec_affine = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine"] error:&err]);
    eng->pipe_rms_norm_rows = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows"] error:&err]);
    eng->pipe_rope_tail = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rope_tail_interleaved"] error:&err]);
    eng->pipe_mla_sdpa = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_decode"] error:&err]);
    eng->pipe_mla_sdpa_f16 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_decode_f16"] error:&err]);
    // F16 precision chain (validated correct, see docs/analysis/dsv4-first-class-support-plan.md)
    eng->pipe_dequant_matvec_affine_f16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_f16out"] error:&err]);
    eng->pipe_rms_norm_rows_f16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows_f16out"] error:&err]);
    eng->pipe_dequant_matvec_affine_f16in_f16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_f16in_f16out"] error:&err]);
    eng->pipe_rms_norm_rows_f16in_f16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows_f16in_f16out"] error:&err]);
    eng->pipe_rope_tail_f16 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rope_tail_interleaved_f16"] error:&err]);
    eng->pipe_matvec_f32_f16in = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec_f32_f16in"] error:&err]);
    eng->pipe_mla_sdpa_f16in_f16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_decode_f16in_f16out"] error:&err]);
    // BF16 precision chain — matches MLX training precision
    eng->pipe_dequant_matvec_affine_bf16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16out"] error:&err]);
    eng->pipe_rms_norm_rows_bf16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows_bf16out"] error:&err]);
    eng->pipe_dequant_matvec_affine_bf16in_bf16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_bf16out"] error:&err]);
    eng->pipe_rms_norm_rows_bf16in_bf16out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_rows_bf16in_bf16out"] error:&err]);
    eng->pipe_rope_tail_bf16 = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rope_tail_interleaved_bf16"] error:&err]);
    eng->pipe_matvec_f32_bf16in = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec_f32_bf16in"] error:&err]);
    eng->pipe_mla_sdpa_bfloat = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_decode_bfloat"] error:&err]);
    eng->pipe_dequant_matvec_affine_bf16in_f32out = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dequant_matvec_affine_bf16in_f32out"] error:&err]);
    eng->pipe_mla_sdpa_prefill_bfloat = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mla_sdpa_prefill_bfloat"] error:&err]);
    eng->pipe_bf16_to_f16_row = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bf16_to_f16_row"] error:&err]);
    eng->pipe_limited_swiglu  = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"limited_swiglu"] error:&err]);
    eng->pipe_f32_to_bf16_vec = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"f32_to_bf16_vec"] error:&err]);
    eng->pipe_bf16_to_f32_vec = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bf16_to_f32_vec"] error:&err]);
    // mHC GPU kernels (f16/bfloat)
    eng->pipe_mhc_pre_f16   = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mhc_pre_f16"] error:&err]);
    eng->pipe_mhc_post_f16  = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mhc_post_f16"] error:&err]);
    eng->pipe_mhc_pre_bfloat  = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mhc_pre_gpu"] error:&err]);
    eng->pipe_mhc_post_bfloat = (void *)([d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mhc_post_bfloat"] error:&err]);
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
    eng->buf_expert_contiguous = (void *)[d newBufferWithLength:(size_t)N_ACTIVE*DIM*sizeof(float) options:MTLResourceStorageModeShared];

    // Gather mode output buffers
    eng->buf_gather_mid = (void *)[d newBufferWithLength:(size_t)N_ACTIVE*INTERMEDIATE*sizeof(float) options:MTLResourceStorageModeShared];
    eng->buf_gather_out = (void *)[d newBufferWithLength:(size_t)N_ACTIVE*DIM*sizeof(float) options:MTLResourceStorageModeShared];
    eng->buf_gather_expert_ids = (void *)[d newBufferWithLength:N_ACTIVE*sizeof(uint32_t) options:MTLResourceStorageModeShared];

    // Persistent scratch buffers for mhc_pre / mhc_post — eliminates per-call allocation
    const int MIX3_size = MHC_MULT * (MHC_MULT + 2);  // 24
    eng->buf_mhc_res_in        = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(float)         options:MTLResourceStorageModeShared];
    eng->buf_mhc_attn_in       = (void *)[d newBufferWithLength:DIM * sizeof(float)                    options:MTLResourceStorageModeShared];
    eng->buf_mhc_attn_norm_bf16= (void *)[d newBufferWithLength:DIM * sizeof(uint16_t)                 options:MTLResourceStorageModeShared];
    eng->buf_mhc_post_weights  = (void *)[d newBufferWithLength:MHC_MULT * sizeof(float)               options:MTLResourceStorageModeShared];
    eng->buf_mhc_comb_weights  = (void *)[d newBufferWithLength:MHC_MULT * MHC_MULT * sizeof(float)    options:MTLResourceStorageModeShared];
    eng->buf_mhc_res_out       = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(uint16_t)      options:MTLResourceStorageModeShared];
    eng->buf_mhc_ffn_in        = (void *)[d newBufferWithLength:DIM * sizeof(float)                    options:MTLResourceStorageModeShared];
    eng->buf_mhc_ffn_norm_bf16 = (void *)[d newBufferWithLength:DIM * sizeof(uint16_t)                 options:MTLResourceStorageModeShared];
    eng->buf_mhc_ffn_res_in    = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(float)         options:MTLResourceStorageModeShared];
    eng->buf_mhc_attn_out_bf16 = (void *)[d newBufferWithLength:DIM * sizeof(uint16_t)                 options:MTLResourceStorageModeShared];
    eng->buf_mhc_res_bf16_in   = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(uint16_t)      options:MTLResourceStorageModeShared];
    eng->buf_mhc_post_res_out  = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(uint16_t)      options:MTLResourceStorageModeShared];
    eng->buf_mhc_ffn_post_out  = (void *)[d newBufferWithLength:MHC_MULT * DIM * sizeof(uint16_t)      options:MTLResourceStorageModeShared];
    (void)MIX3_size;

    // GPU-resident residual buffer (Path B: eliminate CPU↔GPU residual transfers)
    eng->buf_residual_gpu = (void *)[d newBufferWithLength:(size_t)MHC_MULT * DIM * sizeof(float)
                                                    options:MTLResourceStorageModeShared];

    // 2MB-aligned expert I/O buffers
    for (int k = 0; k < N_ACTIVE; k++) {
        posix_memalign((void**)&eng->expert_buf[k], 2*1024*1024, EXPERT_SIZE);
        posix_memalign((void**)&eng->expert_buf_pred[k], 2*1024*1024, EXPERT_SIZE);
    }

    // Initialize expert GPU buffer cache to NULL (populated after SMELT warmup)
    memset(eng->expert_gpu_buf, 0, sizeof(eng->expert_gpu_buf));

    // Initialize gather mode buffers to NULL
    memset(eng->buf_gather_gate_W, 0, sizeof(eng->buf_gather_gate_W));
    memset(eng->buf_gather_gate_s, 0, sizeof(eng->buf_gather_gate_s));
    memset(eng->buf_gather_up_W,   0, sizeof(eng->buf_gather_up_W));
    memset(eng->buf_gather_up_s,   0, sizeof(eng->buf_gather_up_s));
    memset(eng->buf_gather_down_W, 0, sizeof(eng->buf_gather_down_W));
    memset(eng->buf_gather_down_s, 0, sizeof(eng->buf_gather_down_s));
    eng->gather_mode = false;
    // Initialize all pool_pos remapping slots to -1 (not in pool)
    for (int l = 0; l < N_LAYERS; l++)
        for (int e = 0; e < N_EXPERTS; e++)
            eng->smelt_pool_pos[l][e] = -1;

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
    int fd[NUM_IO_THREADS];       // original fd per task (set by dispatcher)
    int claimed_fd[NUM_IO_THREADS]; // fd saved when task is claimed (to preserve original)
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
        // Find and claim an unclaimed task atomically while mutex is held
        for (int i = 0; i < pool->num_tasks; i++) {
            if (pool->fd[i] >= 0) {
                tid = i;
                pool->claimed_fd[i] = pool->fd[i]; // save fd before claiming
                pool->fd[i] = -1;  // mark as claimed immediately
                break;
            }
        }
        pthread_mutex_unlock(&pool->mutex);
        if (tid < 0) continue;

        ssize_t n = pread(pool->claimed_fd[tid], pool->buf[tid], pool->size, pool->offset[tid]);
        (void)n;

        pthread_mutex_lock(&pool->mutex);
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

// Forward declaration for io_pool_dispatch (defined below)
static void io_pool_dispatch(IOPool *pool, int layer_fd, int *expert_ids, int K, uint8_t *buffers[6]);

// ============================================================================
// Expert memory cache — preload experts from SSD into RAM to eliminate I/O
// ============================================================================

int moe_infer_preload_experts(MoEInferEngine *eng, int expert_cache_mb) {
    // Calculate how many experts per layer fit in the budget
    // Each expert is EXPERT_SIZE bytes. Total budget across all layers:
    size_t total_bytes = (size_t)expert_cache_mb * 1024 * 1024;
    int n_per_layer;
    if (expert_cache_mb == 0) {
        // Preload all experts
        n_per_layer = N_EXPERTS;
        total_bytes = (size_t)N_LAYERS * N_EXPERTS * EXPERT_SIZE;
    } else {
        n_per_layer = (int)(total_bytes / ((size_t)N_LAYERS * EXPERT_SIZE));
        if (n_per_layer > N_EXPERTS) n_per_layer = N_EXPERTS;
        if (n_per_layer == 0) {
            fprintf(stderr, "[expert-cache] Budget %dMB too small for even 1 expert per layer (need %luMB)\n",
                expert_cache_mb, (unsigned long)((size_t)N_LAYERS * EXPERT_SIZE / (1024*1024)));
            return 0;
        }
    }

    fprintf(stderr, "[expert-cache] Preloading %d/%d experts per layer (%lu MB total)...\n",
        n_per_layer, N_EXPERTS, (unsigned long)(((size_t)N_LAYERS * n_per_layer * EXPERT_SIZE) / (1024*1024)));
    eng->expert_cache_n_experts = n_per_layer;

    for (int layer = 0; layer < N_LAYERS; layer++) {
        // Allocate lookup table
        eng->expert_mem_cache[layer] = (uint8_t **)calloc(N_EXPERTS, sizeof(uint8_t *));
        if (!eng->expert_mem_cache[layer]) {
            fprintf(stderr, "[expert-cache] OOM allocating lookup table for layer %d\n", layer);
            eng->expert_cache_n_experts = 0;
            return 0;
        }

        // Allocate flat pool for this layer
        size_t pool_bytes = (size_t)n_per_layer * EXPERT_SIZE;
        if (posix_memalign((void**)&eng->expert_mem_pool[layer], 2*1024*1024, pool_bytes) != 0) {
            fprintf(stderr, "[expert-cache] OOM allocating %luMB pool for layer %d\n",
                (unsigned long)(pool_bytes / (1024*1024)), layer);
            eng->expert_cache_n_experts = 0;
            return 0;
        }

        // Read n_per_layer experts starting from 0 (most frequently used in practice)
        // For hash routing layers (0-2), these are exactly the needed experts
        for (int eid = 0; eid < n_per_layer; eid++) {
            uint8_t *dst = eng->expert_mem_pool[layer] + (size_t)eid * EXPERT_SIZE;
            off_t offset = (off_t)eid * EXPERT_SIZE;
            ssize_t n = pread(eng->packed_fd[layer], dst, EXPERT_SIZE, offset);
            if (n != EXPERT_SIZE) {
                fprintf(stderr, "[expert-cache] pread failed: layer=%d expert=%d got=%ld\n", layer, eid, (long)n);
                eng->expert_cache_n_experts = 0;
                return 0;
            }
            eng->expert_mem_cache[layer][eid] = dst;
            eng->smelt_pool_pos[layer][eid] = eid;  // slot == eid for sequential preload
        }
    }

    fprintf(stderr, "[expert-cache] Done. %d experts/layer cached (%lu MB)\n",
        n_per_layer, (unsigned long)(((size_t)N_LAYERS * n_per_layer * EXPERT_SIZE) / (1024*1024)));
    return n_per_layer;
}

// ============================================================================
// SMELT: warmup-based hot expert preloading with routing bias
// ============================================================================

void moe_infer_smelt_init(MoEInferEngine *eng, int warmup_tokens, int n_per_layer, float penalty) {
    memset(eng->routing_counts, 0, sizeof(eng->routing_counts));
    eng->smelt_warmup_tokens = warmup_tokens;
    eng->smelt_n_per_layer   = n_per_layer;
    eng->smelt_tokens_seen   = 0;
    eng->smelt_warmup_done   = false;
    eng->smelt_enabled       = true;
    eng->smelt_penalty       = penalty;
    eng->smelt_in_decode_phase = false;
    fprintf(stderr, "[smelt] Init: warmup=%d tokens, cache=%d experts/layer, penalty=%.0f\n",
        warmup_tokens, n_per_layer, penalty);
}

int moe_infer_smelt_finish_warmup(MoEInferEngine *eng) {
    if (!eng->smelt_enabled || eng->smelt_warmup_done) return eng->expert_cache_n_experts;

    const int n = eng->smelt_n_per_layer;
    if (n <= 0 || n > N_EXPERTS) {
        eng->smelt_warmup_done = true;
        return 0;
    }

    fprintf(stderr, "[smelt] Warmup complete (%d tokens). Preloading experts per layer...\n",
        eng->smelt_tokens_seen);

    // Total memory: hash layers (0-2) cache ALL 256 experts, score layers cache top-N
    size_t total_bytes = 0;
    for (int layer = 0; layer < N_LAYERS; layer++) {
        int n_this_layer = (eng->tid2eid[layer] != NULL) ? N_EXPERTS : n;
        total_bytes += (size_t)n_this_layer * EXPERT_SIZE;
    }
    fprintf(stderr, "[smelt] Total preload: %.1f GB (hash layers: all 256, score layers: top-%d)\n",
        (double)total_bytes / (1024.0*1024.0*1024.0), n);

    // Allocate lookup tables if not already allocated
    for (int layer = 0; layer < N_LAYERS; layer++) {
        if (!eng->expert_mem_cache[layer]) {
            eng->expert_mem_cache[layer] = (uint8_t **)calloc(N_EXPERTS, sizeof(uint8_t *));
            if (!eng->expert_mem_cache[layer]) {
                fprintf(stderr, "[smelt] OOM: lookup table for layer %d\n", layer);
                return 0;
            }
        } else {
            memset(eng->expert_mem_cache[layer], 0, N_EXPERTS * sizeof(uint8_t *));
        }
    }

    int sorted[N_EXPERTS];
    for (int layer = 0; layer < N_LAYERS; layer++) {
        // For hash routing layers (0-2): cache ALL experts (no bias can be applied,
        // so we must ensure every expert that hash routing might select is in RAM).
        // For score-based layers (3-42): cache top-N by routing frequency.
        const bool is_hash_layer = (eng->tid2eid[layer] != NULL);
        const int n_this_layer = is_hash_layer ? N_EXPERTS : n;

        // Build sorted list: for hash layers all experts, for score layers sort by frequency
        for (int i = 0; i < N_EXPERTS; i++) sorted[i] = i;
        if (!is_hash_layer) {
            // Sort by descending routing_counts (simple insertion sort, N=256 is fine)
            for (int i = 1; i < N_EXPERTS; i++) {
                int key = sorted[i];
                uint32_t kc = eng->routing_counts[layer][key];
                int j = i - 1;
                while (j >= 0 && eng->routing_counts[layer][sorted[j]] < kc) {
                    sorted[j+1] = sorted[j]; j--;
                }
                sorted[j+1] = key;
            }
        }

        // Allocate pool for this layer
        if (eng->expert_mem_pool[layer]) {
            free(eng->expert_mem_pool[layer]);
            eng->expert_mem_pool[layer] = NULL;
        }
        size_t pool_bytes = (size_t)n_this_layer * EXPERT_SIZE;
        if (posix_memalign((void**)&eng->expert_mem_pool[layer], 2*1024*1024, pool_bytes) != 0) {
            fprintf(stderr, "[smelt] OOM: pool for layer %d (%lu MB)\n", layer,
                (unsigned long)(pool_bytes / (1024*1024)));
            return 0;
        }

        // Preload experts — also record pool_pos remapping table
        // Initialize all slots to -1 (uncached)
        for (int i = 0; i < N_EXPERTS; i++) eng->smelt_pool_pos[layer][i] = -1;

        int loaded = 0;
        for (int i = 0; i < n_this_layer && i < N_EXPERTS; i++) {
            int eid = sorted[i];
            uint8_t *dst = eng->expert_mem_pool[layer] + (size_t)loaded * EXPERT_SIZE;
            off_t offset = (off_t)eid * EXPERT_SIZE;
            ssize_t bytes = pread(eng->packed_fd[layer], dst, EXPERT_SIZE, offset);
            if (bytes == EXPERT_SIZE) {
                eng->expert_mem_cache[layer][eid] = dst;
                eng->smelt_pool_pos[layer][eid] = loaded;  // record slot position
                loaded++;
            } else {
                fprintf(stderr, "[smelt] pread failed layer=%d eid=%d got=%ld\n", layer, eid, (long)bytes);
            }
        }

        if (layer == 0) {
            fprintf(stderr, "[smelt] L0 (hash): loaded %d/%d experts\n", loaded, n_this_layer);
        } else if (layer == 5) {
            fprintf(stderr, "[smelt] L5 (score) top-%d experts: [", loaded);
            for (int i = 0; i < (loaded < 8 ? loaded : 8); i++)
                fprintf(stderr, "%d(×%u)%s", sorted[i], eng->routing_counts[layer][sorted[i]], i+1<8?",":"");
            fprintf(stderr, "...]\n");
        }
    }

    eng->expert_cache_n_experts = n;  // used as guard in io_pool_dispatch_cached
    eng->smelt_warmup_done = true;
    fprintf(stderr, "[smelt] Done. %.1f GB preloaded, routing bias penalty=%.0f\n",
        (double)total_bytes / (1024.0*1024.0*1024.0), eng->smelt_penalty);

    // === Create persistent GPU MTLBuffer wrappers for all cached experts ===
    // Use MTLResourceStorageModeShared (zero-copy from RAM pool, no extra memory).
    // Private mode would require double the memory (35GB×2 = 70GB > 38GB limit).
    // Shared persistent buffers still eliminate per-call newBufferWithBytesNoCopy overhead.
    {
        id<MTLDevice> d = (id<MTLDevice>)eng->device;
        int n_created = 0;
        for (int layer = 0; layer < N_LAYERS; layer++) {
            if (!eng->expert_mem_cache[layer]) continue;
            for (int eid = 0; eid < N_EXPERTS; eid++) {
                uint8_t *base = eng->expert_mem_cache[layer][eid];
                if (!base) continue;
                eng->expert_gpu_buf[layer][eid][0] = (void *)[d newBufferWithBytesNoCopy:base+GATE_W_OFF length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
                eng->expert_gpu_buf[layer][eid][1] = (void *)[d newBufferWithBytesNoCopy:base+GATE_S_OFF length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                eng->expert_gpu_buf[layer][eid][2] = (void *)[d newBufferWithBytesNoCopy:base+UP_W_OFF   length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
                eng->expert_gpu_buf[layer][eid][3] = (void *)[d newBufferWithBytesNoCopy:base+UP_S_OFF   length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                eng->expert_gpu_buf[layer][eid][4] = (void *)[d newBufferWithBytesNoCopy:base+DOWN_W_OFF length:4194304 options:MTLResourceStorageModeShared deallocator:nil];
                eng->expert_gpu_buf[layer][eid][5] = (void *)[d newBufferWithBytesNoCopy:base+DOWN_S_OFF length:262144  options:MTLResourceStorageModeShared deallocator:nil];
                n_created++;
            }
        }
        fprintf(stderr, "[smelt] Created %d persistent Shared GPU buffers for cached experts\n", n_created * 6);
    }

    // === GPU warmup: touch all expert buffers to establish page-table mappings ===
    // Without this, the first forward pass with any given expert incurs a GPU TLB/page-table
    // setup cost (~100-160ms vs ~12ms warm). A single dummy read per buffer amortizes this.
    // We dispatch a tiny compute pass that reads 1 element from each gate_W buffer.
    // Total data read: n_per_layer × 43 × 4B ≈ negligible; time: <1s for all layers.
    {
        id<MTLDevice> d = (id<MTLDevice>)eng->device;
        id<MTLCommandQueue> q = (id<MTLCommandQueue>)eng->queue;
        fprintf(stderr, "[smelt] GPU warmup: touching all expert buffers...\n");

        // Use a persistent scratch output buffer (1 float, discarded)
        id<MTLBuffer> sink = [d newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

        // For each layer, commit one CB that copies 4 bytes from each expert's gate_W.
        // Use Metal blit encoder (cheapest touch, no kernel needed).
        for (int layer = 0; layer < N_LAYERS; layer++) {
            if (!eng->expert_mem_cache[layer]) continue;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            for (int eid = 0; eid < N_EXPERTS; eid++) {
                void *gbuf = eng->expert_gpu_buf[layer][eid][0];  // gate_W buffer
                if (!gbuf) continue;
                // Copy 4 bytes (one uint32) from the start of this expert's gate_W to sink.
                // This forces the GPU to establish TLB/page-table entry for the buffer.
                [blit copyFromBuffer:(id<MTLBuffer>)gbuf
                       sourceOffset:0
                           toBuffer:sink
                  destinationOffset:0
                               size:4];
            }
            [blit endEncoding];
            [cb commit];
            // Don't waitUntilCompleted per layer — let them pipeline.
            // Wait at the end using a final barrier CB.
        }
        // Final barrier: wait for all warmup CBs to complete
        id<MTLCommandBuffer> barrier = [q commandBuffer];
        [barrier commit];
        [barrier waitUntilCompleted];
        fprintf(stderr, "[smelt] GPU warmup complete — all expert page tables established\n");
    }

    return n;
}

void moe_infer_smelt_set_decode_phase(MoEInferEngine *eng) {
    if (eng && eng->smelt_enabled && !eng->smelt_in_decode_phase) {
        eng->smelt_in_decode_phase = true;
        fprintf(stderr, "[smelt] Decode phase started — SMELT token counting enabled\n");
    }
}

// Background preload thread state
typedef struct {
    MoEInferEngine *eng;
} SmeltPreloadArgs;

static void *smelt_preload_thread(void *arg) {
    SmeltPreloadArgs *a = (SmeltPreloadArgs *)arg;
    moe_infer_smelt_finish_warmup(a->eng);
    free(a);
    return NULL;
}

// Async version: spawn background thread for preloading, return immediately.
// The routing bias will only be active once smelt_warmup_done becomes true.
// Meanwhile, all SSD reads proceed normally (no penalty applied yet).
void moe_infer_smelt_preload_async(MoEInferEngine *eng) {
    if (!eng->smelt_enabled || eng->smelt_warmup_done) return;
    SmeltPreloadArgs *a = (SmeltPreloadArgs *)malloc(sizeof(SmeltPreloadArgs));
    if (!a) { moe_infer_smelt_finish_warmup(eng); return; }
    a->eng = eng;
    pthread_t t;
    if (pthread_create(&t, NULL, smelt_preload_thread, a) != 0) {
        free(a);
        moe_infer_smelt_finish_warmup(eng);
        return;
    }
    pthread_detach(t);
    fprintf(stderr, "[smelt] Async preload started (routing bias inactive until complete)\n");
}

// ============================================================================
// Gather mode: create per-layer full-expert buffers for gatherQmm
// ============================================================================

// Initialize gather mode using the SMELT preloaded expert pool.
// Works with any smelt_n (doesn't require all 256 experts — only K=6 must be cached).
// Creates per-layer NoCopy Metal buffer views over the SMELT RAM pool.
// The gather kernels use pool_pos (slot index) instead of raw expert_id.
// Caller translates: pool_pos = smelt_pool_pos[layer][expert_id] before dispatch.
//
// Returns 1 on success, 0 on failure (SMELT not ready or insufficient cache).
int moe_infer_init_gather_mode(MoEInferEngine *eng) {
    if (!eng->smelt_warmup_done) {
        fprintf(stderr, "[gather] Cannot init gather mode before SMELT warmup\n");
        return 0;
    }

    // Count layers with enough cached experts
    int layers_ok = 0;
    for (int layer = 0; layer < N_LAYERS; layer++) {
        if (!eng->expert_mem_pool[layer]) continue;
        int n_cached = 0;
        for (int eid = 0; eid < N_EXPERTS; eid++) {
            if (eng->smelt_pool_pos[layer][eid] >= 0) n_cached++;
        }
        if (n_cached >= N_ACTIVE) layers_ok++;
    }
    if (layers_ok == 0) {
        fprintf(stderr, "[gather] No layers have >= %d cached experts for gather mode\n", N_ACTIVE);
        return 0;
    }

    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    fprintf(stderr, "[gather] Initializing gather mode (%d/%d layers ready)...\n",
            layers_ok, N_LAYERS);

    // Each layer: create a single NoCopy Metal buffer over the SMELT pool.
    // The pool stores experts in slot order: [slot0][slot1]...[slot(n-1)]
    // Each slot follows packed_experts layout (EXPERT_SIZE bytes):
    //   [GATE_W: 4MB][GATE_S: 256KB][UP_W: 4MB][UP_S: 256KB][DOWN_W: 4MB][DOWN_S: 256KB]
    //
    // Gather kernels use: base = pool + pool_pos * EXPERT_SIZE_U32 + COMPONENT_OFFSET
    // pool_pos comes from smelt_pool_pos[layer][expert_id] (set during SMELT warmup).
    for (int layer = 0; layer < N_LAYERS; layer++) {
        uint8_t *pool = eng->expert_mem_pool[layer];
        if (!pool) continue;

        // Determine actual pool size (max slot + 1) * EXPERT_SIZE
        int n_slots = 0;
        for (int eid = 0; eid < N_EXPERTS; eid++) {
            int s = eng->smelt_pool_pos[layer][eid];
            if (s >= 0 && s + 1 > n_slots) n_slots = s + 1;
        }
        if (n_slots < N_ACTIVE) continue;

        size_t pool_size = (size_t)n_slots * EXPERT_SIZE;
        id<MTLBuffer> pool_buf = [d newBufferWithBytesNoCopy:pool
                                                      length:pool_size
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
        // All 6 "component buffers" alias the same pool (kernel uses internal offsets)
        eng->buf_gather_gate_W[layer] = (void *)pool_buf;
        eng->buf_gather_gate_s[layer] = (void *)pool_buf;
        eng->buf_gather_up_W[layer]   = (void *)pool_buf;
        eng->buf_gather_up_s[layer]   = (void *)pool_buf;
        eng->buf_gather_down_W[layer] = (void *)pool_buf;
        eng->buf_gather_down_s[layer] = (void *)pool_buf;

        if (layer < 3 || layer == 42) {
            fprintf(stderr, "[gather] L%d: %d slots, %.1f MB NoCopy buffer\n",
                    layer, n_slots, (double)pool_size/(1024.0*1024.0));
        }
    }

    eng->gather_mode = true;
    fprintf(stderr, "[gather] Ready. Expert IDs remapped via smelt_pool_pos table.\n");
    return 1;
}

// Dispatch K parallel preads — uses memory cache if available, falls back to SSD.
static void io_pool_dispatch_cached(MoEInferEngine *eng, IOPool *pool, int layer, int *expert_ids, int K,
                                    uint8_t *buffers[6]) {
    if (eng->expert_cache_n_experts > 0 && eng->expert_mem_cache[layer]) {
        // Check if all requested experts are in cache
        int all_cached = 1;
        for (int k = 0; k < K; k++) {
            int eid = expert_ids[k];
            if (eid < 0 || eid >= N_EXPERTS || !eng->expert_mem_cache[layer][eid]) {
                all_cached = 0;
                break;
            }
        }
        if (all_cached) {
            // Zero-copy: point buffer pointers directly at cached data
            for (int k = 0; k < K; k++) {
                buffers[k] = eng->expert_mem_cache[layer][expert_ids[k]];
            }
            return;
        }
        // Partial hit: fill cached ones directly, pread the rest
        for (int k = 0; k < K; k++) {
            int eid = expert_ids[k];
            if (eid >= 0 && eid < N_EXPERTS && eng->expert_mem_cache[layer][eid]) {
                buffers[k] = eng->expert_mem_cache[layer][eid];
            } else {
                // This expert not in cache — must pread into expert_buf[k]
                // We need a writable buffer; reset to the pre-allocated ones
                buffers[k] = eng->expert_buf[k];  // fallback to pread buffer
            }
        }
        // For uncached experts, do selective pread
        uint8_t *fallback_bufs[6];
        int fallback_ids[6];
        int fallback_k_map[6];  // maps fallback_idx → original k
        int n_fallback = 0;
        for (int k = 0; k < K; k++) {
            int eid = expert_ids[k];
            if (!(eid >= 0 && eid < N_EXPERTS && eng->expert_mem_cache[layer][eid])) {
                fallback_ids[n_fallback] = eid;
                fallback_bufs[n_fallback] = eng->expert_buf[k];
                fallback_k_map[n_fallback] = k;
                n_fallback++;
            }
        }
        if (n_fallback > 0) {
            io_pool_dispatch(pool, eng->packed_fd[layer], fallback_ids, n_fallback, fallback_bufs);
            for (int i = 0; i < n_fallback; i++) {
                buffers[fallback_k_map[i]] = fallback_bufs[i];
            }
        }
        return;
    }
    // No cache: regular pread
    io_pool_dispatch(pool, eng->packed_fd[layer], expert_ids, K, buffers);
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

// Async variant: dispatch without waiting. Call io_pool_wait() to synchronize.
static void io_pool_dispatch_start(IOPool *pool, int layer_fd, int *expert_ids, int K,
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
    pthread_mutex_unlock(&pool->mutex);
    // returns immediately — call io_pool_wait() before using buffers
}

static void io_pool_wait(IOPool *pool) {
    pthread_mutex_lock(&pool->mutex);
    while (pool->tasks_done < pool->num_tasks) {
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
    const int gw_off = GATE_W_OFF, gs_off = GATE_S_OFF;
    const int uw_off = UP_W_OFF, us_off = UP_S_OFF;
    const int dw_off = DOWN_W_OFF, ds_off = DOWN_S_OFF;
    const int time_en = (getenv("NATIVE_TIME_LAYERS") != NULL);
    double t0=0, t1=0, t2=0, t3=0;
    if (time_en) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec*1e9+ts.tv_nsec; }
    (void)t3;

    // Helper: get or create MTLBuffer for a given expert slot.
    // Uses persistent GPU buffer if available (post-SMELT), else creates on-the-fly.
    #define EXPERT_BUF(layer, eid, slot, ptr, len) \
        (eng->expert_gpu_buf[layer][eid][slot] ? \
            (id<MTLBuffer>)eng->expert_gpu_buf[layer][eid][slot] : \
            [d newBufferWithBytesNoCopy:(ptr) length:(len) options:MTLResourceStorageModeShared deallocator:nil])

    // Step 1 + 2: expert gate+up+SwiGLU and down_proj (6 separate per-expert kernels)
    // Note: fused_6expert_gate_up was tried but is slower due to reduced GPU parallelism
    // (serial expert loop in kernel vs GPU scheduling multiple dispatches in parallel).
    {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBuffer];

        if (eng->gather_mode && eng->buf_gather_gate_W[layer_idx]) {
            // === GATHER MODE: gatherQmm-equivalent ===
            // Single dispatch covering all K experts simultaneously.
            // GPU reads only selected experts' rows from the full SMELT pool.
            // Translate expert_ids → pool_pos (slot indices in SMELT pool)
            uint32_t *eid_buf = (uint32_t *)[(id<MTLBuffer>)eng->buf_gather_expert_ids contents];
            int all_in_pool = 1;
            for (int k = 0; k < K; k++) {
                int pos = eng->smelt_pool_pos[layer_idx][expert_ids[k]];
                if (pos < 0) { all_in_pool = 0; break; }
                eid_buf[k] = (uint32_t)pos;  // write pool slot, not raw expert_id
            }

            if (!all_in_pool) goto separate_mode;  // fall back if any expert not in pool

            // Encoder 1: gather gate+up+SwiGLU (K experts in parallel via Y dimension)
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_gather_gate_up];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_gate_W[layer_idx] offset:0 atIndex:0]; // pool
                [enc setBuffer:eng->buf_normed offset:0 atIndex:1];                                    // x
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_mid offset:0 atIndex:2];                 // out [K×INT]
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_expert_ids offset:0 atIndex:3];          // eids
                uint k_val = K;
                [enc setBytes:&k_val length:4 atIndex:4];
                // Dispatch: (INTERMEDIATE/8, K, 1) threadgroups × 256 threads
                [enc dispatchThreadgroups:MTLSizeMake(INTERMEDIATE/8, K, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [enc endEncoding];
            }

            // Encoder 2: gather down_proj (K experts in parallel)
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_gather_down];
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_gate_W[layer_idx] offset:0 atIndex:0]; // pool
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_mid offset:0 atIndex:1];                // x_mid
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_out offset:0 atIndex:2];                // out [K×DIM]
                [enc setBuffer:(id<MTLBuffer>)eng->buf_gather_expert_ids offset:0 atIndex:3];
                uint k_val = K;
                [enc setBytes:&k_val length:4 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(DIM/8, K, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [enc endEncoding];
            }
        } else {
            separate_mode:;
            // === SEPARATE MODE: 6 per-expert dispatches ===
            for (int k = 0; k < K; k++) {
                char *base = (char *)expert_bufs[k]; int eid = expert_ids[k];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:eng->pipe_gate_up_swiglu];
                [enc setBuffer:EXPERT_BUF(layer_idx,eid,0,base+gw_off,4194304) offset:0 atIndex:0];
                [enc setBuffer:EXPERT_BUF(layer_idx,eid,1,base+gs_off,262144)  offset:0 atIndex:1];
                [enc setBuffer:EXPERT_BUF(layer_idx,eid,2,base+uw_off,4194304) offset:0 atIndex:2];
                [enc setBuffer:EXPERT_BUF(layer_idx,eid,3,base+us_off,262144)  offset:0 atIndex:3];
                [enc setBuffer:eng->buf_normed offset:0 atIndex:4];
                [enc setBuffer:eng->buf_expert_mid[k] offset:0 atIndex:5];
                uint od=INTERMEDIATE,id_=DIM,gs=32;
                [enc setBytes:&od length:4 atIndex:6]; [enc setBytes:&id_ length:4 atIndex:7]; [enc setBytes:&gs length:4 atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(INTERMEDIATE/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [enc endEncoding];
            }
            for (int k = 0; k < K; k++) {
                char *base = (char *)expert_bufs[k]; int eid = expert_ids[k];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:eng->pipe_dequant_matvec];
                [enc setBuffer:EXPERT_BUF(layer_idx,eid,4,base+dw_off,4194304) offset:0 atIndex:0];
                [enc setBuffer:EXPERT_BUF(layer_idx,eid,5,base+ds_off,262144)  offset:0 atIndex:1];
                [enc setBuffer:eng->buf_expert_mid[k] offset:0 atIndex:2];
                [enc setBuffer:eng->buf_expert_out[k] offset:0 atIndex:3];
                uint od=DIM,id_=INTERMEDIATE,gs=32;
                [enc setBytes:&od length:4 atIndex:4]; [enc setBytes:&id_ length:4 atIndex:5]; [enc setBytes:&gs length:4 atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(DIM/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [enc endEncoding];
            }
        } // end if gather_mode / else separate mode

        // Step 3: blit outputs into contiguous buffer for combine
        // In gather mode: buf_gather_out already holds [K×DIM] contiguous
        // In separate mode: blit from 6 separate buffers
        if (eng->gather_mode && eng->buf_gather_gate_W[layer_idx]) {
            // Gather mode: buf_gather_out is already [K×DIM] contiguous
            // Copy it to buf_expert_contiguous so combine kernel works uniformly
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromBuffer:(id<MTLBuffer>)eng->buf_gather_out
                   sourceOffset:0
                       toBuffer:(id<MTLBuffer>)eng->buf_expert_contiguous
              destinationOffset:0
                           size:(size_t)K*DIM*sizeof(float)];
            [blit endEncoding];
        } else {
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            id<MTLBuffer> contiguous = (id<MTLBuffer>)eng->buf_expert_contiguous;
            for (int k = 0; k < K; k++) {
                [blit copyFromBuffer:(id<MTLBuffer>)eng->buf_expert_out[k]
                       sourceOffset:0
                           toBuffer:contiguous
                  destinationOffset:(size_t)k * DIM * sizeof(float)
                               size:(size_t)DIM * sizeof(float)];
            }
            [blit endEncoding];
        }

        // Step 4: moe_combine in the SAME CB
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:eng->pipe_moe_combine];
            id<MTLBuffer> weights_buf = [d newBufferWithBytes:expert_weights length:K*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> zero_resid  = [d newBufferWithLength:DIM*sizeof(float) options:MTLResourceStorageModeShared];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_expert_contiguous offset:0 atIndex:0];
            [enc setBuffer:weights_buf offset:0 atIndex:1];
            [enc setBuffer:zero_resid  offset:0 atIndex:2];
            [enc setBuffer:eng->buf_hidden offset:0 atIndex:3];
            uint kv = K, hd = DIM;
            [enc setBytes:&kv length:4 atIndex:4];
            [enc setBytes:&hd length:4 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((DIM+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
        }

        if (time_en) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); t1 = ts.tv_sec*1e9+ts.tv_nsec; }
        [cb commit]; [cb waitUntilCompleted];
        if (time_en) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); t2 = ts.tv_sec*1e9+ts.tv_nsec;
            fprintf(stderr, "[MOE-TIME] encode=%.2fms gpu=%.2fms\n", (t1-t0)/1e6, (t2-t1)/1e6); }
    } // end expert block

    #undef EXPERT_BUF
    return 0;
}

// ============================================================================
// Top-K routing on CPU (softmax + topK)
// ============================================================================

// MLX routing: sqrtsoftplus scoring, topK selection, L1-normalize, scale by route_scale.
// Matches DSV4Gate.forward (scoring_func=sqrtsoftplus, norm_topk_prob=true, route_scale=1.5).
// bias: optional [N_EXPERTS] e_score_correction_bias (NULL for hash layers 0-2).
// cache_mask: optional [N_EXPERTS] bool — if non-NULL and smelt_warmup_done, adds penalty
//             to uncached experts to steer routing toward cached ones (SMELT).
static void cpu_moe_route(const float *logits, const float *bias, int n, int K,
                          int *out_indices, float *out_weights,
                          const uint8_t *const *cache_ptr, float smelt_penalty) {
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
    if (bias != NULL || cache_ptr != NULL) {
        biased = (float *)alloca(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            biased[i] = scores[i];
            if (bias) biased[i] += bias[i];
            // SMELT: penalize uncached experts to steer routing toward cached ones
            if (cache_ptr && !cache_ptr[i]) biased[i] -= smelt_penalty;
        }
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

    @autoreleasepool {

    // Build MlaPipes view over the engine's pipelines.
    MlaPipes P;
    P.dev = d;
    P.queue = (id<MTLCommandQueue>)eng->queue;
    P.dequant_matvec_affine = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine;
    P.rms_norm_rows = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows;
    P.rope_tail_interleaved = (id<MTLComputePipelineState>)eng->pipe_rope_tail;
    P.mla_sdpa_decode = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa;
    P.mla_sdpa_decode_f16 = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_f16;
    P.matvec_f32 = (id<MTLComputePipelineState>)eng->pipe_matvec;
    // F16 precision chain (validated correct)
    P.dequant_matvec_affine_f16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_f16out;
    P.rms_norm_rows_f16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_f16out;
    P.dequant_matvec_affine_f16in_f16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_f16in_f16out;
    P.rms_norm_rows_f16in_f16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_f16in_f16out;
    P.rope_tail_interleaved_f16 = (id<MTLComputePipelineState>)eng->pipe_rope_tail_f16;
    P.matvec_f32_f16in = (id<MTLComputePipelineState>)eng->pipe_matvec_f32_f16in;
    P.mla_sdpa_decode_f16in_f16out = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_f16in_f16out;
    // BF16 precision chain — matches MLX training precision
    P.dequant_matvec_affine_bf16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16out;
    P.rms_norm_rows_bf16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_bf16out;
    P.dequant_matvec_affine_bf16in_bf16out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_bf16out;
    P.rms_norm_rows_bf16in_bf16out = (id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_bf16in_bf16out;
    P.rope_tail_interleaved_bf16 = (id<MTLComputePipelineState>)eng->pipe_rope_tail_bf16;
    P.matvec_f32_bf16in = (id<MTLComputePipelineState>)eng->pipe_matvec_f32_bf16in;
    P.mla_sdpa_decode_bfloat = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_bfloat;
    P.dequant_matvec_affine_bf16in_f32out = (id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine_bf16in_f32out;
    P.mla_sdpa_prefill_bfloat = (id<MTLComputePipelineState>)eng->pipe_mla_sdpa_prefill_bfloat;
    P.bf16_to_f16_row = (id<MTLComputePipelineState>)eng->pipe_bf16_to_f16_row;

    // Large scratch buffers are static (forward_layer runs serially).
    static float attn_input[DIM], normed[DIM], attn_out[DIM];
    static float ffn_input[DIM], ffn_out[DIM];
    static uint16_t normed_bf16_direct[DIM];
    float post[MHC_MULT], comb[MHC_MULT * MHC_MULT];
    bool shared_done_early = false;  // set to true if shared expert ran during async pread overlap

    // === Attention sublayer — GPU mhc_pre_f16 (matches MLX bf16 computation) ===
    MhcWeights ahc = { eng->attn_hc_fn[layer], eng->attn_hc_base[layer], eng->attn_hc_scale[layer] };
    if (layer == 0 && getenv("MF_DBG")) {
        double rn=0; for(int z=0;z<MHC_MULT*DIM;z++) rn+=(double)residual[z]*residual[z];
        fprintf(stderr, "[mf-dbg] L0 pos=%d in residual norm=%.4f res[0]=%.6f res[DIM]=%.6f\n", pos, sqrt(rn), residual[0], residual[DIM]);
        const char *dd = getenv("DSV4_DUMP_DIR");
        if (dd) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/L0_residual_metal.bin", dd);
            FILE *f = fopen(path, "wb"); if (f) { fwrite(residual, sizeof(float), MHC_MULT*DIM, f); fclose(f); }
        }
    }
    // Use GPU mhc_pre_gpu — reuse persistent scratch buffers (eliminates alloc/dealloc overhead)
    // === CB-A: mhc_pre(attn) + input_RMSNorm merged into ONE command buffer (2 encoders, 1 wait) ===
    // Path B Step 1: buf_residual_gpu is the source-of-truth for residual.
    // No memcpy needed — embed() already wrote to buf_residual_gpu.
    // Subsequent layers: mhc_post writes back to buf_residual_gpu at end of each layer.
    {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)eng->queue commandBuffer];

        // Encoder 1: mhc_pre(attn) — reads buf_residual_gpu directly (no CPU memcpy)
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_mhc_pre_bfloat];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_attn_hc_fn[layer]    offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_attn_hc_base[layer]  offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_attn_hc_scale[layer] offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu         offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_attn_in          offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_weights     offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_comb_weights     offset:0 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
        }

        // Encoder 2: input_RMSNorm (reads buf_mhc_attn_in written by encoder 1)
        {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_bf16out];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_attn_in           offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_input_norm_gpu[layer] offset:0 atIndex:1];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_attn_norm_bf16    offset:0 atIndex:2];
            uint rd = DIM; float eps = 1e-6f; uint hw = 1;
            [e setBytes:&rd length:4 atIndex:3]; [e setBytes:&eps length:4 atIndex:4]; [e setBytes:&hw length:4 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }

        [cb commit];

        // === Deferred cb3 overlap: wait for previous layer's cb3 while CB-A runs on GPU ===
        // GPU queue is serial: it runs cb3(N-1) then CB-A(N) automatically.
        // CPU waits here so cb3 wait time is hidden inside CB-A GPU execution time.
        // No performance regression vs sync wait: cb3(N-1) finishes before CB-A(N) needs its output.
        if (eng->deferred.active && eng->deferred.cmd_experts) {
            [(id<MTLCommandBuffer>)eng->deferred.cmd_experts waitUntilCompleted];
            [(id<MTLCommandBuffer>)eng->deferred.cmd_experts release];
            eng->deferred.cmd_experts = NULL;
            eng->deferred.active = false;
            // CPU residual readback (buf_residual_gpu already correct on GPU via enc3)
            uint16_t *res_out = (uint16_t *)[(id<MTLBuffer>)eng->buf_mhc_ffn_post_out contents];
            for (int i = 0; i < MHC_MULT * DIM; i++) {
                uint32_t u = ((uint32_t)res_out[i]) << 16;
                memcpy(&residual[i], &u, 4);
            }
        }

        [cb waitUntilCompleted];

        memcpy(attn_input, [(id<MTLBuffer>)eng->buf_mhc_attn_in     contents], DIM * sizeof(float));
        memcpy(post,       [(id<MTLBuffer>)eng->buf_mhc_post_weights contents], MHC_MULT * sizeof(float));
        memcpy(comb,       [(id<MTLBuffer>)eng->buf_mhc_comb_weights contents], MHC_MULT * MHC_MULT * sizeof(float));
        uint16_t *bf16_out = (uint16_t *)[(id<MTLBuffer>)eng->buf_mhc_attn_norm_bf16 contents];
        memcpy(normed_bf16_direct, bf16_out, DIM * sizeof(uint16_t));
        for (int i = 0; i < DIM; i++) {
            uint32_t u = ((uint32_t)normed_bf16_direct[i]) << 16;
            memcpy(&normed[i], &u, 4);
        }
    }

    // Compressor step (before attention, for layers with compress_ratio > 0)
    if (eng->compress_ratio[layer] > 0) {
        moe_infer_compressor_step(eng, layer, pos, normed);
    }

    // Indexer step (ratio=4 layers only — select which comp blocks to attend)
    static bool comp_allowed[MAX_COMP_BLOCKS];
    bool has_comp_selection = false;
    if (eng->compress_ratio[layer] == 4 && eng->comp_state[layer].n_comp > 0) {
        has_comp_selection = moe_infer_indexer_step(eng, layer, pos, normed, NULL, comp_allowed);
    }

    // MLA attention (decode, single token -> cache_len from kv_cache)
    {
        KVCache *kvc = &eng->kv_cache[layer];
        if (!kvc->kv) {
            // Allocate KV cache as MTLBuffer (Shared mode) so it's GPU-accessible.
            // This enables blit-based KV update within CB1, eliminating one GPU sync per layer.
            size_t kvc_size = (size_t)MAX_SEQ_LEN * KV_LORA_RANK * sizeof(uint16_t);
            id<MTLBuffer> kvbuf = [(id<MTLDevice>)eng->device newBufferWithLength:kvc_size
                                       options:MTLResourceStorageModeShared];
            kvc->kv_gpu_buf = (void *)kvbuf;
            kvc->kv = (uint16_t *)[kvbuf contents];
            memset(kvc->kv, 0, kvc_size);
            kvc->len = 0;
        }
        kvc->len += 1;

        // Convert normed (f32) to bf16 for the attention call
        uint16_t normed_bf16[DIM];
        for (int i = 0; i < DIM; i++) {
            // f32 -> bf16: take upper 16 bits of float32
            uint32_t u; memcpy(&u, &normed[i], 4);
            normed_bf16[i] = (uint16_t)(u >> 16);
        }

        const uint32_t n_comp = eng->comp_state[layer].n_comp;
        // Use mixed attention (raw SWA KV + compressed KV blocks) only when the
        // raw KV cache exceeds the sliding window. For short sequences (all tokens
        // fit in SWA_WINDOW), MLX does not compress and uses plain attention — we
        // must do the same to match numerics.
        const int use_comp = (n_comp > 0 && kvc->len > SWA_WINDOW);
        const int tl_attn = (getenv("NATIVE_TIME_LAYERS") != NULL);
        double ta0=0, ta1=0;
        if (tl_attn) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ta0 = ts.tv_sec*1e9+ts.tv_nsec; }
        if (use_comp) {
            // Decode x to f32 for the mixed attention path
            float x_f32[DIM];
            for (int i = 0; i < DIM; i++) {
                uint32_t u = ((uint32_t)normed_bf16_direct[i]) << 16;
                memcpy(&x_f32[i], &u, 4);
            }
            const bool *allowed = has_comp_selection ? comp_allowed : NULL;
            mla_attention_decode_mixed(&P, &eng->attn[layer], x_f32, kvc->kv, kvc->len,
                                       pos, eng->comp_state[layer].comp_kv, (int)n_comp,
                                       allowed, attn_out);
        } else {
            // BF16 attention using directly-computed bf16 normed
            mla_attention_decode_bf16(&P, &eng->attn[layer], normed_bf16_direct, kvc->kv, kvc->len, pos, attn_out, kvc->kv_gpu_buf);
        }
        // Truncate attn_out to bf16
        for (int i = 0; i < DIM; i++) {
            uint32_t u; memcpy(&u, &attn_out[i], 4); u &= 0xFFFF0000U; memcpy(&attn_out[i], &u, 4);
        }
        if (tl_attn) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ta1 = ts.tv_sec*1e9+ts.tv_nsec;
            if (layer == 0) fprintf(stderr, "[ATTN-TIME] L0 pos=%d attn=%.1fms\n", pos, (ta1-ta0)/1e6); }
    }

    // mHC post (attn) — Path B: read residual from buf_residual_gpu (no CPU readback)
    {
        // Convert attn_out to bf16
        uint16_t *attn_out_buf = (uint16_t *)[(id<MTLBuffer>)eng->buf_mhc_attn_out_bf16 contents];
        for (int i = 0; i < DIM; i++) { uint32_t u; memcpy(&u, &attn_out[i], 4); attn_out_buf[i] = (uint16_t)(u >> 16); }
        memcpy([(id<MTLBuffer>)eng->buf_mhc_post_weights contents], post, MHC_MULT*sizeof(float));
        memcpy([(id<MTLBuffer>)eng->buf_mhc_comb_weights contents], comb, MHC_MULT*MHC_MULT*sizeof(float));
        id<MTLCommandBuffer> cb2 = [(id<MTLCommandQueue>)eng->queue commandBuffer];

        // Encoder 1: f32→bf16 on GPU (residual is already in buf_residual_gpu as f32)
        {
            id<MTLComputeCommandEncoder> e = [cb2 computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_f32_to_bf16_vec];
            [e setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu    offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_res_bf16_in offset:0 atIndex:1];
            uint n = MHC_MULT * DIM;
            [e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(MHC_MULT*DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }

        // Encoder 2: mhc_post(attn)
        {
            id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoder];
            [enc2 setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_mhc_post_bfloat];
            [enc2 setBuffer:(id<MTLBuffer>)eng->buf_mhc_attn_out_bf16  offset:0 atIndex:0];
            [enc2 setBuffer:(id<MTLBuffer>)eng->buf_mhc_res_bf16_in    offset:0 atIndex:1];
            [enc2 setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_weights   offset:0 atIndex:2];
            [enc2 setBuffer:(id<MTLBuffer>)eng->buf_mhc_comb_weights   offset:0 atIndex:3];
            [enc2 setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_res_out   offset:0 atIndex:4];
            uint hc2 = MHC_MULT, dim2 = DIM;
            [enc2 setBytes:&hc2 length:4 atIndex:5]; [enc2 setBytes:&dim2 length:4 atIndex:6];
            [enc2 dispatchThreads:MTLSizeMake(DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc2 endEncoding];
        }

        // Encoder 3: bf16→f32, write back to buf_residual_gpu
        {
            id<MTLComputeCommandEncoder> e = [cb2 computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_bf16_to_f32_vec];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_res_out offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu     offset:0 atIndex:1];
            uint n = MHC_MULT * DIM;
            [e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(MHC_MULT*DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }

        [cb2 commit]; [cb2 waitUntilCompleted];

        // CPU readback for downstream use (CPU-side routing, compressor, debug)
        uint16_t *res2_out = (uint16_t *)[(id<MTLBuffer>)eng->buf_mhc_post_res_out contents];
        for (int i = 0; i < MHC_MULT * DIM; i++) { uint32_t u = ((uint32_t)res2_out[i]) << 16; memcpy(&residual[i], &u, 4); }
    }

    // === CMD2 — mhc_pre(ffn) + ffn_RMSNorm + routing_gate in single CB ===
    // Path B: read residual from buf_residual_gpu directly (no CPU memcpy needed).
    {
        id<MTLCommandBuffer> cmd2 = [(id<MTLCommandQueue>)eng->queue commandBuffer];

        // Encoder 1: mhc_pre(ffn) — reads buf_residual_gpu directly
        {
            id<MTLComputeCommandEncoder> enc = [cmd2 computeCommandEncoder];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_mhc_pre_bfloat];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_ffn_hc_fn[layer]    offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_ffn_hc_base[layer]  offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_ffn_hc_scale[layer] offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu        offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_in          offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_weights    offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_comb_weights    offset:0 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
        }
        // Encoder 2: ffn_RMSNorm
        {
            id<MTLComputeCommandEncoder> e = [cmd2 computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_rms_norm_rows_bf16out];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_in           offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_attn_norm_gpu[layer] offset:0 atIndex:1];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_norm_bf16    offset:0 atIndex:2];
            uint rd = DIM; float eps = 1e-6f; uint hw = 1;
            [e setBytes:&rd length:4 atIndex:3]; [e setBytes:&eps length:4 atIndex:4]; [e setBytes:&hw length:4 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }
        // Encoder 3: routing_gate
        if (eng->gate_proj[layer]) {
            id<MTLComputeCommandEncoder> enc = [cmd2 computeCommandEncoder];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_matvec_f32_bf16in];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_gate_proj_gpu[layer] offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_norm_bf16    offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)eng->buf_routing_scores        offset:0 atIndex:2];
            uint od=N_EXPERTS, id_=DIM;
            [enc setBytes:&od length:4 atIndex:3]; [enc setBytes:&id_ length:4 atIndex:4];
            [enc dispatchThreads:MTLSizeMake(N_EXPERTS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
        }
        [cmd2 commit]; [cmd2 waitUntilCompleted];

        // Read back results
        memcpy(ffn_input, [(id<MTLBuffer>)eng->buf_mhc_ffn_in      contents], DIM * sizeof(float));
        memcpy(post,      [(id<MTLBuffer>)eng->buf_mhc_post_weights contents], MHC_MULT * sizeof(float));
        memcpy(comb,      [(id<MTLBuffer>)eng->buf_mhc_comb_weights contents], MHC_MULT * MHC_MULT * sizeof(float));
        uint16_t *bf16_out = (uint16_t *)[(id<MTLBuffer>)eng->buf_mhc_ffn_norm_bf16 contents];
        memcpy(normed_bf16_direct, bf16_out, DIM * sizeof(uint16_t));
        for (int i = 0; i < DIM; i++) {
            uint32_t u = ((uint32_t)bf16_out[i]) << 16;
            memcpy(&normed[i], &u, 4);
        }
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

    // Routing scores ready in buf_routing_scores (written by CMD2 encoder 3)
    float *scores = (float *)[(id<MTLBuffer>)eng->buf_routing_scores contents];
    if (!eng->gate_proj[layer]) {
        for (int i = 0; i < N_EXPERTS; i++) scores[i] = (float)(N_EXPERTS - i);
    } else {
        // Truncate to bf16 to match MLX's bf16 gate computation
        for (int i = 0; i < N_EXPERTS; i++) {
            uint32_t u; memcpy(&u, &scores[i], 4); u &= 0xFFFF0000U; memcpy(&scores[i], &u, 4);
        }
    }

    int expert_ids[N_ACTIVE];
    float expert_weights[N_ACTIVE];
    const bool use_hash_routing = (eng->tid2eid[layer] != NULL && eng->current_token_id >= 0);
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
        // Determine SMELT cache mask: only apply penalty after warmup
        const uint8_t *const *smelt_cache = NULL;
        if (eng->smelt_enabled && eng->smelt_warmup_done && eng->expert_mem_cache[layer]) {
            smelt_cache = (const uint8_t *const *)eng->expert_mem_cache[layer];
        }
        cpu_moe_route(scores, eng->gate_bias[layer], N_EXPERTS, N_ACTIVE, expert_ids, expert_weights,
                      smelt_cache, eng->smelt_penalty);
    }
    // SMELT warmup: accumulate routing statistics (only for score-based routing, not hash)
    if (eng->smelt_enabled && !eng->smelt_warmup_done && !use_hash_routing) {
        for (int k = 0; k < N_ACTIVE; k++) {
            int eid = expert_ids[k];
            if (eid >= 0 && eid < N_EXPERTS) {
                eng->routing_counts[layer][eid]++;
            }
        }
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
    //
    // Async pread overlap: dispatch pread immediately after routing, then do
    // shared expert (GPU) and cb3 (GPU) while SSD I/O runs in background.
    // On cache hit (SMELT), io_pool_dispatch_cached returns instantly anyway.
    {
        float *bn = (float *)[(id<MTLBuffer>)eng->buf_normed contents];
        memcpy(bn, normed, DIM * sizeof(float));
        IOPool *io = (IOPool *)eng->io_pool;
        const int tl = (getenv("NATIVE_TIME_LAYERS") != NULL);
        double ti0=0, ti1=0, ti2=0;
        if (tl) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ti0 = ts.tv_sec*1e9+ts.tv_nsec; }

        // Phase 1: determine buffers (cache hit = zero-copy, no I/O needed)
        uint8_t *expert_data[6];
        for (int k = 0; k < N_ACTIVE; k++) expert_data[k] = eng->expert_buf[k];
        int all_cached = 0;

        if (eng->expert_cache_n_experts > 0 && eng->expert_mem_cache[layer]) {
            // Check cache — if all hit, no pread needed
            all_cached = 1;
            for (int k = 0; k < N_ACTIVE; k++) {
                int eid = expert_ids[k];
                if (eid < 0 || eid >= N_EXPERTS || !eng->expert_mem_cache[layer][eid]) {
                    all_cached = 0; break;
                }
            }
            if (all_cached) {
                for (int k = 0; k < N_ACTIVE; k++)
                    expert_data[k] = eng->expert_mem_cache[layer][expert_ids[k]];
            }
        }

        if (!all_cached) {
            // Async pread start: dispatch immediately, don't wait yet.
            // Shared expert GPU work below will overlap with SSD I/O.
            if (eng->expert_cache_n_experts > 0 && eng->expert_mem_cache[layer]) {
                // Partial cache hit: handle via cached dispatch (sync for misses)
                io_pool_dispatch_cached(eng, io, layer, expert_ids, N_ACTIVE, expert_data);
                all_cached = 1; // treat as sync-done
            } else {
                // No cache: async pread, wait later
                io_pool_dispatch_start(io, eng->packed_fd[layer], expert_ids, N_ACTIVE, expert_data);
            }
        }

        if (tl) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ti1 = ts.tv_sec*1e9+ts.tv_nsec; }

        // Shared expert is moved BEFORE moe_forward_layer when pread is async,
        // so GPU shared expert work overlaps with SSD I/O.
        // Results are accumulated into ffn_out after both complete.
        float shared_out_tmp[DIM];
        memset(shared_out_tmp, 0, sizeof(shared_out_tmp));
        bool shared_done_early = false;

        if (!all_cached && eng->shared[layer].gate.packed != NULL) {
            // Run shared expert on GPU now (overlaps with async pread)
            const int SE_GS = 64;
            const int SE_NG_GU = DIM / SE_GS;
            const int SE_NG_D  = INTERMEDIATE / SE_GS;
            memcpy([(id<MTLBuffer>)eng->buf_mhc_attn_in contents], normed, DIM*sizeof(float));
            id<MTLBuffer> bx    = (id<MTLBuffer>)eng->buf_mhc_attn_in;
            id<MTLBuffer> bgate = (id<MTLBuffer>)eng->buf_hidden;
            id<MTLBuffer> bup   = (id<MTLBuffer>)eng->buf_h_mid;
            id<MTLBuffer> bdown = (id<MTLBuffer>)eng->buf_attn_out;
            {
                id<MTLCommandBuffer> cb_se = [P.queue commandBuffer];
                // gate
                { const QuantWeight *qw = &eng->shared[layer].gate;
                  id bw=[d newBufferWithBytesNoCopy:(void*)qw->packed length:(size_t)qw->out_dim*(qw->in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
                  id bs=[d newBufferWithBytes:(void*)qw->scales length:(size_t)qw->out_dim*SE_NG_GU*sizeof(float) options:MTLResourceStorageModeShared];
                  id bb=[d newBufferWithBytes:(void*)qw->biases length:(size_t)qw->out_dim*SE_NG_GU*sizeof(float) options:MTLResourceStorageModeShared];
                  id<MTLComputeCommandEncoder> e=[cb_se computeCommandEncoder];
                  [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
                  [e setBuffer:bw offset:0 atIndex:0];[e setBuffer:bs offset:0 atIndex:1];[e setBuffer:bb offset:0 atIndex:2];
                  [e setBuffer:bx offset:0 atIndex:3];[e setBuffer:bgate offset:0 atIndex:4];
                  uint od=qw->out_dim,id_=qw->in_dim,gs=SE_GS;
                  [e setBytes:&od length:4 atIndex:5];[e setBytes:&id_ length:4 atIndex:6];[e setBytes:&gs length:4 atIndex:7];
                  [e dispatchThreads:MTLSizeMake(qw->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding]; }
                // up
                { const QuantWeight *qw = &eng->shared[layer].up;
                  id bw=[d newBufferWithBytesNoCopy:(void*)qw->packed length:(size_t)qw->out_dim*(qw->in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
                  id bs=[d newBufferWithBytes:(void*)qw->scales length:(size_t)qw->out_dim*SE_NG_GU*sizeof(float) options:MTLResourceStorageModeShared];
                  id bb=[d newBufferWithBytes:(void*)qw->biases length:(size_t)qw->out_dim*SE_NG_GU*sizeof(float) options:MTLResourceStorageModeShared];
                  id<MTLComputeCommandEncoder> e=[cb_se computeCommandEncoder];
                  [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
                  [e setBuffer:bw offset:0 atIndex:0];[e setBuffer:bs offset:0 atIndex:1];[e setBuffer:bb offset:0 atIndex:2];
                  [e setBuffer:bx offset:0 atIndex:3];[e setBuffer:bup offset:0 atIndex:4];
                  uint od=qw->out_dim,id_=qw->in_dim,gs=SE_GS;
                  [e setBytes:&od length:4 atIndex:5];[e setBytes:&id_ length:4 atIndex:6];[e setBytes:&gs length:4 atIndex:7];
                  [e dispatchThreads:MTLSizeMake(qw->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding]; }
                // swiglu
                { id<MTLComputeCommandEncoder> e=[cb_se computeCommandEncoder];
                  [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_limited_swiglu];
                  [e setBuffer:bgate offset:0 atIndex:0];[e setBuffer:bup offset:0 atIndex:1];
                  uint n=INTERMEDIATE; [e setBytes:&n length:4 atIndex:2];
                  [e dispatchThreads:MTLSizeMake(INTERMEDIATE,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding]; }
                // down
                { const QuantWeight *qw = &eng->shared[layer].down;
                  id bw=[d newBufferWithBytesNoCopy:(void*)qw->packed length:(size_t)qw->out_dim*(qw->in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
                  id bs=[d newBufferWithBytes:(void*)qw->scales length:(size_t)qw->out_dim*SE_NG_D*sizeof(float) options:MTLResourceStorageModeShared];
                  id bb=[d newBufferWithBytes:(void*)qw->biases length:(size_t)qw->out_dim*SE_NG_D*sizeof(float) options:MTLResourceStorageModeShared];
                  id<MTLComputeCommandEncoder> e=[cb_se computeCommandEncoder];
                  [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
                  [e setBuffer:bw offset:0 atIndex:0];[e setBuffer:bs offset:0 atIndex:1];[e setBuffer:bb offset:0 atIndex:2];
                  [e setBuffer:bgate offset:0 atIndex:3];[e setBuffer:bdown offset:0 atIndex:4];
                  uint od=qw->out_dim,id_=qw->in_dim,gs=SE_GS;
                  [e setBytes:&od length:4 atIndex:5];[e setBytes:&id_ length:4 atIndex:6];[e setBytes:&gs length:4 atIndex:7];
                  [e dispatchThreads:MTLSizeMake(qw->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding]; }
                [cb_se commit]; [cb_se waitUntilCompleted];
                float *sv = (float *)[bdown contents];
                for (int j = 0; j < DIM; j++) shared_out_tmp[j] = sv[j];
                shared_done_early = true;
            }
            // Now wait for pread to complete (should be done or nearly done by now)
            io_pool_wait(io);
        } else if (!all_cached) {
            // Sync wait (shouldn't reach here normally)
            io_pool_wait(io);
        }

        moe_forward_layer(eng, layer, expert_data, expert_ids, expert_weights, N_ACTIVE);
        if (tl) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); ti2 = ts.tv_sec*1e9+ts.tv_nsec;
            fprintf(stderr, "[MOE-IO] L%d io=%.2fms moe=%.2fms\n", layer, (ti1-ti0)/1e6, (ti2-ti1)/1e6); }
        memcpy(ffn_out, [(id<MTLBuffer>)eng->buf_hidden contents], DIM * sizeof(float));
        // Add early shared expert result if computed during pread overlap
        if (shared_done_early) {
            for (int j = 0; j < DIM; j++) ffn_out[j] += shared_out_tmp[j];
        }
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
    // gate + up + limited_swiglu + down in ONE command buffer (4 encoders, 1 wait).
    // Saves 1 GPU sync (was 2 waits) = ~13ms/layer × 43 layers = ~560ms/token.
    // NOTE: skipped if already executed above during async pread overlap (shared_done_early).
    if (eng->shared[layer].gate.packed != NULL && !shared_done_early) {
        const int SE_GS = 64;
        const int SE_NG_GU = DIM / SE_GS;        // 64 groups for gate/up [2048,4096]
        const int SE_NG_D  = INTERMEDIATE / SE_GS; // 32 groups for down [4096,2048]
        memcpy([(id<MTLBuffer>)eng->buf_mhc_attn_in contents], normed, DIM*sizeof(float));
        id<MTLBuffer> bx    = (id<MTLBuffer>)eng->buf_mhc_attn_in;
        id<MTLBuffer> bgate = (id<MTLBuffer>)eng->buf_hidden;    // gate output [INTERMEDIATE]
        id<MTLBuffer> bup   = (id<MTLBuffer>)eng->buf_h_mid;     // up output [INTERMEDIATE]
        id<MTLBuffer> bdown = (id<MTLBuffer>)eng->buf_attn_out;  // down output [DIM]
        {
            id<MTLCommandBuffer> cb = [P.queue commandBuffer];
            // Encoder 1: gate projection
            {
                const QuantWeight *qw = &eng->shared[layer].gate;
                id bw = [d newBufferWithBytesNoCopy:(void*)qw->packed length:(size_t)qw->out_dim*(qw->in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
                id bs = [d newBufferWithBytes:(void*)qw->scales length:(size_t)qw->out_dim*SE_NG_GU*sizeof(float) options:MTLResourceStorageModeShared];
                id bb = [d newBufferWithBytes:(void*)qw->biases length:(size_t)qw->out_dim*SE_NG_GU*sizeof(float) options:MTLResourceStorageModeShared];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
                [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bs offset:0 atIndex:1]; [e setBuffer:bb offset:0 atIndex:2];
                [e setBuffer:bx offset:0 atIndex:3]; [e setBuffer:bgate offset:0 atIndex:4];
                uint od=qw->out_dim, id_=qw->in_dim, gs=SE_GS;
                [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
                [e dispatchThreads:MTLSizeMake(qw->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            }
            // Encoder 2: up projection
            {
                const QuantWeight *qw = &eng->shared[layer].up;
                id bw = [d newBufferWithBytesNoCopy:(void*)qw->packed length:(size_t)qw->out_dim*(qw->in_dim/8)*sizeof(uint32_t) options:MTLResourceStorageModeShared deallocator:nil];
                id bs = [d newBufferWithBytes:(void*)qw->scales length:(size_t)qw->out_dim*SE_NG_GU*sizeof(float) options:MTLResourceStorageModeShared];
                id bb = [d newBufferWithBytes:(void*)qw->biases length:(size_t)qw->out_dim*SE_NG_GU*sizeof(float) options:MTLResourceStorageModeShared];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_dequant_matvec_affine];
                [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bs offset:0 atIndex:1]; [e setBuffer:bb offset:0 atIndex:2];
                [e setBuffer:bx offset:0 atIndex:3]; [e setBuffer:bup offset:0 atIndex:4];
                uint od=qw->out_dim, id_=qw->in_dim, gs=SE_GS;
                [e setBytes:&od length:4 atIndex:5]; [e setBytes:&id_ length:4 atIndex:6]; [e setBytes:&gs length:4 atIndex:7];
                [e dispatchThreads:MTLSizeMake(qw->out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            }
            // Encoder 3: limited SwiGLU in-place on bgate (reads bup) — GPU kernel eliminates CPU round-trip
            {
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_limited_swiglu];
                [e setBuffer:bgate offset:0 atIndex:0];
                [e setBuffer:bup   offset:0 atIndex:1];
                uint n = INTERMEDIATE;
                [e setBytes:&n length:4 atIndex:2];
                [e dispatchThreads:MTLSizeMake(INTERMEDIATE,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
            }
            // Encoder 4: down projection (reads bgate = SwiGLU output)
            {
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
                [e endEncoding];
            }
            [cb commit]; [cb waitUntilCompleted];
        }
        float *sv = (float *)[bdown contents];
        for (int j = 0; j < DIM; j++) ffn_out[j] += sv[j];
    } // end if shared expert

    // mHC post (FFN) — Path B: use buf_residual_gpu, no CPU residual readback
    {
        uint16_t *ffn_out_buf = (uint16_t *)[(id<MTLBuffer>)eng->buf_mhc_attn_out_bf16 contents];
        for (int i = 0; i < DIM; i++) { uint32_t u; memcpy(&u, &ffn_out[i], 4); ffn_out_buf[i] = (uint16_t)(u >> 16); }
        memcpy([(id<MTLBuffer>)eng->buf_mhc_post_weights contents], post, MHC_MULT*sizeof(float));
        memcpy([(id<MTLBuffer>)eng->buf_mhc_comb_weights contents], comb, MHC_MULT*MHC_MULT*sizeof(float));
        id<MTLCommandBuffer> cb3 = [(id<MTLCommandQueue>)eng->queue commandBuffer];

        // Encoder 1: f32→bf16 on GPU for residual (buf_residual_gpu → buf_mhc_res_bf16_in)
        {
            id<MTLComputeCommandEncoder> e = [cb3 computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_f32_to_bf16_vec];
            [e setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu    offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_res_bf16_in offset:0 atIndex:1];
            uint n = MHC_MULT * DIM;
            [e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(MHC_MULT*DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }

        // Encoder 2: mhc_post(ffn)
        {
            id<MTLComputeCommandEncoder> enc3 = [cb3 computeCommandEncoder];
            [enc3 setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_mhc_post_bfloat];
            [enc3 setBuffer:(id<MTLBuffer>)eng->buf_mhc_attn_out_bf16 offset:0 atIndex:0];
            [enc3 setBuffer:(id<MTLBuffer>)eng->buf_mhc_res_bf16_in   offset:0 atIndex:1];
            [enc3 setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_weights  offset:0 atIndex:2];
            [enc3 setBuffer:(id<MTLBuffer>)eng->buf_mhc_comb_weights  offset:0 atIndex:3];
            [enc3 setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_post_out  offset:0 atIndex:4];
            uint hc3 = MHC_MULT, dim3 = DIM;
            [enc3 setBytes:&hc3 length:4 atIndex:5]; [enc3 setBytes:&dim3 length:4 atIndex:6];
            [enc3 dispatchThreads:MTLSizeMake(DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc3 endEncoding];
        }

        // Encoder 3: bf16→f32 writeback to buf_residual_gpu (residual updated for next layer)
        {
            id<MTLComputeCommandEncoder> e = [cb3 computeCommandEncoder];
            [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_bf16_to_f32_vec];
            [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_ffn_post_out offset:0 atIndex:0];
            [e setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu     offset:0 atIndex:1];
            uint n = MHC_MULT * DIM;
            [e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(MHC_MULT*DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
        }

        [cb3 commit];
        // DEFERRED: don't wait — buf_residual_gpu will be written by this cb3.
        // The GPU queue serializes cb3 before the next layer's CB-A automatically.
        // We will wait at the START of the next layer (or after the final layer).
        // Save cb3 for deferred completion.
        eng->deferred.active = true;
        eng->deferred.cmd_experts = (void *)[(id<MTLCommandBuffer>)cb3 retain];

        // No CPU readback here — residual is updated via buf_residual_gpu on GPU.
        // The CPU residual[] array will be updated after deferred wait (next layer's start).
    }
    if (layer == 0 && getenv("MF_DBG")) {
        double fn=0; for(int z=0;z<DIM;z++) fn+=(double)ffn_out[z]*ffn_out[z];
        double rn=0; for(int z=0;z<MHC_MULT*DIM;z++) rn+=(double)residual[z]*residual[z];
        fprintf(stderr, "[mf-dbg] L0 ffn_out norm=%.4f, residual after ffn-post norm=%.4f\n", sqrt(fn), sqrt(rn));        const char *dd = getenv("DSV4_DUMP_DIR");
        if (dd) {
            char path[1024]; snprintf(path, sizeof(path), "%s/L0_ffn_out_metal.bin", dd);
            FILE *ff = fopen(path, "wb");
            if (ff) { fwrite(ffn_out, sizeof(float), DIM, ff); fclose(ff); }
            // Also dump the full post-layer-0 residual [MHC_MULT, DIM] for pos=8
            if (pos == 8) {
                snprintf(path, sizeof(path), "%s/L0_residual_out_pos8.bin", dd);
                FILE *fp = fopen(path, "wb");
                if (fp) { fwrite(residual, sizeof(float), MHC_MULT * DIM, fp); fclose(fp); }
            }
        }
    }

    } // end @autoreleasepool
    return 0;
}

int moe_infer_forward(MoEInferEngine *eng, float *hidden, int pos) {
    int max_layers = N_LAYERS;
    const char *nl = getenv("NATIVE_MAX_LAYERS");
    if (nl) max_layers = atoi(nl);
    const char *time_layers = getenv("NATIVE_TIME_LAYERS");
    for (int layer = 0; layer < max_layers && layer < N_LAYERS; layer++) {
        double t0 = 0;
        if (time_layers) {
            struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
            t0 = ts.tv_sec * 1e9 + ts.tv_nsec;
        }
        if (moe_infer_forward_layer(eng, layer, hidden, pos) != 0) return -1;
        if (time_layers) {
            struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
            double t1 = ts.tv_sec * 1e9 + ts.tv_nsec;
            fprintf(stderr, "[TIME] pos=%d layer=%d: %.1fms\n", pos, layer, (t1-t0)/1e6);
        }
    }
    // Wait for the final deferred cb3 (last layer's mhc_post_ffn).
    // This ensures buf_residual_gpu is fully written before the caller reads hidden[].
    if (eng->deferred.active && eng->deferred.cmd_experts) {
        [(id<MTLCommandBuffer>)eng->deferred.cmd_experts waitUntilCompleted];
        [(id<MTLCommandBuffer>)eng->deferred.cmd_experts release];
        eng->deferred.cmd_experts = NULL;
        eng->deferred.active = false;
        // Final CPU readback: sync hidden[] (= residual) from GPU for compress_hc etc.
        uint16_t *res_out = (uint16_t *)[(id<MTLBuffer>)eng->buf_mhc_ffn_post_out contents];
        for (int i = 0; i < MHC_MULT * DIM; i++) {
            uint32_t u = ((uint32_t)res_out[i]) << 16;
            memcpy(&hidden[i], &u, 4);
        }
    }

    // SMELT: count this decode token, trigger warmup completion if threshold reached.
    // Only count tokens after prefill (pos > 0 is not sufficient since prefill also
    // calls forward token-by-token). We detect decode mode by checking the engine's
    // current_pos which is managed by the caller (native_engine.zig sets it after prefill).
    // Use a dedicated flag: smelt_in_decode_phase (set by moe_infer_set_decode_mode).
    if (eng->smelt_enabled && !eng->smelt_warmup_done && eng->smelt_in_decode_phase) {
        eng->smelt_tokens_seen++;
        if (eng->smelt_tokens_seen >= eng->smelt_warmup_tokens) {
            // Async: spawn background thread to pread top-N experts per layer.
            // Routing bias activates once smelt_warmup_done becomes true (after preload).
            // Decode continues at full (un-cached) speed until preload completes.
            moe_infer_smelt_preload_async(eng);
        }
    }
    return 0;
}

// Batch forward pass for N tokens (proper transformer order: all tokens per layer).
// Each token starts from its embed in hidden_batch[t * MHC_MULT * DIM].
// After this call, hidden_batch[t] contains the post-layer-42 residual for token t.
// KV caches are built up correctly for each layer: token 0 first, then token 1, etc.
// token_ids: optional [n_tokens] array of token IDs for per-token hash routing.
//            If NULL, uses eng->current_token_id (set by caller) for all tokens.
int moe_infer_forward_batch(MoEInferEngine *eng, float *hidden_batch, int n_tokens, int start_pos, const int *token_ids) {
    int max_layers = N_LAYERS;
    const char *nl = getenv("NATIVE_MAX_LAYERS");
    if (nl) max_layers = atoi(nl);

    // Dump all token embed outputs (before any layer processing) for diagnostics
    if (getenv("DSV4_DUMP_DIR") && getenv("MF_DBG")) {
        const char *dd = getenv("DSV4_DUMP_DIR");
        char path[1024];
        snprintf(path, sizeof(path), "%s/embed_all_tokens.bin", dd);
        FILE *fe = fopen(path, "wb");
        if (fe) {
            // Write [n_tokens, MHC_MULT, DIM] f32
            fwrite(hidden_batch, sizeof(float), (size_t)n_tokens * MHC_MULT * DIM, fe);
            fclose(fe);
        }
        // Also print first token residual details
        fprintf(stderr, "[embed-dump] n_tokens=%d tok0[0..3]=[%.4f %.4f %.4f %.4f]\n",
            n_tokens, hidden_batch[0], hidden_batch[1], hidden_batch[2], hidden_batch[3]);
        fprintf(stderr, "[embed-dump] tok2[0..3]=[%.4f %.4f %.4f %.4f]\n",
            hidden_batch[2*MHC_MULT*DIM+0], hidden_batch[2*MHC_MULT*DIM+1],
            hidden_batch[2*MHC_MULT*DIM+2], hidden_batch[2*MHC_MULT*DIM+3]);
        fprintf(stderr, "[embed-dump] tok4[0..3]=[%.4f %.4f %.4f %.4f]\n",
            hidden_batch[4*MHC_MULT*DIM+0], hidden_batch[4*MHC_MULT*DIM+1],
            hidden_batch[4*MHC_MULT*DIM+2], hidden_batch[4*MHC_MULT*DIM+3]);
    }

    for (int layer = 0; layer < max_layers && layer < N_LAYERS; layer++) {
        for (int t = 0; t < n_tokens; t++) {
            float *hidden_t = hidden_batch + (size_t)t * (MHC_MULT * DIM);
            // Set token ID per-token so hash routing uses the correct expert table row
            if (token_ids != NULL) eng->current_token_id = (int)token_ids[t];
            if (moe_infer_forward_layer(eng, layer, hidden_t, start_pos + t) != 0) return -1;
        }
        // Per-layer residual dump: write last token's residual as raw f32 bin
        if (getenv("DSV4_DUMP_DIR") && getenv("MF_DBG")) {
            const char *dd = getenv("DSV4_DUMP_DIR");
            char path[1024];
            snprintf(path, sizeof(path), "%s/L%02d_residual_last.bin", dd, layer);
            FILE *fl = fopen(path, "wb");
            if (fl) {
                float *last = hidden_batch + (size_t)(n_tokens-1) * (MHC_MULT * DIM);
                fwrite(last, sizeof(float), MHC_MULT * DIM, fl);
                fclose(fl);
            }
        }
    }

    // SMELT: do NOT count prefill tokens for warmup statistics.
    // Prefill uses hash-routing for layers 0-2 (routing_counts all 0), so
    // triggering warmup here would preload the wrong experts.
    // Warmup is counted only in moe_infer_forward (decode path).
    (void)0; // prefill path intentionally skips smelt_tokens_seen

    // Dump last token's final residual for diagnostics
    if (getenv("DSV4_DUMP_DIR") && getenv("MF_DBG")) {
        const char *dd = getenv("DSV4_DUMP_DIR");
        char path[1024];
        snprintf(path, sizeof(path), "%s/L42_residual_last.bin", dd);
        FILE *f = fopen(path, "wb");
        if (f) {
            float *last = hidden_batch + (size_t)(n_tokens-1) * (MHC_MULT * DIM);
            fwrite(last, sizeof(float), MHC_MULT * DIM, f);
            fclose(f);
        }
        KVCache *kvc0 = &eng->kv_cache[0];
        if (kvc0->kv && kvc0->len > 0) {
            snprintf(path, sizeof(path), "%s/L0_kvcache_prefill.bin", dd);
            FILE *fkv = fopen(path, "wb");
            if (fkv) {
                fwrite(kvc0->kv, sizeof(uint16_t), (size_t)kvc0->len * KV_LORA_RANK, fkv);
                fclose(fkv);
            }
        }
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
    // Initialize SMELT fields (disabled by default)
    memset(eng->routing_counts, 0, sizeof(eng->routing_counts));
    eng->smelt_warmup_tokens = 0;
    eng->smelt_n_per_layer   = 0;
    eng->smelt_tokens_seen   = 0;
    eng->smelt_warmup_done   = false;
    eng->smelt_enabled       = false;
    eng->smelt_in_decode_phase = false;
    eng->smelt_penalty       = 1e9f;
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
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    for (int i = 0; i < N_LAYERS; i++) {
        eng->input_norms[i] = input_norms[i];
        eng->attn_norms[i] = attn_norms[i];
        eng->gate_proj[i]  = gate_proj_w[i];
        eng->gate_bias[i]  = gate_bias_w[i];
        // Upload per-layer norm weights to persistent GPU buffers
        if (input_norms[i] && !eng->buf_input_norm_gpu[i]) {
            eng->buf_input_norm_gpu[i] = (void *)[d newBufferWithBytes:(void*)input_norms[i] length:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        }
        if (attn_norms[i] && !eng->buf_attn_norm_gpu[i]) {
            eng->buf_attn_norm_gpu[i] = (void *)[d newBufferWithBytes:(void*)attn_norms[i] length:DIM*sizeof(float) options:MTLResourceStorageModeShared];
        }
        if (gate_proj_w[i] && !eng->buf_gate_proj_gpu[i]) {
            eng->buf_gate_proj_gpu[i] = (void *)[d newBufferWithBytes:(void*)gate_proj_w[i] length:(size_t)N_EXPERTS*DIM*sizeof(float) options:MTLResourceStorageModeShared];
        }
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
    // If a deferred CMD3 is in-flight from the previous request, wait and release it
    // before clearing state. This prevents use-after-free on GPU buffers.
    if (eng->deferred.active && eng->deferred.cmd_experts) {
        [(id<MTLCommandBuffer>)eng->deferred.cmd_experts waitUntilCompleted];
        [(id<MTLCommandBuffer>)eng->deferred.cmd_experts release];
        eng->deferred.cmd_experts = NULL;
    }
    eng->deferred.active = false;
    eng->deferred.gpu_combined = false;

    // Reset SMELT decode phase flag so next request's prefill doesn't count for warmup.
    // But preserve routing_counts (accumulate across requests for better hot-expert detection)
    // and smelt_tokens_seen (counts total decode tokens seen, not per-request).
    if (eng->smelt_enabled && !eng->smelt_warmup_done) {
        eng->smelt_in_decode_phase = false;
        // NOTE: smelt_tokens_seen and routing_counts are NOT reset — they accumulate across requests
    }

    for (int l = 0; l < N_LAYERS; l++) {
        eng->kv_cache[l].len = 0;
        // Clear KV buffer so stale entries from the previous request
        // cannot bleed into the new sequence when cache_len is small.
        if (eng->kv_cache[l].kv) {
            memset(eng->kv_cache[l].kv, 0,
                   (size_t)MAX_SEQ_LEN * KV_LORA_RANK * sizeof(uint16_t));
        }
        // kv_gpu_buf is NOT freed/released here — it persists for the engine lifetime
        // and is reused for every new request (kv is zeroed above).
        CompressorState *cs = &eng->comp_state[l];
        cs->n_comp = 0;
        cs->n_idx_comp = 0;
        if (cs->state_kv && cs->ratio > 0 && cs->out_dim > 0)
            memset(cs->state_kv, 0, 2 * cs->ratio * cs->out_dim * sizeof(float));
        if (cs->state_score && cs->ratio > 0 && cs->out_dim > 0)
            memset(cs->state_score, 0, 2 * cs->ratio * cs->out_dim * sizeof(float));
        if (cs->comp_kv)
            memset(cs->comp_kv, 0, (size_t)MAX_COMP_BLOCKS * COMP_HEAD_DIM * sizeof(float));
        if (cs->idx_state_kv)  memset(cs->idx_state_kv, 0, 8 * 256 * sizeof(float));
        if (cs->idx_state_score) memset(cs->idx_state_score, 0, 8 * 256 * sizeof(float));
        if (cs->idx_comp_kv)
            memset(cs->idx_comp_kv, 0, (size_t)MAX_COMP_BLOCKS * IDX_HEAD_DIM * sizeof(float));
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

    // Upload mHC weights to persistent GPU buffers immediately (avoids per-call allocation)
    id<MTLDevice> d = (id<MTLDevice>)eng->device;
    const int MIX3 = MHC_MULT * (MHC_MULT + 2);  // 24
    const size_t fn_bytes    = (size_t)MIX3 * MHC_MULT * DIM * sizeof(float);
    const size_t base_bytes  = (size_t)MIX3 * sizeof(float);
    const size_t scale_bytes = 3 * sizeof(float);

    if (attn_fn && !eng->buf_attn_hc_fn[layer]) {
        eng->buf_attn_hc_fn[layer]    = (void *)[d newBufferWithBytes:(void*)attn_fn    length:fn_bytes    options:MTLResourceStorageModeShared];
        eng->buf_attn_hc_base[layer]  = (void *)[d newBufferWithBytes:(void*)attn_base  length:base_bytes  options:MTLResourceStorageModeShared];
        eng->buf_attn_hc_scale[layer] = (void *)[d newBufferWithBytes:(void*)attn_scale length:scale_bytes options:MTLResourceStorageModeShared];
    }
    if (ffn_fn && !eng->buf_ffn_hc_fn[layer]) {
        eng->buf_ffn_hc_fn[layer]    = (void *)[d newBufferWithBytes:(void*)ffn_fn    length:fn_bytes    options:MTLResourceStorageModeShared];
        eng->buf_ffn_hc_base[layer]  = (void *)[d newBufferWithBytes:(void*)ffn_base  length:base_bytes  options:MTLResourceStorageModeShared];
        eng->buf_ffn_hc_scale[layer] = (void *)[d newBufferWithBytes:(void*)ffn_scale length:scale_bytes options:MTLResourceStorageModeShared];
    }
}


// ============================================================================
// Embed / compress / logits helpers (needed by native engine, no MLX)
// ============================================================================

void moe_infer_embed(MoEInferEngine *eng, int token_id, float *hidden_out) {
    const float *row = eng->embed + (size_t)token_id * DIM;
    if (getenv("MF_DBG") && token_id == 128822) {
        fprintf(stderr, "[embed-dbg] tok=128822 embed[0..3]=[%.6f %.6f %.6f %.6f] norm_estimate=%.4f\n",
            row[0], row[1], row[2], row[3],
            sqrtf(row[0]*row[0] + row[1]*row[1] + row[2]*row[2]));
    }
    for (int m = 0; m < MHC_MULT; m++) {
        for (int d = 0; d < DIM; d++) {
            // Truncate embed to bf16 to match MLX embedding lookup precision
            uint32_t u; memcpy(&u, &row[d], 4); u &= 0xFFFF0000U;
            memcpy(&hidden_out[m * DIM + d], &u, 4);
        }
    }
    // Step 1 (Path B): keep buf_residual_gpu in sync with CPU hidden_out.
    // Step 2 will make GPU the source-of-truth and remove the CPU copy.
    if (eng->buf_residual_gpu) {
        memcpy([(id<MTLBuffer>)eng->buf_residual_gpu contents],
               hidden_out, (size_t)MHC_MULT * DIM * sizeof(float));
    }
}

static void cpu_rms_norm(const float *x, const float *w, float *out, int dim, float eps) {
    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / dim) + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv * w[i];
}

int moe_infer_get_logits(MoEInferEngine *eng, const float *hidden, float *logits_out) {
    if (!eng->final_norm || !eng->lm_head || !logits_out) return -1;
    static float normed[DIM];
    cpu_rms_norm(hidden, eng->final_norm, normed, DIM, 1e-6f);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                eng->vocab_size, DIM,
                1.0f, eng->lm_head, DIM,
                normed, 1,
                0.0f, logits_out, 1);
    if (getenv("MF_DBG")) {
        int max_i = 0;
        for (int i=1;i<eng->vocab_size;i++) if (logits_out[i]>logits_out[max_i]) max_i=i;
        fprintf(stderr, "[mf-dbg] logits max=%d val=%.3f\n", max_i, logits_out[max_i]);
    }
    return 0;
}

void moe_infer_compress_hc(MoEInferEngine *eng, const float *residual, float *out) {
    (void)eng;
    for (int d = 0; d < DIM; d++) {
        float sum = 0.0f;
        for (int m = 0; m < MHC_MULT; m++) sum += residual[m * DIM + d];
        out[d] = sum / MHC_MULT;
    }
}

// ============================================================================
// Compressor/Indexer implementation (T2C.2/T2C.3)
// ============================================================================

void moe_infer_set_layer_compressor(MoEInferEngine *eng, int layer,
    uint32_t compress_ratio,
    QuantWeight comp_wkv, QuantWeight comp_wgate,
    const float *comp_ape, const float *comp_norm) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->compress_ratio[layer] = compress_ratio;
    eng->comp_wkv[layer] = comp_wkv;
    eng->comp_wgate[layer] = comp_wgate;
    eng->comp_ape[layer] = comp_ape;
    eng->comp_norm[layer] = comp_norm;
}

void moe_infer_set_layer_indexer(MoEInferEngine *eng, int layer,
    QuantWeight idx_wq_b, QuantWeight idx_weights_proj,
    QuantWeight idx_comp_wkv, QuantWeight idx_comp_wgate,
    const float *idx_comp_ape, const float *idx_comp_norm) {
    if (layer < 0 || layer >= N_LAYERS) return;
    eng->idx_wq_b[layer] = idx_wq_b;
    eng->idx_weights_proj[layer] = idx_weights_proj;
    eng->idx_comp_wkv[layer] = idx_comp_wkv;
    eng->idx_comp_wgate[layer] = idx_comp_wgate;
    eng->idx_comp_ape[layer] = idx_comp_ape;
    eng->idx_comp_norm[layer] = idx_comp_norm;
}

// CPU affine-4bit matvec (safe version): out[out_dim] = W[out_dim, in_dim] @ x[in_dim]
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

// Softmax in-place over x[n]
static void cpu_softmax_inplace(float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

void moe_infer_compressor_step(MoEInferEngine *eng, int layer, int pos,
                                const float *attn_normed) {
    const uint32_t ratio = eng->compress_ratio[layer];
    if (ratio == 0) return;

    CompressorState *cs = &eng->comp_state[layer];

    // Lazy initialize state buffers
    if (cs->state_kv == NULL) {
        const uint32_t out_dim = (ratio == 4) ? CSA_OUT_DIM : HCA_OUT_DIM;
        cs->ratio = ratio;
        cs->out_dim = out_dim;
        cs->state_kv    = (float*)calloc((size_t)2 * ratio * out_dim, sizeof(float));
        cs->state_score = (float*)calloc((size_t)2 * ratio * out_dim, sizeof(float));
        cs->comp_kv     = (float*)calloc((size_t)MAX_COMP_BLOCKS * COMP_HEAD_DIM, sizeof(float));
        cs->n_comp      = 0;
        if (ratio == 4) {
            cs->idx_state_kv    = (float*)calloc((size_t)8 * 256, sizeof(float));
            cs->idx_state_score = (float*)calloc((size_t)8 * 256, sizeof(float));
            cs->idx_comp_kv     = (float*)calloc((size_t)MAX_COMP_BLOCKS * IDX_HEAD_DIM, sizeof(float));
            cs->n_idx_comp      = 0;
        }
    }

    const uint32_t out_dim = cs->out_dim;
    const uint32_t pos_mod = (uint32_t)pos % ratio;
    // For CSA (ratio=4): store in upper half [ratio+pos_mod]; for HCA: at [pos_mod]
    const uint32_t row = (ratio == 4) ? (ratio + pos_mod) : pos_mod;

    float *kv_cur   = (float*)alloca(out_dim * sizeof(float));
    float *gate_cur = (float*)alloca(out_dim * sizeof(float));

    // 1. Project
    cpu_affine_matvec_safe(kv_cur,   &eng->comp_wkv[layer],   attn_normed);
    cpu_affine_matvec_safe(gate_cur, &eng->comp_wgate[layer], attn_normed);

    // 2. Add positional bias
    if (eng->comp_ape[layer]) {
        const float *ape_row = eng->comp_ape[layer] + (size_t)pos_mod * out_dim;
        for (uint32_t j = 0; j < out_dim; j++) gate_cur[j] += ape_row[j];
    }

    // 3. Store in rolling state
    memcpy(cs->state_kv    + (size_t)row * out_dim, kv_cur,   out_dim * sizeof(float));
    memcpy(cs->state_score + (size_t)row * out_dim, gate_cur, out_dim * sizeof(float));

    // 4. Check if we should emit a block
    if (((uint32_t)pos + 1) % ratio != 0) return;
    if (cs->n_comp >= MAX_COMP_BLOCKS) return;

    const uint32_t head_dim = COMP_HEAD_DIM; // 512

    // 5 & 6. Per-dimension softmax pooling (matches ds4 compressor_pool_decode_state).
    // For each dimension j: softmax across the ratio token scores, weighted sum of kv values.
    // For CSA (ratio=4): state is [2*ratio, out_dim] where front half [0..ratio) is the
    // overlap buffer and upper half [ratio..2*ratio) is current window.
    // The score gate for each token-slot r is state_score[r * out_dim + j] for HCA,
    // or state_score[(ratio+r) * out_dim + j] for the current CSA window.
    // width = out_dim for HCA; out_dim = 2*head_dim for CSA (front and back halves
    // per token: state_score[r*out_dim+j] is front-half score, +head_dim is back-half).
    float *pooled = (float*)alloca(head_dim * sizeof(float));
    for (uint32_t j = 0; j < head_dim; j++) {
        float max_score = -1e30f;
        if (ratio == 4) {
            // CSA: per-slot two half-scores (j and j+head_dim)
            for (uint32_t r = 0; r < ratio; r++) {
                float sp = cs->state_score[(size_t)r * out_dim + j];
                float sc = cs->state_score[(size_t)(ratio + r) * out_dim + head_dim + j];
                if (sp > max_score) max_score = sp;
                if (sc > max_score) max_score = sc;
            }
        } else {
            for (uint32_t r = 0; r < ratio; r++) {
                float s = cs->state_score[(size_t)r * out_dim + j];
                if (s > max_score) max_score = s;
            }
        }
        float denom = 0.0f, sum = 0.0f;
        if (ratio == 4) {
            for (uint32_t r = 0; r < ratio; r++) {
                float wp = expf(cs->state_score[(size_t)r * out_dim + j] - max_score);
                float wc = expf(cs->state_score[(size_t)(ratio + r) * out_dim + head_dim + j] - max_score);
                denom += wp + wc;
                sum += wp * cs->state_kv[(size_t)r * out_dim + j];
                sum += wc * cs->state_kv[(size_t)(ratio + r) * out_dim + head_dim + j];
            }
        } else {
            for (uint32_t r = 0; r < ratio; r++) {
                float w = expf(cs->state_score[(size_t)r * out_dim + j] - max_score);
                denom += w;
                sum += w * cs->state_kv[(size_t)r * out_dim + j];
            }
        }
        pooled[j] = denom > 0.0f ? sum / denom : 0.0f;
    }

    // 7. RMSNorm
    if (eng->comp_norm[layer]) {
        cpu_rms_norm(pooled, eng->comp_norm[layer], pooled, (int)head_dim, 1e-6f);
    }

    // 8. Apply YaRN tail RoPE at compressed position
    const int comp_pos = (int)(((uint32_t)pos + 1) - ratio);
    apply_rope_tail(pooled, (int)head_dim, QK_NOPE_DIM, comp_pos, 16.0f);

    // 9. Append to comp_kv
    memcpy(cs->comp_kv + (size_t)cs->n_comp * head_dim, pooled, head_dim * sizeof(float));
    cs->n_comp++;

    // 10. For CSA: shift overlap buffer
    if (ratio == 4) {
        for (uint32_t t = 0; t < ratio; t++) {
            memcpy(cs->state_kv    + (size_t)t * out_dim,
                   cs->state_kv    + (size_t)(ratio + t) * out_dim,
                   out_dim * sizeof(float));
            memcpy(cs->state_score + (size_t)t * out_dim,
                   cs->state_score + (size_t)(ratio + t) * out_dim,
                   out_dim * sizeof(float));
        }
    }
}

bool moe_infer_indexer_step(MoEInferEngine *eng, int layer, int pos,
                              const float *attn_normed, const float *q_a_out,
                              bool *allowed_out) {
    CompressorState *cs = &eng->comp_state[layer];
    const uint32_t n_comp = cs->n_comp;

    if (n_comp == 0) return false;

    uint32_t top_k = 512;
    if (n_comp <= top_k) {
        for (uint32_t c = 0; c < n_comp; c++) allowed_out[c] = true;
        return true;
    }

    // Run indexer's own compressor step (ratio=4, idx_out_dim=256)
    if (eng->idx_comp_wkv[layer].packed != NULL && cs->idx_state_kv != NULL) {
        const uint32_t idx_out_dim = 256;
        const uint32_t idx_pos_mod = (uint32_t)pos % 4;
        const uint32_t idx_row = 4 + idx_pos_mod;

        float *kv_cur   = (float*)alloca(idx_out_dim * sizeof(float));
        float *gate_cur = (float*)alloca(idx_out_dim * sizeof(float));

        cpu_affine_matvec_safe(kv_cur,   &eng->idx_comp_wkv[layer],   attn_normed);
        cpu_affine_matvec_safe(gate_cur, &eng->idx_comp_wgate[layer], attn_normed);

        if (eng->idx_comp_ape[layer]) {
            const float *ape_row = eng->idx_comp_ape[layer] + (size_t)idx_pos_mod * idx_out_dim;
            for (uint32_t j = 0; j < idx_out_dim; j++) gate_cur[j] += ape_row[j];
        }

        memcpy(cs->idx_state_kv    + (size_t)idx_row * idx_out_dim, kv_cur,   idx_out_dim * sizeof(float));
        memcpy(cs->idx_state_score + (size_t)idx_row * idx_out_dim, gate_cur, idx_out_dim * sizeof(float));

        if (((uint32_t)pos + 1) % 4 == 0 && cs->n_idx_comp < MAX_COMP_BLOCKS) {
            float weights_arr[4];
            for (int t = 0; t < 4; t++) {
                float s = 0.0f;
                for (uint32_t d = 0; d < IDX_HEAD_DIM; d++)
                    s += cs->idx_state_score[(size_t)t * idx_out_dim + d];
                weights_arr[t] = s;
            }
            cpu_softmax_inplace(weights_arr, 4);
            float pooled[IDX_HEAD_DIM];
            memset(pooled, 0, sizeof(pooled));
            for (int t = 0; t < 4; t++) {
                for (uint32_t d = 0; d < IDX_HEAD_DIM; d++)
                    pooled[d] += weights_arr[t] * cs->idx_state_kv[(size_t)t * idx_out_dim + d];
            }
            if (eng->idx_comp_norm[layer])
                cpu_rms_norm(pooled, eng->idx_comp_norm[layer], pooled, IDX_HEAD_DIM, 1e-6f);
            memcpy(cs->idx_comp_kv + (size_t)cs->n_idx_comp * IDX_HEAD_DIM, pooled, IDX_HEAD_DIM * sizeof(float));
            cs->n_idx_comp++;
            // Shift CSA overlap buffer for indexer
            for (int t = 0; t < 4; t++) {
                memcpy(cs->idx_state_kv    + (size_t)t * idx_out_dim,
                       cs->idx_state_kv    + (size_t)(4+t) * idx_out_dim, idx_out_dim * sizeof(float));
                memcpy(cs->idx_state_score + (size_t)t * idx_out_dim,
                       cs->idx_state_score + (size_t)(4+t) * idx_out_dim, idx_out_dim * sizeof(float));
            }
        }
    }

    const uint32_t n_idx = cs->n_idx_comp;
    if (n_idx == 0 || q_a_out == NULL) {
        for (uint32_t c = 0; c < n_comp; c++) allowed_out[c] = true;
        return true;
    }

    // Score each idx_comp block using q
    const int n_idx_heads = 64;
    const int idx_head_dim = 128;
    float *q = (float*)alloca((size_t)n_idx_heads * idx_head_dim * sizeof(float));
    cpu_affine_matvec_safe(q, &eng->idx_wq_b[layer], q_a_out);

    float *head_w = (float*)alloca((size_t)n_idx_heads * sizeof(float));
    cpu_affine_matvec_safe(head_w, &eng->idx_weights_proj[layer], attn_normed);
    float scale = 1.0f / sqrtf((float)(n_idx_heads * idx_head_dim));
    for (int h = 0; h < n_idx_heads; h++) head_w[h] *= scale;

    float *scores = (float*)calloc(n_idx, sizeof(float));
    for (uint32_t c = 0; c < n_idx; c++) {
        const float *kv_c = cs->idx_comp_kv + (size_t)c * idx_head_dim;
        float s = 0.0f;
        for (int h = 0; h < n_idx_heads; h++) {
            const float *q_h = q + (size_t)h * idx_head_dim;
            float dot = 0.0f;
            for (int d = 0; d < idx_head_dim; d++) dot += q_h[d] * kv_c[d];
            if (dot > 0.0f) s += dot * head_w[h];
        }
        scores[c] = s;
    }

    memset(allowed_out, 0, n_comp * sizeof(bool));
    uint32_t actual_k = (top_k < n_idx) ? top_k : n_idx;
    for (uint32_t k = 0; k < actual_k; k++) {
        uint32_t best = 0;
        float best_score = -1e30f;
        for (uint32_t c = 0; c < n_idx; c++) {
            if (!allowed_out[c] && scores[c] > best_score) {
                best_score = scores[c];
                best = c;
            }
        }
        if (best < n_comp) allowed_out[best] = true;
    }
    free(scores);
    return true;
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
    // Free expert memory cache
    if (eng->expert_cache_n_experts > 0) {
        for (int layer = 0; layer < N_LAYERS; layer++) {
            if (eng->expert_mem_cache[layer]) free(eng->expert_mem_cache[layer]);
            if (eng->expert_mem_pool[layer]) free(eng->expert_mem_pool[layer]);
        }
    }
    // Free compressor state
    for (int l = 0; l < N_LAYERS; l++) {
        CompressorState *cs = &eng->comp_state[l];
        free(cs->state_kv);
        free(cs->state_score);
        free(cs->comp_kv);
        free(cs->idx_state_kv);
        free(cs->idx_state_score);
        free(cs->idx_comp_kv);
    }
    free(eng);
}
