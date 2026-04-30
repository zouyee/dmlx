# Design: Correctness Properties

**Parent**: `design.md` §Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: NN Layer GPU Numerical Equivalence

*For any* NN layer (RMSNorm, RoPE, SDPA, Embedding, LSTM, GRU, loss functions) and *for any* valid input tensor, the GPU-accelerated implementation (via mlx-c operator chains or fast.zig fused kernels) SHALL produce output with cosine similarity ≥ 0.9999 compared to a pre-computed Python MLX reference output.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 26.1, 26.2**

### Property 2: Forward Pass Arena Cleanup

*For any* model and *for any* valid input tensor, after a forward pass completes and the ScopedArrayArena is deinitialized, all intermediate Arrays tracked by the arena SHALL be released, and only the final output Array SHALL remain live.

**Validates: Requirements 2.1, 2.2**

### Property 3: Generation API Consistency

*For any* model, prompt, and generation config, the sequence of tokens produced by `streamGenerate` (collected via callback) SHALL be identical to the sequence returned by `generate`, and both SHALL have length ≤ `max_tokens`.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4**

### Property 4: Model Registry Lookup Correctness

*For any* architecture name string, looking it up in the Model_Registry SHALL succeed if and only if the architecture is registered. When lookup fails, the error message SHALL contain the queried architecture name.

**Validates: Requirements 6.2, 6.4**

### Property 5: Prompt Cache Round-Trip

*For any* valid KV cache state, saving to a safetensors file and loading it back SHALL produce a KV cache state with keys and values that are element-wise equal to the original. Loading with a mismatched model configuration (different num_layers, head_dim, or num_kv_heads) SHALL return an error.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4**

### Property 6: Fused Operation Numerical Equivalence

*For any* valid input tensors, a compiled fused operation (SwiGLU MLP, AdamW step) SHALL produce output with cosine similarity ≥ 0.9999 compared to the unfused implementation.

**Validates: Requirements 8.1, 8.2, 8.3**

### Property 7: Scheduler Prioritization Invariant

*For any* set of waiting and running requests, the Scheduler's `schedule()` output SHALL include all running (decode-phase) requests before any waiting (prefill-phase) requests. Running requests SHALL always be scheduled if they have allocated blocks.

**Validates: Requirements 9.2**

### Property 8: Block Conservation

*For any* sequence of block allocation and deallocation operations on the Block_Manager, the sum of free blocks and used blocks SHALL always equal the total block count. When a request completes and its blocks are freed, those blocks SHALL appear in the free pool.

**Validates: Requirements 9.3, 9.4, 9.5, 10.1, 10.2, 10.3, 10.5**

### Property 9: Copy-on-Write Isolation

*For any* Block shared by two or more requests (ref_count > 1), when one request modifies the block, the Block_Manager SHALL create a copy before mutation. The other request's view of the block SHALL remain unchanged.

**Validates: Requirements 10.4, 15.3**

### Property 10: Quantize-Dequantize Round-Trip

*For any* tensor (KV cache entry or model weight) and *for any* valid bit-width (4 or 8), quantizing with `mlx_quantize` and then dequantizing with `mlx_dequantize` SHALL produce a tensor with cosine similarity ≥ 0.99 to the original for 8-bit and ≥ 0.95 for 4-bit.

**Validates: Requirements 11.2, 11.3, 18.1, 18.2**

### Property 11: Continuous Batching Attention Isolation

*For any* batch of N request sequences concatenated into a single tensor, the attention mask SHALL ensure that each request's tokens attend only to tokens within the same request. The batched tensor length SHALL equal the sum of individual sequence lengths.

**Validates: Requirements 13.1, 13.2**

### Property 12: Chunked Prefill Correctness

*For any* prompt with length exceeding `max_prefill_tokens`, the Scheduler SHALL split it into ceil(prompt_len / max_prefill_tokens) chunks. While chunked prefill is in progress for one request, decode steps for other active requests SHALL continue to be scheduled.

**Validates: Requirements 14.2, 14.3**

### Property 13: Block Hash Determinism and Prefix Reuse

*For any* sequence of token IDs, `hashBlock(prev_hash, token_ids)` SHALL be deterministic — the same inputs always produce the same hash. When two requests share a token prefix that aligns to block boundaries, the second request SHALL reuse the first request's cached blocks for the shared prefix.

**Validates: Requirements 15.1, 15.2**

### Property 14: N-gram Draft Proposal Correctness

*For any* generated context containing a repeated n-gram suffix, the NgramDrafter SHALL find the matching n-gram and propose the correct continuation tokens from the context.

**Validates: Requirements 16.1**

### Property 15: Speculative Decoding Statistical Equivalence

*For any* draft token sequence and target model probability distribution, the accept/reject decision SHALL follow the speculative sampling algorithm such that accepted tokens are statistically equivalent to sampling directly from the target distribution.

**Validates: Requirements 16.3**

### Property 16: Guided Decoding Constraint Satisfaction

*For any* grammar constraint (JSON schema or regex), the logits mask applied at each generation step SHALL set all disallowed token logits to negative infinity. The resulting generated token sequence SHALL satisfy the constraint.

**Validates: Requirements 17.1, 17.2, 17.3**

### Property 17: QLoRA Forward Correctness

*For any* input tensor x, quantized base weight W_base, and LoRA adapters (A, B, scaling), the QLoRA forward pass SHALL produce output equal to `dequantize(W_base) @ x + (B @ A) @ x * scaling`. Gradients SHALL be computed only for A and B parameters; W_base SHALL remain unchanged after a training step.

**Validates: Requirements 19.2, 19.3**

### Property 18: MoE Top-K Selection

*For any* routing score tensor and top-k value, the MoE_Router SHALL select the k experts with the highest scores. The output tensor shape SHALL match the input tensor shape along the token and hidden dimensions.

**Validates: Requirements 20.1, 20.2**

### Property 19: Model Pool LRU Eviction with Pinning

*For any* set of loaded models in the Model_Pool, when eviction is triggered, the least recently used non-pinned model SHALL be evicted first. Pinned models SHALL never be evicted regardless of their access recency.

**Validates: Requirements 21.3, 21.4**

### Property 20: Tiered KV Cache Evict-Restore Round-Trip

*For any* KV cache block, evicting it to the cold tier (SSD via safetensors) and then restoring it to the hot tier SHALL produce a block with keys and values element-wise equal to the original. When the hot tier exceeds capacity, the least recently accessed blocks SHALL be evicted first.

**Validates: Requirements 22.2, 22.3, 22.4**

### Property 21: Auto max_kv_size Formula

*For any* model configuration (num_layers, num_kv_heads, head_dim, kv_bits) and device memory value, `autoMaxKvSize` SHALL return `(total_RAM - model_bytes - safety_margin) / (2 * num_kv_heads * head_dim * (kv_bits / 8) * num_layers)`.

**Validates: Requirements 24.1, 24.2**

