# mlx-zig Complete Code Review Report

**Review Date:** 2026-05-01  
**Review Scope:** 130+ Zig files under `src/` (~52,572 lines), `build.zig`, `tests/`, `docs/`  
**Zig Version:** 0.16.0 (declared in `build.zig.zon`)  
**Target Platform:** macOS Apple Silicon (Metal)  
**Review Rounds:** 6 independent dimensional reviews  
**Document Status:** Verified against source code

---

## Executive Summary

### Project Positioning
`mlx-zig` is a Zig language binding for Apple's `mlx-c` C library, providing an LLM inference engine. It supports multiple architectures (LLaMA, DeepSeek V4, MiniMax, Nemotron-H, etc.), multiple KV cache strategies, quantization, continuous batching, and server mode.

### Overall Score

| Review Dimension | Score | Notes |
|----------|------|------|
| Architecture Design | 7.5/10 | Well modularized, 200+ ops, multiple cache strategies |
| Deep Implementation | 7.5/10 | Core logic mostly correct, but critical algorithm errors exist |
| Design Patterns | 6.5/10 | VTable correct, but Array ownership ambiguity and many stub modules |
| Build/Security/Resources | 6.5/10 | No CI/CD, dangerous security whitelist, false documentation claims |
| Concurrency/API/Performance | 5.5/10 | Completely single-threaded, ~432 camelCase functions, sampler O(V²) |
| Math/Boundary/Protocol | 4.7/10 | Llama causal mask missing, numerous divide-by-zero/OOB/overflow |
| **DeepSeek V4 Special** | **4/10** | Algorithm defects, 30GB+ cache, numerous boundary conditions |
| **Overall Score** | **5.5/10** | Severe mismatch between architectural ambition and implementation quality |

### Most Critical Findings (Top 10)

1. **🔴 Llama Prefill Bidirectional Attention** — `llama.zig` prefill stage passes `mask=null`, making attention non-causal, breaking autoregressive correctness
2. **🔴 Sampler O(V²) Insertion Sort** — 129K vocab size ~16 billion comparisons per token, CPU bottleneck
3. **🔴 879 `@intCast` ReleaseFast UB** — `build.zig` supports ReleaseFast, all shape conversion overflows are undefined behavior
4. **🔴 Function-Level Static Storage Pointing to Stack Locals** — `server.zig` `StreamState` stores `&sse`, `req.model` pointers
5. **🔴 `custom_kernel.zig` String Dangling Pointer** — `defer free` then immediately returns kernel object containing that pointer
6. **🔴 `last_error_buffer` Global Not Thread-Safe** — multi-threaded MLX operations race on writes
7. **🔴 `tool_executor.zig` Shell Whitelist Bypassable** — `python3 -c "..."` can execute arbitrary code through whitelist
8. **🔴 False Documentation Claims** — `competitive-advantages.md` claims 350 tests pass, but they crash; `FIX-REPORT-DEEPSEEK-V4.md` claims fixed but not fixed
9. **🔴 `grad.zig`/`fused.zig`/`eval.zig` Hardcoded CPU Stream** — GPU idle, 10-100x performance degradation
10. **🔴 DeepSeek V4 YARN RoPE Cache 30GB+** — `max_seq_len=1M` with 61 layers needs ~30.5GB precomputed cache

---

## 1. Architecture & Design Review (First Round)

### 1.1 Positive Aspects
- **Clear modularity**: `ops/` has 200+ operations organized by function, `kvcache/` has 9 strategies via VTable unified interface
- **Broad model architecture coverage**: LLaMA, DeepSeek V4, MiniMax, Nemotron-H, Gemma, GLM4, Qwen, Phi, LLaVA, Flux
- **Quantization support**: W4A16/W8A8, MXFP4, TurboQuant, GPTQ format loading
- **Continuous batching**: Scheduler + BatchBuilder architecture correct
- **Distributed**: MPI-style multi-Mac tensor parallelism

### 1.2 Architectural Defects
- **Model scale out of control**: `deepseek_v4.zig` 2949 lines, `deepseek_v4_loader.zig` 1983 lines — single file bears too many subsystems
- **Too many KV Cache strategies**: 9 backends (Standard, Paged, Quantized, Rotating, Radix, Tiered, PrefixDisk, TurboQuant, DeepSeekV4), no unified factory
- **Deceptive server design**: Claims "continuous batching", but `engineLoop` is single-fiber serial
- **No CI/CD**: No `.github/workflows`, no automated quality gates

---

## 2. Deep Implementation Issues (Second Round)

### 2.1 Thread Safety
- **`last_error_buffer`**: `src/c.zig:32` global 2048-byte buffer, C callback `mlxErrorHandler` writes directly, no lock, no `threadlocal`
- **ModelPool not thread-safe**: `ModelPool` has no Mutex/RwLock, `lru_order` contains dangling pointers after `evictByName`
- **ExpertCache not thread-safe**: `ExpertCache`'s `hits`/`misses` are plain `u64`, concurrent access races

### 2.2 Sorting & Sampling
- **O(n²) insertion sort**: `src/sampling.zig` uses `std.sort.insertion` on full vocab. For 128K vocab, worst case ~16 billion comparisons
- **Top-p normalization**: Re-softmax after filtering, correct

### 2.3 CLI Parsing
- `main.zig` uses manual string comparison to parse arguments, no `std.process.args` structured parsing
- Missing parameter validation (e.g., `--temperature -1` is passed through directly)

---

## 3. Design Patterns & Code Quality (Third Round)

### 3.1 Dangerous `@ptrCast` Assumptions
- `array.zig:strides()`: `const cast_ptr: [*]const i64 = @ptrCast(@alignCast(ptr));` — assumes MLX's `size_t*` aligns and is same size as `i64`. Holds on 64-bit platforms but undocumented
- `safetensors_reader.zig`: `@ptrCast(result)` on mmap memory, no `@alignCast`

### 3.2 Memory Leak Patterns
- `page_allocator` leaks in multiple places: `batch_builder.zig` `emptyBatch` uses `page_allocator` direct allocation, caller does not free
- `Array.zeros`/`ones` accept allocator but ignore (`_ = allocator;`), C API allocates internally
- `EagerContext.deinit` frees stream, but if stream is default stream, may be unsafe

### 3.3 Array Ownership Ambiguity
- No documentation on which functions take ownership vs. borrow
- `closureCallback` frees input Array, but MLX may retain reference for gradients
- `tree.zig`'s `treeMapInPlace` semantics unclear

### 3.4 Stub Modules
- `diffusion/vae.zig`: All convolution weights initialized to zero, decoder output always zero
- `vision/llava.zig`: Vision tower marked "not yet implemented"
- `jang_quantizer.zig`: `analyzeSensitivity` returns empty, `quantizeModel` does not write files

---

## 4. Build System, Security & Resources (Fourth Round)

### 4.1 Build System
- `build.zig` `pkgConfigMlxPrefix` success path does not free `stdout` buffer
- No MLX-C version check: `pkg-config --modversion mlxc` never called
- Unknown binary files tracked in repo: `check_types`(1.8MB), `main`, `test_arr`, `test_mlx_c` and other Mach-O files with no source code

### 4.2 Security Boundaries
- **`tool_executor.zig` SHELL_WHITELIST includes interpreters**: `python3`, `python`, `zig`, `git`, `curl`, `wget`, `rm`
- Only checks first token: `python3 -c "import os; os.system('rm -rf /')"` bypasses whitelist
- `server.zig` HTTP parser: 64KB fixed buffer, no request size limit, Slowloris feasible
- No authentication, no rate limiting

### 4.3 False Documentation Claims
- `docs/competitive-advantages.md`: "350 unit tests all pass" — **false**, `zig build test` crashes
- `docs/FIX-REPORT-DEEPSEEK-V4.md`: Claims fixed fullwidth characters and BOS validation (token 100000) — **not fixed**, source still contains `｜`; BOS validation exists but uses token ID `0` not claimed `100000`

### 4.4 Resource Lifecycle
- `safetensors_reader.zig`: Each `loadTensor` performs open/pread/close, excessive syscalls
- `tiered.zig`/`prefix_disk.zig`: No temp file cleanup, residue after crash
- `mlx_default_cpu_stream_new()` created streams not freed in multiple places (`array.zig:zeros/ones`, `eval.zig`)

---

## 5. Concurrency, API, Performance & Resilience (Fifth Round)

### 5.1 Concurrency Model (4/10)
- **Zero Mutex/RwLock/Thread.Pool** in entire `src/` tree
- Only OS thread: `layer_prefetcher.zig` background prefetch worker thread
- Server uses `std.Io.async(...)` cooperative fibers (GCD), not Zig `async/await`
- `engineLoop` single-fiber serial, all requests compete on single lock-free `ModelState`
- `ModelPool`, `ExpertCache`, `KVCacheStrategy` all non-thread-safe

### 5.2 API Design (5/10)
- **~432 public functions use camelCase**, violating Zig convention (should be snake_case)
- No unified error type: `error.MlxError` + `MemoryError` + `RegistryError` + numerous ad-hoc errors
- `Array` has no operator methods, no method chaining
- NN layer fields all `pub`, exposing implementation details
- Mixed `std.mem.indexOf` and `std.mem.find` (incomplete 0.16 migration)

### 5.3 Performance Critical Paths (5/10)
- **Sampler**: `std.sort.insertion` on 129K elements + `logits.eval()` GPU→CPU sync per token + repetition penalty full-vocab CPU→GPU copy-back
- **RotatingKVCache**: Prefill per-token loop, 4S kernel launches (S=sequence length)
- **BatchBuilder**: Rebuilds O(N²) mask every step, even for pure decode requests
- **Stream mode**: Loads full expert tensors each time (GB-scale disk→GPU copies)
- **No SIMD**: Search for `@Vector`/`simd` returns zero matches

### 5.4 Resilience Design (4/10)
- No signal handling (SIGINT/SIGTERM) — cache not saved, temp files not cleaned
- No request-level timeouts — slow requests block permanently
- No retry, no circuit breaker
- Missing weights = hard failure, no partial load fallback
- `TieredKVCache.evictToSSD` crashes on disk full

### 5.5 Data Format Compatibility (5/10)
- ❌ No native GGUF loader
- ❌ No SentencePiece, TikToken, WordPiece
- ❌ No Jinja2 chat templates (only 4 hardcoded)
- ❌ No image decoder (PNG/JPEG/WebP)
- NPY does not support big-endian, int8/uint8/f16/bf16
- Safetensors custom reader skips `__metadata__`

---

## 6. Math Correctness, Boundary Conditions & Protocol (Sixth Round)

### 6.1 Math Correctness (6/10)

**🔴 P0 — Llama Prefill Missing Causal Mask**
`src/models/llama.zig:657`:
```zig
const logits = try arena.track(try self.forward(input, null, caches));
```
Prefill stage `mask=null`, attention is bidirectional. DeepSeek V4 correctly uses `mask_mode="causal"` at same location.

**🟡 P1 — Softplus Float16 Overflow**
`src/ops/activations.zig:90` threshold 20.0 > float16 `exp` safe limit ~11.5.

**✅ Mathematically Correct Components**: RoPE (all variants), RMSNorm, Softmax, Cross-Entropy, Temperature, Top-p, AdamW.

### 6.2 Boundary Conditions (4/10)

**Boundary Condition Defect List (source-verified):**

| Defect | Location | Trigger | Consequence |
|------|------|----------|------|
| OOB read empty logits | `sampling.zig` | `vocab_size == 0` | Crash |
| `seq_len - 1` underflow | `generation.zig:114` | `seq_len == 0` | Crash |
| `max_new_tokens - 1` underflow | `deepseek_v4.zig:2807` | `max_new_tokens == 0` | Infinite loop |
| `base == 1.0` divide-by-zero | `deepseek_v4.zig:336` | `rope_theta = 1.0` | NaN/Inf |
| `compress_ratio == 0` divide-by-zero | `deepseek_v4.zig:1889` | Config error | Crash |
| `window_size == 0` divide-by-zero | `rotating.zig:127` | Config error | Crash |
| `page_size == 0` divide-by-zero | `paged.zig:358` | Config error | Crash |
| `n_experts - k` underflow | `deepseek_v4.zig:376` | `k > n_experts` | Crash |
| mask buffer overflow | `batch_builder.zig:140` | `total_tokens² > usize.MAX` | Heap corruption |
| No ndim check | `deepseek_v4.zig:1604` | `ndim < 3` | OOB panic |
| RoPE cache OOB | `deepseek_v4.zig:252` | `start_pos+seq_len > max_seq_len` | OOB |

**879 `@intCast` instances**: Under ReleaseFast, overflow = undefined behavior.

### 6.3 Protocol Compliance (4/10)

- OpenAI API missing `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `logprobs`, `n` fields
- `std.json.parseFromSlice` defaults `ignore_unknown_fields = false` — client sending extra fields gets 400
- HTTP parsing: case-sensitive header, no chunked encoding, 64KB fixed buffer, no persistent connections
- SSE: no `id`/`retry`, multi-line data unsafe
- JSON escaping: missing `\b`, `\f` and all control characters `\u00XX`
- Error response format inconsistent: mostly `{"error":"string"}` not OpenAI's nested object
- No authentication

### 6.4 Zig Language Pitfalls (4/10)

**🔴 P0 — Function-Level Static Storage Pointing to Stack Locals**
`server.zig:1122-1132`:
```zig
const StreamState = struct {
    var s_sse: *SSEWriter = undefined;      // ← &sse (stack local)
    var s_model_name: []const u8 = "";      // ← req.model
    // ...
};
```

**🔴 P0 — `custom_kernel.zig` Dangling Pointer**
```zig
const name_z = try allocator.dupeZ(u8, name);
defer allocator.free(name_z);  // freed after return
return .{ .inner = c.c.mlx_fast_metal_kernel_new(name_z.ptr, ...) };
```

**🟡 P1 — `catch {}` Silently Discarding Errors**
- `server.zig:934`: Tokens silently discarded during generation
- `tool_executor.zig:144`: Directory creation failure silently ignored
- `memory.zig`: Eviction failure silently ignored

**🟡 P1 — Resource Leak Paths**
- `ops/fused.zig`: `swigluForward`, `adamwStepForward`, `unfusedAdamWStep` missing `defer`/`errdefer` in multiple places
- `closure.zig`: `closureCallback` leaks `out_arrs` on partial failure; `Closure.apply` `out_vec` no `errdefer`

**🟡 P1 — C API Return Values Ignored**
- `array.zig:165`: `_ = c.c.mlx_array_tostring(...)` — on failure, `str` undefined
- `device.zig:36`: `_ = c.c.mlx_device_get_type(...)`

---

## DeepSeek V4 Module Special Analysis (Priority)

### File Scale & Complexity

| File | Lines | Subsystems |
|------|------|--------|
| `deepseek_v4.zig` | 2949 | MLA, MoE, CSA/HCA, mHC, YARN RoPE, Gate, Router, Compressor, Indexer, Attention, Tokenizer adapter |
| `deepseek_v4_loader.zig` | 1983 | Weight name mapping, quantization/dequantization, Smelt config, layer construction |
| `expert_stream.zig` | ~500 | Stream mode expert loading, prefetch |
| `expert_preload.zig` | ~400 | Preload mode expert loading |
| `kvcache/deepseek_v4_cache.zig` | ~600 | CSA/HCA KV cache, window management |

**Assessment**: Single file 2949 lines carrying 5+ complex subsystems, far exceeding maintainability threshold (~500 lines/file).

### Math & Algorithm Issues

| Issue | Location | Verified | Notes |
|------|------|----------|------|
| Prefill causal mask | — | N/A | DS V4 **correctly** uses `mask_mode="causal"` |
| `findCorrectionDim` divide-by-zero | `z:336` | ✅ Confirmed | `base==1.0` → `@log(1.0)==0` → divide-by-zero |
| YARN RoPE cache memory | `z:192-225` | ✅ Confirmed | `max_seq_len * half_dim * 4B` × 2 caches/layer × 61 layers ≈ **30.5 GB** |
| RoPE cache OOB | `z:252` | ✅ Confirmed | OOB when `start_pos + seq_len > max_seq_len` |
| `createCausalMask` shape overflow | `z:397` | ⚠️ Partial | 4 `@intCast` to `i32` can overflow |
| `topkIndices` underflow | `z:376` | ✅ Confirmed | `usize` underflow when `k > n_experts` |
| `compress_ratio == 0` divide-by-zero | `z:1889` | ✅ Confirmed | `W = ready_len / self.compress_ratio` |

### Memory & Performance Issues

| Issue | Location | Impact |
|------|------|------|
| YARN RoPE precomputation 30GB+ | `z:192-225` | 61 layers × 512MB = exceeds most Mac memory |
| Stream mode full tensor loading | `expert_stream.zig:256` | Full weight loading (GB-scale) per MoE layer |
| Per-layer `hidden.eval()` | `z:~1700` | Prevents cross-layer kernel fusion |
| `RotatingKVCache` per-token | `kvcache/rotating.zig` | 4S kernel launches/prefill |

### Boundary Conditions & Crash Risks

| Issue | Location | Trigger | Consequence |
|------|------|------|------|
| `max_new_tokens==0` infinite loop | `z:2807` | `for (0..max_new_tokens-1)` | Memory exhaustion |
| `DSV4Attention` ndim<2 OOB | `z:1604` | `hidden_states` not 3D | Panic |
| `DSV4YarnRoPE.apply` ndim<2 | `z:242` | Input not at least 2D | Panic |
| `head_dim==0` scale=+inf | `z:1609` | `1.0/@sqrt(0)` | NaN propagation |
| `page_size==0` divide-by-zero | `paged.zig:358` | Config error | Panic |
| `window_size==0` divide-by-zero | `rotating.zig:127` | Config error | Panic |

### Quantization & Loading Issues

| Issue | Location | Verified | Notes |
|------|------|------|------|
| `dequantIfNeeded` hardcoded affine | `loader.zig:122` | ✅ | Ignores configured `mxfp4`/`nvfp4`/`mxfp8` |
| `dispatchGatherMm` only recognizes mxfp4 | `z:967` | ✅ | Other modes fall back to `.affine` |
| `dequantFp4` float division | `loader.zig:678` | ✅ | `divide(packed, 16)` not `rightShift` |
| `dequantFp8` scale mismatch | `loader.zig:726` | ✅ | Pad weights but not scales |
| Expert partial read bypassed | `expert_stream.zig:310` | ✅ | Explicitly commented "bypassed" |
| `expandQuantizedBuffer` shape error | `quantized.zig:373` | ✅ | `seq_len * el_per_int` mistaken for head_dim |

### Code Quality & Technical Debt

| Issue | Location | Notes |
|------|------|------|
| Python pseudocode comments | `z:162` | `// return (pre[..., None] * x.astype(f32)).sum(axis=2)` |
| TODO: pool_base | `z:1943` | `// TODO: Use proper position computation with pool_base` |
| MoE Router integration commented | `z:637` | 15 lines of correct code commented out, using manual top-k |
| Code duplication | Multiple loaders | `getFloat`/`getInt`/`mapLayerComponent` duplicated across 4 files |
| `@intCast` proliferation | Entire module | `usize`→`i32` shape conversions without overflow checks |

---

## Complete Defect Summary

### 🔴 P0 Defects (Crash/Security/Algorithm Errors)

| # | Defect | Source | Verified |
|---|------|------|------|
| 1 | Llama prefill bidirectional attention | Round 6 | ✅ |
| 2 | Sampler O(V²) insertion sort | Round 2 | ✅ |
| 3 | 879 `@intCast` ReleaseFast UB | Round 6 | ✅ |
| 4 | `StreamState` static storage pointing to stack locals | Round 6 | ✅ |
| 5 | `custom_kernel.zig` string dangling pointer | Round 6 | ✅ |
| 6 | `last_error_buffer` global not thread-safe | Round 2 | ✅ |
| 7 | Shell whitelist includes interpreters and is bypassable | Round 4 | ✅ |
| 8 | False documentation claims (350 tests pass, DS V4 fixed) | Round 4 | ✅ |
| 9 | `grad.zig`/`fused.zig`/`eval.zig` hardcoded CPU Stream | Round 3 | ✅ |
| 10 | `expert_remap_test.zig` crashes | Round 4 | ✅ |
| 11 | `model_pool.zig` LRU dangling pointer | Round 3 | ✅ |
| 12 | `npy.zig` `dtype.val` compile error | Round 3 | ✅ |
| 13 | Sampler empty logits OOB | Round 6 | ✅ |
| 14 | `seq_len - 1` underflow | Round 6 | ✅ |
| 15 | `max_new_tokens - 1` underflow infinite loop | Round 6 | ✅ |
| 16 | `findCorrectionDim` base==1.0 divide-by-zero | Round 6 | ✅ |
| 17 | `createCausalMask`/`batch_builder` mask overflow | Round 6 | ⚠️ |
| 18 | `topkIndices` k>n_experts underflow | Round 6 | ✅ |
| 19 | `compress_ratio==0` divide-by-zero | Round 6 | ✅ |
| 20 | `window_size==0` divide-by-zero | Round 6 | ✅ |
| 21 | `page_size==0` divide-by-zero | Round 6 | ✅ |
| 22 | `expandQuantizedBuffer` shape calculation error | Round 6 | ✅ |
| 23 | `@divTrunc` causes quantized buffer too small | Round 6 | ✅ |
| 24 | Unknown binary files in repo | Round 4 | ✅ |
| 25 | `tool_executor.zig` directory creation failure silently ignored | Round 4 | ✅ |
| 26 | `guided.zig` JSON parsing can be spoofed | Round 3 | ✅ |
| 27 | `server.zig` 64KB buffer overflow risk | Round 4 | ✅ |

### 🟡 P1 Defects (Performance/Functionality/Stability)

| # | Defect | Source |
|---|------|------|
| 28 | AdamW 9 scalar allocations per step (comment says ~15) | Round 3 |
| 29 | `trainer.zig` double-free risk | Round 3 |
| 30 | `guided.zig` string search not JSON parsing | Round 3 |
| 31 | `batch_builder.zig` O(N²) mask | Round 3 |
| 32 | `optim.zig` references `compiledAdamWStep` but unused | Round 3 |
| 33 | `closure.zig` ownership bug | Round 3 |
| 34 | `benchmark.zig` 4096 silent truncation | Round 3 |
| 35 | `diffusion/vae.zig` zero weights | Round 3 |
| 36 | `build.zig` memory leak | Round 4 |
| 37 | MLX-C no version check | Round 5 |
| 38 | `zig-pkg/` tracked by git (dual-source risk) | Round 5 |
| 39 | Server no signal handling | Round 5 |
| 40 | No request-level timeout | Round 5 |
| 41 | No MLX error classification | Round 5 |
| 42 | CJK regex range mismatch | Round 5 |
| 43 | HTTP missing charset | Round 5 |
| 44 | `jsonEscapeInto` incomplete | Round 5 |
| 45 | `tool_calling.zig` quote skip bug | Round 5 |
| 46 | `deepseek_v4.zig` YARN RoPE 30GB | Round 6 |
| 47 | Stream mode full tensor loading | Round 6 |
| 48 | `RotatingKVCache` per-token loop | Round 6 |
| 49 | `DSV4YarnRoPE.apply` OOB (ndim/start_pos) | Round 6 |
| 50 | `DSV4Attention` ndim assumption | Round 6 |
| 51 | `dequantIfNeeded` hardcoded affine | Round 6 |
| 52 | `dispatchGatherMm` only recognizes mxfp4 | Round 6 |
| 53 | `dequantFp4` float division | Round 6 |
| 54 | `dequantFp8` scale mismatch | Round 6 |
| 55 | QJL matrix transpose error | Round 6 |
| 56 | `ops/fused.zig` leak paths | Round 6 |
| 57 | `closure.zig` leak paths | Round 6 |
| 58 | C API return values ignored | Round 6 |
| 59 | Softplus float16 overflow | Round 6 |
| 60 | 7 phantom features | Round 6 |
| 61 | OpenAI API fields missing | Round 6 |
| 62 | `ignore_unknown_fields=false` | Round 6 |
| 63 | HTTP case-sensitive headers | Round 6 |
| 64 | SSE multi-line data unsafe | Round 6 |
| 65 | Error response format inconsistent | Round 6 |

### 🟢 P2 Defects (Ergonomics/Conventions/Minor)

| # | Defect | Source |
|---|------|------|
| 66 | ~432 functions camelCase | Round 5 |
| 67 | No unified error type | Round 5 |
| 68 | `Array` no method chaining | Round 5 |
| 69 | NN layer fields all pub | Round 5 |
| 70 | `Array.zeros`/`ones` misleading allocator | Round 5 |
| 71 | `std.mem.indexOf`/`find` mixed usage | Round 5 |
| 72 | No SIMD | Round 5 |
| 73 | Tests no mocking framework | Round 6 |
| 74 | No performance regression tests | Round 6 |
| 75 | KV cache "comptime shape" false claim | Round 5 |
| 76 | No Unicode normalization | Round 5 |
| 77 | `safety_margin_bytes` hardcoded 512MB | Round 4 |
| 78 | `LlamaConfig.rope_theta` default 10000 | Round 4 |
| 79 | `ServerConfig.kv_bits` default 4 | Round 4 |

---

## Fix Recommendations (By Priority)

### Urgent (This Week)
1. **Fix Llama causal mask** — pass causal mask on prefill path at `llama.zig:657`
2. **Fix sampler** — replace `std.sort.insertion` with `std.mem.sort`; or prefer MLX GPU `topk`
3. **Fix all divide-by-zero paths** — add `> 0` validation in `findCorrectionDim`, `compress_ratio`, `window_size`, `page_size`
4. **Fix `max_new_tokens - 1` underflow** — use `saturatingSub` or prefix validation
5. **Fix `StreamState` static storage** — pass via context pointer instead
6. **Remove/isolate unknown binaries** — `check_types`, `main`, `test_*`

### High Priority (This Month)
7. **Fix DeepSeek V4 YARN RoPE memory** — compute on-demand instead of precomputing all 30GB; or document only supporting smaller max_seq_len
8. **Fix `expandQuantizedBuffer` shape error** — `dim` should use `head_dim` not `seq_len * el_per_int`
9. **Fix `custom_kernel.zig` dangling pointer** — ensure string lifetime covers kernel object lifetime
10. **Fix `last_error_buffer` thread safety** — add `threadlocal` or switch to thread-local storage
11. **Fix `tool_executor.zig` whitelist** — remove all interpreters and network tools; or add argument-level validation
12. **Fix false documentation claims** — correct `competitive-advantages.md` and `FIX-REPORT-DEEPSEEK-V4.md`
13. **Add MLX-C version check** — `pkg-config --modversion mlxc`

### Medium Priority (This Quarter)
14. **Fix Stream mode expert loading** — restore partial reads (`PartialTensorReader`) instead of loading full tensors
15. **Fix `RotatingKVCache` per-token loop** — use batch slice_update
16. **Fix BatchBuilder mask rebuild** — cache trivial mask for decode requests
17. **Fix `grad.zig`/`fused.zig`/`eval.zig` CPU stream** — pass correct GPU stream
18. **Unify naming convention** — snake_case public functions
19. **Fix JSON escaping** — add `\b`, `\f` and all control characters `\u00XX`
20. **Fix OpenAI API compatibility** — `ignore_unknown_fields = true`, add missing fields

### Low Priority (Technical Debt)
21. Split `deepseek_v4.zig` and `deepseek_v4_loader.zig`
22. Add unified factory for KV cache
23. Remove/isolate 7 phantom features
24. Add signal handling (SIGINT/SIGTERM)
25. Add request-level timeouts
26. Add CI/CD (GitHub Actions)

---

*This report is based on six independent dimensional reviews of the mlx-zig codebase. All claims have been cross-verified against source code. Report generated: 2026-05-01.*
