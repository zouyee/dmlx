# Server KV Cache Fix — DeepSeek V4 Specialized Caches

## Problem

The server's `loadModel()` function created generic `StandardKVCache` for all architectures (line 490-503 in `server.zig`). This caused DeepSeek V4 to run with incorrect KV cache types, leading to reshape errors with paged/quantized strategies.

DeepSeek V4 uses **heterogeneous attention**:
- Layers with `compress_ratio > 0` need `DeepseekV4Cache` (with Compressor/Indexer state for MLA KV compression)
- Layers with `compress_ratio == 0` need `RotatingWithWindow` (sliding-window attention)

The generic `StandardKVCache` lacks the compressor/indexer state needed by MLA layers.

## Fix

### 1. Server Cache Creation (`src/server.zig`)

**Before:**
```zig
// All architectures use standard caches
for (0..mc.num_layers) |i| {
    caches[i] = try kvcache.createStandard(allocator, layer_config, stream);
}
```

**After:**
```zig
if (std.mem.eql(u8, arch_name, "DeepseekV4ForCausalLM")) {
    // DSv4: per-layer specialized caches (DeepseekV4Cache / RotatingWithWindow)
    const dsv4_config = try root.deepseek_v4_loader.parseDSV4Config(allocator, config_content);
    defer allocator.free(dsv4_config.compress_ratios);
    caches = try root.deepseek_v4_loader.makeV4Caches(allocator, &dsv4_config, stream);
} else {
    // Other architectures: standard caches
    // (unchanged)
}
```

### 2. Dead Code Cleanup (`src/eval.zig`)

Removed unused `defaultStream()` inline helper (same pattern fixed in `grad.zig` and `io/mlx_io.zig` in tasks R1-R2).

### 3. Test Script Update (`scripts/e2e_server.sh`)

Updated the "KNOWN LIMITATION" comment to reflect that specialized caches are now properly created for DeepSeek V4.

## Architecture

```
Engine Loop (batch forward)
  └─ state.vtable.forward()
       └─ DeepseekV4VTableAdapter.forward()
            └─ self.model.forward(input, mask, caches, start_pos, stream)
                 └─ DSV4Attention.forward(hidden, mask, cache, start_pos, stream)
                      └─ cache.updateAndFetch()  ← works with DeepseekV4Cache VTable

Request-level generation (single request)
  └─ state.vtable.nativeGenerate ← nativeGenerateFn
       └─ makeV4Caches() ← creates NEW specialized caches per call
       └─ model.generate(prompt, max_tokens, sampler, caches, stream)
```

### Why Two Cache Creation Points?

1. **Engine Loop**: Uses `state.caches` (created once in `loadModel`) for batched forward passes. These must be specialized for DSv4.

2. **nativeGenerate**: Creates fresh caches per call via `makeV4Caches()`. This was already correct — it bypassed the generic caches entirely.

## Verification

### Build
```bash
zig build  # ✅ compiles without errors
```

### Tests
- `zig build test` — unit tests
- `scripts/e2e_server.sh` — server-mode end-to-end (7 core + 12 extended prompts)
- `scripts/run_benchmark.sh` — performance benchmark vs baseline
