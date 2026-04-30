# Design: Inference Engine Components

**Parent**: `design.md` §Components and Interfaces → 2. Inference Engine Components

## 2.1 Three-Layer Generation API (`src/generation.zig`)

### Types

```zig
pub const GenerateConfig = struct {
    max_tokens: usize = 256,
    temperature: f32 = 0.8,
    top_k: usize = 50,
    top_p: f32 = 1.0,
    stop_tokens: []const u32 = &.{},
    seed: u64 = 0,
    repetition_penalty: f32 = 1.0,
};

pub const SampleResult = struct {
    token: u32,
    logprob: f32,
};

pub const ModelConfig = struct {
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    hidden_size: usize,
    /// Per-layer compression ratios for heterogeneous KV cache sizing (DeepSeek V4).
    /// Empty for models without compressed attention (e.g. LLaMA).
    /// Values: 0 or 1 = sliding-window-only, ~4 = CSA, ~128 = HCA.
    compress_ratios: []const usize = &.{},
};

pub const ForwardWithHiddenResult = struct {
    logits: Array,
    hidden: Array,
};

pub const ModelVTable = struct {
    forward: *const fn (ctx: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!Array,
    /// Optional: forward pass that also returns last-layer hidden states.
    /// Required for EAGLE speculative decoding.
    forwardWithHidden: ?*const fn (ctx: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!ForwardWithHiddenResult = null,
    deinit: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void,
    config: ModelConfig,
    ptr: *anyopaque,
};
```

### Functions

```zig
/// Layer 1: Single-step generation primitive.
/// Returns the sampled token and its log-probability.
pub fn generateStep(
    model: ModelVTable,
    tokens: Array,
    caches: []KVCacheStrategy,
    sampler: *SamplerConfig,
    ctx: EagerContext,
) !SampleResult { ... }

/// Layer 2: Streaming generation with per-token callback
pub fn streamGenerate(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    callback: *const fn (token: u32, is_done: bool) void,
) !void { ... }

/// Layer 3: Complete generation returning full sequence
pub fn generate(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
) ![]u32 { ... }
```

`streamGenerate` and `generate` are implemented in terms of `generateStep`.

### 2.1.4 Speculative Decoding — Prompt Lookup Decoding (PLD)

```zig
/// Stream-generate using a PLD drafter that proposes draft tokens from the prompt.
/// Verifies drafts with the target model; accepted tokens are emitted via callback.
pub fn streamGenerateSpeculative(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    drafter: *PldDrafter,
    callback: *const fn (token: u32, is_done: bool) void,
) !void { ... }
```

### 2.1.5 Speculative Decoding — EAGLE

```zig
/// Stream-generate using an EAGLE draft head that predicts future tokens from
/// last-layer hidden states. Requires `model.forwardWithHidden` to be non-null.
pub fn streamGenerateEagle(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    eagle_drafter: *EagleDrafter,
    callback: *const fn (token: u32, is_done: bool) void,
) !void { ... }
```

## 2.2 Model Registry (`src/model_registry.zig`)

```zig
pub const RegistryError = error{
    UnsupportedArchitecture,
};

/// Lookup a loader function by architecture name.
/// Returns `error.UnsupportedArchitecture` if the name is not in the registry.
pub fn getLoader(arch_name: []const u8) RegistryError!ModelLoader { ... }

/// Compile-time registry mapping architecture names to loader functions.
pub const supported_architectures = [_][]const u8{
    "LlamaForCausalLM",
    "DeepseekV4ForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "GemmaForCausalLM",
    "Glm4ForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "LlavaForConditionalGeneration",
};

pub const model_registry = std.StaticStringMap(ModelLoader).initComptime(.{
    .{ "LlamaForCausalLM", llamaLoader },
    .{ "DeepseekV4ForCausalLM", deepseekV4Loader },
    .{ "MistralForCausalLM", llamaLoader },      // Mistral uses LLaMA arch
    .{ "Qwen2ForCausalLM", llamaLoader },         // Qwen2 uses LLaMA arch with q/k norms
    .{ "Qwen3ForCausalLM", qwen3Loader },
    .{ "GemmaForCausalLM", gemmaLoader },
    .{ "Glm4ForCausalLM", glm4Loader },
    .{ "PhiForCausalLM", phiLoader },
    .{ "Phi3ForCausalLM", phi3Loader },
    .{ "LlavaForConditionalGeneration", llavaLoader },
});

pub const ModelLoader = *const fn (
    allocator: std.mem.Allocator,
    config_json: []const u8,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    io: std.Io,
    smelt: deepseek_v4_loader.SmeltConfig,
) anyerror!ModelVTable;
```

<!-- DIVERGENCE: `ModelLoader` 签名强制所有 loader 接受 `deepseek_v4_loader.SmeltConfig`，
     即使非 SMoE 模型（LLaMA、Gemma 等）也忽略该参数。这是因为 `StaticStringMap`
     要求同质函数指针。设计意图是解耦为通用配置结构（如 `LoadOptions`），但代码
     目前直接透传 SMoE 专用配置。
     **为准方**: 代码。注册表同质性约束当前优先。未来重构注册表为 variant/union
     类型后可消除此耦合。 -->

## 2.3 Prompt Cache (`src/prompt_cache.zig`)

Serializes/deserializes KV cache state using the existing safetensors I/O:

```zig
pub fn savePromptCache(
    allocator: std.mem.Allocator,
    caches: []KVCacheStrategy,
    path: []const u8,
) !void {
    // For each layer, extract keys/values arrays
    // Store as safetensors with metadata: {num_layers, head_dim, num_kv_heads, seq_len}
}

pub fn loadPromptCache(
    allocator: std.mem.Allocator,
    path: []const u8,
    model_config: ModelConfig,
) ![]KVCacheStrategy {
    // Load safetensors, validate metadata against model_config
    // Reconstruct KV cache strategies with loaded data
}
```

## 2.4 Operator Fusion (`src/ops/fused.zig`)

Uses `src/compile.zig` (already bound to `mlx_compile`) to fuse composite operations:

```zig
/// Compiled SwiGLU MLP: gate_proj + silu + up_proj + down_proj as single fused op
pub fn compiledSwiGLU(ctx: EagerContext) !Closure { ... }

/// Compiled AdamW step: ~15 intermediate arrays fused into one kernel launch
pub fn compiledAdamWStep(ctx: EagerContext) !Closure { ... }
```
