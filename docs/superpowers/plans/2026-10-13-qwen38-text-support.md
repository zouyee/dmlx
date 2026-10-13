# Qwen3.8-27B Text-Only Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add text-only inference support for `Qwen3_5ForConditionalGeneration` (`/Users/zouyee/models/Qwen3.8-27B-4bit`) to dmlx, on both CLI chat and server paths.

**Architecture:** New module family mirroring the proven GLM-5.3 structure: `qwen38_loader.zig` (config parse), `qwen38_gdn.zig` (gated delta net, forked from `glm53_kda.zig`), `qwen38_attn.zig` (full attention with partial RoPE + sigmoid output gate), `qwen38.zig` (model assembly + serving model with internal caches). Registered as `"Qwen3_5ForConditionalGeneration"` in `model_registry.zig`. All compute via existing 4-bit affine gs=64 `quantizedMatmul` path.

**Tech Stack:** Zig 0.16, MLX C API via `mlx_z` bindings (`ops`, `fast`, `quantize`, `shape`, `reduce`, `math`, `activations`), safetensors via `TensorIndex`.

**Spec:** `docs/superpowers/specs/2026-10-13-qwen38-text-support-design.md`

---

## Verified facts (do not re-derive)

From the real checkpoint index/headers (`model-0000{1,2,3}-of-00003.safetensors`):

- All weight keys are prefixed `language_model.`; vision keys prefixed `vision_tower.` (ignored).
- Everything is 4-bit affine gs=64 quantized (weight = U32 packed `[out, in/8]`, scales/biases BF16 `[out, in/64]`), including `embed_tokens` and `lm_head`. Only `*.input_layernorm.weight`, `*.post_attention_layernorm.weight`, `model.norm.weight`, `linear_attn.{A_log,dt_bias,conv1d.weight,norm.weight}`, `self_attn.{q_norm,k_norm}.weight` are plain BF16.
- Linear-attn (GDN) layer weights, layer N (`language_model.model.layers.N.linear_attn.`):
  - `in_proj_qkv.weight` packed out=10240 → real out 10240 = q(16×128=2048) + k(2048) + v(48×128=6144)
  - `in_proj_z.weight` out 6144 (output gate), `in_proj_a.weight` out 48 (decay input, per-head), `in_proj_b.weight` out 48 (beta, per-head)
  - `conv1d.weight` `[10240, 4, 1]` (depthwise over qkv channels), `A_log` `[48]`, `dt_bias` `[48]`, `norm.weight` `[128]`, `out_proj` in 6144 → out 5120
- Full-attn layer (N=3,7,…,63; `self_attn.`): `q_proj` out 12288 = 24 heads × 512 (per-head `[q 256 | gate 256]` interleaved, HF Qwen3-Next chunk convention), `k_proj`/`v_proj` out 1024 = 4 kv heads × 256, `q_norm`/`k_norm.weight` `[256]`, `o_proj` in 6144 → 5120
- MLP (every layer): `gate_proj`/`up_proj` out 17408, `down_proj` out 5120 (SwiGLU, `hidden_act=silu`)
- `layer_types`: 64 entries, `"linear_attention"` ×3 then `"full_attention"` (full layers: 3,7,11,…,63)
- Config: hidden 5120, vocab 248320, head_dim 256, 24 q heads, 4 kv heads, `partial_rotary_factor=0.25` (rotary dims=64), `rope_theta=1e7`, `rms_norm_eps=1e-6`, `linear_num_key_heads=16`, `linear_num_value_heads=48`, `linear_key_head_dim=128`, `linear_value_head_dim=128`, `linear_conv_kernel_dim=4`, eos `[248046, 248044]`

Reference implementations to mirror (read before writing code):
- `src/models/glm53_kda.zig` — GDN skeleton: conv window cache, delta-rule recurrence, per-step `eval()`, `prefill` loop
- `src/models/glm53.zig:43-148` — `QuantParts` / `WeightSource` / `fmtName`; `:1078-1116` quantized embedding + lm_head pattern; `:1119-1248` serving model with internal caches + `forwardTokens` contract (L>1 resets state)
- `src/models/glm53_loader.zig` — nested `text_config` JSON parsing pattern, eos list parsing
- `fast.rope(ctx, x, dims, traditional, base, scale, offset, freqs)` — `dims=64` gives partial rotary (`zig-pkg/mlx_z*/src/ops/fast.zig:25`); `fast.scaledDotProductAttention(ctx, q, k, v, scale, mask_mode, mask_arr, sinks)` (`fast.zig:39`)

## Qwen3-Next GDN math (default implementation; every formula is oracle-verified in Task 5)

Per decode token, x `[1, 5120]`:

1. `qkv = silu(depthwise_conv1d(in_proj_qkv(x), k=4))` — sliding window cache `[3, 10240]`
2. Split q `[2048]`/k `[2048]`/v `[6144]` → reshape q,k to `[16,128]`, v to `[48,128]`
3. q,k: **l2norm per head** (`x / max(||x||_2, eps)`, eps=1e-6); q additionally scaled by `1/sqrt(128)`
4. `a = in_proj_a(x)` `[48]`, `b = in_proj_b(x)` `[48]`
5. Decay: `g = exp(-exp(A_log) * softplus(a + dt_bias))` per value head `[48]`; `beta = sigmoid(b)` `[48]`
6. Broadcast q,k from 16 key heads to 48 value heads: head h uses q/k head `h / 3` (kv-group ratio = 48/16 = 3)
7. Delta rule per value head (state `S [48,128,128]` f32, layout `[Hv, Dv, Dk]`):
   `S ← S*g; kv = (S·k).sum(Dk); d = (v−kv)*beta; S ← S + k⊗d; y = (S·q).sum(Dk)`
8. Output gate + norm (gated RMSNorm, fla convention): `y = rmsnorm(y * silu(z), weight=norm.weight, eps=1e-6)` where `z = in_proj_z(x)` `[48,128]`
9. `out = out_proj(flatten(y))` `[1, 5120]`

Note the decay formula differs from GLM (GLM: `exp(lower_bound·sigmoid(exp(A_log)·(a+dt_bias)))`; Qwen: `exp(-exp(A_log)·softplus(a+dt_bias))`). If Task 5 oracle alignment fails, `scripts/qwen38_gdn_ref.py` (Task 2) is the ground truth to reconcile against.

---

## Task 1: Qwen38Config parsing (qwen38_loader.zig)

**Files:**
- Create: `src/models/qwen38_loader.zig`
- Test: inline `test` blocks in the same file

- [ ] **Step 1: Write the failing test + empty module**

Create `src/models/qwen38_loader.zig` with only the test:

```zig
//! Qwen3.8 (qwen3_5) config.json parsing. Language-model hyperparameters live
//! under `text_config`; quantization defaults under top-level `quantization`.
const std = @import("std");

pub const LoadError = error{
    InvalidConfig,
    UnsupportedArchitecture,
};

test "parseQwen38Config: text_config nesting + layer_types" {
    const allocator = std.testing.allocator;
    const json_text =
        \\{"model_type":"qwen3_5","eos_token_id":[248046,248044],
        \\  "quantization":{"group_size":64,"bits":4,"mode":"affine"},
        \\  "text_config":{
        \\    "hidden_size":5120,"num_hidden_layers":64,"vocab_size":248320,
        \\    "intermediate_size":17408,"head_dim":256,
        \\    "num_attention_heads":24,"num_key_value_heads":4,
        \\    "partial_rotary_factor":0.25,"rms_norm_eps":1e-6,
        \\    "linear_num_key_heads":16,"linear_num_value_heads":48,
        \\    "linear_key_head_dim":128,"linear_value_head_dim":128,
        \\    "linear_conv_kernel_dim":4,
        \\    "attn_output_gate":true,
        \\    "rope_parameters":{"rope_theta":10000000},
        \\    "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]
        \\  }}
    ;
    var cfg = try parseQwen38Config(allocator, json_text);
    defer cfg.deinit();
    try std.testing.expectEqual(@as(u32, 64), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 5120), cfg.hidden_size);
    try std.testing.expectEqual(@as(u32, 248320), cfg.vocab_size);
    try std.testing.expectEqual(@as(u32, 16), cfg.linear_num_key_heads);
    try std.testing.expectEqual(@as(u32, 48), cfg.linear_num_value_heads);
    try std.testing.expectEqual(@as(u32, 128), cfg.linear_key_head_dim);
    try std.testing.expectEqual(@as(u32, 128), cfg.linear_value_head_dim);
    try std.testing.expectEqual(@as(u32, 24), cfg.num_attention_heads);
    try std.testing.expectEqual(@as(u32, 4), cfg.num_key_value_heads);
    try std.testing.expectEqual(@as(u32, 256), cfg.head_dim);
    try std.testing.expectEqual(@as(usize, 64), cfg.rotaryDims());
    try std.testing.expectEqual(@as(f32, 1e7), cfg.rope_theta);
    try std.testing.expectEqual(@as(usize, 2), cfg.eos_token_ids.len);
    try std.testing.expectEqual(@as(u32, 248046), cfg.eos_token_ids[0]);
    try std.testing.expect(cfg.isFullAttention(3));
    try std.testing.expect(!cfg.isFullAttention(0));
    try std.testing.expect(!cfg.isFullAttention(64)); // out of range is safe
    try std.testing.expectEqual(@as(u8, 4), cfg.quant_bits_default);
    try std.testing.expectEqual(@as(u32, 64), cfg.quant_group_size);
}

test "parseQwen38Config: missing fields fall back to real-config defaults" {
    const allocator = std.testing.allocator;
    var cfg = try parseQwen38Config(allocator, "{\"model_type\":\"qwen3_5\",\"text_config\":{\"num_hidden_layers\":64}}");
    defer cfg.deinit();
    try std.testing.expectEqual(@as(u32, 5120), cfg.hidden_size);
    try std.testing.expectEqual(@as(u32, 248320), cfg.vocab_size);
    try std.testing.expectEqual(@as(u32, 17408), cfg.intermediate_size);
    try std.testing.expectEqual(@as(usize, 0), cfg.eos_token_ids.len);
    try std.testing.expect(!cfg.isFullAttention(0));
}

test "parseQwen38Config: rejects non-qwen3_5 model_type" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        LoadError.UnsupportedArchitecture,
        parseQwen38Config(allocator, "{\"model_type\":\"llama\"}"),
    );
}
```

- [ ] **Step 2: Run test, verify compile failure**

Run: `zig test src/models/qwen38_loader.zig 2>&1 | head -5`
Expected: FAIL — `parseQwen38Config` undefined.

- [ ] **Step 3: Implement Qwen38Config + parseQwen38Config**

Insert above the tests:

```zig
pub const Qwen38Config = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32 = 5120,
    num_hidden_layers: u32 = 64,
    vocab_size: u32 = 248320,
    intermediate_size: u32 = 17408,
    num_attention_heads: u32 = 24,
    num_key_value_heads: u32 = 4,
    head_dim: u32 = 256,
    partial_rotary_factor: f32 = 0.25,
    rope_theta: f32 = 1e7,
    rms_norm_eps: f32 = 1e-6,
    attn_output_gate: bool = true,
    linear_num_key_heads: u32 = 16,
    linear_num_value_heads: u32 = 48,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,
    eos_token_ids: []u32 = &.{},
    full_attn_layers: []u32 = &.{},
    quant_bits_default: u8 = 4,
    quant_group_size: u32 = 64,

    pub fn deinit(self: *Qwen38Config) void {
        if (self.eos_token_ids.len > 0) self.allocator.free(self.eos_token_ids);
        if (self.full_attn_layers.len > 0) self.allocator.free(self.full_attn_layers);
    }

    pub fn isFullAttention(self: *const Qwen38Config, layer: u32) bool {
        for (self.full_attn_layers) |l| {
            if (l == layer) return true;
        }
        return false;
    }

    /// Rotary dims = head_dim * partial_rotary_factor (64 for the real config).
    pub fn rotaryDims(self: *const Qwen38Config) usize {
        return @intFromFloat(@as(f32, @floatFromInt(self.head_dim)) * self.partial_rotary_factor);
    }

    /// kv-group ratio: value heads per key head (3 for the real config).
    pub fn kvGroupRatio(self: *const Qwen38Config) u32 {
        return self.linear_num_value_heads / self.linear_num_key_heads;
    }
};

const getInt = struct {
    fn call(o: std.json.ObjectMap, key: []const u8) ?i64 {
        const val = o.get(key) orelse return null;
        return switch (val) {
            .integer => |n| n,
            .float => |f| @intFromFloat(f),
            else => null,
        };
    }
}.call;

const getFloat = struct {
    fn call(o: std.json.ObjectMap, key: []const u8) ?f64 {
        const val = o.get(key) orelse return null;
        return switch (val) {
            .float => |f| f,
            .integer => |n| @floatFromInt(n),
            else => null,
        };
    }
}.call;

const getBool = struct {
    fn call(o: std.json.ObjectMap, key: []const u8) ?bool {
        const val = o.get(key) orelse return null;
        return switch (val) {
            .bool => |b| b,
            else => null,
        };
    }
}.call;

fn nestedInt(text: ?std.json.ObjectMap, top: std.json.ObjectMap, key: []const u8) ?i64 {
    if (text) |t| {
        if (getInt(t, key)) |v| return v;
    }
    return getInt(top, key);
}

fn nestedFloat(text: ?std.json.ObjectMap, top: std.json.ObjectMap, key: []const u8) ?f64 {
    if (text) |t| {
        if (getFloat(t, key)) |v| return v;
    }
    return getFloat(top, key);
}

fn toU32(v: ?i64, default: u32) u32 {
    const n = v orelse return default;
    return std.math.cast(u32, n) orelse default;
}

fn toF32(v: ?f64, default: f32) f32 {
    const f = v orelse return default;
    return @floatCast(f);
}

/// Parse Qwen3.8 config.json into Qwen38Config.
/// Caller must call `cfg.deinit()` to free owned slices.
pub fn parseQwen38Config(allocator: std.mem.Allocator, json_text: []const u8) !Qwen38Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return LoadError.InvalidConfig;
    const obj = root.object;

    if (obj.get("model_type")) |mt| {
        if (mt == .string and !std.mem.startsWith(u8, mt.string, "qwen3_5")) {
            return LoadError.UnsupportedArchitecture;
        }
    }

    const text: ?std.json.ObjectMap = blk: {
        const tv = obj.get("text_config") orelse break :blk null;
        if (tv != .object) break :blk null;
        break :blk tv.object;
    };

    var cfg = Qwen38Config{ .allocator = allocator };
    errdefer cfg.deinit();

    cfg.hidden_size = toU32(nestedInt(text, obj, "hidden_size"), cfg.hidden_size);
    cfg.num_hidden_layers = toU32(nestedInt(text, obj, "num_hidden_layers"), cfg.num_hidden_layers);
    cfg.vocab_size = toU32(nestedInt(text, obj, "vocab_size"), cfg.vocab_size);
    cfg.intermediate_size = toU32(nestedInt(text, obj, "intermediate_size"), cfg.intermediate_size);
    cfg.num_attention_heads = toU32(nestedInt(text, obj, "num_attention_heads"), cfg.num_attention_heads);
    cfg.num_key_value_heads = toU32(nestedInt(text, obj, "num_key_value_heads"), cfg.num_key_value_heads);
    cfg.head_dim = toU32(nestedInt(text, obj, "head_dim"), cfg.head_dim);
    cfg.partial_rotary_factor = toF32(nestedFloat(text, obj, "partial_rotary_factor"), cfg.partial_rotary_factor);
    cfg.rms_norm_eps = toF32(nestedFloat(text, obj, "rms_norm_eps"), cfg.rms_norm_eps);
    cfg.attn_output_gate = blk: {
        if (text) |t| {
            if (getBool(t, "attn_output_gate")) |b| break :blk b;
        }
        break :blk cfg.attn_output_gate;
    };
    cfg.linear_num_key_heads = toU32(nestedInt(text, obj, "linear_num_key_heads"), cfg.linear_num_key_heads);
    cfg.linear_num_value_heads = toU32(nestedInt(text, obj, "linear_num_value_heads"), cfg.linear_num_value_heads);
    cfg.linear_key_head_dim = toU32(nestedInt(text, obj, "linear_key_head_dim"), cfg.linear_key_head_dim);
    cfg.linear_value_head_dim = toU32(nestedInt(text, obj, "linear_value_head_dim"), cfg.linear_value_head_dim);
    cfg.linear_conv_kernel_dim = toU32(nestedInt(text, obj, "linear_conv_kernel_dim"), cfg.linear_conv_kernel_dim);

    // rope_theta lives under text_config.rope_parameters.rope_theta.
    if (text) |t| {
        if (t.get("rope_parameters")) |rp| {
            if (rp == .object) {
                cfg.rope_theta = toF32(getFloat(rp.object, "rope_theta"), cfg.rope_theta);
            }
        }
    }

    // eos_token_id: integer or array; text_config first, top-level fallback.
    const eos_val: ?std.json.Value = blk: {
        if (text) |t| {
            if (t.get("eos_token_id")) |v| break :blk v;
        }
        break :blk obj.get("eos_token_id");
    };
    if (eos_val) |val| {
        switch (val) {
            .integer => |n| {
                if (std.math.cast(u32, n)) |id| {
                    const ids = try allocator.alloc(u32, 1);
                    ids[0] = id;
                    cfg.eos_token_ids = ids;
                }
            },
            .array => |arr| {
                var list = std.ArrayList(u32).empty;
                defer list.deinit(allocator);
                for (arr.items) |item| {
                    const n: ?i64 = switch (item) {
                        .integer => |v| v,
                        .float => |f| @intFromFloat(f),
                        else => null,
                    };
                    if (n) |v| {
                        if (std.math.cast(u32, v)) |id| try list.append(allocator, id);
                    }
                }
                if (list.items.len > 0) cfg.eos_token_ids = try list.toOwnedSlice(allocator);
            },
            else => {},
        }
    }

    // layer_types: collect indices of "full_attention" entries.
    if (text) |t| {
        if (t.get("layer_types")) |lt| {
            if (lt == .array) {
                var list = std.ArrayList(u32).empty;
                defer list.deinit(allocator);
                for (lt.array.items, 0..) |item, i| {
                    if (item != .string) return LoadError.InvalidConfig;
                    if (std.mem.eql(u8, item.string, "full_attention")) {
                        try list.append(allocator, @intCast(i));
                    } else if (!std.mem.eql(u8, item.string, "linear_attention")) {
                        std.log.err("qwen38: unknown layer_types[{d}] = \"{s}\"", .{ i, item.string });
                        return LoadError.InvalidConfig;
                    }
                }
                if (list.items.len > 0) cfg.full_attn_layers = try list.toOwnedSlice(allocator);
            }
        }
    }

    // Top-level quantization defaults.
    if (obj.get("quantization")) |q_val| {
        if (q_val == .object) {
            const q = q_val.object;
            cfg.quant_bits_default = @intCast(@min(255, @max(0, getInt(q, "bits") orelse cfg.quant_bits_default)));
            cfg.quant_group_size = toU32(getInt(q, "group_size"), cfg.quant_group_size);
        }
    }

    return cfg;
}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `zig test src/models/qwen38_loader.zig`
Expected: 3/3 pass.

- [ ] **Step 5: Smoke against the real config.json**

Run: `zig test src/models/qwen38_loader.zig --test-filter "real config" 2>/dev/null || true` — instead, add a temporary check via the CLI after Task 5; here just verify parsing manually:

```bash
python3 -c "import json; c=json.load(open('/Users/zouyee/models/Qwen3.8-27B-4bit/config.json')); print(c['model_type'], len(c['text_config']['layer_types']))"
```
Expected: `qwen3_5 64`.

- [ ] **Step 6: Commit**

```bash
git add src/models/qwen38_loader.zig
git commit -m "feat(qwen38): Qwen38Config parsing (text_config nesting + layer_types)"
```

---

## Task 2: GDN numpy reference + cross-check fixture

**Files:**
- Create: `scripts/qwen38_gdn_ref.py`

This script is the numerical ground truth for Task 3's unit test, mirroring `scripts/glm53_kda_ref.py`. It uses the same deterministic weight fill as the Zig test (`(i*37+11) mod 13 - 6) * 0.02`).

- [ ] **Step 1: Write the reference script**

```python
#!/usr/bin/env python3
"""Qwen3.8 GDN (gated delta net) single-step numpy reference.

Weights use the same deterministic fill as src/models/qwen38_gdn.zig tests:
  buf[i] = ((i*37+11) % 13 - 6) * 0.02
Dims (toy): hidden=8, key heads=2, value heads=6 (ratio 3), head_dim=4, conv k=4.
Prints the step output [8] and the final ssm state [6,4,4] flattened.
"""
import numpy as np

HIDDEN, HK, HV, D, K = 8, 2, 6, 4, 4
PQ, PK, PV = HK * D, HK * D, HV * D  # 8, 8, 24
QKV = PQ + PK + PV  # 40

def fill(n):
    return np.array([((i * 37 + 11) % 13 - 6) * 0.02 for i in range(n)], dtype=np.float64)

def rms(x, eps):
    return x / np.sqrt((x ** 2).mean() + eps)

def l2n(x, eps=1e-6):
    return x / np.maximum(np.linalg.norm(x), eps)

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

# weights
in_proj_qkv = fill(QKV * HIDDEN).reshape(QKV, HIDDEN)
conv_w = fill(QKV * K).reshape(QKV, K)
in_proj_a = fill(HV * HIDDEN).reshape(HV, HIDDEN)
in_proj_b = fill(HV * HIDDEN).reshape(HV, HIDDEN)
in_proj_z = fill(PV * HIDDEN).reshape(PV, HIDDEN)
A_log = np.array([-0.5, -0.6, -0.7, -0.8, -0.9, -1.0])
dt_bias = fill(HV)
norm_w = np.array([1.1, 0.9, 1.05, 0.95])
out_proj = fill(HIDDEN * PV).reshape(HIDDEN, PV)

x = np.array([((i * 13 + 5) % 11 - 5) * 0.1 for i in range(HIDDEN)], dtype=np.float64)

# 1. qkv proj + depthwise conv (window = 3 zeros + current) + silu
qkv_lin = in_proj_qkv @ x
window = np.concatenate([np.zeros((K - 1, QKV)), qkv_lin[None, :]], axis=0)
conv = (window * conv_w.T[None, :, :] if False else
        sum(window[j] * conv_w[:, j] for j in range(K)))
def silu(v): return v / (1 + np.exp(-v))
qkv = silu(conv)

q = qkv[:PQ].reshape(HK, D)
k = qkv[PQ:PQ + PK].reshape(HK, D)
v = qkv[PQ + PK:].reshape(HV, D)

# 2. l2norm q/k per head, scale q
scale = 1.0 / np.sqrt(D)
q = np.stack([l2n(q[h]) * scale for h in range(HK)])
k = np.stack([l2n(k[h]) for h in range(HK)])

# 3. gates
a = in_proj_a @ x  # [HV]
b = in_proj_b @ x  # [HV]
z = in_proj_z @ x  # [PV]
g = np.exp(-np.exp(A_log) * softplus(a + dt_bias))  # [HV]
beta = 1 / (1 + np.exp(-b))

# 4. broadcast q/k to value heads (ratio 3)
ratio = HV // HK
qb = np.repeat(q, ratio, axis=0)  # [HV, D]
kb = np.repeat(k, ratio, axis=0)

# 5. delta rule from zero state
S = np.zeros((HV, D, D))
S = S * g[:, None, None]
kv = (S * kb[:, None, :]).sum(-1)      # [HV, D]
d = (v - kv) * beta[:, None]
S = S + kb[:, :, None] * d[:, None, :] # wait: S layout [Hv, Dv, Dk]
# S[h,i,j] += k[h,j] * d[h,i]
S = np.zeros((HV, D, D)) * g[:, None, None]
kv = (S * kb[:, None, :]).sum(-1)
d = (v - kv) * beta[:, None]
S = S + kb[:, None, :] * d[:, :, None].transpose(0, 2, 1) * 0  # placeholder
S = S + d[:, :, None] * kb[:, None, :]
y = (S * qb[:, None, :]).sum(-1)       # [HV, D]

# 6. gated rmsnorm: rmsnorm(y * silu(z)) * norm_w
zg = z.reshape(HV, D)
yg = y * silu(zg)
yn = yg / np.sqrt((yg ** 2).mean(-1, keepdims=True) + 1e-6) * norm_w

out = out_proj @ yn.reshape(-1)
np.set_printoptions(precision=6, suppress=False, linewidth=200)
print("out:", ", ".join(f"{v_:.6f}" for v_ in out))
print("state:", ", ".join(f"{v_:.6f}" for v_ in S.reshape(-1)))
```

- [ ] **Step 2: Run and clean it up**

Run: `python3 scripts/qwen38_gdn_ref.py`
Expected: prints `out:` (8 floats) and `state:` (96 floats). Remove the exploratory duplicate lines (the `placeholder`/dead assignments above) so the delta-rule section reads exactly:

```python
S = np.zeros((HV, D, D))
S = S * g[:, None, None]
kv = (S * kb[:, None, :]).sum(-1)
d = (v - kv) * beta[:, None]
S = S + d[:, :, None] * kb[:, None, :]
y = (S * qb[:, None, :]).sum(-1)
```

State layout convention: `S[h, dv, dk]`; update `S += d[:, :, None] * k[:, None, :]`; readout `y = (S * q[:, None, :]).sum(-1)`; memory read `kv = (S * k[:, None, :]).sum(-1)`. Keep this exact convention — the Zig code in Task 3 matches it.

- [ ] **Step 3: Record reference values**

Copy the printed `out` and `state` arrays into the Task 3 unit test constants.

- [ ] **Step 4: Commit**

```bash
git add scripts/qwen38_gdn_ref.py
git commit -m "test(qwen38): GDN single-step numpy reference"
```

---

## Task 3: Qwen38GDN module (qwen38_gdn.zig)

**Files:**
- Create: `src/models/qwen38_gdn.zig`
- Reference: `src/models/glm53_kda.zig` (fork skeleton), `scripts/qwen38_gdn_ref.py` (numbers)

- [ ] **Step 1: Write the failing test**

Create `src/models/qwen38_gdn.zig` containing only the test scaffolding and tests (adapted from `glm53_kda.zig:352-495`; toy dims hidden=8, key heads=2, value heads=6, head_dim=4, conv k=4 — matches the numpy ref):

```zig
//! Qwen3.8 GDN (gated delta net) linear-attention module.
//!
//! Port of Qwen3-Next `Qwen3NextGatedDeltaNet` single-batch naive path.
//! Differs from GlmKDA: asymmetric heads (key Hk / value Hv, kv-group
//! broadcast), separate in_proj_qkv / in_proj_a / in_proj_b / in_proj_z
//! projections, softplus decay, l2norm q/k, silu output gate.
//! f32 throughout; cross-checked against scripts/qwen38_gdn_ref.py.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const math_mod = @import("mlx").math;
const fast_mod = @import("mlx").fast;
const activations = @import("mlx").activations;
const quantize_mod = @import("mlx").quantize;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const QuantConfig = quantize_mod.QuantConfig;

const TestDims = struct {
    hidden: usize = 8,
    key_heads: usize = 2,
    value_heads: usize = 6,
    head_dim: usize = 4,
    conv_kernel: usize = 4,
};

fn testFill(buf: []f32) void {
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 13) - 6)) * 0.02;
    }
}

fn makeTestWeight(allocator: std.mem.Allocator, shape_: []const i32) !Array {
    var n: usize = 1;
    for (shape_) |s| n *= @intCast(s);
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    testFill(buf);
    return Array.fromData(allocator, f32, buf, shape_);
}

fn makeTestModule(allocator: std.mem.Allocator, ctx: EagerContext, dims: TestDims) !Qwen38GDN {
    const hidden: i32 = @intCast(dims.hidden);
    const HK: i32 = @intCast(dims.key_heads);
    const HV: i32 = @intCast(dims.value_heads);
    const D: i32 = @intCast(dims.head_dim);
    const K: i32 = @intCast(dims.conv_kernel);
    const qkv_dim: i32 = (2 * HK + HV) * D;
    return .{
        .ctx = ctx,
        .hidden_size = dims.hidden,
        .num_key_heads = dims.key_heads,
        .num_value_heads = dims.value_heads,
        .head_dim = dims.head_dim,
        .conv_kernel = dims.conv_kernel,
        .in_proj_qkv = try makeTestWeight(allocator, &[_]i32{ qkv_dim, hidden }),
        .conv1d = try makeTestWeight(allocator, &[_]i32{ qkv_dim, K, 1 }),
        .in_proj_a = try makeTestWeight(allocator, &[_]i32{ HV, hidden }),
        .in_proj_b = try makeTestWeight(allocator, &[_]i32{ HV, hidden }),
        .in_proj_z = try makeTestWeight(allocator, &[_]i32{ HV * D, hidden }),
        .A_log = try Array.fromData(allocator, f32, &[_]f32{ -0.5, -0.6, -0.7, -0.8, -0.9, -1.0 }, &[_]i32{HV}),
        .dt_bias = try makeTestWeight(allocator, &[_]i32{HV}),
        .norm_w = try Array.fromData(allocator, f32, &[_]f32{ 1.1, 0.9, 1.05, 0.95 }, &[_]i32{D}),
        .out_proj = try makeTestWeight(allocator, &[_]i32{ hidden, HV * D }),
    };
}

fn makeTestInput(allocator: std.mem.Allocator) !Array {
    var buf: [8]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    return Array.fromData(allocator, f32, &buf, &[_]i32{ 1, 8 });
}

test "Qwen38GDN.step: shape, determinism, state advance" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};
    var g1 = try makeTestModule(allocator, ctx, dims);
    defer g1.deinit();
    var cache1 = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache1.deinit();

    var g2 = try makeTestModule(allocator, ctx, dims);
    defer g2.deinit();
    var cache2 = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache2.deinit();

    const x = try makeTestInput(allocator);
    defer x.deinit();

    const out1 = try g1.forward(x, &cache1);
    defer out1.deinit();
    const oshape = out1.shape();
    try std.testing.expectEqual(@as(usize, 2), oshape.len);
    try std.testing.expectEqual(@as(i32, 1), oshape[0]);
    try std.testing.expectEqual(@as(i32, 8), oshape[1]);

    const out2 = try g2.forward(x, &cache2);
    defer out2.deinit();
    const diff = try ops.subtract(ctx, out1, out2);
    defer diff.deinit();
    const abs_diff = try ops.abs(ctx, diff);
    defer abs_diff.deinit();
    const max_diff = try reduce_mod.max(ctx, abs_diff, false);
    defer max_diff.deinit();
    try max_diff.eval();
    try std.testing.expectEqual(@as(f32, 0.0), try max_diff.item(f32));

    const state_abs = try ops.abs(ctx, cache1.ssm_state);
    defer state_abs.deinit();
    const state_max = try reduce_mod.max(ctx, state_abs, false);
    defer state_max.deinit();
    try state_max.eval();
    try std.testing.expect(try state_max.item(f32) > 0.0);
}

test "Qwen38GDN.step: numpy reference cross-check" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};
    var g = try makeTestModule(allocator, ctx, dims);
    defer g.deinit();
    var cache = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache.deinit();

    const x = try makeTestInput(allocator);
    defer x.deinit();

    const out = try g.forward(x, &cache);
    defer out.deinit();
    try out.eval();
    try cache.ssm_state.eval();

    // Reference values from scripts/qwen38_gdn_ref.py (Task 2 Step 3 output).
    const expected_out = [_]f32{ /* PASTE 8 values */ };
    const got_out = try out.dataSlice(f32);
    for (expected_out, got_out) |e, o| {
        try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }
    const expected_state = [_]f32{ /* PASTE 96 values */ };
    const got_state = try cache.ssm_state.dataSlice(f32);
    try std.testing.expectEqual(@as(usize, 96), got_state.len);
    for (expected_state, got_state) |e, o| {
        try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }
}

test "Qwen38GDN.prefill: per-token loop equivalence" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};
    var g = try makeTestModule(allocator, ctx, dims);
    defer g.deinit();
    var cache = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache.deinit();

    var buf: [24]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    const x = try Array.fromData(allocator, f32, &buf, &[_]i32{ 1, 3, 8 });
    defer x.deinit();

    const out = try g.prefill(allocator, x, &cache);
    defer out.deinit();
    const oshape = out.shape();
    try std.testing.expectEqual(@as(i32, 1), oshape[0]);
    try std.testing.expectEqual(@as(i32, 3), oshape[1]);
    try std.testing.expectEqual(@as(i32, 8), oshape[2]);
    try out.eval();

    var g_ref = try makeTestModule(allocator, ctx, dims);
    defer g_ref.deinit();
    var cache_ref = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache_ref.deinit();

    var ref_buf: [24]f32 = undefined;
    var t: i32 = 0;
    while (t < 3) : (t += 1) {
        const x_t = try ops.slice(ctx, x, &[_]i32{ 0, t, 0 }, &[_]i32{ 1, t + 1, 8 }, &[_]i32{ 1, 1, 1 });
        defer x_t.deinit();
        const x_flat = try ops.reshape(ctx, x_t, &[_]i32{ 1, 8 });
        defer x_flat.deinit();
        const out_t = try g_ref.forward(x_flat, &cache_ref);
        defer out_t.deinit();
        try out_t.eval();
        const row = try out_t.dataSlice(f32);
        @memcpy(ref_buf[@intCast(t * 8)..@intCast((t + 1) * 8)], row);
    }
    const got = try out.dataSlice(f32);
    for (ref_buf, got) |e, o| {
        try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }
}
```

- [ ] **Step 2: Verify compile failure**

Run: `zig test src/models/qwen38_gdn.zig 2>&1 | head -5`
Expected: FAIL — `Qwen38GDN` / `Qwen38GdnCache` undefined.

- [ ] **Step 3: Implement cache + module**

Insert before the test section. The conv/linear/structure code follows `glm53_kda.zig:56-324` exactly except where noted:

```zig
pub const Qwen38GdnCache = struct {
    conv_state: Array, // [conv_kernel-1, qkv_dim]
    ssm_state: Array, // [value_heads, head_dim, head_dim] f32, layout [Hv, Dv, Dk]

    pub fn init(allocator: std.mem.Allocator, conv_kernel: usize, qkv_dim: usize, value_heads: usize, head_dim: usize) !Qwen38GdnCache {
        return .{
            .conv_state = try array_mod.zeros(allocator, &[_]i32{ @intCast(conv_kernel - 1), @intCast(qkv_dim) }, .float32),
            .ssm_state = try array_mod.zeros(allocator, &[_]i32{ @intCast(value_heads), @intCast(head_dim), @intCast(head_dim) }, .float32),
        };
    }

    pub fn deinit(self: *Qwen38GdnCache) void {
        self.conv_state.deinit();
        self.ssm_state.deinit();
    }
};

pub const Qwen38GDN = struct {
    ctx: EagerContext,
    hidden_size: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    head_dim: usize,
    conv_kernel: usize = 4,
    norm_eps: f32 = 1e-6,
    l2_eps: f32 = 1e-6,

    // Weights (MLX Linear layout: [out_features, in_features]).
    in_proj_qkv: Array,
    conv1d: Array, // [qkv_dim, conv_kernel, 1]
    in_proj_a: Array, // [Hv, hidden]
    in_proj_b: Array, // [Hv, hidden]
    in_proj_z: Array, // [Hv*D, hidden]
    A_log: Array, // [Hv] f32
    dt_bias: Array, // [Hv] f32
    norm_w: Array, // [D]
    out_proj: Array,

    // Optional quantized forms (affine 4-bit gs=64 in the real checkpoint).
    qkv_scales: ?Array = null,
    qkv_biases: ?Array = null,
    qkv_quant: QuantConfig = .{},
    a_scales: ?Array = null,
    a_biases: ?Array = null,
    a_quant: QuantConfig = .{},
    b_scales: ?Array = null,
    b_biases: ?Array = null,
    b_quant: QuantConfig = .{},
    z_scales: ?Array = null,
    z_biases: ?Array = null,
    z_quant: QuantConfig = .{},
    o_scales: ?Array = null,
    o_biases: ?Array = null,
    o_quant: QuantConfig = .{},

    pub fn deinit(self: *Qwen38GDN) void {
        self.in_proj_qkv.deinit();
        self.conv1d.deinit();
        self.in_proj_a.deinit();
        self.in_proj_b.deinit();
        self.in_proj_z.deinit();
        self.A_log.deinit();
        self.dt_bias.deinit();
        self.norm_w.deinit();
        self.out_proj.deinit();
        if (self.qkv_scales) |a| a.deinit();
        if (self.qkv_biases) |a| a.deinit();
        if (self.a_scales) |a| a.deinit();
        if (self.a_biases) |a| a.deinit();
        if (self.b_scales) |a| a.deinit();
        if (self.b_biases) |a| a.deinit();
        if (self.z_scales) |a| a.deinit();
        if (self.z_biases) |a| a.deinit();
        if (self.o_scales) |a| a.deinit();
        if (self.o_biases) |a| a.deinit();
    }

    fn keyProjDim(self: *const Qwen38GDN) usize {
        return self.num_key_heads * self.head_dim;
    }

    fn valueProjDim(self: *const Qwen38GDN) usize {
        return self.num_value_heads * self.head_dim;
    }

    /// x @ w^T, quantizedMatmul when scales are present.
    fn linear(self: *const Qwen38GDN, x: Array, w: Array, scales: ?Array, biases: ?Array, qcfg: QuantConfig) !Array {
        if (scales) |s| {
            const qw = quantize_mod.QuantizedWeight{
                .data = w,
                .scales = s,
                .biases = biases orelse Array.fromHandle(c.c.mlx_array_new()),
                .config = qcfg,
                .original_shape = &[_]i32{},
            };
            return quantize_mod.quantizedMatmul(self.ctx, x, qw, true);
        }
        const w_t = try ops.transpose(self.ctx, w);
        defer w_t.deinit();
        return ops.matmul(self.ctx, x, w_t);
    }

    /// Weight-less l2norm over the last dim: x / max(||x||_2, eps).
    fn l2Norm(self: *const Qwen38GDN, x: Array) !Array {
        const sq = try math_mod.square(self.ctx, x);
        defer sq.deinit();
        const ss = try reduce_mod.sumAxis(self.ctx, sq, -1, true);
        defer ss.deinit();
        const norm = try math_mod.sqrt(self.ctx, ss);
        defer norm.deinit();
        const eps_arr = try ops.scalarF32(self.ctx, self.l2_eps);
        defer eps_arr.deinit();
        const clipped = try ops.maximum(self.ctx, norm, eps_arr);
        defer clipped.deinit();
        return ops.divide(self.ctx, x, clipped);
    }

    /// softplus(x) = log1p(exp(-|x|)) + max(x, 0), numerically stable.
    fn softplus(self: *const Qwen38GDN, x: Array) !Array {
        const ctx = self.ctx;
        const abs_x = try ops.abs(ctx, x);
        defer abs_x.deinit();
        const neg_abs = try ops.negative(ctx, abs_x);
        defer neg_abs.deinit();
        const ex = try ops.exp(ctx, neg_abs);
        defer ex.deinit();
        const one = try ops.scalarF32(ctx, 1.0);
        defer one.deinit();
        const p1 = try ops.add(ctx, ex, one);
        defer p1.deinit();
        const logp = try math_mod.log(ctx, p1);
        defer logp.deinit();
        const zero = try ops.scalarF32(ctx, 0.0);
        defer zero.deinit();
        const relu = try ops.maximum(ctx, x, zero);
        defer relu.deinit();
        return ops.add(ctx, logp, relu);
    }

    /// One decode token. x: [1, hidden]. Returns [1, hidden]. Advances cache.
    /// Caller MUST eval the returned output each step (decode path does) so
    /// lazy graphs do not accumulate across tokens.
    pub fn forward(self: *Qwen38GDN, x: Array, cache: *Qwen38GdnCache) !Array {
        const ctx = self.ctx;
        const HK: i32 = @intCast(self.num_key_heads);
        const HV: i32 = @intCast(self.num_value_heads);
        const D: i32 = @intCast(self.head_dim);
        const PK: i32 = @intCast(self.keyProjDim());
        const PV: i32 = @intCast(self.valueProjDim());
        const QKV: i32 = 2 * PK + PV;
        const K: i32 = @intCast(self.conv_kernel);

        // 1. qkv projection + depthwise causal conv (k=4) + silu.
        const qkv_lin = try self.linear(x, self.in_proj_qkv, self.qkv_scales, self.qkv_biases, self.qkv_quant);
        defer qkv_lin.deinit();

        const conv_input = try shape_mod.concatenateAxis(ctx, &[_]Array{ cache.conv_state, qkv_lin }, 0);
        defer conv_input.deinit(); // [K, QKV]

        var conv_acc = blk: {
            const x_0 = try ops.slice(ctx, conv_input, &[_]i32{ 0, 0 }, &[_]i32{ 1, QKV }, &[_]i32{ 1, 1 });
            defer x_0.deinit();
            const w_0_s = try ops.slice(ctx, self.conv1d, &[_]i32{ 0, 0, 0 }, &[_]i32{ QKV, 1, 1 }, &[_]i32{ 1, 1, 1 });
            defer w_0_s.deinit();
            const w_0 = try ops.reshape(ctx, w_0_s, &[_]i32{QKV});
            defer w_0.deinit();
            break :blk try ops.multiply(ctx, x_0, w_0);
        };
        {
            var j: i32 = 1;
            while (j < K) : (j += 1) {
                const x_j = try ops.slice(ctx, conv_input, &[_]i32{ j, 0 }, &[_]i32{ j + 1, QKV }, &[_]i32{ 1, 1 });
                defer x_j.deinit();
                const w_j_s = try ops.slice(ctx, self.conv1d, &[_]i32{ 0, j, 0 }, &[_]i32{ QKV, j + 1, 1 }, &[_]i32{ 1, 1, 1 });
                defer w_j_s.deinit();
                const w_j = try ops.reshape(ctx, w_j_s, &[_]i32{QKV});
                defer w_j.deinit();
                const term = try ops.multiply(ctx, x_j, w_j);
                defer term.deinit();
                const acc_new = try ops.add(ctx, conv_acc, term);
                conv_acc.deinit();
                conv_acc = acc_new;
            }
        }
        defer conv_acc.deinit();

        // Advance conv window: keep last K-1 rows.
        const tail = try ops.slice(ctx, conv_input, &[_]i32{ 1, 0 }, &[_]i32{ K, QKV }, &[_]i32{ 1, 1 });
        defer tail.deinit();
        const new_conv_state = try ops.copy(ctx, tail);
        cache.conv_state.deinit();
        cache.conv_state = new_conv_state;
        try cache.conv_state.eval();

        const qkv = try activations.silu(ctx, conv_acc);
        defer qkv.deinit(); // [1, QKV]

        // 2. split q/k/v; l2norm q,k per head; scale q by 1/sqrt(D).
        const qkv_flat = try ops.reshape(ctx, qkv, &[_]i32{QKV});
        defer qkv_flat.deinit();
        const q_parts = try ops.slice(ctx, qkv_flat, &[_]i32{0}, &[_]i32{PK}, &[_]i32{1});
        defer q_parts.deinit();
        const k_parts = try ops.slice(ctx, qkv_flat, &[_]i32{PK}, &[_]i32{2 * PK}, &[_]i32{1});
        defer k_parts.deinit();
        const v_parts = try ops.slice(ctx, qkv_flat, &[_]i32{2 * PK}, &[_]i32{QKV}, &[_]i32{1});
        defer v_parts.deinit();

        const q_resh = try ops.reshape(ctx, q_parts, &[_]i32{ HK, D });
        defer q_resh.deinit();
        const k_resh = try ops.reshape(ctx, k_parts, &[_]i32{ HK, D });
        defer k_resh.deinit();
        const v = try ops.reshape(ctx, v_parts, &[_]i32{ HV, D });
        defer v.deinit();

        const q_norm = try self.l2Norm(q_resh);
        defer q_norm.deinit();
        const k_norm = try self.l2Norm(k_resh);
        defer k_norm.deinit();
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const scale_arr = try ops.scalarF32(ctx, scale);
        defer scale_arr.deinit();
        const q = try ops.multiply(ctx, q_norm, scale_arr);
        defer q.deinit();
        const k = k_norm; // rename for symmetry
        defer k.deinit();

        // 3. per-head gates: a (decay input), b (beta), z (output gate).
        const a = try self.linear(x, self.in_proj_a, self.a_scales, self.a_biases, self.a_quant);
        defer a.deinit(); // [1, HV]
        const b = try self.linear(x, self.in_proj_b, self.b_scales, self.b_biases, self.b_quant);
        defer b.deinit(); // [1, HV]
        const z = try self.linear(x, self.in_proj_z, self.z_scales, self.z_biases, self.z_quant);
        defer z.deinit(); // [1, HV*D]

        // g = exp(-exp(A_log) * softplus(a + dt_bias))  [HV]
        const a_flat = try ops.reshape(ctx, a, &[_]i32{HV});
        defer a_flat.deinit();
        const a_dt = try ops.add(ctx, a_flat, self.dt_bias);
        defer a_dt.deinit();
        const sp = try self.softplus(a_dt);
        defer sp.deinit();
        const e_a_log = try ops.exp(ctx, self.A_log);
        defer e_a_log.deinit();
        const g_pre = try ops.multiply(ctx, e_a_log, sp);
        defer g_pre.deinit();
        const g_neg = try ops.negative(ctx, g_pre);
        defer g_neg.deinit();
        const g = try ops.exp(ctx, g_neg); // [HV]
        defer g.deinit();

        const b_flat = try ops.reshape(ctx, b, &[_]i32{HV});
        defer b_flat.deinit();
        const beta = try ops.sigmoid(ctx, b_flat); // [HV]
        defer beta.deinit();

        // 4. broadcast q/k to value heads (kv-group ratio = HV/HK).
        const ratio: i32 = @intCast(self.num_value_heads / self.num_key_heads);
        const q4 = try ops.reshape(ctx, q, &[_]i32{ HK, 1, D });
        defer q4.deinit();
        const q_b = try shape_mod.broadcastTo(ctx, q4, &[_]i32{ HK, ratio, D });
        defer q_b.deinit();
        const q_full = try ops.reshape(ctx, q_b, &[_]i32{ HV, D });
        defer q_full.deinit();
        const k4 = try ops.reshape(ctx, k, &[_]i32{ HK, 1, D });
        defer k4.deinit();
        const k_b0 = try shape_mod.broadcastTo(ctx, k4, &[_]i32{ HK, ratio, D });
        defer k_b0.deinit();
        const k_full = try ops.reshape(ctx, k_b0, &[_]i32{ HV, D });
        defer k_full.deinit();

        // 5. gated delta rule step. State S [Hv, Dv, Dk].
        const g_decay = try ops.reshape(ctx, g, &[_]i32{ HV, 1, 1 });
        defer g_decay.deinit();
        const s_decayed = try ops.multiply(ctx, cache.ssm_state, g_decay);
        defer s_decayed.deinit();

        const k_b = try ops.reshape(ctx, k_full, &[_]i32{ HV, 1, D });
        defer k_b.deinit();
        const sk = try ops.multiply(ctx, s_decayed, k_b);
        defer sk.deinit();
        const kv_mem = try reduce_mod.sumAxis(ctx, sk, -1, false); // [HV, Dv]
        defer kv_mem.deinit();

        const v_sub = try ops.subtract(ctx, v, kv_mem);
        defer v_sub.deinit();
        const beta_b = try ops.reshape(ctx, beta, &[_]i32{ HV, 1 });
        defer beta_b.deinit();
        const delta = try ops.multiply(ctx, v_sub, beta_b); // [HV, Dv]
        defer delta.deinit();

        const delta_b = try ops.expandDims(ctx, delta, -1); // [HV, Dv, 1]
        defer delta_b.deinit();
        const k_delta = try ops.multiply(ctx, k_b, delta_b);
        defer k_delta.deinit();
        const s_new = try ops.add(ctx, s_decayed, k_delta);
        cache.ssm_state.deinit();
        cache.ssm_state = s_new;
        // Force materialization each step (recurrent lazy graph stays O(1) deep).
        try cache.ssm_state.eval();

        const q_b2 = try ops.reshape(ctx, q_full, &[_]i32{ HV, 1, D });
        defer q_b2.deinit();
        const sq = try ops.multiply(ctx, cache.ssm_state, q_b2);
        defer sq.deinit();
        const y = try reduce_mod.sumAxis(ctx, sq, -1, false); // [HV, Dv]
        defer y.deinit();

        // 6. gated RMSNorm: rmsnorm(y * silu(z)) * norm_w -> out_proj.
        const z_resh = try ops.reshape(ctx, z, &[_]i32{ HV, D });
        defer z_resh.deinit();
        const z_act = try activations.silu(ctx, z_resh);
        defer z_act.deinit();
        const y_gated = try ops.multiply(ctx, y, z_act);
        defer y_gated.deinit();
        const y_normed = try fast_mod.rmsNorm(ctx, y_gated, self.norm_w, self.norm_eps);
        defer y_normed.deinit();
        const y_flat = try ops.reshape(ctx, y_normed, &[_]i32{ 1, PV });
        defer y_flat.deinit();

        return self.linear(y_flat, self.out_proj, self.o_scales, self.o_biases, self.o_quant);
    }

    /// Prefill: naive per-token loop over x [1, L, hidden]. Returns [1, L, hidden].
    pub fn prefill(self: *Qwen38GDN, allocator: std.mem.Allocator, x: Array, cache: *Qwen38GdnCache) !Array {
        const xshape = x.shape();
        const L: i32 = @intCast(xshape[1]);
        const hidden: i32 = @intCast(self.hidden_size);

        var outs = try std.ArrayList(Array).initCapacity(allocator, @intCast(L));
        defer outs.deinit(allocator);
        defer for (outs.items) |o| o.deinit();

        var t: i32 = 0;
        while (t < L) : (t += 1) {
            const x_t = try ops.slice(self.ctx, x, &[_]i32{ 0, t, 0 }, &[_]i32{ 1, t + 1, hidden }, &[_]i32{ 1, 1, 1 });
            defer x_t.deinit();
            const x_flat = try ops.reshape(self.ctx, x_t, &[_]i32{ 1, hidden });
            defer x_flat.deinit();
            const out_t = try self.forward(x_flat, cache);
            errdefer out_t.deinit();
            const out_exp = try ops.expandDims(self.ctx, out_t, 1);
            outs.appendAssumeCapacity(out_exp);
            out_t.deinit();
        }
        return shape_mod.concatenateAxis(self.ctx, outs.items, 1);
    }
};
```

Verify before compiling: `ops.maximum`, `ops.negative`, `ops.divide`, `ops.copy`, `math_mod.log`, `math_mod.sqrt`, `shape_mod.broadcastTo` exist in the mlx_z bindings — grep `zig-pkg/mlx_z-*/src/ops/*.zig` for each; if a name differs (e.g. `max` instead of `maximum`), use the real name. This is the only permitted deviation from the listing above.

- [ ] **Step 4: Run tests**

Run: `zig test src/models/qwen38_gdn.zig`
Expected: 3/3 pass (numpy cross-check within 1e-5). If the cross-check fails, first re-verify the numpy script's delta-rule lines against the cleaned-up version in Task 2 Step 2, then check the gate ordering in step 6 (gate-then-norm).

- [ ] **Step 5: Commit**

```bash
git add src/models/qwen38_gdn.zig
git commit -m "feat(qwen38): GDN linear-attention module (asymmetric heads, softplus decay)"
```

---

## Task 4: Full-attention layer (qwen38_attn.zig)

**Files:**
- Create: `src/models/qwen38_attn.zig`

Full attention: q_proj out 12288 = 24 heads × `[q 256 | gate 256]` interleaved; per-head q/k RMSNorm; partial RoPE dims=64; GQA 24/4; internal KV cache; sigmoid output gate before o_proj.

- [ ] **Step 1: Write the failing test**

```zig
//! Qwen3.8 full-attention layer: per-head q/k RMSNorm, partial RoPE
//! (dims = head_dim * partial_rotary_factor), GQA, sigmoid output gate
//! (fused into q_proj, interleaved per head), model-internal KV cache.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const fast_mod = @import("mlx").fast;
const quantize_mod = @import("mlx").quantize;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const QuantConfig = quantize_mod.QuantConfig;

pub const Qwen38AttnCache = struct {
    keys: ?Array = null, // [1, n_kv, L, head_dim]
    values: ?Array = null, // [1, n_kv, L, head_dim]

    pub fn deinit(self: *Qwen38AttnCache) void {
        if (self.keys) |*k| k.deinit();
        if (self.values) |*v| v.deinit();
        self.keys = null;
        self.values = null;
    }

    pub fn len(self: *const Qwen38AttnCache) i32 {
        const k = self.keys orelse return 0;
        return k.shape()[2];
    }
};

test "Qwen38Attn.step: shapes, cache growth, determinism" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Toy dims: hidden=16, q heads=2, kv heads=1, head_dim=8, rotary dims=2.
    var attn = try Qwen38Attn.makeTestModule(allocator, ctx);
    defer attn.deinit();
    var cache = Qwen38AttnCache{};
    defer cache.deinit();

    var buf: [16]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    const x = try Array.fromData(allocator, f32, &buf, &[_]i32{ 1, 16 });
    defer x.deinit();

    const out1 = try attn.forward(x, &cache);
    defer out1.deinit();
    try out1.eval();
    const oshape = out1.shape();
    try std.testing.expectEqual(@as(i32, 1), oshape[0]);
    try std.testing.expectEqual(@as(i32, 16), oshape[1]);
    try std.testing.expectEqual(@as(i32, 1), cache.len());

    const out2 = try attn.forward(x, &cache);
    defer out2.deinit();
    try out2.eval();
    try std.testing.expectEqual(@as(i32, 2), cache.len());
}
```

- [ ] **Step 2: Verify compile failure**

Run: `zig test src/models/qwen38_attn.zig 2>&1 | head -5`
Expected: FAIL — `Qwen38Attn` undefined.

- [ ] **Step 3: Implement Qwen38Attn**

```zig
pub const Qwen38Attn = struct {
    ctx: EagerContext,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dims: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    output_gate: bool = true,

    // q_proj packs per-head [q head_dim | gate head_dim]: out = H * 2 * D.
    q_proj: Array,
    k_proj: Array,
    v_proj: Array,
    o_proj: Array,
    q_norm: Array, // [head_dim]
    k_norm: Array, // [head_dim]

    q_scales: ?Array = null,
    q_biases: ?Array = null,
    q_quant: QuantConfig = .{},
    k_scales: ?Array = null,
    k_biases: ?Array = null,
    k_quant: QuantConfig = .{},
    v_scales: ?Array = null,
    v_biases: ?Array = null,
    v_quant: QuantConfig = .{},
    o_scales: ?Array = null,
    o_biases: ?Array = null,
    o_quant: QuantConfig = .{},

    pub fn deinit(self: *Qwen38Attn) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.o_proj.deinit();
        self.q_norm.deinit();
        self.k_norm.deinit();
        if (self.q_scales) |a| a.deinit();
        if (self.q_biases) |a| a.deinit();
        if (self.k_scales) |a| a.deinit();
        if (self.k_biases) |a| a.deinit();
        if (self.v_scales) |a| a.deinit();
        if (self.v_biases) |a| a.deinit();
        if (self.o_scales) |a| a.deinit();
        if (self.o_biases) |a| a.deinit();
    }

    fn linear(self: *const Qwen38Attn, x: Array, w: Array, scales: ?Array, biases: ?Array, qcfg: QuantConfig) !Array {
        if (scales) |s| {
            const qw = quantize_mod.QuantizedWeight{
                .data = w,
                .scales = s,
                .biases = biases orelse Array.fromHandle(c.c.mlx_array_new()),
                .config = qcfg,
                .original_shape = &[_]i32{},
            };
            return quantize_mod.quantizedMatmul(self.ctx, x, qw, true);
        }
        const w_t = try ops.transpose(self.ctx, w);
        defer w_t.deinit();
        return ops.matmul(self.ctx, x, w_t);
    }

    /// x: [1, hidden] single token OR [1, L, hidden] prefill.
    /// Returns the same leading shape with hidden last dim. Advances cache.
    pub fn forward(self: *Qwen38Attn, x: Array, cache: *Qwen38AttnCache) !Array {
        const ctx = self.ctx;
        const H: i32 = @intCast(self.num_heads);
        const HKV: i32 = @intCast(self.num_kv_heads);
        const D: i32 = @intCast(self.head_dim);
        const xshape = x.shape();
        const prefill = xshape.len == 3;
        const L: i32 = if (prefill) xshape[1] else 1;
        const offset: i32 = cache.len();

        // q path: [.., H*2*D] -> [.., H, 2*D] -> chunk into q / gate.
        const q_lin = try self.linear(x, self.q_proj, self.q_scales, self.q_biases, self.q_quant);
        defer q_lin.deinit();
        const q_resh = if (prefill)
            try ops.reshape(ctx, q_lin, &[_]i32{ 1, L, H, 2 * D })
        else
            try ops.reshape(ctx, q_lin, &[_]i32{ 1, H, 2 * D });
        defer q_resh.deinit();
        const q_raw = if (prefill)
            try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, L, H, D }, &[_]i32{ 1, 1, 1, 1 })
        else
            try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, H, D }, &[_]i32{ 1, 1, 1 });
        defer q_raw.deinit();
        const gate = if (self.output_gate) blk: {
            const g = if (prefill)
                try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, 0, D }, &[_]i32{ 1, L, H, 2 * D }, &[_]i32{ 1, 1, 1, 1 })
            else
                try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, D }, &[_]i32{ 1, H, 2 * D }, &[_]i32{ 1, 1, 1 });
            break :blk g;
        } else null;
        defer if (gate) |g| g.deinit();

        const k_lin = try self.linear(x, self.k_proj, self.k_scales, self.k_biases, self.k_quant);
        defer k_lin.deinit();
        const v_lin = try self.linear(x, self.v_proj, self.v_scales, self.v_biases, self.v_quant);
        defer v_lin.deinit();
        const k_resh = if (prefill)
            try ops.reshape(ctx, k_lin, &[_]i32{ 1, L, HKV, D })
        else
            try ops.reshape(ctx, k_lin, &[_]i32{ 1, HKV, D });
        defer k_resh.deinit();
        const v_resh = if (prefill)
            try ops.reshape(ctx, v_lin, &[_]i32{ 1, L, HKV, D })
        else
            try ops.reshape(ctx, v_lin, &[_]i32{ 1, HKV, D });
        defer v_resh.deinit();

        // Per-head RMSNorm, then partial RoPE (dims = rotary_dims).
        const q_n = try fast_mod.rmsNorm(ctx, q_raw, self.q_norm, self.rms_norm_eps);
        defer q_n.deinit();
        const k_n = try fast_mod.rmsNorm(ctx, k_resh, self.k_norm, self.rms_norm_eps);
        defer k_n.deinit();
        const rd: i32 = @intCast(self.rotary_dims);
        const q_r = try fast_mod.rope(ctx, q_n, rd, false, self.rope_theta, 1.0, offset, null);
        defer q_r.deinit();
        const k_r = try fast_mod.rope(ctx, k_n, rd, false, self.rope_theta, 1.0, offset, null);
        defer k_r.deinit();

        // Transpose to [1, heads, L, D] and append to cache.
        const q_t = if (prefill)
            try ops.transposeAxes(ctx, q_r, &[_]i32{ 0, 2, 1, 3 })
        else
            try ops.expandDims(ctx, q_r, 2); // [1, H, 1, D]
        defer q_t.deinit();
        const k_t = if (prefill)
            try ops.transposeAxes(ctx, k_r, &[_]i32{ 0, 2, 1, 3 })
        else
            try ops.expandDims(ctx, k_r, 2);
        defer k_t.deinit();
        const v_t = if (prefill)
            try ops.transposeAxes(ctx, v_resh, &[_]i32{ 0, 2, 1, 3 })
        else
            try ops.expandDims(ctx, v_resh, 2);
        defer v_t.deinit();

        const k_all = if (cache.keys) |ck| blk: {
            defer ck.deinit();
            break :blk try shape_mod.concatenateAxis(ctx, &[_]Array{ ck, k_t }, 2);
        } else try ops.copy(ctx, k_t);
        const v_all = if (cache.values) |cv| blk: {
            defer cv.deinit();
            break :blk try shape_mod.concatenateAxis(ctx, &[_]Array{ cv, v_t }, 2);
        } else try ops.copy(ctx, v_t);
        cache.keys = k_all;
        cache.values = v_all;
        try cache.keys.?.eval();
        try cache.values.?.eval();

        // GQA: repeat kv heads to q head count.
        const rep: i32 = @intCast(self.num_heads / self.num_kv_heads);
        const k_rep = if (rep > 1) blk: {
            const e = try ops.expandDims(ctx, cache.keys.?, 2); // [1, HKV, 1, L, D]
            defer e.deinit();
            const b = try shape_mod.broadcastTo(ctx, e, &[_]i32{ 1, HKV, rep, cache.len(), D });
            defer b.deinit();
            break :blk try ops.reshape(ctx, b, &[_]i32{ 1, H, cache.len(), D });
        } else try ops.copy(ctx, cache.keys.?);
        defer k_rep.deinit();
        const v_rep = if (rep > 1) blk: {
            const e = try ops.expandDims(ctx, cache.values.?, 2);
            defer e.deinit();
            const b = try shape_mod.broadcastTo(ctx, e, &[_]i32{ 1, HKV, rep, cache.len(), D });
            defer b.deinit();
            break :blk try ops.reshape(ctx, b, &[_]i32{ 1, H, cache.len(), D });
        } else try ops.copy(ctx, cache.values.?);
        defer v_rep.deinit();

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const attn = try fast_mod.scaledDotProductAttention(
            ctx,
            q_t,
            k_rep,
            v_rep,
            scale,
            if (prefill) "causal" else "none",
            null,
            null,
        );
        defer attn.deinit(); // [1, H, L, D]

        // Back to [1, L, H*D] (or [1, H*D]).
        const attn_t = if (prefill)
            try ops.transposeAxes(ctx, attn, &[_]i32{ 0, 2, 1, 3 })
        else
            try ops.squeeze(ctx, attn);
        defer attn_t.deinit();
        const attn_flat = if (prefill)
            try ops.reshape(ctx, attn_t, &[_]i32{ 1, L, H * D })
        else
            try ops.reshape(ctx, attn_t, &[_]i32{ 1, H * D });
        defer attn_flat.deinit();

        // Sigmoid output gate.
        const gated = if (gate) |g| blk: {
            const g_flat = if (prefill)
                try ops.reshape(ctx, g, &[_]i32{ 1, L, H * D })
            else
                try ops.reshape(ctx, g, &[_]i32{ 1, H * D });
            defer g_flat.deinit();
            const sig = try ops.sigmoid(ctx, g_flat);
            defer sig.deinit();
            break :blk try ops.multiply(ctx, attn_flat, sig);
        } else try ops.copy(ctx, attn_flat);
        defer gated.deinit();

        return self.linear(gated, self.o_proj, self.o_scales, self.o_biases, self.o_quant);
    }

    /// Test-only constructor: toy dims hidden=16, H=2, HKV=1, D=8, rotary=2.
    pub fn makeTestModule(allocator: std.mem.Allocator, ctx: EagerContext) !Qwen38Attn {
        const fill = struct {
            fn call(buf: []f32) void {
                for (buf, 0..) |*v, i| {
                    v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 13) - 6)) * 0.02;
                }
            }
        }.call;
        const mk = struct {
            fn call(alloc: std.mem.Allocator, shape_: []const i32, fillFn: *const fn ([]f32) void) !Array {
                var n: usize = 1;
                for (shape_) |s| n *= @intCast(s);
                const buf = try alloc.alloc(f32, n);
                defer alloc.free(buf);
                fillFn(buf);
                return Array.fromData(alloc, f32, buf, shape_);
            }
        }.call;
        const norm_buf = try allocator.alloc(f32, 8);
        defer allocator.free(norm_buf);
        for (norm_buf, 0..) |*v, i| v.* = 1.0 + 0.01 * @as(f32, @floatFromInt(i));
        return .{
            .ctx = ctx,
            .hidden_size = 16,
            .num_heads = 2,
            .num_kv_heads = 1,
            .head_dim = 8,
            .rotary_dims = 2,
            .rope_theta = 10000.0,
            .rms_norm_eps = 1e-6,
            .q_proj = try mk(allocator, &[_]i32{ 2 * 2 * 8, 16 }, fill),
            .k_proj = try mk(allocator, &[_]i32{ 1 * 8, 16 }, fill),
            .v_proj = try mk(allocator, &[_]i32{ 1 * 8, 16 }, fill),
            .o_proj = try mk(allocator, &[_]i32{ 16, 2 * 8 }, fill),
            .q_norm = try Array.fromData(allocator, f32, norm_buf, &[_]i32{8}),
            .k_norm = try Array.fromData(allocator, f32, norm_buf, &[_]i32{8}),
        };
    }
};
```

Same binding-name caveat as Task 3 Step 3: grep mlx_z for `transposeAxes`, `squeeze`, `expandDims` and use the real names.

- [ ] **Step 4: Run tests**

Run: `zig test src/models/qwen38_attn.zig`
Expected: 1/1 pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/qwen38_attn.zig
git commit -m "feat(qwen38): full-attention layer (partial RoPE, q/k norm, sigmoid output gate)"
```

---

## Task 5: Model assembly + oracle alignment

**Files:**
- Create: `src/models/qwen38.zig`, `scripts/qwen38_oracle_dump.py`, `scripts/compare_qwen38_oracle.py`
- Modify: `build.zig` (register a `test_qwen38_oracle_dump` exe next to the GLM one — find `test_glm53_oracle_dump` in build.zig and clone the block)

Structure follows `glm53.zig`: a harness `Qwen38Model` (reads weights fresh via `WeightSource.fromIndex`) used by the oracle dump binary, plus `embedTokens` / `finalLogits` helpers on `WeightSource`.

- [ ] **Step 1: Weight loading helpers**

In `src/models/qwen38.zig`, import and re-export the GLM shared helpers — they are generic:

```zig
const glm53 = @import("glm53.zig");
pub const WeightSource = glm53.WeightSource;
pub const QuantParts = glm53.QuantParts;
```

Weight-name helper (stack buffers only, `glm53.zig:140` pattern):

```zig
fn fmtName(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}
```

- [ ] **Step 2: embedTokens / finalLogits**

Same pattern as `glm53.zig:1078-1116` (quantized embed → `dequantizeAs(.bfloat16)` → `takeAxis` rows → reshape `[1, L, hidden]`; final: `rmsNorm` with `language_model.model.norm.weight` → quantized lm_head matmul). Prefixes: `language_model.model.embed_tokens`, `language_model.model.norm.weight`, `language_model.lm_head`.

- [ ] **Step 3: Layer forward + model forward**

```zig
pub const Qwen38LayerCache = union(enum) {
    gdn: qwen38_gdn.Qwen38GdnCache,
    attn: qwen38_attn.Qwen38AttnCache,

    pub fn deinit(self: *Qwen38LayerCache) void {
        switch (self.*) {
            .gdn => |*gc| gc.deinit(),
            .attn => |*ac| ac.deinit(),
        }
    }
};
```

Per layer: `h = h + attn_branch(rmsnorm(h, input_layernorm))` then `h = h + swiglu(rmsnorm(h, post_attention_layernorm))`. SwiGLU: `down(silu(gate(x)) * up(x))`, all quantized via `ws.quantized(prefix, in_features)` + `QuantParts.linear().apply(ctx, x)` (see `glm53.zig:55-58` and its use in `glm53_attention.zig`). Branch prefix: `language_model.model.layers.{i}.linear_attn` / `.self_attn` / `.mlp`.

GDN/attn modules are constructed per call in the harness (weights read fresh via `ws`), exactly like `glm53.zig`'s branch helpers (`kdaAttnBranch` at `:259`, `mlaAttnBranch` at `:406`). GDN loader fills both plain and quantized fields: load `in_proj_qkv.weight` + `.scales` + `.biases` into `qkv_*` (in_features = hidden), same for `in_proj_a/b/z`, `out_proj`; plain loads for `conv1d.weight`, `A_log`, `dt_bias`, `norm.weight`.

Model forward contract (mirrors `Glm53Model.forward`, `glm53.zig:889`): `pub fn forward(self: *Qwen38Model, ids: []const i32, opts: ForwardOpts) !Array` with `ForwardOpts{ max_layer: ?u32 = null, hook: ?SegmentHook = null }` for segment-wise oracle dumping — copy the `SegmentHook` idea from `glm53.zig:820`.

- [ ] **Step 4: Oracle dump script**

`scripts/qwen38_oracle_dump.py` — follow `scripts/glm53_oracle_dump.py` structure: load `/Users/zouyee/models/Qwen3.8-27B-4bit` with mlx-lm (verify support first: `python3 -c "import mlx_lm; print(mlx_lm.__version__)"`; if mlx-lm lacks qwen3_5, fall back to `transformers` — `transformers_version` in config says `5.8.0.dev0`, so a dev install may be needed: `pip install -U "git+https://github.com/huggingface/transformers"`). Dump for prompt `"The capital of France is"` (fixed token ids printed in the dump): embeddings, per-segment hidden states (after layer 0 GDN, after layer 3 full attn), final logits. Save to `artifacts/qwen38_oracle/*.npy` plus a `meta.json` with token ids and segment boundaries.

- [ ] **Step 5: Zig dump harness + compare**

Clone `src/test_glm53_oracle_dump.zig` → `src/test_qwen38_oracle_dump.zig` (it reads config, runs `Qwen38Model.forward` with hooks, writes `.npy` segments). Register in build.zig. `scripts/compare_qwen38_oracle.py` — cosine similarity per segment, threshold ≥ 0.9999 (copy structure from `scripts/compare_glm53_oracle.py`).

Run:

```bash
python3 scripts/qwen38_oracle_dump.py --out artifacts/qwen38_oracle
zig build run-qwen38-oracle -- --model /Users/zouyee/models/Qwen3.8-27B-4bit --out artifacts/qwen38_oracle_zig
python3 scripts/compare_qwen38_oracle.py artifacts/qwen38_oracle artifacts/qwen38_oracle_zig
```

Expected: every segment cos ≥ 0.9999. If the layer-0 GDN segment fails, suspect (in order): decay formula, q/k l2norm eps, gate-then-norm order; if the layer-3 attn segment fails, suspect: q/gate interleave convention (per-head `[q|gate]` chunk vs global halves), RoPE `traditional` flag, q/k norm placement relative to RoPE.

- [ ] **Step 6: Full forward + logits cos check + commit**

Extend the compare to full 64-layer logits. Then:

```bash
git add src/models/qwen38.zig src/test_qwen38_oracle_dump.zig scripts/qwen38_oracle_dump.py scripts/compare_qwen38_oracle.py build.zig
git commit -m "feat(qwen38): model assembly + oracle segment alignment (cos >= 0.9999)"
```

---

## Task 6: Serving model + registry + CLI/server wiring

**Files:**
- Modify: `src/models/qwen38.zig` (add `Qwen38ServingModel`), `src/model_registry.zig`, `src/server/state.zig`, `src/server/openai.zig`, `src/server/anthropic.zig`, `src/main.zig`

- [ ] **Step 1: Qwen38ServingModel**

Mirror `Glm53ServingModel` (`glm53.zig:1140-1249`): resident `weights: std.StringHashMap(Array)` (all `language_model.*` tensors loaded once via `TensorIndex.loadTensor`), `WeightSource.fromMap` COW borrowing, per-layer `Qwen38LayerCache`, `reset()`, `forwardTokens(ids: []const u32) !Array` with the contract "L>1 resets state; L=1 is a decode step". No expert streaming (dense model) — no `PartialTensorReader`/`ExpertStreamProvider` fields.

- [ ] **Step 2: Registry entry**

In `src/model_registry.zig`:
- add `const qwen38_loader = @import("models/qwen38_loader.zig");` and `const qwen38 = @import("models/qwen38.zig");`
- add `.{ "Qwen3_5ForConditionalGeneration", qwen38Loader },` to `model_registry` and `"Qwen3_5ForConditionalGeneration",` to `supported_architectures`; update the warn string in `getLoader` and the registry tests (the "all architectures" test list)
- `qwen38Loader` mirrors `glm53Loader` (`model_registry.zig:446-560`) minus expert streaming: parse config → `loadOrBuildIndex` → load all `language_model.*` tensors into the resident map → build `Qwen38ServingModel` → adapter with `forward` = ids → `forwardTokens` → logits (copy `Glm53VTableAdapter`, drop reader/fd_pool/stream_provider/layer_meta fields)
- `ModelConfig` for the vtable: `num_layers = cfg.num_hidden_layers`, `num_kv_heads = 1`, `head_dim = cfg.head_dim`, `vocab_size`, `hidden_size`, `compress_ratios = &.{}`
- add `pub fn Qwen38ConfigFromJson(allocator, config_json) !qwen38_loader.Qwen38Config` next to `Glm53ConfigFromJson` (`:98`)

- [ ] **Step 3: Server wiring**

In `src/server/state.zig`:
1. Template (`:153-158`): add a branch — `else if (std.mem.indexOf(u8, config_content, "\"qwen3_5\"") != null) root.tokenizer.ChatTemplate.initChatML(allocator)` (verify the ChatML init fn name in `src/tokenizer/chat_template.zig` — it's the constructor matching `applyChatML`)
2. Cache skip (`:295`): `const is_glm53 = ...` becomes `const is_internal_cache = is_glm53 or std.mem.eql(u8, arch_name, "Qwen3_5ForConditionalGeneration")`; use it in the `:304` branch
3. Stop tokens: `engine.RequestConfig` gets `stop_tokens` from the model's eos list. Parse `Qwen38Config` in `loadModel` for this arch, store `eos_token_ids` on `ServerState` (new field `stop_tokens: []const u32 = &.{}`), and in `src/server/openai.zig:129` and the anthropic equivalent set `.stop_tokens = state.stop_tokens` in `RequestConfig` (verify `RequestConfig` has this field — `src/engine/request_state.zig:44`)

- [ ] **Step 4: CLI wiring**

In `src/main.zig`, clone `runGlm53Chat` (`:815`) as `runQwen38Chat`: parse `Qwen38ConfigFromJson`, ChatML template (`initChatML`), generation config `.stop_tokens = cfg.eos_token_ids` (`:866` pattern), defaults temperature 1.0 / top_k 20 / top_p 0.95. Dispatch: find where `runGlm53Chat` is selected (`:576`) and add the `Qwen3_5ForConditionalGeneration` branch.

- [ ] **Step 5: Build + full test suite + registry tests**

Run: `zig build test`
Expected: all pass, including updated registry tests.

- [ ] **Step 6: Commit**

```bash
git add src/models/qwen38.zig src/model_registry.zig src/server/state.zig src/server/openai.zig src/server/anthropic.zig src/main.zig
git commit -m "feat(qwen38): serving model + registry + CLI/server wiring (ChatML, multi-EOS)"
```

---

## Task 7: End-to-end acceptance

- [ ] **Step 1: CLI smoke**

```bash
zig build run -- chat --model /Users/zouyee/models/Qwen3.8-27B-4bit --prompt "The capital of France is" --max-tokens 32
```
Expected: coherent completion containing "Paris". (Verify the exact CLI flag names with `zig build run -- --help`.)

- [ ] **Step 2: Vision token rejection**

Feed a prompt whose tokenization includes 248056 (or unit-test the path): expect explicit `vision input not supported` error, not garbage output.

- [ ] **Step 3: 7-prompt e2e + server stability**

Serve: `zig build run -- server --model /Users/zouyee/models/Qwen3.8-27B-4bit --port 8080`. Run 7 prompts (mix of short QA, code, long instruction) through both streaming and non-streaming `/v1/chat/completions`; verify coherent text, clean EOS termination (finish_reason `stop`, no runaway), no segfault across sequential requests (state reset works).

- [ ] **Step 4: Regression**

Run: `zig build test` (full suite — llama/glm53/dsv4 paths stay green) and a GLM-5.3 serve smoke if the checkpoint is present at `~/models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx`.

- [ ] **Step 5: Commit + update HANDOFF.md**

```bash
git commit -m "test(qwen38): e2e acceptance" --allow-empty
```
