#!/usr/bin/env python3
"""
Verify native layer-0 attention output by comparing with MLX golden.
Uses the /tmp/mlx_ref_new/ golden (France prompt) to:
1. Check that MLX golden L0_attn_normed → L0_attn_out is consistent
2. Check the per-token attention computation numerically
3. Identify which step diverges

Usage: python3 scripts/verify_layer0_attn.py
"""
import numpy as np
import os, sys, json, struct

REF = "/tmp/mlx_ref_new"
MODEL = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit"

# Model constants
DIM = 4096
N_HEADS = 64
HEAD_DIM = 512
Q_LORA_RANK = 1024
KV_LORA_RANK = 512
QK_ROPE_DIM = 64
QK_NOPE_DIM = 448
O_GROUPS = 8
O_LORA_RANK = 1024
ATTN_GS = 64
EPS = 1e-6
PI = 3.14159265358979323846

def bf16_arr(x):
    """Convert float32 array to bf16 (truncate lower 16 bits) and back."""
    u = x.view(np.uint32) & np.uint32(0xFFFF0000)
    return u.view(np.float32)

def rms_norm(x, w, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return (x / rms) * w

def rms_norm_bf16(x, w):
    """MLX-style: norm in f32, output in bf16."""
    out = rms_norm(x.astype(np.float64), w).astype(np.float32)
    return bf16_arr(out)

def yarn_cos_sin(pos, half_rope=QK_ROPE_DIM//2):
    base = 10000.0; factor = 16.0; beta_fast = 32.0; beta_slow = 1.0
    orig_max = 65536.0
    cd_fast = QK_ROPE_DIM * np.log(orig_max / (beta_fast * 2 * PI)) / (2 * np.log(base))
    cd_slow = QK_ROPE_DIM * np.log(orig_max / (beta_slow * 2 * PI)) / (2 * np.log(base))
    low = int(max(0, np.floor(cd_fast)))
    high = int(min(QK_ROPE_DIM - 1, np.ceil(cd_slow)))
    freqs = np.zeros(half_rope)
    for i in range(half_rope):
        freq = 1.0 / (base ** (2.0 * i / QK_ROPE_DIM))
        if low == high:
            ramp = 0.0 if i <= low else 1.0
        else:
            ramp = min(1.0, max(0.0, (i - low) / (high - low)))
        smooth = 1.0 - ramp
        freqs[i] = freq / factor * (1 - smooth) + freq * smooth
    return np.cos(pos * freqs).astype(np.float32), np.sin(pos * freqs).astype(np.float32)

def rope_tail(q, cos, sin, n_heads, head_dim=HEAD_DIM, nope=QK_NOPE_DIM, rope=QK_ROPE_DIM):
    """Apply tail RoPE to q [n_heads, head_dim]."""
    half = rope // 2
    out = q.copy()
    for h in range(n_heads):
        for i in range(half):
            j0 = nope + i; j1 = nope + half + i
            x0 = float(q[h, j0]); x1 = float(q[h, j1])
            c = float(cos[i]); s = float(sin[i])
            out[h, j0] = np.float32(x0 * c - x1 * s)
            out[h, j1] = np.float32(x0 * s + x1 * c)
    return out

def dequant_affine(packed, scales, biases, out_dim, in_dim, gs=ATTN_GS):
    """Dequantize affine 4-bit weights."""
    n_groups = in_dim // gs
    ppg = gs // 8  # packed uint32s per group
    result = np.zeros(out_dim, dtype=np.float32)
    packed_view = packed.reshape(out_dim, in_dim // 8)
    for r in range(out_dim):
        for g in range(n_groups):
            sc = scales[r, g]; bi = biases[r, g]
            for p in range(ppg):
                pw = packed_view[r, g * ppg + p]
                for k in range(8):
                    nib = (pw >> (k * 4)) & 0xF
                    result[r] += (sc * nib + bi) * 0  # placeholder, need x
    return result  # this would need x; skip for now

def matvec_affine(W_packed, scales, biases, x, out_dim, in_dim, gs=ATTN_GS):
    """Compute y = W @ x where W is affine-quantized 4-bit."""
    n_groups = in_dim // gs
    ppg = gs // 8
    packed_cols = in_dim // 8
    result = np.zeros(out_dim, dtype=np.float64)
    for r in range(out_dim):
        wp = W_packed[r * packed_cols : (r+1) * packed_cols]
        for g in range(n_groups):
            sc = float(scales[r * n_groups + g])
            bi = float(biases[r * n_groups + g])
            base_x = g * gs
            for p in range(ppg):
                pw = int(wp[g * ppg + p])
                for k in range(8):
                    nib = (pw >> (k * 4)) & 0xF
                    val = sc * nib + bi
                    result[r] += val * float(x[base_x + p * 8 + k])
    return result.astype(np.float32)

print("Loading MLX golden from", REF)
attn_normed = np.load(f"{REF}/L0_attn_normed.npy")  # [1,9,4096] f32
attn_out_mlx = np.load(f"{REF}/L0_attn_out.npy")    # [1,9,4096] f32

print("attn_normed shape:", attn_normed.shape)
print("attn_out_mlx shape:", attn_out_mlx.shape)

# Token 0: normed input and expected output
tok0_normed = attn_normed[0, 0, :]  # [4096]
tok0_attn_out = attn_out_mlx[0, 0, :]  # [4096] — expected output from MLX

print(f"\nToken 0 normed norm: {np.linalg.norm(tok0_normed):.4f}")
print(f"Token 0 expected attn_out norm: {np.linalg.norm(tok0_attn_out):.4f}")
print(f"Token 0 attn_out[:5]: {tok0_attn_out[:5]}")

# Load layer 0 attention weights from safetensors
print("\n--- Loading model weights ---")
try:
    import safetensors.numpy as st
    # Find which shard has layer 0 attention weights
    index_path = f"{MODEL}/model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Key weights for layer 0 attention
    keys_needed = [
        "model.layers.0.attn.wq_a.weight",
        "model.layers.0.attn.wq_a.weight_scale_inv",  # or scales
        "model.layers.0.attn.q_norm.weight",
        "model.layers.0.attn.wq_b.weight",
        "model.layers.0.attn.wkv_a_proj_with_mqa.weight",
        "model.layers.0.attn.kv_norm.weight",
        "model.layers.0.attn.attn_sink",
    ]
    
    shards_needed = set()
    for k in keys_needed:
        if k in weight_map:
            shards_needed.add(weight_map[k])
    
    print(f"Loading shards: {shards_needed}")
    
    loaded = {}
    for shard in shards_needed:
        shard_path = f"{MODEL}/{shard}"
        if os.path.exists(shard_path):
            d = st.load_file(shard_path)
            loaded.update(d)
    
    print(f"Loaded {len(loaded)} tensors")
    # Print available layer 0 attention keys
    l0_keys = [k for k in loaded if 'layers.0.attn' in k]
    for k in sorted(l0_keys)[:20]:
        print(f"  {k}: {loaded[k].shape} {loaded[k].dtype}")

except ImportError:
    print("safetensors not installed, skipping weight load")
    print("Install: pip install safetensors")
    sys.exit(0)
except Exception as e:
    print(f"Error loading weights: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
