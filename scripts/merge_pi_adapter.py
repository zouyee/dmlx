#!/usr/bin/env python3
"""Offline pi-adapted LoRA merge into an MLX 4-bit model dir.

Replicates dmlx's load-time merge (src/models/lora_merge.zig) so the merged
directory can be served by engines without LoRA support (omlx, TensorFold).
Per-module streaming: dequant -> + scale*(B@A) -> requantize affine.
"""
import json, os, shutil, sys
import mlx.core as mx
from safetensors import safe_open
import mlx.core as mx
import numpy as np

BASE = os.path.expanduser(sys.argv[1])
ADAPTER = os.path.expanduser(sys.argv[2])
OUT = os.path.expanduser(sys.argv[3])

cfg = json.load(open(os.path.join(ADAPTER, "adapter_config.json")))
SCALE = cfg["lora_alpha"] / cfg["r"]

# ---- adapter tensors ----
adapter = {}
with safe_open(os.path.join(ADAPTER, "adapter_model.safetensors"), "np") as f:
    for k in f.keys():
        adapter[k] = f.get_tensor(k)

def resident_stem(peft_stem: str) -> str:
    s = peft_stem
    if s.startswith("base_model.model."):
        s = s[len("base_model.model."):]
    if s.startswith("model.language_model."):
        s = "language_model.model." + s[len("model.language_model."):]
    return s

targets = {}  # resident stem -> (A, B)
for k in adapter:
    if not k.endswith(".lora_A.weight"):
        continue
    stem = k[: -len(".lora_A.weight")]
    b_key = stem + ".lora_B.weight"
    if b_key not in adapter:
        print("warn: no B for", stem)
        continue
    targets[resident_stem(stem)] = (adapter[k], adapter[b_key])
print(f"{len(targets)} adapter modules")

# ---- model layout ----
index = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
weight_map = index["weight_map"]
shard_names = sorted(set(weight_map.values()))

os.makedirs(OUT, exist_ok=True)
for f in os.listdir(BASE):
    if f.endswith(".safetensors"):
        continue
    src = os.path.join(BASE, f)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(OUT, f))

merged = 0
for shard in shard_names:
    path = os.path.join(BASE, shard)
    tensors = mx.load(path)  # safetensors loader (preserves dtypes incl. bf16)
    for stem, (a_np, b_np) in targets.items():
        w_name, s_name, bi_name = stem + ".weight", stem + ".scales", stem + ".biases"
        if w_name not in tensors:
            continue
        w_q = tensors.pop(w_name)
        scales = tensors.pop(s_name)
        biases = tensors.pop(bi_name)
        out_f, in_f = b_np.shape[0], a_np.shape[1]
        bits = w_q.shape[-1] * 32 // in_f
        gs = in_f // scales.shape[-1]
        w = mx.dequantize(w_q, scales, biases, group_size=gs, bits=bits)
        a = mx.array(a_np.astype(np.float32))
        b = mx.array(b_np.astype(np.float32))
        delta = (b @ a) * SCALE
        w_m = w + delta
        q_w, q_s, q_b = mx.quantize(w_m, group_size=gs, bits=bits, mode="affine")
        tensors[w_name] = q_w
        tensors[s_name] = q_s.astype(mx.bfloat16)
        tensors[bi_name] = q_b.astype(mx.bfloat16)
        merged += 1
        if merged % 64 == 0:
            print(f"merged {merged}/{len(targets)} ({stem})", flush=True)
    # add untouched keys still in this shard (already loaded above)
    mx.save_safetensors(os.path.join(OUT, shard), tensors)
    print(f"shard {shard} written ({len(tensors)} tensors)", flush=True)

print(f"DONE merged={merged} -> {OUT}")
