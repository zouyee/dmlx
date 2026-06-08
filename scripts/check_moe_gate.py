#!/usr/bin/env python3
"""Check Metal MoE gate routing against MLX for L0_ffn_normed."""
import struct, json, glob, sys
import numpy as np

MODEL = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit"
N_EXPERTS = 256
N_ACTIVE = 6
DIM = 4096


def find_tensor(name):
    for f in sorted(glob.glob(MODEL + "/model-*-of-00033.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            base = 8 + n
            if name in hdr:
                v = hdr[name]
                o0, o1 = v["data_offsets"]
                fh.seek(base + o0)
                raw = fh.read(o1 - o0)
                return v["dtype"], v["shape"], raw
    return None


def to_f32(dt, sh, raw):
    if dt == "BF16":
        u = np.frombuffer(raw, "<u2").astype(np.uint32)
        a = (u << 16).view(np.float32)
    elif dt == "F32":
        a = np.frombuffer(raw, "<f4")
    else:
        raise ValueError(dt)
    return a.reshape(sh)


def main():
    normed = np.load("/tmp/mlx_dump/L0_ffn_normed.npy").astype(np.float32).reshape(-1)
    print(f"normed shape={normed.shape} norm={np.linalg.norm(normed):.4f}")

    # Load layer 0 gate
    gate_w = to_f32(*find_tensor("model.layers.0.ffn.gate.weight"))
    print(f"gate_w shape={gate_w.shape}")

    # MLX-style matmul (x @ W.T) -> scores
    scores = normed @ gate_w.T
    print(f"scores min={scores.min():.4f} max={scores.max():.4f}")

    # Load gate bias if exists
    bias_info = find_tensor("model.layers.0.moe_gate.bias")
    if bias_info:
        bias = to_f32(*bias_info).reshape(-1)
        scores = scores + bias
        print(f"added bias")

    # Top-k with softmax
    topk_idx = np.argsort(scores)[-N_ACTIVE:][::-1]
    topk_scores = scores[topk_idx]
    topk_weights = np.exp(topk_scores - topk_scores.max())
    topk_weights /= topk_weights.sum()

    print(f"MLX  expert_ids={list(topk_idx)} weights={[f'{w:.3f}' for w in topk_weights]}")
    print(f"Metal expert_ids=[243,54,238,198,214,150] weights=[0.281,0.264,0.261]")


if __name__ == "__main__":
    main()
