#!/usr/bin/env python3
"""Verify Metal mxfp4 LUT against MLX dequantize."""
import sys, os
sys.path.insert(0, "/opt/homebrew/lib/python3.12/site-packages")
import mlx.core as mx
import numpy as np

MODEL = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit"

# Load a single expert weight from layer 0
w = mx.load(os.path.join(MODEL, "model-00001-of-00033.safetensors"))
# Better: use the specific key
import json, glob, struct

def load_tensor(name):
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
                if v["dtype"] == "U32":
                    return np.frombuffer(raw, "<u4").reshape(v["shape"])
                elif v["dtype"] == "U8":
                    return np.frombuffer(raw, "<u1").reshape(v["shape"])
    raise ValueError(name)

# Load layer 0 gate_proj for expert 0
w_packed = load_tensor("model.layers.0.ffn.switch_mlp.gate_proj.weight")  # [256, 2048, 512] U32
s_u8 = load_tensor("model.layers.0.ffn.switch_mlp.gate_proj.scales")     # [256, 2048, 128] U8

print("weight shape:", w_packed.shape, "dtype:", w_packed.dtype)
print("scales shape:", s_u8.shape, "dtype:", s_u8.dtype)

# Take expert 0, row 0
exp0_w = w_packed[0, 0]  # [512] uint32
exp0_s = s_u8[0, 0]      # [128] uint8

# Unpack with Metal LUT
METAL_LUT = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
                      -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0], np.float32)

# Unpack nibbles
nibbles = np.empty(4096, np.uint8)
for i in range(8):
    nibbles[i::8] = (exp0_w & 0xF).astype(np.uint8)
    exp0_w = exp0_w >> 4

metal_deq = np.empty(4096, np.float32)
for g in range(128):
    sf = 2.0 ** (float(exp0_s[g]) - 127.0)
    metal_deq[g*32:(g+1)*32] = METAL_LUT[nibbles[g*32:(g+1)*32]] * sf

print("Metal dequant first 32:", metal_deq[:32])
print("Metal dequant stats: min=%.4f max=%.4f mean=%.4f" % (metal_deq.min(), metal_deq.max(), metal_deq.mean()))

# Now use MLX to dequantize the same tensor
import mlx.core as mx
# Build mlx array from the raw data
w_full = mx.array(w_packed)  # [256, 2048, 512] uint32
s_full = mx.array(s_u8)      # [256, 2048, 128] uint8

# We need to call mx.dequantize with the right params.
# mx.dequantize(w, scales, biases, group_size, bits)
# For mxfp4, biases should be zeros or None, bits=4
try:
    # Expert 0, all rows
    w_e0 = w_full[0]  # [2048, 512] uint32
    s_e0 = s_full[0]  # [2048, 128] uint8
    deq_mlx = mx.dequantize(w_e0, s_e0, group_size=32, bits=4)
    deq_np = np.array(deq_mlx)
    print("MLX dequant row 0 first 32:", deq_np[0, :32])
    print("MLX dequant stats: min=%.4f max=%.4f mean=%.4f" % (deq_np.min(), deq_np.max(), deq_np.mean()))
except Exception as e:
    print("MLX dequantize failed:", e)
    import traceback
    traceback.print_exc()
