#!/usr/bin/env python3
"""Verify mxfp4 LUT by recomputing gate scores."""
import struct, json, glob
import numpy as np

MODEL = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit"
N_EXPERTS = 256
DIM = 4096

def load_raw(name):
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
    raise ValueError(name)


def to_arr(dt, sh, raw):
    if dt == "U32":
        return np.frombuffer(raw, "<u4").reshape(sh)
    elif dt == "U8":
        return np.frombuffer(raw, "<u1").reshape(sh)
    elif dt == "BF16":
        u = np.frombuffer(raw, "<u2").astype(np.uint32)
        return (u << 16).view(np.float32).reshape(sh)
    else:
        raise ValueError(dt)


def dequant_row(packed_row, scales, lut):
    # packed_row: [512] uint32 -> 4096 nibbles
    # scales: [128] uint8 -> 128 groups of 32
    nibbles = np.empty(4096, np.uint8)
    tmp = packed_row.copy()
    for i in range(8):
        nibbles[i::8] = (tmp & 0xF).astype(np.uint8)
        tmp >>= 4
    out = np.empty(4096, np.float32)
    for g in range(128):
        sf = 2.0 ** (float(scales[g]) - 127.0)
        out[g*32:(g+1)*32] = lut[nibbles[g*32:(g+1)*32]] * sf
    return out


def main():
    normed = np.load("/tmp/mlx_dump/L0_ffn_normed.npy").astype(np.float32).reshape(-1)

    w_packed = to_arr(*load_raw("model.layers.0.ffn.switch_mlp.gate_proj.weight"))
    s_u8 = to_arr(*load_raw("model.layers.0.ffn.switch_mlp.gate_proj.scales"))

    # LUTs
    current_lut = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
                            -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0], np.float32)
    e2m1_lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)

    # Compute scores for top experts
    top_ids = [243, 54, 238, 198, 214, 150]
    for eid in top_ids:
        row = w_packed[eid, 0]  # first row of expert eid
        sc = s_u8[eid, 0]
        cur_w = dequant_row(row, sc, current_lut)
        e2m1_w = dequant_row(row, sc, e2m1_lut)
        cur_score = cur_w @ normed
        e2m1_score = e2m1_w @ normed
        print(f"expert {eid:3d}: current_score={cur_score:8.4f}  e2m1_score={e2m1_score:8.4f}")


if __name__ == "__main__":
    main()
