#!/usr/bin/env python3
"""Export a trained bf16 MTP head (artifacts/mtpd/ckpt/*.safetensors) into the
dmlx drafter checkpoint layout: 4-bit gs=64 affine quantization, keys exactly
as in ~/models/Qwen3.8-27B-MTP-4bit (fc.*, layers.0.*, norm.weight,
pre_fc_norm_{embedding,hidden}.weight), plus config/tokenizer copied from the
reference drafter dir.

Usage: ~/venvs/dmlx-omlx/bin/python scripts/mtpd_export.py \
    --ckpt artifacts/mtpd/ckpt/best.safetensors \
    --out ~/models/Qwen3.8-27B-MTP-4bit-distilled
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

REF = Path.home() / "models/Qwen3.8-27B-MTP-4bit"
COPY_FILES = [
    "config.json", "chat_template.jinja", "generation_config.json",
    "tokenizer_config.json", "tokenizer.json", "vocab.json", "README.md",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    params = dict(mx.load(args.ckpt))
    out: dict[str, mx.array] = {}
    for key, w in sorted(params.items()):
        if w.ndim == 2:  # linear weight -> quantize
            qw, scales, biases = mx.quantize(w, group_size=args.group_size, bits=args.bits)
            out[f"{key}.weight"] = qw
            out[f"{key}.scales"] = scales
            out[f"{key}.biases"] = biases
        else:  # norms stay bf16, as in the reference checkpoint
            out[key] = w
    mx.eval(out)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "model.safetensors"), out)
    index = {"metadata": {"total_size": sum(v.nbytes for v in out.values())},
             "weight_map": {k: "model.safetensors" for k in out}}
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    for name in COPY_FILES:
        src = REF / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    print(f"wrote {len(out)} tensors -> {out_dir}", flush=True)
    # Report key agreement with the reference layout.
    ref_map = json.load(open(REF / "model.safetensors.index.json"))["weight_map"]
    missing = sorted(set(ref_map) - set(out))
    extra = sorted(set(out) - set(ref_map))
    print(f"missing vs reference: {missing}")
    print(f"extra vs reference:   {extra}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
