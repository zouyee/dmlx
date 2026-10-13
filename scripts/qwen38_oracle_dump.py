#!/usr/bin/env python3
"""Dump Qwen3.8-27B-4bit intermediate activations from mlx-lm's qwen3_5
implementation as a numerical oracle (golden reference) for the dmlx Zig
implementation (src/models/qwen38*.zig).

Loads /Users/zouyee/models/Qwen3.8-27B-4bit with mlx-lm (the local checkout at
/Users/zouyee/work/code/mlx-lm, which has models/qwen3_5.py), tokenizes the
fixed prompt "The capital of France is", and dumps float32 .npy segments plus
meta.json:

    embed              [1, L, hidden]   bf16 embedding rows
    layer0_gdn_out     [1, L, hidden]   residual after layer 0 GDN block
    layer0_mlp_out     [1, L, hidden]   residual after layer 0 MLP block
    layer3_attn_out    [1, L, hidden]   residual after layer 3 full-attn block
    layer3_mlp_out     [1, L, hidden]   residual after layer 3 MLP block
    hidden_final       [1, L, hidden]   post final RMSNorm
    logits             [1, L, vocab]

Usage:
    PYTHONPATH=/Users/zouyee/work/code/mlx-lm python3 scripts/qwen38_oracle_dump.py \
        --out artifacts/qwen38_oracle [--model /Users/zouyee/models/Qwen3.8-27B-4bit]

Requires a python env with a working `mlx` build (the system brew python's mlx
is broken: `import mlx.core` raises ModuleNotFoundError: mlx._reprlib_fix).

Debug env vars:
    QWEN38_DUMP_ALL=1    dump every layer's attn/mlp output (per-layer sweep)
    QWEN38_NAIVE_GDN=1   force the ops-based GDN recurrence (no fused kernel)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

PROMPT = "The capital of France is"
DUMP_ALL = os.environ.get("QWEN38_DUMP_ALL", "0") == "1"


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("/Users/zouyee/models/Qwen3.8-27B-4bit"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/qwen38_oracle"))
    args = parser.parse_args()

    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from importlib.metadata import version

    if os.environ.get("QWEN38_NAIVE_GDN", "0") == "1":
        # Force the ops-based GDN recurrence (no fused Metal kernel) so the
        # oracle matches the Zig ops-loop implementation step for step.
        import mlx_lm.models.gated_delta as gd

        original = gd.gated_delta_update

        def naive(*a, **kw):
            kw["use_kernel"] = False
            return original(*a, **kw)

        gd.gated_delta_update = naive
        import mlx_lm.models.qwen3_5 as q35

        q35.gated_delta_update = naive
        print("GDN kernel disabled (naive ops path)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(str(args.model))
    print(f"model type: {type(model).__name__}", flush=True)

    ids = tokenizer.encode(PROMPT)
    print(f"prompt: {PROMPT!r} -> token ids {ids}", flush=True)
    tokens = mx.array([ids])

    lm = model.language_model  # TextModel
    text = lm.model  # Qwen3_5TextModel

    dumps = {}
    h = text.embed_tokens(tokens)
    dumps["embed"] = h
    mx.eval(h)

    caches = model.make_cache()
    fa_mask = create_attention_mask(h, caches[text.fa_idx])
    ssm_mask = create_ssm_mask(h, caches[text.ssm_idx])

    for i, layer in enumerate(text.layers):
        mask = ssm_mask if layer.is_linear else fa_mask
        if layer.is_linear:
            r = layer.linear_attn(layer.input_layernorm(h), mask, caches[i])
            kind = "gdn"
        else:
            r = layer.self_attn(layer.input_layernorm(h), mask, caches[i])
            kind = "attn"
        h = h + r
        mx.eval(h)
        if DUMP_ALL or i in (0, 3):
            dumps[f"layer{i}_{kind}_out"] = h
        mlp_in = layer.post_attention_layernorm(h)
        h = h + layer.mlp(mlp_in)
        mx.eval(h)
        if DUMP_ALL or i in (0, 3):
            dumps[f"layer{i}_mlp_out"] = h
        print(f"layer {i} ({kind}) done", flush=True)

    hidden_final = text.norm(h)
    dumps["hidden_final"] = hidden_final
    logits = lm.lm_head(hidden_final)
    dumps["logits"] = logits
    mx.eval(logits)

    for name, tensor in dumps.items():
        array = np.array(tensor.astype(mx.float32))
        np.save(args.out / f"{name}.npy", array)
        print(f"saved {name}.npy {array.shape} {array.dtype}", flush=True)

    meta = {
        "prompt": PROMPT,
        "input_ids": [ids],
        "model_dir": str(args.model),
        "mlx_version": version("mlx"),
        "mlx_lm_source": "/Users/zouyee/work/code/mlx-lm",
        "numpy_version": np.__version__,
        "segments": {
            name: {"shape": list(t.shape), "dtype": "float32"}
            for name, t in dumps.items()
        },
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
