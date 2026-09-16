#!/usr/bin/env python3
"""Dump GLM-5.3 intermediate activations from the source MLX checkpoint as a
numerical oracle (golden reference) for the dmlx Zig implementation.

Loads the text tower of ~/models/GLM-5.3-Flash-MLX-mixed-4_8bit with
mlx-vlm's glm5_next modules, runs a fixed 7-token prompt layer by layer with
per-layer weight eviction (the checkpoint is ~125GB, far larger than RAM),
and saves float32 .npy fixtures plus meta.json.

Usage:
    python3 scripts/glm53_oracle_dump.py MODEL_DIR OUT_DIR [--self-check]

Example (regenerate the committed fixtures):
    python3 scripts/glm53_oracle_dump.py \
        ~/models/GLM-5.3-Flash-MLX-mixed-4_8bit \
        tests/fixtures/glm53_oracle --self-check

Requires: mlx-vlm >= 0.6.17 (developed against 0.7.1), mlx >= 0.32, numpy.
The script intentionally bypasses mlx_vlm.utils.load_model: it builds
LanguageModel directly, remaps the checkpoint's KDA keys (forget_gate.*,
fused conv1d) that the installed sanitize() does not cover, derives per-module
quantization from the packed weight shapes, and never calls mx.eval on the
full parameter set.
"""

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

INPUT_IDS = [17, 42, 314, 2718, 7, 99, 1234]

KDA_DUMP_LAYERS = (0, 30)
MLA_DUMP_LAYERS = (3, 43)
MOE_DUMP_LAYER = 10
TOPK_DUMP_LAYER = 3


def remap_kda_weights(weights, text_config):
    """Remap this checkpoint's KDA key layout to the installed glm5_next
    module names (installed sanitize() only handles q/k/v_proj concat)."""
    for i, layer_type in enumerate(text_config.layer_types):
        if layer_type != "linear_attention":
            continue
        p = f"language_model.model.layers.{i}.self_attn"
        conv = weights.pop(f"{p}.conv1d.weight", None)
        if conv is not None:
            # Already fused and already (channels, kernel, 1) — nn.Conv1d layout.
            weights[f"{p}.qkv_conv.conv.weight"] = conv
        for suffix in ("weight", "scales", "biases"):
            parts = []
            for src in (f"{p}.forget_gate.f_a_proj", f"{p}.b_proj", f"{p}.g_a_proj"):
                key = f"{src}.{suffix}"
                if key not in weights:
                    parts = None
                    break
                parts.append(weights.pop(key))
            if parts is not None:
                weights[f"{p}.fbg_a_proj.{suffix}"] = mx.concatenate(parts, axis=0)
        for old, new in (
            ("forget_gate.f_b_proj", "f_b_proj"),
            ("forget_gate.g_b_proj", "g_b_proj"),
        ):
            for suffix in ("weight", "scales", "biases"):
                key = f"{p}.{old}.{suffix}"
                if key in weights:
                    weights[f"{p}.{new}.{suffix}"] = weights.pop(key)
        for name in ("A_log", "dt_bias"):
            key = f"{p}.forget_gate.{name}"
            if key in weights:
                weights[f"{p}.{name}"] = weights.pop(key)


def quantize_like_checkpoint(model, weights):
    """Quantize modules to match the checkpoint's packed tensors, deriving
    group_size/bits from tensor shapes (handles the mixed 4/8-bit layout)."""

    def predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        w = weights.get(f"language_model.{path}.weight")
        s = weights.get(f"language_model.{path}.scales")
        if w is None or s is None:
            return False
        in_dims = module.weight.shape[-1]
        if w.shape[-1] * 32 % in_dims or in_dims % s.shape[-1]:
            return False
        return {
            "group_size": in_dims // s.shape[-1],
            "bits": w.shape[-1] * 32 // in_dims,
        }

    nn.quantize(model, group_size=64, bits=4, class_predicate=predicate)


def build_model(model_path):
    from mlx_vlm.models.glm5_next.config import TextConfig
    from mlx_vlm.models.glm5_next.language import LanguageModel

    config = json.loads((model_path / "config.json").read_text())
    text_config = TextConfig.from_dict(config["text_config"])
    model = LanguageModel(text_config)

    index = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weights = {}
    for shard in sorted(set(index.values())):
        for key, value in mx.load(str(model_path / shard)).items():
            if key.startswith("language_model."):
                weights[key] = value
    remap_kda_weights(weights, text_config)
    weights = model.sanitize(weights)
    quantize_like_checkpoint(model, weights)
    weights = {
        key[len("language_model.") :]: value for key, value in weights.items()
    }
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    return model, weights


def drop_keys(weights, fragment):
    for key in [k for k in weights if fragment in k]:
        del weights[key]


def run_forward(model, weights):
    """Single 7-token prefill with per-layer weight eviction. Returns dumps."""
    from mlx_vlm.models.linear import linear

    dumps = {}
    text = model.model
    ids = mx.array([INPUT_IDS])
    h = text.embed_tokens(ids)
    dumps["embed"] = h
    mx.eval(h)
    drop_keys(weights, "model.embed_tokens.")
    text.embed_tokens = None

    h = mx.repeat(h[:, :, None], text.config.hc_mult, axis=2)
    topk = None
    layers = list(text.layers)
    for i, layer in enumerate(layers):
        if layer.is_linear:
            h = layer.attn_hc.apply_branch(
                h, layer.input_layernorm, layer.self_attn, None, None
            )
            if i in KDA_DUMP_LAYERS:
                dumps[f"layer{i}_kda_out"] = h
            h = layer.ffn_hc.apply_branch(h, layer.post_attention_layernorm, layer.mlp)
        else:
            h, topk = layer.attn_hc.apply_branch(
                h, layer.input_layernorm, layer.self_attn, None, None, topk
            )
            if i in MLA_DUMP_LAYERS:
                dumps[f"layer{i}_mla_out"] = h
            if i == TOPK_DUMP_LAYER:
                dumps[f"layer{TOPK_DUMP_LAYER}_indexer_topk"] = topk
            h = layer.ffn_hc.apply_branch(h, layer.post_attention_layernorm, layer.mlp)
        if i == MOE_DUMP_LAYER:
            dumps[f"layer{MOE_DUMP_LAYER}_moe_out"] = h
        mx.eval(h)
        layers[i] = None
        text.layers[i] = None
        drop_keys(weights, f"model.layers.{i}.")
        gc.collect()
        mx.clear_cache()

    hidden_final = text.norm(h.mean(axis=2))
    dumps["hidden_final"] = hidden_final
    logits = linear(model.lm_head, hidden_final)
    dumps["logits"] = logits
    mx.eval(logits)
    return dumps


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run a second forward pass and require logits cosine >= 0.9999",
    )
    args = parser.parse_args()

    import mlx_vlm  # noqa: F401  (version check below)
    from importlib.metadata import version

    mlx_version = version("mlx")
    vlm_version = version("mlx-vlm")
    vlm_tuple = tuple(int(x) for x in vlm_version.split(".")[:3])
    if vlm_tuple < (0, 6, 17):
        sys.exit(f"mlx-vlm >= 0.6.17 required, found {vlm_version}")

    args.model_dir = args.model_dir.expanduser()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)

    t0 = time.time()
    model, weights = build_model(args.model_dir)
    print(f"model built in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    dumps = run_forward(model, weights)
    print(
        f"forward pass 1 in {time.time() - t0:.1f}s, "
        f"peak metal {mx.get_peak_memory() / 2**30:.2f} GiB, "
        f"peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GiB",
        flush=True,
    )

    for name, tensor in dumps.items():
        if name.endswith("topk"):
            array = np.array(tensor.astype(mx.int32))
        else:
            array = np.array(tensor.astype(mx.float32))
        np.save(args.out_dir / f"{name}.npy", array)
        print(f"saved {name}.npy {array.shape} {array.dtype}", flush=True)

    meta = {
        "input_ids": [INPUT_IDS],
        "model_dir": str(args.model_dir.expanduser()),
        "command": command,
        "mlx_version": mlx_version,
        "mlx_vlm_version": vlm_version,
        "numpy_version": np.__version__,
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": "int32" if name.endswith("topk") else "float32",
            }
            for name, tensor in dumps.items()
        },
    }

    if args.self_check:
        logits_ref = np.array(dumps["logits"].astype(mx.float32))
        del dumps, model, weights
        gc.collect()
        mx.clear_cache()
        model2, weights2 = build_model(args.model_dir)
        t0 = time.time()
        dumps2 = run_forward(model2, weights2)
        print(f"forward pass 2 in {time.time() - t0:.1f}s", flush=True)
        logits2 = np.array(dumps2["logits"].astype(mx.float32))
        cos = cosine(logits_ref, logits2)
        print(f"self-check logits cosine: {cos:.8f}", flush=True)
        if cos < 0.9999:
            sys.exit(f"self-check FAILED: logits cosine {cos:.8f} < 0.9999")
        meta["self_check_logits_cosine"] = cos

    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
