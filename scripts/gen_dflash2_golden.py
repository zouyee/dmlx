#!/usr/bin/env python3
"""Generate golden fixtures for the Zig DFlash2 port.

Builds a tiny DFlash2DraftModel with seeded random weights via mlx-vlm's
reference classes, runs one draft block manually (recording every
intermediate tensor), and writes:

  tests/golden/dflash2/meta.json           config + scalar inputs
  tests/golden/dflash2/weights.safetensors model weights (HF key naming)
  tests/golden/dflash2/trace.safetensors   inputs + per-stage intermediates

Run with the dmlx-omlx venv python (has mlx + mlx-vlm editable):
  ~/venvs/dmlx-omlx/bin/python scripts/gen_dflash2_golden.py
"""

import json
import os

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.speculative.drafters.dflash2.dflash2 import (
    DFlash2DraftModel,
    _grouped_dynamic_convolve,
)
from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config

SEED = 20261118
CTX_LEN = 13
ANCHOR = 42

CONFIG = dict(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    rms_norm_eps=1e-6,
    vocab_size=1000,
    max_position_embeddings=4096,
    rope_theta=10000000.0,
    block_size=8,
    mask_token_id=999,
    target_layer_ids=[1, 3, 5, 7, 9],
    num_target_layers=12,
    layer_types=["sliding_attention", "sliding_attention"],
    sliding_window=64,
    conv_kernel_size=2,
    conv_group_size=16,
    selector_rank=32,
    selector_top_k=8,
    is_causal=False,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "golden", "dflash2",
)


def rand(*shape, scale=0.02):
    return (mx.random.normal(shape) * scale).astype(mx.bfloat16)


def main():
    mx.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = DFlash2Config(**CONFIG)
    model = DFlash2DraftModel(cfg)

    # Seeded random weights, keyed with the HF checkpoint naming.
    weights = {}
    weights["fc.weight"] = rand(cfg.hidden_size, 5 * cfg.hidden_size)
    weights["hidden_norm.weight"] = rand(cfg.hidden_size, scale=1.0)
    for i, layer in enumerate(model.layers):
        p = f"layers.{i}"
        weights[f"{p}.input_layernorm.weight"] = rand(cfg.hidden_size, scale=1.0)
        weights[f"{p}.post_attention_layernorm.weight"] = rand(cfg.hidden_size, scale=1.0)
        a = layer.self_attn
        weights[f"{p}.self_attn.q_proj.weight"] = rand(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
        weights[f"{p}.self_attn.k_proj.weight"] = rand(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
        weights[f"{p}.self_attn.v_proj.weight"] = rand(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
        weights[f"{p}.self_attn.o_proj.weight"] = rand(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)
        weights[f"{p}.self_attn.q_norm.weight"] = rand(cfg.head_dim, scale=1.0)
        weights[f"{p}.self_attn.k_norm.weight"] = rand(cfg.head_dim, scale=1.0)
        for sub in ("attention_conv", "mlp_conv"):
            conv = getattr(layer, sub)
            weights[f"{p}.{sub}.base_kernel"] = rand(2, cfg.conv_kernel_size, cfg.hidden_size)
            groups = cfg.hidden_size // cfg.conv_group_size
            weights[f"{p}.{sub}.kernel_projection.weight"] = rand(
                2 * cfg.conv_kernel_size * groups, cfg.hidden_size
            )
        weights[f"{p}.mlp.gate_proj.weight"] = rand(cfg.intermediate_size, cfg.hidden_size)
        weights[f"{p}.mlp.up_proj.weight"] = rand(cfg.intermediate_size, cfg.hidden_size)
        weights[f"{p}.mlp.down_proj.weight"] = rand(cfg.hidden_size, cfg.intermediate_size)
    weights["norm.weight"] = rand(cfg.hidden_size, scale=1.0)
    s = model.candidate_selector
    weights["candidate_selector.predecessor_codebook"] = rand(cfg.vocab_size, cfg.selector_rank)
    weights["candidate_selector.successor_codebook"] = rand(cfg.vocab_size, cfg.selector_rank)
    weights["candidate_selector.hidden_projection.weight"] = rand(cfg.selector_rank, cfg.hidden_size)
    # Borrowed-from-target tables (not in the HF checkpoint; test doubles).
    weights["embed_tokens.weight"] = rand(cfg.vocab_size, cfg.hidden_size)
    weights["lm_head.weight"] = rand(cfg.vocab_size, cfg.hidden_size)

    sanitized = model.sanitize(
        {k: v for k, v in weights.items() if not k.startswith(("embed_tokens", "lm_head"))}
    )
    model.load_weights(list(sanitized.items()))
    model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    model.embed_tokens.weight = weights["embed_tokens.weight"]
    model.embed_scale = 1.0
    model.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    model.lm_head.weight = weights["lm_head.weight"]

    # Inputs.
    concat_dim = len(cfg.target_layer_ids) * cfg.hidden_size
    ctx_hidden = rand(1, CTX_LEN, concat_dim, scale=1.0)
    block_ids = mx.array([[ANCHOR] + [cfg.mask_token_id] * (cfg.block_size - 1)], dtype=mx.int32)

    trace = {"ctx_hidden": ctx_hidden, "block_ids": block_ids}

    # Manual _hidden() with recording.
    caches = model.make_cache()
    h = model._embed_input_tokens(block_ids)
    h_ctx = model.hidden_norm(model.fc(ctx_hidden))
    trace["block_embed"] = h
    trace["ctx_projected"] = h_ctx

    for idx, (layer, cache) in enumerate(zip(model.layers, caches)):
        # Attention sublayer, mirroring DFlash2DecoderLayer._forward.
        residual = h
        xn = layer.input_layernorm(h)
        trace[f"l{idx}.attn.normed"] = xn
        prepared, dyn = layer.attention_conv.prepare(xn)
        trace[f"l{idx}.attn.conv_prepared"] = prepared
        trace[f"l{idx}.attn.conv_dynamic"] = dyn
        attn_out = layer.self_attn(prepared, h_ctx, model.rope, cache)
        trace[f"l{idx}.attn.out"] = attn_out
        h = residual + layer.attention_conv.finish(attn_out, dyn)
        trace[f"l{idx}.post_attn"] = h

        residual = h
        xn = layer.post_attention_layernorm(h)
        trace[f"l{idx}.mlp.normed"] = xn
        prepared, dyn = layer.mlp_conv.prepare(xn)
        trace[f"l{idx}.mlp.conv_prepared"] = prepared
        trace[f"l{idx}.mlp.conv_dynamic"] = dyn
        mlp_out = layer.mlp(prepared)
        trace[f"l{idx}.mlp.out"] = mlp_out
        h = residual + layer.mlp_conv.finish(mlp_out, dyn)
        trace[f"l{idx}.out"] = h

        # Record ring state (K/V after update_and_fetch) for the Zig ring test.
        k, v = cache.keys, cache.values
        trace[f"l{idx}.ring_keys"] = k
        trace[f"l{idx}.ring_values"] = v
        trace[f"l{idx}.ring_offset"] = mx.array([cache.offset], dtype=mx.int32)

    h_final = model.norm(h)
    trace["final_hidden"] = h_final
    draft_hidden = h_final[:, 1:]
    logits = model._logits(draft_hidden)
    trace["logits"] = logits

    class _Greedy:
        def __call__(self, x):
            return mx.argmax(x, axis=-1)

    selected = model.candidate_selector.select(
        draft_hidden, logits, mx.array([ANCHOR], dtype=mx.int32), _Greedy()
    )
    trace["selected_tokens"] = selected.astype(mx.int32)

    # Also record q/k post-RoPE for layer 0 (RoPE parity check in Zig).
    layer0 = model.layers[0]
    xn = layer0.input_layernorm(trace["block_embed"])
    prepared0, _ = layer0.attention_conv.prepare(xn)
    B, L = 1, cfg.block_size
    S = CTX_LEN
    q = layer0.self_attn.q_proj(prepared0)
    q = layer0.self_attn.q_norm(q.reshape(B, L, cfg.num_attention_heads, -1)).transpose(0, 2, 1, 3)
    q_rope = model.rope(q, offset=S)
    trace["l0.q_post_rope"] = q_rope
    ck = layer0.self_attn.k_proj(h_ctx)
    ck = layer0.self_attn.k_norm(ck.reshape(B, S, cfg.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    trace["l0.ctx_k_post_rope"] = model.rope(ck, offset=0)

    mx.save_safetensors(os.path.join(OUT_DIR, "weights.safetensors"), weights)
    mx.save_safetensors(os.path.join(OUT_DIR, "trace.safetensors"), trace)
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({"seed": SEED, "ctx_len": CTX_LEN, "anchor": ANCHOR, "config": CONFIG}, f, indent=2)
    print(f"wrote {OUT_DIR}: weights={len(weights)} trace={len(trace)}")


if __name__ == "__main__":
    main()
