#!/usr/bin/env python3
"""Dump DeepSeek-V4-Flash per-layer hidden states from MLX for native alignment.

Matches native chat-mode tokenization by applying the DeepSeek chat template
before selecting the first effective token.

Usage:
    DSV4_DUMP_DIR=/tmp/dsv4_mlx_dump python3 scripts/dump_mlx_activations.py "What is the capital of France?"
"""
import os
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import CacheList


def main():
    if len(sys.argv) < 2:
        print("Usage: DSV4_DUMP_DIR=/tmp/dump python3 dump_mlx_activations.py <prompt>")
        sys.exit(1)

    prompt = sys.argv[1]
    dump_dir = os.environ.get("DSV4_DUMP_DIR")
    if not dump_dir:
        print("DSV4_DUMP_DIR not set", file=sys.stderr)
        sys.exit(1)
    os.makedirs(dump_dir, exist_ok=True)

    model_path = os.path.expanduser("~/models/DeepSeek-V4-Flash-4bit")
    print(f"Loading model from {model_path} ...")
    # lazy=True avoids a GPU timeout during weight materialization on this model.
    model, tokenizer = load(model_path, lazy=True)

    # Apply the same DeepSeek chat template that native dmlx uses for non-raw chat.
    # Native formatting: BOS + "<｜User｜>" + prompt + "<｜Assistant｜>" + "</think>"
    formatted = (
        "<｜begin▁of▁sentence｜>"
        "<｜User｜>"
        + prompt
        + "<｜Assistant｜>"
        + "</think>"
    )
    ids = tokenizer.encode(formatted)
    print(f"Formatted prompt token ids: {ids}")

    # Strip BOS/EOS framing to match native_engine.zig.
    while ids and ids[0] == 0:
        ids = ids[1:]
    while ids and ids[-1] == 1:
        ids = ids[:-1]
    if not ids:
        print("No effective tokens after stripping BOS/EOS", file=sys.stderr)
        sys.exit(1)
    first_token_id = ids[0]
    print(f"First effective token id: {first_token_id}")

    inputs = mx.array([[first_token_id]])

    from mlx_lm.models import deepseek_v4

    original_call = deepseek_v4.DeepseekV4Model.__call__

    def dump_call(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)
        mx.eval(h)
        _write_tensor(h, dump_dir, 0, "in")

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList) else first_cache
        )
        mask = deepseek_v4.create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        if self.pipeline_rank < self.pipeline_size - 1:
            h = mx.distributed.recv_like(h, (self.pipeline_rank + 1))

        for idx, (layer, layer_cache) in enumerate(zip(self.pipeline_layers, cache)):
            h = layer(h, mask, layer_cache, inputs)
            mx.eval(h)
            _write_tensor(h, dump_dir, idx, "out")
            if idx + 1 < len(self.pipeline_layers):
                _write_tensor(h, dump_dir, idx + 1, "in")

        if self.pipeline_rank != 0:
            h = mx.distributed.send(h, (self.pipeline_rank - 1) % self.pipeline_size)
            cache_item = cache[-1]
            if isinstance(cache_item, CacheList):
                cache_item = cache_item[0]
            if cache_item is not None:
                cache_item.keys = mx.depends(cache_item.keys, h)

        if self.pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(self.hc_head(h))

    deepseek_v4.DeepseekV4Model.__call__ = dump_call

    print("Running single-token forward ...")
    out = model(inputs, cache=None)
    mx.eval(out)
    print(f"Done. Dumps written to {dump_dir}")


def _write_tensor(h: mx.array, dump_dir: str, layer: int, kind: str):
    f32 = h.astype(mx.float32)
    mx.eval(f32)
    arr = np.array(f32)
    path = os.path.join(dump_dir, f"L{layer:02d}_{kind}_pos0.bin")
    arr.tofile(path)
    print(f"  wrote {path} shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
