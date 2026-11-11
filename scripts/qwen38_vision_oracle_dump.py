#!/usr/bin/env python3
"""Dump Qwen3.8-27B-4bit VISION intermediates from mlx-vlm's qwen3_5
implementation as the numerical oracle for the dmlx Zig vision path
(src/vision/qwen38_preprocess.zig, src/vision/qwen38_vit.zig,
src/models/qwen38_mrope.zig, Qwen38ServingModel.forwardTokensWithVision).

Loads /Users/zouyee/models/Qwen3.8-27B-4bit with mlx-vlm, preprocesses one
fixture image through Qwen3VLImageProcessor (smart_resize + PIL BICUBIC +
normalize + merge-block patchify), runs the vision tower and the full
multimodal prefill, and dumps float32 .npy segments plus meta.json:

    pixels             [n_patches, 3*2*16*16]  processor output (pre-bf16)
    grid_thw           [3]                     int64 (t, h, w)
    tower_features     [n_merged, 5120]        vision tower output
    merged_embeddings  [L, hidden]             embed rows with image spliced
    position_ids       [3, L]                  int64 MRoPE ids
    cos / sin          [L, rotary_dims]        interleaved MRoPE tables
    logits_last        [vocab]                 last-position logits
    rope_delta         scalar                  in meta.json

Usage:
    /Users/zouyee/venvs/dmlx-omlx/bin/python scripts/qwen38_vision_oracle_dump.py \
        tests/fixtures/vision/photo.png artifacts/qwen38_vision_oracle \
        [--model /Users/zouyee/models/Qwen3.8-27B-4bit]

The fixture is a lossless PNG on purpose: at 512x384 smart_resize is the
identity, so the only pixel delta vs the Zig (stb_image) path would be JPEG
IDCT decoder differences. PNG decodes identically on both sides.

To regenerate the fixture image:
    python3 scripts/qwen38_vision_oracle_dump.py --make-fixture tests/fixtures/vision/photo.png
"""

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

# Chat wrapper around the image block. Token ids are recorded in meta.json so
# the Zig side replays exactly the same sequence (no tokenizer dependency).
PROMPT_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|>{pads}<|vision_end|>\n"
    "Describe this image.<|im_end|>\n<|im_start|>assistant\n"
)


def make_fixture(path: Path) -> None:
    """Deterministic 512x384 photo-like PNG: sky/ground gradient, a sun disc,
    a dark rectangle 'barn', diagonal stripes, and seeded smooth noise."""
    from PIL import Image

    rng = np.random.default_rng(42)
    h, w = 384, 512
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # Sky gradient (blue-ish, brighter toward top-left) over ground (green).
    sky = np.stack(
        [120 + 60 * xx / w, 170 + 40 * yy / h, 230 - 30 * yy / h], axis=-1
    )
    ground = np.stack(
        [60 + 40 * xx / w, 110 + 30 * np.sin(xx / 31.0), 50 + 20 * yy / h],
        axis=-1,
    )
    img = np.where((yy < h * 0.55)[..., None], sky, ground)
    # Sun disc.
    sun = (xx - 400) ** 2 + (yy - 80) ** 2 < 40**2
    img[sun] = [250, 235, 170]
    # Barn rectangle.
    img[250:340, 60:200] = [110, 60, 45]
    img[230:250, 45:215] = [80, 45, 40]  # roof strip
    # Diagonal fence stripes.
    stripe = ((xx + 2 * yy) % 48) < 8
    img[stripe & (yy > 300)] = [200, 200, 195]
    # Smooth low-frequency noise so bicubic != bilinear matter.
    coarse = rng.standard_normal((h // 8 + 1, w // 8 + 1, 3))
    noise = np.array(
        Image.fromarray(
            ((coarse - coarse.min()) / (np.ptp(coarse) + 1e-9) * 255).astype(np.uint8)
        ).resize((w, h), Image.BICUBIC),
        dtype=np.float64,
    )
    img = np.clip(img + (noise - 127.5) * 0.25, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    # PNG: deterministic, lossless, and decoder-identical in PIL and
    # stb_image (no quality flag — that is JPEG-only).
    Image.fromarray(img).save(path)
    print(f"wrote fixture {path} ({path.stat().st_size} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", type=Path,
                        default=Path("/Users/zouyee/models/Qwen3.8-27B-4bit"))
    parser.add_argument("--make-fixture", action="store_true",
                        help="write the fixture image and exit (no model load)")
    args = parser.parse_args()

    if args.make_fixture:
        make_fixture(args.image)
        return

    from mlx_vlm.utils import load
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

    args.out.mkdir(parents=True, exist_ok=True)

    # --- Preprocessing oracle (pure numpy/PIL, no model needed) ---
    proc = Qwen3VLImageProcessor()
    pp = proc([str(args.image)])
    pixel_values = np.asarray(pp["pixel_values"], dtype=np.float32)
    grid_thw = np.asarray(pp["image_grid_thw"], dtype=np.int64)[0]
    gt, gh, gw = (int(v) for v in grid_thw)
    n_pads = gt * (gh // proc.merge_size) * (gw // proc.merge_size)
    print(f"grid_thw={grid_thw.tolist()} n_patches={pixel_values.shape[0]} n_pads={n_pads}",
          flush=True)

    model, tokenizer = load(str(args.model))
    print(f"model type: {type(model).__name__}", flush=True)

    if os.environ.get("QWEN38_NAIVE_GDN", "0") == "1":
        # Force the ops-based GDN recurrence (no fused Metal kernel) so the
        # oracle matches the Zig ops-loop implementation step for step —
        # same convention as scripts/qwen38_oracle_dump.py.
        import mlx_vlm.models.qwen3_5.gated_delta as gd

        original = gd.gated_delta_update

        def naive(*a, **kw):
            kw["use_kernel"] = False
            return original(*a, **kw)

        gd.gated_delta_update = naive
        import mlx_vlm.models.qwen3_5.language as q35lang

        if hasattr(q35lang, "gated_delta_update"):
            q35lang.gated_delta_update = naive
        # Cheap guard against a future mlx-vlm refactor silently no-opping
        # the patch: every alias must point at the wrapper now.
        assert gd.gated_delta_update is naive
        assert not hasattr(q35lang, "gated_delta_update") or \
            q35lang.gated_delta_update is naive
        print("GDN kernel disabled (naive ops path)", flush=True)

    # --- Prompt with the image-pad block ---
    image_pad = "<|image_pad|>"
    prompt = PROMPT_TEMPLATE.format(pads=image_pad * n_pads)
    ids = tokenizer.tokenizer.encode(prompt)
    assert ids.count(model.config.image_token_index) == n_pads, (
        ids.count(model.config.image_token_index), n_pads)
    input_ids = mx.array([ids])
    L = len(ids)
    print(f"prompt tokens: {L} ({n_pads} image pads)", flush=True)

    pv_mx = mx.array(pixel_values)
    thw_mx = mx.array(grid_thw[None, :])

    dumps = {}

    # --- Vision tower ---
    vt = model.vision_tower
    trace = {}
    if os.environ.get("QWEN38_VISION_TRACE", "0") == "1":
        # Per-stage intermediates for Zig-side bisection
        # (src/vision/qwen38_vit.zig forwardTraced emits the same names).
        pe = vt.patch_embed(pv_mx.astype(vt.patch_embed.proj.weight.dtype))
        pos = vt.fast_pos_embed_interpolate(thw_mx)
        hpos = pe + pos
        rope = vt.rot_pos_emb(thw_mx)
        seq_len = hpos.shape[0]
        hh = hpos.reshape(seq_len, -1)
        rope_flat = rope.reshape(seq_len, -1)
        cu = mx.cumsum(
            mx.pad(mx.array([gh * gw]), (1, 0), mode="constant", constant_values=0)
        )
        trace["patch_embed"] = pe
        trace["pos_embed"] = pos
        trace["pos_added"] = hh
        trace["rope_freqs"] = rope_flat
        for i, blk in enumerate(vt.blocks):
            hh = blk(hh, cu_seqlens=cu, rotary_pos_emb=rope_flat)
            if i < 4 or i + 1 == len(vt.blocks):
                trace[f"block{i}"] = hh
        trace["merger_normed"] = vt.merger.norm(hh)
        trace["tower_trace_out"] = vt.merger(hh)
        mx.eval(*list(trace.values()))
    tower_out, _ = model.vision_tower(pv_mx.astype(
        model.vision_tower.patch_embed.proj.weight.dtype), thw_mx)
    mx.eval(tower_out)
    if trace:
        import numpy as _np
        a = _np.array(trace["tower_trace_out"].astype(mx.float32))
        b = _np.array(tower_out.astype(mx.float32))
        assert _np.allclose(a, b, atol=1e-3), "manual trace path diverges from tower __call__"
        for name, t in trace.items():
            if name != "tower_trace_out":
                dumps[f"trace_{name}"] = t
    dumps["tower_features"] = tower_out
    print(f"tower features {tower_out.shape}", flush=True)

    # --- Merged embeddings + MRoPE position ids (reference path) ---
    emb_feat = model.get_input_embeddings(
        input_ids, pixel_values=pv_mx, image_grid_thw=thw_mx)
    merged = emb_feat.inputs_embeds
    position_ids = emb_feat.position_ids  # [3, 1, L]
    rope_deltas = emb_feat.rope_deltas
    mx.eval(merged, position_ids)
    dumps["merged_embeddings"] = merged[0]
    dumps["position_ids"] = position_ids[:, 0, :].astype(mx.int64)

    # --- Per-slot cos/sin from a full-attention layer's rotary embedding ---
    text = model.language_model.model
    attn_layer = next(l for l in text.layers if not l.is_linear)
    cos, sin = attn_layer.self_attn.rotary_emb(mx.array(0, mx.float32), position_ids)
    mx.eval(cos, sin)
    # [3?, B, L, rd] -> squeeze to [L, rd]; channels share slots via the
    # interleaved selector, so all three channel rows are identical.
    dumps["cos"] = cos.reshape(-1, cos.shape[-2], cos.shape[-1])[0]
    dumps["sin"] = sin.reshape(-1, sin.shape[-2], sin.shape[-1])[0]
    print(f"cos {dumps['cos'].shape} rope_delta {rope_deltas}", flush=True)

    # --- Full multimodal prefill -> last-position logits ---
    cache = model.language_model.make_cache()
    out = model.language_model(
        input_ids, inputs_embeds=merged, position_ids=position_ids, cache=cache,
        capture_layer_ids=[0, 3, 8, 16, 24, 32, 40, 48, 56, 63], return_hidden=True)
    logits_last = out.logits[0, -1]
    mx.eval(logits_last)
    dumps["logits_last"] = logits_last
    if out.hidden_states:
        # hidden_sink: per-layer PRE-norm residuals (capture ids), then the
        # post-final-norm hidden appended by LanguageModel.__call__.
        layer_ids = [0, 3, 8, 16, 24, 32, 40, 48, 56, 63]
        # hidden_sink entries arrive in capture order followed by the final
        # post-norm hidden — assert the layout before zipping.
        assert len(out.hidden_states) == len(layer_ids) + 1, (
            f"expected {len(layer_ids) + 1} hidden captures, "
            f"got {len(out.hidden_states)}")
        for li, h in zip(layer_ids, out.hidden_states):
            dumps[f"lm_hidden_layer{li}"] = h[0]
        dumps["lm_hidden_final_normed"] = out.hidden_states[-1][0]
    top5 = np.argsort(np.array(logits_last.astype(mx.float32)))[-5:][::-1]
    print(f"logits_last {logits_last.shape} top5={top5.tolist()}", flush=True)

    dumps["pixels"] = pv_mx
    for name, tensor in dumps.items():
        arr = np.array(tensor.astype(mx.float32)) if tensor.dtype != mx.int64 \
            else np.array(tensor)
        np.save(args.out / f"{name}.npy", arr)
        print(f"saved {name}.npy {arr.shape} {arr.dtype}", flush=True)

    meta = {
        "prompt_template": PROMPT_TEMPLATE,
        "image": str(args.image),
        "input_ids": ids,
        "n_image_pads": n_pads,
        "grid_thw": grid_thw.tolist(),
        "merge_size": proc.merge_size,
        "rope_delta": int(np.array(rope_deltas).reshape(-1)[0]),
        "image_token_id": int(model.config.image_token_index),
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "model_dir": str(args.model),
        "mlx_version": version("mlx"),
        "mlx_vlm_version": version("mlx_vlm"),
        "segments": {name: {"shape": list(t.shape)} for name, t in dumps.items()},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
