#!/usr/bin/env python3
"""Make omlx's NATIVE MTP work on the community Qwen3.8-27B-4bit checkpoint.

Why this is needed
------------------
The engine comparison (plan doc §13) could only run omlx as a *no-speculation*
row, which makes the speculative axis meaningless: our MTP vs their plain
decode is not a comparison. The stated reason was "the community 4-bit export
stripped the mtp.* weights". That was wrong. The weights are not missing — they
ship as a SEPARATE checkpoint (Qwen3.8-27B-MTP-4bit) in FLAT layout:

    fc.weight, layers.0.self_attn.q_proj.weight, norm.weight,
    pre_fc_norm_hidden.weight, pre_fc_norm_embedding.weight, ...

omlx's `combine_mtp_donor` only recognizes donors whose keys already carry an
`mtp.` prefix (`oq._strip_mtp_key_prefix` / `_MTP_WEIGHT_PREFIXES`), so it
reports "mtp.* weights stripped" for a flat donor. And omlx's head module
(`patches/mlx_lm_mtp/qwen35_model.py`) is built from exactly those five
component names, so `mtp.` + flat key is precisely the layout it wants.

What this does
--------------
1. Writes a NORMALIZED donor (prefix every drafter tensor with `mtp.`).
2. Builds a recipient directory that SYMLINKS the 14 GB of main-checkpoint
   shards (only config.json / index are real files, because the combine
   rewrites them) — so this costs ~240 MB, not ~15 GB.
3. Calls omlx's own `combine_mtp_donor`, which handles the parts that are easy
   to get wrong by hand: per-layer quantization entries synthesized under the
   recipient's key naming (the donor's global bits may differ), and the raw-HF
   vs MLX RMSNorm `+1` rule.

Pre-flight checks that all pass on this pair (omlx's `validate_mtp_donor_pair`
enforces them): family token `qwen3_5` on both, `mtp_num_hidden_layers=1`,
every `_MTP_GEOMETRY_FIELDS` entry equal, and byte-identical `tokenizer.json`
(sha256 06b95093...) — required because the head reuses the recipient's
embed_tokens / lm_head.

The original checkpoints are never modified.

Usage:
    ~/venvs/dmlx-omlx/bin/python scripts/omlx_enable_native_mtp.py [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HOME = Path.home()
MAIN = HOME / "models/Qwen3.8-27B-4bit"
DRAFTER = HOME / "models/Qwen3.8-27B-MTP-4bit"
DONOR = Path("/tmp/omlx_mtp_donor")
OUT = HOME / "models/Qwen3.8-27B-4bit-mtp"

# Files a served checkpoint needs besides the weight shards.
SIDECARS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    # This checkpoint is a VLM wrapper (Qwen3_5ForConditionalGeneration), so
    # the processor configs must come along or the loader cannot build the
    # processor even for text-only serving.
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


def build_donor() -> None:
    import mlx.core as mx

    DONOR.mkdir(parents=True, exist_ok=True)
    flat = mx.load(str(DRAFTER / "model.safetensors"))
    prefixed = {f"mtp.{k}": v for k, v in flat.items()}
    mx.save_safetensors(
        str(DONOR / "model.safetensors"), prefixed, metadata={"format": "mlx"}
    )
    with (DONOR / "model.safetensors.index.json").open("w") as f:
        json.dump(
            {
                "metadata": {"total_size": sum(v.nbytes for v in prefixed.values())},
                "weight_map": {k: "model.safetensors" for k in prefixed},
            },
            f,
            indent=1,
        )
    shutil.copy2(DRAFTER / "config.json", DONOR / "config.json")
    for name in SIDECARS:
        src = DRAFTER / name
        if src.exists():
            shutil.copy2(src, DONOR / name)
    print(f"donor: {len(prefixed)} tensors -> {DONOR}")


def build_recipient(force: bool) -> None:
    if OUT.exists():
        if not force:
            sys.exit(f"{OUT} exists; pass --force to rebuild")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    # Weight shards: symlink (read-only, never rewritten by the combine).
    shards = sorted(MAIN.glob("*.safetensors"))
    if not shards:
        sys.exit(f"no shards in {MAIN}")
    for shard in shards:
        (OUT / shard.name).symlink_to(shard)
    # config.json + index: real copies, the combine rewrites both.
    shutil.copy2(MAIN / "config.json", OUT / "config.json")
    index = MAIN / "model.safetensors.index.json"
    if index.exists():
        shutil.copy2(index, OUT / "model.safetensors.index.json")
    else:
        # Single-shard checkpoints ship no index; synthesize one so the
        # combine has a weight_map to merge the mtp keys into.
        import mlx.core as mx

        weight_map = {}
        for shard in shards:
            for key in mx.load(str(shard)).keys():
                weight_map[key] = shard.name
        with (OUT / "model.safetensors.index.json").open("w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)
    for name in SIDECARS:
        src = MAIN / name
        if src.exists():
            shutil.copy2(src, OUT / name)
    print(f"recipient: {len(shards)} shards symlinked -> {OUT}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for p in (MAIN, DRAFTER):
        if not p.is_dir():
            sys.exit(f"missing checkpoint: {p}")

    build_donor()
    build_recipient(args.force)

    from omlx.oq import combine_mtp_donor
    from omlx.utils.model_loading import _checkpoint_has_mtp_weights

    combine_mtp_donor(OUT, DONOR)

    ok = _checkpoint_has_mtp_weights(OUT)
    print(f"\n_checkpoint_has_mtp_weights({OUT.name}) = {ok}")
    if not ok:
        print("FAILED: omlx still does not see mtp weights", file=sys.stderr)
        return 1
    quant = (json.loads((OUT / "config.json").read_text()).get("quantization") or {})
    mtp_entries = [k for k in quant if k.startswith("mtp.")]
    print(f"synthesized per-layer quant entries for mtp.*: {len(mtp_entries)}")
    print(f"\nServe with:  omlx.cli serve --model-dir {OUT.parent} "
          f"(model id: {OUT.name})  + mtp_enabled=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
