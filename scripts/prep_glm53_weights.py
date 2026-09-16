#!/usr/bin/env python3
"""prep_glm53_weights.py — offline sanitize + layout prep for GLM-5.3-Flash MLX weights.

Usage:
    python3 scripts/prep_glm53_weights.py SRC_DIR DST_DIR
    python3 scripts/prep_glm53_weights.py --verify SRC_DIR DST_DIR

Produces DST_DIR laid out for dmlx expert streaming:
  - All source shards hard-linked (no double disk usage); only newly produced
    (concatenated / renamed) tensors are written to extra `dmlx-merge-*.safetensors`.
  - A fresh `model.safetensors.index.json` maps every output tensor to its shard.
  - `dmlx_manifest.json` records per-tensor bits/group_size and transform stats.

Ground truth: key layout of ~/models/GLM-5.3-Flash-MLX-mixed-4_8bit (mixed affine
4/8bit, probed 2026-09-15 with the Step-1 key-pattern scan, 45 layers, prefix
`language_model.model.`):

  - Experts are ALREADY STACKED: `...mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`
    with shape [288, out, packed_in] (e.g. gate_proj.weight [288, 2048, 512] U32 = 4bit,
    group 64; scales/biases [288, 2048, 64] BF16). No per-expert tensors exist.
  - MLA layers (full_attn_layers = [3,7,11,15,19,23,27,31,35,39,43]):
    kv_b split ALREADY done: `self_attn.embed_q.*` ([64,512,64] U32, 8bit g64) and
    `self_attn.unembed_out.*` ([64,256,128] U32, 8bit g64) already exist.
    `self_attn.q_a_proj.*` ([1536,...]) and `self_attn.kv_a_proj_with_mqa.*` ([512,...])
    are SEPARATE -> concatenated here into `self_attn.qkv_a_proj.*`.
  - KDA layers (all other layers, incl. dense layers 0-2):
    `self_attn.{q,k,v}_proj.*` separate -> `self_attn.qkv_proj.*`.
    `self_attn.forget_gate.f_a_proj.*` + `self_attn.b_proj.*` + `self_attn.g_a_proj.*`
    separate -> `self_attn.fbg_a_proj.*` (f_b_proj / g_b_proj left untouched).
    Depthwise convs ALREADY merged as `self_attn.conv1d.weight` ([24576, 4, 1] BF16,
    i.e. 3*8192 channels, [C, K, 1] layout) -> renamed to `self_attn.qkv_conv.weight`.
  - Dense MLP (layers 0-2) `mlp.{gate,up}_proj.*` and MoE `mlp.shared_experts.{gate,up}_proj.*`
    separate -> `gate_up_proj.*` (concat axis 0, per suffix weight/scales/biases).
  - HC params ALREADY named `attn_hc.{fn,base,scale}` / `ffn_hc.{fn,base,scale}`.
  - fp32/bf16 unquantized keep-list (hard-linked as-is): forget_gate.A_log (F32),
    forget_gate.dt_bias (F32), mlp.gate.e_score_correction_bias (F32),
    attn_hc.* / ffn_hc.* (BF16/F32), indexer.index_kpool_compress_{ape,gate} (BF16),
    mlp.gate.weight (BF16 router), all layernorms, conv1d (BF16).
  - Quantization is affine (biases present): bits = packed_u32*32/in_features,
    group_size = in_features/scales.shape[-1]. Mixed 4bit (MoE/dense MLP, attn proj)
    and 8bit (embed_tokens, lm_head, embed_q/unembed_out).
"""

import argparse
import json
import os
import re
import sys

import mlx.core as mx
from safetensors import safe_open

PREP_VERSION = "1.0.0"
N_LAYERS = 45
FIRST_K_DENSE = 3
N_ROUTED_EXPERTS = 288
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
LAYER_PREFIX = "language_model.model.layers"
PRODUCED_SHARD_FMT = "dmlx-merge-{:05d}.safetensors"
SUFFIXES = ("weight", "scales", "biases")

# Unquantized tensors that must be preserved as-is (never dequantized).
FP32_KEEP_PATTERNS = (
    "A_log",
    "dt_bias",
    "e_score_correction_bias",
    "attn_hc.",
    "ffn_hc.",
    "index_kpool_compress_",
)


def is_kda(layer_idx):
    return layer_idx not in FULL_ATTN_LAYERS


def build_transform_plan():
    """Return list of ops: {kind, layer, sources: {suffix: [keys]}, dest_base}.

    kind == "concat": dest is f"{dest_base}.{suffix}" = concat(sources[suffix], axis=0)
    kind == "rename": dest is f"{dest_base}.weight" = copy of sources["weight"][0]
    """
    ops = []
    for i in range(N_LAYERS):
        p = f"{LAYER_PREFIX}.{i}"
        if i < FIRST_K_DENSE:
            ops.append(dict(
                kind="concat", layer=i, dest_base=f"{p}.mlp.gate_up_proj",
                sources={s: [f"{p}.mlp.gate_proj.{s}", f"{p}.mlp.up_proj.{s}"]
                         for s in SUFFIXES}))
        else:
            ops.append(dict(
                kind="concat", layer=i, dest_base=f"{p}.mlp.shared_experts.gate_up_proj",
                sources={s: [f"{p}.mlp.shared_experts.gate_proj.{s}",
                             f"{p}.mlp.shared_experts.up_proj.{s}"] for s in SUFFIXES}))
        if is_kda(i):
            ops.append(dict(
                kind="concat", layer=i, dest_base=f"{p}.self_attn.qkv_proj",
                sources={s: [f"{p}.self_attn.{proj}.{s}"
                             for proj in ("q_proj", "k_proj", "v_proj")]
                         for s in SUFFIXES}))
            ops.append(dict(
                kind="concat", layer=i, dest_base=f"{p}.self_attn.fbg_a_proj",
                sources={s: [f"{p}.self_attn.forget_gate.f_a_proj.{s}",
                             f"{p}.self_attn.b_proj.{s}",
                             f"{p}.self_attn.g_a_proj.{s}"] for s in SUFFIXES}))
            ops.append(dict(
                kind="rename", layer=i, dest_base=f"{p}.self_attn.qkv_conv",
                sources={"weight": [f"{p}.self_attn.conv1d.weight"]}))
        else:
            ops.append(dict(
                kind="concat", layer=i, dest_base=f"{p}.self_attn.qkv_a_proj",
                sources={s: [f"{p}.self_attn.q_a_proj.{s}",
                             f"{p}.self_attn.kv_a_proj_with_mqa.{s}"] for s in SUFFIXES}))
    return ops


def scan_shards(src_dir):
    """Return (key -> shard_filename, shard -> [keys])."""
    key_to_shard = {}
    shard_keys = {}
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(src_dir, name)
        with safe_open(path, framework="mlx") as sf:
            keys = list(sf.keys())
        shard_keys[name] = keys
        for k in keys:
            key_to_shard[k] = name
    return key_to_shard, shard_keys


def validate_sources(key_to_shard, ops):
    missing = []
    for op in ops:
        for keys in op["sources"].values():
            for k in keys:
                if k not in key_to_shard:
                    missing.append(k)
    # MoE expert stacks must already exist and cover every MoE layer
    for i in range(FIRST_K_DENSE, N_LAYERS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for s in SUFFIXES:
                k = f"{LAYER_PREFIX}.{i}.mlp.switch_mlp.{proj}.{s}"
                if k not in key_to_shard:
                    missing.append(k)
    # MLA layers must already have the kv_b split products
    for i in FULL_ATTN_LAYERS:
        for proj in ("embed_q", "unembed_out"):
            for s in SUFFIXES:
                k = f"{LAYER_PREFIX}.{i}.self_attn.{proj}.{s}"
                if k not in key_to_shard:
                    missing.append(k)
    if missing:
        raise SystemExit(
            "Source model is missing %d expected tensors (incomplete download?).\n"
            "First missing keys:\n  %s" % (len(missing), "\n  ".join(missing[:20])))


class ShardLoader:
    """Lazy per-shard loader backed by mx.load (safetensors' own mlx path
    cannot decode bfloat16). Keeps at most `max_open` shards in memory."""

    def __init__(self, root, key_to_shard, max_open=2):
        self.root = root
        self.key_to_shard = key_to_shard
        self.max_open = max_open
        self._cache = {}

    def get(self, key):
        shard = self.key_to_shard[key]
        if shard not in self._cache:
            if len(self._cache) >= self.max_open:
                self._cache.clear()
            self._cache[shard] = mx.load(os.path.join(self.root, shard))
        return self._cache[shard][key]

    def close(self):
        self._cache.clear()


def hardlink_tree(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src_dir)):
        if name in (".cache", "model.safetensors.index.json", "dmlx_manifest.json"):
            continue
        s = os.path.join(src_dir, name)
        if not os.path.isfile(s):
            continue
        d = os.path.join(dst_dir, name)
        if os.path.exists(d):
            if os.stat(d).st_ino == os.stat(s).st_ino:
                continue
            os.remove(d)
        os.link(s, d)
        n += 1
    return n


def infer_quant(weight_shape, scales_shape):
    """Infer (bits, group_size) for an affine-quantized weight, or None."""
    packed = weight_shape[-1]
    groups = scales_shape[-1]
    for bits in (8, 4, 2):
        in_features = packed * 32 // bits
        if packed * 32 % bits:
            continue
        if in_features % groups == 0 and in_features // groups in (32, 64, 128):
            return bits, in_features // groups
    return None


def run_prep(src_dir, dst_dir):
    key_to_shard, _ = scan_shards(src_dir)
    ops = build_transform_plan()
    validate_sources(key_to_shard, ops)

    consumed = set()
    produced = {}  # key -> shard name
    stats = {"concat_ops": 0, "rename_ops": 0, "produced_tensors": 0}

    n_linked = hardlink_tree(src_dir, dst_dir)
    print(f"hard-linked {n_linked} files into {dst_dir}")

    quant_info = {}
    loader = ShardLoader(src_dir, key_to_shard)

    # Authoritative per-module quantization from config.json (default + overrides
    # keyed by module prefix, e.g. "language_model.model.layers.0.self_attn.q_proj").
    with open(os.path.join(src_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg_quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    default_quant = {"bits": cfg_quant.get("bits", 4),
                     "group_size": cfg_quant.get("group_size", 64)}

    def quant_for(base, src_bases=()):
        for b in (base, *src_bases):
            if b in cfg_quant:
                e = cfg_quant[b]
                return {"bits": e["bits"], "group_size": e["group_size"]}
        return dict(default_quant)

    for op in ops:
        if op["kind"] == "concat":
            src_bases = [sk[: -len(".weight")] for sk in op["sources"]["weight"]]
            qs = [quant_for(b) for b in src_bases]
            if any(q != qs[0] for q in qs):
                raise SystemExit(
                    f"concat sources with mixed quantization for {op['dest_base']}: "
                    + ", ".join(f"{b}={q}" for b, q in zip(src_bases, qs)))
        out = {}
        for suffix, keys in op["sources"].items():
            arrays = [loader.get(k) for k in keys]
            consumed.update(keys)
            if op["kind"] == "concat":
                out[f"{op['dest_base']}.{suffix}"] = mx.concatenate(arrays, axis=0)
            else:
                out[f"{op['dest_base']}.{suffix}"] = arrays[0]
        mx.eval(*out.values())
        # one shard per op keeps things simple and memory-light
        shard_name = PRODUCED_SHARD_FMT.format(len(produced) + 1)
        path = os.path.join(dst_dir, shard_name)
        mx.save_safetensors(path, out)
        for k, v in out.items():
            produced[k] = shard_name
            stats["produced_tensors"] += 1
            base, _, suffix = k.rpartition(".")
            if suffix == "weight" and f"{base}.scales" in out:
                src_bases = [sk[: -len(".weight")]
                             for sk in op["sources"]["weight"]]
                q = quant_for(base, src_bases)
                # sanity: packed U32 must be consistent with bits/in-features
                q_inf = infer_quant(v.shape, out[f"{base}.scales"].shape)
                if q_inf and (q_inf[0], q_inf[1]) != (q["bits"], q["group_size"]):
                    print(f"  WARN quant mismatch {base}: config={q} shape-inferred={q_inf}")
                quant_info[base] = q
        stats["concat_ops" if op["kind"] == "concat" else "rename_ops"] += 1
        print(f"  [{op['kind']}] layer {op['layer']:2d} -> {shard_name}: "
              + ", ".join(f"{k.rpartition('.')[2]}={tuple(v.shape)}" for k, v in out.items()))

    # Existing quantized tensors: record bits/group_size in the manifest too.
    for k in list(key_to_shard):
        if not k.endswith(".scales") or k in consumed:
            continue
        base = k[: -len(".scales")]
        wk = f"{base}.weight"
        if wk in consumed or wk not in key_to_shard:
            continue
        quant_info[base] = quant_for(base)

    # Rewrite the index: unchanged keys -> original shard; produced -> new shard.
    weight_map = {}
    for k, shard in key_to_shard.items():
        if k not in consumed:
            weight_map[k] = shard
    weight_map.update(produced)
    index = {"metadata": {"total_size": 0}, "weight_map": dict(sorted(weight_map.items()))}
    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    manifest = {
        "prep_version": PREP_VERSION,
        "source": os.path.abspath(src_dir),
        "n_layers": N_LAYERS,
        "full_attn_layers": FULL_ATTN_LAYERS,
        "first_k_dense_replace": FIRST_K_DENSE,
        "n_routed_experts": N_ROUTED_EXPERTS,
        "quant": dict(sorted(quant_info.items())),
        "fp32_keep_patterns": list(FP32_KEEP_PATTERNS),
        "stats": {
            **stats,
            "consumed_tensors": len(consumed),
            "unchanged_tensors": len(weight_map) - len(produced),
            "total_output_tensors": len(weight_map),
        },
    }
    with open(os.path.join(dst_dir, "dmlx_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote model.safetensors.index.json and dmlx_manifest.json")
    print(json.dumps(manifest["stats"], indent=2))


def load_key(loader, key):
    return loader.get(key)


def run_verify(src_dir, dst_dir):
    import random

    src_map, _ = scan_shards(src_dir)
    with open(os.path.join(dst_dir, "model.safetensors.index.json")) as f:
        dst_map = json.load(f)["weight_map"]
    src_loader = ShardLoader(src_dir, src_map)
    dst_loader = ShardLoader(dst_dir, dst_map)
    with open(os.path.join(dst_dir, "dmlx_manifest.json")) as f:
        manifest = json.load(f)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        ok = ok and cond

    print("check 1: MoE expert stacks have shape[0] == 288")
    for i in range(FIRST_K_DENSE, N_LAYERS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            k = f"{LAYER_PREFIX}.{i}.mlp.switch_mlp.{proj}.weight"
            if k not in dst_map:
                check(f"layer {i} {proj}", False, "missing")
                continue
            t = load_key(dst_loader, k)
            check(f"layer {i} {proj}", t.shape[0] == N_ROUTED_EXPERTS,
                  f"shape={tuple(t.shape)}")

    print("check 2: concat products are bit-exact vs sources")
    rng = random.Random(0)
    moe_layer = rng.choice([i for i in range(FIRST_K_DENSE, N_LAYERS)])
    p = f"{LAYER_PREFIX}.{moe_layer}"
    merged = load_key(dst_loader, f"{p}.mlp.shared_experts.gate_up_proj.weight")
    g = load_key(src_loader, f"{p}.mlp.shared_experts.gate_proj.weight")
    u = load_key(src_loader, f"{p}.mlp.shared_experts.up_proj.weight")
    rows = g.shape[0]
    check("shared gate_up (gate half)", mx.array_equal(merged[:rows], g))
    check("shared gate_up (up half)", mx.array_equal(merged[rows:], u))
    # expert stack slice vs itself through the index (layout round-trip)
    ek = f"{p}.mlp.switch_mlp.gate_proj.weight"
    stack_dst = load_key(dst_loader, ek)
    stack_src = load_key(src_loader, ek)
    check("expert stack expert 7 slice bit-exact",
          mx.array_equal(stack_dst[7], stack_src[7]),
          f"layer {moe_layer} expert 7")
    kda_layer = rng.choice([i for i in range(N_LAYERS) if is_kda(i)])
    p = f"{LAYER_PREFIX}.{kda_layer}"
    qkv = load_key(dst_loader, f"{p}.self_attn.qkv_proj.weight")
    off = 0
    for proj in ("q_proj", "k_proj", "v_proj"):
        src_t = load_key(src_loader, f"{p}.self_attn.{proj}.weight")
        r = src_t.shape[0]
        check(f"qkv_proj[{proj}] layer {kda_layer}",
              mx.array_equal(qkv[off:off + r], src_t))
        off += r
    fbg = load_key(dst_loader, f"{p}.self_attn.fbg_a_proj.weight")
    off = 0
    for sk in (f"{p}.self_attn.forget_gate.f_a_proj.weight",
               f"{p}.self_attn.b_proj.weight",
               f"{p}.self_attn.g_a_proj.weight"):
        src_t = load_key(src_loader, sk)
        r = src_t.shape[0]
        check(f"fbg_a_proj[{sk.rpartition('.')[0].rpartition('.')[2]}] layer {kda_layer}",
              mx.array_equal(fbg[off:off + r], src_t))
        off += r
    conv_dst = load_key(dst_loader, f"{p}.self_attn.qkv_conv.weight")
    conv_src = load_key(src_loader, f"{p}.self_attn.conv1d.weight")
    check(f"qkv_conv rename bit-exact layer {kda_layer}",
          mx.array_equal(conv_dst, conv_src))

    print("check 3: MLA qkv_a_proj matches sources; embed_q/unembed_out sane")
    for i in FULL_ATTN_LAYERS:
        p = f"{LAYER_PREFIX}.{i}"
        merged = load_key(dst_loader, f"{p}.self_attn.qkv_a_proj.weight")
        qa = load_key(src_loader, f"{p}.self_attn.q_a_proj.weight")
        kva = load_key(src_loader, f"{p}.self_attn.kv_a_proj_with_mqa.weight")
        check(f"layer {i} qkv_a bit-exact",
              mx.array_equal(merged[: qa.shape[0]], qa)
              and mx.array_equal(merged[qa.shape[0]:], kva))
        # value-level agreement after dequantize (cos >= 0.9999 per half)
        qi = manifest["quant"].get(f"{p}.self_attn.qkv_a_proj")
        if qi:
            w = load_key(dst_loader, f"{p}.self_attn.qkv_a_proj.weight")
            s = load_key(dst_loader, f"{p}.self_attn.qkv_a_proj.scales")
            b = load_key(dst_loader, f"{p}.self_attn.qkv_a_proj.biases")
            deq = mx.dequantize(w, s, b, group_size=qi["group_size"], bits=qi["bits"])
            src_deq = []
            for proj in ("q_a_proj", "kv_a_proj_with_mqa"):
                w2 = load_key(src_loader, f"{p}.self_attn.{proj}.weight")
                s2 = load_key(src_loader, f"{p}.self_attn.{proj}.scales")
                b2 = load_key(src_loader, f"{p}.self_attn.{proj}.biases")
                src_deq.append(mx.dequantize(w2, s2, b2,
                                             group_size=qi["group_size"],
                                             bits=qi["bits"]))
            src_deq = mx.concatenate(src_deq, axis=0).astype(mx.float32).reshape(-1)
            deq = deq.astype(mx.float32).reshape(-1)
            cos = (mx.sum(deq * src_deq)
                   / (mx.linalg.norm(deq) * mx.linalg.norm(src_deq))).item()
            check(f"layer {i} qkv_a dequant cos >= 0.9999", cos >= 0.9999, f"cos={cos:.6f}")
        for proj in ("embed_q", "unembed_out"):
            k = f"{p}.self_attn.{proj}.weight"
            t = load_key(dst_loader, k)
            qi = manifest["quant"].get(f"{p}.self_attn.{proj}")
            check(f"layer {i} {proj} present+quant-recorded",
                  t.shape[0] == 64 and qi is not None,
                  f"shape={tuple(t.shape)} quant={qi}")

    print("check 4: fp32/bf16 keep-list tensors present and unquantized")
    n_keep = 0
    for k in dst_map:
        if any(pat in k for pat in FP32_KEEP_PATTERNS):
            t = load_key(dst_loader, k)
            good = t.dtype in (mx.float32, mx.bfloat16)
            check(k, good, f"dtype={t.dtype}")
            n_keep += 1
            if not good:
                break
    print(f"  keep-list tensors verified: {n_keep}")

    print("VERIFY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        run_verify(args.src, args.dst)
    else:
        run_prep(args.src, args.dst)


if __name__ == "__main__":
    main()
