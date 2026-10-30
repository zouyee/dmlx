#!/usr/bin/env python3
"""MTP-D style self-distillation of the Qwen3.8 MTP drafter head (plan:
docs/superpowers/plans/2026-10-30-mtp-drafter-training.md).

Loads the grafted omlx checkpoint (~/models/Qwen3.8-27B-4bit-mtp) through the
omlx mlx_lm_mtp patch, freezes the 4-bit trunk, dequantizes the single MTP
head to bf16, and trains it on on-policy corpus data (scripts/mtpd_gen_corpus.py)
with a chain-aligned objective:

    step k:  out_k = mtp(out_{k-1}, tokens[k:])      (out_0 = trunk pre-norm hidden)
    loss  =  CE(out_k -> tokens[k+1:])  +  lambda * KL(trunk_logits[k:] || out_k) * T^2

i.e. chain step k learns to predict token t+k+1 from the head's OWN previous
output hidden — exactly the inputs the inference-time draft chain consumes —
with the trunk's next-token distribution as the soft teacher (FastMTP-style
recursive alignment + MTP-D on-policy distillation).

Run with the omlx venv:  ~/venvs/dmlx-omlx/bin/python scripts/mtpd_train.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

GRAFTED = str(Path.home() / "models/Qwen3.8-27B-4bit-mtp")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_grafted(path: str):
    from omlx.patches import mlx_lm_mtp
    from omlx.patches.mlx_lm_mtp import qwen35_model as q35_patch

    q35_patch.apply()
    mlx_lm_mtp.set_mtp_active(True)
    from mlx_lm.utils import load

    model, tokenizer = load(path)
    lm = model.language_model if hasattr(model, "language_model") else model
    assert hasattr(lm, "mtp"), "MTP head not attached — patch activation failed"
    return model, lm, tokenizer


def dequantize_tree(params):
    """Recursively dequantize {weight, scales, biases} leaves to plain weights."""
    out = {}
    for k, v in params.items():
        if isinstance(v, dict):
            out[k] = dequantize_tree(v)
        elif k == "scales" or k == "biases":
            continue
        else:
            out[k] = v
    return out


def dequantize_mtp(lm):
    """Replace the (4-bit) MTP head with a bf16 trainable copy."""
    from mlx_lm.models import qwen3_5 as q35

    flat = dict(tree_flatten(lm.mtp.parameters()))
    deq = {}
    for key, w in flat.items():
        if key.endswith(".weight") and f"{key[:-7]}.scales" in flat:
            base = key[:-7]
            w = mx.dequantize(
                w, flat[f"{base}.scales"], flat[f"{base}.biases"],
                group_size=64, bits=4,
            )
        elif key.endswith(".scales") or key.endswith(".biases"):
            continue
        deq[key] = w.astype(mx.bfloat16)
    mx.eval(deq)

    head = q35.MTPModule(lm.args)
    head.update(_unflatten(deq))
    return head


def _unflatten(flat):
    tree = {}
    for key, value in flat.items():
        parts = key.split(".")
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    def convert(node):
        if all(not isinstance(v, dict) for v in node.values()):
            return node
        out = {}
        for k, v in node.items():
            out[k] = convert(v) if isinstance(v, dict) else v
        if all(k.isdigit() for k in out):
            return [out[str(i)] for i in range(len(out))]
        return out
    return convert(tree)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_examples(tokenizer, corpus_path: str, seq_len: int):
    examples = []
    with open(corpus_path) as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True,
        )
        text = row["text"]
        # Strip think blocks: at decode time the head drafts through whatever
        # the model emits; keeping the raw stream matches the on-policy goal.
        comp_ids = tokenizer.encode(text, add_special_tokens=False)
        ids = list(prompt_ids) + comp_ids
        if len(ids) < 48:
            continue
        ids = ids[: seq_len + 1]
        mask = [0] * min(len(prompt_ids), len(ids)) + [1] * (len(ids) - len(prompt_ids))
        examples.append((ids, mask))
    return examples


# ---------------------------------------------------------------------------
# Training pieces
# ---------------------------------------------------------------------------

def trunk_forward(lm, ids):
    """Pre-norm trunk hidden + teacher logits (bf16, chunked lm_head)."""
    import mlx.core as mx

    toks = mx.array(ids[:-1])[None]  # [1, S-1]
    hidden = lm.model(toks)  # pre-norm [1, S-1, H]
    normed = lm.model.norm(hidden)
    V = 256  # lm_head chunk
    chunks = []
    for s in range(0, normed.shape[1], V):
        chunks.append(lm.lm_head(normed[:, s : s + V]).astype(mx.bfloat16))
    teacher = mx.concatenate(chunks, axis=1)
    mx.eval(hidden, teacher)
    return hidden, teacher


def make_loss(lm, chain: int, lam: float, temp: float, chunk: int):
    embed = lm.model.embed_tokens
    lm_head = lm.lm_head

    def loss_fn(model, hidden, toks, target_mask, teacher):
        lm_ = model.language_model if hasattr(model, "language_model") else model
        mtp = lm_.mtp
        L = hidden.shape[1]  # S-1
        ce_total = mx.array(0.0)
        kl_total = mx.array(0.0)
        n_tok = mx.maximum(target_mask.sum(), 1.0)
        cur_h = hidden
        for k in range(chain):
            Lk = L - 1 - k
            if Lk < 8:
                break
            out = mtp(cur_h[:, :Lk], toks[k : k + Lk][None], embed)  # [1, Lk, H]
            targets = toks[k + 1 : k + 1 + Lk]  # predicts tokens[k+1 ...]
            tmask = target_mask[k + 1 : k + 1 + Lk]
            teach = teacher[:, k + 1 : k + 1 + Lk]
            for s in range(0, Lk, chunk):
                e = min(s + chunk, Lk)
                logits = lm_head(out[:, s:e]).astype(mx.float32) / temp
                tgt = targets[s:e]
                m = tmask[s:e].astype(mx.float32)
                logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                ce = -mx.take_along_axis(logp, tgt[None, :, None], axis=-1).squeeze(-1)
                ce_total = ce_total + (ce * m * temp * temp).sum()
                t_logits = teach[:, s:e].astype(mx.float32) / temp
                t_logp = t_logits - mx.logsumexp(t_logits, axis=-1, keepdims=True)
                t_prob = mx.exp(mx.stop_gradient(t_logp))
                kl = (t_prob * (mx.stop_gradient(t_logp) - logp)).sum(axis=-1)
                kl_total = kl_total + (kl * m * temp * temp).sum()
            cur_h = out
        return (ce_total + lam * kl_total) / n_tok

    return loss_fn


@mx.compile
def _argmax_agree(a, b):
    return (a == b).astype(mx.float32)


def heldout_proxy(lm, held, chain: int, chunk: int = 512):
    """Per-chain-step top-1 agreement with the trunk's own next token (hard
    proxy for p1..p_{chain})."""
    embed = lm.model.embed_tokens
    agree = [0.0] * chain
    count = [0] * chain
    for ids, mask in held:
        hidden, _ = trunk_forward(lm, ids)
        toks = mx.array(ids[1:])  # length S-1
        L = hidden.shape[1]
        cur_h = hidden
        for k in range(chain):
            Lk = L - 1 - k
            if Lk < 8:
                break
            out = lm.mtp(cur_h[:, :Lk], toks[k : k + Lk][None], embed)
            pred = mx.argmax(lm.lm_head(out), axis=-1).squeeze(0)
            targets = toks[k + 1 : k + 1 + Lk]
            m = mx.array(mask[k + 1 : k + 1 + Lk], dtype=mx.bool_)
            agree[k] += mx.sum(mx.where(m, pred == targets, False)).item()
            count[k] += int(m.sum().item())
            cur_h = out
    return [agree[k] / max(count[k], 1) for k in range(chain)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=GRAFTED)
    ap.add_argument("--corpus", default="artifacts/mtpd/corpus.jsonl")
    ap.add_argument("--out", default="artifacts/mtpd/ckpt")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--chain", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--heldout", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--init-from", default=None, help="bf16 mtp safetensors to init from")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading grafted model ...", flush=True)
    model, lm, tokenizer = load_grafted(args.model)
    print("dequantizing MTP head to bf16 ...", flush=True)
    lm.mtp = dequantize_mtp(lm)
    n_params = sum(v.size for _, v in tree_flatten(lm.mtp.parameters()))
    print(f"MTP head trainable params: {n_params/1e6:.1f}M", flush=True)

    if args.init_from:
        lm.mtp.load_weights(args.init_from)

    print("building examples ...", flush=True)
    examples = build_examples(tokenizer, args.corpus, args.seq_len)
    import random

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    held = examples[: args.heldout]
    train = examples[args.heldout :]
    print(f"train={len(train)} heldout={len(held)}", flush=True)

    base_proxy = heldout_proxy(lm, held, args.chain)
    print("baseline heldout top-1 per chain step:",
          [f"{a:.4f}" for a in base_proxy], flush=True)
    if args.eval_only:
        return

    model.freeze()
    lm.mtp.unfreeze()

    steps_per_epoch = len(train)
    total_steps = steps_per_epoch * args.epochs
    # AdamW on a pretrained bf16 head: the first-step sign-like transient at
    # lr 1e-4 was measured to collapse heldout chain top-1 0.79 -> 0.35 in ONE
    # step (zero-grad control clean; lr 1e-6 healthy). Low LR + warmup.
    schedule = optim.join_schedules(
        [optim.linear_schedule(0.0, args.lr, args.warmup),
         optim.cosine_decay(args.lr, max(total_steps - args.warmup, 1), 0.1 * args.lr)],
        [args.warmup],
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)
    loss_fn = make_loss(lm, args.chain, args.lam, args.temp, args.chunk)
    value_and_grad = nn.value_and_grad(model, loss_fn)

    state = [model.state, optimizer.state]
    step = 0
    t0 = time.time()
    log_path = out_dir / "train_log.jsonl"
    best_proxy = sum(base_proxy[:2]) / 2

    for epoch in range(args.epochs):
        rng.shuffle(train)
        for ids, mask in train:
            hidden, teacher = trunk_forward(lm, ids)
            toks = mx.array(ids[1:])
            tmask = mx.array(mask[1:], dtype=mx.float32)
            loss, grads = value_and_grad(model, hidden, toks, tmask, teacher)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            mx.eval(state)
            step += 1
            if step % 10 == 0:
                rec = {"step": step, "loss": loss.item(),
                       "lr": optimizer.learning_rate.item(),
                       "tok_s": (step * args.seq_len) / (time.time() - t0)}
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(rec, flush=True)
            if step % args.eval_every == 0:
                proxy = heldout_proxy(lm, held, args.chain)
                print(f"[step {step}] heldout top-1: {[f'{a:.4f}' for a in proxy]}", flush=True)
                score = sum(proxy[:2]) / 2
                if score > best_proxy:
                    best_proxy = score
                    flat = dict(tree_flatten(lm.mtp.parameters()))
                    mx.save_safetensors(str(out_dir / "best.safetensors"), flat)
            if step % args.ckpt_every == 0:
                flat = dict(tree_flatten(lm.mtp.parameters()))
                mx.save_safetensors(str(out_dir / f"step{step}.safetensors"), flat)

    flat = dict(tree_flatten(lm.mtp.parameters()))
    mx.save_safetensors(str(out_dir / "final.safetensors"), flat)
    final_proxy = heldout_proxy(lm, held, args.chain)
    print("final heldout top-1:", [f"{a:.4f}" for a in final_proxy], flush=True)
    print("baseline was:      ", [f"{a:.4f}" for a in base_proxy], flush=True)
    print(f"best score {best_proxy:.4f}; checkpoints in {out_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
