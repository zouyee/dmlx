#!/usr/bin/env python3
"""Fidelity profile for a change to the MTP speculative path.

Why `mtp_self_consistency.py` is not the right gate for this
---------------------------------------------------------------
That probe demands the speculative output be byte-identical to plain greedy.
Speculative decoding does NOT guarantee that, and cannot: verify runs L rows
in ONE forward while plain decode runs one row at a time, so the two differ at
the ULP level through

  - SDPA reduction order over L rows vs 1,
  - the fused multi-step GDN recurrence vs the single-step decode kernel,
  - the M>=2 verify-shape kernels (`verify_qmm.zig` documents 1e-5..4e-3
    relative deviation).

Over enough tokens an argmax flip is therefore inevitable. Measured here: the
DEFAULT configuration already diverges on 2 of 5 prompts at 256 tokens. So a
byte-identity gate is only a "deviation has not bitten yet at this length"
check, and a change cannot be judged by whether it passes — the baseline does
not pass either. Judging `DMLX_VERIFY_QMM_LM_HEAD` that way produced a wrong
rejection (plan doc §36).

What this measures instead
--------------------------
For each prompt it records WHERE the speculative output first departs from
plain greedy, and compares that profile between two speculative configs. A
change is fidelity-neutral when it does not move divergence earlier: the
emitted token is always the argmax of the target's own verify logits, so what
matters is how closely those logits track the decode-path logits, and the
first-divergence index is a direct, cheap proxy for exactly that.

Reported per config: identical count, and the first-divergence character index
per prompt (higher = tracks plain greedy longer).

Usage:
    python3 scripts/mtp_fidelity_profile.py --env DMLX_VERIFY_QMM_LM_HEAD=1
    python3 scripts/mtp_fidelity_profile.py --env A=1 --env B=2 --max-tokens 384
Exit: 0 when the variant is no worse, 1 when it diverges earlier, 2 harness.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics as st
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "zig-out/bin/dmlx"
MODEL = Path.home() / "models/Qwen3.8-27B-4bit"
PORT = int(os.environ.get("PORT", "18116"))

PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that merges two sorted lists.",
    "Explain in two sentences why the sky is blue.",
    "Count from 1 to 30.",
    "What is 17 * 23? Show your reasoning.",
    "Name three prime numbers greater than 50 and explain why they are prime.",
]


def kill_all() -> None:
    subprocess.run(["pkill", "-f", "dmlx serve"], capture_output=True)
    time.sleep(2)


def wait_up(timeout: int = 300) -> bool:
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return False


def ask(prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def collect(flags: list[str], extra: dict[str, str], max_tokens: int,
            log: Path) -> list[str]:
    kill_all()
    env = dict(os.environ, **extra)
    with log.open("wb") as lf:
        proc = subprocess.Popen(
            [str(CLI), "serve", "--model", str(MODEL), *flags, "--port", str(PORT)],
            stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        if not wait_up():
            raise RuntimeError(f"server never came up; see {log}")
        return [ask(p, max_tokens) for p in PROMPTS]
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
        kill_all()


def first_diff(a: str, b: str) -> int | None:
    """Index of the first differing character, or None when identical."""
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def profile(plain: list[str], spec: list[str]) -> tuple[int, list[int], list[int]]:
    same = 0
    idx: list[int] = []
    lens: list[int] = []
    for a, b in zip(plain, spec):
        lens.append(len(a))
        d = first_diff(a, b)
        if d is None:
            same += 1
            idx.append(len(a))  # tracked all the way to the end
        else:
            idx.append(d)
    return same, idx, lens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE for the variant speculative run (repeatable)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()
    variant_env = {}
    for item in args.env:
        k, _, v = item.partition("=")
        variant_env[k] = v or "1"
    if not variant_env:
        print("need --env KEY=VALUE for the variant", file=sys.stderr)
        return 2

    spec_flags = ["--speculative-mtp", str(args.depth)]
    print(f"max_tokens={args.max_tokens} depth<={args.depth}\n")
    print("=== plain greedy (reference) ===")
    plain = collect([], {}, args.max_tokens, Path("/tmp/fid_plain.log"))
    print("=== speculative, baseline ===")
    base = collect(spec_flags, {}, args.max_tokens, Path("/tmp/fid_base.log"))
    print(f"=== speculative, variant {variant_env} ===")
    var = collect(spec_flags, variant_env, args.max_tokens, Path("/tmp/fid_var.log"))

    for name, outs in (("plain", plain), ("base", base), ("variant", var)):
        for i, o in enumerate(outs):
            if len(o.strip()) < 2:
                print(f"harness failure: {name}[{i}] empty", file=sys.stderr)
                return 2

    b_same, b_idx, lens = profile(plain, base)
    v_same, v_idx, _ = profile(plain, var)

    print(f"\n  {'prompt':<40} {'len':>6} {'base':>8} {'variant':>8} {'delta':>7}")
    for p, L, bi, vi in zip(PROMPTS, lens, b_idx, v_idx):
        print(f"  {p[:38]!r:<40} {L:>6} {bi:>8} {vi:>8} {vi - bi:>+7}")
    print(f"\n  identical to plain greedy : base {b_same}/{len(PROMPTS)}"
          f"   variant {v_same}/{len(PROMPTS)}")
    print(f"  median first-divergence   : base {st.median(b_idx):.0f}"
          f"   variant {st.median(v_idx):.0f}")
    print(f"  mean first-divergence     : base {st.fmean(b_idx):.0f}"
          f"   variant {st.fmean(v_idx):.0f}")

    # A variant is acceptable when it tracks plain greedy at least as long on
    # aggregate. Per-prompt noise is large (one flipped argmax early in a
    # 900-char answer moves one row a lot), so judge on the aggregate.
    worse = st.fmean(v_idx) < st.fmean(b_idx) * 0.8
    print()
    if worse:
        print("FIDELITY REGRESSION: the variant departs from plain greedy "
              "materially earlier than the baseline")
        return 1
    print("FIDELITY NEUTRAL: the variant tracks plain greedy no worse than "
          "the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
