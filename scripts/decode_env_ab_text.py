#!/usr/bin/env python3
"""Byte-compare plain greedy decode under two env configurations.

The GDN decode-step kernels are the one part of the model that NEITHER
existing correctness gate covers:
  - the unit tests run at toy dims (D=4), where the fused kernels are not
    even eligible (`gdnConvTailWide` requires D==128);
  - the mlx-lm oracle dump is a PREFILL pass (L=5), so it exercises the
    prefill kernels, not the decode step.

So a decode-kernel rewrite has to be validated by generating text and
diffing it. At temperature 0 the two configs must agree byte-for-byte: same
math, and any reduction-order difference that flips an argmax is exactly the
kind of regression this is looking for.

Runs serially (one server at a time). Usage:
    python3 scripts/decode_env_ab_text.py A=1 ...   # baseline is no extra env
    python3 scripts/decode_env_ab_text.py --env DMLX_DECODE_WIDE_TAIL=1
Exit: 0 identical, 1 divergence, 2 harness failure.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "zig-out/bin/dmlx"
MODEL = Path.home() / "models/Qwen3.8-27B-4bit"
PORT = int(os.environ.get("PORT", "18114"))

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
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def collect(extra: dict[str, str], max_tokens: int, log: Path) -> list[str]:
    kill_all()
    env = dict(os.environ, **extra)
    with log.open("wb") as lf:
        proc = subprocess.Popen(
            [str(CLI), "serve", "--model", str(MODEL), "--port", str(PORT)],
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE applied to the variant run (repeatable)")
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()
    extra = {}
    for item in args.env:
        k, _, v = item.partition("=")
        extra[k] = v or "1"
    if not extra:
        print("need at least one --env KEY=VALUE", file=sys.stderr)
        return 2

    print("=== baseline (no extra env) ===")
    base = collect({}, args.max_tokens, Path("/tmp/ab_text_base.log"))
    print(f"=== variant {extra} ===")
    var = collect(extra, args.max_tokens, Path("/tmp/ab_text_var.log"))

    for name, outs in (("baseline", base), ("variant", var)):
        for i, o in enumerate(outs):
            if len(o.strip()) < 2:
                print(f"harness failure: {name}[{i}] empty: {o!r}", file=sys.stderr)
                return 2

    bad = 0
    for p, x, y in zip(PROMPTS, base, var):
        if x == y:
            print(f"  PASS  {p[:44]!r:<48} ({len(x)} chars)")
        else:
            bad += 1
            k = next((i for i, (c1, c2) in enumerate(zip(x, y)) if c1 != c2),
                     min(len(x), len(y)))
            print(f"  FAIL  {p[:44]!r}  first diff at char {k}")
            print(f"        base: {x[max(0, k-40):k+40]!r}")
            print(f"        var : {y[max(0, k-40):k+40]!r}")
    print()
    if bad:
        print(f"DECODE A/B FAIL: {bad}/{len(PROMPTS)} diverged")
        return 1
    print(f"DECODE A/B PASS: {len(PROMPTS)}/{len(PROMPTS)} byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
