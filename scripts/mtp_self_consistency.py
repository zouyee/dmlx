#!/usr/bin/env python3
"""MTP self-consistency probe.

At temperature 0, speculative decoding must produce output IDENTICAL to plain
greedy decoding: the verify step accepts a draft only when it matches the
target's own argmax, so the emitted sequence is by construction the greedy one.
Any divergence means a real bug — a wrong committed GDN state, a bad rollback,
or a numerics deviation large enough to flip an argmax.

This is the real-dimension test that toy-dim unit tests cannot give. It is the
test that actually covers `commitVerifyStates` / the lazy conv-window path
(plan doc §22.5) and the verify_qmm kernels (§numerics).

Runs the two configs SERIALLY (never two serves at once, per project rules).

Usage:
    python3 scripts/mtp_self_consistency.py [--model DIR] [--port N]
                                            [--max-tokens N] [--depth N]
Exit: 0 all prompts identical, 1 divergence, 2 harness failure.
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

PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that merges two sorted lists.",
    "Explain in two sentences why the sky is blue.",
    "Count from 1 to 20.",
    "What is 17 * 23? Show your reasoning.",
]


def wait_ready(port: int, timeout_s: int = 300) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return False


def complete(port: int, prompt: str, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": "m",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def kill_serves() -> None:
    subprocess.run(["pkill", "-f", "dmlx serve"], check=False)
    time.sleep(2)


def collect(cli: Path, model: str, port: int, flags: list[str], max_tokens: int,
            log: Path) -> list[str]:
    kill_serves()
    cmd = [str(cli), "serve", "--model", model, *flags, "--port", str(port)]
    with log.open("wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
    try:
        if not wait_ready(port):
            raise RuntimeError(f"server never became ready; see {log}")
        return [complete(port, p, max_tokens) for p in PROMPTS]
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=30)
        kill_serves()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models/Qwen3.8-27B-4bit"))
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()

    cli = Path(__file__).resolve().parent.parent / "zig-out/bin/dmlx"
    if not cli.exists():
        print(f"missing {cli}", file=sys.stderr)
        return 2

    print("=== plain greedy (no speculation) ===")
    plain = collect(cli, args.model, args.port, [], args.max_tokens,
                    Path("/tmp/selfcons_plain_serve.log"))
    print(f"=== speculative MTP (depth<={args.depth}) ===")
    spec = collect(cli, args.model, args.port,
                   ["--speculative-mtp", str(args.depth)], args.max_tokens,
                   Path("/tmp/selfcons_spec_serve.log"))

    # Harness guard: an earlier shell version of this probe reported PASS on two
    # EMPTY files after a quoting bug swallowed every request. Empty or short
    # completions are a harness failure, never a pass.
    for name, outs in (("plain", plain), ("spec", spec)):
        for i, o in enumerate(outs):
            if len(o.strip()) < 2:
                print(f"harness failure: {name}[{i}] is empty/trivial: {o!r}",
                      file=sys.stderr)
                return 2

    bad = 0
    for p, a, b in zip(PROMPTS, plain, spec):
        if a == b:
            print(f"  PASS  {p[:48]!r:<52} ({len(a)} chars)")
        else:
            bad += 1
            print(f"  FAIL  {p[:48]!r}")
            print(f"        plain: {a!r}")
            print(f"        spec : {b!r}")
    print()
    if bad:
        print(f"SELF-CONSISTENCY FAIL: {bad}/{len(PROMPTS)} prompts diverged")
        return 1
    print(f"SELF-CONSISTENCY PASS: {len(PROMPTS)}/{len(PROMPTS)} prompts byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
