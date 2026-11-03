#!/usr/bin/env python3
"""Find what separates the two math tok/s modes.

The interleaved A/B of the round-1 cost exclusion (artifacts/
ab_round1cost_20261103.json) came back bimodal on the math prompt and only
there: with round-1 cost COUNTED the six samples split cleanly into
{29.0, 29.1, 29.2} and {35.5, 35.5, 35.7}, while with it EXCLUDED all six sat
in the slow cluster. code/prose/count were flat to within 1% in both arms.

A 22% gap between two clean clusters of the same binary on the same prompt is
not noise, it is the depth controller settling into two different fixed
points. This probe runs the math prompt N times per arm with the controller's
own per-request summary enabled, and tabulates tok/s against the depth
histogram and the conditional-acceptance vector, so the fast mode can be
identified by the depth it chooses rather than guessed at.

Each request gets a fresh nonce (defeats prefix reuse) but the server is NOT
restarted between requests inside an arm: the controller's state is
per-request, and restarting would also discard the page cache and confound
the comparison with load time.

Usage:
    python3 scripts/mtp_math_mode_probe.py --requests 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "zig-out/bin/dmlx"
MODEL = Path.home() / "models/Qwen3.8-27B-4bit"
PORT = int(os.environ.get("PORT", "18131"))
PROMPT = "What is 17 * 23? Show your working briefly."

DEPTH_RE = re.compile(r"MTP depth ((?:N=\d+:\d+r/[\d.]+ms/tok\(E\[t\]=[\d.]+\)\s*)+)")
ACC_RE = re.compile(r"MTP cond-acceptance (p1=[\d.]+.*)")
REQ_RE = re.compile(r"MTP request \d+: acceptance ([\d.]+)% \((\d+)/(\d+) drafts\); "
                    r"ms/round draft=([\d.]+) verify=([\d.]+)")


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


def one_request(max_tokens: int) -> tuple[float, int]:
    """Return (decode_tok_s, generated_tokens) for one non-streaming request.

    Non-streaming is mandatory: dmlx only runs the speculative path off the
    streaming fast path, so a streaming client would measure the wrong engine.
    """
    body = json.dumps({
        "model": "m",
        "messages": [{"role": "user",
                      "content": f"{PROMPT} (ref {uuid.uuid4().hex[:8]})"}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        payload = json.load(r)
    dt = time.perf_counter() - t0
    text = payload["choices"][0]["message"]["content"]
    # The server reports generated_tokens in its own log; approximate with the
    # engine's usage block when present, else fall back to a word-ish count.
    n = (payload.get("usage") or {}).get("completion_tokens") or max(1, len(text) // 4)
    return n / dt, n


def run_arm(name: str, env_extra: dict[str, str], requests: int,
            max_tokens: int) -> dict:
    kill_all()
    log_path = Path(f"/tmp/mode_probe_{name}.log")
    env = {**os.environ, **env_extra, "DMLX_MTP_DEBUG": "1"}
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [str(CLI), "serve", "--model", str(MODEL),
             "--speculative-mtp", "7", "--port", str(PORT)],
            stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        if not wait_up():
            raise RuntimeError(f"{name}: server never became healthy")
        # One throwaway request: the first request of a process pays model
        # warmup and drafter priming, which is the very effect under test and
        # must not land in the sample.
        one_request(max_tokens)
        marker = log_path.stat().st_size
        rows = []
        for _ in range(requests):
            tok_s, n = one_request(max_tokens)
            rows.append({"tok_s": tok_s, "tokens": n})
        time.sleep(0.5)
        tail = log_path.read_text()[marker:]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    depths = DEPTH_RE.findall(tail)
    accs = ACC_RE.findall(tail)
    reqs = REQ_RE.findall(tail)
    for i, r in enumerate(rows):
        r["depth"] = depths[i].strip() if i < len(depths) else "?"
        r["acc"] = accs[i].strip() if i < len(accs) else "?"
        r["round_ms"] = (f"draft={reqs[i][3]} verify={reqs[i][4]}"
                         if i < len(reqs) else "?")
    return {"name": name, "env": env_extra, "rows": rows, "log": str(log_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="artifacts/mtp_math_modes.json")
    args = ap.parse_args()

    arms = [
        ("excl", {}),                                      # round-1 cost skipped (current)
        ("count", {"DMLX_MTP_COUNT_ROUND1_COST": "1"}),    # round-1 cost counted (pre-fix)
    ]
    results = []
    for name, env in arms:
        print(f"\n=== arm {name}  env={env or 'baseline'}", flush=True)
        res = run_arm(name, env, args.requests, args.max_tokens)
        results.append(res)
        for i, r in enumerate(res["rows"]):
            print(f"  [{i}] {r['tok_s']:5.1f} tok/s  {r['tokens']:>4}tok  "
                  f"{r['round_ms']}\n       {r['depth']}\n       {r['acc']}",
                  flush=True)
        v = [r["tok_s"] for r in res["rows"]]
        print(f"  -> median {st.median(v):.1f}  min {min(v):.1f}  max {max(v):.1f}")
    kill_all()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"prompt": PROMPT, "arms": results}, indent=1))
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
