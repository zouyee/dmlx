#!/usr/bin/env python3
"""Decode throughput versus CONTEXT LENGTH, for both engines.

The hypothesis this tests, which would explain every number we have:

  nothink   dmlx +8.5% over omlx   (context = short prompt + 128 tokens)
  thinking  dmlx -7.8% under omlx  (context = prompt + a long reasoning
                                    block + 128 tokens)

Same two engines, same quantised-matmul kernel (our sg8 is a port of omlx's
vk_group_sg8), and every component of our verify round is now priced and is
either ours or under 3% — so a component difference cannot explain it. What
differs between the two regimes is not the engine, it is how much context the
attention layers and the GDN states carry while decoding.

If omlx's decode degrades more slowly with context, the gap is context
scaling and it lives somewhere specific: the 16 full-attention layers' KV
handling. If both curves have the same shape, context is not the explanation
and the thinking gap has to be content (the two engines generate different
text, and acceptance is measured on whatever each produced).

Method. Thinking is DISABLED on both engines so the generated length is
controlled and short; context is varied by padding the PROMPT instead. The
padding is unique per request (a nonce plus an index) so no prefix cache can
serve it, and the task is identical at every length, so the only thing moving
is how many tokens the decoder must attend over.

Reports decode tok/s per context length and the degradation from the shortest
to the longest, which is the quantity the hypothesis is about.

Usage:
    python3 scripts/decode_vs_context.py --lengths 64,512,2048,4096
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OMLX_PY = os.path.expanduser("~/venvs/dmlx-omlx/bin/python")
MODEL = Path.home() / "models/Qwen3.8-27B-4bit"
OMLX_MODEL_DIR = Path.home() / "models"
OMLX_MODEL_ID = "Qwen3.8-27B-4bit-mtp"
PORT = int(os.environ.get("PORT", "18141"))

# One sentence of filler, ~12 tokens, repeated to reach a target length. The
# content is deliberately dull: it should not invite the model to comment on
# it, only to carry it in the KV cache.
FILLER = ("Item {i}: the quarterly figure was recorded and filed without "
          "further comment. ")
# The task must reliably produce MORE tokens than the larger cap, or the
# two-point difference collapses (both requests stop early, n_long == n_short,
# and the rate is nan). Counting is also the most predictable thing the model
# can do, which keeps draft acceptance high and similar on both engines so the
# measurement is about context, not about who guesses better.
TASK = ("\n\nIgnoring all of the above, do exactly this and nothing else: "
        "count from 1 to 300, separated by commas.")


def kill_all() -> None:
    for pat in ("dmlx serve", "omlx.cli"):
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(3)


def wait_up(timeout: int = 420) -> bool:
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return False


def build_prompt(target_tokens: int, nonce: str) -> str:
    """Roughly `target_tokens` of unique filler, then the same task."""
    out = [f"Log {nonce}.\n"]
    n = 0
    i = 0
    while n < target_tokens:
        s = FILLER.format(i=f"{nonce}-{i}")
        out.append(s)
        n += 12
        i += 1
    return "".join(out) + TASK


def measure(model_id: str, prompt: str, max_tokens: int,
            extra_body: dict | None = None) -> tuple[float, int, float]:
    """Return (decode_tok_s, tokens, ttft_s) using the two-point method.

    Two requests at different caps share the same prefill, so differencing
    them removes prefill entirely without needing a streaming client — which
    matters because dmlx only runs its speculative path off the non-streaming
    route.
    """
    def one(cap: int) -> tuple[float, int]:
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cap,
            "temperature": 0,
            "stream": False,
        }
        # Thinking is disabled per engine the way compare_engines does it:
        # omlx takes chat_template_kwargs in the body, dmlx takes the
        # DMLX_CHAT_EMPTY_THINK env var and rejects the body field with a 400.
        if extra_body:
            body.update(extra_body)
        req = urllib.request.Request(
            f"http://localhost:{PORT}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=2400) as r:
            payload = json.load(r)
        dt = time.perf_counter() - t0
        n = (payload.get("usage") or {}).get("completion_tokens") or cap
        return dt, n

    t_short, n_short = one(8)
    t_long, n_long = one(max_tokens)
    d_tok = n_long - n_short
    d_t = t_long - t_short
    rate = d_tok / d_t if d_t > 0 and d_tok > 0 else float("nan")
    return rate, n_long, t_short


def run_engine(name: str, cmd: list[str], model_id: str, lengths: list[int],
               max_tokens: int, reps: int, env: dict,
               extra_body: dict | None) -> dict:
    kill_all()
    Path("artifacts/logs").mkdir(parents=True, exist_ok=True)
    log = open(f"artifacts/logs/ctx_{name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, **env})
    try:
        if not wait_up():
            raise RuntimeError(f"{name}: server never became healthy")
        # Warm the weights once so the first length does not pay load cost.
        measure(model_id, build_prompt(64, "warm"), 32, extra_body)
        out = {}
        for L in lengths:
            rates = []
            for r in range(reps):
                nonce = uuid.uuid4().hex[:8]
                rate, _, ttft = measure(
                    model_id, build_prompt(L, nonce), max_tokens, extra_body)
                rates.append(rate)
                print(f"    ctx~{L:>5}  rep{r}  {rate:6.1f} tok/s  "
                      f"(ttft {ttft:.2f}s)", flush=True)
            out[L] = st.median(rates)
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="64,512,2048")
    ap.add_argument("--max-tokens", type=int, default=72)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="artifacts/decode_vs_context.json")
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    engines = {
        "dmlx": ([str(HERE / "zig-out/bin/dmlx"), "serve", "--model",
                  str(MODEL), "--speculative-mtp", "7", "--port", str(PORT)],
                 "m", {"DMLX_CHAT_EMPTY_THINK": "1"}, None),
        "omlx-mtp": ([OMLX_PY, "-m", "omlx.cli", "serve", "--model-dir",
                      str(OMLX_MODEL_DIR), "--no-cache", "--port", str(PORT)],
                     OMLX_MODEL_ID, {},
                     {"chat_template_kwargs": {"enable_thinking": False}}),
    }

    res = {}
    for name, (cmd, model_id, env, extra) in engines.items():
        print(f"=== {name}", flush=True)
        res[name] = run_engine(name, cmd, model_id, lengths,
                               args.max_tokens, args.reps, env, extra)
    kill_all()

    print(f"\n  {'ctx tokens':>11}  {'dmlx':>8}  {'omlx':>8}  {'dmlx/omlx':>10}")
    for L in lengths:
        a, b = res["dmlx"][L], res["omlx-mtp"][L]
        print(f"  {L:>11}  {a:>8.1f}  {b:>8.1f}  {100*(a/b-1):>+9.1f}%")
    for name in res:
        v = res[name]
        lo, hi = v[lengths[0]], v[lengths[-1]]
        print(f"  {name:<9} degradation {lengths[0]} -> {lengths[-1]}: "
              f"{100*(hi/lo-1):+.1f}%")
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\nJSON: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
