#!/usr/bin/env python3
"""Do dmlx and omlx-mtp actually generate the SAME text at temperature 0?

Why this is not a detail. compare_engines reports decode tok/s per prompt but
never stores the generated text, so nothing in the comparison has ever checked
that the two engines are doing the same work. On the math prompt they emit 88
chunks (dmlx) against 101 (omlx), so they are demonstrably producing different
sequences. Two reasons that could matter, both of which would make the tok/s
numbers incomparable rather than merely noisy:

  - Sampling. The harness sends temperature=0, but omlx carries persisted
    per-model settings (temperature 1.0, top_p 0.95, top_k 20 for this
    checkpoint) and only honours the request when `force_sampling` is false.
    If omlx is sampling, its acceptance rate and its output are not the
    greedy ones dmlx produces.
  - Chat template / thinking block. A shorter thinking preamble keeps the
    context shorter, and verify cost grows with context, so an engine that
    thinks less decodes faster at equal engine speed.

This runs each engine in turn (never concurrently), sends the same prompts at
temperature 0, and reports token counts, the first divergence offset, and
whether repeated runs of the SAME engine are self-identical. An engine that
disagrees with itself is sampling.

Exit 0 when both engines are self-consistent, 1 when either is not.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CMP_PY = os.path.expanduser("~/venvs/dmlx-cmp/bin/python")
OMLX_PY = os.path.expanduser("~/venvs/dmlx-omlx/bin/python")
MODEL = Path.home() / "models/Qwen3.8-27B-4bit"
OMLX_MODEL_DIR = Path.home() / "models"
OMLX_MODEL_ID = "Qwen3.8-27B-4bit-mtp"
PORT = int(os.environ.get("PORT", "18137"))

PROMPTS = {
    "math": "What is 17 * 23? Show your working briefly.",
    "prose": "Explain in one paragraph why unified memory helps local LLM inference.",
}


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


def ask(model_id: str, prompt: str, max_tokens: int) -> tuple[str, int]:
    """One non-streaming greedy request. Returns (text, completion_tokens).

    Non-streaming is mandatory: dmlx only runs its speculative path off the
    non-streaming route.
    """
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        payload = json.load(r)
    msg = payload["choices"][0]["message"]
    # Thinking models split the answer across reasoning_content and content;
    # concatenate so the comparison covers everything that was decoded.
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    n = (payload.get("usage") or {}).get("completion_tokens") or 0
    return text, n


def run_engine(name: str, cmd: list[str], model_id: str, env: dict,
               max_tokens: int, reps: int) -> dict:
    kill_all()
    log = open(f"artifacts/logs/parity_{name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, **env})
    try:
        if not wait_up():
            raise RuntimeError(f"{name}: server never became healthy")
        out: dict = {}
        for key, prompt in PROMPTS.items():
            runs = [ask(model_id, prompt, max_tokens) for _ in range(reps)]
            out[key] = {
                "tokens": [n for _, n in runs],
                "texts": [t for t, _ in runs],
                "self_identical": len({t for t, _ in runs}) == 1,
            }
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def first_divergence(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--reps", type=int, default=2,
                    help="repeats per prompt; an engine that disagrees with "
                         "itself at temperature 0 is sampling")
    args = ap.parse_args()
    Path("artifacts/logs").mkdir(parents=True, exist_ok=True)

    engines = {
        "dmlx": (
            [str(HERE / "zig-out/bin/dmlx"), "serve", "--model", str(MODEL),
             "--speculative-mtp", "7", "--port", str(PORT)],
            "m", {},
        ),
        "omlx-mtp": (
            [OMLX_PY, "-m", "omlx.cli", "serve", "--model-dir",
             str(OMLX_MODEL_DIR), "--no-cache", "--port", str(PORT)],
            OMLX_MODEL_ID, {},
        ),
    }

    results = {}
    for name, (cmd, model_id, env) in engines.items():
        print(f"=== {name}", flush=True)
        results[name] = run_engine(name, cmd, model_id, env,
                                   args.max_tokens, args.reps)
        for key, r in results[name].items():
            flag = "self-identical" if r["self_identical"] else "*** DIFFERS RUN TO RUN (sampling) ***"
            print(f"  {key:<6} tokens={r['tokens']}  {flag}", flush=True)
    kill_all()

    print("\n=== cross-engine ===")
    ok = True
    for key in PROMPTS:
        a = results["dmlx"][key]
        b = results["omlx-mtp"][key]
        ta, tb = a["texts"][0], b["texts"][0]
        d = first_divergence(ta, tb)
        print(f"  {key:<6} dmlx {a['tokens'][0]:>4} tok / {len(ta):>5} chars   "
              f"omlx {b['tokens'][0]:>4} tok / {len(tb):>5} chars   "
              f"first divergence: {'identical' if d is None else d}")
        if not a["self_identical"] or not b["self_identical"]:
            ok = False
    Path("artifacts/engine_output_parity.json").write_text(
        json.dumps(results, indent=1))
    print("\nJSON: artifacts/engine_output_parity.json")
    if not ok:
        print("FAIL: an engine is not deterministic at temperature 0, so its "
              "tok/s is not comparable to a greedy engine's.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
