#!/usr/bin/env python3
"""What IS the thinking-mode -7.8%? Each engine's own acceptance on its own text.

By elimination the thinking gap is not engine speed. Every component of the
dmlx verify round is priced and is ours or under 3% of the round; the
quantised matmul is omlx's own kernel (sg8 is a port of vk_group_sg8);
acceptance at matched fixed depth is ours (E[t] 3.92 against 3.08); the
drafter commit semantics are ours; and on an identical controlled task dmlx
decodes faster than omlx at every context length tested (+9.6% at 64 tokens,
+26.8% at 1024).

What remains is that in thinking mode the two engines write DIFFERENT text.
Both are non-deterministic at temperature 0, and speculative acceptance is
measured on whatever each engine itself produced — so an engine whose
free-form reasoning its own drafter happens to predict better records higher
tok/s without being faster.

This reads it off each engine's own instrumentation rather than inferring it.
For every harness prompt, in THINKING mode, it records:

  - tok/s, from the engine's own reported timing
  - the engine's own accepted-draft fraction, parsed from its log
  - how many tokens of reasoning it wrote

If omlx's advantage tracks its own acceptance on its own output while our
speed on identical content is higher, the -7.8% is a property of the text,
not of the engine, and the way to "fix" it is a harness that forces one
shared continuation — not a change to the engine.

Usage:
    python3 scripts/thinking_acceptance.py
"""
from __future__ import annotations

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
OMLX_PY = os.path.expanduser("~/venvs/dmlx-omlx/bin/python")
MODEL = Path.home() / "models/Qwen3.8-27B-4bit"
OMLX_MODEL_DIR = Path.home() / "models"
OMLX_MODEL_ID = "Qwen3.8-27B-4bit-mtp"
PORT = int(os.environ.get("PORT", "18171"))
MAX_TOKENS = 128

PROMPTS = {
    "code": "Write a Python function that merges two sorted lists.",
    "prose": "Explain in one paragraph why unified memory helps local LLM inference.",
    "math": "What is 17 * 23? Show your working briefly.",
    "count": "Count from 1 to 50.",
}

# dmlx: "MTP request N: acceptance 61.4% (54/88 drafts)"
DMLX_ACC = re.compile(r"acceptance ([\d.]+)% \((\d+)/(\d+) drafts\)")
# omlx: "MTP[0] ... tok/cycle=3.08 accept=25/29 (86.2%)"
OMLX_ACC = re.compile(r"tok/cycle=([\d.]+) accept=(\d+)/(\d+) \(([\d.]+)%\)")


def kill_all() -> None:
    for pat in ("dmlx serve", "omlx.cli"):
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(3)


def ask(model_id: str, prompt: str, cap: int) -> tuple[float, int, int]:
    """Returns (wall_s, reasoning_tokens_estimate, completion_tokens)."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cap,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=2400) as r:
        payload = json.load(r)
    dt = time.perf_counter() - t0
    msg = payload["choices"][0]["message"]
    think = msg.get("reasoning_content") or ""
    n = (payload.get("usage") or {}).get("completion_tokens") or cap
    return dt, len(think) // 4, int(n)


def wait_ready(model_id: str, timeout: int = 900) -> bool:
    """Ready means it can complete; omlx answers /health before loading."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ask(model_id, "ready?", 1)
            return True
        except Exception:
            time.sleep(3)
    return False


def run(name: str, cmd: list[str], model_id: str) -> dict:
    kill_all()
    Path("artifacts/logs").mkdir(parents=True, exist_ok=True)
    path = Path(f"artifacts/logs/think_{name}.log")
    log = path.open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(model_id):
            raise RuntimeError(f"{name}: never answered a completion")
        out = {}
        for key, prompt in PROMPTS.items():
            mark = path.stat().st_size
            dt, think_tok, n = ask(
                model_id, f"{prompt} (ref {uuid.uuid4().hex[:6]})", MAX_TOKENS)
            time.sleep(1.0)
            tail = path.read_text(errors="ignore")[mark:]
            acc = None
            m = DMLX_ACC.search(tail)
            if m:
                acc = float(m.group(1))
            else:
                ms = OMLX_ACC.findall(tail)
                if ms:
                    acc = float(ms[-1][3])
            out[key] = {"wall_s": dt, "tokens": n,
                        "reasoning_tok_est": think_tok, "accept_pct": acc,
                        "tok_s": n / dt}
            print(f"  {key:<6} {n:>4} tok in {dt:5.2f}s = {n/dt:5.1f} tok/s  "
                  f"reasoning~{think_tok:>4} tok  accept="
                  f"{'n/a' if acc is None else f'{acc:.1f}%'}", flush=True)
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def main() -> int:
    engines = {
        "dmlx": ([str(HERE / "zig-out/bin/dmlx"), "serve", "--model",
                  str(MODEL), "--speculative-mtp", "7", "--port", str(PORT)],
                 "m"),
        "omlx-mtp": ([OMLX_PY, "-m", "omlx.cli", "serve", "--model-dir",
                      str(OMLX_MODEL_DIR), "--no-cache", "--port", str(PORT)],
                     OMLX_MODEL_ID),
    }
    res = {}
    for name, (cmd, mid) in engines.items():
        print(f"=== {name} (thinking mode)", flush=True)
        res[name] = run(name, cmd, mid)
    kill_all()

    print(f"\n  {'prompt':<7} {'dmlx t/s':>9} {'omlx t/s':>9} "
          f"{'dmlx acc':>9} {'omlx acc':>9} {'dmlx think':>11} {'omlx think':>11}")
    for k in PROMPTS:
        a, b = res["dmlx"][k], res["omlx-mtp"][k]
        fa = "n/a" if a["accept_pct"] is None else f"{a['accept_pct']:.1f}%"
        fb = "n/a" if b["accept_pct"] is None else f"{b['accept_pct']:.1f}%"
        print(f"  {k:<7} {a['tok_s']:>9.1f} {b['tok_s']:>9.1f} {fa:>9} {fb:>9} "
              f"{a['reasoning_tok_est']:>11} {b['reasoning_tok_est']:>11}")
    Path("artifacts/thinking_acceptance.json").write_text(json.dumps(res, indent=1))
    print("\nJSON: artifacts/thinking_acceptance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
