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
# dmlx: "ms/round draft=8.3 verify=80.1 accept=0.3 commit=0.1"
DMLX_ROUND = re.compile(r"ms/round draft=([\d.]+) verify=([\d.]+)")
# dmlx: "N=2:28r/31.4ms/tok(E[t]=2.61) N=7:3r/39.3ms/tok(E[t]=3.34)"
DMLX_DEPTH = re.compile(r"N=(\d+):(\d+)r/([\d.]+)ms/tok\(E\[t\]=([\d.]+)\)")


def kill_all() -> None:
    for pat in ("dmlx serve", "omlx.cli"):
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(3)


NOTHINK_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def ask(model_id: str, prompt: str, cap: int,
        extra: dict | None = None) -> tuple[float, int, int]:
    """Returns (wall_s, reasoning_tokens_estimate, completion_tokens)."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cap,
        "temperature": 0,
        "stream": False,
    }
    if extra:
        body.update(extra)
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


def wait_ready(model_id: str, timeout: int = 900,
               extra: dict | None = None) -> bool:
    """Ready means it can complete; omlx answers /health before loading."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ask(model_id, "ready?", 1, extra)
            return True
        except Exception:
            time.sleep(3)
    return False


def run(name: str, cmd: list[str], model_id: str,
        env: dict, extra: dict | None) -> dict:
    kill_all()
    Path("artifacts/logs").mkdir(parents=True, exist_ok=True)
    path = Path(f"artifacts/logs/think_{name}.log")
    log = path.open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, **env})
    try:
        if not wait_ready(model_id, extra=extra):
            raise RuntimeError(f"{name}: never answered a completion")
        out = {}
        for key, prompt in PROMPTS.items():
            mark = path.stat().st_size
            dt, think_tok, n = ask(
                model_id, f"{prompt} (ref {uuid.uuid4().hex[:6]})", MAX_TOKENS,
                extra)
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
            rnd = None
            mr = DMLX_ROUND.search(tail)
            if mr:
                rnd = float(mr.group(1)) + float(mr.group(2))
            et = depth = None
            ds = DMLX_DEPTH.findall(tail)
            if ds:
                tot = sum(int(r) for _, r, _, _ in ds) or 1
                et = sum(int(r) * float(e) for _, r, _, e in ds) / tot
                depth = sum(int(r) * int(nn) for nn, r, _, _ in ds) / tot
            else:
                mo = OMLX_ACC.findall(tail)
                if mo:
                    et = float(mo[-1][0])          # omlx tok/cycle
            out[key] = {"wall_s": dt, "tokens": n,
                        "reasoning_tok_est": think_tok, "accept_pct": acc,
                        "round_ms": rnd, "e_t": et, "mean_depth": depth,
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
    """Four configurations in one session: each engine x {thinking, nothink}.

    Splits the measured asymmetry — removing thinking content speeds dmlx
    +29% but omlx only +16% — into its two factors, since
    tok/s = E[tokens per round] / round_ms. Each engine is read with its OWN
    instrumentation: dmlx reports per-depth E[t] and the draft/verify round
    split, omlx reports tok/cycle and accepted drafts.
    """
    dmlx_cmd = [str(HERE / "zig-out/bin/dmlx"), "serve", "--model",
                str(MODEL), "--speculative-mtp", "7", "--port", str(PORT)]
    omlx_cmd = [OMLX_PY, "-m", "omlx.cli", "serve", "--model-dir",
                str(OMLX_MODEL_DIR), "--no-cache", "--port", str(PORT)]
    configs = [
        ("dmlx-think",   dmlx_cmd, "m",            {}, None),
        ("dmlx-nothink", dmlx_cmd, "m",            {"DMLX_CHAT_EMPTY_THINK": "1"}, None),
        ("omlx-think",   omlx_cmd, OMLX_MODEL_ID,  {}, None),
        ("omlx-nothink", omlx_cmd, OMLX_MODEL_ID,  {}, NOTHINK_BODY),
    ]
    res = {}
    for name, cmd, mid, env, extra in configs:
        print(f"=== {name}", flush=True)
        res[name] = run(name, cmd, mid, env, extra)
    kill_all()

    print(f"\n  {'config':<14}{'tok/s':>7}{'E[t]':>7}{'depth':>7}"
          f"{'round ms':>9}{'accept':>8}")
    for n, r in res.items():
        v = [x for x in r.values() if x["tok_s"]]
        f = lambda k: [x[k] for x in r.values() if x.get(k)]
        g = lambda k: (sum(f(k)) / len(f(k))) if f(k) else float("nan")
        print(f"  {n:<14}{sum(x['tok_s'] for x in v)/len(v):>7.1f}"
              f"{g('e_t'):>7.2f}{g('mean_depth'):>7.2f}{g('round_ms'):>9.1f}"
              f"{g('accept_pct'):>7.1f}%")
    for eng in ("dmlx", "omlx"):
        t = res[f"{eng}-think"]; n = res[f"{eng}-nothink"]
        mt = sum(x["tok_s"] for x in t.values()) / len(t)
        mn = sum(x["tok_s"] for x in n.values()) / len(n)
        print(f"  {eng}: nothink is {100*(mn/mt-1):+.1f}% over thinking")
    Path("artifacts/asymmetry.json").write_text(json.dumps(res, indent=1))
    print("\nJSON: artifacts/asymmetry.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
