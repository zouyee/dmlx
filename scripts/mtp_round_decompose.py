#!/usr/bin/env python3
"""Decompose the dmlx-vs-omlx speculative gap on the SAME prompt.

A tok/s difference is the product of two independent factors:

    tok/s = tokens_per_round / ms_per_round

§26 showed omlx-mtp ahead by 29-43% on predictable content and attributed it to
"deeper chains". That was wrong: omlx's own log shows depth only up to d4, the
same ceiling we ran. So the gap must be in one of the two factors, and they
must be read on the SAME prompt — comparing our code-prompt numbers against
their prose-prompt log is how the depth story got invented.

Both engines report what is needed:
  - omlx: "MTP[n] tokens=N cycles=C tok/cycle=X accept=a/b depth[...] timing[...]"
  - dmlx: `DMLX_MTP_DEBUG=1` prints one "MTP round: n=.. ms=.. k=.." per round,
    plus the depth/cond-acceptance summary.

Serial runs only (one server at a time). Usage:
    python3 scripts/mtp_round_decompose.py [--prompt-kind code|prose|math|count]
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
OMLX_PY = os.environ.get("OMLX_PY") or os.path.expanduser("~/venvs/dmlx-omlx/bin/python")
OMLX_MODEL_DIR = Path.home() / "models"
OMLX_MODEL_ID = "Qwen3.8-27B-4bit-mtp"
PORT = int(os.environ.get("PORT", "18112"))
MAX_TOKENS = 128

PROMPTS = {
    "code": "Write a Python function that merges two sorted lists.",
    "prose": "Explain in two sentences why the sky is blue.",
    "math": "What is 17 * 23? Show your reasoning.",
    "count": "Count from 1 to 50.",
}


def kill_all() -> None:
    for pat in ("dmlx serve", "omlx", "mlx_lm.server", "mlx_vlm.server"):
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(3)


def wait_up(paths=("/health", "/v1/models"), timeout: int = 420) -> bool:
    for _ in range(timeout):
        for p in paths:
            try:
                with urllib.request.urlopen(f"http://localhost:{PORT}{p}", timeout=2):
                    return True
            except (urllib.error.URLError, OSError):
                pass
        time.sleep(1)
    return False


def ask(model_id: str, prompt: str) -> dict:
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return {"wall": time.perf_counter() - t0,
            "tokens": d.get("usage", {}).get("completion_tokens", 0)}


def run_server(cmd: list[str], env: dict, log: Path, model_id: str,
               prompt: str) -> dict:
    kill_all()
    with log.open("wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    try:
        if not wait_up():
            raise RuntimeError(f"server never came up; see {log}")
        ask(model_id, prompt)                 # warm
        return ask(model_id, prompt)          # measured
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


def parse_dmlx(log: Path) -> dict:
    text = log.read_text(errors="replace")
    rounds = [(int(m.group(1)), float(m.group(2)), int(m.group(3)))
              for m in re.finditer(r"MTP round: n=(\d+) ms=([\d.]+).*?k=(\d+)", text)]
    rounds = rounds[len(rounds) // 2:]        # measured request only
    if not rounds:
        return {}
    # tokens per round = 1 bonus/rejection token + k accepted drafts.
    toks = [1 + k for _, _, k in rounds]
    out = {
        "rounds": len(rounds),
        "ms_per_round": st.median(m for _, m, _ in rounds),
        "tok_per_round": st.fmean(toks),
        "depth_used": st.fmean(n for n, _, _ in rounds),
    }
    p = re.findall(r"p(\d+)=([\d.]+)\((\d+)/(\d+)\)", text)
    if p:
        out["p"] = {int(i): (float(v), f"{a}/{r}") for i, v, a, r in p}
    return out


def parse_omlx(log: Path) -> dict:
    lines = [ln for ln in log.read_text(errors="replace").splitlines()
             if "MTP[" in ln and "tok/cycle" in ln]
    if not lines:
        return {}
    ln = lines[-1]                            # measured request
    def grab(pat, cast=float):
        m = re.search(pat, ln)
        return cast(m.group(1)) if m else None
    cycles = grab(r"cycles=(\d+)", int)
    backbone = grab(r"backbone=([\d.]+)ms")
    mtp = grab(r"mtp=([\d.]+)ms")
    cache = grab(r"cache=([\d.]+)ms")
    out = {
        "rounds": cycles,
        "tok_per_round": grab(r"tok/cycle=([\d.]+)"),
        "accept_pct": grab(r"\(([\d.]+)%\)"),
    }
    if cycles:
        for name, v in (("backbone", backbone), ("drafter", mtp), ("cache", cache)):
            if v is not None:
                out[f"{name}_ms_per_round"] = v / cycles
        if None not in (backbone, mtp, cache):
            out["ms_per_round"] = (backbone + mtp + cache) / cycles
    m = re.search(r"depth\[([^\]]+)\]", ln)
    if m:
        out["depth"] = m.group(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-kind", default="code", choices=sorted(PROMPTS))
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()
    prompt = PROMPTS[args.prompt_kind]
    print(f"prompt-kind={args.prompt_kind}  depth-ceiling={args.depth}  "
          f"max_tokens={MAX_TOKENS}\n{prompt!r}\n")

    dlog = Path("/tmp/decomp_dmlx.log")
    dres = run_server(
        [str(CLI), "serve", "--model", str(MODEL),
         "--speculative-mtp", str(args.depth), "--port", str(PORT)],
        dict(os.environ, DMLX_MTP_DEBUG="1"), dlog, "m", prompt)
    d = parse_dmlx(dlog)

    olog = Path("/tmp/decomp_omlx.log")
    ores = run_server(
        [OMLX_PY, "-m", "omlx.cli", "serve", "--model-dir", str(OMLX_MODEL_DIR),
         "--port", str(PORT)],
        dict(os.environ, OMLX_HEADLESS="1"), olog, OMLX_MODEL_ID, prompt)
    o = parse_omlx(olog)

    if not d or not o:
        print(f"harness failure: dmlx parsed={bool(d)} omlx parsed={bool(o)} "
              f"(logs {dlog}, {olog})", file=sys.stderr)
        return 2

    def row(label, dv, ov, fmt="{:.2f}"):
        ds = fmt.format(dv) if isinstance(dv, (int, float)) else str(dv)
        os_ = fmt.format(ov) if isinstance(ov, (int, float)) else str(ov)
        ratio = ""
        if isinstance(dv, (int, float)) and isinstance(ov, (int, float)) and ov:
            ratio = f"{dv / ov:.2f}x"
        print(f"  {label:<26} {ds:>12} {os_:>12}   {ratio:>7}")

    print(f"  {'':<26} {'dmlx':>12} {'omlx-mtp':>12}   {'d/o':>7}")
    row("tokens per round", d["tok_per_round"], o["tok_per_round"])
    row("ms per round", d["ms_per_round"], o.get("ms_per_round", float("nan")))
    row("  of which drafter ms", float("nan"), o.get("drafter_ms_per_round", float("nan")))
    row("  of which backbone ms", float("nan"), o.get("backbone_ms_per_round", float("nan")))
    row("rounds", d["rounds"], o["rounds"], "{:.0f}")
    row("mean depth used", d.get("depth_used", float("nan")), float("nan"))
    print(f"\n  omlx depth[]: {o.get('depth')}")
    if "p" in d:
        print("  dmlx p_i    : " + "  ".join(
            f"p{i}={v:.3f}({c})" for i, (v, c) in sorted(d["p"].items())))
    print(f"\n  request wall: dmlx {dres['wall']:.2f}s / {dres['tokens']} tok"
          f"   omlx {ores['wall']:.2f}s / {ores['tokens']} tok")
    dt = dres["wall"] / max(dres["tokens"], 1) * 1000
    ot = ores["wall"] / max(ores["tokens"], 1) * 1000
    print(f"  ms per emitted token (incl prefill): dmlx {dt:.1f}  omlx {ot:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
