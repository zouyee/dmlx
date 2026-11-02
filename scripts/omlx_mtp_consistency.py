#!/usr/bin/env python3
"""Validate the grafted omlx MTP checkpoint by self-consistency.

Speculative decoding at temperature 0 must emit exactly the greedy sequence.
So omlx serving the GRAFTED checkpoint (native Lightning MTP active) must
produce output identical to omlx serving the ORIGINAL checkpoint (no MTP
weights, plain decode) — the backbone shards are literally the same files via
symlink, so any divergence means the graft mis-mapped a tensor, applied the
wrong RMSNorm convention, or wired the wrong quantization parameters.

Without this check a graft that silently produced a *worse* drafter would look
like a legitimately slower competitor, which would make our comparison
flattering and wrong.

Runs the two servers serially. Usage:
    ~/venvs/dmlx-omlx/bin/python scripts/omlx_mtp_consistency.py
Exit: 0 identical, 1 divergence, 2 harness failure.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OMLX_PY = os.environ.get("OMLX_PY") or os.path.expanduser("~/venvs/dmlx-omlx/bin/python")
MODEL_DIR = Path.home() / "models"
PLAIN_ID = "Qwen3.8-27B-4bit"
MTP_ID = "Qwen3.8-27B-4bit-mtp"
PORT = int(os.environ.get("PORT", "18111"))
MAX_TOKENS = 96

PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that merges two sorted lists.",
    "Explain in two sentences why the sky is blue.",
]


def kill_all() -> None:
    for pat in ("omlx", "dmlx serve"):
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(3)


def wait_models(timeout: int = 420) -> bool:
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(
                f"http://localhost:{PORT}/v1/models", timeout=2
            ):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return False


def ask(model_id: str, prompt: str) -> str:
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def run(model_id: str, log: Path) -> list[str]:
    kill_all()
    env = dict(os.environ, OMLX_HEADLESS="1")
    with log.open("wb") as lf:
        proc = subprocess.Popen(
            [OMLX_PY, "-m", "omlx.cli", "serve", "--model-dir", str(MODEL_DIR),
             "--port", str(PORT)],
            stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
    try:
        if not wait_models():
            raise RuntimeError(f"omlx never came up; see {log}")
        return [ask(model_id, p) for p in PROMPTS]
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
    print(f"=== omlx, plain checkpoint ({PLAIN_ID}) ===")
    plain = run(PLAIN_ID, Path("/tmp/omlx_cons_plain.log"))
    print(f"=== omlx, grafted checkpoint ({MTP_ID}, native MTP) ===")
    mtp_log = Path("/tmp/omlx_cons_mtp.log")
    mtp = run(MTP_ID, mtp_log)

    # The MTP path must have actually engaged, else this proves nothing.
    evidence = [
        ln for ln in mtp_log.read_text(errors="replace").splitlines()
        if "MTP path activated" in ln or "Speculative backend selected" in ln
    ]
    if not evidence:
        print("harness failure: no MTP activation in the log; the grafted run "
              "was not actually speculative", file=sys.stderr)
        return 2
    for ln in evidence[:3]:
        print("  evidence:", ln.split(" - ")[-1])

    for name, outs in (("plain", plain), ("mtp", mtp)):
        for i, o in enumerate(outs):
            if len(o.strip()) < 2:
                print(f"harness failure: {name}[{i}] empty: {o!r}", file=sys.stderr)
                return 2

    bad = 0
    print()
    for p, a, b in zip(PROMPTS, plain, mtp):
        if a == b:
            print(f"  PASS  {p[:46]!r:<50} ({len(a)} chars)")
        else:
            bad += 1
            # Report the first divergence position: a late split is a numerics
            # tail, an immediate one is a wiring error.
            k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"  FAIL  {p[:46]!r}  first diff at char {k}")
            print(f"        plain: {a[max(0, k - 40):k + 40]!r}")
            print(f"        mtp  : {b[max(0, k - 40):k + 40]!r}")
    print()
    if bad:
        print(f"GRAFT CONSISTENCY FAIL: {bad}/{len(PROMPTS)} diverged")
        return 1
    print(f"GRAFT CONSISTENCY PASS: {len(PROMPTS)}/{len(PROMPTS)} byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
