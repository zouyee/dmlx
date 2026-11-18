#!/usr/bin/env python3
"""pi-adapted client perf bench: TTFT + decode tok/s, thinking on/off.

Usage: pi_perf_bench.py <port> <label> [--long N]
Serial, one server at a time. Prints one line per scenario.
"""
import json, os, sys, time, urllib.request

port = int(sys.argv[1])
label = sys.argv[2]
MODEL = os.environ.get("MODEL", "qwen3.8-pi")

CODE_TASK = (
    "The CLI tool in this repo crashes when the config file has an empty [paths] "
    "section. Find the bug, fix it, and add a regression test. Repo layout: "
    "src/config.py parses the ini, src/cli.py dispatches."
)

def make_prompt(n_tokens_approx):
    # agentic mid-session context: system + growing history of tool results
    base = (
        "You are a coding agent. Prefer small, verifiable steps.\n\n"
        "USER: " + CODE_TASK + "\n"
    )
    if n_tokens_approx <= 200:
        return base
    filler_unit = (
        "TOOL[bash] $ pytest -x\n"
        "FAILED tests/test_config.py::test_empty_paths - KeyError: 'paths'\n"
        "src/config.py:42: section = cfg['paths']\n"
        "ASSISTANT: reading config.py around the KeyError.\n"
        "TOOL[read] src/config.py:30-60\n"
        "def load(path):\n    cfg = parse_ini(path)\n"
        "    section = cfg['paths']\n    return [Path(p) for p in section]\n"
    )
    # ~75 tokens per unit
    reps = max(1, n_tokens_approx // 75)
    return base + "\n" + filler_unit * reps

def run(prompt, max_tokens, temperature, think_note):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n_tok = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                if line.strip() == b"data: [DONE]":
                    break
                try:
                    d = json.loads(line[5:])
                except Exception:
                    continue
                ch = d.get("choices") or [{}]
                delta = ch[0].get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                if piece:
                    n_tok += 1
                    if ttft is None:
                        ttft = time.time() - t0
    except Exception as e:
        print(f"{label} {think_note}: ERROR {e}")
        return None
    wall = time.time() - t0
    decode = (n_tok - 1) / max(1e-9, wall - (ttft or wall))
    print(f"{label:24s} {think_note:14s} prompt~{len(prompt)//4:5d}tok "
          f"ttft={ttft or 0:6.2f}s decode={decode:6.1f} tok/s n={n_tok} wall={wall:6.1f}s")
    return ttft, decode

# scenarios: short agentic turn, mid-session (2k), long-session (8k)
run(make_prompt(100), 256, 0.8, "short")
run(make_prompt(2000), 256, 0.8, "mid-2k")
run(make_prompt(8000), 256, 0.8, "long-8k")
