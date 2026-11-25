#!/usr/bin/env python3
"""Copy-bands AB/BA client: thinking on/off x short/2k/8k, reps per pass.

Same protocol as artifacts/clean_h2h_20261120/clean_h2h.py plus finish_reason
capture. Prints one line per request for the analyzer.
"""
import json, sys, time, urllib.request

port = int(sys.argv[1])
label = sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
temp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8

def make_prompt(n_tokens_approx):
    base = (
        "You are debugging a Python CLI tool that crashes with KeyError when a "
        "config section is missing. Read the transcript and fix it.\n"
    )
    filler = (
        "src/config.py:42: section = cfg['paths']\n"
        "ASSISTANT: reading config.py around the KeyError.\n"
        "TOOL[read] src/config.py:30-60\n"
        "def load(path):\n    cfg = parse_ini(path)\n"
        "    section = cfg['paths']\n    return [Path(p) for p in section]\n"
    )
    reps_n = max(1, n_tokens_approx // 75)
    return base + "\n" + filler * reps_n

def run(prompt, max_tokens, temperature, thinking, tag):
    body = {"model": "m", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n_tok = 0
    finish = "?"
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
                ch = (d.get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                if piece:
                    n_tok += 1
                    if ttft is None:
                        ttft = time.time() - t0
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    except Exception as e:
        print(f"{label} {tag}: ERROR {e}", flush=True)
        return
    wall = time.time() - t0
    decode = (n_tok - 1) / max(1e-9, wall - (ttft or wall))
    print(f"{label:12s} {tag:22s} ttft={ttft or 0:6.2f}s decode={decode:6.1f} "
          f"n={n_tok} finish={finish} wall={wall:6.1f}s", flush=True)

run("hi", 8, 0.0, False, "warmup")
for thinking in (False, True):
    tname = "think-on" if thinking else "think-off"
    for size, tok in (("short", 100), ("mid-2k", 2000), ("long-8k", 8000)):
        for rep in range(1, reps + 1):
            run(make_prompt(tok), 256, temp, thinking, f"{tname}/{size}/r{rep}")
