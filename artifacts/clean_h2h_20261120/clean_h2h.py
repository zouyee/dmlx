#!/usr/bin/env python3
"""Clean H2H: thinking on/off x short/mid/long, 2 runs each. Server-side
tok/s read from engine logs; this script reports client-side."""
import json, sys, time, urllib.request

port = int(sys.argv[1])
label = sys.argv[2]

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
    reps = max(1, n_tokens_approx // 75)
    return base + "\n" + filler * reps

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
        print(f"{label} {tag}: ERROR {e}", flush=True)
        return
    wall = time.time() - t0
    decode = (n_tok - 1) / max(1e-9, wall - (ttft or wall))
    print(f"{label:10s} {tag:18s} ttft={ttft or 0:6.2f}s decode={decode:6.1f} tok/s n={n_tok} wall={wall:6.1f}s", flush=True)

# warmup
run("hi", 8, 0.0, False, "warmup")
for thinking in (False, True):
    tname = "think-on" if thinking else "think-off"
    for size, tok in (("short", 100), ("mid-2k", 2000), ("long-8k", 8000)):
        for rep in (1, 2):
            run(make_prompt(tok), 256, 0.8, thinking, f"{tname}/{size}/r{rep}")
