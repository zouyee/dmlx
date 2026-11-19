#!/usr/bin/env python3
"""Realistic-workload decode benchmark: coherent code prompt, streaming,
decode tok/s = (tokens-1)/(wall-TTFT). Usage: decode_realistic.py <base_url> <label> [max_tokens]"""
import json, sys, time, urllib.request

base_url, label = sys.argv[1], sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128

# Realistic context: actual source code from this repo (structured, predictable).
with open("/Users/zouyee/work/code/dmlx/src/server.zig") as f:
    code = f.read()[:6000]
prompt = f"Here is part of a Zig source file:\n\n```zig\n{code}\n```\n\nContinue the file with the next function, keeping the same style."

body = json.dumps({
    "model": "m",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": True,
}).encode()
req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
ttft = None
chunks = 0
with urllib.request.urlopen(req, timeout=900) as r:
    for line in r:
        if not line.startswith(b"data:"):
            continue
        p = line[5:].strip()
        if p == b"[DONE]":
            break
        try:
            chunk = json.loads(p)
        except json.JSONDecodeError:
            continue
        for ch in chunk.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content") or d.get("reasoning_content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
wall = time.perf_counter() - t0
tg = (chunks - 1) / (wall - ttft) if ttft and chunks > 1 else 0
print(f"[{label}] chunks={chunks} ttft={ttft:.2f}s decode_wall={wall-ttft:.2f}s tg={tg:.1f} tok/s")
