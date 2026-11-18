#!/usr/bin/env python3
"""SSE timestamping client for the client/server tok/s gap measurement.

Usage: gap_client.py <port> <label> <prompt_file|SHORT> <max_tokens> [reps]
Writes <label>.json + prints a summary line per rep.
"""
import json, sys, time, urllib.request

port = int(sys.argv[1]); label = sys.argv[2]
prompt_src = sys.argv[3]; max_tokens = int(sys.argv[4]); reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3

if prompt_src == "SHORT":
    prompt = "请从 1 数到 100，每个数字一行"
else:
    prompt = open(prompt_src).read()

out = []
for rep in range(reps):
    body = json.dumps({
        "model": "qwen3.8-27b-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    chunks = []  # (t_since_t0, n_tokens_so_far)
    n = 0
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            now = time.time() - t0
            if not line.startswith(b"data:"):
                continue
            if line.strip() == b"data: [DONE]":
                break
            try:
                d = json.loads(line[5:])
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                n += 1
                if ttft is None:
                    ttft = now
                chunks.append((now, n))
    wall = time.time() - t0
    ptok = usage["prompt_tokens"] if usage else len(prompt) // 4
    ctok = usage["completion_tokens"] if usage else n
    decode = ctok / max(wall - ttft, 1e-9) if ttft else 0
    itls = [chunks[i][0] - chunks[i-1][0] for i in range(1, len(chunks))]
    itl_med = sorted(itls)[len(itls)//2] if itls else 0
    rec = {"label": label, "rep": rep, "prompt_tokens": ptok, "completion_tokens": ctok,
           "ttft_s": ttft, "wall_s": wall, "client_tps": decode, "itl_median_s": itl_med,
           "first_chunks": chunks[:5]}
    out.append(rec)
    print(f"{label} rep{rep}: prompt={ptok} completion={ctok} ttft={ttft:.3f}s "
          f"wall={wall:.2f}s client={decode:.2f} tok/s itl_med={itl_med*1000:.1f}ms", flush=True)

json.dump(out, open(f"artifacts/qwen38_client_server_gap_20261118/{label}.json", "w"), indent=1)
