#!/usr/bin/env python3
"""Realistic-workload head-to-head: agent-style scenarios, streaming decode.

Scenarios (all temperature 0):
  1. code_gen    — 1.6k-token code context, ask to continue writing code
  2. multi_turn  — same code context + 2-turn history, follow-up question
                   (prefix-cache reuse on turn 2)
  3. prose_sum   — 2k-token prose context, ask for a summary

Usage: realistic_h2h.py <base_url> <model> <label> [--out file.json]
Run once per server, sequentially. Prints per-scenario TTFT and decode tok/s.
"""
import json, sys, time, urllib.request

def read(path, n):
    with open(path) as f:
        return f.read()[:n]

CODE = read("/Users/zouyee/work/code/dmlx/src/server.zig", 6000)
PROSE = read("/Users/zouyee/work/code/dmlx/README.md", 8000)

SCENARIOS = [
    ("code_gen", [
        {"role": "user", "content": f"Here is part of a Zig source file:\n\n```zig\n{CODE}\n```\n\nContinue the file with the next function, keeping the same style. Write at least 100 lines."},
    ]),
    ("multi_turn", [
        {"role": "user", "content": f"Here is part of a Zig source file:\n\n```zig\n{CODE}\n```\n\nWhat does this file do? One paragraph."},
        {"role": "assistant", "content": "This file is the entry point of the DMLX inference server. It wires up model loading, the scheduler, the HTTP accept loop, a background memory guard, and a startup warmup that replays the last stored prompt to rebuild the KV prefix snapshot. Request handling is delegated to submodules for OpenAI-compatible chat completions, streaming, embeddings, and responses."},
        {"role": "user", "content": "Now add a /v1/models endpoint handler. Write the code."},
    ]),
    ("prose_sum", [
        {"role": "user", "content": f"Read this document:\n\n{PROSE}\n\nSummarize the key points as a bullet list."},
    ]),
]


def chat(base_url, model, messages, max_tokens=256):
    body = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
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
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
    wall = time.perf_counter() - t0
    tg = (chunks - 1) / (wall - ttft) if ttft and chunks > 1 else 0.0
    return {"ttft": ttft, "wall": wall, "chunks": chunks, "tg": tg,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens")}


def main():
    base_url, model, label = sys.argv[1], sys.argv[2], sys.argv[3]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    results = {"label": label, "scenarios": []}
    # warmup
    chat(base_url, model, [{"role": "user", "content": "hi"}], 4)
    for name, messages in SCENARIOS:
        # two reps; rep2 also measures prefix-cache benefit
        for rep in (1, 2):
            r = chat(base_url, model, messages)
            r.update({"scenario": name, "rep": rep})
            results["scenarios"].append(r)
            print(f"[{label}] {name} rep{rep}: ttft={r['ttft']:.2f}s tg={r['tg']:.1f} tok/s "
                  f"chunks={r['chunks']} prompt={r['prompt_tokens']}", flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
