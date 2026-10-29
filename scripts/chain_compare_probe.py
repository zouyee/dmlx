#!/usr/bin/env python3
"""Fire the same RAW prompt at both engines via /v1/completions so the chat
template cannot differ (dmlx starts with <think>, omlx skipped it when using
/v1/chat/completions). Usage: python3 scripts/chain_compare_probe.py <port> <out.json>"""
import json, sys, urllib.request

PROMPT = ("<|im_start|>user\nWrite a Python function to merge two sorted lists."
          "<|im_end|>\n<|im_start|>assistant\n")

port, out = int(sys.argv[1]), sys.argv[2]
body = json.dumps({
    "model": sys.argv[3] if len(sys.argv) > 3 else "q",
    "prompt": PROMPT, "stream": False, "temperature": 0, "max_tokens": 96,
}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/completions",
                             data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=180) as r:
    resp = json.load(r)
text = resp["choices"][0].get("text")
if text is None:  # chat-shaped response fallback
    text = resp["choices"][0]["message"]["content"]
json.dump({"text": text, "usage": resp.get("usage")}, open(out, "w"))
print(f"len={len(text)} usage={resp.get('usage')}")
