#!/usr/bin/env python3
import json, sys, time, urllib.request

port = int(sys.argv[1])
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
body = {
    "model": "m",
    "messages": [{"role": "user", "content": "Write a Python function that reads an INI config section safely and explain it briefly."}],
    "max_tokens": 64,
    "temperature": temp,
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": False},
}
if temp > 0:
    body["top_k"] = 50
    body["top_p"] = 1.0
req = urllib.request.Request(
    f"http://localhost:{port}/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
pieces = []
finish = None
with urllib.request.urlopen(req, timeout=900) as r:
    for line in r:
        if not line.startswith(b"data:"):
            continue
        if line.strip() == b"data: [DONE]":
            break
        data = json.loads(line[5:])
        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
        if piece:
            pieces.append(piece)
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
print(json.dumps({"pieces": len(pieces), "finish": finish, "wall_s": time.time() - t0,
                  "text": "".join(pieces)}, ensure_ascii=False))
