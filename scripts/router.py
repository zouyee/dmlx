#!/usr/bin/env python3
"""Local-first routing proxy for dmlx (OpenAI-compatible).

Routes /v1/chat/completions between a local dmlx server and a cloud
OpenAI-compatible upstream:

  - small prompt, no streaming-tool gap  -> local dmlx (free, private)
  - large prompt, or tools+stream        -> cloud upstream

Measured on Apple M4 Pro 48GB (see README "What runs on your Mac"):
local prefill is ~110 tok/s, so interactive use wants <= 8k tokens.

Config (env):
  ROUTER_LOCAL_URL     default http://127.0.0.1:8080
  ROUTER_CLOUD_URL     default https://api.openai.com
  ROUTER_CLOUD_KEY     required for cloud routing (else local-only)
  ROUTER_CLOUD_MODEL   model name sent to cloud (default: request model)
  ROUTER_LOCAL_MAX_TOK prompt-token ceiling for local routing (default 8192)
  ROUTER_PORT          listen port (default 8400)

Routing decisions are logged as: route=<local|cloud> reason=... prompt_tokens=~N
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCAL_URL = os.environ.get("ROUTER_LOCAL_URL", "http://127.0.0.1:8080").rstrip("/")
CLOUD_URL = os.environ.get("ROUTER_CLOUD_URL", "https://api.openai.com").rstrip("/")
CLOUD_KEY = os.environ.get("ROUTER_CLOUD_KEY", "")
CLOUD_MODEL = os.environ.get("ROUTER_CLOUD_MODEL", "")
LOCAL_MAX_TOK = int(os.environ.get("ROUTER_LOCAL_MAX_TOK", "8192"))
PORT = int(os.environ.get("ROUTER_PORT", "8400"))

CHAT_PATH = "/v1/chat/completions"


def estimate_tokens(body):
    """Accurate count via dmlx /v1/tokenize; falls back to chars/4."""
    text = "".join(
        part if isinstance(part, str) else " ".join(p.get("text", "") for p in part if isinstance(p, dict))
        for m in body.get("messages", [])
        for part in [m.get("content", "")]
    )
    try:
        payload = json.dumps({"model": body.get("model", "default"), "text": text}).encode()
        r = urllib.request.urlopen(urllib.request.Request(
            LOCAL_URL + "/v1/tokenize", data=payload,
            headers={"Content-Type": "application/json"}), timeout=10)
        j = json.load(r)
        return j.get("count") or j.get("tokens") or len(text) // 4
    except Exception:
        return len(text) // 4


def decide(body, tokens):
    """Return (route, reason)."""
    if not CLOUD_KEY:
        return "local", "no cloud key configured (local-only mode)"
    if tokens > LOCAL_MAX_TOK:
        return "cloud", f"prompt ~{tokens} tok > local ceiling {LOCAL_MAX_TOK} (prefill ~110 tok/s)"
    return "local", f"prompt ~{tokens} tok fits local budget"


def forward(base_url, path, body, headers, stream):
    req = urllib.request.Request(base_url + path, data=body, method="POST")
    for k, v in headers.items():
        if k.lower() not in ("host", "content-length", "authorization", "connection"):
            req.add_header(k, v)
    return urllib.request.urlopen(req, timeout=None if stream else 600)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _proxy(self, target_base, body, extra_headers, t0, route, reason, tokens):
        headers = {"Content-Type": "application/json"}
        headers.update(extra_headers)
        streaming = b'"stream":true' in body.replace(b" ", b"") or b'"stream": true' in body
        upstream = forward(target_base, CHAT_PATH, body, headers, streaming)
        self.send_response(upstream.status)
        ctype = upstream.headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", ctype)
        self.send_header("X-Dmlx-Route", route)
        if streaming and "text/event-stream" in ctype:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = upstream.read1(4096) if hasattr(upstream, "read1") else upstream.read(4096)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        else:
            data = upstream.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        dt = time.time() - t0
        print(f"route={route} reason={reason} prompt_tokens=~{tokens} stream={streaming} {dt:.1f}s", flush=True)

    def do_POST(self):
        if self.path != CHAT_PATH:
            self.send_error(404, "router proxies /v1/chat/completions only")
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        try:
            parsed = json.loads(body)
        except Exception:
            self.send_error(400, "invalid JSON")
            return
        t0 = time.time()
        tokens = estimate_tokens(parsed)
        route, reason = decide(parsed, tokens)
        if route == "cloud":
            if CLOUD_MODEL:
                parsed["model"] = CLOUD_MODEL
                body = json.dumps(parsed).encode()
            try:
                self._proxy(CLOUD_URL, body, {"Authorization": f"Bearer {CLOUD_KEY}"}, t0, route, reason, tokens)
            except Exception as e:
                print(f"cloud upstream failed ({e}), falling back to local", flush=True)
                self._proxy(LOCAL_URL, body, {}, t0, "local-fallback", f"cloud error: {e}", tokens)
        else:
            self._proxy(LOCAL_URL, body, {}, t0, route, reason, tokens)

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            try:
                upstream = urllib.request.urlopen(LOCAL_URL + self.path, timeout=5)
                data = upstream.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_error(502, f"local dmlx unreachable: {e}")
            return
        self.send_error(404)


if __name__ == "__main__":
    print(f"dmlx router on :{PORT}", flush=True)
    print(f"  local: {LOCAL_URL} (ceiling {LOCAL_MAX_TOK} tok)", flush=True)
    print(f"  cloud: {CLOUD_URL} ({'key set' if CLOUD_KEY else 'NO KEY - local-only'})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
