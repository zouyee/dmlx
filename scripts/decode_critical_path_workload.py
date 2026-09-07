#!/usr/bin/env python3
"""Deterministic HTTP workloads for the DSV4 decode critical-path collector."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

KINDS = ("standard", "long", "correctness", "exact_q4")
CORRECTNESS_CASES = (
    ("2+2=", "4"),
    ("The capital of France is", "Paris"),
    ("What temperature does water freeze at in Celsius? Just give the number.", "0"),
    ("Is the Earth round? Reply with only yes or no.", "yes"),
    ("3*3=", "9"),
    ("10-5=", "5"),
    ("What is capital of France?", "Paris"),
)
TIMEOUT_SECONDS = 600


class WorkloadError(RuntimeError):
    """A fail-closed workload or response validation error."""


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def require_integer(value: Any, where: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise WorkloadError(f"{where}: expected integer >= {minimum}")
    return value


def parse_response(value: Any, where: str) -> tuple[str, int, int]:
    if type(value) is not dict:
        raise WorkloadError(f"{where}: response must be an object")
    choices = value.get("choices")
    usage = value.get("usage")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise WorkloadError(f"{where}.choices: expected exactly one choice object")
    message = choices[0].get("message")
    if type(message) is not dict or type(message.get("content")) is not str:
        raise WorkloadError(f"{where}.choices[0].message.content: expected string")
    if type(usage) is not dict:
        raise WorkloadError(f"{where}.usage: expected object")
    prompt_tokens = require_integer(usage.get("prompt_tokens"), where + ".usage.prompt_tokens", 1)
    completion_tokens = require_integer(usage.get("completion_tokens"), where + ".usage.completion_tokens")
    return message["content"], prompt_tokens, completion_tokens


def post(base_url: str, prompt: str, max_tokens: int, where: str) -> tuple[dict[str, Any], str, int, int, int]:
    body = compact_json({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.monotonic_ns()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WorkloadError(f"{where}: HTTP request failed: {exc}") from exc
    elapsed_ns = max(1, time.monotonic_ns() - started_ns)
    if status != 200:
        raise WorkloadError(f"{where}: expected HTTP 200, got {status}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkloadError(f"{where}: response is not valid JSON: {exc}") from exc
    text, prompt_tokens, completion_tokens = parse_response(value, where)
    return value, text, prompt_tokens, completion_tokens, elapsed_ns


def run_standard(base_url: str, label: str) -> dict[str, Any]:
    for index in range(5):
        _, _, _, completion, _ = post(base_url, "Hi", 5, f"{label}.warmup[{index}]")
        if completion != 5:
            raise WorkloadError(f"{label}.warmup[{index}]: expected completion_tokens=5")
    samples = []
    prompt_tokens = []
    for index in range(3):
        _, _, prompt_count, completion, elapsed_ns = post(base_url, "Hi", 5, f"{label}.sample[{index}]")
        if completion != 5:
            raise WorkloadError(f"{label}.sample[{index}]: expected completion_tokens=5")
        samples.append({"max_tokens": 5, "completion_tokens": completion, "elapsed_ns": elapsed_ns})
        prompt_tokens.append(prompt_count)
    throughputs = [5_000_000_000 / sample["elapsed_ns"] for sample in samples]
    median = sorted(throughputs)[1]
    return {"passed": True, "metrics": {"label": label, "warmups": 5, "runs": 3,
        "prompt_tokens": prompt_tokens, "throughputs": throughputs},
        "operand": {"samples": samples, "median_throughput": median}}


def run_long(base_url: str, label: str) -> dict[str, Any]:
    # Protocol v5: three sequential 100-token requests, median throughput,
    # mirroring the standard operand. A single 100-token request is too noisy
    # to resolve the frozen 5% gate (observed cross-round spread -6.5%..+8.8%
    # while the timeline probe's true overhead is ~2-4%).
    samples = []
    prompt_tokens = []
    for index in range(3):
        _, _, prompt_count, completion, elapsed_ns = post(
            base_url, "Count from 1 to 50.", 100, f"{label}.long[{index}]")
        if completion != 100:
            raise WorkloadError(f"{label}.long[{index}]: expected completion_tokens=100, got {completion}")
        samples.append({"max_tokens": 100, "completion_tokens": completion, "elapsed_ns": elapsed_ns})
        prompt_tokens.append(prompt_count)
    throughputs = [100_000_000_000 / sample["elapsed_ns"] for sample in samples]
    median = sorted(throughputs)[1]
    return {"passed": True, "metrics": {"label": label, "runs": 3,
        "prompt_tokens": prompt_tokens, "throughputs": throughputs},
        "operand": {"samples": samples, "median_throughput": median}}


def run_correctness(base_url: str, label: str) -> dict[str, Any]:
    cases = []
    for index, (prompt, expected) in enumerate(CORRECTNESS_CASES):
        _, text, prompt_tokens, completion_tokens, elapsed_ns = post(
            base_url, prompt, 30, f"{label}.correctness[{index}]")
        passed = expected.casefold() in text.casefold()
        cases.append({"index": index + 1, "prompt": prompt, "expected": expected,
                      "text": text, "passed": passed, "prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens, "elapsed_ns": elapsed_ns})
    return {"passed": all(case["passed"] for case in cases),
            "metrics": {"label": label, "passed": sum(case["passed"] for case in cases),
                        "total": len(cases), "cases": cases}, "operand": None}


def run_exact_q4(base_url: str, label: str, request_marker: str) -> dict[str, Any]:
    if not request_marker:
        raise WorkloadError("exact_q4 requires a non-empty --request-marker")
    response, text, prompt_tokens, completion_tokens, elapsed_ns = post(
        base_url, "The capital of France is", 30, f"{label}.exact_q4")
    passed = "paris" in text.casefold()
    return {"passed": passed, "metrics": {"label": label, "text": text,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "elapsed_ns": elapsed_ns}, "operand": None, "response": response,
        "request_marker": request_marker}


def run_kind(kind: str, base_url: str, label: str, request_marker: str) -> dict[str, Any]:
    if kind == "standard":
        return run_standard(base_url, label)
    if kind == "long":
        return run_long(base_url, label)
    if kind == "correctness":
        return run_correctness(base_url, label)
    if kind == "exact_q4":
        return run_exact_q4(base_url, label, request_marker)
    raise WorkloadError(f"unknown kind: {kind}")


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            request = json.loads(self.rfile.read(length))
            prompt = request["messages"][0]["content"]
            max_tokens = request["max_tokens"]
            if self.server.inject_error:  # type: ignore[attr-defined]
                payload = {"choices": [], "usage": {}}
            else:
                answers = {prompt: expected for prompt, expected in CORRECTNESS_CASES}
                text = answers.get(prompt, "Paris" if "France" in prompt else "Hi")
                payload = {"choices": [{"message": {"content": text}}],
                           "usage": {"prompt_tokens": 4, "completion_tokens": max_tokens}}
            encoded = compact_json(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except Exception as exc:  # pragma: no cover - protects the self-test server
            self.send_error(500, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        return


def self_test() -> dict[str, Any]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    server.inject_error = False  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    checks: dict[str, bool] = {}
    try:
        outputs = {kind: run_kind(kind, base_url, "self-test", "marker-原样") for kind in KINDS}
        checks["standard_schema"] = (
            set(outputs["standard"]) == {"passed", "metrics", "operand"}
            and len(outputs["standard"]["operand"]["samples"]) == 3)
        checks["long_schema"] = (
            set(outputs["long"]) == {"passed", "metrics", "operand"}
            and len(outputs["long"]["operand"]["samples"]) == 3
            and all(sample["completion_tokens"] == 100
                    for sample in outputs["long"]["operand"]["samples"]))
        checks["correctness_schema"] = (
            outputs["correctness"]["passed"] is True and outputs["correctness"]["operand"] is None
            and outputs["correctness"]["metrics"]["total"] == 7)
        checks["exact_q4_schema"] = (
            set(outputs["exact_q4"]) == {"passed", "metrics", "operand", "response", "request_marker"}
            and outputs["exact_q4"]["passed"] is True and outputs["exact_q4"]["operand"] is None)
        checks["marker_round_trip"] = outputs["exact_q4"]["request_marker"] == "marker-原样"
        try:
            run_kind("exact_q4", base_url, "self-test", "")
            checks["empty_marker_error"] = False
        except WorkloadError:
            checks["empty_marker_error"] = True
        server.inject_error = True  # type: ignore[attr-defined]
        try:
            run_kind("long", base_url, "self-test", "")
            checks["malformed_response_error"] = False
        except WorkloadError:
            checks["malformed_response_error"] = True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    passed = all(checks.values())
    return {"passed": passed, "checks": checks}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--base-url")
    parser.add_argument("--label")
    parser.add_argument("--request-marker", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [name for name in ("kind", "base_url", "label") if not getattr(args, name)]
        if missing:
            parser.error("required unless --self-test: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = self_test() if args.self_test else run_kind(
            args.kind, args.base_url, args.label, args.request_marker)
        sys.stdout.write(compact_json(result) + "\n")
        return 0 if result.get("passed") is True else 1
    except Exception as exc:
        print(f"decode-critical-path-workload: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
