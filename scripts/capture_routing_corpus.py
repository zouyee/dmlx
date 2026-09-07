#!/usr/bin/env python3
"""Capture and validate a privacy-safe exact-Q4 routing corpus.

The capture path is diagnostic-only: one native server, serial requests,
NATIVE_SMELT_N=20, prefetch disabled, and all batch/DSpark flags disabled.
Prompt text and responses are never written to corpus artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compare_batch_oracle import OracleError, compute_fingerprints
from replay_routing_trace import TraceValidationError, replay_trace

SCHEMA_VERSION = 1
CORPUS_KIND = "dmlx-exact-q4-routing-corpus"
TELEMETRY_MARKER = "[NATIVE_TELEMETRY] "
SMELT_N = 20
TRACE_CAPACITY = 65_536
PHASE_NAMES = {0: "prefill", 1: "decode", 2: "verify"}
CACHE_NAMES = {1: "hot", 2: "prefetch", 3: "batch_pool", 4: "cold"}
TRACE_FIELDS = {
    "header": {
        "kind", "schema_version", "model_fingerprint_kind",
        "model_fingerprint_u64", "n_layers", "n_experts", "top_k",
        "cache_capacity", "expert_size_bytes", "record_capacity",
    },
    "request_begin": {"kind", "schema_version", "request_id", "initial_phase"},
    "route": {
        "kind", "schema_version", "request_id", "event_seq", "phase",
        "token_position", "layer", "expert_ids", "weights", "cache_states",
        "route_bytes", "pread_bytes", "io_wait_ns", "short_reads", "io_status",
    },
    "request_end": {
        "kind", "schema_version", "request_id", "records_attempted",
        "records_buffered", "records_written", "records_dropped",
        "invalid_records", "write_errors",
    },
}
FORBIDDEN_TRACE_KEYS = {
    "prompt", "prompt_text", "content", "messages", "request_body",
    "token_ids", "token_text", "response", "completion",
}
TELEMETRY_TOP_FIELDS = {
    "schema_version", "kind", "scope", "request", "batch_path",
    "dspark_verification", "expert_io", "dispatch", "phases",
}
TELEMETRY_REQUEST_FIELDS = {
    "prompt_tokens", "generated_tokens", "request_start_ns",
    "first_token_at_ns", "engine_first_token_latency_ns", "ttft_ns",
    "prefill_ns", "decode_start_ns", "decode_ns", "request_end_ns",
    "request_wall_ns", "itl_ns",
}
TELEMETRY_BATCH_FIELDS = {"scalar_layer_calls", "batched_moe_layers"}
TELEMETRY_DSPARK_FIELDS = {
    "candidate_count", "accepted_count", "verification_batches",
    "verification_ns",
}
TELEMETRY_IO_FIELDS = {
    "measurement_scope", "pread_calls", "bytes_requested", "bytes_read",
    "io_wait_ns", "hot_hits", "prefetch_hits", "batch_pool_hits",
    "cold_misses", "bytes_avoided", "short_reads",
}
TELEMETRY_DISPATCH_FIELDS = {
    "measurement_scope", "dispatches", "command_buffers", "threadgroups",
    "scheduled_lanes", "active_lanes", "max_threads_per_threadgroup",
    "active_lane_utilization_proxy", "workspace_high_water_bytes",
    "attention_gpu_stage_ns", "peak_rss_bytes",
}
TELEMETRY_PHASE_FIELDS = {"name", "layers"}
TELEMETRY_LAYER_FIELDS = {
    "layer", "calls", "unique_experts", "hot_hits", "prefetch_hits",
    "batch_pool_hits", "cold_misses", "pread_calls", "bytes_requested",
    "bytes_read", "short_reads", "io_wait_ns", "bytes_avoided",
    "moe_dispatches", "moe_command_buffers", "threadgroups",
    "scheduled_lanes", "active_lanes", "max_threads_per_threadgroup",
}
BATCH_FLAGS = {
    "NATIVE_BATCH_PREFILL": "0",
    "NATIVE_BATCH_MLA": "0",
    "NATIVE_EXPERT_BUCKETS": "0",
    "NATIVE_BATCH_EXPERT_KERNELS": "0",
    "NATIVE_DSPARK_BATCH_VERIFY": "0",
}


class CorpusError(RuntimeError):
    """Raised when capture evidence is incomplete, unsafe, or inconsistent."""


@dataclass(frozen=True)
class CaptureCase:
    name: str
    shape: str
    max_tokens: int
    prompt: str


def _cases() -> list[CaptureCase]:
    # These are deterministic synthetic workloads, not user data. Their text is
    # used only in the HTTP request and is never persisted in corpus artifacts.
    long_context = " ".join(f"item-{index:02d}" for index in range(32))
    return [
        CaptureCase(
            "short-prefill-short-decode",
            "short_prefill_short_decode",
            5,
            "State two concise properties of prime numbers.",
        ),
        CaptureCase(
            "long-prefill-short-decode",
            "long_prefill_short_decode",
            5,
            "Summarize the pattern in this synthetic sequence: " + long_context,
        ),
        CaptureCase(
            "short-prefill-long-decode",
            "short_prefill_long_decode",
            30,
            "Produce a detailed numbered checklist for validating a software release.",
        ),
    ]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise CorpusError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _write_private(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot read JSON {path}: {error}") from error
    if type(value) is not dict:
        raise CorpusError(f"JSON root must be an object: {path}")
    return value


def _new_phase_layer() -> dict[str, int]:
    return {
        "route_records": 0,
        "hot_hits": 0,
        "prefetch_hits": 0,
        "batch_pool_hits": 0,
        "cold_misses": 0,
        "pread_calls": 0,
        "bytes_requested": 0,
        "bytes_read": 0,
        "bytes_avoided": 0,
        "io_wait_ns": 0,
        "short_reads": 0,
    }


def inspect_trace(path: Path) -> dict[str, Any]:
    """Run V5 validation, enforce a numeric-only whitelist, and summarize phases."""
    replay = replay_trace(path, SMELT_N)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot inspect trace {path}: {error}") from error

    phase_layers: dict[str, dict[str, dict[str, int]]] = {
        name: {} for name in PHASE_NAMES.values()
    }
    request_ids: list[int] = []
    expert_size = replay["trace"]["expert_size_bytes"]
    for line_number, record in enumerate(records, 1):
        if type(record) is not dict:
            raise CorpusError(f"{path}:{line_number}: record must be an object")
        kind = record.get("kind")
        if kind not in TRACE_FIELDS:
            raise CorpusError(f"{path}:{line_number}: unknown kind {kind!r}")
        keys = set(record)
        if keys != TRACE_FIELDS[kind]:
            extra = sorted(keys - TRACE_FIELDS[kind])
            missing = sorted(TRACE_FIELDS[kind] - keys)
            raise CorpusError(
                f"{path}:{line_number}: schema field mismatch; extra={extra}, missing={missing}"
            )
        if keys & FORBIDDEN_TRACE_KEYS:
            raise CorpusError(f"{path}:{line_number}: prompt/content field is forbidden")
        for key, value in record.items():
            if isinstance(value, str) and key not in {
                "kind", "model_fingerprint_kind"
            }:
                raise CorpusError(
                    f"{path}:{line_number}: unexpected string field {key!r}"
                )
        if kind == "request_begin":
            request_ids.append(record["request_id"])
        if kind != "route":
            continue
        phase_name = PHASE_NAMES[record["phase"]]
        layer_key = str(record["layer"])
        metrics = phase_layers[phase_name].setdefault(layer_key, _new_phase_layer())
        metrics["route_records"] += 1
        metrics["bytes_read"] += record["pread_bytes"]
        metrics["io_wait_ns"] += record["io_wait_ns"]
        metrics["short_reads"] += record["short_reads"]
        for cache_state in record["cache_states"]:
            cache_name = CACHE_NAMES[cache_state]
            if cache_name == "hot":
                metrics["hot_hits"] += 1
                metrics["bytes_avoided"] += expert_size
            elif cache_name == "prefetch":
                metrics["prefetch_hits"] += 1
                metrics["bytes_avoided"] += expert_size
            elif cache_name == "batch_pool":
                metrics["batch_pool_hits"] += 1
                metrics["bytes_avoided"] += expert_size
            else:
                metrics["cold_misses"] += 1
                metrics["pread_calls"] += 1
                metrics["bytes_requested"] += expert_size

    phase_totals: dict[str, dict[str, int]] = {}
    for phase_name, layers in phase_layers.items():
        total = _new_phase_layer()
        for metrics in layers.values():
            for key, value in metrics.items():
                total[key] += value
        phase_totals[phase_name] = total

    return {
        "replay": replay,
        "request_ids": request_ids,
        "phase_layers": phase_layers,
        "phase_totals": phase_totals,
        "privacy": {
            "strict_schema_whitelist": True,
            "numeric_routing_metadata_only": True,
            "prompt_or_content_fields": 0,
        },
    }


def _telemetry_phase_layers(telemetry: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {
        name: {} for name in PHASE_NAMES.values()
    }
    phases = telemetry.get("phases")
    if type(phases) is not list:
        raise CorpusError("telemetry.phases must be an array")
    for phase in phases:
        if type(phase) is not dict or phase.get("name") not in result:
            raise CorpusError("telemetry contains an invalid phase")
        layers = phase.get("layers")
        if type(layers) is not list:
            raise CorpusError("telemetry phase layers must be an array")
        for layer in layers:
            if type(layer) is not dict or type(layer.get("layer")) is not int:
                raise CorpusError("telemetry contains an invalid layer")
            key = str(layer["layer"])
            if key in result[phase["name"]]:
                raise CorpusError("telemetry contains a duplicate phase/layer")
            result[phase["name"]][key] = layer
    return result


def _exact_object(value: Any, fields: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CorpusError(f"{context} must be an object")
    if set(value) != fields:
        raise CorpusError(
            f"{context} schema mismatch; "
            f"extra={sorted(set(value) - fields)}, missing={sorted(fields - set(value))}"
        )
    return value


def _require_nonnegative_int(value: Any, context: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) is not int or value < 0:
        raise CorpusError(f"{context} must be a non-negative integer")


def _validate_telemetry_schema(value: Any) -> None:
    telemetry = _exact_object(value, TELEMETRY_TOP_FIELDS, "telemetry")
    if telemetry["schema_version"] != 1:
        raise CorpusError("telemetry.schema_version must be 1")
    if telemetry["kind"] != "dmlx-native-request-telemetry":
        raise CorpusError("unexpected telemetry kind")
    if telemetry["scope"] != "native_engine_after_tokenization":
        raise CorpusError("unexpected telemetry scope")

    request = _exact_object(
        telemetry["request"], TELEMETRY_REQUEST_FIELDS, "telemetry.request"
    )
    for key in ("prompt_tokens", "generated_tokens", "request_start_ns"):
        _require_nonnegative_int(request[key], f"telemetry.request.{key}")
    for key in (
        "first_token_at_ns", "engine_first_token_latency_ns", "ttft_ns",
        "prefill_ns", "decode_start_ns", "decode_ns", "request_end_ns",
        "request_wall_ns",
    ):
        _require_nonnegative_int(
            request[key], f"telemetry.request.{key}", nullable=True
        )
    itl = request["itl_ns"]
    if itl is not None:
        if type(itl) is not list:
            raise CorpusError("telemetry.request.itl_ns must be an array or null")
        for index, item in enumerate(itl):
            _require_nonnegative_int(item, f"telemetry.request.itl_ns[{index}]")

    for object_key, fields in (
        ("batch_path", TELEMETRY_BATCH_FIELDS),
        ("dspark_verification", TELEMETRY_DSPARK_FIELDS),
    ):
        child = _exact_object(telemetry[object_key], fields, f"telemetry.{object_key}")
        for key, item in child.items():
            _require_nonnegative_int(item, f"telemetry.{object_key}.{key}")

    io = _exact_object(telemetry["expert_io"], TELEMETRY_IO_FIELDS, "telemetry.expert_io")
    if io["measurement_scope"] != "critical_path_routed_expert_resolution":
        raise CorpusError("unexpected expert I/O measurement scope")
    for key, item in io.items():
        if key != "measurement_scope":
            _require_nonnegative_int(item, f"telemetry.expert_io.{key}")

    dispatch = _exact_object(
        telemetry["dispatch"], TELEMETRY_DISPATCH_FIELDS, "telemetry.dispatch"
    )
    if dispatch["measurement_scope"] != "routed_moe_only":
        raise CorpusError("unexpected dispatch measurement scope")
    for key, item in dispatch.items():
        if key == "measurement_scope":
            continue
        if key == "active_lane_utilization_proxy":
            if item is not None and (
                type(item) not in (int, float) or isinstance(item, bool) or item < 0
            ):
                raise CorpusError(f"telemetry.dispatch.{key} must be numeric or null")
        else:
            _require_nonnegative_int(
                item,
                f"telemetry.dispatch.{key}",
                nullable=key in {
                    "workspace_high_water_bytes", "attention_gpu_stage_ns",
                    "peak_rss_bytes",
                },
            )

    phases = telemetry["phases"]
    if type(phases) is not list or len(phases) != len(PHASE_NAMES):
        raise CorpusError("telemetry.phases must contain exactly three phases")
    seen_phases: set[str] = set()
    for phase_index, raw_phase in enumerate(phases):
        phase = _exact_object(
            raw_phase, TELEMETRY_PHASE_FIELDS, f"telemetry.phases[{phase_index}]"
        )
        phase_name = phase["name"]
        if phase_name not in PHASE_NAMES.values() or phase_name in seen_phases:
            raise CorpusError("telemetry contains an invalid or duplicate phase")
        seen_phases.add(phase_name)
        if type(phase["layers"]) is not list:
            raise CorpusError(f"telemetry phase {phase_name} layers must be an array")
        seen_layers: set[int] = set()
        for layer_index, raw_layer in enumerate(phase["layers"]):
            layer = _exact_object(
                raw_layer,
                TELEMETRY_LAYER_FIELDS,
                f"telemetry phase {phase_name} layer[{layer_index}]",
            )
            for key, item in layer.items():
                _require_nonnegative_int(
                    item, f"telemetry phase {phase_name} layer[{layer_index}].{key}"
                )
            if layer["layer"] in seen_layers:
                raise CorpusError(f"telemetry phase {phase_name} has a duplicate layer")
            seen_layers.add(layer["layer"])


def reconcile_telemetry(
    trace: dict[str, Any],
    telemetry: dict[str, Any],
    *,
    require_prefetch_disabled: bool = True,
) -> dict[str, Any]:
    """Require exact equality for counters sourced from the same route events."""
    _validate_telemetry_schema(telemetry)
    if len(trace["request_ids"]) != 1:
        raise CorpusError("per-case trace must contain exactly one request")
    io = telemetry.get("expert_io")
    if type(io) is not dict:
        raise CorpusError("telemetry.expert_io must be an object")

    replay = trace["replay"]
    states = replay["records"]["cache_states"]
    expert_size = replay["trace"]["expert_size_bytes"]
    expected = {
        "hot_hits": states["hot"],
        "prefetch_hits": states["prefetch"],
        "batch_pool_hits": states["batch_pool"],
        "cold_misses": states["cold"],
        "pread_calls": states["cold"],
        "bytes_requested": states["cold"] * expert_size,
        "bytes_read": replay["policy"]["read_bytes"],
        "io_wait_ns": sum(
            phase["io_wait_ns"] for phase in trace["phase_totals"].values()
        ),
        "bytes_avoided": (
            states["hot"] + states["prefetch"] + states["batch_pool"]
        ) * expert_size,
        "short_reads": 0,
    }
    for key, value in expected.items():
        if io.get(key) != value:
            raise CorpusError(
                f"telemetry mismatch for expert_io.{key}: expected {value}, got {io.get(key)!r}"
            )

    telemetry_layers = _telemetry_phase_layers(telemetry)
    layer_fields = {
        "route_records": "calls",
        "hot_hits": "hot_hits",
        "prefetch_hits": "prefetch_hits",
        "batch_pool_hits": "batch_pool_hits",
        "cold_misses": "cold_misses",
        "pread_calls": "pread_calls",
        "bytes_requested": "bytes_requested",
        "bytes_read": "bytes_read",
        "bytes_avoided": "bytes_avoided",
        "io_wait_ns": "io_wait_ns",
        "short_reads": "short_reads",
    }
    for phase_name in PHASE_NAMES.values():
        trace_layers = trace["phase_layers"][phase_name]
        observed_layers = telemetry_layers[phase_name]
        if set(trace_layers) != set(observed_layers):
            raise CorpusError(
                f"phase {phase_name} layer coverage mismatch: "
                f"trace={sorted(trace_layers)}, telemetry={sorted(observed_layers)}"
            )
        for layer_key, trace_metrics in trace_layers.items():
            observed = observed_layers[layer_key]
            for trace_key, telemetry_key in layer_fields.items():
                if observed.get(telemetry_key) != trace_metrics[trace_key]:
                    raise CorpusError(
                        f"phase {phase_name} layer {layer_key} {telemetry_key} mismatch: "
                        f"expected {trace_metrics[trace_key]}, got {observed.get(telemetry_key)!r}"
                    )

    if require_prefetch_disabled and (
        states["prefetch"] != 0 or io["prefetch_hits"] != 0
    ):
        raise CorpusError("exact schema-v1 corpus must have prefetch disabled")
    if replay["records"]["records_dropped"] != 0:
        raise CorpusError("trace dropped records")
    return {
        "status": "PASS",
        "scope": "aggregate_and_phase_layer_exact",
        "matched_fields": sorted(expected),
        "matched_phase_layer_fields": sorted(layer_fields.values()),
    }


def _split_trace(master_path: Path, count: int) -> list[list[str]]:
    try:
        lines = master_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CorpusError(f"cannot read master trace: {error}") from error
    if not lines:
        raise CorpusError("master trace is empty")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise CorpusError(f"master trace header is invalid: {error}") from error
    if header.get("kind") != "header":
        raise CorpusError("master trace does not start with a header")

    envelopes: list[list[str]] = []
    active: list[str] | None = None
    for line in lines[1:]:
        try:
            kind = json.loads(line).get("kind")
        except json.JSONDecodeError as error:
            raise CorpusError(f"master trace contains invalid JSON: {error}") from error
        if kind == "request_begin":
            if active is not None:
                raise CorpusError("nested request envelope in master trace")
            active = [json.dumps(header, separators=(",", ":")), line]
        elif active is None:
            raise CorpusError("record outside request envelope in master trace")
        else:
            active.append(line)
            if kind == "request_end":
                envelopes.append(active)
                active = None
    if active is not None:
        raise CorpusError("truncated request envelope in master trace")
    if len(envelopes) != count:
        raise CorpusError(f"expected {count} trace envelopes, got {len(envelopes)}")
    return envelopes


def _extract_telemetry(log_path: Path, count: int) -> list[dict[str, Any]]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise CorpusError(f"cannot read server log: {error}") from error
    result: list[dict[str, Any]] = []
    for line in lines:
        if TELEMETRY_MARKER not in line:
            continue
        payload = line.split(TELEMETRY_MARKER, 1)[1]
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise CorpusError(f"invalid telemetry JSON: {error}") from error
        if type(value) is not dict:
            raise CorpusError("telemetry record must be an object")
        result.append(value)
    if len(result) != count:
        raise CorpusError(f"expected {count} telemetry records, got {len(result)}")
    return result


def _kill_all_servers() -> None:
    subprocess.run(
        ["pkill", "-f", "dmlx serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    time.sleep(1)


def _assert_no_server() -> None:
    result = subprocess.run(
        ["pgrep", "-f", "dmlx serve"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        raise CorpusError(f"dmlx serve process remains: {result.stdout.strip()}")


def _http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise CorpusError(f"HTTP request failed for {url}: {error}") from error
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise CorpusError(f"HTTP response is not JSON for {url}: {error}") from error
    if type(value) is not dict:
        raise CorpusError(f"HTTP response must be an object for {url}")
    return value


def _wait_ready(base_url: str, process: subprocess.Popen[bytes], timeout: int = 240) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CorpusError(f"dmlx serve exited during startup with {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise CorpusError("dmlx serve did not become healthy before timeout")


def _stop_server(base_url: str, process: subprocess.Popen[bytes]) -> None:
    try:
        request = urllib.request.Request(f"{base_url}/shutdown", data=b"", method="POST")
        urllib.request.urlopen(request, timeout=5).read()
    except (urllib.error.URLError, TimeoutError):
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    _kill_all_servers()


def _git_revision(project_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and len(revision) == 40 else "unknown"


def _hardware() -> dict[str, Any]:
    def sysctl(name: str) -> str:
        result = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    memory = sysctl("hw.memsize")
    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(memory) if memory.isdigit() else None,
        "os": "macOS",
    }


def _restore_stats(stats_path: Path, backup: Path | None) -> None:
    if backup is None:
        stats_path.unlink(missing_ok=True)
    else:
        shutil.copy2(backup, stats_path)


def capture(args: argparse.Namespace) -> Path:
    project_dir = Path(__file__).resolve().parents[1]
    binary = args.binary.resolve()
    model = args.model.resolve()
    packed = args.packed_dir.resolve() if args.packed_dir else model / "packed_experts"
    output_dir = args.output_dir.resolve()
    if not binary.is_file():
        raise CorpusError(f"dmlx binary does not exist: {binary}")
    if not model.is_dir() or not packed.is_dir():
        raise CorpusError("model or packed expert directory does not exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(stat.S_IRWXU)

    cases = _cases()
    master_trace = output_dir / ".capture-master.jsonl"
    server_log = output_dir / ".capture-server.log"
    metadata_path = output_dir / "corpus-metadata.json"
    for path in [master_trace, server_log, metadata_path]:
        path.unlink(missing_ok=True)
    for case in cases:
        (output_dir / f"{case.name}.jsonl").unlink(missing_ok=True)
        (output_dir / f"{case.name}.telemetry.json").unlink(missing_ok=True)

    try:
        fingerprints = compute_fingerprints(model, packed)
    except OracleError as error:
        raise CorpusError(f"model fingerprint failed: {error}") from error

    stats_path = packed / ".smelt_routing_stats.bin"
    process: subprocess.Popen[bytes] | None = None
    base_url = f"http://127.0.0.1:{args.port}"
    with tempfile.TemporaryDirectory(prefix="dmlx-routing-stats-") as temp_dir:
        backup = Path(temp_dir) / "routing-stats.bin" if stats_path.is_file() else None
        if backup is not None:
            shutil.copy2(stats_path, backup)
        initial_stats_sha256 = _sha256_file(stats_path) if stats_path.is_file() else None
        try:
            _kill_all_servers()
            _assert_no_server()
            env = os.environ.copy()
            env.update({
                "NATIVE_SMELT_N": str(SMELT_N),
                "NATIVE_ROUTING_TRACE": str(master_trace),
                "NATIVE_ROUTING_TRACE_CAPACITY": str(TRACE_CAPACITY),
                "NATIVE_TELEMETRY": "1",
                "NATIVE_PREFETCH": "0",
                **BATCH_FLAGS,
            })
            command = [
                str(binary), "serve", "--model", str(model), "--port", str(args.port),
                "--native", "--expert-packed-dir", str(packed), "--max-tokens", "30",
                "--temperature", "0",
            ]
            with server_log.open("wb") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=project_dir,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                _wait_ready(base_url, process)
                for case in cases:
                    response = _http_json(
                        f"{base_url}/v1/chat/completions",
                        {
                            "model": "default",
                            "messages": [{"role": "user", "content": case.prompt}],
                            "max_tokens": case.max_tokens,
                            "temperature": 0,
                        },
                        timeout=args.request_timeout,
                    )
                    choices = response.get("choices")
                    if type(choices) is not list or not choices:
                        raise CorpusError(f"case {case.name} returned no choices")
                _stop_server(base_url, process)
                process = None
        finally:
            original_error = sys.exc_info()[1]
            cleanup_errors: list[tuple[str, Exception]] = []
            if process is not None:
                try:
                    _stop_server(base_url, process)
                except Exception as error:  # cleanup must continue independently
                    cleanup_errors.append(("stop server", error))
            for description, action in (
                ("restore routing stats", lambda: _restore_stats(stats_path, backup)),
                ("global server kill", _kill_all_servers),
                ("process exclusion check", _assert_no_server),
            ):
                try:
                    action()
                except Exception as error:  # preserve the first failure
                    cleanup_errors.append((description, error))
            if cleanup_errors:
                details = "; ".join(
                    f"{description}: {error}" for description, error in cleanup_errors
                )
                message = f"capture cleanup failures: {details}"
                if original_error is not None:
                    if hasattr(original_error, "add_note"):
                        original_error.add_note(message)
                else:
                    raise CorpusError(message) from cleanup_errors[0][1]

    restored_stats_sha256 = _sha256_file(stats_path) if stats_path.is_file() else None
    if restored_stats_sha256 != initial_stats_sha256:
        raise CorpusError(
            "routing stats were not restored to their pre-capture content"
        )

    envelopes = _split_trace(master_trace, len(cases))
    telemetry_records = _extract_telemetry(server_log, len(cases))
    case_metadata: list[dict[str, Any]] = []
    try:
        for index, (case, envelope, telemetry) in enumerate(
            zip(cases, envelopes, telemetry_records), 1
        ):
            trace_path = output_dir / f"{case.name}.jsonl"
            telemetry_path = output_dir / f"{case.name}.telemetry.json"
            _write_private(trace_path, "\n".join(envelope) + "\n")
            _write_private(
                telemetry_path,
                json.dumps(telemetry, sort_keys=True, separators=(",", ":")) + "\n",
            )
            trace_summary = inspect_trace(trace_path)
            reconciliation = reconcile_telemetry(trace_summary, telemetry)
            phase_totals = trace_summary["phase_totals"]
            if phase_totals["prefill"]["route_records"] == 0:
                raise CorpusError(f"case {case.name} has no prefill routes")
            if phase_totals["decode"]["route_records"] == 0:
                raise CorpusError(f"case {case.name} has no decode routes")
            request = telemetry.get("request")
            if type(request) is not dict:
                raise CorpusError(f"case {case.name} telemetry has no request metrics")
            case_metadata.append({
                "name": case.name,
                "shape": case.shape,
                "serial_request_index": index,
                "max_tokens": case.max_tokens,
                "prompt_sha256": _sha256_bytes(case.prompt.encode("utf-8")),
                "prompt_char_count": len(case.prompt),
                "prompt_tokens": request.get("prompt_tokens"),
                "generated_tokens": request.get("generated_tokens"),
                "trace_file": trace_path.name,
                "trace_sha256": _sha256_file(trace_path),
                "telemetry_file": telemetry_path.name,
                "telemetry_sha256": _sha256_file(telemetry_path),
                "trace_model_fingerprint_u64": trace_summary["replay"]["trace"]["model_fingerprint_u64"],
                "phase_route_records": {
                    phase: totals["route_records"] for phase, totals in phase_totals.items()
                },
                "route_records": trace_summary["replay"]["records"]["route_records"],
                "expert_routes": trace_summary["replay"]["records"]["expert_routes"],
                "hot_hits": trace_summary["replay"]["policy"]["hits"],
                "cold_or_nonhot_misses": trace_summary["replay"]["policy"]["misses"],
                "read_bytes": trace_summary["replay"]["policy"]["read_bytes"],
                "records_dropped": trace_summary["replay"]["records"]["records_dropped"],
                "privacy": trace_summary["privacy"],
                "telemetry_reconciliation": reconciliation,
                "v5": "PASS",
            })
    finally:
        # The raw server log contains debug token ids and is not corpus evidence.
        server_log.unlink(missing_ok=True)
        master_trace.unlink(missing_ok=True)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": CORPUS_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_revision": _git_revision(project_dir),
        "tooling": {
            "project_root": str(project_dir),
            "capture_script_sha256": _sha256_file(Path(__file__).resolve()),
            "replay_script_sha256": _sha256_file(
                project_dir / "scripts" / "replay_routing_trace.py"
            ),
            "fingerprint_script_sha256": _sha256_file(
                project_dir / "scripts" / "compare_batch_oracle.py"
            ),
        },
        "model": {
            "path": str(model),
            "packed_experts_path": str(packed),
            **fingerprints,
            "routing_stats_sha256": initial_stats_sha256,
        },
        "hardware": _hardware(),
        "configuration": {
            "mode": "native",
            "single_server": True,
            "serial_requests": True,
            "server_max_tokens": 30,
            "NATIVE_SMELT_N": SMELT_N,
            "NATIVE_PREFETCH": 0,
            "NATIVE_ROUTING_TRACE_CAPACITY": TRACE_CAPACITY,
            "NATIVE_TELEMETRY": 1,
            **{key: int(value) for key, value in BATCH_FLAGS.items()},
        },
        "privacy_contract": {
            "prompt_text_persisted": False,
            "response_text_persisted": False,
            "raw_server_log_retained": False,
            "trace_fields": "strict schema-v1 numeric whitelist",
            "prompt_identity": "sha256 and shape/count metadata only",
        },
        "cases": case_metadata,
        "validation": {
            "all_v5_pass": True,
            "all_prefill_and_decode_present": True,
            "all_telemetry_reconciled": True,
            "all_privacy_checks_pass": True,
            "process_cleanup": "PASS",
        },
    }
    _assert_no_server()
    _write_private(
        metadata_path,
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    validate_metadata(metadata_path)
    return metadata_path


def _resolve_evidence(root: Path, relative: Any, context: str) -> Path:
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise CorpusError(f"{context} has an invalid relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise CorpusError(f"{context} escapes corpus directory") from error
    if not candidate.is_file():
        raise CorpusError(f"{context} is missing: {candidate}")
    return candidate


def validate_metadata(metadata_path: Path) -> dict[str, Any]:
    metadata_path = metadata_path.resolve()
    metadata = _load_json(metadata_path)
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("kind") != CORPUS_KIND:
        raise CorpusError("unsupported corpus metadata schema")
    tooling = metadata.get("tooling")
    project_dir = Path(__file__).resolve().parents[1]
    expected_tooling = {
        "project_root": str(project_dir),
        "capture_script_sha256": _sha256_file(Path(__file__).resolve()),
        "replay_script_sha256": _sha256_file(
            project_dir / "scripts" / "replay_routing_trace.py"
        ),
        "fingerprint_script_sha256": _sha256_file(
            project_dir / "scripts" / "compare_batch_oracle.py"
        ),
    }
    if tooling != expected_tooling:
        raise CorpusError("corpus tooling provenance does not match current sources")
    config = metadata.get("configuration")
    if type(config) is not dict:
        raise CorpusError("metadata.configuration must be an object")
    required_config = {
        "mode": "native",
        "single_server": True,
        "serial_requests": True,
        "server_max_tokens": 30,
        "NATIVE_SMELT_N": 20,
        "NATIVE_PREFETCH": 0,
        "NATIVE_TELEMETRY": 1,
        **{key: 0 for key in BATCH_FLAGS},
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise CorpusError(
                f"metadata configuration {key} must be {expected!r}, got {config.get(key)!r}"
            )
    privacy = metadata.get("privacy_contract")
    if type(privacy) is not dict or privacy.get("prompt_text_persisted") is not False:
        raise CorpusError("metadata does not declare the prompt privacy contract")
    cases = metadata.get("cases")
    if type(cases) is not list or len(cases) < 3:
        raise CorpusError("corpus must contain at least three cases")

    seen_hashes: set[str] = set()
    total_prefill = 0
    total_decode = 0
    for case in cases:
        if type(case) is not dict:
            raise CorpusError("corpus case must be an object")
        if "prompt" in case or "content" in case or "response" in case:
            raise CorpusError("corpus metadata contains prompt/response content")
        prompt_hash = case.get("prompt_sha256")
        if type(prompt_hash) is not str or len(prompt_hash) != 64:
            raise CorpusError("corpus case has an invalid prompt hash")
        trace_path = _resolve_evidence(metadata_path.parent, case.get("trace_file"), "trace")
        telemetry_path = _resolve_evidence(
            metadata_path.parent, case.get("telemetry_file"), "telemetry"
        )
        if _sha256_file(trace_path) != case.get("trace_sha256"):
            raise CorpusError(f"trace hash mismatch: {trace_path}")
        if _sha256_file(telemetry_path) != case.get("telemetry_sha256"):
            raise CorpusError(f"telemetry hash mismatch: {telemetry_path}")
        if case["trace_sha256"] in seen_hashes:
            raise CorpusError("corpus contains duplicate traces")
        seen_hashes.add(case["trace_sha256"])
        trace_summary = inspect_trace(trace_path)
        telemetry = _load_json(telemetry_path)
        reconcile_telemetry(trace_summary, telemetry)
        if case.get("v5") != "PASS" or case.get("records_dropped") != 0:
            raise CorpusError("case did not retain a clean V5 result")
        phase_totals = trace_summary["phase_totals"]
        total_prefill += phase_totals["prefill"]["route_records"]
        total_decode += phase_totals["decode"]["route_records"]
        if phase_totals["prefill"]["route_records"] == 0:
            raise CorpusError(f"trace lacks prefill coverage: {trace_path}")
        if phase_totals["decode"]["route_records"] == 0:
            raise CorpusError(f"trace lacks decode coverage: {trace_path}")
    if total_prefill == 0 or total_decode == 0:
        raise CorpusError("corpus does not cover both prefill and decode")
    return {
        "status": "PASS",
        "metadata": str(metadata_path),
        "cases": len(cases),
        "trace_sha256": sorted(seen_hashes),
        "prefill_route_records": total_prefill,
        "decode_route_records": total_decode,
    }


def run_self_test() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    fixture = project_dir / "tests" / "fixtures" / "routing_trace_small.jsonl"
    summary = inspect_trace(fixture)
    replay = summary["replay"]
    states = replay["records"]["cache_states"]
    expert_size = replay["trace"]["expert_size_bytes"]
    phases = []
    for phase_name, layers in summary["phase_layers"].items():
        telemetry_layers = []
        for layer_key, metrics in layers.items():
            telemetry_layers.append({
                "layer": int(layer_key),
                "calls": metrics["route_records"],
                "unique_experts": 0,
                "hot_hits": metrics["hot_hits"],
                "prefetch_hits": metrics["prefetch_hits"],
                "batch_pool_hits": metrics["batch_pool_hits"],
                "cold_misses": metrics["cold_misses"],
                "pread_calls": metrics["pread_calls"],
                "bytes_requested": metrics["bytes_requested"],
                "bytes_read": metrics["bytes_read"],
                "bytes_avoided": metrics["bytes_avoided"],
                "io_wait_ns": metrics["io_wait_ns"],
                "short_reads": metrics["short_reads"],
                "moe_dispatches": 0,
                "moe_command_buffers": 0,
                "threadgroups": 0,
                "scheduled_lanes": 0,
                "active_lanes": 0,
                "max_threads_per_threadgroup": 0,
            })
        phases.append({"name": phase_name, "layers": telemetry_layers})
    telemetry = {
        "schema_version": 1,
        "kind": "dmlx-native-request-telemetry",
        "scope": "native_engine_after_tokenization",
        "request": {
            "prompt_tokens": 1,
            "generated_tokens": 1,
            "request_start_ns": 1,
            "first_token_at_ns": 1,
            "engine_first_token_latency_ns": 0,
            "ttft_ns": None,
            "prefill_ns": None,
            "decode_start_ns": None,
            "decode_ns": None,
            "request_end_ns": 1,
            "request_wall_ns": 0,
            "itl_ns": None,
        },
        "batch_path": {"scalar_layer_calls": 0, "batched_moe_layers": 0},
        "dspark_verification": {
            "candidate_count": 0,
            "accepted_count": 0,
            "verification_batches": 0,
            "verification_ns": 0,
        },
        "expert_io": {
            "measurement_scope": "critical_path_routed_expert_resolution",
            "hot_hits": states["hot"],
            "prefetch_hits": states["prefetch"],
            "batch_pool_hits": states["batch_pool"],
            "cold_misses": states["cold"],
            "pread_calls": states["cold"],
            "bytes_requested": states["cold"] * expert_size,
            "bytes_read": replay["policy"]["read_bytes"],
            "io_wait_ns": sum(
                value["io_wait_ns"] for value in summary["phase_totals"].values()
            ),
            "bytes_avoided": (
                states["hot"] + states["prefetch"] + states["batch_pool"]
            ) * expert_size,
            "short_reads": 0,
        },
        "dispatch": {
            "measurement_scope": "routed_moe_only",
            "dispatches": 0,
            "command_buffers": 0,
            "threadgroups": 0,
            "scheduled_lanes": 0,
            "active_lanes": 0,
            "max_threads_per_threadgroup": 0,
            "active_lane_utilization_proxy": None,
            "workspace_high_water_bytes": None,
            "attention_gpu_stage_ns": None,
            "peak_rss_bytes": None,
        },
        "phases": phases,
    }
    reconcile_telemetry(summary, telemetry, require_prefetch_disabled=False)

    with tempfile.TemporaryDirectory(prefix="dmlx-routing-corpus-test-") as directory:
        unsafe = Path(directory) / "unsafe.jsonl"
        lines = fixture.read_text(encoding="utf-8").splitlines()
        route = json.loads(lines[2])
        route["prompt"] = "must be rejected"
        lines[2] = json.dumps(route, separators=(",", ":"))
        _write_private(unsafe, "\n".join(lines) + "\n")
        try:
            inspect_trace(unsafe)
        except CorpusError:
            pass
        else:
            raise AssertionError("privacy self-test accepted a prompt field")

        bad = json.loads(json.dumps(telemetry))
        bad["expert_io"]["bytes_read"] += 1
        try:
            reconcile_telemetry(summary, bad, require_prefetch_disabled=False)
        except CorpusError:
            pass
        else:
            raise AssertionError("reconciliation self-test accepted mismatched bytes")

        unknown_numeric = json.loads(json.dumps(telemetry))
        unknown_numeric["request"]["input_ids"] = [101, 102, 103]
        try:
            reconcile_telemetry(
                summary, unknown_numeric, require_prefetch_disabled=False
            )
        except CorpusError:
            pass
        else:
            raise AssertionError("telemetry schema accepted unknown numeric token content")

        bad_layer = json.loads(json.dumps(telemetry))
        layer = next(
            layer
            for phase in bad_layer["phases"]
            for layer in phase["layers"]
        )
        layer["bytes_requested"] += 1
        try:
            reconcile_telemetry(summary, bad_layer, require_prefetch_disabled=False)
        except CorpusError:
            pass
        else:
            raise AssertionError("phase/layer reconciliation accepted mismatched bytes")
    print("routing corpus capture self-test: PASS", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Capture or validate a privacy-safe exact-Q4 routing corpus."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/dmlx-routing-traces")
    )
    parser.add_argument(
        "--model", type=Path, default=Path.home() / "models/DeepSeek-V4-Flash-4bit"
    )
    parser.add_argument("--packed-dir", type=Path)
    parser.add_argument("--binary", type=Path, default=project / "zig-out/bin/dmlx")
    parser.add_argument("--port", type=int, default=18961)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--validate-metadata", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    capture_mode = not args.self_test and args.validate_metadata is None
    try:
        if args.self_test:
            run_self_test()
            return 0
        if args.validate_metadata:
            result = validate_metadata(args.validate_metadata)
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0
        metadata_path = capture(args)
        result = validate_metadata(metadata_path)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (CorpusError, TraceValidationError, AssertionError) as error:
        print(f"routing corpus error: {error}", file=sys.stderr)
        _kill_all_servers()
        return 2
    finally:
        if capture_mode:
            output_dir = args.output_dir.resolve()
            (output_dir / ".capture-server.log").unlink(missing_ok=True)
            (output_dir / ".capture-master.jsonl").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
