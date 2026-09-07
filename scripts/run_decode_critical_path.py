#!/usr/bin/env python3
"""Source-bound collector and offline validator for DSV4 decode evidence.

Collection is opt-in only via ``--collect``. Validation, self-test, plan, and
``--dry-run`` never build, kill, or start a server.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, NoReturn

SCHEMA_VERSION = 1
LAYERS = 43
LAYER_STAGES = (
    "attention_cb", "dsa_cpu", "cb2cmd2_cb", "io_exposed_wait", "shared_expert_cb",
    "routed_moe_cb", "cb3_cb",
)
TOKEN_STAGES = ("token_forward", "logits_cb", "token_sample")
STAGES = ("token_forward",) + LAYER_STAGES + TOKEN_STAGES[1:]
EVENT_FIELDS = frozenset((
    "schema_version", "token", "layer", "stage", "status", "begin_ns",
    "encode_done_ns", "pre_commit_ns", "post_commit_ns", "post_wait_ns",
    "end_ns",
))
CAPTURE_FIELDS = frozenset(("abi_version", "valid", "count", "events"))
CAPTURE_EVENT_FIELDS = frozenset((
    "token", "layer", "stage", "status", "begin_ns", "encode_done_ns",
    "pre_commit_ns", "post_commit_ns", "post_wait_ns", "end_ns",
))
TIME_FIELDS = (
    "begin_ns", "encode_done_ns", "pre_commit_ns", "post_commit_ns",
    "post_wait_ns", "end_ns",
)
EXPECTED_EVENTS_PER_TOKEN = 1 + LAYERS * len(LAYER_STAGES) + 2
MAX_UNCLASSIFIED_RATIO = 0.05
# Non-gated warmup process (protocol v3): runs once before off1 with the
# identical server shape so the first gated child no longer executes on a cold
# page cache. Its evidence is bound but never enters runs, the gate, or the
# diagnosis count.
WARMUP_LABEL = "warmup0"
RUN_ORDER = ("off1", "on1", "off2", "on2")
DIAGNOSIS_ORDER = ("diagnosis1", "diagnosis2", "diagnosis3")
SIDECAR_ORDER = ("exact_q4_sidecar1", "exact_q4_sidecar2", "exact_q4_sidecar3")
PAIR_LABELS = tuple(f"{diagnosis}_{sidecar}" for diagnosis, sidecar in zip(DIAGNOSIS_ORDER, SIDECAR_ORDER))
PAIRED_DIAGNOSIS = dict(zip(SIDECAR_ORDER, DIAGNOSIS_ORDER))
MODEL = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit"
PACKED = MODEL + "/packed_experts"
MAX_RSS_EXCLUSIVE = 47_244_640_256
# Memory sampling (protocol v4): each sample records both the total resident
# size (rss_bytes) and the Mach phys_footprint (footprint_bytes). Total RSS is
# unstable on macOS because file-mapped page cache is attributed to whichever
# process touches it first, so the gate uses only footprint_bytes; rss_bytes
# is retained purely as provenance.
RUSAGE_INFO_V4 = 4
_LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
SHA256_FIELDS = frozenset((
    "binary_sha256", "dirty_workspace_sha256", "protocol_sha256",
    "collector_sha256", "semantic_sha256",
))
PROVENANCE_FIELDS = frozenset((
    "binary_sha256", "git_head", "dirty_workspace_sha256", "model_identity",
    "tokenizer_identity", "packed_manifest_identity", "protocol_sha256",
    "collector_sha256", "routing_seed_identity", "pid", "executable_identity",
    "environment", "argv", "rss_series", "lifecycle",
))
IDENTITY_FIELDS = frozenset(("path", "sha256"))
LIFECYCLE_FIELDS = frozenset((
    "cleanup_before", "verify_none_before", "started", "stopped", "waited",
    "verify_none_after", "active_publishers_final", "overflow",
    "incomplete_commands", "final_status", "started_ns", "stopped_ns", "verified_none_ns",
))
RUN_FIELDS = frozenset(("label", "timeline_enabled", "provenance", "timeline_sha256"))
ARTIFACT_FIELDS = frozenset((
    "schema_version", "kind", "protocol", "runs", "diagnostics",
    "overhead_gate_passed", "semantic_sha256",
))


class ValidationError(ValueError):
    pass


class ChildCollectionError(RuntimeError):
    def __init__(self, message: str, record: dict[str, Any]):
        super().__init__(message)
        self.record = record


def fail(message: str) -> NoReturn:
    raise ValidationError(message)


def closed(obj: Any, fields: frozenset[str], where: str) -> dict[str, Any]:
    if type(obj) is not dict:
        fail(f"{where}: expected object")
    missing = fields - obj.keys()
    extra = obj.keys() - fields
    if missing or extra:
        fail(f"{where}: field mismatch missing={sorted(missing)} unknown={sorted(extra)}")
    return obj


def integer(value: Any, where: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{where}: expected integer >= {minimum}")
    return value


def boolean(value: Any, where: str) -> bool:
    if type(value) is not bool:
        fail(f"{where}: expected boolean")
    return value


def string(value: Any, where: str, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        fail(f"{where}: expected {'non-empty ' if nonempty else ''}string")
    return value


def sha256(value: Any, where: str) -> str:
    text = string(value, where)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        fail(f"{where}: expected lowercase SHA-256")
    return text


def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"JSON: duplicate key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"{path}: cannot read strict JSON: {exc}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def semantic_hash(artifact: dict[str, Any]) -> str:
    payload = dict(artifact)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def normalize_capture(value: Any, line: int) -> list[dict[str, Any]]:
    capture = closed(value, CAPTURE_FIELDS, f"capture line {line}")
    if integer(capture["abi_version"], f"capture line {line}.abi_version") != 2:
        fail(f"capture line {line}: unsupported abi_version")
    if integer(capture["valid"], f"capture line {line}.valid") != 1:
        fail(f"capture line {line}: invalid producer snapshot")
    count = integer(capture["count"], f"capture line {line}.count")
    raw_events = capture["events"]
    if type(raw_events) is not list or len(raw_events) != count:
        fail(f"capture line {line}: count/events mismatch")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_events):
        event = closed(raw, CAPTURE_EVENT_FIELDS, f"capture line {line}.events[{index}]")
        stage_id = integer(event["stage"], f"capture line {line}.events[{index}].stage")
        if stage_id >= len(STAGES):
            fail(f"capture line {line}.events[{index}]: unknown stage {stage_id}")
        status = integer(event["status"], f"capture line {line}.events[{index}].status")
        layer = integer(event["layer"], f"capture line {line}.events[{index}].layer")
        converted = dict(event)
        converted.update(
            schema_version=SCHEMA_VERSION,
            stage=STAGES[stage_id],
            status="completed" if status == 1 else f"invalid:{status}",
            layer=None if layer == 65535 else layer,
        )
        normalized.append(validate_event(converted, line))
    return normalized


def validate_event(value: Any, line: int) -> dict[str, Any]:
    event = closed(value, EVENT_FIELDS, f"timeline line {line}")
    if integer(event["schema_version"], f"line {line}.schema_version") != SCHEMA_VERSION:
        fail(f"timeline line {line}: unsupported schema_version")
    token = integer(event["token"], f"line {line}.token")
    del token
    stage = string(event["stage"], f"line {line}.stage")
    if stage not in STAGES:
        fail(f"timeline line {line}: unknown stage {stage!r}")
    if event["status"] != "completed":
        fail(f"timeline line {line}: status must be 'completed'")
    layer = event["layer"]
    if stage in LAYER_STAGES:
        if type(layer) is not int or isinstance(layer, bool) or not 0 <= layer < LAYERS:
            fail(f"timeline line {line}: layer stage needs integer layer 0..42")
    elif layer is not None:
        fail(f"timeline line {line}: token-level stage requires null layer")
    times = [integer(event[name], f"line {line}.{name}") for name in TIME_FIELDS]
    if any(left > right for left, right in zip(times, times[1:])):
        fail(f"timeline line {line}: non-monotonic timestamps")
    return event


def interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def validate_token(token: int, events: list[dict[str, Any]]) -> dict[str, Any]:
    if len(events) != EXPECTED_EVENTS_PER_TOKEN:
        fail(f"token {token}: expected {EXPECTED_EVENTS_PER_TOKEN} events, got {len(events)}")
    seen: set[tuple[str, Any]] = set()
    for event in events:
        key = (event["stage"], event["layer"])
        if key in seen:
            fail(f"token {token}: duplicate stage/layer {key}")
        seen.add(key)
    expected = {(stage, layer) for stage in LAYER_STAGES for layer in range(LAYERS)}
    expected |= {(stage, None) for stage in TOKEN_STAGES}
    missing = expected - seen
    extra = seen - expected
    if missing or extra:
        fail(f"token {token}: cardinality mismatch missing={sorted(missing, key=str)} extra={sorted(extra, key=str)}")
    envelope = next(event for event in events if event["stage"] == "token_forward")
    begin, end = envelope["begin_ns"], envelope["end_ns"]
    duration = end - begin
    if duration <= 0:
        fail(f"token {token}: TOKEN_FORWARD duration must be positive")
    clipped: list[tuple[int, int]] = []
    for event in events:
        if event is envelope:
            continue
        child_begin, child_end = event["begin_ns"], event["end_ns"]
        # Containment is strict validation; clipping is still used for accounting.
        if child_begin < begin or child_end > end:
            fail(f"token {token}: child {event['stage']}/{event['layer']} outside TOKEN_FORWARD")
        clipped_begin, clipped_end = max(begin, child_begin), min(end, child_end)
        if clipped_end < clipped_begin:
            fail(f"token {token}: negative clipped interval")
        clipped.append((clipped_begin, clipped_end))
    child_union = interval_union_ns(clipped)
    residual = duration - child_union
    if residual < 0:
        fail(f"token {token}: negative unclassified_ns")
    ratio = residual / duration
    if ratio > MAX_UNCLASSIFIED_RATIO:
        fail(f"token {token}: unclassified ratio {ratio:.9f} exceeds 0.05")
    return {"token": token, "token_forward_ns": duration,
            "child_union_ns": child_union, "unclassified_ns": residual,
            "unclassified_ratio": ratio}


def validate_timeline(path: Path) -> dict[str, Any]:
    # Each capture line is one request's producer snapshot whose token ordinals
    # restart at zero; contiguity is therefore validated per segment, never
    # across requests. Consecutive plain event lines form a single segment.
    segments: list[list[dict[str, Any]]] = []
    segment_plain: list[bool] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_no, raw in enumerate(stream, 1):
                if not raw.strip():
                    fail(f"timeline line {line_no}: blank lines forbidden")
                try:
                    value = json.loads(raw, object_pairs_hook=no_duplicate_pairs)
                except json.JSONDecodeError as exc:
                    fail(f"timeline line {line_no}: invalid JSON: {exc}")
                if type(value) is dict and set(value) == CAPTURE_FIELDS:
                    segments.append(normalize_capture(value, line_no))
                    segment_plain.append(False)
                else:
                    event = validate_event(value, line_no)
                    if segments and segment_plain[-1]:
                        segments[-1].append(event)
                    else:
                        segments.append([event])
                        segment_plain.append(True)
    except (OSError, UnicodeError) as exc:
        fail(f"{path}: cannot read timeline: {exc}")
    accounting: list[dict[str, Any]] = []
    total = 0
    for events in segments:
        if not events:
            continue
        total += len(events)
        grouped: dict[int, list[dict[str, Any]]] = {}
        token_order: list[int] = []
        previous_token: int | None = None
        closed_tokens: set[int] = set()
        for event in events:
            token = event["token"]
            if previous_token is not None and token != previous_token:
                closed_tokens.add(previous_token)
                if token in closed_tokens:
                    fail(f"timeline: token {token} is non-contiguous")
            if token not in grouped:
                token_order.append(token)
                grouped[token] = []
            grouped[token].append(event)
            previous_token = token
        first = token_order[0]
        if token_order != list(range(first, first + len(token_order))):
            fail(f"timeline: non-continuous token ordinals {token_order}")
        accounting.extend(validate_token(token, grouped[token]) for token in token_order)
    if total == 0:
        fail("timeline: empty input")
    return {"valid": True, "events": total, "retained_tokens": len(accounting),
            "coverage": 1.0, "accounting": accounting}


def validate_identity(value: Any, where: str) -> None:
    identity = closed(value, IDENTITY_FIELDS, where)
    string(identity["path"], where + ".path")
    sha256(identity["sha256"], where + ".sha256")


def validate_environment(value: Any, enabled: bool, where: str) -> None:
    if type(value) is not dict or any(type(k) is not str or type(v) is not str for k, v in value.items()):
        fail(f"{where}: expected string-to-string object")
    required = {"NATIVE_SMELT_N": "20"}
    if enabled:
        required["DSV4_DECODE_TIMELINE_ENABLE"] = "1"
    elif value.get("DSV4_DECODE_TIMELINE_ENABLE") not in (None, "0"):
        fail(f"{where}: off run timeline value must be absent or '0'")
    for key, expected in required.items():
        if value.get(key) != expected:
            fail(f"{where}: {key} must equal {expected!r}")
    for key, val in value.items():
        lowered = key.lower() + "=" + val.lower()
        if "preload" in lowered or "smelt-strategy=preload" in lowered:
            fail(f"{where}: preload is forbidden")


def validate_argv(value: Any, where: str) -> None:
    if type(value) is not list or not value or any(type(item) is not str for item in value):
        fail(f"{where}: expected non-empty string array")
    joined = "\0".join(value).lower()
    if "serve" not in value or "--native" not in value:
        fail(f"{where}: native serve argv required")
    if "--expert-packed-dir" not in value or PACKED not in value:
        fail(f"{where}: exact packed experts required")
    if "--model" not in value or MODEL not in value:
        fail(f"{where}: exact model required")
    if "preload" in joined:
        fail(f"{where}: preload forbidden")


def validate_lifecycle(value: Any, where: str) -> None:
    lifecycle = closed(value, LIFECYCLE_FIELDS, where)
    for field in ("cleanup_before", "verify_none_before", "started", "stopped", "waited", "verify_none_after"):
        if boolean(lifecycle[field], where + "." + field) is not True:
            fail(f"{where}.{field}: must be true")
    if integer(lifecycle["active_publishers_final"], where + ".active_publishers_final") != 0:
        fail(f"{where}: active publishers remain")
    if boolean(lifecycle["overflow"], where + ".overflow"):
        fail(f"{where}: overflow invalidates run")
    if integer(lifecycle["incomplete_commands"], where + ".incomplete_commands") != 0:
        fail(f"{where}: incomplete commands invalidate run")
    if lifecycle["final_status"] != "verified_none":
        fail(f"{where}.final_status: must be 'verified_none'")
    started_ns = integer(lifecycle["started_ns"], where + ".started_ns", 1)
    stopped_ns = integer(lifecycle["stopped_ns"], where + ".stopped_ns", started_ns)
    integer(lifecycle["verified_none_ns"], where + ".verified_none_ns", stopped_ns)


def validate_provenance(value: Any, enabled: bool, where: str) -> None:
    provenance = closed(value, PROVENANCE_FIELDS, where)
    for field in SHA256_FIELDS - {"semantic_sha256"}:
        sha256(provenance[field], where + "." + field)
    string(provenance["git_head"], where + ".git_head")
    for field in ("model_identity", "tokenizer_identity", "packed_manifest_identity", "routing_seed_identity", "executable_identity"):
        validate_identity(provenance[field], where + "." + field)
    if provenance["model_identity"]["path"] != MODEL:
        fail(f"{where}.model_identity.path: wrong model")
    if provenance["packed_manifest_identity"]["path"] != PACKED:
        fail(f"{where}.packed_manifest_identity.path: wrong packed directory")
    integer(provenance["pid"], where + ".pid", 1)
    validate_environment(provenance["environment"], enabled, where + ".environment")
    validate_argv(provenance["argv"], where + ".argv")
    rss = provenance["rss_series"]
    if type(rss) is not list or not rss:
        fail(f"{where}.rss_series: expected non-empty array")
    previous = -1
    for index, sample in enumerate(rss):
        sample = closed(sample, frozenset(("monotonic_ns", "rss_bytes", "footprint_bytes")), f"{where}.rss_series[{index}]")
        timestamp = integer(sample["monotonic_ns"], f"{where}.rss_series[{index}].monotonic_ns")
        for field in ("rss_bytes", "footprint_bytes"):
            resident = integer(sample[field], f"{where}.rss_series[{index}].{field}")
            if resident >= MAX_RSS_EXCLUSIVE:
                fail(f"{where}.rss_series[{index}]: {field} violates exclusive 44 GiB limit")
        if timestamp <= previous:
            fail(f"{where}.rss_series: timestamps must increase")
        previous = timestamp
    validate_lifecycle(provenance["lifecycle"], where + ".lifecycle")


def validate_run(value: Any, expected_label: str, enabled: bool, where: str) -> int:
    run = closed(value, RUN_FIELDS, where)
    if run["label"] != expected_label:
        fail(f"{where}.label: expected {expected_label!r}")
    if boolean(run["timeline_enabled"], where + ".timeline_enabled") != enabled:
        fail(f"{where}: wrong timeline_enabled")
    sha256(run["timeline_sha256"], where + ".timeline_sha256")
    validate_provenance(run["provenance"], enabled, where + ".provenance")
    return run["provenance"]["pid"]


def hash_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        fail(f"{path}: cannot hash source: {exc}")


def validate_source(value: Any, where: str) -> Path:
    source = closed(value, frozenset(("path", "sha256")), where)
    path = Path(string(source["path"], where + ".path")).resolve()
    expected = sha256(source["sha256"], where + ".sha256")
    if not path.is_file():
        fail(f"{where}: source is unavailable: {path}")
    actual = hash_file(path)
    if actual != expected:
        fail(f"{where}: source hash mismatch expected={expected} actual={actual}")
    return path


def finite_number(value: Any, where: str, positive: bool = False) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        fail(f"{where}: expected finite number")
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        fail(f"{where}: expected finite number")
    if positive and number <= 0:
        fail(f"{where}: expected positive number")
    return number


def load_workload(source: Any, where: str, kind: str) -> dict[str, Any]:
    path = validate_source(source, where + ".source")
    value = load_json(path)
    required = frozenset(("argv", "environment", "executable", "script", "passed", "metrics", "operand", "binding"))
    data = closed(value, required, where + ".content")
    if type(data["argv"]) is not list or not data["argv"] or any(type(x) is not str for x in data["argv"]):
        fail(f"{where}.argv: expected non-empty argv")
    executable = validate_source(data["executable"], where + ".executable")
    invoked = Path(data["argv"][0])
    resolved_invoked = invoked.resolve() if invoked.is_absolute() or "/" in str(invoked) else Path(shutil.which(str(invoked)) or "").resolve()
    if not resolved_invoked.is_file() or hash_file(resolved_invoked) != hash_file(executable):
        fail(f"{where}: workload executable identity mismatch")
    script_ref = data["script"]
    if script_ref is not None:
        script_path = validate_source(script_ref, where + ".script")
        if len(data["argv"]) < 2 or Path(data["argv"][1]).resolve() != script_path:
            fail(f"{where}: workload script identity mismatch")
    binding = closed(data["binding"], frozenset(("request_marker", "response_sha256")), where + ".binding")
    marker = string(binding["request_marker"], where + ".binding.request_marker", nonempty=False)
    response_hash = string(binding["response_sha256"], where + ".binding.response_sha256", nonempty=False)
    if kind == "exact_q4":
        if not marker:
            fail(f"{where}: exact-Q4 request marker required")
        sha256(response_hash, where + ".binding.response_sha256")
    if type(data["environment"]) is not dict or any(type(k) is not str or type(v) is not str for k, v in data["environment"].items()):
        fail(f"{where}.environment: expected string map")
    boolean(data["passed"], where + ".passed")
    if type(data["metrics"]) is not dict:
        fail(f"{where}.metrics: expected object")
    operand = data["operand"]
    if kind == "standard":
        operand = closed(operand, frozenset(("samples", "median_throughput")), where + ".operand")
        samples = operand["samples"]
        if type(samples) is not list or len(samples) != 3:
            fail(f"{where}.operand.samples: exactly three max-5 samples required")
        throughputs = []
        for index, sample in enumerate(samples):
            sample = closed(sample, frozenset(("max_tokens", "completion_tokens", "elapsed_ns")), f"{where}.operand.samples[{index}]")
            if integer(sample["max_tokens"], f"{where}.samples[{index}].max_tokens") != 5:
                fail(f"{where}: standard max_tokens must be 5")
            if integer(sample["completion_tokens"], f"{where}.samples[{index}].completion_tokens") != 5:
                fail(f"{where}: standard completion_tokens must be 5")
            elapsed = integer(sample["elapsed_ns"], f"{where}.samples[{index}].elapsed_ns", 1)
            throughputs.append(5_000_000_000 / elapsed)
        computed = sorted(throughputs)[1]
        if finite_number(operand["median_throughput"], where + ".operand.median_throughput", True) != computed:
            fail(f"{where}: standard median does not match samples")
        data["computed_throughput"] = computed
    elif kind == "long":
        # Protocol v5: median of three sequential 100-token requests.
        operand = closed(operand, frozenset(("samples", "median_throughput")), where + ".operand")
        samples = operand["samples"]
        if type(samples) is not list or len(samples) != 3:
            fail(f"{where}.operand.samples: exactly three 100-token samples required")
        throughputs = []
        for index, sample in enumerate(samples):
            sample = closed(sample, frozenset(("max_tokens", "completion_tokens", "elapsed_ns")), f"{where}.operand.samples[{index}]")
            if integer(sample["max_tokens"], f"{where}.samples[{index}].max_tokens") != 100:
                fail(f"{where}: long max_tokens must be 100")
            if integer(sample["completion_tokens"], f"{where}.samples[{index}].completion_tokens") != 100:
                fail(f"{where}: long completion_tokens must be 100")
            elapsed = integer(sample["elapsed_ns"], f"{where}.samples[{index}].elapsed_ns", 1)
            throughputs.append(100_000_000_000 / elapsed)
        computed = sorted(throughputs)[1]
        if finite_number(operand["median_throughput"], where + ".operand.median_throughput", True) != computed:
            fail(f"{where}: long median does not match samples")
        data["computed_throughput"] = computed
    elif operand is not None:
        fail(f"{where}.operand: must be null for {kind}")
    return data


def canonical_pair_workload(workload: dict[str, Any], template: Any, where: str) -> dict[str, Any]:
    if type(template) is not list or not template or any(type(item) is not str or not item for item in template):
        fail(f"{where}.template: expected non-empty string array")
    script = workload["script"]
    identity = {
        "executable_sha256": sha256(workload["executable"]["sha256"], where + ".executable_sha256"),
        "script_sha256": None if script is None else sha256(script["sha256"], where + ".script_sha256"),
        "canonical_request_operand": workload["operand"],
        "template": template,
    }
    identity["sha256"] = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    return identity


def validate_pair_workload_identity(value: Any, workload: dict[str, Any], where: str) -> dict[str, Any]:
    identity = closed(value, frozenset(("executable_sha256", "script_sha256", "canonical_request_operand", "template", "sha256")), where)
    expected = canonical_pair_workload(workload, identity["template"], where)
    if identity != expected:
        fail(f"{where}: canonical workload identity mismatch")
    return identity


def validate_run_v2(value: Any, expected_label: str, enabled: bool, where: str,
                    common: dict[str, str], pair_label: str | None = None) -> dict[str, Any]:
    fields = frozenset(("label", "timeline_enabled", "pid", "executable", "observed_executable", "argv", "environment",
                        "timeline", "workloads", "pair_label", "pair_workload_identity", "rss", "lifecycle"))
    run = closed(value, fields, where)
    if run["label"] != expected_label or boolean(run["timeline_enabled"], where + ".timeline_enabled") != enabled:
        fail(f"{where}: wrong label or timeline mode")
    integer(run["pid"], where + ".pid", 1)
    executable = validate_source(run["executable"], where + ".executable")
    observed_executable = validate_source(run["observed_executable"], where + ".observed_executable")
    if hash_file(executable) != common["binary"] or hash_file(observed_executable) != common["binary"]:
        fail(f"{where}: executable or observed process image differs from frozen binary")
    validate_argv(run["argv"], where + ".argv")
    validate_environment(run["environment"], enabled, where + ".environment")
    timeline_ref = closed(run["timeline"], frozenset(("state", "source")), where + ".timeline")
    expected_state = "present" if enabled else "absent"
    if timeline_ref["state"] != expected_state:
        fail(f"{where}.timeline: expected state {expected_state!r}")
    if enabled:
        timeline_path = validate_source(timeline_ref["source"], where + ".timeline.source")
        timeline = validate_timeline(timeline_path)
    else:
        if timeline_ref["source"] is not None:
            fail(f"{where}.timeline.source: disabled run must bind explicit absence")
        timeline = None
    workloads = run["workloads"]
    expected_kinds = ("standard", "long", "correctness") + (("exact_q4",) if pair_label is not None else ())
    if type(workloads) is not dict or set(workloads) != set(expected_kinds):
        fail(f"{where}.workloads: expected {expected_kinds}")
    loaded = {kind: load_workload(workloads[kind], f"{where}.workloads.{kind}", kind) for kind in expected_kinds}
    if not all(item["passed"] for item in loaded.values()):
        fail(f"{where}: workload gate failed")
    pair_identity = None
    if pair_label is None:
        if run["pair_label"] is not None or run["pair_workload_identity"] is not None:
            fail(f"{where}: non-diagnosis run cannot carry pair binding")
    else:
        if run["pair_label"] != pair_label:
            fail(f"{where}: canonical pair label mismatch")
        pair_identity = validate_pair_workload_identity(run["pair_workload_identity"], loaded["exact_q4"],
                                                        where + ".pair_workload_identity")
    rss_path = validate_source(run["rss"], where + ".rss")
    rss_value = load_json(rss_path)
    if type(rss_value) is not list or not rss_value:
        fail(f"{where}.rss: expected non-empty samples")
    previous = -1
    peak = 0
    for index, sample in enumerate(rss_value):
        sample = closed(sample, frozenset(("monotonic_ns", "rss_bytes", "footprint_bytes")), f"{where}.rss[{index}]")
        stamp = integer(sample["monotonic_ns"], f"{where}.rss[{index}].monotonic_ns")
        for field in ("rss_bytes", "footprint_bytes"):
            resident = integer(sample[field], f"{where}.rss[{index}].{field}")
            if resident >= MAX_RSS_EXCLUSIVE:
                fail(f"{where}.rss: {field} exceeds 44 GiB exclusive limit")
        footprint = sample["footprint_bytes"]
        if stamp <= previous:
            fail(f"{where}.rss: timestamps must increase")
        previous, peak = stamp, max(peak, footprint)
    lifecycle_path = validate_source(run["lifecycle"], where + ".lifecycle")
    lifecycle_value = load_json(lifecycle_path)
    validate_lifecycle(lifecycle_value, where + ".lifecycle.content")
    return {"pid": run["pid"], "standard": loaded["standard"]["computed_throughput"],
            "long": loaded["long"]["computed_throughput"], "footprint_peak": peak,
            "timeline": timeline, "pair_workload_identity": pair_identity,
            "started_ns": lifecycle_value["started_ns"],
            "verified_none_ns": lifecycle_value["verified_none_ns"]}


def load_exact_q4_module() -> Any:
    path = Path(__file__).resolve().with_name("benchmark_report.py")
    spec = importlib.util.spec_from_file_location("dsv4_benchmark_report", path)
    if spec is None or spec.loader is None:
        fail("cannot load benchmark_report exact-Q4 validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_exact_q4_raw(path: Path, where: str) -> list[int]:
    ordinals: list[int] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        fail(f"{where}: cannot read exact-Q4 raw: {exc}")
    if not lines:
        fail(f"{where}: empty exact-Q4 raw")
    for index, line in enumerate(lines):
        value = json.loads(line, object_pairs_hook=no_duplicate_pairs)
        value = closed(value, frozenset(("producer", "requests", "stages", "reads", "mla_snapshot", "mla_paths")), f"{where}[{index}]")
        producer = value["producer"]
        for field in ("dropped_stages", "dropped_reads", "incomplete", "active_read_publishers"):
            if integer(producer.get(field), f"{where}[{index}].producer.{field}") != 0:
                fail(f"{where}[{index}]: producer counter {field} is nonzero")
        requests = value["requests"]
        if type(requests) is not list or len(requests) != 1:
            fail(f"{where}[{index}]: exactly one request required")
        ordinal = integer(requests[0].get("request_ordinal"), f"{where}[{index}].request_ordinal")
        if ordinal in ordinals:
            fail(f"{where}: duplicate request ordinal")
        for read in value["reads"]:
            if integer(read.get("source"), f"{where}.read.source") == 3:
                requested = integer(read.get("requested_bytes"), f"{where}.read.requested_bytes")
                actual = integer(read.get("read_bytes"), f"{where}.read.read_bytes")
                if actual != requested or integer(read.get("error_code"), f"{where}.read.error_code") != 0:
                    fail(f"{where}: exact-Q4 short/error read")
        ordinals.append(ordinal)
    return ordinals


def validate_sidecar(value: Any, label: str, paired: str, pair_label: str, where: str,
                     common: dict[str, str], diagnosis: dict[str, Any],
                     diagnosis_summary: dict[str, Any]) -> dict[str, int]:
    fields = frozenset(("label", "paired_diagnosis", "pair_label", "pair_workload_identity", "timeline_enabled", "pid", "executable",
                        "observed_executable", "argv", "environment", "workload", "raw",
                        "raw_terminal", "summary", "pair_source_identities", "rss", "lifecycle"))
    sidecar = closed(value, fields, where)
    if sidecar["label"] != label or sidecar["paired_diagnosis"] != paired or sidecar["pair_label"] != pair_label:
        fail(f"{where}: wrong sidecar order/pair semantics")
    if boolean(sidecar["timeline_enabled"], where + ".timeline_enabled"):
        fail(f"{where}: sidecar timeline must be off")
    pid = integer(sidecar["pid"], where + ".pid", 1)
    for field in ("executable", "observed_executable"):
        path = validate_source(sidecar[field], where + "." + field)
        if hash_file(path) != common["binary"]:
            fail(f"{where}: {field} differs from frozen binary")
    validate_argv(sidecar["argv"], where + ".argv")
    validate_environment(sidecar["environment"], False, where + ".environment")
    env = sidecar["environment"]
    if env.get("NATIVE_EXACT_Q4_CAPTURE") != "1" or not env.get("NATIVE_EXACT_Q4_RAW_JSONL"):
        fail(f"{where}: exact-Q4 capture env missing")
    workload = load_workload(sidecar["workload"], where + ".workload", "exact_q4")
    if not workload["passed"]:
        fail(f"{where}: exact-Q4 workload gate failed")
    sidecar_identity = validate_pair_workload_identity(sidecar["pair_workload_identity"], workload,
                                                       where + ".pair_workload_identity")
    if sidecar_identity != diagnosis_summary["pair_workload_identity"]:
        fail(f"{where}: diagnosis/sidecar canonical workload mismatch")
    raw_path = validate_source(sidecar["raw"], where + ".raw")
    ordinals = validate_exact_q4_raw(raw_path, where + ".raw.content")
    terminal_path = validate_source(sidecar["raw_terminal"], where + ".raw_terminal")
    terminal = closed(load_json(terminal_path), frozenset(("raw_sha256", "request_ordinals", "producer_snapshots", "short_reads", "pending_owners")), where + ".raw_terminal.content")
    if terminal["raw_sha256"] != sidecar["raw"]["sha256"] or terminal["request_ordinals"] != ordinals:
        fail(f"{where}: raw terminal binding mismatch")
    if integer(terminal["short_reads"], where + ".raw_terminal.short_reads") != 0 or integer(terminal["pending_owners"], where + ".raw_terminal.pending_owners") != 0:
        fail(f"{where}: raw terminal counters failed")
    if type(terminal["producer_snapshots"]) is not list or len(terminal["producer_snapshots"]) != len(ordinals):
        fail(f"{where}: raw producer snapshots missing")
    summary_path = validate_source(sidecar["summary"], where + ".summary")
    summary = load_json(summary_path)
    try:
        load_exact_q4_module().validate_exact_q4_latency(summary)
    except Exception as exc:
        fail(f"{where}: invalid exact-Q4 summary: {exc}")
    if len(ordinals) != 1 or len(summary["requests"]) != 1:
        fail(f"{where}: sidecar must bind exactly one marked request")
    summary_identity = summary["identity"]
    expected_summary_identity = {"binary_sha256": common["binary"],
                                 "model_sha256": common["model_config"],
                                 "tokenizer_sha256": common["tokenizer"],
                                 "packed_sha256": common["packed_manifest"],
                                 "role": label}
    for field, expected_value in expected_summary_identity.items():
        if summary_identity.get(field) != expected_value:
            fail(f"{where}: summary identity mismatch for {field}")
    identities = closed(sidecar["pair_source_identities"], frozenset(("binary", "routing_seed")), where + ".pair_source_identities")
    if identities["binary"] != common["binary"] or identities["routing_seed"] != common["routing_seed"]:
        fail(f"{where}: pair source identity mismatch")
    if diagnosis["executable"]["sha256"] != identities["binary"]:
        fail(f"{where}: paired diagnosis binary mismatch")
    rss_path = validate_source(sidecar["rss"], where + ".rss")
    rss = load_json(rss_path)
    if type(rss) is not list or not rss:
        fail(f"{where}.rss: expected samples")
    lifecycle_value = load_json(validate_source(sidecar["lifecycle"], where + ".lifecycle"))
    validate_lifecycle(lifecycle_value, where + ".lifecycle.content")
    return {"pid": pid, "started_ns": lifecycle_value["started_ns"],
            "verified_none_ns": lifecycle_value["verified_none_ns"]}


def validate_failed_child(value: Any, where: str) -> dict[str, Any]:
    fields = frozenset(("label", "role", "pid", "log", "rss", "lifecycle", "observed_executable",
                        "cleanup", "error"))
    record = closed(value, fields, where)
    string(record["label"], where + ".label")
    if record["role"] not in ("warmup", "overhead", "diagnosis", "sidecar"):
        fail(f"{where}.role: invalid child role")
    pid = record["pid"]
    if pid is not None:
        integer(pid, where + ".pid", 1)
    string(record["error"], where + ".error")
    validate_source(record["log"], where + ".log")
    rss = load_json(validate_source(record["rss"], where + ".rss"))
    if type(rss) is not list:
        fail(f"{where}.rss.content: expected array")
    lifecycle = load_json(validate_source(record["lifecycle"], where + ".lifecycle"))
    if pid is not None:
        validate_lifecycle(lifecycle, where + ".lifecycle.content")
    else:
        lifecycle = closed(lifecycle, LIFECYCLE_FIELDS, where + ".lifecycle.content")
        if boolean(lifecycle["started"], where + ".lifecycle.content.started"):
            fail(f"{where}: null PID cannot have started lifecycle")
        for field in ("cleanup_before", "verify_none_before", "stopped", "waited", "verify_none_after"):
            if boolean(lifecycle[field], where + ".lifecycle.content." + field) is not True:
                fail(f"{where}.lifecycle.content.{field}: must be true")
        if lifecycle["final_status"] != "verified_none":
            fail(f"{where}.lifecycle.content.final_status: must be verified_none")
    if record["observed_executable"] is not None:
        validate_source(record["observed_executable"], where + ".observed_executable")
    cleanup = closed(record["cleanup"], frozenset(("stopped", "waited", "verified_none", "error")), where + ".cleanup")
    for field in ("stopped", "waited", "verified_none"):
        if boolean(cleanup[field], where + ".cleanup." + field) is not True:
            fail(f"{where}.cleanup.{field}: must be true")
    if cleanup["error"] is not None:
        fail(f"{where}.cleanup.error: must be null")
    return {**record, "started_ns": lifecycle["started_ns"],
            "verified_none_ns": lifecycle["verified_none_ns"]}


def validate_physical_order(children: list[tuple[str, dict[str, Any]]],
                            failed: tuple[str, dict[str, Any]] | None = None) -> None:
    """Require strict serial process lifecycles in protocol order."""
    previous_label: str | None = None
    previous_end: int | None = None
    seen_pids: set[int] = set()
    for label, summary in children:
        pid = summary.get("pid")
        if pid is not None:
            pid = integer(pid, f"{label}.pid", 1)
            if pid in seen_pids:
                fail("artifact: duplicate child PID")
            seen_pids.add(pid)
        started = integer(summary.get("started_ns"), f"{label}.started_ns", 1)
        verified_none = integer(summary.get("verified_none_ns"), f"{label}.verified_none_ns", started)
        if previous_end is not None and previous_end >= started:
            fail(f"artifact: physical lifecycle overlap {previous_label} -> {label}")
        previous_label, previous_end = label, verified_none
    if failed is not None and failed[1].get("pid") is not None:
        label, summary = failed
        started = integer(summary.get("started_ns"), f"{label}.started_ns", 1)
        if previous_end is not None and previous_end >= started:
            fail(f"artifact: physical lifecycle overlap {previous_label} -> failed {label}")


def validate_closed_decision(status: Any, error: Any, gate_passed: bool,
                             diagnostics: Any, sidecars: Any) -> str:
    if status == "complete":
        if error is not None or not gate_passed:
            fail("artifact: complete terminal requires a passing gate and null error")
    elif status == "NOT_PROVEN":
        if error is not None or gate_passed:
            fail("artifact: NOT_PROVEN terminal requires a failed gate and null error")
        if diagnostics != [] or sidecars != []:
            fail("artifact: NOT_PROVEN strictly forbids diagnostics and sidecars")
    else:
        fail("artifact.terminal.status: expected complete or NOT_PROVEN")
    return status


def validate_partial_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    fields = frozenset(("schema_version", "kind", "sources", "warmup", "runs", "diagnostics",
                        "exact_q4_sidecars", "computed_gate", "failed_child", "terminal", "semantic_sha256"))
    artifact = closed(artifact, fields, "artifact")
    if integer(artifact["schema_version"], "artifact.schema_version") != 3 or artifact["kind"] != "dsv4_decode_critical_path_evidence.partial":
        fail("artifact: unsupported partial schema")
    terminal = closed(artifact["terminal"], frozenset(("status", "error")), "artifact.terminal")
    if terminal["status"] != "failed":
        fail("artifact.terminal.status: expected failed")
    string(terminal["error"], "artifact.terminal.error")
    if type(artifact["sources"]) is not dict:
        fail("artifact.sources: expected object")
    for name, source in artifact["sources"].items():
        string(name, "artifact.sources key")
        validate_source(source, "artifact.sources." + name)
    for field in ("runs", "diagnostics", "exact_q4_sidecars"):
        if type(artifact[field]) is not list:
            fail(f"artifact.{field}: expected array")
    warmup = artifact["warmup"]
    if warmup is not None and type(warmup) is not dict:
        fail("artifact.warmup: expected object or null")
    if len(artifact["runs"]) > len(RUN_ORDER) or len(artifact["diagnostics"]) > len(DIAGNOSIS_ORDER) or len(artifact["exact_q4_sidecars"]) > len(SIDECAR_ORDER):
        fail("artifact: completed child count exceeds protocol")
    run_labels = [item.get("label") if type(item) is dict else None for item in artifact["runs"]]
    diagnosis_labels = [item.get("label") if type(item) is dict else None for item in artifact["diagnostics"]]
    sidecar_labels = [item.get("label") if type(item) is dict else None for item in artifact["exact_q4_sidecars"]]
    if run_labels != list(RUN_ORDER[:len(run_labels)]) or diagnosis_labels != list(DIAGNOSIS_ORDER[:len(diagnosis_labels)]) or sidecar_labels != list(SIDECAR_ORDER[:len(sidecar_labels)]):
        fail("artifact: completed children are not protocol-order prefixes")
    if len(sidecar_labels) > len(diagnosis_labels) or len(diagnosis_labels) > len(sidecar_labels) + 1:
        fail("artifact: partial diagnosis/sidecar prefix relation is impossible")
    completed_count = len(run_labels) + len(diagnosis_labels) + len(sidecar_labels) + (1 if warmup is not None else 0)
    completed_order: list[tuple[str, dict[str, Any]]] = []
    if completed_count:
        source_names = ("binary", "git_head", "dirty_diff", "model_config", "tokenizer",
                        "packed_manifest", "protocol", "collector", "routing_seed")
        if set(artifact["sources"]) != set(source_names):
            fail(f"artifact.sources: completed children require {source_names}")
        common = {name: hash_file(validate_source(artifact["sources"][name], "artifact.sources." + name))
                  for name in source_names}
        if common["collector"] != hash_file(Path(__file__).resolve()):
            fail("artifact.sources.collector: validator is not the recorded collector")
        if warmup is not None:
            completed_order.append((WARMUP_LABEL, validate_run_v2(warmup, WARMUP_LABEL, False,
                                                                  "artifact.warmup", common)))
        run_summaries = [validate_run_v2(run, RUN_ORDER[index], RUN_ORDER[index].startswith("on"),
                                         f"artifact.runs[{index}]", common)
                         for index, run in enumerate(artifact["runs"])]
        if len(run_summaries) == len(RUN_ORDER):
            recomputed_gate = {
                "standard_ratios": [run_summaries[1]["standard"] / run_summaries[0]["standard"],
                                    run_summaries[3]["standard"] / run_summaries[2]["standard"]],
                "long_ratios": [run_summaries[1]["long"] / run_summaries[0]["long"],
                                run_summaries[3]["long"] / run_summaries[2]["long"]],
                "footprint_deltas": [abs(run_summaries[1]["footprint_peak"] - run_summaries[0]["footprint_peak"]),
                                      abs(run_summaries[3]["footprint_peak"] - run_summaries[2]["footprint_peak"])],
            }
            recomputed_gate["passed"] = all(0.95 <= ratio <= 1.05 for ratio in recomputed_gate["standard_ratios"] + recomputed_gate["long_ratios"]) and all(delta <= 268_435_456 for delta in recomputed_gate["footprint_deltas"])
            if artifact["computed_gate"] != recomputed_gate or not recomputed_gate["passed"]:
                fail("artifact.computed_gate: invalid completed overhead gate")
        elif artifact["computed_gate"] is not None:
            fail("artifact.computed_gate: must be null before all overhead runs complete")
        diagnosis_summaries = []
        for index, diagnosis in enumerate(artifact["diagnostics"]):
            diagnosis_summaries.append(validate_run_v2(diagnosis, DIAGNOSIS_ORDER[index], True,
                                        f"artifact.diagnostics[{index}]", common, PAIR_LABELS[index]))
        sidecar_summaries = []
        for index, sidecar in enumerate(artifact["exact_q4_sidecars"]):
            sidecar_summaries.append(validate_sidecar(
                sidecar, SIDECAR_ORDER[index], DIAGNOSIS_ORDER[index], PAIR_LABELS[index],
                f"artifact.exact_q4_sidecars[{index}]", common,
                artifact["diagnostics"][index], diagnosis_summaries[index]))
        completed_order += [(RUN_ORDER[index], summary) for index, summary in enumerate(run_summaries)]
        for index, diagnosis_summary in enumerate(diagnosis_summaries):
            completed_order.append((DIAGNOSIS_ORDER[index], diagnosis_summary))
            if index < len(sidecar_summaries):
                completed_order.append((SIDECAR_ORDER[index], sidecar_summaries[index]))
        validate_physical_order(completed_order)
    elif artifact["computed_gate"] is not None:
        fail("artifact.computed_gate: must be null without completed runs")
    failed = validate_failed_child(artifact["failed_child"], "artifact.failed_child")
    if warmup is None:
        if run_labels or diagnosis_labels or sidecar_labels:
            fail("artifact: completed children cannot precede the warmup process")
        expected_label, expected_role = WARMUP_LABEL, "warmup"
    elif len(run_labels) < len(RUN_ORDER):
        if diagnosis_labels or sidecar_labels:
            fail("artifact: diagnostics cannot precede completed overhead runs")
        expected_label, expected_role = RUN_ORDER[len(run_labels)], "overhead"
    elif len(diagnosis_labels) == len(sidecar_labels) and len(diagnosis_labels) < len(DIAGNOSIS_ORDER):
        expected_label, expected_role = DIAGNOSIS_ORDER[len(diagnosis_labels)], "diagnosis"
    elif len(diagnosis_labels) == len(sidecar_labels) + 1:
        expected_label, expected_role = SIDECAR_ORDER[len(sidecar_labels)], "sidecar"
    else:
        fail("artifact: partial diagnosis/sidecar state is impossible")
    if failed["label"] != expected_label or failed["role"] != expected_role:
        fail("artifact.failed_child: does not identify the next protocol child")
    validate_physical_order(completed_order, (failed["label"], failed))
    expected = sha256(artifact["semantic_sha256"], "artifact.semantic_sha256")
    actual = semantic_hash(artifact)
    if expected != actual:
        fail(f"artifact.semantic_sha256: mismatch expected {actual}")
    return {"valid": True, "terminal": "failed", "failed_child": failed["label"], "semantic_sha256": actual}


def validate_complete_artifact(path: Path) -> dict[str, Any]:
    artifact = closed(load_json(path), frozenset(("schema_version", "kind", "sources", "warmup", "runs",
                                                   "diagnostics", "exact_q4_sidecars", "computed_gate",
                                                   "terminal", "semantic_sha256")), "artifact")
    if integer(artifact["schema_version"], "artifact.schema_version") != 3:
        fail("artifact: unsupported schema_version")
    terminal = closed(artifact["terminal"], frozenset(("status", "error")), "artifact.terminal")
    if artifact["kind"] != "dsv4_decode_critical_path_evidence":
        fail("artifact.kind: unexpected value")
    source_names = ("binary", "git_head", "dirty_diff", "model_config", "tokenizer",
                    "packed_manifest", "protocol", "collector", "routing_seed")
    sources = artifact["sources"]
    if type(sources) is not dict or set(sources) != set(source_names):
        fail(f"artifact.sources: expected {source_names}")
    common: dict[str, str] = {}
    for name in source_names:
        source_path = validate_source(sources[name], f"artifact.sources.{name}")
        common[name] = hash_file(source_path)
    if common["collector"] != hash_file(Path(__file__).resolve()):
        fail("artifact.sources.collector: validator is not the recorded collector")
    runs, diagnostics, sidecars = artifact["runs"], artifact["diagnostics"], artifact["exact_q4_sidecars"]
    if type(runs) is not list or len(runs) != 4 or type(diagnostics) is not list or type(sidecars) is not list:
        fail("artifact: exactly four overhead runs and diagnostic arrays required")
    warmup = artifact["warmup"]
    if type(warmup) is not dict:
        fail("artifact.warmup: completed evidence requires the warmup record")
    warmup_summary = validate_run_v2(warmup, WARMUP_LABEL, False, "artifact.warmup", common)
    summaries = [validate_run_v2(run, RUN_ORDER[index], RUN_ORDER[index].startswith("on"),
                                 f"artifact.runs[{index}]", common)
                 for index, run in enumerate(runs)]
    standard_ratios = [summaries[1]["standard"] / summaries[0]["standard"], summaries[3]["standard"] / summaries[2]["standard"]]
    long_ratios = [summaries[1]["long"] / summaries[0]["long"], summaries[3]["long"] / summaries[2]["long"]]
    footprint_deltas = [abs(summaries[1]["footprint_peak"] - summaries[0]["footprint_peak"]),
                         abs(summaries[3]["footprint_peak"] - summaries[2]["footprint_peak"])]
    passed = all(0.95 <= ratio <= 1.05 for ratio in standard_ratios + long_ratios) and all(delta <= 268_435_456 for delta in footprint_deltas)
    gate = closed(artifact["computed_gate"], frozenset(("standard_ratios", "long_ratios", "footprint_deltas", "passed")), "artifact.computed_gate")
    if gate["standard_ratios"] != standard_ratios or gate["long_ratios"] != long_ratios or gate["footprint_deltas"] != footprint_deltas or gate["passed"] is not passed:
        fail("artifact.computed_gate: does not match recomputed gate")
    status = validate_closed_decision(terminal["status"], terminal["error"], passed, diagnostics, sidecars)
    completed_order = [(WARMUP_LABEL, warmup_summary)] + [(RUN_ORDER[index], summary) for index, summary in enumerate(summaries)]
    if status == "complete":
        if len(diagnostics) != 3 or len(sidecars) != 3:
            fail("artifact: complete evidence requires three diagnostic and three exact-Q4 sidecar runs")
        for index, label in enumerate(DIAGNOSIS_ORDER):
            pair_label = PAIR_LABELS[index]
            summary = validate_run_v2(diagnostics[index], label, True, f"artifact.diagnostics[{index}]", common, pair_label)
            sidecar_summary = validate_sidecar(sidecars[index], SIDECAR_ORDER[index], label, pair_label,
                                               f"artifact.exact_q4_sidecars[{index}]", common,
                                               diagnostics[index], summary)
            completed_order.extend(((label, summary), (SIDECAR_ORDER[index], sidecar_summary)))
    validate_physical_order(completed_order)
    expected = sha256(artifact["semantic_sha256"], "artifact.semantic_sha256")
    actual = semantic_hash(artifact)
    if expected != actual:
        fail(f"artifact.semantic_sha256: mismatch expected {actual}")
    return {"valid": True, "terminal": status, "runs": 4,
            "diagnostics": len(diagnostics), "exact_q4_sidecars": len(sidecars),
            "gate": gate, "semantic_sha256": actual}


def validate_artifact(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if type(value) is not dict:
        fail("artifact: expected object")
    if value.get("kind") == "dsv4_decode_critical_path_evidence.partial":
        return validate_partial_artifact(value)
    return validate_complete_artifact(path)


def orchestration_plan(binary: str) -> dict[str, Any]:
    child_states = [
        "cleanup_start", "verify_none_before", "restore_routing_seed", "start_child",
        "record_pid_executable", "sample_rss_and_workload", "stop_exact_pid", "wait_child",
        "cleanup_stop", "verify_none_after",
    ]
    base = {
        "binary": binary, "build": {"mode": "ReleaseFast", "count": 1, "frozen": True},
        "server": {"native": True, "model": MODEL, "packed_experts": PACKED,
                   "environment": {"NATIVE_SMELT_N": "20"}, "full_preload_forbidden": True,
                   "maximum_concurrent": 1},
        "child_state_machine": child_states,
        "cleanup_command": 'pkill -f "dmlx serve" 2>/dev/null; sleep 1',
        "execution_supported": False,
    }
    return {"schema_version": 3, "warmup": {"label": WARMUP_LABEL, "timeline_enabled": False,
                                            "gated": False,
                                            "purpose": "page-cache warm-up so off1 is not cold"},
            "overhead": list(RUN_ORDER),
            "diagnosis_authorization": "only_after_overhead_gate_passed",
            "diagnostic_pairs": [{"diagnosis": diagnosis, "exact_q4_sidecar": sidecar}
                                 for diagnosis, sidecar in zip(DIAGNOSIS_ORDER, SIDECAR_ORDER)],
            "frozen": base}


def source_ref(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": hash_file(resolved)}


def run_checked(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
                stdout: Any = subprocess.PIPE) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                              stderr=subprocess.PIPE, text=True, check=True, shell=False)
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"command failed {argv!r}: {exc}")


def verify_no_server() -> None:
    result = subprocess.run(["pgrep", "-f", "dmlx serve"], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            check=False, shell=False)
    if result.returncode == 0 and result.stdout.strip():
        fail(f"dmlx serve still active: {result.stdout.strip()}")
    if result.returncode not in (0, 1):
        fail("pgrep failed while verifying process exclusion")


def cleanup_servers() -> None:
    subprocess.run(["pkill", "-f", "dmlx serve"], stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, shell=False)
    time.sleep(1.0)
    verify_no_server()


def parse_argv_template(text: str, option: str) -> list[str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        fail(f"{option}: must be a JSON argv array: {exc}")
    if type(value) is not list or not value or any(type(item) is not str or not item for item in value):
        fail(f"{option}: must be a non-empty JSON string array")
    return value


def render_argv(template: list[str], values: dict[str, str]) -> list[str]:
    try:
        return [part.format_map(values) for part in template]
    except (KeyError, ValueError) as exc:
        fail(f"invalid workload placeholder: {exc}")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def observed_process_executable(pid: int, frozen: Path) -> dict[str, str]:
    result = run_checked(["lsof", "-a", "-p", str(pid), "-d", "txt", "-Fn"])
    candidates = [Path(line[1:]).resolve() for line in result.stdout.splitlines() if line.startswith("n/")]
    frozen_hash = hash_file(frozen)
    for candidate in candidates:
        if candidate.is_file() and hash_file(candidate) == frozen_hash:
            return source_ref(candidate)
    fail(f"PID {pid}: lsof did not observe frozen executable image")


def process_memory_sample(pid: int) -> dict[str, int]:
    # proc_pid_rusage(RUSAGE_INFO_V4): ri_resident_size is u64 index 8,
    # ri_phys_footprint is u64 index 9 (ri_uuid is uint8_t[16]).
    buffer = (ctypes.c_uint64 * 64)()
    if _LIBC.proc_pid_rusage(pid, RUSAGE_INFO_V4, buffer) != 0:
        fail(f"cannot sample memory for PID {pid}")
    return {"rss_bytes": int(buffer[8]), "footprint_bytes": int(buffer[9])}


def wait_ready(process: subprocess.Popen[Any], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            fail(f"server PID {process.pid} exited before readiness")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 300:
                    json.loads(response.read().decode("utf-8"))
                    return
        except (OSError, UnicodeError, json.JSONDecodeError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    fail(f"server PID {process.pid} readiness timeout")


def workspace_source_fingerprint(repo: Path, output: Path | None = None) -> str:
    head = run_checked(["git", "rev-parse", "HEAD"], cwd=repo).stdout
    tracked = run_checked(["git", "diff", "--binary", "HEAD", "--"], cwd=repo).stdout
    names = run_checked(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo).stdout
    records = []
    for relative in sorted(item for item in names.split("\0") if item):
        candidate = (repo / relative).resolve()
        if output is not None and (candidate == output or output in candidate.parents):
            continue
        if candidate.is_file():
            records.append((relative, hash_file(candidate)))
    return hashlib.sha256(canonical_bytes({"head": head, "tracked": tracked, "untracked": records})).hexdigest()


def collect_source_files(args: argparse.Namespace, output: Path, frozen: Path) -> tuple[dict[str, Any], bytes | None]:
    git_head = run_checked(["git", "rev-parse", "HEAD"], cwd=Path(args.repo)).stdout.strip()
    head_path = output / "git-head.txt"; head_path.write_text(git_head + "\n", encoding="utf-8")
    tracked = run_checked(["git", "diff", "--binary", "HEAD", "--"], cwd=Path(args.repo)).stdout
    untracked = run_checked(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=Path(args.repo)).stdout
    untracked_names = sorted(x for x in untracked.split("\0") if x)
    untracked_sources = []
    repo = Path(args.repo).resolve()
    for relative in untracked_names:
        candidate = (repo / relative).resolve()
        if candidate == output or output in candidate.parents:
            continue
        if not candidate.is_file() or repo not in candidate.parents:
            fail(f"unsupported untracked source while freezing dirty state: {relative}")
        untracked_sources.append({"path": relative, "sha256": hash_file(candidate)})
    dirty_path = output / "canonical-dirty-state.json"
    write_json(dirty_path, {"tracked_diff": tracked, "untracked_sources": untracked_sources})
    seed = Path(args.routing_seed)
    seed_bytes = seed.read_bytes() if seed.exists() else None
    seed_snapshot = output / "routing-seed.snapshot"
    if seed_bytes is None:
        write_json(seed_snapshot, {"state": "absent", "path": str(seed.resolve())})
    else:
        seed_snapshot.write_bytes(seed_bytes)
    paths = {"binary": frozen, "git_head": head_path, "dirty_diff": dirty_path,
             "model_config": Path(args.model_config), "tokenizer": Path(args.tokenizer),
             "packed_manifest": Path(args.packed_manifest), "protocol": Path(args.protocol),
             "collector": Path(__file__), "routing_seed": seed_snapshot}
    return {name: source_ref(path) for name, path in paths.items()}, seed_bytes


def restore_seed(path: Path, snapshot: bytes | None) -> None:
    if snapshot is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".restore.tmp")
        temporary.write_bytes(snapshot)
        os.replace(temporary, path)


def collect_workload(kind: str, template: list[str], label: str, base_url: str,
                     directory: Path, environment: dict[str, str], marker: str = "") -> dict[str, str]:
    output = directory / f"{kind}.json"
    if not any("{base_url}" in item for item in template):
        fail(f"{label}/{kind}: workload template must contain {{base_url}}")
    argv = render_argv(template, {"base_url": base_url, "label": label, "output": str(output),
                                  "request_marker": marker})
    executable_text = shutil.which(argv[0]) if "/" not in argv[0] else argv[0]
    if not executable_text or not Path(executable_text).resolve().is_file():
        fail(f"{label}/{kind}: workload executable unavailable")
    executable = source_ref(Path(executable_text))
    script = source_ref(Path(argv[1])) if len(argv) > 1 and Path(argv[1]).is_file() else None
    try:
        result = subprocess.run(argv, env=environment, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                check=False, shell=False)
    except OSError as exc:
        fail(f"{label}/{kind}: workload spawn failed {argv!r}: {exc}")
    if result.returncode != 0:
        stdout_path, stderr_path = directory / f"{kind}.stdout.txt", directory / f"{kind}.stderr.txt"
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        fail(f"{label}/{kind}: workload exited {result.returncode} "
             f"(captured {stdout_path.name}, {stderr_path.name}); "
             f"stderr tail: {result.stderr.strip()[-2000:]!r}; "
             f"stdout tail: {result.stdout.strip()[-2000:]!r}")
    try:
        payload = json.loads(result.stdout, object_pairs_hook=no_duplicate_pairs)
    except json.JSONDecodeError as exc:
        fail(f"{label}/{kind}: workload stdout must be one JSON object: {exc}")
    if type(payload) is not dict:
        fail(f"{label}/{kind}: workload stdout must be an object")
    passed = boolean(payload.pop("passed", False), f"{label}/{kind}.passed")
    metrics = payload.pop("metrics", {})
    operand = payload.pop("operand", None)
    response = payload.pop("response", None)
    response_hash = hashlib.sha256(canonical_bytes(response)).hexdigest() if response is not None else ""
    echoed_marker = payload.pop("request_marker", "")
    if marker and (echoed_marker != marker or response is None):
        fail(f"{label}/{kind}: workload did not bind request marker/response")
    if payload:
        fail(f"{label}/{kind}: unknown workload fields {sorted(payload)}")
    write_json(output, {"argv": argv, "environment": environment, "executable": executable,
                        "script": script, "passed": passed, "metrics": metrics, "operand": operand,
                        "binding": {"request_marker": echoed_marker, "response_sha256": response_hash}})
    load_workload(source_ref(output), f"{label}/{kind}", kind)
    return source_ref(output)


def _collect_child(args: argparse.Namespace, label: str, enabled: bool, frozen: Path,
                  seed_snapshot: bytes | None, templates: dict[str, list[str]], output: Path,
                  pair_label: str | None = None) -> dict[str, Any]:
    cleanup_servers()
    seed_path = Path(args.routing_seed)
    restore_seed(seed_path, seed_snapshot)
    verify_no_server()
    directory = output / label; directory.mkdir()
    timeline = directory / "timeline.ndjson"
    log_path = directory / "server.log"
    environment = dict(os.environ)
    environment.update({"NATIVE_SMELT_N": "20", "DSV4_DECODE_TIMELINE_ENABLE": "1" if enabled else "0"})
    if enabled:
        environment["DSV4_DECODE_TIMELINE_RAW_JSONL"] = str(timeline)
    argv = [str(frozen), "serve", "--model", MODEL, "--expert-packed-dir", PACKED,
            "--native", "--port", str(args.port), "--max-tokens", "100", "--temperature", "0"]
    lifecycle = {name: False for name in ("cleanup_before", "verify_none_before", "started", "stopped", "waited", "verify_none_after")}
    lifecycle.update(active_publishers_final=0, overflow=False, incomplete_commands=0,
                     final_status="failed", started_ns=0, stopped_ns=0, verified_none_ns=0)
    lifecycle["cleanup_before"] = lifecycle["verify_none_before"] = True
    rss: list[dict[str, int]] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, cwd=args.repo, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, text=True, shell=False)
        write_json(directory / "pid.json", {"pid": process.pid})
        lifecycle["started"] = True
        lifecycle["started_ns"] = time.monotonic_ns()
        try:
            wait_ready(process, args.port, args.readiness_timeout)
            observed_executable = observed_process_executable(process.pid, frozen)
            write_json(directory / "observed-executable.json", observed_executable)
            rss.append({"monotonic_ns": time.monotonic_ns(), **process_memory_sample(process.pid)})
            workloads: dict[str, Any] = {}
            workload_kinds = ("standard", "long", "correctness") + (("exact_q4",) if pair_label is not None else ())
            for kind in workload_kinds:
                marker = (hashlib.sha256(f"{pair_label}:{hash_file(frozen)}".encode()).hexdigest()
                          if kind == "exact_q4" else "")
                workloads[kind] = collect_workload(kind, templates[kind], label,
                                                   f"http://127.0.0.1:{args.port}", directory, environment, marker)
                rss.append({"monotonic_ns": time.monotonic_ns(), **process_memory_sample(process.pid)})
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            lifecycle["stopped"] = True
            lifecycle["stopped_ns"] = time.monotonic_ns()
            try:
                process.wait(timeout=30); lifecycle["waited"] = True
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(); lifecycle["waited"] = True
            cleanup_error: Exception | None = None
            try:
                cleanup_servers(); lifecycle["verify_none_after"] = True
                lifecycle["verified_none_ns"] = time.monotonic_ns()
                lifecycle["final_status"] = "verified_none"
            except Exception as exc:
                cleanup_error = exc
            finally:
                write_json(directory / "rss.json", rss)
                write_json(directory / "lifecycle.json", lifecycle)
            if cleanup_error is not None:
                raise cleanup_error
    rss_path = directory / "rss.json"
    lifecycle_path = directory / "lifecycle.json"
    if enabled:
        if not timeline.is_file():
            fail(f"{label}: producer did not write timeline {timeline}")
        validate_timeline(timeline)
        timeline_ref: dict[str, Any] = {"state": "present", "source": source_ref(timeline)}
    else:
        if timeline.exists():
            fail(f"{label}: disabled producer unexpectedly wrote timeline")
        timeline_ref = {"state": "absent", "source": None}
    pair_identity = None
    if pair_label is not None:
        exact = load_workload(workloads["exact_q4"], label + ".exact_q4", "exact_q4")
        pair_identity = canonical_pair_workload(exact, templates["exact_q4"], label + ".pair_workload_identity")
    return {"label": label, "timeline_enabled": enabled, "pid": process.pid,
            "executable": source_ref(frozen), "observed_executable": observed_executable,
            "argv": argv, "environment": environment,
            "timeline": timeline_ref, "workloads": workloads, "pair_label": pair_label,
            "pair_workload_identity": pair_identity, "rss": source_ref(rss_path),
            "lifecycle": source_ref(lifecycle_path)}


def _collect_sidecar(args: argparse.Namespace, label: str, paired_diagnosis: str, pair_label: str, frozen: Path,
                    seed_snapshot: bytes | None, template: list[str], output: Path,
                    sources: dict[str, Any]) -> dict[str, Any]:
    cleanup_servers()
    restore_seed(Path(args.routing_seed), seed_snapshot)
    verify_no_server()
    directory = output / label; directory.mkdir()
    raw = directory / "exact-q4.raw.jsonl"
    summary = directory / "exact-q4.summary.json"
    identity_path = directory / "exact-q4.identity.json"
    log_path = directory / "server.log"
    environment = dict(os.environ)
    environment.update({"NATIVE_SMELT_N": "20", "DSV4_DECODE_TIMELINE_ENABLE": "0",
                        "NATIVE_EXACT_Q4_CAPTURE": "1", "NATIVE_EXACT_Q4_RAW_JSONL": str(raw)})
    argv = [str(frozen), "serve", "--model", MODEL, "--expert-packed-dir", PACKED,
            "--native", "--port", str(args.port), "--max-tokens", "100", "--temperature", "0"]
    lifecycle = {name: False for name in ("cleanup_before", "verify_none_before", "started", "stopped", "waited", "verify_none_after")}
    lifecycle.update(active_publishers_final=0, overflow=False, incomplete_commands=0,
                     final_status="failed", started_ns=0, stopped_ns=0, verified_none_ns=0)
    lifecycle["cleanup_before"] = lifecycle["verify_none_before"] = True
    rss: list[dict[str, int]] = []
    workloads: dict[str, Any] = {}
    observed_executable: dict[str, str] | None = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, cwd=args.repo, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, text=True, shell=False)
        write_json(directory / "pid.json", {"pid": process.pid})
        lifecycle["started"] = True
        lifecycle["started_ns"] = time.monotonic_ns()
        try:
            wait_ready(process, args.port, args.readiness_timeout)
            observed_executable = observed_process_executable(process.pid, frozen)
            write_json(directory / "observed-executable.json", observed_executable)
            rss.append({"monotonic_ns": time.monotonic_ns(), **process_memory_sample(process.pid)})
            marker = hashlib.sha256(f"{pair_label}:{sources['binary']['sha256']}".encode()).hexdigest()
            workloads["exact_q4"] = collect_workload("exact_q4", template, label,
                f"http://127.0.0.1:{args.port}", directory, environment, marker)
            rss.append({"monotonic_ns": time.monotonic_ns(), **process_memory_sample(process.pid)})
            ordinals = validate_exact_q4_raw(raw, label + ".raw")
            raw_records = [json.loads(line, object_pairs_hook=no_duplicate_pairs)
                           for line in raw.read_text(encoding="utf-8").splitlines()]
            producer_snapshots = [record["producer"] for record in raw_records]
            short_reads = sum(1 for record in raw_records for read in record["reads"]
                              if read["source"] == 3 and (read["read_bytes"] != read["requested_bytes"] or read["error_code"] != 0))
            pending_owners = sum(record["producer"]["active_read_publishers"] for record in raw_records)
            raw_terminal = directory / "exact-q4.raw-terminal.json"
            write_json(raw_terminal, {"raw_sha256": hash_file(raw), "request_ordinals": ordinals,
                                      "producer_snapshots": producer_snapshots,
                                      "short_reads": short_reads, "pending_owners": pending_owners})
            identity = {"workspace_revision": Path(sources["git_head"]["path"]).read_text().strip(),
                        "binary_sha256": sources["binary"]["sha256"],
                        "model_sha256": sources["model_config"]["sha256"],
                        "tokenizer_sha256": sources["tokenizer"]["sha256"],
                        "packed_sha256": sources["packed_manifest"]["sha256"],
                        "command": " ".join(argv), "environment": environment,
                        "role": label, "pair_ordinal": SIDECAR_ORDER.index(label)}
            write_json(identity_path, identity)
            module = load_exact_q4_module()
            document = module.exact_q4_document_from_raw_jsonl(raw, identity_path, ordinals)
            write_json(summary, document)
            module.validate_exact_q4_latency(load_json(summary))
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            lifecycle["stopped"] = True
            lifecycle["stopped_ns"] = time.monotonic_ns()
            try:
                process.wait(timeout=30); lifecycle["waited"] = True
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(); lifecycle["waited"] = True
            cleanup_error: Exception | None = None
            try:
                cleanup_servers(); lifecycle["verify_none_after"] = True
                lifecycle["verified_none_ns"] = time.monotonic_ns()
                lifecycle["final_status"] = "verified_none"
            except Exception as exc:
                cleanup_error = exc
            finally:
                write_json(directory / "rss.json", rss)
                write_json(directory / "lifecycle.json", lifecycle)
            if cleanup_error is not None:
                raise cleanup_error
    if observed_executable is None:
        fail(f"{label}: executable observation missing")
    workload = workloads["exact_q4"]
    loaded_workload = load_workload(workload, label + ".exact_q4", "exact_q4")
    pair_identity = canonical_pair_workload(loaded_workload, template, label + ".pair_workload_identity")
    return {"label": label, "paired_diagnosis": paired_diagnosis, "pair_label": pair_label,
            "pair_workload_identity": pair_identity, "timeline_enabled": False,
            "pid": process.pid, "executable": source_ref(frozen),
            "observed_executable": observed_executable, "argv": argv, "environment": environment,
            "workload": workload, "raw": source_ref(raw),
            "raw_terminal": source_ref(directory / "exact-q4.raw-terminal.json"),
            "summary": source_ref(summary),
            "pair_source_identities": {"binary": sources["binary"]["sha256"],
                "routing_seed": sources["routing_seed"]["sha256"]},
            "rss": source_ref(directory / "rss.json"),
            "lifecycle": source_ref(directory / "lifecycle.json")}


def failed_child_record(output: Path, label: str, role: str, error: Exception) -> dict[str, Any]:
    directory = output / label
    pid_path = directory / "pid.json"
    pid = integer(load_json(pid_path).get("pid"), label + ".pid", 1) if pid_path.is_file() else None
    log_path, rss_path, lifecycle_path = (directory / "server.log", directory / "rss.json", directory / "lifecycle.json")
    if pid is None:
        cleanup_servers()
        now = time.monotonic_ns()
        if not rss_path.is_file():
            write_json(rss_path, [])
        write_json(lifecycle_path, {"cleanup_before": True, "verify_none_before": True, "started": False,
                                   "stopped": True, "waited": True, "verify_none_after": True,
                                   "active_publishers_final": 0, "overflow": False, "incomplete_commands": 0,
                                   "final_status": "verified_none", "started_ns": 0, "stopped_ns": now,
                                   "verified_none_ns": time.monotonic_ns()})
    lifecycle = load_json(lifecycle_path)
    observed_path = directory / "observed-executable.json"
    observed = load_json(observed_path) if observed_path.is_file() else None
    cleanup_error = None if lifecycle.get("verify_none_after") else "cleanup/verify-none incomplete"
    return {"label": label, "role": role, "pid": pid, "log": source_ref(log_path),
            "rss": source_ref(rss_path), "lifecycle": source_ref(lifecycle_path),
            "observed_executable": observed,
            "cleanup": {"stopped": lifecycle.get("stopped") is True,
                        "waited": lifecycle.get("waited") is True,
                        "verified_none": lifecycle.get("verify_none_after") is True,
                        "error": cleanup_error},
            "error": str(error)}


def collect_child(args: argparse.Namespace, label: str, enabled: bool, frozen: Path,
                  seed_snapshot: bytes | None, templates: dict[str, list[str]], output: Path,
                  pair_label: str | None = None) -> dict[str, Any]:
    try:
        return _collect_child(args, label, enabled, frozen, seed_snapshot, templates, output, pair_label)
    except Exception as exc:
        role = "warmup" if label == WARMUP_LABEL else ("diagnosis" if pair_label is not None else "overhead")
        raise ChildCollectionError(str(exc), failed_child_record(output, label, role, exc)) from exc


def collect_sidecar(args: argparse.Namespace, label: str, paired_diagnosis: str, pair_label: str, frozen: Path,
                    seed_snapshot: bytes | None, template: list[str], output: Path,
                    sources: dict[str, Any]) -> dict[str, Any]:
    try:
        return _collect_sidecar(args, label, paired_diagnosis, pair_label, frozen, seed_snapshot,
                                template, output, sources)
    except Exception as exc:
        raise ChildCollectionError(str(exc), failed_child_record(output, label, "sidecar", exc)) from exc


def gate_from_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(run: dict[str, Any], kind: str) -> dict[str, Any]:
        return load_workload(run["workloads"][kind], f"collect.{run['label']}.{kind}", kind)
    standards = [metrics(run, "standard")["computed_throughput"] for run in runs]
    longs = [metrics(run, "long")["computed_throughput"] for run in runs]
    peaks = [max(item["footprint_bytes"] for item in load_json(Path(run["rss"]["path"]))) for run in runs]
    standard_ratios = [standards[1] / standards[0], standards[3] / standards[2]]
    long_ratios = [longs[1] / longs[0], longs[3] / longs[2]]
    footprint_deltas = [abs(peaks[1] - peaks[0]), abs(peaks[3] - peaks[2])]
    passed = all(0.95 <= x <= 1.05 for x in standard_ratios + long_ratios) and all(x <= 268_435_456 for x in footprint_deltas)
    return {"standard_ratios": standard_ratios, "long_ratios": long_ratios,
            "footprint_deltas": footprint_deltas, "passed": passed}


def collect(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        fail(f"collect output already exists: {output}")
    repo = Path(args.repo).resolve()
    pre_workspace = workspace_source_fingerprint(repo, output)
    pre_inputs = {name: hash_file(Path(path)) for name, path in {
        "model_config": args.model_config, "tokenizer": args.tokenizer,
        "packed_manifest": args.packed_manifest, "protocol": args.protocol,
        "collector": __file__,
    }.items()}
    output.mkdir(parents=True)
    warmup: dict[str, Any] | None = None
    runs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    failed_child: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    sources: dict[str, Any] = {}
    try:
        templates = {kind: parse_argv_template(getattr(args, kind + "_command"), "--" + kind.replace("_", "-") + "-command")
                     for kind in ("standard", "long", "correctness", "exact_q4")}
        run_checked(["zig", "build", "-Doptimize=ReleaseFast", "-Ddecode_timeline_probe=true"], cwd=repo)
        if workspace_source_fingerprint(repo, output) != pre_workspace:
            fail("source workspace changed during build")
        for name, expected in pre_inputs.items():
            if hash_file(Path({"model_config": args.model_config, "tokenizer": args.tokenizer,
                               "packed_manifest": args.packed_manifest, "protocol": args.protocol,
                               "collector": __file__}[name])) != expected:
                fail(f"source changed during build: {name}")
        built = repo / "zig-out/bin/dmlx"
        frozen = output / "frozen-dmlx"; shutil.copy2(built, frozen)
        sources, seed_snapshot = collect_source_files(args, output, frozen)
        warmup = collect_child(args, WARMUP_LABEL, False, frozen, seed_snapshot, templates, output)
        for label in RUN_ORDER:
            runs.append(collect_child(args, label, label.startswith("on"), frozen, seed_snapshot, templates, output))
        gate = gate_from_runs(runs)
        if not gate["passed"]:
            artifact = {"schema_version": 3, "kind": "dsv4_decode_critical_path_evidence",
                        "sources": sources, "warmup": warmup, "runs": runs, "diagnostics": [],
                        "exact_q4_sidecars": [], "computed_gate": gate,
                        "terminal": {"status": "NOT_PROVEN", "error": None}, "semantic_sha256": ""}
            artifact["semantic_sha256"] = semantic_hash(artifact)
            artifact_path = output / "artifact.json"; write_json(artifact_path, artifact)
            validate_artifact(artifact_path)
            return {"artifact": str(artifact_path), "decision": "NOT_PROVEN",
                    "semantic_sha256": artifact["semantic_sha256"]}
        for index, (diagnosis, sidecar) in enumerate(zip(DIAGNOSIS_ORDER, SIDECAR_ORDER)):
            pair_label = PAIR_LABELS[index]
            diagnostics.append(collect_child(args, diagnosis, True, frozen, seed_snapshot, templates, output, pair_label))
            sidecars.append(collect_sidecar(args, sidecar, diagnosis, pair_label, frozen, seed_snapshot,
                                            templates["exact_q4"], output, sources))
        artifact = {"schema_version": 3, "kind": "dsv4_decode_critical_path_evidence",
                    "sources": sources, "warmup": warmup, "runs": runs, "diagnostics": diagnostics,
                    "exact_q4_sidecars": sidecars, "computed_gate": gate,
                    "terminal": {"status": "complete", "error": None}, "semantic_sha256": ""}
        artifact["semantic_sha256"] = semantic_hash(artifact)
        artifact_path = output / "artifact.json"; write_json(artifact_path, artifact)
        validate_artifact(artifact_path)
        return {"artifact": str(artifact_path), "semantic_sha256": artifact["semantic_sha256"]}
    except Exception as exc:
        if isinstance(exc, ChildCollectionError):
            failed_child = exc.record
        if failed_child is None:
            raise
        partial = {"schema_version": 3, "kind": "dsv4_decode_critical_path_evidence.partial",
                   "sources": sources, "warmup": warmup, "runs": runs, "diagnostics": diagnostics,
                   "exact_q4_sidecars": sidecars, "computed_gate": gate,
                   "failed_child": failed_child,
                   "terminal": {"status": "failed", "error": str(exc)}, "semantic_sha256": ""}
        partial["semantic_sha256"] = semantic_hash(partial)
        partial_path = output / "artifact.partial.json"
        write_json(partial_path, partial)
        validate_artifact(partial_path)
        raise


def synthetic_events(residual_ratio: float = 0.0) -> list[dict[str, Any]]:
    begin, end = 1_000, 101_000
    events: list[dict[str, Any]] = []
    def event(stage: str, layer: int | None, start: int, stop: int) -> dict[str, Any]:
        points = [start, start, start, start, stop, stop]
        return dict(zip(EVENT_FIELDS_ORDER, [1, 7, layer, stage, "completed", *points]))
    events.append(event("token_forward", None, begin, end))
    child_end = end - int((end - begin) * residual_ratio)
    # All layer events may overlap; union semantics count the span once.
    for layer in range(LAYERS):
        for stage in LAYER_STAGES:
            events.append(event(stage, layer, begin, child_end))
    events.append(event("logits_cb", None, begin, child_end))
    events.append(event("token_sample", None, begin, child_end))
    return events


EVENT_FIELDS_ORDER = (
    "schema_version", "token", "layer", "stage", "status", "begin_ns",
    "encode_done_ns", "pre_commit_ns", "post_commit_ns", "post_wait_ns", "end_ns",
)


def write_ndjson(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events), encoding="utf-8")


def self_test() -> dict[str, Any]:
    outcomes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="dsv4-decode-validator-") as directory:
        root = Path(directory)
        fixtures: dict[str, list[dict[str, Any]]] = {"valid": synthetic_events()}
        duplicate = synthetic_events()
        duplicate.append(dict(duplicate[-1]))
        fixtures["duplicate"] = duplicate
        missing = synthetic_events()
        missing.pop()
        fixtures["missing"] = missing
        nonmonotonic = synthetic_events()
        nonmonotonic[1]["post_commit_ns"] = nonmonotonic[1]["pre_commit_ns"] - 1
        fixtures["nonmonotonic"] = nonmonotonic
        fixtures["high_residual"] = synthetic_events(0.051)
        zero_envelope = synthetic_events()
        for item in zero_envelope:
            item.update(begin_ns=1_000, encode_done_ns=1_000, pre_commit_ns=1_000,
                        post_commit_ns=1_000, post_wait_ns=1_000, end_ns=1_000)
        fixtures["zero_envelope"] = zero_envelope
        for name, events in fixtures.items():
            path = root / (name + ".ndjson")
            write_ndjson(path, events)
            try:
                validate_timeline(path)
                passed = True
            except ValidationError:
                passed = False
            expected = name == "valid"
            if passed != expected:
                fail(f"self-test {name}: expected valid={expected}, got {passed}")
            outcomes[name] = "accepted" if passed else "rejected"

        wrapper_events = []
        for event in synthetic_events():
            wrapper_events.append({
                "token": event["token"],
                "layer": 65535 if event["layer"] is None else event["layer"],
                "stage": STAGES.index(event["stage"]),
                "status": 1,
                **{name: event[name] for name in TIME_FIELDS},
            })
        wrapper_path = root / "producer-wrapper.ndjson"
        wrapper_path.write_text(json.dumps({
            "abi_version": 2,
            "valid": 1,
            "count": len(wrapper_events),
            "events": wrapper_events,
        }, separators=(",", ":")) + "\n", encoding="utf-8")
        validate_timeline(wrapper_path)
        outcomes["producer_wrapper"] = "accepted"

        # Each capture line is an independent request whose token ordinals
        # restart at zero; concatenated requests must validate per segment.
        wrapper_line = wrapper_path.read_text(encoding="utf-8")
        multi_wrapper_path = root / "producer-wrapper-multi.ndjson"
        multi_wrapper_path.write_text(wrapper_line + wrapper_line, encoding="utf-8")
        validate_timeline(multi_wrapper_path)
        outcomes["producer_wrapper_multi"] = "accepted"

        # A token that repeats after another token within one capture stays invalid.
        broken_events = list(wrapper_events) + [dict(wrapper_events[0])]
        broken_path = root / "producer-wrapper-broken.ndjson"
        broken_path.write_text(json.dumps({
            "abi_version": 2,
            "valid": 1,
            "count": len(broken_events),
            "events": broken_events,
        }, separators=(",", ":")) + "\n", encoding="utf-8")
        try:
            validate_timeline(broken_path)
            fail("self-test non-contiguous capture was accepted")
        except ValidationError:
            outcomes["producer_wrapper_noncontiguous"] = "rejected"

        source_path = root / "source.txt"
        source_path.write_text("bound", encoding="utf-8")
        fabricated = {"path": str(source_path), "sha256": "0" * 64}
        try:
            validate_source(fabricated, "fabricated")
            fail("self-test fabricated hash was accepted")
        except ValidationError:
            outcomes["fabricated_hash"] = "rejected"

        fake_runs = []
        for index in range(4):
            standard = root / f"gate-standard-{index}.json"
            long = root / f"gate-long-{index}.json"
            rss = root / f"gate-rss-{index}.json"
            throughput_elapsed = 1_000_000_000 if index != 1 else 2_000_000_000
            standard_samples = [{"max_tokens": 5, "completion_tokens": 5, "elapsed_ns": throughput_elapsed}] * 3
            standard_operand = {"samples": standard_samples, "median_throughput": 5_000_000_000 / throughput_elapsed}
            long_samples = [{"max_tokens": 100, "completion_tokens": 100, "elapsed_ns": throughput_elapsed}] * 3
            long_operand = {"samples": long_samples, "median_throughput": 100_000_000_000 / throughput_elapsed}
            write_json(standard, {"argv": [sys.executable], "environment": {},
                                  "executable": source_ref(Path(sys.executable)), "script": None,
                                  "passed": True,
                                  "metrics": {}, "operand": standard_operand,
                                  "binding": {"request_marker": "", "response_sha256": ""}})
            write_json(long, {"argv": [sys.executable], "environment": {},
                              "executable": source_ref(Path(sys.executable)), "script": None,
                              "passed": True,
                              "metrics": {}, "operand": long_operand,
                              "binding": {"request_marker": "", "response_sha256": ""}})
            write_json(rss, [{"monotonic_ns": 1, "rss_bytes": 1024, "footprint_bytes": 1024}])
            fake_runs.append({"label": RUN_ORDER[index], "workloads": {"standard": source_ref(standard), "long": source_ref(long)}, "rss": source_ref(rss)})
        if gate_from_runs(fake_runs)["passed"]:
            fail("self-test false overhead gate was accepted")
        outcomes["false_gate"] = "rejected"

        if validate_closed_decision("NOT_PROVEN", None, False, [], []) != "NOT_PROVEN":
            fail("self-test valid NOT_PROVEN decision was not accepted")
        outcomes["valid_not_proven"] = "accepted"
        try:
            validate_closed_decision("NOT_PROVEN", None, True, [], [])
            fail("self-test forged NOT_PROVEN gate was accepted")
        except ValidationError:
            outcomes["forged_not_proven_gate"] = "rejected"

        ordered_overhead = [(label, {"pid": index + 1, "started_ns": index * 20 + 1,
                                     "verified_none_ns": index * 20 + 10})
                            for index, label in enumerate(RUN_ORDER)]
        validate_physical_order(ordered_overhead)
        overlapping_overhead = json.loads(json.dumps(ordered_overhead))
        overlapping_overhead[1][1]["started_ns"] = ordered_overhead[0][1]["verified_none_ns"]
        try:
            validate_physical_order([(label, summary) for label, summary in overlapping_overhead])
            fail("self-test overlapping overhead was accepted")
        except ValidationError:
            outcomes["overlapping_overhead"] = "rejected"

        partial_prefix = ordered_overhead + [
            (DIAGNOSIS_ORDER[0], {"pid": 5, "started_ns": 81, "verified_none_ns": 100}),
            (SIDECAR_ORDER[0], {"pid": 6, "started_ns": 100, "verified_none_ns": 120}),
        ]
        try:
            validate_physical_order(partial_prefix)
            fail("self-test overlapping partial diagnosis/sidecar was accepted")
        except ValidationError:
            outcomes["partial_overlapping_diagnosis_sidecar"] = "rejected"

        exact_path = root / "pair-exact.json"
        exact_payload = {"argv": [sys.executable], "environment": {},
                         "executable": source_ref(Path(sys.executable)), "script": None,
                         "passed": True, "metrics": {}, "operand": None,
                         "binding": {"request_marker": "marker", "response_sha256": "1" * 64}}
        write_json(exact_path, exact_payload)
        loaded_exact = load_workload(source_ref(exact_path), "self-test.pair", "exact_q4")
        pair_identity = canonical_pair_workload(loaded_exact, [sys.executable, "{base_url}"], "self-test.pair")
        mismatched = dict(pair_identity)
        mismatched["template"] = [sys.executable, "different"]
        try:
            validate_pair_workload_identity(mismatched, loaded_exact, "self-test.pair-mismatch")
            fail("self-test pair workload mismatch was accepted")
        except ValidationError:
            outcomes["pair_workload_mismatch"] = "rejected"

        failed_log = root / "failed.log"; failed_log.write_text("readiness failed\n", encoding="utf-8")
        failed_rss = root / "failed-rss.json"; write_json(failed_rss, [])
        failed_lifecycle = root / "failed-lifecycle.json"
        lifecycle_fixture = {"cleanup_before": True, "verify_none_before": True, "started": True,
                             "stopped": True, "waited": True, "verify_none_after": True,
                             "active_publishers_final": 0, "overflow": False, "incomplete_commands": 0,
                             "final_status": "verified_none", "started_ns": 1, "stopped_ns": 2,
                             "verified_none_ns": 3}
        write_json(failed_lifecycle, lifecycle_fixture)
        failed_record = {"label": WARMUP_LABEL, "role": "warmup", "pid": 42,
                         "log": source_ref(failed_log), "rss": source_ref(failed_rss),
                         "lifecycle": source_ref(failed_lifecycle), "observed_executable": None,
                         "cleanup": {"stopped": True, "waited": True, "verified_none": True, "error": None},
                         "error": "readiness failed"}
        partial = {"schema_version": 3, "kind": "dsv4_decode_critical_path_evidence.partial",
                   "sources": {}, "warmup": None, "runs": [], "diagnostics": [], "exact_q4_sidecars": [],
                   "computed_gate": None, "failed_child": failed_record,
                   "terminal": {"status": "failed", "error": "readiness failed"}, "semantic_sha256": ""}
        partial["semantic_sha256"] = semantic_hash(partial)
        if validate_partial_artifact(partial)["terminal"] != "failed":
            fail("self-test valid partial was not accepted")
        outcomes["valid_partial"] = "accepted"
        # Without a completed warmup record, off1 cannot be the failed child.
        skipped_warmup = json.loads(json.dumps(partial))
        skipped_warmup["failed_child"]["label"] = "off1"
        skipped_warmup["failed_child"]["role"] = "overhead"
        skipped_warmup["semantic_sha256"] = semantic_hash(skipped_warmup)
        try:
            validate_partial_artifact(skipped_warmup)
            fail("self-test warmup-skipping partial was accepted")
        except ValidationError:
            outcomes["skipped_warmup"] = "rejected"
        forged = json.loads(json.dumps(partial))
        forged["failed_child"]["cleanup"]["waited"] = False
        forged["semantic_sha256"] = semantic_hash(forged)
        try:
            validate_partial_artifact(forged)
            fail("self-test forged partial was accepted")
        except ValidationError:
            outcomes["forged_partial"] = "rejected"
        sidecar_without_diagnosis = json.loads(json.dumps(partial))
        sidecar_without_diagnosis["runs"] = [{"label": label} for label in RUN_ORDER]
        sidecar_without_diagnosis["exact_q4_sidecars"] = [{"label": SIDECAR_ORDER[0]}]
        sidecar_without_diagnosis["failed_child"]["label"] = DIAGNOSIS_ORDER[0]
        sidecar_without_diagnosis["failed_child"]["role"] = "diagnosis"
        sidecar_without_diagnosis["semantic_sha256"] = semantic_hash(sidecar_without_diagnosis)
        try:
            validate_partial_artifact(sidecar_without_diagnosis)
            fail("self-test sidecar-only partial was accepted")
        except ValidationError:
            outcomes["sidecar_without_diagnosis"] = "rejected"
    return {"self_test": "passed", "fixtures": outcomes}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    group = result.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", metavar="TIMELINE.ndjson", type=Path,
                       help="strictly validate a raw timeline NDJSON file offline")
    group.add_argument("--validate-artifact", metavar="ARTIFACT.json", type=Path,
                       help="re-open and validate a source-bound collection artifact")
    group.add_argument("--semantic-hash", metavar="ARTIFACT.json", type=Path,
                       help="print canonical SHA-256 excluding semantic_sha256")
    group.add_argument("--plan", action="store_true", help="print the non-executing serial plan")
    group.add_argument("--dry-run", action="store_true", help="alias for a non-executing collection plan")
    group.add_argument("--self-test", action="store_true", help="run fail-closed offline fixtures")
    group.add_argument("--collect", action="store_true",
                       help="EXPLICITLY execute four overhead children and three diagnosis/sidecar pairs")
    result.add_argument("--frozen-binary", default="./frozen/dmlx")
    result.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    result.add_argument("--output")
    result.add_argument("--port", type=int, default=8930)
    result.add_argument("--readiness-timeout", type=float, default=300.0)
    result.add_argument("--routing-seed", default=PACKED + "/.smelt_routing_stats.bin")
    result.add_argument("--model-config")
    result.add_argument("--tokenizer")
    result.add_argument("--packed-manifest")
    result.add_argument("--protocol", default=str(Path(__file__).resolve().parent.parent / "docs/en/analysis/dsv4-decode-critical-path-protocol.json"))
    for kind in ("standard", "long", "correctness", "exact-q4"):
        result.add_argument(f"--{kind}-command", dest=kind.replace("-", "_") + "_command",
                            help="required JSON argv template; placeholders: {base_url}, {label}, {output}, {request_marker}")
    return result


def require_collect_arguments(args: argparse.Namespace) -> None:
    required = ("output", "model_config", "tokenizer", "packed_manifest", "standard_command",
                "long_command", "correctness_command", "exact_q4_command")
    missing = ["--" + name.replace("_", "-") for name in required if not getattr(args, name)]
    if missing:
        fail("--collect requires " + ", ".join(missing))
    if not 1 <= args.port <= 65535 or args.readiness_timeout <= 0:
        fail("--collect requires a valid port and positive readiness timeout")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.validate is not None:
            output = validate_timeline(args.validate)
        elif args.validate_artifact is not None:
            output = validate_artifact(args.validate_artifact)
        elif args.semantic_hash is not None:
            value = load_json(args.semantic_hash)
            if type(value) is not dict:
                fail("artifact: expected object")
            output = {"semantic_sha256": semantic_hash(value)}
        elif args.collect:
            require_collect_arguments(args)
            output = collect(args)
        elif args.plan or args.dry_run:
            output = orchestration_plan(args.frozen_binary)
            output["frozen"]["execution_supported"] = True
            output["execution_requires_explicit_collect"] = True
            output["required_workload_templates"] = ["standard", "long", "correctness", "exact_q4"]
        else:
            output = self_test()
        print(json.dumps(output, sort_keys=True, separators=(",", ":")))
        return 0
    except (ValidationError, ValueError, ChildCollectionError) as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
