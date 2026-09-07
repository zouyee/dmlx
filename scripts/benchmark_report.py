#!/usr/bin/env python3
"""Write and validate dmlx benchmark schema v1 JSON artifacts.

The writer consumes BM_* environment variables exported by run_benchmark.sh.
Unavailable measurements are serialized as JSON null, never synthetic zeroes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from check_expert_io_invariants import InvariantError, summarize_telemetry
except ModuleNotFoundError:  # Support importing as scripts.benchmark_report.
    from scripts.check_expert_io_invariants import InvariantError, summarize_telemetry

SCHEMA_VERSION = 1
TELEMETRY_SCHEMA_VERSION = 1
STANDARD_RUN_COUNT = 3
DEFAULT_OUTPUT = "docs/en/analysis/performance-benchmark.json"

BATCH_ENVIRONMENT_KEYS = (
    "NATIVE_BATCH_PREFILL",
    "NATIVE_BATCH_MLA",
    "NATIVE_EXPERT_BUCKETS",
    "NATIVE_BATCH_EXPERT_KERNELS",
    "NATIVE_DSPARK_BATCH_VERIFY",
)

AGGREGATE_EXPERT_IO_KEYS = (
    "measurement_scope",
    "pread_calls",
    "bytes_requested",
    "bytes_read",
    "io_wait_ms",
    "hot_hits",
    "prefetch_hits",
    "batch_pool_hits",
    "cold_misses",
    "bytes_avoided",
    "short_reads",
)
AGGREGATE_DISPATCH_KEYS = (
    "measurement_scope",
    "dispatches",
    "command_buffers",
    "threadgroups",
    "scheduled_lanes",
    "active_lanes",
    "max_threads_per_threadgroup",
    "active_lane_utilization_proxy",
    "workspace_high_water_bytes",
    "attention_gpu_stage_ns",
    "peak_rss_bytes",
)


class ValidationError(ValueError):
    pass


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() in {"", "?", "unknown", "null"}:
        return None
    return float(value)


def _optional_int(value: str | None) -> int | None:
    if value is None or value.strip() in {"", "?", "unknown", "null"}:
        return None
    return int(value)


def _required_int(env: Mapping[str, str], name: str) -> int:
    value = _optional_int(env.get(name))
    if value is None:
        raise ValidationError(f"missing required integer environment variable: {name}")
    return value


def _required_float(env: Mapping[str, str], name: str) -> float:
    value = _optional_float(env.get(name))
    if value is None:
        raise ValidationError(f"missing required float environment variable: {name}")
    return value


def _csv_floats(value: str | None, name: str) -> list[float]:
    if not value:
        raise ValidationError(f"missing required CSV environment variable: {name}")
    try:
        return [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise ValidationError(f"invalid float CSV in {name}: {value}") from exc


def _csv_ints(value: str | None, name: str) -> list[int]:
    if not value:
        raise ValidationError(f"missing required CSV environment variable: {name}")
    try:
        return [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise ValidationError(f"invalid integer CSV in {name}: {value}") from exc


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{label} must be a nonnegative integer")
    return value


def _nullable_nonnegative(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{label} must be nonnegative or null")
    return float(value)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _load_telemetry_jsonl(value: str | None) -> list[dict[str, Any]] | None:
    if value is None or value.strip() in {"", "?", "unknown", "null"}:
        return None
    path = Path(value)
    if not path.is_file():
        raise ValidationError(f"BM_TELEMETRY_JSONL is not a file: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"invalid telemetry JSON on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise ValidationError(f"telemetry line {line_number} must be a JSON object")
        records.append(record)
    if len(records) != STANDARD_RUN_COUNT:
        raise ValidationError(
            f"exactly {STANDARD_RUN_COUNT} standard telemetry records are required, got {len(records)}"
        )
    return records


def _validate_layer_metrics(layer: Mapping[str, Any], label: str) -> None:
    layer_index = _nonnegative_int(layer.get("layer"), f"{label}.layer")
    if layer_index >= 43:
        raise ValidationError(f"{label}.layer must be below 43")
    calls = _nonnegative_int(layer.get("calls"), f"{label}.calls")
    if calls == 0:
        raise ValidationError(f"{label}.calls must be positive for an emitted layer")
    unique_experts = _nonnegative_int(layer.get("unique_experts"), f"{label}.unique_experts")
    if unique_experts > min(256, calls * 6):
        raise ValidationError(f"{label}.unique_experts exceeds the routed-expert bound")

    hot_hits = _nonnegative_int(layer.get("hot_hits"), f"{label}.hot_hits")
    prefetch_hits = _nonnegative_int(layer.get("prefetch_hits"), f"{label}.prefetch_hits")
    batch_pool_hits = _nonnegative_int(layer.get("batch_pool_hits"), f"{label}.batch_pool_hits")
    cold_misses = _nonnegative_int(layer.get("cold_misses"), f"{label}.cold_misses")
    if hot_hits + prefetch_hits + batch_pool_hits + cold_misses != calls * 6:
        raise ValidationError(f"{label} cache classes must partition calls * 6 routes")

    pread_calls = _nonnegative_int(layer.get("pread_calls"), f"{label}.pread_calls")
    bytes_requested = _nonnegative_int(
        layer.get("bytes_requested"), f"{label}.bytes_requested"
    )
    bytes_read = _nonnegative_int(layer.get("bytes_read"), f"{label}.bytes_read")
    short_reads = _nonnegative_int(layer.get("short_reads"), f"{label}.short_reads")
    _nonnegative_int(layer.get("io_wait_ns"), f"{label}.io_wait_ns")
    _nonnegative_int(layer.get("bytes_avoided"), f"{label}.bytes_avoided")
    if "cache_admissions" in layer:
        admissions = _nonnegative_int(
            layer.get("cache_admissions"), f"{label}.cache_admissions"
        )
        evictions = _nonnegative_int(
            layer.get("cache_evictions"), f"{label}.cache_evictions"
        )
        if admissions != evictions:
            raise ValidationError(f"{label} cache admissions/evictions mismatch")
    unique_cold_experts = cold_misses
    if "unique_cold_experts" in layer:
        unique_cold_experts = _nonnegative_int(
            layer.get("unique_cold_experts"), f"{label}.unique_cold_experts"
        )
        if unique_cold_experts > cold_misses:
            raise ValidationError(
                f"{label}.unique_cold_experts must not exceed cold_misses"
            )
    if pread_calls > unique_cold_experts:
        raise ValidationError(f"{label}.pread_calls exceeds unique cold experts")
    if short_reads != 0:
        raise ValidationError(f"{label}.short_reads must be zero")
    if bytes_read != bytes_requested:
        raise ValidationError(f"{label}.bytes_read must equal bytes_requested")
    if pread_calls == 0 and bytes_read != 0:
        raise ValidationError(f"{label} has bytes without a pread")

    dispatches = _nonnegative_int(layer.get("moe_dispatches"), f"{label}.moe_dispatches")
    command_buffers = _nonnegative_int(
        layer.get("moe_command_buffers"), f"{label}.moe_command_buffers"
    )
    threadgroups = _nonnegative_int(layer.get("threadgroups"), f"{label}.threadgroups")
    scheduled_lanes = _nonnegative_int(
        layer.get("scheduled_lanes"), f"{label}.scheduled_lanes"
    )
    active_lanes = _nonnegative_int(layer.get("active_lanes"), f"{label}.active_lanes")
    max_threads = _nonnegative_int(
        layer.get("max_threads_per_threadgroup"),
        f"{label}.max_threads_per_threadgroup",
    )
    if active_lanes > scheduled_lanes:
        raise ValidationError(f"{label}.active_lanes exceeds scheduled_lanes")
    if dispatches == 0 and any(
        value != 0
        for value in (threadgroups, scheduled_lanes, active_lanes, max_threads)
    ):
        raise ValidationError(f"{label} has dispatch geometry without dispatches")
    if calls > 0 and command_buffers == 0:
        raise ValidationError(f"{label} routed MoE calls require command buffers")


def _sum_phase_layers(telemetry: Mapping[str, Any], label: str) -> dict[str, int]:
    phases = telemetry.get("phases")
    if not isinstance(phases, list) or len(phases) != 3:
        raise ValidationError(f"{label}.phases must contain prefill, decode, and verify")
    expected_names = ["prefill", "decode", "verify"]
    sums = {
        "hot_hits": 0,
        "prefetch_hits": 0,
        "batch_pool_hits": 0,
        "cold_misses": 0,
        "pread_calls": 0,
        "bytes_requested": 0,
        "bytes_read": 0,
        "short_reads": 0,
        "io_wait_ns": 0,
        "bytes_avoided": 0,
        "dispatches": 0,
        "command_buffers": 0,
        "threadgroups": 0,
        "scheduled_lanes": 0,
        "active_lanes": 0,
        "max_threads_per_threadgroup": 0,
    }
    for phase_index, phase in enumerate(phases):
        phase_label = f"{label}.phases[{phase_index}]"
        if not isinstance(phase, Mapping) or phase.get("name") != expected_names[phase_index]:
            raise ValidationError(f"{phase_label} has an invalid phase name/order")
        layers = phase.get("layers")
        if not isinstance(layers, list):
            raise ValidationError(f"{phase_label}.layers must be a list")
        seen_layers: set[int] = set()
        for item_index, layer in enumerate(layers):
            item_label = f"{phase_label}.layers[{item_index}]"
            if not isinstance(layer, Mapping):
                raise ValidationError(f"{item_label} must be an object")
            _validate_layer_metrics(layer, item_label)
            layer_index = int(layer["layer"])
            if layer_index in seen_layers:
                raise ValidationError(f"{phase_label} repeats layer {layer_index}")
            seen_layers.add(layer_index)
            for key in (
                "hot_hits",
                "prefetch_hits",
                "batch_pool_hits",
                "cold_misses",
                "pread_calls",
                "bytes_requested",
                "bytes_read",
                "short_reads",
                "io_wait_ns",
                "bytes_avoided",
                "threadgroups",
                "scheduled_lanes",
                "active_lanes",
            ):
                sums[key] += int(layer[key])
            sums["dispatches"] += int(layer["moe_dispatches"])
            sums["command_buffers"] += int(layer["moe_command_buffers"])
            sums["max_threads_per_threadgroup"] = max(
                sums["max_threads_per_threadgroup"],
                int(layer["max_threads_per_threadgroup"]),
            )
    return sums


def validate_request_telemetry(
    telemetry: Mapping[str, Any], run: Mapping[str, Any], label: str
) -> None:
    if telemetry.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
        raise ValidationError(f"{label} has an unsupported schema_version")
    if telemetry.get("kind") != "dmlx-native-request-telemetry":
        raise ValidationError(f"{label}.kind is invalid")
    if telemetry.get("scope") != "native_engine_after_tokenization":
        raise ValidationError(f"{label}.scope is invalid")

    request = telemetry.get("request")
    if not isinstance(request, Mapping):
        raise ValidationError(f"{label}.request is required")
    prompt_tokens = _nonnegative_int(request.get("prompt_tokens"), f"{label}.request.prompt_tokens")
    generated_tokens = _nonnegative_int(
        request.get("generated_tokens"), f"{label}.request.generated_tokens"
    )
    http_prompt_tokens = run.get("prompt_tokens")
    if (
        not isinstance(http_prompt_tokens, int)
        or isinstance(http_prompt_tokens, bool)
        or prompt_tokens > http_prompt_tokens
        or http_prompt_tokens - prompt_tokens > 2
    ):
        raise ValidationError(
            f"{label} effective prompt count is inconsistent with HTTP template tokens"
        )
    if generated_tokens != run.get("completion_tokens"):
        raise ValidationError(f"{label} generated token count does not match the HTTP response")

    batch_path = telemetry.get("batch_path")
    if not isinstance(batch_path, Mapping):
        raise ValidationError(f"{label}.batch_path is required")
    _nonnegative_int(
        batch_path.get("scalar_layer_calls"),
        f"{label}.batch_path.scalar_layer_calls",
    )
    _nonnegative_int(
        batch_path.get("batched_moe_layers"),
        f"{label}.batch_path.batched_moe_layers",
    )

    cache_policy = telemetry.get("cache_policy")
    if cache_policy is not None:
        if not isinstance(cache_policy, Mapping):
            raise ValidationError(f"{label}.cache_policy must be an object")
        name = cache_policy.get("name")
        experimental = cache_policy.get("experimental")
        if name not in ("fixed_hot_observed", "lru_observed_hot_seeded"):
            raise ValidationError(f"{label}.cache_policy.name is invalid")
        if not isinstance(experimental, bool):
            raise ValidationError(f"{label}.cache_policy.experimental must be boolean")
        if experimental != (name == "lru_observed_hot_seeded"):
            raise ValidationError(f"{label}.cache_policy name/flag mismatch")
        capacity = _nonnegative_int(
            cache_policy.get("capacity_per_score_layer"),
            f"{label}.cache_policy.capacity_per_score_layer",
        )
        resident_slots = _nonnegative_int(
            cache_policy.get("resident_slots"),
            f"{label}.cache_policy.resident_slots",
        )
        resident_bytes = _nonnegative_int(
            cache_policy.get("resident_bytes"),
            f"{label}.cache_policy.resident_bytes",
        )
        _nonnegative_int(
            cache_policy.get("metadata_bytes"),
            f"{label}.cache_policy.metadata_bytes",
        )
        admissions = _nonnegative_int(
            cache_policy.get("admissions"),
            f"{label}.cache_policy.admissions",
        )
        evictions = _nonnegative_int(
            cache_policy.get("evictions"),
            f"{label}.cache_policy.evictions",
        )
        if resident_slots != 3 * 256 + 40 * capacity or resident_bytes <= 0:
            raise ValidationError(f"{label}.cache_policy residency geometry is invalid")
        if admissions != evictions:
            raise ValidationError(f"{label}.cache_policy admissions/evictions mismatch")
        if not experimental and (admissions != 0 or evictions != 0):
            raise ValidationError(f"{label} fixed-hot telemetry contains churn")

    dspark = telemetry.get("dspark_verification")
    if not isinstance(dspark, Mapping):
        raise ValidationError(f"{label}.dspark_verification is required")
    candidates = _nonnegative_int(
        dspark.get("candidate_count"),
        f"{label}.dspark_verification.candidate_count",
    )
    accepted = _nonnegative_int(
        dspark.get("accepted_count"),
        f"{label}.dspark_verification.accepted_count",
    )
    verification_batches = _nonnegative_int(
        dspark.get("verification_batches"),
        f"{label}.dspark_verification.verification_batches",
    )
    verification_ns = _nonnegative_int(
        dspark.get("verification_ns"),
        f"{label}.dspark_verification.verification_ns",
    )
    if accepted > candidates:
        raise ValidationError(f"{label} accepted more DSpark candidates than it verified")
    if candidates == 0:
        if accepted != 0 or verification_batches != 0 or verification_ns != 0:
            raise ValidationError(f"{label} has DSpark metrics without candidates")
    elif verification_batches == 0 or verification_ns == 0:
        raise ValidationError(f"{label} candidates lack DSpark batch/time evidence")

    start = _nonnegative_int(request.get("request_start_ns"), f"{label}.request.request_start_ns")
    first = _nonnegative_int(request.get("first_token_at_ns"), f"{label}.request.first_token_at_ns")
    decode_start = _nonnegative_int(
        request.get("decode_start_ns"), f"{label}.request.decode_start_ns"
    )
    end = _nonnegative_int(request.get("request_end_ns"), f"{label}.request.request_end_ns")
    engine_first_latency = _nonnegative_int(
        request.get("engine_first_token_latency_ns"),
        f"{label}.request.engine_first_token_latency_ns",
    )
    for unavailable in ("ttft_ns", "prefill_ns"):
        if unavailable not in request or request.get(unavailable) is not None:
            raise ValidationError(
                f"{label}.request.{unavailable} is unavailable and must be null"
            )
    decode = _nonnegative_int(request.get("decode_ns"), f"{label}.request.decode_ns")
    wall = _nonnegative_int(request.get("request_wall_ns"), f"{label}.request.request_wall_ns")
    itl = request.get("itl_ns")
    if not isinstance(itl, list) or len(itl) != max(0, generated_tokens - 1):
        raise ValidationError(f"{label}.request.itl_ns length is invalid")
    itl_values = [
        _nonnegative_int(value, f"{label}.request.itl_ns[{index}]")
        for index, value in enumerate(itl)
    ]
    last_token = first + sum(itl_values)
    if not (start <= first <= decode_start <= last_token <= end):
        raise ValidationError(f"{label} request phases are not monotonic/non-overlapping")
    if engine_first_latency != first - start:
        raise ValidationError(f"{label} engine first-token duration is inconsistent")
    if decode != last_token - decode_start:
        raise ValidationError(f"{label} decode duration is inconsistent")
    if wall != end - start:
        raise ValidationError(f"{label} request wall duration is inconsistent")

    sums = _sum_phase_layers(telemetry, label)
    expert_io = telemetry.get("expert_io")
    if not isinstance(expert_io, Mapping):
        raise ValidationError(f"{label}.expert_io is required")
    if expert_io.get("measurement_scope") != "critical_path_routed_expert_resolution":
        raise ValidationError(f"{label}.expert_io.measurement_scope is invalid")
    for key in (
        "hot_hits",
        "prefetch_hits",
        "batch_pool_hits",
        "cold_misses",
        "pread_calls",
        "bytes_requested",
        "bytes_read",
        "short_reads",
        "io_wait_ns",
        "bytes_avoided",
    ):
        value = _nonnegative_int(expert_io.get(key), f"{label}.expert_io.{key}")
        if value != sums[key]:
            raise ValidationError(f"{label}.expert_io.{key} does not match phase layers")
    if _nonnegative_int(expert_io.get("short_reads"), f"{label}.expert_io.short_reads") != 0:
        raise ValidationError(f"{label} contains a short expert read")
    if expert_io["bytes_read"] != expert_io["bytes_requested"]:
        raise ValidationError(f"{label} bytes_read must equal bytes_requested")

    dispatch = telemetry.get("dispatch")
    if not isinstance(dispatch, Mapping):
        raise ValidationError(f"{label}.dispatch is required")
    if dispatch.get("measurement_scope") != "routed_moe_only":
        raise ValidationError(f"{label}.dispatch.measurement_scope is invalid")
    for key in (
        "dispatches",
        "command_buffers",
        "threadgroups",
        "scheduled_lanes",
        "active_lanes",
        "max_threads_per_threadgroup",
    ):
        value = _nonnegative_int(dispatch.get(key), f"{label}.dispatch.{key}")
        if value != sums[key]:
            raise ValidationError(f"{label}.dispatch.{key} does not match phase layers")
    scheduled = int(dispatch["scheduled_lanes"])
    active = int(dispatch["active_lanes"])
    utilization = dispatch.get("active_lane_utilization_proxy")
    if scheduled > 0:
        if not isinstance(utilization, (int, float)) or isinstance(utilization, bool):
            raise ValidationError(f"{label}.dispatch utilization must be numeric")
        if abs(float(utilization) - active / scheduled) > 1e-8:
            raise ValidationError(f"{label}.dispatch utilization is inconsistent")
    elif utilization is not None:
        raise ValidationError(f"{label}.dispatch utilization must be null without lanes")
    workspace_high_water = dispatch.get("workspace_high_water_bytes")
    if workspace_high_water is not None:
        _nonnegative_int(
            workspace_high_water,
            f"{label}.dispatch.workspace_high_water_bytes",
        )
    peak_rss = dispatch.get("peak_rss_bytes")
    if peak_rss is not None:
        _nonnegative_int(peak_rss, f"{label}.dispatch.peak_rss_bytes")
    if dispatch.get("attention_gpu_stage_ns") is not None:
        raise ValidationError(
            f"{label}.dispatch.attention_gpu_stage_ns is unavailable and must be null"
        )

    try:
        summarize_telemetry([telemetry])
    except InvariantError as exc:
        raise ValidationError(f"{label}: {exc}") from exc


def _aggregate_telemetry(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requests = [record["request"] for record in records]
    itl_ms = [
        float(value) / 1_000_000.0
        for request in requests
        for value in request["itl_ns"]
    ]
    expert_keys = (
        "pread_calls",
        "bytes_requested",
        "bytes_read",
        "hot_hits",
        "prefetch_hits",
        "batch_pool_hits",
        "cold_misses",
        "bytes_avoided",
        "short_reads",
    )
    expert = {
        key: sum(int(record["expert_io"][key]) for record in records)
        for key in expert_keys
    }
    expert["measurement_scope"] = "critical_path_routed_expert_resolution"
    expert["io_wait_ms"] = sum(
        int(record["expert_io"]["io_wait_ns"]) for record in records
    ) / 1_000_000.0

    dispatch_sum_keys = (
        "dispatches",
        "command_buffers",
        "threadgroups",
        "scheduled_lanes",
        "active_lanes",
    )
    dispatch = {
        key: sum(int(record["dispatch"][key]) for record in records)
        for key in dispatch_sum_keys
    }
    dispatch["measurement_scope"] = "routed_moe_only"
    dispatch["max_threads_per_threadgroup"] = max(
        int(record["dispatch"]["max_threads_per_threadgroup"])
        for record in records
    )
    scheduled = dispatch["scheduled_lanes"]
    dispatch["active_lane_utilization_proxy"] = (
        dispatch["active_lanes"] / scheduled if scheduled > 0 else None
    )
    workspace_values = [
        int(record["dispatch"]["workspace_high_water_bytes"])
        for record in records
        if record["dispatch"].get("workspace_high_water_bytes") is not None
    ]
    rss_values = [
        int(record["dispatch"]["peak_rss_bytes"])
        for record in records
        if record["dispatch"].get("peak_rss_bytes") is not None
    ]
    dispatch["workspace_high_water_bytes"] = (
        max(workspace_values) if workspace_values else None
    )
    dispatch["attention_gpu_stage_ns"] = None
    dispatch["peak_rss_bytes"] = max(rss_values) if rss_values else None

    candidate_count = sum(
        int(record["dspark_verification"]["candidate_count"])
        for record in records
    )
    accepted_count = sum(
        int(record["dspark_verification"]["accepted_count"])
        for record in records
    )
    verification_batches = sum(
        int(record["dspark_verification"]["verification_batches"])
        for record in records
    )
    verification_ns = sum(
        int(record["dspark_verification"]["verification_ns"])
        for record in records
    )
    dspark_verification = {
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "verification_batches": verification_batches,
        "verification_ms": verification_ns / 1_000_000.0,
        "acceptance_rate": (
            accepted_count / candidate_count if candidate_count > 0 else None
        ),
        "mean_batch_latency_ms": (
            verification_ns / verification_batches / 1_000_000.0
            if verification_batches > 0
            else None
        ),
    }

    cache_records = [
        record.get("cache_policy") for record in records
        if record.get("cache_policy") is not None
    ]
    cache_policy = None
    if cache_records:
        identity_keys = (
            "name", "experimental", "capacity_per_score_layer",
            "resident_slots", "resident_bytes", "metadata_bytes",
        )
        reference = cache_records[0]
        if len(cache_records) != len(records) or any(
            any(item.get(key) != reference.get(key) for key in identity_keys)
            for item in cache_records[1:]
        ):
            raise ValidationError("standard telemetry cache policy metadata differs")
        cache_policy = {key: reference[key] for key in identity_keys}
        cache_policy["name"] = {
            "fixed_hot_observed": "fixed_hot",
            "lru_observed_hot_seeded": "bounded_lru",
        }[str(reference["name"])]
        cache_policy["admissions"] = sum(int(item["admissions"]) for item in cache_records)
        cache_policy["evictions"] = sum(int(item["evictions"]) for item in cache_records)

    return {
        # True HTTP TTFT and an independently bounded prefill phase are not
        # available from the native-engine-after-tokenization scope.
        "ttft_ms": None,
        "prefill_ms": None,
        "first_token_ms": None,
        "peak_rss_bytes": dispatch["peak_rss_bytes"],
        "engine_first_token_ms": statistics.median(
            float(request["engine_first_token_latency_ns"]) / 1_000_000.0
            for request in requests
        ),
        "steady_state_decode_ms": statistics.median(
            float(request["decode_ns"]) / 1_000_000.0 for request in requests
        ),
        "itl_ms": {
            "p50": _percentile(itl_ms, 0.50),
            "p95": _percentile(itl_ms, 0.95),
            "max": max(itl_ms) if itl_ms else None,
        },
        "expert_io": expert,
        "expert_io_invariants": summarize_telemetry(records),
        "cache_policy": cache_policy,
        "dispatch": dispatch,
        "dspark_verification": dspark_verification,
    }


def build_document(env: Mapping[str, str]) -> dict[str, Any]:
    standard_tps = _csv_floats(env.get("BM_STANDARD_TPS_CSV"), "BM_STANDARD_TPS_CSV")
    standard_elapsed = _csv_floats(
        env.get("BM_STANDARD_ELAPSED_CSV"), "BM_STANDARD_ELAPSED_CSV"
    )
    standard_prompt_tokens = _csv_ints(
        env.get("BM_STANDARD_PROMPT_TOKENS_CSV"), "BM_STANDARD_PROMPT_TOKENS_CSV"
    )
    standard_completion_tokens = _csv_ints(
        env.get("BM_STANDARD_TOKENS_CSV"), "BM_STANDARD_TOKENS_CSV"
    )
    lengths = {
        len(standard_tps),
        len(standard_elapsed),
        len(standard_prompt_tokens),
        len(standard_completion_tokens),
    }
    if len(lengths) != 1 or next(iter(lengths)) != STANDARD_RUN_COUNT:
        raise ValidationError("exactly three standard run values are required in every CSV")

    telemetry_records = _load_telemetry_jsonl(env.get("BM_TELEMETRY_JSONL"))
    standard_runs: list[dict[str, Any]] = []
    for index, (prompt_tokens, completion_tokens, elapsed, tps) in enumerate(
        zip(
            standard_prompt_tokens,
            standard_completion_tokens,
            standard_elapsed,
            standard_tps,
        )
    ):
        run = {
            "run_index": index + 1,
            "requested_max_tokens": 5,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "wall_seconds": elapsed,
            "tokens_per_second": tps,
            "telemetry": telemetry_records[index] if telemetry_records is not None else None,
        }
        if telemetry_records is not None:
            validate_request_telemetry(
                telemetry_records[index], run, f"standard run {index + 1} telemetry"
            )
        standard_runs.append(run)

    startup_ms = _optional_float(env.get("BM_STARTUP_MS"))
    if startup_ms is None:
        startup_seconds = _optional_float(env.get("BM_STARTUP_SECS"))
        startup_ms = startup_seconds * 1000.0 if startup_seconds is not None else None
    perf_seconds = _optional_float(env.get("BM_PERF_SECS"))
    total_seconds = _optional_float(env.get("BM_TOTAL_SECS"))

    measured = _aggregate_telemetry(telemetry_records) if telemetry_records else None
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "date_local": env.get("BM_DATE") or None,
            "mode": env.get("BM_MODE", "native"),
            "model_path": env.get("BM_MODEL_PATH") or None,
            "packed_experts_path": env.get("BM_PACKED_DIR") or None,
            "dspark_weights_path": env.get("BM_DSPARK_DIR") or None,
            "model_fingerprint": env.get("BM_MODEL_FINGERPRINT") or None,
            "workspace_revision": env.get("BM_COMMIT") or None,
            "branch": env.get("BM_BRANCH") or None,
            "command": env.get("BM_COMMAND") or None,
            "environment": {
                "NATIVE_SMELT_N": _optional_int(env.get("BM_SMELT_N")),
                "NATIVE_TELEMETRY": _optional_int(env.get("BM_TELEMETRY_ENABLED")),
                "NATIVE_EXPERIMENTAL_CACHE": _optional_int(
                    env.get("BM_EXPERIMENTAL_CACHE")
                ) or 0,
                "NATIVE_BATCH_PREFILL": _required_int(env, "BM_BATCH_PREFILL"),
                "NATIVE_BATCH_MLA": _required_int(env, "BM_BATCH_MLA"),
                "NATIVE_EXPERT_BUCKETS": _required_int(env, "BM_EXPERT_BUCKETS"),
                "NATIVE_BATCH_EXPERT_KERNELS": _required_int(
                    env, "BM_BATCH_EXPERT_KERNELS"
                ),
                "NATIVE_DSPARK_BATCH_VERIFY": _required_int(
                    env, "BM_DSPARK_BATCH_VERIFY"
                ),
                "NATIVE_PREFETCH": _optional_int(env.get("BM_PREFETCH")) or 0,
                "NATIVE_SMELT_PREFETCH": _optional_int(env.get("BM_SMELT_PREFETCH")) or 0,
                "SKIP_PURGE": _optional_int(env.get("BM_SKIP_PURGE")) or 0,
            },
            "machine": {
                "hardware": env.get("BM_HW") or None,
                "memory": env.get("BM_MEM") or None,
            },
            "warmup": {
                "initial_requests": _required_int(env, "BM_WARMUP_INITIAL"),
                "rewarm_requests": _required_int(env, "BM_WARMUP_REWARM"),
            },
            "measured_requests": len(standard_runs),
            "unit_tests": env.get("BM_UNIT") or None,
        },
        "phases": {
            "standard_decode": {
                "role": "standard",
                "requested_max_tokens": 5,
                "included_in_standard_median": True,
                "runs": standard_runs,
                "median_tokens_per_second": statistics.median(standard_tps),
            },
            "long_decode_diagnostic": {
                "role": "diagnostic",
                "requested_max_tokens": 100,
                "included_in_standard_median": False,
                "run": {
                    "prompt_tokens": _required_int(env, "BM_LONG_PROMPT_TOKENS"),
                    "completion_tokens": _required_int(env, "BM_LONG_TOKENS"),
                    "wall_seconds": _required_float(env, "BM_LONG_ELAPSED"),
                    "tokens_per_second": _required_float(env, "BM_NATIVE_DECODE_TPS"),
                },
            },
            "correctness": {
                "paris_pass": env.get("BM_NATIVE_CORRECT") == "1",
                "e2e_pass": _required_int(env, "BM_E2E_PASS"),
                "e2e_fail": _required_int(env, "BM_E2E_FAIL"),
            },
        },
        "metrics": {
            "startup_ms": startup_ms,
            "ttft_ms": measured["ttft_ms"] if measured else None,
            "prefill_ms": measured["prefill_ms"] if measured else None,
            "first_token_ms": measured["first_token_ms"] if measured else None,
            "engine_first_token_ms": measured["engine_first_token_ms"] if measured else None,
            "steady_state_decode_ms": measured["steady_state_decode_ms"] if measured else None,
            "end_to_end_wall_ms": total_seconds * 1000.0 if total_seconds is not None else None,
            "performance_phase_wall_ms": perf_seconds * 1000.0 if perf_seconds is not None else None,
            "peak_rss_bytes": measured["peak_rss_bytes"] if measured else None,
            "itl_ms": measured["itl_ms"] if measured else {"p50": None, "p95": None, "max": None},
            "expert_io": measured["expert_io"] if measured else {
                "measurement_scope": None,
                "pread_calls": None,
                "bytes_requested": None,
                "bytes_read": None,
                "io_wait_ms": None,
                "hot_hits": None,
                "prefetch_hits": None,
                "batch_pool_hits": None,
                "cold_misses": None,
                "bytes_avoided": None,
                "short_reads": None,
            },
            "expert_io_invariants": measured["expert_io_invariants"] if measured else None,
            "cache_policy": measured["cache_policy"] if measured else None,
            "dspark_verification": measured["dspark_verification"] if measured else {
                "candidate_count": None,
                "accepted_count": None,
                "verification_batches": None,
                "verification_ms": None,
                "acceptance_rate": None,
                "mean_batch_latency_ms": None,
            },
            "dispatch": measured["dispatch"] if measured else {
                "measurement_scope": None,
                "dispatches": None,
                "command_buffers": None,
                "threadgroups": None,
                "scheduled_lanes": None,
                "active_lanes": None,
                "max_threads_per_threadgroup": None,
                "active_lane_utilization_proxy": None,
                "workspace_high_water_bytes": None,
                "attention_gpu_stage_ns": None,
                "peak_rss_bytes": None,
            },
        },
    }
    validate_document(document)
    return document


def _validate_rate(label: str, tokens: Any, elapsed: Any, tps: Any) -> float:
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValidationError(f"{label} completion_tokens must be a positive integer")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed <= 0:
        raise ValidationError(f"{label} wall_seconds must be positive")
    if not isinstance(tps, (int, float)) or isinstance(tps, bool) or tps <= 0:
        raise ValidationError(f"{label} tokens_per_second must be positive")
    expected = tokens / float(elapsed)
    if abs(float(tps) - expected) > max(0.001, expected * 0.001):
        raise ValidationError(f"{label} timing is inconsistent with completion_tokens / wall_seconds")
    return float(tps)


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValidationError(f"{label} must be null")
        return
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        raise ValidationError(f"{label} must be numeric")
    if abs(float(actual) - float(expected)) > max(1e-8, abs(float(expected)) * 1e-9):
        raise ValidationError(f"{label} does not match standard telemetry")


def validate_document(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("unsupported benchmark schema_version")

    run = document.get("run")
    phases = document.get("phases")
    metrics = document.get("metrics")
    if not isinstance(run, Mapping) or not isinstance(phases, Mapping) or not isinstance(metrics, Mapping):
        raise ValidationError("run, phases, and metrics objects are required")

    timestamp = run.get("timestamp_utc")
    if not isinstance(timestamp, str):
        raise ValidationError("run.timestamp_utc is required")
    try:
        datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValidationError("run.timestamp_utc must be ISO-8601") from exc

    environment = run.get("environment")
    if not isinstance(environment, Mapping):
        raise ValidationError("run.environment is required")
    smelt_n = environment.get("NATIVE_SMELT_N")
    if not isinstance(smelt_n, int) or isinstance(smelt_n, bool) or smelt_n <= 0:
        raise ValidationError("run.environment.NATIVE_SMELT_N must be positive")
    if environment.get("NATIVE_TELEMETRY") != 1:
        raise ValidationError("run.environment.NATIVE_TELEMETRY must be 1")
    experimental_cache = environment.get("NATIVE_EXPERIMENTAL_CACHE", 0)
    if experimental_cache not in (0, 1):
        raise ValidationError(
            "run.environment.NATIVE_EXPERIMENTAL_CACHE must be 0 or 1"
        )
    for key in BATCH_ENVIRONMENT_KEYS:
        if environment.get(key) not in (0, 1):
            raise ValidationError(f"run.environment.{key} must be 0 or 1")
    fully_batched_prefill = all(
        environment[key] == 1
        for key in (
            "NATIVE_BATCH_PREFILL",
            "NATIVE_BATCH_MLA",
            "NATIVE_EXPERT_BUCKETS",
            "NATIVE_BATCH_EXPERT_KERNELS",
        )
    )
    dspark_enabled = environment["NATIVE_DSPARK_BATCH_VERIFY"] == 1
    if experimental_cache == 1 and any(
        environment[key] == 1 for key in BATCH_ENVIRONMENT_KEYS
    ):
        raise ValidationError(
            "experimental cache is incompatible with enabled batch paths"
        )
    if dspark_enabled and not all(
        environment[key] == 1
        for key in (
            "NATIVE_BATCH_MLA",
            "NATIVE_EXPERT_BUCKETS",
            "NATIVE_BATCH_EXPERT_KERNELS",
        )
    ):
        raise ValidationError(
            "enabled DSpark verification requires MLA, buckets and expert kernels"
        )
    enabled_batched_path = fully_batched_prefill or dspark_enabled
    dspark_path = run.get("dspark_weights_path")
    if dspark_enabled:
        if not isinstance(dspark_path, str) or not dspark_path:
            raise ValidationError(
                "enabled DSpark verification requires run.dspark_weights_path"
            )
        if Path(dspark_path).name != "dspark_weights":
            raise ValidationError(
                "enabled DSpark verification requires a dspark_weights directory"
            )
    elif dspark_path is not None:
        if not isinstance(dspark_path, str) or not dspark_path:
            raise ValidationError("run.dspark_weights_path must be a nonempty string or null")

    standard = phases.get("standard_decode")
    diagnostic = phases.get("long_decode_diagnostic")
    if not isinstance(standard, Mapping) or not isinstance(diagnostic, Mapping):
        raise ValidationError("standard and diagnostic phases are required")
    if standard.get("role") != "standard":
        raise ValidationError("standard_decode role must be standard")
    if standard.get("requested_max_tokens") != 5 or standard.get("included_in_standard_median") is not True:
        raise ValidationError("standard_decode must be the 5-token median source")
    if diagnostic.get("role") != "diagnostic":
        raise ValidationError("long_decode_diagnostic role must be diagnostic")
    if diagnostic.get("requested_max_tokens") != 100 or diagnostic.get("included_in_standard_median") is not False:
        raise ValidationError("long decode must be a separate 100-token diagnostic")

    runs = standard.get("runs")
    if not isinstance(runs, list) or len(runs) != STANDARD_RUN_COUNT:
        raise ValidationError("exactly three standard runs are required")
    if run.get("measured_requests") != len(runs):
        raise ValidationError("run.measured_requests does not match standard runs")

    tps_values: list[float] = []
    telemetry_records: list[Mapping[str, Any]] = []
    telemetry_presence: list[bool] = []
    for index, item in enumerate(runs, start=1):
        if not isinstance(item, Mapping):
            raise ValidationError("standard run must be an object")
        if item.get("run_index") != index or item.get("requested_max_tokens") != 5:
            raise ValidationError("standard run index/requested token count is invalid")
        prompt_tokens = item.get("prompt_tokens")
        if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens <= 0:
            raise ValidationError("standard prompt_tokens must be a positive integer")
        if item.get("completion_tokens") != 5:
            raise ValidationError("standard runs must each produce exactly 5 completion tokens")
        tps_values.append(
            _validate_rate(
                f"standard run {index}",
                item.get("completion_tokens"),
                item.get("wall_seconds"),
                item.get("tokens_per_second"),
            )
        )
        telemetry = item.get("telemetry")
        telemetry_presence.append(telemetry is not None)
        if telemetry is not None:
            if not isinstance(telemetry, Mapping):
                raise ValidationError(f"standard run {index} telemetry must be an object or null")
            validate_request_telemetry(telemetry, item, f"standard run {index} telemetry")
            batch_path = telemetry["batch_path"]
            if enabled_batched_path:
                if (
                    batch_path["scalar_layer_calls"] != 0
                    or batch_path["batched_moe_layers"] <= 0
                ):
                    raise ValidationError(
                        f"standard run {index} did not prove the enabled batched path"
                    )
            elif batch_path["batched_moe_layers"] != 0:
                raise ValidationError(
                    f"standard run {index} used batched MoE while required flags were disabled"
                )
            dspark_metrics = telemetry["dspark_verification"]
            if dspark_enabled:
                if (
                    dspark_metrics["candidate_count"] <= 0
                    or dspark_metrics["verification_batches"] <= 0
                    or dspark_metrics["verification_ns"] <= 0
                ):
                    raise ValidationError(
                        f"standard run {index} did not prove the enabled DSpark verification path"
                    )
            elif any(
                dspark_metrics[key] != 0
                for key in (
                    "candidate_count",
                    "accepted_count",
                    "verification_batches",
                    "verification_ns",
                )
            ):
                raise ValidationError(
                    f"standard run {index} used DSpark verification while its flag was disabled"
                )
            telemetry_records.append(telemetry)
    if any(telemetry_presence) and not all(telemetry_presence):
        raise ValidationError("standard telemetry must be present for all three runs or none")

    expected_median = statistics.median(tps_values)
    median = standard.get("median_tokens_per_second")
    if not isinstance(median, (int, float)) or abs(float(median) - expected_median) > 1e-9:
        raise ValidationError("standard median does not match standard runs")

    diagnostic_run = diagnostic.get("run")
    if not isinstance(diagnostic_run, Mapping):
        raise ValidationError("long_decode_diagnostic.run is required")
    prompt_tokens = diagnostic_run.get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens <= 0:
        raise ValidationError("diagnostic prompt_tokens must be a positive integer")
    completion_tokens = diagnostic_run.get("completion_tokens")
    if not isinstance(completion_tokens, int) or isinstance(completion_tokens, bool) or not (0 < completion_tokens <= 100):
        raise ValidationError("diagnostic completion_tokens must be in [1, 100]")
    _validate_rate(
        "long decode diagnostic",
        completion_tokens,
        diagnostic_run.get("wall_seconds"),
        diagnostic_run.get("tokens_per_second"),
    )

    for name in ("startup_ms", "end_to_end_wall_ms", "performance_phase_wall_ms", "peak_rss_bytes"):
        _nullable_nonnegative(metrics.get(name), name)

    measured = _aggregate_telemetry(telemetry_records) if telemetry_records else None
    for name in (
        "ttft_ms",
        "prefill_ms",
        "first_token_ms",
        "peak_rss_bytes",
        "engine_first_token_ms",
        "steady_state_decode_ms",
    ):
        _assert_close(metrics.get(name), measured[name] if measured else None, name)

    itl = metrics.get("itl_ms")
    if not isinstance(itl, Mapping):
        raise ValidationError("metrics.itl_ms is required")
    expected_itl = measured["itl_ms"] if measured else {"p50": None, "p95": None, "max": None}
    for key in ("p50", "p95", "max"):
        _assert_close(itl.get(key), expected_itl[key], f"itl_ms.{key}")

    expert_io = metrics.get("expert_io")
    dispatch = metrics.get("dispatch")
    dspark_metrics = metrics.get("dspark_verification")
    if (
        not isinstance(expert_io, Mapping)
        or not isinstance(dispatch, Mapping)
        or not isinstance(dspark_metrics, Mapping)
    ):
        raise ValidationError(
            "metrics.expert_io, metrics.dispatch and metrics.dspark_verification are required"
        )
    expected_invariants = measured["expert_io_invariants"] if measured else None
    if metrics.get("expert_io_invariants") != expected_invariants:
        raise ValidationError(
            "expert_io_invariants does not match standard telemetry"
        )
    if measured is None:
        for label, aggregate, required_keys in (
            ("expert_io", expert_io, AGGREGATE_EXPERT_IO_KEYS),
            ("dispatch", dispatch, AGGREGATE_DISPATCH_KEYS),
            (
                "dspark_verification",
                dspark_metrics,
                (
                    "candidate_count",
                    "accepted_count",
                    "verification_batches",
                    "verification_ms",
                    "acceptance_rate",
                    "mean_batch_latency_ms",
                ),
            ),
        ):
            for key in required_keys:
                if key not in aggregate:
                    raise ValidationError(
                        f"{label}.{key} is required when telemetry is unavailable"
                    )
                if aggregate[key] is not None:
                    raise ValidationError(
                        f"{label}.{key} must be null when telemetry is unavailable"
                    )
    else:
        expected_expert = measured["expert_io"]
        expected_dispatch = measured["dispatch"]
        for key in expected_expert:
            expected = expected_expert[key]
            actual = expert_io.get(key)
            if isinstance(expected, str) or expected is None:
                if actual != expected:
                    raise ValidationError(f"expert_io.{key} does not match telemetry availability")
            else:
                _assert_close(actual, expected, f"expert_io.{key}")
        for key in expected_dispatch:
            expected = expected_dispatch[key]
            actual = dispatch.get(key)
            if isinstance(expected, str) or expected is None:
                if actual != expected:
                    raise ValidationError(f"dispatch.{key} does not match telemetry availability")
            else:
                _assert_close(actual, expected, f"dispatch.{key}")
        expected_dspark = measured["dspark_verification"]
        for key, expected in expected_dspark.items():
            _assert_close(
                dspark_metrics.get(key),
                expected,
                f"dspark_verification.{key}",
            )


def write_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_canonical_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _sample_layer(layer: int, calls: int, hot: int, cold: int, scale: int) -> dict[str, int]:
    dispatches = calls * 3
    threadgroups = calls * 10
    scheduled = threadgroups * 256
    active = scheduled - calls * 16
    bytes_requested = cold * 13_369_344
    return {
        "layer": layer,
        "calls": calls,
        "unique_experts": min(256, calls * 6),
        "hot_hits": hot,
        "prefetch_hits": 0,
        "batch_pool_hits": 0,
        "cold_misses": cold,
        "pread_calls": cold,
        "bytes_requested": bytes_requested,
        "bytes_read": bytes_requested,
        "short_reads": 0,
        "io_wait_ns": cold * 1_000_000 * scale,
        "bytes_avoided": hot * 13_369_344,
        "moe_dispatches": dispatches,
        "moe_command_buffers": calls,
        "threadgroups": threadgroups,
        "scheduled_lanes": scheduled,
        "active_lanes": active,
        "max_threads_per_threadgroup": 256,
    }


def _sample_telemetry(index: int) -> dict[str, Any]:
    base = 1_000_000_000_000 + index * 10_000_000_000
    first = base + (1_000_000_000 + index * 10_000_000)
    decode_start = first + 20_000_000
    itl = [600_000_000, 500_000_000, 500_000_000, 400_000_000]
    last = first + sum(itl)
    end = last + 30_000_000
    prefill_layer = _sample_layer(0, 1, 6, 0, index + 1)
    decode_layer = _sample_layer(0, 4, 20, 4, index + 1)
    layers = [prefill_layer, decode_layer]

    def total(key: str) -> int:
        return sum(int(layer[key]) for layer in layers)

    scheduled = total("scheduled_lanes")
    active = total("active_lanes")
    return {
        "schema_version": 1,
        "kind": "dmlx-native-request-telemetry",
        "scope": "native_engine_after_tokenization",
        "batch_path": {
            "scalar_layer_calls": 0,
            "batched_moe_layers": 0,
        },
        "dspark_verification": {
            "candidate_count": 0,
            "accepted_count": 0,
            "verification_batches": 0,
            "verification_ns": 0,
        },
        "request": {
            "prompt_tokens": 8,
            "generated_tokens": 5,
            "request_start_ns": base,
            "first_token_at_ns": first,
            "engine_first_token_latency_ns": first - base,
            "ttft_ns": None,
            "prefill_ns": None,
            "decode_start_ns": decode_start,
            "decode_ns": last - decode_start,
            "request_end_ns": end,
            "request_wall_ns": end - base,
            "itl_ns": itl,
        },
        "expert_io": {
            "measurement_scope": "critical_path_routed_expert_resolution",
            "pread_calls": total("pread_calls"),
            "bytes_requested": total("bytes_requested"),
            "bytes_read": total("bytes_read"),
            "io_wait_ns": total("io_wait_ns"),
            "hot_hits": total("hot_hits"),
            "prefetch_hits": total("prefetch_hits"),
            "batch_pool_hits": total("batch_pool_hits"),
            "cold_misses": total("cold_misses"),
            "bytes_avoided": total("bytes_avoided"),
            "short_reads": 0,
        },
        "dispatch": {
            "measurement_scope": "routed_moe_only",
            "dispatches": total("moe_dispatches"),
            "command_buffers": total("moe_command_buffers"),
            "threadgroups": total("threadgroups"),
            "scheduled_lanes": scheduled,
            "active_lanes": active,
            "max_threads_per_threadgroup": 256,
            "active_lane_utilization_proxy": active / scheduled,
            "workspace_high_water_bytes": 12_582_912 + index * 4096,
            "attention_gpu_stage_ns": None,
            "peak_rss_bytes": 32 * 1024 * 1024 * 1024 + index * 1024,
        },
        "phases": [
            {"name": "prefill", "layers": [prefill_layer]},
            {"name": "decode", "layers": [decode_layer]},
            {"name": "verify", "layers": []},
        ],
    }


def _expect_validation_error(action: Any, message: str) -> None:
    try:
        action()
    except ValidationError:
        return
    raise AssertionError(message)


def self_test() -> None:
    env = {
        "BM_DATE": "2026-01-01",
        "BM_MODE": "native",
        "BM_MODEL_PATH": "/model",
        "BM_PACKED_DIR": "/model/packed_experts",
        "BM_COMMIT": "abc123",
        "BM_BRANCH": "main",
        "BM_COMMAND": "NATIVE_SMELT_N=20 bash scripts/run_benchmark.sh",
        "BM_HW": "Apple test",
        "BM_MEM": "48GB",
        "BM_SMELT_N": "20",
        "BM_TELEMETRY_ENABLED": "1",
        "BM_BATCH_PREFILL": "0",
        "BM_BATCH_MLA": "0",
        "BM_EXPERT_BUCKETS": "0",
        "BM_BATCH_EXPERT_KERNELS": "0",
        "BM_DSPARK_BATCH_VERIFY": "0",
        "BM_UNIT": "PASS",
        "BM_WARMUP_INITIAL": "5",
        "BM_WARMUP_REWARM": "4",
        "BM_STANDARD_TPS_CSV": "1.111111,1.250000,1.315789",
        "BM_STANDARD_ELAPSED_CSV": "4.5,4.0,3.8",
        "BM_STANDARD_PROMPT_TOKENS_CSV": "8,8,8",
        "BM_STANDARD_TOKENS_CSV": "5,5,5",
        "BM_LONG_PROMPT_TOKENS": "14",
        "BM_LONG_TOKENS": "100",
        "BM_LONG_ELAPSED": "70.0",
        "BM_NATIVE_DECODE_TPS": "1.428571",
        "BM_NATIVE_CORRECT": "1",
        "BM_E2E_PASS": "7",
        "BM_E2E_FAIL": "0",
        "BM_STARTUP_MS": "9123.5",
        "BM_PERF_SECS": "153",
        "BM_TOTAL_SECS": "245",
    }

    disabled = build_document(env)
    validate_document(json.loads(json.dumps(disabled)))
    if disabled["metrics"]["ttft_ms"] is not None:
        raise AssertionError("disabled telemetry produced measured timing")
    if disabled["metrics"]["expert_io_invariants"] is not None:
        raise AssertionError("disabled telemetry produced an invariant summary")

    missing_disabled_key = json.loads(json.dumps(disabled))
    del missing_disabled_key["metrics"]["expert_io"]["pread_calls"]
    _expect_validation_error(
        lambda: validate_document(missing_disabled_key),
        "validator accepted a missing disabled expert I/O field",
    )

    nonnull_disabled_aggregate = json.loads(json.dumps(disabled))
    nonnull_disabled_aggregate["metrics"]["dispatch"]["dispatches"] = 1
    _expect_validation_error(
        lambda: validate_document(nonnull_disabled_aggregate),
        "validator accepted a non-null disabled dispatch field",
    )

    with tempfile.TemporaryDirectory() as directory:
        telemetry_path = Path(directory) / "telemetry.jsonl"
        records = [_sample_telemetry(index) for index in range(3)]
        telemetry_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        enabled_env = dict(env, BM_TELEMETRY_JSONL=str(telemetry_path), BM_TELEMETRY_ENABLED="1")
        enabled = build_document(enabled_env)
        validate_document(json.loads(json.dumps(enabled)))
        if enabled["metrics"]["engine_first_token_ms"] is None:
            raise AssertionError("enabled telemetry did not produce engine first-token timing")
        if enabled["metrics"]["expert_io_invariants"] is None:
            raise AssertionError("enabled telemetry did not produce an invariant summary")
        if enabled["metrics"]["peak_rss_bytes"] is None:
            raise AssertionError("enabled telemetry did not preserve peak RSS")
        if enabled["metrics"]["dispatch"]["workspace_high_water_bytes"] is None:
            raise AssertionError("enabled telemetry did not preserve workspace high-water")

        invalid_flag = json.loads(json.dumps(enabled))
        invalid_flag["run"]["environment"]["NATIVE_BATCH_PREFILL"] = 2
        _expect_validation_error(
            lambda: validate_document(invalid_flag),
            "validator accepted a non-boolean effective batch flag",
        )

        mismatched_enabled_path = json.loads(json.dumps(enabled))
        for key in (
            "NATIVE_BATCH_PREFILL",
            "NATIVE_BATCH_MLA",
            "NATIVE_EXPERT_BUCKETS",
            "NATIVE_BATCH_EXPERT_KERNELS",
        ):
            mismatched_enabled_path["run"]["environment"][key] = 1
        _expect_validation_error(
            lambda: validate_document(mismatched_enabled_path),
            "validator accepted enabled flags without batched path counters",
        )

        enabled_batched = json.loads(json.dumps(mismatched_enabled_path))
        for run in enabled_batched["phases"]["standard_decode"]["runs"]:
            run["telemetry"]["batch_path"] = {
                "scalar_layer_calls": 0,
                "batched_moe_layers": 43,
            }
        validate_document(enabled_batched)

        dspark_records = json.loads(json.dumps(records))
        for record in dspark_records:
            record["batch_path"] = {
                "scalar_layer_calls": 0,
                "batched_moe_layers": 43,
            }
            record["dspark_verification"] = {
                "candidate_count": 4,
                "accepted_count": 2,
                "verification_batches": 1,
                "verification_ns": 2_000_000,
            }
        telemetry_path.write_text(
            "".join(json.dumps(record) + "\n" for record in dspark_records),
            encoding="utf-8",
        )
        dspark_env = dict(
            enabled_env,
            BM_BATCH_MLA="1",
            BM_EXPERT_BUCKETS="1",
            BM_BATCH_EXPERT_KERNELS="1",
            BM_DSPARK_BATCH_VERIFY="1",
            BM_DSPARK_DIR="/model/dspark_weights",
        )
        dspark_document = build_document(dspark_env)
        validate_document(json.loads(json.dumps(dspark_document)))
        if dspark_document["metrics"]["dspark_verification"]["candidate_count"] != 12:
            raise AssertionError("DSpark candidate aggregate is incorrect")

        accepted_overflow = json.loads(json.dumps(dspark_document))
        accepted_overflow["phases"]["standard_decode"]["runs"][0]["telemetry"][
            "dspark_verification"
        ]["accepted_count"] = 5
        _expect_validation_error(
            lambda: validate_document(accepted_overflow),
            "validator accepted DSpark accepted_count > candidate_count",
        )

        missing_dspark_path = json.loads(json.dumps(dspark_document))
        missing_dspark_path["run"]["dspark_weights_path"] = None
        _expect_validation_error(
            lambda: validate_document(missing_dspark_path),
            "validator accepted enabled DSpark without an explicit weights path",
        )

        wrong_dspark_basename = json.loads(json.dumps(dspark_document))
        wrong_dspark_basename["run"]["dspark_weights_path"] = "/model/not_weights"
        _expect_validation_error(
            lambda: validate_document(wrong_dspark_basename),
            "validator accepted enabled DSpark with the wrong weights basename",
        )

        no_dspark_evidence = json.loads(json.dumps(dspark_document))
        for run in no_dspark_evidence["phases"]["standard_decode"]["runs"]:
            run["telemetry"]["dspark_verification"] = {
                "candidate_count": 0,
                "accepted_count": 0,
                "verification_batches": 0,
                "verification_ns": 0,
            }
        _expect_validation_error(
            lambda: validate_document(no_dspark_evidence),
            "validator accepted enabled DSpark without path evidence",
        )

        # Restore the default-off records for the remaining malformed-input tests.
        telemetry_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

        stale_summary = json.loads(json.dumps(enabled))
        stale_summary["metrics"]["expert_io_invariants"]["totals"]["pread_calls"] += 1
        _expect_validation_error(
            lambda: validate_document(stale_summary),
            "validator accepted a stale expert I/O invariant summary",
        )
        for unavailable in ("ttft_ms", "prefill_ms", "first_token_ms"):
            if enabled["metrics"][unavailable] is not None:
                raise AssertionError(f"enabled telemetry fabricated unavailable {unavailable}")

        malformed_path = Path(directory) / "malformed.jsonl"
        malformed_path.write_text("{not-json}\n", encoding="utf-8")
        _expect_validation_error(
            lambda: build_document(dict(env, BM_TELEMETRY_JSONL=str(malformed_path))),
            "parser accepted malformed telemetry",
        )

        nonmonotonic = json.loads(json.dumps(records))
        nonmonotonic[0]["request"]["decode_start_ns"] = (
            nonmonotonic[0]["request"]["first_token_at_ns"] - 1
        )
        telemetry_path.write_text(
            "".join(json.dumps(record) + "\n" for record in nonmonotonic),
            encoding="utf-8",
        )
        _expect_validation_error(
            lambda: build_document(enabled_env),
            "validator accepted nonmonotonic request phases",
        )

        false_timing = json.loads(json.dumps(records))
        false_timing[0]["request"]["ttft_ns"] = 1
        false_timing[0]["request"]["prefill_ns"] = 1
        telemetry_path.write_text(
            "".join(json.dumps(record) + "\n" for record in false_timing),
            encoding="utf-8",
        )
        _expect_validation_error(
            lambda: build_document(enabled_env),
            "validator accepted fabricated TTFT/prefill timing",
        )

        invalid_io = json.loads(json.dumps(records))
        invalid_io[0]["expert_io"]["pread_calls"] += 1
        telemetry_path.write_text(
            "".join(json.dumps(record) + "\n" for record in invalid_io),
            encoding="utf-8",
        )
        _expect_validation_error(
            lambda: build_document(enabled_env),
            "validator accepted invalid expert I/O totals",
        )

        short_read = json.loads(json.dumps(records))
        short_read[0]["phases"][0]["layers"][0]["short_reads"] = 1
        short_read[0]["expert_io"]["short_reads"] = 1
        telemetry_path.write_text(
            "".join(json.dumps(record) + "\n" for record in short_read),
            encoding="utf-8",
        )
        _expect_validation_error(
            lambda: build_document(enabled_env),
            "validator accepted a reconciled nonzero expert short-read count",
        )

    invalid = json.loads(json.dumps(disabled))
    invalid["metrics"]["prefill_ms"] = 0
    _expect_validation_error(
        lambda: validate_document(invalid),
        "validator accepted synthetic zero for unavailable prefill",
    )

    invalid = json.loads(json.dumps(disabled))
    invalid["phases"]["long_decode_diagnostic"]["included_in_standard_median"] = True
    _expect_validation_error(
        lambda: validate_document(invalid),
        "validator accepted diagnostic in standard median",
    )

    invalid = json.loads(json.dumps(disabled))
    invalid["phases"]["standard_decode"]["runs"][0]["wall_seconds"] = 1.0
    _expect_validation_error(
        lambda: validate_document(invalid),
        "validator accepted inconsistent timing/rate",
    )


def _promotion_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValidationError("promotion statistics require at least one value")
    numeric = [float(value) for value in values]
    mean = statistics.fmean(numeric)
    return {
        "count": len(numeric),
        "values": numeric,
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "spread": max(numeric) - min(numeric),
        "coefficient_of_variation": statistics.pstdev(numeric) / mean if mean else None,
    }


def _promotion_nullable_stats(values: Sequence[Any], label: str) -> dict[str, Any]:
    present = [value is not None for value in values]
    if not any(present):
        return {"availability": "unavailable", "values": [None for _ in values], "statistics": None}
    if not all(present):
        raise ValidationError(f"{label} availability differs between repeated runs")
    numeric: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValidationError(f"{label} must be nonnegative or null")
        numeric.append(float(value))
    return {"availability": "measured", "values": numeric, "statistics": _promotion_stats(numeric)}


def _promotion_percent_delta(candidate: float, scalar: float) -> float | None:
    if scalar == 0:
        return None
    return (candidate - scalar) / scalar * 100.0


def _canonical_benchmark_sha256(document: Mapping[str, Any]) -> str:
    canonical = json.loads(json.dumps(document))
    run = canonical.get("run")
    if not isinstance(run, dict):
        raise ValidationError("canonical benchmark identity requires run object")
    run.pop("timestamp_utc", None)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return __import__("hashlib").sha256(payload).hexdigest()


def _load_promotion_sources(paths: Sequence[Path], label: str) -> list[tuple[Path, dict[str, Any]]]:
    if len(paths) < 2:
        raise ValidationError(f"promotion {label} requires at least two repeated artifacts")
    loaded: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    canonical_identities: set[str] = set()
    for supplied in paths:
        path = supplied.expanduser().resolve()
        if path in seen:
            raise ValidationError(f"promotion {label} artifact is duplicated: {path}")
        seen.add(path)
        if not path.is_file():
            raise ValidationError(f"promotion {label} artifact is not a file: {path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid promotion source JSON {path}: {exc.msg}") from exc
        if not isinstance(document, dict):
            raise ValidationError(f"promotion source must be an object: {path}")
        validate_document(document)
        canonical_identity = _canonical_benchmark_sha256(document)
        if canonical_identity in canonical_identities:
            raise ValidationError(
                f"promotion {label} contains a timestamp/path-only clone: {path}"
            )
        canonical_identities.add(canonical_identity)
        loaded.append((path, document))
    return loaded


def _promotion_metric(documents: Sequence[Mapping[str, Any]], *keys: str) -> dict[str, Any]:
    values: list[Any] = []
    for document in documents:
        value: Any = document["metrics"]
        for key in keys:
            if not isinstance(value, Mapping):
                raise ValidationError(f"promotion metric path is not an object: {'.'.join(keys)}")
            value = value.get(key)
        values.append(value)
    return _promotion_nullable_stats(values, ".".join(keys))


def _promotion_group(
    loaded: Sequence[tuple[Path, Mapping[str, Any]]], expected_flag: int
) -> dict[str, Any]:
    documents = [document for _, document in loaded]
    for path, document in loaded:
        environment = document["run"]["environment"]
        if any(environment[key] != expected_flag for key in BATCH_ENVIRONMENT_KEYS):
            raise ValidationError(
                f"promotion artifact has contaminated batch flags (expected all {expected_flag}): {path}"
            )
        dspark_path = document["run"].get("dspark_weights_path")
        if expected_flag == 0 and dspark_path is not None:
            raise ValidationError(f"scalar promotion artifact must not load DSpark weights: {path}")
        if expected_flag == 1 and Path(str(dspark_path)).name != "dspark_weights":
            raise ValidationError(f"fully-batched artifact must use dspark_weights: {path}")

    sources = []
    hashlib = __import__("hashlib")
    for path, document in loaded:
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "canonical_run_sha256": _canonical_benchmark_sha256(document),
                "timestamp_utc": document["run"]["timestamp_utc"],
            }
        )

    correctness_runs = [
        {
            "paris_pass": document["phases"]["correctness"]["paris_pass"],
            "e2e_pass": document["phases"]["correctness"]["e2e_pass"],
            "e2e_fail": document["phases"]["correctness"]["e2e_fail"],
            "unit_tests": document["run"].get("unit_tests"),
        }
        for document in documents
    ]
    return {
        "sources": sources,
        "correctness": {
            "runs": correctness_runs,
            "all_pass": all(
                item["paris_pass"] is True
                and item["e2e_pass"] == 7
                and item["e2e_fail"] == 0
                and isinstance(item["unit_tests"], str)
                and item["unit_tests"].startswith("PASS")
                for item in correctness_runs
            ),
        },
        "standard_tokens_per_second": _promotion_stats(
            [document["phases"]["standard_decode"]["median_tokens_per_second"] for document in documents]
        ),
        "long_diagnostic_tokens_per_second": _promotion_stats(
            [document["phases"]["long_decode_diagnostic"]["run"]["tokens_per_second"] for document in documents]
        ),
        "metrics": {
            "ttft_ms": _promotion_metric(documents, "ttft_ms"),
            "prefill_ms": _promotion_metric(documents, "prefill_ms"),
            "engine_first_token_ms": _promotion_metric(documents, "engine_first_token_ms"),
            "steady_state_decode_ms": _promotion_metric(documents, "steady_state_decode_ms"),
            "itl_p50_ms": _promotion_metric(documents, "itl_ms", "p50"),
            "itl_p95_ms": _promotion_metric(documents, "itl_ms", "p95"),
            "verification_candidates": _promotion_metric(documents, "dspark_verification", "candidate_count"),
            "verification_accepted": _promotion_metric(documents, "dspark_verification", "accepted_count"),
            "verification_batches": _promotion_metric(documents, "dspark_verification", "verification_batches"),
            "verification_ms": _promotion_metric(documents, "dspark_verification", "verification_ms"),
            "verification_mean_batch_ms": _promotion_metric(documents, "dspark_verification", "mean_batch_latency_ms"),
            "expert_pread_calls": _promotion_metric(documents, "expert_io", "pread_calls"),
            "expert_bytes_requested": _promotion_metric(documents, "expert_io", "bytes_requested"),
            "expert_bytes_read": _promotion_metric(documents, "expert_io", "bytes_read"),
            "expert_io_wait_ms": _promotion_metric(documents, "expert_io", "io_wait_ms"),
            "moe_dispatches": _promotion_metric(documents, "dispatch", "dispatches"),
            "moe_command_buffers": _promotion_metric(documents, "dispatch", "command_buffers"),
            "workspace_high_water_bytes": _promotion_metric(documents, "dispatch", "workspace_high_water_bytes"),
            "peak_rss_bytes": _promotion_metric(documents, "peak_rss_bytes"),
        },
    }


def _promotion_median(group: Mapping[str, Any], metric: str) -> float | None:
    entry = group["metrics"][metric]
    statistics_entry = entry.get("statistics")
    return float(statistics_entry["median"]) if statistics_entry is not None else None


def _sha256_path(path: Path) -> str:
    return __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def _batch_oracle_api() -> tuple[Any, Any, Any]:
    try:
        from compare_batch_oracle import compute_fingerprints, validate_fixture, validate_result
    except ModuleNotFoundError:
        from scripts.compare_batch_oracle import compute_fingerprints, validate_fixture, validate_result
    return compute_fingerprints, validate_fixture, validate_result


def build_oracle_evidence(
    result_path: Path,
    fixture_path: Path,
    model_path: Path,
    packed_path: Path,
    runner_path: Path,
    log_path: Path,
    workspace_revision: str | None,
) -> dict[str, Any]:
    result_path = result_path.expanduser().resolve()
    fixture_path = fixture_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    packed_path = packed_path.expanduser().resolve()
    runner_path = runner_path.expanduser().resolve()
    log_path = log_path.expanduser().resolve()
    for label, path in (
        ("result", result_path),
        ("fixture", fixture_path),
        ("runner", runner_path),
        ("log", log_path),
    ):
        if not path.is_file():
            raise ValidationError(f"batch oracle {label} is not a file: {path}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid batch oracle JSON: {exc.msg}") from exc
    if not isinstance(result, dict) or not isinstance(fixture, dict):
        raise ValidationError("batch oracle result and fixture must be objects")
    compute_fingerprints, validate_fixture, validate_result = _batch_oracle_api()
    validate_result(result)
    fingerprints = compute_fingerprints(model_path, packed_path)
    validate_fixture(fixture, result, fingerprints)
    log_text = log_path.read_text(encoding="utf-8")
    suite_markers = {
        "scalar_final_logits_and_partitions": "golden fixture PASS: fingerprints match",
        "production_v2_b1_b2_b3_b5_b8_and_partitions": "production-v2 oracle: PASS B=1,2,3,5,8",
        "scalar_rollback_path": "real batch oracle: PASS (B=1,2,3,5,tail=5+2)",
    }
    suites = {name: marker in log_text for name, marker in suite_markers.items()}
    evidence = {
        "schema_version": 1,
        "kind": "dmlx-batch-oracle-validation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "workspace_revision": workspace_revision,
        "command": "NATIVE_SMELT_N=20 bash scripts/run_batch_oracle.sh",
        "configuration": {
            "NATIVE_SMELT_N": 20,
            "model_path": str(model_path),
            "packed_experts_path": str(packed_path),
            "model_fingerprint": fingerprints["model_manifest_sha256"],
            "tokenizer_sha256": fingerprints["tokenizer_sha256"],
            "fingerprint_method": fingerprints["fingerprint_method"],
        },
        "suites": suites,
        "sources": {
            "result": {"path": str(result_path), "sha256": _sha256_path(result_path)},
            "fixture": {"path": str(fixture_path), "sha256": _sha256_path(fixture_path)},
            "runner": {"path": str(runner_path), "sha256": _sha256_path(runner_path)},
            "log": {"path": str(log_path), "sha256": _sha256_path(log_path)},
        },
    }
    validate_oracle_evidence(evidence)
    return evidence


def validate_oracle_evidence(evidence: Mapping[str, Any]) -> None:
    if evidence.get("schema_version") != 1 or evidence.get("kind") != "dmlx-batch-oracle-validation":
        raise ValidationError("unsupported batch oracle evidence")
    if evidence.get("status") != "PASS":
        raise ValidationError("batch oracle evidence status must be PASS")
    generated = evidence.get("generated_utc")
    if not isinstance(generated, str):
        raise ValidationError("batch oracle generated_utc is required")
    try:
        datetime.fromisoformat(generated)
    except ValueError as exc:
        raise ValidationError("batch oracle generated_utc must be ISO-8601") from exc
    configuration = evidence.get("configuration")
    suites = evidence.get("suites")
    sources = evidence.get("sources")
    if not isinstance(configuration, Mapping) or not isinstance(suites, Mapping) or not isinstance(sources, Mapping):
        raise ValidationError("batch oracle configuration, suites and sources are required")
    if configuration.get("NATIVE_SMELT_N") != 20:
        raise ValidationError("batch oracle must use NATIVE_SMELT_N=20")
    fingerprint = configuration.get("model_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValidationError("batch oracle model_fingerprint must be a SHA-256 digest")
    required_suites = {
        "scalar_final_logits_and_partitions",
        "production_v2_b1_b2_b3_b5_b8_and_partitions",
        "scalar_rollback_path",
    }
    if set(suites) != required_suites or any(suites[name] is not True for name in required_suites):
        raise ValidationError("batch oracle evidence must prove all V4 suites")
    source_paths: dict[str, Path] = {}
    for name in ("result", "fixture", "runner", "log"):
        source = sources.get(name)
        if not isinstance(source, Mapping):
            raise ValidationError(f"batch oracle source {name} is required")
        path_value = source.get("path")
        digest = source.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            raise ValidationError(f"batch oracle source {name} provenance is malformed")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file() or _sha256_path(path) != digest:
            raise ValidationError(f"batch oracle source {name} is missing or modified")
        source_paths[name] = path
    try:
        result = json.loads(source_paths["result"].read_text(encoding="utf-8"))
        fixture = json.loads(source_paths["fixture"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid batch oracle source JSON: {exc.msg}") from exc
    if not isinstance(result, dict) or not isinstance(fixture, dict):
        raise ValidationError("batch oracle source roots must be objects")
    _, validate_fixture, validate_result = _batch_oracle_api()
    validate_result(result)
    fingerprints = {
        "model_manifest_sha256": fingerprint,
        "tokenizer_sha256": configuration.get("tokenizer_sha256"),
        "fingerprint_method": configuration.get("fingerprint_method"),
    }
    validate_fixture(fixture, result, fingerprints)
    log_text = source_paths["log"].read_text(encoding="utf-8")
    expected_suites = {
        "scalar_final_logits_and_partitions": "golden fixture PASS: fingerprints match" in log_text,
        "production_v2_b1_b2_b3_b5_b8_and_partitions": "production-v2 oracle: PASS B=1,2,3,5,8" in log_text,
        "scalar_rollback_path": "real batch oracle: PASS (B=1,2,3,5,tail=5+2)" in log_text,
    }
    if dict(suites) != expected_suites or not all(expected_suites.values()):
        raise ValidationError("batch oracle suites do not match retained V4 log evidence")


def _load_oracle_evidence(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValidationError(f"batch oracle evidence is not a file: {resolved}")
    try:
        evidence = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid batch oracle evidence JSON: {exc.msg}") from exc
    if not isinstance(evidence, dict):
        raise ValidationError("batch oracle evidence root must be an object")
    validate_oracle_evidence(evidence)
    return resolved, evidence


def build_promotion_document(
    scalar_paths: Sequence[Path], candidate_paths: Sequence[Path], oracle_path: Path
) -> dict[str, Any]:
    scalar_loaded = _load_promotion_sources(scalar_paths, "scalar")
    candidate_loaded = _load_promotion_sources(candidate_paths, "fully-batched")
    oracle_resolved, oracle_evidence = _load_oracle_evidence(oracle_path)
    all_loaded = [*scalar_loaded, *candidate_loaded]

    comparable_fields = (
        "workspace_revision",
        "mode",
        "model_path",
        "model_fingerprint",
        "packed_experts_path",
        "machine",
        "warmup",
        "measured_requests",
    )
    reference = all_loaded[0][1]["run"]
    for path, document in all_loaded:
        run = document["run"]
        if run.get("mode") != "native":
            raise ValidationError(f"promotion source {path} must use native mode")
        fingerprint = run.get("model_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValidationError(f"promotion source {path} requires a SHA-256 model fingerprint")
        for field in comparable_fields:
            if run.get(field) != reference.get(field):
                raise ValidationError(f"promotion source {path} differs in run.{field}")
        if run["environment"]["NATIVE_SMELT_N"] != 20:
            raise ValidationError(f"promotion source {path} must use NATIVE_SMELT_N=20")
    oracle_configuration = oracle_evidence["configuration"]
    if oracle_evidence.get("workspace_revision") != reference.get("workspace_revision"):
        raise ValidationError("batch oracle and benchmark workspace revisions differ")
    for oracle_key, run_key in (
        ("model_path", "model_path"),
        ("packed_experts_path", "packed_experts_path"),
        ("model_fingerprint", "model_fingerprint"),
    ):
        if oracle_configuration.get(oracle_key) != reference.get(run_key):
            raise ValidationError(f"batch oracle and benchmark {run_key} differ")
    timestamps = [document["run"]["timestamp_utc"] for _, document in all_loaded]
    if len(set(timestamps)) != len(timestamps):
        raise ValidationError("promotion sources must be independent runs with unique timestamps")

    scalar = _promotion_group(scalar_loaded, 0)
    candidate = _promotion_group(candidate_loaded, 1)
    standard_scalar = float(scalar["standard_tokens_per_second"]["median"])
    standard_candidate = float(candidate["standard_tokens_per_second"]["median"])
    long_scalar = float(scalar["long_diagnostic_tokens_per_second"]["median"])
    long_candidate = float(candidate["long_diagnostic_tokens_per_second"]["median"])

    metric_deltas: dict[str, Any] = {}
    for metric in (
        "ttft_ms",
        "prefill_ms",
        "engine_first_token_ms",
        "steady_state_decode_ms",
        "itl_p50_ms",
        "itl_p95_ms",
        "verification_ms",
        "expert_pread_calls",
        "expert_bytes_requested",
        "expert_bytes_read",
        "expert_io_wait_ms",
        "moe_dispatches",
        "moe_command_buffers",
        "workspace_high_water_bytes",
        "peak_rss_bytes",
    ):
        scalar_value = _promotion_median(scalar, metric)
        candidate_value = _promotion_median(candidate, metric)
        metric_deltas[metric] = {
            "scalar_median": scalar_value,
            "candidate_median": candidate_value,
            "candidate_delta_percent": (
                _promotion_percent_delta(candidate_value, scalar_value)
                if scalar_value is not None and candidate_value is not None
                else None
            ),
        }

    prefill_delta = metric_deltas["prefill_ms"]["candidate_delta_percent"]
    scalar_decode = _promotion_median(scalar, "steady_state_decode_ms")
    verify_latency = _promotion_median(candidate, "verification_mean_batch_ms")
    verify_candidates = _promotion_median(candidate, "verification_candidates")
    verify_batches = _promotion_median(candidate, "verification_batches")
    scalar_equivalent_verify_ms = None
    verification_gain_percent = None
    if (
        scalar_decode is not None
        and verify_latency is not None
        and verify_candidates is not None
        and verify_batches is not None
        and verify_batches > 0
    ):
        candidates_per_batch = verify_candidates / verify_batches
        scalar_equivalent_verify_ms = scalar_decode / 4.0 * candidates_per_batch
        verification_gain_percent = -_promotion_percent_delta(
            verify_latency, scalar_equivalent_verify_ms
        ) if scalar_equivalent_verify_ms > 0 else None

    candidate_rss = candidate["metrics"]["peak_rss_bytes"]["statistics"]
    safe_rss = candidate_rss is not None and candidate_rss["max"] <= 44 * 1024**3
    gates = {
        "correctness_all_runs": scalar["correctness"]["all_pass"] and candidate["correctness"]["all_pass"],
        "batch_oracle": oracle_evidence["status"] == "PASS",
        "material_prefill_gain": prefill_delta is not None and prefill_delta <= -5.0,
        "material_verification_gain": verification_gain_percent is not None and verification_gain_percent >= 5.0,
        "safe_peak_rss": safe_rss,
        "no_material_standard_regression": _promotion_percent_delta(standard_candidate, standard_scalar) >= -5.0,
        "no_material_steady_decode_regression": (
            metric_deltas["steady_state_decode_ms"]["candidate_delta_percent"] is not None
            and metric_deltas["steady_state_decode_ms"]["candidate_delta_percent"] <= 5.0
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": 1,
        "kind": "dmlx-batch-default-promotion",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "workspace_revision": reference.get("workspace_revision"),
            "model_path": reference.get("model_path"),
            "packed_experts_path": reference.get("packed_experts_path"),
            "model_fingerprint": reference.get("model_fingerprint"),
            "machine": reference.get("machine"),
            "warmup": reference.get("warmup"),
            "NATIVE_SMELT_N": 20,
            "repeats_per_group": len(scalar_loaded),
        },
        "oracle": {
            "source": {
                "path": str(oracle_resolved),
                "sha256": _sha256_path(oracle_resolved),
            },
            "status": oracle_evidence["status"],
            "suites": oracle_evidence["suites"],
            "model_fingerprint": oracle_configuration["model_fingerprint"],
        },
        "policy": {
            "material_gain_percent": 5.0,
            "maximum_regression_percent": 5.0,
            "maximum_peak_rss_bytes": 44 * 1024**3,
            "unavailable_metrics_fail_required_gain": True,
        },
        "comparison": {
            "scalar": scalar,
            "fully_batched": candidate,
            "candidate_vs_scalar": {
                "standard_tokens_per_second_delta_percent": _promotion_percent_delta(standard_candidate, standard_scalar),
                "long_diagnostic_tokens_per_second_delta_percent": _promotion_percent_delta(long_candidate, long_scalar),
                "metrics": metric_deltas,
                "verification_baseline": {
                    "scalar_equivalent_ms": scalar_equivalent_verify_ms,
                    "candidate_mean_batch_ms": verify_latency,
                    "gain_percent": verification_gain_percent,
                    "method": "scalar median decode_ms / 4 inter-token intervals * candidate_count / verification_batches",
                },
            },
        },
        "gates": gates,
        "decision": {
            "promote_fully_batched_defaults": not failed,
            "selected_defaults": "fully_batched" if not failed else "scalar_off",
            "failed_gates": failed,
        },
    }


def validate_promotion_document(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != 1 or document.get("kind") != "dmlx-batch-default-promotion":
        raise ValidationError("unsupported promotion document")
    generated = document.get("generated_utc")
    if not isinstance(generated, str):
        raise ValidationError("promotion generated_utc is required")
    try:
        datetime.fromisoformat(generated)
    except ValueError as exc:
        raise ValidationError("promotion generated_utc must be ISO-8601") from exc
    comparison = document.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ValidationError("promotion comparison is required")
    scalar = comparison.get("scalar")
    candidate = comparison.get("fully_batched")
    if not isinstance(scalar, Mapping) or not isinstance(candidate, Mapping):
        raise ValidationError("promotion scalar and fully_batched groups are required")
    try:
        scalar_paths = [Path(item["path"]) for item in scalar["sources"]]
        candidate_paths = [Path(item["path"]) for item in candidate["sources"]]
        oracle_path = Path(document["oracle"]["source"]["path"])
    except (KeyError, TypeError) as exc:
        raise ValidationError("promotion source provenance or gates are malformed") from exc
    rebuilt = build_promotion_document(scalar_paths, candidate_paths, oracle_path)
    rebuilt["generated_utc"] = generated
    if rebuilt != document:
        raise ValidationError("promotion document does not match validated source artifacts")


def promotion_self_test() -> None:
    base_env = {
        "BM_DATE": "2026-01-01",
        "BM_MODE": "native",
        "BM_MODEL_PATH": "/model",
        "BM_PACKED_DIR": "/model/packed_experts",
        "BM_COMMIT": "abc123",
        "BM_BRANCH": "main",
        "BM_COMMAND": "benchmark",
        "BM_HW": "Apple test",
        "BM_MEM": "48GB",
        "BM_SMELT_N": "20",
        "BM_TELEMETRY_ENABLED": "1",
        "BM_UNIT": "PASS",
        "BM_WARMUP_INITIAL": "5",
        "BM_WARMUP_REWARM": "4",
        "BM_STANDARD_TPS_CSV": "1.111111,1.250000,1.315789",
        "BM_STANDARD_ELAPSED_CSV": "4.5,4.0,3.8",
        "BM_STANDARD_PROMPT_TOKENS_CSV": "8,8,8",
        "BM_STANDARD_TOKENS_CSV": "5,5,5",
        "BM_LONG_PROMPT_TOKENS": "14",
        "BM_LONG_TOKENS": "100",
        "BM_LONG_ELAPSED": "70.0",
        "BM_NATIVE_DECODE_TPS": "1.428571",
        "BM_NATIVE_CORRECT": "1",
        "BM_E2E_PASS": "7",
        "BM_E2E_FAIL": "0",
        "BM_STARTUP_MS": "9000",
        "BM_PERF_SECS": "153",
        "BM_TOTAL_SECS": "245",
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model_path = root / "model"
        packed_path = model_path / "packed_experts"
        model_path.mkdir()
        packed_path.mkdir()
        (model_path / "config.json").write_text("{}\n", encoding="utf-8")
        (model_path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (model_path / "model.safetensors").write_bytes(b"model")
        (packed_path / "expert.bin").write_bytes(b"expert")
        try:
            from compare_batch_oracle import (
                _fixture_from_result,
                _sample_result,
                compute_fingerprints,
            )
        except ModuleNotFoundError:
            from scripts.compare_batch_oracle import (
                _fixture_from_result,
                _sample_result,
                compute_fingerprints,
            )
        fingerprints = compute_fingerprints(model_path, packed_path)
        oracle_result = root / "oracle-result.json"
        oracle_fixture = root / "oracle-fixture.json"
        oracle_runner = root / "run_batch_oracle.sh"
        oracle_log = root / "oracle-run.log"
        result_document = _sample_result()
        write_atomic(oracle_result, result_document)
        write_atomic(
            oracle_fixture,
            _fixture_from_result(result_document, fingerprints, "promotion self-test"),
        )
        oracle_runner.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        oracle_log.write_text(
            "golden fixture PASS: fingerprints match\n"
            "production-v2 oracle: PASS B=1,2,3,5,8\n"
            "real batch oracle: PASS (B=1,2,3,5,tail=5+2)\n",
            encoding="utf-8",
        )
        oracle_path = root / "oracle-evidence.json"
        write_atomic(
            oracle_path,
            build_oracle_evidence(
                oracle_result,
                oracle_fixture,
                model_path,
                packed_path,
                oracle_runner,
                oracle_log,
                "abc123",
            ),
        )
        base_env.update(
            BM_MODEL_PATH=str(model_path.resolve()),
            BM_PACKED_DIR=str(packed_path.resolve()),
            BM_MODEL_FINGERPRINT=fingerprints["model_manifest_sha256"],
        )
        scalar_records = [_sample_telemetry(index) for index in range(3)]
        candidate_records = json.loads(json.dumps(scalar_records))
        for record in candidate_records:
            record["batch_path"] = {"scalar_layer_calls": 0, "batched_moe_layers": 43}
            record["dspark_verification"] = {
                "candidate_count": 4,
                "accepted_count": 2,
                "verification_batches": 1,
                "verification_ns": 2_000_000,
            }
        scalar_telemetry = root / "scalar.jsonl"
        candidate_telemetry = root / "candidate.jsonl"
        scalar_telemetry.write_text("".join(json.dumps(item) + "\n" for item in scalar_records), encoding="utf-8")
        candidate_telemetry.write_text("".join(json.dumps(item) + "\n" for item in candidate_records), encoding="utf-8")
        scalar_paths: list[Path] = []
        candidate_paths: list[Path] = []
        for index in range(2):
            scalar_env = dict(
                base_env,
                BM_BATCH_PREFILL="0",
                BM_BATCH_MLA="0",
                BM_EXPERT_BUCKETS="0",
                BM_BATCH_EXPERT_KERNELS="0",
                BM_DSPARK_BATCH_VERIFY="0",
                BM_DSPARK_DIR="",
                BM_TELEMETRY_JSONL=str(scalar_telemetry),
                BM_TOTAL_SECS=str(245 + index),
            )
            candidate_env = dict(
                base_env,
                BM_BATCH_PREFILL="1",
                BM_BATCH_MLA="1",
                BM_EXPERT_BUCKETS="1",
                BM_BATCH_EXPERT_KERNELS="1",
                BM_DSPARK_BATCH_VERIFY="1",
                BM_DSPARK_DIR="/model/dspark_weights",
                BM_TELEMETRY_JSONL=str(candidate_telemetry),
                BM_TOTAL_SECS=str(245 + index),
            )
            scalar_document = build_document(scalar_env)
            candidate_document = build_document(candidate_env)
            scalar_document["run"]["timestamp_utc"] = f"2026-01-01T00:00:0{index}+00:00"
            candidate_document["run"]["timestamp_utc"] = f"2026-01-01T00:00:1{index}+00:00"
            scalar_path = root / f"scalar-{index}.json"
            candidate_path = root / f"candidate-{index}.json"
            write_atomic(scalar_path, scalar_document)
            write_atomic(candidate_path, candidate_document)
            scalar_paths.append(scalar_path)
            candidate_paths.append(candidate_path)
        promotion = build_promotion_document(scalar_paths, candidate_paths, oracle_path)
        validate_promotion_document(json.loads(json.dumps(promotion)))
        if promotion["decision"]["promote_fully_batched_defaults"]:
            raise AssertionError("promotion self-test unexpectedly promoted unavailable prefill evidence")
        _expect_validation_error(
            lambda: build_promotion_document(scalar_paths[:1], candidate_paths, oracle_path),
            "promotion accepted fewer than two scalar runs",
        )
        clone = json.loads(scalar_paths[0].read_text(encoding="utf-8"))
        clone["run"]["timestamp_utc"] = "2026-01-01T00:00:59+00:00"
        clone_path = root / "scalar-timestamp-clone.json"
        write_atomic(clone_path, clone)
        _expect_validation_error(
            lambda: build_promotion_document(
                [scalar_paths[0], clone_path], candidate_paths, oracle_path
            ),
            "promotion accepted a timestamp-only clone as an independent run",
        )
        mode_contaminated = json.loads(scalar_paths[1].read_text(encoding="utf-8"))
        mode_contaminated["run"]["mode"] = "mlx"
        mode_path = root / "scalar-mode-contaminated.json"
        write_atomic(mode_path, mode_contaminated)
        _expect_validation_error(
            lambda: build_promotion_document(
                [scalar_paths[0], mode_path], candidate_paths, oracle_path
            ),
            "promotion accepted a non-native source",
        )
        fingerprint_contaminated = json.loads(
            scalar_paths[1].read_text(encoding="utf-8")
        )
        fingerprint_contaminated["run"]["model_fingerprint"] = "f" * 64
        fingerprint_path = root / "scalar-fingerprint-contaminated.json"
        write_atomic(fingerprint_path, fingerprint_contaminated)
        _expect_validation_error(
            lambda: build_promotion_document(
                [scalar_paths[0], fingerprint_path], candidate_paths, oracle_path
            ),
            "promotion accepted a mismatched model fingerprint",
        )
        contaminated = json.loads(candidate_paths[0].read_text(encoding="utf-8"))
        contaminated["run"]["environment"]["NATIVE_BATCH_PREFILL"] = 0
        write_atomic(candidate_paths[0], contaminated)
        _expect_validation_error(
            lambda: build_promotion_document(scalar_paths, candidate_paths, oracle_path),
            "promotion accepted contaminated candidate flags",
        )
        forged_oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        oracle_log.write_text(
            "golden fixture PASS: fingerprints match\n"
            "real batch oracle: PASS (B=1,2,3,5,tail=5+2)\n",
            encoding="utf-8",
        )
        forged_oracle["sources"]["log"]["sha256"] = _sha256_path(oracle_log)
        _expect_validation_error(
            lambda: validate_oracle_evidence(forged_oracle),
            "oracle evidence accepted a missing production-v2 suite marker",
        )


def _cache_metric_value(document: Mapping[str, Any], *keys: str) -> Any:
    value: Any = document["metrics"]
    for key in keys:
        if not isinstance(value, Mapping):
            raise ValidationError(f"cache promotion metric path is not an object: {'.'.join(keys)}")
        value = value.get(key)
    return value


def _cache_derived(document: Mapping[str, Any]) -> tuple[float, float | None]:
    completion_tokens = sum(
        int(run["completion_tokens"])
        for run in document["phases"]["standard_decode"]["runs"]
    )
    if completion_tokens <= 0:
        raise ValidationError("cache promotion standard completion token sum must be positive")
    bytes_read = _cache_metric_value(document, "expert_io", "bytes_read")
    if bytes_read is None:
        raise ValidationError("cache promotion requires expert bytes_read telemetry")
    hot_hits = _cache_metric_value(document, "expert_io", "hot_hits")
    cold_misses = _cache_metric_value(document, "expert_io", "cold_misses")
    if hot_hits is None or cold_misses is None:
        raise ValidationError("cache promotion requires hot-hit and cold-miss telemetry")
    denominator = float(hot_hits + cold_misses)
    return float(bytes_read) / completion_tokens, (float(hot_hits) / denominator if denominator else None)


def _cache_promotion_group(
    loaded: Sequence[tuple[Path, Mapping[str, Any]]], experimental: int
) -> dict[str, Any]:
    documents = [document for _, document in loaded]
    expected_name = "bounded_lru" if experimental else "fixed_hot"
    expected_metadata = 88088
    sources: list[dict[str, Any]] = []
    hashlib = __import__("hashlib")
    for path, document in loaded:
        environment = document["run"]["environment"]
        if environment.get("NATIVE_EXPERIMENTAL_CACHE") != experimental:
            raise ValidationError(f"cache promotion artifact has wrong experimental-cache flag: {path}")
        if environment.get("NATIVE_PREFETCH") != 0 or environment.get("NATIVE_SMELT_PREFETCH") != 0:
            raise ValidationError(f"cache promotion artifact has prefetch enabled: {path}")
        if any(environment.get(key) != 0 for key in BATCH_ENVIRONMENT_KEYS):
            raise ValidationError(f"cache promotion artifact has enabled batch flags: {path}")
        if document["run"].get("dspark_weights_path") is not None:
            raise ValidationError(f"cache promotion artifact must not load DSpark weights: {path}")
        policy = document["metrics"].get("cache_policy")
        if not isinstance(policy, Mapping):
            raise ValidationError(f"cache promotion artifact lacks cache policy telemetry: {path}")
        if policy.get("name") != expected_name or policy.get("experimental") is not bool(experimental):
            raise ValidationError(f"cache promotion artifact has wrong cache policy: {path}")
        if policy.get("metadata_bytes") != expected_metadata:
            raise ValidationError(f"cache promotion artifact has wrong cache metadata bytes: {path}")
        if experimental:
            if policy.get("admissions") != policy.get("evictions"):
                raise ValidationError(f"LRU cache admissions and evictions differ: {path}")
        elif policy.get("admissions") != 0 or policy.get("evictions") != 0:
            raise ValidationError(f"fixed-hot cache contains churn: {path}")
        sources.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "canonical_run_sha256": _canonical_benchmark_sha256(document),
            "timestamp_utc": document["run"]["timestamp_utc"],
        })

    metric_paths = {
        "itl_p50_ms": ("itl_ms", "p50"),
        "itl_p95_ms": ("itl_ms", "p95"),
        "expert_pread_calls": ("expert_io", "pread_calls"),
        "expert_bytes_read": ("expert_io", "bytes_read"),
        "expert_io_wait_ms": ("expert_io", "io_wait_ms"),
        "hot_hits": ("expert_io", "hot_hits"),
        "cold_misses": ("expert_io", "cold_misses"),
        "cache_admissions": ("cache_policy", "admissions"),
        "cache_evictions": ("cache_policy", "evictions"),
        "metadata_bytes": ("cache_policy", "metadata_bytes"),
        "resident_slots": ("cache_policy", "resident_slots"),
        "resident_expert_bytes": ("cache_policy", "resident_bytes"),
        "peak_rss_bytes": ("peak_rss_bytes",),
    }
    derived = [_cache_derived(document) for document in documents]
    correctness = [{
        "paris_pass": document["phases"]["correctness"]["paris_pass"],
        "e2e_pass": document["phases"]["correctness"]["e2e_pass"],
        "e2e_fail": document["phases"]["correctness"]["e2e_fail"],
        "unit_tests": document["run"].get("unit_tests"),
    } for document in documents]
    return {
        "sources": sources,
        "correctness": {"runs": correctness, "all_pass": all(
            item["paris_pass"] is True and item["e2e_pass"] == 7
            and item["e2e_fail"] == 0
            and isinstance(item["unit_tests"], str)
            and item["unit_tests"].startswith("PASS")
            for item in correctness
        )},
        "standard_tokens_per_second": _promotion_stats([
            document["phases"]["standard_decode"]["median_tokens_per_second"]
            for document in documents
        ]),
        "long_diagnostic_tokens_per_second": _promotion_stats([
            document["phases"]["long_decode_diagnostic"]["run"]["tokens_per_second"]
            for document in documents
        ]),
        "metrics": {
            **{name: _promotion_nullable_stats(
                [_cache_metric_value(document, *path) for document in documents], name
            ) for name, path in metric_paths.items()},
            "bytes_per_completion_token": _promotion_nullable_stats(
                [item[0] for item in derived], "bytes_per_completion_token"
            ),
            "hit_rate": _promotion_nullable_stats([item[1] for item in derived], "hit_rate"),
        },
    }


def build_cache_promotion_document(
    fixed_paths: Sequence[Path], candidate_paths: Sequence[Path]
) -> dict[str, Any]:
    fixed_loaded = _load_promotion_sources(fixed_paths, "cache fixed-hot")
    lru_loaded = _load_promotion_sources(candidate_paths, "cache bounded-LRU")
    if len(fixed_loaded) != len(lru_loaded):
        raise ValidationError("cache promotion groups must contain the same repeat count")
    all_loaded = [*fixed_loaded, *lru_loaded]
    reference = all_loaded[0][1]["run"]
    comparable = ("workspace_revision", "mode", "model_path", "model_fingerprint",
                  "packed_experts_path", "machine", "warmup", "measured_requests")
    for path, document in all_loaded:
        run = document["run"]
        if run.get("mode") != "native":
            raise ValidationError(f"cache promotion source {path} must use native mode")
        fingerprint = run.get("model_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValidationError(f"cache promotion source {path} requires a model fingerprint")
        for field in comparable:
            if run.get(field) != reference.get(field):
                raise ValidationError(f"cache promotion source {path} differs in run.{field}")
        if run["environment"].get("NATIVE_SMELT_N") != 20:
            raise ValidationError(f"cache promotion source {path} must use NATIVE_SMELT_N=20")
    timestamps = [document["run"]["timestamp_utc"] for _, document in all_loaded]
    if len(set(timestamps)) != len(timestamps):
        raise ValidationError("cache promotion sources require unique timestamps")
    purge_states = {document["run"]["environment"].get("SKIP_PURGE") for _, document in all_loaded}
    if len(purge_states) != 1:
        raise ValidationError("cache promotion sources must use a matched SKIP_PURGE state")

    fixed = _cache_promotion_group(fixed_loaded, 0)
    lru = _cache_promotion_group(lru_loaded, 1)
    geometry = {(document["metrics"]["cache_policy"]["resident_slots"],
                 document["metrics"]["cache_policy"]["resident_bytes"])
                for _, document in all_loaded}
    fixed_tps = fixed["standard_tokens_per_second"]["values"]
    lru_tps = lru["standard_tokens_per_second"]["values"]
    fixed_bytes = fixed["metrics"]["expert_bytes_read"]["statistics"]
    lru_bytes = lru["metrics"]["expert_bytes_read"]["statistics"]
    fixed_rss = fixed["metrics"]["peak_rss_bytes"]["statistics"]
    lru_rss = lru["metrics"]["peak_rss_bytes"]["statistics"]
    gates = {
        "correctness_all_runs": fixed["correctness"]["all_pass"] and lru["correctness"]["all_pass"],
        "matching_cache_geometry": len(geometry) == 1,
        "safe_peak_rss": fixed_rss is not None and lru_rss is not None
            and max(fixed_rss["max"], lru_rss["max"]) <= 44 * 1024**3,
        "lru_minimum_standard_tps": lru["standard_tokens_per_second"]["median"] >= 2.0,
        "paired_reproducible_improvement": all(candidate > baseline for baseline, candidate in zip(fixed_tps, lru_tps)),
        "median_standard_improvement": lru["standard_tokens_per_second"]["median"] > fixed["standard_tokens_per_second"]["median"],
        "no_io_bytes_direction_regression": fixed_bytes is not None and lru_bytes is not None
            and lru_bytes["median"] <= fixed_bytes["median"],
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": 1,
        "kind": "dmlx-cache-default-promotion",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "workspace_revision": reference.get("workspace_revision"),
            "model_path": reference.get("model_path"),
            "packed_experts_path": reference.get("packed_experts_path"),
            "model_fingerprint": reference.get("model_fingerprint"),
            "machine": reference.get("machine"),
            "warmup": reference.get("warmup"),
            "NATIVE_SMELT_N": 20,
            "repeats_per_group": len(fixed_loaded),
        },
        "policy": {"minimum_lru_standard_tps": 2.0, "stretch_standard_tps": 2.5,
                   "maximum_peak_rss_bytes": 44 * 1024**3},
        "comparison": {"fixed_hot": fixed, "bounded_lru": lru},
        "stretch": {"lru_standard_tps_2_5_achieved": lru["standard_tokens_per_second"]["median"] >= 2.5},
        "gates": gates,
        "decision": {
            "promote_experimental_cache_defaults": not failed,
            "selected_default": "bounded_lru" if not failed else "fixed_hot_n20",
            "failed_gates": failed,
        },
    }


def validate_cache_promotion_document(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != 1 or document.get("kind") != "dmlx-cache-default-promotion":
        raise ValidationError("unsupported cache promotion document")
    generated = document.get("generated_utc")
    if not isinstance(generated, str):
        raise ValidationError("cache promotion generated_utc is required")
    try:
        datetime.fromisoformat(generated)
        fixed_paths = [Path(item["path"]) for item in document["comparison"]["fixed_hot"]["sources"]]
        lru_paths = [Path(item["path"]) for item in document["comparison"]["bounded_lru"]["sources"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValidationError("cache promotion provenance is malformed") from exc
    rebuilt = build_cache_promotion_document(fixed_paths, lru_paths)
    rebuilt["generated_utc"] = generated
    if rebuilt != document:
        raise ValidationError("cache promotion document does not match bound source artifacts")


def cache_promotion_self_test() -> None:
    base_env = {
        "BM_DATE": "2026-01-01", "BM_MODE": "native", "BM_MODEL_PATH": "/model",
        "BM_PACKED_DIR": "/model/packed_experts", "BM_MODEL_FINGERPRINT": "a" * 64,
        "BM_COMMIT": "abc", "BM_BRANCH": "main", "BM_COMMAND": "benchmark",
        "BM_HW": "Apple M4", "BM_MEM": "48GB", "BM_SMELT_N": "20",
        "BM_TELEMETRY_ENABLED": "1", "BM_UNIT": "PASS", "BM_WARMUP_INITIAL": "5",
        "BM_WARMUP_REWARM": "4", "BM_STANDARD_ELAPSED_CSV": "2.5,2.5,2.5",
        "BM_STANDARD_PROMPT_TOKENS_CSV": "8,8,8", "BM_STANDARD_TOKENS_CSV": "5,5,5",
        "BM_LONG_PROMPT_TOKENS": "14", "BM_LONG_TOKENS": "100", "BM_LONG_ELAPSED": "40",
        "BM_NATIVE_DECODE_TPS": "2.5", "BM_NATIVE_CORRECT": "1", "BM_E2E_PASS": "7",
        "BM_E2E_FAIL": "0", "BM_STARTUP_MS": "9000", "BM_PERF_SECS": "100",
        "BM_BATCH_PREFILL": "0", "BM_BATCH_MLA": "0", "BM_EXPERT_BUCKETS": "0",
        "BM_BATCH_EXPERT_KERNELS": "0", "BM_DSPARK_BATCH_VERIFY": "0", "BM_DSPARK_DIR": "",
        "BM_PREFETCH": "0", "BM_SMELT_PREFETCH": "0", "BM_SKIP_PURGE": "0",
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixed_records = [_sample_telemetry(index) for index in range(3)]
        lru_records = json.loads(json.dumps(fixed_records))
        for record in fixed_records:
            record["cache_policy"] = {
                "name": "fixed_hot_observed", "experimental": False,
                "capacity_per_score_layer": 20, "resident_slots": 1568,
                "resident_bytes": 1024, "metadata_bytes": 88088,
                "admissions": 0, "evictions": 0,
            }
        for record in lru_records:
            record["cache_policy"] = {
                "name": "lru_observed_hot_seeded", "experimental": True,
                "capacity_per_score_layer": 20, "resident_slots": 1568,
                "resident_bytes": 1024, "metadata_bytes": 88088,
                "admissions": 2, "evictions": 2,
            }
        fixed_telemetry, lru_telemetry = root / "fixed.jsonl", root / "lru.jsonl"
        fixed_telemetry.write_text("".join(json.dumps(item) + "\n" for item in fixed_records), encoding="utf-8")
        lru_telemetry.write_text("".join(json.dumps(item) + "\n" for item in lru_records), encoding="utf-8")
        fixed_paths: list[Path] = []
        lru_paths: list[Path] = []
        for index in range(2):
            fixed_env = dict(base_env, BM_EXPERIMENTAL_CACHE="0", BM_TELEMETRY_JSONL=str(fixed_telemetry),
                             BM_STANDARD_TPS_CSV="2.0,2.0,2.0", BM_TOTAL_SECS=str(200 + index))
            lru_env = dict(base_env, BM_EXPERIMENTAL_CACHE="1", BM_TELEMETRY_JSONL=str(lru_telemetry),
                           BM_STANDARD_TPS_CSV="2.2,2.2,2.2", BM_STANDARD_ELAPSED_CSV="2.272727,2.272727,2.272727",
                           BM_TOTAL_SECS=str(210 + index))
            fixed_doc, lru_doc = build_document(fixed_env), build_document(lru_env)
            fixed_doc["run"]["timestamp_utc"] = f"2026-01-01T00:00:0{index}+00:00"
            lru_doc["run"]["timestamp_utc"] = f"2026-01-01T00:00:1{index}+00:00"
            validate_document(fixed_doc); validate_document(lru_doc)
            fixed_path, lru_path = root / f"fixed-{index}.json", root / f"lru-{index}.json"
            write_atomic(fixed_path, fixed_doc); write_atomic(lru_path, lru_doc)
            fixed_paths.append(fixed_path); lru_paths.append(lru_path)
        report = build_cache_promotion_document(fixed_paths, lru_paths)
        validate_cache_promotion_document(json.loads(json.dumps(report)))
        tampered = json.loads(json.dumps(report)); tampered["gates"]["safe_peak_rss"] = False
        _expect_validation_error(lambda: validate_cache_promotion_document(tampered),
                                 "cache promotion accepted aggregate tampering")
        contaminated = json.loads(lru_paths[0].read_text(encoding="utf-8"))
        contaminated["run"]["environment"]["NATIVE_BATCH_PREFILL"] = 1
        write_atomic(lru_paths[0], contaminated)
        _expect_validation_error(lambda: build_cache_promotion_document(fixed_paths, lru_paths),
                                 "cache promotion accepted source contamination")


# Exact-Q4 latency schema v1 is independent of native request telemetry v1.
EXACT_Q4_KIND = "dmlx.exact-q4-latency.v1"
EXACT_Q4_SCHEMA_VERSION = 1
UINT64_MAX = (1 << 64) - 1
EXACT_Q4_COVERAGE_NUMERATOR = 95
EXACT_Q4_COVERAGE_DENOMINATOR = 100
EXACT_Q4_STAGE_NAMES = (
    "attention_encode", "attention_gpu_execution", "attention_gpu_wait",
    "routing", "io_queue", "io_pread", "io_exposed_wait", "moe_encode",
    "moe_gpu_execution", "moe_gpu_wait", "cache_admission",
    "host_alloc_wrapper", "logits_sample", "other_accounted",
)
EXACT_Q4_STAGE_IDS = {name: index for index, name in enumerate(EXACT_Q4_STAGE_NAMES)}
EXACT_Q4_FLAG_PARENT_ONLY = 1
EXACT_Q4_FLAG_GPU_KNOWN = 2
EXACT_Q4_FLAG_GPU_UNKNOWN = 4
EXACT_Q4_FLAG_JOIN_BLOCKED = 8
EXACT_Q4_FLAG_CPU_ACTIVE = 16
EXACT_Q4_FLAG_RAW_DIAGNOSTIC = 32
EXACT_Q4_VALID_FLAGS = 63
EXACT_Q4_READ_SOURCES = {"fixed_hot": 0, "prefetch": 1, "batch_pool": 2, "pread": 3}
EXACT_Q4_TOTAL_KEYS = (
    "host_overhead_ns", "cpu_active_ns", "gpu_execution_ns",
    "io_exposed_wait_ns", "gpu_unknown_cpu_wait_ns", "other_accounted_ns",
    "unaccounted_ns",
)


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValidationError(f"{label} keys differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return value


def _u64(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= UINT64_MAX:
        raise ValidationError(f"{label} must be a uint64")
    return value


def _uint(value: Any, maximum: int, label: str) -> int:
    result = _u64(value, label)
    if result > maximum:
        raise ValidationError(f"{label} exceeds {maximum}")
    return result


def _exact_add(left: int, right: int, label: str) -> int:
    if right > UINT64_MAX - left:
        raise ValidationError(f"{label} overflows uint64")
    return left + right


def exact_q4_hash_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    projection = json.loads(json.dumps(document))
    projection["content_sha256"] = ""
    return projection


def exact_q4_canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def exact_q4_content_sha256(document: Mapping[str, Any]) -> str:
    import hashlib
    return hashlib.sha256(exact_q4_canonical_bytes(exact_q4_hash_projection(document))).hexdigest()


def _validate_exact_event(
    event: Mapping[str, Any], request_ordinal: int, token_ordinal: int,
    request_begin: int, request_end: int, token_begin: int, token_end: int,
    expected_ordinal: int, allow_legacy_logits_layer: bool = False,
) -> dict[str, int]:
    event = _exact_keys(event, {
        "request_ordinal", "token_ordinal", "event_ordinal", "layer", "stage",
        "flags", "calls", "begin_ns", "end_ns", "bytes",
    }, "stage event")
    if _uint(event["request_ordinal"], (1 << 32) - 1, "event.request_ordinal") != request_ordinal:
        raise ValidationError("stage event request identity mismatch")
    if _uint(event["token_ordinal"], (1 << 32) - 1, "event.token_ordinal") != token_ordinal:
        raise ValidationError("stage event token identity mismatch")
    if _uint(event["event_ordinal"], (1 << 32) - 1, "event.event_ordinal") != expected_ordinal:
        raise ValidationError("stage event ordinals must be contiguous and unique")
    layer = _uint(event["layer"], (1 << 16) - 1, "event.layer")
    stage = _uint(event["stage"], len(EXACT_Q4_STAGE_NAMES) - 1, "event.stage")
    if stage == 12:
        if layer != (1 << 16) - 1 and not allow_legacy_logits_layer:
            raise ValidationError("LOGITS_SAMPLE must use the UINT16_MAX layer sentinel")
    elif layer > 42:
        raise ValidationError("layer-scoped stage must use layer 0..42")
    flags = _uint(event["flags"], (1 << 32) - 1, "event.flags")
    if flags & ~EXACT_Q4_VALID_FLAGS:
        raise ValidationError("stage event has unknown flags")
    calls = _uint(event["calls"], (1 << 32) - 1, "event.calls")
    begin, end = _u64(event["begin_ns"], "event.begin_ns"), _u64(event["end_ns"], "event.end_ns")
    _u64(event["bytes"], "event.bytes")
    if begin >= end:
        raise ValidationError("stage event must have positive duration")
    # Strict rule: only parent-only spans may enclose a token, and no event may
    # escape its request. Child spans are wholly inside the token interval.
    if begin < request_begin or end > request_end:
        raise ValidationError("stage event is outside its request")
    parent = bool(flags & EXACT_Q4_FLAG_PARENT_ONLY)
    if parent:
        if begin > token_begin or end < token_end:
            raise ValidationError("parent-only span must enclose its token")
    elif begin < token_begin or end > token_end:
        raise ValidationError("non-parent stage event is outside its token")
    gpu_stage = stage in (1, 8)
    gpu_wait = stage in (2, 9)
    raw_stage = stage in (4, 5)
    if gpu_stage != bool(flags & EXACT_Q4_FLAG_GPU_KNOWN):
        raise ValidationError("known GPU flag/stage mismatch")
    if bool(flags & EXACT_Q4_FLAG_GPU_UNKNOWN) != gpu_wait:
        raise ValidationError("unknown GPU flag/stage mismatch")
    if bool(flags & EXACT_Q4_FLAG_JOIN_BLOCKED) != (stage == 6):
        raise ValidationError("join-blocked flag/stage mismatch")
    cpu_stages = stage in (0, 3, 7, 10, 11, 12)
    if bool(flags & EXACT_Q4_FLAG_CPU_ACTIVE) != cpu_stages:
        raise ValidationError("CPU-active flag/stage mismatch")
    if bool(flags & EXACT_Q4_FLAG_RAW_DIAGNOSTIC) != raw_stage:
        raise ValidationError("raw-diagnostic flag/stage mismatch")
    if calls == 0:
        raise ValidationError("positive-duration stage event requires calls")
    return {"layer": layer, "stage": stage, "flags": flags, "begin": max(begin, token_begin), "end": min(end, token_end)}


def _validate_exact_read(
    read: Mapping[str, Any], request_ordinal: int, token_ordinal: int,
    request_begin: int, request_end: int, expected_ordinal: int,
) -> dict[str, int]:
    read = _exact_keys(read, {
        "request_ordinal", "token_ordinal", "event_ordinal", "layer", "expert",
        "route_rank", "source", "reserved", "error_code", "flags", "submit_ns",
        "worker_start_ns", "complete_ns", "requested_bytes", "read_bytes",
    }, "expert read event")
    for key, expected in (("request_ordinal", request_ordinal), ("token_ordinal", token_ordinal), ("event_ordinal", expected_ordinal)):
        if _uint(read[key], (1 << 32) - 1, f"read.{key}") != expected:
            raise ValidationError("read identity/ordinal mismatch")
    _uint(read["layer"], 42, "read.layer")
    _uint(read["expert"], 255, "read.expert")
    _uint(read["route_rank"], 5, "read.route_rank")
    source = _uint(read["source"], 3, "read.source")
    if read["reserved"] != 0 or isinstance(read["reserved"], bool):
        raise ValidationError("read.reserved must be zero")
    error = read["error_code"]
    if not isinstance(error, int) or isinstance(error, bool) or not -(1 << 31) <= error < (1 << 31):
        raise ValidationError("read.error_code must be int32")
    flags = _uint(read["flags"], (1 << 32) - 1, "read.flags")
    if flags != 1:
        raise ValidationError("read event must be completed with no unknown flags")
    submit, worker, complete = (_u64(read[key], f"read.{key}") for key in ("submit_ns", "worker_start_ns", "complete_ns"))
    requested, actual = _u64(read["requested_bytes"], "read.requested_bytes"), _u64(read["read_bytes"], "read.read_bytes")
    if not (request_begin <= submit <= worker <= complete <= request_end):
        raise ValidationError("read lifecycle is nonmonotonic or outside request")
    if source != EXACT_Q4_READ_SOURCES["pread"] and any((requested, actual, error)):
        raise ValidationError("non-pread source contains physical read results")
    if source == EXACT_Q4_READ_SOURCES["pread"]:
        if requested == 0 or (error == 0 and actual != requested) or (error != 0 and actual >= requested):
            raise ValidationError("pread byte/error lifecycle is inconsistent")
    return {"pread_calls": int(source == 3), "requested_bytes": requested, "read_bytes": actual}


def evaluate_exact_q4_token(
    token: Mapping[str, Any], request_ordinal: int, request_begin: int, request_end: int,
    verify_derived: bool = True, allow_legacy_logits_layer: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    token = _exact_keys(token, {"token_ordinal", "begin_ns", "end_ns", "steady", "events", "derived"}, "token")
    ordinal = _uint(token["token_ordinal"], (1 << 32) - 1, "token.token_ordinal")
    begin, end = _u64(token["begin_ns"], "token.begin_ns"), _u64(token["end_ns"], "token.end_ns")
    if not isinstance(token["steady"], bool):
        raise ValidationError("token.steady must be boolean")
    if begin >= end or begin < request_begin or end > request_end:
        raise ValidationError("token interval is zero/reversed/outside request")
    events_value = token["events"]
    if not isinstance(events_value, list):
        raise ValidationError("token.events must be a list")
    events = [_validate_exact_event(item, request_ordinal, ordinal, request_begin, request_end, begin, end, index, allow_legacy_logits_layer) for index, item in enumerate(events_value)]
    boundaries = {begin, end}
    raw_sum = 0
    for source, event in zip(events_value, events):
        boundaries.update((event["begin"], event["end"]))
        raw_sum = _exact_add(raw_sum, int(source["end_ns"]) - int(source["begin_ns"]), "raw stage sum")
    ordered = sorted(boundaries)
    if ordered[0] != begin or ordered[-1] != end:
        raise ValidationError("event boundaries escape token")
    totals = {key: 0 for key in EXACT_Q4_TOTAL_KEYS}
    violations = 0
    raw_union = 0
    gpu_unknown = False

    def owner(event: Mapping[str, int]) -> tuple[int, str] | None:
        nonlocal gpu_unknown
        stage, flags = event["stage"], event["flags"]
        if flags & EXACT_Q4_FLAG_PARENT_ONLY or stage in (4, 5):
            return None
        if stage in (10, 11): return (1, "host_overhead_ns")
        if stage in (0, 3, 7, 12): return (2, "cpu_active_ns")
        if stage in (1, 8): return (3, "gpu_execution_ns")
        if stage == 6: return (4, "io_exposed_wait_ns")
        if stage in (2, 9):
            gpu_unknown = True
            return (5, "gpu_unknown_cpu_wait_ns")
        if stage == 13: return (6, "other_accounted_ns")
        return None

    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            raise ValidationError("unordered attribution boundary")
        active = [event for event in events if event["begin"] <= left and event["end"] >= right]
        if active:
            raw_union = _exact_add(raw_union, right - left, "raw stage union")
        candidates = [(precedence, event["stage"], category) for event in active if (entry := owner(event)) is not None for precedence, category in (entry,)]
        if not candidates:
            category = "unaccounted_ns"
        else:
            precedence = min(item[0] for item in candidates)
            tied = [item for item in candidates if item[0] == precedence]
            if len(tied) > 1:
                violations += 1
            category = min(tied, key=lambda item: item[1])[2]
        totals[category] = _exact_add(totals[category], right - left, "exclusive total")
    wall = end - begin
    if sum(totals.values()) != wall:
        raise ValidationError("exclusive totals do not equal token wall")
    explained = wall - totals["unaccounted_ns"]
    derived = {
        "exclusive_ns": totals,
        "wall_ns": wall,
        "explained_ns": explained,
        "coverage_numerator_ns": explained,
        "coverage_denominator_ns": wall,
        "schema_violations": violations,
        "gpu_time_unknown": gpu_unknown,
        "raw_stage_sum_ns": raw_sum,
        "raw_stage_union_ns": raw_union,
    }
    if violations:
        raise ValidationError("same-precedence overlap produced a schema violation")
    if token["steady"] and explained * EXACT_Q4_COVERAGE_DENOMINATOR < wall * EXACT_Q4_COVERAGE_NUMERATOR:
        raise ValidationError("steady token coverage is below 95%")
    if verify_derived and token["derived"] != derived:
        raise ValidationError("token derived attribution was tampered")
    return derived, {"wall": wall, "explained": explained}


def validate_exact_q4_latency(
    document: Mapping[str, Any], *, allow_canonical_fixture_legacy_logits: bool = False,
) -> None:
    document = _exact_keys(document, {"kind", "schema_version", "identity", "requests", "derived", "content_sha256"}, "exact-Q4 artifact")
    if document["kind"] != EXACT_Q4_KIND or not isinstance(document["schema_version"], int) or isinstance(document["schema_version"], bool) or document["schema_version"] != EXACT_Q4_SCHEMA_VERSION:
        raise ValidationError("unsupported exact-Q4 kind/version")
    identity = _exact_keys(document["identity"], {"workspace_revision", "binary_sha256", "model_sha256", "tokenizer_sha256", "packed_sha256", "command", "environment", "role", "pair_ordinal"}, "identity")
    for key in ("workspace_revision", "binary_sha256", "model_sha256", "tokenizer_sha256", "packed_sha256", "command", "role"):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ValidationError(f"identity.{key} must be a nonempty string")
    for key in ("binary_sha256", "model_sha256", "tokenizer_sha256", "packed_sha256"):
        if len(identity[key]) != 64 or any(char not in "0123456789abcdef" for char in identity[key]):
            raise ValidationError(f"identity.{key} must be lowercase SHA-256")
    if not isinstance(identity["environment"], Mapping) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in identity["environment"].items()):
        raise ValidationError("identity.environment must map strings to strings")
    _uint(identity["pair_ordinal"], (1 << 32) - 1, "identity.pair_ordinal")
    requests = document["requests"]
    if not isinstance(requests, list) or not requests:
        raise ValidationError("requests must be a nonempty list")
    aggregate_totals = {key: 0 for key in EXACT_Q4_TOTAL_KEYS}
    aggregate_wall = aggregate_explained = pread_calls = requested_bytes = read_bytes = 0
    seen_requests: set[int] = set()
    for request_index, request_value in enumerate(requests):
        request = _exact_keys(request_value, {"request_ordinal", "begin_ns", "end_ns", "tokens", "reads"}, "request")
        request_ordinal = _uint(request["request_ordinal"], (1 << 32) - 1, "request.request_ordinal")
        if request_ordinal in seen_requests or request_ordinal != request_index:
            raise ValidationError("request ordinals must be contiguous and unique")
        seen_requests.add(request_ordinal)
        request_begin, request_end = _u64(request["begin_ns"], "request.begin_ns"), _u64(request["end_ns"], "request.end_ns")
        if request_begin >= request_end:
            raise ValidationError("request interval must be positive")
        tokens = request["tokens"]
        if not isinstance(tokens, list) or not tokens:
            raise ValidationError("request.tokens must be nonempty")
        seen_tokens: set[int] = set()
        for token_index, token in enumerate(tokens):
            if not isinstance(token, Mapping): raise ValidationError("token must be an object")
            ordinal = _uint(token.get("token_ordinal"), (1 << 32) - 1, "token.token_ordinal")
            if ordinal in seen_tokens or ordinal != token_index:
                raise ValidationError("token ordinals must be contiguous and unique")
            seen_tokens.add(ordinal)
            derived, summary = evaluate_exact_q4_token(
                token, request_ordinal, request_begin, request_end,
                allow_legacy_logits_layer=allow_canonical_fixture_legacy_logits,
            )
            aggregate_wall = _exact_add(aggregate_wall, summary["wall"], "aggregate wall")
            aggregate_explained = _exact_add(aggregate_explained, summary["explained"], "aggregate explained")
            for key, value in derived["exclusive_ns"].items(): aggregate_totals[key] = _exact_add(aggregate_totals[key], value, "aggregate exclusive total")
        reads = request["reads"]
        if not isinstance(reads, list): raise ValidationError("request.reads must be a list")
        for read_index, read in enumerate(reads):
            if not isinstance(read, Mapping): raise ValidationError("read must be an object")
            token_ordinal = read.get("token_ordinal")
            if not isinstance(token_ordinal, int) or isinstance(token_ordinal, bool) or token_ordinal not in seen_tokens:
                raise ValidationError("read token identity is invalid")
            counts = _validate_exact_read(read, request_ordinal, token_ordinal, request_begin, request_end, read_index)
            pread_calls = _exact_add(pread_calls, counts["pread_calls"], "pread calls")
            requested_bytes = _exact_add(requested_bytes, counts["requested_bytes"], "requested bytes")
            read_bytes = _exact_add(read_bytes, counts["read_bytes"], "read bytes")
    derived = {
        "exclusive_ns": aggregate_totals,
        "wall_ns": aggregate_wall,
        "explained_ns": aggregate_explained,
        "coverage_numerator_ns": aggregate_explained,
        "coverage_denominator_ns": aggregate_wall,
        "schema_violations": 0,
        "physical_io": {"pread_calls": pread_calls, "requested_bytes": requested_bytes, "read_bytes": read_bytes},
    }
    if aggregate_wall == 0 or aggregate_explained * 100 < aggregate_wall * 95:
        raise ValidationError("aggregate coverage is below 95%")
    if document["derived"] != derived:
        raise ValidationError("artifact derived totals were tampered")
    digest = document["content_sha256"]
    if not isinstance(digest, str) or digest != exact_q4_content_sha256(document):
        raise ValidationError("artifact content SHA-256 mismatch")


def finalize_exact_q4_latency(document: dict[str, Any]) -> dict[str, Any]:
    document["content_sha256"] = ""
    document["content_sha256"] = exact_q4_content_sha256(document)
    return document


def exact_q4_document_from_producer_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build and validate one canonical exact-Q4 document from a raw producer snapshot."""
    payload = _exact_keys(payload, {"identity", "producer", "requests", "stages", "reads"}, "raw producer payload")
    producer = _exact_keys(payload["producer"], {
        "abi_version", "valid", "enabled", "finalized", "stage_capacity",
        "read_capacity", "stage_count", "read_count", "dropped_stages",
        "dropped_reads", "incomplete", "active_read_publishers",
    }, "raw producer metadata")
    expected = {
        "abi_version": 1, "valid": 1, "enabled": 1, "finalized": 1,
        "dropped_stages": 0, "dropped_reads": 0, "incomplete": 0,
        "active_read_publishers": 0,
    }
    for key, value in expected.items():
        if producer[key] != value or isinstance(producer[key], bool):
            raise ValidationError(f"raw producer metadata rejects {key}")
    stages, reads, intervals = payload["stages"], payload["reads"], payload["requests"]
    if not isinstance(stages, list) or not isinstance(reads, list) or not isinstance(intervals, list):
        raise ValidationError("raw producer arrays must be lists")
    stage_count = _uint(producer["stage_count"], (1 << 32) - 1, "producer.stage_count")
    read_count = _uint(producer["read_count"], (1 << 32) - 1, "producer.read_count")
    if stage_count != len(stages) or read_count != len(reads):
        raise ValidationError("raw producer counts differ from arrays")
    if stage_count > _uint(producer["stage_capacity"], (1 << 32) - 1, "producer.stage_capacity") or read_count > _uint(producer["read_capacity"], (1 << 32) - 1, "producer.read_capacity"):
        raise ValidationError("raw producer count exceeds capacity")
    requests: list[dict[str, Any]] = []
    aggregate_totals = {key: 0 for key in EXACT_Q4_TOTAL_KEYS}
    aggregate_wall = aggregate_explained = 0
    for request_index, raw_request in enumerate(intervals):
        raw_request = _exact_keys(raw_request, {"request_ordinal", "begin_ns", "end_ns", "tokens"}, "raw request interval")
        request_ordinal = _uint(raw_request["request_ordinal"], (1 << 32) - 1, "raw request ordinal")
        if request_ordinal != request_index:
            raise ValidationError("raw request ordinals must be contiguous")
        request_begin, request_end = _u64(raw_request["begin_ns"], "raw request begin"), _u64(raw_request["end_ns"], "raw request end")
        raw_tokens = raw_request["tokens"]
        if not isinstance(raw_tokens, list) or not raw_tokens:
            raise ValidationError("raw request tokens must be nonempty")
        tokens = []
        for token_index, raw_token in enumerate(raw_tokens):
            raw_token = _exact_keys(raw_token, {"token_ordinal", "begin_ns", "end_ns", "steady"}, "raw token interval")
            if raw_token["token_ordinal"] != token_index:
                raise ValidationError("raw token ordinals must be contiguous")
            token_events = [dict(event) for event in stages if event.get("request_ordinal") == request_ordinal and event.get("token_ordinal") == token_index]
            token = {**raw_token, "events": token_events, "derived": {}}
            derived, summary = evaluate_exact_q4_token(token, request_ordinal, request_begin, request_end, verify_derived=False)
            token["derived"] = derived
            tokens.append(token)
            aggregate_wall = _exact_add(aggregate_wall, summary["wall"], "aggregate wall")
            aggregate_explained = _exact_add(aggregate_explained, summary["explained"], "aggregate explained")
            for key, value in derived["exclusive_ns"].items():
                aggregate_totals[key] = _exact_add(aggregate_totals[key], value, "aggregate exclusive total")
        request_reads = [dict(read) for read in reads if read.get("request_ordinal") == request_ordinal]
        requests.append({"request_ordinal": request_ordinal, "begin_ns": request_begin, "end_ns": request_end, "tokens": tokens, "reads": request_reads})
    if sum(len(token["events"]) for request in requests for token in request["tokens"]) != stage_count or sum(len(request["reads"]) for request in requests) != read_count:
        raise ValidationError("raw producer event identity has no declared interval")
    pread_calls = requested_bytes = read_bytes = 0
    for request in requests:
        for index, read in enumerate(request["reads"]):
            counts = _validate_exact_read(read, request["request_ordinal"], read["token_ordinal"], request["begin_ns"], request["end_ns"], index)
            pread_calls = _exact_add(pread_calls, counts["pread_calls"], "pread calls")
            requested_bytes = _exact_add(requested_bytes, counts["requested_bytes"], "requested bytes")
            read_bytes = _exact_add(read_bytes, counts["read_bytes"], "read bytes")
    document = {"kind": EXACT_Q4_KIND, "schema_version": EXACT_Q4_SCHEMA_VERSION, "identity": dict(payload["identity"]), "requests": requests, "derived": {"exclusive_ns": aggregate_totals, "wall_ns": aggregate_wall, "explained_ns": aggregate_explained, "coverage_numerator_ns": aggregate_explained, "coverage_denominator_ns": aggregate_wall, "schema_violations": 0, "physical_io": {"pread_calls": pread_calls, "requested_bytes": requested_bytes, "read_bytes": read_bytes}}, "content_sha256": ""}
    finalize_exact_q4_latency(document)
    validate_exact_q4_latency(document)
    return document


def _exact_q4_fixture_document() -> dict[str, Any]:
    def event(ordinal: int, stage: int, flags: int, begin: int, end: int, layer: int = 0) -> dict[str, int]:
        return {"request_ordinal": 0, "token_ordinal": 0, "event_ordinal": ordinal, "layer": layer, "stage": stage, "flags": flags, "calls": 1, "begin_ns": begin, "end_ns": end, "bytes": 0}
    events = [
        event(0, 13, 0, 100, 198),
        event(1, 11, EXACT_Q4_FLAG_CPU_ACTIVE, 100, 105),
        event(2, 10, EXACT_Q4_FLAG_CPU_ACTIVE, 105, 110),
        event(3, 0, EXACT_Q4_FLAG_CPU_ACTIVE, 110, 120),
        event(4, 3, EXACT_Q4_FLAG_CPU_ACTIVE, 120, 130),
        event(5, 1, EXACT_Q4_FLAG_GPU_KNOWN, 130, 140),
        event(6, 2, EXACT_Q4_FLAG_GPU_UNKNOWN, 128, 145),
        event(7, 6, EXACT_Q4_FLAG_JOIN_BLOCKED, 145, 155),
        event(8, 4, EXACT_Q4_FLAG_RAW_DIAGNOSTIC, 130, 160),
        event(9, 5, EXACT_Q4_FLAG_RAW_DIAGNOSTIC, 132, 150),
        event(10, 7, EXACT_Q4_FLAG_CPU_ACTIVE, 155, 165),
        event(11, 8, EXACT_Q4_FLAG_GPU_KNOWN, 165, 175),
        event(12, 9, EXACT_Q4_FLAG_GPU_UNKNOWN, 163, 180),
        event(13, 12, EXACT_Q4_FLAG_CPU_ACTIVE, 180, 190),
        event(14, 13, EXACT_Q4_FLAG_PARENT_ONLY, 90, 210),
    ]
    token = {"token_ordinal": 0, "begin_ns": 100, "end_ns": 200, "steady": True, "events": events, "derived": {}}
    token["derived"], _ = evaluate_exact_q4_token(
        token, 0, 90, 210, verify_derived=False,
        allow_legacy_logits_layer=True,
    )

    def read(ordinal: int, source: int, submit: int, worker: int, complete: int, requested: int = 0, actual: int = 0, error: int = 0) -> dict[str, int]:
        return {"request_ordinal": 0, "token_ordinal": 0, "event_ordinal": ordinal, "layer": 0, "expert": 7 + ordinal, "route_rank": ordinal % 6, "source": source, "reserved": 0, "error_code": error, "flags": 1, "submit_ns": submit, "worker_start_ns": worker, "complete_ns": complete, "requested_bytes": requested, "read_bytes": actual}
    reads = [
        read(0, EXACT_Q4_READ_SOURCES["fixed_hot"], 110, 111, 112),
        read(1, EXACT_Q4_READ_SOURCES["prefetch"], 113, 114, 115),
        read(2, EXACT_Q4_READ_SOURCES["batch_pool"], 116, 117, 118),
        read(3, EXACT_Q4_READ_SOURCES["pread"], 130, 132, 150, 4096, 4096),
        read(4, EXACT_Q4_READ_SOURCES["pread"], 151, 152, 154, 4096, 1024, 5),
    ]
    document = {"kind": EXACT_Q4_KIND, "schema_version": 1, "identity": {"workspace_revision": "fixture", "binary_sha256": "1" * 64, "model_sha256": "2" * 64, "tokenizer_sha256": "3" * 64, "packed_sha256": "4" * 64, "command": "fixture", "environment": {"NATIVE_SMELT_N": "20"}, "role": "schema_fixture", "pair_ordinal": 0}, "requests": [{"request_ordinal": 0, "begin_ns": 90, "end_ns": 210, "tokens": [token], "reads": reads}], "derived": {}, "content_sha256": ""}
    totals = token["derived"]["exclusive_ns"]
    document["derived"] = {"exclusive_ns": totals, "wall_ns": 100, "explained_ns": 100 - totals["unaccounted_ns"], "coverage_numerator_ns": 100 - totals["unaccounted_ns"], "coverage_denominator_ns": 100, "schema_violations": 0, "physical_io": {"pread_calls": 2, "requested_bytes": 8192, "read_bytes": 5120}}
    return finalize_exact_q4_latency(document)


def exact_q4_latency_self_test() -> int:
    import copy
    fixture = _exact_q4_fixture_document()
    raw_fixture = {
        "identity": copy.deepcopy(fixture["identity"]),
        "producer": {
            "abi_version": 1, "valid": 1, "enabled": 1, "finalized": 1,
            "stage_capacity": 32, "read_capacity": 8,
            "stage_count": len(fixture["requests"][0]["tokens"][0]["events"]),
            "read_count": len(fixture["requests"][0]["reads"]),
            "dropped_stages": 0, "dropped_reads": 0, "incomplete": 0,
            "active_read_publishers": 0,
        },
        "requests": [{
            "request_ordinal": 0, "begin_ns": 90, "end_ns": 210,
            "tokens": [{"token_ordinal": 0, "begin_ns": 100, "end_ns": 200, "steady": True}],
        }],
        "stages": copy.deepcopy(fixture["requests"][0]["tokens"][0]["events"]),
        "reads": copy.deepcopy(fixture["requests"][0]["reads"]),
    }
    raw_fixture["stages"][13]["layer"] = (1 << 16) - 1
    raw_document = exact_q4_document_from_producer_snapshot(raw_fixture)
    if raw_document["derived"]["physical_io"] != {"pread_calls": 2, "requested_bytes": 8192, "read_bytes": 5120}:
        raise AssertionError("raw producer exact calls/bytes differ")
    if raw_document["derived"]["explained_ns"] * 100 < raw_document["derived"]["wall_ns"] * 95:
        raise AssertionError("raw producer coverage is below 95%")
    validate_exact_q4_latency(raw_document)
    raw_cases: list[tuple[str, Any]] = []
    def raw_mutate(name: str, action: Any) -> None:
        candidate = copy.deepcopy(raw_fixture); action(candidate); raw_cases.append((name, candidate))
    raw_mutate("reversed", lambda d: d["requests"][0]["tokens"][0].__setitem__("end_ns", 99))
    raw_mutate("same-precedence", lambda d: d["stages"].insert(3, {**d["stages"][3], "event_ordinal": 3, "stage": 3}))
    for index, event in enumerate(raw_cases[-1][1]["stages"]): event["event_ordinal"] = index
    raw_cases[-1][1]["producer"]["stage_count"] += 1
    raw_mutate("physical-io", lambda d: d["reads"][3].__setitem__("read_bytes", 1))
    raw_mutate("producer-valid", lambda d: d["producer"].__setitem__("valid", 0))
    raw_mutate("producer-count", lambda d: d["producer"].__setitem__("stage_count", 1))
    for name, candidate in raw_cases:
        try:
            exact_q4_document_from_producer_snapshot(candidate)
        except ValidationError:
            continue
        raise AssertionError(f"raw producer helper accepted invalid case: {name}")
    tampered = copy.deepcopy(raw_document)
    tampered["derived"]["wall_ns"] += 1
    try:
        validate_exact_q4_latency(tampered)
    except ValidationError:
        pass
    else:
        raise AssertionError("raw producer derived tamper accepted")
    tampered = copy.deepcopy(raw_document)
    tampered["content_sha256"] = "0" * 64
    try:
        validate_exact_q4_latency(tampered)
    except ValidationError:
        pass
    else:
        raise AssertionError("raw producer hash tamper accepted")
    validate_exact_q4_latency(
        fixture, allow_canonical_fixture_legacy_logits=True)
    try:
        validate_exact_q4_latency(fixture)
    except ValidationError:
        pass
    else:
        raise AssertionError("artifact-controlled schema_fixture role authorized legacy logits layer")
    fixture_token = fixture["requests"][0]["tokens"][0]
    expected_totals = {
        "host_overhead_ns": 10,
        "cpu_active_ns": 40,
        "gpu_execution_ns": 20,
        "io_exposed_wait_ns": 10,
        "gpu_unknown_cpu_wait_ns": 10,
        "other_accounted_ns": 8,
        "unaccounted_ns": 2,
    }
    if fixture_token["derived"]["exclusive_ns"] != expected_totals:
        raise AssertionError("exact-Q4 stage ownership/precedence totals differ")
    if {event["stage"] for event in fixture_token["events"]} != set(range(len(EXACT_Q4_STAGE_NAMES))):
        raise AssertionError("exact-Q4 fixture does not exercise every declared stage")
    fixture_reads = fixture["requests"][0]["reads"]
    if {read["source"] for read in fixture_reads} != set(EXACT_Q4_READ_SOURCES.values()):
        raise AssertionError("exact-Q4 fixture does not exercise every read source")
    if any(any(read[key] for key in ("requested_bytes", "read_bytes", "error_code")) for read in fixture_reads if read["source"] != EXACT_Q4_READ_SOURCES["pread"]):
        raise AssertionError("exact-Q4 non-pread fixture contains physical read results")
    failed_preads = [read for read in fixture_reads if read["source"] == EXACT_Q4_READ_SOURCES["pread"] and read["error_code"] != 0]
    if len(failed_preads) != 1 or not failed_preads[0]["read_bytes"] < failed_preads[0]["requested_bytes"]:
        raise AssertionError("exact-Q4 fixture lacks one coherent failed short pread")
    if fixture["derived"]["physical_io"] != {"pread_calls": 2, "requested_bytes": 8192, "read_bytes": 5120}:
        raise AssertionError("exact-Q4 physical I/O totals differ")
    fixture_path = Path(__file__).resolve().parent.parent / "tests/fixtures/dmlx_exact_q4_latency_v1.json"
    fixture_bytes = fixture_path.read_bytes()
    if fixture_bytes != exact_q4_canonical_bytes(fixture):
        raise AssertionError("committed exact-Q4 fixture differs from canonical generated fixture")
    validate_exact_q4_latency(
        json.loads(fixture_bytes), allow_canonical_fixture_legacy_logits=True)
    if exact_q4_canonical_bytes(fixture) != exact_q4_canonical_bytes(json.loads(exact_q4_canonical_bytes(fixture))):
        raise AssertionError("exact-Q4 canonical bytes are nondeterministic")
    try:
        _exact_add(UINT64_MAX, 1, "self-test")
    except ValidationError:
        pass
    else:
        raise AssertionError("exact-Q4 uint64 addition accepted overflow")
    cases: list[tuple[str, Any]] = []
    def mutate(name: str, action: Any) -> None:
        candidate = copy.deepcopy(fixture); action(candidate); cases.append((name, candidate))
    mutate("kind", lambda d: d.__setitem__("kind", "bad"))
    mutate("version", lambda d: d.__setitem__("schema_version", 2))
    mutate("missing-key", lambda d: d.pop("identity"))
    mutate("extra-key", lambda d: d.__setitem__("extra", 1))
    mutate("type", lambda d: d.__setitem__("requests", {}))
    mutate("hash", lambda d: d.__setitem__("content_sha256", "0" * 64))
    mutate("derived", lambda d: d["derived"].__setitem__("wall_ns", 99))
    mutate("token-derived", lambda d: d["requests"][0]["tokens"][0]["derived"].__setitem__("wall_ns", 99))
    mutate("reversed", lambda d: d["requests"][0]["tokens"][0]["events"][0].__setitem__("end_ns", 99))
    mutate("zero-wall", lambda d: d["requests"][0]["tokens"][0].__setitem__("end_ns", 100))
    mutate("out-of-request", lambda d: d["requests"][0]["tokens"][0]["events"][0].__setitem__("begin_ns", 80))
    mutate("uint64", lambda d: d["requests"][0].__setitem__("end_ns", UINT64_MAX + 1))
    mutate("duplicate-request", lambda d: d["requests"].append(copy.deepcopy(d["requests"][0])))
    mutate("duplicate-token", lambda d: d["requests"][0]["tokens"].append(copy.deepcopy(d["requests"][0]["tokens"][0])))
    mutate("token-ordinal-list", lambda d: d["requests"][0]["tokens"][0].__setitem__("token_ordinal", []))
    mutate("token-ordinal-object", lambda d: d["requests"][0]["tokens"][0].__setitem__("token_ordinal", {}))
    mutate("token-ordinal-bool", lambda d: d["requests"][0]["tokens"][0].__setitem__("token_ordinal", True))
    mutate("token-ordinal-negative", lambda d: d["requests"][0]["tokens"][0].__setitem__("token_ordinal", -1))
    mutate("token-ordinal-overflow", lambda d: d["requests"][0]["tokens"][0].__setitem__("token_ordinal", 1 << 32))
    mutate("event-ordinal", lambda d: d["requests"][0]["tokens"][0]["events"][1].__setitem__("event_ordinal", 0))
    mutate("read-ordinal", lambda d: d["requests"][0]["reads"][0].__setitem__("event_ordinal", 1))
    mutate("stage", lambda d: d["requests"][0]["tokens"][0]["events"][0].__setitem__("stage", 14))
    mutate("flags", lambda d: d["requests"][0]["tokens"][0]["events"][0].__setitem__("flags", 64))
    mutate("source", lambda d: d["requests"][0]["reads"][0].__setitem__("source", 4))
    mutate("read-order", lambda d: d["requests"][0]["reads"][0].__setitem__("worker_start_ns", 151))
    mutate("read-bytes", lambda d: d["requests"][0]["reads"][3].__setitem__("read_bytes", 1))
    mutate("read-error", lambda d: d["requests"][0]["reads"][3].__setitem__("error_code", 1))
    mutate("non-pread-requested", lambda d: d["requests"][0]["reads"][0].__setitem__("requested_bytes", 1))
    mutate("non-pread-read", lambda d: d["requests"][0]["reads"][1].__setitem__("read_bytes", 1))
    mutate("non-pread-error", lambda d: d["requests"][0]["reads"][2].__setitem__("error_code", 1))
    mutate("routing-flag", lambda d: d["requests"][0]["tokens"][0]["events"][4].__setitem__("flags", 0))
    mutate("cache-admission-flag", lambda d: d["requests"][0]["tokens"][0]["events"][2].__setitem__("flags", 0))
    mutate("logits-sample-flag", lambda d: d["requests"][0]["tokens"][0]["events"][13].__setitem__("flags", 0))
    mutate("moe-gpu-execution-flag", lambda d: d["requests"][0]["tokens"][0]["events"][11].__setitem__("flags", 0))
    mutate("moe-gpu-wait-flag", lambda d: d["requests"][0]["tokens"][0]["events"][12].__setitem__("flags", 0))
    mutate("same-precedence", lambda d: d["requests"][0]["tokens"][0]["events"].insert(3, {**d["requests"][0]["tokens"][0]["events"][3], "event_ordinal": 3, "stage": 3}))
    # Make all subsequent ordinals contiguous so the conflict, not identity, rejects.
    conflict = cases[-1][1]
    for index, item in enumerate(conflict["requests"][0]["tokens"][0]["events"]): item["event_ordinal"] = index
    mutate("coverage-token", lambda d: (d["requests"][0]["tokens"][0].__setitem__("events", d["requests"][0]["tokens"][0]["events"][:1]), d["requests"][0]["tokens"][0]["events"][0].__setitem__("end_ns", 190), d["requests"][0]["tokens"][0].__setitem__("derived", {})))
    mutate("identity-hash-type", lambda d: d["identity"].__setitem__("binary_sha256", 1))
    mutate("reserved", lambda d: d["requests"][0]["reads"][0].__setitem__("reserved", 1))
    mutate("read-flags", lambda d: d["requests"][0]["reads"][0].__setitem__("flags", 0))
    mutate("parent-clipping", lambda d: d["requests"][0]["tokens"][0]["events"][14].__setitem__("begin_ns", 101))
    mutate("event-extra", lambda d: d["requests"][0]["tokens"][0]["events"][0].__setitem__("extra", 1))
    mutate("zero-calls", lambda d: d["requests"][0]["tokens"][0]["events"][0].__setitem__("calls", 0))
    mutate("boolean-version", lambda d: d.__setitem__("schema_version", True))
    mutate("overflow-sum", lambda d: (d["requests"][0]["tokens"][0]["events"][0].__setitem__("begin_ns", 0), d["requests"][0]["tokens"][0]["events"][0].__setitem__("end_ns", UINT64_MAX), d["requests"][0].__setitem__("begin_ns", 0), d["requests"][0].__setitem__("end_ns", UINT64_MAX)))
    aggregate_low = copy.deepcopy(fixture)
    aggregate_token = aggregate_low["requests"][0]["tokens"][0]
    aggregate_token["steady"] = False
    aggregate_token["events"] = aggregate_token["events"][:1]
    aggregate_token["events"][0]["end_ns"] = 190
    aggregate_token["derived"], _ = evaluate_exact_q4_token(
        aggregate_token, 0, 90, 210, verify_derived=False)
    cases.append(("coverage-aggregate", aggregate_low))
    for name, candidate in cases:
        try:
            validate_exact_q4_latency(candidate)
        except ValidationError:
            continue
        raise AssertionError(f"exact-Q4 validator accepted invalid case: {name}")
    return len(cases) + 4


def _validate_exact_q4_mla(raw: Mapping[str, Any]) -> None:
    snapshot = _exact_keys(raw.get("mla_snapshot"), {
        "abi_version", "valid", "enabled", "finalized", "request_ordinal",
        "token_ordinal", "capacity", "count", "dropped", "reserved",
    }, "MLA snapshot")
    paths = raw.get("mla_paths")
    if not isinstance(paths, list):
        raise ValidationError("mla_paths must be a list")
    for key, expected in (("abi_version", 1), ("valid", 1), ("enabled", 1),
                          ("finalized", 1), ("dropped", 0), ("reserved", 0)):
        if snapshot[key] != expected or isinstance(snapshot[key], bool):
            raise ValidationError(f"MLA snapshot rejects {key}")
    if snapshot["count"] != len(paths) or snapshot["count"] > snapshot["capacity"]:
        raise ValidationError("MLA snapshot count/capacity mismatch")
    request = raw["requests"]
    if not isinstance(request, list) or len(request) != 1:
        raise ValidationError("raw JSONL record must contain exactly one request")
    tokens = request[0].get("tokens")
    if not isinstance(tokens, list) or len(paths) != len(tokens) * 43:
        raise ValidationError("MLA path count must equal token count times 43 layers")
    expected_token_ordinal = tokens[-1]["token_ordinal"] if tokens else 0
    if snapshot["request_ordinal"] != request[0]["request_ordinal"] or snapshot["token_ordinal"] != expected_token_ordinal:
        raise ValidationError("MLA snapshot request/token identity mismatch")
    valid_flags = 1 | 2 | 4 | 8 | 16
    for index, path in enumerate(paths):
        path = _exact_keys(path, {"request_ordinal", "token_ordinal", "event_ordinal", "layer", "path", "flags"}, "MLA path")
        token, layer = divmod(index, 43)
        if (path["request_ordinal"] != request[0]["request_ordinal"] or path["token_ordinal"] != token or path["event_ordinal"] != index or path["layer"] != layer):
            raise ValidationError("MLA path identity/order mismatch")
        if path["path"] not in range(4) or not isinstance(path["flags"], int) or path["flags"] & ~valid_flags:
            raise ValidationError("MLA path enum/flags invalid")
        flags = path["flags"]
        required_forbidden = {
            0: (1 | 2 | 4, 8 | 16), 1: (1, 4 | 8 | 16),
            2: (1 | 2 | 16, 4 | 8), 3: (8, 1 | 4 | 16),
        }
        required, forbidden = required_forbidden[path["path"]]
        if flags & required != required or flags & forbidden:
            raise ValidationError("MLA path semantic flags invalid")


def exact_q4_document_from_raw_jsonl(
    raw_path: Path, identity_path: Path, selected_ordinals: list[int],
) -> dict[str, Any]:
    if not selected_ordinals or len(set(selected_ordinals)) != len(selected_ordinals):
        raise ValidationError("exact-Q4 selection must be nonempty and unique")
    records = []
    with raw_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid raw JSONL line {line_number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise ValidationError("raw JSONL records must be objects")
            records.append(value)
    by_ordinal = {}
    for record in records:
        record = _exact_keys(record, {
            "producer", "requests", "stages", "reads", "mla_snapshot", "mla_paths",
        }, "raw JSONL record")
        producer = _exact_keys(record["producer"], {
            "abi_version", "valid", "enabled", "finalized", "request_ordinal",
            "token_ordinal", "stage_capacity", "read_capacity", "stage_count",
            "read_count", "dropped_stages", "dropped_reads", "incomplete",
            "active_read_publishers",
        }, "raw JSONL producer metadata")
        _validate_exact_q4_mla(record)
        request = record["requests"][0]
        ordinal = request["request_ordinal"]
        expected_token = request["tokens"][-1]["token_ordinal"] if request["tokens"] else 0
        if producer["request_ordinal"] != ordinal or producer["token_ordinal"] != expected_token:
            raise ValidationError("raw producer request/token identity mismatch")
        if producer["stage_count"] != len(record["stages"]) or producer["read_count"] != len(record["reads"]):
            raise ValidationError("raw producer counts differ from JSONL arrays")
        if ordinal in by_ordinal:
            raise ValidationError("duplicate raw request ordinal")
        by_ordinal[ordinal] = record
    if any(ordinal not in by_ordinal for ordinal in selected_ordinals):
        raise ValidationError("selected raw request ordinal is missing")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    requests, stages, reads = [], [], []
    stage_capacity = read_capacity = 0
    for new_request, source_ordinal in enumerate(selected_ordinals):
        record = by_ordinal[source_ordinal]
        source_request = record["requests"][0]
        normalized = dict(source_request)
        normalized["request_ordinal"] = new_request
        requests.append(normalized)
        read_base = len(reads)
        stage_ordinals: dict[int, int] = {}
        for event in record["stages"]:
            item = dict(event); item["request_ordinal"] = new_request
            token_ordinal = item["token_ordinal"]
            item["event_ordinal"] = stage_ordinals.get(token_ordinal, 0)
            stage_ordinals[token_ordinal] = item["event_ordinal"] + 1
            stages.append(item)
        for event in record["reads"]:
            item = dict(event); item["request_ordinal"] = new_request
            item["event_ordinal"] = len(reads) - read_base; reads.append(item)
        stage_capacity += record["producer"]["stage_capacity"]
        read_capacity += record["producer"]["read_capacity"]
    payload = {
        "identity": identity,
        "producer": {"abi_version": 1, "valid": 1, "enabled": 1, "finalized": 1,
            "stage_capacity": stage_capacity, "read_capacity": read_capacity,
            "stage_count": len(stages), "read_count": len(reads), "dropped_stages": 0,
            "dropped_reads": 0, "incomplete": 0, "active_read_publishers": 0},
        "requests": requests, "stages": stages, "reads": reads,
    }
    return exact_q4_document_from_producer_snapshot(payload)


EXACT_Q4_TASK03_KIND = "dmlx.exact-q4-task-0.3.v1"
EXACT_Q4_RETAINED_TPS = 1.207636


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_exact_q4_task03_document(source_specs: list[str], drift_root_cause: str | None, *, _skip_validate: bool = False) -> dict[str, Any]:
    expected_roles = ["baseline1", "enabled1", "baseline2", "enabled2"]
    if len(source_specs) != 4:
        raise ValidationError("Task 0.3 requires exactly four ordered sources")
    runs = []
    identity_key = None
    enabled_identity_hashes = None
    for pair_ordinal, (spec, expected_role) in enumerate(zip(source_specs, expected_roles), 1):
        parts = spec.split(":", 4)
        if len(parts) != 5 or parts[0] != expected_role:
            raise ValidationError("Task 0.3 source order must be baseline1, enabled1, baseline2, enabled2")
        benchmark_path = Path(parts[1]); standard_path = None if parts[2] == "-" else Path(parts[2]); long_path = None if parts[3] == "-" else Path(parts[3]); identity_path = Path(parts[4])
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")); validate_document(benchmark)
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        config = benchmark["run"]
        flags = config["environment"]
        required_flags = {"NATIVE_EXPERIMENTAL_CACHE": 0, "NATIVE_BATCH_PREFILL": 0,
            "NATIVE_BATCH_MLA": 0, "NATIVE_EXPERT_BUCKETS": 0,
            "NATIVE_BATCH_EXPERT_KERNELS": 0, "NATIVE_DSPARK_BATCH_VERIFY": 0,
            "NATIVE_PREFETCH": 0, "NATIVE_SMELT_PREFETCH": 0}
        if flags["NATIVE_SMELT_N"] != 20 or any(flags.get(k) != v for k, v in required_flags.items()):
            raise ValidationError("Task 0.3 source is not fixed-hot N=20 scalar/off")
        comparable_config = (
            config["workspace_revision"], config["mode"], config["model_path"],
            config["packed_experts_path"], config.get("model_fingerprint"),
            config["machine"], config["warmup"], config["measured_requests"],
            config["command"], flags.get("SKIP_PURGE"),
        )
        identity = _exact_keys(identity, {
            "workspace_revision", "binary_sha256", "model_sha256", "tokenizer_sha256",
            "packed_sha256", "command", "environment", "role", "pair_ordinal",
        }, "Task 0.3 run identity")
        expected_pair = (pair_ordinal + 1) // 2
        if identity["role"] != expected_role or identity["pair_ordinal"] != expected_pair:
            raise ValidationError("Task 0.3 run identity role/pair mismatch")
        if not str(identity["workspace_revision"]).startswith(str(config["workspace_revision"])):
            raise ValidationError("Task 0.3 run identity workspace differs from benchmark")
        if identity["model_sha256"] != config["model_fingerprint"]:
            raise ValidationError("Task 0.3 run identity model differs from benchmark")
        capture_expected = "1" if expected_role.startswith("enabled") else "0"
        expected_identity_env = {key: str(value) for key, value in flags.items() if key != "NATIVE_TELEMETRY"}
        expected_identity_env["NATIVE_EXACT_Q4_CAPTURE"] = capture_expected
        if identity["environment"] != expected_identity_env:
            raise ValidationError("Task 0.3 run identity environment differs from benchmark")
        identity_hashes = tuple(identity[key] for key in (
            "model_sha256", "tokenizer_sha256", "packed_sha256",
        ))
        identity_common = (comparable_config, identity_hashes, identity["command"])
        if identity_key is None: identity_key = identity_common
        elif identity_common != identity_key: raise ValidationError("Task 0.3 source identities/environments differ")
        enabled = expected_role.startswith("enabled")
        if enabled != (standard_path is not None and long_path is not None):
            raise ValidationError("enabled runs require standard and long canonical artifacts; baselines require none")
        canonical = {}
        if enabled:
            for label, path, count in (("standard", standard_path, 3), ("long", long_path, 1)):
                raw = path.read_bytes(); doc = json.loads(raw)
                if raw != exact_q4_canonical_bytes(doc): raise ValidationError("noncanonical exact-Q4 source")
                validate_exact_q4_latency(doc)
                if len(doc["requests"]) != count: raise ValidationError(f"{label} exact-Q4 request count mismatch")
                canonical_identity = doc["identity"]
                expected_pair = (pair_ordinal + 1) // 2
                if canonical_identity != identity:
                    raise ValidationError("canonical exact-Q4 identity differs from bound run identity")
                if canonical_identity["role"] != expected_role or canonical_identity["pair_ordinal"] != expected_pair:
                    raise ValidationError("canonical exact-Q4 role/pair identity mismatch")
                if not str(canonical_identity["workspace_revision"]).startswith(str(config["workspace_revision"])):
                    raise ValidationError("canonical workspace revision differs from benchmark")
                if canonical_identity["model_sha256"] != config["model_fingerprint"]:
                    raise ValidationError("canonical model identity differs from benchmark fingerprint")
                for env_key, value in flags.items():
                    if env_key == "NATIVE_TELEMETRY":
                        continue
                    if canonical_identity["environment"].get(env_key) != str(value):
                        raise ValidationError("canonical environment differs from benchmark")
                hashes = tuple(canonical_identity[key] for key in ("model_sha256", "tokenizer_sha256", "packed_sha256"))
                if enabled_identity_hashes is None:
                    enabled_identity_hashes = hashes
                elif hashes != enabled_identity_hashes:
                    raise ValidationError("enabled canonical binary/model/tokenizer/packed identities differ")
                canonical[label] = {"path": str(path.resolve()), "physical_sha256": _file_sha256(path),
                    "content_sha256": doc["content_sha256"], "physical_io": doc["derived"]["physical_io"],
                    "coverage_numerator_ns": doc["derived"]["coverage_numerator_ns"],
                    "coverage_denominator_ns": doc["derived"]["coverage_denominator_ns"]}
            io = benchmark["metrics"]["expert_io"]
            if canonical["standard"]["physical_io"] != {"pread_calls": io["pread_calls"], "requested_bytes": io["bytes_requested"], "read_bytes": io["bytes_read"]}:
                raise ValidationError("exact-Q4 physical I/O differs from telemetry-v1 benchmark metrics")
        correctness = benchmark["phases"]["correctness"]
        rss = benchmark["metrics"]["peak_rss_bytes"]
        runs.append({"role": expected_role, "pair_ordinal": (pair_ordinal + 1) // 2,
            "benchmark": {"path": str(benchmark_path.resolve()), "physical_sha256": _file_sha256(benchmark_path)},
            "identity": {"path": str(identity_path.resolve()), "physical_sha256": _file_sha256(identity_path)},
            "canonical": canonical, "standard_median_tokens_per_second": benchmark["phases"]["standard_decode"]["median_tokens_per_second"],
            "peak_rss_bytes": rss, "correctness_pass": correctness["paris_pass"] and correctness["e2e_pass"] == 7 and correctness["e2e_fail"] == 0})
    baseline_ok = all(abs(run["standard_median_tokens_per_second"] / EXACT_Q4_RETAINED_TPS - 1) <= .05 for run in runs if run["role"].startswith("baseline"))
    if not baseline_ok and not drift_root_cause:
        raise ValidationError("disabled baseline drift exceeds ±5% without root cause")
    gates = {"baseline_within_retained_5_percent_or_root_caused": baseline_ok or bool(drift_root_cause),
        "enabled_coverage_at_least_95_percent": all(c["coverage_numerator_ns"] * 100 >= c["coverage_denominator_ns"] * 95 for r in runs if r["canonical"] for c in r["canonical"].values()),
        "peak_rss_below_44_gib": all(r["peak_rss_bytes"] is not None and r["peak_rss_bytes"] < 44 * 1024**3 for r in runs),
        "correctness_all_runs": all(r["correctness_pass"] for r in runs)}
    document = {"kind": EXACT_Q4_TASK03_KIND, "schema_version": 1, "retained_baseline_tokens_per_second": EXACT_Q4_RETAINED_TPS,
        "drift_root_cause": drift_root_cause, "runs": runs, "gates": gates, "decision": {"pass": all(gates.values()), "failed_gates": [k for k, v in gates.items() if not v]}, "content_sha256": ""}
    document["content_sha256"] = hashlib.sha256(json.dumps({**document, "content_sha256": ""}, sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest()
    if not _skip_validate:
        validate_exact_q4_task03_document(document)
    return document


def validate_exact_q4_task03_document(document: Mapping[str, Any]) -> None:
    if document.get("kind") != EXACT_Q4_TASK03_KIND or document.get("schema_version") != 1:
        raise ValidationError("unsupported Task 0.3 aggregate")
    if [run.get("role") for run in document.get("runs", [])] != ["baseline1", "enabled1", "baseline2", "enabled2"]:
        raise ValidationError("Task 0.3 aggregate source order invalid")
    if [run.get("pair_ordinal") for run in document["runs"]] != [1, 1, 2, 2]:
        raise ValidationError("Task 0.3 pair ordinals invalid")
    expected = hashlib.sha256(json.dumps({**document, "content_sha256": ""}, sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest()
    if document.get("content_sha256") != expected:
        raise ValidationError("Task 0.3 aggregate content hash mismatch")
    for run in document["runs"]:
        if _file_sha256(Path(run["benchmark"]["path"])) != run["benchmark"]["physical_sha256"]:
            raise ValidationError("Task 0.3 benchmark source hash mismatch")
        if _file_sha256(Path(run["identity"]["path"])) != run["identity"]["physical_sha256"]:
            raise ValidationError("Task 0.3 identity source hash mismatch")
        for source in run["canonical"].values():
            path = Path(source["path"]); raw = path.read_bytes(); exact = json.loads(raw)
            if _file_sha256(path) != source["physical_sha256"] or exact["content_sha256"] != source["content_sha256"]:
                raise ValidationError("Task 0.3 canonical source hash mismatch")
    if document["decision"]["pass"] != all(document["gates"].values()):
        raise ValidationError("Task 0.3 decision differs from gates")
    specs = []
    for run in document["runs"]:
        canonical = run["canonical"]
        specs.append(":".join((run["role"], run["benchmark"]["path"],
            canonical.get("standard", {}).get("path", "-"),
            canonical.get("long", {}).get("path", "-"),
            run["identity"]["path"])))
    rebuilt = build_exact_q4_task03_document(
        specs, document.get("drift_root_cause"), _skip_validate=True)
    if rebuilt != document:
        raise ValidationError("Task 0.3 aggregate differs from strict source recomputation")


def exact_q4_task03_self_test() -> int:
    import copy
    paths = []
    for layer in range(43):
        paths.append({"request_ordinal": 7, "token_ordinal": 0, "event_ordinal": layer,
                      "layer": layer, "path": 0, "flags": 1 | 2 | 4})
    raw = {"requests": [{"request_ordinal": 7, "tokens": [{"token_ordinal": 0}]}],
           "mla_snapshot": {"abi_version": 1, "valid": 1, "enabled": 1,
               "finalized": 1, "request_ordinal": 7, "token_ordinal": 0,
               "capacity": 100, "count": 43, "dropped": 0, "reserved": 0},
           "mla_paths": paths}
    _validate_exact_q4_mla(raw)
    cases = []
    for mutator in (
        lambda d: d["mla_snapshot"].__setitem__("dropped", 1),
        lambda d: d["mla_paths"][1].__setitem__("event_ordinal", 0),
        lambda d: d["mla_paths"][1].__setitem__("layer", 2),
        lambda d: d["mla_paths"][1].__setitem__("flags", 8),
        lambda d: d["mla_snapshot"].__setitem__("request_ordinal", 8),
        lambda d: d["mla_snapshot"].__setitem__("token_ordinal", 1),
        lambda d: d["mla_paths"].pop(),
    ):
        candidate = copy.deepcopy(raw); mutator(candidate); cases.append(candidate)
    for candidate in cases:
        try: _validate_exact_q4_mla(candidate)
        except ValidationError: continue
        raise AssertionError("Task 0.3 MLA tamper case was accepted")
    return len(cases)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--promotion-scalar", type=Path, action="append", default=[])
    parser.add_argument("--promotion-candidate", type=Path, action="append", default=[])
    parser.add_argument("--promotion-output", type=Path)
    parser.add_argument("--promotion-oracle", type=Path)
    parser.add_argument("--validate-promotion", type=Path)
    parser.add_argument("--cache-promotion-fixed", type=Path, action="append", default=[])
    parser.add_argument("--cache-promotion-candidate", type=Path, action="append", default=[])
    parser.add_argument("--cache-promotion-output", type=Path)
    parser.add_argument("--validate-cache-promotion", type=Path)
    parser.add_argument("--oracle-evidence-output", type=Path)
    parser.add_argument("--oracle-result", type=Path)
    parser.add_argument("--oracle-fixture", type=Path)
    parser.add_argument("--oracle-model", type=Path)
    parser.add_argument("--oracle-packed", type=Path)
    parser.add_argument("--oracle-runner", type=Path)
    parser.add_argument("--oracle-log", type=Path)
    parser.add_argument("--oracle-workspace-revision")
    parser.add_argument("--validate-oracle-evidence", type=Path)
    parser.add_argument("--validate-exact-q4-latency", type=Path)
    parser.add_argument("--exact-q4-raw-jsonl", type=Path)
    parser.add_argument("--exact-q4-identity", type=Path)
    parser.add_argument("--exact-q4-select")
    parser.add_argument("--exact-q4-output", type=Path)
    parser.add_argument("--exact-q4-task03-source", action="append", default=[])
    parser.add_argument("--exact-q4-task03-output", type=Path)
    parser.add_argument("--exact-q4-task03-drift-root-cause")
    parser.add_argument("--validate-exact-q4-task03", type=Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        promotion_self_test()
        cache_promotion_self_test()
        exact_count = exact_q4_latency_self_test()
        task03_count = exact_q4_task03_self_test()
        print(f"benchmark_report self-test: PASS (exact-Q4 cases: {exact_count}, Task 0.3 tamper cases: {task03_count})")
        return 0
    if args.validate_exact_q4_task03 is not None:
        raw = args.validate_exact_q4_task03.read_bytes()
        document = json.loads(raw)
        if raw != (json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"):
            raise ValidationError("Task 0.3 aggregate is not canonical JSON")
        validate_exact_q4_task03_document(document)
        print(f"exact-Q4 Task 0.3 aggregate valid: {args.validate_exact_q4_task03}")
        return 0
    if args.exact_q4_task03_source or args.exact_q4_task03_output is not None:
        if args.exact_q4_task03_output is None:
            raise ValidationError("--exact-q4-task03-output is required")
        document = build_exact_q4_task03_document(args.exact_q4_task03_source, args.exact_q4_task03_drift_root_cause)
        write_canonical_atomic(args.exact_q4_task03_output, document)
        print(f"exact-Q4 Task 0.3 aggregate: {args.exact_q4_task03_output}")
        return 0
    if args.exact_q4_raw_jsonl is not None or args.exact_q4_output is not None:
        if args.exact_q4_raw_jsonl is None or args.exact_q4_identity is None or args.exact_q4_select is None or args.exact_q4_output is None:
            raise ValidationError("raw conversion requires --exact-q4-raw-jsonl, --exact-q4-identity, --exact-q4-select, and --exact-q4-output")
        try:
            selected = [int(value) for value in args.exact_q4_select.split(",")]
        except ValueError as exc:
            raise ValidationError("--exact-q4-select must be comma-separated request ordinals") from exc
        document = exact_q4_document_from_raw_jsonl(args.exact_q4_raw_jsonl, args.exact_q4_identity, selected)
        write_canonical_atomic(args.exact_q4_output, document)
        print(f"exact-Q4 canonical artifact: {args.exact_q4_output}")
        return 0
    if args.validate_exact_q4_latency is not None:
        raw = args.validate_exact_q4_latency.read_bytes()
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid exact-Q4 JSON: {exc}") from exc
        if raw != exact_q4_canonical_bytes(document):
            raise ValidationError("exact-Q4 artifact is not canonical compact UTF-8 with one newline")
        canonical_fixture_path = (
            Path(__file__).resolve().parent.parent
            / "tests/fixtures/dmlx_exact_q4_latency_v1.json"
        ).resolve()
        validate_exact_q4_latency(
            document,
            allow_canonical_fixture_legacy_logits=(
                args.validate_exact_q4_latency.resolve() == canonical_fixture_path
            ),
        )
        print(f"exact-Q4 latency artifact valid: {args.validate_exact_q4_latency}")
        return 0
    if args.validate is not None:
        with args.validate.open(encoding="utf-8") as handle:
            validate_document(json.load(handle))
        print(f"benchmark report valid: {args.validate}")
        return 0
    if args.validate_oracle_evidence is not None:
        with args.validate_oracle_evidence.open(encoding="utf-8") as handle:
            validate_oracle_evidence(json.load(handle))
        print(f"batch oracle evidence valid: {args.validate_oracle_evidence}")
        return 0
    if args.validate_promotion is not None:
        with args.validate_promotion.open(encoding="utf-8") as handle:
            validate_promotion_document(json.load(handle))
        print(f"benchmark promotion report valid: {args.validate_promotion}")
        return 0
    if args.validate_cache_promotion is not None:
        with args.validate_cache_promotion.open(encoding="utf-8") as handle:
            validate_cache_promotion_document(json.load(handle))
        print(f"cache promotion report valid: {args.validate_cache_promotion}")
        return 0
    if args.oracle_evidence_output is not None:
        required = {
            "--oracle-result": args.oracle_result,
            "--oracle-fixture": args.oracle_fixture,
            "--oracle-model": args.oracle_model,
            "--oracle-packed": args.oracle_packed,
            "--oracle-runner": args.oracle_runner,
            "--oracle-log": args.oracle_log,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValidationError(
                "oracle evidence generation requires " + ", ".join(missing)
            )
        evidence = build_oracle_evidence(
            args.oracle_result,
            args.oracle_fixture,
            args.oracle_model,
            args.oracle_packed,
            args.oracle_runner,
            args.oracle_log,
            args.oracle_workspace_revision,
        )
        write_atomic(args.oracle_evidence_output, evidence)
        print(f"batch oracle evidence JSON: {args.oracle_evidence_output}")
        return 0
    if args.cache_promotion_fixed or args.cache_promotion_candidate or args.cache_promotion_output is not None:
        if args.cache_promotion_output is None:
            raise ValidationError("--cache-promotion-output is required for cache promotion aggregation")
        promotion = build_cache_promotion_document(
            args.cache_promotion_fixed, args.cache_promotion_candidate
        )
        write_atomic(args.cache_promotion_output, promotion)
        print(f"cache promotion JSON: {args.cache_promotion_output}")
        return 0
    if args.promotion_scalar or args.promotion_candidate or args.promotion_output is not None:
        if args.promotion_output is None or args.promotion_oracle is None:
            raise ValidationError(
                "--promotion-output and --promotion-oracle are required for promotion aggregation"
            )
        promotion = build_promotion_document(
            args.promotion_scalar, args.promotion_candidate, args.promotion_oracle
        )
        write_atomic(args.promotion_output, promotion)
        print(f"benchmark promotion JSON: {args.promotion_output}")
        return 0

    document = build_document(os.environ)
    output = Path(args.output)
    write_atomic(output, document)
    print(f"benchmark JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
