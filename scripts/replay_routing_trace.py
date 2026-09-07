#!/usr/bin/env python3
"""Validate schema-v1 routing traces and replay the observed fixed-hot baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from cache_policy import (
    CachePolicyError,
    RouteEvent,
    compare_policies,
    run_self_test as run_cache_policy_self_test,
)

SCHEMA_VERSION = 1
TOP_K = 6
FINGERPRINT_KIND = "fnv1a64_artifact_metadata_v1"
CACHE_HOT = 1
CACHE_PREFETCH = 2
CACHE_BATCH_POOL = 3
CACHE_COLD = 4
CACHE_STATE_NAMES = {
    CACHE_HOT: "hot",
    CACHE_PREFETCH: "prefetch",
    CACHE_BATCH_POOL: "batch_pool",
    CACHE_COLD: "cold",
}
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


class TraceValidationError(ValueError):
    """Raised when a trace is malformed, incomplete, or internally inconsistent."""


def _fail(line_number: int, message: str) -> TraceValidationError:
    return TraceValidationError(f"line {line_number}: {message}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise TraceValidationError(f"non-standard JSON constant {value!r}")


def _parse_json_line(text: str, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TraceValidationError) as error:
        raise _fail(line_number, f"invalid JSON: {error}") from error
    if type(value) is not dict:
        raise _fail(line_number, "record must be a JSON object")
    return value


def _required(record: dict[str, Any], key: str, line_number: int) -> Any:
    if key not in record:
        raise _fail(line_number, f"missing required field {key!r}")
    return record[key]


def _integer(
    record: dict[str, Any],
    key: str,
    line_number: int,
    *,
    minimum: int = 0,
    maximum: int = UINT64_MAX,
) -> int:
    value = _required(record, key, line_number)
    if type(value) is not int:
        raise _fail(line_number, f"field {key!r} must be an integer, not {type(value).__name__}")
    if value < minimum or value > maximum:
        raise _fail(line_number, f"field {key!r}={value} is outside [{minimum}, {maximum}]")
    return value


def _string(record: dict[str, Any], key: str, line_number: int) -> str:
    value = _required(record, key, line_number)
    if type(value) is not str or not value:
        raise _fail(line_number, f"field {key!r} must be a non-empty string")
    return value


def _array(record: dict[str, Any], key: str, line_number: int, length: int) -> list[Any]:
    value = _required(record, key, line_number)
    if type(value) is not list:
        raise _fail(line_number, f"field {key!r} must be an array")
    if len(value) != length:
        raise _fail(line_number, f"field {key!r} must contain exactly {length} entries")
    return value


def _validate_schema(record: dict[str, Any], line_number: int) -> None:
    version = _integer(record, "schema_version", line_number, maximum=UINT32_MAX)
    if version != SCHEMA_VERSION:
        raise _fail(line_number, f"unsupported schema_version {version}; expected {SCHEMA_VERSION}")


def _validate_header(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    if record.get("kind") != "header":
        raise _fail(line_number, "first record must be a header")
    _validate_schema(record, line_number)
    fingerprint_kind = _string(record, "model_fingerprint_kind", line_number)
    if fingerprint_kind != FINGERPRINT_KIND:
        raise _fail(line_number, f"unsupported model fingerprint kind {fingerprint_kind!r}")

    fingerprint = _integer(record, "model_fingerprint_u64", line_number)
    n_layers = _integer(record, "n_layers", line_number, minimum=1, maximum=4096)
    n_experts = _integer(record, "n_experts", line_number, minimum=TOP_K, maximum=65536)
    top_k = _integer(record, "top_k", line_number, minimum=1, maximum=UINT32_MAX)
    cache_capacity = _integer(record, "cache_capacity", line_number, maximum=UINT32_MAX)
    expert_size = _integer(record, "expert_size_bytes", line_number, minimum=1)
    record_capacity = _integer(record, "record_capacity", line_number, minimum=1)

    if top_k != TOP_K:
        raise _fail(line_number, f"top_k must be {TOP_K}, got {top_k}")
    if cache_capacity > n_experts:
        raise _fail(line_number, "cache_capacity cannot exceed n_experts")

    return {
        "model_fingerprint_kind": fingerprint_kind,
        "model_fingerprint_u64": fingerprint,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "top_k": top_k,
        "cache_capacity": cache_capacity,
        "expert_size_bytes": expert_size,
        "record_capacity": record_capacity,
    }


def _new_totals() -> dict[str, Any]:
    return {
        "requests": 0,
        "route_records": 0,
        "expert_routes": 0,
        "hits": 0,
        "misses": 0,
        "read_bytes": 0,
        "cache_states": {name: 0 for name in CACHE_STATE_NAMES.values()},
        "records_attempted": 0,
        "records_dropped": 0,
        "invalid_records": 0,
    }


def _validate_begin(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    _validate_schema(record, line_number)
    return {
        "request_id": _integer(record, "request_id", line_number),
        "initial_phase": _integer(record, "initial_phase", line_number, maximum=2),
        "line": line_number,
        "route_count": 0,
        "event_sequences": [],
    }


def _validate_integer_array(
    values: list[Any],
    field: str,
    line_number: int,
    *,
    minimum: int,
    maximum: int,
) -> list[int]:
    result: list[int] = []
    for index, value in enumerate(values):
        if type(value) is not int:
            raise _fail(line_number, f"{field}[{index}] must be an integer")
        if value < minimum or value > maximum:
            raise _fail(
                line_number,
                f"{field}[{index}]={value} is outside [{minimum}, {maximum}]",
            )
        result.append(value)
    return result


def _validate_route(
    record: dict[str, Any],
    line_number: int,
    header: dict[str, Any],
    request: dict[str, Any],
    totals: dict[str, Any],
) -> None:
    _validate_schema(record, line_number)
    request_id = _integer(record, "request_id", line_number)
    if request_id != request["request_id"]:
        raise _fail(line_number, "route request_id does not match active request")

    event_seq = _integer(record, "event_seq", line_number)
    sequences: list[int] = request["event_sequences"]
    if sequences and event_seq <= sequences[-1]:
        raise _fail(line_number, "event_seq must be strictly increasing within a request")

    # Schema v1 declares event_seq as the total order. It does not declare
    # token-major versus layer-major traversal, and phase values may cycle
    # during speculative verification. Validate these fields, but do not infer
    # an ordering that the producer did not encode.
    _integer(record, "phase", line_number, maximum=2)
    _integer(record, "token_position", line_number, maximum=UINT32_MAX)
    _integer(record, "layer", line_number, maximum=header["n_layers"] - 1)

    expert_ids = _validate_integer_array(
        _array(record, "expert_ids", line_number, TOP_K),
        "expert_ids",
        line_number,
        minimum=0,
        maximum=header["n_experts"] - 1,
    )
    if len(set(expert_ids)) != TOP_K:
        raise _fail(line_number, "expert_ids must contain six distinct experts")

    weights = _array(record, "weights", line_number, TOP_K)
    for index, weight in enumerate(weights):
        if type(weight) not in (int, float) or isinstance(weight, bool):
            raise _fail(line_number, f"weights[{index}] must be numeric")
        if not math.isfinite(float(weight)):
            raise _fail(line_number, f"weights[{index}] must be finite")

    cache_states = _validate_integer_array(
        _array(record, "cache_states", line_number, TOP_K),
        "cache_states",
        line_number,
        minimum=CACHE_HOT,
        maximum=CACHE_COLD,
    )
    route_bytes = _validate_integer_array(
        _array(record, "route_bytes", line_number, TOP_K),
        "route_bytes",
        line_number,
        minimum=0,
        maximum=header["expert_size_bytes"],
    )
    pread_bytes = _integer(record, "pread_bytes", line_number)
    _integer(record, "io_wait_ns", line_number)
    short_reads = _integer(record, "short_reads", line_number, maximum=UINT32_MAX)
    io_status = _integer(record, "io_status", line_number, minimum=-(1 << 31), maximum=(1 << 31) - 1)

    if sum(route_bytes) != pread_bytes:
        raise _fail(line_number, "sum(route_bytes) does not equal pread_bytes")
    if short_reads != 0 or io_status != 0:
        raise _fail(line_number, "failed or short expert I/O cannot be replayed")
    for index, (state, byte_count) in enumerate(zip(cache_states, route_bytes)):
        if state != CACHE_COLD and byte_count != 0:
            raise _fail(line_number, f"route_bytes[{index}] must be zero for a non-cold route")
        if state == CACHE_COLD and byte_count != header["expert_size_bytes"]:
            raise _fail(line_number, f"cold route_bytes[{index}] must equal expert_size_bytes")

    sequences.append(event_seq)
    request["route_count"] += 1
    hot_count = cache_states.count(CACHE_HOT)
    totals["route_records"] += 1
    totals["expert_routes"] += TOP_K
    totals["hits"] += hot_count
    totals["misses"] += TOP_K - hot_count
    totals["read_bytes"] += pread_bytes
    for state in cache_states:
        totals["cache_states"][CACHE_STATE_NAMES[state]] += 1


def _validate_end(
    record: dict[str, Any],
    line_number: int,
    header: dict[str, Any],
    request: dict[str, Any],
    totals: dict[str, Any],
) -> None:
    _validate_schema(record, line_number)
    request_id = _integer(record, "request_id", line_number)
    if request_id != request["request_id"]:
        raise _fail(line_number, "request_end request_id does not match active request")

    attempted = _integer(record, "records_attempted", line_number)
    buffered = _integer(record, "records_buffered", line_number)
    written = _integer(record, "records_written", line_number)
    dropped = _integer(record, "records_dropped", line_number)
    invalid = _integer(record, "invalid_records", line_number)
    write_errors = _integer(record, "write_errors", line_number)
    visible = request["route_count"]

    if visible == 0:
        raise _fail(line_number, "request contains no route records")
    if buffered != visible or written != visible:
        raise _fail(line_number, "footer buffered/written counts do not match visible routes")
    if buffered > header["record_capacity"]:
        raise _fail(line_number, "records_buffered exceeds header record_capacity")
    if attempted != buffered + dropped:
        raise _fail(line_number, "records_attempted must equal records_buffered + records_dropped")
    if invalid > dropped:
        raise _fail(line_number, "invalid_records cannot exceed records_dropped")
    if write_errors != 0:
        raise _fail(line_number, "trace has write errors and is not complete enough to replay")

    sequences: list[int] = request["event_sequences"]
    if any(sequence >= attempted for sequence in sequences):
        raise _fail(line_number, "event_seq is outside records_attempted range")
    if attempted - len(sequences) != dropped:
        raise _fail(line_number, "event_seq gaps are not explained by records_dropped")

    totals["requests"] += 1
    totals["records_attempted"] += attempted
    totals["records_dropped"] += dropped
    totals["invalid_records"] += invalid


def replay_trace(path: Path, smelt_n: int) -> dict[str, Any]:
    if type(smelt_n) is not int or smelt_n < 0:
        raise TraceValidationError("smelt_n must be a non-negative integer")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise TraceValidationError(f"cannot read trace {path}: {error}") from error
    if not lines:
        raise TraceValidationError("trace is empty")
    if any(not line.strip() for line in lines):
        empty_line = next(index for index, line in enumerate(lines, 1) if not line.strip())
        raise _fail(empty_line, "blank lines are not valid JSONL records")

    header = _validate_header(_parse_json_line(lines[0], 1), 1)
    if smelt_n != header["cache_capacity"]:
        raise TraceValidationError(
            f"--smelt-n {smelt_n} does not match trace cache_capacity "
            f"{header['cache_capacity']}; schema v1 has no fixed-hot membership metadata "
            "for counterfactual replay"
        )

    totals = _new_totals()
    active_request: dict[str, Any] | None = None
    seen_request_ids: set[int] = set()

    for line_number, text in enumerate(lines[1:], 2):
        record = _parse_json_line(text, line_number)
        kind = record.get("kind")
        if kind == "header":
            raise _fail(line_number, "header may appear only once at the start")
        if kind == "request_begin":
            if active_request is not None:
                raise _fail(line_number, "request_begin encountered before prior request_end")
            active_request = _validate_begin(record, line_number)
            if active_request["request_id"] in seen_request_ids:
                raise _fail(line_number, "request_id must be unique within a trace")
            seen_request_ids.add(active_request["request_id"])
        elif kind == "route":
            if active_request is None:
                raise _fail(line_number, "route record is outside a request envelope")
            _validate_route(record, line_number, header, active_request, totals)
        elif kind == "request_end":
            if active_request is None:
                raise _fail(line_number, "request_end has no matching request_begin")
            _validate_end(record, line_number, header, active_request, totals)
            active_request = None
        else:
            raise _fail(line_number, f"unknown record kind {kind!r}")

    if active_request is not None:
        raise TraceValidationError(
            f"truncated trace: request {active_request['request_id']} begun on line "
            f"{active_request['line']} has no request_end"
        )
    if totals["requests"] == 0:
        raise TraceValidationError("trace contains no complete request")

    expert_routes = totals["expert_routes"]
    hit_rate = totals["hits"] / expert_routes if expert_routes else None
    return {
        "schema_version": SCHEMA_VERSION,
        "trace": {
            "path": str(path),
            "record_order": "event_seq",
            **header,
        },
        "policy": {
            "name": "fixed_hot_observed_v1",
            "smelt_n": smelt_n,
            "hits": totals["hits"],
            "misses": totals["misses"],
            "hit_rate": hit_rate,
            "read_bytes": totals["read_bytes"],
            "slot_churn": 0,
            "accounting": "HOT is a hit; every other observed cache state is a miss",
        },
        "records": {
            "requests": totals["requests"],
            "route_records": totals["route_records"],
            "expert_routes": expert_routes,
            "cache_states": totals["cache_states"],
            "records_attempted": totals["records_attempted"],
            "records_dropped": totals["records_dropped"],
            "invalid_records": totals["invalid_records"],
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise TraceValidationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _policy_events(path: Path) -> list[RouteEvent]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise TraceValidationError(f"cannot read trace {path}: {error}") from error
    events: list[RouteEvent] = []
    for line_number, text in enumerate(lines[1:], 2):
        record = _parse_json_line(text, line_number)
        if record.get("kind") != "route":
            continue
        events.append(
            RouteEvent(
                order_index=len(events),
                request_id=record["request_id"],
                event_seq=record["event_seq"],
                phase=record["phase"],
                token_position=record["token_position"],
                layer=record["layer"],
                expert_ids=tuple(record["expert_ids"]),
                cache_states=tuple(record["cache_states"]),
                route_bytes=tuple(record["route_bytes"]),
            )
        )
    return events


def replay_trace_with_policies(
    path: Path,
    smelt_n: int,
    full_resident_layers: int,
) -> dict[str, Any]:
    baseline = replay_trace(path, smelt_n)
    records = baseline["records"]
    if records["records_dropped"] != 0 or records["invalid_records"] != 0:
        raise TraceValidationError(
            "cache-policy replay requires a complete trace with zero dropped or "
            "invalid route records"
        )
    header = baseline["trace"]
    try:
        comparison = compare_policies(
            _policy_events(path),
            n_layers=header["n_layers"],
            n_experts=header["n_experts"],
            capacity=smelt_n,
            expert_size_bytes=header["expert_size_bytes"],
            full_resident_layers=full_resident_layers,
        )
    except CachePolicyError as error:
        raise TraceValidationError(f"cache-policy replay failed: {error}") from error
    return {
        **baseline,
        "policy_comparison": comparison,
    }


_METRIC_INTEGER_FIELDS = (
    "route_records",
    "expert_routes",
    "hits",
    "misses",
    "read_bytes",
    "admissions",
    "evictions",
    "slot_churn",
)


def _empty_policy_metrics() -> dict[str, int]:
    return {key: 0 for key in _METRIC_INTEGER_FIELDS}


def _add_policy_metrics(destination: dict[str, int], source: dict[str, Any]) -> None:
    for key in _METRIC_INTEGER_FIELDS:
        value = source.get(key)
        if type(value) is not int or value < 0:
            raise TraceValidationError(f"policy metric {key} is invalid")
        destination[key] += value


def _finalize_policy_metrics(metrics: dict[str, int]) -> dict[str, Any]:
    accesses = metrics["expert_routes"]
    return {
        **metrics,
        "hit_rate": metrics["hits"] / accesses if accesses else None,
        "bytes_per_expert_route": metrics["read_bytes"] / accesses if accesses else None,
    }


def _aggregate_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    if not comparisons:
        raise TraceValidationError("corpus has no policy comparisons")
    reference_geometry = comparisons[0]["geometry"]
    policy_names = [policy["name"] for policy in comparisons[0]["policies"]]
    aggregate: dict[str, dict[str, Any]] = {}
    for policy in comparisons[0]["policies"]:
        aggregate[policy["name"]] = {
            "name": policy["name"],
            "role": policy["role"],
            "eligible_for_online_ab": policy["eligible_for_online_ab"],
            "assumptions": policy["assumptions"],
            "memory": policy["memory"],
            "totals": _empty_policy_metrics(),
            "by_phase": {},
            "by_layer": {},
            "by_phase_layer": {},
            "cases_better_than_fixed_hot": 0,
        }

    for comparison in comparisons:
        if comparison["geometry"] != reference_geometry:
            raise TraceValidationError("corpus policy geometry is inconsistent")
        if [policy["name"] for policy in comparison["policies"]] != policy_names:
            raise TraceValidationError("corpus policy set/order is inconsistent")
        per_case = {policy["name"]: policy for policy in comparison["policies"]}
        baseline_bytes = per_case["fixed_hot_observed"]["totals"]["read_bytes"]
        for name, source_policy in per_case.items():
            destination = aggregate[name]
            if (
                source_policy["role"] != destination["role"]
                or source_policy["memory"] != destination["memory"]
                or source_policy["assumptions"] != destination["assumptions"]
            ):
                raise TraceValidationError(f"policy metadata drift for {name}")
            _add_policy_metrics(destination["totals"], source_policy["totals"])
            if source_policy["totals"]["read_bytes"] < baseline_bytes:
                destination["cases_better_than_fixed_hot"] += 1
            for phase, metrics in source_policy["by_phase"].items():
                target = destination["by_phase"].setdefault(
                    phase, _empty_policy_metrics()
                )
                _add_policy_metrics(target, metrics)
            for layer, metrics in source_policy["by_layer"].items():
                target = destination["by_layer"].setdefault(
                    layer, _empty_policy_metrics()
                )
                _add_policy_metrics(target, metrics)
            for phase, layers in source_policy["by_phase_layer"].items():
                target_layers = destination["by_phase_layer"].setdefault(phase, {})
                for layer, metrics in layers.items():
                    target = target_layers.setdefault(layer, _empty_policy_metrics())
                    _add_policy_metrics(target, metrics)

    policies: list[dict[str, Any]] = []
    for name in policy_names:
        policy = aggregate[name]
        policy["totals"] = _finalize_policy_metrics(policy["totals"])
        policy["by_phase"] = {
            phase: _finalize_policy_metrics(metrics)
            for phase, metrics in sorted(policy["by_phase"].items())
        }
        policy["by_layer"] = {
            layer: _finalize_policy_metrics(metrics)
            for layer, metrics in sorted(
                policy["by_layer"].items(), key=lambda item: int(item[0])
            )
        }
        policy["by_phase_layer"] = {
            phase: {
                layer: _finalize_policy_metrics(metrics)
                for layer, metrics in sorted(
                    layers.items(), key=lambda item: int(item[0])
                )
            }
            for phase, layers in sorted(policy["by_phase_layer"].items())
        }
        policies.append(policy)

    by_name = {policy["name"]: policy for policy in policies}
    baseline = by_name["fixed_hot_observed"]["totals"]
    for policy in policies:
        totals = policy["totals"]
        policy["delta_vs_fixed_hot"] = {
            "read_bytes": totals["read_bytes"] - baseline["read_bytes"],
            "read_bytes_percent": (
                (totals["read_bytes"] - baseline["read_bytes"])
                * 100.0
                / baseline["read_bytes"]
                if baseline["read_bytes"]
                else None
            ),
            "hit_rate_percentage_points": (
                (totals["hit_rate"] - baseline["hit_rate"]) * 100.0
                if totals["hit_rate"] is not None and baseline["hit_rate"] is not None
                else None
            ),
            "slot_churn": totals["slot_churn"] - baseline["slot_churn"],
        }
        policy["all_cases_better_than_fixed_hot"] = (
            policy["cases_better_than_fixed_hot"] == len(comparisons)
        )

    candidates = [
        policy
        for policy in policies
        if policy["role"] in {"candidate", "reference", "isolated_oracle_candidate"}
    ]
    candidates.sort(
        key=lambda policy: (
            policy["totals"]["read_bytes"],
            policy["totals"]["slot_churn"],
            policy["name"],
        )
    )
    ranking = [
        {
            "rank": rank,
            "name": policy["name"],
            "role": policy["role"],
            "eligible_for_online_ab": policy["eligible_for_online_ab"],
            "cases_better_than_fixed_hot": policy["cases_better_than_fixed_hot"],
            "all_cases_better_than_fixed_hot": policy["all_cases_better_than_fixed_hot"],
            "read_bytes": policy["totals"]["read_bytes"],
            "read_bytes_delta_vs_fixed_hot": policy["delta_vs_fixed_hot"]["read_bytes"],
            "hit_rate": policy["totals"]["hit_rate"],
            "slot_churn": policy["totals"]["slot_churn"],
        }
        for rank, policy in enumerate(candidates, 1)
    ]
    return {
        "geometry": reference_geometry,
        "case_count": len(comparisons),
        "policies": policies,
        "candidate_ranking": ranking,
        "theoretical_lower_bound_policy": "belady_same_seed_lower_bound",
    }


def _expected_corpus_tooling() -> dict[str, str]:
    project_dir = Path(__file__).resolve().parents[1]
    return {
        "project_root": str(project_dir),
        "capture_script_sha256": _sha256_file(
            project_dir / "scripts" / "capture_routing_corpus.py"
        ),
        "replay_script_sha256": _sha256_file(Path(__file__).resolve()),
        "fingerprint_script_sha256": _sha256_file(
            project_dir / "scripts" / "compare_batch_oracle.py"
        ),
    }


def replay_corpus(
    metadata_path: Path,
    smelt_n: int,
    full_resident_layers: int,
) -> dict[str, Any]:
    try:
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(
            metadata_bytes,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TraceValidationError(f"cannot read corpus metadata: {error}") from error
    if (
        type(metadata) is not dict
        or metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("kind") != "dmlx-exact-q4-routing-corpus"
    ):
        raise TraceValidationError("unsupported corpus metadata schema")
    if smelt_n != 20:
        raise TraceValidationError("formal corpus replay requires --smelt-n 20")
    configuration = metadata.get("configuration")
    required_configuration = {
        "mode": "native",
        "single_server": True,
        "serial_requests": True,
        "server_max_tokens": 30,
        "NATIVE_SMELT_N": 20,
        "NATIVE_PREFETCH": 0,
        "NATIVE_TELEMETRY": 1,
        "NATIVE_ROUTING_TRACE_CAPACITY": 65_536,
        "NATIVE_BATCH_PREFILL": 0,
        "NATIVE_BATCH_MLA": 0,
        "NATIVE_EXPERT_BUCKETS": 0,
        "NATIVE_BATCH_EXPERT_KERNELS": 0,
        "NATIVE_DSPARK_BATCH_VERIFY": 0,
    }
    if configuration != required_configuration:
        raise TraceValidationError("corpus configuration is not the formal N=20 capture")
    expected_privacy = {
        "prompt_identity": "sha256 and shape/count metadata only",
        "prompt_text_persisted": False,
        "raw_server_log_retained": False,
        "response_text_persisted": False,
        "trace_fields": "strict schema-v1 numeric whitelist",
    }
    if metadata.get("privacy_contract") != expected_privacy:
        raise TraceValidationError("corpus privacy contract does not match formal capture")
    tooling = metadata.get("tooling")
    expected_tooling = _expected_corpus_tooling()
    if tooling != expected_tooling:
        raise TraceValidationError("corpus tooling provenance does not match current sources")
    current_replay_hash = expected_tooling["replay_script_sha256"]
    cases = metadata.get("cases")
    if type(cases) is not list or not cases:
        raise TraceValidationError("corpus metadata has no cases")

    root = metadata_path.resolve().parent
    case_results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    trace_hashes: list[str] = []
    expected_index = 1
    for case in cases:
        if type(case) is not dict or case.get("serial_request_index") != expected_index:
            raise TraceValidationError("corpus case order is invalid")
        expected_index += 1
        relative = case.get("trace_file")
        if type(relative) is not str or not relative or Path(relative).is_absolute():
            raise TraceValidationError("corpus trace path is invalid")
        trace_path = (root / relative).resolve()
        try:
            trace_path.relative_to(root)
        except ValueError as error:
            raise TraceValidationError("corpus trace escapes metadata directory") from error
        if not trace_path.is_file():
            raise TraceValidationError(f"corpus trace is missing: {trace_path}")
        trace_hash = _sha256_file(trace_path)
        if trace_hash != case.get("trace_sha256"):
            raise TraceValidationError(f"corpus trace hash mismatch: {trace_path}")
        result = replay_trace_with_policies(
            trace_path, smelt_n, full_resident_layers
        )
        if result["trace"]["model_fingerprint_u64"] != case.get(
            "trace_model_fingerprint_u64"
        ):
            raise TraceValidationError("corpus trace fingerprint mismatch")
        comparison = result["policy_comparison"]
        comparisons.append(comparison)
        trace_hashes.append(trace_hash)
        case_results.append(
            {
                "name": case.get("name"),
                "shape": case.get("shape"),
                "serial_request_index": case.get("serial_request_index"),
                "trace_path": str(trace_path),
                "trace_sha256": trace_hash,
                "v5": {
                    "records": result["records"],
                    "fixed_hot_observed": result["policy"],
                },
                "policy_comparison": comparison,
            }
        )

    aggregate = _aggregate_comparisons(comparisons)
    return {
        "schema_version": 1,
        "kind": "dmlx-exact-q4-cache-policy-corpus-replay",
        "runtime_mutation": False,
        "source": {
            "corpus_metadata_path": str(metadata_path.resolve()),
            "corpus_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "trace_sha256": trace_hashes,
            "replay_script_sha256": current_replay_hash,
            "cache_policy_script_sha256": _sha256_file(
                Path(__file__).resolve().with_name("cache_policy.py")
            ),
            "project_revision": metadata.get("project_revision"),
            "model": metadata.get("model"),
        },
        "configuration": {
            "smelt_n": smelt_n,
            "full_resident_layers": full_resident_layers,
            "case_reset": "each split trace starts from its reconstructed observed-hot seed",
            "candidate_gate": "eligible policy must beat fixed-hot physical bytes in every case",
        },
        "cases": case_results,
        "aggregate": aggregate,
    }


def _write_json_private(path: Path, value: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        path.chmod(0o600)
    except OSError as error:
        raise TraceValidationError(f"cannot write replay output {path}: {error}") from error


def _expect_invalid(path: Path, smelt_n: int, description: str) -> None:
    try:
        replay_trace(path, smelt_n)
    except TraceValidationError:
        return
    raise AssertionError(f"{description} trace was accepted")


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_test() -> None:
    run_cache_policy_self_test()

    fixture = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "routing_trace_small.jsonl"
    )
    result = replay_trace(fixture, 20)
    policy = result["policy"]
    records = result["records"]
    assert policy["hits"] == 17
    assert policy["misses"] == 19
    assert policy["read_bytes"] == 1500
    assert policy["slot_churn"] == 0
    assert records["requests"] == 1
    assert records["route_records"] == 6
    assert records["expert_routes"] == 36

    policy_result = replay_trace_with_policies(fixture, 20, 0)
    comparison = policy_result["policy_comparison"]
    by_name = {item["name"]: item for item in comparison["policies"]}
    expected = {
        "fixed_hot_observed": (17, 19, 1500, 0),
        "lru_observed_hot_seeded": (17, 19, 1900, 19),
        "lfu_observed_hot_seeded": (17, 19, 1900, 19),
        "trust_os_no_user_cache": (0, 36, 3600, 0),
        "static_frequency_oracle": (36, 0, 0, 0),
        "belady_same_seed_lower_bound": (17, 19, 1900, 19),
    }
    for name, (hits, misses, read_bytes, churn) in expected.items():
        totals = by_name[name]["totals"]
        assert totals["hits"] == hits, (name, totals)
        assert totals["misses"] == misses, (name, totals)
        assert totals["read_bytes"] == read_bytes, (name, totals)
        assert totals["slot_churn"] == churn, (name, totals)
    assert comparison == replay_trace_with_policies(fixture, 20, 0)[
        "policy_comparison"
    ]
    assert by_name["fixed_hot_observed"]["by_phase"]["prefill"][
        "read_bytes"
    ] == 1000
    assert by_name["fixed_hot_observed"]["by_layer"]["2"][
        "read_bytes"
    ] == 500
    assert comparison["candidate_ranking"][0]["name"] == "static_frequency_oracle"

    source_lines = fixture.read_text(encoding="utf-8").splitlines()
    with tempfile.TemporaryDirectory(prefix="dmlx-routing-replay-") as directory:
        root = Path(directory)

        corrupt = root / "corrupt.jsonl"
        _write_lines(corrupt, source_lines[:2] + ["{not-json}"])
        _expect_invalid(corrupt, 20, "corrupt")

        invalid_utf8 = root / "invalid-utf8.jsonl"
        invalid_utf8.write_bytes(b"\xff\xfe")
        _expect_invalid(invalid_utf8, 20, "invalid UTF-8")

        truncated = root / "truncated.jsonl"
        _write_lines(truncated, source_lines[:-1])
        _expect_invalid(truncated, 20, "truncated")

        out_of_range_lines = list(source_lines)
        route = json.loads(out_of_range_lines[2])
        route["expert_ids"][0] = 32
        out_of_range_lines[2] = json.dumps(route, separators=(",", ":"))
        out_of_range = root / "out-of-range.jsonl"
        _write_lines(out_of_range, out_of_range_lines)
        _expect_invalid(out_of_range, 20, "out-of-range")

        bad_event_lines = list(source_lines)
        route = json.loads(bad_event_lines[3])
        route["event_seq"] = 0
        bad_event_lines[3] = json.dumps(route, separators=(",", ":"))
        bad_event = root / "bad-event-order.jsonl"
        _write_lines(bad_event, bad_event_lines)
        _expect_invalid(bad_event, 20, "out-of-order event_seq")

        # Schema v1 permits producer-defined layer-major traversal and phase
        # cycling; event_seq remains the only declared total order.
        layer_major_indices = (2, 5, 3, 6, 4, 7)
        layer_major_lines = source_lines[:2]
        for event_seq, source_index in enumerate(layer_major_indices):
            route = json.loads(source_lines[source_index])
            route["event_seq"] = event_seq
            route["phase"] = (1, 2, 1, 2, 1, 1)[event_seq]
            layer_major_lines.append(json.dumps(route, separators=(",", ":")))
        layer_major_lines.append(source_lines[-1])
        layer_major = root / "layer-major-phase-cycle.jsonl"
        _write_lines(layer_major, layer_major_lines)
        layer_major_result = replay_trace(layer_major, 20)
        assert layer_major_result["records"]["route_records"] == 6

        explained_gap_lines = source_lines[:4] + source_lines[5:-1]
        footer = json.loads(source_lines[-1])
        footer.update(records_buffered=5, records_written=5, records_dropped=1)
        explained_gap_lines.append(json.dumps(footer, separators=(",", ":")))
        explained_gap = root / "explained-gap.jsonl"
        _write_lines(explained_gap, explained_gap_lines)
        gap_result = replay_trace(explained_gap, 20)
        assert gap_result["records"]["records_dropped"] == 1
        try:
            replay_trace_with_policies(explained_gap, 20, 0)
        except TraceValidationError as error:
            assert "complete trace" in str(error)
        else:
            raise AssertionError("policy replay accepted a dropped route event")

        second_envelope: list[str] = []
        for text in source_lines[1:]:
            record = json.loads(text)
            record["request_id"] = 9002
            second_envelope.append(json.dumps(record, separators=(",", ":")))
        multiple_requests = root / "multiple-requests.jsonl"
        _write_lines(multiple_requests, source_lines + second_envelope)
        multiple_result = replay_trace(multiple_requests, 20)
        assert multiple_result["records"]["requests"] == 2

        inconsistent_lines = list(source_lines)
        route = json.loads(inconsistent_lines[7])
        route["cache_states"][3] = CACHE_COLD
        route["route_bytes"][3] = 100
        route["pread_bytes"] += 100
        inconsistent_lines[7] = json.dumps(route, separators=(",", ":"))
        inconsistent = root / "inconsistent-hot-membership.jsonl"
        _write_lines(inconsistent, inconsistent_lines)
        try:
            replay_trace_with_policies(inconsistent, 20, 0)
        except TraceValidationError:
            pass
        else:
            raise AssertionError("inconsistent HOT membership was accepted")

        try:
            replay_trace_with_policies(fixture, 20, 1)
        except TraceValidationError:
            pass
        else:
            raise AssertionError("incorrect full-resident-layer assumption was accepted")

        valid_metadata = {
            "schema_version": SCHEMA_VERSION,
            "kind": "dmlx-exact-q4-routing-corpus",
            "configuration": {
                "mode": "native",
                "single_server": True,
                "serial_requests": True,
                "server_max_tokens": 30,
                "NATIVE_SMELT_N": 20,
                "NATIVE_PREFETCH": 0,
                "NATIVE_TELEMETRY": 1,
                "NATIVE_ROUTING_TRACE_CAPACITY": 65_536,
                "NATIVE_BATCH_PREFILL": 0,
                "NATIVE_BATCH_MLA": 0,
                "NATIVE_EXPERT_BUCKETS": 0,
                "NATIVE_BATCH_EXPERT_KERNELS": 0,
                "NATIVE_DSPARK_BATCH_VERIFY": 0,
            },
            "privacy_contract": {
                "prompt_identity": "sha256 and shape/count metadata only",
                "prompt_text_persisted": False,
                "raw_server_log_retained": False,
                "response_text_persisted": False,
                "trace_fields": "strict schema-v1 numeric whitelist",
            },
            "tooling": _expected_corpus_tooling(),
            "cases": [{}],
        }

        def expect_metadata_rejected(
            value: dict[str, Any], description: str, expected_message: str
        ) -> None:
            path = root / f"bad-metadata-{description}.json"
            _write_json_private(path, value)
            try:
                replay_corpus(path, 20, 0)
            except TraceValidationError as error:
                assert expected_message in str(error), (description, error)
            else:
                raise AssertionError(f"corpus accepted {description} drift")

        bad_schema = json.loads(json.dumps(valid_metadata))
        bad_schema["schema_version"] = SCHEMA_VERSION + 1
        expect_metadata_rejected(bad_schema, "schema", "unsupported corpus metadata")
        for field in (
            "capture_script_sha256",
            "replay_script_sha256",
            "fingerprint_script_sha256",
        ):
            bad_tooling = json.loads(json.dumps(valid_metadata))
            bad_tooling["tooling"][field] = "0" * 64
            expect_metadata_rejected(
                bad_tooling, field, "tooling provenance does not match"
            )
        bad_configuration = json.loads(json.dumps(valid_metadata))
        bad_configuration["configuration"]["NATIVE_PREFETCH"] = 1
        expect_metadata_rejected(
            bad_configuration, "configuration", "not the formal N=20 capture"
        )
        bad_capacity = json.loads(json.dumps(valid_metadata))
        bad_capacity["configuration"]["NATIVE_ROUTING_TRACE_CAPACITY"] = 1
        expect_metadata_rejected(
            bad_capacity, "trace-capacity", "not the formal N=20 capture"
        )
        extra_configuration = json.loads(json.dumps(valid_metadata))
        extra_configuration["configuration"]["UNDECLARED_FORMAL_FLAG"] = 1
        expect_metadata_rejected(
            extra_configuration, "extra-configuration", "not the formal N=20 capture"
        )
        for field in ("response_text_persisted", "raw_server_log_retained"):
            bad_privacy = json.loads(json.dumps(valid_metadata))
            bad_privacy["privacy_contract"][field] = True
            expect_metadata_rejected(
                bad_privacy, field, "privacy contract does not match"
            )

        _expect_invalid(fixture, 19, "counterfactual smelt-n")

    print("routing trace replay self-test: PASS", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate schema-v1 routing traces and optionally compare deterministic "
            "offline cache policies."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--trace", type=Path, help="schema-v1 routing trace JSONL")
    source.add_argument(
        "--corpus-metadata",
        type=Path,
        help="Task 6.1 routing corpus metadata JSON",
    )
    parser.add_argument("--smelt-n", type=int, default=20, help="observed fixed-hot capacity")
    parser.add_argument(
        "--compare-policies",
        action="store_true",
        help="compare fixed-hot, LRU, LFU, Trust-OS and offline bounds",
    )
    parser.add_argument(
        "--full-resident-layers",
        type=int,
        default=0,
        help="leading layers whose cache capacity equals n_experts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the JSON result with mode 0600",
    )
    parser.add_argument("--format", choices=("json",), default="json")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    arguments = _build_parser().parse_args()
    try:
        if arguments.self_test:
            run_self_test()
            return 0
        if arguments.full_resident_layers < 0:
            raise TraceValidationError("--full-resident-layers must be non-negative")
        if arguments.corpus_metadata is not None:
            result = replay_corpus(
                arguments.corpus_metadata,
                arguments.smelt_n,
                arguments.full_resident_layers,
            )
        elif arguments.trace is not None:
            if arguments.compare_policies:
                result = replay_trace_with_policies(
                    arguments.trace,
                    arguments.smelt_n,
                    arguments.full_resident_layers,
                )
            else:
                if arguments.full_resident_layers != 0:
                    raise TraceValidationError(
                        "--full-resident-layers requires --compare-policies or "
                        "--corpus-metadata"
                    )
                result = replay_trace(arguments.trace, arguments.smelt_n)
        else:
            raise TraceValidationError(
                "--trace or --corpus-metadata is required unless --self-test is used"
            )
        if arguments.output is not None:
            _write_json_private(arguments.output, result)
        json.dump(result, sys.stdout, sort_keys=True, separators=(",", ":"))
        sys.stdout.write("\n")
        return 0
    except (TraceValidationError, AssertionError) as error:
        print(f"routing trace replay error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
