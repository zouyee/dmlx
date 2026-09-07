#!/usr/bin/env python3
"""Validate schema-v1 native expert-I/O telemetry and report invariants."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
TELEMETRY_KIND = "dmlx-native-request-telemetry"
PHASE_NAMES = ("prefill", "decode", "verify")
LAYER_COUNT = 43
ROUTES_PER_CALL = 6

LAYER_TOTAL_KEYS = (
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
)


class InvariantError(ValueError):
    """Raised when native telemetry violates an expert-I/O invariant."""


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InvariantError(f"{label} must be a nonnegative integer")
    return value


def _unique_cold_experts(layer: Mapping[str, Any], label: str) -> int:
    cold_misses = _nonnegative_int(layer.get("cold_misses"), f"{label}.cold_misses")
    if "unique_cold_experts" not in layer:
        return cold_misses
    unique_cold = _nonnegative_int(
        layer.get("unique_cold_experts"), f"{label}.unique_cold_experts"
    )
    if unique_cold > cold_misses:
        raise InvariantError(
            f"{label}.unique_cold_experts must not exceed cold_misses"
        )
    return unique_cold


def _validate_layer(layer: Mapping[str, Any], label: str) -> dict[str, int]:
    layer_index = _nonnegative_int(layer.get("layer"), f"{label}.layer")
    if layer_index >= LAYER_COUNT:
        raise InvariantError(f"{label}.layer must be below {LAYER_COUNT}")

    calls = _nonnegative_int(layer.get("calls"), f"{label}.calls")
    if calls == 0:
        raise InvariantError(f"{label}.calls must be positive for an emitted layer")
    unique_experts = _nonnegative_int(
        layer.get("unique_experts"), f"{label}.unique_experts"
    )
    if unique_experts > min(256, calls * ROUTES_PER_CALL):
        raise InvariantError(f"{label}.unique_experts exceeds the routed-expert bound")

    values = {
        key: _nonnegative_int(layer.get(key), f"{label}.{key}")
        for key in LAYER_TOTAL_KEYS
    }
    if (
        values["hot_hits"]
        + values["prefetch_hits"]
        + values["batch_pool_hits"]
        + values["cold_misses"]
        != calls * ROUTES_PER_CALL
    ):
        raise InvariantError(f"{label} cache classes must partition calls * 6 routes")

    unique_cold = _unique_cold_experts(layer, label)
    if values["pread_calls"] > unique_cold:
        raise InvariantError(
            f"{label}.pread_calls exceeds unique cold experts ({unique_cold})"
        )
    if values["short_reads"] != 0:
        raise InvariantError(f"{label}.short_reads must be zero")
    if values["bytes_read"] != values["bytes_requested"]:
        raise InvariantError(f"{label}.bytes_read must equal bytes_requested")
    if values["pread_calls"] == 0 and values["bytes_read"] != 0:
        raise InvariantError(f"{label} has physical bytes without a pread")

    return {
        "layer": layer_index,
        "unique_experts": unique_experts,
        "unique_cold_experts": unique_cold,
        "hot": values["hot_hits"],
        "prefetch": values["prefetch_hits"],
        "batch": values["batch_pool_hits"],
        "cold": values["cold_misses"],
        "pread_calls": values["pread_calls"],
        "physical_bytes": values["bytes_read"],
        "avoided_bytes": values["bytes_avoided"],
    }


def _validate_request(record: Mapping[str, Any], label: str) -> list[dict[str, int]]:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise InvariantError(f"{label} has an unsupported schema_version")
    if record.get("kind") != TELEMETRY_KIND:
        raise InvariantError(f"{label}.kind is invalid")
    if record.get("scope") != "native_engine_after_tokenization":
        raise InvariantError(f"{label}.scope is invalid")

    request = record.get("request")
    if not isinstance(request, Mapping):
        raise InvariantError(f"{label}.request must be an object")
    _nonnegative_int(request.get("prompt_tokens"), f"{label}.request.prompt_tokens")
    _nonnegative_int(
        request.get("generated_tokens"), f"{label}.request.generated_tokens"
    )

    phases = record.get("phases")
    if not isinstance(phases, list) or len(phases) != len(PHASE_NAMES):
        raise InvariantError(f"{label}.phases must contain prefill, decode, and verify")

    layer_rows: list[dict[str, int]] = []
    layer_sums = {key: 0 for key in LAYER_TOTAL_KEYS}
    unique_cold_sum = 0
    for phase_index, (phase, expected_name) in enumerate(zip(phases, PHASE_NAMES)):
        phase_label = f"{label}.phases[{phase_index}]"
        if not isinstance(phase, Mapping) or phase.get("name") != expected_name:
            raise InvariantError(f"{phase_label} has an invalid phase name/order")
        layers = phase.get("layers")
        if not isinstance(layers, list):
            raise InvariantError(f"{phase_label}.layers must be a list")
        seen_layers: set[int] = set()
        for layer_index, layer in enumerate(layers):
            item_label = f"{phase_label}.layers[{layer_index}]"
            if not isinstance(layer, Mapping):
                raise InvariantError(f"{item_label} must be an object")
            row = _validate_layer(layer, item_label)
            if row["layer"] in seen_layers:
                raise InvariantError(
                    f"{phase_label} repeats layer {row['layer']}"
                )
            seen_layers.add(row["layer"])
            layer_rows.append(row)
            unique_cold_sum += row["unique_cold_experts"]
            for key in LAYER_TOTAL_KEYS:
                layer_sums[key] += int(layer[key])

    expert_io = record.get("expert_io")
    if not isinstance(expert_io, Mapping):
        raise InvariantError(f"{label}.expert_io must be an object")
    if expert_io.get("measurement_scope") != "critical_path_routed_expert_resolution":
        raise InvariantError(f"{label}.expert_io.measurement_scope is invalid")
    for key, expected in layer_sums.items():
        actual = _nonnegative_int(expert_io.get(key), f"{label}.expert_io.{key}")
        if actual != expected:
            raise InvariantError(
                f"{label}.expert_io.{key} does not match phase layers"
            )
    if "unique_cold_experts" in expert_io:
        aggregate_unique_cold = _nonnegative_int(
            expert_io.get("unique_cold_experts"),
            f"{label}.expert_io.unique_cold_experts",
        )
        if aggregate_unique_cold > int(expert_io["cold_misses"]):
            raise InvariantError(
                f"{label}.expert_io.unique_cold_experts must not exceed cold_misses"
            )
        if aggregate_unique_cold != unique_cold_sum:
            raise InvariantError(
                f"{label}.expert_io.unique_cold_experts does not match phase layers"
            )
    if int(expert_io["short_reads"]) != 0:
        raise InvariantError(f"{label}.expert_io.short_reads must be zero")
    if int(expert_io["bytes_read"]) != int(expert_io["bytes_requested"]):
        raise InvariantError(
            f"{label}.expert_io.bytes_read must equal bytes_requested"
        )
    return layer_rows


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    by_layer: dict[int, dict[str, int]] = {}
    summary_keys = (
        "unique_experts",
        "unique_cold_experts",
        "hot",
        "prefetch",
        "batch",
        "cold",
        "pread_calls",
        "physical_bytes",
        "avoided_bytes",
    )
    for request_index, record in enumerate(records):
        rows = _validate_request(record, f"telemetry[{request_index}]")
        for row in rows:
            layer_index = row["layer"]
            aggregate = by_layer.setdefault(
                layer_index,
                {"layer": layer_index, **{key: 0 for key in summary_keys}},
            )
            for key in summary_keys:
                aggregate[key] += row[key]

    layers = [by_layer[index] for index in sorted(by_layer)]
    totals = {key: sum(layer[key] for layer in layers) for key in summary_keys}
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "dmlx-expert-io-invariant-summary",
        "request_count": len(records),
        "layers": layers,
        "totals": totals,
    }


def summarize_telemetry(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Validate request telemetry and return a deterministic per-layer summary."""
    return _summary(records)


def _artifact_records(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if document.get("kind") == TELEMETRY_KIND:
        return [document]

    phases = document.get("phases")
    if document.get("schema_version") != SCHEMA_VERSION or not isinstance(phases, Mapping):
        raise InvariantError("JSON object is neither schema-v1 telemetry nor a benchmark artifact")
    standard = phases.get("standard_decode")
    if not isinstance(standard, Mapping) or not isinstance(standard.get("runs"), list):
        raise InvariantError("benchmark artifact is missing standard_decode.runs")
    telemetry_values: list[Any] = []
    for index, run in enumerate(standard["runs"]):
        if not isinstance(run, Mapping):
            raise InvariantError(f"benchmark standard run {index + 1} must be an object")
        telemetry_values.append(run.get("telemetry"))
    present = [value is not None for value in telemetry_values]
    if any(present) and not all(present):
        raise InvariantError("benchmark telemetry must be present for every run or none")
    if not any(present):
        metrics = document.get("metrics")
        if not isinstance(metrics, Mapping):
            raise InvariantError("disabled benchmark telemetry requires metrics")
        if (
            "expert_io_invariants" not in metrics
            or metrics["expert_io_invariants"] is not None
        ):
            raise InvariantError(
                "disabled benchmark telemetry requires null expert_io_invariants"
            )
        expert_io = metrics.get("expert_io")
        dispatch = metrics.get("dispatch")
        if not isinstance(expert_io, Mapping) or not isinstance(dispatch, Mapping):
            raise InvariantError(
                "disabled benchmark telemetry requires expert_io and dispatch objects"
            )
        for label, aggregate, required_keys in (
            ("expert_io", expert_io, AGGREGATE_EXPERT_IO_KEYS),
            ("dispatch", dispatch, AGGREGATE_DISPATCH_KEYS),
        ):
            for key in required_keys:
                if key not in aggregate:
                    raise InvariantError(
                        f"disabled benchmark telemetry requires metrics.{label}.{key}"
                    )
                if aggregate[key] is not None:
                    raise InvariantError(
                        f"disabled benchmark telemetry requires null metrics.{label}.{key}"
                    )
        return []
    records: list[Mapping[str, Any]] = []
    for index, value in enumerate(telemetry_values):
        if not isinstance(value, Mapping):
            raise InvariantError(f"benchmark run {index + 1} telemetry must be an object")
        records.append(value)
    _summary(records)
    _validate_benchmark_expert_io(document, records)
    return records


def _validate_benchmark_expert_io(
    document: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    metrics = document.get("metrics")
    expert_io = metrics.get("expert_io") if isinstance(metrics, Mapping) else None
    if not isinstance(expert_io, Mapping):
        raise InvariantError("benchmark metrics.expert_io must be an object")
    for key in (
        "pread_calls",
        "bytes_requested",
        "bytes_read",
        "hot_hits",
        "prefetch_hits",
        "batch_pool_hits",
        "cold_misses",
        "bytes_avoided",
        "short_reads",
    ):
        expected = sum(
            _nonnegative_int(
                record["expert_io"].get(key), f"telemetry expert_io.{key}"
            )
            for record in records
            if isinstance(record.get("expert_io"), Mapping)
        )
        actual = _nonnegative_int(expert_io.get(key), f"metrics.expert_io.{key}")
        if actual != expected:
            raise InvariantError(
                f"metrics.expert_io.{key} does not match benchmark request telemetry"
            )
    expected_wait_ms = sum(int(record["expert_io"]["io_wait_ns"]) for record in records) / 1_000_000.0
    actual_wait_ms = expert_io.get("io_wait_ms")
    if (
        not isinstance(actual_wait_ms, (int, float))
        or isinstance(actual_wait_ms, bool)
        or abs(float(actual_wait_ms) - expected_wait_ms) > 1e-9
    ):
        raise InvariantError(
            "metrics.expert_io.io_wait_ms does not match benchmark request telemetry"
        )


def _records_from_value(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return _artifact_records(value)
    if isinstance(value, list):
        if not value:
            raise InvariantError("telemetry array contains no records")
        records: list[Mapping[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping) or item.get("kind") != TELEMETRY_KIND:
                raise InvariantError(f"telemetry[{index}] must be a native telemetry object")
            records.append(item)
        return records
    raise InvariantError("telemetry input must be a JSON object or array")


def load_records(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        return _records_from_value(json.loads(text))
    except json.JSONDecodeError:
        records: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            prefix = "[NATIVE_TELEMETRY] "
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise InvariantError(
                    f"invalid JSONL telemetry on line {line_number}: {exc.msg}"
                ) from exc
            records.extend(_records_from_value(value))
        if not records:
            raise InvariantError("telemetry input contains no records")
        return records


def _sample_record() -> dict[str, Any]:
    layer = {
        "layer": 0,
        "calls": 2,
        "unique_experts": 8,
        "hot_hits": 4,
        "prefetch_hits": 2,
        "batch_pool_hits": 2,
        "cold_misses": 4,
        "pread_calls": 4,
        "bytes_requested": 400,
        "bytes_read": 400,
        "short_reads": 0,
        "io_wait_ns": 1000,
        "bytes_avoided": 800,
    }
    expert_io = {
        "measurement_scope": "critical_path_routed_expert_resolution",
        **{key: layer[key] for key in LAYER_TOTAL_KEYS},
    }
    return {
        "schema_version": 1,
        "kind": TELEMETRY_KIND,
        "scope": "native_engine_after_tokenization",
        "request": {"prompt_tokens": 8, "generated_tokens": 5},
        "expert_io": expert_io,
        "phases": [
            {"name": "prefill", "layers": [layer]},
            {"name": "decode", "layers": []},
            {"name": "verify", "layers": []},
        ],
    }


def _expect_invalid(record: Mapping[str, Any], description: str) -> None:
    try:
        summarize_telemetry([record])
    except InvariantError:
        return
    raise AssertionError(f"accepted synthetic invalid telemetry: {description}")


def _expect_invalid_artifact(
    document: Mapping[str, Any], description: str
) -> None:
    try:
        _artifact_records(document)
    except InvariantError:
        return
    raise AssertionError(f"accepted synthetic invalid artifact: {description}")


def self_test() -> None:
    valid = _sample_record()
    summary = summarize_telemetry([valid])
    if summary is None or summary["totals"]["physical_bytes"] != 400:
        raise AssertionError("valid telemetry did not produce the expected summary")

    disabled_artifact = {
        "schema_version": SCHEMA_VERSION,
        "phases": {
            "standard_decode": {
                "runs": [{"telemetry": None} for _ in range(3)],
            }
        },
        "metrics": {
            "expert_io": {key: None for key in AGGREGATE_EXPERT_IO_KEYS},
            "expert_io_invariants": None,
            "dispatch": {key: None for key in AGGREGATE_DISPATCH_KEYS},
        },
    }
    if _artifact_records(disabled_artifact) != []:
        raise AssertionError("valid disabled artifact produced telemetry records")

    missing_disabled_key = copy.deepcopy(disabled_artifact)
    del missing_disabled_key["metrics"]["expert_io"]["pread_calls"]
    _expect_invalid_artifact(missing_disabled_key, "missing disabled expert I/O field")

    nonnull_disabled_aggregate = copy.deepcopy(disabled_artifact)
    nonnull_disabled_aggregate["metrics"]["dispatch"]["dispatches"] = 1
    _expect_invalid_artifact(
        nonnull_disabled_aggregate, "non-null disabled dispatch field"
    )

    excess_preads = copy.deepcopy(valid)
    excess_preads["phases"][0]["layers"][0]["pread_calls"] = 5
    excess_preads["expert_io"]["pread_calls"] = 5
    _expect_invalid(excess_preads, "excess preads")

    byte_mismatch = copy.deepcopy(valid)
    byte_mismatch["phases"][0]["layers"][0]["bytes_read"] = 399
    byte_mismatch["expert_io"]["bytes_read"] = 399
    _expect_invalid(byte_mismatch, "byte mismatch")

    short_read = copy.deepcopy(valid)
    short_read["phases"][0]["layers"][0]["short_reads"] = 1
    short_read["expert_io"]["short_reads"] = 1
    _expect_invalid(short_read, "nonzero short read")

    class_mismatch = copy.deepcopy(valid)
    class_mismatch["phases"][0]["layers"][0]["hot_hits"] = 3
    class_mismatch["expert_io"]["hot_hits"] = 3
    _expect_invalid(class_mismatch, "class mismatch")

    aggregate_mismatch = copy.deepcopy(valid)
    aggregate_mismatch["expert_io"]["bytes_avoided"] += 1
    _expect_invalid(aggregate_mismatch, "aggregate mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate schema-v1 native expert-I/O telemetry invariants"
    )
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("check_expert_io_invariants self-test: PASS")
        return 0
    if args.input is None:
        parser.error("input is required unless --self-test is used")

    try:
        records = load_records(args.input)
        summary = summarize_telemetry(records)
    except (InvariantError, OSError) as exc:
        print(f"expert I/O invariant check FAILED: {exc}", file=sys.stderr)
        return 1
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
