#!/usr/bin/env python3
"""Task 2.2 candidate selection: frozen source-bound replay over a completed
Task 2.1 evidence artifact.

Reads a complete collector artifact, re-validates it with the collector
module, replays the three diagnosis timelines with the collector's own
parsing/validation helpers, and applies the frozen candidate-budget rules
from the protocol file bound in the artifact:

  per retained token: eligible-stage intervals clipped to TOKEN_FORWARD,
  unioned once per family; recoverable ratio = union_ns * cap / token_forward_ns
  per process:        median recoverable ratio over retained tokens
  cross process:      minimum of the three process medians
  dependency gate:    a family whose protocol entry requires a named removable
                      dependency receives budget 0 unless the bound dependency
                      analysis marks it feasible
  selection:          families with cross-process minimum >= the frozen
                      threshold; the highest minimum wins, ties break by the
                      frozen family order; zero qualifiers => NOT_TRIGGERED

Everything is fail-closed: any schema, hash, or rule violation raises
SystemExit with a non-zero status.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any, NoReturn

COLLECTOR_PATH = Path(__file__).resolve().with_name("run_decode_critical_path.py")
SELECTION_FIELDS = frozenset((
    "schema_version", "kind", "source_artifact", "dependency_analysis",
    "protocol", "threshold", "families", "retained_token_counts",
    "union_numerator_ns", "process_medians", "cross_process_minimum",
    "dependency_feasibility", "threshold_result", "selection",
    "semantic_sha256",
))


def fail(message: str) -> NoReturn:
    raise SystemExit(f"selection error: {message}")


def load_collector() -> Any:
    spec = importlib.util.spec_from_file_location("dsv4_collector", COLLECTOR_PATH)
    if spec is None or spec.loader is None:
        fail("cannot load collector module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"cannot load collector module: {exc}")
    return module


def replay_tokens(collector: Any, timeline_path: Path) -> list[dict[str, Any]]:
    """Parse and fully validate a diagnosis timeline, returning per-token
    grouped events in capture order."""
    tokens: list[dict[str, Any]] = []
    try:
        with timeline_path.open("r", encoding="utf-8") as stream:
            segments: list[list[dict[str, Any]]] = []
            segment_plain: list[bool] = []
            for line_no, raw in enumerate(stream, 1):
                if not raw.strip():
                    fail(f"{timeline_path}:{line_no}: blank lines forbidden")
                value = json.loads(raw)
                if type(value) is dict and set(value) == collector.CAPTURE_FIELDS:
                    segments.append(collector.normalize_capture(value, line_no))
                    segment_plain.append(False)
                else:
                    event = collector.validate_event(value, line_no)
                    if segments and segment_plain[-1]:
                        segments[-1].append(event)
                    else:
                        segments.append([event])
                        segment_plain.append(True)
    except SystemExit:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"{timeline_path}: cannot replay: {exc}")
    for events in segments:
        if not events:
            continue
        grouped: dict[int, list[dict[str, Any]]] = {}
        order: list[int] = []
        for event in events:
            if event["token"] not in grouped:
                grouped[event["token"]] = []
                order.append(event["token"])
            grouped[event["token"]].append(event)
        for token in order:
            collector.validate_token(token, grouped[token])  # fail-closed accounting
            tokens.append({"token": token, "events": grouped[token]})
    if not tokens:
        fail(f"{timeline_path}: no retained tokens")
    return tokens


def family_token_ratios(events: list[dict[str, Any]], stages: tuple[str, ...]) -> tuple[int, int, float]:
    envelope = next(event for event in events if event["stage"] == "token_forward")
    begin, end = envelope["begin_ns"], envelope["end_ns"]
    duration = end - begin
    if duration <= 0:
        fail("token_forward duration must be positive")
    intervals = [(event["begin_ns"], event["end_ns"])
                 for event in events if event["stage"] in stages]
    union_ns = sum_union(intervals) if intervals else 0
    return union_ns, duration, union_ns / duration


def sum_union(intervals: list[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    total, start, end = 0, ordered[0][0], ordered[0][1]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def select_candidate(
    per_process_tokens: dict[str, list[dict[str, Any]]],
    families: dict[str, dict[str, Any]],
    family_order: list[str],
    threshold: float,
    dependency: dict[str, Any],
) -> dict[str, Any]:
    retained_token_counts = {label: len(tokens) for label, tokens in per_process_tokens.items()}
    union_numerator: dict[str, dict[str, int]] = {}
    medians: dict[str, dict[str, float]] = {}
    minima: dict[str, float] = {}
    feasibility: dict[str, dict[str, Any]] = {}
    for name in family_order:
        spec = families[name]
        stages = tuple(spec["eligible_stages"])
        cap = spec["recoverable_cap"]
        requires_dependency = bool(spec.get("requires_named_removable_dependency"))
        entry = dependency.get(name)
        feasible = bool(entry and entry.get("feasible")) if requires_dependency else True
        if requires_dependency and feasible:
            stage_deps = entry.get("stage_dependencies", {})
            for stage in stages:
                dep = stage_deps.get(stage)
                if not dep or not dep.get("removable") or not dep.get("first_later_consumer") \
                        or not dep.get("ownership_transfer"):
                    feasible = False
                    break
        feasibility[name] = {"requires_named_removable_dependency": requires_dependency,
                             "feasible": feasible,
                             "budget_zero": requires_dependency and not feasible}
        union_numerator[name] = {}
        medians[name] = {}
        for label, tokens in per_process_tokens.items():
            total_union = 0
            ratios = []
            for token in tokens:
                union_ns, duration, _ = family_token_ratios(token["events"], stages)
                total_union += union_ns
                ratios.append((union_ns * cap) / duration)
            union_numerator[name][label] = total_union
            medians[name][label] = statistics.median(ratios)
        minima[name] = 0.0 if (requires_dependency and not feasible) else min(medians[name].values())
    qualifiers = [name for name in family_order if minima[name] >= threshold]
    if qualifiers:
        selected = max(qualifiers, key=lambda name: (minima[name], -family_order.index(name)))
        selection = {"decision": "selected", "family": selected,
                     "cross_process_minimum": minima[selected]}
    else:
        selected = None
        selection = {"decision": "NOT_TRIGGERED", "family": None, "cross_process_minimum": 0.0}
    threshold_result = {name: {"cross_process_minimum": minima[name],
                               "qualifies": name in qualifiers,
                               "selected": name == selected}
                        for name in family_order}
    return {"retained_token_counts": retained_token_counts,
            "union_numerator_ns": union_numerator,
            "process_medians": medians,
            "cross_process_minimum": minima,
            "dependency_feasibility": feasibility,
            "threshold_result": threshold_result,
            "selection": selection}


def load_protocol_budget(artifact: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    sources = artifact["sources"]
    protocol_ref = sources.get("protocol")
    if type(protocol_ref) is not dict:
        fail("artifact.sources.protocol: missing")
    protocol_path = Path(protocol_ref["path"])
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_ref["sha256"]:
        fail("protocol file hash differs from the artifact binding")
    protocol = json.loads(protocol_path.read_bytes())
    budget = protocol.get("candidate_budget")
    if type(budget) is not dict or budget.get("token_denominator") != "token_forward_ns":
        fail("protocol candidate_budget missing or wrong denominator")
    if budget.get("per_process_aggregate") != "median_retained_token_ratio" or \
            budget.get("cross_process_aggregate") != "minimum_of_three_process_medians":
        fail("protocol aggregation rules differ from this replay")
    threshold = protocol.get("candidate_predicted_wall_reduction_min")
    if type(threshold) not in (int, float) or threshold <= 0:
        fail("protocol threshold missing")
    order = protocol.get("candidate_families")
    families = budget.get("families")
    if type(order) is not list or type(families) is not dict or any(name not in families for name in order):
        fail("protocol family order/families mismatch")
    return {"families": families, "order": order, "threshold": float(threshold),
            "path": str(protocol_path), "sha256": protocol_ref["sha256"]}


def run(artifact_path: Path, dependency_path: Path, output_path: Path) -> dict[str, Any]:
    collector = load_collector()
    summary = collector.validate_complete_artifact(artifact_path)
    if not summary.get("valid") or summary.get("terminal") != "complete":
        fail("source artifact is not a complete validated Task 2.1 artifact")
    artifact = json.loads(artifact_path.read_bytes())
    protocol = load_protocol_budget(artifact, artifact_path)
    dependency = json.loads(dependency_path.read_bytes())
    if type(dependency) is not dict or type(dependency.get("families")) is not dict:
        fail("dependency analysis: expected object with families")
    per_process_tokens: dict[str, list[dict[str, Any]]] = {}
    timeline_hashes: dict[str, dict[str, str]] = {}
    for diagnosis in artifact["diagnostics"]:
        label = diagnosis["label"]
        timeline = diagnosis["timeline"]
        if timeline.get("state") != "present":
            fail(f"{label}: diagnosis without timeline")
        ref = timeline["source"]
        timeline_path = Path(ref["path"])
        if hashlib.sha256(timeline_path.read_bytes()).hexdigest() != ref["sha256"]:
            fail(f"{label}: timeline hash differs from the artifact binding")
        timeline_hashes[label] = dict(ref)
        per_process_tokens[label] = replay_tokens(collector, timeline_path)
    result = select_candidate(per_process_tokens, protocol["families"], protocol["order"],
                              protocol["threshold"], dependency["families"])
    selection_artifact: dict[str, Any] = {
        "schema_version": 1,
        "kind": "dsv4_decode_critical_path_selection",
        "source_artifact": {"path": str(artifact_path),
                            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                            "semantic_sha256": artifact["semantic_sha256"],
                            "diagnosis_timelines": timeline_hashes},
        "dependency_analysis": {"path": str(dependency_path),
                                "sha256": hashlib.sha256(dependency_path.read_bytes()).hexdigest()},
        "protocol": {"path": protocol["path"], "sha256": protocol["sha256"]},
        "threshold": protocol["threshold"],
        "families": {name: {"eligible_stages": protocol["families"][name]["eligible_stages"],
                            "recoverable_cap": protocol["families"][name]["recoverable_cap"]}
                     for name in protocol["order"]},
        "retained_token_counts": result["retained_token_counts"],
        "union_numerator_ns": result["union_numerator_ns"],
        "process_medians": result["process_medians"],
        "cross_process_minimum": result["cross_process_minimum"],
        "dependency_feasibility": result["dependency_feasibility"],
        "threshold_result": result["threshold_result"],
        "selection": result["selection"],
    }
    selection_artifact["semantic_sha256"] = collector.semantic_hash(selection_artifact)
    if output_path.exists():
        fail(f"selection output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(collector.canonical_bytes(selection_artifact) + b"\n")
    return selection_artifact


def synthesize_token(stage_spans: dict[str, tuple[int, int]], duration: int) -> dict[str, Any]:
    events = []
    begin = 1_000_000
    events.append({"stage": "token_forward", "layer": None,
                   "begin_ns": begin, "end_ns": begin + duration})
    for stage, (offset, length) in stage_spans.items():
        events.append({"stage": stage, "layer": 0,
                       "begin_ns": begin + offset, "end_ns": begin + offset + length})
    return {"token": 0, "events": events}


def self_test() -> dict[str, Any]:
    families = {
        "routed_moe_kernel_or_cb": {"eligible_stages": ["routed_moe_cb"], "recoverable_cap": 0.20},
        "cpu_gpu_readback_or_sync": {"eligible_stages": ["attention_cb", "cb2cmd2_cb"],
                                     "recoverable_cap": 0.50,
                                     "requires_named_removable_dependency": True},
        "lm_head_top1": {"eligible_stages": ["logits_cb"], "recoverable_cap": 0.50},
    }
    order = ["routed_moe_kernel_or_cb", "cpu_gpu_readback_or_sync", "lm_head_top1"]
    checks: dict[str, Any] = {}

    def tokens(routed: int, attn: int, logits: int) -> dict[str, Any]:
        spans = {"routed_moe_cb": (0, routed), "attention_cb": (0, attn), "logits_cb": (0, logits)}
        return synthesize_token(spans, 1000)

    processes = {label: [tokens(400, 800, 10)] for label in ("d1", "d2", "d3")}
    full_dep = {"cpu_gpu_readback_or_sync": {"feasible": True, "stage_dependencies": {
        stage: {"removable": True, "first_later_consumer": "c", "ownership_transfer": "t"}
        for stage in ("attention_cb", "cb2cmd2_cb")}}}
    # ratios: routed 400*0.2/1000=0.08, sync 800*0.5/1000=0.40, lm 10*0.5/1000=0.005
    result = select_candidate(processes, families, order, 0.05, full_dep)
    checks["selects_highest_minimum"] = result["selection"]["family"] == "cpu_gpu_readback_or_sync"
    checks["minima"] = (abs(result["cross_process_minimum"]["cpu_gpu_readback_or_sync"] - 0.40) < 1e-12
                        and abs(result["cross_process_minimum"]["routed_moe_kernel_or_cb"] - 0.08) < 1e-12)

    # Dependency infeasible -> sync family budget zero -> routed family selected.
    infeasible = {"cpu_gpu_readback_or_sync": {"feasible": False}}
    result2 = select_candidate(processes, families, order, 0.05, infeasible)
    checks["infeasible_zeroes_budget"] = (
        result2["cross_process_minimum"]["cpu_gpu_readback_or_sync"] == 0.0
        and result2["selection"]["family"] == "routed_moe_kernel_or_cb")

    # Missing stage dependency entry also zeroes the budget.
    partial_dep = {"cpu_gpu_readback_or_sync": {"feasible": True, "stage_dependencies": {
        "attention_cb": {"removable": True, "first_later_consumer": "c", "ownership_transfer": "t"}}}}
    result3 = select_candidate(processes, families, order, 0.05, partial_dep)
    checks["missing_stage_dependency_zeroes_budget"] = \
        result3["selection"]["family"] == "routed_moe_kernel_or_cb"

    # Everything below threshold -> NOT_TRIGGERED.
    quiet = {label: [tokens(100, 60, 10)] for label in ("d1", "d2", "d3")}
    result4 = select_candidate(quiet, families, order, 0.05, full_dep)
    checks["not_triggered"] = result4["selection"]["decision"] == "NOT_TRIGGERED"

    # Tie on minima -> earlier family order wins.
    tie = {label: [tokens(1000, 400, 10)] for label in ("d1", "d2", "d3")}
    result5 = select_candidate(tie, families, order, 0.05, full_dep)
    checks["tie_break_order"] = result5["selection"]["family"] == "routed_moe_kernel_or_cb"

    checks["passed"] = all(v is True for k, v in checks.items() if k != "passed")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", help="completed Task 2.1 artifact.json")
    parser.add_argument("--dependency-analysis", help="dependency analysis JSON")
    parser.add_argument("--output", help="selection artifact output path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        checks = self_test()
        print(json.dumps({"self_test": "passed" if checks["passed"] else "failed",
                          "checks": checks}, sort_keys=True))
        raise SystemExit(0 if checks["passed"] else 1)
    if not (args.artifact and args.dependency_analysis and args.output):
        fail("--artifact, --dependency-analysis, and --output are required")
    result = run(Path(args.artifact), Path(args.dependency_analysis), Path(args.output))
    print(json.dumps({"selection": result["selection"],
                      "cross_process_minimum": result["cross_process_minimum"],
                      "semantic_sha256": result["semantic_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
