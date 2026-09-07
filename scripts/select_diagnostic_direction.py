#!/usr/bin/env python3
"""Select one non-authoritative investigation direction from benchmark telemetry."""
import argparse
import hashlib
import json
from pathlib import Path

KINDS = ("attention", "routed_moe")
REASONS = ("selector_unavailable_count", "nonfinite_count", "zero_count", "reversed_count", "host_order_count")

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def exact_nonnegative_int(value) -> bool:
    return type(value) is int and value >= 0

def select(doc: dict, source_sha256: str) -> dict:
    reasons = []
    if not isinstance(doc, dict):
        reasons.append("top_level_not_object")
        doc = {}
    phases = doc.get("phases")
    if not isinstance(phases, dict):
        reasons.append("phases_not_object")
        phases = {}
    correctness = phases.get("correctness")
    if not isinstance(correctness, dict):
        reasons.append("correctness_not_object")
        correctness = {}
    standard = phases.get("standard_decode")
    if not isinstance(standard, dict):
        reasons.append("standard_decode_not_object")
        standard = {}
    runs = standard.get("runs")
    if not isinstance(runs, list):
        reasons.append("standard_runs_not_array")
        runs = []
    if correctness.get("e2e_pass") != 7 or correctness.get("e2e_fail") != 0 or not correctness.get("paris_pass"):
        reasons.append("correctness_not_7_of_7")
    if len(runs) != 3:
        reasons.append("standard_run_count_not_three")
    samples = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            reasons.append(f"run_{index+1}_not_object")
            continue
        telemetry = run.get("telemetry")
        if not isinstance(telemetry, dict):
            reasons.append(f"run_{index+1}_telemetry_not_object")
            continue
        metric = telemetry.get("sampled_gpu_cb_duration")
        if not isinstance(metric, dict) or metric.get("status") != "available":
            reasons.append(f"run_{index+1}_sampled_duration_unavailable")
            continue
        if metric.get("measurement_scope") != "sampled_command_buffer_duration_mass":
            reasons.append(f"run_{index+1}_measurement_scope_mismatch")
            continue
        retained = 0
        valid = True
        sample = {}
        for kind in KINDS:
            child = metric.get(kind)
            if not isinstance(child, dict):
                valid = False
                continue
            duration = child.get("duration_ns_sum")
            count = child.get("valid_count")
            reason_values = [child.get(name) for name in REASONS]
            if (child.get("duration_sum_overflow") is not False or
                    not exact_nonnegative_int(duration) or
                    not exact_nonnegative_int(count) or count == 0 or
                    not all(exact_nonnegative_int(value) for value in reason_values)):
                valid = False
                continue
            invalid = sum(reason_values)
            if invalid != 0:
                valid = False
            retained += count + invalid
            sample[kind] = duration
        total_seen = metric.get("total_seen")
        retained_count = metric.get("retained_count")
        overflow_count = metric.get("overflow_count")
        if (not exact_nonnegative_int(total_seen) or
                not exact_nonnegative_int(retained_count) or
                not exact_nonnegative_int(overflow_count) or
                overflow_count != 0 or retained_count != retained or
                total_seen != retained + overflow_count):
            valid = False
        if not valid or len(sample) != len(KINDS):
            reasons.append(f"run_{index+1}_invalid_sample_accounting")
            continue
        samples.append(sample)
    decision = "NOT_TRIGGERED"
    target = None
    medians = {}
    if not reasons and len(samples) == 3:
        for kind in KINDS:
            values = sorted(sample[kind] for sample in samples)
            medians[kind] = values[1]
        if medians["attention"] > medians["routed_moe"]:
            decision, target = "ATTENTION_INVESTIGATION", "production scalar merged attention command buffer"
        elif medians["routed_moe"] > medians["attention"]:
            decision, target = "ROUTED_MOE_INVESTIGATION", "production scalar routed-MoE command buffer"
        else:
            reasons.append("sampled_duration_tie")
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict):
        reasons.append("metrics_not_object")
        metrics = {}
    expert_io = metrics.get("expert_io")
    if not isinstance(expert_io, dict):
        reasons.append("expert_io_not_object")
        expert_io = {}
    if reasons and decision != "NOT_TRIGGERED":
        decision, target = "NOT_TRIGGERED", None
    return {
        "schema_version": 1,
        "decision": decision,
        "target": target,
        "source_sha256": source_sha256,
        "standard_run_sampled_duration_ns": samples,
        "median_sampled_duration_ns": medians,
        "expert_io_scope": expert_io,
        "rejection_reasons": reasons,
        "claims": {
            "authoritative_candidate": False,
            "wall_time_percentage": False,
            "exclusive_gpu_time": False,
            "scheduling_slack_measured": False,
            "unblocks_task_2_2": False,
        },
    }

def self_test() -> None:
    child = {
        "duration_sum_overflow": False, "valid_count": 1,
        "selector_unavailable_count": 0, "nonfinite_count": 0,
        "zero_count": 0, "reversed_count": 0, "host_order_count": 0,
    }
    runs = []
    for attention, routed_moe in ((10, 5), (12, 6), (11, 7)):
        metric = {
            "measurement_scope": "sampled_command_buffer_duration_mass",
            "status": "available", "total_seen": 2, "retained_count": 2,
            "overflow_count": 0,
            "attention": dict(child, duration_ns_sum=attention),
            "routed_moe": dict(child, duration_ns_sum=routed_moe),
        }
        runs.append({"telemetry": {"sampled_gpu_cb_duration": metric}})
    base = {
        "phases": {
            "correctness": {"e2e_pass": 7, "e2e_fail": 0, "paris_pass": True},
            "standard_decode": {"runs": runs},
        },
        "metrics": {"expert_io": {}},
    }
    clone = lambda value: json.loads(json.dumps(value))
    assert select(base, "0" * 64)["decision"] == "ATTENTION_INVESTIGATION"

    routed = clone(base)
    for run in routed["phases"]["standard_decode"]["runs"]:
        metric = run["telemetry"]["sampled_gpu_cb_duration"]
        metric["attention"]["duration_ns_sum"] = 1
        metric["routed_moe"]["duration_ns_sum"] = 2
    assert select(routed, "0" * 64)["decision"] == "ROUTED_MOE_INVESTIGATION"

    tie = clone(base)
    for run in tie["phases"]["standard_decode"]["runs"]:
        metric = run["telemetry"]["sampled_gpu_cb_duration"]
        metric["routed_moe"]["duration_ns_sum"] = metric["attention"]["duration_ns_sum"]
    assert select(tie, "0" * 64)["decision"] == "NOT_TRIGGERED"

    mutations = (
        ("overflow_count", 1),
        ("attention.duration_ns_sum", True),
        ("attention.valid_count", 0),
        ("attention.zero_count", "0"),
    )
    for path, value in mutations:
        invalid = clone(base)
        metric = invalid["phases"]["standard_decode"]["runs"][0]["telemetry"]["sampled_gpu_cb_duration"]
        if "." in path:
            parent, key = path.split(".")
            metric[parent][key] = value
        else:
            metric[path] = value
        assert select(invalid, "0" * 64)["decision"] == "NOT_TRIGGERED"
    for malformed in ([], {"phases": []}, {"phases": {"standard_decode": {"runs": [None, None, None]}}}):
        report = select(malformed, "0" * 64)
        assert report["decision"] == "NOT_TRIGGERED"
        assert report["rejection_reasons"]
    print("diagnostic direction self-test: PASS")

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.input is None or args.output is None:
        parser.error("INPUT and --output are required")
    doc=json.loads(args.input.read_text(encoding="utf-8"))
    report=select(doc,file_sha256(args.input))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"diagnostic direction: {report['decision']}")

if __name__ == "__main__": main()
