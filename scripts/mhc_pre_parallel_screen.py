#!/usr/bin/env python3
"""Collect or fail-closed verify the standalone mhc_pre parallel screen."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/mhc_pre_parallel_gpu_test.m"
METAL = ROOT / "scripts/mhc_pre_parallel_kernels.metal"
PRODUCTION = ROOT / "src/models/moe_kernel.metal"
PROTOCOL = ROOT / "docs/en/analysis/dsv4-mhc-pre-parallel-screen-protocol.json"
EVIDENCE = ROOT / "docs/en/analysis/dsv4-mhc-pre-parallel-screen-evidence.json"
BINARY = Path("/tmp/mhc_pre_parallel_gpu_test")
ARMS = ("serial_one_lane_per_row", "simdgroup_per_row_wave8", "eight_lane_per_row")
CANDIDATES = ARMS[1:]
TOPOLOGY = "1cb/1encoder/1dispatch/1commit/1wait/1x256"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_int(value: object) -> bool:
    return type(value) is int


def identities() -> dict[str, str]:
    return {
        "harness_source_sha256": sha256(HARNESS),
        "harness_metal_sha256": sha256(METAL),
        "production_kernel_sha256": sha256(PRODUCTION),
    }


def validate_run(run: dict, ordinal: int, expected_hashes: dict[str, str]) -> dict[str, int]:
    require(run.get("schema_version") == 1, f"run {ordinal}: schema")
    require(exact_int(run.get("process_ordinal")) and run["process_ordinal"] == ordinal, f"run {ordinal}: ordinal")
    require(exact_int(run.get("process_id")) and run["process_id"] > 0, f"run {ordinal}: pid")
    require(run.get("oracle_status") == "PASS", f"run {ordinal}: oracle")
    require(run.get("topology") == TOPOLOGY, f"run {ordinal}: topology")
    for key, value in expected_hashes.items():
        require(run.get(key) == value, f"run {ordinal}: {key}")
    events = run.get("events")
    require(isinstance(events, list) and len(events) == 93, f"run {ordinal}: event count")
    measured: dict[str, list[int]] = defaultdict(list)
    cursor = 0
    for phase, iterations in (("warmup", 10), ("measured", 21)):
        for iteration in range(iterations):
            for step in range(3):
                event = events[cursor]; cursor += 1
                arm = ARMS[(ordinal - 1 + iteration + step) % 3]
                require(event.get("phase") == phase, f"run {ordinal}: phase at {cursor}")
                require(exact_int(event.get("process_ordinal")) and event["process_ordinal"] == ordinal, f"run {ordinal}: event ordinal at {cursor}")
                require(exact_int(event.get("iteration")) and exact_int(event.get("step")) and event["iteration"] == iteration and event["step"] == step, f"run {ordinal}: coordinates at {cursor}")
                require(event.get("arm") == arm, f"run {ordinal}: arm at {cursor}")
                duration = event.get("duration_ns")
                if phase == "warmup":
                    require(duration is None, f"run {ordinal}: warmup duration at {cursor}")
                else:
                    require(exact_int(duration) and duration > 0, f"run {ordinal}: measured duration at {cursor}")
                    measured[arm].append(duration)
    require(all(len(measured[arm]) == 21 for arm in ARMS), f"run {ordinal}: per-arm samples")
    return {arm: int(statistics.median(measured[arm])) for arm in ARMS}


def decision(medians: list[dict[str, int]]) -> tuple[dict, str]:
    per_candidate: dict[str, dict] = {}
    viable: list[str] = []
    for candidate in CANDIDATES:
        ratios = [Fraction(row[candidate], row[ARMS[0]]) for row in medians]
        is_viable = all(ratio <= Fraction(4, 5) for ratio in ratios)
        score = ratios[0] * ratios[1] * ratios[2]
        per_candidate[candidate] = {
            "ratios": [{"numerator": r.numerator, "denominator": r.denominator, "decimal": float(r)} for r in ratios],
            "viable": is_viable,
            "exact_product_score": {"numerator": score.numerator, "denominator": score.denominator, "decimal": float(score)},
        }
        if is_viable:
            viable.append(candidate)
    if not viable:
        result = "REJECTED"
    elif len(viable) == 1:
        result = f"RECOMMEND_{viable[0].upper()}"
    else:
        a, b = viable
        sa = Fraction(per_candidate[a]["exact_product_score"]["numerator"], per_candidate[a]["exact_product_score"]["denominator"])
        sb = Fraction(per_candidate[b]["exact_product_score"]["numerator"], per_candidate[b]["exact_product_score"]["denominator"])
        if abs(sa - sb) <= Fraction(1, 100):
            result = "NOT_PROVEN"
        else:
            result = f"RECOMMEND_{(a if sa < sb else b).upper()}"
    return per_candidate, result


def validate_evidence(evidence: dict) -> str:
    protocol = json.loads(PROTOCOL.read_text())
    require(protocol.get("status") == "frozen_before_implementation_and_measurement", "protocol status")
    expected_hashes = identities()
    require(evidence.get("schema_version") == 1, "evidence schema")
    require(evidence.get("protocol") == PROTOCOL.name, "protocol reference")
    require(evidence.get("build_identity") == expected_hashes, "build identity")
    runs = evidence.get("runs")
    require(isinstance(runs, list) and len(runs) == 3, "run count")
    require(len({run.get("process_id") for run in runs}) == 3, "distinct process IDs")
    medians = [validate_run(run, ordinal, expected_hashes) for ordinal, run in enumerate(runs, 1)]
    require(evidence.get("medians_ns") == medians, "median summary")
    candidates, result = decision(medians)
    require(evidence.get("candidates") == candidates, "candidate summary")
    require(evidence.get("result") == result, "result")
    require(evidence.get("authority") == "standalone_mapping_recommendation_only_no_production_authority", "authority")
    return result


def tamper_self_test(evidence: dict) -> None:
    cases = []
    pid_bool = copy.deepcopy(evidence)
    pid_bool["runs"][0]["process_id"] = True
    cases.append(("boolean PID", pid_bool))
    duration_bool = copy.deepcopy(evidence)
    measured = next(event for event in duration_bool["runs"][0]["events"] if event["phase"] == "measured")
    measured["duration_ns"] = True
    cases.append(("boolean duration", duration_bool))
    for name, tampered in cases:
        try:
            validate_evidence(tampered)
        except ValueError:
            continue
        raise ValueError(f"tamper self-test accepted {name}")


def collect() -> dict:
    compile_command = [
        "clang", "-Wall", "-Wextra", "-Werror", "-framework", "Metal", "-framework", "Foundation",
        "-fobjc-arc", str(HARNESS.relative_to(ROOT)), "-o", str(BINARY),
    ]
    subprocess.run(compile_command, cwd=ROOT, check=True)
    runs = []
    for ordinal in range(1, 4):
        completed = subprocess.run([str(BINARY), str(ordinal)], cwd=ROOT, check=True, text=True, capture_output=True)
        runs.append(json.loads(completed.stdout))
    expected_hashes = identities()
    medians = [validate_run(run, ordinal, expected_hashes) for ordinal, run in enumerate(runs, 1)]
    candidates, result = decision(medians)
    evidence = {
        "schema_version": 1,
        "protocol": PROTOCOL.name,
        "collection_command": "python3 scripts/mhc_pre_parallel_screen.py --collect",
        "verification_command": "python3 scripts/mhc_pre_parallel_screen.py",
        "compile_command": " ".join(compile_command),
        "build_identity": expected_hashes,
        "runs": runs,
        "medians_ns": medians,
        "candidates": candidates,
        "result": result,
        "authority": "standalone_mapping_recommendation_only_no_production_authority",
    }
    EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n")
    validate_evidence(evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="compile, run three serial processes, and replace evidence")
    args = parser.parse_args()
    evidence = collect() if args.collect else json.loads(EVIDENCE.read_text())
    result = validate_evidence(evidence)
    tamper_self_test(evidence)
    print(f"mhc_pre parallel screen PASS: {result}; boolean tamper tests PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"mhc_pre parallel screen FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
