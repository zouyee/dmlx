#!/usr/bin/env python3
"""Fail-closed verifier for bounded mhc_pre operational-candidate evidence."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/en/analysis/dsv4-mhc-pre-investigation-protocol.json"
EVIDENCE = ROOT / "docs/en/analysis/dsv4-mhc-pre-investigation-evidence.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    evidence = json.loads(EVIDENCE.read_text())
    runs = evidence.get("runs", [])
    if len(runs) != protocol["independent_process_runs"]:
        fail("independent process count mismatch")
    identity = evidence.get("build_identity", {})
    expected = {
        "harness_sha256": sha256(ROOT / "scripts/mhc_pre_gpu_test.m"),
        "production_kernel_sha256": sha256(ROOT / "src/models/moe_kernel.metal"),
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            fail(f"{key} mismatch")
    pids = [run.get("process_id") for run in runs]
    if any(not isinstance(pid, int) or pid <= 0 for pid in pids) or len(set(pids)) != len(pids):
        fail("missing or non-distinct process identities")
    passes = []
    for run in runs:
        if run.get("oracle_status") != "PASS":
            fail(f"run {run.get('ordinal')} oracle did not pass")
        for key in ("full_raw_ns", "dot_raw_ns", "phase_b_raw_ns"):
            values = run.get(key, [])
            if len(values) != protocol["measured_repetitions_per_arm"] or any(v <= 0 for v in values):
                fail(f"run {run.get('ordinal')} invalid {key}")
        full = statistics.median(run["full_raw_ns"])
        dot = statistics.median(run["dot_raw_ns"])
        phase_b = statistics.median(run["phase_b_raw_ns"])
        ratio = dot / full
        if (full, dot, phase_b) != (run["full_median_ns"], run["dot_median_ns"], run["phase_b_median_ns"]):
            fail(f"run {run.get('ordinal')} median mismatch")
        if not math.isclose(ratio, run["dot_full_ratio"], rel_tol=0, abs_tol=5e-7):
            fail(f"run {run.get('ordinal')} ratio mismatch")
        passes.append(ratio >= protocol["threshold"])
    if not all(passes) or evidence.get("result") != "OPERATIONAL_CANDIDATE_RETAINED":
        fail("all-three operational gate failed closed")
    if evidence.get("decision", {}).get("all_processes_pass") is not True:
        fail("decision summary mismatch")
    print("mhc_pre evidence PASS", [round(run["dot_full_ratio"], 6) for run in runs])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"mhc_pre evidence FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
