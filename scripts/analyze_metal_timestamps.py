#!/usr/bin/env python3
"""Deterministic offline proof of Metal-to-monotonic clock correlation."""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, getcontext
from itertools import combinations
from pathlib import Path
from typing import Any

from gpu_timestamp_artifact import ArtifactError, CAPACITY, seal as seal_artifact, validate as validate_artifact

getcontext().prec = 60
D = Decimal
REPORT_VERSION = 1
TOL = D(100_000)
SLOPE_LO, SLOPE_HI = D(999_000_000), D(1_001_000_000)
PROOF_LO, PROOF_HI = D(999_900_000), D(1_000_100_000)
KIND_NAMES = {1: "attention", 2: "routed_moe"}
GATE_KEYS = {
    "enabled", "no_overflow", "minimum_counts", "coverage", "local_validity",
    "nonempty_region", "slope_projection", "uncertainty_p95", "uncertainty_max",
    "containment", "regression",
}
REPORT_KEYS = {
    "schema_version", "source", "policy", "counts", "sample_bounds", "feasible_region",
    "selected_mapping", "uncertainty", "regressions", "containment", "gates",
    "failed_gates", "decision", "content_sha256",
}

class AnalysisError(ValueError):
    pass

def load_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise AnalysisError(f"non-finite JSON constant rejected: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)

def dec(value: Any, name: str = "number") -> D:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise AnalysisError(f"{name} must be a non-boolean number")
    try:
        result = D(str(value))
    except Exception as error:
        raise AnalysisError(f"{name} is not decimal") from error
    if not result.is_finite():
        raise AnalysisError(f"{name} must be finite")
    return result

def ds(value: D) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")

def canonical_bytes(payload: dict[str, Any]) -> bytes:
    semantic = dict(payload)
    semantic.pop("content_sha256", None)
    return json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")

def report_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()

def seal_report(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["content_sha256"] = report_hash(result)
    return result

def locally_valid(sample: dict[str, Any]) -> bool:
    if sample["valid"] != 1 or sample["reason"] != 0:
        return False
    start, end = dec(sample["gpu_start_s"]), dec(sample["gpu_end_s"])
    return (sample["pre_commit_ns"] <= sample["post_commit_ns"] <= sample["post_wait_ns"]
            and start > 0 and end > start)

def r7(values: list[D], percentile: D) -> D:
    if not values:
        return D(0)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    h = (D(len(ordered)) - 1) * percentile
    lo = int(h)
    fraction = h - lo
    return ordered[lo] + fraction * (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo])

# Halfplane is A*a + B*b <= C.
def halfplanes(samples: list[dict[str, Any]]) -> list[tuple[D, D, D, str]]:
    result = [(D(-1), D(0), -SLOPE_LO, "slope_min"),
              (D(1), D(0), SLOPE_HI, "slope_max")]
    for sample in samples:
        seq = sample["sequence"]
        start, end = dec(sample["gpu_start_s"]), dec(sample["gpu_end_s"])
        lower, upper = D(sample["pre_commit_ns"]) - TOL, D(sample["post_wait_ns"]) + TOL
        result.append((-start, D(-1), -lower, f"sample[{seq}].start_lower"))
        result.append((end, D(1), upper, f"sample[{seq}].end_upper"))
    return result

def feasible(point: tuple[D, D], planes: list[tuple[D, D, D, str]]) -> bool:
    a, b = point
    return all(A * a + B * b <= C for A, B, C, _ in planes)

def polygon_vertices(planes: list[tuple[D, D, D, str]]) -> list[tuple[D, D]]:
    # The slope prior is bounded. Derive a safe finite b rectangle from all
    # non-slope constraints, then clip it once per halfplane (O(n*v)).
    b_candidates = []
    for A, B, C, _ in planes:
        if B != 0:
            for a in (SLOPE_LO, SLOPE_HI):
                b_candidates.append((C - A * a) / B)
    if not b_candidates:
        return []
    b_lo, b_hi = min(b_candidates), max(b_candidates)
    polygon = [(SLOPE_LO, b_lo), (SLOPE_HI, b_lo), (SLOPE_HI, b_hi), (SLOPE_LO, b_hi)]
    for A, B, C, _ in planes:
        if not polygon:
            break
        clipped: list[tuple[D, D]] = []
        previous = polygon[-1]
        previous_value = A * previous[0] + B * previous[1] - C
        for current in polygon:
            current_value = A * current[0] + B * current[1] - C
            previous_inside, current_inside = previous_value <= 0, current_value <= 0
            if previous_inside != current_inside:
                denominator = previous_value - current_value
                t = previous_value / denominator
                clipped.append((previous[0] + t * (current[0] - previous[0]),
                                previous[1] + t * (current[1] - previous[1])))
            if current_inside:
                clipped.append(current)
            previous, previous_value = current, current_value
        polygon = clipped
    ordered: list[tuple[D, D]] = []
    for point in polygon:
        if not ordered or point != ordered[-1]:
            ordered.append(point)
    if len(ordered) > 1 and ordered[0] == ordered[-1]:
        ordered.pop()
    return ordered

def solve3(rows: tuple[tuple[D, D, D, D], ...]) -> tuple[D, D, D] | None:
    matrix = [list(row) for row in rows]
    for col in range(3):
        pivot = next((row for row in range(col, 3) if matrix[row][col] != 0), None)
        if pivot is None:
            return None
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        divisor = matrix[col][col]
        matrix[col] = [value / divisor for value in matrix[col]]
        for row in range(3):
            if row == col:
                continue
            factor = matrix[row][col]
            matrix[row] = [matrix[row][i] - factor * matrix[col][i] for i in range(4)]
    return matrix[0][3], matrix[1][3], matrix[2][3]

def selected_center(planes: list[tuple[D, D, D, str]], samples: list[dict[str, Any]],
                    vertices: list[tuple[D, D]]) -> dict[str, str]:
    mids = [((D(s["pre_commit_ns"]) - TOL + D(s["post_wait_ns"]) + TOL) / 2
             - D(1_000_000_000) * (dec(s["gpu_start_s"]) + dec(s["gpu_end_s"])) / 2)
            for s in samples]
    b_ref = r7(mids, D("0.5"))
    lowers = [D(s["pre_commit_ns"]) - TOL for s in samples]
    uppers = [D(s["post_wait_ns"]) + TOL for s in samples]
    Bscale = max(D(1), max(uppers) - min(lowers))
    raw_area = (sum(vertices[i][0] * vertices[(i + 1) % len(vertices)][1]
                    - vertices[(i + 1) % len(vertices)][0] * vertices[i][1]
                    for i in range(len(vertices))) if len(vertices) >= 3 else D(0))
    area = abs(raw_area)
    if area == 0:
        a, b = min(vertices)
        alpha = (a - D(1_000_000_000)) / D(1_000_000)
        beta = (b - b_ref) / Bscale
        return {"b_ref": ds(b_ref), "B": ds(Bscale), "alpha": ds(alpha), "beta": ds(beta),
                "a": ds(a), "b": ds(b), "radius": "0"}
    normalized_vertices = [((a - D(1_000_000_000)) / D(1_000_000),
                            (b - b_ref) / Bscale) for a, b in vertices]
    orientation = D(1) if raw_area > 0 else D(-1)
    normalized: list[tuple[D, D, D, D]] = []
    for index, (x1, y1) in enumerate(normalized_vertices):
        x2, y2 = normalized_vertices[(index + 1) % len(normalized_vertices)]
        dx, dy = x2 - x1, y2 - y1
        p, q, c = orientation * dy, -orientation * dx, orientation * (dy * x1 - dx * y1)
        normalized.append((p, q, abs(p) + abs(q), c))
    normalized.append((D(0), D(0), D(-1), D(0))) # r >= 0
    candidates: list[tuple[D, D, D]] = []
    for rows in combinations(normalized, 3):
        point = solve3(rows)
        if point is not None and all(p*point[0] + q*point[1] + w*point[2] <= c
                                     for p, q, w, c in normalized):
            candidates.append(point)
    if not candidates:
        raise AnalysisError("unable to select Chebyshev center")
    alpha, beta, radius = min(candidates, key=lambda x: (-x[2], x[0], x[1]))
    a = D(1_000_000_000) + D(1_000_000) * alpha
    b = b_ref + Bscale * beta
    return {"b_ref": ds(b_ref), "B": ds(Bscale), "alpha": ds(alpha), "beta": ds(beta),
            "a": ds(a), "b": ds(b), "radius": ds(radius)}

def analyze(artifact: dict[str, Any]) -> dict[str, Any]:
    validate_artifact(artifact)
    retained = artifact["samples"]
    valid = [sample for sample in retained if locally_valid(sample)]
    by_kind = {kind: [s for s in valid if s["kind"] == kind] for kind in KIND_NAMES}
    total = len(retained); valid_count = len(valid)
    coverage = D(valid_count) / D(total) if total else D(0)
    counts = {
        "retained": total, "valid": valid_count, "invalid": total - valid_count,
        "coverage": ds(coverage), "invalid_rate": ds(D(1) - coverage),
        "by_kind": {KIND_NAMES[k]: {"retained": sum(s["kind"] == k for s in retained),
                                     "valid": len(by_kind[k]),
                                     "invalid": sum(s["kind"] == k for s in retained) - len(by_kind[k])}
                    for k in KIND_NAMES},
        "by_reason": {str(reason): sum(s["reason"] == reason for s in retained) for reason in range(6)},
    }
    planes = halfplanes(valid)
    vertices = polygon_vertices(planes) if valid else []
    region_nonempty = bool(vertices)
    region: dict[str, Any] = {
        "halfplanes": [{"A": ds(A), "B": ds(B), "C": ds(C), "label": label}
                       for A, B, C, label in planes],
        "vertices": [{"a": ds(a), "b": ds(b)} for a, b in vertices],
        "bbox": None,
    }
    center = None
    if vertices:
        region["bbox"] = {"a_min": ds(min(v[0] for v in vertices)), "a_max": ds(max(v[0] for v in vertices)),
                          "b_min": ds(min(v[1] for v in vertices)), "b_max": ds(max(v[1] for v in vertices))}
        center = selected_center(planes, valid, vertices)
    uncertainty: dict[str, Any] = {}
    uncertainty_p95_ok = uncertainty_max_ok = region_nonempty
    for kind, name in KIND_NAMES.items():
        spreads: list[D] = []
        if vertices:
            for sample in by_kind[kind]:
                for field in ("gpu_start_s", "gpu_end_s"):
                    gpu = dec(sample[field]); mapped = [a * gpu + b for a, b in vertices]
                    spreads.append(max(mapped) - min(mapped))
        p50, p95, maximum = r7(spreads, D("0.5")), r7(spreads, D("0.95")), max(spreads, default=D(0))
        uncertainty[name] = {"endpoint_count": len(spreads), "p50_ns": ds(p50),
                             "p95_ns": ds(p95), "max_ns": ds(maximum)}
        uncertainty_p95_ok &= p95 <= D(50_000)
        uncertainty_max_ok &= maximum <= D(200_000)
    regressions: dict[str, Any] = {}
    regression_ok = True
    for kind, name in KIND_NAMES.items():
        failures = []
        previous = None
        for sample in by_kind[kind]:
            current = (dec(sample["gpu_start_s"]), dec(sample["gpu_end_s"]))
            if previous is not None:
                if current[0] < previous[0]: failures.append({"sequence": sample["sequence"], "endpoint": "start"})
                if current[1] < previous[1]: failures.append({"sequence": sample["sequence"], "endpoint": "end"})
            previous = current
        regressions[name] = failures
        regression_ok &= not failures
    containment_failures: list[int] = []
    if center:
        a, b = D(center["a"]), D(center["b"])
        for sample in valid:
            mapped_start = a * dec(sample["gpu_start_s"]) + b
            mapped_end = a * dec(sample["gpu_end_s"]) + b
            if mapped_start < D(sample["pre_commit_ns"]) - TOL or mapped_end > D(sample["post_wait_ns"]) + TOL:
                containment_failures.append(sample["sequence"])
    containment_coverage = ((D(len(valid) - len(containment_failures)) / D(len(valid))) if valid else D(0))
    slope_ok = bool(vertices) and min(v[0] for v in vertices) >= PROOF_LO and max(v[0] for v in vertices) <= PROOF_HI
    gates = {
        "enabled": artifact["enabled"], "no_overflow": artifact["overflow_count"] == 0,
        "minimum_counts": all(len(by_kind[k]) >= 100 for k in KIND_NAMES),
        "coverage": coverage >= D("0.99"), "local_validity": all(locally_valid(s) for s in valid),
        "nonempty_region": region_nonempty, "slope_projection": slope_ok,
        "uncertainty_p95": bool(uncertainty_p95_ok), "uncertainty_max": bool(uncertainty_max_ok),
        "containment": containment_coverage >= D("0.99"), "regression": regression_ok,
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    gpu_values = [dec(s[f]) for s in valid for f in ("gpu_start_s", "gpu_end_s")]
    host_values = [D(s[f]) for s in valid for f in ("pre_commit_ns", "post_wait_ns")]
    report = {
        "schema_version": REPORT_VERSION,
        "source": {"content_sha256": artifact["content_sha256"], "identity": artifact["identity"],
                   "enabled": artifact["enabled"], "capacity": artifact["capacity"],
                   "total_seen": artifact["total_seen"], "overflow_count": artifact["overflow_count"]},
        "policy": {"decimal_precision": getcontext().prec, "tolerance_ns": 100000,
                   "slope_prior": [ds(SLOPE_LO), ds(SLOPE_HI)],
                   "slope_gate": [ds(PROOF_LO), ds(PROOF_HI)], "percentile": "linear_R7"},
        "counts": counts,
        "sample_bounds": None if not valid else {"gpu_min": ds(min(gpu_values)), "gpu_max": ds(max(gpu_values)),
                                                  "host_min": ds(min(host_values)), "host_max": ds(max(host_values))},
        "feasible_region": region, "selected_mapping": center, "uncertainty": uncertainty,
        "regressions": regressions,
        "containment": {"valid_denominator": len(valid), "failures": containment_failures,
                        "coverage": ds(containment_coverage)},
        "gates": gates, "failed_gates": failed,
        "decision": "PROVEN" if not failed else "NOT_PROVEN",
    }
    return seal_report(report)

def validate_report(report: Any, artifact: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or set(report) != REPORT_KEYS:
        raise AnalysisError("report top-level fields do not match schema v1")
    if report["schema_version"] != REPORT_VERSION:
        raise AnalysisError("unsupported report schema_version")
    if not isinstance(report["content_sha256"], str) or report["content_sha256"] != report_hash(report):
        raise AnalysisError("report content_sha256 mismatch")
    expected = analyze(artifact)
    if report != expected:
        raise AnalysisError("report does not match source-derived replay")
    return report

def identity() -> dict[str, Any]:
    return {"pid": 4242, "process_start_ns": 1, "device_registry_id": 1, "device_name": "fixture",
            "os_version": "fixture", "binary_sha256": "1" * 64, "git_commit": "fixture", "worktree_dirty": False}

def make_artifact(per_kind: int = 100, *, slope: D = D(1_000_000_000), margin: int = -100_000,
                  invalid: list[tuple[int, int]] | None = None, overflow: int = 0,
                  regress: bool = False, outlier: bool = False) -> dict[str, Any]:
    samples = []
    invalid = invalid or []
    for index in range(per_kind * 2):
        kind = index % 2 + 1; ordinal = index // 2
        start = D("10") + D(ordinal) * D(10) + D(kind) / D(100000)
        if regress and kind == 1 and ordinal == per_kind - 1: start -= D(100)
        end = start + D("0.001")
        base = slope * start + D(5_000_000_000)
        pre = int(base - margin); wait = int(slope * end + D(5_000_000_000) + margin)
        if outlier and index == per_kind * 2 - 1:
            pre += 10_000_000; wait += 10_000_000
        reason = next((r for i, r in invalid if i == index), 0)
        valid_flag = 0 if reason else 1
        gs, ge = (D(0), D(0)) if reason in (1, 2) else (start, end)
        if reason == 3: gs = ge = D(0)
        if reason == 4: ge = gs
        if reason == 5: wait = pre - 1
        samples.append({"sequence": index, "request_id": index // 20, "token_index": ordinal, "layer": ordinal % 43,
                        "kind": kind, "pre_commit_ns": pre, "post_commit_ns": pre if reason != 5 else pre + 2,
                        "post_wait_ns": wait, "gpu_start_s": float(gs), "gpu_end_s": float(ge),
                        "valid": valid_flag, "reason": reason})
    return seal_artifact({"schema_version": 1, "identity": identity(), "enabled": True, "capacity": CAPACITY,
                          "total_seen": len(samples) + overflow, "overflow_count": overflow, "samples": samples})

def expect_gate(artifact: dict[str, Any], gate: str, expected: bool) -> dict[str, Any]:
    report = analyze(artifact); assert report["gates"][gate] is expected, (gate, report["failed_gates"])
    validate_report(report, artifact)
    return report

def self_test() -> None:
    proven = make_artifact()
    report = analyze(proven)
    assert report["decision"] == "PROVEN"
    assert canonical_bytes(report) == canonical_bytes(analyze(proven))
    # Offset and exact/adjacent drift boundaries.
    expect_gate(make_artifact(slope=PROOF_LO), "slope_projection", True)
    expect_gate(make_artifact(slope=PROOF_HI), "slope_projection", True)
    expect_gate(make_artifact(slope=PROOF_LO - 1), "slope_projection", False)
    expect_gate(make_artifact(slope=PROOF_HI + 1), "slope_projection", False)
    # Count, overflow, global coverage, all raw invalid projections, regression, mixed streams.
    expect_gate(make_artifact(per_kind=99), "minimum_counts", False)
    expect_gate(make_artifact(overflow=1), "no_overflow", False)
    expect_gate(make_artifact(invalid=[(0, 1), (1, 2)]), "coverage", True)  # 198/200 = 99%
    expect_gate(make_artifact(invalid=[(0, 1), (1, 2), (2, 3)]), "coverage", False)
    for reason in (2, 3, 4, 5): analyze(make_artifact(invalid=[(0, reason)]))
    expect_gate(make_artifact(regress=True), "regression", False)
    expect_gate(make_artifact(outlier=True), "nonempty_region", False)
    # Empty denominator semantics.
    empty = make_artifact(per_kind=0); assert analyze(empty)["counts"]["coverage"] == "0"
    # Artifact-driven uncertainty boundaries and adjacent failures.
    p95_boundary = analyze(make_artifact(margin=-75_000))
    assert p95_boundary["uncertainty"]["attention"]["p95_ns"] == "50000"
    assert p95_boundary["gates"]["uncertainty_p95"]
    expect_gate(make_artifact(margin=-74_999), "uncertainty_p95", False)
    assert analyze(make_artifact(margin=-1))["gates"]["uncertainty_max"]
    expect_gate(make_artifact(margin=0), "uncertainty_max", False)
    # A non-degenerate polygon must produce a positive normalized L-infinity radius.
    assert D(analyze(make_artifact(margin=0))["selected_mapping"]["radius"]) > 0
    # Closed containment equality: the point-region mapping touches both envelope limits.
    selected = report["selected_mapping"]; first = proven["samples"][0]
    mapped_start = D(selected["a"]) * dec(first["gpu_start_s"]) + D(selected["b"])
    mapped_end = D(selected["a"]) * dec(first["gpu_end_s"]) + D(selected["b"])
    assert mapped_start == D(first["pre_commit_ns"]) - TOL
    assert mapped_end == D(first["post_wait_ns"]) + TOL
    assert report["gates"]["containment"]
    tamper_cases = []
    for mutation in (
        lambda r: r.__setitem__("decision", "NOT_PROVEN"),
        lambda r: r["gates"].__setitem__("containment", False),
        lambda r: r["source"]["identity"].__setitem__("device_name", "tampered"),
        lambda r: r.__setitem__("extra", 1),
        lambda r: r.__setitem__("schema_version", 2),
    ):
        candidate = json.loads(json.dumps(report)); mutation(candidate)
        candidate["content_sha256"] = report_hash(candidate); tamper_cases.append(candidate)
    hash_tamper = json.loads(json.dumps(report)); hash_tamper["content_sha256"] = "0" * 64; tamper_cases.append(hash_tamper)
    for candidate in tamper_cases:
        try: validate_report(candidate, proven)
        except AnalysisError: pass
        else: raise AssertionError("tampered report accepted")
    # Source artifact content/identity tamper is rejected by raw validation/replay.
    source_tamper = json.loads(json.dumps(proven)); source_tamper["identity"]["device_name"] = "tampered"
    try: validate_report(report, source_tamper)
    except (AnalysisError, ArtifactError): pass
    else: raise AssertionError("tampered source accepted")
    print("analyze_metal_timestamps self-test: PASS")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-report", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.validate_report:
        if not args.artifact: parser.error("--validate-report requires --artifact")
        validate_report(load_json(args.validate_report), load_json(args.artifact))
        print(f"GPU timestamp analysis report valid: {args.validate_report}"); return
    if not args.input: parser.error("provide INPUT, --validate-report, or --self-test")
    report = analyze(load_json(args.input))
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output: args.output.write_text(text, encoding="utf-8")
    else: print(text, end="")

if __name__ == "__main__":
    main()
