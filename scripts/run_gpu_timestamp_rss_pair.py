#!/usr/bin/env python3
"""Frozen G1.3 paired process-RSS protocol. Stdlib only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any

SCHEMA = "dmlx.g1_3.paired_process_rss.v1"
ORDER = ("B1", "I1", "B2", "I2")
PERIOD_NS = 200_000_000
MAX_GAP_NS = 750_000_000  # pre-registered scheduling tolerance, not a 0.2s exactness claim
POST_RESPONSE_NS = 1_000_000_000
GIB = 1 << 30
MIB = 1 << 20
POLICY = {
    "sample_period_ns": PERIOD_NS,
    "max_adjacent_gap_ns": MAX_GAP_NS,
    "post_final_response_min_ns": POST_RESPONSE_NS,
    "peak_upper_exclusive_bytes": 44 * GIB,
    "pair_absolute_max_bytes": 256 * MIB,
    "pair_relative_max_ppm": 15_000,
    "mean_absolute_max_bytes": 128 * MIB,
    "mean_relative_max_ppm": 10_000,
    "baseline_drift_max_bytes": GIB,
    "baseline_drift_max_ppm": 50_000,
}
EXPERIMENT_ENV = {
    "NATIVE_SMELT_N": "20",
    "NATIVE_SMELT_STRATEGY": "stream",
    "NATIVE_EXPERT_PREFETCH": "0",
    "NATIVE_EXPERT_CACHE_EXPERIMENT": "0",
    "NATIVE_BATCH_EXPERIMENT": "0",
    "NATIVE_DSPARK": "0",
    "DSV4_GPU_TIMESTAMP_ENABLE": "0",
}
BUILD_COMMANDS = {
    "baseline": ["zig", "build", "-Doptimize=ReleaseFast", "-Dgpu_timestamp_probe=false"],
    "instrumented": ["zig", "build", "-Doptimize=ReleaseFast", "-Dgpu_timestamp_probe=true"],
}
BINARY_KEYS = {"build_command", "probe_compiled", "frozen_path", "sha256", "size",
               "sha256_before", "size_before", "sha256_after", "size_after"}
WORKLOAD = [
    {"label": "warmup", "endpoint": "/v1/chat/completions", "model": "default",
     "prompt": "Hi", "max_tokens": 5, "temperature": 0},
    {"label": "paris", "endpoint": "/v1/chat/completions", "model": "default",
     "prompt": "The capital of France is", "max_tokens": 30, "temperature": 0},
    {"label": "arithmetic", "endpoint": "/v1/chat/completions", "model": "default",
     "prompt": "2+2=", "max_tokens": 30, "temperature": 0},
]

class InvalidArtifact(ValueError):
    pass


def canonical_bytes(value: dict[str, Any]) -> bytes:
    unsigned = dict(value)
    unsigned.pop("semantic_sha256", None)
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def seal(value: dict[str, Any]) -> dict[str, Any]:
    value["semantic_sha256"] = hashlib.sha256(canonical_bytes(value)).hexdigest()
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dirty_workspace_sha256(workspace: pathlib.Path) -> str:
    """Bind tracked changes plus every non-ignored untracked path and its content."""
    digest = hashlib.sha256()
    tracked = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=workspace)
    digest.update(b"tracked-diff\0"); digest.update(len(tracked).to_bytes(8, "big")); digest.update(tracked)
    raw_paths = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=workspace)
    paths = sorted(path for path in raw_paths.split(b"\0") if path)
    for encoded in paths:
        relative = os.fsdecode(encoded); candidate = workspace / relative
        digest.update(b"untracked-path\0"); digest.update(len(encoded).to_bytes(8, "big")); digest.update(encoded)
        if candidate.is_symlink():
            content = os.fsencode(os.readlink(candidate)); kind = b"symlink\0"
        elif candidate.is_file():
            content = candidate.read_bytes(); kind = b"file\0"
        else:
            raise RuntimeError(f"unsupported untracked workspace entry: {relative}")
        digest.update(kind); digest.update(len(content).to_bytes(8, "big")); digest.update(content)
    return digest.hexdigest()


def ppm_le(numerator: int, denominator: int, limit: int) -> bool:
    return denominator > 0 and numerator * 1_000_000 <= denominator * limit


def derive(peaks: dict[str, int]) -> tuple[dict[str, Any], dict[str, bool]]:
    if tuple(peaks) != ORDER:
        raise InvalidArtifact("segment peak order is not B1,I1,B2,I2")
    p = [peaks[name] for name in ORDER]
    deltas = [p[1] - p[0], p[3] - p[2]]
    absolutes = [abs(value) for value in deltas]
    pairs = [{
        "baseline": ORDER[index * 2], "instrumented": ORDER[index * 2 + 1],
        "delta_bytes": deltas[index], "absolute_bytes": absolutes[index],
        "relative_numerator_bytes": absolutes[index],
        "relative_denominator_bytes": p[index * 2],
    } for index in range(2)]
    drift = abs(p[2] - p[0])
    derived = {
        "peaks_bytes": peaks,
        "pairs": pairs,
        "mean_absolute_numerator_bytes": sum(absolutes),
        "mean_absolute_denominator": 2,
        "mean_relative_sum": [{"numerator_bytes": absolutes[i],
                               "denominator_bytes": p[i * 2]} for i in range(2)],
        "mean_relative_denominator": 2,
        "baseline_drift_bytes": drift,
        "baseline_drift_denominator_bytes": p[0],
    }
    gates: dict[str, bool] = {}
    for name, peak in peaks.items():
        gates[f"peak_{name}_range"] = 0 < peak < POLICY["peak_upper_exclusive_bytes"]
    for i, pair in enumerate(pairs, 1):
        gates[f"pair_{i}_absolute"] = pair["absolute_bytes"] <= POLICY["pair_absolute_max_bytes"]
        gates[f"pair_{i}_relative"] = ppm_le(pair["relative_numerator_bytes"], pair["relative_denominator_bytes"], POLICY["pair_relative_max_ppm"])
    gates["mean_absolute"] = sum(absolutes) <= 2 * POLICY["mean_absolute_max_bytes"]
    # (a1/b1+a2/b2)/2 <= ppm/1e6, evaluated with integers.
    b1, b2 = p[0], p[2]
    gates["mean_relative"] = (absolutes[0] * b2 + absolutes[1] * b1) * 1_000_000 <= 2 * POLICY["mean_relative_max_ppm"] * b1 * b2 if b1 > 0 and b2 > 0 else False
    gates["baseline_drift_absolute"] = drift <= POLICY["baseline_drift_max_bytes"]
    gates["baseline_drift_relative"] = ppm_le(drift, p[0], POLICY["baseline_drift_max_ppm"])
    return derived, gates


def has_post_response_sample(series: list[dict[str, int]], final_response_ns: int) -> bool:
    return bool(series and series[-1]["monotonic_ns"] - final_response_ns >= POST_RESPONSE_NS)


def validate_series(segment: dict[str, Any]) -> int:
    series = segment.get("rss_series")
    lifecycle = segment.get("lifecycle")
    if not isinstance(series, list) or not series:
        raise InvalidArtifact("empty RSS series")
    if not isinstance(lifecycle, dict) or not lifecycle.get("spawn_sampled") or not lifecycle.get("post_response_sampled"):
        raise InvalidArtifact("missing first/last lifecycle marker")
    previous = -1
    for sample in series:
        if set(sample) != {"monotonic_ns", "rss_bytes"}:
            raise InvalidArtifact("invalid RSS sample schema")
        timestamp, rss = sample["monotonic_ns"], sample["rss_bytes"]
        if not isinstance(timestamp, int) or not isinstance(rss, int) or rss <= 0:
            raise InvalidArtifact("invalid RSS sample value")
        if previous >= 0 and not (0 < timestamp - previous <= POLICY["max_adjacent_gap_ns"]):
            raise InvalidArtifact("non-monotonic or incomplete RSS series")
        previous = timestamp
    if series[0]["monotonic_ns"] > lifecycle["spawn_sample_deadline_ns"]:
        raise InvalidArtifact("series does not begin immediately after spawn")
    if not has_post_response_sample(series, lifecycle["final_response_ns"]):
        raise InvalidArtifact("series does not extend one second after final response")
    return max(sample["rss_bytes"] for sample in series)


def validate_artifact(path: pathlib.Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text())
    required = {"schema_version", "policy", "provenance", "binaries", "segments",
                "derived", "gates", "decision", "semantic_sha256"}
    if set(artifact) != required or artifact["schema_version"] != SCHEMA or artifact["policy"] != POLICY:
        raise InvalidArtifact("artifact schema/policy mismatch")
    if artifact["semantic_sha256"] != hashlib.sha256(canonical_bytes(artifact)).hexdigest():
        raise InvalidArtifact("semantic SHA-256 mismatch")
    root = path.resolve().parent
    provenance = artifact["provenance"]
    provenance_keys = {"workspace", "git_head", "dirty_diff_sha256", "git_status", "zig_version",
        "sdk", "platform", "model_path", "model_identity", "tokenizer_identity",
        "packed_experts_path", "packed_experts_identity", "vm_stat_before", "memory_pressure_before", "provenance_file"}
    identity_keys = provenance_keys - {"git_status"}
    if (not provenance_keys.issubset(provenance) or
            not all(isinstance(provenance[key], str) for key in provenance_keys) or
            not all(provenance[key].strip() and not provenance[key].startswith("UNAVAILABLE:")
                    for key in identity_keys) or
            not isinstance(provenance.get("port"), int) or not 1 <= provenance["port"] <= 65535):
        raise InvalidArtifact("incomplete or unavailable provenance")
    provenance_sidecar = root / provenance["provenance_file"]
    if not provenance_sidecar.is_file() or json.loads(provenance_sidecar.read_text()) != provenance:
        raise InvalidArtifact("provenance sidecar mismatch")
    binaries = artifact["binaries"]
    if not set(binaries).issubset({"baseline", "instrumented"}):
        raise InvalidArtifact("binary roles mismatch")
    for role, binary in binaries.items():
        if (not isinstance(binary, dict) or set(binary) != BINARY_KEYS or
                binary["build_command"] != BUILD_COMMANDS[role] or
                binary["probe_compiled"] is not (role == "instrumented") or
                not isinstance(binary["frozen_path"], str) or not binary["frozen_path"] or
                not isinstance(binary["sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", binary["sha256"]) or
                any(not isinstance(binary[key], str) for key in ("sha256_before", "sha256_after")) or
                any(not isinstance(binary[key], int) or binary[key] <= 0
                    for key in ("size", "size_before", "size_after"))):
            raise InvalidArtifact("binary provenance schema or build role mismatch")
        frozen = pathlib.Path(binary["frozen_path"])
        if binary["sha256"] != binary["sha256_before"] or binary["size"] != binary["size_before"]:
            raise InvalidArtifact("initial frozen binary identity mismatch")
        if binary["sha256_before"] != binary["sha256_after"] or binary["size_before"] != binary["size_after"]:
            raise InvalidArtifact("frozen binary changed during session")
        if not frozen.is_file() or sha256_file(frozen) != binary["sha256"] or frozen.stat().st_size != binary["size"]:
            raise InvalidArtifact("frozen binary provenance mismatch")
    if set(binaries) == {"baseline", "instrumented"} and binaries["baseline"]["sha256"] == binaries["instrumented"]["sha256"]:
        raise InvalidArtifact("baseline and instrumented binary hashes are equal")
    segments = artifact["segments"]
    failed = isinstance(provenance.get("failure"), str) and bool(provenance["failure"])
    names = [segment.get("name") for segment in segments]
    if names != list(ORDER[:len(names)]) or len(names) > len(ORDER):
        raise InvalidArtifact("segment lifecycle order mismatch")
    peaks: dict[str, int] = {}
    for segment in segments:
        expected_role = "baseline" if segment["name"].startswith("B") else "instrumented"
        if segment.get("binary_role") != expected_role or expected_role not in binaries:
            raise InvalidArtifact("segment role mismatch")
        expected_argv = [binaries[expected_role]["frozen_path"], "serve", "--model", provenance["model_path"],
            "--expert-packed-dir", provenance["packed_experts_path"], "--native", "--port", str(provenance["port"]),
            "--max-tokens", "30", "--temperature", "0"]
        argv = segment.get("argv", [])
        if argv != expected_argv:
            raise InvalidArtifact("segment argv differs from frozen workload")
        if segment.get("workload") != WORKLOAD:
            raise InvalidArtifact("segment workload differs from frozen chat contract")
        raw_path = root / segment.get("rss_jsonl", "")
        try: raw_series = [json.loads(line) for line in raw_path.read_text().splitlines()]
        except Exception as error: raise InvalidArtifact(f"RSS sidecar unreadable: {error}") from error
        if raw_series != segment.get("rss_series"):
            raise InvalidArtifact("RSS sidecar mismatch")
        responses_path = root / segment.get("responses", "")
        try: responses = json.loads(responses_path.read_text())
        except Exception as error: raise InvalidArtifact(f"response sidecar unreadable: {error}") from error
        checks = {"warmup_correct": bool(response_text(responses.get("warmup", {})).strip()),
            "paris_correct": "paris" in response_text(responses.get("paris", {})).lower(),
            "arithmetic_correct": bool(re.search(r"(^|\D)4(\D|$)", response_text(responses.get("arithmetic", {}))))}
        for key, value in checks.items():
            if segment.get("validity", {}).get(key) != value:
                raise InvalidArtifact("response correctness gate mismatch")
        try:
            peaks[segment["name"]] = validate_series(segment)
        except InvalidArtifact:
            if not failed or segment.get("validity", {}).get("series_complete", True):
                raise
    if failed:
        if artifact["decision"] != "NOT_PROVEN" or len(segments) == 4 and all(all(s.get("validity", {}).values()) for s in segments):
            raise InvalidArtifact("failure artifact does not prove an incomplete session")
        if artifact["derived"] or artifact["gates"]:
            raise InvalidArtifact("partial failure artifact must not claim derived gates")
        return artifact
    if set(binaries) != {"baseline", "instrumented"} or names != list(ORDER):
        raise InvalidArtifact("complete artifact lacks binaries or segments")
    if not all(all(segment.get("validity", {}).values()) for segment in segments):
        raise InvalidArtifact("complete segment validity mismatch")
    derived, numerical = derive(peaks)
    gates = {**{f"segment_{s['name']}_{k}": v for s in segments for k, v in s["validity"].items()}, **numerical}
    if artifact["derived"] != derived or artifact["gates"] != gates:
        raise InvalidArtifact("derived values or gates were not independently reproducible")
    expected = "PROVEN" if all(gates.values()) else "NOT_PROVEN"
    if artifact["decision"] != expected:
        raise InvalidArtifact("decision mismatch")
    return artifact


def self_test() -> None:
    peaks = {"B1": 10 * GIB, "I1": 10 * GIB + 64 * MIB,
             "B2": 10 * GIB, "I2": 10 * GIB - 64 * MIB}
    _, gates = derive(peaks)
    assert all(gates.values())
    boundary = dict(peaks); boundary["I1"] = boundary["B1"] + 256 * MIB
    _, gates = derive(boundary); assert gates["pair_1_absolute"]
    outside = dict(boundary); outside["I1"] += 1
    _, gates = derive(outside); assert not gates["pair_1_absolute"]
    try: derive({"I1": 1, "B1": 1, "B2": 1, "I2": 1})
    except InvalidArtifact: pass
    else: raise AssertionError("bad lifecycle order accepted")
    sample = {"name": "B1", "rss_series": [{"monotonic_ns": 1, "rss_bytes": 1}],
              "lifecycle": {"spawn_sampled": True, "post_response_sampled": False,
                            "spawn_sample_deadline_ns": 1, "final_response_ns": 0}}
    try: validate_series(sample)
    except InvalidArtifact: pass
    else: raise AssertionError("bad lifecycle markers accepted")
    assert not has_post_response_sample([], 1)
    assert not has_post_response_sample([{"monotonic_ns": POST_RESPONSE_NS, "rss_bytes": 1}], 1)
    assert has_post_response_sample([{"monotonic_ns": POST_RESPONSE_NS + 1, "rss_bytes": 1}], 1)
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        binaries: dict[str, Any] = {}
        for role, data in (("baseline", b"baseline"), ("instrumented", b"instrumented")):
            binary = root / role; binary.write_bytes(data)
            digest = sha256_file(binary)
            binaries[role] = {"build_command": BUILD_COMMANDS[role], "probe_compiled": role == "instrumented",
                "frozen_path": str(binary), "sha256": digest, "size": len(data),
                "sha256_before": digest, "size_before": len(data),
                "sha256_after": digest, "size_after": len(data)}
        segments = []
        provenance = {key: "fixture" for key in ("workspace", "git_head", "dirty_diff_sha256", "git_status",
            "zig_version", "sdk", "platform", "model_identity", "tokenizer_identity",
            "packed_experts_identity", "vm_stat_before", "memory_pressure_before")}
        provenance.update({"model_path": "/model", "packed_experts_path": "/experts", "port": 8930, "provenance_file": "provenance.json"})
        (root / "provenance.json").write_text(json.dumps(provenance))
        for index, name in enumerate(ORDER):
            base = index * 2_000_000_000
            series = [{"monotonic_ns": base + offset, "rss_bytes": peaks[name]}
                      for offset in (0, 500_000_000, 1_100_000_000)]
            validity = {"warmup_correct": True, "paris_correct": True, "arithmetic_correct": True}
            role = "baseline" if name.startswith("B") else "instrumented"
            rss_name, response_name = f"{name}.rss.jsonl", f"{name}.responses.json"
            (root / rss_name).write_text("".join(json.dumps(value) + "\n" for value in series))
            (root / response_name).write_text(json.dumps({"warmup": {"choices": [{"message": {"content": "hello"}}]},
                "paris": {"choices": [{"message": {"content": "Paris"}}]},
                "arithmetic": {"choices": [{"message": {"content": "4"}}]}}))
            segments.append({"name": name, "binary_role": role,
                "argv": [binaries[role]["frozen_path"], "serve", "--model", "/model", "--expert-packed-dir", "/experts",
                         "--native", "--port", "8930", "--max-tokens", "30", "--temperature", "0"],
                "workload": WORKLOAD,
                "rss_jsonl": rss_name, "responses": response_name, "rss_series": series,
                "lifecycle": {"spawn_sampled": True, "post_response_sampled": True,
                "spawn_sample_deadline_ns": base, "final_response_ns": base + 100_000_000}, "validity": validity})
        derived, numerical = derive(peaks)
        gates = {**{f"segment_{s['name']}_{key}": value for s in segments for key, value in s["validity"].items()}, **numerical}
        artifact = seal({"schema_version": SCHEMA, "policy": POLICY, "provenance": provenance,
            "binaries": binaries, "segments": segments, "derived": derived, "gates": gates,
            "decision": "PROVEN"})
        path = root / "artifact.json"; path.write_text(json.dumps(artifact))
        validate_artifact(path)
        injected = json.loads(path.read_text())
        injected["segments"][0]["argv"].insert(-4, "--smelt-strategy")
        injected["segments"][0]["argv"].insert(-4, "preload")
        path.write_text(json.dumps(seal(injected)))
        try: validate_artifact(path)
        except InvalidArtifact: pass
        else: raise AssertionError("injected argv accepted")
        path.write_text(json.dumps(artifact))
        bad_workload = json.loads(json.dumps(artifact))
        bad_workload["segments"][0]["workload"][0]["endpoint"] = "/v1/completions"
        path.write_text(json.dumps(seal(bad_workload)))
        try: validate_artifact(path)
        except InvalidArtifact: pass
        else: raise AssertionError("wrong chat workload accepted")
        assert chat_payload(WORKLOAD[1]) == {
            "model": "default", "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 30, "temperature": 0}
        try: response_text({"choices": [{"text": "legacy"}]})
        except InvalidArtifact: pass
        else: raise AssertionError("legacy completion response accepted")
        path.write_text(json.dumps(artifact))
        bad_build = json.loads(json.dumps(artifact))
        bad_build["binaries"]["baseline"]["build_command"][-1] = "-Dgpu_timestamp_probe=true"
        path.write_text(json.dumps(seal(bad_build)))
        try: validate_artifact(path)
        except InvalidArtifact: pass
        else: raise AssertionError("wrong baseline build command accepted")
        bad_role = json.loads(json.dumps(artifact))
        bad_role["binaries"]["baseline"]["probe_compiled"] = True
        path.write_text(json.dumps(seal(bad_role)))
        try: validate_artifact(path)
        except InvalidArtifact: pass
        else: raise AssertionError("wrong probe role accepted")
        unavailable = json.loads(json.dumps(artifact))
        unavailable["provenance"]["zig_version"] = "UNAVAILABLE: fixture"
        (root / "provenance.json").write_text(json.dumps(unavailable["provenance"]))
        path.write_text(json.dumps(seal(unavailable)))
        try: validate_artifact(path)
        except InvalidArtifact: pass
        else: raise AssertionError("unavailable identity accepted")
        (root / "provenance.json").write_text(json.dumps(provenance))
        path.write_text(json.dumps(artifact))
        artifact["derived"]["peaks_bytes"]["B1"] += 1
        path.write_text(json.dumps(artifact))
        try: validate_artifact(path)
        except InvalidArtifact: pass
        else: raise AssertionError("tampered artifact accepted")
        failed_provenance = dict(provenance); failed_provenance["failure"] = "fixture build failure"
        (root / "provenance.json").write_text(json.dumps(failed_provenance))
        failed = seal({"schema_version": SCHEMA, "policy": POLICY, "provenance": failed_provenance,
            "binaries": {}, "segments": [], "derived": {}, "gates": {}, "decision": "NOT_PROVEN"})
        path.write_text(json.dumps(failed)); validate_artifact(path)
    print("paired RSS harness self-test: PASS")


def run(command: list[str], cwd: pathlib.Path, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, check=True, capture_output=True, **kwargs)


def command_output(command: list[str], cwd: pathlib.Path) -> str:
    try: return run(command, cwd).stdout.strip()
    except Exception as error: return f"UNAVAILABLE: {error}"


def process_executable(pid: int) -> str:
    return command_output(["ps", "-p", str(pid), "-o", "comm="], pathlib.Path.cwd())


def no_server(workspace: pathlib.Path) -> bool:
    result = subprocess.run(["pgrep", "-f", "dmlx serve"], cwd=workspace,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode != 0


def cleanup(workspace: pathlib.Path) -> bool:
    subprocess.run(["pkill", "-f", "dmlx serve"], cwd=workspace,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    return no_server(workspace)


def rss_bytes(pid: int) -> int:
    text = subprocess.check_output(["ps", "-p", str(pid), "-o", "rss="], text=True).strip()
    return int(text) * 1024


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} for {request.method} {url}: {body}") from error


def response_text(response: Any) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str): raise TypeError("chat content is not a string")
        return content
    except Exception as error:
        raise InvalidArtifact(f"invalid chat completion response: {error}") from error


def chat_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {"model": case["model"],
            "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": case["max_tokens"], "temperature": case["temperature"]}


def collect_segment(name: str, role: str, binary: pathlib.Path, workspace: pathlib.Path,
                    out: pathlib.Path, model: pathlib.Path, experts: pathlib.Path,
                    port: int) -> dict[str, Any]:
    if not cleanup(workspace): raise RuntimeError("residual dmlx serve before segment")
    for candidate in workspace.glob("**/*routing*stats*"):
        if candidate.is_file(): candidate.unlink()
    env = os.environ.copy(); env.update(EXPERIMENT_ENV)
    for key in ("DSV4_DUMP_DIR", "MF_STAGE_TIME", "METAL_FULL", "METAL_MOE"):
        env.pop(key, None)
    argv = [str(binary), "serve", "--model", str(model), "--expert-packed-dir", str(experts),
            "--native", "--port", str(port), "--max-tokens", "30", "--temperature", "0"]
    log_path = out / f"{name}.log"; responses_path = out / f"{name}.responses.json"
    log = log_path.open("wb")
    process = subprocess.Popen(argv, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT)
    spawned_ns = time.monotonic_ns(); series: list[dict[str, int]] = []
    stop = threading.Event(); sampler_error: list[str] = []
    def sampler() -> None:
        deadline = time.monotonic_ns()
        while not stop.is_set():
            now = time.monotonic_ns()
            try: series.append({"monotonic_ns": now, "rss_bytes": rss_bytes(process.pid)})
            except Exception as error:
                if process.poll() is None: sampler_error.append(str(error))
                break
            deadline += PERIOD_NS
            stop.wait(max(0.0, (deadline - time.monotonic_ns()) / 1e9))
    thread = threading.Thread(target=sampler, daemon=True); thread.start()
    validity = {key: False for key in ("clean_start", "direct_pid", "executable_before", "readiness",
        "warmup_correct", "paris_correct", "arithmetic_correct", "executable_after", "series_complete", "cleanup")}
    validity["clean_start"] = True
    responses: dict[str, Any] = {}
    try:
        expected = str(binary.resolve()); actual = process_executable(process.pid)
        validity["direct_pid"] = process.pid > 0 and argv[0] == str(binary)
        validity["executable_before"] = actual == expected
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if process.poll() is not None: raise RuntimeError("server exited before readiness")
            try:
                request_json(f"http://127.0.0.1:{port}/health", timeout=5); validity["readiness"] = True; break
            except Exception: time.sleep(1)
        if not validity["readiness"]: raise RuntimeError("health readiness timeout")
        time.sleep(5)
        for case in WORKLOAD:
            label = case["label"]
            try:
                response = request_json(f"http://127.0.0.1:{port}{case['endpoint']}",
                    chat_payload(case), 600)
            except Exception as error:
                raise RuntimeError(f"{label} request failed at {case['endpoint']}: {error}") from error
            responses[label] = response; text = response_text(response)
            if label == "warmup": validity["warmup_correct"] = bool(text.strip())
            elif label == "paris": validity["paris_correct"] = "paris" in text.lower()
            else: validity["arithmetic_correct"] = bool(re.search(r"(^|\D)4(\D|$)", text))
            if not validity[f"{label}_correct"]: raise RuntimeError(f"{label} correctness failed")
        final_ns = time.monotonic_ns()
        post_response_deadline_ns = final_ns + POST_RESPONSE_NS + MAX_GAP_NS
        while (not has_post_response_sample(series, final_ns) and
               time.monotonic_ns() < post_response_deadline_ns and
               process.poll() is None and thread.is_alive()):
            time.sleep(0.02)
        validity["executable_after"] = process_executable(process.pid) == expected
    finally:
        final_ns = locals().get("final_ns", time.monotonic_ns())
        stop.set(); thread.join(2)
        if process.poll() is None: process.send_signal(signal.SIGTERM)
        try: process.wait(10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close(); responses_path.write_text(json.dumps(responses, indent=2, sort_keys=True) + "\n")
        (out / f"{name}.rss.jsonl").write_text("".join(
            json.dumps(sample, sort_keys=True) + "\n" for sample in series))
        validity["cleanup"] = cleanup(workspace)
    lifecycle = {"spawned_ns": spawned_ns, "spawn_sample_deadline_ns": spawned_ns + MAX_GAP_NS,
                 "final_response_ns": final_ns, "spawn_sampled": bool(series),
                 "post_response_sampled": has_post_response_sample(series, final_ns)}
    try:
        validate_series({"rss_series": series, "lifecycle": lifecycle})
        validity["series_complete"] = not sampler_error
    except InvalidArtifact: validity["series_complete"] = False
    return {"name": name, "binary_role": role, "argv": argv, "pid": process.pid,
            "resolved_executable": str(binary.resolve()), "log": log_path.name,
            "workload": WORKLOAD, "responses": responses_path.name,
            "rss_jsonl": f"{name}.rss.jsonl", "lifecycle": lifecycle,
            "rss_series": series, "validity": validity}


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--validate-artifact", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path.home()/"models/DeepSeek-V4-Flash-4bit")
    parser.add_argument("--expert-packed-dir", type=pathlib.Path)
    parser.add_argument("--port", type=int, default=8930)
    args = parser.parse_args()
    if args.self_test: self_test(); return 0
    if args.validate_artifact:
        artifact = validate_artifact(args.validate_artifact); print(artifact["decision"]); return 0
    workspace = pathlib.Path(__file__).resolve().parents[1]
    out = args.output or workspace/"artifacts"/f"g1_3_rss_{time.strftime('%Y%m%dT%H%M%S')}"
    if out.exists(): raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    model = args.model.resolve(); experts = (args.expert_packed_dir or model/"packed_experts").resolve()
    artifact: dict[str, Any] = {"schema_version": SCHEMA, "policy": POLICY,
        "provenance": {}, "binaries": {}, "segments": [], "derived": {}, "gates": {}, "decision": "NOT_PROVEN"}
    try:
        dirty_sha256 = dirty_workspace_sha256(workspace)
        artifact["provenance"] = {"workspace": str(workspace), "git_head": command_output(["git", "rev-parse", "HEAD"], workspace),
            "dirty_diff_sha256": dirty_sha256, "git_status": command_output(["git", "status", "--porcelain=v1"], workspace),
            "zig_version": command_output(["zig", "version"], workspace), "sdk": command_output(["xcrun", "--show-sdk-version"], workspace),
            "platform": platform.platform(), "model_path": str(model), "model_identity": command_output(["stat", "-f", "%N:%z:%m", str(model/"config.json")], workspace),
            "tokenizer_identity": command_output(["stat", "-f", "%N:%z:%m", str(model/"tokenizer.json")], workspace),
            "packed_experts_path": str(experts), "packed_experts_identity": command_output(["stat", "-f", "%N:%z:%m", str(experts)], workspace),
            "vm_stat_before": command_output(["vm_stat"], workspace), "memory_pressure_before": command_output(["memory_pressure"], workspace),
            "port": args.port, "provenance_file": "provenance.json"}
        if not model.is_dir() or not experts.is_dir(): raise RuntimeError("model/packed expert path missing")
        for role, command in BUILD_COMMANDS.items():
            run(command, workspace)
            source = workspace/"zig-out/bin/dmlx"; frozen = out/f"dmlx-{role}"
            shutil.copy2(source, frozen); frozen.chmod(0o555)
            artifact["binaries"][role] = {"build_command": command, "probe_compiled": role == "instrumented", "frozen_path": str(frozen.resolve()),
                "sha256": sha256_file(frozen), "size": frozen.stat().st_size,
                "sha256_before": sha256_file(frozen), "size_before": frozen.stat().st_size,
                "sha256_after": "", "size_after": 0}
        if artifact["binaries"]["baseline"]["sha256"] == artifact["binaries"]["instrumented"]["sha256"]: raise RuntimeError("binary hashes unexpectedly equal")
        for name in ORDER:
            role = "baseline" if name.startswith("B") else "instrumented"
            segment = collect_segment(name, role, pathlib.Path(artifact["binaries"][role]["frozen_path"]), workspace, out, model, experts, args.port)
            artifact["segments"].append(segment)
            if not all(segment["validity"].values()): raise RuntimeError(f"{name} invalid; whole session cannot resume")
        for binary in artifact["binaries"].values():
            frozen = pathlib.Path(binary["frozen_path"]); binary["sha256_after"] = sha256_file(frozen); binary["size_after"] = frozen.stat().st_size
        peaks = {s["name"]: max(x["rss_bytes"] for x in s["rss_series"]) for s in artifact["segments"]}
        artifact["derived"], numerical = derive(peaks)
        artifact["gates"] = {**{f"segment_{s['name']}_{k}": v for s in artifact["segments"] for k, v in s["validity"].items()}, **numerical}
        artifact["decision"] = "PROVEN" if all(artifact["gates"].values()) else "NOT_PROVEN"
        artifact["provenance"].update({"vm_stat_after": command_output(["vm_stat"], workspace), "memory_pressure_after": command_output(["memory_pressure"], workspace)})
    except Exception as error:
        artifact["provenance"]["failure"] = str(error)
        artifact["decision"] = "NOT_PROVEN"
    finally:
        cleanup(workspace)
        for binary in artifact.get("binaries", {}).values():
            frozen = pathlib.Path(binary["frozen_path"])
            if frozen.exists(): binary["sha256_after"], binary["size_after"] = sha256_file(frozen), frozen.stat().st_size
        (out / "provenance.json").write_text(json.dumps(artifact["provenance"], indent=2, sort_keys=True) + "\n")
        (out/"artifact.json").write_text(json.dumps(seal(artifact), indent=2, sort_keys=True) + "\n")
    if artifact["decision"] != "PROVEN": print(f"NOT_PROVEN: {out/'artifact.json'}", file=sys.stderr); return 1
    validate_artifact(out/"artifact.json"); print(f"PROVEN: {out/'artifact.json'}"); return 0

if __name__ == "__main__":
    raise SystemExit(main())
