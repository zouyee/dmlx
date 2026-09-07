#!/usr/bin/env python3
"""Canonical raw Metal timestamp artifact schema, hashing, and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CAPACITY = 4096
KINDS = {1, 2}
REASONS = set(range(6))
SAMPLE_KEYS = {
    "sequence", "request_id", "token_index", "layer", "kind",
    "pre_commit_ns", "post_commit_ns", "post_wait_ns",
    "gpu_start_s", "gpu_end_s", "valid", "reason",
}
IDENTITY_KEYS = {
    "pid", "process_start_ns", "device_registry_id", "device_name",
    "os_version", "binary_sha256", "git_commit", "worktree_dirty",
}
TOP_KEYS = {
    "schema_version", "identity", "enabled", "capacity", "total_seen",
    "overflow_count", "samples", "content_sha256",
}


class ArtifactError(ValueError):
    pass


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    semantic = dict(payload)
    semantic.pop("content_sha256", None)
    return json.dumps(
        semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def seal(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed["content_sha256"] = content_sha256(sealed)
    return sealed


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ArtifactError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ArtifactError(f"{name} must be hexadecimal") from error
    return value


def _reason_matches(sample: dict[str, Any]) -> bool:
    pre = sample["pre_commit_ns"]
    post = sample["post_commit_ns"]
    wait = sample["post_wait_ns"]
    start = sample["gpu_start_s"]
    end = sample["gpu_end_s"]
    valid = sample["valid"]
    reason = sample["reason"]
    host_ordered = pre <= post <= wait
    finite = math.isfinite(start) and math.isfinite(end)
    if valid == 1:
        return reason == 0 and host_ordered and finite and start > 0 and end > start
    if reason == 1:
        return host_ordered and start == 0 and end == 0
    if reason == 2:
        # In-memory nonfinite observations are normalized at export to this
        # JSON-safe projection. The reason code, never the projected values,
        # preserves the distinction from selector-unavailable reason 1.
        return host_ordered and start == 0 and end == 0
    if reason == 3:
        return host_ordered and finite and (start <= 0 or end <= 0)
    if reason == 4:
        return host_ordered and finite and start > 0 and end <= start
    if reason == 5:
        return not host_ordered
    return False


def validate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != TOP_KEYS:
        raise ArtifactError("top-level fields do not match schema v1")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ArtifactError("unsupported schema_version")
    if not isinstance(payload["enabled"], bool):
        raise ArtifactError("enabled must be boolean")
    if payload["capacity"] != CAPACITY:
        raise ArtifactError("capacity must be 4096")

    identity = payload["identity"]
    if not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS:
        raise ArtifactError("identity fields do not match schema v1")
    _integer(identity["pid"], "identity.pid", 1)
    _integer(identity["process_start_ns"], "identity.process_start_ns", 1)
    _integer(identity["device_registry_id"], "identity.device_registry_id", 1)
    for field in ("device_name", "os_version", "git_commit"):
        if not isinstance(identity[field], str) or not identity[field]:
            raise ArtifactError(f"identity.{field} must be non-empty")
    _sha256(identity["binary_sha256"], "identity.binary_sha256")
    if not isinstance(identity["worktree_dirty"], bool):
        raise ArtifactError("identity.worktree_dirty must be boolean")

    total_seen = _integer(payload["total_seen"], "total_seen")
    overflow = _integer(payload["overflow_count"], "overflow_count")
    samples = payload["samples"]
    if not isinstance(samples, list) or len(samples) > CAPACITY:
        raise ArtifactError("samples must be an array within capacity")
    if total_seen != len(samples) + overflow:
        raise ArtifactError("total_seen must equal retained samples plus overflow")

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != SAMPLE_KEYS:
            raise ArtifactError(f"sample[{index}] fields do not match schema v1")
        for field in (
            "sequence", "request_id", "token_index", "layer", "kind",
            "pre_commit_ns", "post_commit_ns", "post_wait_ns", "valid", "reason",
        ):
            _integer(sample[field], f"sample[{index}].{field}")
        if sample["sequence"] != index:
            raise ArtifactError(f"sample[{index}] sequence is not contiguous")
        if sample["layer"] >= 43 or sample["kind"] not in KINDS:
            raise ArtifactError(f"sample[{index}] layer/kind invalid")
        if sample["valid"] not in (0, 1) or sample["reason"] not in REASONS:
            raise ArtifactError(f"sample[{index}] validity invalid")
        for field in ("gpu_start_s", "gpu_end_s"):
            if isinstance(sample[field], bool) or not isinstance(sample[field], (int, float)):
                raise ArtifactError(f"sample[{index}].{field} must be numeric")
        if not _reason_matches(sample):
            raise ArtifactError(f"sample[{index}] reason does not match observations")

    expected = _sha256(payload["content_sha256"], "content_sha256")
    if expected != content_sha256(payload):
        raise ArtifactError("content_sha256 mismatch")
    return payload


def fixture_payload() -> dict[str, Any]:
    return seal({
        "schema_version": 1,
        "identity": {
            "pid": 4242,
            "process_start_ns": 1000000000,
            "device_registry_id": 123456,
            "device_name": "Apple M4 Pro",
            "os_version": "macOS fixture",
            "binary_sha256": "1" * 64,
            "git_commit": "fixture-commit",
            "worktree_dirty": False,
        },
        "enabled": True,
        "capacity": CAPACITY,
        "total_seen": 3,
        "overflow_count": 0,
        "samples": [
            {
                "sequence": 0, "request_id": 7, "token_index": 0,
                "layer": 0, "kind": 1, "pre_commit_ns": 100,
                "post_commit_ns": 110, "post_wait_ns": 300,
                "gpu_start_s": 1.25, "gpu_end_s": 1.5,
                "valid": 1, "reason": 0,
            },
            {
                "sequence": 1, "request_id": 7, "token_index": 0,
                "layer": 0, "kind": 2, "pre_commit_ns": 400,
                "post_commit_ns": 410, "post_wait_ns": 700,
                "gpu_start_s": 0.0, "gpu_end_s": 0.0,
                "valid": 0, "reason": 1,
            },
            {
                "sequence": 2, "request_id": 7, "token_index": 1,
                "layer": 1, "kind": 1, "pre_commit_ns": 800,
                "post_commit_ns": 810, "post_wait_ns": 900,
                "gpu_start_s": 0.0, "gpu_end_s": 0.0,
                "valid": 0, "reason": 2,
            },
        ],
    })


def self_test() -> None:
    fixture = fixture_payload()
    validate(fixture)
    assert canonical_bytes(fixture) == canonical_bytes(dict(reversed(list(fixture.items()))))
    for mutate in (
        lambda p: p["samples"][0].__setitem__("sequence", 9),
        lambda p: p.__setitem__("overflow_count", 1),
        lambda p: p["identity"].__setitem__("device_name", ""),
        lambda p: p["samples"][1].__setitem__("reason", 0),
        lambda p: p["samples"][2].__setitem__("gpu_start_s", 1.0),
    ):
        candidate = json.loads(json.dumps(fixture))
        mutate(candidate)
        try:
            validate(candidate)
        except ArtifactError:
            pass
        else:
            raise AssertionError("tampered artifact accepted")
    print("gpu_timestamp_artifact self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-fixture", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.print_fixture:
        print(json.dumps(fixture_payload(), indent=2, sort_keys=True, ensure_ascii=False))
    if args.artifact:
        validate(json.loads(args.artifact.read_text()))
        print(f"GPU timestamp artifact valid: {args.artifact}")
    if not (args.self_test or args.print_fixture or args.artifact):
        parser.error("provide --self-test, --print-fixture, or an artifact path")


if __name__ == "__main__":
    main()
