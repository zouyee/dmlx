#!/usr/bin/env python3
"""Validate the real-model scalar/chunk oracle and its frozen logits fixture."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import tempfile
from typing import Any

SCHEMA_VERSION = 1
EXPECTED_INPUT_IDS = [17, 42, 314, 2718, 7, 99, 1234]
EXPECTED_PARTITIONS = [
    ("scalar-sized", [1, 1, 1, 1, 1, 1, 1]),
    ("batch-2-tail", [2, 2, 2, 1]),
    ("odd-3-tail", [3, 3, 1]),
    ("max-5-tail", [5, 2]),
]
TOLERANCE_SOURCE = "batch_oracle.c:F32/final_logits"
GENERATION_METHOD = "production embed/forward/hyper-head/get-logits before sampling"
REFERENCE_PATH = "scalar"
FIXTURE_COMMAND = (
    "NATIVE_SMELT_N=20 zig build run-batch-oracle-model "
    "-Doptimize=ReleaseFast -- --output <result.json>"
)
FINGERPRINT_METHOD = (
    "sha256(config content + model index content + indexed model artifact contents + "
    "sorted packed artifact contents)"
)


class OracleError(RuntimeError):
    pass


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleError(f"cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OracleError(f"JSON root must be an object: {path}")
    return value


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise OracleError(f"{context} missing required field {key!r}")
    return mapping[key]


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OracleError(f"{context} must be a non-empty string")
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleError(f"{context} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise OracleError(f"{context} must be finite, got {value!r}")
    return converted


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise OracleError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _artifact_path(root: pathlib.Path, relative: str, context: str) -> pathlib.Path:
    if not relative or pathlib.PurePosixPath(relative).is_absolute():
        raise OracleError(f"{context} contains an invalid artifact path {relative!r}")
    root_resolved = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise OracleError(f"{context} artifact escapes model root: {relative!r}") from exc
    if not path.is_file():
        raise OracleError(f"{context} artifact is missing: {path}")
    return path


def compute_fingerprints(model_path: pathlib.Path, packed_dir: pathlib.Path) -> dict[str, str]:
    """Hash the content of every model and packed artifact used by the oracle."""
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    single_model_path = model_path / "model.safetensors"
    tokenizer_path = model_path / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise OracleError(f"model is missing config.json or tokenizer.json: {model_path}")
    if not packed_dir.is_dir():
        raise OracleError(f"packed experts directory does not exist: {packed_dir}")

    identity_lines = [f"config.json\0sha256={_sha256_file(config_path)}\n"]
    if index_path.is_file():
        index = _load_json(index_path)
        weight_map = _require(index, "weight_map", "model index")
        if not isinstance(weight_map, dict) or not weight_map:
            raise OracleError("model index weight_map must be a non-empty object")
        artifact_names: set[str] = set()
        for tensor_name, artifact_name in weight_map.items():
            _nonempty_string(tensor_name, "model index tensor name")
            artifact_names.add(
                _nonempty_string(artifact_name, f"model index weight_map[{tensor_name!r}]")
            )
        identity_lines.append(
            f"model.safetensors.index.json\0sha256={_sha256_file(index_path)}\n"
        )
        for artifact_name in sorted(artifact_names):
            artifact = _artifact_path(model_path, artifact_name, "model index")
            identity_lines.append(
                f"model/{artifact_name}\0sha256={_sha256_file(artifact)}\n"
            )
    elif single_model_path.is_file():
        identity_lines.append(
            f"model.safetensors\0sha256={_sha256_file(single_model_path)}\n"
        )
    else:
        raise OracleError("model has neither model.safetensors.index.json nor model.safetensors")

    packed_files = sorted(
        path
        for path in packed_dir.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".") for part in path.relative_to(packed_dir).parts
        )
    )
    if not packed_files:
        raise OracleError(f"packed experts manifest is empty: {packed_dir}")
    for path in packed_files:
        relative = path.relative_to(packed_dir).as_posix()
        identity_lines.append(
            f"packed_experts/{relative}\0sha256={_sha256_file(path)}\n"
        )

    digest = hashlib.sha256("".join(identity_lines).encode("utf-8")).hexdigest()
    return {
        "model_manifest_sha256": digest,
        "tokenizer_sha256": _sha256_file(tokenizer_path),
        "fingerprint_method": FINGERPRINT_METHOD,
    }


def _validate_tolerance(value: Any, context: str) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise OracleError(f"{context} must be an object")
    if _require(value, "source", context) != TOLERANCE_SOURCE:
        raise OracleError(f"{context}.source must be {TOLERANCE_SOURCE!r}")
    absolute = _finite_number(_require(value, "absolute", context), f"{context}.absolute")
    relative = _finite_number(_require(value, "relative", context), f"{context}.relative")
    if absolute <= 0.0 or relative <= 0.0:
        raise OracleError(f"{context} values must be positive")
    return absolute, relative


def _validate_result_generation(value: Any) -> None:
    if not isinstance(value, dict):
        raise OracleError("result.generation must be an object")
    method = _nonempty_string(
        _require(value, "method", "result.generation"), "result.generation.method"
    )
    if method != GENERATION_METHOD:
        raise OracleError(f"result.generation.method must be {GENERATION_METHOD!r}")
    max_generated = _require(value, "max_generated_tokens", "result.generation")
    if (
        not isinstance(max_generated, int)
        or isinstance(max_generated, bool)
        or max_generated != 0
    ):
        raise OracleError("result generation must stop before sampling (max_generated_tokens=0)")


def _validate_fixture_generation(value: Any) -> None:
    if not isinstance(value, dict):
        raise OracleError("fixture.generation must be an object")
    generated_at = _nonempty_string(
        _require(value, "generated_at_utc", "fixture.generation"),
        "fixture.generation.generated_at_utc",
    )
    try:
        parsed = dt.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OracleError("fixture.generation.generated_at_utc must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise OracleError("fixture.generation.generated_at_utc must include a timezone")
    if _require(value, "method", "fixture.generation") != GENERATION_METHOD:
        raise OracleError(f"fixture.generation.method must be {GENERATION_METHOD!r}")
    if _require(value, "reference_path", "fixture.generation") != REFERENCE_PATH:
        raise OracleError("fixture.generation.reference_path must be 'scalar'")
    _nonempty_string(
        _require(value, "regeneration_reason", "fixture.generation"),
        "fixture.generation.regeneration_reason",
    )
    if _require(value, "command", "fixture.generation") != FIXTURE_COMMAND:
        raise OracleError("fixture.generation.command does not match the supported command")


def validate_result(result: dict[str, Any]) -> tuple[list[float], tuple[float, float]]:
    if _require(result, "schema_version", "result") != SCHEMA_VERSION:
        raise OracleError("result schema_version mismatch")
    if _require(result, "runner", "result") != "dmlx-native-batch-oracle":
        raise OracleError("unexpected result runner")
    _validate_result_generation(_require(result, "generation", "result"))
    if _require(result, "input_ids", "result") != EXPECTED_INPUT_IDS:
        raise OracleError("result input_ids do not match the frozen oracle input")
    terminal_position = _require(result, "final_token_position", "result")
    if terminal_position != len(EXPECTED_INPUT_IDS) - 1:
        raise OracleError("result final_token_position does not identify the terminal input token")

    vocab_size = _require(result, "vocab_size", "result")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise OracleError("result vocab_size must be a positive integer")
    tolerance = _validate_tolerance(
        _require(result, "central_tolerance", "result"), "result.central_tolerance"
    )

    reference = _require(result, "reference", "result")
    if (
        not isinstance(reference, dict)
        or _require(reference, "path", "result.reference") != REFERENCE_PATH
    ):
        raise OracleError("result reference must be the scalar path")
    raw_logits = _require(reference, "final_logits", "result.reference")
    if not isinstance(raw_logits, list) or len(raw_logits) != vocab_size:
        raise OracleError(
            f"result final_logits length mismatch: expected {vocab_size}, got "
            f"{len(raw_logits) if isinstance(raw_logits, list) else 'non-list'}"
        )
    logits = [
        _finite_number(value, f"result.reference.final_logits[{index}]")
        for index, value in enumerate(raw_logits)
    ]
    expected_argmax = max(range(vocab_size), key=logits.__getitem__)
    if _require(reference, "argmax", "result.reference") != expected_argmax:
        raise OracleError("result scalar argmax does not match final_logits")

    partitions = _require(result, "partitions", "result")
    if not isinstance(partitions, list) or len(partitions) != len(EXPECTED_PARTITIONS):
        raise OracleError("result partition count mismatch")
    for index, ((expected_name, expected_chunks), partition) in enumerate(
        zip(EXPECTED_PARTITIONS, partitions, strict=True)
    ):
        context = f"result.partitions[{index}]"
        if not isinstance(partition, dict):
            raise OracleError(f"{context} must be an object")
        if partition.get("name") != expected_name or partition.get("chunks") != expected_chunks:
            raise OracleError(
                f"partition mismatch at {index}: expected {expected_name} {expected_chunks}, "
                f"got {partition.get('name')} {partition.get('chunks')}"
            )
        if partition.get("status") != "ok":
            raise OracleError(f"{context} did not pass centralized C comparison")
        if partition.get("argmax") != expected_argmax:
            raise OracleError(f"{context} argmax differs from scalar reference")
        if partition.get("final_token_position") != terminal_position:
            raise OracleError(f"{context} final_token_position mismatch")
        _finite_number(partition.get("max_absolute_error"), f"{context}.max_absolute_error")
        _finite_number(partition.get("max_relative_error"), f"{context}.max_relative_error")
    return logits, tolerance


def _fixture_from_result(
    result: dict[str, Any], fingerprints: dict[str, str], reason: str
) -> dict[str, Any]:
    logits, _ = validate_result(result)
    _nonempty_string(reason, "regeneration reason")
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_kind": "dsv4-native-final-logits",
        "model": fingerprints,
        "generation": {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": result["generation"]["method"],
            "reference_path": REFERENCE_PATH,
            "regeneration_reason": reason,
            "command": FIXTURE_COMMAND,
        },
        "input_ids": result["input_ids"],
        "final_token_position": result["final_token_position"],
        "vocab_size": result["vocab_size"],
        "central_tolerance": result["central_tolerance"],
        "partition_suite": [chunks for _, chunks in EXPECTED_PARTITIONS],
        "reference": {
            "argmax": result["reference"]["argmax"],
            "final_logits": logits,
        },
    }


def _compare_logits(
    expected: list[float], actual: list[float], absolute: float, relative: float
) -> tuple[float, float]:
    if len(expected) != len(actual):
        raise OracleError(
            f"fixture logits length mismatch: fixture={len(expected)} result={len(actual)}"
        )
    first_failure: tuple[int, float, float, float, float] | None = None
    max_absolute = 0.0
    max_relative = 0.0
    for index, (reference, candidate) in enumerate(zip(expected, actual, strict=True)):
        if not math.isfinite(reference) or not math.isfinite(candidate):
            raise OracleError(f"nonfinite golden comparison at logit index {index}")
        absolute_error = abs(reference - candidate)
        scale = max(abs(reference), abs(candidate))
        relative_error = absolute_error / scale if scale > 0.0 else 0.0
        max_absolute = max(max_absolute, absolute_error)
        max_relative = max(max_relative, relative_error)
        if absolute_error > absolute + relative * scale and first_failure is None:
            first_failure = (index, reference, candidate, absolute_error, relative_error)
    if first_failure is not None:
        index, reference, candidate, abs_error, rel_error = first_failure
        raise OracleError(
            "fixture logits mismatch: "
            f"token={len(EXPECTED_INPUT_IDS) - 1} first_index={index} "
            f"reference={reference:.9g} actual={candidate:.9g} "
            f"abs={abs_error:.9g} rel={rel_error:.9g} "
            f"max_abs={max_absolute:.9g} max_rel={max_relative:.9g} "
            f"atol={absolute:.9g} rtol={relative:.9g}"
        )
    return max_absolute, max_relative


def validate_fixture(
    fixture: dict[str, Any],
    result: dict[str, Any],
    fingerprints: dict[str, str],
) -> tuple[float, float]:
    result_logits, tolerance = validate_result(result)
    if _require(fixture, "schema_version", "fixture") != SCHEMA_VERSION:
        raise OracleError("fixture schema_version mismatch")
    if _require(fixture, "fixture_kind", "fixture") != "dsv4-native-final-logits":
        raise OracleError("unexpected fixture_kind")
    _validate_fixture_generation(_require(fixture, "generation", "fixture"))
    fixture_model = _require(fixture, "model", "fixture")
    if not isinstance(fixture_model, dict):
        raise OracleError("fixture.model must be an object")
    for field in ("model_manifest_sha256", "tokenizer_sha256", "fingerprint_method"):
        expected = fingerprints[field]
        actual = fixture_model.get(field)
        if actual != expected:
            raise OracleError(
                f"fingerprint mismatch for {field}: fixture={actual!r} current={expected!r}"
            )
    if fixture.get("input_ids") != result["input_ids"]:
        raise OracleError("fixture input_ids mismatch")
    if fixture.get("final_token_position") != result["final_token_position"]:
        raise OracleError("fixture final_token_position mismatch")
    if fixture.get("vocab_size") != result["vocab_size"]:
        raise OracleError("fixture vocab_size mismatch")
    if fixture.get("partition_suite") != [chunks for _, chunks in EXPECTED_PARTITIONS]:
        raise OracleError("fixture partition suite mismatch")
    fixture_tolerance = _validate_tolerance(
        _require(fixture, "central_tolerance", "fixture"), "fixture.central_tolerance"
    )
    if fixture_tolerance != tolerance:
        raise OracleError(
            f"central tolerance changed: fixture={fixture_tolerance} result={tolerance}; "
            "review and explicitly regenerate rather than silently accepting"
        )
    reference = _require(fixture, "reference", "fixture")
    if not isinstance(reference, dict):
        raise OracleError("fixture.reference must be an object")
    fixture_logits_raw = _require(reference, "final_logits", "fixture.reference")
    if not isinstance(fixture_logits_raw, list):
        raise OracleError("fixture.reference.final_logits must be a list")
    fixture_logits = [
        _finite_number(value, f"fixture.reference.final_logits[{index}]")
        for index, value in enumerate(fixture_logits_raw)
    ]
    if reference.get("argmax") != result["reference"]["argmax"]:
        raise OracleError(
            f"fixture argmax mismatch: fixture={reference.get('argmax')} "
            f"result={result['reference']['argmax']}"
        )
    return _compare_logits(fixture_logits, result_logits, *tolerance)


def _atomic_write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _sample_result() -> dict[str, Any]:
    partitions = []
    for name, chunks in EXPECTED_PARTITIONS:
        partitions.append(
            {
                "name": name,
                "chunks": chunks,
                "argmax": 2,
                "status": "ok",
                "final_token_position": len(EXPECTED_INPUT_IDS) - 1,
                "max_absolute_error": 0.0,
                "max_relative_error": 0.0,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "runner": "dmlx-native-batch-oracle",
        "generation": {"method": GENERATION_METHOD, "max_generated_tokens": 0},
        "input_ids": EXPECTED_INPUT_IDS,
        "final_token_position": len(EXPECTED_INPUT_IDS) - 1,
        "vocab_size": 3,
        "central_tolerance": {
            "source": TOLERANCE_SOURCE,
            "absolute": 0.01,
            "relative": 0.01,
        },
        "reference": {"path": REFERENCE_PATH, "argmax": 2, "final_logits": [0.0, 1.0, 2.0]},
        "partitions": partitions,
    }


def _expect_oracle_error(action: Any, message_fragment: str, description: str) -> None:
    try:
        action()
    except OracleError as exc:
        if message_fragment not in str(exc):
            raise
    else:
        raise AssertionError(f"{description} unexpectedly passed")


def _test_content_fingerprint() -> None:
    with tempfile.TemporaryDirectory(prefix="dmlx-oracle-fingerprint-") as directory:
        root = pathlib.Path(directory)
        model = root / "model"
        packed = model / "packed_experts"
        packed.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "tokenizer.json").write_text("{}", encoding="utf-8")
        (model / "model-00001-of-00001.safetensors").write_bytes(b"model-a")
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"tensor": "model-00001-of-00001.safetensors"}}),
            encoding="utf-8",
        )
        packed_artifact = packed / "layer_00.bin"
        packed_artifact.write_bytes(b"packed-a")

        original = compute_fingerprints(model, packed)["model_manifest_sha256"]
        packed_artifact.write_bytes(b"packed-b")
        packed_changed = compute_fingerprints(model, packed)["model_manifest_sha256"]
        if packed_changed == original:
            raise AssertionError("same-size packed content change did not alter fingerprint")

        packed_artifact.write_bytes(b"packed-a")
        (model / "model-00001-of-00001.safetensors").write_bytes(b"model-b")
        model_changed = compute_fingerprints(model, packed)["model_manifest_sha256"]
        if model_changed == original:
            raise AssertionError("same-size model content change did not alter fingerprint")


def run_self_test() -> None:
    fingerprints = {
        "model_manifest_sha256": "model-a",
        "tokenizer_sha256": "tokenizer-a",
        "fingerprint_method": FINGERPRINT_METHOD,
    }
    result = _sample_result()
    fixture = _fixture_from_result(result, fingerprints, "self-test")
    validate_fixture(fixture, result, fingerprints)

    corrupt = copy.deepcopy(fixture)
    del corrupt["reference"]
    _expect_oracle_error(
        lambda: validate_fixture(corrupt, result, fingerprints),
        "missing required field",
        "corrupt fixture",
    )

    mismatch = copy.deepcopy(fixture)
    mismatch["model"]["tokenizer_sha256"] = "tokenizer-b"
    _expect_oracle_error(
        lambda: validate_fixture(mismatch, result, fingerprints),
        "fingerprint mismatch",
        "fingerprint mismatch",
    )

    bad_partition = copy.deepcopy(result)
    bad_partition["partitions"][2]["chunks"] = [3, 2, 2]
    _expect_oracle_error(
        lambda: validate_fixture(fixture, bad_partition, fingerprints),
        "partition mismatch",
        "partition mismatch",
    )

    nonfinite = copy.deepcopy(fixture)
    nonfinite["reference"]["final_logits"][1] = float("nan")
    _expect_oracle_error(
        lambda: validate_fixture(nonfinite, result, fingerprints),
        "finite",
        "nonfinite fixture",
    )

    result_without_method = copy.deepcopy(result)
    del result_without_method["generation"]["method"]
    _expect_oracle_error(
        lambda: validate_result(result_without_method),
        "missing required field",
        "result without generation method",
    )

    fixture_without_generation = copy.deepcopy(fixture)
    del fixture_without_generation["generation"]
    _expect_oracle_error(
        lambda: validate_fixture(fixture_without_generation, result, fingerprints),
        "missing required field",
        "fixture without generation provenance",
    )

    fixture_without_reason = copy.deepcopy(fixture)
    del fixture_without_reason["generation"]["regeneration_reason"]
    _expect_oracle_error(
        lambda: validate_fixture(fixture_without_reason, result, fingerprints),
        "missing required field",
        "fixture without regeneration reason",
    )

    _test_content_fingerprint()
    print("batch oracle comparator self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=pathlib.Path)
    parser.add_argument("--fixture", type=pathlib.Path)
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--packed-dir", type=pathlib.Path)
    parser.add_argument("--regenerate-fixture", action="store_true")
    parser.add_argument("--regeneration-reason")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if not all((args.result, args.fixture, args.model, args.packed_dir)):
        parser.error("--result, --fixture, --model, and --packed-dir are required")
    if args.regenerate_fixture and not args.regeneration_reason:
        parser.error("--regenerate-fixture requires --regeneration-reason")
    if args.regeneration_reason and not args.regenerate_fixture:
        parser.error("--regeneration-reason is valid only with --regenerate-fixture")

    result = _load_json(args.result)
    fingerprints = compute_fingerprints(args.model, args.packed_dir)
    validate_result(result)
    print(f"model_manifest_sha256={fingerprints['model_manifest_sha256']}")
    print(f"tokenizer_sha256={fingerprints['tokenizer_sha256']}")
    for partition in result["partitions"]:
        print(
            f"partition PASS: {partition['name']} chunks={partition['chunks']} "
            f"token={partition['final_token_position']} "
            f"argmax={partition['argmax']} max_abs={partition['max_absolute_error']:.9g} "
            f"max_rel={partition['max_relative_error']:.9g}"
        )

    if args.regenerate_fixture:
        fixture = _fixture_from_result(result, fingerprints, args.regeneration_reason)
        _atomic_write_json(args.fixture, fixture)
        print(f"explicitly regenerated scalar golden fixture: {args.fixture}")
        return 0

    if not args.fixture.is_file():
        raise OracleError(
            f"fixture does not exist: {args.fixture}; ordinary validation never creates or updates it"
        )
    fixture = _load_json(args.fixture)
    max_absolute, max_relative = validate_fixture(fixture, result, fingerprints)
    print(
        "golden fixture PASS: fingerprints match; "
        f"full_logits={result['vocab_size']} max_abs={max_absolute:.9g} "
        f"max_rel={max_relative:.9g}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OracleError as exc:
        print(f"batch oracle validation FAILED: {exc}", file=os.sys.stderr)
        raise SystemExit(1)
