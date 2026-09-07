#!/usr/bin/env python3
"""Strict validator for the reduced mixed2 fixture-sidecar contract.

Normal CLI and self-test validation always require the sidecar so header/file/core
bindings are checked.  Sidecar-free parsing is intentionally limited to isolated
binary-corruption categorization in ``self_test``; it is not a complete validator.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import struct
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXTURE_BIN = ROOT / "tests/fixtures/mixed_2bit_small.bin"
FIXTURE_JSON = ROOT / "tests/fixtures/mixed_2bit_small.json"
LOGITS_FIXTURE = ROOT / "tests/fixtures/dsv4_native_logits.json"
RFC_PATH = ROOT / "docs/en/analysis/mixed-2bit-rfc.md"
QUALITY_JSON = ROOT / "docs/en/analysis/mixed-2bit-quality.json"
QUALITY_SCHEMA = "dmlx.mixed2.quality-report.v1"
spec = importlib.util.spec_from_file_location("convert_experts_2bit", HERE / "convert_experts_2bit.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SIDECAR_KEYS = {
    "schema", "contract_scope", "runtime_qualified", "profile", "format_major",
    "format_minor", "tensor_shape", "allocation_entries", "source_q4_records_hex",
    "expected_decoded", "binary", "manifest_core_sha256", "manifest_binding_sha256",
}
BINARY_KEYS = set(m.MIXED2_FIXTURE_BINARY_FIELDS)
ALLOCATION_KEYS = {"encoding", "row", "group_start", "group_count"}


def fail(category, detail):
    raise ValueError(f"{category}: {detail}")


def sha(data):
    return hashlib.sha256(data).digest()


def align_up(value):
    return (value + m.MIXED2_ALIGNMENT - 1) // m.MIXED2_ALIGNMENT * m.MIXED2_ALIGNMENT


def validate_sidecar_shape(sidecar):
    if not isinstance(sidecar, dict) or set(sidecar) != SIDECAR_KEYS:
        fail("manifest", "unexpected or missing top-level keys")
    if (sidecar["schema"] != "dmlx.mixed2.fixture.v1" or
            sidecar["contract_scope"] != "reduced-single-expert-gate-tensor-fixture-only" or
            sidecar["runtime_qualified"] is not False or
            sidecar["profile"] != "reduced-tensor-production-wire"):
        fail("manifest", "fixture schema/scope/qualification/profile")
    if sidecar["format_major"] != 1 or sidecar["format_minor"] != 0:
        fail("manifest", "fixture format version")
    if sidecar["tensor_shape"] != [1, 256]:
        fail("manifest", "fixture tensor shape")
    if not isinstance(sidecar["binary"], dict) or set(sidecar["binary"]) != BINARY_KEYS:
        fail("manifest", "binary schema")
    allocations = sidecar["allocation_entries"]
    if not isinstance(allocations, list) or any(not isinstance(x, dict) or set(x) != ALLOCATION_KEYS for x in allocations):
        fail("manifest", "allocation schema")
    if not isinstance(sidecar["source_q4_records_hex"], list):
        fail("manifest", "source Q4 records")
    expected = sidecar["expected_decoded"]
    if not isinstance(expected, dict) or set(expected) != {"q4", "int2"}:
        fail("manifest", "expected decode schema")


def parse_and_validate(data, sidecar=None):
    """Validate structure and hashes; ``sidecar=None`` is test-internal only."""
    if len(data) < m.MIXED2_HEADER_SIZE:
        fail("bounds", "truncated header")
    h = data[:m.MIXED2_HEADER_SIZE]
    if sidecar is not None:
        validate_sidecar_shape(sidecar)
        _, core_hash, binding = m.mixed2_fixture_bindings(sidecar)
        if binding != sidecar["manifest_binding_sha256"]:
            fail("manifest-binding", "final sidecar binding mismatch")
        if core_hash != sidecar["manifest_core_sha256"]:
            fail("manifest-core", "fixture core projection mismatch")
        if h[352:384].hex() != sidecar["manifest_core_sha256"]:
            fail("header-core-identity", "identity #9 does not equal fixture core")
    if h[:16] != m.MIXED2_MAGIC:
        fail("magic", "invalid mixed2 magic")
    major, minor = struct.unpack_from("<HH", h, 16)
    if (major, minor) != m.MIXED2_VERSION:
        fail("version", f"unsupported {major}.{minor}")
    endian, hlen, elen = struct.unpack_from("<III", h, 20)
    if endian != m.MIXED2_ENDIAN_TAG:
        fail("endian", "invalid endian marker")
    if hlen != 512 or elen != 84:
        fail("shape", "fixed header/table-entry sizes")
    layer, components, experts, active = struct.unpack_from("<HHHH", h, 32)
    if layer != 0 or components != 3 or experts != 256 or active != 6:
        fail("shape", "fixed production header fields")
    count = struct.unpack_from("<I", h, 40)[0]
    toff, tlen, poff, plen, flen = struct.unpack_from("<QQQQQ", h, 44)
    alignment, flags, reserved = struct.unpack_from("<III", h, 84)
    if alignment != 64 or flags != 0:
        fail("enum", "alignment/flags")
    if reserved or any(h[480:512]):
        fail("padding", "nonzero header reserved bytes")
    if toff != 512 or tlen != count * 84:
        fail("bounds", "table geometry")
    if poff != align_up(toff + tlen) or poff % 64:
        fail("bounds", "payload alignment")
    if flen != len(data):
        fail("trailing-bytes" if flen < len(data) else "bounds", "file length mismatch")
    if poff + plen != flen or toff + tlen > poff:
        fail("bounds", "regions outside file")
    if any(data[toff + tlen:poff]):
        fail("padding", "nonzero table-to-payload gap")
    table, payload = data[toff:toff + tlen], data[poff:poff + plen]
    if sha(table) != h[384:416]:
        fail("table-hash", "SHA-256 mismatch")
    if sha(payload) != h[416:448]:
        fail("payload-hash", "SHA-256 mismatch")
    hz = bytearray(h)
    hz[448:480] = b"\0" * 32
    if sha(m.MIXED2_LAYER_DOMAIN + bytes(hz) + table + payload) != h[448:480]:
        fail("layer-binding", "SHA-256 mismatch")

    entries = []
    expected_offset = poff
    for i in range(count):
        fields = struct.unpack_from("<HBBBBBBIIIIIIHHQQ32s", table, i * 84)
        (expert, role, enc, bits, sdtype, bdtype, rsv, rows, cols, row,
         row_count, group, groups, logical, qsub, off, length, digest) = fields
        if expert >= 256 or role not in (0, 1, 2):
            fail("enum", f"entry {i} expert/role")
        if rsv:
            fail("padding", f"entry {i} reserved")
        if rows == 0 or cols == 0 or cols % 64 or row >= rows or row_count != 1:
            fail("shape", f"entry {i} dimensions/row_count")
        if logical != 64 or group + groups > cols // 64 or groups == 0:
            fail("bounds", f"entry {i} groups")
        if enc == m.MIXED2_ENCODING_Q4:
            expected = (4, m.MIXED2_SCALE_E8M0, m.MIXED2_BIAS_NONE, 32, groups * 34)
        elif enc == m.MIXED2_ENCODING_INT2:
            expected = (2, m.MIXED2_SCALE_E8M0, m.MIXED2_BIAS_BF16, 0, groups * 19)
        else:
            fail("encoding", f"entry {i} unknown {enc}")
        if (bits, sdtype, bdtype, qsub, length) != expected:
            fail("dtype", f"entry {i} encoding metadata")
        if off != expected_offset:
            fail("layout", f"entry {i} offset is not exact aligned recurrence")
        if off + length > flen:
            fail("bounds", f"entry {i} payload")
        record_bytes = data[off:off + length]
        if sha(record_bytes) != digest:
            fail("entry-hash", f"entry {i} SHA-256 mismatch")
        entries.append(dict(expert=expert, role=role, encoding=enc, rows=rows, cols=cols,
                            row=row, group=group, groups=groups, offset=off, length=length,
                            bytes=record_bytes))
        next_offset = align_up(off + length)
        if i + 1 < count and any(data[off + length:next_offset]):
            fail("padding", f"nonzero inter-entry gap after entry {i}")
        expected_offset = next_offset
    if entries and entries[-1]["offset"] + entries[-1]["length"] != flen:
        fail("trailing-bytes", "terminal padding is forbidden")
    if not entries and poff != flen:
        fail("trailing-bytes", "empty payload")

    if entries != sorted(entries, key=lambda e: (e["expert"], e["role"], e["row"], e["group"])):
        fail("order", "table is not canonical")
    by_coord = entries
    for a, b in zip(by_coord, by_coord[1:]):
        same_row = (a["expert"], a["role"], a["row"]) == (b["expert"], b["role"], b["row"])
        if same_row and a["group"] + a["groups"] > b["group"]:
            fail("overlap", "logical group overlap")
        if same_row and a["group"] + a["groups"] == b["group"] and a["encoding"] == b["encoding"]:
            fail("maximal-segments", "adjacent equal encodings")

    if sidecar is not None:
        allocations = [{"encoding": "q4" if e["encoding"] == 1 else "int2", "row": e["row"],
                        "group_start": e["group"], "group_count": e["groups"]} for e in entries]
        if sidecar["allocation_entries"] != allocations:
            fail("manifest-allocation", "entries do not exactly match parsed table")
        rows, cols = sidecar["tensor_shape"]
        if any(e["expert"] != 0 or e["role"] != 0 or e["rows"] != rows or e["cols"] != cols for e in entries):
            fail("manifest-allocation", "reduced fixture expert/component/shape mismatch")
        covered = [(e["row"], group) for e in entries for group in range(e["group"], e["group"] + e["groups"])]
        required = [(row, group) for row in range(rows) for group in range(cols // 64)]
        if covered != required:
            fail("manifest-allocation", "logical groups are not sorted, maximal, and covered exactly once")
        binary = sidecar["binary"]
        checks = {"file_size": len(data), "table_offset": toff, "table_length": tlen,
                  "payload_offset": poff, "payload_length": plen,
                  "header_sha256": sha(h).hex(), "table_sha256": sha(table).hex(),
                  "payload_sha256": sha(payload).hex(), "layer_binding_sha256": h[448:480].hex(),
                  "file_sha256": sha(data).hex()}
        for key in m.MIXED2_FIXTURE_BINARY_FIELDS:
            if binary[key] != checks[key]:
                fail(key.replace("_sha256", "-hash").replace("_", "-"), "sidecar binary metadata mismatch")

    decoded = []
    for e in entries:
        size = 34 if e["encoding"] == 1 else 19
        decoder = m.mixed2_decode_q4_record if size == 34 else m.mixed2_decode_int2_record
        decoded.append([decoder(e["bytes"][i:i + size]) for i in range(0, e["length"], size)])
    return entries, decoded


def reseal(raw, table_changed=False, payload_changed=False):
    """Reseal only enclosing binary hashes needed to expose an inner corruption."""
    b = bytearray(raw)
    toff, tlen, poff, plen = struct.unpack_from("<QQQQ", b, 44)
    if table_changed:
        b[384:416] = sha(bytes(b[toff:toff + tlen]))
    if payload_changed:
        b[416:448] = sha(bytes(b[poff:poff + plen]))
    b[448:480] = b"\0" * 32
    b[448:480] = sha(m.MIXED2_LAYER_DOMAIN + bytes(b[:512]) + bytes(b[toff:toff+tlen]) + bytes(b[poff:poff+plen]))
    return bytes(b)


def reseal_sidecar(sidecar):
    result = copy.deepcopy(sidecar)
    result["manifest_binding_sha256"] = m.MIXED2_ZERO_HASH
    _, core_hash, binding = m.mixed2_fixture_bindings(result)
    result["manifest_core_sha256"] = core_hash
    result["manifest_binding_sha256"] = binding
    return result


def sidecar_for_binary(sidecar, data):
    result = copy.deepcopy(sidecar)
    h = data[:512]
    toff, tlen, poff, plen = struct.unpack_from("<QQQQ", h, 44)
    result["binary"] = {"file_size": len(data), "table_offset": toff, "table_length": tlen,
                        "payload_offset": poff, "payload_length": plen,
                        "header_sha256": sha(h).hex(), "table_sha256": sha(data[toff:toff+tlen]).hex(),
                        "payload_sha256": sha(data[poff:poff+plen]).hex(),
                        "layer_binding_sha256": h[448:480].hex(), "file_sha256": sha(data).hex()}
    return reseal_sidecar(result)


def expect_failure(name, data, category, sidecar=None):
    try:
        parse_and_validate(data, sidecar)
    except ValueError as exc:
        if not str(exc).startswith(category + ":"):
            raise AssertionError(f"{name}: expected {category}, got {exc}") from exc
        return
    raise AssertionError(f"{name}: corruption accepted")


def expect_encode_failure(name, values, category):
    try:
        m.mixed2_encode_int2(values)
    except ValueError as exc:
        if not str(exc).startswith(category + ":"):
            raise AssertionError(f"{name}: expected {category}, got {exc}") from exc
        return
    raise AssertionError(f"{name}: invalid int2 input accepted")


def self_test(bin_path=FIXTURE_BIN, json_path=FIXTURE_JSON):
    generated, meta = m.build_mixed2_small_fixture()
    generated2, meta2 = m.build_mixed2_small_fixture()
    assert generated == generated2 and m.mixed2_canonical_json(meta) == m.mixed2_canonical_json(meta2)
    checked = bin_path.read_bytes()
    checked_meta = load_sidecar(json_path)
    canonical_sidecar = json_path.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "sidecar.json"
        tmp_path.write_bytes(b" " + canonical_sidecar)
        try:
            load_sidecar(tmp_path)
        except ValueError as exc:
            assert str(exc).startswith("manifest-canonical:")
        else:
            raise AssertionError("fixture: noncanonical whitespace sidecar accepted")
        tmp_path.write_bytes(b'{"schema":"duplicate",' + canonical_sidecar[1:])
        try:
            load_sidecar(tmp_path)
        except ValueError as exc:
            assert str(exc).startswith("manifest-canonical:")
        else:
            raise AssertionError("fixture: duplicate-key sidecar accepted")
    assert checked == generated, "fixture: checked-in binary differs from generator"
    assert json_path.read_bytes() == m.mixed2_canonical_json(meta) + b"\n", "fixture: sidecar differs"
    entries, decoded = parse_and_validate(checked, checked_meta)
    source_q4 = [bytes.fromhex(x) for x in checked_meta["source_q4_records_hex"]]
    assert entries[0]["bytes"] == b"".join(source_q4)
    assert [entries[0]["bytes"][i:i + 34] for i in range(0, entries[0]["length"], 34)] == source_q4
    assert decoded == [checked_meta["expected_decoded"]["q4"], checked_meta["expected_decoded"]["int2"]]
    assert checked[352:384].hex() == checked_meta["manifest_core_sha256"]

    constant = m.mixed2_encode_int2([-0.0] * 64)
    assert constant[:16] == b"\0" * 16 and constant[16] == 127
    assert constant[17:19] == b"\0\x80", "zero-range bias must preserve negative zero"
    halfway_constant = m.mixed2_encode_int2([1.00390625] * 64)
    assert halfway_constant[:16] == b"\0" * 16 and halfway_constant[16] == 127
    assert halfway_constant[17:19] == b"\x80\x3f", "zero-range BF16 tie must round to even"
    tie = m.mixed2_encode_int2([0.0, 0.5, 1.5, 2.5, 3.0] + [0.0] * 59)
    assert tie[0] == 0xA0 and tie[1] & 3 == 3, "int2 q ties must use RNE"
    expect_encode_failure("nan", [math.nan] + [0.0] * 63, "finite")
    expect_encode_failure("infinity", [math.inf] + [0.0] * 63, "finite")
    expect_encode_failure("exponent high", [0.0, math.ldexp(3.0, 129)] + [0.0] * 62, "scale-range")
    expect_encode_failure("exponent low", [0.0, math.ldexp(3.0, -128)] + [0.0] * 62, "scale-range")
    max_f64 = float.fromhex("0x1.fffffffffffffp+1023")
    expect_encode_failure("finite range overflow", [-max_f64, max_f64] + [0.0] * 62, "scale-range")
    expect_encode_failure("finite BF16 overflow", [max_f64] * 64, "bf16-range")

    cases = 0
    def corrupt(name, data, category, sidecar=None):
        nonlocal cases
        expect_failure(name, data, category, sidecar)
        cases += 1

    x = bytearray(checked); x[0] ^= 1; corrupt("magic", bytes(x), "magic")
    x = bytearray(checked); struct.pack_into("<H", x, 16, 2); corrupt("version", bytes(x), "version")
    x = bytearray(checked); x[384] ^= 1; corrupt("table hash", bytes(x), "table-hash")
    x = bytearray(checked); x[416] ^= 1; corrupt("payload hash", bytes(x), "payload-hash")
    x = bytearray(checked); x[512 + 52] ^= 1; corrupt("entry hash", reseal(x, table_changed=True), "entry-hash")
    x = bytearray(checked); x[448] ^= 1; corrupt("layer binding", bytes(x), "layer-binding")
    corrupt("truncation", checked[:-1], "bounds")
    corrupt("trailing", checked + b"x", "trailing-bytes")
    toff, tlen, poff, _ = struct.unpack_from("<QQQQ", checked, 44)
    x = bytearray(checked); x[toff + tlen] = 1; corrupt("table gap", bytes(x), "padding")
    first_off, first_len = entries[0]["offset"], entries[0]["length"]
    x = bytearray(checked); x[first_off + first_len] = 1
    corrupt("inter-entry gap", reseal(x, payload_changed=True), "padding")
    x = bytearray(checked); struct.pack_into("<Q", x, 512 + 84 + 36, entries[1]["offset"] + 64)
    corrupt("offset recurrence", reseal(x, table_changed=True), "layout")
    x = bytearray(checked); struct.pack_into("<I", x, 512 + 84 + 24, 1)
    corrupt("overlap", reseal(x, table_changed=True), "overlap")
    x = bytearray(checked); x[512 + 84 + 3] = 99
    corrupt("encoding", reseal(x, table_changed=True), "encoding")

    bad = copy.deepcopy(checked_meta); bad["binary"]["header_sha256"] = "0" * 64
    corrupt("sidecar header hash", checked, "header-hash", reseal_sidecar(bad))
    bad = copy.deepcopy(checked_meta); bad["binary"]["file_sha256"] = "0" * 64
    corrupt("sidecar file hash", checked, "file-hash", reseal_sidecar(bad))
    bad = copy.deepcopy(checked_meta); bad["manifest_binding_sha256"] = "0" * 64
    corrupt("sidecar final binding", checked, "manifest-binding", bad)
    x = bytearray(checked); x[352] ^= 1; x = reseal(x)
    corrupt("header core identity", x, "header-core-identity", sidecar_for_binary(checked_meta, x))

    print(f"SELF-TEST PASS: {len(entries)} entries, {sum(len(x) for x in decoded)} records, "
          f"{cases} corruption cases, 8 int2 edge cases, 2 canonical-sidecar cases")


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("manifest-canonical", f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    fail("manifest-canonical", f"nonstandard JSON constant {value}")


def load_sidecar(path):
    raw = path.read_bytes()
    try:
        sidecar = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs,
                             parse_constant=_reject_json_constant)
    except UnicodeDecodeError as exc:
        fail("manifest-canonical", "sidecar is not UTF-8")
    except json.JSONDecodeError as exc:
        fail("manifest-canonical", f"invalid JSON at byte {exc.pos}")
    canonical = m.mixed2_canonical_json(sidecar) + b"\n"
    if raw != canonical:
        fail("manifest-canonical", "sidecar bytes are not exact canonical JSON plus newline")
    return sidecar


def build_quality_report():
    """Audit frozen Task 7.3 evidence without inventing unavailable measurements."""
    fixture_data = FIXTURE_BIN.read_bytes()
    fixture_meta = load_sidecar(FIXTURE_JSON)
    entries, decoded = parse_and_validate(fixture_data, fixture_meta)
    logits_raw = LOGITS_FIXTURE.read_bytes()
    logits = json.loads(logits_raw)
    candidate_keys = {"mixed", "mixed_candidate", "uniform_int2", "matched_control"}
    present_candidate_keys = sorted(candidate_keys.intersection(logits))

    thresholds = {
        "storage": {"mixed_to_q4_ratio_max": "0.75"},
        "tensor": {
            "weighted_nrmse_max": "0.090",
            "weighted_snr_db_min": "20.9151498112135",
            "int2_unit_nrmse_p99_max": "0.35",
            "int2_unit_nrmse_max": "0.75",
            "absolute_error_rms_multiplier_max": "8",
        },
        "logits": {
            "top1_agreement_min": "0.98",
            "top5_overlap_mean_min": "0.970",
            "top5_overlap_p1_min": "0.600",
            "js_mean_max": "0.0020",
            "js_p99_max": "0.020",
            "centered_nrmse_mean_max": "0.060",
            "centered_nrmse_p99_max": "0.150",
            "mixed_vs_control_js_reduction_min": "0.20",
            "mixed_vs_control_centered_nrmse_reduction_min": "0.15",
            "mixed_vs_control_top1_points_min": "1.0",
        },
        "e2e": {
            "canonical_q4_passes": "7/7",
            "canonical_mixed_passes": "7/7",
            "large_suite_min_prompts": 256,
            "overall_delta_points_min": "-1.0",
            "stratum_delta_points_min": "-3.0",
            "paired_q4_pass_ratio_min": "0.95",
        },
    }
    unavailable_reason = (
        "full-model Q4/uniform-int2/matched-control/mixed outputs with identical held-out "
        "token IDs and settings are not present"
    )
    gates = [
        {
            "id": "artifact-and-provenance",
            "status": "failed",
            "reason": (
                "the only mixed artifact is a reduced fixture with runtime_qualified=false; "
                "frozen calibration/held-out manifests, imatrix, allocation trace, full-model "
                "mixed artifact, matched control, pinned MPFR evaluator, and Unicode 15.1 "
                "manifest/table bindings are unavailable"
            ),
            "measured": {
                "fixture_entries": len(entries),
                "fixture_records": sum(len(group) for group in decoded),
                "fixture_runtime_qualified": fixture_meta["runtime_qualified"],
                "fixture_validation": "pass",
            },
        },
        {"id": "tensor-reconstruction", "status": "unavailable", "reason": unavailable_reason},
        {
            "id": "heldout-logits",
            "status": "unavailable",
            "reason": (
                "the checked-in logits fixture contains only the authoritative Q4 reference; "
                f"candidate/control keys present: {present_candidate_keys}"
            ),
        },
        {"id": "canonical-seven", "status": "unavailable", "reason": unavailable_reason},
        {"id": "large-heldout-suite", "status": "unavailable", "reason": unavailable_reason},
        {"id": "serialized-storage", "status": "unavailable", "reason": unavailable_reason},
    ]
    return {
        "schema": QUALITY_SCHEMA,
        "decision": "REJECTED",
        "runtime_work_authorized": False,
        "stop_tasks": ["7.4", "7.5", "7.6"],
        "stop_reason": "first mandatory artifact-and-provenance gate failed",
        "policy": {
            "fail_closed": True,
            "quality_thresholds_frozen": True,
            "speed_can_waive_quality": False,
            "q4_remains_authoritative_default": True,
        },
        "inputs": {
            "rfc": {"path": "docs/en/analysis/mixed-2bit-rfc.md", "sha256": sha(RFC_PATH.read_bytes()).hex()},
            "fixture_binary": {"path": "tests/fixtures/mixed_2bit_small.bin", "sha256": sha(fixture_data).hex()},
            "fixture_sidecar": {"path": "tests/fixtures/mixed_2bit_small.json", "sha256": sha(FIXTURE_JSON.read_bytes()).hex()},
            "q4_logits": {
                "path": "tests/fixtures/dsv4_native_logits.json",
                "sha256": sha(logits_raw).hex(),
                "fixture_kind": logits.get("fixture_kind"),
                "model_manifest_sha256": logits.get("model", {}).get("model_manifest_sha256"),
            },
        },
        "thresholds": thresholds,
        "gates": gates,
    }


def quality_report_bytes():
    return json.dumps(build_quality_report(), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8") + b"\n"


def quality_self_test():
    first = quality_report_bytes()
    second = quality_report_bytes()
    assert first == second
    report = json.loads(first)
    assert report["decision"] == "REJECTED" and report["runtime_work_authorized"] is False
    assert report["gates"][0]["status"] == "failed"
    assert all(gate["status"] == "unavailable" for gate in report["gates"][1:])
    assert report["inputs"]["fixture_binary"]["sha256"] == sha(FIXTURE_BIN.read_bytes()).hex()
    print(f"QUALITY SELF-TEST PASS: {len(report['gates'])} gates, decision REJECTED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, default=FIXTURE_BIN)
    ap.add_argument("--sidecar", type=Path, default=FIXTURE_JSON)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--quality-self-test", action="store_true")
    ap.add_argument("--quality-report", type=Path)
    ap.add_argument("--regenerate-fixture", action="store_true")
    ap.add_argument("--reason")
    args = ap.parse_args()
    if args.quality_self_test:
        quality_self_test()
        return
    if args.quality_report is not None:
        output = quality_report_bytes()
        args.quality_report.parent.mkdir(parents=True, exist_ok=True)
        args.quality_report.write_bytes(output)
        print(f"quality report written: {args.quality_report} ({sha(output).hex()}, REJECTED)")
        return
    if args.regenerate_fixture:
        if not args.reason or not args.reason.strip():
            ap.error("--regenerate-fixture requires nonempty --reason")
        data, meta = m.build_mixed2_small_fixture()
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        args.fixture.write_bytes(data)
        args.sidecar.write_bytes(m.mixed2_canonical_json(meta) + b"\n")
        print(f"regenerated fixture ({args.reason}): {args.fixture}")
        return
    if args.self_test:
        self_test(args.fixture, args.sidecar)
        return
    entries, decoded = parse_and_validate(args.fixture.read_bytes(), load_sidecar(args.sidecar))
    print(f"VALID: {args.fixture} ({len(entries)} entries, {sum(len(x) for x in decoded)} records)")


if __name__ == "__main__":
    main()
