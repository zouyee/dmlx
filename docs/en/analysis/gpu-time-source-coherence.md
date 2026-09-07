# GPU-Time Optimization G0.1 — Workspace/Source Coherence Audit

**Decision:** `RECONCILIATION_REQUIRED`  
**Date:** 2026-08-22  
**No server or GPU-timing edit was performed.**

## Summary

The current workspace is not coherent with the accepted exact-Q4 telemetry spec. The
public C/Zig contract, native caller, focused tests, build targets, accepted semantic
review, and prior artifacts all describe a complete producer/runtime implementation, but
the current `src/metal_infer/engine.c` contains zero `moe_infer_exact_q4_*` definitions
or calls.

The reviewed implementation is recoverable from Git object storage:

- unreachable WIP commit: `505749335c2e5c73fb5a51b928b7109757acf7af`
- commit timestamp: `2026-08-20 16:33:58 +0800`
- commit subject: `WIP on main: 9a7e1ef feat(native): 2-bit expert quantization tooling — scheme FAILED at quality gate, infra preserved`
- `src/metal_infer/engine.c` blob: `2aba30b88eb58b9e64efda12d14e49b52c52c320`
- blob size: 502477 bytes / 10812 lines
- exact-Q4 definitions: **27 unique functions** — 18 production APIs and 9 test-only helpers/suites

Task 0.2 must reconcile `engine.c` from this reviewed recovery source before any raw GPU
timestamp work.

## Contract matrix

| Contract | `engine.h` | `engine.zig` | current `engine.c` | native caller/test | recovery blob |
|---|---|---|---|---|---|
| configure/enable/query | declared | extern + wrappers | **missing** | capture init/deinit + producer tests | implemented |
| begin request/token | declared | extern + wrappers | **missing** | `ExactQ4Request.begin/beginToken` | implemented |
| append stage | declared | extern + wrapper | **missing** | logits publication + stage tests | implemented |
| atomic layer commit | declared | extern + wrapper | **missing** | scalar transaction tests | implemented |
| reserve/start/complete/abort read | declared | extern + wrappers | **missing** | I/O tests | implemented |
| request drain/abort | declared | extern + wrapper | **missing** | native error path | implemented |
| finalize/snapshot/path snapshot | declared | extern + wrappers | **missing** | raw artifact writer + tests | implemented |
| reset/deconfigure | declared | extern + wrappers | **missing** | request/deinit paths + tests | implemented |
| producer test helpers | declared under test macros | N/A | **missing** | focused producer/stage/I/O targets | implemented |

## Evidence

### Header contract

`src/metal_infer/engine.h` declares the full API at lines 1383–1442, including:

- `moe_infer_exact_q4_configure`
- `moe_infer_exact_q4_begin_request` / `begin_token`
- `moe_infer_exact_q4_append_stage` / `commit_layer`
- complete per-read ownership lifecycle
- `drain_and_abort_request`, `finalize`, snapshots, reset/deconfigure
- deterministic producer/I/O test helpers

### Zig bridge and runtime caller

`src/metal_infer/engine.zig` declares/wraps the same API at lines 234–269 and
1180–1290. `src/native_engine.zig` calls configure/enable, begins requests/tokens,
appends logits, finalizes/snapshots/writes artifacts, drains failed requests before KV
reset, then resets/deconfigures.

### Focused tests/build

- `test_exact_q4_producer.c` directly exercises lifecycle, races, ownership, disabled
  overhead and bounded storage.
- `test_exact_q4_stages.c` exercises atomic layer commit, path sidecar, ordering,
  capacity failure and reset.
- `test_exact_q4_io.c` calls the production I/O suite.
- `build.zig` links all three targets against the current `src/metal_infer/engine.c`.

With current `engine.c`, these symbols cannot link. Therefore the current tree cannot
reproduce previously accepted exact-Q4 evidence despite retaining interfaces/tests.

### Accepted prior review

`semantic-review/2026-08-18-104147-pr-local.md` was `APPROVED` and specifically verified:

- C producer concurrency and abort gate;
- `moe_infer_exact_q4_commit_layer` atomic publication;
- scalar layer stage/path wiring;
- native request totality and drain-before-reset;
- producer/I/O/stage focused tests and build PASS.

This proves the implementation existed in the reviewed workspace.

### Recovery source discovery

Normal branch history and all named stashes contain no `engine.c` with
`moe_infer_exact_q4_commit_layer`. `git fsck --no-reflogs --unreachable` found blob
`2aba30b…`, and unreachable commit `5057493…` maps that blob to
`src/metal_infer/engine.c`. The blob contains all 27 exact-Q4 definitions (18 production
APIs plus 9 test-only helpers/suites), including `commit_layer` at line 308,
`drain_and_abort_request` at line 8695, and focused suites at lines 9950/10798.

A second blob `240168a…` contains declarations only and is the corresponding header, not
a runtime implementation.

## Required Task 0.2 reconciliation

1. Materialize recovery blob `2aba30b…` as the candidate `engine.c` in a temporary review
   context; do not blindly overwrite the working tree.
2. Diff it against current `engine.c` and classify all post-review changes, especially:
   Task 1.2 buffer hoisting/deinit fixes and unrelated 2-bit/DSpark work.
3. Restore producer/runtime/stage/I/O code while preserving independently newer changes.
4. Resolve API signature drift against current `engine.h`, `mla_attention.h`, and DSpark
   declarations.
5. Run build plus producer/I/O/stage focused tests and report self-tests.
6. Only after semantic review APPROVED may G1 timestamp work begin.

## Decision rationale

`COHERENT` is impossible because required production definitions are missing.
`BLOCKED_BY_REJECTED_DEPENDENCY` is unnecessary because a reviewed recovery source was
found. The correct nonterminal result is **`RECONCILIATION_REQUIRED`**, mandating Task
0.2.
