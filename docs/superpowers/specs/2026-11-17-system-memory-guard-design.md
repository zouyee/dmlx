# System Memory Guard — Design

Date: 2026-11-17
Status: draft (auto-mode; assumptions listed below)

## Context

The user's Mac OOM-reboots while running dmlx. Root cause gap in current code:
`src/memory.zig` only enforces a **process RSS** budget, and only **reactively at
request admission** (`enforceMemoryLimit` in `src/server/http.zig`). Nothing watches
**system-wide** memory. On Apple Silicon, MLX wired memory (`main.zig` sets a wired
limit at ~85% of RAM) plus other apps can drive the whole machine into swap
exhaustion / jetsam panic between two requests.

## Goal

A background monitor that watches total machine memory and prevents dmlx from
pushing the machine over the edge: shed load early instead of letting macOS
run out of memory and reboot.

## Design

### System memory sampling (`src/memory.zig`)

- `getSystemFreeMemoryBytes()` — `host_statistics64(HOST_VM_INFO64)`:
  reclaimable = free + inactive + purgeable + speculative pages × page size.
- `getSwapUsageBytes()` — `sysctl vm.swapusage` (total - free = used).
- `classifyPressure(used_pct, swap_used_bytes, total_ram) PressureLevel` —
  pure function, unit-tested:
  - `.ok`: used < warn threshold and swap < 25% of RAM
  - `.warn`: used ≥ warn threshold, or swap ≥ 25% of RAM
  - `.critical`: used ≥ critical threshold, or swap ≥ 50% of RAM

### Monitor loop (`src/server.zig`)

- Spawned as an `io.async` fiber in `start()` alongside `acceptLoop`.
- Polls every 2 s, updates an atomic `PressureLevel` on `ServerState`.
- `.warn`: log once per level change.
- `.critical`: additionally calls `memory_mod.enforceMemoryLimit(...)` to
  evict pool models / offload KV, and logs loudly.
- Loop exits when `engine_running` flips false (graceful shutdown).

### Admission gate (`src/server/http.zig`)

- At the existing memory-check point for `/v1/chat/completions` (and the other
  generate endpoints already calling `enforceMemoryLimit`): if guard level is
  `.warn` or `.critical`, return 503 `memory_pressure` before doing any work.

### Configuration

- `ServerConfig.system_mem_guard_pct: f32 = 0.92` — warn when system used
  fraction ≥ this; critical at `min(pct + 0.03, 0.99)`.
- CLI: `--system-mem-guard <pct|0>` on `serve`; `0` disables the guard.
  (`run` is a one-shot CLI without the HTTP server; the guard is server-side.)
- Default **on**. This is the whole point — most users will never discover a flag.

## Out of scope (YAGNI)

- Killing in-flight generations mid-decode (complex cancellation; admission
  gating + eviction is enough for the reported failure mode).
- Prometheus/metrics export of pressure level.
- Linux support (project targets macOS/Apple Silicon).

## Assumptions (auto mode, decided without asking)

1. Default threshold 92% system-used, critical at 95% — leaves ~4 GB free on a
   48 GB machine at warn time.
2. Swap growth is treated as pressure even if "used" looks fine, because macOS
   OOM-reboots correlate with runaway swap.
3. Guard applies to all generate endpoints that already call
   `enforceMemoryLimit`, not just chat completions.

## Testing

- Unit tests in `src/memory.zig` for `classifyPressure` boundary cases and for
  the sysctl/host_statistics queries returning sane values on real hardware.
- `make test` must pass.
