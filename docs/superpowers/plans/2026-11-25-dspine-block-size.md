# DSpine Configurable Block Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support `DMLX_DSPINE_BLOCK=2..8` with default 8 and compare block 5 against 8 in one binary.

**Architecture:** Parse the block once at DSpine enable and propagate `P=block-1` through the existing adapter. Keep fixed arrays only as capacity storage; all numerical work and engine draft counts use the selected block.

**Tech Stack:** Zig 0.16, mlx-c, existing dmlx AB/BA harness.

---

### Task 1: Config and adapter block propagation

**Files:**
- Modify: `src/models/qwen38_dspine.zig`
- Modify: `src/model_registry.zig`

- [ ] **Step 1: Add failing tests**

```zig
test "DSpineConfig accepts block five" {
    const raw = @embedFile("testdata/dspine/meta.json");
    const cfg = try DSpineConfig.parse(std.testing.allocator, raw, 5);
    try std.testing.expectEqual(@as(usize, 5), cfg.block_size);
    try std.testing.expectEqual(@as(usize, 16), cfg.trained_block_size);
}

test "DSpineConfig rejects blocks outside two through eight" {
    const raw = @embedFile("testdata/dspine/meta.json");
    try std.testing.expectError(error.UnsupportedBlockSize, DSpineConfig.parse(std.testing.allocator, raw, 1));
    try std.testing.expectError(error.UnsupportedBlockSize, DSpineConfig.parse(std.testing.allocator, raw, 9));
}
```

- [ ] **Step 2: Run the complete test suite red**

Run `env -u MLX_METAL_FAST_SYNCH MLX_C_PREFIX=/opt/homebrew zig build test`.
Expected: block=5 and invalid-boundary tests fail.

- [ ] **Step 3: Implement the range and environment parser**

Use this validation in `DSpineConfig.parse`:

```zig
if (served_block < 2 or served_block > 8) return error.UnsupportedBlockSize;
```

Add this adapter helper and call it before `DSpine.load`; pass the result into
a new `DSpine.loadBlock(..., served_block)` API while keeping `load` as the
block=8 wrapper for existing callers:

```zig
fn dspineBlockFromEnv() !usize {
    const raw = std.c.getenv("DMLX_DSPINE_BLOCK") orelse return 8;
    const text = std.mem.span(raw);
    const block = std.fmt.parseInt(usize, text, 10) catch return error.InvalidDSpineBlock;
    if (block < 2 or block > 8) return error.InvalidDSpineBlock;
    return block;
}
```

Set `self.mtp_ops.num_drafts = block - 1`. In `dspineComputeBlock`, use
`const P: i32 = @intCast(m.cfg.block_size - 1);`, allocate mask IDs with
`@intCast(P)`, read only `P * 16` candidates, and call:

```zig
const positions: usize = m.cfg.block_size - 1;
const greedy = try qwen38_dspine.greedyChainFromHost(
    self.dspine_table_host, self.dspine_candidates[0 .. positions * 16],
    positions, 16, self.allocator);
self.dspine_block_len = positions;
```

Pass `positions` to `sparseLogitRowFromHost`. Trace only
`self.dspine_block[0..positions]` and
`self.dspine_candidates[0 .. positions * 16]`.

- [ ] **Step 4: Run formatting, tests, and ReleaseFast**

Run `zig fmt --check src/models/qwen38_dspine.zig src/model_registry.zig`,
then the test command above, then
`env -u MLX_METAL_FAST_SYNCH MLX_C_PREFIX=/opt/homebrew zig build -Doptimize=ReleaseFast`.
Expected: all pass; block=8 default remains unchanged.

### Task 2: Block=5 numerical and server validation

**Files:**
- Modify: `artifacts/dspine_phase1_20261125/dspine_reference.py`
- Create: `artifacts/dspine_phase1_20261125/block5/` raw logs

- [ ] **Step 1: Parameterize the official reference**

Read `DSPINE_BLOCK` with default 8 and use it for generator construction,
`round_T`, block IDs, and stats. Keep the exported dmlx tap and forced anchor
support unchanged.

- [ ] **Step 2: Compare block=5 proposals**

Run dmlx with `DMLX_DSPINE_BLOCK=5 DMLX_DSPINE_TRACE=1`, then official with
`DSPINE_BLOCK=5 FORCE_ANCHOR=1596 DMLX_TAP_PATH=<dmlx_tap.safetensors>`.
Expected: four proposal IDs and all 5x16 candidate IDs match exactly.

- [ ] **Step 3: Run server smokes and full checks**

Run greedy and sampled block=5 requests at max 64, then repeat formatting,
full tests, and ReleaseFast. Expected: valid finish reasons, no predecessor or
cache errors, and all build/test checks pass.

### Task 3: Static block 5 versus 8 measurements

**Files:**
- Create: `artifacts/dspine_phase1_20261125/block5/run_abba.sh`
- Create: `artifacts/dspine_phase1_20261125/block5/run_greedy.sh`
- Create: `artifacts/dspine_phase1_20261125/block5/REPORT.md`
- Modify: `HANDOFF.md`
- Modify: `artifacts/dspine_phase1_20261125/REPORT.md`

- [ ] **Step 1: Run sampled AB/BA**

A = DSpine block=5, B = DSpine block=8, same ReleaseFast binary, A→B then B→A,
n=4 per arm per cell, all other settings copied from the Phase 1 matrix.
Expected logs contain M=5 and M=8 verify rounds respectively, with comparable
output lengths and finish reasons.

- [ ] **Step 2: Run greedy controls and memory**

Run one greedy repetition per cell for both blocks and the short MLX MemProbe
control. Record active and peak memory.

- [ ] **Step 3: Write the verdict**

Update the Phase 1 report and HANDOFF with per-cell medians/ranges, E[t],
phase timings, memory, and whether block=5 should replace 8 as the default.
Do not change the default unless the measured evidence justifies it. Leave all
source and benchmark changes uncommitted unless the user explicitly asks for a
commit.
