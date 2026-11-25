# Qwen3.8 DSpine Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in DSpine block=8 drafter for Qwen3.8 and validate it against DFlash2 in one ReleaseFast binary.

**Architecture:** A new `qwen38_dspine.zig` owns the checked loader, causal block forward, code bridge, serial refinement tail, and projected-KV ring. `Qwen38VTableAdapter` selects it explicitly and otherwise preserves DF2/MTP behavior; target verify and engine sampling remain unchanged.

**Tech Stack:** Zig 0.16, mlx-c/MLX, safetensors, the existing dmlx test and server harness, official `dspine_mlx` only as a golden reference.

---

## File map

- Create `src/models/qwen38_dspine.zig`: DSpine config, weights, forward,
  bridge, refinement, context ring, and focused unit tests.
- Create `src/models/testdata/dspine/meta.json` and small golden tensors:
  released-config-shaped metadata and official-reference numerical fixtures.
- Create `scripts/fixtures/gen_dspine_golden.py`: reproducible small-tensor
  golden generator using the official `dspine_mlx` formulas.
- Modify `src/model_registry.zig`: adapter state, explicit DSpine enable,
  prefill/draft/commit/reset/deinit branches, and text-only guard.
- Modify `src/tests.zig`: import the new test module.
- Modify `HANDOFF.md`: final capability, environment variable, measurements,
  and remaining limits after verification.
- Create `artifacts/dspine_phase1_20261125/`: raw smoke/ABBA logs and report.

No engine-loop or target-kernel changes are planned. Do not commit or push
unless the user explicitly requests it.

### Task 1: Config and checked weight layout

**Files:**
- Create: `src/models/qwen38_dspine.zig`
- Modify: `src/tests.zig`
- Test: `src/models/qwen38_dspine.zig`

- [ ] **Step 1: Add the failing config tests**

Import the module from `src/tests.zig` and add tests for the released fields:

```zig
test "DSpineConfig.parse released checkpoint and served block eight" {
    const raw = @embedFile("testdata/dspine/meta.json");
    const cfg = try DSpineConfig.parse(std.testing.allocator, raw, 8);
    try std.testing.expectEqual(@as(usize, 8), cfg.block_size);
    try std.testing.expectEqual(@as(usize, 16), cfg.trained_block_size);
    try std.testing.expectEqual(@as(usize, 16), cfg.head_topk);
    try std.testing.expectEqual([5]usize{ 1, 16, 31, 46, 61 }, cfg.target_layer_ids);
}

test "DSpineConfig rejects unsupported served block" {
    const raw = @embedFile("testdata/dspine/meta.json");
    try std.testing.expectError(error.UnsupportedBlockSize,
        DSpineConfig.parse(std.testing.allocator, raw, 16));
}
```

- [ ] **Step 2: Run the focused test and confirm the red state**

Run:

```bash
env -u MLX_METAL_FAST_SYNCH MLX_C_PREFIX=/opt/homebrew zig build test -- --filter DSpineConfig
```

Expected: compilation fails because `models/qwen38_dspine.zig` and its config
API do not exist.

- [ ] **Step 3: Implement strict config parsing**

Define the fixed Phase 1 contract and parse both top-level and
`dflash_config`/`selected_prefix_code_bridge` fields:

```zig
pub const DSpineConfig = struct {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    block_size: usize,
    trained_block_size: usize,
    mask_token_id: u32,
    target_layer_ids: [5]usize,
    num_target_layers: usize,
    sliding_window: usize,
    code_size: usize,
    bridge_rank: usize,
    bridge_eps: f32,
    head_topk: usize,

    pub fn parse(allocator: std.mem.Allocator, raw: []const u8, served_block: usize) !DSpineConfig {
        if (served_block != 8) return error.UnsupportedBlockSize;
        // Parse and validate every field used by the forward; require five
        // layers, no convolution, causal serial independent bridge writes,
        // K=16, five target taps, and trained_block >= served_block.
    }
};
```

The loader must strip an optional `model.` prefix, validate the scalar version
tensors, load `fc`, `hidden_norm`, `norm`, five attention/MLP layers, and these
bridge names:

```text
selected_prefix_bridge.{prev,gate,predecessor_gate,out}.{weight,bias}
selected_prefix_bridge.{read_weight,early_read_weights,feature_write_weights,codebook}
selected_prefix_bridge.{independent_write_version,terminal_disabled_version}
```

Use `MtpLinear` for quantizable backbone projections and plain `Array` for BF16
bridge tensors. Return an error on missing tensors, invalid versions, or a
nonzero terminal `out.weight`.

- [ ] **Step 4: Run config tests and inspect ownership**

Run the Step 2 command. Expected: both config tests pass; the testing allocator
reports no leaks.

### Task 2: Code-bridge numerical golden

**Files:**
- Create: `scripts/fixtures/gen_dspine_golden.py`
- Create: `src/models/testdata/dspine/bridge.safetensors`
- Modify: `src/models/qwen38_dspine.zig`

- [ ] **Step 1: Generate a small official-reference fixture**

Use tiny dimensions `H=8, C=4, R=2, T=3, layers=2`, deterministic inputs, and
the official bridge operation order. Save inputs, weights, per-layer outputs,
and last-layer `raw/rms/packed`. The generator must expose this callable:

```python
def bridge_reference(hidden, residual, input_ids, layer, next_norm_weight):
    raw = (hidden + residual).astype(np.float32).astype(np.bfloat16)
    # Follow dspine_mlx/bridge.py lines 110-163 operation by operation.
    # Return next_normed, out_residual, raw, rms, packed.
```

Run with the `~/venvs/dspine` environment and save safetensors through MLX;
do not hand-type floating-point golden values.

- [ ] **Step 2: Add and run the failing bridge test**

```zig
test "DSpine bridge matches official small golden" {
    const f = try loadBridgeGolden(std.testing.allocator);
    defer f.deinit();
    const got = try bridgeWrite(testCtx(), &f.weights, f.hidden, f.residual,
        f.input_ids, 0, f.next_norm, f.eps);
    defer got.deinit();
    try expectRelMaxDiff(f.expected_normed, got.normed, 2e-2, 1e-3);
    try expectRelMaxDiff(f.expected_raw, got.stash.raw, 0, 0);
}
```

Run the focused test. Expected: failure because `bridgeWrite` is undefined.

- [ ] **Step 3: Implement `bridgeWrite` and its stash**

Use this public result contract so the final layer can feed refinement:

```zig
pub const BridgeStash = struct {
    raw: Array,
    rms: Array,
    packed: Array,
    pub fn deinit(self: *BridgeStash) void { /* deinit each Array */ }
};
pub const BridgeOut = struct {
    normed: Array,
    residual: Array,
    stash: ?BridgeStash,
    pub fn deinit(self: *BridgeOut) void { /* release owned arrays/stash */ }
};
```

Implement exactly the normalization, anchor replacement, packed split,
adjacent message, not-anchor mask, scaled write, and next-norm sequence in the
design. Keep bridge arrays BF16; cast to FP32 only where the reference does.

- [ ] **Step 4: Re-run bridge and config tests**

Run:

```bash
env -u MLX_METAL_FAST_SYNCH MLX_C_PREFIX=/opt/homebrew zig build test -- --filter DSpine
```

Expected: config and bridge tests pass.

### Task 3: Causal context ring and five-layer block forward

**Files:**
- Modify: `src/models/qwen38_dspine.zig`
- Create: `src/models/testdata/dspine/forward.safetensors`

- [ ] **Step 1: Add failing mask/ring tests**

Test a ring with committed positions `3,4,5`, block length three, and window
three. The expected boolean mask is:

```text
query 0: context 3,4,5 + block 0
query 1: context 3,4,5 + block 0,1
query 2: context 3,4,5 + block 0,1,2
```

Also cover ring wrap, accepted-only `observe`, `reset`, and shadow trimming:

```zig
try std.testing.expect(!maskValue(mask, 0, ring_capacity + 1));
try std.testing.expect(maskValue(mask, 2, ring_capacity + 1));
try std.testing.expectEqual(@as(i64, 0), ring.fed);
ring.observe(projected_two_rows);
try std.testing.expectEqual(@as(i64, 2), ring.fed);
ring.reset();
try std.testing.expectEqual(@as(i64, 0), ring.fed);
```

- [ ] **Step 2: Run the tests red, then implement the ring**

Run the focused tests and confirm the missing-API failure. Implement:

```zig
pub const DSpineContext = struct {
    fed: i64,
    capacity: i32,
    ring_slots: i64,
    keys: [5]Array,
    values: [5]Array,
    pos: []i64,
    rows_shadow: ?Array,

    pub fn observe(self: *DSpineContext, tap_hidden: Array) !void;
    pub fn seedFromProjectedAt(self: *DSpineContext, rows: Array, start: i64) !void;
    pub fn draftMask(self: *DSpineContext, allocator: std.mem.Allocator) !Array;
    pub fn reset(self: *DSpineContext) void;
    pub fn deinit(self: *DSpineContext) void;
};
```

`draftMask` returns `[1,1,block,capacity+block]`: context visibility depends on
occupied/window positions; block key `j` is visible to query `i` iff `j <= i`.

- [ ] **Step 3: Add the official small forward golden and run it red**

Generate a two-layer causal block forward with the same tiny config, including
context K/V and per-layer bridge writes. Add:

```zig
const got = try blockHiddenRing(ctx, &model, &ring, embed, block_ids);
defer got.hidden.deinit();
try expectRelMaxDiff(expected.hidden, got.hidden, 2e-2, 1e-3);
try expectRelMaxDiff(expected.stash_raw, got.stash.?.raw, 0, 0);
```

Expected initial result: missing `blockHiddenRing` failure.

- [ ] **Step 4: Implement the forward and re-run**

Define:

```zig
pub const BlockOut = struct {
    hidden: Array,       // post-final-norm rows 1..7
    stash: BridgeStash, // final-layer pre-write state
    pub fn deinit(self: *BlockOut) void;
};

pub fn blockHiddenRing(ctx: EagerContext, model: *const DSpine,
    ring: *DSpineContext, embed_table: Array, block_ids: Array) !BlockOut;
```

Run all DSpine tests. Expected: ring and forward goldens pass, including causal
mask coverage.

### Task 4: Direct serial refinement and conditional sparse q

**Files:**
- Modify: `src/models/qwen38_dspine.zig`
- Create: `src/models/testdata/dspine/refine.safetensors`

- [ ] **Step 1: Add the failing refinement tests**

The fixture must contain seven or fewer positions, K=16 or a tiny-K unit
variant, known sender IDs, target-head rows, and the official sampled-table
output. Test both chain modes:

```zig
const toks = try greedyChain(table, candidates, 0, allocator);
try std.testing.expectEqualSlices(u32, expected_greedy, toks);
const row = try sparseLogitRow(ctx, table, candidates, position,
    previous_token, vocab, allocator);
try std.testing.expectEqual(@as(usize, vocab), row.len);
try std.testing.expect(std.math.isNegativeInf(row[non_candidate]));
try std.testing.expectEqual(expected_score, row[expected_candidate]);
try std.testing.expectError(error.PredecessorNotCandidate,
    sparseLogitRow(ctx, table, candidates, 1, missing_token, vocab, allocator));
```

Run and confirm failure because these APIs do not exist.

- [ ] **Step 2: Implement direct table construction**

Define a head-row accessor that gathers and, when necessary, dequantizes the
shared target LM head to BF16:

```zig
pub fn refinementTable(ctx: EagerContext, model: *const DSpine,
    lm_head: MtpLinear, hidden_rows: Array, stash: BridgeStash,
    candidates: Array, anchor: u32) !Array; // [P,K,K] f32
```

Use the direct predecessor-codebook formula from the design. Do not allocate
TG/TSV vocabulary tables and do not run a second full-vocabulary head.

- [ ] **Step 3: Implement chain walking and sparse rows**

`greedyChain` selects `argmax(table[p, sender, :])`, maps through
`candidates[p]`, then uses the selected candidate index as the next sender.
`sparseLogitRow` uses anchor sender zero at position zero; otherwise it finds
the actual previous token in `candidates[p-1]`, scatters K scores into a
vocabulary-sized f32 buffer, and fills every other entry with `-inf`.

- [ ] **Step 4: Run all numerical tests**

Run the DSpine filter. Expected: table values match the official fixture;
greedy and sparse-q tests pass, including the missing-predecessor error.

### Task 5: Adapter integration and round ownership

**Files:**
- Modify: `src/model_registry.zig`
- Modify: `src/models/qwen38.zig` only if the generic tap helper requires a
  second fixed tap-ID constant; prefer passing the DSpine config slice.

- [ ] **Step 1: Add the explicit selection and state**

Add DSpine pointers and round storage alongside, but independent of, DF2:

```zig
dspine: ?*qwen38_dspine.DSpine = null,
dspine_ctx: ?qwen38_dspine.DSpineContext = null,
dspine_block: [7]u32 = undefined,
dspine_candidates: [7 * 16]u32 = undefined,
dspine_table: ?Array = null,
dspine_block_len: usize = 0,
dspine_block_idx: usize = 0,
```

In `mtpEnable`, check `DMLX_QWEN38_DSPINE_DIR` before the DF2 variable, load at
served block eight, set `num_drafts=7`, and do not install DF2 tree/predraft
callbacks. A failed load returns false and leaves both drafters disabled.

- [ ] **Step 2: Implement prefill and block computation**

Mirror the existing target-tap prefill contract with DSpine tap IDs and its own
projected-row shadow. `computeDSpineBlock` must:

```text
embed anchor+7 masks -> blockHiddenRing -> shared head -> top16 candidates
-> refinementTable -> one small eval/readback -> greedy tokens or sampled table
```

For sampled rounds keep the table/candidates. For greedy rounds fill seven
IDs and serve `draftGreedy` without additional evaluations. Use one cleanup
function on prefill, commit, rollback, and failure.

- [ ] **Step 3: Implement conditional sampled draft and commit**

At sampled position zero, produce the anchor-conditioned sparse row. At later
positions use the `previous_token` argument to choose the sender. Return a
copy of `hidden`, as DF2 does. On commit, call the unchanged target
`commitVerify`, observe only `tap_hidden[:, :accepted]`, and clear the round.
Do not observe on rollback.

- [ ] **Step 4: Add the text-only guard and deinit**

DSpine vision prefill returns `error.DSpineVisionUnsupported`. Adapter deinit
releases round state, ring, model, cache IDs/rows, and persisted DSpine shadow;
DF2 deinit remains byte-for-byte behaviorally unchanged.

- [ ] **Step 5: Compile and run all unit tests**

Run:

```bash
zig fmt --check src/models/qwen38_dspine.zig src/model_registry.zig src/tests.zig
env -u MLX_METAL_FAST_SYNCH MLX_C_PREFIX=/opt/homebrew zig build test
```

Expected: formatting and the complete test suite pass. If a test exposes an
ownership error, fix ownership before running the model.

### Task 6: Real-checkpoint smoke and reference comparison

**Files:**
- Create: `artifacts/dspine_phase1_20261125/dspine_reference.py`
- Create: `artifacts/dspine_phase1_20261125/smoke_client.py`
- Create: raw server/client logs under the same directory

- [ ] **Step 1: Build the deliverable**

Run:

```bash
env -u MLX_METAL_FAST_SYNCH MLX_C_PREFIX=/opt/homebrew zig build -Doptimize=ReleaseFast
```

Expected: `zig-out/bin/dmlx` builds successfully in the isolated worktree.

- [ ] **Step 2: Run fixed-prompt greedy reference and dmlx smoke**

Use the released target/draft snapshots and the same fixed code-repair prompt.
Record official DSpine proposal IDs and dmlx draft/accept logs. Start dmlx with:

```bash
DMLX_QWEN38_DSPINE_DIR="$HOME/.cache/huggingface/hub/models--tencent--Qwen3.8-27B-DSpine-b16/snapshots/260a6c23070ca873ff1a5f33b7119ac3f6a245ee" \
zig-out/bin/dmlx serve --model "$HOME/models/Qwen3.8-27B-4bit" \
  --speculative-mtp 7 --system-mem-guard 0 --temperature 0 --top-k 0
```

Expected: block=8/M=8 rounds, no load/numerical/cache error, valid stop or
length finish, and target-greedy output agreement. Proposal-ID mismatches must
be diagnosed against the bridge/refinement goldens before benchmarking.

- [ ] **Step 3: Run sampled smoke**

Repeat with temperature 0.8/top-k 50/top-p 1.0 and a fixed seed. Confirm that
positions after zero consume the actual previous candidate, no
`PredecessorNotCandidate` occurs in normal operation, no NaN normalization
occurs, and output length/finish reason are valid.

- [ ] **Step 4: Check memory and clean process state**

Record server peak RSS/MLX metrics and confirm no other dmlx/MLX GPU process is
running before or after the smoke. Do not proceed to ABBA while correctness or
termination checks fail.

### Task 7: Same-binary six-scenario AB/BA and verdict

**Files:**
- Create: `artifacts/dspine_phase1_20261125/run_abba.sh`
- Create: `artifacts/dspine_phase1_20261125/abba_client.py`
- Create: `artifacts/dspine_phase1_20261125/analyze.py`
- Create: `artifacts/dspine_phase1_20261125/REPORT.md`
- Modify: `HANDOFF.md`

- [ ] **Step 1: Run sequential AB/BA**

Arm A is DF2+copies+bands; arm B is DSpine block=8. Use the same binary,
prompts, sampling, and four repetitions per cell per arm as the copy-bands
baseline, with pass order A→B then B→A. Run thinking on/off x short/2k/8k at
temperature 0.8/top-k 50/top-p 1.0/max 256, one server/GPU workload at a time.

- [ ] **Step 2: Validate output and mechanism metrics**

For every request record decode tok/s, generated tokens, finish reason,
acceptance/tokens per round, draft/verify/accept timings, and memory. Require
comparable output lengths/finish reasons and no cache/termination regression.
Run greedy six-cell controls after the sampled matrix.

- [ ] **Step 3: Write the verdict and update handoff**

Report medians and ranges per cell, drift across passes, and the explicit
promote/experimental/revert decision. Do not claim the Phase 0 cross-stack
numbers as dmlx gains. Update `HANDOFF.md` with the exact opt-in variable,
block=8/K=16 limits, text-only status, raw artifact path, and measured verdict.

- [ ] **Step 4: Final clean verification and workspace audit**

Re-run formatting, the complete test command, and ReleaseFast build after the
last source change. Inspect `git status --short`; retain only source, test,
fixture, design/plan, handoff, and benchmark artifacts relevant to this task.
Leave all pre-existing user files untouched and leave commits/pushes to an
explicit user request.
