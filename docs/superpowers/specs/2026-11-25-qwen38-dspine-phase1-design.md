# Qwen3.8 DSpine Phase 1 integration design

Date: 2026-11-25
Status: implemented as experimental opt-in; measured verdict in `artifacts/dspine_phase1_20261125/REPORT.md`

## Context and goal

Phase 0 measured the official/community MLX DSpine stack on M4 Pro. With a
4-bit draft and block=8 it beat the strongest dmlx DFlash2+copy-bands baseline
in all six decode-only cells (+6.7% to +25.6%, two DSpine repetitions per
cell). Verify was still 88-99 ms/round in that stack. Phase 1 will determine
whether the gain survives inside dmlx, where the M=8 target-verify path is
already tuned.

The deliverable is an **opt-in, text-first DSpine block=8 vertical slice** in
the same dmlx binary as DFlash2. DFlash2 remains the fallback and its behavior
must not change. This phase does not add tree verify, block>8, new target
kernels, 2-bit weights, concurrency, or a new default drafter.

## Approaches considered

1. **Independent DSpine module behind the existing MtpOps contract
   (selected).** Add `qwen38_dspine.zig`, adapter-owned DSpine state, and an
   explicit checkpoint environment variable. Reuse target prefill taps,
   verify/accept/commit, and the engine's sampler. This isolates the numerical
   and cache changes from the working DFlash2 path.
2. Generalize DFlash2 into a configurable block-drafter family. This would
   share more source, but DSpine differs in causal block attention, per-layer
   bridge writes, serial refinement, and conditional sampled proposals.
   Generalizing those branches would put the existing DF2 path at regression
   risk during the feasibility gate.
3. Keep DSpine in a Python sidecar. This avoids Zig numerical work but adds a
   second runtime and per-round IPC/GPU ownership problems; it cannot validate
   the intended dmlx verify path.

## Configuration and lifecycle

- `DMLX_QWEN38_DSPINE_DIR=<checkpoint>` selects DSpine and takes precedence
  over `DMLX_QWEN38_DFLASH2_DIR`; a log line makes the selection explicit.
- The checkpoint declares trained block=16, but Phase 1 serves block=8:
  anchor + seven proposals, target verify M=8. `--speculative-mtp` therefore
  enables seven DSpine drafts, matching the existing DF2 round contract.
- Draft backbone linear weights (`fc` and five transformer layers) use 4-bit
  affine group-size-64 quantization, matching Phase 0. Bridge weights,
  codebook, read/write projections, and norms remain BF16.
- DSpine target taps come from its own config (`1,16,31,46,61` for the released
  checkpoint), not DF2's tap IDs. Prompt-cache rows use a DSpine-specific
  in-memory shadow; DF2 rows are never substituted. Disk persistence is outside
  Phase 1, so a fresh process suppresses a backbone snapshot restore unless the
  required DSpine rows can be rebuilt.
- Prefill resets the DSpine ring and round state, observes accepted target
  hidden rows, and returns the normal last-logits/last-hidden result. Commit
  observes only accepted verify rows. Rollback touches the target through the
  existing verify implementation; no uncommitted DSpine ring rows are added.
- The initial integration is text-only. A vision request must return an
  explicit unsupported error rather than silently using the wrong tap layout.

## DSpine block forward

`qwen38_dspine.zig` owns config parsing, checked weight loading, the five-layer
forward, the code bridge, the serial refinement tail, and the projected-KV
ring.

For each round:

1. Embed `[anchor, mask x 7]` with the target embedding table.
2. Project target tap rows once per committed token with `fc + hidden_norm`
   and cache per-layer RoPE-applied K plus raw V in an 8192-position ring.
3. For each draft layer, run Qwen3 attention with causal attention **inside
   the block** and full visibility of valid in-window context, followed by the
   SwiGLU MLP and residual stream.
4. After every layer, apply the DSpine code-bridge write. Preserve BF16
   rounding points from `dspine_mlx/bridge.py`: raw residual, FP32 RMS,
   normalized BF16 read, normalized codes, anchor-code substitution, packed
   predecessor/gate/value projections, adjacent-causal message, scaled write,
   and the next layer's RMSNorm. Save the last layer's raw, RMS, and packed
   gate state for refinement.
5. Drop the anchor row and run the shared target LM head once to obtain the
   top-16 candidate IDs at each of the seven proposal positions.

The existing tuned verify-qmm arm may cover the block projections because all
shapes have at most eight rows. Draft numerics affect proposals only; target
verification remains authoritative.

## Serial refinement and sampled q contract

Build the released checkpoint's direct `[position, sender, candidate]` score
table with K=16. For each position and possible predecessor (anchor at position
zero, otherwise one of the previous position's candidates):

- gather the predecessor codebook and apply predecessor-gate/value projections;
- refine the raw hidden with the last feature-write matrix;
- normalize and score only the K gathered target-head rows.

This is the direct formula used by `RefinementTail.sampled_table`. It is used
for greedy selection as well, avoiding the reference implementation's
vocabulary-wide TG/TSV tables (about 762 MB of BF16) and their extra startup
work. It does not add a second full-vocabulary LM-head pass.

For **greedy** rounds, walk the table's argmax chain and return seven tokens
through the existing GPU-draft interface; the host table/candidate readback is
a single small evaluation.

For **sampled** rounds, the engine draws one proposal at a time. The adapter
keeps the table and candidate IDs for the round. Position zero uses the anchor
sender row. Each later `draft(hidden, previous_token)` call finds
`previous_token` in the previous candidate row and selects that sender's K
scores. It returns a vocabulary-shaped sparse-logit row containing those K
scores and `-inf` elsewhere. Thus the actual sampled predecessor conditions
the next proposal without pre-sampling inside the adapter and without seven
full LM-head evaluations. The existing target/draft rejection sampling and
adjusted resampling remain unchanged.

A previous token that is absent from its candidate row is an internal contract
error; fail the draft rather than silently selecting a different sender.

## Error handling and ownership

- Reject unsupported config features (grouped conv, non-serial bridge, missing
  independent writes, non-causal blocks, wrong tap count, head_topk other than
  16, or served block other than eight) during enable/load.
- All Arrays and host buffers have one owner. Round cleanup releases score
  tables, candidate buffers, sparse logits, and pending state on commit,
  rollback, prefill, and draft failure.
- DSpine and DF2 state are mutually exclusive. Load/init failure returns false
  from `MtpOps.enable`, preserving the existing non-speculative fallback.
- No DSpine predraft or tree callbacks in Phase 1; the engine uses its normal
  non-predraft chain path. Existing copy proposals remain orthogonal and may
  bypass the drafter exactly as they do for DF2.

## Verification

### Unit and numerical tests (added before implementation)

- Parse a released-config-shaped JSON and enforce block=8/K=16/tap IDs.
- Small-tensor code-bridge golden covering anchor substitution, adjacent
  writes, BF16/FP32 rounding, and final normalization.
- Small serial-table golden covering predecessor selection, sparse q rows,
  greedy chain walking, and the absent-predecessor error.
- Ring/mask tests proving causal block visibility, sliding-window eviction,
  prefill reset, accepted-only observe, and no DF2 cache reuse.
- Loader shape/name checks, including bridge weights remaining unquantized.

Where practical, generate golden values with the official `dspine_mlx` code at
commit `75cade7ab0307fc2daac8eda95337705226b217e` and record the generator in
the testdata directory.

### End-to-end checks

- Clean Zig test run and ReleaseFast build.
- Real-checkpoint greedy smoke: compare proposal tokens and final output with
  the official MLX stack on a fixed prompt; inspect acceptance and termination.
- Sampled smoke: exercise every round's predecessor-conditioned q path, with
  no NaN/-inf normalization failure, and comparable length/finish reason.
- Same ReleaseFast binary, DFlash2+copies+bands versus DSpine, sequential
  AB/BA, thinking on/off x short/2k/8k, temperature 0.8/top-k 50/max 256,
  four repetitions per arm per cell. Record decode throughput, tokens/round,
  per-phase timing, output lengths, finish reasons, and peak memory.
- Greedy six-cell controls and block=8 shape confirmation. No M>8 kernel work
  is triggered by this phase.

## Decision gate

Phase 1 is a feasibility integration, not a promise of a default-mode change.
Promote DSpine only if same-binary results reproduce a meaningful gain over
DFlash2 without correctness, termination, cache, or memory regressions. If the
gain disappears, retain the code only as an explicitly experimental opt-in or
revert it; do not expand kernels or make it the default to chase the Phase 0
cross-stack delta.
