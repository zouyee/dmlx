# DSpine configurable block size design

Date: 2026-11-25
Status: implemented; block=8 retained after sampled/greedy measurements (`artifacts/dspine_phase1_20261125/block5/REPORT.md`)

## Context

The official MLX guide recommends block=5 for 4-bit Qwen3.8-27B because larger
verify batches step up in cost. dmlx Phase 0 on M4 Pro favored block=8 in the
community stack, while dmlx Phase 1 has different tuned verify kernels and a
thinking-on/2k regression. A same-binary static block comparison is needed.

## Design

Add optional `DMLX_DSPINE_BLOCK`, accepting integer values 2 through 8 and
defaulting to 8. `DSpineConfig.parse` accepts this range and preserves the
checkpoint's trained block=16 separately. The adapter sets
`MtpOps.num_drafts=block-1` and logs the selected block.

The existing fixed arrays remain maximum-capacity storage (seven drafts and
7x16 candidates). Every active computation uses `P=block-1`: mask IDs, block
forward, candidate/table readback, greedy chain length, sampled sparse-q
position count, trace slices, and round cleanup. Copy proposals retain their
independent M=8 cap.

No engine-loop or target-kernel change. DF2 and sequential MTP behavior are
unchanged. Invalid environment values fail DSpine enable explicitly instead of
silently selecting another block.

## Verification

- Config tests accept 5 and 8, reject 1, 9, and malformed environment text.
- Real-checkpoint block=5 same-tap comparison against official dspine_mlx:
  four proposals and all 5x16 candidate IDs must match.
- Greedy and sampled block=5 server smokes complete valid outputs without
  cache/termination errors.
- Same ReleaseFast binary, DSpine block=5 versus block=8, sampled six-scenario
  AB/BA with n=4 per arm per cell and one-repetition greedy controls. Record
  decode, output/finish comparability, E[t], phase timings, and memory.
- Default remains 8 unless the measurements justify changing it; update
  HANDOFF and the Phase 1 report with the actual result.
