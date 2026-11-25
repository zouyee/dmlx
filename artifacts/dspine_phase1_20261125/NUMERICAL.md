# DSpine Phase 1 numerical validation — 2026-11-25

## Same-input drafter comparison

The official MLX target and dmlx target do not produce the same greedy anchor
or raw hidden states for the fixed smoke prompt. A target-only dmlx control
produced the same `We need answer...` text as the DSpine-enabled server, so the
text difference is upstream of speculative decoding, not an output mutation by
DSpine.

To isolate the drafter, dmlx exported its raw five-layer target tap to
`dmlx_tap.safetensors`. The official `dspine_mlx` drafter was then run on that
exact input with anchor token 1596, 4-bit draft backbone, and block=8.

Results (`reference_same_dmlx_tap.log` versus `server_tensors.log`):

- All seven greedy proposal IDs matched exactly:
  `[1144, 4087, 1156, 25, 328, 7734, 264]`.
- All 7 x 16 = 112 initial candidate IDs matched exactly and in the same order.
- Projected-context first eight values matched exactly:
  `[1.046875, -0.515625, -0.97265625, 0.94921875, -0.341796875, 0.6015625, 0.435546875, -0.181640625]`.
- Post-forward hidden differences were limited to small BF16-rounding-scale
  values. Example, official versus dmlx:
  `[0.118164, 4.875, 3.921875, 0.941406, -3.796875, 2.25, -0.4453125, 2.40625]`
  versus
  `[0.108887, 4.90625, 3.890625, 0.9140625, -3.828125, 2.234375, -0.4453125, 2.46875]`.

This validates the Zig loader, projected context, causal block forward,
per-layer bridge, candidate selection, and serial greedy refinement on real
checkpoint tensors. It also explains why comparing official and dmlx final
text without forcing the same target features was not a valid drafter test.

## Server smokes

- Greedy, max 64: `finish=length`, no draft/load/cache/termination error.
- Sampled temperature 0.8/top-k 50, max 64: `finish=length`, no sparse-q or
  predecessor-contract error.
- Target-only greedy control: same output regime as DSpine-enabled dmlx.

Raw logs: `client_greedy.log`, `client_sampled.log`, `client_target_only.log`,
`server_greedy.log`, `reference_*.log`, and `server_*.log` in this directory.
