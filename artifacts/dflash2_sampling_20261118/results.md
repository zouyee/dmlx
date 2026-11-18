# DFlash2 sampled (temp>0) path — 2026-11-18

Change: dflash2 retains block logits [P,V] f32 per round and serves them via
MtpOps.draft; engine's existing rejection sampler (min(1,p/q), (p-q)+ resample)
handles acceptance. greedy_only gate removed.

Measured (M4 Pro 48GB, qwen3.8-27b-4bit, drafter 4-bit, serial):
- temp=0.8 code 256tok: dflash2 17.8 tok/s vs plain decode 13.2 tok/s (+35%)
- acceptance @temp0.8: 24-36% (vs ~40%+ greedy)
- accept phase 23-31ms/round (per-row vocab pulls) — main remaining cost
- greedy regression: 27.5-28.3 tok/s on 256tok code prompt (short run,
  depth-controller cold; path code untouched), outputs stable
- streaming + stop paths exercised OK
