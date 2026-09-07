# DSV4 bounded attention investigation

## Decision

The production-normal 14-stage timestamp probe disproves the bounded `wq_b dominance`
hypothesis. Across three serial clean-start native N=20 requests, `mhc_pre` had the largest
sampled encoder-duration mass and `wq_b` was second. This is stage-duration mass only: it
does not establish exclusive GPU time, wall-time ownership, scheduling slack, or recoverable
latency, and it does not authorize a kernel optimization or unblock Task 2.2 / Tasks 3.1–5.2.

## Frozen conditions

- ReleaseFast binary SHA-256: `9fd2e119bff3e8c3d6819f203767c88003754a8fccc27b3aec86034f29efb00e`
- Native engine, packed experts, `NATIVE_SMELT_N=20`
- `NATIVE_TELEMETRY=1`, `NATIVE_ATTENTION_STAGE_TIME=1`
- `MF_STAGE_TIME` unset, temperature 0, max tokens 5
- One server at a time; each accepted request used a clean server start
- All three responses were exactly `Paris.` and stage status was `available`

## Production evidence

Every stage had `valid_count=430` in every run.

| Run | mhc_pre (ns) | wq_b (ns) | wo_a (ns) | wo_b (ns) | Largest |
|---:|---:|---:|---:|---:|---|
| 1 | 321471657 | 104494537 | 96977579 | 80477706 | mhc_pre |
| 2 | 296851880 | 97068966 | 88769125 | 74379134 | mhc_pre |
| 3 | 284568941 | 93537629 | 83218796 | 69633461 | mhc_pre |

The complete fixed-order telemetry contained all 14 stages with positive sums and equal
counts. The sealed extracted evidence is
`docs/en/analysis/dsv4-attention-stage-production-evidence.json`; it records the frozen
configuration, binary identity, response correctness, full vectors, and source-log SHA-256
values for all three runs. Metal returned valid overlapping intervals for adjacent encoders (observed between
`q_rope` and `wkv`), so pair-local start/end ordering is validated while cross-encoder
non-overlap is intentionally not inferred.

## Validation

- `zig build test-attention-stage-timestamp -Doptimize=ReleaseFast`: PASS
- `zig build test-exact-q4-producer -Doptimize=ReleaseFast`: PASS, disabled ratio 0.005912
- `zig build -Doptimize=ReleaseFast`: PASS
- `bash scripts/run_kernel_tests.sh`: PASS
- `bash scripts/run_mla_attention_test.sh`: PASS

## Terminal boundary

`wq_b dominance = DISPROVEN`. The next optimization direction is not selected by this
investigation. Any investigation of `mhc_pre` requires a separate bounded hypothesis and
explicit authorization; the current evidence alone is insufficient to change production
kernels.

## Bounded `mhc_pre` operational candidate investigation

A standalone-only harness invokes the exact production
`mhc_pre_split_weighted_sum_norm` 9-buffer ABI and 1×256 launch. Test-only dot-only and
Phase-B-only variants retain one CB, one encoder, one dispatch, one commit and one wait per
sample. All timings are stage-boundary `GPUTimestamp`; production engine and kernel paths are
unchanged.

The first composite/fixed-order attempt was rejected by semantic review and contributes no
evidence. The separately frozen final protocol uses a true dense-dot arm with residual norm
precomputed outside the timed encoder, rotating interleaved arm order, 10 warmups, and 21
retained raw samples per arm. The final harness passed an internal CPU reference for all 24
isolated dot outputs and the production fused `post`, `comb`, `attn_input`, and `normed`
outputs in every independent process. Complete vectors are sealed in
`dsv4-mhc-pre-investigation-evidence.json`:

| Run | Full median ns | Dot median ns | Phase-B median ns | Dot/full |
|---:|---:|---:|---:|---:|
| 1 | 478166 | 440333 | 16042 | 0.920879 |
| 2 | 566791 | 510542 | 18167 | 0.900759 |
| 3 | 497833 | 446417 | 14458 | 0.896720 |

All three isolated-arm ratios exceed the independently frozen 0.70 threshold. Therefore the
24×16384 dense-f32 dot path is **RETAINED AS THE NEXT INVESTIGATION CANDIDATE**. This is not
a claim that dots own a majority fraction of fused `mhc_pre`: separately compiled variant
durations are non-additive and may exceed one, so the ratio cannot be interpreted as a
fractional decomposition. This narrows the next engineering candidate to parallelizing or
restructuring those dot products, but it does not prove production wall-time savings,
exclusive GPU ownership, recoverable latency, or authorize implementation. A production
kernel change remains a separate decision requiring explicit authorization and correctness/
performance gates.
## Standalone parallel dense-dot candidate screen

Task 2.9 compiled three kernels exclusively from the harness-owned
`scripts/mhc_pre_parallel_kernels.metal`; production Metal was read only for SHA-256 identity
and was neither compiled nor modified. Every process retained 30 warmup and 63 measured
events under the frozen process-offset permutation, and all 24 outputs passed finite,
bitwise-repeatability, and scalar CPU absolute-error gates.

| Run | Serial median ns | SIMDgroup median ns | SIMD/serial | 24×8 median ns | 24×8/serial |
|---:|---:|---:|---:|---:|---:|
| 1 | 446875 | 396834 | 0.888020 | 355916 | 0.796455 |
| 2 | 477333 | 459333 | 0.962290 | 403625 | 0.845584 |
| 3 | 298209 | 290959 | 0.975688 | 241333 | 0.809275 |

The frozen viability rule requires a candidate ratio `<=0.80` in every process. The
SIMDgroup candidate fails all three runs; the 24×8 candidate passes only run 1. The decision
is therefore **REJECTED**. The pilot run is excluded from evidence, thresholds are unchanged,
and neither mapping is recommended for a production-implementation authorization request.
This standalone rejection does not establish production performance, wall/exclusive/
recoverable time, or alter/unlock any production task. Complete events and exact-rational
replay are sealed in `dsv4-mhc-pre-parallel-screen-evidence.json` and verified by
`python3 scripts/mhc_pre_parallel_screen.py`.
