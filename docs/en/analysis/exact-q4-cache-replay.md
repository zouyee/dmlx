---
decision_date: 2026-08-15
status: online_ab_candidate_selected
selected_candidate: lru_observed_hot_seeded
runtime_default_changed: false
corpus_metadata_sha256: ee5f73376f5e750d37facc1368bd992256dfee6b00cb990692f37924a04856eb
replay_artifact_sha256: 7bb9c946e8b1757ae3354945191d02e70494b2be39d77d3266cc19f9ca87da6a
---

# Exact-Q4 cache replay decision

## Decision

Advance **bounded, per-layer, observed-hot-seeded LRU** to a controlled online A/B behind `NATIVE_EXPERIMENTAL_CACHE=1`. Do not change the default fixed-hot N=20 policy in this task.

LRU is the strongest deployable policy in the formal replay: it beats fixed-hot physical read bytes in every captured shape, reduces aggregate application-visible expert reads by 59.905%, raises the modeled hit rate from 15.558% to 66.143%, and uses the same 1,568 expert slots and 20,963,131,392 modeled resident bytes (19.523 GiB). This satisfies the R6.3 replay and modeled-memory entry gate, but it is not evidence of an online speedup by itself.

The future-aware static-frequency and Belady results are ceilings only. They cannot enter online A/B. LFU is deployable but reads more bytes and churns more slots than LRU in aggregate. Trust-OS/no-user-cache regresses physical bytes. No preload, higher-capacity default, routing change, prefetch path, or runtime implementation is approved here.

## Evidence boundary

The decision uses these private, local artifacts:

- corpus metadata: `/tmp/dmlx-routing-traces/corpus-metadata.json`, SHA-256 `ee5f73376f5e750d37facc1368bd992256dfee6b00cb990692f37924a04856eb`;
- policy replay: `/tmp/dmlx-routing-traces/cache-policy-replay.json`, SHA-256 `7bb9c946e8b1757ae3354945191d02e70494b2be39d77d3266cc19f9ca87da6a`;
- replay source: `scripts/replay_routing_trace.py`, SHA-256 `662519f98b3b5991583e532369d5d7217266c1077af5caaed84cf58d928e5697`;
- cache-policy source: `scripts/cache_policy.py`, SHA-256 `0667f0fff0e0eece39977f15e4c54661c21b3176f51c918cadf9adf8784a701e`;
- full-content model manifest: `807672b318f49dd74567d69a603dbf35434c8d71c900f246150d79a18bc1b6d0`;
- hardware: Apple M4 Pro with 48 GiB unified memory;
- formal capture: native mode, one server, serial requests, N=20, prefetch and all batch/DSpark flags off.

Both retained artifacts are mode `0600`. Corpus replay requires exact schema, configuration, privacy and capture/replay/fingerprint tooling objects, path-confined trace files, matching trace hashes and fingerprints, and zero dropped or invalid route records.

## Replay contract and limitations

The result is deterministic under the following contract:

1. `event_seq` is the only declared route order. Token-, layer-, or phase-major order is not inferred.
2. Each top-6 route is classified atomically against one pre-admission cache snapshot. Admissions happen afterward in rank order.
3. Caches are independent by layer. DSV4 hash-routing layers 0–2 are modeled fully resident at 256 experts; score-routing layers 3–42 use capacity 20.
4. Fixed-hot uses observed HOT membership and actual synchronous `route_bytes`. Counterfactual misses cost one complete 13,369,344-byte expert read.
5. LRU and LFU start with observed HOT members plus anonymous occupied slots. Anonymous slots preserve startup occupancy without inventing expert hits.
6. Each split trace resets to its reconstructed initial seed. State is not carried between independent corpus cases.
7. Startup preload bytes are excluded from request read bytes but resident memory is reported. Slot churn means resident evictions/replacements.
8. Trust-OS charges one application-visible full `pread` per routed expert. It does not claim that an OS page-cache hit is a zero-byte read.
9. Static frequency uses the entire evaluated trace to choose residents. Belady uses future route knowledge from the same dynamic-policy seed. Both leak the future and are ineligible online.
10. Replay models hits, bytes, churn and resident slots. It does not model lookup cost, locking, metadata allocation, SSD latency distribution, OS page-cache behavior, GPU overlap or end-to-end tok/s.

## Aggregate result

All percentages below were independently recomputed from integer case totals rather than copied from the artifact aggregate.

| Policy | Role | Online eligible | Hit rate | Physical read bytes | Delta vs fixed | Slot churn | Modeled resident bytes | Cases better than fixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed hot observed | Baseline | No | 15.558% | 498,061,541,376 | — | 0 | 20,963,131,392 | 0/3 |
| **LRU, observed-hot seeded** | Candidate | **Yes** | **66.143%** | **199,697,891,328** | **-59.905%** | 14,937 | 20,963,131,392 | **3/3** |
| LFU, observed-hot seeded | Candidate | Yes | 58.520% | 244,658,995,200 | -50.878% | 18,300 | 20,963,131,392 | 3/3 |
| Trust-OS/no-user-cache | Reference | Yes | 0.000% | 589,828,718,592 | +18.425% | 0 | 0 | 0/3 |
| Static-frequency oracle | Future-aware ceiling | No | 73.784% | 154,629,832,704 | -68.954% | 0 | 20,963,131,392 | 3/3 |
| Belady, same-seed lower bound | Future-aware floor | No | 77.549% | 132,423,352,320 | -73.412% | 7,055 | 20,963,131,392 | 3/3 |

LRU captures 81.601% of the byte reduction between fixed-hot and the same-seed Belady floor. Its reads are still 50.803% above that floor, so meaningful theoretical headroom remains; that headroom does not identify another implementable policy.

Static frequency can beat same-seed Belady in an individual short trace because the two policies have different startup information. Static frequency preselects its initial residents using the evaluated future, while Belady is constrained to the observed-hot/anonymous seed used by online dynamic candidates. It is therefore not a contradiction and not a promotion argument for static frequency.

## Robustness by workload

| Independent corpus case | Fixed-hot bytes | LRU bytes | LRU delta | LFU bytes | LFU delta | LRU churn |
|---|---:|---:|---:|---:|---:|---:|
| Short prefill, short decode | 43,076,026,368 | 25,080,889,344 | -41.775% | 24,465,899,520 | -43.203% | 1,876 |
| Long prefill, short decode | 328,284,241,920 | 117,543,272,448 | -64.195% | 156,768,927,744 | -52.246% | 8,792 |
| Short prefill, long decode | 126,701,273,088 | 57,073,729,536 | -54.954% | 63,424,167,936 | -49.942% | 4,269 |

LFU is slightly better only in the shortest case. LRU is materially better in both long-prefill and long-decode evidence and has lower aggregate churn, so LRU is the single candidate selected for implementation.

| Phase | Fixed-hot bytes | LRU bytes | LRU delta | LRU hit rate | LRU churn | Belady-floor bytes |
|---|---:|---:|---:|---:|---:|---:|
| Prefill | 391,360,806,912 | 156,862,513,152 | -59.919% | 66.062% | 11,733 | 107,903,975,424 |
| Decode | 106,700,734,464 | 42,835,378,176 | -59.855% | 66.436% | 3,204 | 24,519,376,896 |

The decode-specific result matters for R6.5: the reduction is not an aggregate artifact caused only by long prefill.

## Memory decision

Formal geometry is `3 × 256 + 40 × 20 = 1,568` expert slots. At 13,369,344 bytes per packed expert, modeled cache residency is 20,963,131,392 bytes (19.523 GiB) for fixed-hot, LRU, LFU and both oracle policies. LRU changes membership, not capacity, and therefore does not buy fewer reads by increasing the modeled expert cache.

This is only a modeled equality. The online implementation must separately report LRU metadata, synchronization and transient allocation overhead plus peak process RSS. It must retain the existing safe RSS gate and may not assume reclaimable OS page cache is guaranteed free memory. Full preload remains prohibited: all 256 experts across all 43 layers would require roughly 147 GB, far beyond the 48 GiB machine.

N=51 is not a substitute decision. With the same three fully resident hash layers it would require `3 × 256 + 40 × 51 = 2,808` slots, or 37,541,117,952 bytes (34.963 GiB), before other model, Metal, OS and page-cache needs. It approaches the machine limit, is outside the formal N=20 replay geometry, and must not become a default through this work.

## Is 2.0 or 2.5 tok/s plausible?

The latest matched scalar evidence in `performance-benchmark.md` reports:

- 1.287128 tok/s group median for standard 5-token requests;
- 1.867517 tok/s median for a separate 100-token diagnostic.

Reaching 2.0 tok/s would require +55.385% from the standard result or +7.094% from the long diagnostic. Reaching 2.5 would require +94.231% or +33.868%, respectively. These are different scopes: the short standard result includes costs that are not steady decode, while the 100-token result is explicitly diagnostic and cannot replace the standard promotion gate.

LRU reduces replayed decode expert bytes by 59.855%. If all elapsed time scaled only with these bytes, the byte ratio would be a 2.494× upper bound. That is not a throughput forecast. Cache lookup and eviction work, 14,937 replacements, actual `pread` latency, page-cache residency, Metal work and CPU/GPU/I/O overlap prevent a direct bytes-to-tok/s conversion.

The decision is therefore:

- **2.0 tok/s is plausible enough to justify controlled online A/B, but unproven.** The replay removes enough decode bytes to clear the required speedup in an I/O-only model, and prior scalar telemetry shows expert I/O is material.
- **2.5 tok/s remains a stretch hypothesis, not expected performance.** It requires stronger translation of byte savings into wall-clock savings and has less margin for dynamic-policy overhead.
- Neither target is achieved by this replay. Task 6.5 must report an evidence-based miss and bottleneck attribution if online A/B does not reach them.
- The failed batching promotion is independent evidence. Its 63.417% standard regression and DSpark verification cost neither invalidate the LRU replay nor count toward the exact-Q4 decode target.

## Online A/B contract

Task 6.4 may implement only the selected bounded LRU policy, behind an exact-value experimental flag that is off by default. Task 6.5 may consider promotion only if serial, repeated A/B under matched warm/cold conditions demonstrates all of the following:

1. exact-Q4 logits, routed experts and generated tokens are unchanged;
2. capacity stays at N=20 for score layers and no full preload is introduced;
3. observed bytes and hit-rate direction agree with replay across standard and long-decode cases;
4. standard decode tok/s and ITL improve reproducibly, with median and spread rather than a best run;
5. bytes/token, `pread` calls, I/O wait, slot churn, LRU metadata bytes and peak RSS are reported;
6. peak RSS remains inside the existing safety gate and no memory pressure is hidden in OS page cache;
7. failures, output drift, an I/O/RSS regression or non-robust speedup immediately disables the flag and retains fixed-hot N=20.

The offline result does not authorize default promotion. That decision belongs to the online A/B gate.

## Task 6.4 online implementation evidence

The selected policy was implemented as an exact-value startup opt-in: only the string `NATIVE_EXPERIMENTAL_CACHE=1` enables it; unset and every other value retain fixed-hot N=20. Score layers reuse the existing 20 slots, hash layers remain immutable/full-resident, and no preload or additional expert pool is allocated. The online state uses 88,088 bytes of fixed metadata. Route membership is snapshotted before mutation, all current route IDs remain pinned through admission, hits become MRU in rank order, and misses are admitted in rank order only after complete reads. Residency changes do not alter routing, packed expert bytes, or Q4 kernels. Experimental LRU is rejected at startup when prefetch, SMELT capacity below six, or any batch/DSpark path is enabled.

A serial one-run implementation check on Apple M4 Pro 48GB used `NATIVE_SMELT_N=20`, all batch/DSpark/prefetch flags off, and separate processes for fixed-hot and LRU. These artifacts are local evidence, not promotion evidence:

- fixed-hot: `/tmp/dmlx-cache-fixed-final.json`, SHA-256 `a0a5a9b3d2666ff3284800c578d02bfe7760d18a18b326daacf1aeeba23e729d`;
- bounded LRU: `/tmp/dmlx-cache-lru-final.json`, SHA-256 `fdce1eb55689544bed84e075627fbc8bf8dade85c13396dc5d7bd22bc8555e23`.

| Online metric (three standard 5-token requests) | Fixed-hot N=20 | Bounded LRU N=20 | Direction |
|---|---:|---:|---:|
| Expert bytes read | 70,269,272,064 | 32,527,613,952 | -53.711% |
| Hot hits | 936 | 3,759 | +301.603% |
| Cold misses / preads | 5,256 | 2,433 | -53.711% |
| Expert I/O wait | 5,832.405 ms | 5,558.939 ms | -4.689% |
| Slot admissions / evictions | 0 / 0 | 2,433 / 2,433 | bounded churn |
| Peak RSS | 20,261,273,600 | 18,996,559,872 | -1,264,713,728 bytes |
| Standard median | 1.283451 tok/s | 0.597782 tok/s | -53.424% |
| Separate 100-token diagnostic | 1.912546 tok/s | 1.007435 tok/s | -47.325% |

Both runs passed Paris and the 7/7 E2E suite with identical displayed standard response text, retained exactly 1,568 resident slots and 20,963,131,392 resident expert bytes, and left no `dmlx serve` process. The online byte/hit direction agrees with replay and RSS remains below the 48GB machine budget, so Task 6.4 establishes that the bounded policy is implementable and observable. However, wall-clock throughput regressed severely despite lower physical bytes. The flag therefore remains default-off and is **not promotable**. This single check is not the repeated matched A/B required by Task 6.5; it is already sufficient to make 2.0/2.5 tok/s unsupported for this implementation and to trigger the rollback/default-retention gate unless a separately justified implementation removes the online replacement cost.

Validation completed: V0 ReleaseFast build; kernel and MLA gates; V1 `SMOKE PASS`; V4 `BATCH ORACLE PASS`; V2 fixed-hot and opt-in LRU artifacts; `zig build test`; benchmark schema self-test; JSON parsing; shell syntax; `git diff --check`; and final single-server cleanup.

## Task 6.5 repeated online A/B and promotion decision

The promotion gate ran two independent, strictly serial pairs in fixed1/LRU1/fixed2/LRU2 order on Apple M4 Pro 48GB. Every child used the full-content model fingerprint `807672b318f49dd74567d69a603dbf35434c8d71c900f246150d79a18bc1b6d0`, N=20, `SKIP_PURGE=1`, prefetch/DSpark/all batch flags off, identical 5+4 warmup, and a separate process. The aggregate binds every source by absolute path, file SHA-256, timestamp and timestamp-independent canonical run SHA-256, and rebuilds exactly from those sources.

| Repeated metric | Fixed-hot N=20 | Bounded LRU N=20 | LRU direction |
|---|---:|---:|---:|
| Standard tok/s values | 1.177255, 1.238016 | 0.669182, 0.680901 | both paired runs slower |
| Standard median / spread | 1.207636 / 0.060761 | 0.675042 / 0.011719 | -44.102% median |
| Long diagnostic median / spread | 1.857532 / 0.214932 | 1.124937 / 0.033237 | -39.439% median |
| ITL p50 median | 392.796 ms | 722.754 ms | +84.002% |
| ITL p95 median | 473.613 ms | 825.921 ms | +74.387% |
| Expert bytes / completion token | 4,684,618,137.6 | 2,168,507,596.8 | -53.710% |
| Expert bytes read median | 70,269,272,064 | 32,527,613,952 | -53.710% |
| Hit-rate median | 15.116% | 60.707% | replay direction confirmed |
| Expert I/O wait median | 6,027.972 ms | 5,538.980 ms | -8.112% |
| Admissions / evictions median | 0 / 0 | 2,433 / 2,433 | bounded churn |
| Peak RSS maximum | 21,078,016,000 | 19,284,639,744 | below 44 GiB gate |

All four runs passed Paris, the 7/7 E2E suite and unit tests. Every run retained 1,568 resident slots, 20,963,131,392 resident expert bytes and 88,088 metadata bytes, and no `dmlx serve` remained. The byte, hit-rate, I/O-wait, geometry, correctness and RSS gates pass, but LRU fails the 2.0 tok/s target, fails both paired improvement checks, and regresses the repeated standard median. It also misses the 2.5 tok/s stretch target. The evidence-based Task 6.5 decision is therefore **do not promote**: fixed-hot N=20 remains authoritative and default, while `NATIVE_EXPERIMENTAL_CACHE=1` remains diagnostic-only.

Retained evidence:

- aggregate: `docs/en/analysis/performance-cache-promotion.json`, SHA-256 `bc9ba24895116df11635daaedd7585794a5a3284523084429f10d2178b0f7e18`;
- fixed run 1: SHA-256 `049319ff116eea3a1e07b0e50b32a44b038e06db901402f63e147d40251d570c`;
- LRU run 1: SHA-256 `549ec01482222991ac40e86d42c4f9c272264e6799cb5978229d0c88864ff3db`;
- fixed run 2: SHA-256 `8c5c2ef247757c1774447d1d7e914a9ca09472d40ed8964bd4bcf5c3caaba904`;
- LRU run 2: SHA-256 `1d703be193a7805d697e3db14419050ddcbc23b70bc0585731355ec8f4b9eeb5`.

## Rejected paths

| Path | Decision and reason |
|---|---|
| LFU | Do not implement alongside LRU. It reads 22.5% more bytes than LRU and churns 3,363 more slots in aggregate. |
| Trust-OS/no-user-cache | Reject for this model: it saves user-cache memory but increases application-visible bytes by 18.425%. |
| Static-frequency oracle | Reject online: its membership uses the evaluated future. It is a ceiling, not a deployable winner. |
| Belady | Reject online: future route knowledge is unavailable. It is only a lower bound. |
| Full preload | Prohibited by R6.7 and the 48 GiB budget; estimated expert residency is roughly 147 GB. |
| N=51 or another high-memory default | Reject as a default. It is outside formal geometry and leaves unsafe headroom for model, Metal, OS and page cache. |
| Predictive/background `pread` | No supporting comparison in this replay. Do not revive before its own trace-driven policy and contention evidence. |
| Runtime routing penalty or bias | Reject: it changes route selection rather than residency and threatens exact-Q4 output invariance. |
| Large/global dynamic LRU | Reject. The selected policy is bounded per layer at the existing capacity; unbounded or higher-capacity LRU has no memory evidence. |
| Cold-expert `mmap` | No evidence here; page-fault behavior is outside replay and prior experiments are not a promotion basis for DSV4. |
| `F_RDADVISE` or `dispatch_io` | No trace/replay win for this corpus; do not re-enter the default route under R6.6. |
| Treat batching as the decode solution | Reject. Batch/DSpark promotion failed its own gate and must remain separate from the exact-Q4 cache A/B. |

## Independent reproduction

Validate provenance and regenerate the formal artifact without starting a server:

```bash
python3 scripts/capture_routing_corpus.py \
  --validate-metadata /tmp/dmlx-routing-traces/corpus-metadata.json
python3 scripts/replay_routing_trace.py \
  --corpus-metadata /tmp/dmlx-routing-traces/corpus-metadata.json \
  --smelt-n 20 \
  --full-resident-layers 3 \
  --format json \
  --output /tmp/dmlx-routing-traces/cache-policy-replay.json
```

Independently recompute aggregate integer metrics from case-level results:

```bash
python3 - <<'PY'
import json
from collections import defaultdict

path = "/tmp/dmlx-routing-traces/cache-policy-replay.json"
artifact = json.load(open(path, encoding="utf-8"))
sums = defaultdict(lambda: defaultdict(int))
fields = ("expert_routes", "hits", "misses", "read_bytes", "slot_churn")
for case in artifact["cases"]:
    for policy in case["policy_comparison"]["policies"]:
        for field in fields:
            sums[policy["name"]][field] += policy["totals"][field]
for name, totals in sorted(sums.items()):
    totals["hit_rate"] = totals["hits"] / totals["expert_routes"]
    print(name, dict(totals))
PY
```

The recomputed sums must match `aggregate.policies[*].totals` exactly before this decision is reused.
