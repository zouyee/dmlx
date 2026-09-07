# RFC: Calibration-Guided Mixed 2/4-Bit Routed Experts

- **Status:** Proposed for dsv4-native-roadmap Task 7.1; implementation may begin at Task 7.2 only after review.
- **Decision:** Specify a deterministic, quality-gated, asymmetric 2/4-bit expert format and workflow. **The existing Q4 MXFP4 gs=32 path remains authoritative and the default.** Mixed precision is opt-in and experimental through Task 7.6.
- **Scope:** Routed-expert gate/up/down weights, calibration and importance capture, allocation, conversion/reference decode, artifact validation, offline quality gates, staged native integration, and rollback.
- **Normative language:** MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are requirements as defined by RFC 2119.

## 1. Decision, invariants, and non-goals

This RFC rejects uniform int2 as the replacement policy. The existing uniform int2 gs=64 implementation is only a control and format/decoder baseline. A mixed artifact MUST retain sensitive regions as Q4 and use affine int2 only where calibration and held-out gates permit it.

The target geometry is immutable for format major version 1:

| Property | Value |
|---|---:|
| Transformer layers | 43 |
| Routed experts per layer | 256 |
| Active routed experts per token | K=6 |
| Hidden dimension | 4096 |
| Expert intermediate dimension | 2048 |

For every layer, experts MUST be ordered by expert ID `0..255`. Within an expert, components MUST be ordered `gate`, `up`, `down`. Logical matrices are row-major with the input/K dimension contiguous:

| Semantic role | Common alias | Logical shape `[rows, input_columns]` | Operation |
|---|---|---:|---|
| gate projection | w1 | `[2048, 4096]` | `gate(x)` before activation |
| up projection | w3 | `[2048, 4096]` | `up(x)` multiplied by activated gate |
| down projection | w2 | `[4096, 2048]` | projects intermediate to hidden |

Changing layer count, expert count, shapes, component order, or K requires a new format major version. K is execution metadata rather than a stored matrix dimension, but MUST be recorded and validated as 6.

Non-goals are quantizing shared experts, attention, router weights, KV cache, or non-expert weights; changing routing or model numerics; supporting chat/MLX production paths; claiming speedup before measurement; and replacing, rewriting, or ambiguously auto-detecting Q4. Version 1 supports only 2-bit and 4-bit blocks. Three-bit encoding MUST NOT be emitted or accepted unless a later RFC defines an encoding and increments the compatible format version.

## 2. Existing formats are baselines, not the mixed contract

### 2.1 Authoritative Q4 MXFP4 gs=32

Q4 groups are 32 consecutive input-column values within one row. Each value is an E2M1 nibble with decode table

`[+0, +0.5, +1, +1.5, +2, +3, +4, +6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]`.

Element `2i` occupies the low nibble and element `2i+1` the high nibble of byte `i`. Equivalently, eight consecutive values occupy successive low-to-high nibbles of a little-endian `u32`. Each group has one unsigned E8M0 scale byte and no additive bias:

`decode(q, s) = E2M1[q] * 2^(int(s) - 127)`.

The E8M0 exponent bias MUST be **127, never 128**. Retained Q4 blocks in a mixed artifact MUST preserve the source nibble and scale bytes exactly; the converter MUST NOT requantize them. This exact-byte retention avoids introducing conversion drift but does not imply that mixed-model logits equal Q4 logits.

### 2.2 Uniform affine int2 gs=64 prototype

The current prototype partitions each row into 64-column groups. For finite weights `w`, it starts from `b=min(w)` and `s=(max(w)-min(w))/3`, rounds `s` to the nearest power of two, stores `s_byte=clip(round(log2(s))+127,0,255)`, round-to-nearest-even converts `b` to BF16, reconstructs both, then computes

`q = clamp(round_to_nearest_even((w - b_bf16) / 2^(s_byte-127)), 0, 3)`,

with decode `b_bf16 + q * 2^(s_byte-127)`. The bias plane is little-endian BF16. Sixteen consecutive q values are packed into one little-endian `u32`; value `i` occupies bits `[2i,2i+1]`. Thus each byte stores four values in least-significant-two-bits-first order.

This uniform prototype and its sidecar manifest are not format v1. They lack strict identity, bounds, and checksum binding and MUST NOT be interpreted as mixed artifacts. Its behavior for zero-range/nonfinite groups is superseded by Section 7.

## 3. Reproducible calibration corpus

Calibration and held-out manifests MUST be immutable, version-controlled metadata (content may reside separately for privacy), disjoint by prompt identity and provenance, and frozen before importance capture. Evaluation prompts, expected answers, held-out logits, and any future benchmark data MUST NOT influence importance statistics, allocation, hyperparameters, thresholds, retries, or manual exceptions. Duplicate and near-duplicate detection SHOULD operate on normalized private text before hashing; any collision across splits MUST fail corpus validation.

A canonical JSON manifest MUST use UTF-8, sorted object keys, compact separators, no insignificant whitespace, integers for integral values, and lowercase hexadecimal SHA-256. Entries remain in explicit array order; tools MUST NOT sort them implicitly. Schema:

```json
{
  "schema": "dmlx.calibration-corpus.v1",
  "corpus_id": "...",
  "split": "calibration|heldout",
  "entries": [{
    "ordinal": 0,
    "id": "stable-public-id",
    "bucket": "short-prefill|long-prefill|decode",
    "language": "BCP-47",
    "domain": "math|code|factual|general|...",
    "privacy_sha256": "sha256(salt_id || canonical_private_prompt)",
    "token_ids": [1, 2],
    "decode_steps": 0,
    "source_license_id": "..."
  }],
  "tokenizer_sha256": "...",
  "model_identity_sha256": "...",
  "tool_git_commit": "40-hex",
  "tool_tree_sha256": "...",
  "config_sha256": "...",
  "salt_id": "non-secret-versioned-id",
  "seeds": {"corpus": 0, "capture": 0, "search": 0},
  "commands": ["..."]
}
```

Exact token IDs are normative; prompt retokenization MUST NOT occur during capture. The private prompt and salt MUST NOT be stored in public artifacts. `model_identity_sha256` MUST bind the complete Q4 manifest and all source layer payload hashes; tokenizer identity MUST bind tokenizer files and configuration. The manifest hash MUST be recorded by every derived artifact.

Minimum calibration and independently sourced held-out coverage MUST each include: (1) short prefill, 1-128 input tokens; (2) long prefill, at least 1024 tokens and including the maximum validated capture length; (3) autoregressive decode, at least 30 generated positions per selected prompt; (4) multilingual data spanning at least English, Simplified Chinese, and one additional language; (5) math, code, factual recall, and general prose; and (6) routing-diverse samples. Coverage is sufficient only if every layer routes at least one token to every expert or the rare-expert fallback in Section 4 is applied. The corpus report MUST publish counts by split/bucket/language/domain, token-length quantiles, per-layer/expert route counts, and hashes, without prompt text.

The Task 7.2 implementation MUST provide a deterministic manifest validator and a capture command equivalent to the planned interfaces in Section 11. Any corpus amendment creates a new corpus ID/hash and invalidates all downstream imatrix, allocation, conversion, and quality artifacts.

## 4. Imatrix capture and deterministic aggregation

For every processed token and layer, capture the six routed expert IDs and their post-router weights in execution order. For each routed `(layer l, expert e, component c, input column j)`, accumulate activation importance. Version 1 MUST use diagonal activation energy

`A[l,e,c,j] += p(l,e,t) * x(l,e,c,t,j)^2`,

`W[l,e,c,j] += p(l,e,t)`, and `I=A/max(W, epsilon)`.

Here `p` is the actual nonnegative routing weight consumed by the MoE operation. For gate and up, `x` is their shared hidden input; for down, `x` is the activated-gate-times-up intermediate input. This is an empirical diagonal Hessian proxy; tools MUST label it `activation_energy_diag_v1`, not an exact Hessian. An implementation MAY additionally emit unweighted energy and row/group summaries, but allocation MUST identify exactly which statistic it uses.

Capture MUST record routed IDs/weights before expert loading, and MUST verify IDs are in `[0,255]`, exactly six distinct IDs are present per layer/token, weights and activations are finite, and route weights use the same dtype/normalization as authoritative Q4 execution. Violations fail capture.

Determinism requirements:

- Accumulators MUST be IEEE float64 on CPU. Each sample, token, layer, route slot, component, and column MUST be reduced in that lexicographic order. Parallel workers MAY produce per-entry shards, but final merge MUST follow manifest ordinal then tensor index; unordered atomics are forbidden.
- Counts MUST be uint64 with checked overflow. Route-weight sums and squared-energy sums MUST reject NaN/Inf and overflow.
- Final normalization MUST occur once after merging. `epsilon` MUST be the smallest positive normal float64 only as a division guard; a zero weight produces `I=0` and an explicit uncovered flag.
- Optional per-row/per-group statistics MUST be derived deterministically from column accumulators or accumulated with the same order and dtype.
- The artifact MUST bind corpus, model, tokenizer, capture executable/tree, compiler/runtime, capture config, ordered shard, and source-Q4 hashes.

An expert is rare in a layer when it has fewer than 128 routed calibration tokens or route-weight sum below `1e-4` of that layer's total. Uncovered and rare experts MUST default entirely to retained Q4. A per-component group with fewer than 128 contributing routed tokens also MUST remain Q4. No global floor may fabricate observations. Allocation MAY use `I_floor = percentile_1(nonzero I in the same layer/component)` only for numerically zero observed columns; this floor MUST NOT relax the rare/uncovered Q4 fallback.

The imatrix file MUST include schema/version, dimensions, statistic name, normalization, thresholds, ordered accumulator semantics, coverage flags, and SHA-256 for every source and payload. Two runs from identical inputs MUST be byte-identical.

## 5. Asymmetric allocation search

### 5.1 Allocation unit and candidates

Version 1 allocation units are contiguous 64-column groups within one matrix row. This aligns int2 gs=64. A Q4 allocation unit contains two retained source MXFP4 gs=32 blocks. Each unit is homogeneously either:

- `MXFP4_E2M1_E8M0_GS32`, bit width 4, two groups, exact retained bytes; or
- `AFFINE_UINT2_E8M0_BF16_GS64`, bit width 2, one group.

Segments in the allocation table MUST be maximal contiguous runs of same-class row/groups and MUST NOT cross row, component, expert, or layer boundaries. Gate, up, and down are scored and protected independently; no tensor-wide bit choice is allowed. Allocation maps MUST nevertheless identify every row/group explicitly after segment expansion.

### 5.2 Objective, exact serialized cost, budget, and controls

The converter MUST first compute candidate int2 reconstructions deterministically and a weighted distortion for each unit `u`:

`D_u = sum_(r,j in u) I[c,j] * (Q2(w[r,j]) - Q4(w[r,j]))^2`.

If row-specific statistics are captured, the declared statistic MAY replace broadcast column importance. The coordinate of a unit is `(layer, expert, component, row, group)`, with component order `gate < up < down`; this tuple's ascending order is the canonical coordinate order.

An expanded allocation map is the encoding ID for every unit in canonical coordinate order. Its canonical serialized form is the compact canonical JSON object `{"schema":"dmlx.mixed2-allocation-map.v1","geometry":<immutable geometry object>,"units":[<encoding IDs in canonical coordinate order>]}` using Section 3 JSON rules. Its SHA-256 is the allocation-map identity. For each layer, serialization MUST coalesce the expanded map into maximal same-encoding segments that are contiguous in group coordinate and do not cross a row, component, expert, or layer. Segment order is canonical coordinate order. There is exactly one table entry per segment. Implementations MUST NOT persist a nonmaximal segmentation.

Define `ArtifactBytes(S)` as the exact sum of canonical `manifest.json` byte length and all 43 complete layer-file lengths produced from allocation state `S` by Section 7, including headers, identity/hash records, table entries, table-to-payload padding, inter-entry payload padding, and record bytes. It excludes the containing directory's filesystem metadata and allocation/search sidecars. Candidate costing uses an **allocation-cost projection**: set `identities.allocation_map`, `identities.allocation_trace`, and `identities.threshold_set` to `Z`; set `qualification.report_sha256=Z` and `qualification.status="UNQUALIFIED"`; all other identities are their frozen input values. Candidate map bytes themselves are excluded as sidecars. Implementations compute exact table counts, offsets, lengths, padding, manifest length, and layer lengths but MAY length-model record data and fixed-width hashes rather than materialize bytes; both methods MUST return the same integer. Because decimal byte counts in `manifest.json` can change its own byte length, construction MUST iterate manifest serialization and layer-length calculation until all stored lengths are unchanged; fixed-width hashes cannot change length. Failure to reach a fixed point within 16 iterations is an implementation error. `ArtifactBytes` is the only allocation cost; payload-only or fixed per-unit proxies are forbidden.

For state `S` and eligible int2 unit `u`, form `S+u` by promoting `u` to Q4, then fully coalesce and serialize it. The exact marginal cost is `delta(u,S) = ArtifactBytes(S+u) - ArtifactBytes(S)`. Every selection iteration MUST simulate every remaining eligible promotion from the current state and recompute all deltas; cached priorities from a prior state MUST NOT be used. This captures segment splits/merges, table-count changes, JSON digit changes, and alignment padding.

Mandatory-Q4 units are rare/uncovered units, units whose int2 candidate is nonfinite, and independently predeclared sensitive units. Apply them to the uniform-int2 map in canonical coordinate order to form initial state `S0`; if `ArtifactBytes(S0) > B`, allocation fails. At each iteration partition candidates by delta. If any candidate has `delta <= 0`, select among those by larger `D_u`, then canonical coordinate ascending, and promote it if the candidate size is `<= B`. Otherwise rank positive-delta candidates by descending exact float64 `D_u/delta`, then larger `D_u`, then smaller `delta`, then canonical coordinate ascending. Floating comparisons use stored IEEE float64 values without epsilon adjustment; nonfinite `D_u`, ratio, or serialized cost fails allocation. If the selected candidate would exceed `B`, stop immediately: no lower-ranked candidate is substituted. After every promotion, recompute the full candidate set. The result therefore cannot exceed `B` but is intentionally greedy rather than a knapsack optimum.

The search trace MUST be canonical JSON and record for every iteration: iteration number, current `ArtifactBytes`, every candidate coordinate, `D_u`, `delta`, candidate total bytes, resulting segment count, selected coordinate, selection class (`nonpositive-delta` or `positive-ratio`), and post-selection bytes; a terminal record gives `budget-reached`, `next-selected-exceeds-budget`, or `no-candidates`. Each float64 is a JSON string of exactly 16 lowercase hexadecimal digits representing the IEEE-754 binary64 bit pattern in big-endian/network digit order; `+0` is `"0000000000000000"`, and `-0` MUST be canonicalized to `+0`. The expanded map, coalesced tables, and trace MUST each be SHA-256 hashed.

The search inputs MUST predeclare exact maximum artifact bytes `B`. Controls are all-Q4 and uniform int2 serialized through the identical v1 container. Mixed MUST satisfy `bytes_mixed = ArtifactBytes(S_mixed) <= B`, `bytes_mixed < bytes_Q4`, and `bytes_mixed <= 0.750 * bytes_Q4`; all accounting includes manifest, every header/table, and all padding. Section 7's 34-byte versus 19-byte contiguous records make this target arithmetically possible, but this RFC makes no claim that quality gates are attainable at that size.

The matched-byte control is constructed independently, not with distortion ranking. Start from uniform int2 plus the same mandatory-Q4 units, then visit every remaining unit exactly once in canonical coordinate order. For each unit, simulate promotion, maximal coalescing, and complete `ArtifactBytes`; accept it whenever candidate bytes are `<= bytes_mixed`, including when its delta is zero or negative, otherwise leave it int2 and continue. Recompute serialized cost after every accepted or rejected trial. After all coordinates have been visited, the output MUST be the **terminal expanded allocation map** containing every accepted promotion, and `bytes_matched` MUST be `ArtifactBytes` of that terminal map. An implementation MAY record the greatest accepted byte size diagnostically, but MUST NOT emit, hash, evaluate, or substitute the earlier maximum-size map. No alternative combination search is permitted. Require `bytes_mixed - bytes_matched <= max(34, ceil(0.001 * bytes_mixed))` bytes; 34 is one Q4 record. If the tolerance is unattainable, matched-control comparison is `UNATTAINABLE` and Task 7.3 fails rather than silently changing the control.

Task 7.2 MUST include this normative matched-control allocator fixture. Synthetic promotable coordinates `u0=(0,0,gate,0,0)`, `u1=(0,0,gate,0,1)`, and `u2=(0,0,gate,0,2)` are visited in that order from an empty accepted-promotion set; all other geometry is fixed fixture context. Set `bytes_mixed=120`. Exact reserialization starts at 100 bytes and, after accepting each promotion, yields `100 -> 120 -> 112 -> 119`: `u0`, `u1`, and `u2` are all accepted, with `u0` establishing the earlier diagnostic maximum of 120, `u1` exercising the accepted negative delta to 112, and `u2` producing the 119-byte terminal state. The expected accepted-coordinate sequence is `[u0,u1,u2]`; the terminal expanded map has Q4 at all three coordinates and preserves the fixture's declared encoding at every other coordinate; its coalesced table is one Q4 run covering groups `0..2`; and `bytes_matched=119`, not 120. The fixture MUST assert the accepted order, every trial byte count, terminal expanded map, terminal coalesced table, and their canonical hashes.

No calibration retries may inspect held-out/evaluation results. Thresholds, budget, sensitive-unit rule, importance definition, and tie rules MUST be frozen before held-out evaluation. Task 7.3 either passes the unchanged set or stops Wave 7. A failed held-out gate MUST remain recorded; thresholds MUST NOT be relaxed, prompts removed, or data added after seeing held-out results.

## 6. Predeclared quality gates and stop logic

All gates below are mandatory engineering risk budgets, not empirical claims. Conversion/reference-decode gates run before model evaluation. Q4, uniform int2, matched control, and mixed MUST use identical held-out token IDs and settings. Report raw values and deterministic aggregates; do not describe mixed logits as exact Q4.

### 6.0 Canonical metric and matching conventions

All metric inputs and intermediate operations MUST be finite IEEE binary64 unless a failure rule below says otherwise. The normative evaluator rounds to binary64, round-to-nearest ties-to-even, after every addition, subtraction, multiplication, division, `sqrt`, `exp`, natural `log`, and `log10`; `log10` is one correctly rounded operation and MUST NOT be implemented as `log(x)/log(10)` unless the resulting bit is identical. FMA/contraction, reassociation, unordered parallel reduction, and extended-precision retention across operations are forbidden. Task 7.2 MUST provide a pinned reference evaluator that computes each elementary function in MPFR at sufficient precision and then rounds once to binary64; its source, executable, source-tree, compiler/toolchain, MPFR source/build, and configuration SHA-256 values MUST be bound by both `metric_spec` and `threshold_set`. Production results MUST match its frozen bits; host `libm` or platform operations are permitted only when they do. This is a result contract, not a claim that Python, C, or any platform `libm` is universally correctly rounded.

A reduction accumulator starts at binary64 `+0`. Tensor reductions iterate flattened coordinates lexicographically by layer, component (`gate`, `up`, `down`), expert, row, group, then element/input-column. For each coordinate, form all subtractions and squares with a rounding after each operation; form the weight-times-square with one rounded multiplication; then perform exactly one rounded binary64 addition into the accumulator. Scalar per-position reductions use position then unsigned token ID; aggregate reductions use sample-manifest ordinal then position. Mean top-5 overlap adds each per-position binary64 overlap in that order and performs one final rounded division by the binary64 count. Every mean elsewhere follows the same add-then-final-divide schedule. Threshold comparisons compare the final binary64 metric directly with the binary64 threshold encoded by the `threshold_set`; `<=` and `>=` are inclusive, with no decimal rerounding, epsilon, or tolerance.

For a nonempty list `v` of length `n`, nearest-rank percentile `p` sorts ascending by numeric value (stable original index for equal values), uses one-based rank `r=max(1,ceil(p*n))`, and returns sorted element `r`; `p` is in `[0,1]`. Empty input fails. Thus p1 means `p=0.01` and p99 means `p=0.99`; no interpolation occurs. Section 4's `percentile_1` uses this rule.

Top-k sorts token IDs by descending logit, then ascending unsigned token ID for equal logits, and takes the first `k`. Argmax is Top-1. Stable softmax for logits `z[0..V-1]` computes `m=max(z)` in token-ID order, `e_i=exp(z_i-m)` in token-ID order, `s=sum_i e_i` in token-ID order, and `p_i=e_i/s`; nonfinite input, nonpositive/nonfinite `s`, or nonfinite output fails.

Define `xlogy(x,q)` as `0` when `x==0` (including `q==0`), `x*log(x/q)` when `x>0` and `q>0`, and failure when `x>0` and `q==0` or either argument is negative/nonfinite. With `M_i=(P_i+Q_i)/2`, `JS(P,Q)=0.5*sum_i xlogy(P_i,M_i)+0.5*sum_i xlogy(Q_i,M_i)`, each sum in token-ID order. Identity distributions therefore produce exactly `+0.0` under the reference implementation; any invalid summand fails.

For source vector `x` and candidate `y`, centered logit NRMSE computes means `mx=sum(x)/n`, `my=sum(y)/n`, numerator `sqrt(sum(((y_i-my)-(x_i-mx))^2)/n)`, and denominator `max(sqrt(sum((x_i-mx)^2)/n),1e-12)`, all in index order; result is numerator/denominator. Aggregate mean reduces per-position values in position order. Aggregate percentiles use nearest-rank above.

Answer normalization uses the immutable **Unicode 15.1.0** profile. NFKC is exactly Unicode Standard Annex #15 for Unicode 15.1.0. Full default case folding uses Unicode 15.1.0 `CaseFolding.txt`, applying status `C` and `F` mappings (with `F` taking precedence where present) and never Turkic status `T`. Whitespace is exactly the `White_Space` property from Unicode 15.1.0 `PropList.txt`. Letter/number boundaries are General_Category values `L*` or `N*` from Unicode 15.1.0 `UnicodeData.txt` and `DerivedCoreProperties.txt`; underscore remains an explicit boundary character. Apply NFKC, full folding, replacement of each maximal `White_Space` sequence by one ASCII space, and trimming, in that order; then replace every standalone decimal-integer token matching `(?<![L*N*_])[+-]?[0-9]+(?![L*N*_])`, where `L*`, `N*`, and `_` denote the frozen classifications, by its mathematical base-10 integer (`-0` and `+0` become `0`, leading zeroes removed). Required text matches only as a complete token sequence bounded on both sides by start/end or a character outside `L*`, `N*`, and underscore. A required integer matches one normalized integer token exactly. Punctuation is not removed. Invalid Unicode fails.

Task 7.2 MUST vendor either the exact Unicode 15.1.0 source files and generated runtime table or the exact source files plus a reproducibly generated table. The normative source set MUST include `UnicodeData.txt`, `DerivedCoreProperties.txt`, `PropList.txt`, `CaseFolding.txt`, `DerivedNormalizationProps.txt`, `CompositionExclusions.txt`, and `NormalizationTest.txt`. A canonical Unicode manifest MUST record Unicode version, source URL/path, and lowercase SHA-256 for every source; generated-table SHA-256; and generator source-tree, executable, toolchain, and configuration SHA-256. Both `metric_spec` and `threshold_set` identities MUST bind that manifest and every listed identity. Runtime/platform Unicode APIs MUST be rejected unless their complete effective data-table hash is identical to the generated-table hash. The official Unicode 15.1.0 `NormalizationTest.txt` conformance suite MUST pass bit-for-bit before artifacts are produced.

For a larger suite of `N` prompts, `Q_i,M_i in {0,1}` are Q4 and mixed exact-match passes. Overall scores are `sum Q_i/N` and `sum M_i/N`; mixed score delta in percentage points is `100*(sum M_i-sum Q_i)/N`. Paired pass count is `sum(Q_i*M_i)` and MUST be `>= ceil(0.95*sum Q_i)`; if `sum Q_i=0`, the suite is invalid rather than vacuously passing. For each predeclared stratum `s`, denominator `N_s` is the count of suite entries assigned to that stratum before execution; it MUST be nonzero, and delta is `100*(sum_s M_i-sum_s Q_i)/N_s`. Multi-label entries contribute once to each declared stratum and once overall.

Task 7.2 MUST self-test these normative semantic vectors before producing artifacts:

| Convention | Input | Expected result |
|---|---|---|
| nearest-rank | `[9,1,5,3]`, p=`0.01,0.50,0.99` | `1,3,9` |
| top-k tie | logits `[2,2,1,2]`, token IDs `0..3`, k=3 | `[0,1,3]` |
| normalization | `"  ＰＡＲＩＳ\t"` / required `"paris"` | match |
| full folding | `"Straße"` / required `"strasse"`; `"Σςσ"` / required `"σσσ"` | match; match |
| whitespace | `"alpha\u00a0\u00a0beta"` / required `"alpha beta"` | match after NBSP collapse |
| Unicode boundary | `"éparis"` / required `"paris"`; `"paris٢"` / required `"paris"`; `"(paris)"` | no match; no match; match (`é` is `L*`, `٢` is `N*`) |
| integer normalization | `"answer: −000"` is not ASCII-minus grammar; `"answer: -000"` | first does not match integer `0`; second matches `0` |
| paired rule | Q4 passes `100`, paired passes `94,95` | fail, pass (`ceil(95)=95`) |

The following numeric fixture inputs are the exact binary64 encodings of the shown integers/decimals. Every expected value is exactly 16 lowercase hexadecimal digits in the Section 5.2 bit convention. The checked-in Task 7.2 reference calculation MUST spell out the operation schedule above, use MPFR then binary64 rounding for elementary operations, print these bits, and be independently rerun by a second implementation before its source and result hashes are frozen.

| Numeric path | Input / operation | Expected binary64 bits in operation order |
|---|---|---|
| stable softmax | logits `[1,2,3]` | `m=4008000000000000`; `e=[3fc152aaa3bf81cc,3fd78b56362cef38,3ff0000000000000]`; `sum=3ff80d2ae2032c08`; `P=[3fb70c3e5f682bda,3fcf534335ca4bcf,3fe549a766a06790]` |
| positive `xlogy` | `xlogy(0.25,0.5)` | `bfc62e42fefa39ef` (the separate `x=0` branch is `0000000000000000`) |
| nonidentity JS | `P` above, `Q=reverse(P)` | `M=[3fd82b2f328d6d0b,3fcf534335ca4bcf,3fd82b2f328d6d0b]`; P terms `[bfc085d3ff6010d3,0000000000000000,3fd81b65a338cfaa]`; Q terms reversed; each sum and final JS `3fcfb0f747118e81` |
| centered NRMSE | `x=[1,2,4]`, `y=[1.5,1.5,5]` | `mx=4002aaaaaaaaaaab`; `my=4005555555555555`; numerator-square sum `3ff2aaaaaaaaaaab`; denominator-square sum `4012aaaaaaaaaaaa`; numerator `3fe3f49c0b9ad4db`; denominator `3ff3f49c0b9ad4db`; result `3fe0000000000000` |
| weighted NRMSE/SNR | flattened `x=[1,2,4]`, `y=[1.5,1,5]`, `a=[0.5,2,1.5]` | `A=4010000000000000`; weighted error sum `400d000000000000`; weighted source sum `4040400000000000`; RMSE `3fee768d399dc470`; RMS `4006cdb2bbb212eb`; NRMSE `3fd55fd368d22859`; ratio `4007f437f578c348`; SNR dB `40230d2f9016b01e` |
| top-5 overlap mean | overlaps from ordered sets are `[1,0.8,1]` | per-position bits `[3ff0000000000000,3fe999999999999a,3ff0000000000000]`; ordered mean `3feddddddddddddd`; nearest-rank p1 is `3fe999999999999a` |

For every threshold literal, Task 7.2 MUST test the adjacent representable binary64 values `(below, at, above)`. The rows below cover every comparison family; shared literals use the same row. For an upper bound, below/at pass and above fails; for a lower bound, below fails and at/above pass. Absolute-error multiplier 8 uses the same inclusive upper-bound rule after its rounded multiplication. Relative-improvement and percentage-point rows use lower-bound rules. Zero-denominator special cases in Section 6.2 remain mandatory separate failures.

| Threshold use | Below / at / above binary64 bits |
|---|---|
| upper `0.09` weighted NRMSE | `3fb70a3d70a3d709 / 3fb70a3d70a3d70a / 3fb70a3d70a3d70b` |
| lower `20.9151498112135` SNR | `4034ea47420e1a40 / 4034ea47420e1a41 / 4034ea47420e1a42` |
| upper unit `0.35`, `0.75`, multiplier `8` | `3fd6666666666665/3fd6666666666666/3fd6666666666667`; `3fe7ffffffffffff/3fe8000000000000/3fe8000000000001`; `401fffffffffffff/4020000000000000/4020000000000001` |
| lower top-1 `0.98`, mean top-5 `0.97`, p1 top-5 `0.60` | `3fef5c28f5c28f5b/3fef5c28f5c28f5c/3fef5c28f5c28f5d`; `3fef0a3d70a3d709/3fef0a3d70a3d70a/3fef0a3d70a3d70b`; `3fe3333333333332/3fe3333333333333/3fe3333333333334` |
| upper JS `0.002`, `0.020` | `3f60624dd2f1a9fb/3f60624dd2f1a9fc/3f60624dd2f1a9fd`; `3f947ae147ae147a/3f947ae147ae147b/3f947ae147ae147c` |
| upper centered NRMSE `0.060`, `0.150` | `3faeb851eb851eb7/3faeb851eb851eb8/3faeb851eb851eb9`; `3fc3333333333332/3fc3333333333333/3fc3333333333334` |
| lower relative reductions `0.20`, `0.15`; top-1 points `1.0` | `3fc9999999999999/3fc999999999999a/3fc999999999999b`; `3fc3333333333332/3fc3333333333333/3fc3333333333334`; `3fefffffffffffff/3ff0000000000000/3ff0000000000001` |
| upper total-byte ratio `0.75`; paired ratio `0.95` | `3fe7ffffffffffff/3fe8000000000000/3fe8000000000001`; `3fee666666666665/3fee666666666666/3fee666666666667` |

### 6.1 Tensor reconstruction gates

For Q4-dequantized `x_i`, mixed decode `y_i`, and `a_i=max(I_i,0)`, iterate `i` in the flattened canonical tensor-coordinate order from Section 6.0. Let `A` be the ordered sum of `a_i`; `A==0` fails the weighted metric. For each coordinate compute rounded `d_i=x_i-y_i`, rounded `d_i^2=d_i*d_i`, rounded weighted-error term `a_i*d_i^2`, and rounded weighted-source term `a_i*(x_i*x_i)`, then add each term once to its respective accumulator. Define `RMSE_a=sqrt((sum error terms)/A)`, `RMS_a=sqrt((sum source terms)/A)`, and `weighted_NRMSE=RMSE_a/max(RMS_a,1e-12)`, with each displayed operation rounded as Section 6.0 requires. Define `weighted_SNR_dB=20*log10(max(RMS_a,1e-12)/max(RMSE_a,1e-12))`, where division, `log10`, and multiplication each round separately. Unit NRMSE uses uniform binary64 weights and the same schedule.

Every decoded value MUST be finite. Retained Q4 bytes and decode MUST match exactly. For each layer/component, weighted NRMSE MUST be `<=0.090`; weighted SNR MUST be `>=20.9151498112135 dB`, derived without rounding from `20*log10(1/0.09)`. Because the shared floors can diverge only in degenerate near-zero cases, SNR is a diagnostic/consistency check, not an independent risk budget. Across int2 units, nearest-rank p99 unit NRMSE MUST be `<=0.35`, maximum `<=0.75`, and no finite-value absolute error may exceed `8*max(RMS(unit source),1e-6)`.

### 6.2 Held-out logits gates

At each scored position compare float32 logits using Section 6.0. Mandatory aggregate gates are:

| Metric | Mandatory threshold |
|---|---:|
| Argmax/top-1 agreement | `>= 98.0%` |
| Mean top-5 set overlap `|Top5_Q4 ∩ Top5_mixed|/5` | `>= 0.970` |
| Nearest-rank p1 per-position top-5 overlap | `>= 0.600` |
| Mean JS, natural log | `<= 0.0020` |
| Nearest-rank p99 JS | `<= 0.020` |
| Centered logit NRMSE | mean `<=0.060`, nearest-rank p99 `<=0.150` |

Nonfinite values, count mismatch, or route-shape mismatch fail. Relative to matched control, mixed MUST reduce mean JS by at least 20%, reduce mean centered-logit NRMSE by at least 15%, and improve top-1 agreement by at least 1.0 percentage point. Compute each ordered mean first; each reduction then performs one rounded subtraction `control-mixed` followed by one rounded division by `control`. Compute each agreement as one rounded division of ordered integer count converted exactly to binary64 by its count, then compute percentage-point improvement as one rounded subtraction followed by one rounded multiplication by binary64 `100`. Compare only those final bits to the threshold bits. A zero control denominator requires mixed also zero and counts as diagnostic equality, not the required material improvement, so comparison fails. If control top-1 is 100%, mixed MUST also be 100%; otherwise the one-point rule applies. All three are mandatory material-dominance budgets.

### 6.3 E2E gates

The canonical seven use temperature 0, `max_tokens=30`, and Section 6.0 matching:

| ID | Prompt | Required answer |
|---|---|---|
| `arith-add` | `2+2=` | `4` |
| `france-completion` | `The capital of France is` | `Paris` |
| `water-freeze` | `At what temperature does water freeze in Celsius? Answer with a number.` | `0` |
| `earth-round` | `Is the Earth round? Answer yes or no.` | `yes` |
| `arith-mul` | `3*3=` | `9` |
| `arith-sub` | `10-5=` | `5` |
| `france-question` | `What is capital of France?` | `Paris` |

The deprecated 16-case script is supplemental only. Mixed and Q4 MUST each pass 7/7 in the same run. A larger immutable suite of at least 256 prompts MUST satisfy Section 3 strata and remain disjoint. Mixed overall delta MUST be `>= -1.0` point, every stratum delta `>= -3.0` points, and paired passes MUST satisfy Section 6.0. Controls MUST be reported.

### 6.4 Threshold provenance and freeze

No pilot result exists as of this RFC. The thresholds are predeclared engineering risk budgets and MUST NOT be presented as measured feasibility, calibrated confidence bounds, or claims of attainability at 25% saving.

| Threshold family | Source / rationale | Attainability claim |
|---|---|---|
| NRMSE `0.09`; SNR `20.9151498112135 dB` | Declared relative reconstruction-error budget; SNR is mathematically derived and diagnostic | None |
| unit p99/max/absolute-error tails | Declared catastrophic-local-error budgets | None |
| top-1 `98%`, top-5, canonical 7/7, paired `ceil(0.95*Q4 passes)` | Declared token-decision stability budgets | None |
| mean JS `.002`, p99 JS `.020` | Declared average and tail distribution-drift budgets | None |
| centered NRMSE mean/p99 | Declared logit-shape drift budgets | None |
| larger-suite `-1` point overall / `-3` per stratum | Declared task-level regression budgets with explicit denominators | None |
| mixed-over-control `20%/15%/+1 point` | Declared material-dominance budgets to justify asymmetric allocation | None |
| total bytes `<=75%` Q4 | Declared material storage-value budget | Arithmetically possible only; quality feasibility unknown |

Task 7.2 MUST run frozen synthetic and small-fixture feasibility self-tests covering every Section 6.0 semantic, Unicode 15.1.0, numeric intermediate/final-bit, and adjacent-threshold vector; Unicode source/generated-table hashes and official normalization conformance; layout/hash vectors; the Section 5.2 terminal matched-map allocator fixture including its accepted negative delta; and each metric's pass/fail boundary and zero-denominator rule. These test mechanics, not observed model quality, validate implementability. Task 7.3 is the first held-out feasibility test and MUST either pass these unchanged thresholds or stop Wave 7. Thresholds cannot be relaxed, reinterpreted, or replaced after held-out results; a later experiment requires a new RFC/version and new untouched held-out data.

### 6.5 Exact stop logic

Run artifact/conversion integrity, tensor reconstruction, logits, canonical seven, larger suite, then Task 7.6 operational measurements. At first mandatory failure, write a signed/hash-bound report, mark `REJECTED`, and stop Tasks 7.4-7.6 except labeled non-acceptance diagnostics. No averaging, waiver, prompt removal, seed shopping, or post-hoc exception is permitted.

## 7. Artifact format `packed_experts_mixed2_v1`

Mixed artifacts MUST reside in a distinct directory named `packed_experts_mixed2_v1/`. They MUST never overwrite files in the Q4 packed-expert directory. Directory names are not format detection: a caller MUST explicitly opt in and the loader MUST validate the manifest and every required layer before allocating layer payload buffers.

### 7.1 Canonical manifest and acyclic hash DAG

`manifest.json` uses Section 3 canonical JSON. Its complete top-level fields are exactly:

- `schema="dmlx.packed-experts-mixed2.v1"`, `format_major=1`, `format_minor=0`, `endianness="little"`, `alignment=64`, `creation_tool`, and `creation_version`; the last two are nonempty printable ASCII strings of at most 128 bytes;
- `geometry` containing exactly `layer_count=43`, `expert_count=256`, `active_experts=6`, `hidden_dimension=4096`, `intermediate_dimension=2048`, and `component_order=["gate","up","down"]`;
- `identities` containing lowercase SHA-256 strings for exactly `model`, `source_q4`, `converter_executable`, `converter_tree`, `converter_config`, `calibration_corpus`, `imatrix`, `allocation_map`, `allocation_trace`, `metric_spec`, and `threshold_set`;
- `encoding_registry`, exactly the array `[Q4,Q2]`, where `Q4={"id":1,"name":"MXFP4_E2M1_E8M0_GS32","bit_width":4,"logical_group_size":64,"q4_sub_group_size":32,"record_size":34,"scale_dtype":"E8M0_U8","bias_dtype":"NONE"}` and `Q2={"id":2,"name":"AFFINE_UINT2_E8M0_BF16_GS64","bit_width":2,"logical_group_size":64,"q4_sub_group_size":0,"record_size":19,"scale_dtype":"E8M0_U8","bias_dtype":"BF16_LE"}`;
- `layers`, an ordered 43-element array. Each object contains exactly `index`, `path`, `file_size`, `header_length`, `table_offset`, `table_length`, `payload_offset`, `payload_length`, `table_sha256`, `payload_sha256`, `header_sha256`, and `file_sha256`;
- `accounting` containing exactly `total_artifact_bytes`, `manifest_bytes`, `layer_file_bytes`, `header_bytes`, `table_bytes`, `payload_record_bytes`, `payload_padding_bytes`, `table_to_payload_padding_bytes`, `all_q4_artifact_bytes`, `saved_bytes`, `savings_ppm`, `byte_budget`, `layer_encoding_counts`, `component_encoding_counts`, and `encoding_bytes`. All are nonnegative JSON integers. `saved_bytes=all_q4_artifact_bytes-total_artifact_bytes`; `savings_ppm=floor(1000000*saved_bytes/all_q4_artifact_bytes)`. The count maps are arrays of exact records: layer records `{layer,encoding_id,count}` ordered layer then encoding ID, component records `{component,encoding_id,count}` ordered component then encoding ID, and byte records `{encoding_id,record_bytes}` ordered encoding ID;
- `qualification` containing exactly `canonical_seven_id`, `canonical_seven_sha256`, `heldout_suite_id`, `heldout_suite_sha256`, `report_sha256`, and `status`, where status is `UNQUALIFIED`, `QUALIFIED`, or `REJECTED`;
- `manifest_core_sha256` and `manifest_binding_sha256`.

No extension fields are allowed in v1. All generated SHA-256 fields are 64 lowercase hexadecimal characters. The zero hash `Z` is exactly 64 ASCII `0` characters. Paths obey the prior relative-path restrictions: no absolute path, empty component, `.`, `..`, backslash, NUL, scheme, drive prefix, symlink, or directory escape.

The hash graph and projections are acyclic and normative:

1. **Core projection.** Deep-copy the final manifest shape and set every layer's `file_size`, `header_length`, `table_offset`, `table_length`, `payload_offset`, and `payload_length` to integer `0`; set every layer's `table_sha256`, `payload_sha256`, `header_sha256`, and `file_sha256` to `Z`; set scalar `accounting.total_artifact_bytes`, `manifest_bytes`, `layer_file_bytes`, `header_bytes`, `table_bytes`, `payload_record_bytes`, `payload_padding_bytes`, `table_to_payload_padding_bytes`, `all_q4_artifact_bytes`, `saved_bytes`, `savings_ppm`, and `byte_budget` to integer `0`; within each accounting record set only `count` or `record_bytes` to `0` while preserving `layer`, `component`, and `encoding_id`; set `qualification.report_sha256=Z` and `qualification.status="UNQUALIFIED"`; set both top-level `manifest_core_sha256=Z` and `manifest_binding_sha256=Z`. Do not remove fields and do not zero immutable qualification suite IDs/hashes. `manifest_core_sha256 = SHA256(ASCII("DMLX-MIX2-CORE-V1\0") || canonical_json(core_projection))`.
2. **Headers and files.** Write each header with only `manifest_core_sha256`, never the final binding. Generate canonical tables and payloads, calculate region hashes, then layer binding, header hash, and complete file hash as Section 7.2 specifies. This step cannot depend on final manifest-derived fields.
3. **Final manifest.** Fill all layer offsets/lengths/hashes, accounting, and qualification report/status. Set `manifest_core_sha256` to step 1 and `manifest_binding_sha256=Z`; canonicalize the whole manifest and compute `manifest_binding_sha256 = SHA256(ASCII("DMLX-MIX2-MANIFEST-V1\0") || canonical_json(final_manifest_with_only_manifest_binding_zeroed))`. Store that value and write canonical JSON. No header embeds this binding.

Construction order is: immutable manifest skeleton; core projection/hash; coalesced tables and contiguous payload layouts; table/payload hashes; headers with layer-binding field zero; layer bindings; final headers; header/file hashes; accounting and qualification fields; final binding; `manifest.json`. Qualification may transition only by rebuilding the final manifest and binding; layer files remain unchanged because headers bind only the core.

Validation order is: parse canonical manifest and schema; set only `manifest_binding_sha256=Z` and recompute final binding; construct the exact core projection above and recompute core hash; verify each header embeds that core hash; then validate bounds/layout and recompute table, entry, payload, layer-binding, header, and complete-file hashes; finally verify accounting and qualification report/status. Validation MUST finish bounded metadata checks before payload allocation.

### 7.2 Per-layer binary

All integers are little-endian. A layer file consists of the exact 512-byte fixed header below, a packed fixed-width allocation table, one table-to-payload alignment gap, and payload. The v1 header fields, in order, are:

| Field | Type / rule |
|---|---|
| magic | 16 bytes: ASCII `DMLXMIX2V1` then six NUL bytes |
| major, minor | `u16`, exactly `1,0` |
| endian marker | `u32`, exactly `0x01020304` as little-endian bytes |
| header length, table-entry length | `u32`, exactly `512,84` |
| layer index | `u16`, `0..42` |
| component count, expert count, K | `u16`, `3,256,6` |
| table count | `u32`, exact expanded/segmented count declared by manifest |
| table offset/length | `u64`, aligned and bounded |
| payload offset/length | `u64`, aligned and bounded |
| file length | `u64`, exactly OS-reported regular-file size |
| alignment | `u32`, exactly 64 in v1 |
| flags/reserved | flags `u32=0`, then four reserved bytes all zero; this brings the pre-hash prefix to 96 bytes |
| identity hashes | nine raw 32-byte SHA-256 values, in order: `identities.model`, `source_q4`, `converter_executable`, `converter_tree`, `converter_config`, `calibration_corpus`, `imatrix`, `allocation_map`, then `manifest_core_sha256` |
| table SHA-256 | raw 32 bytes over exactly `table_length` bytes |
| payload SHA-256 | raw 32 bytes over exactly `payload_length` bytes, including inter-entry zero padding |
| layer-binding SHA-256 | raw 32 bytes over `ASCII("DMLX-MIX2-LAYER-V1\0") || header_with_only_layer_binding_zeroed || table_region || payload_region` |
| trailing reserved | 32 zero bytes, making `header_length=512` |

The header contains no final-manifest binding. `header_sha256` is SHA-256 of the final `header_length` bytes including the nonzero layer binding; `file_sha256` is SHA-256 of all `file_length` bytes. The table and payload hashes stored in the header MUST equal the final manifest values.

Each table entry MUST contain: expert ID `u16`; semantic role enum `u8` (`0=gate,1=up,2=down`); encoding ID `u8`; bit width `u8`; scale dtype enum `u8`; bias dtype enum `u8`; reserved `u8=0`; rows and columns `u32`; row start/count `u32`; group start/count `u32`; logical group size `u16`; Q4 sub-group size `u16`; payload offset/length `u64`; and entry payload SHA-256. It MUST describe a maximal homogeneous contiguous row/group segment. The ordered table itself is the allocation map representation and MUST sort by expert, component, row start, group start.

Before allocation or mapping, checked arithmetic MUST validate all counts, dimensions, enums, products, sums, offsets, alignment, and lengths against `u64`, `size_t`, configured maxima, manifest values, and file size. Header/table/payload regions and entry ranges MUST be bounded and nonoverlapping. `table_offset=header_length`; `table_length=table_count*table_entry_length`; `payload_offset=align_up(table_offset+table_length,64)`; bytes in that gap MUST be zero. Entry ranges and only their defined post-entry alignment gaps exactly cover the payload region. Expanded entries cover every logical unit once. Regular file type, filename/layer agreement, exactly 43 unique layers, zero reserved fields/padding, known versions/encodings, all hashes, no trailing bytes, and unchanged post-open size are mandatory before payload allocation.

### 7.3 Encodings and edge behavior

`MXFP4_E2M1_E8M0_GS32` stores, for each row and each covered 64-column unit, the first 32-column block's 16 packed nibble bytes then scale byte, followed by the second block in the same order. Row-major unit order is normative. These 34 bytes MUST equal source Q4 bytes exactly. There is no bias. Table fields MUST be bit width 4, group size 64, Q4 sub-group 32, scale dtype E8M0-U8, bias dtype NONE.

`AFFINE_UINT2_E8M0_BF16_GS64` stores each covered row/group record as 16 packed q bytes, one E8M0 scale byte, and one little-endian BF16 bias (19 bytes), in row-major unit order. Quantization and packing follow Section 2.2 with these resolved edge rules:

- Input source values MUST be finite; any NaN/Inf fails conversion.
- If `max==min`, emit `q=0` for all 64 values, bias `BF16_RNE(min)`, and scale byte `127` (scale 1). Decode is the rounded bias; no `log2(0)` occurs.
- Otherwise, compute min/max and `(max-min)/3` in float64 in input index order; choose exponent by round-to-nearest-even of `log2(scale)`, require it in `[-127,128]`, store exponent+127, and fail rather than silently clip an out-of-range finite group.
- Bias MUST be BF16 round-to-nearest-even with preserved sign of zero. Reconstructed E8M0 scale and BF16 bias MUST be used for q calculation. Ties in q use IEEE round-to-nearest-even.

Records are contiguous inside each table entry; there is no per-record padding. For an entry, `record_count = row_count * group_count` with checked arithmetic, where the represented maximal segment has one row in v1 (`row_count=1`) and consecutive 64-column groups; readers MUST reject any other v1 value. `record_size=34` for Q4 and `19` for int2, and `entry.payload_length = record_count * record_size` exactly.

Let `P=payload_offset`. For entries `i=0..table_count-1` in table order, define `entry_start_0=P`; for `i>0`, `entry_start_i=align_up(entry_end_(i-1),64)`; `entry_end_i=entry_start_i+entry.payload_length`; and table `payload_offset` fields equal `entry_start_i`. Thus every entry start is 64-byte aligned. The half-open gap `[entry_end_(i-1),entry_start_i)` is zero padding after the prior entry; it is excluded from every entry payload length and entry payload hash but included in the layer payload hash. There is no terminal padding after the final entry: layer `payload_length=entry_end_last-P` and `file_length=P+payload_length`. For an empty table, which valid model geometry never produces, payload length would be zero.

The entry payload hash covers exactly `[entry_start_i,entry_end_i)`. The layer payload hash covers exactly `[P,P+payload_length)`, including all inter-entry gaps. A decoder iterates `k=0..record_count-1`, reads record `entry_start_i+k*record_size`, and never aligns between records. It advances to the next entry using that entry's stored/algebraically verified aligned start. All table entries themselves are fixed-width and contiguous with no per-entry table padding; only the table-to-payload gap is aligned.

For byte accounting, `table_length=table_count*table_entry_length`, each entry contributes exact record bytes, and padding contributes `entry_start_i-entry_end_(i-1)` plus the table-to-payload gap. The artifact total additionally includes every header and canonical manifest byte. Because each converted unit saves 15 record bytes before dynamic table/padding effects, a 25% artifact saving is possible for sufficiently many int2 units; it is not guaranteed. Converter and decoder MUST reject nonzero padding, offset disagreement, nonmaximal segments, and payload-length disagreement.

## 8. Loader, coexistence, exact-value opt-in, and rollback

Q4 remains authoritative/default. Merely pointing `--expert-packed-dir` at a mixed directory MUST NOT cause format sniffing or selection. Task 7.4 MUST introduce a separate explicit planned option such as `--mixed-expert-packed-dir <packed_experts_mixed2_v1>` together with `--experimental-mixed2`; both MUST be required. Environment-only implicit selection SHOULD NOT be supported. A future stable exact-value selector MAY be `--expert-format q4-mxfp4|mixed2-v1`, with default exactly `q4-mxfp4`; its names and values become API and require separate review.

“Exact-value opt-in” means unknown values MUST fail, aliases MUST NOT be guessed, and selected format MUST match validated manifest encoding. It does not mean numerical equality between mixed and Q4. A malformed, missing, unqualified, unknown-version, unknown-encoding, or corrupt mixed artifact MUST produce a clear error before allocation and MUST NOT silently or automatically fall back to Q4. Explicit Q4 selection remains available as a separate process invocation.

Rollback is operationally simple: remove the experimental mixed flags and use the unchanged Q4 directory/path. Tasks 7.2-7.6 MUST NOT mutate Q4 files or change default selection. Any quality, correctness, integrity, RSS, or performance regression disables mixed publication and restores Q4-only release configuration; no artifact migration is required.

## 9. Security and robustness requirements

Readers MUST treat manifests and binaries as hostile. In addition to Section 7 checks, implementations MUST cap manifest size, table count, allocation count, and layer bytes before allocation; reject duplicate JSON keys, invalid UTF-8, noncanonical numbers, unsupported Unicode/path normalization, short reads, mutable-file size changes between validation and mapping, and hash changes after open. Files SHOULD be opened relative to a directory descriptor with no-follow semantics. Validation MUST complete using bounded metadata buffers before large payload allocation or Metal resource creation. Error messages MUST identify layer/entry/check class without dumping private prompts or raw model data.

## 10. Risks and resolved design questions

| Risk/question | v1 resolution |
|---|---|
| Uniform int2 loses important experts/columns | Calibration-weighted 64-column units; independent component protection; rare units remain Q4 |
| Calibration overfit/leakage | Immutable disjoint calibration/held-out sets; frozen config; no post-held-out tuning |
| Sparse routing leaves experts unseen | Published coverage; `<128` tokens or low route mass forces Q4 |
| Metadata erases savings | Exact `ArtifactBytes` includes manifest, headers, tables, records, and padding; contiguous records preserve int2 savings; minimum 25% total saving |
| Dynamic segment costs change promotion order | Every candidate is fully reserialized and reranked after each promotion; matched control has a separate coordinate scan |
| Arbitrary mixed layouts complicate kernels | Maximal homogeneous row-local segments; aligned entry starts; contiguous 34/19-byte records; only 2/4 bit in v1 |
| Corruption causes unsafe allocation | Acyclic core-header-file-final-manifest hash DAG; exact projection rules; all checks before allocation |
| Metric implementations disagree | Frozen percentile, softmax/top-k, JS, centered NRMSE, answer, paired-pass formulas, and reference vectors |
| Thresholds imply unsupported feasibility | Predeclared engineering risk budgets only; no pilot claim; Task 7.2 self-tests mechanics and Task 7.3 passes unchanged or stops Wave 7 |
| Q4 ambiguity or accidental replacement | Separate directory and explicit dual opt-in; Q4 remains default; no malformed fallback |
| Seven prompts are weak or conflict with 16 cases | Current seven are the authoritative canonical E2E smoke gate; deprecated 16 are supplemental; separate >=256 held-out suite carries coverage |
| E8M0 and zero-range ambiguity | Bias 127; deterministic zero-range int2 record; nonfinite input fails |
| Performance benefit unknown | No claim; Task 7.6 measures bytes/token, throughput, ITL, and RSS after quality passes |

The following remain implementation choices for Task 7.2 but MUST be frozen in its config and hashes: exact public corpus sources/licenses, corpus cardinality above minimum coverage, maximum long-prefill length supported by capture, byte budget beyond the 25% saving floor, and optional row-specific importance capture. They do not block converter/reference-decoder work because format, statistics, allocation unit, deterministic order, and gates are fixed here.

## 11. Staged handoff and acceptance matrix

| Task | Deliverables | Acceptance / stop condition |
|---|---|---|
| **7.2 Converter and reference** | Corpus validator; deterministic imatrix capture/merge; exact-cost allocator and terminal-map matched control; v1 converter; CPU decoder; canonical manifest/binary reader; vendored Unicode 15.1.0 sources/generated tables and hash manifest; pinned MPFR reference evaluator; hash/layout/allocation/Unicode/numeric boundary vectors; corruption/version fixtures | Frozen synthetic/small-fixture self-tests pass; two identical runs byte-match; core/final bindings and region/file vectors match; 34/19-byte contiguous-record layouts and split/merge deltas match; the `100->120->112->119` negative-delta fixture accepts all coordinates and emits terminal bytes 119/map/order; every Unicode source, generated table, generator, evaluator, toolchain, and config hash matches and Unicode/non-ASCII vectors pass; every numeric intermediate/final bit and below/at/above comparison matches; Q4 retained bytes match; malformed fixtures fail before allocation; Q4 files unchanged |
| **7.3 Offline quality** | Frozen Q4, uniform-int2, matched-byte, and mixed reports; tensor/logit/E2E metrics; source and threshold-set hashes | Matched tolerance is attainable, every unchanged Section 6 gate passes, and total storage saving is >=25%; otherwise mark REJECTED and stop all Wave 7 Tasks 7.4-7.6 |
| **7.4 Loader** | Explicit default-off selection; strict bounded parser/checksums; coexistence tests | Default remains Q4; exact opt-in required; malformed mixed fails with no fallback and before payload allocation |
| **7.5 Metal decode** | Gate/up/down mixed decode; tail/alignment tests; CPU-reference comparisons; NaN/Inf tests | For each isolated record, Metal and CPU MUST produce identical 32-bit IEEE float output bits for all 64 values for both encodings; retained Q4 source bytes remain exact. End-to-end accumulation is diagnostic here and adds no separate tolerance gate. All existing kernel tests pass; Q4 tests/default remain unchanged |
| **7.6 Serial promotion** | Serial Q4 then mixed frozen-logit/E2E run; bytes/token, tok/s, ITL, peak RSS report; rollback record | All quality gates still pass, 7/7 canonical passes, larger suite passes, integrity passes; no unsupported performance claim. Promotion remains experimental unless separately approved |

Task 7.5 changes to Metal kernels MUST run both existing kernel suites. Task 7.6 servers MUST run serially, never concurrently, and diagnostic comparisons MUST use `max_tokens=1`; canonical E2E uses 30 and performance benchmark uses 5.

## 12. Planned reproducibility interfaces

The following commands are **proposed interfaces and do not necessarily exist yet**. Task 7.2 MAY refine option spelling but MUST preserve inputs, outputs, hashes, and deterministic semantics.

```bash
# Planned: validate immutable, disjoint manifests and exact token IDs.
python3 tools/mixed2/corpus.py validate \
  --calibration config/mixed2/calibration-v1.json \
  --heldout config/mixed2/heldout-v1.json \
  --canonical-e2e config/mixed2/canonical-e2e-7-v1.json

# Planned: capture deterministic shards, then merge in manifest order.
python3 tools/mixed2/capture_imatrix.py \
  --model ~/models/DeepSeek-V4-Flash-4bit \
  --experts ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
  --corpus config/mixed2/calibration-v1.json \
  --config config/mixed2/capture-v1.json --output work/imatrix-shards
python3 tools/mixed2/merge_imatrix.py \
  --corpus config/mixed2/calibration-v1.json \
  --input work/imatrix-shards --output work/imatrix-v1.bin

# Planned: frozen conformance self-tests before any held-out evaluation.
python3 tools/mixed2/selftest.py \
  --hash-spec dmlx-mix2-hash-v1 --layout-spec dmlx-mix2-layout-v1 \
  --allocation-spec dmlx-mix2-allocation-v1 --metric-spec dmlx-mix2-metrics-v1 \
  --fixtures config/mixed2/conformance-v1.json

# Planned: deterministic search, conversion, validation, and reference decode.
python3 tools/mixed2/allocate.py --imatrix work/imatrix-v1.bin \
  --config config/mixed2/allocation-v1.json \
  --layout-spec dmlx-mix2-layout-v1 \
  --output work/allocation-v1.json --trace work/allocation-trace-v1.json
python3 tools/mixed2/convert.py \
  --source ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
  --allocation work/allocation-v1.json --config config/mixed2/converter-v1.json \
  --hash-spec dmlx-mix2-hash-v1 --layout-spec dmlx-mix2-layout-v1 \
  --output ~/models/DeepSeek-V4-Flash-4bit/packed_experts_mixed2_v1
python3 tools/mixed2/validate.py \
  ~/models/DeepSeek-V4-Flash-4bit/packed_experts_mixed2_v1 --deep \
  --hash-spec dmlx-mix2-hash-v1 --layout-spec dmlx-mix2-layout-v1
python3 tools/mixed2/reference_decode.py \
  --artifact ~/models/DeepSeek-V4-Flash-4bit/packed_experts_mixed2_v1 \
  --layout-spec dmlx-mix2-layout-v1 --output work/reference-decode-v1

# Planned: offline mandatory gate; must run before runtime Tasks 7.4-7.6.
python3 tools/mixed2/evaluate.py \
  --q4 ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
  --uniform-int2 work/uniform-int2-v1 \
  --matched-control work/matched-control-v1 \
  --mixed ~/models/DeepSeek-V4-Flash-4bit/packed_experts_mixed2_v1 \
  --heldout config/mixed2/heldout-v1.json \
  --canonical-e2e config/mixed2/canonical-e2e-7-v1.json \
  --config config/mixed2/quality-gates-v1.json \
  --metric-spec dmlx-mix2-metrics-v1 \
  --threshold-set predeclared-engineering-risk-v1 \
  --output work/quality-v1.json
```

For future Task 7.6 runtime reproduction, the operator MUST first ensure no `dmlx serve` exists, start Q4 and mixed runs separately, and stop one before starting the other. Planned mixed invocation:

```bash
pkill -f "dmlx serve" 2>/dev/null; sleep 1
./zig-out/bin/dmlx serve \
  --model ~/models/DeepSeek-V4-Flash-4bit \
  --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
  --mixed-expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts_mixed2_v1 \
  --experimental-mixed2 --native --port 8930 --max-tokens 30 --temperature 0
```

The final report MUST print command lines, environment variables, git commit/tree status, host/toolchain identity, all input/output hashes, byte counts, route coverage, allocation counts, and each mandatory gate as `PASS` or `FAIL`. This RFC authorizes no runtime change; it defines the handoff contract for Tasks 7.2-7.6.
