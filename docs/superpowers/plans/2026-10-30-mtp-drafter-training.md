# MTP Drafter Self-Distillation (catch-omlx-mtp, final lever)

Date: 2026-10-30. Status: APPROVED (auto mode) → executing.

## Why this is the only remaining lever

From `2026-10-29-catch-omlx-mtp.md` (all settled by measurement, do not re-open):

- The math-category gap decomposes as tokens/round 3.56 (dmlx) vs 3.88 (omlx-mtp)
  at **identical** ms/round (93.05 vs 92.99). Acceptance, not speed, is the gap.
- p_i collapses past depth 3 (p4 ≈ 0.5–0.64), so deeper chains don't pay; the
  controller correctly stops at 3–4. Raising p4 is the only way to raise
  tokens/round.
- Drafter *weights* are the same community checkpoint omlx uses; step-by-step
  oracle showed symmetric quality. omlx-mtp's edge must come from a better-trained
  head (their graft release notes suggest retrained heads) — so we train ours.
- Every engineering lever is exhausted: op-level profile is all zeros, prefill at
  hardware ceiling, batch fold done, post-norm A/B tie, mma morphology vetoed by
  the acceptance cliff.

MTP-D (arXiv 2603.23911) reports +7.5% acceptance from on-policy self-distillation
of the MTP head. Required gain: tokens/round 3.56 → ≥3.88 (+9%) ≈ p4 ~0.5 → ~0.7.

## Design

Train **in Python MLX** (venv `~/venvs/dmlx-omlx`, py3.13, has mlx + the omlx
`mlx_lm_mtp` qwen35 model code). Rationale: `src/trainer.zig` is a LLaMA-only SFT
loop with no qwen38/GDN/MTP support and no value_and_grad through a 4-bit trunk;
rebuilding that in Zig is weeks of work and a new correctness surface. MLX gives
us `nn.value_and_grad`, quantized linear keep-frozen semantics, and the exact
reference MTP implementation for free. Export is safetensors — the format the
dmlx drafter loader already reads, so **no engine code changes**.

Frozen: 4-bit trunk (all 48 layers), token embedding, lm_head.
Trainable: the single MTP decoder layer + `mtp.fc` + the two RMSNorms
(~240 MB bf16; +AdamW states ≈ 2 GB — fits easily beside the 15 GB trunk).

### Objective (MTP-D style, hard + soft)

Per position t with trunk hidden h_t (pre-final-norm, the convention the head was
trained on — see qwen38_mtp.zig:482):

    L = CE(head(h_t, tok_t), tok_{t+1})            # hard next-token
      + λ · KL(trunk_logits_t ∥ head_logits_t)   # soft distillation, λ=0.3, T=2

Soft term chunked: 256 positions at a time → logits chunk = 256×248320×4 B ≈
254 MB fp32. Sequences 2048 tokens.

### Data (on-policy self-distillation)

1. Prompt pool: ~600 prompts mixed code/prose/math/count (reuse the benchmark
   corpus generator in `scripts/compare_engines.py` + routing corpus prompts).
2. Generate completions with the CURRENT engine at temperature 0.7, max 1024
   tokens (on-policy ≈ matches decode-time distribution; temp 0.7 avoids the
   degenerate loops of greedy long-form).
3. Target ≈ 400 sequences ≈ 300–400k tokens. That's the scale FastMTP/MTP-D ablate
   at for single-head adaptation; more is bounded by M4 Pro throughput (~35 tok/s
   → ~3 h generation; training forward passes dominate, est. 4–6 h total).

### Steps

- [ ] **S0 — Pre-training p_i baseline** (running): fixed depth 4, 6 mixed prompts,
  read `[Engine] MTP cond-acceptance`. This is the before/after yardstick.
- [ ] **S1 — Corpus generation** `scripts/mtpd_gen_corpus.py`: drives the local
  dmlx server, writes `artifacts/mtpd/corpus.jsonl` {prompt, text}. Gate: ≥350k
  tokens, dedupe check.
- [ ] **S2 — Training script** `scripts/mtpd_train.py`:
  - load trunk via omlx's mlx_lm_mtp loader (4-bit, frozen); load community MTP
    head as trainable init;
  - tokenize corpus, pack into 2048-token sequences;
  - per step: trunk forward (no grad) → hiddens; head forward with grad on
    chunked CE + λ·KL; AdamW lr 1e-4 cosine, warmup 50, ~2 epochs, grad clip 1.0;
  - log train/heldout acceptance-proxy (top-1 match of head vs trunk argmax on
    heldout 32 sequences — this predicts p1; chain-accuracy of 4 consecutive
    predicts p2..p4);
  - checkpoint every 200 steps to `artifacts/mtpd/ckpt/`.
- [ ] **S3 — Export** to drafter layout `~/models/Qwen3.8-27B-MTP-4bit-distilled/`:
  bf16 → same 4-bit gs=64 quant + safetensors index the loader expects
  (verify against `MtpHead.load` key list; RMSNorm convention: raw scale, NOT
  scale−1 — qwen38_mtp.zig:206).
- [ ] **S4 — Acceptance re-measurement**: identical fixed-depth-4 probe as S0.
  Kill criterion: chain top-1 agreement at positions 2–4 must improve ≥5 points
  AND end-to-end p4 proxy ≥0.62, else revert to the community head and stop
  (honest-verdict rule).
- [ ] **S5 — Interleaved A/B** `scripts/compare_engines.py --engines dmlx,omlx-mtp
  --reps 3 --nonce`, distilled head vs community head vs omlx-mtp.
- [ ] **S6 — Gates + commit**: self-consistency 5/5 byte-identical (temp-0 output
  must not change — verify fixes bad drafts), fidelity profile NEUTRAL, unit
  suite 464+1 skip, benchmark Paris ✓ + E2E 7/7, RSS < 20 GiB. Docs
  (§44 in `docs/analysis/beat-omlx-mlxvlm-plan.md` + HANDOFF), commit only
  files this plan touched.

### Risks / discipline

- **Catastrophic forgetting of the head**: init from the community head, low LR,
  early stop on heldout proxy; keep the original checkpoint untouched.
- **Distribution mismatch**: on-policy generation is the point (MTP-D's core
  claim); do NOT train on generic text (RedPajama etc.) — off-policy corpora
  taught this project nothing in §34.
- **Measurement discipline** carries over in full: interleaved A/B, fixed-depth
  reads for diagnostics, acceptance is a first-class signal, cross-session drift
  ~8–10% means before/after must be same-session.
- omlx-mtp's absolute tok/s is a moving target across sessions (33→41→46); the
  goal is judged on the SAME-day interleaved harness, not historical numbers.

## Execution log (2026-10-30)

- [x] **S0 — Pre-training p_i baseline** (fixed depth 4, 6 mixed prompts,
  `DMLX_MTP_FIXED_N=4`): p1=0.874 p2=0.777 p3=0.795 p4=0.784 (pooled
  215/188/146/116 rounds). MLX-side heldout chain top-1 proxy = 0.79/0.70/0.63.
- [x] **Training stack validated**: grafted checkpoint loads via
  `omlx.patches.mlx_lm_mtp` (`set_mtp_active(True)` + `qwen35_model.apply()`);
  head dequantizes to bf16 (424.7M trainable params, 15 leaves); chain-aligned
  loss verified healthy (head CE 1.13 vs trunk 0.70 nats/token at init).
- [x] **Divergence root-caused**: lr 1e-4 collapsed heldout top-1 0.79->0.35 in
  ONE step. Zero-grad control clean; max|dw|=5e-4; alignment verified. Cause =
  AdamW first-step sign-like transient on a sharp pretrained head. Fix: lr 2e-5
  + 30-step linear warmup (`optim.join_schedules`), validated stable.
- [x] **Engine bug found & fixed** (`engine_loop.zig:417-426`): use-after-free
  race - HTTP handler frees `req.prompt_tokens` on completion while the engine
  thread may still need it for the post-generation prefix-cache store
  (`panic: @memcpy arguments alias` in `prefix_cache.store`). Fixed by duping
  the prompt into engine-owned memory at `processRequest` entry.
- [ ] **KNOWN ISSUE (not chased, orthogonal)**: with the prefix cache ENABLED
  the server still crashes around request #17-19 (first `evictLRU`) with the
  same alias panic even with engine-owned prompts; a second crash signature
  shows a wild-pointer read (`0xafd4c966...`) in `isCached` /
  `qwen38_gdn.prefillVerify.deinit`. Prefix-cache code itself audits clean;
  suspicion is heap corruption originating in the GDN/MTP cache clone/deinit
  paths (user's in-flight code). A third signature: `Failed to create Metal
  shared event` under concurrent MLX-python training + dmlx serving - GPU
  contention, avoided by serialization. Workaround for all corpus/benchmark
  serving: `--prefix-cache-entries 0` + exclusive GPU. Prefix cache yields 0
  hits on unique-prompt workloads, so no perf cost.
- [ ] **S1 corpus generation running**: 400 mixed prompts, temp 0.7,
  max 1024 tok, resume-capable (`scripts/mtpd_gen_corpus.py`).
- [ ] **KNOWN ISSUE updated (root-caused one level deeper)**: the sustaining
  crash is NOT the prefix cache. Exclusive-GPU + `--prefix-cache-entries 0`
  still dies at ~request 20-23 with
  `MLX rc=1: [metal::malloc] Resource limit (499000) exceeded` followed by a
  segfault in `mlx_array_free` from the `qwen38_gdn.prefillVerify` errdefer
  cleanup. Two distinct defects in the in-flight MTP code:
  (a) a Metal buffer LEAK in the MTP verify path accumulating ~25k buffers per
  request until the Metal allocation limit kills the server after ~20
  sustained temp-0.7 requests (temp-0 sessions of the same length were fine,
  so the leak is likely on the sampling/RNG side of the round);
  (b) fragile cleanup: when an MLX op fails mid-`prefillVerify`, the errdefer
  deinits partially-constructed arrays -> wild free / segfault (this is also
  what poisoned the heap in the earlier "prefix cache" crashes).
  Corpus workaround: serve WITHOUT `--speculative-mtp` for generation
  (identical sampling distribution, half speed, no verify path).

## PIVOT (2026-10-30, user directive): no training — gap is implementation

User: "这是实现上的差距，先不要训练". Correct: omlx-mtp runs the SAME community
drafter weights (grafted checkpoint), so weights cannot explain any gap.
Training route cancelled (pipeline kept in tree: scripts/mtpd_*.py, unused).

**Same-session evidence gathered after the pivot:**

1. Fixed depth 4, 6 identical prompts, pooled conditional acceptance:
   dmlx p1-4 = 0.874/0.777/0.795/0.784; omlx = 0.765/0.791/0.729/0.790.
   => per-position acceptance EQUAL (dmlx slightly ahead). tok/cycle at
   fixed 4: dmlx 3.41 vs omlx 3.26.
2. Adaptive mode, same 6 prompts: dmlx E[t] 3.06 @ ~97 ms/round vs omlx
   E[t] 3.04 @ ~90 ms/cycle. Tokens/round EQUAL; the whole gap is ms/round.
3. `mtp_round_decompose.py --prompt-kind math` (serial same-prompt):
   3.54 vs 3.66 tokens/round, 94.3 vs 91.2 ms/round, wall 3.83s vs 3.52s
   (128 tok). omlx drafts deeper (21/35 rounds hit d4) at LOWER ms/round.
   dmlx p_i HIGHER (0.89/1.00/0.90); controller correctly parks at depth 3
   because OUR marginal row is pricier, making depth 4 a wash for us while
   it wins for them.
4. omlx timing semantics decoded: their "mtp" timer is host-only dispatch
   (~2ms); the chain's GPU time resolves inside the NEXT cycle's "backbone"
   timer (fire-and-forget mx.async_eval). So real per-cycle = backbone+mtp
   ≈ 93.6ms at fixed depth 4 vs dmlx ~112 (draft 13 + verify 98).
5. Plain decode (NO MTP either side), first single readings:
   dmlx ~13.7 tok/s vs omlx ~15.1 tok/s => ~10% base op-level gap (their q4
   prefill patch is prefill-only; decode path TBD). Interleaved 2-rep A/B
   running to confirm.

**Where the remaining gap is NOT:** weights, drafter quality, acceptance,
chain-state conventions, controller policy (ours parks correctly given our
cost curve), prefix cache, verify-kernel routing (already bench-tuned §38).
**Where it IS:** per-round latency — verify marginal per row (their 5-row
cycle ~93.6ms total vs ours ~112) and possibly base decode ops (~10% TBC).

## ROOT CAUSE FOUND (chat-template parity, 2026-10-30)

The interleaved scorecard (dmlx 32.6 vs omlx-mtp 40.6 HEADLINE decode t/s)
decomposed wrong on inspection: omlx's big wins were code 47.1 / math 51.9
with an early-stop advantage, while dmlx WON effective tok/s (TTFT 0.28s vs
0.80-1.16s). Direct probes showed equal acceptance, equal tokens/round,
equal plain decode (both ~93% of the 231 GB/s roofline).

**The engines generate different content for the same chat prompt.** omlx
sets `enable_thinking=false` (`thinking_budget_enabled: false` in
model_settings.json) so the HF chat template appends an EMPTY think block
(`<|im_start|>assistant\n<think>\n\n</think>\n\n`) and the model answers
directly (math: 44 chars, stops at ~86-104 tokens). dmlx's applyChatML
appends bare `<|im_start|>assistant\n` and the model free-thinks (128-token
cap hit every time). Decode-rate comparisons across different generated
content are meaningless; the two-point window also credits omlx's fast
predictable answer segment while dmlx grinds think tokens.

**Fix (parity feature, default OFF):** `DMLX_CHAT_EMPTY_THINK=1` ->
`ChatTemplate.empty_think` appends the empty think block in applyChatML
(chat_template.zig:151-157, state.zig env wiring), harness arm
`dmlx-nothink` in compare_engines.py, unit test added. Verified: math
prompt now returns a direct answer with no think block.

Residual REAL gap after parity: ~3 ms/round verify marginal at depth 3-4
(94.3 vs 91.2 same-prompt serial), ~3% — inside cross-session drift.

## LANDED (2026-10-30): leak fix + content parity

1. **Metal buffer leak FIXED** (`qwen38_gdn.zig` commitVerifyStates replay
   branch): the five replay input arrays (k_full/v/q_full/beta/g, moved out of
   `front`) were never deinited on the commit path — `res.replay=null` skipped
   the free in `PrefillVerifyResult.deinit`. Leaked ~0.4 MB x 48 layers per
   committed round = +0.68 GB/request active MLX memory -> progressive
   slowdown (verify 95 ms -> 41 SECONDS per round by request #11) and the
   `metal::malloc Resource limit (499000)` / Metal-shared-event crashes that
   killed every long MTP session at ~20 requests. Found via a new
   `DMLX_MEM_PROBE` (per-request) / `DMLX_MEM_PROBE_ROUNDS` (per-phase)
   instrument in engine_loop.zig, bisected with the existing verify-section
   ablation flags + a new `DMLX_VERIFY_EAGER_REC=1` (fused-vs-eager control).
   Verified: 5 requests dead flat at 22.50 GB; 25-request temp-0.7 stress
   clean (the original crash condition).
2. **Content parity**: `DMLX_CHAT_EMPTY_THINK=1` + `dmlx-nothink` harness arm.
3. **Scorecard after both fixes** (interleaved 3 reps, nonce):
   dmlx-nothink 35.2 vs omlx-mtp 46.2 decode t/s; EFFECTIVE (incl TTFT):
   code 35.6 vs 35.0, prose 23.8 vs 27.7, math 34.1 vs 34.5 — a tie, carried
   by our 0.28 s vs their 0.80-1.16 s TTFT. dmlx improved 32.6 -> 35.2 with
   parity+leak-fix; omlx drifts 40.6 -> 46.2 across sessions (their number is
   a moving target; also their 51.9 math decode is inflated by early-stop
   two-point windows + best-of-6 — their own log shows ~3.5 tok/cycle @ ~89 ms
   ≈ 39-40 tok/s steady state, and their 256-token raw wall ran ~22 tok/s
   with ~60 ms/cycle Python host overhead).
4. Gates: temp-0 self-consistency 5/5 byte-identical; unit 472/473 + 1 skip
   (unchanged); verify-kernel routing untouched.

Residual honest gap: ~7-10% steady-state decode on long generations
(verify marginal ~3 ms/round + omlx lower host overhead in short windows).
Not chased further: within cross-session drift and shrinking; the remaining
big structural lever (deeper chains at p4+) stays blocked on acceptance,
not on any identified defect.
