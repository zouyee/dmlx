# DSpine draft optimization notes — 2026-11-25

## Initial same-binary AB/BA

Raw evidence: `round1/`.

DSpine was not a uniform win over DF2+copies+bands: sampled decode deltas were
-3.3%, +0.2%, +9.3%, +3.9%, -17.0%, and +5.7% across the six cells. Every
request in both arms produced 256 tokens with finish=length.

Mechanism medians over each full pass:

- DF2: E[t] about 2.56-2.66, draft 12.7-13.1 ms/round.
- DSpine: E[t] about 2.79-2.93, draft 31.5-32.7 ms/round.

The acceptance improvement was too small to cover the draft implementation
cost.

## Draft profile

`DMLX_DSPINE_PROF=1` splits each block into forward, shared LM head, top-k,
and serial refinement. After warmup on a short greedy prompt, the stable
medians were approximately:

- forward: 10.8 ms
- LM head: 3.8 ms
- top-k: 1.2 ms
- refinement: 1.0 ms
- total block: 16.8 ms

The sampled server-reported draft cost was nevertheless about 35 ms/round,
while its block-profile total stayed about 17 ms. The extra cost was therefore
in the per-position sampled interface, not the five-layer forward.

## Root cause and fix

DSpine exposes only K=16 finite proposal logits and fills the rest of the
248320-entry vocabulary with `-inf`. `topKTopPSampler` treated equal `-inf`
scores as insertion candidates after its top-50 list filled, producing an
O(vocab x 50) path per drafted token. Dense DF2 logits mostly take the O(1)
early-skip path.

The sampler now skips NaN and negative-infinity scores, computes probabilities
over the actual retained count, and reports `NoFiniteLogits` when no candidate
has nonzero probability. A unit test covers sparse distributions and the
all-`-inf` error.

After the fix, three stable sampled smoke requests reported draft medians of
about 21-22 ms/round instead of about 35 ms. Full tests and ReleaseFast passed
before the second AB/BA. Raw profile logs: `server_prof.log`,
`server_prof_after_sampler.log`.


## Second same-binary AB/BA

Raw evidence: `round2/`; summary: `analysis_round2.txt`.

The sparse-sampler fix reduced DSpine's full-pass draft median to 17.2-18.0
ms/round. Sampled decode deltas versus DF2+copies+bands became:

- think-off short/2k/8k: +10.4%, +5.5%, +9.2%
- think-on short/2k/8k: +12.4%, -6.6%, +12.6%

All 48 measured requests per arm completed 256 tokens with finish=length.
DSpine E[t] was 2.90 in both passes. The think-on/2k regression has broad
range overlap, so it is not evidence of a systematic correctness failure, but
it does prevent a default-mode recommendation. The appropriate verdict is
experimental opt-in, not replacement of DF2.


## Greedy controls

One repetition per cell (`analysis_greedy.txt`), all requests finish=length at
256 tokens:

- think-off short/2k/8k: -4.9%, -6.6%, -6.4%
- think-on short/2k/8k: -3.4%, +11.6%, -9.4%

Greedy does not show the sampled path's broad gain. DSpine should therefore be
documented as a sampled-oriented experimental option, not a generally faster
replacement for DF2.
