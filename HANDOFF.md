# Handoff — Qwen3.8-27B text-only support (2026-10-20)

Branch `glm53-mlx-support`, HEAD after Task 7: see `git log` (commits `fb826f1`..`ae35a88` + acceptance commit).

## What works

- **Qwen3.8-27B-4bit (qwen3_5, hybrid GDN + full attention)** end to end:
  config parsing (`src/models/qwen38_loader.zig`), GDN linear-attention module
  (`qwen38_gdn.zig`), partial-RoPE full attention (`qwen38_attn.zig`), model
  assembly aligned to the mlx-lm oracle (cos >= 0.9999, `qwen38.zig`), serving
  model + registry arch `Qwen3_5ForConditionalGeneration` + CLI/server wiring
  (ChatML template, multi-EOS stop tokens 248046/248044).
- CLI: `dmlx chat --model /Users/zouyee/models/Qwen3.8-27B-4bit --prompt ... --max-tokens N`
  → coherent completions, clean `<|im_end|>` termination.
- Server: `dmlx server --model ... --port 8080`, `/v1/chat/completions`
  streaming + non-streaming; 7-prompt acceptance (short QA / code / long
  instruction) all returned `finish_reason: "stop"`, server survives
  sequential requests (per-layer GDN/attention caches reset on each prefill).
- Vision input rejected explicitly: token ids 248053–248057
  (`<|vision_start|>`, `<|vision_end|>`, `<|vision_pad|>`, `<|image_pad|>`,
  `<|video_pad|>`) raise `error.VisionInputNotSupported` from
  `Qwen38ServingModel.forwardTokens` (`qwen38.rejectVisionTokens`).

## Measured performance (M4 Pro, 48 GB)

- CLI: model load + 32 tokens in ~19 s total.
- Server decode: ~4–5.5 tok/s (27 tokens/6.5 s, 128 tokens/23.8 s incl.
  ~13–19 token prefill). Prefill ~13 tokens in <1 s.

## Fixes made during acceptance (Task 7)

- **Sampler O(n²) fix (`src/sampling.zig`)**: `topKSampler` /
  `topKTopPSampler` full-sorted the entire vocab with insertion sort — fine at
  32k vocab, ~minutes/token at Qwen3.8's 248,320 vocab. Now a stable O(n·k)
  top-k selection (`selectTopK`); `topPSampler` uses stable `std.sort.block`.
  Semantics preserved (stable ordering, ties keep lower token id).
- **finish_reason threaded to the wire** (commit after acceptance): the engine
  always computed the real verdict but `openai.zig` / `streaming.zig` /
  `anthropic.zig` hardcoded `"stop"` / `"end_turn"`. Now threaded end-to-end:
  `StreamCallbackCtx` got `stop_tokens` (EOS → `.stop`, cap → `.length`), and
  the non-streaming/speculative/guided/DSV4 engine paths use
  `finishReasonForTokens` in `engine_loop.zig`. Anthropic maps
  `.stop`→`end_turn`, `.length`→`max_tokens`. Verified live on Qwen3.8:
  EOS → `stop`, max_tokens=8 → `length` (streaming final chunk too),
  Anthropic → `end_turn` / `max_tokens`.

## Known gaps / concerns

- **Streaming SSE text loses spaces**: the engine decodes each token
  individually (`engine_loop.zig` `tokenizer.decode(&[_]u32{token})`) and
  `bpe.zig` decode strips the leading space (step 3), so streamed deltas
  concatenate without spaces ("Theuserwants…"). Non-streaming output is
  correct. Pre-existing, affects all models; needs incremental decode state.
- glm53-inherited loader error-path leaks (minor, only on failure paths).
- CLI `--temperature 0.8` is the global default *and* a sentinel: qwen38 chat
  maps "left at default" to the recommended temp 1.0 / top_k 20 / top_p 0.95,
  so an explicit `--temperature 0.8` cannot be distinguished from the default.
- `zig build test` reports a failed step due to the Zig 0.16 `--listen=-`
  runner quirk; real signal is the test binary tail line
  (`440 passed; 1 skipped; 0 failed`).

## Regression status

- Full unit suite: 440 passed, 1 skipped, 0 failed (baseline was 438; +2 new
  vision-rejection tests).
- GLM-5.3 smoke (checkpoint present at
  `~/models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx`): served on :8081, one
  non-streaming request → coherent text, `finish_reason: stop`, server alive.
  Note: `dmlx chat` CLI dispatch fails for GLM (`detectModelType` reads the
  first `model_type` key, which is the nested `glm5_next_text`); the server
  path detects by substring and works. Pre-existing.
