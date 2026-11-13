# Qwen3.8-27B Vision (Image Input) Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add image-input support for `Qwen3_5ForConditionalGeneration` (`/Users/zouyee/models/Qwen3.8-27B-4bit`) to dmlx: CLI (`dmlx chat --image`) and server (`/v1/chat/completions` with `image_url` parts), oracle-verified against mlx-vlm.

**Architecture:** New vision tower module `src/vision/qwen38_vit.zig` (bf16, unquantized — the checkpoint ships the tower in BF16), host-side preprocessing `src/vision/qwen38_preprocess.zig` (stb_image decode + bicubic resize), MRoPE position plumbing `src/models/qwen38_mrope.zig` (3-channel interleaved position ids, host-computed cos/sin), and a vision entry point on `Qwen38ServingModel` (`forwardTokensWithVision`). mlx-vlm 0.7.11 (`~/venvs/dmlx-omlx`) is the oracle; parity harness mirrors the text-side `qwen38_oracle` workflow.

**Tech Stack:** Zig 0.16, MLX C API via `mlx_z` (`ops`, `fast`, `shape`, `reduce`), stb_image (single-header C) for JPEG/PNG decode, mlx-vlm as test oracle.

**Out of scope (v1):** video input (`video_token_id` stays rejected), deepstack (this checkpoint has `deepstack_visual_indexes: []`), vision feature caching, multi-image batching in the ViT (images encoded sequentially), ViT perf tuning (one-shot per image; measure first).

---

## Verified facts (do not re-derive)

From the real checkpoint and config (`/Users/zouyee/models/Qwen3.8-27B-4bit`):

- Vision tower weights: 333 tensors prefixed `vision_tower.`, **all BF16, none quantized** (no `.scales`); biases present on Linears and LayerNorms.
- `vision_tower.patch_embed.proj.weight` BF16 `[1152, 2, 16, 16, 3]` = `[out, t, kh, kw, in]` — already MLX conv3d layout (NDHWC input), **no transpose** (the mlx-vlm sanitize transpose is a no-op for this layout).
- `vision_tower.pos_embed.weight` `[2304, 1152]` (48×48 grid), learned absolute pos embed, bilinearly interpolated to the actual grid.
- vision_config: `depth=27, hidden=1152, heads=16 (head_dim 72), intermediate=4304 (check at load), patch=16, temporal_patch=2, merge=2, out_hidden=5120, in_channels=3, num_position_embeddings=2304, model_type qwen3_5`. Act: GELU(tanh) in blocks, GELU in merger. LayerNorm eps=1e-6.
- Token ids: `vision_start=248053, vision_end=248054, image_pad=248056, video_pad=248057` (248055 = object_ref, unused). Prompt shape: `<|vision_start|><|image_pad|>×N<|vision_end|>` where N = grid_h/2 × grid_w/2 (merge²) per image (grid_t=1 for images).
- `rope_parameters`: `rope_theta=1e7, partial_rotary_factor=0.25 (rotary dims=64), mrope_interleaved=true, mrope_section=[11,11,10]` (sums to 32 = rotary dims/2). Language MRoPE: 3-channel position ids (t,h,w); for each of the 32 frequency slots, the channel is chosen by interleaved section assignment; apply = standard rotate_half rope on the first 64 dims, remaining 192 dims pass through (`rope_utils._apply_interleaved_rotary_pos_emb_axis1`).
- Text-only prompts: all 3 channels equal the scalar position — collapses exactly to the existing 1D offset rope (why today's text path is correct).
- Decode/verify after a vision prefill: `rope_delta = max(position_ids)+1 − seq_len` (scalar); all later positions use `offset = cache_len + rope_delta` — the existing scalar-offset rope works unchanged once the delta is folded in.
- Vision tower attention is **non-causal**, per image (cu_seqlens split), full bf16 SDPA.
- Reference sources (read before coding each task):
  - `~/venvs/dmlx-omlx/lib/python3.13/site-packages/mlx_vlm/models/qwen3_vl/vision.py` — tower (PatchEmbed/rot_pos_emb/fast_pos_embed_interpolate/blocks/merger)
  - `.../qwen3_5/qwen3_5.py` — `get_input_embeddings`, `merge_input_ids_with_image_features` (masked_scatter), `sanitize`
  - `.../qwen3_5/language.py:1465` `get_rope_index` — MRoPE position id construction
  - `.../models/rope_utils.py` — `compute_mrope_frequencies`, `_interleaved_position_selector`, `rotate_half`
  - `.../qwen3_vl/processing_qwen3_vl.py` — `_smart_resize_image` (HF smart_resize port), temporal duplication, grid_thw
- dmlx integration points:
  - `src/models/qwen38.zig` — `Qwen38ServingModel`, `rejectVisionTokens` (must allow 248053/248054/248056 when features are supplied), attn rope call sites in `qwen38_attn.zig:1455-1458` (scalar offset today)
  - `fast.rope(ctx, x, dims, traditional, base, scale, offset, freqs)` and `fast.ropeDynamic(... offset: Array ...)` (`zig-pkg/mlx_z*/src/ops/fast.zig:25,32`)
  - `src/vision/vit.zig` + `src/vision/preprocess.zig` — llava CLIP precedent (LayerNorm/patchify idioms)
  - `src/model_registry.zig` — `LlavaVTableAdapter` shows the vision-model vtable precedent; Qwen38 adapter at `:617` (`forward` → `forwardTokensLast`)
  - Oracle workflow precedent: `scripts/qwen38_oracle_dump.py` + `scripts/compare_qwen38_oracle.py` + `artifacts/qwen38_oracle/`

## Conventions

- TDD: every Zig module lands with toy-weight unit tests in the same file (`zig build test`); oracle parity tests skip gracefully when artifacts are absent (same pattern as `TinyLlama golden data not found` skip).
- Commit per task, English messages (project language is English).
- The vision tower runs at bf16 with fp32 norms; do **not** route tower Linears through `qmm_v8`/`verify_qmm` (unquantized weights — plain `ops.matmul`).

---

## Task 1: Vision config + loader extension

**Files:**
- Modify: `src/models/qwen38_loader.zig` (add `Qwen38VisionConfig`, parse `vision_config`, `image_token_id`/`video_token_id`/`vision_start_token_id`/`vision_end_token_id`, and `text_config.rope_parameters.mrope_section`/`mrope_interleaved`/`rope_theta`)
- Test: inline `test` blocks in `qwen38_loader.zig`

- [x] **Step 1: failing test** — parse a config JSON fixture string containing the real `vision_config` block + `rope_parameters`; assert `vision.depth==27`, `vision.hidden==1152`, `mrope_section==[11,11,10]`, `image_token_id==248056`, `rope_theta==1e7`.

- [x] **Step 2: run test, verify failure** — `zig build test 2>&1 | grep -c "qwen38_loader"` shows the new test failing (fields missing).

- [x] **Step 3: implement** — mirror the existing nested-JSON parsing (`text_config`) for `vision_config`; `rope_parameters` overrides `rope_theta` when present. Defaults: `mrope_section = .{11,11,10}`, `mrope_interleaved = true`.

- [x] **Step 4: run test, verify pass.**

- [x] **Step 5: commit** — `git commit -m "qwen38: parse vision_config and mrope rope_parameters"`

## Task 2: MRoPE position ids (host-side, `src/models/qwen38_mrope.zig`)

**Files:**
- Create: `src/models/qwen38_mrope.zig`
- Test: inline

Port `get_rope_index` (image-only subset) as pure Zig integer logic:

```zig
pub const MropeResult = struct {
    position_ids: [3][]i32,  // t/h/w channels, len = seq_len
    rope_delta: i32,         // max(pos)+1 - seq_len
};
/// ids: full prompt token ids; image_blocks: one per image, in order of
/// appearance: grid_t=1, grid_h/=merge, grid_w/=merge already divided.
pub fn computePositionIds(
    allocator: std.mem.Allocator,
    ids: []const u32,
    image_blocks: []const [3]i32, // (t, h_merge, w_merge) per image
    image_token: u32,             // 248056
    vision_start: u32,            // 248053
) !MropeResult
```

Semantics (mirror `language.py:1465`): walk tokens; text runs get `arange` on all 3 channels starting at `st_idx = max(prev)+1`; each image block (token span `vision_start, image_pad×(t·h·w), ...`) gets t/h/w meshgrid positions + `text_len + st_idx`; after the block, `st = ed + t·h·w`. `rope_delta = max over channels + 1 − seq_len`.

Also the frequency selector (host, computed once): `interleaved` layout over 32 slots from section `[11,11,10]` — slot j takes channel `(j mod 3)` within its section run (mirror `_interleaved_position_selector` in rope_utils.py — read it, do not guess).

- [x] **Step 1: failing test** — hand-computed case: ids = `[A A vs_start img img img img vs_end B B]` with one image block (t=1,h=2,w=2 → 4 pads) ⇒ expected 3×10 position matrix (compute by hand from the reference algorithm) and `rope_delta = (2+2-1... )` — verify against the Python oracle first:
  `python3 -c` snippet running `LanguageModel.get_rope_index` on the same ids, print both, paste expected into the test.
- [x] **Step 2: run, verify fail.**
- [x] **Step 3: implement** `computePositionIds` + selector.
- [x] **Step 4: run, verify pass.**
- [x] **Step 5: commit** — `git commit -m "qwen38: MRoPE position ids (image subset) + interleaved frequency selector"`

## Task 3: Vision tower (`src/vision/qwen38_vit.zig`)

**Files:**
- Create: `src/vision/qwen38_vit.zig`
- Test: inline toy-weight tests + oracle parity test (Task 7 supplies artifacts; write the test now, skipping when absent)

Structure (mirror `qwen3_vl/vision.py`, single image per call — `cu_seqlens` degenerates to one span, no split needed):

```zig
pub const Qwen38VitConfig = struct { depth: usize, hidden: i32, heads: i32,
    intermediate: i32, patch: i32, temporal_patch: i32, merge: i32,
    out_hidden: i32, num_pos: i32, in_ch: i32 };
pub const Qwen38Vit = struct {
    // patch_embed proj [1152,2,16,16,3] + bias; pos_embed [2304,1152];
    // blocks: norm1(LN w/ bias) → attn(qkv fused Linear w/ bias → vision rope
    // → SDPA non-causal → proj) → norm2 → mlp(fc1→gelu_tanh→fc2);
    // merger: LN(hidden) → fc1 → gelu → fc2 → [n_merged, 5120]
    pub fn init(allocator, ctx, cfg: Qwen38VitConfig, weights: *const std.StringHashMap(Array)) !Qwen38Vit;
    /// pixels: [grid_t*grid_h*grid_w, in_ch*t*p*p] bf16 (preprocess output);
    /// returns [ (grid_h/merge)*(grid_w/merge), 5120 ] bf16
    pub fn forward(self: *Qwen38Vit, pixels: Array, grid_t: i32, grid_h: i32, grid_w: i32) !Array;
    pub fn deinit(self: *Qwen38Vit) void;
};
```

Key sub-parts (all elementwise/shape ops, no custom kernels):
- Patch embed: prefer `mlx_conv3d` (mlx-c has it, `ops.h:249`); weight already `[out,t,kh,kw,in]`. Fallback if conv3d binding missing in mlx_z: reshape pixels to patches and one `ops.matmul` (stride==kernel ⇒ unfold+GEMM equivalence).
- `fast_pos_embed_interpolate`: compute the 4-corner indices + weights on host (f64), then `pos_embed(idx)` via `shape_mod.takeAxis` ×4, weighted sum, then the merge-order permute (reshape/transposes — copy the exact reshape chain from vision.py:354-372).
- Vision rotary (`rot_pos_emb`): freq table `outer(arange(max_hw), inv_freq)`; 2D coords per patch in merge-block order (vision.py:232-281); apply = `x*cos + rotate_half(x)*sin` with cos/sin tiled ×2 (vision.py:23-40). `rotate_half(x) = cat(-x[d/2:], x[:d/2])`.
- Block attention: qkv `[n, 3*1152]` → split → heads 16×72 → rope → `fast.scaledDotProductAttention(scale=72**-0.5, mask="")`.

- [x] **Step 1: failing toy test** — 1-block tower, hidden 16/heads 2/patch 2/merge 1, random weights, assert output shape and determinism (same input → bitwise same output).
- [x] **Step 2: run, verify fail (module missing).**
- [x] **Step 3: implement.**
- [x] **Step 4: run, verify pass.**
- [x] **Step 5: commit** — `git commit -m "qwen38: vision tower (ViT-27B, bf16) with toy-weight tests"`

## Task 4: Image preprocessing (`src/vision/qwen38_preprocess.zig` + stb)

**Files:**
- Create: `src/vision/stb_image_impl.c` (`#define STB_IMAGE_IMPLEMENTATION` + `#include "stb_image.h"`, vendored header `src/vision/stb_image.h`)
- Create: `src/vision/qwen38_preprocess.zig`
- Modify: `build.zig` (add the C file to the CLI/server targets — follow the existing `addCSourceFile` pattern at `build.zig:962`)
- Test: inline

```zig
pub const PreprocessOut = struct {
    pixels: Array,          // [grid_h*grid_w, 3*2*16*16] bf16 (t duplicated)
    grid_thw: [3]i32,       // [1, h/16, w/16]
    n_image_pad: usize,     // (h/32)*(w/32)
};
pub fn preprocessImageFile(allocator, ctx, path: []const u8) !PreprocessOut;
pub fn preprocessImageBytes(allocator, ctx, bytes: []const u8) !PreprocessOut;
```

- smart_resize (port `_smart_resize_image` exactly): round h/w to multiples of 32 (`patch×merge`), clamp total pixels to [65536, 16777216] (preprocessor_config.json), preserving aspect via `round(h*factor/32)*32`.
- Resize: bilinear via Accelerate `vImageScale_ARGB8888`+float conversion, or straightforward Zig bilinear on f32 planes (image ≤ 4096×4096 — fine on CPU). Reference uses PIL **bicubic**; v1 uses bilinear and Task 7's parity gate (cos ≥ 0.999 on tower features) decides acceptability — if it fails, port bicubic (16-tap separable).
- Normalize (x/255 − 0.5)/0.5; duplicate along temporal (`t=2` copy of the single frame); flatten patches in NDHWC patch order.

- [x] **Step 1: failing test** — synthesize a 64×96 PNG via Python PIL into `tests/fixtures/` (commit the fixture); assert `grid_thw == .{1,4,6}`, `n_image_pad == 12`, pixel array shape `[24, 1536]`, mean ≈ expected within 1e-3.
- [x] **Step 2: run, verify fail.**
- [x] **Step 3: implement.**
- [x] **Step 4: run, verify pass.**
- [x] **Step 5: commit** — `git commit -m "qwen38: image preprocessing (stb decode + smart_resize + patchify)"`

## Task 5: Serving model vision entry (`src/models/qwen38.zig`, `qwen38_attn.zig`)

**Files:**
- Modify: `src/models/qwen38.zig` — new `forwardTokensWithVision`
- Modify: `src/models/qwen38_attn.zig` — `forward` gains an optional `mrope: ?MropeApply` parameter (cos/sin arrays + precomputed layout); when null, today's scalar-offset rope path runs untouched
- Test: inline (toy model, monkey shapes)

```zig
pub const VisionInput = struct {
    features: Array,        // [total_pads, 5120] bf16, concat of all images
    blocks: []const [3]i32, // (t, h_merge, w_merge) per image
};
pub fn forwardTokensWithVision(
    self: *Qwen38ServingModel,
    ids: []const u32,
    vision: VisionInput,
) !Array // last-position logits, same contract as forwardTokensLast
```

Flow: `embed_tokens(ids)` → masked scatter of `features` at `ids == 248056` positions (host builds the index list; use `shape_mod.takeAxis` on features + `scatter`-style update on embeds — check mlx_z for `scatter`/`putAlongAxis`; fallback: slice-assign contiguous spans per image) → `mrope.computePositionIds` → host-computed cos/sin `[L, 32]`→selected → tiled to `[L, 64]` → per attn layer `forward(..., mrope)` applying `q_rot*cos + rotate_half(q_rot)*sin` on the first 64 dims (q and k), concat pass-through dims → store `self.rope_delta = result.rope_delta`.

Rope offset plumbing for AFTER a vision prefill: today decode/verify use `offset = cache.len()`; change to `offset = cache.len() + self.rope_delta` (default delta = 0 — text path unchanged). Sites: `qwen38_attn.zig:1362` (`offset: i32 = cache.len()`), verify continuation, and `qwen38_mtp.zig` drafter offsets. GDN layers: no rope, untouched.

`rejectVisionTokens`: allow 248053/248054/248056 when called from `forwardTokensWithVision`; plain `forwardTokens*` keeps rejecting (exact current behavior).

- [x] **Step 1: failing test** — toy 2-layer model (like the chunked-prefill test's `mkModel`): prefill 4 text tokens normally → logits A; same 4 tokens through `forwardTokensWithVision` with an empty image list ⇒ logits identical to A (delta=0 path). Plus a delta case: hand-built `blocks` entry shifting positions; assert decode-step rope offset includes delta (compare against reference computed with explicit cos/sin).
- [x] **Step 2: run, verify fail.**
- [x] **Step 3: implement.**
- [x] **Step 4: run, verify pass + full suite (`zig build test`) green.**
- [x] **Step 5: commit** — `git commit -m "qwen38: forwardTokensWithVision — embed merge + MRoPE + rope_delta decode offsets"`

## Task 6: Oracle dump + parity harness

**Files:**
- Create: `scripts/qwen38_vision_oracle_dump.py` (mlx-vlm; dumps for a fixture image: preprocessed pixels + grid_thw, tower features, merged embeddings, position_ids (3×L), per-slot cos/sin, and final logits for the image prompt)
- Create: `scripts/compare_qwen38_vision_oracle.py`
- Create: `tests/fixtures/vision/` artifacts via the dump script (gitignored or committed-small, follow `tests/golden/` precedent)
- Test: `src/test_qwen38_vision_oracle.zig` (registered in `build.zig` like `test_qwen38_oracle_dump.zig`), skips when artifacts absent

Parity gates: tower features cos ≥ 0.999; merged embeddings exact-token positions equal; position_ids exact; final logits cos ≥ 0.999 (bf16 accumulate tolerance, same convention as text oracle).

- [x] **Step 1: write dump script; run it** — `~/venvs/dmlx-omlx/bin/python scripts/qwen38_vision_oracle_dump.py tests/fixtures/vision/photo.jpg artifacts/qwen38_vision_oracle/` (fixture: a real 512×384 photo, committed).
- [x] **Step 2: write the zig parity test reading artifacts (expect FAIL on tower features first run).**
- [x] **Step 3: reconcile until gates pass** — the usual suspects, in order: resize resampling (bilinear vs bicubic), patch order (NDHWC vs NCDHW), merge-order permute, norm `+1.0` shift does **not** apply to vision LayerNorm (that shift is an RMSNorm-only loading convention — `should_offset_norm_weight` guards on it), rope selector layout.
- [x] **Step 4: commit** — `git commit -m "qwen38: vision oracle parity harness (tower + mrope + logits)"`

## Task 7: CLI + server wiring

**Files:**
- Modify: `src/main.zig` — `dmlx chat --image <path>` (repeatable), builds VisionInput via `preprocessImageFile` + tower + `forwardTokensWithVision`
- Modify: `src/server/` chat handler — parse OpenAI content parts: `{"type":"text"}` and `{"type":"image_url","image_url":{"url":"data:image/...;base64,..."}}` (data URLs v1; also accept absolute file paths for local dev), run preprocessing inline, route to the vision forward
- Modify: `src/model_registry.zig` — expose a vision-capable vtable entry (or bypass: the server detects qwen38 + images and calls the serving model directly, mirroring how MTP is threaded)
- Test: `scripts/ab_vision_smoke.sh` — starts server, posts an image prompt, asserts non-error `finish_reason:"stop"`

- [x] **Step 1: CLI first** — `dmlx chat --model ~/models/Qwen3.8-27B-4bit --image tests/fixtures/vision/photo.jpg --prompt "Describe this image." --max-tokens 64` produces a coherent description (eyeball + record in commit message).
- [x] **Step 2: server** — curl an image_url request, compare against mlx-vlm's answer for the same image/prompt (semantic equivalence, not token-exact).
- [x] **Step 3: MTP regression** — same image prompt with `--speculative-mtp 4`: acceptance rate sane (≥50%), output coherent; text-only MTP acceptance unchanged (rerun one arm of `scripts/ab_verify_round.sh`).
- [x] **Step 4: commit** — `git commit -m "qwen38: CLI --image and server image_url support"`

## Task 8: Docs + handoff

- [x] **Step 1:** HANDOFF.md new section: what works (CLI/server/MTP-with-images), gates passed, known limits (no video, bilinear resize note, sequential image encoding, vision tower unquantized bf16 memory +~1.3GB).
- [x] **Step 2:** README capability line update (English, one line).
- [x] **Step 3:** commit — `git commit -m "docs: qwen38 vision support handoff"`

---

## Self-review notes

- Spec coverage: config(1) → mrope(2) → tower(3) → preprocess(4) → plumbing(5) → parity(6) → wiring(7) → docs(8). Video/deepstack explicitly out of scope; `rejectVisionTokens` keeps rejecting 248057 (video) and 248055.
- Type consistency: `VisionInput`, `Qwen38Vit.forward` signature, `MropeResult` are used uniformly across Tasks 2/3/5/6/7.
- Risk register (with mitigation): resize resampling mismatch (Task 6 gate + bicubic fallback in Task 4); rope selector layout wrong (Task 2 test vs oracle); MTP/verify offsets with delta (Task 5 test + Task 7 step 3); server base64 memory blowup (reject images > 20 MB decoded).
