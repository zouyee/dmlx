#!/usr/bin/env python3
import json, os, sys, time
import mlx.core as mx
from dspine_mlx.generate import DSpineGenerator, GenStats, load_draft, quantize_draft
from dspine_mlx.target import TargetAdapter

TARGET = "mlx-community/Qwen3.8-27B-4bit"
DRAFT = "tencent/Qwen3.8-27B-DSpine-b16"
BLOCK = int(os.environ.get("DSPINE_BLOCK", "8"))
prompt_text = "Write a Python function that reads an INI config section safely and explain it briefly."
draft, cfg = load_draft(DRAFT)
draft = quantize_draft(draft, bits=4, group_size=64, mode="affine")
target = TargetAdapter.load(TARGET, cfg.target_layer_ids)
draft.bind(target.model)
target.enable_prompt_cache(False)
gen = DSpineGenerator(target, draft, block_size=BLOCK, profile=True)
if os.environ.get("DSPINE_TRACE"):
    original_draft_round = gen._draft_round
    def traced_draft_round(*args, **kwargs):
        drafts, q_rows, cand = original_draft_round(*args, **kwargs)
        print(f"REF drafts={drafts.tolist()}", flush=True)
        return drafts, q_rows, cand
    gen._draft_round = traced_draft_round
ids = target.tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt_text}],
    add_generation_prompt=True, enable_thinking=False)
stop_ids = [t for t in (target.tokenizer.eos_token_id,
                        getattr(target.tokenizer, "pad_token_id", None)) if t is not None]
if os.environ.get("FORCE_ANCHOR"):
    target.reset()
    if os.environ.get("DMLX_TAP_PATH"):
        hidden = mx.load(os.environ["DMLX_TAP_PATH"])["tap"]
        logits = mx.zeros((1, 1, 1), dtype=mx.float32)
    else:
        logits, hidden = target.prefill(mx.array([ids], dtype=mx.int32))
    mx.eval(logits, hidden)
    stats = GenStats(prompt_tokens=len(ids), block_size=BLOCK)
    caches = draft.make_cache()
    anchor = int(os.environ["FORCE_ANCHOR"])
    if os.environ.get("DSPINE_TENSORS"):
        print("REF target_hidden", hidden[0, -1, :8].tolist(), flush=True)
        projected = draft.hidden_norm(draft.fc(hidden))
        print("REF context", projected[0, -1, :8].tolist(), flush=True)
    block_ids = mx.array([[anchor] + [draft.mask_token_id] * (BLOCK - 1)], dtype=mx.int32)
    out, stash = draft(hidden, block_ids, caches)
    rows = out[0, 1:]
    drafts, candidates = gen.tail.greedy(rows, stash, anchor)
    if os.environ.get("DSPINE_TENSORS"):
        print("REF hidden", rows[0, :8].tolist(), flush=True)
        print("REF raw", stash["raw"][1, :8].tolist(), flush=True)
    print(json.dumps({"forced_anchor": anchor,
                      "drafts": drafts.tolist(),
                      "candidates": candidates.tolist()}, ensure_ascii=False))
    sys.exit(0)
t0 = time.perf_counter()
tokens, st = gen.generate(ids, max_tokens=64, temperature=0.0, top_k=0,
                          top_p=1.0, stop_ids=stop_ids)
print(json.dumps({
    "tokens": tokens,
    "text": target.tokenizer.decode(tokens),
    "rounds": st.rounds,
    "tokens_per_round": st.tokens_per_round,
    "decode_tps": st.decode_tps,
    "draft_s": st.draft_seconds,
    "verify_s": st.verify_seconds,
    "tail_s": st.tail_seconds,
    "wall_s": time.perf_counter() - t0,
}, ensure_ascii=False))
