#!/usr/bin/env python3
"""DSpine Phase 0: six-scenario acceptance/speed probe on M4 Pro.

Same prompts and sampling as the dmlx copy-bands bench (clean_h2h filler,
thinking on/off x short/2k/8k, max_tokens 256). Target: mlx-community
Qwen3.8-27B-4bit; draft: tencent Qwen3.8-27B-DSpine-b16 quantized to 4-bit.
Prompt cache stays on across configs of one cell, so each cell prefills once.

Usage: bench_six.py <out.tsv>
"""
import sys
import time

import mlx.core as mx

from dspine_mlx.generate import DSpineGenerator, load_draft, quantize_draft
from dspine_mlx.target import TargetAdapter

TARGET = "mlx-community/Qwen3.8-27B-4bit"
DRAFT = "tencent/Qwen3.8-27B-DSpine-b16"


def make_prompt(n_tokens_approx):
    base = (
        "You are debugging a Python CLI tool that crashes with KeyError when a "
        "config section is missing. Read the transcript and fix it.\n"
    )
    filler = (
        "src/config.py:42: section = cfg['paths']\n"
        "ASSISTANT: reading config.py around the KeyError.\n"
        "TOOL[read] src/config.py:30-60\n"
        "def load(path):\n    cfg = parse_ini(path)\n"
        "    section = cfg['paths']\n    return [Path(p) for p in section]\n"
    )
    reps = max(1, n_tokens_approx // 75)
    return base + "\n" + filler * reps


def main():
    out_path = sys.argv[1]
    draft, cfg = load_draft(DRAFT)
    draft = quantize_draft(draft, bits=4, group_size=64, mode="affine")
    target = TargetAdapter.load(TARGET, cfg.target_layer_ids)
    draft.bind(target.model)
    tok = target.tokenizer
    target.enable_prompt_cache(True, ctx_rows=cfg.sliding_window)
    stop_ids = [t for t in (
        tok.eos_token_id,
        getattr(tok, "pad_token_id", None),
    ) if t is not None]

    cells = [
        ("think-off", "short", 100, False),
        ("think-off", "2k", 2000, False),
        ("think-off", "8k", 8000, False),
        ("think-on", "short", 100, True),
        ("think-on", "2k", 2000, True),
        ("think-on", "8k", 8000, True),
    ]
    # (temperature, top_k, blocks, reps)
    configs = [
        (0.8, 50, [5, 8], 2),
        (0.0, 0, [5, 8, 16], 1),
    ]

    with open(out_path, "w") as out:
        out.write("cell\tctx\ttemp\tblock\trep\ttokens\trounds\taccept/round\t"
                  "tokens/round\tdecode_tps\tprefill_s\tverify_s\tdraft_s\t"
                  "tail_s\tpeak_gb\n")
        for tname, size, ptok, thinking in cells:
            text = make_prompt(ptok)
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, enable_thinking=thinking)
            for temp, top_k, blocks, reps in configs:
                for block in blocks:
                    gen = DSpineGenerator(target, draft, block_size=block,
                                          profile=True)
                    for rep in range(1, reps + 1):
                        mx.clear_cache()
                        t0 = time.perf_counter()
                        tokens, st = gen.generate(
                            ids, max_tokens=256, temperature=temp,
                            top_k=top_k, top_p=1.0, stop_ids=stop_ids)
                        wall = time.perf_counter() - t0
                        row = (f"{tname}\t{size}\t{temp}\t{block}\t{rep}\t"
                               f"{st.generated_tokens}\t{st.rounds}\t"
                               f"{st.accepted_per_round:.2f}\t"
                               f"{st.tokens_per_round:.2f}\t"
                               f"{st.decode_tps:.2f}\t{st.prefill_seconds:.2f}\t"
                               f"{st.verify_seconds:.2f}\t{st.draft_seconds:.2f}\t"
                               f"{st.tail_seconds:.2f}\t{st.peak_memory_gb:.1f}")
                        print(row.replace("\t", " "), flush=True)
                        out.write(row + "\n")
                        out.flush()
    print("done")


if __name__ == "__main__":
    main()
