#!/usr/bin/env python3
"""Common-basis throughput comparison across local inference engines.

Why this exists
---------------
Cross-engine numbers recorded in HANDOFF were NOT on a common basis: dmlx's
figures were effective tok/s over a whole request (including prefill and the
model's <think> block) while mlx-vlm's were its own generation-only rate, and
the "acceptance %" each engine prints is a different statistic. That made
"we are 4% behind" unfalsifiable. Three separate conclusions in the
2026-10-24 session had to be retracted because of basis mismatches.

So every engine here is driven through its OWN OpenAI-compatible server by
the SAME client with the SAME metric code, and all three rates are derived
from streaming timestamps:

    prefill_tok_s = prompt_tokens / ttft
    decode_tok_s  = (completion_tokens - 1) / (t_last - t_first)
    effective_tok_s = completion_tokens / wall

`decode_tok_s` excludes prefill by construction (it starts at the first
emitted token), which is the quantity to compare against another engine's
"generation tok/s". `effective_tok_s` is what a user feels. Report both;
never mix them.

Protocol, per the project's hard constraints:
  - exactly ONE server runs at a time (co-resident servers cost dmlx ~10% to
    memory pressure, and 48 GB cannot hold two 27B models);
  - engines are interleaved across repetitions so thermal drift hits all of
    them equally (measured: drift moved a whole run 2.5%, interleaving cuts
    run-to-run spread to ~0.2%);
  - warmup requests before any measurement (Metal PSO compilation);
  - the prompt set spans the predictability spectrum. In particular it does
    NOT rely on "Count from 1 to 50.", which is anomalously predictable and
    on which an earlier draft-depth default was overfitted.

Usage:
    python3 scripts/compare_engines.py --engines dmlx,mlx-lm --reps 3
    python3 scripts/compare_engines.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import statistics as st
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# Vocabulary for synthetic prompts. Plain common words so tokenization stays
# close to 1 token/word on this tokenizer and the target lengths are roughly
# honoured; the actual count is read back from each engine's usage field.
WORDS = (
    "time person year way day thing man world life hand part child eye woman "
    "place work week case point government company number group problem fact "
    "water money story month book line city business night question house "
    "power country area school study job word issue side kind head service "
    "friend father hour game line end member law car city community name "
    "president team minute idea kid body information back parent face others "
    "level office door health person art war history party result change "
    "morning reason research girl guy moment air teacher force education"
).split()

MODEL = os.path.expanduser("~/models/Qwen3.8-27B-4bit")
DRAFTER = os.path.expanduser("~/models/Qwen3.8-27B-MTP-4bit")
# Grafted checkpoint that activates omlx's native MTP; see
# scripts/omlx_enable_native_mtp.py. Backbone shards are symlinks into MODEL.
OMLX_MTP_MODEL = os.path.expanduser("~/models/Qwen3.8-27B-4bit-mtp")
PORT = 18099

# Two interpreters, because the engines' Python requirements are disjoint:
# mlx-vlm 0.7.8 needs the 3.14 stack we built on top of the system
# site-packages, while omlx declares `requires-python >=3.11,<3.14` (which is
# why it was never installed into any interpreter on this machine).
CMP_PY = os.path.expanduser("~/venvs/dmlx-cmp/bin/python")    # 3.14: mlx + mlx_lm + mlx_vlm
OMLX_PY = os.path.expanduser("~/venvs/dmlx-omlx/bin/python")  # 3.13: omlx (editable from ~/work/code/omlx)

PROMPTS = {
    "code": "Write a Python function that merges two sorted lists.",
    "prose": "Explain in one paragraph why unified memory helps local LLM inference.",
    "math": "What is 17 * 23? Show your working briefly.",
    # Kept only as an upper-bound reference: it is anomalously predictable
    # (measured conditional acceptance p1..p4 all ~1.0), so tuning on it
    # overfits. Excluded from the headline aggregate.
    "count": "Count from 1 to 50.",
}
HEADLINE = ("code", "prose", "math")

# Prefix-cache nonce.
#
# omlx keeps a 24 GB persisted prefix/GDN-state cache under ~/.omlx/cache and
# scans it at startup to "reuse existing cache files". This harness sends the
# SAME four prompts on every rep of every run, so omlx can serve a prefix it
# already has — and its measured headline rose monotonically across three
# otherwise-identical runs (33.2 -> 37.3 -> 39.1) while its SSD index grew
# (scanned=68/indexed=0 on the first run, scanned=133/indexed=39 later).
#
# A nonce PREPENDED to the user message invalidates any persisted prefix from
# the first token, for every engine equally, while leaving the prompt's nature
# (and therefore its predictability class) intact. Enabled with --nonce.
NONCE: str = ""


def prompt_text(kind: str) -> str:
    """The prompt as sent, with the prefix-cache nonce when enabled."""
    base = PROMPTS[kind]
    return f"[{NONCE}] {base}" if NONCE else base


@dataclass
class Engine:
    name: str
    cmd: list[str]
    # The `model` field each engine expects in the request body. It is NOT
    # uniform: mlx-lm resolves it against HuggingFace (so it must be the real
    # path, or every request 404s), while omlx is a multi-model server whose
    # ids are the SUBDIRECTORY NAMES under --model-dir. dmlx ignores it.
    model_id: str = MODEL
    env: dict = field(default_factory=dict)
    note: str = ""

    def available(self) -> tuple[bool, str]:
        exe = self.cmd[0]
        if exe.endswith("python") or exe.endswith("python3"):
            if not os.path.exists(exe):
                return False, f"missing interpreter {exe}"
            mod = self.cmd[2] if len(self.cmd) > 2 and self.cmd[1] == "-m" else None
            if mod:
                r = subprocess.run(
                    [exe, "-c", f"import importlib.util as u,sys; sys.exit(0 if u.find_spec('{mod.split('.')[0]}') else 1)"],
                    capture_output=True,
                )
                if r.returncode != 0:
                    return False, f"{mod} not importable in {exe}"
            return True, ""
        if shutil.which(exe) is None and not os.path.exists(exe):
            return False, f"{exe} not found"
        return True, ""

    def version(self) -> str:
        exe = self.cmd[0]
        if self.name == "dmlx":
            r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
            return "git:" + r.stdout.strip()
        mod = {
            "mlx-lm": "mlx_lm", "mlx-vlm": "mlx_vlm",
            "mlx-vlm-nospec": "mlx_vlm", "omlx": "omlx",
        }.get(self.name)
        if mod and os.path.exists(exe):
            r = subprocess.run(
                [exe, "-c", f"import {mod};print(getattr({mod},'__version__','?'))"],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                return r.stdout.strip()
        return "?"


def engines() -> dict[str, Engine]:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        # dmlx: adaptive MTP depth (flag is the upper BOUND) + verify kernels.
        #
        # The ceiling is 7 because the verify-shape kernels cover M = 2..8,
        # i.e. draft depths 1..7; depth 8 would put the verify forward at M=9
        # where `pick` returns null and every projection falls back to stock.
        # It was 4 (matching omlx's default) until §38 flattened cost(n) —
        # the verify qmm is nearly flat from M=6 to M=8, so deep chains became
        # cheap. Interleaved A/B, 3 reps: ceiling 4 -> 7 moves the headline
        # 29.0 -> 30.2, count 39.3 -> 48.1 and prose 26.1 -> 29.9, with code
        # 27.3 -> 26.9. Single per-prompt runs disagreed with themselves by
        # 15%, so only the interleaved number is trustworthy here.
        "dmlx": Engine(
            name="dmlx",
            cmd=[os.path.join(here, "zig-out/bin/dmlx"), "serve",
                 "--model", MODEL, "--speculative-mtp", "7",
                 "--port", str(PORT)],
            note="adaptive MTP depth (ceiling 7), verify-shape kernels armed",
        ),
        # Same engine, higher adaptive depth CEILING. The flag is an upper
        # bound, not a fixed depth, so this asks whether the controller can
        # profitably go deeper now that §38's routing fixes flattened cost(n)
        # (the verify qmm is nearly flat from M=6 to M=8). Measured through
        # the interleaved harness because per-prompt single runs disagreed
        # with themselves by 15%.
        "dmlx7": Engine(
            name="dmlx7",
            cmd=[os.path.join(here, "zig-out/bin/dmlx"), "serve",
                 "--model", MODEL, "--speculative-mtp", "7",
                 "--port", str(PORT)],
            note="adaptive MTP, ceiling 7",
        ),
        # A/B arm for the trunk-hidden convention: DMLX_MTP_TRUNK_POSTNORM=1
        # feeds the head the backbone hidden AFTER the final RMSNorm (omlx's
        # `HEAD_HIDDEN_POST_NORM = True` convention). Plan
        # 2026-10-29-catch-omlx-mtp Task 2/4.
        "dmlx-pn": Engine(
            name="dmlx-pn",
            cmd=[os.path.join(here, "zig-out/bin/dmlx"), "serve",
                 "--model", MODEL, "--speculative-mtp", "7",
                 "--port", str(PORT)],
            env={"DMLX_MTP_TRUNK_POSTNORM": "1"},
            note="adaptive MTP ceiling 7 + trunk post-norm head input",
        ),
        # A/B arm: revert the batched commit fold to the sequential replay
        # (ee8cee2). The fold measured perf-neutral at fixed depth on one
        # prompt, but the first full-harness run after landing it read ~8%
        # lower; isolate it before believing that.
        "dmlx-nofold": Engine(
            name="dmlx-nofold",
            cmd=[os.path.join(here, "zig-out/bin/dmlx"), "serve",
                 "--model", MODEL, "--speculative-mtp", "7",
                 "--port", str(PORT)],
            env={"DMLX_MTP_NO_BATCH_FOLD": "1"},
            note="adaptive MTP ceiling 7, sequential commit replay",
        ),
        # A/B arm: MTP-D distilled drafter head (plan
        # 2026-10-30-mtp-drafter-training). Same engine, same trunk; only the
        # drafter checkpoint differs (DMLX_QWEN38_MTP_DIR).
        "dmlx-distilled": Engine(
            name="dmlx-distilled",
            cmd=[os.path.join(here, "zig-out/bin/dmlx"), "serve",
                 "--model", MODEL, "--speculative-mtp", "7",
                 "--port", str(PORT)],
            env={"DMLX_QWEN38_MTP_DIR":
                 os.path.expanduser("~/models/Qwen3.8-27B-MTP-4bit-distilled")},
            note="adaptive MTP ceiling 7, self-distilled drafter",
        ),
        # A/B arm: empty-think generation prompt (HF enable_thinking=false,
        # omlx's thinking_budget_enabled=false default). Parity of GENERATED
        # CONTENT: without it dmlx free-thinks (longer outputs) while omlx
        # answers directly, which confounds decode-rate comparisons.
        "dmlx-nothink": Engine(
            name="dmlx-nothink",
            cmd=[os.path.join(here, "zig-out/bin/dmlx"), "serve",
                 "--model", MODEL, "--speculative-mtp", "7",
                 "--port", str(PORT)],
            env={"DMLX_CHAT_EMPTY_THINK": "1"},
            note="adaptive MTP ceiling 7 + empty-think prompt (omlx parity)",
        ),
        "dmlx-nospec": Engine(
            name="dmlx-nospec",
            cmd=[os.path.join(here, "zig-out/bin/dmlx"), "serve",
                 "--model", MODEL, "--port", str(PORT)],
            note="no speculation (decode baseline)",
        ),
        "mlx-lm": Engine(
            name="mlx-lm",
            cmd=[CMP_PY, "-m", "mlx_lm.server", "--model", MODEL,
                 "--port", str(PORT)],
            note="no speculation (mlx-lm has no MTP for this arch)",
        ),
        # The speculative competitor. Same community drafter dmlx uses, so
        # the drafter is held constant and only the engine differs.
        "mlx-vlm": Engine(
            name="mlx-vlm",
            cmd=[CMP_PY, "-m", "mlx_vlm.server", "--model", MODEL,
                 "--draft-model", DRAFTER, "--draft-kind", "mtp",
                 "--port", str(PORT)],
            note="MTP via --draft-kind mtp, same drafter as dmlx",
        ),
        "mlx-vlm-nospec": Engine(
            name="mlx-vlm-nospec",
            cmd=[CMP_PY, "-m", "mlx_vlm.server", "--model", MODEL,
                 "--port", str(PORT)],
            note="no speculation",
        ),
        # omlx runs in its own 3.13 venv (see OMLX_PY). This row is the
        # no-speculation decode/prefill reference: the community 4-bit export
        # ships no `mtp.*` tensors in the MAIN checkpoint, which is what
        # omlx's native MTP looks for.
        "omlx": Engine(
            name="omlx",
            cmd=[OMLX_PY, "-m", "omlx.cli", "serve",
                 "--model-dir", os.path.dirname(MODEL),
                 "--port", str(PORT)],
            model_id=os.path.basename(MODEL),
            note="no speculation (no embedded mtp.* in this checkpoint)",
        ),
        # omlx WITH its native Lightning MTP, on the grafted checkpoint built
        # by scripts/omlx_enable_native_mtp.py (drafter tensors re-prefixed
        # `mtp.` and merged as one extra shard via omlx's own
        # `combine_mtp_donor`; the 14 GB of backbone shards are symlinks).
        # omlx auto-sets `mtp_enabled` for this model because
        # `lightning_mtp_supported()` is true — confirmed in its log:
        # "Speculative backend selected: Lightning MTP (qwen3_5, active)" and
        # "MTP path activated for uid=...". Same drafter weights dmlx and
        # mlx-vlm use, so all three speculative rows hold the drafter constant
        # and only the engine differs.
        "omlx-mtp": Engine(
            name="omlx-mtp",
            cmd=[OMLX_PY, "-m", "omlx.cli", "serve",
                 "--model-dir", os.path.dirname(OMLX_MTP_MODEL),
                 "--port", str(PORT)],
            model_id=os.path.basename(OMLX_MTP_MODEL),
            note="native Lightning MTP, adaptive depth",
        ),
    }


def wait_health(timeout: int = 240) -> bool:
    for _ in range(timeout):
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(f"http://localhost:{PORT}{path}", timeout=2):
                    return True
            except Exception:
                pass
        time.sleep(1)
    return False


def one_request(prompt: str, max_tokens: int, model_id: str = MODEL) -> dict | None:
    """One NON-streaming completion; returns wall time and usage counts.

    Non-streaming is mandatory, not a convenience: dmlx's MTP speculative
    path (`processMtpRequest`) is implemented for non-streaming requests
    only, so a streaming harness silently measures dmlx WITHOUT speculation
    (measured 14.8 tok/s streaming vs 26.5 non-streaming — i.e. it would have
    reported the plain-decode baseline and called it the speculative result).
    """
    # The model id must be the real path: mlx-lm's server resolves the
    # request's `model` field against HuggingFace, so a placeholder like "d"
    # makes it 404 on every request. dmlx ignores the field.
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    last_err = None
    for path in ("/v1/chat/completions", "/chat/completions"):
        req = urllib.request.Request(
            f"http://localhost:{PORT}{path}",
            data=body, headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                obj = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as e:
            last_err = e
            continue
        wall = time.monotonic() - t0
        usage = obj.get("usage") or {}
        comp = usage.get("completion_tokens")
        prompt_toks = usage.get("prompt_tokens")
        if comp is None:
            return None
        return {"wall": wall, "completion_tokens": int(comp),
                "prompt_tokens": int(prompt_toks) if prompt_toks else None}
    print(f"      ! request failed: {type(last_err).__name__}: {last_err}", file=sys.stderr)
    return None


def random_prompt(target_words: int, rng: "random.Random") -> str:
    """A fresh random-word prompt, to defeat prefix caching.

    Prefix caches would otherwise make the second and later measurements of
    the same prompt report a prefill that never happened (omlx's had to be
    confirmed defeated via cached_tokens=0 in the 2026-10-22 measurements).
    """
    return " ".join(rng.choice(WORDS) for _ in range(target_words))


def ttft_probe(prompt: str, model_id: str = MODEL) -> dict | None:
    """Directly measured TTFT: the wall time of a max_tokens=1 request.

    This is a MEASURED quantity, unlike the earlier derived estimate
    (`wall(8 tokens) - 7/decode_rate`), which depended on the decode estimate
    and only ever saw ~20-token prompts. wall(1) = prefill + one decode step
    + fixed request overhead, which is what a user experiences as TTFT.

    Streaming is deliberately avoided: dmlx's speculative path is
    non-streaming only, and mixing the two would change which code path runs.
    Prefill is identical either way, so max_tokens=1 is both valid and
    uniform across engines.
    """
    r = one_request(prompt, 1, model_id)
    if r is None:
        return None
    return {"ttft_s": r["wall"], "prompt_tokens": r["prompt_tokens"]}


def prefill_sweep(model_id: str, lengths: list[int], reps: int,
                  rng: "random.Random") -> dict:
    """Prefill throughput from two-point differencing over PROMPT length.

    wall(L) at max_tokens=1 is prefill(L) + one decode step + fixed overhead,
    so differencing two prompt lengths cancels the decode step and every
    fixed cost:

        prefill_tok_s = (pt_long - pt_short) / (wall_long - wall_short)

    `pt_*` are the engine's own reported prompt_tokens, so differing
    tokenizations do not bias the comparison. This is the axis the harness was
    missing: short-prompt TTFT is dominated by fixed overhead and says nothing
    about prefill throughput at agent-sized contexts.
    """
    out: dict = {"points": {}}
    for L in lengths:
        best = None
        for _ in range(reps):
            p = random_prompt(L, rng)  # fresh words every attempt
            r = ttft_probe(p, model_id)
            if r is None or not r["prompt_tokens"]:
                continue
            if best is None or r["ttft_s"] < best["ttft_s"]:
                best = r
        if best:
            out["points"][L] = best
    pts = sorted(out["points"].items())
    if len(pts) >= 2:
        (_, lo), (_, hi) = pts[0], pts[-1]
        dpt = hi["prompt_tokens"] - lo["prompt_tokens"]
        dt = hi["ttft_s"] - lo["ttft_s"]
        if dpt > 0 and dt > 0:
            out["prefill_tok_s"] = dpt / dt
            out["span"] = [lo["prompt_tokens"], hi["prompt_tokens"]]
    return out


def two_point(prompt: str, short: int, long_: int, model_id: str = MODEL) -> dict | None:
    """Prefill-free decode rate from two non-streaming requests.

    wall(n) = prefill + (n - 1) * step + fixed, so

        decode_tok_s = (c_long - c_short) / (t_long - t_short)

    cancels prefill and every fixed per-request cost without needing
    streaming timestamps or any engine-reported timing field. The same
    formula is applied to every engine, which is the point.
    """
    a = one_request(prompt, short, model_id)
    if a is None:
        return None
    b = one_request(prompt, long_, model_id)
    if b is None:
        return None
    dc = b["completion_tokens"] - a["completion_tokens"]
    dt = b["wall"] - a["wall"]
    if dc < 16 or dt <= 0:
        return None  # generation stopped early; rate would be meaningless
    decode = dc / dt
    # Back out prefill: the short request's wall minus the decode it did.
    ttft = max(a["wall"] - (a["completion_tokens"] - 1) / decode, 0.0)
    return {
        "decode_tok_s": decode,
        "effective_tok_s": b["completion_tokens"] / b["wall"],
        "ttft_s": ttft,
        "chunks": b["completion_tokens"],
        "wall": b["wall"],
        "prompt_tokens": b["prompt_tokens"],
    }


def measure(engine: Engine, reps_prompt: int, max_tokens: int,
            prefill_lengths: list[int] | None = None) -> dict:
    """Start the engine, warm it, measure every prompt, stop it."""
    subprocess.run(["pkill", "-f", "dmlx serve"], capture_output=True)
    subprocess.run(["pkill", "-f", "mlx_lm.server"], capture_output=True)
    subprocess.run(["pkill", "-f", "mlx_vlm.server"], capture_output=True)
    subprocess.run(["pkill", "-f", "omlx"], capture_output=True)
    time.sleep(2)

    env = dict(os.environ)
    env.update(engine.env)
    log = open(f"/tmp/cmp_{engine.name}.log", "w")
    proc = subprocess.Popen(engine.cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    out: dict = {"ok": False, "per_prompt": {}}
    try:
        if not wait_health():
            out["error"] = "server did not become healthy"
            return out
        for _ in range(3):
            one_request("Hi", 5, engine.model_id)
        for key in PROMPTS:
            runs = []
            for _ in range(reps_prompt):
                r = two_point(prompt_text(key), 8, max_tokens, engine.model_id)
                if r:
                    runs.append(r)
            if not runs:
                continue
            best = max(runs, key=lambda r: r["decode_tok_s"])  # warm best-of
            out["per_prompt"][key] = best
        # Directly measured TTFT at a short prompt, plus the prefill-throughput
        # sweep. Seeded per engine-visit so every engine sees different random
        # text (defeating prefix caches) but the same length targets.
        if prefill_lengths:
            rng = random.Random(0x5EED ^ int(time.time()))
            short_ttft = []
            for _ in range(max(reps_prompt, 2)):
                r = ttft_probe(random_prompt(16, rng), engine.model_id)
                if r:
                    short_ttft.append(r["ttft_s"])
            if short_ttft:
                out["ttft_short_s"] = min(short_ttft)
            out["prefill"] = prefill_sweep(engine.model_id, prefill_lengths,
                                           max(reps_prompt, 2), rng)
        out["ok"] = bool(out["per_prompt"])
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        log.close()
        time.sleep(2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="dmlx,mlx-lm")
    ap.add_argument("--reps", type=int, default=2, help="interleaved passes over the engine set")
    ap.add_argument("--reps-prompt", type=int, default=2, help="requests per prompt inside a pass")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="/tmp/compare_engines.json")
    ap.add_argument("--prefill-lengths", default="512,2048",
                    help="comma-separated prompt word counts for the TTFT/prefill "
                         "sweep; empty string disables it")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--allow-skips", action="store_true",
                    help="report even if some requested engines are unavailable")
    ap.add_argument("--nonce", action="store_true",
                    help="prepend a unique token to every prompt so no engine "
                         "can serve a persisted prefix cache (see NONCE)")
    args = ap.parse_args()
    prefill_lengths = [int(x) for x in args.prefill_lengths.split(",") if x.strip()]

    reg = engines()
    if args.list:
        for name, e in reg.items():
            ok, why = e.available()
            print(f"{name:<12} {'OK ' if ok else 'NO '} {e.version():<10} {why or e.note}")
        return 0

    if args.nonce:
        global NONCE
        NONCE = f"run{int(time.time()) % 100000:05d}"
        print(f"prefix-cache nonce: {NONCE} (every prompt is textually novel)")
    skipped: list[tuple[str, str]] = []
    want = [n.strip() for n in args.engines.split(",") if n.strip()]
    for n in want:
        if n not in reg:
            print(f"unknown engine: {n} (have {', '.join(reg)})", file=sys.stderr)
            return 2
    runnable = []
    for n in want:
        ok, why = reg[n].available()
        if ok:
            runnable.append(n)
        else:
            skipped.append((n, why))
            print(f"skip {n}: {why}", file=sys.stderr)
    if not runnable:
        print("no runnable engines", file=sys.stderr)
        return 1
    # A silently-skipped competitor turns this tool into a dmlx-only run that
    # still prints a comparison-shaped table. That has already happened once:
    # /tmp was cleaned, both venvs vanished, all four competitors were skipped,
    # and the "comparison" contained only our own rows. Refuse by default;
    # --allow-skips is for deliberately measuring a subset.
    if skipped and not args.allow_skips:
        print(
            "refusing to report a partial comparison: "
            + "; ".join(f"{n} ({why})" for n, why in skipped)
            + "\nFix the environment (see scripts/setup_compare_envs.sh) or pass "
              "--allow-skips to measure a subset deliberately.",
            file=sys.stderr,
        )
        return 2

    # Interleave: one pass touches every engine, so drift is shared.
    acc: dict[str, list[dict]] = {n: [] for n in runnable}
    for rep in range(args.reps):
        for n in runnable:
            print(f"[pass {rep + 1}/{args.reps}] {n} ...", flush=True)
            r = measure(reg[n], args.reps_prompt, args.max_tokens, prefill_lengths)
            if r["ok"]:
                acc[n].append(r)
            else:
                print(f"   failed: {r.get('error', 'no data')}", file=sys.stderr)

    doc = {
        "schema": "dmlx-engine-comparison-v1",
        "model": MODEL,
        "max_tokens": args.max_tokens,
        "metric_defs": {
            "prefill_tok_s": "prompt_tokens / ttft (not reported per-prompt: prompt_tokens is engine-reported and not uniformly available over SSE)",
            "decode_tok_s": "(streamed_chunks - 1) / (t_last_chunk - t_first_chunk) — excludes prefill",
            "effective_tok_s": "streamed_chunks / (t_last_chunk - t_request_start) — includes prefill and thinking",
            "caveat": "chunks are SSE content deltas, which equal tokens for all engines here but are not guaranteed to; verify per engine before citing",
        },
        "engines": {},
    }
    print(f"\n{'engine':<12}{'version':<10}{'prompt':<8}{'decode t/s':>12}{'effective':>11}{'ttft s':>9}")
    for n in runnable:
        if not acc[n]:
            continue
        ev = {"version": reg[n].version(), "note": reg[n].note, "per_prompt": {}}
        for key in PROMPTS:
            vals = [p["per_prompt"][key] for p in acc[n] if key in p["per_prompt"]]
            if not vals:
                continue
            ev["per_prompt"][key] = {
                "decode_tok_s": st.median(v["decode_tok_s"] for v in vals),
                "effective_tok_s": st.median(v["effective_tok_s"] for v in vals),
                "ttft_s": st.median(v["ttft_s"] for v in vals),
                "chunks": st.median(v["chunks"] for v in vals),
                "samples": len(vals),
            }
            m = ev["per_prompt"][key]
            tag = key + ("*" if key not in HEADLINE else "")
            print(f"{n:<12}{ev['version']:<10}{tag:<8}{m['decode_tok_s']:>12.1f}"
                  f"{m['effective_tok_s']:>11.1f}{m['ttft_s']:>9.2f}")
        head = [ev["per_prompt"][k]["decode_tok_s"] for k in HEADLINE if k in ev["per_prompt"]]
        if head:
            ev["headline_decode_tok_s"] = st.mean(head)
            print(f"{n:<12}{'':<10}{'HEADLINE':<8}{ev['headline_decode_tok_s']:>12.1f}")
        # TTFT / prefill, aggregated across passes (min = warm best).
        tt = [p["ttft_short_s"] for p in acc[n] if "ttft_short_s" in p]
        if tt:
            ev["ttft_short_s"] = min(tt)
        pf = [p["prefill"] for p in acc[n] if p.get("prefill", {}).get("prefill_tok_s")]
        if pf:
            ev["prefill_tok_s"] = max(x["prefill_tok_s"] for x in pf)
            ev["prefill_span"] = pf[0].get("span")
            ev["prefill_points"] = {
                str(L): v["ttft_s"] for L, v in sorted(pf[0]["points"].items())
            }
        doc["engines"][n] = ev

    # TTFT / prefill table — the axis the first version of this harness did
    # not cover (it only had a DERIVED ttft from a ~20-token prompt).
    have_tt = [n for n in runnable if "ttft_short_s" in doc["engines"].get(n, {})]
    if have_tt:
        print(f"\n{'engine':<12}{'TTFT short':>12}{'prefill t/s':>13}{'span':>14}", end="")
        for L in prefill_lengths:
            print(f"{'ttft@' + str(L):>12}", end="")
        print()
        for n in have_tt:
            e = doc["engines"][n]
            span = e.get("prefill_span")
            print(f"{n:<12}{e['ttft_short_s']:>11.3f}s"
                  f"{e.get('prefill_tok_s', 0):>13.1f}"
                  f"{(str(span[0]) + '-' + str(span[1])) if span else '-':>14}", end="")
            for L in prefill_lengths:
                v = e.get("prefill_points", {}).get(str(L))
                print(f"{v:>11.3f}s" if v else f"{'-':>12}", end="")
            print()

    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\n* excluded from HEADLINE (anomalously predictable prompt)")
    print(f"JSON: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
