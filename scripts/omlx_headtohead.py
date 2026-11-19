#!/usr/bin/env python3
"""Head-to-head PP/TG benchmark: dmlx vs omlx (or any OpenAI-compatible server).

Measures, against one running server:
  - cold prefill tok/s at several prompt sizes (unique prompts, cache-proof)
  - decode tok/s via streaming timestamps (TTFT vs total)
  - warm-cache prefill tok/s (same prompt twice; measures prefix cache)

Usage:
  python3 scripts/omlx_headtohead.py --base-url http://localhost:PORT --label dmlx \
      [--model m] [--out result.json]

Run once per server with identical flags; compare the JSON outputs.
"""
import argparse, json, random, sys, time, urllib.request

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu "
         "xi omicron pi rho sigma tau upsilon phi chi psi omega").split()


def make_prompt(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def chat(base_url, model, prompt, max_tokens, stream):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    if not stream:
        with urllib.request.urlopen(req, timeout=3600) as r:
            data = json.loads(r.read())
        wall = time.perf_counter() - t0
        u = data.get("usage", {})
        return {"wall": wall, "ttft": None,
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens")}
    ttft = None
    completion_tokens = 0
    prompt_tokens = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
                completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    if not chunk.get("usage"):
                        completion_tokens += 1
    wall = time.perf_counter() - t0
    return {"wall": wall, "ttft": ttft,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="m")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefill-sizes", default="1024,4096,8192")
    ap.add_argument("--decode-prompt", type=int, default=1024)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    # Rough words-per-token calibration: measure once, scale prompt sizes.
    probe = chat(args.base_url, args.model, make_prompt(1000, 7), 1, False)
    wpt = 1000 / max(probe["prompt_tokens"] or 1000, 1)

    result = {"label": args.label, "prefill": [], "decode": [], "cache": {}}
    sizes = [int(s) for s in args.prefill_sizes.split(",")]

    # Cold prefill: unique prompt each rep -> no prefix-cache hits.
    for size in sizes:
        words = int(size * wpt)
        best = None
        for rep in range(args.reps):
            r = chat(args.base_url, args.model, make_prompt(words, 1000 * size + rep), 1, False)
            pt = r["prompt_tokens"] or size
            pps = pt / r["wall"]
            if best is None or pps > best["pps"]:
                best = {"target": size, "prompt_tokens": pt, "wall": r["wall"], "pps": pps}
        result["prefill"].append(best)
        print(f"[{args.label}] prefill {best['prompt_tokens']} tok: {best['pps']:.1f} tok/s ({best['wall']:.2f}s)", flush=True)

    # Decode: streaming, TTFT vs total.
    words = int(args.decode_prompt * wpt)
    best = None
    for rep in range(args.reps):
        r = chat(args.base_url, args.model, make_prompt(words, 42 + rep), args.decode_tokens, True)
        if r["ttft"] and r["completion_tokens"] > 1:
            tg = (r["completion_tokens"] - 1) / (r["wall"] - r["ttft"])
            row = {"prompt_tokens": r["prompt_tokens"], "ttft": r["ttft"],
                   "tg": tg, "completion_tokens": r["completion_tokens"]}
            if best is None or tg > best["tg"]:
                best = row
    if best:
        result["decode"].append(best)
        print(f"[{args.label}] decode @{best['prompt_tokens']} ctx: {best['tg']:.1f} tok/s (TTFT {best['ttft']:.2f}s)", flush=True)

    # Warm cache: same prompt twice; second run shows cache-hit prefill speed.
    words = int(4096 * wpt)
    prompt = make_prompt(words, 999)
    cold = chat(args.base_url, args.model, prompt, 1, False)
    warm = chat(args.base_url, args.model, prompt, 1, False)
    pt = warm["prompt_tokens"] or 4096
    result["cache"] = {
        "cold_pps": (cold["prompt_tokens"] or pt) / cold["wall"],
        "warm_wall": warm["wall"], "warm_pps": pt / warm["wall"],
    }
    print(f"[{args.label}] cache-hit prefill: {result['cache']['warm_pps']:.1f} tok/s "
          f"(cold {result['cache']['cold_pps']:.1f})", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
