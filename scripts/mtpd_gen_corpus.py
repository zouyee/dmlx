#!/usr/bin/env python3
"""MTP-D corpus generation: drive the local dmlx server at temp 0.7 over a
mixed prompt pool and record {prompt, text} pairs (on-policy self-distillation
data for the MTP drafter head).

Usage:
    python3 scripts/mtpd_gen_corpus.py [--port 18099] [--out artifacts/mtpd/corpus.jsonl]
                                       [--max-tokens 1024] [--limit N]
Assumes `dmlx serve` is already running on the port (start it with
--speculative-mtp 4 for speed; sampling distribution is unaffected).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request

CODE_TASKS = [
    "merge two sorted lists", "compute the nth Fibonacci number iteratively",
    "implement binary search", "reverse a linked list", "find the longest common subsequence",
    "implement a LRU cache", "parse a CSV file into dicts", "implement Dijkstra's algorithm",
    "validate balanced parentheses", "implement quicksort", "compute edit distance",
    "serialize a binary tree", "implement a thread pool", "find all anagrams in a list",
    "implement a bloom filter", "compute the convex hull of points",
    "implement a rate limiter", "topological sort of a DAG", "implement a trie",
    "find the median of a stream", "implement RSA key generation",
    "write a URL shortener", "implement matrix multiplication", "solve the n-queens problem",
    "implement a simple regex matcher", "compute histogram of word frequencies",
    "implement a JSON parser", "write a tic-tac-toe game", "implement A* search",
    "write a markdown to HTML converter",
]
CODE_LANGS = ["Python", "Rust", "Go", "C++", "TypeScript", "Java"]
PROSE_TOPICS = [
    "why unified memory helps local LLM inference", "the causes of World War I",
    "how photosynthesis works", "the history of the internet", "why sleep matters",
    "the economics of renewable energy", "how vaccines work", "the fall of the Roman Empire",
    "what makes a good API design", "the psychology of habit formation",
    "how compilers optimize code", "the future of electric vehicles",
    "why databases use B-trees", "the physics of rainbows", "how GPS works",
    "the ethics of autonomous vehicles", "why the sky is blue",
    "how neural networks learn", "the history of cryptography", "why islands have unique species",
    "the pros and cons of remote work", "how antibiotics were discovered",
    "why caching is hard", "the cultural impact of the printing press",
    "how earthquakes are measured", "the role of mitochondria in cells",
    "why airplanes fly", "how blockchain consensus works", "the sociology of cities",
    "why exercise improves cognition",
]
MATH_PROBLEMS = [
    "What is {a} * {b}? Show your working briefly.",
    "Solve step by step: {a} + {b} - {c}.",
    "Compute the derivative of x^{n} * e^x.",
    "Compute the integral of x^{n} * e^x dx.",
    "Solve for x: {a}x + {b} = {c}.",
    "What is the greatest common divisor of {a} and {b}?",
    "Simplify: ({a} + {b})^2 - ({a} - {b})^2.",
    "How many ways can you choose 3 items from {n}?",
    "What is {a}% of {b}?",
    "Sum the arithmetic series 1 + 2 + ... + {n}.",
    "Convert {a} from decimal to binary.",
    "What is log2({n})? Explain briefly.",
    "A train travels {a} km in {b} hours. What is its average speed?",
    "Find the roots of x^2 - {a}x + {b} = 0.",
    "What is the probability of rolling a total of 7 with two dice?",
]
REASONING = [
    "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Explain.",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain.",
    "You have 8 balls; one is heavier. How many weighings on a balance do you need? Explain.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
    "A lily pad doubles in size every day and covers the lake in 48 days. When was it half covered?",
    "Explain why 0.999... equals 1.",
    "Why is the derivative of e^x equal to e^x? Give intuition.",
    "Explain the pigeonhole principle with an example.",
    "Why can you not trisect an angle with ruler and compass?",
    "Explain the halting problem to a programmer.",
]


def build_prompts(rng: random.Random) -> list[str]:
    pool: list[str] = []
    for lang in CODE_LANGS:
        for t in CODE_TASKS:
            pool.append(f"Write a {lang} function to {t}.")
            pool.append(f"Write a {lang} program that demonstrates how to {t}.")
    for topic in PROSE_TOPICS:
        pool.append(f"Explain {topic} in a few paragraphs.")
        pool.append(f"Write a short essay about {topic}.")
        pool.append(f"Summarize {topic} for a beginner.")
    for tmpl in MATH_PROBLEMS:
        for _ in range(4):
            pool.append(tmpl.format(
                a=rng.randint(7, 997), b=rng.randint(7, 997),
                c=rng.randint(100, 9999), n=rng.choice([2, 3, 4, 5, 8, 16, 64, 128]),
            ))
    pool.extend(REASONING * 4)
    rng.shuffle(pool)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--out", default="artifacts/mtpd/corpus.jsonl")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()

    prompts = build_prompts(random.Random(args.seed))[: args.limit]
    done_prompts = set()
    try:
        with open(args.out) as f:
            for line in f:
                done_prompts.add(json.loads(line)["prompt"])
    except FileNotFoundError:
        pass

    n_tok = 0
    t0 = time.time()
    with open(args.out, "a") as out:
        for i, p in enumerate(prompts):
            if p in done_prompts:
                continue
            body = json.dumps({
                "model": "m",
                "messages": [{"role": "user", "content": p}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"http://localhost:{args.port}/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    resp = json.load(r)
            except Exception as e:  # noqa: BLE001 - keep going on transient errors
                print(f"[{i}] ERROR {e}", flush=True)
                continue
            msg = resp["choices"][0].get("message") or {}
            text = msg.get("content") or resp["choices"][0].get("text") or ""
            usage = resp.get("usage") or {}
            n_tok += usage.get("completion_tokens", 0)
            out.write(json.dumps({"prompt": p, "text": text}) + "\n")
            out.flush()
            if (i + 1) % 10 == 0:
                dt = time.time() - t0
                print(f"[{i+1}/{len(prompts)}] tokens={n_tok} "
                      f"rate={n_tok/max(dt,1):.1f} tok/s", flush=True)
    print(f"DONE tokens={n_tok} out={args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
