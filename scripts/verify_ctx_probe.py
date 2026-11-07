#!/usr/bin/env python3
"""Verify-round cost probe at long context: sends a filler-word prompt of
PROBE_WORDS words (~1.33 tokens/word) with max_tokens=128 greedy, N times,
against a running dmlx MTP server on PORT. The server's per-request log line
([Engine] MTP request ... ms/round ... verify=X) carries the verify cost;
this script only drives load and reports wall time. Compare two server runs
(e.g. with and without DMLX_VERIFY_NO_ATTN_CORE=1) by their logs.

Usage: PORT=18201 PROBE_WORDS=12000 python3 scripts/verify_ctx_probe.py [reps]
"""
import json, os, random, statistics, sys, time, urllib.request

PORT = int(os.environ.get("PORT", "18201"))
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
WORDS = int(os.environ.get("PROBE_WORDS", "12000"))
MAXTOK = int(os.environ.get("PROBE_MAXTOK", "128"))


def mk_prompt(n, seed):
    random.seed(seed)
    W = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta',
         'theta', 'iota', 'kappa')
    return ' '.join(random.choice(W) for _ in range(n))


def probe(prompt):
    body = json.dumps({
        'model': 'm',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': MAXTOK, 'temperature': 0, 'stream': False,
    }).encode()
    r = urllib.request.Request(
        f'http://localhost:{PORT}/v1/chat/completions', data=body,
        headers={'Content-Type': 'application/json'})
    t = time.perf_counter()
    urllib.request.urlopen(r, timeout=1800).read()
    return time.perf_counter() - t


probe(mk_prompt(64, 1))  # warm
times = [probe(mk_prompt(WORDS, 1000 + i)) for i in range(REPS)]
mean = statistics.mean(times)
print(f'words={WORDS} maxtok={MAXTOK} reps={REPS}')
print('times:', ' '.join(f'{t:.2f}' for t in times))
print(f'mean={mean:.3f}s' + (f' stdev={statistics.stdev(times):.3f}s' if len(times) > 1 else ''))
