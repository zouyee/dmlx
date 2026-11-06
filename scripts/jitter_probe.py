#!/usr/bin/env python3
"""Prefill jitter probe: N back-to-back long-prompt max_tokens=1 requests
against one dmlx server instance. Reports per-request wall time, mean, stdev.

Usage: python3 scripts/jitter_probe.py [mtp|plain] [reps]
Server must already be running on PORT (default 18201).
"""
import json, os, random, statistics, sys, time, urllib.request

PORT = int(os.environ.get("PORT", "18201"))
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
WORDS = int(os.environ.get("PROBE_WORDS", "1600"))  # ~2100 tokens

def mk_prompt(n, seed):
    random.seed(seed)
    W = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta',
         'theta', 'iota', 'kappa')
    return ' '.join(random.choice(W) for _ in range(n))

def probe(prompt):
    body = json.dumps({
        'model': 'm',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 1, 'temperature': 0, 'stream': False,
    }).encode()
    r = urllib.request.Request(
        f'http://localhost:{PORT}/v1/chat/completions', data=body,
        headers={'Content-Type': 'application/json'})
    t = time.perf_counter()
    urllib.request.urlopen(r, timeout=1800).read()
    return time.perf_counter() - t

mode = sys.argv[1] if len(sys.argv) > 1 else 'mtp'
probe(mk_prompt(32, 1))  # warm
probe(mk_prompt(32, 2))
times = [probe(mk_prompt(WORDS, 1000 + i)) for i in range(REPS)]
mean = statistics.mean(times)
sd = statistics.stdev(times)
print(f'mode={mode} reps={REPS} words={WORDS}')
print('times:', ' '.join(f'{t:.2f}' for t in times))
print(f'mean={mean:.3f}s stdev={sd:.3f}s cv={sd/mean*100:.1f}% '
      f'min={min(times):.2f} max={max(times):.2f}')
