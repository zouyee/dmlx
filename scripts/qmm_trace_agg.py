#!/usr/bin/env python3
"""Aggregate DMLX_QMM_TRACE lines: total/count/mean ms by (M, N, via)."""
import re, sys, collections

pat = re.compile(r"\[QMM_TRACE\] (\w+) x=\[(\d+),(\d+),(\d+)\][^\n]*?N=(\d+)[^\n]*? ([\d.]+)ms")
agg = collections.defaultdict(lambda: [0.0, 0])
for line in open(sys.argv[1], errors="replace"):
    m = pat.search(line)
    if not m:
        continue
    via, _b, M, _k, N, ms = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), float(m.group(6))
    agg[(int(M), int(N), via)][0] += ms
    agg[(int(M), int(N), via)][1] += 1

rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
total = sum(v[0] for v in agg.values())
print(f"{'M':>6} {'N':>6} {'via':>6} {'count':>5} {'total_ms':>10} {'mean_ms':>8}")
for (M, N, via), (tot, cnt) in rows[:25]:
    print(f"{M:>6} {N:>6} {via:>6} {cnt:>5} {tot:>10.1f} {tot/cnt:>8.1f}")
print(f"grand total {total:.1f} ms")
