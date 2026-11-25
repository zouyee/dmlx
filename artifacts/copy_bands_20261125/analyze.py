#!/usr/bin/env python3
"""Copy-bands AB/BA analyzer: client decode medians + server-side mechanism
evidence (copy share/acceptance, per-request verify cost).

Usage: analyze.py <artifact_dir>
"""
import re
import statistics
import sys
from collections import defaultdict

OUT = sys.argv[1]
ARMS = {"A": "copies", "B": "copies+bands"}
LABEL2ARM = {"pass1-A": "A", "pass2-A": "A", "pass1-B": "B", "pass2-B": "B"}

# ---- client side ----
rows = []
try:
    with open(f"{OUT}/client_pass1.log") as f:
        lines = f.readlines()
except FileNotFoundError:
    lines = []
# client_{tag}.log fallback (older harness)
import glob as g
for p in sorted(g.glob(f"{OUT}/client_*.log")):
    with open(p) as f:
        lines.extend(f.readlines())

pat = re.compile(
    r"(\S+)\s+(think-\w+)/(\S+)/r(\d+)\s+ttft=\s*([\d.]+)s decode=\s*([\d.]+) "
    r"n=(\d+) finish=(\S+)")
for line in lines:
    m = pat.search(line)
    if not m:
        continue
    label, think, size, rep, ttft, dec, n, finish = m.groups()
    arm = LABEL2ARM.get(label)
    if not arm:
        continue
    rows.append(dict(arm=arm, cell=f"{think}/{size}", rep=int(rep),
                     ttft=float(ttft), decode=float(dec), n=int(n), finish=finish))

cells = sorted({r["cell"] for r in rows})
print("== client decode tok/s (median of n=4, [min-max]) ==")
print(f"{'cell':18s} {'A(copies)':>20s} {'B(+bands)':>20s} {'delta':>8s}")
summary = {}
for cell in cells:
    vals = {}
    for arm in ("A", "B"):
        v = sorted(r["decode"] for r in rows if r["cell"] == cell and r["arm"] == arm)
        vals[arm] = v
    if not vals["A"] or not vals["B"]:
        print(f"{cell:18s} A={vals['A']} B={vals['B']}")
        continue
    ma, mb = statistics.median(vals["A"]), statistics.median(vals["B"])
    d = 100.0 * (mb - ma) / ma
    summary[cell] = d
    print(f"{cell:18s} {ma:7.1f} [{min(vals['A']):.1f}-{max(vals['A']):.1f}]{len(vals['A']):>3d} "
          f"{mb:7.1f} [{min(vals['B']):.1f}-{max(vals['B']):.1f}]{len(vals['B']):>3d} {d:+7.1f}%")

print("\n== output comparability (n tokens: A mean vs B mean; finish reasons) ==")
for cell in cells:
    for arm in ("A", "B"):
        rs = [r for r in rows if r["cell"] == cell and r["arm"] == arm]
        if not rs:
            continue
        ns = [r["n"] for r in rs]
        fs = defaultdict(int)
        for r in rs:
            fs[r["finish"]] += 1
        print(f"  {cell:18s} {arm}: n mean={statistics.mean(ns):.0f} range=[{min(ns)}-{max(ns)}] "
              f"finish={dict(fs)}")

# ---- server side ----
print("\n== server mechanism evidence ==")
for tag, arm in LABEL2ARM.items():
    try:
        with open(f"{OUT}/server_{tag}.log", errors="replace") as f:
            log = f.read()
    except FileNotFoundError:
        continue
    copies = re.findall(r"MTP copies: rounds=(\d+) share=([\d.]+)% acceptance=([\d.]+)% \((\d+)/(\d+)\)", log)
    reqs = re.findall(r"MTP request \d+: acceptance [\d.]+% \(\d+/\d+ drafts\); ms/round draft=([\d.]+) verify=([\d.]+) accept=([\d.]+)", log)
    toks = re.findall(r"tokens_per_sec=([\d.]+)", log)
    share = [float(c[1]) for c in copies]
    acc = [float(c[2]) for c in copies]
    ver = [float(r[1]) for r in reqs]
    print(f"  {tag} ({ARMS[arm]}): requests={len(reqs)} "
          f"copy_rounds={sum(int(c[0]) for c in copies)} "
          f"copy_share_med={statistics.median(share) if share else 0:.0f}% "
          f"copy_acc_med={statistics.median(acc) if acc else 0:.0f}% "
          f"verify_ms_med={statistics.median(ver) if ver else 0:.1f}")
