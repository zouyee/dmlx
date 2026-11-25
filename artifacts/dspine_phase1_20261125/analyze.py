#!/usr/bin/env python3
"""DSpine Phase 1 AB/BA analyzer."""
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
LABEL2ARM = {"pass1-A": "A", "pass2-A": "A", "pass1-B": "B", "pass2-B": "B"}
ARM_NAME = {"A": os.environ.get("ARM_A_NAME", "DF2+bands"),
            "B": os.environ.get("ARM_B_NAME", "DSpine")}
pat = re.compile(
    r"(\S+)\s+(think-\w+)/(\S+)/r(\d+)\s+ttft=\s*([\d.]+)s decode=\s*([\d.]+) "
    r"n=(\d+) finish=(\S+)")
rows = []
for path in sorted(glob.glob(f"{OUT}/client_pass*.log")):
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            label, think, size, rep, ttft, dec, n, finish = m.groups()
            arm = LABEL2ARM.get(label)
            if arm:
                rows.append(dict(arm=arm, cell=f"{think}/{size}", decode=float(dec),
                                 n=int(n), finish=finish))
cells = [f"{think}/{size}" for think in ("think-off", "think-on")
         for size in ("short", "mid-2k", "long-8k")]
print("== client decode tok/s (median n=4 [min-max]) ==")
for cell in cells:
    vals = {}
    for arm in ("A", "B"):
        vals[arm] = sorted(r["decode"] for r in rows if r["arm"] == arm and r["cell"] == cell)
    if len(vals["A"]) != 4 or len(vals["B"]) != 4:
        print(f"{cell:22s} incomplete A={vals['A']} B={vals['B']}")
        continue
    ma, mb = statistics.median(vals["A"]), statistics.median(vals["B"])
    print(f"{cell:22s} A={ma:6.2f} [{min(vals['A']):.2f}-{max(vals['A']):.2f}] "
          f"B={mb:6.2f} [{min(vals['B']):.2f}-{max(vals['B']):.2f}] "
          f"delta={(mb/ma-1)*100:+6.1f}%")

print("\n== output comparability ==")
for cell in cells:
    for arm in ("A", "B"):
        rs = [r for r in rows if r["arm"] == arm and r["cell"] == cell]
        if not rs:
            continue
        finishes = defaultdict(int)
        for r in rs:
            finishes[r["finish"]] += 1
        ns = [r["n"] for r in rs]
        print(f"{cell:22s} {arm}: n={ns} finish={dict(finishes)}")

print("\n== server mechanism (request medians) ==")
for label in LABEL2ARM:
    path = f"{OUT}/server_{label}.log"
    if not os.path.exists(path):
        continue
    with open(path, errors="replace") as f:
        text = f.read()
    reqs = re.findall(r"MTP request \d+: acceptance ([\d.]+)% \(\d+/\d+ drafts\); ms/round draft=([\d.]+) verify=([\d.]+) accept=([\d.]+)", text)
    depth_lines = re.findall(r"MTP depth .*", text)
    ets = [float(re.findall(r"E\[t\]=([\d.]+)", line)[-1]) for line in depth_lines
           if re.findall(r"E\[t\]=([\d.]+)", line)]
    if reqs:
        cols = list(zip(*[(float(a), float(d), float(v), float(ac)) for a, d, v, ac in reqs]))
        print(f"{label} ({ARM_NAME[LABEL2ARM[label]]}): requests={len(reqs)} "
              f"accept%={statistics.median(cols[0]):.1f} E[t]={statistics.median(ets):.2f} "
              f"draft_ms={statistics.median(cols[1]):.1f} verify_ms={statistics.median(cols[2]):.1f} "
              f"accept_ms={statistics.median(cols[3]):.1f}")
    mem = f"{OUT}/memory_{label}.txt"
    if os.path.exists(mem):
        with open(mem) as f:
            content = f.read().strip()
        print(f"  memory: {content} (RSS KiB)")
