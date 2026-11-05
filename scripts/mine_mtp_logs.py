#!/usr/bin/env python3
"""Mine CMP_KEEP_PASS_LOGS server logs for MTP round economics.

dmlx lines:
  [Engine] MTP cond-acceptance p1=...(~a/b) ...
  [Engine] MTP request N: acceptance A% (x/y drafts); ms/round draft=.. verify=.. accept=.. commit=..
omlx lines:
  ... MTP[N] finish=.. tokens=T cycles=C tok/cycle=.. accept=x/y (P%) depth[d1=..] d0=.. emits[..] timing[backbone=.. mtp=.. ..]

Prompt mapping: per pass the harness runs code, prose, math, count,
reps_prompt=2 each, and every measured request is preceded by an 8-token
probe (two_point). We classify by tokens: <=16 -> probe, else decode, and
assign decode requests to prompts in cyclic order.
"""
import re, sys, glob, statistics

PROMPT_ORDER = ["code", "prose", "math", "count"]

dm_req = re.compile(r"MTP request (\d+): acceptance ([\d.]+)% \((\d+)/(\d+) drafts\); ms/round draft=([\d.]+) verify=([\d.]+) accept=([\d.]+) commit=([\d.]+)")
dm_cond = re.compile(r"cond-acceptance (.*)")
om_req = re.compile(r"MTP\[(\d+)\] finish=\w+ tokens=(\d+) cycles=(\d+) tok/cycle=([\d.]+) accept=(\d+)/(\d+) \(([\d.]+)%\).*?timing\[backbone=([\d.]+)ms mtp=([\d.]+)ms sample=([\d.]+)ms cache=([\d.]+)ms\]")
om_depth = re.compile(r"depth\[(.*?)\]")

def mine_dmlx(path):
    reqs = []
    conds = []
    for line in open(path, errors="replace"):
        m = dm_req.search(line)
        if m:
            reqs.append(dict(acc=float(m.group(2)), drafts=int(m.group(4)),
                             draft_ms=float(m.group(5)), verify_ms=float(m.group(6)),
                             accept_ms=float(m.group(7)), commit_ms=float(m.group(8))))
            continue
        c = dm_cond.search(line)
        if c:
            conds.append(c.group(1))
    return reqs, conds

def mine_omlx(path):
    reqs = []
    for line in open(path, errors="replace"):
        m = om_req.search(line)
        if m:
            d = om_depth.search(line)
            reqs.append(dict(tokens=int(m.group(2)), cycles=int(m.group(3)),
                             tok_cycle=float(m.group(4)),
                             acc=float(m.group(7)),
                             backbone_ms=float(m.group(8)), mtp_ms=float(m.group(9)),
                             depth=d.group(1) if d else ""))
    return reqs

def report_dmlx(path):
    reqs, conds = mine_dmlx(path)
    dec = [r for r in reqs if r["drafts"] > 16]  # long requests only
    print(f"--- {path}")
    for i, r in enumerate(dec):
        prompt = PROMPT_ORDER[(i // 2) % 4] if len(dec) >= 8 else "?"
        round_ms = r["draft_ms"] + r["verify_ms"] + r["accept_ms"] + r["commit_ms"]
        tok_round = (r["drafts"] * r["acc"] / 100 + 1)
        print(f"  req[{prompt:6}] acc={r['acc']:5.1f}% drafts={r['drafts']:4} "
              f"round={round_ms:6.1f}ms (draft {r['draft_ms']:5.1f} verify {r['verify_ms']:5.1f}) "
              f"~tok/round={tok_round:.2f} est_tps={1000*tok_round/round_ms:.1f}")
    if conds:
        print(f"  last cond-acceptance: {conds[-1]}")

def report_omlx(path):
    reqs = [r for r in mine_omlx(path) if r["tokens"] > 16]
    print(f"--- {path}")
    for i, r in enumerate(reqs):
        prompt = PROMPT_ORDER[(i // 2) % 4] if len(reqs) >= 8 else "?"
        cyc_ms = (r["backbone_ms"] + r["mtp_ms"]) / max(r["cycles"], 1)
        print(f"  req[{prompt:6}] acc={r['acc']:5.1f}% tok/cycle={r['tok_cycle']:.2f} "
              f"cycle={cyc_ms:6.1f}ms (backbone {r['backbone_ms']/max(r['cycles'],1):5.1f} mtp {r['mtp_ms']/max(r['cycles'],1):4.1f}) "
              f"est_tps={1000*r['tok_cycle']/cyc_ms:.1f} depth[{r['depth']}]")

for pat in sys.argv[1:]:
    for path in sorted(glob.glob(pat)):
        if "omlx" in path:
            report_omlx(path)
        else:
            report_dmlx(path)
