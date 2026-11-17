#!/bin/bash
# Automated git bisect for the "second 1k prefill hangs" regression.
# Run inside the clean worktree. Log: /tmp/bisect_drive.log
set -u
WT="/Users/zouyee/work/code/dmlx/.worktrees/h2h-head"
PROBE="/Users/zouyee/work/code/dmlx/scripts/bisect_probe.sh"
cd "$WT"
git bisect reset 2>/dev/null
git bisect start HEAD 651caba > /tmp/bisect_drive.log 2>&1 || { echo "START_FAILED"; exit 1; }
for step in $(seq 1 12); do
  commit=$(git rev-parse --short HEAD)
  echo "=== step $step building $commit ===" >> /tmp/bisect_drive.log
  if ! zig build -Doptimize=ReleaseFast >> /tmp/bisect_drive.log 2>&1; then
    out=$(git bisect skip 2>&1)
  else
    rm -f /tmp/bisect_verdict.txt
    bash "$PROBE" "$WT" > /tmp/bisect_probe_stdout.log 2>&1
    verdict=$(cat /tmp/bisect_verdict.txt 2>/dev/null)
    verdict=${verdict:-SKIP}
    echo "=== $commit -> $verdict ===" >> /tmp/bisect_drive.log
    case "$verdict" in
      PASS) out=$(git bisect good 2>&1) ;;
      HANG) out=$(git bisect bad 2>&1) ;;
      *)    out=$(git bisect skip 2>&1) ;;
    esac
  fi
  echo "$out" >> /tmp/bisect_drive.log
  if echo "$out" | grep -q "is the first bad commit"; then
    echo "DONE" >> /tmp/bisect_drive.log
    break
  fi
done
pkill -9 -f "zig-out/bin/dmlx serve" 2>/dev/null
tail -30 /tmp/bisect_drive.log
