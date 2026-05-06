#!/usr/bin/env python3
"""
Upload benchmark results to the benchmark-results branch.

This script:
  1. Reads the latest benchmark result JSON from baselines/
  2. Appends it to benchmark-results/history.jsonl
  3. Commits and pushes to the benchmark-results branch

Usage:
    python scripts/upload_benchmark.py

Environment:
    BENCHMARK_RESULTS_BRANCH — target branch (default: benchmark-results)
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BASELINE_DIR = PROJECT_DIR / "scripts" / "baselines"
RESULTS_BRANCH = os.environ.get("BENCHMARK_RESULTS_BRANCH", "benchmark-results")


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd,
        cwd=cwd or PROJECT_DIR,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout.strip()


def get_git_commit() -> str:
    return run(["git", "rev-parse", "--short", "HEAD"])


def get_git_branch() -> str:
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def find_latest_baseline() -> Path | None:
    """Find the most recently modified baseline JSON file."""
    files = list(BASELINE_DIR.glob("performance_*.json"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def ensure_results_branch() -> Path:
    """Ensure the benchmark-results branch exists locally, create if not."""
    # Check if branch exists locally
    branches = run(["git", "branch", "--list", RESULTS_BRANCH], check=False)
    if RESULTS_BRANCH not in branches:
        # Check if branch exists on origin
        remote_branches = run(
            ["git", "ls-remote", "--heads", "origin", RESULTS_BRANCH],
            check=False,
        )
        if remote_branches:
            run(["git", "fetch", "origin", f"{RESULTS_BRANCH}:{RESULTS_BRANCH}"])
        else:
            # Create orphan branch
            run(["git", "checkout", "--orphan", RESULTS_BRANCH])
            run(["git", "rm", "-rf", "."], check=False)
            # Create a placeholder README
            readme = PROJECT_DIR / "README.md"
            readme.write_text(
                "# Benchmark Results\\n\\n"
                "This branch stores performance benchmark results for the dmlx project.\\n"
                "Do not edit manually — results are uploaded by `scripts/upload_benchmark.py`.\\n"
            )
            run(["git", "add", "README.md"])
            run(["git", "commit", "-m", "init: benchmark-results branch"])
            run(["git", "checkout", "-"])  # go back to original branch

    # Get the worktree path
    worktree_dir = PROJECT_DIR / ".benchmark-worktree"
    if not worktree_dir.exists():
        run(["git", "worktree", "add", str(worktree_dir), RESULTS_BRANCH])
    else:
        # Update worktree
        run(["git", "-C", str(worktree_dir), "pull", "origin", RESULTS_BRANCH], check=False)

    return worktree_dir


def upload() -> None:
    baseline_file = find_latest_baseline()
    if not baseline_file:
        print("❌ No baseline file found in scripts/baselines/")
        print("   Run `make benchmark` first to generate a baseline.")
        sys.exit(1)

    print(f"📄 Found baseline: {baseline_file.name}")

    # Read baseline data
    data = json.loads(baseline_file.read_text())

    # Enrich with upload metadata
    data["_uploaded_at"] = datetime.now(timezone.utc).isoformat()
    data["_source_branch"] = get_git_branch()
    data["_baseline_filename"] = baseline_file.name

    # Prepare results branch
    print(f"🔀 Preparing {RESULTS_BRANCH} branch...")
    worktree = ensure_results_branch()

    # Append to history.jsonl
    history_file = worktree / "history.jsonl"
    with history_file.open("a") as f:
        f.write(json.dumps(data, separators=(",", ":")) + "\n")

    # Also save a per-commit snapshot
    commit = data.get("git_commit", "unknown")
    snapshot_file = worktree / f"snapshot_{commit}.json"
    snapshot_file.write_text(json.dumps(data, indent=2))

    # Commit and push
    print(f"💾 Committing results...")
    run(["git", "-C", str(worktree), "add", "."])

    # Check if there are changes to commit
    status = run(["git", "-C", str(worktree), "status", "--porcelain"], check=False)
    if not status:
        print("⚠️  No changes to upload (results already up to date).")
        return

    run(
        [
            "git",
            "-C",
            str(worktree),
            "commit",
            "-m",
            f"perf: benchmark @{commit} on {data.get('hostname', 'unknown')}",
        ]
    )

    print(f"🚀 Pushing to origin/{RESULTS_BRANCH}...")
    run(["git", "-C", str(worktree), "push", "origin", RESULTS_BRANCH])

    print(f"✅ Uploaded! Dashboard will update shortly.")
    print(f"   URL: https://dmlx.ai/ (after Pages deploy)")


def main() -> None:
    # Check we're in a git repo
    try:
        run(["git", "rev-parse", "--git-dir"])
    except subprocess.CalledProcessError:
        print("❌ Not in a git repository.")
        sys.exit(1)

    upload()


if __name__ == "__main__":
    main()
