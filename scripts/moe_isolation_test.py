#!/usr/bin/env python3
"""
Standalone MoE isolation test for metal-full DSV4.

1. Load MLX golden L0_ffn_normed.npy.
2. Start `dmlx serve` with MOE_TEST_INJECT_NORMED pointing at the golden input.
3. Trigger a single /v1/chat/completions request (any prompt works; only the
   layer-0 MoE matters because we override its normed input).
4. Compare the dumped L0_ffn_out_metal.bin against MLX golden L0_ffn_out.npy.
"""
import os
import sys
import subprocess
import time
import signal
import glob
import numpy as np
import requests
import tempfile

DUMP_DIR = "/tmp/mlx_dump"
MLX_NORMED = os.path.join(DUMP_DIR, "L0_ffn_normed.npy")
MLX_FFN_OUT = os.path.join(DUMP_DIR, "L0_ffn_out.npy")
METAL_FFN_OUT = os.path.join(DUMP_DIR, "L0_ffn_out_metal.bin")

MODEL = os.path.expanduser("~/models/DeepSeek-V4-Flash-4bit")
EXEC = os.path.abspath("zig-out/bin/dmlx")


def kill_existing_dmlx():
    try:
        subprocess.run(["pkill", "-f", "dmlx serve"], check=False, capture_output=True)
        time.sleep(0.5)
    except Exception:
        pass


def ensure_dump_dir():
    os.makedirs(DUMP_DIR, exist_ok=True)
    for p in [METAL_FFN_OUT]:
        if os.path.exists(p):
            os.remove(p)


def prepare_injected_input():
    if not os.path.exists(MLX_NORMED):
        print(f"[ERR] Missing golden file: {MLX_NORMED}")
        sys.exit(1)
    arr = np.load(MLX_NORMED)
    print(f"[INFO] Loaded L0_ffn_normed shape={arr.shape} dtype={arr.dtype}")
    arr = arr.astype(np.float32).reshape(-1)
    if arr.shape[0] != 4096:
        print(f"[ERR] Expected 4096 floats, got {arr.shape[0]}")
        sys.exit(1)
    bin_path = os.path.join(DUMP_DIR, "L0_ffn_normed_injected.bin")
    arr.tofile(bin_path)
    print(f"[INFO] Wrote injected input to {bin_path} ({arr.shape[0]} floats)")
    return bin_path


def start_server(inject_bin: str):
    env = os.environ.copy()
    env["DSV4_DUMP_DIR"] = DUMP_DIR
    env["MOE_TEST_INJECT_NORMED"] = inject_bin
    env["MF_DBG"] = "1"
    # full-metal engine path (attention + mHC + MoE in engine.c)
    # Optional: limit logging noise
    # env["MLX_METAL_DEBUG"] = "0"
    PACKED_DIR = os.path.join(MODEL, "packed_experts")
    cmd = [
        EXEC, "serve",
        "--model", MODEL,
        "--port", "8930",
        "--max-tokens", "64",
        "--temperature", "0",
        "--expert-packed-dir", PACKED_DIR,
        "--metal-full",
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Wait for server startup (cold load + weight warmup can take 60-90s)
    for i in range(180):
        try:
            r = requests.get("http://127.0.0.1:8930/health", timeout=1)
            if r.status_code == 200:
                print(f"[INFO] Server ready after {i}s")
                return proc
        except requests.exceptions.RequestException:
            pass
        ret = proc.poll()
        if ret is not None:
            stdout, stderr = proc.communicate()
            print(f"[ERR] Server exited early with code {ret}")
            print(stdout)
            print(stderr, file=sys.stderr)
            sys.exit(1)
        time.sleep(1)
    print("[ERR] Server failed to become ready")
    try:
        stdout, stderr = proc.communicate(timeout=5)
        print(stdout)
        print(stderr, file=sys.stderr)
    except Exception:
        pass
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)
    sys.exit(1)


def send_request():
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "2+2="}],
        "max_tokens": 5,
        "temperature": 0,
    }
    r = requests.post("http://127.0.0.1:8930/v1/chat/completions", json=payload, timeout=60)
    print(f"[INFO] Request status={r.status_code}")
    try:
        print("[INFO] Response:", r.json())
    except Exception as e:
        print("[WARN] Could not parse JSON response:", e)


def compare_outputs():
    if not os.path.exists(MLX_FFN_OUT):
        print(f"[ERR] Missing golden: {MLX_FFN_OUT}")
        sys.exit(1)
    if not os.path.exists(METAL_FFN_OUT):
        print(f"[ERR] Missing metal output: {METAL_FFN_OUT}")
        sys.exit(1)

    golden = np.load(MLX_FFN_OUT).astype(np.float32).reshape(-1)
    metal = np.fromfile(METAL_FFN_OUT, dtype=np.float32)

    if golden.shape[0] != metal.shape[0]:
        print(f"[ERR] Shape mismatch: golden={golden.shape} metal={metal.shape}")
        sys.exit(1)

    diff = np.abs(golden - metal)
    rel = diff / (np.abs(golden) + 1e-8)
    rel_l2 = np.sqrt(np.sum((golden - metal) ** 2)) / (np.sqrt(np.sum(golden ** 2)) + 1e-8)

    print(f"[RESULT] golden shape: {golden.shape}")
    print(f"[RESULT] max abs diff:  {diff.max():.6e}")
    print(f"[RESULT] mean abs diff: {diff.mean():.6e}")
    print(f"[RESULT] max rel diff:  {rel.max():.6e}")
    print(f"[RESULT] mean rel diff: {rel.mean():.6e}")
    print(f"[RESULT] rel L2:        {rel_l2:.6e}")

    # Per-component quick view
    worst = int(np.argmax(diff))
    print(f"[RESULT] worst idx={worst} golden={golden[worst]:.6e} metal={metal[worst]:.6e} diff={diff[worst]:.6e}")

    # Save diff for offline analysis
    diff_path = os.path.join(DUMP_DIR, "L0_ffn_out_diff.npy")
    np.save(diff_path, golden - metal)
    print(f"[INFO] Saved residual to {diff_path}")

    return rel_l2


def main():
    if not os.path.exists(EXEC):
        print(f"[ERR] dmlx binary not found at {EXEC}; run `zig build -Doptimize=ReleaseFast` first")
        sys.exit(1)

    ensure_dump_dir()
    inject_bin = prepare_injected_input()
    kill_existing_dmlx()

    proc = None
    try:
        proc = start_server(inject_bin)
        send_request()
        # Give a moment for dump writes to flush
        time.sleep(0.5)
        rel_l2 = compare_outputs()
        if rel_l2 < 1e-4:
            print("[PASS] MoE isolation: output matches MLX golden (rel_L2 < 1e-4)")
        elif rel_l2 < 1e-2:
            print(f"[WARN] MoE isolation: small drift (rel_L2={rel_l2:.3e}); acceptable?")
        else:
            print(f"[FAIL] MoE isolation: significant drift (rel_L2={rel_l2:.3e}); bug likely in MoE path")
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        kill_existing_dmlx()


if __name__ == "__main__":
    main()
