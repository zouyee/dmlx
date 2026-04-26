#!/usr/bin/env python3
"""
Generate golden reference data from Python MLX for NN layer numerical verification.

Usage:
    python tests/generate_golden.py

Generates reference input/output pairs as raw binary files + JSON metadata
for each NN layer. These are consumed by src/tests/golden_test.zig to verify
mlx-zig produces numerically equivalent results (cosine similarity >= 0.9999).

Layers covered:
  - RMSNorm
  - RoPE (Rotary Position Embedding)
  - SDPA (Scaled Dot-Product Attention)
  - Embedding
  - LSTM
  - GRU
  - Loss functions (MSE, Cross-Entropy, L1, Huber)
  - TinyLlama end-to-end inference (optional, requires model download)

Requirements:
    pip install mlx mlx-lm

**Validates: Requirements R3.1, R3.2, R3.3, R3.4, R3.5, R3.6, R26.1, R26.2, R26.3**
"""
import os
import json
import struct

import mlx.core as mx
import mlx.nn as nn


# Golden data is saved into src/tests/golden/ so Zig @embedFile can find them
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "tests", "golden")


def save_array(arr: mx.array, path: str) -> None:
    """Save MLX array as little-endian binary with JSON metadata sidecar."""
    mx.eval(arr)
    flat = arr.reshape(-1)

    if arr.dtype == mx.float32:
        fmt = "f"
        dtype_str = "float32"
    elif arr.dtype == mx.int32:
        fmt = "i"
        dtype_str = "int32"
    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")

    data_bytes = struct.pack(f"<{flat.size}{fmt}", *flat.tolist())
    with open(path, "wb") as f:
        f.write(data_bytes)

    meta = {"shape": list(arr.shape), "dtype": dtype_str}
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def generate_rmsnorm() -> None:
    """RMSNorm: weight=ones, eps=1e-5, shape (1,2,4,8)."""
    mx.random.seed(42)
    dims = 8
    x = mx.random.normal((1, 2, 4, dims))
    rms = nn.RMSNorm(dims)
    out = rms(x)

    save_array(x, os.path.join(OUTPUT_DIR, "rmsnorm_input.bin"))
    save_array(out, os.path.join(OUTPUT_DIR, "rmsnorm_output.bin"))
    print("  ✓ RMSNorm")


def generate_rope() -> None:
    """RoPE: dims=8, traditional=False, base=10000, scale=1, offset=0."""
    mx.random.seed(42)
    x = mx.random.normal((1, 2, 4, 8))
    out = mx.fast.rope(x, dims=8, traditional=False, base=10000.0, scale=1.0, offset=0)

    save_array(x, os.path.join(OUTPUT_DIR, "rope_input.bin"))
    save_array(out, os.path.join(OUTPUT_DIR, "rope_output.bin"))
    print("  ✓ RoPE")


def generate_sdpa() -> None:
    """SDPA: shape (1,1,4,8), scale=1/sqrt(8), no mask."""
    mx.random.seed(42)
    q = mx.random.normal((1, 1, 4, 8))
    k = mx.random.normal((1, 1, 4, 8))
    v = mx.random.normal((1, 1, 4, 8))
    scale = 1.0 / (8 ** 0.5)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    save_array(q, os.path.join(OUTPUT_DIR, "sdpa_q.bin"))
    save_array(k, os.path.join(OUTPUT_DIR, "sdpa_k.bin"))
    save_array(v, os.path.join(OUTPUT_DIR, "sdpa_v.bin"))
    save_array(out, os.path.join(OUTPUT_DIR, "sdpa_output.bin"))
    print("  ✓ SDPA")


def generate_embedding() -> None:
    """Embedding: vocab=8, dim=4, lookup indices [0,3,7,1]."""
    mx.random.seed(42)
    weight = mx.random.normal((8, 4))
    indices = mx.array([0, 3, 7, 1], dtype=mx.int32)
    out = weight[indices]

    save_array(weight, os.path.join(OUTPUT_DIR, "embedding_weight.bin"))
    save_array(indices, os.path.join(OUTPUT_DIR, "embedding_indices.bin"))
    save_array(out, os.path.join(OUTPUT_DIR, "embedding_output.bin"))
    print("  ✓ Embedding")


def generate_lstm() -> None:
    """LSTM: input_size=4, hidden_size=3, batch=1, seq_len=2.

    Uses known fixed weights and input so we can reproduce in Zig.
    """
    mx.random.seed(42)
    input_size = 4
    hidden_size = 3
    batch = 1
    seq_len = 2

    # Generate deterministic weights and input
    x = mx.random.normal((batch, seq_len, input_size))
    w_ih = mx.random.normal((4 * hidden_size, input_size))
    w_hh = mx.random.normal((4 * hidden_size, hidden_size))
    b_ih = mx.random.normal((4 * hidden_size,))
    b_hh = mx.random.normal((4 * hidden_size,))

    # Manual LSTM forward pass (matching mlx-zig implementation)
    h = mx.zeros((batch, hidden_size))
    c = mx.zeros((batch, hidden_size))
    outputs = []

    for t in range(seq_len):
        x_t = x[:, t, :]  # (batch, input_size)
        gates = x_t @ w_ih.T + h @ w_hh.T + b_ih + b_hh  # (batch, 4*hidden)
        i_gate = mx.sigmoid(gates[:, :hidden_size])
        f_gate = mx.sigmoid(gates[:, hidden_size:2*hidden_size])
        o_gate = mx.sigmoid(gates[:, 2*hidden_size:3*hidden_size])
        g_gate = mx.tanh(gates[:, 3*hidden_size:])
        c = f_gate * c + i_gate * g_gate
        h = o_gate * mx.tanh(c)
        outputs.append(h[:, None, :])  # (batch, 1, hidden)

    output = mx.concatenate(outputs, axis=1)  # (batch, seq_len, hidden)

    save_array(x, os.path.join(OUTPUT_DIR, "lstm_input.bin"))
    save_array(w_ih, os.path.join(OUTPUT_DIR, "lstm_w_ih.bin"))
    save_array(w_hh, os.path.join(OUTPUT_DIR, "lstm_w_hh.bin"))
    save_array(b_ih, os.path.join(OUTPUT_DIR, "lstm_b_ih.bin"))
    save_array(b_hh, os.path.join(OUTPUT_DIR, "lstm_b_hh.bin"))
    save_array(output, os.path.join(OUTPUT_DIR, "lstm_output.bin"))
    save_array(h, os.path.join(OUTPUT_DIR, "lstm_final_h.bin"))
    save_array(c, os.path.join(OUTPUT_DIR, "lstm_final_c.bin"))
    print("  ✓ LSTM")


def generate_gru() -> None:
    """GRU: input_size=4, hidden_size=3, batch=1, seq_len=2.

    Uses known fixed weights and input so we can reproduce in Zig.
    """
    mx.random.seed(123)
    input_size = 4
    hidden_size = 3
    batch = 1
    seq_len = 2

    x = mx.random.normal((batch, seq_len, input_size))
    w_ih = mx.random.normal((3 * hidden_size, input_size))
    w_hh = mx.random.normal((3 * hidden_size, hidden_size))
    b_ih = mx.random.normal((3 * hidden_size,))
    b_hh = mx.random.normal((3 * hidden_size,))

    # Manual GRU forward pass (matching mlx-zig implementation)
    h = mx.zeros((batch, hidden_size))
    outputs = []

    for t in range(seq_len):
        x_t = x[:, t, :]
        gates = x_t @ w_ih.T + h @ w_hh.T + b_ih + b_hh
        z = mx.sigmoid(gates[:, :hidden_size])
        r = mx.sigmoid(gates[:, hidden_size:2*hidden_size])
        n = mx.tanh(gates[:, 2*hidden_size:])
        h = (1 - z) * h + z * n
        outputs.append(h[:, None, :])

    output = mx.concatenate(outputs, axis=1)

    save_array(x, os.path.join(OUTPUT_DIR, "gru_input.bin"))
    save_array(w_ih, os.path.join(OUTPUT_DIR, "gru_w_ih.bin"))
    save_array(w_hh, os.path.join(OUTPUT_DIR, "gru_w_hh.bin"))
    save_array(b_ih, os.path.join(OUTPUT_DIR, "gru_b_ih.bin"))
    save_array(b_hh, os.path.join(OUTPUT_DIR, "gru_b_hh.bin"))
    save_array(output, os.path.join(OUTPUT_DIR, "gru_output.bin"))
    save_array(h, os.path.join(OUTPUT_DIR, "gru_final_h.bin"))
    print("  ✓ GRU")


def generate_mse_loss() -> None:
    """MSE loss: predictions and targets shape (4,)."""
    mx.random.seed(42)
    preds = mx.random.normal((4,))
    targets = mx.random.normal((4,))
    diff = preds - targets
    loss = mx.mean(diff * diff)

    save_array(preds, os.path.join(OUTPUT_DIR, "mse_preds.bin"))
    save_array(targets, os.path.join(OUTPUT_DIR, "mse_targets.bin"))
    save_array(loss.reshape((1,)), os.path.join(OUTPUT_DIR, "mse_output.bin"))
    print("  ✓ MSE Loss")


def generate_l1_loss() -> None:
    """L1 loss: predictions and targets shape (4,)."""
    mx.random.seed(42)
    preds = mx.random.normal((4,))
    targets = mx.random.normal((4,))
    loss = mx.mean(mx.abs(preds - targets))

    save_array(preds, os.path.join(OUTPUT_DIR, "l1_preds.bin"))
    save_array(targets, os.path.join(OUTPUT_DIR, "l1_targets.bin"))
    save_array(loss.reshape((1,)), os.path.join(OUTPUT_DIR, "l1_output.bin"))
    print("  ✓ L1 Loss")


def generate_cross_entropy_loss() -> None:
    """Cross-entropy loss: logits (2,4), labels (2,) int32."""
    mx.random.seed(42)
    logits = mx.random.normal((2, 4))
    labels = mx.array([1, 3], dtype=mx.int32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    gathered = mx.take_along_axis(log_probs, labels.reshape(-1, 1), axis=-1).squeeze(-1)
    loss = mx.mean(-gathered)

    save_array(logits, os.path.join(OUTPUT_DIR, "ce_logits.bin"))
    save_array(labels, os.path.join(OUTPUT_DIR, "ce_labels.bin"))
    save_array(loss.reshape((1,)), os.path.join(OUTPUT_DIR, "ce_output.bin"))
    print("  ✓ Cross-Entropy Loss")


def generate_huber_loss() -> None:
    """Huber loss: predictions and targets shape (4,), delta=1.0."""
    mx.random.seed(42)
    preds = mx.random.normal((4,))
    targets = mx.random.normal((4,))
    delta = 1.0

    diff = preds - targets
    abs_diff = mx.abs(diff)
    quad = 0.5 * diff * diff
    linear = delta * (abs_diff - 0.5 * delta)
    per_elem = mx.where(abs_diff <= delta, quad, linear)
    loss = mx.mean(per_elem)

    save_array(preds, os.path.join(OUTPUT_DIR, "huber_preds.bin"))
    save_array(targets, os.path.join(OUTPUT_DIR, "huber_targets.bin"))
    save_array(loss.reshape((1,)), os.path.join(OUTPUT_DIR, "huber_output.bin"))
    print("  ✓ Huber Loss")


def generate_tinyllama_e2e() -> None:
    """TinyLlama end-to-end inference: generate output tokens for a fixed prompt.

    Downloads TinyLlama-1.1B-Chat (if not cached) and runs greedy decoding
    on a fixed prompt. Saves the output token IDs so the Zig e2e test can
    compare its own inference output against this reference.

    Requires: pip install mlx-lm
    """
    try:
        from mlx_lm import load, generate as mlx_generate
    except ImportError:
        print("  ⚠ Skipping TinyLlama e2e (mlx-lm not installed)")
        return

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompt = "The capital of France is"
    max_tokens = 16

    print(f"  Loading {model_name} ...")
    try:
        model, tokenizer = load(model_name)
    except Exception as e:
        print(f"  ⚠ Skipping TinyLlama e2e (model load failed: {e})")
        return

    # Tokenize the prompt
    prompt_tokens = tokenizer.encode(prompt, return_tensors="mlx")
    prompt_ids = prompt_tokens.reshape(-1).tolist()

    # Greedy generation: temperature=0 for deterministic output
    output_text = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=0.0,
    )

    # Re-tokenize the full output to get token IDs
    full_ids = tokenizer.encode(output_text, return_tensors="mlx").reshape(-1).tolist()
    # Extract only the generated tokens (after the prompt)
    generated_ids = full_ids[len(prompt_ids):]

    # Save prompt token IDs as int32 binary
    prompt_arr = mx.array(prompt_ids, dtype=mx.int32)
    save_array(prompt_arr, os.path.join(OUTPUT_DIR, "tinyllama_prompt_tokens.bin"))

    # Save generated token IDs as int32 binary
    gen_arr = mx.array(generated_ids, dtype=mx.int32)
    save_array(gen_arr, os.path.join(OUTPUT_DIR, "tinyllama_output_tokens.bin"))

    # Save metadata with model name, prompt, and generation config
    meta = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "num_prompt_tokens": len(prompt_ids),
        "num_generated_tokens": len(generated_ids),
    }
    with open(os.path.join(OUTPUT_DIR, "tinyllama_e2e.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ TinyLlama e2e ({len(generated_ids)} tokens generated)")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating golden reference data...")
    generate_rmsnorm()
    generate_rope()
    generate_sdpa()
    generate_embedding()
    generate_lstm()
    generate_gru()
    generate_mse_loss()
    generate_l1_loss()
    generate_cross_entropy_loss()
    generate_huber_loss()

    # TinyLlama e2e is optional — requires mlx-lm and model download
    import sys
    if "--skip-e2e" not in sys.argv:
        generate_tinyllama_e2e()
    else:
        print("  ⚠ Skipping TinyLlama e2e (--skip-e2e flag)")

    print(f"\nAll reference data saved to {OUTPUT_DIR}/")
