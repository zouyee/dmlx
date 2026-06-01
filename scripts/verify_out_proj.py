#!/usr/bin/env python3
"""S5 go/no-go: verify grouped output-projection LAYOUT matches MLX op chain.

The output projection (DSV4Attention, grouped wo_a path) for decode (B=1,S=1):
  attn_out: [H=64, head_dim=512]  (after inverse tail RoPE)
  -> split into n_groups=8 groups of heads_per_group=8 heads
  -> per group g: group_vec = concat(head[g*8 .. g*8+8] each 512) = 4096
  -> out_g = wo_a[g] @ group_vec                 # [o_lora_rank=1024]
  -> concat 8 groups -> [8192]
  -> wo_b @ [8192] -> [4096]

The numerical pieces (affine matvec, inverse RoPE) are already proven in
S1/S2. The ONLY new risk is the reshape/transpose ORDERING. This script
replicates the exact MLX op chain (reshape [b,ng,hpg,s,hd] -> transpose
[1,0,3,2,4] -> reshape [ng, b*s, hpg*hd]) and compares against the simple
scalar indexing the Metal host code will use (group = h//hpg, head-major
flatten). Match => GO.

Usage: python3 scripts/verify_out_proj.py
"""
import sys
import numpy as np

H = 64
HEAD_DIM = 512
N_GROUPS = 8
HPG = H // N_GROUPS          # heads per group = 8
O_LORA = 1024
GROUP_FEAT = HPG * HEAD_DIM  # 4096


def mlx_style_groupvecs(attn_out):
    """Replicate the MLX reshape/transpose/reshape chain (B=1,S=1)."""
    b, s = 1, 1
    x = attn_out.reshape(b, H, s, HEAD_DIM)            # [1,64,1,512]
    x = x.reshape(b, N_GROUPS, HPG, s, HEAD_DIM)       # [1,8,8,1,512]
    x = np.transpose(x, (1, 0, 3, 2, 4))               # [8,1,1,8,512]
    x = x.reshape(N_GROUPS, b * s, HPG * HEAD_DIM)     # [8,1,4096]
    return x[:, 0, :]                                   # [8, 4096]


def kernel_style_groupvecs(attn_out):
    """Simple scalar indexing the Metal host will use."""
    gv = np.empty((N_GROUPS, GROUP_FEAT), np.float32)
    for g in range(N_GROUPS):
        for hh in range(HPG):
            h = g * HPG + hh
            gv[g, hh * HEAD_DIM:(hh + 1) * HEAD_DIM] = attn_out[h]
    return gv


def main():
    rng = np.random.default_rng(0)
    attn_out = rng.standard_normal((H, HEAD_DIM)).astype(np.float32)

    a = mlx_style_groupvecs(attn_out)
    b = kernel_style_groupvecs(attn_out)
    diff = np.abs(a - b)
    print(f"group-vec layout: shape{a.shape} max_abs_diff={diff.max():.3e}")

    # Also verify the full projection matches a direct dense equivalent.
    # Build random wo_a [8,1024,4096], wo_b [4096,8192]; compute both ways.
    wo_a = rng.standard_normal((N_GROUPS, O_LORA, GROUP_FEAT)).astype(np.float32) * 0.02
    wo_b = rng.standard_normal((4096, N_GROUPS * O_LORA)).astype(np.float32) * 0.02

    # kernel-style: per-group matvec then concat then wo_b
    parts = [wo_a[g] @ b[g] for g in range(N_GROUPS)]   # each [1024]
    cat = np.concatenate(parts)                          # [8192]
    out_kernel = wo_b @ cat                              # [4096]

    # mlx-style group vecs through same math
    parts2 = [wo_a[g] @ a[g] for g in range(N_GROUPS)]
    out_mlx = wo_b @ np.concatenate(parts2)
    pdiff = np.abs(out_kernel - out_mlx)
    print(f"full out-proj: max_abs_diff={pdiff.max():.3e}")

    ok = diff.max() < 1e-6 and pdiff.max() < 1e-4
    print("RESULT:", "GO — grouped out-proj layout matches MLX" if ok else "NO-GO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
