// DeepSeek V4 MoE Metal kernels — MXFP4 E2M1 format: group_size=32, uint8 E8M0 scales, no biases.
// Verified against MLX fp4_e2m1 (../mlx/mlx/backend/metal/kernels/fp4.h):
//   w = NIBBLE_TO_FLOAT[nibble] * exp2(scale - 127)
// Note: MLX fp8_e8m0 uses bias=127 (not 128!). Our packed scales use the same convention.
// E2M1 positive LUT: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
// gate/up: [2048, 4096], down: [4096, 2048]
#include <metal_stdlib>
using namespace metal;

// P2a: MLX-aligned fp8_e8m0 scale decode — single bit-shift, no FPU transcendental.
// Equivalent to exp2((float)s - 127.0f) but uses IEEE 754 bit manipulation:
//   (uint)s << 23 places the 8-bit biased exponent directly into float exponent field.
// ~5x faster than exp2() on Apple Silicon (avoids FPU transcendental pipeline).
// Source: mlx/backend/metal/kernels/fp_quantized.h dequantize_scale<T, group_size>()
static inline float fp8_e8m0_to_float(uint8_t s) {
    return as_type<float>((uint)s << 23);
}

constant float NIBBLE_TO_FLOAT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Read bf16 value from uint16_t pointer and convert to f32.
inline float bf16_to_f32(device const uint16_t* p) {
    return float(*(device const bfloat*)p);
}

// fused_gate_up_swiglu — SIMD-optimized MXFP4 fused gate+up+SwiGLU.
//
// Strategy (matches flash-moe dequant_matvec_4bit_fast pattern):
//   - One threadgroup of 256 threads = 8 SIMD groups of 32.
//   - Each SIMD group handles ONE output row.
//   - 32 threads in a SIMD group stripe across num_groups (128 groups for gs=32, in=4096).
//   - Input vector x (4096 floats = 16KB) is cached in threadgroup shared memory.
//   - simd_sum reduction: single instruction, no explicit barrier.
//
// Dispatch: MTLSizeMake(out_dim/8, 1, 1) threadgroups × 256 threads.
// (For out_dim=2048: 256 threadgroups × 256 threads = 65536 threads)
//
// Memory savings vs naive: x read ONCE per threadgroup (16KB) vs 256× per row.
// Speedup vs naive: ~20-40× from combined bandwidth + parallelism improvements.
kernel void fused_gate_up_swiglu(
    device const uint32_t* gate_W    [[buffer(0)]],
    device const uint8_t*  gate_s    [[buffer(1)]],
    device const uint32_t* up_W      [[buffer(2)]],
    device const uint8_t*  up_s      [[buffer(3)]],
    device const float*    x         [[buffer(4)]],
    device float*          out       [[buffer(5)]],
    constant uint&         out_dim   [[buffer(6)]],
    constant uint&         in_dim    [[buffer(7)]],
    constant uint&         group_size [[buffer(8)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles ROWS_PER_TG=8 output rows (one per SIMD group).
    const uint ROWS_PER_TG = 8;
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache input x in threadgroup shared memory (4096 floats = 16KB).
    // ALL 256 threads cooperate — mandatory before any early return.
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* g_row = gate_W + row * packed_cols;
    device const uint8_t*  g_s   = gate_s + row * num_groups;
    device const uint32_t* u_row = up_W   + row * packed_cols;
    device const uint8_t*  u_s   = up_s   + row * num_groups;

    float gate_acc = 0.0f, up_acc = 0.0f;
    uint packed_per_group = group_size / 8;

    // P2b: packs_per_thread=2 (MLX fp_qmv_fast alignment).
    // Process 2 packed uint32_t words per inner step → 2 independent load chains,
    // GPU scheduler can overlap both; halves inner-loop iterations for gs=32 (4→2).
    // Stripe across groups: lane k processes groups k, k+32, k+64, ...
    for (uint g = simd_lane; g < num_groups; g += 32) {
        float gsf = exp2((float)g_s[g] - 127.0f);
        float usf = exp2((float)u_s[g] - 127.0f);
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        // Unroll 2 packs per step (packs_per_thread=2, 8 nibbles each = 16 values/step)
        for (uint p = 0; p < packed_per_group; p += 2) {
            uint32_t gpw0 = g_row[bp + p];
            uint32_t gpw1 = g_row[bp + p + 1];
            uint32_t upw0 = u_row[bp + p];
            uint32_t upw1 = u_row[bp + p + 1];
            uint x0 = bx + p * 8;
            uint x1 = x0 + 8;
            // Pack 0
            gate_acc += NIBBLE_TO_FLOAT[(gpw0 >>  0) & 0xF] * gsf * x_shared[x0 + 0];
            gate_acc += NIBBLE_TO_FLOAT[(gpw0 >>  4) & 0xF] * gsf * x_shared[x0 + 1];
            gate_acc += NIBBLE_TO_FLOAT[(gpw0 >>  8) & 0xF] * gsf * x_shared[x0 + 2];
            gate_acc += NIBBLE_TO_FLOAT[(gpw0 >> 12) & 0xF] * gsf * x_shared[x0 + 3];
            gate_acc += NIBBLE_TO_FLOAT[(gpw0 >> 16) & 0xF] * gsf * x_shared[x0 + 4];
            gate_acc += NIBBLE_TO_FLOAT[(gpw0 >> 20) & 0xF] * gsf * x_shared[x0 + 5];
            gate_acc += NIBBLE_TO_FLOAT[(gpw0 >> 24) & 0xF] * gsf * x_shared[x0 + 6];
            gate_acc += NIBBLE_TO_FLOAT[(gpw0 >> 28) & 0xF] * gsf * x_shared[x0 + 7];
            // Pack 1
            gate_acc += NIBBLE_TO_FLOAT[(gpw1 >>  0) & 0xF] * gsf * x_shared[x1 + 0];
            gate_acc += NIBBLE_TO_FLOAT[(gpw1 >>  4) & 0xF] * gsf * x_shared[x1 + 1];
            gate_acc += NIBBLE_TO_FLOAT[(gpw1 >>  8) & 0xF] * gsf * x_shared[x1 + 2];
            gate_acc += NIBBLE_TO_FLOAT[(gpw1 >> 12) & 0xF] * gsf * x_shared[x1 + 3];
            gate_acc += NIBBLE_TO_FLOAT[(gpw1 >> 16) & 0xF] * gsf * x_shared[x1 + 4];
            gate_acc += NIBBLE_TO_FLOAT[(gpw1 >> 20) & 0xF] * gsf * x_shared[x1 + 5];
            gate_acc += NIBBLE_TO_FLOAT[(gpw1 >> 24) & 0xF] * gsf * x_shared[x1 + 6];
            gate_acc += NIBBLE_TO_FLOAT[(gpw1 >> 28) & 0xF] * gsf * x_shared[x1 + 7];
            // Pack 0 (up)
            up_acc += NIBBLE_TO_FLOAT[(upw0 >>  0) & 0xF] * usf * x_shared[x0 + 0];
            up_acc += NIBBLE_TO_FLOAT[(upw0 >>  4) & 0xF] * usf * x_shared[x0 + 1];
            up_acc += NIBBLE_TO_FLOAT[(upw0 >>  8) & 0xF] * usf * x_shared[x0 + 2];
            up_acc += NIBBLE_TO_FLOAT[(upw0 >> 12) & 0xF] * usf * x_shared[x0 + 3];
            up_acc += NIBBLE_TO_FLOAT[(upw0 >> 16) & 0xF] * usf * x_shared[x0 + 4];
            up_acc += NIBBLE_TO_FLOAT[(upw0 >> 20) & 0xF] * usf * x_shared[x0 + 5];
            up_acc += NIBBLE_TO_FLOAT[(upw0 >> 24) & 0xF] * usf * x_shared[x0 + 6];
            up_acc += NIBBLE_TO_FLOAT[(upw0 >> 28) & 0xF] * usf * x_shared[x0 + 7];
            // Pack 1 (up)
            up_acc += NIBBLE_TO_FLOAT[(upw1 >>  0) & 0xF] * usf * x_shared[x1 + 0];
            up_acc += NIBBLE_TO_FLOAT[(upw1 >>  4) & 0xF] * usf * x_shared[x1 + 1];
            up_acc += NIBBLE_TO_FLOAT[(upw1 >>  8) & 0xF] * usf * x_shared[x1 + 2];
            up_acc += NIBBLE_TO_FLOAT[(upw1 >> 12) & 0xF] * usf * x_shared[x1 + 3];
            up_acc += NIBBLE_TO_FLOAT[(upw1 >> 16) & 0xF] * usf * x_shared[x1 + 4];
            up_acc += NIBBLE_TO_FLOAT[(upw1 >> 20) & 0xF] * usf * x_shared[x1 + 5];
            up_acc += NIBBLE_TO_FLOAT[(upw1 >> 24) & 0xF] * usf * x_shared[x1 + 6];
            up_acc += NIBBLE_TO_FLOAT[(upw1 >> 28) & 0xF] * usf * x_shared[x1 + 7];
        }
    }

    // SIMD reduction: sum 32 lanes → lane 0 holds the final value.
    float gate_val = simd_sum(gate_acc);
    float up_val   = simd_sum(up_acc);

    if (simd_lane == 0) {
        // Limited SwiGLU (swiglu_limit=10, matches MLX DSV4SwitchGLU.limitedSwiGLU)
        const float limit = 10.0f;
        float g_c = min(gate_val, limit);
        float u_c = min(max(up_val, -limit), limit);
        out[row] = (bfloat)((g_c / (1.0f + exp(-g_c))) * u_c);
    }
}

// fused_gate_up_swiglu_v2 — ds4 no-x_shared coalesced pattern for MoE gate+up.
//
// Design (ds4 Q8_0 pattern adapted for MXFP4, group_size=32):
//   - NR0=2 rows per threadgroup, NSG=4 simdgroups, 32 threads each → 128 threads/TG
//   - Shared memory: 32 * 2 * sizeof(float) = 256 bytes (reduction only)
//   - NO x_shared — no 16KB occupancy penalty
//   - Coalesced access: 4 threads per group word, 8 groups per SIMD group
//   - Each thread reads 8 consecutive x values, adjacent threads read adjacent blocks
//
// For gate/up [2048, 4096]: 2048/2 = 1024 threadgroups, 128 threads each = 131K threads.
//   num_groups=128, packed_per_group=4. Each SIMD group: 32 threads → 8 groups per iteration,
//   4 iterations total.
kernel void fused_gate_up_swiglu_v2(
    device const uint32_t* gate_W    [[buffer(0)]],
    device const uint8_t*  gate_s    [[buffer(1)]],
    device const uint32_t* up_W      [[buffer(2)]],
    device const uint8_t*  up_s      [[buffer(3)]],
    device const float*    x         [[buffer(4)]],
    device float*          out       [[buffer(5)]],
    constant uint&         out_dim   [[buffer(6)]],
    constant uint&         in_dim    [[buffer(7)]],
    constant uint&         group_size [[buffer(8)]],
    threadgroup float*     shmem     [[threadgroup(0)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 2, NSG = 4, NW = 32, NQ = 8, TPG = 4;

    const uint num_groups = in_dim / group_size;
    const uint packed_per_group = group_size / 8;  // = 4 for gs=32
    const uint packed_cols = in_dim / 8;

    const int row0 = (int)tgpig.x * NR0;
    const short ix = tiisg / TPG, il = tiisg % TPG;
    const int g0 = (int)sgitg * NQ + (int)ix;

    device const uint32_t *gr[NR0], *ur[NR0];
    device const uint8_t  *gs[NR0], *us[NR0];
    for (short row = 0; row < NR0; row++) {
        int r = row0 + row;
        if (r < (int)out_dim) {
            gr[row] = gate_W + r * packed_cols;
            gs[row] = gate_s + r * num_groups;
            ur[row] = up_W   + r * packed_cols;
            us[row] = up_s   + r * num_groups;
        }
    }

    float gate_sum[NR0] = { 0.0f };
    float up_sum[NR0]   = { 0.0f };

    for (int gg = g0; gg < (int)num_groups; gg += NSG * NQ) {
        uint xb = (uint)gg * group_size + (uint)il * 8;
        float xv0 = x[xb+0], xv1 = x[xb+1], xv2 = x[xb+2], xv3 = x[xb+3];
        float xv4 = x[xb+4], xv5 = x[xb+5], xv6 = x[xb+6], xv7 = x[xb+7];

        for (short row = 0; row < NR0; row++) {
            int r = row0 + row;
            if (r >= (int)out_dim) continue;

            float gsf = exp2((float)gs[row][gg] - 127.0f);
            float usf = exp2((float)us[row][gg] - 127.0f);
            uint gpw = gr[row][gg * packed_per_group + (uint)il];
            uint upw = ur[row][gg * packed_per_group + (uint)il];

            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>> 0)&0xF] * gsf * xv0;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>> 4)&0xF] * gsf * xv1;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>> 8)&0xF] * gsf * xv2;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>>12)&0xF] * gsf * xv3;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>>16)&0xF] * gsf * xv4;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>>20)&0xF] * gsf * xv5;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>>24)&0xF] * gsf * xv6;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>>28)&0xF] * gsf * xv7;

            up_sum[row] += NIBBLE_TO_FLOAT[(upw>> 0)&0xF] * usf * xv0;
            up_sum[row] += NIBBLE_TO_FLOAT[(upw>> 4)&0xF] * usf * xv1;
            up_sum[row] += NIBBLE_TO_FLOAT[(upw>> 8)&0xF] * usf * xv2;
            up_sum[row] += NIBBLE_TO_FLOAT[(upw>>12)&0xF] * usf * xv3;
            up_sum[row] += NIBBLE_TO_FLOAT[(upw>>16)&0xF] * usf * xv4;
            up_sum[row] += NIBBLE_TO_FLOAT[(upw>>20)&0xF] * usf * xv5;
            up_sum[row] += NIBBLE_TO_FLOAT[(upw>>24)&0xF] * usf * xv6;
            up_sum[row] += NIBBLE_TO_FLOAT[(upw>>28)&0xF] * usf * xv7;
        }
    }

    threadgroup float *shmem_f32[NR0*2];
    for (short row = 0; row < NR0; row++) {
        shmem_f32[row*2]   = shmem + NW * (row*2);
        shmem_f32[row*2+1] = shmem + NW * (row*2+1);
        if (sgitg == 0) {
            shmem_f32[row*2][tiisg] = 0.0f;
            shmem_f32[row*2+1][tiisg] = 0.0f;
        }
        gate_sum[row] = simd_sum(gate_sum[row]);
        up_sum[row]   = simd_sum(up_sum[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; row++) {
        if (tiisg == 0) {
            shmem_f32[row*2][sgitg]   = gate_sum[row];
            shmem_f32[row*2+1][sgitg] = up_sum[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float limit = 10.0f;
    for (short row = 0; row < NR0; row++) {
        const int d = row0 + row;
        if (d >= (int)out_dim) continue;
        float gv = simd_sum(shmem_f32[row*2][tiisg]);
        float uv = simd_sum(shmem_f32[row*2+1][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float g_c = min(gv, limit);
            float u_c = min(max(uv, -limit), limit);
            out[d] = (bfloat)((g_c / (1.0f + exp(-g_c))) * u_c);
        }
    }
}

kernel void fused_gate_up_swiglu_v2_affine(
    device const uint32_t* gate_W    [[buffer(0)]],
    device const uint16_t* gate_s    [[buffer(1)]],  // bf16 scales
    device const uint16_t* gate_b    [[buffer(2)]],  // bf16 biases
    device const uint32_t* up_W      [[buffer(3)]],
    device const uint16_t* up_s      [[buffer(4)]],  // bf16 scales
    device const uint16_t* up_b      [[buffer(5)]],  // bf16 biases
    device const float*    x         [[buffer(6)]],
    device float*          out       [[buffer(7)]],
    constant uint&         out_dim   [[buffer(8)]],
    constant uint&         in_dim    [[buffer(9)]],
    constant uint&         group_size [[buffer(10)]],
    threadgroup float*     shmem     [[threadgroup(0)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 2, NSG = 4, NW = 32, NQ = 4, TPG = 8;

    const uint num_groups = in_dim / group_size;
    const uint packed_per_group = group_size / 8;  // = 8 for gs=64
    const uint packed_cols = in_dim / 8;

    const int row0 = (int)tgpig.x * NR0;
    const short ix = tiisg / TPG, il = tiisg % TPG;
    const int g0 = (int)sgitg * NQ + (int)ix;

    device const uint32_t *gr[NR0], *ur[NR0];
    device const uint16_t *gs[NR0], *gb[NR0], *us[NR0], *ub[NR0];
    for (short row = 0; row < NR0; row++) {
        int r = row0 + row;
        if (r < (int)out_dim) {
            gr[row] = gate_W + r * packed_cols;
            gs[row] = gate_s + r * num_groups;
            gb[row] = gate_b + r * num_groups;
            ur[row] = up_W   + r * packed_cols;
            us[row] = up_s   + r * num_groups;
            ub[row] = up_b   + r * num_groups;
        }
    }

    float gate_sum[NR0] = { 0.0f };
    float up_sum[NR0]   = { 0.0f };

    for (int gg = g0; gg < (int)num_groups; gg += NSG * NQ) {
        uint xb = (uint)gg * group_size + (uint)il * 8;
        float xv0 = x[xb+0], xv1 = x[xb+1], xv2 = x[xb+2], xv3 = x[xb+3];
        float xv4 = x[xb+4], xv5 = x[xb+5], xv6 = x[xb+6], xv7 = x[xb+7];

        for (short row = 0; row < NR0; row++) {
            int r = row0 + row;
            if (r >= (int)out_dim) continue;

            float gsf = bf16_to_f32(&gs[row][gg]);
            float gbf = bf16_to_f32(&gb[row][gg]);
            float usf = bf16_to_f32(&us[row][gg]);
            float ubf = bf16_to_f32(&ub[row][gg]);

            uint gpw = gr[row][gg * packed_per_group + (uint)il];
            uint upw = ur[row][gg * packed_per_group + (uint)il];

            gate_sum[row] += fma(float((gpw>> 0)&0xF), gsf, gbf) * xv0;
            gate_sum[row] += fma(float((gpw>> 4)&0xF), gsf, gbf) * xv1;
            gate_sum[row] += fma(float((gpw>> 8)&0xF), gsf, gbf) * xv2;
            gate_sum[row] += fma(float((gpw>>12)&0xF), gsf, gbf) * xv3;
            gate_sum[row] += fma(float((gpw>>16)&0xF), gsf, gbf) * xv4;
            gate_sum[row] += fma(float((gpw>>20)&0xF), gsf, gbf) * xv5;
            gate_sum[row] += fma(float((gpw>>24)&0xF), gsf, gbf) * xv6;
            gate_sum[row] += fma(float((gpw>>28)&0xF), gsf, gbf) * xv7;

            up_sum[row]   += fma(float((upw>> 0)&0xF), usf, ubf) * xv0;
            up_sum[row]   += fma(float((upw>> 4)&0xF), usf, ubf) * xv1;
            up_sum[row]   += fma(float((upw>> 8)&0xF), usf, ubf) * xv2;
            up_sum[row]   += fma(float((upw>>12)&0xF), usf, ubf) * xv3;
            up_sum[row]   += fma(float((upw>>16)&0xF), usf, ubf) * xv4;
            up_sum[row]   += fma(float((upw>>20)&0xF), usf, ubf) * xv5;
            up_sum[row]   += fma(float((upw>>24)&0xF), usf, ubf) * xv6;
            up_sum[row]   += fma(float((upw>>28)&0xF), usf, ubf) * xv7;
        }
    }

    threadgroup float *shmem_f32[NR0*2];
    for (short row = 0; row < NR0; row++) {
        shmem_f32[row*2]   = shmem + NW * (row*2);
        shmem_f32[row*2+1] = shmem + NW * (row*2+1);
        if (sgitg == 0) {
            shmem_f32[row*2][tiisg] = 0.0f;
            shmem_f32[row*2+1][tiisg] = 0.0f;
        }
        gate_sum[row] = simd_sum(gate_sum[row]);
        up_sum[row]   = simd_sum(up_sum[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; row++) {
        if (tiisg == 0) {
            shmem_f32[row*2][sgitg]   = gate_sum[row];
            shmem_f32[row*2+1][sgitg] = up_sum[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float limit = 10.0f;
    for (short row = 0; row < NR0; row++) {
        const int d = row0 + row;
        if (d >= (int)out_dim) continue;
        float gv = simd_sum(shmem_f32[row*2][tiisg]);
        float uv = simd_sum(shmem_f32[row*2+1][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float g_c = min(gv, limit);
            float u_c = min(max(uv, -limit), limit);
            out[d] = (bfloat)((g_c / (1.0f + exp(-g_c))) * u_c);
        }
    }
}

// dequant_matvec_4bit_affine — ds4 no-x_shared affine 4-bit down_proj [4096, 2048] gs=64.
//
// Same strategy as fused_gate_up_swiglu: 8 SIMD groups per threadgroup,
// each SIMD group handles one output row with simd_sum reduction.
// x (2048 floats = 8KB) cached in threadgroup shared memory.
//
// Dispatch: MTLSizeMake(out_dim/8, 1, 1) threadgroups × 256 threads.
// (For out_dim=4096: 512 threadgroups × 256 threads)

// dequant_matvec_4bit_affine — ds4 no-x_shared affine 4-bit down_proj [4096, 2048] gs=64.

kernel void dequant_matvec_4bit_affine(
    device const uint32_t* W_packed [[buffer(0)]],
    device const uint16_t* scales   [[buffer(1)]],  // bf16
    device const uint16_t* biases   [[buffer(2)]],  // bf16
    device const float*    x        [[buffer(3)]],
    device float*          out      [[buffer(4)]],
    constant uint&         out_dim  [[buffer(5)]],
    constant uint&         in_dim   [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    threadgroup float*     shmem    [[threadgroup(0)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 2, NSG = 4, NW = 32;
    const short TPG = 8;  // threads per group: packed_per_group = group_size/8 = 8 for gs=64
    const short NQ  = 4;  // groups per SIMD group: NW/TPG = 32/8 = 4

    const uint num_groups = in_dim / group_size;
    const uint packed_per_group = group_size / 8;
    const uint packed_cols = in_dim / 8;

    const int row0 = (int)tgpig.x * NR0;
    const short ix = tiisg / TPG, il = tiisg % TPG;
    const int g0 = (int)sgitg * NQ + (int)ix;

    device const uint32_t *wr[NR0];
    device const uint16_t *sr[NR0];
    device const uint16_t *br[NR0];
    for (short row = 0; row < NR0; row++) {
        int r = row0 + row;
        if (r < (int)out_dim) {
            wr[row] = W_packed + r * packed_cols;
            sr[row] = scales   + r * num_groups;
            br[row] = biases   + r * num_groups;
        }
    }

    float sumf[NR0] = { 0.0f };

    for (int gg = g0; gg < (int)num_groups; gg += NSG * NQ) {
        uint xb = (uint)gg * group_size + (uint)il * 8;
        float xv0 = x[xb+0], xv1 = x[xb+1], xv2 = x[xb+2], xv3 = x[xb+3];
        float xv4 = x[xb+4], xv5 = x[xb+5], xv6 = x[xb+6], xv7 = x[xb+7];

        for (short row = 0; row < NR0; row++) {
            int r = row0 + row;
            if (r >= (int)out_dim) continue;

            float sf = bf16_to_f32(&sr[row][gg]);
            float bf = bf16_to_f32(&br[row][gg]);
            uint32_t pw = wr[row][gg * packed_per_group + (uint)il];

            sumf[row] += fma(float((pw>> 0)&0xF), sf, bf) * xv0;
            sumf[row] += fma(float((pw>> 4)&0xF), sf, bf) * xv1;
            sumf[row] += fma(float((pw>> 8)&0xF), sf, bf) * xv2;
            sumf[row] += fma(float((pw>>12)&0xF), sf, bf) * xv3;
            sumf[row] += fma(float((pw>>16)&0xF), sf, bf) * xv4;
            sumf[row] += fma(float((pw>>20)&0xF), sf, bf) * xv5;
            sumf[row] += fma(float((pw>>24)&0xF), sf, bf) * xv6;
            sumf[row] += fma(float((pw>>28)&0xF), sf, bf) * xv7;
        }
    }

    threadgroup float *shmem_f32[NR0];
    for (short row = 0; row < NR0; row++) {
        shmem_f32[row] = shmem + NW * row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; row++) {
        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; row++) {
        const int d = row0 + row;
        if (d >= (int)out_dim) continue;
        float tot = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) out[d] = (bfloat)tot;
    }
}

kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed [[buffer(0)]],
    device const uint8_t*  scales   [[buffer(1)]],
    device const float*    x        [[buffer(2)]],
    device float*          out      [[buffer(3)]],
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         in_dim   [[buffer(5)]],
    constant uint&         group_size [[buffer(6)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint ROWS_PER_TG = 8;
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols    = in_dim / 8;
    uint num_groups     = in_dim / group_size;
    uint packed_per_grp = group_size / 8;

    // Cache x in threadgroup shared memory (up to 4096 floats = 16KB).
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* wr = W_packed + row * packed_cols;
    device const uint8_t*  sc = scales   + row * num_groups;

    float acc = 0.0f;
    for (uint g = simd_lane; g < num_groups; g += 32) {
        float sf = exp2((float)sc[g] - 127.0f);
        uint bp = g * packed_per_grp;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_grp; p++) {
            uint32_t pw  = wr[bp + p];
            uint x_base  = bx + p * 8;
            acc += NIBBLE_TO_FLOAT[(pw >>  0) & 0xF] * sf * x_shared[x_base + 0];
            acc += NIBBLE_TO_FLOAT[(pw >>  4) & 0xF] * sf * x_shared[x_base + 1];
            acc += NIBBLE_TO_FLOAT[(pw >>  8) & 0xF] * sf * x_shared[x_base + 2];
            acc += NIBBLE_TO_FLOAT[(pw >> 12) & 0xF] * sf * x_shared[x_base + 3];
            acc += NIBBLE_TO_FLOAT[(pw >> 16) & 0xF] * sf * x_shared[x_base + 4];
            acc += NIBBLE_TO_FLOAT[(pw >> 20) & 0xF] * sf * x_shared[x_base + 5];
            acc += NIBBLE_TO_FLOAT[(pw >> 24) & 0xF] * sf * x_shared[x_base + 6];
            acc += NIBBLE_TO_FLOAT[(pw >> 28) & 0xF] * sf * x_shared[x_base + 7];
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = (bfloat)sum;
    }
}

// fused_gate_up_swiglu_bfloat_in: bfloat input, f32 output.
// Used when the hidden state comes from MLX (bf16 → f32 → bfloat truncation).
// Matches MLX's limited SwiGLU with swiglu_limit=10.
kernel void fused_gate_up_swiglu_bfloat_in(
    device const uint32_t* gate_W   [[buffer(0)]],
    device const uint8_t*  gate_s   [[buffer(1)]],
    device const uint32_t* up_W     [[buffer(2)]],
    device const uint8_t*  up_s     [[buffer(3)]],
    device const bfloat*   x        [[buffer(4)]],  // bfloat input
    device float*          out      [[buffer(5)]],  // f32 output
    constant uint&         out_dim  [[buffer(6)]],
    constant uint&         in_dim   [[buffer(7)]],
    constant uint&         group_size [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* g_row = gate_W + tid * packed_cols;
    device const uint8_t*  g_s   = gate_s + tid * num_groups;
    device const uint32_t* u_row = up_W   + tid * packed_cols;
    device const uint8_t*  u_s   = up_s   + tid * num_groups;

    float gate_val = 0.0f, up_val = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float gsf = exp2((float)g_s[g] - 127.0f);
        float usf = exp2((float)u_s[g] - 127.0f);
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gpw = g_row[bp + p], upw = u_row[bp + p];
            for (uint i = 0; i < 8; i++) {
                float g_w = NIBBLE_TO_FLOAT[(gpw >> (i * 4)) & 0xF] * gsf;
                float u_w = NIBBLE_TO_FLOAT[(upw >> (i * 4)) & 0xF] * usf;
                float xv = float(x[bx + p * 8 + i]);
                gate_val += g_w * xv;
                up_val   += u_w * xv;
            }
        }
    }
    const float limit = 10.0f;
    float g_c = min(gate_val, limit);
    float u_c = min(max(up_val, -limit), limit);
    float act = g_c / (1.0f + exp(-g_c));
    out[tid] = act * u_c;
}

// dequant_matvec_4bit_bfloat_in: mxfp4 down_proj with bfloat input, f32 output.
// Used in the bf16 MoE path to match MLX's bf16 expert computation.
kernel void dequant_matvec_4bit_bfloat_in(
    device const uint32_t* W_packed [[buffer(0)]],
    device const uint8_t*  scales   [[buffer(1)]],
    device const bfloat*   x        [[buffer(2)]],  // bfloat input
    device float*          out      [[buffer(3)]],  // f32 output
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         in_dim   [[buffer(5)]],
    constant uint&         group_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const uint8_t*  sc = scales   + tid * num_groups;
    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float sf = exp2((float)sc[g] - 127.0f);
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            for (uint i = 0; i < 8; i++) {
                float w_val = NIBBLE_TO_FLOAT[(pw >> (i * 4)) & 0xF] * sf;
                acc += w_val * float(x[bx + p * 8 + i]);
            }
        }
    }
    out[tid] = acc;
}

// fused_6expert_gate_up_swiglu — processes 6 experts sharing one x_shared load.
// Hardcoded for DSV4-Flash: out_dim=2048, in_dim=4096, group_size=32 (no constant params needed).
// Saves 5 × 4MB redundant x reads vs 6 separate dispatches (x is identical for all experts).
// Buffers 0-5: gate_W[0..5], 6-11: up_W[0..5], 12-17: gate_s[0..5], 18-23: up_s[0..5]
// Buffer 24: x, Buffers 25-30: out[0..5]
// Dispatch: MTLSizeMake(256, 1, 1) × 256 threads (256 = INTERMEDIATE/8)
kernel void fused_6expert_gate_up_swiglu(
    device const uint32_t* gW0 [[buffer(0)]], device const uint32_t* gW1 [[buffer(1)]],
    device const uint32_t* gW2 [[buffer(2)]], device const uint32_t* gW3 [[buffer(3)]],
    device const uint32_t* gW4 [[buffer(4)]], device const uint32_t* gW5 [[buffer(5)]],
    device const uint32_t* uW0 [[buffer(6)]], device const uint32_t* uW1 [[buffer(7)]],
    device const uint32_t* uW2 [[buffer(8)]], device const uint32_t* uW3 [[buffer(9)]],
    device const uint32_t* uW4 [[buffer(10)]], device const uint32_t* uW5 [[buffer(11)]],
    device const uint8_t*  gS0 [[buffer(12)]], device const uint8_t* gS1 [[buffer(13)]],
    device const uint8_t*  gS2 [[buffer(14)]], device const uint8_t* gS3 [[buffer(15)]],
    device const uint8_t*  gS4 [[buffer(16)]], device const uint8_t* gS5 [[buffer(17)]],
    device const uint8_t*  uS0 [[buffer(18)]], device const uint8_t* uS1 [[buffer(19)]],
    device const uint8_t*  uS2 [[buffer(20)]], device const uint8_t* uS3 [[buffer(21)]],
    device const uint8_t*  uS4 [[buffer(22)]], device const uint8_t* uS5 [[buffer(23)]],
    device const float*    x   [[buffer(24)]],
    device float* out0 [[buffer(25)]], device float* out1 [[buffer(26)]],
    device float* out2 [[buffer(27)]], device float* out3 [[buffer(28)]],
    device float* out4 [[buffer(29)]], device float* out5 [[buffer(30)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Hardcoded DSV4-Flash dimensions: gate/up [2048, 4096], group_size=32
    const uint OUT_DIM = 2048, IN_DIM = 4096, GS = 32;
    const uint ROWS_PER_TG = 8;
    const uint packed_cols = IN_DIM / 8;   // 512
    const uint num_groups  = IN_DIM / GS;  // 128
    const uint ppg         = GS / 8;       // 4

    uint row = tgid * ROWS_PER_TG + simd_group;

    // Load x ONCE — shared across all 6 experts in this threadgroup
    threadgroup float x_shared[4096];
    for (uint i = lid; i < IN_DIM; i += 256) x_shared[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= OUT_DIM) return;

    device const uint32_t* const gWs[6] = {gW0, gW1, gW2, gW3, gW4, gW5};
    device const uint32_t* const uWs[6] = {uW0, uW1, uW2, uW3, uW4, uW5};
    device const uint8_t*  const gSs[6] = {gS0, gS1, gS2, gS3, gS4, gS5};
    device const uint8_t*  const uSs[6] = {uS0, uS1, uS2, uS3, uS4, uS5};
    device float* const outs[6] = {out0, out1, out2, out3, out4, out5};

    for (int ei = 0; ei < 6; ei++) {
        device const uint32_t* g_row = gWs[ei] + row * packed_cols;
        device const uint8_t*  g_s   = gSs[ei] + row * num_groups;
        device const uint32_t* u_row = uWs[ei] + row * packed_cols;
        device const uint8_t*  u_s   = uSs[ei] + row * num_groups;

        float ga = 0.0f, ua = 0.0f;
        for (uint g = simd_lane; g < num_groups; g += 32) {
            float gsf = exp2((float)g_s[g] - 127.0f);
            float usf = exp2((float)u_s[g] - 127.0f);
            uint bp = g * ppg, bx = g * GS;
            for (uint p = 0; p < ppg; p++) {
                uint32_t gpw = g_row[bp+p], upw = u_row[bp+p];
                uint xb = bx + p*8;
                ga += NIBBLE_TO_FLOAT[(gpw>> 0)&0xF]*gsf*x_shared[xb+0];
                ga += NIBBLE_TO_FLOAT[(gpw>> 4)&0xF]*gsf*x_shared[xb+1];
                ga += NIBBLE_TO_FLOAT[(gpw>> 8)&0xF]*gsf*x_shared[xb+2];
                ga += NIBBLE_TO_FLOAT[(gpw>>12)&0xF]*gsf*x_shared[xb+3];
                ga += NIBBLE_TO_FLOAT[(gpw>>16)&0xF]*gsf*x_shared[xb+4];
                ga += NIBBLE_TO_FLOAT[(gpw>>20)&0xF]*gsf*x_shared[xb+5];
                ga += NIBBLE_TO_FLOAT[(gpw>>24)&0xF]*gsf*x_shared[xb+6];
                ga += NIBBLE_TO_FLOAT[(gpw>>28)&0xF]*gsf*x_shared[xb+7];
                ua += NIBBLE_TO_FLOAT[(upw>> 0)&0xF]*usf*x_shared[xb+0];
                ua += NIBBLE_TO_FLOAT[(upw>> 4)&0xF]*usf*x_shared[xb+1];
                ua += NIBBLE_TO_FLOAT[(upw>> 8)&0xF]*usf*x_shared[xb+2];
                ua += NIBBLE_TO_FLOAT[(upw>>12)&0xF]*usf*x_shared[xb+3];
                ua += NIBBLE_TO_FLOAT[(upw>>16)&0xF]*usf*x_shared[xb+4];
                ua += NIBBLE_TO_FLOAT[(upw>>20)&0xF]*usf*x_shared[xb+5];
                ua += NIBBLE_TO_FLOAT[(upw>>24)&0xF]*usf*x_shared[xb+6];
                ua += NIBBLE_TO_FLOAT[(upw>>28)&0xF]*usf*x_shared[xb+7];
            }
        }
        float gv = simd_sum(ga), uv = simd_sum(ua);
        if (simd_lane == 0) {
            const float lim = 10.0f;
            float gc = min(gv, lim), uc = min(max(uv, -lim), lim);
            outs[ei][row] = (gc / (1.0f + exp(-gc))) * uc;
        }
    }
}

// fused_6expert_down — 6 expert down_proj in one dispatch (hardcoded DSV4-Flash dims).
// Each expert has its own intermediate input. Dispatch: MTLSizeMake(512*6, 1, 1) × 256 threads.
// Buffers 0-5: W[0..5], 6-11: s[0..5], 12-17: x_mid[0..5], 18-23: out[0..5]
kernel void fused_6expert_down(
    device const uint32_t* W0 [[buffer(0)]], device const uint32_t* W1 [[buffer(1)]],
    device const uint32_t* W2 [[buffer(2)]], device const uint32_t* W3 [[buffer(3)]],
    device const uint32_t* W4 [[buffer(4)]], device const uint32_t* W5 [[buffer(5)]],
    device const uint8_t*  S0 [[buffer(6)]], device const uint8_t*  S1 [[buffer(7)]],
    device const uint8_t*  S2 [[buffer(8)]], device const uint8_t*  S3 [[buffer(9)]],
    device const uint8_t*  S4 [[buffer(10)]], device const uint8_t* S5 [[buffer(11)]],
    device const float* x0 [[buffer(12)]], device const float* x1 [[buffer(13)]],
    device const float* x2 [[buffer(14)]], device const float* x3 [[buffer(15)]],
    device const float* x4 [[buffer(16)]], device const float* x5 [[buffer(17)]],
    device float* out0 [[buffer(18)]], device float* out1 [[buffer(19)]],
    device float* out2 [[buffer(20)]], device float* out3 [[buffer(21)]],
    device float* out4 [[buffer(22)]], device float* out5 [[buffer(23)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Hardcoded DSV4-Flash: down [4096, 2048], group_size=32
    const uint OUT_DIM = 4096, IN_DIM = 2048, GS = 32;
    const uint ROWS_PER_TG  = 8;
    const uint TG_PER_EXPERT = OUT_DIM / ROWS_PER_TG;  // 512

    uint ei  = tgid / TG_PER_EXPERT;
    uint rtg = tgid % TG_PER_EXPERT;
    uint row = rtg * ROWS_PER_TG + simd_group;

    if (ei >= 6 || row >= OUT_DIM) return;

    device const uint32_t* const Ws[6] = {W0, W1, W2, W3, W4, W5};
    device const uint8_t*  const Ss[6] = {S0, S1, S2, S3, S4, S5};
    device const float*    const xs[6] = {x0, x1, x2, x3, x4, x5};
    device float* const outs[6] = {out0, out1, out2, out3, out4, out5};

    threadgroup float x_shared[2048];
    for (uint i = lid; i < IN_DIM; i += 256) x_shared[i] = xs[ei][i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint packed_cols = IN_DIM / 8;   // 256
    const uint num_groups  = IN_DIM / GS;  // 64
    const uint ppg         = GS / 8;       // 4

    device const uint32_t* w_row = Ws[ei] + row * packed_cols;
    device const uint8_t*  s_row = Ss[ei] + row * num_groups;

    float acc = 0.0f;
    for (uint g = simd_lane; g < num_groups; g += 32) {
        float sf = exp2((float)s_row[g] - 127.0f);
        uint bp = g * ppg, bx = g * GS;
        for (uint p = 0; p < ppg; p++) {
            uint32_t pw = w_row[bp+p];
            uint xb = bx + p*8;
            acc += NIBBLE_TO_FLOAT[(pw>> 0)&0xF]*sf*x_shared[xb+0];
            acc += NIBBLE_TO_FLOAT[(pw>> 4)&0xF]*sf*x_shared[xb+1];
            acc += NIBBLE_TO_FLOAT[(pw>> 8)&0xF]*sf*x_shared[xb+2];
            acc += NIBBLE_TO_FLOAT[(pw>>12)&0xF]*sf*x_shared[xb+3];
            acc += NIBBLE_TO_FLOAT[(pw>>16)&0xF]*sf*x_shared[xb+4];
            acc += NIBBLE_TO_FLOAT[(pw>>20)&0xF]*sf*x_shared[xb+5];
            acc += NIBBLE_TO_FLOAT[(pw>>24)&0xF]*sf*x_shared[xb+6];
            acc += NIBBLE_TO_FLOAT[(pw>>28)&0xF]*sf*x_shared[xb+7];
        }
    }
    if (simd_lane == 0) outs[ei][row] = simd_sum(acc);
}


// ============================================================================
// Gather MoE kernels — gatherQmm equivalent for MXFP4 expert weights
// ============================================================================
//
// Instead of 6 separate per-expert dispatches (each reading a full 13.4MB expert blob),
// these kernels take the ENTIRE layer's weights as one buffer and use expert_ids
// to gather only the K=6 selected experts' rows.
//
// Data layout (same as packed_experts/layer_XX.bin components):
//   gate_W_all: [N_EXPERTS=256, INTERMEDIATE=2048, PACKED_COLS=512] uint32  (1 GB)
//   gate_s_all: [N_EXPERTS=256, INTERMEDIATE=2048, N_GROUPS=128] uint8      (64 MB)
//   (same for up_W_all, up_s_all, down_W_all, down_s_all)
//
// The key insight: GPU only reads the K=6 selected experts' rows, not all 256.
// With K=6: reads 6/256 ≈ 2.3% of each weight matrix per forward pass.
// Per layer: gate_W + up_W = 2 × 6/256 × 1GB = ~47MB (vs 80MB for separate kernel)
// But more importantly: down_W = 6/256 × 2GB = ~47MB.
// Total: ~94MB vs 228MB current (sparse access pattern benefits GPU cache).

// gather_gate_up_swiglu: fused gate+up+SwiGLU for K experts selected from N_EXPERTS.
//
// Uses packed_experts layout: pool = [e0: gate_W(4MB) gate_s(256KB) up_W(4MB) up_s(256KB) ...]
//                                     [e1: ...] ... [e255: ...]
// EXPERT_SIZE = 13369344 bytes, offsets: GATE_W=0, GATE_S=4194304, UP_W=4456448, UP_S=8650752
//
// Dispatch: MTLSizeMake(INTERMEDIATE/8, K, 1) threadgroups × 256 threads.
// K experts run in parallel via Y dimension, all sharing the same x input.
kernel void gather_gate_up_swiglu(
    device const uint32_t* pool     [[buffer(0)]],   // entire layer pool (NoCopy from SMELT RAM)
    device const float*    x        [[buffer(1)]],   // [IN_DIM=4096] shared input
    device float*          out      [[buffer(2)]],   // [K × INTERMEDIATE] output (after SwiGLU)
    constant uint*         eids     [[buffer(3)]],   // [K] selected expert indices
    constant uint&         K        [[buffer(4)]],
    uint3 tgid       [[threadgroup_position_in_grid]],
    uint3 lid3       [[thread_position_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    const uint INTERMEDIATE   = 2048;
    const uint IN_DIM         = 4096;
    const uint GS             = 32;
    const uint PACKED_COLS    = IN_DIM / 8;       // 512 uint32 per row
    const uint N_GROUPS       = IN_DIM / GS;      // 128 scale groups per row
    const uint PPG            = GS / 8;           // 4 packed words per group
    const uint ROWS_PER_TG    = 8;
    // Byte offsets within one expert blob (matching packed_experts constants)
    const uint EXPERT_SIZE_U32 = 13369344u / 4u;  // 3342336 uint32 elements
    const uint GATE_S_BYTE    = 4194304u;          // byte offset of gate_s
    const uint UP_W_U32       = 4456448u / 4u;    // 1114112 uint32 elements
    const uint UP_S_BYTE      = 8650752u;          // byte offset of up_s

    uint k_idx = tgid.y;
    if (k_idx >= K) return;
    uint eid = eids[k_idx];
    uint row = tgid.x * ROWS_PER_TG + simd_group;
    uint lid = lid3.x;  // linearized thread ID within threadgroup

    // gate_W base for this expert: pool + eid * EXPERT_SIZE_U32 + 0 + row * PACKED_COLS
    device const uint32_t* g_row = pool + eid * EXPERT_SIZE_U32 + row * PACKED_COLS;
    device const uint32_t* u_row = pool + eid * EXPERT_SIZE_U32 + UP_W_U32 + row * PACKED_COLS;
    // Scales via byte arithmetic (avoids uint8* cast which may have alignment issues in Metal)
    const uint gs_byte_base = (uint)(eid * 13369344u) + GATE_S_BYTE + row * N_GROUPS;
    const uint us_byte_base = (uint)(eid * 13369344u) + UP_S_BYTE   + row * N_GROUPS;

    // Cache x in shared memory — reused for all K experts via Y-dim parallelism
    // (each (tgid.x, tgid.y) threadgroup loads x independently, but same x data)
    threadgroup float x_shared[4096];
    for (uint i = lid; i < IN_DIM; i += 256) x_shared[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= INTERMEDIATE) return;

    float ga = 0.0f, ua = 0.0f;
    for (uint g = simd_lane; g < N_GROUPS; g += 32) {
        // Read scale bytes: use shift/mask instead of division/modulo for ALU efficiency.
        uint gs_bidx = gs_byte_base + g;
        uint us_bidx = us_byte_base + g;
        float gsf = exp2((float)(uint8_t)((pool[gs_bidx >> 2] >> ((gs_bidx & 3) << 3)) & 0xFF) - 127.0f);
        float usf = exp2((float)(uint8_t)((pool[us_bidx >> 2] >> ((us_bidx & 3) << 3)) & 0xFF) - 127.0f);
        uint bp = g * PPG, bx = g * GS;
        for (uint p = 0; p < PPG; p++) {
            uint32_t gpw = g_row[bp+p], upw = u_row[bp+p];
            uint xb = bx + p*8;
            ga += NIBBLE_TO_FLOAT[(gpw>> 0)&0xF]*gsf*x_shared[xb+0];
            ga += NIBBLE_TO_FLOAT[(gpw>> 4)&0xF]*gsf*x_shared[xb+1];
            ga += NIBBLE_TO_FLOAT[(gpw>> 8)&0xF]*gsf*x_shared[xb+2];
            ga += NIBBLE_TO_FLOAT[(gpw>>12)&0xF]*gsf*x_shared[xb+3];
            ga += NIBBLE_TO_FLOAT[(gpw>>16)&0xF]*gsf*x_shared[xb+4];
            ga += NIBBLE_TO_FLOAT[(gpw>>20)&0xF]*gsf*x_shared[xb+5];
            ga += NIBBLE_TO_FLOAT[(gpw>>24)&0xF]*gsf*x_shared[xb+6];
            ga += NIBBLE_TO_FLOAT[(gpw>>28)&0xF]*gsf*x_shared[xb+7];
            ua += NIBBLE_TO_FLOAT[(upw>> 0)&0xF]*usf*x_shared[xb+0];
            ua += NIBBLE_TO_FLOAT[(upw>> 4)&0xF]*usf*x_shared[xb+1];
            ua += NIBBLE_TO_FLOAT[(upw>> 8)&0xF]*usf*x_shared[xb+2];
            ua += NIBBLE_TO_FLOAT[(upw>>12)&0xF]*usf*x_shared[xb+3];
            ua += NIBBLE_TO_FLOAT[(upw>>16)&0xF]*usf*x_shared[xb+4];
            ua += NIBBLE_TO_FLOAT[(upw>>20)&0xF]*usf*x_shared[xb+5];
            ua += NIBBLE_TO_FLOAT[(upw>>24)&0xF]*usf*x_shared[xb+6];
            ua += NIBBLE_TO_FLOAT[(upw>>28)&0xF]*usf*x_shared[xb+7];
        }
    }
    float gv = simd_sum(ga), uv = simd_sum(ua);
    if (simd_lane == 0) {
        const float lim = 10.0f;
        float gc = min(gv, lim), uc = min(max(uv, -lim), lim);
        out[k_idx * INTERMEDIATE + row] = (bfloat)((gc / (1.0f + exp(-gc))) * uc);
    }
}

// gather_down: down projection for K experts from packed_experts pool.
// DOWN_W offset = 8912896 bytes, DOWN_S = 13107200 bytes within each expert blob.
// Dispatch: MTLSizeMake(DIM/8, K, 1) threadgroups × 256 threads.
kernel void gather_down(
    device const uint32_t* pool     [[buffer(0)]],   // entire layer pool
    device const float*    x_mid    [[buffer(1)]],   // [K × INTERMEDIATE] gate*up results
    device float*          out      [[buffer(2)]],   // [K × DIM]
    constant uint*         eids     [[buffer(3)]],
    constant uint&         K        [[buffer(4)]],
    uint3 tgid       [[threadgroup_position_in_grid]],
    uint3 lid3       [[thread_position_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    const uint DIM            = 4096;
    const uint INTERMEDIATE   = 2048;
    const uint GS             = 32;
    const uint PACKED_COLS    = INTERMEDIATE / 8;   // 256 uint32 per row
    const uint N_GROUPS       = INTERMEDIATE / GS;  // 64
    const uint PPG            = GS / 8;             // 4
    const uint ROWS_PER_TG    = 8;
    const uint EXPERT_SIZE_U32 = 13369344u / 4u;
    const uint DOWN_W_U32     = 8912896u / 4u;      // 2228224
    const uint DOWN_S_BYTE    = 13107200u;

    uint k_idx = tgid.y;
    if (k_idx >= K) return;
    uint eid = eids[k_idx];
    uint row = tgid.x * ROWS_PER_TG + simd_group;
    uint lid = lid3.x;
    device const uint32_t* w_row = pool + eid * EXPERT_SIZE_U32 + DOWN_W_U32 + row * PACKED_COLS;
    // Use uint8 view of pool for scales (byte-level access)
    // Note: casting to uint8* — use separate buffer approach to avoid Metal restrictions
    // Instead, read scale byte from pool using byte arithmetic on uint32 words
    // DOWN_S at byte: eid*13369344 + 13107200 + row*N_GROUPS + g
    // In uint32 words (poolword index) = (eid*13369344 + 13107200 + row*N_GROUPS + g) / 4
    // byte within word = same % 4
    // Precompute base byte offset for this row's scales
    const uint s_byte_base = (uint)(eid * 13369344u) + DOWN_S_BYTE + row * N_GROUPS;

    // Load this expert's intermediate output (gate×up result) into shared memory
    threadgroup float x_shared[2048];
    device const float* my_xmid = x_mid + k_idx * INTERMEDIATE;
    for (uint i = lid; i < INTERMEDIATE; i += 256) x_shared[i] = my_xmid[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= DIM) return;

    float acc = 0.0f;
    for (uint g = simd_lane; g < N_GROUPS; g += 32) {
        // Read scale byte: shift/mask instead of division/modulo.
        uint s_byte_idx = s_byte_base + g;
        uint s_word = pool[s_byte_idx >> 2];
        uint8_t s_byte = (uint8_t)((s_word >> ((s_byte_idx & 3) << 3)) & 0xFF);
        float sf = exp2((float)s_byte - 127.0f);
        uint bp = g * PPG, bx = g * GS;
        for (uint p = 0; p < PPG; p++) {
            uint32_t pw = w_row[bp+p];
            uint xb = bx + p*8;
            acc += NIBBLE_TO_FLOAT[(pw>> 0)&0xF]*sf*x_shared[xb+0];
            acc += NIBBLE_TO_FLOAT[(pw>> 4)&0xF]*sf*x_shared[xb+1];
            acc += NIBBLE_TO_FLOAT[(pw>> 8)&0xF]*sf*x_shared[xb+2];
            acc += NIBBLE_TO_FLOAT[(pw>>12)&0xF]*sf*x_shared[xb+3];
            acc += NIBBLE_TO_FLOAT[(pw>>16)&0xF]*sf*x_shared[xb+4];
            acc += NIBBLE_TO_FLOAT[(pw>>20)&0xF]*sf*x_shared[xb+5];
            acc += NIBBLE_TO_FLOAT[(pw>>24)&0xF]*sf*x_shared[xb+6];
            acc += NIBBLE_TO_FLOAT[(pw>>28)&0xF]*sf*x_shared[xb+7];
        }
    }
    float gd_sum = simd_sum(acc);  // collective: must be executed by ALL lanes
    if (simd_lane == 0) out[k_idx * DIM + row] = (bfloat)gd_sum;
}

// ============================================================================
// Gather6 MoE kernels — pointer-array variant for non-contiguous expert blobs
// ============================================================================
//
// Unlike gather_gate_up_swiglu/gather_down (which index into one contiguous
// SMELT pool by eid), these take 6 independent expert blob buffers (indices
// 0..5; MSL 3.1 rejects arrays-of-buffers as kernel params, so they are six
// separate parameters selected by a uniform switch on tgid.y). Used by the
// decode path where experts come from a mix of SMELT cache / prefetch / pread
// buffers with no contiguous layout.
// Replaces 12 per-expert dispatches + 6 blits with 2 dispatches total.
//
// Dispatch: MTLSizeMake(INTERMEDIATE/8, K=6, 1) threadgroups × 256 threads.
kernel void gather6_gate_up_swiglu(
    device const uint32_t* blob0    [[buffer(0)]],   // packed expert blob, slot 0
    device const uint32_t* blob1    [[buffer(1)]],   // slot 1
    device const uint32_t* blob2    [[buffer(2)]],   // slot 2
    device const uint32_t* blob3    [[buffer(3)]],   // slot 3
    device const uint32_t* blob4    [[buffer(4)]],   // slot 4
    device const uint32_t* blob5    [[buffer(5)]],   // slot 5
    device const float*    x        [[buffer(6)]],   // [IN_DIM=4096] shared input
    device float*          out      [[buffer(7)]],   // [6 × INTERMEDIATE] output (after SwiGLU)
    uint3 tgid       [[threadgroup_position_in_grid]],
    uint3 lid3       [[thread_position_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    const uint INTERMEDIATE   = 2048;
    const uint IN_DIM         = 4096;
    const uint GS             = 32;
    const uint PACKED_COLS    = IN_DIM / 8;       // 512 uint32 per row
    const uint N_GROUPS       = IN_DIM / GS;      // 128 scale groups per row
    const uint PPG            = GS / 8;           // 4 packed words per group
    const uint ROWS_PER_TG    = 8;
    // Byte offsets within one expert blob (matching packed_experts constants)
    const uint GATE_S_BYTE    = 4194304u;          // byte offset of gate_s
    const uint UP_W_U32       = 4456448u / 4u;    // 1114112 uint32 elements
    const uint UP_S_BYTE      = 8650752u;          // byte offset of up_s

    uint k_idx = tgid.y;
    uint row = tgid.x * ROWS_PER_TG + simd_group;
    uint lid = lid3.x;  // linearized thread ID within threadgroup

    // Uniform select of this slot's blob (tgid.y is constant per threadgroup)
    device const uint32_t* blob = blob0;
    switch (k_idx) {
        case 1: blob = blob1; break;
        case 2: blob = blob2; break;
        case 3: blob = blob3; break;
        case 4: blob = blob4; break;
        case 5: blob = blob5; break;
        default: break;
    }
    device const uint32_t* g_row = blob + row * PACKED_COLS;
    device const uint32_t* u_row = blob + UP_W_U32 + row * PACKED_COLS;
    // Scales via byte arithmetic relative to this slot's blob base
    const uint gs_byte_base = GATE_S_BYTE + row * N_GROUPS;
    const uint us_byte_base = UP_S_BYTE   + row * N_GROUPS;

    // Cache x in shared memory — reused for all K experts via Y-dim parallelism
    threadgroup float x_shared[4096];
    for (uint i = lid; i < IN_DIM; i += 256) x_shared[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= INTERMEDIATE) return;

    float ga = 0.0f, ua = 0.0f;
    for (uint g = simd_lane; g < N_GROUPS; g += 32) {
        uint gs_bidx = gs_byte_base + g;
        uint us_bidx = us_byte_base + g;
        float gsf = exp2((float)(uint8_t)((blob[gs_bidx >> 2] >> ((gs_bidx & 3) << 3)) & 0xFF) - 127.0f);
        float usf = exp2((float)(uint8_t)((blob[us_bidx >> 2] >> ((us_bidx & 3) << 3)) & 0xFF) - 127.0f);
        uint bp = g * PPG, bx = g * GS;
        for (uint p = 0; p < PPG; p++) {
            uint32_t gpw = g_row[bp+p], upw = u_row[bp+p];
            uint xb = bx + p*8;
            ga += NIBBLE_TO_FLOAT[(gpw>> 0)&0xF]*gsf*x_shared[xb+0];
            ga += NIBBLE_TO_FLOAT[(gpw>> 4)&0xF]*gsf*x_shared[xb+1];
            ga += NIBBLE_TO_FLOAT[(gpw>> 8)&0xF]*gsf*x_shared[xb+2];
            ga += NIBBLE_TO_FLOAT[(gpw>>12)&0xF]*gsf*x_shared[xb+3];
            ga += NIBBLE_TO_FLOAT[(gpw>>16)&0xF]*gsf*x_shared[xb+4];
            ga += NIBBLE_TO_FLOAT[(gpw>>20)&0xF]*gsf*x_shared[xb+5];
            ga += NIBBLE_TO_FLOAT[(gpw>>24)&0xF]*gsf*x_shared[xb+6];
            ga += NIBBLE_TO_FLOAT[(gpw>>28)&0xF]*gsf*x_shared[xb+7];
            ua += NIBBLE_TO_FLOAT[(upw>> 0)&0xF]*usf*x_shared[xb+0];
            ua += NIBBLE_TO_FLOAT[(upw>> 4)&0xF]*usf*x_shared[xb+1];
            ua += NIBBLE_TO_FLOAT[(upw>> 8)&0xF]*usf*x_shared[xb+2];
            ua += NIBBLE_TO_FLOAT[(upw>>12)&0xF]*usf*x_shared[xb+3];
            ua += NIBBLE_TO_FLOAT[(upw>>16)&0xF]*usf*x_shared[xb+4];
            ua += NIBBLE_TO_FLOAT[(upw>>20)&0xF]*usf*x_shared[xb+5];
            ua += NIBBLE_TO_FLOAT[(upw>>24)&0xF]*usf*x_shared[xb+6];
            ua += NIBBLE_TO_FLOAT[(upw>>28)&0xF]*usf*x_shared[xb+7];
        }
    }
    float gv = simd_sum(ga), uv = simd_sum(ua);
    if (simd_lane == 0) {
        const float lim = 10.0f;
        float gc = min(gv, lim), uc = min(max(uv, -lim), lim);
        out[k_idx * INTERMEDIATE + row] = (bfloat)((gc / (1.0f + exp(-gc))) * uc);
    }
}

// gather6_down: down projection from 6 independent expert blobs.
// DOWN_W offset = 8912896 bytes, DOWN_S = 13107200 bytes within each expert blob.
// Dispatch: MTLSizeMake(DIM/8, K=6, 1) threadgroups × 256 threads.
kernel void gather6_down(
    device const uint32_t* blob0    [[buffer(0)]],   // packed expert blob, slot 0
    device const uint32_t* blob1    [[buffer(1)]],   // slot 1
    device const uint32_t* blob2    [[buffer(2)]],   // slot 2
    device const uint32_t* blob3    [[buffer(3)]],   // slot 3
    device const uint32_t* blob4    [[buffer(4)]],   // slot 4
    device const uint32_t* blob5    [[buffer(5)]],   // slot 5
    device const float*    x_mid    [[buffer(6)]],   // [6 × INTERMEDIATE] gate*up results
    device float*          out      [[buffer(7)]],   // [6 × DIM]
    uint3 tgid       [[threadgroup_position_in_grid]],
    uint3 lid3       [[thread_position_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    const uint DIM            = 4096;
    const uint INTERMEDIATE   = 2048;
    const uint GS             = 32;
    const uint PACKED_COLS    = INTERMEDIATE / 8;   // 256 uint32 per row
    const uint N_GROUPS       = INTERMEDIATE / GS;  // 64
    const uint PPG            = GS / 8;             // 4
    const uint ROWS_PER_TG    = 8;
    const uint DOWN_W_U32     = 8912896u / 4u;      // 2228224
    const uint DOWN_S_BYTE    = 13107200u;

    uint k_idx = tgid.y;
    uint row = tgid.x * ROWS_PER_TG + simd_group;
    uint lid = lid3.x;
    // Uniform select of this slot's blob (tgid.y is constant per threadgroup)
    device const uint32_t* blob = blob0;
    switch (k_idx) {
        case 1: blob = blob1; break;
        case 2: blob = blob2; break;
        case 3: blob = blob3; break;
        case 4: blob = blob4; break;
        case 5: blob = blob5; break;
        default: break;
    }
    device const uint32_t* w_row = blob + DOWN_W_U32 + row * PACKED_COLS;
    // Scale byte base for this row, relative to this slot's blob base
    const uint s_byte_base = DOWN_S_BYTE + row * N_GROUPS;

    // Load this expert's intermediate output (gate×up result) into shared memory
    threadgroup float x_shared[2048];
    device const float* my_xmid = x_mid + k_idx * INTERMEDIATE;
    for (uint i = lid; i < INTERMEDIATE; i += 256) x_shared[i] = my_xmid[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= DIM) return;

    float acc = 0.0f;
    for (uint g = simd_lane; g < N_GROUPS; g += 32) {
        uint s_byte_idx = s_byte_base + g;
        uint s_word = blob[s_byte_idx >> 2];
        uint8_t s_byte = (uint8_t)((s_word >> ((s_byte_idx & 3) << 3)) & 0xFF);
        float sf = exp2((float)s_byte - 127.0f);
        uint bp = g * PPG, bx = g * GS;
        for (uint p = 0; p < PPG; p++) {
            uint32_t pw = w_row[bp+p];
            uint xb = bx + p*8;
            acc += NIBBLE_TO_FLOAT[(pw>> 0)&0xF]*sf*x_shared[xb+0];
            acc += NIBBLE_TO_FLOAT[(pw>> 4)&0xF]*sf*x_shared[xb+1];
            acc += NIBBLE_TO_FLOAT[(pw>> 8)&0xF]*sf*x_shared[xb+2];
            acc += NIBBLE_TO_FLOAT[(pw>>12)&0xF]*sf*x_shared[xb+3];
            acc += NIBBLE_TO_FLOAT[(pw>>16)&0xF]*sf*x_shared[xb+4];
            acc += NIBBLE_TO_FLOAT[(pw>>20)&0xF]*sf*x_shared[xb+5];
            acc += NIBBLE_TO_FLOAT[(pw>>24)&0xF]*sf*x_shared[xb+6];
            acc += NIBBLE_TO_FLOAT[(pw>>28)&0xF]*sf*x_shared[xb+7];
        }
    }
    float g6d_sum = simd_sum(acc);  // collective: must be executed by ALL lanes
    if (simd_lane == 0) out[k_idx * DIM + row] = (bfloat)g6d_sum;
}


kernel void moe_combine(
    device const float* expert_outs [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device const float* residual    [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant uint&      K           [[buffer(4)]],
    constant uint&      hidden_dim  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= hidden_dim) return;
    float sum = residual[tid];
    for (uint k = 0; k < K; k++) {
        sum += expert_outs[k * hidden_dim + tid] * weights[k];
    }
    output[tid] = sum;
}

// dequant_matvec_affine_v2: coalesced affine 4-bit dequant matvec (ds4 Q8_0 pattern).
//
// Design (ds4 kernel_mul_mv_q8_0_f32 adapted for affine 4-bit, group_size=64):
//   - NR0=2 rows per threadgroup, NSG=4 simdgroups, 32 threads each → 128 threads/TG
//   - Shared memory: 32 * 2 * sizeof(float) = 256 bytes
//   - NO x_shared — no occupancy risk
//   - Coalesced access: 8 threads per group, each reads 1 uint32 word
//     Adjacent threads read adjacent uint32 words from W and adjacent floats from x.
//   - FMA optimization: pre-compute scale*x, bias*x per nibble
//
// For wo_b [4096, 8192]: 2048 threadgroups, 128 threads each = 262,144 threads.
//   num_groups=128, packed_per_group=8.
//   Each SIMD group: 32 threads → 4 groups per iteration, 32 iterations.
kernel void dequant_matvec_affine_v2(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const float*    scales     [[buffer(1)]],  // [out_dim, num_groups]
    device const float*    biases     [[buffer(2)]],  // [out_dim, num_groups]
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    threadgroup float*     shmem      [[threadgroup(0)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 2, NSG = 4, NW = 32, NQ = 4, TPG = 8;

    const uint num_groups = in_dim / group_size;
    const uint packed_per_group = group_size / 8;
    const uint packed_cols = in_dim / 8;

    const int row0 = (int)tgpig.x * NR0;
    const short ix = tiisg / TPG, il = tiisg % TPG;
    const int g0 = (int)sgitg * NQ + (int)ix;

    device const uint32_t *wr[NR0];
    device const float    *sr[NR0];
    device const float    *br[NR0];
    for (short row = 0; row < NR0; row++) {
        int r = row0 + row;
        if (r < (int)out_dim) {
            wr[row] = W_packed + r * packed_cols;
            sr[row] = scales   + r * num_groups;
            br[row] = biases   + r * num_groups;
        }
    }

    float sumf[NR0] = { 0.0f };

    for (int gg = g0; gg < (int)num_groups; gg += NSG * NQ) {
        uint xb = (uint)gg * group_size + (uint)il * 8;
        float xv0 = x[xb+0], xv1 = x[xb+1], xv2 = x[xb+2], xv3 = x[xb+3];
        float xv4 = x[xb+4], xv5 = x[xb+5], xv6 = x[xb+6], xv7 = x[xb+7];

        for (short row = 0; row < NR0; row++) {
            int r = row0 + row;
            if (r >= (int)out_dim) continue;
            float scale = sr[row][gg], bias = br[row][gg];
            float sx0 = scale*xv0, bx0 = bias*xv0;
            float sx1 = scale*xv1, bx1 = bias*xv1;
            float sx2 = scale*xv2, bx2 = bias*xv2;
            float sx3 = scale*xv3, bx3 = bias*xv3;
            float sx4 = scale*xv4, bx4 = bias*xv4;
            float sx5 = scale*xv5, bx5 = bias*xv5;
            float sx6 = scale*xv6, bx6 = bias*xv6;
            float sx7 = scale*xv7, bx7 = bias*xv7;
            uint32_t pw = wr[row][gg * packed_per_group + (uint)il];
            sumf[row] += fma(float((pw>> 0)&0xF), sx0, bx0);
            sumf[row] += fma(float((pw>> 4)&0xF), sx1, bx1);
            sumf[row] += fma(float((pw>> 8)&0xF), sx2, bx2);
            sumf[row] += fma(float((pw>>12)&0xF), sx3, bx3);
            sumf[row] += fma(float((pw>>16)&0xF), sx4, bx4);
            sumf[row] += fma(float((pw>>20)&0xF), sx5, bx5);
            sumf[row] += fma(float((pw>>24)&0xF), sx6, bx6);
            sumf[row] += fma(float((pw>>28)&0xF), sx7, bx7);
        }
    }

    threadgroup float *shmem_f32[NR0];
    for (short row = 0; row < NR0; row++) {
        shmem_f32[row] = shmem + NW * row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; row++) {
        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; row++) {
        const int d = row0 + row;
        if (d >= (int)out_dim) continue;
        float tot = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) out[d] = (bfloat)tot;
    }
}

// dequant_matvec_affine: out = W @ x, W affine-4bit quantized (gs=64, bf16 scales+biases).
// Matches MLX affine dequant: w = scale_g * nibble + bias_g  (nibble in [0,15], no LUT).
// W_packed: [out_dim, in_dim/8] uint32 (8 nibbles per uint32).
// scales/biases: [out_dim, in_dim/group_size] — passed as float (caller converts bf16->f32).
kernel void dequant_matvec_affine(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const float*    x        [[buffer(3)]],
    device float*          out      [[buffer(4)]],
    constant uint&         out_dim  [[buffer(5)]],
    constant uint&         in_dim   [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const float*    sc = scales   + tid * num_groups;
    device const float*    bi = biases   + tid * num_groups;

    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = sc[g];
        float bias  = bi[g];
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            uint x_base = bx + p * 8;
            for (uint i = 0; i < 8; i++) {
                float nib = (float)((pw >> (i * 4)) & 0xF);
                acc += (scale * nib + bias) * x[x_base + i];
            }
        }
    }
    out[tid] = acc;
}

// dequant_matvec_affine_bf16out: same as dequant_matvec_affine but output is bfloat.
// This matches MLX's behavior: affine matmul result is stored as bfloat16, which is
// the critical precision alignment needed for exact expert selection match.
kernel void dequant_matvec_affine_bf16out(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const float*    x        [[buffer(3)]],
    device bfloat*         out      [[buffer(4)]],
    constant uint&         out_dim  [[buffer(5)]],
    constant uint&         in_dim   [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const float*    sc = scales   + tid * num_groups;
    device const float*    bi = biases   + tid * num_groups;

    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = sc[g];
        float bias  = bi[g];
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            uint x_base = bx + p * 8;
            for (uint i = 0; i < 8; i++) {
                float nib = (float)((pw >> (i * 4)) & 0xF);
                acc += (scale * nib + bias) * x[x_base + i];
            }
        }
    }
    out[tid] = (bfloat)acc;
}

// rms_norm_rows_bf16out: rms_norm_rows but output is bfloat16 (matching MLX's bf16 intermediate).
// Used for Q chain (wq_a output, q_norm output, wq_b output) to match MLX bf16 attention.
kernel void rms_norm_rows_bf16out(
    device const float*  x          [[buffer(0)]],
    device const float*  weight     [[buffer(1)]],
    device bfloat*       out        [[buffer(2)]],
    constant uint&       row_dim    [[buffer(3)]],
    constant float&      eps        [[buffer(4)]],
    constant uint&       has_weight [[buffer(5)]],
    uint row     [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    float ss = 0.0f;
    for (uint i = lid; i < row_dim; i += tg_size) {
        float v = x[row * row_dim + i]; ss += v * v;
    }
    shared_sum[lid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared_sum[lid] += shared_sum[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_inv = rsqrt(shared_sum[0] / float(row_dim) + eps);
    for (uint i = lid; i < row_dim; i += tg_size) {
        float v = x[row * row_dim + i] * rms_inv;
        out[row * row_dim + i] = (bfloat)(has_weight ? v * weight[i] : v);
    }
}

// bf16_to_f32: convert a bfloat buffer to float (element-wise copy with widening).
kernel void bf16_to_f32(
    device const bfloat* src [[buffer(0)]],
    device float*        dst [[buffer(1)]],
    constant uint&       n   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    dst[tid] = float(src[tid]);
}

// f32_to_bf16: convert float buffer to bfloat (truncation).
kernel void f32_to_bf16(
    device const float*  src [[buffer(0)]],
    device bfloat*       dst [[buffer(1)]],
    constant uint&       n   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    dst[tid] = (bfloat)src[tid];
}

// dequant_matvec_affine_bf16in_f32out: affine 4bit matmul with bfloat input, float output.
// Needed for the first step of the bf16 Q chain where attn_input is bfloat.
kernel void dequant_matvec_affine_bf16in_f32out(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const bfloat*   x        [[buffer(3)]],  // bfloat input
    device float*          out      [[buffer(4)]],  // float output
    constant uint&         out_dim  [[buffer(5)]],
    constant uint&         in_dim   [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const float*    sc = scales   + tid * num_groups;
    device const float*    bi = biases   + tid * num_groups;
    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = sc[g], bias = bi[g];
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            for (uint i = 0; i < 8; i++) {
                float nib = (float)((pw >> (i * 4)) & 0xF);
                acc += (scale * nib + bias) * float(x[bx + p * 8 + i]);
            }
        }
    }
    out[tid] = acc;
}

// dequant_matvec_affine_bf16in_f32out_grp8: grouped variant for wo_a.
// The full weight is [8*out_g, in_dim]; row tid belongs to group
// g = tid / out_g and reads x from x[g*in_dim ..]. Writes straight into the
// concat layout (out[tid]) — replaces 8 per-group dispatches + 8 input blits
// + concat blit with ONE dispatch. Per-row accumulation order is identical
// to dequant_matvec_affine_bf16in_f32out (bit-exact).
kernel void dequant_matvec_affine_bf16in_f32out_grp8(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const bfloat*   x        [[buffer(3)]],  // [8, in_dim] grouped input
    device float*          out      [[buffer(4)]],  // [8*out_g] concat output
    constant uint&         out_g    [[buffer(5)]],  // rows per group (O_LORA_RANK)
    constant uint&         in_dim   [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const float*    sc = scales   + tid * num_groups;
    device const float*    bi = biases   + tid * num_groups;
    device const bfloat*   xg = x + (tid / out_g) * in_dim;
    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = sc[g], bias = bi[g];
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            for (uint i = 0; i < 8; i++) {
                float nib = (float)((pw >> (i * 4)) & 0xF);
                acc += (scale * nib + bias) * float(xg[bx + p * 8 + i]);
            }
        }
    }
    out[tid] = acc;
}

// dequant_matvec_affine_bf16in_bf16out: affine 4bit matmul with bfloat input AND output.
// Used for wq_b, wkv, wo_b in the full bf16 attention chain.
kernel void dequant_matvec_affine_bf16in_bf16out(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const bfloat*   x        [[buffer(3)]],  // bfloat input
    device bfloat*         out      [[buffer(4)]],  // bfloat output
    constant uint&         out_dim  [[buffer(5)]],
    constant uint&         in_dim   [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const float*    sc = scales   + tid * num_groups;
    device const float*    bi = biases   + tid * num_groups;
    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = sc[g], bias = bi[g];
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            for (uint i = 0; i < 8; i++) {
                float nib = (float)((pw >> (i * 4)) & 0xF);
                acc += (scale * nib + bias) * float(x[bx + p * 8 + i]);
            }
        }
    }
    out[tid] = (bfloat)acc;
}

// dequant_matvec_affine_bf16in_bf16out_v2: SIMD-parallel affine 4-bit matmul, bfloat in/out.
// Optimized for MLA attention weights (wq_a, wq_b, wkv, wo_b):
//   - wq_b [32768, 1024] gs=64: 16MB weight — biggest attention bottleneck
//   - Same coalesced pattern as dequant_matvec_affine_v2, adapted for bf16 I/O
// Strategy: NR0=2 rows/TG, NSG=4 SIMD groups, 32 threads each (128 threads/TG).
//   8 threads per group: 4 threads for TPG_x sub-groups × 8 packed_per_group reads.
//   Adjacent threads read adjacent W words (coalesced) + adjacent x elements.
//   simd_sum reduction via threadgroup memory (256B).
// Dispatch: MTLSizeMake((out_dim+1)/2, 1, 1) TGs × MTLSizeMake(32,4,1) threads.
kernel void dequant_matvec_affine_bf16in_bf16out_v2(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const float*    scales     [[buffer(1)]],
    device const float*    biases     [[buffer(2)]],
    device const bfloat*   x          [[buffer(3)]],  // bfloat input
    device bfloat*         out        [[buffer(4)]],  // bfloat output
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    threadgroup float*     shmem      [[threadgroup(0)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 2, NSG = 4, NW = 32, NQ = 4, TPG = 8;

    const uint num_groups     = in_dim / group_size;
    const uint packed_per_group = group_size / 8;
    const uint packed_cols    = in_dim / 8;

    const int row0  = (int)tgpig.x * NR0;
    const short ix  = tiisg / TPG;
    const short il  = tiisg % TPG;
    const int g0    = (int)sgitg * NQ + (int)ix;

    device const uint32_t *wr[NR0];
    device const float    *sr[NR0];
    device const float    *br[NR0];
    for (short row = 0; row < NR0; row++) {
        int r = row0 + row;
        if (r < (int)out_dim) {
            wr[row] = W_packed + r * packed_cols;
            sr[row] = scales   + r * num_groups;
            br[row] = biases   + r * num_groups;
        }
    }

    float sumf[NR0] = { 0.0f };

    for (int gg = g0; gg < (int)num_groups; gg += NSG * NQ) {
        uint xb = (uint)gg * group_size + (uint)il * 8;
        // Load 8 bfloat x values and convert to f32
        float xv0 = float(x[xb+0]), xv1 = float(x[xb+1]);
        float xv2 = float(x[xb+2]), xv3 = float(x[xb+3]);
        float xv4 = float(x[xb+4]), xv5 = float(x[xb+5]);
        float xv6 = float(x[xb+6]), xv7 = float(x[xb+7]);

        for (short row = 0; row < NR0; row++) {
            int r = row0 + row;
            if (r >= (int)out_dim) continue;
            float scale = sr[row][gg], bias = br[row][gg];
            float sx0 = scale*xv0, bx0 = bias*xv0;
            float sx1 = scale*xv1, bx1 = bias*xv1;
            float sx2 = scale*xv2, bx2 = bias*xv2;
            float sx3 = scale*xv3, bx3 = bias*xv3;
            float sx4 = scale*xv4, bx4 = bias*xv4;
            float sx5 = scale*xv5, bx5 = bias*xv5;
            float sx6 = scale*xv6, bx6 = bias*xv6;
            float sx7 = scale*xv7, bx7 = bias*xv7;
            uint32_t pw = wr[row][gg * packed_per_group + (uint)il];
            sumf[row] += fma(float((pw>> 0)&0xF), sx0, bx0);
            sumf[row] += fma(float((pw>> 4)&0xF), sx1, bx1);
            sumf[row] += fma(float((pw>> 8)&0xF), sx2, bx2);
            sumf[row] += fma(float((pw>>12)&0xF), sx3, bx3);
            sumf[row] += fma(float((pw>>16)&0xF), sx4, bx4);
            sumf[row] += fma(float((pw>>20)&0xF), sx5, bx5);
            sumf[row] += fma(float((pw>>24)&0xF), sx6, bx6);
            sumf[row] += fma(float((pw>>28)&0xF), sx7, bx7);
        }
    }

    // Reduce across SIMD groups via threadgroup memory
    threadgroup float *sf[NR0];
    for (short row = 0; row < NR0; row++) {
        sf[row] = shmem + NW * row;
        if (sgitg == 0) sf[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; row++) {
        if (tiisg == 0) sf[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; row++) {
        const int d = row0 + row;
        if (d >= (int)out_dim) continue;
        float tot = simd_sum(sf[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) out[d] = (bfloat)tot;
    }
}

// rms_norm_rows_bf16in_bf16out: RMSNorm with bfloat input AND bfloat output.
// Used for q_norm and per-head norm in the full bf16 attention chain.
kernel void rms_norm_rows_bf16in_bf16out(
    device const bfloat* x          [[buffer(0)]],
    device const float*  weight     [[buffer(1)]],
    device bfloat*       out        [[buffer(2)]],
    constant uint&       row_dim    [[buffer(3)]],
    constant float&      eps        [[buffer(4)]],
    constant uint&       has_weight [[buffer(5)]],
    uint row     [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    float ss = 0.0f;
    for (uint i = lid; i < row_dim; i += tg_size) {
        float v = float(x[row * row_dim + i]); ss += v * v;
    }
    shared_sum[lid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared_sum[lid] += shared_sum[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_inv = rsqrt(shared_sum[0] / float(row_dim) + eps);
    for (uint i = lid; i < row_dim; i += tg_size) {
        float v = float(x[row * row_dim + i]) * rms_inv;
        out[row * row_dim + i] = (bfloat)(has_weight ? v * weight[i] : v);
    }
}

// rope_tail_interleaved_bf16: RoPE with bfloat input AND bfloat output.
// Used in the full bf16 attention chain for Q (and optionally K) RoPE.
kernel void rope_tail_interleaved_bf16(
    device bfloat*         q          [[buffer(0)]],  // in-place [n_heads, head_dim]
    device const float*    cos_vals   [[buffer(1)]],  // [half_rope]
    device const float*    sin_vals   [[buffer(2)]],  // [half_rope]
    constant uint&         n_heads    [[buffer(3)]],
    constant uint&         head_dim   [[buffer(4)]],
    constant uint&         n_nope     [[buffer(5)]],
    constant uint&         n_rope     [[buffer(6)]],
    constant uint&         inverse    [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint half_rope = n_rope / 2;
    if (tid >= n_heads * half_rope) return;
    uint head = tid / half_rope, ic = tid % half_rope;
    uint j0 = head * head_dim + n_nope + 2 * ic;
    uint j1 = j0 + 1;
    float cos_v = cos_vals[ic];
    float sin_v = inverse ? -sin_vals[ic] : sin_vals[ic];
    float x0 = float(q[j0]);
    float x1 = float(q[j1]);
    q[j0] = (bfloat)(x0 * cos_v - x1 * sin_v);
    q[j1] = (bfloat)(x0 * sin_v + x1 * cos_v);
}

// matvec_q8_0_f32: ds4 kernel_mul_mv_q8_0_f32 adapted for dmlx wo_a.
//
// Q8_0 format: each block has a float scale (d) and 32 int8 values.
// Weight = d * qs[i] for element i in the block.
//
// Design (from ds4 dense.metal:108-176):
//   - NR0=2 rows per threadgroup, NSG=4 simdgroups, 32 threads each → 128 threads/TG
//   - Shared memory: 32 * 2 * sizeof(float) = 256 bytes (very small)
//   - Coalesced access: thread (ix, il) loads NQ=8 elements from each block
//   - Reduction: simd_sum per row → threadgroup scatter → final simd_sum
//
// For wo_a [1024, 4096]: 1024/2 = 512 threadgroups, 128 threads each = 65,536 threads.
// Per thread: 4096 / (4*8) = 128 blocks / (NSG*NQ) = 4 outer loop iterations.
kernel void matvec_q8_0_f32(
    device const char*    W        [[buffer(0)]],  // Q8_0 blocks: [out_dim, in_dim/32]
    device const bfloat*  x        [[buffer(1)]],  // [in_dim]
    device float*         out      [[buffer(2)]],  // [out_dim]
    constant uint&        out_dim  [[buffer(3)]],
    constant uint&        in_dim   [[buffer(4)]],
    threadgroup float*    shmem    [[threadgroup(0)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 2;       // rows per threadgroup
    const short NSG = 4;       // simdgroups per threadgroup
    const short NW  = 32;      // SIMD width
    const short NQ  = 8;       // elements per thread per block
    const short QK  = 32;      // block size
    const short NB  = 36;      // bytes per Q8_0 block (4B scale + 32B int8)

    const int nb = (int)in_dim / QK;  // blocks per row (= 128 for in_dim=4096)
    const int row0 = (int)tgpig.x * NR0;

    // Thread indexing within SIMD group
    const short ix = tiisg / (NW / NQ);   // block offset within simdgroup (0..3)
    const short il = tiisg % (NW / NQ);   // lane within block (0..3)
    const int ib0 = (int)sgitg * NQ + (int)ix;

    // Input vector access pattern — bfloat→float conversion on load
    const int offset_y = ib0 * QK + (int)il * NQ;
    device const bfloat *yb = x + offset_y;

    // Weight pointers for 2 rows
    device const char *ax[NR0];
    for (short row = 0; row < NR0; row++) {
        if (row0 + row < (int)out_dim) {
            ax[row] = W + (row0 + row) * nb * NB;
        }
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    // Main loop: iterate over blocks, each thread processes NQ elements per block
    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        // Load NQ input values with bf16→f32 conversion
        for (short i = 0; i < NQ; i++) {
            yl[i] = float(yb[i]);
        }

        for (short row = 0; row < NR0; row++) {
            if (row0 + row >= (int)out_dim) continue;
            // Block layout: [4B scale][32B int8]
            device const char *blk = ax[row] + ib * NB;
            float d = *(device const float *)blk;
            device const int8_t *qs = (device const int8_t *)(blk + 4) + (int)il * NQ;

            float sumq = 0.0f;
            for (short i = 0; i < NQ; i++) {
                sumq += (float)qs[i] * yl[i];
            }
            sumf[row] += sumq * d;
        }

        yb += NSG * NQ * QK;  // advance input pointer
    }

    // Reduction (ds4 helper_mv_reduce_and_write pattern)
    threadgroup float *shmem_f32[NR0];
    for (short row = 0; row < NR0; row++) {
        shmem_f32[row] = shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; row++) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; row++) {
        const int d = row0 + row;
        if (d >= (int)out_dim) continue;
        float tot = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[d] = tot;
        }
    }
}

// matvec_f32_bf16in: dense f32 matmul with bfloat input. (naive, 1-thread-per-row)
// Used for wo_a with small out_dim — kept for MLA attention path.
kernel void matvec_f32_bf16in(
    device const float*  W   [[buffer(0)]],
    device const bfloat* x   [[buffer(1)]],
    device float*        out [[buffer(2)]],
    constant uint&       out_dim [[buffer(3)]],
    constant uint&       in_dim  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    float acc = 0.0f;
    for (uint i = 0; i < in_dim; i++) acc += W[tid * in_dim + i] * float(x[i]);
    out[tid] = acc;
}

// matvec_f32_bf16in_simd: SIMD-parallel routing gate matmul.
// Optimized for routing gate [N_EXPERTS=256, DIM=4096] × [DIM] bfloat → [N_EXPERTS] f32.
// Strategy: ROWS_PER_TG=8 rows per threadgroup (one SIMD group per row),
//   x cached as f32 in threadgroup shared memory (eliminates redundant bf16→f32 converts),
//   32-thread simd_sum reduction → 32× fewer serial FMAs vs naive, coalesced W access.
// Dispatch: MTLSizeMake((out_dim+7)/8, 1, 1) threadgroups × 256 threads.
kernel void matvec_f32_bf16in_simd(
    device const float*  W       [[buffer(0)]],
    device const bfloat* x       [[buffer(1)]],
    device float*        out     [[buffer(2)]],
    constant uint&       out_dim [[buffer(3)]],
    constant uint&       in_dim  [[buffer(4)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_grp  [[simdgroup_index_in_threadgroup]]
) {
    const uint ROWS_PER_TG = 8;
    uint row = tgid * ROWS_PER_TG + simd_grp;

    // Cache x as f32 in shared memory (4096 bf16 → f32, 16KB per TG).
    // All 256 threads cooperate before any early return.
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = float(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const float* w_row = W + row * in_dim;
    float acc = 0.0f;
    // 32 threads in SIMD group stripe across in_dim: coalesced access.
    for (uint i = simd_lane; i < in_dim; i += 32) {
        acc += w_row[i] * x_shared[i];
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// mhc_pre_gpu: full mhc_pre computation on GPU with bfloat output for out_input.
// This matches MLX's HyperHead.forward() which computes in f32 internally but
// returns .astype(x.dtype) = bfloat16. The bfloat output is critical for
// matching MLX's precision in the attention chain.
//
// Dispatch: one threadgroup of 256 threads. Uses threadgroup memory for all
// intermediate results. Inputs:
//   fn_weight: [MIX3, HC*DIM] = [24, 16384] f32
//   base: [MIX3] = [24] f32
//   scale: [3] f32
//   residual: [HC, DIM] f32 (= [4, 4096])
// Outputs:
//   out_input: [DIM] f32 (bf16-truncated)
//   out_post: [HC] f32
//   out_comb: [HC*HC] f32 (sinkhorn-normalized)
//
// Constants: HC=4, DIM=4096, MIX3=24, EPS=1e-6
kernel void mhc_pre_gpu(
    device const float* fn_weight [[buffer(0)]],  // [24, 16384]
    device const float* base      [[buffer(1)]],  // [24]
    device const float* scale_v   [[buffer(2)]],  // [3]
    device const float* residual  [[buffer(3)]],  // [4, 4096]
    device float*       out_input [[buffer(4)]],  // [4096] bfloat-truncated f32
    device float*       out_post  [[buffer(5)]],  // [4]
    device float*       out_comb  [[buffer(6)]],  // [16]
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint HC = 4, DIM = 4096, MHC_H = HC * DIM, MIX3 = 24;
    const float EPS = 1e-6f, POST_MULT = 2.0f;

    // Threadgroup scratch: mixes[24], pre_mix[4], comb[16]
    threadgroup float mixes[24];
    threadgroup float pre_mix[4];
    threadgroup float comb_mat[16];  // [4][4]
    threadgroup float sum_sq;
    threadgroup float rms_norm_factor;

    // Step 1: compute sum(residual^2) / MHC_H
    float local_ss = 0.0f;
    for (uint i = lid; i < MHC_H; i += tg_size) local_ss += residual[i] * residual[i];
    threadgroup float ss_buf[256];
    ss_buf[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) ss_buf[lid] += ss_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        sum_sq = ss_buf[0];
        rms_norm_factor = rsqrt(sum_sq / float(MHC_H) + EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm = rms_norm_factor;

    // Step 2: compute mixes[r] = (fn[r,:] @ residual) * norm, for r in 0..23
    for (uint r = lid; r < MIX3; r += tg_size) {
        device const float* fn_r = fn_weight + r * MHC_H;
        float acc = 0.0f;
        for (uint i = 0; i < MHC_H; i++) acc += fn_r[i] * residual[i];
        mixes[r] = acc * norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: compute pre_mix, post, comb from mixes
    if (lid == 0) {
        float s0 = scale_v[0], s1 = scale_v[1], s2 = scale_v[2];
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[m] * s0 + base[m];
            pre_mix[m] = 1.0f / (1.0f + exp(-biased)) + EPS;
        }
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[HC + m] * s1 + base[HC + m];
            out_post[m] = (1.0f / (1.0f + exp(-biased))) * POST_MULT;
        }
        for (uint c = 0; c < HC * HC; c++) {
            comb_mat[c] = mixes[2 * HC + c] * s2 + base[2 * HC + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Sinkhorn normalization on comb_mat[4][4]
    // (Single-threaded, tiny compute — 4x4 matrix, 20 iterations)
    if (lid == 0) {
        // Initial softmax per row + eps
        for (uint i = 0; i < HC; i++) {
            float m_val = comb_mat[i * HC];
            for (uint j = 1; j < HC; j++) if (comb_mat[i*HC+j] > m_val) m_val = comb_mat[i*HC+j];
            float s_val = 0.0f;
            for (uint j = 0; j < HC; j++) { comb_mat[i*HC+j] = exp(comb_mat[i*HC+j] - m_val); s_val += comb_mat[i*HC+j]; }
            for (uint j = 0; j < HC; j++) comb_mat[i*HC+j] = comb_mat[i*HC+j] / s_val + EPS;
        }
        // Initial col-norm
        for (uint j = 0; j < HC; j++) {
            float cs = 0.0f; for (uint i = 0; i < HC; i++) cs += comb_mat[i*HC+j];
            cs += EPS; for (uint i = 0; i < HC; i++) comb_mat[i*HC+j] /= cs;
        }
        // 19 more row/col-norm iterations
        for (uint it = 0; it < 19; it++) {
            for (uint i = 0; i < HC; i++) {
                float rs = 0.0f; for (uint j = 0; j < HC; j++) rs += comb_mat[i*HC+j];
                rs += EPS; for (uint j = 0; j < HC; j++) comb_mat[i*HC+j] /= rs;
            }
            for (uint j = 0; j < HC; j++) {
                float cs = 0.0f; for (uint i = 0; i < HC; i++) cs += comb_mat[i*HC+j];
                cs += EPS; for (uint i = 0; i < HC; i++) comb_mat[i*HC+j] /= cs;
            }
        }
        // Write sinkhorn result to out_comb
        for (uint c = 0; c < HC * HC; c++) out_comb[c] = comb_mat[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: out_input[d] = sum_m pre_mix[m] * residual[m,d], truncated to bfloat
    for (uint d = lid; d < DIM; d += tg_size) {
        float acc = 0.0f;
        for (uint m = 0; m < HC; m++) acc += pre_mix[m] * residual[m * DIM + d];
        // Truncate to bfloat and back to f32 (matches MLX's .astype(x.dtype) = bfloat16)
        out_input[d] = float((bfloat)acc);
    }
}

// mhc_pre_bfloat: mhc_pre with bfloat residual input and bfloat output.
// residual: [HC, DIM] bfloat
// out_input: [DIM] bfloat (the sublayer input)
// out_post, out_comb: f32 (same as mhc_pre_gpu)
// All computation in f32, bfloat only at input/output boundaries.
kernel void mhc_pre_bfloat(
    device const float*  fn_weight [[buffer(0)]],  // [24, 16384] f32
    device const float*  base      [[buffer(1)]],  // [24] f32
    device const float*  scale_v   [[buffer(2)]],  // [3] f32
    device const bfloat* residual  [[buffer(3)]],  // [HC, DIM] bfloat
    device bfloat*       out_input [[buffer(4)]],  // [DIM] bfloat
    device float*        out_post  [[buffer(5)]],  // [HC] f32
    device float*        out_comb  [[buffer(6)]],  // [HC*HC] f32
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint HC = 4, DIM = 4096, MHC_H = HC * DIM, MIX3 = 24;
    const float EPS = 1e-6f, POST_MULT = 2.0f;

    threadgroup float mixes[24];
    threadgroup float pre_mix[4];
    threadgroup float comb_mat[16];
    threadgroup float ss_buf[256];

    // Step 1: compute mean(residual^2) using bfloat values (cast to float for accumulation)
    float local_ss = 0.0f;
    for (uint i = lid; i < MHC_H; i += tg_size) {
        float v = float(residual[i]); local_ss += v * v;
    }
    ss_buf[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) ss_buf[lid] += ss_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = rsqrt(ss_buf[0] / float(MHC_H) + EPS);

    // Step 2: mixes = (fn @ float(residual)) * norm
    for (uint r = lid; r < MIX3; r += tg_size) {
        device const float* fn_r = fn_weight + r * MHC_H;
        float acc = 0.0f;
        for (uint i = 0; i < MHC_H; i++) acc += fn_r[i] * float(residual[i]);
        mixes[r] = acc * norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: pre_mix, post, comb (single-threaded, same as mhc_pre_gpu)
    if (lid == 0) {
        float s0 = scale_v[0], s1 = scale_v[1], s2 = scale_v[2];
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[m] * s0 + base[m];
            pre_mix[m] = 1.0f / (1.0f + exp(-biased)) + EPS;
        }
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[HC + m] * s1 + base[HC + m];
            out_post[m] = (1.0f / (1.0f + exp(-biased))) * POST_MULT;
        }
        for (uint c = 0; c < HC * HC; c++)
            comb_mat[c] = mixes[2 * HC + c] * s2 + base[2 * HC + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Sinkhorn (single-threaded)
    if (lid == 0) {
        for (uint i = 0; i < HC; i++) {
            float m_val = comb_mat[i*HC]; for (uint j=1;j<HC;j++) if (comb_mat[i*HC+j]>m_val) m_val=comb_mat[i*HC+j];
            float s_val = 0.0f; for (uint j=0;j<HC;j++) { comb_mat[i*HC+j]=exp(comb_mat[i*HC+j]-m_val); s_val+=comb_mat[i*HC+j]; }
            for (uint j=0;j<HC;j++) comb_mat[i*HC+j]=comb_mat[i*HC+j]/s_val+EPS;
        }
        for (uint j=0;j<HC;j++) { float cs=0.0f; for (uint i=0;i<HC;i++) cs+=comb_mat[i*HC+j]; cs+=EPS; for (uint i=0;i<HC;i++) comb_mat[i*HC+j]/=cs; }
        for (uint it=0;it<19;it++) {
            for (uint i=0;i<HC;i++) { float rs=0.0f; for(uint j=0;j<HC;j++) rs+=comb_mat[i*HC+j]; rs+=EPS; for(uint j=0;j<HC;j++) comb_mat[i*HC+j]/=rs; }
            for (uint j=0;j<HC;j++) { float cs=0.0f; for(uint i=0;i<HC;i++) cs+=comb_mat[i*HC+j]; cs+=EPS; for(uint i=0;i<HC;i++) comb_mat[i*HC+j]/=cs; }
        }
        for (uint c=0;c<HC*HC;c++) out_comb[c]=comb_mat[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: out_input[d] = bfloat( sum_m pre_mix[m] * float(residual[m,d]) )
    for (uint d = lid; d < DIM; d += tg_size) {
        float acc = 0.0f;
        for (uint m = 0; m < HC; m++) acc += pre_mix[m] * float(residual[m * DIM + d]);
        out_input[d] = (bfloat)acc;
    }
}

// mhc_post_bfloat: mhc_post with bfloat residual I/O and bfloat sublayer output.
// out[m,d] = post[m] * float(x[d]) + sum_k comb[k,m] * float(residual[k,d])
// Result stored as bfloat → matches MLX's .astype(x.dtype) = bfloat16
// x: [DIM] bfloat (sublayer output: attn_out or ffn_out)
// residual: [HC, DIM] bfloat
// post: [HC] f32
// comb: [HC*HC] f32
// out_residual: [HC, DIM] bfloat (can alias residual for in-place)
kernel void mhc_post_bfloat(
    device const bfloat* x            [[buffer(0)]],  // [DIM] bfloat
    device const bfloat* residual     [[buffer(1)]],  // [HC, DIM] bfloat
    device const float*  post         [[buffer(2)]],  // [HC] f32
    device const float*  comb         [[buffer(3)]],  // [HC*HC] f32 (row-major [k][m])
    device bfloat*       out_residual [[buffer(4)]],  // [HC, DIM] bfloat
    constant uint&       hc           [[buffer(5)]],
    constant uint&       dim          [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    uint d = tid;
    float xv = float(x[d]);
    for (uint m = 0; m < hc; m++) {
        float acc = post[m] * xv;
        for (uint k = 0; k < hc; k++) {
            // comb[k][m] = comb[k*hc + m]
            acc += comb[k * hc + m] * float(residual[k * dim + d]);
        }
        out_residual[m * dim + d] = (bfloat)acc;
    }
}


// mhc_post_ffn_expand4: ds4 kernel_dsv4_hc_expand4 adapted for dmlx.
//
// Replaces the 3-encoder mhc_post_ffn cb3 with a single pure-f32 dispatch.
// Each thread handles one dimension, computing all 4 HC output streams.
// ZERO shared memory — no occupancy risk.
//
// Corresponds to ds4 dsv4_hc.metal:579-620 with HC=4, decode mode.
//
// block_out: ffn output [DIM] f32
// residual:  current residual [4, DIM] f32
// post:      per-HC gate coefficients [4] f32
// comb:      HC×HC comb matrix [4, 4] f32 (row-major: comb[k*4+m] = comb[k][m])
// dst:       new residual [4, DIM] f32 (in-place: dst can alias residual)
kernel void mhc_post_ffn_expand4(
    device const float* block_out  [[buffer(0)]],  // [DIM]
    device const float* residual   [[buffer(1)]],  // [4, DIM]
    device const float* post       [[buffer(2)]],  // [4]
    device const float* comb       [[buffer(3)]],  // [4, 4] row-major: comb[k*4+m]=comb[k][m]
    device float*       dst        [[buffer(4)]],  // [4, DIM]
    constant uint&      dim        [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) return;

    // Read block_out and residual with bfloat truncation to match the old
    // 3-encoder cb3 precision path:
    //   old: f32 → bf16 (encoder 1) → mhc_post_bfloat → bf16→f32 (encoder 3)
    //   new: emulate the bf16 round-trip within the kernel
    float block_v = float(bfloat(block_out[gid]));

    float r0 = float(bfloat(residual[0 * dim + gid]));
    float r1 = float(bfloat(residual[1 * dim + gid]));
    float r2 = float(bfloat(residual[2 * dim + gid]));
    float r3 = float(bfloat(residual[3 * dim + gid]));

    // HC stream 0
    float acc0 = block_v * post[0];
    acc0 += comb[0] * r0 + comb[4] * r1 + comb[8]  * r2 + comb[12] * r3;
    dst[0 * dim + gid] = float(bfloat(acc0));

    // HC stream 1
    float acc1 = block_v * post[1];
    acc1 += comb[1] * r0 + comb[5] * r1 + comb[9]  * r2 + comb[13] * r3;
    dst[1 * dim + gid] = float(bfloat(acc1));

    // HC stream 2
    float acc2 = block_v * post[2];
    acc2 += comb[2] * r0 + comb[6] * r1 + comb[10] * r2 + comb[14] * r3;
    dst[2 * dim + gid] = float(bfloat(acc2));

    // HC stream 3
    float acc3 = block_v * post[3];
    acc3 += comb[3] * r0 + comb[7] * r1 + comb[11] * r2 + comb[15] * r3;
    dst[3 * dim + gid] = float(bfloat(acc3));
}

// mhc_pre_split_weighted_sum_norm: ds4 kernel_dsv4_hc_split_weighted_sum_norm4 adapted for dmlx.
//
// Replaces CB-A's 2 encoders (mhc_pre_gpu + rms_norm_rows_bf16out) with a single
// dispatch that computes HC coefficients, collapses 4 residual streams, and applies
// RMSNorm — all in one kernel.
//
// Corresponds to ds4 dsv4_hc.metal:395-536.
//
// fn_weight: [24, 16384] f32  (MIX3 × MHC_H)
// base:      [24] f32
// scale_v:   [3] f32  (pre, post, comb scales)
// residual:  [4, DIM] f32  (4 HC streams × 4096)
// out_post:  [4] f32  (HC gate coefficients, for later mhc_post)
// out_comb:  [16] f32  (Sinkhorn'd comb matrix, for later mhc_post)
// attn_input:[DIM] f32  (collapsed row, for CPU diagnostics/compressor)
// norm_weight: [DIM] f32  (RMSNorm weight)
// normed:    [DIM] bf16  (RMSNorm'd output, feeds attention Q/KV chain)
//
// Threadgroup memory: ~17.5KB (row_shmem 16KB + mixes/pre_mix/comb/ss_buf ~1.5KB).
// 256 threads per threadgroup. 1 threadgroup for decode (1 token).
kernel void mhc_pre_split_weighted_sum_norm(
    device const float*  fn_weight   [[buffer(0)]],  // [24, 16384]
    device const float*  base        [[buffer(1)]],  // [24]
    device const float*  scale_v     [[buffer(2)]],  // [3]
    device const float*  residual    [[buffer(3)]],  // [4, DIM]
    device float*        out_post    [[buffer(4)]],  // [4]
    device float*        out_comb    [[buffer(5)]],  // [16]
    device float*        attn_input  [[buffer(6)]],  // [DIM]
    device const float*  norm_weight [[buffer(7)]],  // [DIM]
    device bfloat*       normed      [[buffer(8)]],  // [DIM]
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint HC = 4, DIM = 4096, MHC_H = HC * DIM, MIX3 = 24;
    const float EPS = 1e-6f, POST_MULT = 2.0f, NORM_EPS = 1e-6f;

    // Shared memory layout:
    //   row_shmem[0..DIM-1]:    collapsed row for RMSNorm (4096 floats = 16KB)
    //   mixes[DIM..DIM+23]:     dot products [24] (96B)
    //   pre_mix[DIM+24..DIM+27]: gate coefficients [4] (16B)
    //   comb_mat[DIM+28..DIM+43]: Sinkhorn matrix [16] (64B)
    //   ss_buf/DIM+44..]: sum-of-squares reduction (256 floats = 1KB)
    // Total: 16384 + 96 + 16 + 64 + 1024 = 17584 bytes < 32KB

    // Reusable shared memory: ss_buf used for both residual norm reduction (Phase A)
    // and collapsed-row sum-of-squares reduction (Phase B).
    threadgroup float ss_buf[256];

    // --- Phase A: mhc_pre (compute HC coefficients) ---

    // Step 1: compute sum(residual^2) / MHC_H → norm factor
    float local_ss = 0.0f;
    for (uint i = lid; i < MHC_H; i += tg_size) local_ss += residual[i] * residual[i];
    ss_buf[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) ss_buf[lid] += ss_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = rsqrt(ss_buf[0] / float(MHC_H) + EPS);

    // Note: can't use dynamic shared memory index on pre-declared threadgroup arrays,
    // so we use separate named threadgroup arrays for the Phase A data.
    threadgroup float mixes[24];
    threadgroup float pre_mix[4];
    threadgroup float comb_mat[16];

    // Step 2: mixes[r] = (fn[r,:] @ residual) * norm
    for (uint r = lid; r < MIX3; r += tg_size) {
        device const float* fn_r = fn_weight + r * MHC_H;
        float acc = 0.0f;
        for (uint i = 0; i < MHC_H; i++) acc += fn_r[i] * residual[i];
        mixes[r] = acc * norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: compute pre_mix, post, comb from mixes (single-threaded)
    if (lid == 0) {
        float s0 = scale_v[0], s1 = scale_v[1], s2 = scale_v[2];
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[m] * s0 + base[m];
            pre_mix[m] = 1.0f / (1.0f + exp(-biased)) + EPS;
        }
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[HC + m] * s1 + base[HC + m];
            out_post[m] = (1.0f / (1.0f + exp(-biased))) * POST_MULT;
        }
        for (uint c = 0; c < HC * HC; c++) {
            comb_mat[c] = mixes[2 * HC + c] * s2 + base[2 * HC + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Sinkhorn normalization on comb_mat (single-threaded, 4×4 matrix)
    if (lid == 0) {
        for (uint i = 0; i < HC; i++) {
            float m_val = comb_mat[i * HC];
            for (uint j = 1; j < HC; j++) if (comb_mat[i*HC+j] > m_val) m_val = comb_mat[i*HC+j];
            float s_val = 0.0f;
            for (uint j = 0; j < HC; j++) { comb_mat[i*HC+j] = exp(comb_mat[i*HC+j] - m_val); s_val += comb_mat[i*HC+j]; }
            for (uint j = 0; j < HC; j++) comb_mat[i*HC+j] = comb_mat[i*HC+j] / s_val + EPS;
        }
        for (uint j = 0; j < HC; j++) {
            float cs = 0.0f; for (uint i = 0; i < HC; i++) cs += comb_mat[i*HC+j];
            cs += EPS; for (uint i = 0; i < HC; i++) comb_mat[i*HC+j] /= cs;
        }
        for (uint it = 0; it < 19; it++) {
            for (uint i = 0; i < HC; i++) {
                float rs = 0.0f; for (uint j = 0; j < HC; j++) rs += comb_mat[i*HC+j];
                rs += EPS; for (uint j = 0; j < HC; j++) comb_mat[i*HC+j] /= rs;
            }
            for (uint j = 0; j < HC; j++) {
                float cs = 0.0f; for (uint i = 0; i < HC; i++) cs += comb_mat[i*HC+j];
                cs += EPS; for (uint i = 0; i < HC; i++) comb_mat[i*HC+j] /= cs;
            }
        }
        for (uint c = 0; c < HC * HC; c++) out_comb[c] = comb_mat[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase B: weighted sum + RMSNorm (ds4 kernel_dsv4_hc_split_weighted_sum_norm4 pattern) ---
    // Compute collapsed row in shared memory, accumulate sum_sq on the fly,
    // then apply RMSNorm.

    // Step 5: weighted sum → row_shmem (with bf16 truncation to match
    //   old mhc_pre_gpu's out_input[d] = float((bfloat)acc)), accumulate sum_sq
    threadgroup float row_shmem[4096];
    float row_ss = 0.0f;
    for (uint d = lid; d < DIM; d += tg_size) {
        float acc = 0.0f;
        for (uint m = 0; m < HC; m++) acc += pre_mix[m] * residual[m * DIM + d];
        float v = float(bfloat(acc));  // bf16 truncation — matches old 2-encoder path
        row_shmem[d] = v;
        row_ss += v * v;
    }
    // Reduce sum_sq across threads
    ss_buf[lid] = row_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) ss_buf[lid] += ss_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm_scale = rsqrt(ss_buf[0] / float(DIM) + NORM_EPS);

    // Step 6: write attn_input + apply RMSNorm → normed
    for (uint d = lid; d < DIM; d += tg_size) {
        float v = row_shmem[d];
        attn_input[d] = v;                               // f32 collapsed row (CPU diagnostics)
        normed[d] = bfloat(v * norm_scale * norm_weight[d]); // bf16 normed output (attention input)
    }
}

// f32_to_bf16_vec: convert n f32 values → bfloat in-place (GPU-side f32→bf16 conversion).
// Used in Path B to avoid CPU residual readback before mhc_post.
// Dispatch: (n + 255) / 256 threadgroups × 256 threads.
kernel void f32_to_bf16_vec(
    device const float*   src [[buffer(0)]],
    device bfloat*        dst [[buffer(1)]],
    constant uint&        n   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    dst[tid] = bfloat(src[tid]);
}

// bf16_to_f32_vec: convert n bfloat values → f32 (for residual writeback).
// Dispatch: (n + 255) / 256 threadgroups × 256 threads.
kernel void bf16_to_f32_vec(
    device const bfloat*  src [[buffer(0)]],
    device float*         dst [[buffer(1)]],
    constant uint&        n   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    dst[tid] = float(src[tid]);
}

// limited_swiglu: in-place GPU SwiGLU for shared expert — eliminates CPU round-trip.
// gate[i] = gate[i] / (1 + exp(-gate[i])) * up[i], with clamping to avoid overflow.
// Writes result into gate_buf (in-place); up_buf is read-only.
// Dispatch: (n + 255) / 256 threadgroups × 256 threads.
kernel void limited_swiglu(
    device float*       gate_buf [[buffer(0)]],  // [n] gate (overwritten with SwiGLU output)
    device const float* up_buf   [[buffer(1)]],  // [n] up
    constant uint&      n        [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    const float limit = 10.0f;
    float g = min(gate_buf[tid], limit);
    float u = min(max(up_buf[tid], -limit), limit);
    gate_buf[tid] = (g / (1.0f + exp(-g))) * u;
}

// bf16_to_f16_row: convert KV_LORA_RANK bfloat values → half, writing into kv_cache row.
// Used within CB1 to eliminate CPU KV-cache round-trip (GPU blit of bf16 + convert in-place).
// Dispatch: 1 threadgroup × KV_LORA_RANK threads.
kernel void bf16_to_f16_row(
    device const bfloat* src      [[buffer(0)]],   // [KV_LORA_RANK] bfloat — bkv_n output
    device half*         dst_cache[[buffer(1)]],   // [MAX_SEQ_LEN, KV_LORA_RANK] half — full kv_cache
    constant uint&       row_idx  [[buffer(2)]],   // which row to write (= cache_len - 1)
    constant uint&       rank     [[buffer(3)]],   // KV_LORA_RANK
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= rank) return;
    dst_cache[row_idx * rank + tid] = half(float(src[tid]));
}

// mla_sdpa_decode_bfloat: ds4-style decode SDPA — float4 dot product, f16 KV cache.
//
// Matches ds4 kernel_flash_attn_ext_vec_f16_dk512_dv512:
//  - Q: bf16 input loaded via float4 (cast from bfloat)
//  - KV: f16 (half) — same precision as ds4/MLX
//  - dot(float4, float4) FMA path
//  - 32 threads (1 simdgroup), 1 head per threadgroup
//  - SLOTS=4: each thread owns 4 float4 = 16 f32 dims; 32*16=512 total
kernel void mla_sdpa_decode_bfloat(
    device const bfloat* q       [[buffer(0)]],  // [n_heads, head_dim] bf16
    device const half*   kv      [[buffer(1)]],  // [n_kv, head_dim]    f16
    device const float*  sinks   [[buffer(2)]],  // [n_heads]           f32
    device       bfloat* out     [[buffer(3)]],  // [n_heads, head_dim] bf16
    constant uint&       n_heads [[buffer(4)]],
    constant uint&       head_dim[[buffer(5)]],
    constant uint&       n_kv    [[buffer(6)]],
    constant float&      scale   [[buffer(7)]],
    uint  head  [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]]
) {
    if (head >= n_heads) return;

    // head_dim = 512, fixed. NW=32 threads, SLOTS=4 float4s per thread.
    // Thread tiisg owns float4 indices: {tiisg, tiisg+32, tiisg+64, tiisg+96}
    const uint NW    = 32;
    const uint DK4   = 512 / 4;    // = 128 float4s per row
    // const uint SLOTS = DK4 / NW;   // = 4

    // Load Q row for this head as float4 array
    // Use explicit conversion from bfloat* via float4 cast
    float4 qv[4]; // SLOTS=4
    {
        device const bfloat* qh = q + (uint64_t)head * 512u;
        for (uint ii = 0; ii < 4; ii++) {
            uint base = (ii * NW + tiisg) * 4;
            qv[ii] = float4(float(qh[base+0]), float(qh[base+1]),
                            float(qh[base+2]), float(qh[base+3]));
        }
    }

    float4 so[4]; // SLOTS=4
    for (uint ii = 0; ii < 4; ii++) so[ii] = float4(0.0f);

    float S = 0.0f;
    float M = -FLT_MAX / 2.0f;

    for (uint k = 0; k < n_kv; k++) {
        device const half* kvk = kv + (uint64_t)k * 512u;

        // Partial dot product: sum over this thread's 4 float4 slots
        float mqk = 0.0f;
        for (uint ii = 0; ii < 4; ii++) {
            uint base = (ii * NW + tiisg) * 4;
            float4 kv4 = float4(float(kvk[base+0]), float(kvk[base+1]),
                                float(kvk[base+2]), float(kvk[base+3]));
            mqk += dot(qv[ii], kv4);
        }
        // All 32 lanes get the full dot product
        float score = simd_sum(mqk) * scale;

        // Online softmax update (same value in all lanes)
        float m_new = max(M, score);
        float ms    = exp(M - m_new);
        float vs    = exp(score - m_new);

        S = S * ms + vs;  // single KV row: denominator += one exp
        M = m_new;

        // Accumulate weighted KV
        for (uint ii = 0; ii < 4; ii++) {
            uint base = (ii * NW + tiisg) * 4;
            float4 kv4 = float4(float(kvk[base+0]), float(kvk[base+1]),
                                float(kvk[base+2]), float(kvk[base+3]));
            so[ii] = so[ii] * ms + kv4 * vs;
        }
    }

    // Sink: denominator-only, zero value contribution
    {
        float sink  = sinks[head];
        float m_new = max(M, sink);
        float ms    = exp(M - m_new);
        float vs    = exp(sink - m_new);
        S = S * ms + vs;
        M = m_new;
        for (uint ii = 0; ii < 4; ii++) so[ii] *= ms;
    }

    float inv_s = (S > 0.0f) ? 1.0f / S : 0.0f;
    {
        device bfloat* oh = out + (uint64_t)head * 512u;
        for (uint ii = 0; ii < 4; ii++) {
            float4 v = so[ii] * inv_s;
            uint base = (ii * NW + tiisg) * 4;
            oh[base+0] = (bfloat)v.x;
            oh[base+1] = (bfloat)v.y;
            oh[base+2] = (bfloat)v.z;
            oh[base+3] = (bfloat)v.w;
        }
    }
}

// mla_sdpa_decode_f16: same as mla_sdpa_decode but KV cache is f16 (half).
// Q remains f32, KV is stored as f16, output is f32.
// This matches ds4's precision path: Q(f32) · KV(f16→f32) with f32 accumulation.
kernel void mla_sdpa_decode_f16(
    device const float* q        [[buffer(0)]],
    device const half*  kv       [[buffer(1)]],
    device const float* sinks    [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant uint&      n_heads  [[buffer(4)]],
    constant uint&      head_dim [[buffer(5)]],
    constant uint&      n_kv     [[buffer(6)]],
    constant float&     scale    [[buffer(7)]],
    uint  head [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tg   [[threads_per_threadgroup]]
) {
    if (head >= n_heads) return;
    threadgroup float red[32];
    device const float* qh = q + (uint64_t)head * head_dim;

    threadgroup float t_m;
    threadgroup float t_s;
    threadgroup float t_score;
    if (lid == 0) { t_m = -INFINITY; t_s = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[8];
    uint n_slots = (head_dim + tg - 1) / tg;
    for (uint i = 0; i < n_slots; i++) acc[i] = 0.0f;

    for (uint k = 0; k < n_kv; k++) {
        device const half* kvk = kv + (uint64_t)k * head_dim;
        // ds4-style: Q and K are cast to half before dot product,
        // matching the f16 precision SDPA path that ds4 uses.
        float partial = 0.0f;
        for (uint d = lid; d < head_dim; d += tg) {
            half qh_h = (half)qh[d];
            half kh_h = kvk[d];
            partial += float(qh_h) * float(kh_h);
        }
        float dot = simd_sum(partial);
        uint lane = lid % 32, sg = lid / 32;
        if (lane == 0) red[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tot = 0.0f;
            uint n_sg = (tg + 31) / 32;
            for (uint g = 0; g < n_sg; g++) tot += red[g];
            t_score = tot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = t_score;
        float m_old = t_m;
        float m_new = max(m_old, score);
        float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
        float p = exp(score - m_new);
        for (uint i = 0; i < n_slots; i++) {
            uint d = lid + i * tg;
            if (d < head_dim) acc[i] = acc[i] * corr + p * float(kvk[d]);
        }
        if (lid == 0) { t_s = t_s * corr + p; t_m = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Fold per-head sink into the denominator
    if (lid == 0) {
        float sink = sinks[head];
        float m_old = t_m;
        float m_new = max(m_old, sink);
        float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
        t_s = t_s * corr + exp(sink - m_new);
        t_m = m_new;
        red[0] = corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float corr_final = red[0];
    float inv_s = (t_s == 0.0f) ? 0.0f : 1.0f / t_s;
    device float* oh = out + (uint64_t)head * head_dim;
    for (uint i = 0; i < n_slots; i++) {
        uint d = lid + i * tg;
        if (d < head_dim) oh[d] = (acc[i] * corr_final) * inv_s;
    }
}

kernel void rms_norm_sum_sq(
    device const float* x       [[buffer(0)]],
    device float*       sum_sq  [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];
    float acc = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) { float val = x[i]; acc += val * val; }
    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32, simd_group = lid / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) sum_sq[0] = val;
    }
}

// rms_norm_apply: out = x * rsqrt(sum_sq/dim + eps) * weight
kernel void rms_norm_apply(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* sum_sq  [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      dim     [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    out[tid] = x[tid] * rms * weight[tid];
}

// matvec_f32: out = W @ x. W: [out_dim, in_dim], x: [in_dim]
kernel void matvec_f32(
    device const float*    W      [[buffer(0)]],
    device const float*    x      [[buffer(1)]],
    device float*          out    [[buffer(2)]],
    constant uint&         out_dim [[buffer(3)]],
    constant uint&         in_dim  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    float acc = 0.0f;
    for (uint j = 0; j < in_dim; j++) acc += W[tid * in_dim + j] * x[j];
    out[tid] = acc;
}

// ===========================================================================
// S2: MLA Q-chain kernels
// ===========================================================================

// rms_norm_rows: per-row RMSNorm over `row_dim`, for `n_rows` rows.
// out[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * (weight ? weight[:] : 1)
// One threadgroup per row; 256 threads cooperatively reduce.
// weight may be null (pass has_weight=0) for the weightless per-head norm.
kernel void rms_norm_rows(
    device const float* x        [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant uint&      row_dim  [[buffer(3)]],
    constant float&     eps      [[buffer(4)]],
    constant uint&      has_weight [[buffer(5)]],
    uint  row [[threadgroup_position_in_grid]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  tg  [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];
    device const float* xr = x + (uint64_t)row * row_dim;
    device float*       orow = out + (uint64_t)row * row_dim;

    float acc = 0.0f;
    for (uint i = lid; i < row_dim; i += tg) { float v = xr[i]; acc += v * v; }
    float s = simd_sum(acc);
    uint lane = lid % 32, sg = lid / 32;
    if (lane == 0) shared[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    uint n_sg = (tg + 31) / 32;
    if (lid < n_sg) total = shared[lid];
    total = simd_sum(total);
    // broadcast via shared[0]
    if (lid == 0) shared[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = rsqrt(shared[0] / float(row_dim) + eps);
    for (uint i = lid; i < row_dim; i += tg) {
        float w = (has_weight != 0u) ? weight[i] : 1.0f;
        orow[i] = xr[i] * rms * w;
    }
}

// rope_tail_interleaved: apply YaRN tail RoPE to the last rope_dim of each head.
// q: [n_heads, head_dim]; rotates dims [nope_dim .. head_dim) in interleaved
// pairs (2i, 2i+1) using precomputed cos/sin (length rope_dim/2) for this pos.
// Matches DSV4YarnRoPE (interleaved, NOT split-half). inverse negates sin.
kernel void rope_tail_interleaved(
    device float*       q        [[buffer(0)]],
    device const float* cos_t    [[buffer(1)]],
    device const float* sin_t    [[buffer(2)]],
    constant uint&      n_heads  [[buffer(3)]],
    constant uint&      head_dim [[buffer(4)]],
    constant uint&      nope_dim [[buffer(5)]],
    constant uint&      rope_dim [[buffer(6)]],
    constant uint&      inverse  [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint half_rope = rope_dim / 2;
    uint total = n_heads * half_rope;
    if (tid >= total) return;
    uint h = tid / half_rope;
    uint i = tid % half_rope;
    float c = cos_t[i];
    float s = sin_t[i];
    if (inverse != 0u) s = -s;
    device float* row = q + (uint64_t)h * head_dim;
    uint j0 = nope_dim + 2 * i;
    uint j1 = nope_dim + 2 * i + 1;
    float x0 = row[j0];
    float x1 = row[j1];
    row[j0] = x0 * c - x1 * s;
    row[j1] = x0 * s + x1 * c;
}

// ===========================================================================
// S4: MLA SDPA (decode, single query token) + attention sink
// ===========================================================================

// mla_sdpa_decode: one query token attends to `n_kv` cached KV rows (MQA: a
// single KV head shared by all N_HEADS query heads), with per-head attention
// sink folded into the softmax denominator (matches MLX fast SDPA sinks).
//
//   score_k = (q_h . kv[k]) * scale         for k in [0, n_kv)
//   denom   = sum_k exp(score_k - M) + exp(sink_h - M)   (M = running max)
//   out_h   = sum_k softmax_k * kv[k]
//
// One threadgroup per head; 256 threads cooperatively reduce over head_dim.
// q:   [n_heads, head_dim]
// kv:  [n_kv, head_dim]   (shared single KV head)
// out: [n_heads, head_dim]
// sinks: [n_heads]
kernel void mla_sdpa_decode(
    device const float* q        [[buffer(0)]],
    device const float* kv       [[buffer(1)]],
    device const float* sinks    [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant uint&      n_heads  [[buffer(4)]],
    constant uint&      head_dim [[buffer(5)]],
    constant uint&      n_kv     [[buffer(6)]],
    constant float&     scale    [[buffer(7)]],
    uint  head [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tg   [[threads_per_threadgroup]]
) {
    if (head >= n_heads) return;
    threadgroup float red[32];
    device const float* qh = q + (uint64_t)head * head_dim;

    // Online softmax accumulation over keys. Each thread owns a partial output
    // slice acc[d] for d = lid, lid+tg, ...  Running max M and denom S are
    // threadgroup-wide scalars kept in red[0]/red[1] via reduction per key.
    // For simplicity (correctness-first), recompute the dot product cooperatively.
    threadgroup float t_m;     // running max
    threadgroup float t_s;     // running denom
    threadgroup float t_score; // current key score
    if (lid == 0) { t_m = -INFINITY; t_s = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // per-thread output accumulator over its strided dims
    float acc[8];
    uint n_slots = (head_dim + tg - 1) / tg;
    for (uint i = 0; i < n_slots; i++) acc[i] = 0.0f;

    for (uint k = 0; k < n_kv; k++) {
        device const float* kvk = kv + (uint64_t)k * head_dim;
        // cooperative dot product q_h . kv[k]
        float partial = 0.0f;
        for (uint d = lid; d < head_dim; d += tg) partial += qh[d] * kvk[d];
        float dot = simd_sum(partial);
        uint lane = lid % 32, sg = lid / 32;
        if (lane == 0) red[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tot = 0.0f;
            uint n_sg = (tg + 31) / 32;
            for (uint g = 0; g < n_sg; g++) tot += red[g];
            t_score = tot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = t_score;
        float m_old = t_m;
        float m_new = max(m_old, score);
        float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
        float p = exp(score - m_new);
        // rescale acc and add p * kv[k]
        for (uint i = 0; i < n_slots; i++) {
            uint d = lid + i * tg;
            if (d < head_dim) acc[i] = acc[i] * corr + p * kvk[d];
        }
        if (lid == 0) { t_s = t_s * corr + p; t_m = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Fold per-head sink into the denominator (no output contribution).
    if (lid == 0) {
        float sink = sinks[head];
        float m_old = t_m;
        float m_new = max(m_old, sink);
        float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
        t_s = t_s * corr + exp(sink - m_new);
        t_m = m_new;
        red[0] = corr; // broadcast correction for acc rescale
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float corr_final = red[0];
    float inv_s = (t_s == 0.0f) ? 0.0f : 1.0f / t_s;
    device float* oh = out + (uint64_t)head * head_dim;
    for (uint i = 0; i < n_slots; i++) {
        uint d = lid + i * tg;
        if (d < head_dim) oh[d] = (acc[i] * corr_final) * inv_s;
    }
}

// ============================================================================
// F16-precision kernels (ds4-style). Generated from bf16 variants above.
// ============================================================================
kernel void dequant_matvec_affine_f16out(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const float*    x        [[buffer(3)]],
    device half*         out      [[buffer(4)]],
    constant uint&         out_dim  [[buffer(5)]],
    constant uint&         in_dim   [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const float*    sc = scales   + tid * num_groups;
    device const float*    bi = biases   + tid * num_groups;

    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = sc[g];
        float bias  = bi[g];
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            uint x_base = bx + p * 8;
            for (uint i = 0; i < 8; i++) {
                float nib = (float)((pw >> (i * 4)) & 0xF);
                acc += (scale * nib + bias) * x[x_base + i];
            }
        }
    }
    out[tid] = (half)acc;
}

kernel void rms_norm_rows_f16out(
    device const float*  x          [[buffer(0)]],
    device const float*  weight     [[buffer(1)]],
    device half*       out        [[buffer(2)]],
    constant uint&       row_dim    [[buffer(3)]],
    constant float&      eps        [[buffer(4)]],
    constant uint&       has_weight [[buffer(5)]],
    uint row     [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    float ss = 0.0f;
    for (uint i = lid; i < row_dim; i += tg_size) {
        float v = x[row * row_dim + i]; ss += v * v;
    }
    shared_sum[lid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared_sum[lid] += shared_sum[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_inv = rsqrt(shared_sum[0] / float(row_dim) + eps);
    for (uint i = lid; i < row_dim; i += tg_size) {
        float v = x[row * row_dim + i] * rms_inv;
        out[row * row_dim + i] = (half)(has_weight ? v * weight[i] : v);
    }
}

kernel void dequant_matvec_affine_f16in_f16out(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const half*   x        [[buffer(3)]],  // half input
    device half*         out      [[buffer(4)]],  // half output
    constant uint&         out_dim  [[buffer(5)]],
    constant uint&         in_dim   [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const float*    sc = scales   + tid * num_groups;
    device const float*    bi = biases   + tid * num_groups;
    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = sc[g], bias = bi[g];
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            for (uint i = 0; i < 8; i++) {
                float nib = (float)((pw >> (i * 4)) & 0xF);
                acc += (scale * nib + bias) * float(x[bx + p * 8 + i]);
            }
        }
    }
    out[tid] = (half)acc;
}

kernel void rms_norm_rows_f16in_f16out(
    device const half* x          [[buffer(0)]],
    device const float*  weight     [[buffer(1)]],
    device half*       out        [[buffer(2)]],
    constant uint&       row_dim    [[buffer(3)]],
    constant float&      eps        [[buffer(4)]],
    constant uint&       has_weight [[buffer(5)]],
    uint row     [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    float ss = 0.0f;
    for (uint i = lid; i < row_dim; i += tg_size) {
        float v = float(x[row * row_dim + i]); ss += v * v;
    }
    shared_sum[lid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared_sum[lid] += shared_sum[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_inv = rsqrt(shared_sum[0] / float(row_dim) + eps);
    for (uint i = lid; i < row_dim; i += tg_size) {
        float v = float(x[row * row_dim + i]) * rms_inv;
        out[row * row_dim + i] = (half)(has_weight ? v * weight[i] : v);
    }
}

kernel void rope_tail_interleaved_f16(
    device half*         q          [[buffer(0)]],  // in-place [n_heads, head_dim]
    device const float*    cos_vals   [[buffer(1)]],  // [half_rope]
    device const float*    sin_vals   [[buffer(2)]],  // [half_rope]
    constant uint&         n_heads    [[buffer(3)]],
    constant uint&         head_dim   [[buffer(4)]],
    constant uint&         n_nope     [[buffer(5)]],
    constant uint&         n_rope     [[buffer(6)]],
    constant uint&         inverse    [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint half_rope = n_rope / 2;
    if (tid >= n_heads * half_rope) return;
    uint head = tid / half_rope, ic = tid % half_rope;
    uint j0 = head * head_dim + n_nope + 2 * ic;
    uint j1 = j0 + 1;
    float cos_v = cos_vals[ic];
    float sin_v = inverse ? -sin_vals[ic] : sin_vals[ic];
    float x0 = float(q[j0]);
    float x1 = float(q[j1]);
    q[j0] = (half)(x0 * cos_v - x1 * sin_v);
    q[j1] = (half)(x0 * sin_v + x1 * cos_v);
}

kernel void matvec_f32_f16in(
    device const float*  W   [[buffer(0)]],
    device const half* x   [[buffer(1)]],
    device float*        out [[buffer(2)]],
    constant uint&       out_dim [[buffer(3)]],
    constant uint&       in_dim  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    float acc = 0.0f;
    for (uint i = 0; i < in_dim; i++) acc += W[tid * in_dim + i] * float(x[i]);
    out[tid] = acc;
}

kernel void mhc_pre_gpu_f16(
    device const float* fn_weight [[buffer(0)]],  // [24, 16384]
    device const float* base      [[buffer(1)]],  // [24]
    device const float* scale_v   [[buffer(2)]],  // [3]
    device const float* residual  [[buffer(3)]],  // [4, 4096]
    device float*       out_input [[buffer(4)]],  // [4096] half-truncated f32
    device float*       out_post  [[buffer(5)]],  // [4]
    device float*       out_comb  [[buffer(6)]],  // [16]
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint HC = 4, DIM = 4096, MHC_H = HC * DIM, MIX3 = 24;
    const float EPS = 1e-6f, POST_MULT = 2.0f;

    // Threadgroup scratch: mixes[24], pre_mix[4], comb[16]
    threadgroup float mixes[24];
    threadgroup float pre_mix[4];
    threadgroup float comb_mat[16];  // [4][4]
    threadgroup float sum_sq;
    threadgroup float rms_norm_factor;

    // Step 1: compute sum(residual^2) / MHC_H
    float local_ss = 0.0f;
    for (uint i = lid; i < MHC_H; i += tg_size) local_ss += residual[i] * residual[i];
    threadgroup float ss_buf[256];
    ss_buf[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) ss_buf[lid] += ss_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        sum_sq = ss_buf[0];
        rms_norm_factor = rsqrt(sum_sq / float(MHC_H) + EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm = rms_norm_factor;

    // Step 2: compute mixes[r] = (fn[r,:] @ residual) * norm, for r in 0..23
    for (uint r = lid; r < MIX3; r += tg_size) {
        device const float* fn_r = fn_weight + r * MHC_H;
        float acc = 0.0f;
        for (uint i = 0; i < MHC_H; i++) acc += fn_r[i] * residual[i];
        mixes[r] = acc * norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: compute pre_mix, post, comb from mixes
    if (lid == 0) {
        float s0 = scale_v[0], s1 = scale_v[1], s2 = scale_v[2];
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[m] * s0 + base[m];
            pre_mix[m] = 1.0f / (1.0f + exp(-biased)) + EPS;
        }
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[HC + m] * s1 + base[HC + m];
            out_post[m] = (1.0f / (1.0f + exp(-biased))) * POST_MULT;
        }
        for (uint c = 0; c < HC * HC; c++) {
            comb_mat[c] = mixes[2 * HC + c] * s2 + base[2 * HC + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Sinkhorn normalization on comb_mat[4][4]
    // (Single-threaded, tiny compute — 4x4 matrix, 20 iterations)
    if (lid == 0) {
        // Initial softmax per row + eps
        for (uint i = 0; i < HC; i++) {
            float m_val = comb_mat[i * HC];
            for (uint j = 1; j < HC; j++) if (comb_mat[i*HC+j] > m_val) m_val = comb_mat[i*HC+j];
            float s_val = 0.0f;
            for (uint j = 0; j < HC; j++) { comb_mat[i*HC+j] = exp(comb_mat[i*HC+j] - m_val); s_val += comb_mat[i*HC+j]; }
            for (uint j = 0; j < HC; j++) comb_mat[i*HC+j] = comb_mat[i*HC+j] / s_val + EPS;
        }
        // Initial col-norm
        for (uint j = 0; j < HC; j++) {
            float cs = 0.0f; for (uint i = 0; i < HC; i++) cs += comb_mat[i*HC+j];
            cs += EPS; for (uint i = 0; i < HC; i++) comb_mat[i*HC+j] /= cs;
        }
        // 19 more row/col-norm iterations
        for (uint it = 0; it < 19; it++) {
            for (uint i = 0; i < HC; i++) {
                float rs = 0.0f; for (uint j = 0; j < HC; j++) rs += comb_mat[i*HC+j];
                rs += EPS; for (uint j = 0; j < HC; j++) comb_mat[i*HC+j] /= rs;
            }
            for (uint j = 0; j < HC; j++) {
                float cs = 0.0f; for (uint i = 0; i < HC; i++) cs += comb_mat[i*HC+j];
                cs += EPS; for (uint i = 0; i < HC; i++) comb_mat[i*HC+j] /= cs;
            }
        }
        // Write sinkhorn result to out_comb
        for (uint c = 0; c < HC * HC; c++) out_comb[c] = comb_mat[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: out_input[d] = sum_m pre_mix[m] * residual[m,d], truncated to half
    for (uint d = lid; d < DIM; d += tg_size) {
        float acc = 0.0f;
        for (uint m = 0; m < HC; m++) acc += pre_mix[m] * residual[m * DIM + d];
        // Truncate to half and back to f32 (matches MLX's .astype(x.dtype) = half16)
        out_input[d] = float((half)acc);
    }
}

kernel void mhc_pre_f16(
    device const float*  fn_weight [[buffer(0)]],  // [24, 16384] f32
    device const float*  base      [[buffer(1)]],  // [24] f32
    device const float*  scale_v   [[buffer(2)]],  // [3] f32
    device const half* residual  [[buffer(3)]],  // [HC, DIM] half
    device half*       out_input [[buffer(4)]],  // [DIM] half
    device float*        out_post  [[buffer(5)]],  // [HC] f32
    device float*        out_comb  [[buffer(6)]],  // [HC*HC] f32
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint HC = 4, DIM = 4096, MHC_H = HC * DIM, MIX3 = 24;
    const float EPS = 1e-6f, POST_MULT = 2.0f;

    threadgroup float mixes[24];
    threadgroup float pre_mix[4];
    threadgroup float comb_mat[16];
    threadgroup float ss_buf[256];

    // Step 1: compute mean(residual^2) using half values (cast to float for accumulation)
    float local_ss = 0.0f;
    for (uint i = lid; i < MHC_H; i += tg_size) {
        float v = float(residual[i]); local_ss += v * v;
    }
    ss_buf[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) ss_buf[lid] += ss_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = rsqrt(ss_buf[0] / float(MHC_H) + EPS);

    // Step 2: mixes = (fn @ float(residual)) * norm
    for (uint r = lid; r < MIX3; r += tg_size) {
        device const float* fn_r = fn_weight + r * MHC_H;
        float acc = 0.0f;
        for (uint i = 0; i < MHC_H; i++) acc += fn_r[i] * float(residual[i]);
        mixes[r] = acc * norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: pre_mix, post, comb (single-threaded, same as mhc_pre_gpu)
    if (lid == 0) {
        float s0 = scale_v[0], s1 = scale_v[1], s2 = scale_v[2];
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[m] * s0 + base[m];
            pre_mix[m] = 1.0f / (1.0f + exp(-biased)) + EPS;
        }
        for (uint m = 0; m < HC; m++) {
            float biased = mixes[HC + m] * s1 + base[HC + m];
            out_post[m] = (1.0f / (1.0f + exp(-biased))) * POST_MULT;
        }
        for (uint c = 0; c < HC * HC; c++)
            comb_mat[c] = mixes[2 * HC + c] * s2 + base[2 * HC + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Sinkhorn (single-threaded)
    if (lid == 0) {
        for (uint i = 0; i < HC; i++) {
            float m_val = comb_mat[i*HC]; for (uint j=1;j<HC;j++) if (comb_mat[i*HC+j]>m_val) m_val=comb_mat[i*HC+j];
            float s_val = 0.0f; for (uint j=0;j<HC;j++) { comb_mat[i*HC+j]=exp(comb_mat[i*HC+j]-m_val); s_val+=comb_mat[i*HC+j]; }
            for (uint j=0;j<HC;j++) comb_mat[i*HC+j]=comb_mat[i*HC+j]/s_val+EPS;
        }
        for (uint j=0;j<HC;j++) { float cs=0.0f; for (uint i=0;i<HC;i++) cs+=comb_mat[i*HC+j]; cs+=EPS; for (uint i=0;i<HC;i++) comb_mat[i*HC+j]/=cs; }
        for (uint it=0;it<19;it++) {
            for (uint i=0;i<HC;i++) { float rs=0.0f; for(uint j=0;j<HC;j++) rs+=comb_mat[i*HC+j]; rs+=EPS; for(uint j=0;j<HC;j++) comb_mat[i*HC+j]/=rs; }
            for (uint j=0;j<HC;j++) { float cs=0.0f; for(uint i=0;i<HC;i++) cs+=comb_mat[i*HC+j]; cs+=EPS; for(uint i=0;i<HC;i++) comb_mat[i*HC+j]/=cs; }
        }
        for (uint c=0;c<HC*HC;c++) out_comb[c]=comb_mat[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: out_input[d] = half( sum_m pre_mix[m] * float(residual[m,d]) )
    for (uint d = lid; d < DIM; d += tg_size) {
        float acc = 0.0f;
        for (uint m = 0; m < HC; m++) acc += pre_mix[m] * float(residual[m * DIM + d]);
        out_input[d] = (half)acc;
    }
}

kernel void mhc_post_f16(
    device const half* x            [[buffer(0)]],  // [DIM] half
    device const half* residual     [[buffer(1)]],  // [HC, DIM] half
    device const float*  post         [[buffer(2)]],  // [HC] f32
    device const float*  comb         [[buffer(3)]],  // [HC*HC] f32 (row-major [k][m])
    device half*       out_residual [[buffer(4)]],  // [HC, DIM] half
    constant uint&       hc           [[buffer(5)]],
    constant uint&       dim          [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    uint d = tid;
    float xv = float(x[d]);
    for (uint m = 0; m < hc; m++) {
        float acc = post[m] * xv;
        for (uint k = 0; k < hc; k++) {
            // comb[k][m] = comb[k*hc + m]
            acc += comb[k * hc + m] * float(residual[k * dim + d]);
        }
        out_residual[m * dim + d] = (half)acc;
    }
}

kernel void mla_sdpa_decode_f16in_f16out(
    device const half* q        [[buffer(0)]],
    device const half* kv       [[buffer(1)]],
    device const float*  sinks    [[buffer(2)]],
    device half*       out      [[buffer(3)]],
    constant uint&       n_heads  [[buffer(4)]],
    constant uint&       head_dim [[buffer(5)]],
    constant uint&       n_kv     [[buffer(6)]],
    constant float&      scale    [[buffer(7)]],
    uint  head [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tg   [[threads_per_threadgroup]]
) {
    if (head >= n_heads) return;
    threadgroup float red[32];
    device const half* qh = q + (uint64_t)head * head_dim;

    threadgroup float t_m;
    threadgroup float t_s;
    threadgroup float t_score;
    if (lid == 0) { t_m = -INFINITY; t_s = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[8];
    uint n_slots = (head_dim + tg - 1) / tg;
    for (uint i = 0; i < n_slots; i++) acc[i] = 0.0f;

    for (uint k = 0; k < n_kv; k++) {
        device const half* kvk = kv + (uint64_t)k * head_dim;
        float partial = 0.0f;
        for (uint d = lid; d < head_dim; d += tg) partial += float(qh[d]) * float(kvk[d]);
        float dot = simd_sum(partial);
        uint lane = lid % 32, sg = lid / 32;
        if (lane == 0) red[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float tot = 0.0f;
            uint n_sg = (tg + 31) / 32;
            for (uint g = 0; g < n_sg; g++) tot += red[g];
            t_score = tot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = t_score;
        float m_old = t_m;
        float m_new = max(m_old, score);
        float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
        float p = exp(score - m_new);
        for (uint i = 0; i < n_slots; i++) {
            uint d = lid + i * tg;
            if (d < head_dim) acc[i] = acc[i] * corr + p * float(kvk[d]);
        }
        if (lid == 0) { t_s = t_s * corr + p; t_m = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Sink
    if (lid == 0) {
        float sink = sinks[head];
        float m_old = t_m;
        float m_new = max(m_old, sink);
        float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
        t_s = t_s * corr + exp(sink - m_new);
        t_m = m_new;
        red[0] = corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float corr_final = red[0];
    float inv_s = (t_s == 0.0f) ? 0.0f : 1.0f / t_s;
    device half* oh = out + (uint64_t)head * head_dim;
    for (uint i = 0; i < n_slots; i++) {
        uint d = lid + i * tg;
        if (d < head_dim) oh[d] = (half)((acc[i] * corr_final) * inv_s);
    }
}


kernel void dequant_matvec_affine_f16in_f32out(
    device const uint32_t* W_packed [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const half*   x        [[buffer(3)]],  // half input
    device float*          out      [[buffer(4)]],  // float output
    constant uint&         out_dim  [[buffer(5)]],
    constant uint&         in_dim   [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const float*    sc = scales   + tid * num_groups;
    device const float*    bi = biases   + tid * num_groups;
    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = sc[g], bias = bi[g];
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            for (uint i = 0; i < 8; i++) {
                float nib = (float)((pw >> (i * 4)) & 0xF);
                acc += (scale * nib + bias) * float(x[bx + p * 8 + i]);
            }
        }
    }
    out[tid] = acc;
}


// ============================================================================
// MoE f16-precision kernels (ds4-style end-to-end chain)
// ============================================================================

// fused_gate_up_swiglu_f16: same as fused_gate_up_swiglu but with f16 input/output.
// All computation is in f32, but operands are cast from half on read and
// truncated to half on write.
kernel void fused_gate_up_swiglu_f16(
    device const uint32_t* gate_W   [[buffer(0)]],
    device const uint8_t*  gate_s   [[buffer(1)]],
    device const uint32_t* up_W     [[buffer(2)]],
    device const uint8_t*  up_s     [[buffer(3)]],
    device const half*     x        [[buffer(4)]],
    device half*           out      [[buffer(5)]],
    constant uint&         out_dim  [[buffer(6)]],
    constant uint&         in_dim   [[buffer(7)]],
    constant uint&         group_size [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* g_row = gate_W + tid * packed_cols;
    device const uint8_t*  g_s   = gate_s + tid * num_groups;
    device const uint32_t* u_row = up_W   + tid * packed_cols;
    device const uint8_t*  u_s   = up_s   + tid * num_groups;

    float gate_val = 0.0f, up_val = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float gsf = exp2((float)g_s[g] - 127.0f);
        float usf = exp2((float)u_s[g] - 127.0f);
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gpw = g_row[bp + p], upw = u_row[bp + p];
            for (uint i = 0; i < 8; i++) {
                float g_w = NIBBLE_TO_FLOAT[(gpw >> (i * 4)) & 0xF] * gsf;
                float u_w = NIBBLE_TO_FLOAT[(upw >> (i * 4)) & 0xF] * usf;
                float xv = float(x[bx + p * 8 + i]);
                gate_val += g_w * xv;
                up_val   += u_w * xv;
            }
        }
    }
    const float limit = 10.0f;
    float g_c = min(gate_val, limit);
    float u_c = min(max(up_val, -limit), limit);
    float act = g_c / (1.0f + exp(-g_c));
    out[tid] = (half)(act * u_c);
}

// dequant_matvec_4bit_f16out: same as dequant_matvec_4bit but output is half.
kernel void dequant_matvec_4bit_f16out(
    device const uint32_t* W_packed [[buffer(0)]],
    device const uint8_t*  scales   [[buffer(1)]],
    device const float*    x        [[buffer(2)]],
    device half*           out      [[buffer(3)]],
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         in_dim   [[buffer(5)]],
    constant uint&         group_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const uint8_t*  sc = scales   + tid * num_groups;

    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float sf = exp2((float)sc[g] - 127.0f);
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            for (uint i = 0; i < 8; i++) {
                float w_val = NIBBLE_TO_FLOAT[(pw >> (i * 4)) & 0xF] * sf;
                acc += w_val * x[bx + p * 8 + i];
            }
        }
    }
    out[tid] = (half)acc;
}

// dequant_matvec_4bit_f16in_f32out: same as dequant_matvec_4bit but input is half.
kernel void dequant_matvec_4bit_f16in_f32out(
    device const uint32_t* W_packed [[buffer(0)]],
    device const uint8_t*  scales   [[buffer(1)]],
    device const half*     x        [[buffer(2)]],
    device float*          out      [[buffer(3)]],
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         in_dim   [[buffer(5)]],
    constant uint&         group_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* wr = W_packed + tid * packed_cols;
    device const uint8_t*  sc = scales   + tid * num_groups;

    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float sf = exp2((float)sc[g] - 127.0f);
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t pw = wr[bp + p];
            for (uint i = 0; i < 8; i++) {
                float w_val = NIBBLE_TO_FLOAT[(pw >> (i * 4)) & 0xF] * sf;
                acc += w_val * float(x[bx + p * 8 + i]);
            }
        }
    }
    out[tid] = acc;
}

// moe_combine_f16: weighted sum of K expert outputs + residual, all f16.
kernel void moe_combine_f16(
    device const half*  expert_outs [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device const half*  residual    [[buffer(2)]],
    device half*        output      [[buffer(3)]],
    constant uint&      K           [[buffer(4)]],
    constant uint&      hidden_dim  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= hidden_dim) return;
    float sum = float(residual[tid]);
    for (uint k = 0; k < K; k++) {
        sum += float(expert_outs[k * hidden_dim + tid]) * weights[k];
    }
    output[tid] = (half)sum;
}

// fused_gate_up_swiglu_f32in_f16out: f32 input, f16 output.
// Used when the preceding RMSNorm is still f32 but we want f16 expert intermediates.
kernel void fused_gate_up_swiglu_f32in_f16out(
    device const uint32_t* gate_W   [[buffer(0)]],
    device const uint8_t*  gate_s   [[buffer(1)]],
    device const uint32_t* up_W     [[buffer(2)]],
    device const uint8_t*  up_s     [[buffer(3)]],
    device const float*    x        [[buffer(4)]],
    device half*           out      [[buffer(5)]],
    constant uint&         out_dim  [[buffer(6)]],
    constant uint&         in_dim   [[buffer(7)]],
    constant uint&         group_size [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* g_row = gate_W + tid * packed_cols;
    device const uint8_t*  g_s   = gate_s + tid * num_groups;
    device const uint32_t* u_row = up_W   + tid * packed_cols;
    device const uint8_t*  u_s   = up_s   + tid * num_groups;

    float gate_val = 0.0f, up_val = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float gsf = exp2((float)g_s[g] - 127.0f);
        float usf = exp2((float)u_s[g] - 127.0f);
        uint bp = g * packed_per_group;
        uint bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gpw = g_row[bp + p], upw = u_row[bp + p];
            for (uint i = 0; i < 8; i++) {
                float g_w = NIBBLE_TO_FLOAT[(gpw >> (i * 4)) & 0xF] * gsf;
                float u_w = NIBBLE_TO_FLOAT[(upw >> (i * 4)) & 0xF] * usf;
                float xv = x[bx + p * 8 + i];
                gate_val += g_w * xv;
                up_val   += u_w * xv;
            }
        }
    }
    const float limit = 10.0f;
    float g_c = min(gate_val, limit);
    float u_c = min(max(up_val, -limit), limit);
    float act = g_c / (1.0f + exp(-g_c));
    out[tid] = (half)(act * u_c);
}

// ===========================================================================

// ===========================================================================
// mla_sdpa_prefill_bfloat: Batch prefill SDPA for MLA (corrected v2).
//
// One threadgroup = 32 threads (1 simdgroup) processes up to NQ=8 query tokens
// for one attention head. Matches ds4/MLX simdgroup reduction order.
//
// Thread ownership:
//   - Thread lid owns head_dim slots: lid, lid+32, lid+64, ..., lid+480 (16 slots total)
//   - acc[qi][slot]  maps to output dim = lid + slot*32
//
// Online softmax:
//   - t_m[qi], t_s[qi], t_score[qi], t_corr[qi], t_p[qi] are per-query threadgroup state
//   - lid==0 updates t_m, t_s after each simd_sum
//   - t_corr, t_p are broadcast back via threadgroup mem so all threads can rescale acc
//
// Inputs:
//   q    : [n_tok, n_heads, HEAD_DIM] bfloat  (HEAD_DIM = 512)
//   kv   : [n_tok, HEAD_DIM]          bfloat  (MQA: single KV head broadcast)
//   sinks: [n_heads]                  float   (per-head sink logit)
//   out  : [n_tok, n_heads, HEAD_DIM] bfloat
//
// Causal: token qi attends to kv[0 .. q_base+qi] inclusive.
// Grid: [ceil(n_tok/NQ), n_heads, 1] threadgroups, 32 threads each.
// ===========================================================================

#define PREFILL_NQ  8    // query tokens per threadgroup
#define PREFILL_HD  512  // HEAD_DIM
#define PREFILL_SLOTS 16 // HEAD_DIM / 32 = slots per thread

kernel void mla_sdpa_prefill_bfloat(
    device const bfloat* q      [[buffer(0)]],  // [n_tok, n_heads, PREFILL_HD]
    device const bfloat* kv     [[buffer(1)]],  // [n_tok, PREFILL_HD]
    device const float*  sinks  [[buffer(2)]],  // [n_heads]
    device       bfloat* out    [[buffer(3)]],  // [n_tok, n_heads, PREFILL_HD]
    constant uint&       n_tok  [[buffer(4)]],
    constant uint&       n_heads[[buffer(5)]],
    constant float&      scale  [[buffer(6)]],
    // 1D flat dispatch: tg_id = q_block * n_heads + head
    uint  tg_id [[threadgroup_position_in_grid]],
    uint  lid   [[thread_position_in_threadgroup]]
) {
    // Decode q_block and head from flat index
    const uint q_block = tg_id / n_heads;
    const uint head    = tg_id % n_heads;
    const uint q_base  = q_block * PREFILL_NQ;

    if (q_base >= n_tok || head >= n_heads) return;

    const uint nq = min((uint)PREFILL_NQ, n_tok - q_base);
    const uint q_stride = n_heads * PREFILL_HD; // stride between consecutive token rows in q

    // --- Threadgroup state for online softmax ---
    // All updated by lid==0 only; broadcast via t_corr/t_p for acc rescaling.
    threadgroup float t_m    [PREFILL_NQ];  // running max per query
    threadgroup float t_s    [PREFILL_NQ];  // running sum-of-exp per query
    threadgroup float t_score[PREFILL_NQ];  // current dot product score
    threadgroup float t_corr [PREFILL_NQ];  // correction factor (exp(m_old - m_new))
    threadgroup float t_p    [PREFILL_NQ];  // exp(score - m_new)

    if (lid < PREFILL_NQ) {
        t_m[lid] = -INFINITY;
        t_s[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Per-query output accumulator ---
    // Thread lid owns dims: lid + slot*32, for slot in [0, PREFILL_SLOTS)
    float acc[PREFILL_NQ][PREFILL_SLOTS];
    for (uint qi = 0; qi < PREFILL_NQ; qi++)
        for (uint s = 0; s < PREFILL_SLOTS; s++)
            acc[qi][s] = 0.0f;

    // Iterate over all KV positions (causal: up to last token in this block)
    const uint max_kv = q_base + nq;

    for (uint k = 0; k < max_kv; k++) {
        const device bfloat* kv_k = kv + (size_t)k * PREFILL_HD;

        // --- Compute dot products Q[qi] · kv[k] for all causally-allowed qi ---
        for (uint qi = 0; qi < nq; qi++) {
            if (k > q_base + qi) {
                // Causal masking: this kv position is beyond token qi's horizon
                if (lid == 0) { t_score[qi] = -INFINITY; }
                continue;
            }

            const device bfloat* q_qi = q + (size_t)(q_base + qi) * q_stride + head * PREFILL_HD;

            // Cooperative dot: thread lid accumulates dims lid, lid+32, ..., lid+480
            float partial = 0.0f;
            for (uint s = 0; s < PREFILL_SLOTS; s++) {
                uint d = lid + s * 32;
                partial += float(q_qi[d]) * float(kv_k[d]);
            }
            // Reduce within simdgroup → every thread in simdgroup gets the full dot
            float dot = simd_sum(partial);

            if (lid == 0) {
                t_score[qi] = dot * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Online softmax update + acc rescale ---
        // lid==0 computes new m, corr, p and writes them; all threads read them.
        if (lid == 0) {
            for (uint qi = 0; qi < nq; qi++) {
                float score = t_score[qi];
                if (score == -INFINITY) {
                    t_corr[qi] = 1.0f;
                    t_p[qi]    = 0.0f;
                    continue;
                }
                float m_old = t_m[qi];
                float m_new = max(m_old, score);
                float corr  = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
                float p     = exp(score - m_new);
                t_m[qi] = m_new;
                t_s[qi] = t_s[qi] * corr + p;
                t_corr[qi] = corr;
                t_p[qi]    = p;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // All threads rescale their acc slots and accumulate new weighted kv
        for (uint qi = 0; qi < nq; qi++) {
            float corr = t_corr[qi];
            float p    = t_p[qi];
            if (p == 0.0f && corr == 1.0f) continue; // masked position
            for (uint s = 0; s < PREFILL_SLOTS; s++) {
                uint d = lid + s * 32;
                acc[qi][s] = acc[qi][s] * corr + p * float(kv_k[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Fold per-head sink into denominator (sink has zero-valued KV vector) ---
    {
        float sink = sinks[head];
        if (lid == 0) {
            for (uint qi = 0; qi < nq; qi++) {
                float m_old = t_m[qi];
                float m_new = max(m_old, sink);
                float corr  = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
                // sink adds to denominator only (KV value is zero)
                t_s[qi] = t_s[qi] * corr + exp(sink - m_new);
                t_m[qi] = m_new;
                t_corr[qi] = corr;
                t_p[qi]    = 0.0f; // sink contribution to acc is zero
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Rescale acc by corr due to updated max
        for (uint qi = 0; qi < nq; qi++) {
            float corr = t_corr[qi];
            if (corr == 1.0f) continue;
            for (uint s = 0; s < PREFILL_SLOTS; s++) {
                acc[qi][s] *= corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Write normalized output ---
    for (uint qi = 0; qi < nq; qi++) {
        float inv_s = (t_s[qi] > 0.0f) ? (1.0f / t_s[qi]) : 0.0f;
        device bfloat* out_qi = out + (size_t)(q_base + qi) * n_heads * PREFILL_HD + head * PREFILL_HD;
        for (uint s = 0; s < PREFILL_SLOTS; s++) {
            uint d = lid + s * 32;
            out_qi[d] = (bfloat)(acc[qi][s] * inv_s);
        }
    }
}

#undef PREFILL_NQ
#undef PREFILL_HD
#undef PREFILL_SLOTS

// ============================================================================
// moe_route_gpu: unified GPU MoE routing (replaces cpu_moe_route).
// Single dispatch: sqrtsoftplus + SMELT penalty + bitonic top-6 + L1-normalize.
// 256 threads x 1 threadgroup, one thread per expert.
// ============================================================================
kernel void moe_route_gpu(
    device const float*   logits        [[buffer(0)]],  // [256] gate logits (f32)
    device const float*   bias          [[buffer(1)]],  // [256] e_score_correction_bias
    device const uint8_t* cached        [[buffer(2)]],  // [256] 1=in SMELT pool
    device int32_t*       selected      [[buffer(3)]],  // [6]  output: top-6 expert IDs
    device float*         weights       [[buffer(4)]],  // [6]  output: routing weights
    constant float&       smelt_penalty [[buffer(5)]],
    constant uint&        has_bias      [[buffer(6)]],
    constant uint&        has_smelt     [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint N = 256u, K = 6u;
    const float ROUTE_SCALE = 1.5f;

    // sqrtsoftplus score (matches MLX DSV4Gate scoring_func=sqrtsoftplus)
    float l = logits[tid];
    float sp = l > 0.0f ? l + log(1.0f + exp(-l)) : log(1.0f + exp(l));
    float score = sqrt(sp);

    // biased score for top-K selection only
    float biased = score;
    if (has_bias)  biased += bias[tid];
    if (has_smelt && !cached[tid]) biased -= smelt_penalty;

    threadgroup float   tg_scores[256];
    threadgroup float   tg_raw[256];
    threadgroup int32_t tg_idx[256];
    threadgroup float   ksum[8];

    tg_scores[tid] = biased;
    tg_raw[tid]    = score;
    tg_idx[tid]    = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort descending
    for (uint k = 2u; k <= N; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            uint other = tid ^ j;
            if (other > tid) {
                int32_t ai = tg_idx[tid], bi2 = tg_idx[other];
                float sa = tg_scores[(uint)ai], sb2 = tg_scores[(uint)bi2];
                bool ascending = ((tid & k) == 0u);
                if (ascending ? sa < sb2 : sa > sb2) {
                    tg_idx[tid] = bi2; tg_idx[other] = ai;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid < K) {
        selected[tid] = tg_idx[tid];
        ksum[tid] = tg_raw[(uint)tg_idx[tid]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < K; i++) s += ksum[i];
        s = max(s, 1e-20f);
        for (uint i = 0; i < K; i++) ksum[i] = ksum[i] / s * ROUTE_SCALE;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < K) weights[tid] = ksum[tid];
}

// ============================================================================
// LUT-based affine 4-bit kernels (same formula as MXFP4, group_size=64)
// These use the same NIBBLE_TO_FLOAT LUT and E8M0 scales as MXFP4, but with
// group_size=64 instead of 32. Format: value = LUT[nibble] * exp2(scale - 127)
// No biases — scale-only quantization with non-uniform LUT.
// ============================================================================

// fused_gate_up_swiglu_v2_lut — LUT-based gate+up for affine 4-bit (gs=64)
kernel void fused_gate_up_swiglu_v2_lut(
    device const uint32_t* gate_W    [[buffer(0)]],
    device const uint8_t*  gate_s    [[buffer(1)]],  // E8M0 scales
    device const uint32_t* up_W      [[buffer(2)]],
    device const uint8_t*  up_s      [[buffer(3)]],  // E8M0 scales
    device const float*    x         [[buffer(4)]],
    device float*          out       [[buffer(5)]],
    constant uint&         out_dim   [[buffer(6)]],
    constant uint&         in_dim    [[buffer(7)]],
    constant uint&         group_size [[buffer(8)]],
    threadgroup float*     shmem     [[threadgroup(0)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 2, NSG = 4, NW = 32, NQ = 4, TPG = 8;

    const uint num_groups = in_dim / group_size;
    const uint packed_per_group = group_size / 8;  // = 8 for gs=64
    const uint packed_cols = in_dim / 8;

    const int row0 = (int)tgpig.x * NR0;
    const short ix = tiisg / TPG, il = tiisg % TPG;
    const int g0 = (int)sgitg * NQ + (int)ix;

    device const uint32_t *gr[NR0], *ur[NR0];
    device const uint8_t  *gs[NR0], *us[NR0];
    for (short row = 0; row < NR0; row++) {
        int r = row0 + row;
        if (r < (int)out_dim) {
            gr[row] = gate_W + r * packed_cols;
            gs[row] = gate_s + r * num_groups;
            ur[row] = up_W   + r * packed_cols;
            us[row] = up_s   + r * num_groups;
        }
    }

    float gate_sum[NR0] = { 0.0f };
    float up_sum[NR0]   = { 0.0f };

    for (int gg = g0; gg < (int)num_groups; gg += NSG * NQ) {
        uint xb = (uint)gg * group_size + (uint)il * 8;
        float xv0 = x[xb+0], xv1 = x[xb+1], xv2 = x[xb+2], xv3 = x[xb+3];
        float xv4 = x[xb+4], xv5 = x[xb+5], xv6 = x[xb+6], xv7 = x[xb+7];

        for (short row = 0; row < NR0; row++) {
            int r = row0 + row;
            if (r >= (int)out_dim) continue;

            float gsf = exp2((float)gs[row][gg] - 127.0f);
            float usf = exp2((float)us[row][gg] - 127.0f);

            uint gpw = gr[row][gg * packed_per_group + (uint)il];
            uint upw = ur[row][gg * packed_per_group + (uint)il];

            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>> 0)&0xF] * gsf * xv0;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>> 4)&0xF] * gsf * xv1;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>> 8)&0xF] * gsf * xv2;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>>12)&0xF] * gsf * xv3;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>>16)&0xF] * gsf * xv4;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>>20)&0xF] * gsf * xv5;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>>24)&0xF] * gsf * xv6;
            gate_sum[row] += NIBBLE_TO_FLOAT[(gpw>>28)&0xF] * gsf * xv7;

            up_sum[row]   += NIBBLE_TO_FLOAT[(upw>> 0)&0xF] * usf * xv0;
            up_sum[row]   += NIBBLE_TO_FLOAT[(upw>> 4)&0xF] * usf * xv1;
            up_sum[row]   += NIBBLE_TO_FLOAT[(upw>> 8)&0xF] * usf * xv2;
            up_sum[row]   += NIBBLE_TO_FLOAT[(upw>>12)&0xF] * usf * xv3;
            up_sum[row]   += NIBBLE_TO_FLOAT[(upw>>16)&0xF] * usf * xv4;
            up_sum[row]   += NIBBLE_TO_FLOAT[(upw>>20)&0xF] * usf * xv5;
            up_sum[row]   += NIBBLE_TO_FLOAT[(upw>>24)&0xF] * usf * xv6;
            up_sum[row]   += NIBBLE_TO_FLOAT[(upw>>28)&0xF] * usf * xv7;
        }
    }

    threadgroup float *shmem_f32[NR0*2];
    for (short row = 0; row < NR0; row++) {
        shmem_f32[row*2]   = shmem + NW * (row*2);
        shmem_f32[row*2+1] = shmem + NW * (row*2+1);
        if (sgitg == 0) {
            shmem_f32[row*2][tiisg] = 0.0f;
            shmem_f32[row*2+1][tiisg] = 0.0f;
        }
        gate_sum[row] = simd_sum(gate_sum[row]);
        up_sum[row]   = simd_sum(up_sum[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; row++) {
        if (tiisg == 0) {
            shmem_f32[row*2][sgitg]   = gate_sum[row];
            shmem_f32[row*2+1][sgitg] = up_sum[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float limit = 10.0f;
    for (short row = 0; row < NR0; row++) {
        const int d = row0 + row;
        if (d >= (int)out_dim) continue;
        float gv = simd_sum(shmem_f32[row*2][tiisg]);
        float uv = simd_sum(shmem_f32[row*2+1][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float g_c = min(gv, limit);
            float u_c = min(max(uv, -limit), limit);
            out[d] = (bfloat)((g_c / (1.0f + exp(-g_c))) * u_c);
        }
    }
}

// dequant_matvec_4bit_lut — LUT-based down_proj for affine 4-bit (gs=64)
kernel void dequant_matvec_4bit_lut(
    device const uint32_t* W_packed [[buffer(0)]],
    device const uint8_t*  scales   [[buffer(1)]],  // E8M0 scales
    device const float*    x        [[buffer(2)]],
    device float*          out      [[buffer(3)]],
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         in_dim   [[buffer(5)]],
    constant uint&         group_size [[buffer(6)]],
    threadgroup float*     shmem    [[threadgroup(0)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 2, NSG = 4, NW = 32;
    const short TPG = 8;  // threads per group: packed_per_group = group_size/8 = 8 for gs=64
    const short NQ  = 4;  // groups per SIMD group: NW/TPG = 32/8 = 4

    const uint num_groups = in_dim / group_size;
    const uint packed_per_group = group_size / 8;
    const uint packed_cols = in_dim / 8;

    const int row0 = (int)tgpig.x * NR0;
    const short ix = tiisg / TPG, il = tiisg % TPG;
    const int g0 = (int)sgitg * NQ + (int)ix;

    device const uint32_t *wr[NR0];
    device const uint8_t  *sr[NR0];
    for (short row = 0; row < NR0; row++) {
        int r = row0 + row;
        if (r < (int)out_dim) {
            wr[row] = W_packed + r * packed_cols;
            sr[row] = scales   + r * num_groups;
        }
    }

    float sumf[NR0] = { 0.0f };

    for (int gg = g0; gg < (int)num_groups; gg += NSG * NQ) {
        uint xb = (uint)gg * group_size + (uint)il * 8;
        float xv0 = x[xb+0], xv1 = x[xb+1], xv2 = x[xb+2], xv3 = x[xb+3];
        float xv4 = x[xb+4], xv5 = x[xb+5], xv6 = x[xb+6], xv7 = x[xb+7];

        for (short row = 0; row < NR0; row++) {
            int r = row0 + row;
            if (r >= (int)out_dim) continue;

            float sf = exp2((float)sr[row][gg] - 127.0f);
            uint32_t pw = wr[row][gg * packed_per_group + (uint)il];

            sumf[row] += NIBBLE_TO_FLOAT[(pw>> 0)&0xF] * sf * xv0;
            sumf[row] += NIBBLE_TO_FLOAT[(pw>> 4)&0xF] * sf * xv1;
            sumf[row] += NIBBLE_TO_FLOAT[(pw>> 8)&0xF] * sf * xv2;
            sumf[row] += NIBBLE_TO_FLOAT[(pw>>12)&0xF] * sf * xv3;
            sumf[row] += NIBBLE_TO_FLOAT[(pw>>16)&0xF] * sf * xv4;
            sumf[row] += NIBBLE_TO_FLOAT[(pw>>20)&0xF] * sf * xv5;
            sumf[row] += NIBBLE_TO_FLOAT[(pw>>24)&0xF] * sf * xv6;
            sumf[row] += NIBBLE_TO_FLOAT[(pw>>28)&0xF] * sf * xv7;
        }
    }

    threadgroup float *shmem_f32[NR0];
    for (short row = 0; row < NR0; row++) {
        shmem_f32[row] = shmem + NW * row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; row++) {
        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; row++) {
        const int d = row0 + row;
        if (d >= (int)out_dim) continue;
        float tot = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) out[d] = (bfloat)tot;
    }
}

// ============================================================================
// INT8 + E8M0 Scale dequantized matvec (for DSpark MTP experts)
// ============================================================================
//
// Weight format: int8 [out_dim, in_dim], row-major.
// Scale format: uint8 E8M0 [out_dim, in_dim / block_size], row-major.
// Dequant: w_float = (float)int8_val * 2^(scale_byte - 127)
//        = (float)int8_val * as_type<float>((uint)scale_byte << 23)
//
// Dispatch: threadgroups = out_dim / ROWS_PER_TG, threads per TG = 256 (8 simdgroups × 32).
// Each simdgroup handles one output row. 32 threads cooperatively reduce in_dim.
// x_shared: input vector cached in threadgroup memory (up to 4096 floats = 16KB).
//
// block_size for MTP experts: 16 (weight_cols / scale_cols = 2048/128 or 1024/64).

kernel void dequant_matvec_int8_e8m0(
    device const int8_t*   W        [[buffer(0)]],   // [out_dim, in_dim] int8
    device const uint8_t*  scales   [[buffer(1)]],   // [out_dim, in_dim/block_size] uint8 E8M0
    device const float*    x        [[buffer(2)]],   // [in_dim] float32
    device float*          out      [[buffer(3)]],   // [out_dim] float32
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         in_dim   [[buffer(5)]],
    constant uint&         block_size [[buffer(6)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint ROWS_PER_TG = 8;
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint num_groups = in_dim / block_size;

    // Cache x in threadgroup shared memory
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const int8_t*  wr = W      + row * in_dim;
    device const uint8_t* sc = scales  + row * num_groups;

    float acc = 0.0f;
    // Each simd lane processes a subset of groups
    for (uint g = simd_lane; g < num_groups; g += 32) {
        // E8M0 scale decode via bit-shift: 2^(byte - 127) = as_type<float>((uint)byte << 23)
        float sf = as_type<float>((uint)sc[g] << 23);
        uint base_idx = g * block_size;

        // Unroll by 4 for ILP
        float local_acc = 0.0f;
        for (uint i = 0; i < block_size; i += 4) {
            uint idx = base_idx + i;
            local_acc += (float)wr[idx + 0] * x_shared[idx + 0];
            local_acc += (float)wr[idx + 1] * x_shared[idx + 1];
            local_acc += (float)wr[idx + 2] * x_shared[idx + 2];
            local_acc += (float)wr[idx + 3] * x_shared[idx + 3];
        }
        acc += local_acc * sf;
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// Fused gate + up + SwiGLU for INT8 + E8M0 (DSpark MTP experts).
// Same logic as fused_gate_up_swiglu but reads int8 weights + e8m0 scales.
// block_size = 16 for MTP experts.
// Uses ROWS_PER_TG=8, 256 threads, x_shared for input caching.

kernel void fused_gate_up_swiglu_int8_e8m0(
    device const int8_t*   gate_W   [[buffer(0)]],   // [out_dim, in_dim] int8
    device const uint8_t*  gate_s   [[buffer(1)]],   // [out_dim, in_dim/bs] e8m0
    device const int8_t*   up_W     [[buffer(2)]],   // [out_dim, in_dim] int8
    device const uint8_t*  up_s     [[buffer(3)]],   // [out_dim, in_dim/bs] e8m0
    device const float*    x        [[buffer(4)]],   // [in_dim] float32
    device float*          out      [[buffer(5)]],   // [out_dim] float32 (SwiGLU output)
    constant uint&         out_dim  [[buffer(6)]],
    constant uint&         in_dim   [[buffer(7)]],
    constant uint&         block_size [[buffer(8)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint ROWS_PER_TG = 8;
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint num_groups = in_dim / block_size;

    // Cache x in shared memory
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const int8_t*  g_row = gate_W + row * in_dim;
    device const uint8_t* g_s   = gate_s + row * num_groups;
    device const int8_t*  u_row = up_W   + row * in_dim;
    device const uint8_t* u_s   = up_s   + row * num_groups;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint g = simd_lane; g < num_groups; g += 32) {
        float gsf = as_type<float>((uint)g_s[g] << 23);
        float usf = as_type<float>((uint)u_s[g] << 23);
        uint base_idx = g * block_size;

        float g_local = 0.0f, u_local = 0.0f;
        for (uint i = 0; i < block_size; i += 4) {
            uint idx = base_idx + i;
            float x0 = x_shared[idx + 0], x1 = x_shared[idx + 1];
            float x2 = x_shared[idx + 2], x3 = x_shared[idx + 3];
            g_local += (float)g_row[idx + 0] * x0 + (float)g_row[idx + 1] * x1
                     + (float)g_row[idx + 2] * x2 + (float)g_row[idx + 3] * x3;
            u_local += (float)u_row[idx + 0] * x0 + (float)u_row[idx + 1] * x1
                     + (float)u_row[idx + 2] * x2 + (float)u_row[idx + 3] * x3;
        }
        gate_acc += g_local * gsf;
        up_acc += u_local * usf;
    }

    float gate_val = simd_sum(gate_acc);
    float up_val = simd_sum(up_acc);

    if (simd_lane == 0) {
        // Limited SwiGLU (swiglu_limit=10)
        const float limit = 10.0f;
        float g_c = min(gate_val, limit);
        float u_c = min(max(up_val, -limit), limit);
        float act = g_c / (1.0f + exp(-g_c));  // silu(gate)
        out[row] = act * u_c;
    }
}
