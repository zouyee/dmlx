/// GLM-5.3 full-attention block: NoPE MLA with absorbed projections plus the
/// DSA (DeepSeek Sparse Attention) indexer, and the dense SwiGLU MLP used by
/// the first `first_k_dense_replace` layers.
///
/// Faithful port of mlx-vlm `Glm5NextAttention` / `Glm5NextIndexer` /
/// `Glm5NextMLP` (reference: /tmp/glm5_next_ref/language.py). No RoPE anywhere
/// (qk_rope_head_dim = 0, mla_use_nope = true).
///
/// Numerics notes:
///   - All projections run through fused `quantizedMatmul` on the packed
///     checkpoint tensors (same approach as the KDA module; layer-0 oracle
///     cosine 0.99996 was achieved this way).
///   - The per-head `embed_q` / `unembed_out` MultiLinear weights are
///     dequantized once at load; the absorbed matmuls then run as plain
///     broadcast matmuls, equivalent to mx.quantized_matmul up to reduction
///     order.
///   - Attention scores/softmax run in f32 (oracle uses the fused bf16 SDPA
///     kernel; f32 is strictly more accurate and lands well within the 0.999
///     oracle gate).
///
/// Cache: `Glm53LatentCache` carries the latent KV [B,1,N,512] (standard
/// concat-append semantics) and `Glm53PoolCache` the kpool window accumulator
/// (PoolingCache semantics for the single-row, no-left-padding,
/// non-speculative case). The Task-8 oracle harness only exercises the
/// cache=None prefill path. Planned for Task 9 (model assembly): the cached
/// decode indexer forward that consumes Glm53PoolCache (accumulate_windows →
/// compress → pool selection against the growing pool cache), mirroring the
/// oracle's cache!=None branch.
const std = @import("std");
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const math_mod = @import("mlx").math;
const sort_mod = @import("mlx").sort;
const comparison_mod = @import("mlx").comparison;
const creation_mod = @import("mlx").creation;
const fast_mod = @import("mlx").fast;
const activations = @import("mlx").activations;
const quantize_mod = @import("mlx").quantize;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const QuantConfig = quantize_mod.QuantConfig;

/// clip(x, lo, hi) via minimum/maximum (the mlx_z math.clip binding is broken
/// for optional bounds — it passes null where the C API wants a value).
fn clipMinMax(ctx: EagerContext, x: Array, lo: Array, hi: Array) !Array {
    const lower = try math_mod.maximum(ctx, x, lo);
    defer lower.deinit();
    return math_mod.minimum(ctx, lower, hi);
}

/// Decode-block threshold from mlx_vlm.models.linear: sequences of at most
/// this many tokens take the per-query gather+SDPA loop (which is also what
/// the oracle's 7-token prefill exercises).
pub const DECODE_BLOCK_SIZE = 8;

// ============================================================
// Quantized linear helper
// ============================================================

/// Packed affine-quantized Linear (MLX layout: weight [out, in] + scales +
/// biases). Arrays are borrowed — the caller owns them.
pub const QuantLinear = struct {
    w: Array,
    scales: Array,
    biases: Array,
    cfg: QuantConfig,

    /// x @ dequantize(w)^T via the fused quantized matmul.
    pub fn apply(self: *const QuantLinear, ctx: EagerContext, x: Array) !Array {
        const qw = quantize_mod.QuantizedWeight{
            .data = self.w,
            .scales = self.scales,
            .biases = self.biases,
            .config = self.cfg,
            .original_shape = &[_]i32{},
        };
        return quantize_mod.quantizedMatmul(ctx, x, qw, true);
    }
};

// ============================================================
// Dense MLP (first_k_dense_replace layers)
// ============================================================

pub const Glm53Mlp = struct {
    ctx: EagerContext,
    gate_up_proj: QuantLinear,
    down_proj: QuantLinear,
    swiglu_limit: f32,

    /// x: [B, L, hidden] -> [B, L, hidden]
    /// gate, up = split(gate_up_proj(x)); down(silu(min(gate, limit)) * clip(up, ±limit))
    pub fn forward(self: *const Glm53Mlp, x: Array) !Array {
        const ctx = self.ctx;
        const gu = try self.gate_up_proj.apply(ctx, x);
        defer gu.deinit();
        const sh = gu.shape();
        const b = sh[0];
        const l = sh[1];
        const inter = @divTrunc(sh[2], 2);

        const gate = try ops.slice(ctx, gu, &[_]i32{ 0, 0, 0 }, &[_]i32{ b, l, inter }, &[_]i32{ 1, 1, 1 });
        defer gate.deinit();
        const up = try ops.slice(ctx, gu, &[_]i32{ 0, 0, inter }, &[_]i32{ b, l, 2 * inter }, &[_]i32{ 1, 1, 1 });
        defer up.deinit();

        const limit_pos = try ops.scalarF32(ctx, self.swiglu_limit);
        defer limit_pos.deinit();
        const limit_neg = try ops.scalarF32(ctx, -self.swiglu_limit);
        defer limit_neg.deinit();

        const gate_c = try math_mod.minimum(ctx, gate, limit_pos);
        defer gate_c.deinit();
        const up_c = try clipMinMax(ctx, up, limit_neg, limit_pos);
        defer up_c.deinit();
        const act = try activations.silu(ctx, gate_c);
        defer act.deinit();
        const gated = try ops.multiply(ctx, act, up_c);
        defer gated.deinit();
        return self.down_proj.apply(ctx, gated);
    }
};

// ============================================================
// Indexer (DSA top-k selection)
// ============================================================

pub const Glm53Indexer = struct {
    ctx: EagerContext,
    n_heads: i32, // 32
    head_dim: i32, // 128
    index_topk: i32, // 2048
    kpool: i32, // 4
    always_select_tail: bool,

    wq_b: QuantLinear, // [n_heads*head_dim, q_lora_rank]
    wk: QuantLinear, // [head_dim, hidden]
    weights_proj: QuantLinear, // [n_heads, hidden]
    k_norm_w: Array, // [head_dim]
    k_norm_b: Array, // [head_dim]
    /// [kpool, head_dim] f32 positional additive term for pool compression.
    ape: Array,
    /// Transposed copy of the checkpoint's [head_dim, hidden] f32 gate
    /// projection, stored as [hidden, head_dim] for `x_f32 @ compress_gate_t`.
    compress_gate_t: Array,

    const k_norm_eps: f32 = 1e-6;

    /// values: [B, L, rest...]; indices: [B, ...] i32. Gathers along the L
    /// axis with a per-batch offset, mirroring the oracle's `_batch_gather`.
    fn batchGather(ctx: EagerContext, allocator: std.mem.Allocator, values: Array, indices: Array) !Array {
        const vsh = values.shape();
        const ish = indices.shape();
        const b = vsh[0];
        const l = vsh[1];

        var off_shape = try allocator.alloc(i32, ish.len);
        defer allocator.free(off_shape);
        off_shape[0] = b;
        for (1..ish.len) |i| off_shape[i] = 1;
        const batch_arange = try creation_mod.arange(ctx, 0, @floatFromInt(b), 1, .int32);
        defer batch_arange.deinit();
        const offs = try ops.reshape(ctx, batch_arange, off_shape);
        defer offs.deinit();
        const l_arr = try Array.fromData(allocator, i32, &[_]i32{l}, &[_]i32{1});
        defer l_arr.deinit();
        const offs_scaled = try ops.multiply(ctx, offs, l_arr);
        defer offs_scaled.deinit();
        const shifted = try ops.add(ctx, indices, offs_scaled);
        defer shifted.deinit();

        var total: i64 = 1;
        for (ish) |d| total *= d;
        const flat_idx = try ops.reshape(ctx, shifted, &[_]i32{@intCast(total)});
        defer flat_idx.deinit();

        var vflat_shape = try allocator.alloc(i32, vsh.len - 1);
        defer allocator.free(vflat_shape);
        vflat_shape[0] = b * l;
        var rest_total: i64 = 1;
        for (vsh[2..], 0..) |d, i| {
            vflat_shape[1 + i] = d;
            rest_total *= d;
        }
        const vflat = try ops.reshape(ctx, values, vflat_shape);
        defer vflat.deinit();
        const gathered = try shape_mod.takeAxis(ctx, vflat, flat_idx, 0);
        defer gathered.deinit();

        var out_shape = try allocator.alloc(i32, ish.len + (vsh.len - 2));
        defer allocator.free(out_shape);
        @memcpy(out_shape[0..ish.len], ish);
        @memcpy(out_shape[ish.len..], vsh[2..]);
        return ops.reshape(ctx, gathered, out_shape);
    }

    /// `_pooled_states`: group keys/gates into kpool windows and compress each
    /// window by softmax(gates + ape). All positions valid (no padding).
    /// keys: [B,L,D], gates: [B,L,D] (keys dtype), valid: [B,L] bool.
    /// Returns pool_keys [B,PC,D], pool_indices [B,PC,KP] i32, pool_valid [B,PC].
    const PooledStates = struct { pool_keys: Array, pool_indices: Array, pool_valid: Array };

    fn pooledStates(self: *const Glm53Indexer, allocator: std.mem.Allocator, keys: Array, gates: Array, valid: Array) !PooledStates {
        const ctx = self.ctx;
        const ksh = keys.shape();
        const b = ksh[0];
        const l = ksh[1];
        const d = ksh[2];
        const kp = self.kpool;
        const pc: i32 = @intCast(@divTrunc(l + kp - 1, kp));

        // first_key = where(any(valid), argmax(valid), L)
        const any_valid = try comparison_mod.anyAxis(ctx, valid, -1, false);
        defer any_valid.deinit();
        const valid_i32 = try ops.astype(ctx, valid, .int32);
        defer valid_i32.deinit();
        const argmax_valid = try reduce_mod.argmaxAxis(ctx, valid_i32, -1, false);
        defer argmax_valid.deinit();
        const l_scalar = try Array.fromData(allocator, i32, &[_]i32{l}, &[_]i32{1});
        defer l_scalar.deinit();
        const first_key = try ops.where(ctx, any_valid, argmax_valid, l_scalar);
        defer first_key.deinit(); // [B] i32

        // indices = first_key[:,None,None] + arange(pc*kp).reshape(1,pc,kp)
        const offsets = try creation_mod.arange(ctx, 0, @floatFromInt(pc * kp), 1, .int32);
        defer offsets.deinit();
        const offsets3 = try ops.reshape(ctx, offsets, &[_]i32{ 1, pc, kp });
        defer offsets3.deinit();
        const fk3 = try ops.reshape(ctx, first_key, &[_]i32{ b, 1, 1 });
        defer fk3.deinit();
        var indices = try ops.add(ctx, fk3, offsets3); // [B,PC,KP] i32
        errdefer indices.deinit();

        const l_minus1 = try Array.fromData(allocator, i32, &[_]i32{l - 1}, &[_]i32{1});
        defer l_minus1.deinit();
        const zero_i = try Array.fromData(allocator, i32, &[_]i32{0}, &[_]i32{1});
        defer zero_i.deinit();
        const safe = try clipMinMax(ctx, indices, zero_i, l_minus1);
        defer safe.deinit();

        const grouped_keys = try batchGather(ctx, allocator, keys, safe);
        defer grouped_keys.deinit(); // [B,PC,KP,D]
        const grouped_gates = try batchGather(ctx, allocator, gates, safe);
        defer grouped_gates.deinit();

        const valid_e = try ops.expandDims(ctx, valid, -1);
        defer valid_e.deinit();
        const gv4 = try batchGather(ctx, allocator, valid_e, safe);
        defer gv4.deinit(); // [B,PC,KP,1] bool
        const gv3 = try ops.reshape(ctx, gv4, &[_]i32{ b, pc, kp });
        defer gv3.deinit();
        const idx_lt_l = try comparison_mod.less(ctx, indices, l_scalar);
        defer idx_lt_l.deinit();
        const grouped_valid = try comparison_mod.logicalAnd(ctx, gv3, idx_lt_l);
        defer grouped_valid.deinit(); // [B,PC,KP]

        const pool_valid = try comparison_mod.allAxis(ctx, grouped_valid, -1, false);
        errdefer pool_valid.deinit(); // [B,PC]

        const minus_one = try Array.fromData(allocator, i32, &[_]i32{-1}, &[_]i32{1});
        defer minus_one.deinit();
        const indices_masked = try ops.where(ctx, grouped_valid, indices, minus_one);
        indices.deinit();
        indices = indices_masked;

        // logits = gates.f32 + ape; mask invalid to -inf; softmax over window.
        const gates_f32 = try ops.astype(ctx, grouped_gates, .float32);
        defer gates_f32.deinit();
        const ape4 = try ops.reshape(ctx, self.ape, &[_]i32{ 1, 1, kp, d });
        defer ape4.deinit();
        const logits0 = try ops.add(ctx, gates_f32, ape4);
        defer logits0.deinit();
        const gv4b = try ops.expandDims(ctx, grouped_valid, -1);
        defer gv4b.deinit();
        const neg_inf = try ops.scalarF32(ctx, -std.math.inf(f32));
        defer neg_inf.deinit();
        const logits = try ops.where(ctx, gv4b, logits0, neg_inf);
        defer logits.deinit();
        const probs_raw = try ops.softmaxPrecise(ctx, logits, &[_]i32{2});
        defer probs_raw.deinit();
        const probs_f32 = try math_mod.nanToNum(ctx, probs_raw, 0.0, null, null);
        defer probs_f32.deinit();
        const probs = try ops.astype(ctx, probs_f32, keys.dtype());
        defer probs.deinit();
        const weighted = try ops.multiply(ctx, probs, grouped_keys);
        defer weighted.deinit();
        const pool_keys = try reduce_mod.sumAxis(ctx, weighted, 2, false);
        errdefer pool_keys.deinit();

        return .{ .pool_keys = pool_keys, .pool_indices = indices, .pool_valid = pool_valid };
    }

    /// No-cache forward (oracle `cache is None` path): full prefill selection.
    /// x: [B,L,hidden] (branch input, pre-norm output); q_resid: [B,L,q_lora_rank].
    /// Returns topk indices [B, L, index_topk + tail_width] i32.
    pub fn forward(self: *const Glm53Indexer, allocator: std.mem.Allocator, x: Array, q_resid: Array) !Array {
        const ctx = self.ctx;
        const xsh = x.shape();
        const b = xsh[0];
        const l = xsh[1];
        const kp = self.kpool;
        const d = self.head_dim;

        // k = k_norm(wk(x)); gate = x.f32 @ compress_gate_t
        const k_proj = try self.wk.apply(ctx, x);
        defer k_proj.deinit();
        const k = try fast_mod.layerNorm(ctx, k_proj, self.k_norm_w, self.k_norm_b, k_norm_eps);
        defer k.deinit(); // [B,L,D]
        const x_f32 = try ops.astype(ctx, x, .float32);
        defer x_f32.deinit();
        const gate_f32 = try ops.matmul(ctx, x_f32, self.compress_gate_t);
        defer gate_f32.deinit();
        const gate = try ops.astype(ctx, gate_f32, k.dtype());
        defer gate.deinit();

        const valid = try creation_mod.ones(ctx, &[_]i32{ b, l }, .bool_);
        defer valid.deinit();

        var pooled = try self.pooledStates(allocator, k, gate, valid);
        defer pooled.pool_keys.deinit();
        defer pooled.pool_indices.deinit();
        defer pooled.pool_valid.deinit();
        const pc: i32 = @intCast(@divTrunc(l + kp - 1, kp));

        // query_positions = arange(L) (kv_length == L with no cache)
        const query_positions = try creation_mod.arange(ctx, 0, @floatFromInt(l), 1, .int32);
        defer query_positions.deinit(); // [L]

        // pool_end = clip(pool_indices[..., -1], 0, L-1)
        const last_idx = try ops.slice(ctx, pooled.pool_indices, &[_]i32{ 0, 0, kp - 1 }, &[_]i32{ b, pc, kp }, &[_]i32{ 1, 1, 1 });
        defer last_idx.deinit();
        const pool_end2 = try ops.reshape(ctx, last_idx, &[_]i32{ b, pc });
        defer pool_end2.deinit();
        const l_minus1 = try Array.fromData(allocator, i32, &[_]i32{l - 1}, &[_]i32{1});
        defer l_minus1.deinit();
        const zero_i = try Array.fromData(allocator, i32, &[_]i32{0}, &[_]i32{1});
        defer zero_i.deinit();
        const pool_end = try clipMinMax(ctx, pool_end2, zero_i, l_minus1);
        defer pool_end.deinit(); // [B,PC]

        // candidates = pool_valid[:,None] & (pool_end[:,None] <= query_positions[None,:,None])
        const pv3 = try ops.expandDims(ctx, pooled.pool_valid, 1);
        defer pv3.deinit(); // [B,1,PC]
        const pe3 = try ops.expandDims(ctx, pool_end, 1);
        defer pe3.deinit(); // [B,1,PC]
        const qp2 = try ops.reshape(ctx, query_positions, &[_]i32{ 1, l, 1 });
        defer qp2.deinit();
        const end_visible = try comparison_mod.lessEqual(ctx, pe3, qp2);
        defer end_visible.deinit();
        const candidates = try comparison_mod.logicalAnd(ctx, pv3, end_visible);
        defer candidates.deinit(); // [B,L,PC]

        const select_k = @min(@divTrunc(self.index_topk, kp), pc);
        // Unified ownership: both are released exactly once by these defers,
        // on the success and the error path alike.
        var selected_opt: ?Array = null;
        var selected_valid_opt: ?Array = null;
        defer if (selected_opt) |a| a.deinit();
        defer if (selected_valid_opt) |a| a.deinit();
        if (select_k == pc) {
            // Select every pool; validity carries the mask.
            const ar = try creation_mod.arange(ctx, 0, @floatFromInt(pc), 1, .int32);
            defer ar.deinit();
            const ar3 = try ops.reshape(ctx, ar, &[_]i32{ 1, 1, pc });
            defer ar3.deinit();
            selected_opt = try ops.broadcastTo(ctx, ar3, &[_]i32{ b, l, pc });
            selected_valid_opt = try ops.copy(ctx, candidates);
        } else {
            // q: [B,L,H,D]; weights: [B,L,H] f32 scaled by H^-0.5
            const q_idx = try self.wq_b.apply(ctx, q_resid);
            defer q_idx.deinit();
            const q4 = try ops.reshape(ctx, q_idx, &[_]i32{ b, l, self.n_heads, d });
            defer q4.deinit();
            const qf = try ops.astype(ctx, q4, .float32);
            defer qf.deinit();
            const w_proj = try self.weights_proj.apply(ctx, x);
            defer w_proj.deinit();
            const w_f32 = try ops.astype(ctx, w_proj, .float32);
            defer w_f32.deinit();
            const h_scale = try ops.scalarF32(ctx, std.math.pow(f32, @floatFromInt(self.n_heads), -0.5));
            defer h_scale.deinit();
            const weights = try ops.multiply(ctx, w_f32, h_scale);
            defer weights.deinit();

            // scores = sum_h(relu(q @ pool_keys^T) * weights * softmax_scale)
            const pk = try ops.astype(ctx, pooled.pool_keys, .float32);
            defer pk.deinit(); // [B,PC,D]
            const pk4 = try ops.expandDims(ctx, pk, 1);
            defer pk4.deinit(); // [B,1,PC,D]
            const pk_t = try ops.transposeAxes(ctx, pk4, &[_]i32{ 0, 1, 3, 2 });
            defer pk_t.deinit(); // [B,1,D,PC]
            const s_raw = try ops.matmul(ctx, qf, pk_t);
            defer s_raw.deinit(); // [B,L,H,PC]
            const zero_f = try ops.scalarF32(ctx, 0);
            defer zero_f.deinit();
            const s_relu = try math_mod.maximum(ctx, s_raw, zero_f);
            defer s_relu.deinit();
            const soft_scale = try ops.scalarF32(ctx, std.math.pow(f32, @floatFromInt(d), -0.5));
            defer soft_scale.deinit();
            const w_scaled = try ops.multiply(ctx, weights, soft_scale);
            defer w_scaled.deinit();
            const w4 = try ops.expandDims(ctx, w_scaled, -1);
            defer w4.deinit();
            const s_weighted = try ops.multiply(ctx, s_relu, w4);
            defer s_weighted.deinit();
            const scores = try reduce_mod.sumAxis(ctx, s_weighted, 2, false);
            defer scores.deinit(); // [B,L,PC]

            const f32_min = try ops.scalarF32(ctx, -std.math.floatMax(f32));
            defer f32_min.deinit();
            const scores_masked = try ops.where(ctx, candidates, scores, f32_min);
            defer scores_masked.deinit();
            const scores_neg = try ops.negative(ctx, scores_masked);
            defer scores_neg.deinit();
            const part = try sort_mod.argpartitionAxis(ctx, scores_neg, select_k - 1, -1);
            defer part.deinit();
            const sel = try ops.slice(ctx, part, &[_]i32{ 0, 0, 0 }, &[_]i32{ b, l, select_k }, &[_]i32{ 1, 1, 1 });
            defer sel.deinit();
            selected_opt = try ops.astype(ctx, sel, .int32);
            selected_valid_opt = try shape_mod.takeAlongAxis(ctx, candidates, selected_opt.?, -1);
        }
        const selected = selected_opt.?;
        const selected_valid = selected_valid_opt.?;

        // Restore chronological pool order; invalid slots go last.
        const pc_scalar = try Array.fromData(allocator, i32, &[_]i32{pc}, &[_]i32{1});
        defer pc_scalar.deinit();
        const sort_key = try ops.where(ctx, selected_valid, selected, pc_scalar);
        defer sort_key.deinit();
        const order = try sort_mod.argsortAxis(ctx, sort_key, -1);
        defer order.deinit();
        const selected_sorted = try shape_mod.takeAlongAxis(ctx, selected, order, -1);
        defer selected_sorted.deinit();
        const valid_sorted = try shape_mod.takeAlongAxis(ctx, selected_valid, order, -1);
        defer valid_sorted.deinit();

        // Expand selected pools into token indices.
        const pc_minus1 = try Array.fromData(allocator, i32, &[_]i32{pc - 1}, &[_]i32{1});
        defer pc_minus1.deinit();
        const safe_selected = try clipMinMax(ctx, selected_sorted, zero_i, pc_minus1);
        defer safe_selected.deinit(); // [B,L,K]
        const pools4 = try ops.expandDims(ctx, pooled.pool_indices, 1);
        defer pools4.deinit(); // [B,1,PC,KP]
        const expanded_pools = try ops.broadcastTo(ctx, pools4, &[_]i32{ b, l, pc, kp });
        defer expanded_pools.deinit();
        const sel4 = try ops.expandDims(ctx, safe_selected, -1);
        defer sel4.deinit(); // [B,L,K,1]
        const gathered_pools = try shape_mod.takeAlongAxis(ctx, expanded_pools, sel4, 2);
        defer gathered_pools.deinit(); // [B,L,K,KP]
        const width = select_k * kp;
        var topk = try ops.reshape(ctx, gathered_pools, &[_]i32{ b, l, width });
        errdefer topk.deinit();
        const valid4 = try ops.expandDims(ctx, valid_sorted, -1);
        defer valid4.deinit();
        const valid_rep = try ops.broadcastTo(ctx, valid4, &[_]i32{ b, l, select_k, kp });
        defer valid_rep.deinit();
        const valid_flat = try ops.reshape(ctx, valid_rep, &[_]i32{ b, l, width });
        defer valid_flat.deinit();
        const minus_one = try Array.fromData(allocator, i32, &[_]i32{-1}, &[_]i32{1});
        defer minus_one.deinit();
        {
            const masked = try ops.where(ctx, valid_flat, topk, minus_one);
            topk.deinit();
            topk = masked;
        }

        // Pad to index_topk width with -1. width = select_k * kpool <=
        // index_topk by construction (select_k = min(index_topk/kpool, pc)).
        std.debug.assert(width <= self.index_topk);
        if (width < self.index_topk) {
            const padded = try shape_mod.pad(ctx, topk, &[_]i32{2}, &[_]i32{0}, &[_]i32{self.index_topk - width}, minus_one, "constant");
            topk.deinit();
            topk = padded;
        }

        // always_select_tail: append the (kpool-1) predecessors of the current
        // partial pool so causal attention always sees the running window.
        if (self.always_select_tail and kp > 1) {
            const tail_width = kp - 1;
            // No-cache prefill: valid is all-ones, so cumsum(valid) sampled at
            // query_positions is just position+1 — skip the identity gather.
            const one_i = try Array.fromData(allocator, i32, &[_]i32{1}, &[_]i32{1});
            defer one_i.deinit();
            const qp_e = try ops.expandDims(ctx, query_positions, 0);
            defer qp_e.deinit();
            const qp_b = try ops.broadcastTo(ctx, qp_e, &[_]i32{ b, l });
            defer qp_b.deinit();
            const visible = try ops.add(ctx, qp_b, one_i);
            defer visible.deinit(); // [B,L]
            const kp_arr = try Array.fromData(allocator, i32, &[_]i32{kp}, &[_]i32{1});
            defer kp_arr.deinit();
            const tail_count = try math_mod.remainder(ctx, visible, kp_arr);
            defer tail_count.deinit(); // [B,L]

            // first_key (all-valid here, but keep the general form)
            const any_valid = try comparison_mod.anyAxis(ctx, valid, -1, false);
            defer any_valid.deinit();
            const valid_i32 = try ops.astype(ctx, valid, .int32);
            defer valid_i32.deinit();
            const argmax_valid = try reduce_mod.argmaxAxis(ctx, valid_i32, -1, false);
            defer argmax_valid.deinit();
            const l_scalar = try Array.fromData(allocator, i32, &[_]i32{l}, &[_]i32{1});
            defer l_scalar.deinit();
            const first_key = try ops.where(ctx, any_valid, argmax_valid, l_scalar);
            defer first_key.deinit(); // [B]

            const fk2 = try ops.expandDims(ctx, first_key, -1);
            defer fk2.deinit();
            const vis_base = try ops.add(ctx, fk2, visible);
            defer vis_base.deinit();
            const tail_start = try ops.subtract(ctx, vis_base, tail_count);
            defer tail_start.deinit(); // [B,L]
            const ts3 = try ops.expandDims(ctx, tail_start, -1);
            defer ts3.deinit();
            const tail_arange = try creation_mod.arange(ctx, 0, @floatFromInt(tail_width), 1, .int32);
            defer tail_arange.deinit();
            const tail = try ops.add(ctx, ts3, tail_arange);
            defer tail.deinit(); // [B,L,TW]

            const tw3 = try ops.reshape(ctx, tail_arange, &[_]i32{ 1, 1, tail_width });
            defer tw3.deinit();
            const tc3 = try ops.expandDims(ctx, tail_count, -1);
            defer tc3.deinit();
            const tv1 = try comparison_mod.less(ctx, tw3, tc3);
            defer tv1.deinit();
            const tv2 = try comparison_mod.less(ctx, tail, l_scalar);
            defer tv2.deinit();
            var tail_valid = try comparison_mod.logicalAnd(ctx, tv1, tv2);
            defer tail_valid.deinit();
            const safe_tail = try clipMinMax(ctx, tail, zero_i, l_minus1);
            defer safe_tail.deinit();
            const valid_e = try ops.expandDims(ctx, valid, -1);
            defer valid_e.deinit();
            const gathered_valid = try batchGather(ctx, allocator, valid_e, safe_tail);
            defer gathered_valid.deinit(); // [B,L,TW,1]
            const gv3 = try ops.reshape(ctx, gathered_valid, &[_]i32{ b, l, tail_width });
            defer gv3.deinit();
            {
                const tv = try comparison_mod.logicalAnd(ctx, tail_valid, gv3);
                tail_valid.deinit();
                tail_valid = tv;
            }
            const tail_masked = try ops.where(ctx, tail_valid, tail, minus_one);
            defer tail_masked.deinit();
            const with_tail = try shape_mod.concatenateAxis(ctx, &[_]Array{ topk, tail_masked }, -1);
            topk.deinit();
            topk = with_tail;
        }

        // query_valid is all-true in the no-padding harness path; cast to the
        // oracle's return dtype (mx.int32).
        const topk_i32 = try ops.astype(ctx, topk, .int32);
        topk.deinit();
        return topk_i32;
    }
};

// ============================================================
// NoPE MLA attention
// ============================================================

pub const MlaResult = struct {
    out: Array, // [B, L, hidden]
    topk: Array, // [B, L, index_topk + tail] i32
};

pub const Glm53Mla = struct {
    ctx: EagerContext,
    num_heads: i32, // 64
    q_lora_rank: i32, // 1536
    kv_lora_rank: i32, // 512
    qk_nope_head_dim: i32, // 256
    v_head_dim: i32, // 256
    rms_norm_eps: f32,

    qkv_a_proj: QuantLinear, // [q_lora+kv_lora, hidden]
    q_a_layernorm_w: Array,
    q_b_proj: QuantLinear, // [num_heads*qk_nope, q_lora]
    kv_a_layernorm_w: Array,
    /// Dequantized MultiLinear weights, transposed for direct matmul:
    /// embed_q_t: [H, qk_nope, kv_lora] (query = x @ embed_q_t)
    /// embed_q_w: [H, kv_lora, qk_nope] (keys = latent @ embed_q_w)
    /// unembed_out_t: [H, kv_lora, v_head] (out/value = x @ unembed_out_t)
    embed_q_t: Array,
    embed_q_w: Array,
    unembed_out_t: Array,
    o_proj: QuantLinear, // [hidden, num_heads*v_head]
    indexer: Glm53Indexer,

    fn scale(self: *const Glm53Mla) f32 {
        return std.math.pow(f32, @floatFromInt(self.qk_nope_head_dim), -0.5);
    }

    /// No-cache forward (oracle cache=None path).
    /// x: [B, L, hidden] (post input_layernorm). Returns branch output and
    /// the indexer topk indices.
    pub fn forward(self: *const Glm53Mla, allocator: std.mem.Allocator, x: Array) !MlaResult {
        const ctx = self.ctx;
        const xsh = x.shape();
        const b = xsh[0];
        const l = xsh[1];

        const qkv_a = try self.qkv_a_proj.apply(ctx, x);
        defer qkv_a.deinit();
        const q_a = try ops.slice(ctx, qkv_a, &[_]i32{ 0, 0, 0 }, &[_]i32{ b, l, self.q_lora_rank }, &[_]i32{ 1, 1, 1 });
        defer q_a.deinit();
        const kv_a = try ops.slice(ctx, qkv_a, &[_]i32{ 0, 0, self.q_lora_rank }, &[_]i32{ b, l, self.q_lora_rank + self.kv_lora_rank }, &[_]i32{ 1, 1, 1 });
        defer kv_a.deinit();

        const q_resid = try fast_mod.rmsNorm(ctx, q_a, self.q_a_layernorm_w, self.rms_norm_eps);
        defer q_resid.deinit();
        const q_flat = try self.q_b_proj.apply(ctx, q_resid);
        defer q_flat.deinit();
        const q4 = try ops.reshape(ctx, q_flat, &[_]i32{ b, l, self.num_heads, self.qk_nope_head_dim });
        defer q4.deinit();
        const q = try ops.transposeAxes(ctx, q4, &[_]i32{ 0, 2, 1, 3 });
        defer q.deinit(); // [B,H,L,D]

        const latent3 = try fast_mod.rmsNorm(ctx, kv_a, self.kv_a_layernorm_w, self.rms_norm_eps);
        defer latent3.deinit();
        const latent = try ops.expandDims(ctx, latent3, 1);
        defer latent.deinit(); // [B,1,L,512]

        const topk = try self.indexer.forward(allocator, x, q_resid);
        errdefer topk.deinit();

        const attended = try self.attend(allocator, q, latent, topk);
        defer attended.deinit(); // [B,H,L,V]

        const out_t = try ops.transposeAxes(ctx, attended, &[_]i32{ 0, 2, 1, 3 });
        defer out_t.deinit();
        const out_flat = try ops.reshape(ctx, out_t, &[_]i32{ b, l, self.num_heads * self.v_head_dim });
        defer out_flat.deinit();
        const out = try self.o_proj.apply(ctx, out_flat);
        return .{ .out = out, .topk = topk };
    }

    /// Sparse attention over the latent cache. L <= DECODE_BLOCK_SIZE takes the
    /// per-query gather loop in the 512-dim latent space (this is also what the
    /// oracle's short prefill runs); longer sequences pre-project keys/values
    /// into the 256-dim q/v space and gather there (oracle `_attend` else-branch,
    /// naive per-query instead of the fused indexed-sparse kernel).
    fn attend(self: *const Glm53Mla, allocator: std.mem.Allocator, q: Array, latent: Array, topk: Array) !Array {
        const ctx = self.ctx;
        const qsh = q.shape();
        const b = qsh[0];
        const h = qsh[1];
        const l = qsh[2];
        const lsh = latent.shape();
        const n = lsh[2];
        const tsh = topk.shape();
        const t = tsh[2];
        const s = self.scale();

        if (l <= DECODE_BLOCK_SIZE) {
            // Project all queries into latent space once: [B,H,L,512].
            const qf32 = try ops.astype(ctx, q, .float32);
            defer qf32.deinit();
            const eqt = try ops.astype(ctx, self.embed_q_t, .float32);
            defer eqt.deinit();
            const queries = try ops.matmul(ctx, qf32, eqt);
            defer queries.deinit(); // [B,H,L,512] f32

            const latent_f32 = try ops.astype(ctx, latent, .float32);
            defer latent_f32.deinit();

            const zero_i = try Array.fromData(allocator, i32, &[_]i32{0}, &[_]i32{1});
            defer zero_i.deinit();
            const n_minus1 = try Array.fromData(allocator, i32, &[_]i32{n - 1}, &[_]i32{1});
            defer n_minus1.deinit();
            const neg_inf = try ops.scalarF32(ctx, -std.math.inf(f32));
            defer neg_inf.deinit();
            const scale_arr = try ops.scalarF32(ctx, s);
            defer scale_arr.deinit();

            var parts = try allocator.alloc(Array, @intCast(l));
            var made: usize = 0;
            defer {
                for (parts[0..made]) |p| p.deinit();
                allocator.free(parts);
            }

            var i: i32 = 0;
            while (i < l) : (i += 1) {
                const idx = try ops.slice(ctx, topk, &[_]i32{ 0, i, 0 }, &[_]i32{ b, i + 1, t }, &[_]i32{ 1, 1, 1 });
                defer idx.deinit(); // [B,1,T]
                const ge0 = try comparison_mod.greaterEqual(ctx, idx, zero_i);
                defer ge0.deinit();
                const ltn_inc = try comparison_mod.lessEqual(ctx, idx, n_minus1);
                defer ltn_inc.deinit();
                const valid = try comparison_mod.logicalAnd(ctx, ge0, ltn_inc);
                defer valid.deinit(); // [B,1,T] bool
                const safe = try clipMinMax(ctx, idx, zero_i, n_minus1);
                defer safe.deinit();
                const gather_idx = try ops.expandDims(ctx, safe, -1);
                defer gather_idx.deinit(); // [B,1,T,1]
                const selected = try shape_mod.takeAlongAxis(ctx, latent_f32, gather_idx, 2);
                defer selected.deinit(); // [B,1,T,512] f32

                const qi = try ops.slice(ctx, queries, &[_]i32{ 0, 0, i, 0 }, &[_]i32{ b, h, i + 1, self.kv_lora_rank }, &[_]i32{ 1, 1, 1, 1 });
                defer qi.deinit(); // [B,H,1,512]
                const sel_t = try ops.transposeAxes(ctx, selected, &[_]i32{ 0, 1, 3, 2 });
                defer sel_t.deinit(); // [B,1,512,T]
                const scores_raw = try ops.matmul(ctx, qi, sel_t);
                defer scores_raw.deinit(); // [B,H,1,T]
                const scores_sc = try ops.multiply(ctx, scores_raw, scale_arr);
                defer scores_sc.deinit();
                const mask4 = try ops.reshape(ctx, valid, &[_]i32{ b, 1, 1, t });
                defer mask4.deinit();
                const scores = try ops.where(ctx, mask4, scores_sc, neg_inf);
                defer scores.deinit();
                const probs = try ops.softmaxPrecise(ctx, scores, &[_]i32{-1});
                defer probs.deinit();
                const att = try ops.matmul(ctx, probs, selected);
                defer att.deinit(); // [B,H,1,512] f32

                // unembed to v space in the attention output dtype. The oracle's
                // fused SDPA emits bf16 here (query dtype), so we round att to
                // bf16 before the unembed matmul to mirror that numerics path —
                // intentional asymmetry vs the f32 long-prefill path below.
                const att_bf16 = try ops.astype(ctx, att, .bfloat16);
                defer att_bf16.deinit();
                const out_i = try ops.matmul(ctx, att_bf16, self.unembed_out_t);
                parts[@intCast(i)] = out_i; // [B,H,1,V]; freed by the parts defer below
                made += 1;
            }
            return shape_mod.concatenateAxis(ctx, parts, 2);
        }

        // Long prefill: pre-project keys/values, gather per query in 256-dim.
        const eqw = try ops.astype(ctx, self.embed_q_w, .float32);
        defer eqw.deinit();
        const latent_f32 = try ops.astype(ctx, latent, .float32);
        defer latent_f32.deinit();
        const keys = try ops.matmul(ctx, latent_f32, eqw);
        defer keys.deinit(); // [B,H,N,256] f32
        const uot = try ops.astype(ctx, self.unembed_out_t, .float32);
        defer uot.deinit();
        const values = try ops.matmul(ctx, latent_f32, uot);
        defer values.deinit(); // [B,H,N,256] f32
        const qf32 = try ops.astype(ctx, q, .float32);
        defer qf32.deinit();

        const zero_i = try Array.fromData(allocator, i32, &[_]i32{0}, &[_]i32{1});
        defer zero_i.deinit();
        const n_minus1 = try Array.fromData(allocator, i32, &[_]i32{n - 1}, &[_]i32{1});
        defer n_minus1.deinit();
        const neg_inf = try ops.scalarF32(ctx, -std.math.inf(f32));
        defer neg_inf.deinit();
        const scale_arr = try ops.scalarF32(ctx, s);
        defer scale_arr.deinit();

        var parts = try allocator.alloc(Array, @intCast(l));
        var made: usize = 0;
        defer {
            for (parts[0..made]) |p| p.deinit();
            allocator.free(parts);
        }

        var i: i32 = 0;
        while (i < l) : (i += 1) {
            const idx = try ops.slice(ctx, topk, &[_]i32{ 0, i, 0 }, &[_]i32{ b, i + 1, t }, &[_]i32{ 1, 1, 1 });
            defer idx.deinit();
            const ge0 = try comparison_mod.greaterEqual(ctx, idx, zero_i);
            defer ge0.deinit();
            const ltn_inc = try comparison_mod.lessEqual(ctx, idx, n_minus1);
            defer ltn_inc.deinit();
            const valid = try comparison_mod.logicalAnd(ctx, ge0, ltn_inc);
            defer valid.deinit();
            const safe = try clipMinMax(ctx, idx, zero_i, n_minus1);
            defer safe.deinit();
            const gidx = try ops.reshape(ctx, safe, &[_]i32{ b, 1, t, 1 });
            defer gidx.deinit();
            const gidx_h = try ops.broadcastTo(ctx, gidx, &[_]i32{ b, h, t, 1 });
            defer gidx_h.deinit();
            const sel_k = try shape_mod.takeAlongAxis(ctx, keys, gidx_h, 2);
            defer sel_k.deinit(); // [B,H,T,256]
            const sel_v = try shape_mod.takeAlongAxis(ctx, values, gidx_h, 2);
            defer sel_v.deinit();

            const qi = try ops.slice(ctx, qf32, &[_]i32{ 0, 0, i, 0 }, &[_]i32{ b, h, i + 1, self.qk_nope_head_dim }, &[_]i32{ 1, 1, 1, 1 });
            defer qi.deinit();
            const k_t = try ops.transposeAxes(ctx, sel_k, &[_]i32{ 0, 1, 3, 2 });
            defer k_t.deinit();
            const scores_raw = try ops.matmul(ctx, qi, k_t);
            defer scores_raw.deinit();
            const scores_sc = try ops.multiply(ctx, scores_raw, scale_arr);
            defer scores_sc.deinit();
            const mask4 = try ops.reshape(ctx, valid, &[_]i32{ b, 1, 1, t });
            defer mask4.deinit();
            const scores = try ops.where(ctx, mask4, scores_sc, neg_inf);
            defer scores.deinit();
            const probs = try ops.softmaxPrecise(ctx, scores, &[_]i32{-1});
            defer probs.deinit();
            // [B,H,1,256] f32 (kept f32: this naive per-query path replaces the
            // oracle's fused bf16 sparse-attention kernel; f32 accumulation is
            // strictly more accurate and the layer-3 oracle gate (cos >= 0.999)
            // leaves ample margin). Freed by the parts defer below.
            const out_i = try ops.matmul(ctx, probs, sel_v);
            parts[@intCast(i)] = out_i;
            made += 1;
        }
        return shape_mod.concatenateAxis(ctx, parts, 2);
    }
};

// ============================================================
// Tests
// ============================================================

const test_c = @import("mlx").c;

fn makeTestQuantLinear(ctx: EagerContext, allocator: std.mem.Allocator, out_f: i32, in_f: i32, seed: i64) !QuantLinear {
    const n: usize = @intCast(out_f * in_f);
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 7 + seed, 13) - 6)) * 0.02;
    }
    const w = try Array.fromData(allocator, f32, buf, &[_]i32{ out_f, in_f });
    defer w.deinit();
    const qw = try quantize_mod.quantize(ctx, w, .{ .bits = 8, .group_size = 32, .mode = .affine });
    allocator.free(qw.original_shape); // QuantLinear keeps only the arrays
    return .{ .w = qw.data, .scales = qw.scales, .biases = qw.biases, .cfg = .{ .bits = 8, .group_size = 32, .mode = .affine } };
}

fn makeTestArray(allocator: std.mem.Allocator, shape: []const i32, seed: i64) !Array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 5 + seed, 17) - 8)) * 0.05;
    }
    return Array.fromData(allocator, f32, buf, shape);
}

test "Glm53Indexer.forward: 打分支 (select_k < pc) 形状与因果不变量" {
    test_c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Small dims that force the argpartition scoring path: pc=3 pools,
    // select_k = index_topk/kpool = 2 < pc.
    const hidden: i32 = 64;
    const head_dim: i32 = 32;
    const n_heads: i32 = 4;
    const l: i32 = 6;
    const kpool: i32 = 2;
    const index_topk: i32 = 4;

    var indexer = Glm53Indexer{
        .ctx = ctx,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .index_topk = index_topk,
        .kpool = kpool,
        .always_select_tail = true,
        .wq_b = try makeTestQuantLinear(ctx, allocator, n_heads * head_dim, hidden, 1),
        .wk = try makeTestQuantLinear(ctx, allocator, head_dim, hidden, 2),
        .weights_proj = try makeTestQuantLinear(ctx, allocator, n_heads, hidden, 3),
        .k_norm_w = blk: {
            const ones_buf = try allocator.alloc(f32, @intCast(head_dim));
            defer allocator.free(ones_buf);
            @memset(ones_buf, 1.0);
            break :blk try Array.fromData(allocator, f32, ones_buf, &[_]i32{head_dim});
        },
        .k_norm_b = try creation_mod.zeros(ctx, &[_]i32{head_dim}, .float32),
        .ape = try creation_mod.zeros(ctx, &[_]i32{ kpool, head_dim }, .float32),
        .compress_gate_t = try makeTestArray(allocator, &[_]i32{ hidden, head_dim }, 4),
    };
    defer {
        indexer.wq_b.w.deinit();
        indexer.wq_b.scales.deinit();
        indexer.wq_b.biases.deinit();
        indexer.wk.w.deinit();
        indexer.wk.scales.deinit();
        indexer.wk.biases.deinit();
        indexer.weights_proj.w.deinit();
        indexer.weights_proj.scales.deinit();
        indexer.weights_proj.biases.deinit();
        indexer.k_norm_w.deinit();
        indexer.k_norm_b.deinit();
        indexer.ape.deinit();
        indexer.compress_gate_t.deinit();
    }

    const x = try makeTestArray(allocator, &[_]i32{ 1, l, hidden }, 5);
    defer x.deinit();
    const q_resid = try makeTestArray(allocator, &[_]i32{ 1, l, hidden }, 6);
    defer q_resid.deinit();

    const topk = try indexer.forward(allocator, x, q_resid);
    defer topk.deinit();

    // Shape [1, L, index_topk + kpool - 1], dtype int32.
    const sh = topk.shape();
    try std.testing.expectEqual(@as(usize, 3), sh.len);
    try std.testing.expectEqual(@as(i32, 1), sh[0]);
    try std.testing.expectEqual(l, sh[1]);
    try std.testing.expectEqual(index_topk + kpool - 1, sh[2]);
    try std.testing.expect(topk.dtype() == .int32);

    // Causal invariant: every non-masked index in row p is in [0, p].
    try topk.eval();
    const data = try topk.dataSlice(i32);
    const width: usize = @intCast(index_topk + kpool - 1);
    for (0..@intCast(l)) |p| {
        for (data[p * width .. (p + 1) * width]) |v| {
            try std.testing.expect(v >= -1);
            try std.testing.expect(v <= @as(i32, @intCast(p)));
        }
    }
}

// ============================================================
// Caches (wired for Task 9+ model assembly; not exercised by the
// Task-8 oracle harness, which runs the cache=None prefill path).
// ============================================================

/// Latent KV cache: plain concat-append of [B,1,L,512] latents.
pub const Glm53LatentCache = struct {
    latent: ?Array = null,

    pub fn deinit(self: *Glm53LatentCache) void {
        if (self.latent) |a| a.deinit();
        self.latent = null;
    }

    pub fn offset(self: *const Glm53LatentCache) i32 {
        const a = self.latent orelse return 0;
        return a.shape()[2];
    }

    /// Append new latent [B,1,L,512]; returns the full [B,1,N,512] latent.
    /// The returned array is borrowed — owned by the cache, valid until the
    /// next updateAndFetch/deinit; do NOT deinit it. `new_latent` stays owned
    /// by the caller.
    pub fn updateAndFetch(self: *Glm53LatentCache, ctx: EagerContext, new_latent: Array) !Array {
        if (self.latent) |old| {
            const joined = try shape_mod.concatenateAxis(ctx, &[_]Array{ old, new_latent }, 2);
            old.deinit();
            self.latent = joined;
        } else {
            self.latent = try ops.copy(ctx, new_latent);
        }
        return self.latent.?;
    }
};

/// kpool window accumulator (mlx-vlm PoolingCache semantics, single-row /
/// no-left-padding / non-speculative case). Raw indexer keys/gates accumulate
/// until a full window of `ratio` tokens is available; complete windows are
/// emitted for compression, the partial tail stays in the buffer.
pub const Glm53PoolCache = struct {
    ratio: i32,
    buf_k: ?Array = null, // [B, ratio, D]
    buf_gate: ?Array = null,
    remainder: i32 = 0,
    pooled: ?Array = null, // [B, N_pool, D] compressed pool keys

    pub fn deinit(self: *Glm53PoolCache) void {
        if (self.buf_k) |a| a.deinit();
        if (self.buf_gate) |a| a.deinit();
        if (self.pooled) |a| a.deinit();
        self.* = .{ .ratio = self.ratio };
    }

    pub fn poolCount(self: *const Glm53PoolCache) i32 {
        const a = self.pooled orelse return 0;
        return a.shape()[1];
    }

    /// Accumulate new keys/gates [B,L,D]; returns the ready full windows
    /// ([B,U,D] each, U multiple of ratio, possibly 0 rows). The returned
    /// arrays are owned by the caller. On error the cache state is unchanged.
    pub fn accumulateWindows(self: *Glm53PoolCache, ctx: EagerContext, k: Array, gate: Array) !struct { k: Array, gate: Array } {
        const b = k.shape()[0];
        const l = k.shape()[1];
        const dk = k.shape()[2];
        const dg = gate.shape()[2];
        const total = self.remainder + l;
        const usable = @divTrunc(total, self.ratio) * self.ratio;
        const new_remainder = total - usable;

        var ready_k: ?Array = null;
        var ready_gate: ?Array = null;
        errdefer if (ready_k) |a| a.deinit();
        errdefer if (ready_gate) |a| a.deinit();
        if (usable == 0) {
            ready_k = try creation_mod.zeros(ctx, &[_]i32{ b, 0, dk }, k.dtype());
            ready_gate = try creation_mod.zeros(ctx, &[_]i32{ b, 0, dg }, gate.dtype());
        } else if (self.remainder == 0) {
            ready_k = try ops.slice(ctx, k, &[_]i32{ 0, 0, 0 }, &[_]i32{ b, usable, dk }, &[_]i32{ 1, 1, 1 });
            ready_gate = try ops.slice(ctx, gate, &[_]i32{ 0, 0, 0 }, &[_]i32{ b, usable, dg }, &[_]i32{ 1, 1, 1 });
        } else {
            const buf_k_part = try ops.slice(ctx, self.buf_k.?, &[_]i32{ 0, 0, 0 }, &[_]i32{ b, self.remainder, dk }, &[_]i32{ 1, 1, 1 });
            defer buf_k_part.deinit();
            const buf_g_part = try ops.slice(ctx, self.buf_gate.?, &[_]i32{ 0, 0, 0 }, &[_]i32{ b, self.remainder, dg }, &[_]i32{ 1, 1, 1 });
            defer buf_g_part.deinit();
            const consume = usable - self.remainder;
            const new_k_part = try ops.slice(ctx, k, &[_]i32{ 0, 0, 0 }, &[_]i32{ b, consume, dk }, &[_]i32{ 1, 1, 1 });
            defer new_k_part.deinit();
            const new_g_part = try ops.slice(ctx, gate, &[_]i32{ 0, 0, 0 }, &[_]i32{ b, consume, dg }, &[_]i32{ 1, 1, 1 });
            defer new_g_part.deinit();
            ready_k = try shape_mod.concatenateAxis(ctx, &[_]Array{ buf_k_part, new_k_part }, 1);
            ready_gate = try shape_mod.concatenateAxis(ctx, &[_]Array{ buf_g_part, new_g_part }, 1);
        }

        // New remainder buffer, built into locals first so a failure mid-way
        // leaves the cache untouched. When a window completed the old remainder
        // was consumed into `ready_*`, so the new buffer is purely the tail of
        // the new input; otherwise it is the carried-over buffer plus the tail.
        var new_buf_k: ?Array = null;
        var new_buf_gate: ?Array = null;
        errdefer if (new_buf_k) |a| a.deinit();
        errdefer if (new_buf_gate) |a| a.deinit();
        if (new_remainder > 0) {
            const tail_len = @min(new_remainder, l);
            const tail_k = try ops.slice(ctx, k, &[_]i32{ 0, l - tail_len, 0 }, &[_]i32{ b, l, dk }, &[_]i32{ 1, 1, 1 });
            defer tail_k.deinit();
            const tail_g = try ops.slice(ctx, gate, &[_]i32{ 0, l - tail_len, 0 }, &[_]i32{ b, l, dg }, &[_]i32{ 1, 1, 1 });
            defer tail_g.deinit();
            const carry = new_remainder - tail_len; // from old buffer
            if (carry > 0) {
                const carry_k = try ops.slice(ctx, self.buf_k.?, &[_]i32{ 0, 0, 0 }, &[_]i32{ b, carry, dk }, &[_]i32{ 1, 1, 1 });
                defer carry_k.deinit();
                const carry_g = try ops.slice(ctx, self.buf_gate.?, &[_]i32{ 0, 0, 0 }, &[_]i32{ b, carry, dg }, &[_]i32{ 1, 1, 1 });
                defer carry_g.deinit();
                new_buf_k = try shape_mod.concatenateAxis(ctx, &[_]Array{ carry_k, tail_k }, 1);
                new_buf_gate = try shape_mod.concatenateAxis(ctx, &[_]Array{ carry_g, tail_g }, 1);
            } else {
                new_buf_k = try ops.copy(ctx, tail_k);
                new_buf_gate = try ops.copy(ctx, tail_g);
            }
        }

        // Commit: swap buffers and advance the remainder count.
        if (self.buf_k) |a| a.deinit();
        if (self.buf_gate) |a| a.deinit();
        self.buf_k = new_buf_k;
        self.buf_gate = new_buf_gate;
        new_buf_k = null;
        new_buf_gate = null;
        self.remainder = new_remainder;
        return .{ .k = ready_k.?, .gate = ready_gate.? };
    }

    /// Append compressed pool keys [B,P,D]; returns all pooled keys.
    /// The returned array is borrowed — owned by the cache, valid until the
    /// next updateAndFetch/deinit; do NOT deinit it. `new_pools` stays owned
    /// by the caller.
    pub fn updateAndFetch(self: *Glm53PoolCache, ctx: EagerContext, new_pools: Array) !Array {
        if (new_pools.shape()[1] == 0) {
            if (self.pooled) |p| return p;
            // Keep ownership uniform: store an internal (empty) copy instead
            // of aliasing the caller's array.
            self.pooled = try ops.copy(ctx, new_pools);
            return self.pooled.?;
        }
        if (self.pooled) |old| {
            const joined = try shape_mod.concatenateAxis(ctx, &[_]Array{ old, new_pools }, 1);
            old.deinit();
            self.pooled = joined;
        } else {
            self.pooled = try ops.copy(ctx, new_pools);
        }
        return self.pooled.?;
    }
};
