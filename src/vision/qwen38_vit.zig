//! Qwen3.8 vision tower (ViT, bf16): conv3d patch embed, interpolated
//! learned pos-embeds, 2D rotary in merge-block order, non-causal full
//! attention blocks, and a spatial-merge MLP head. Faithful port of
//! mlx_vlm/models/qwen3_vl/vision.py for a single image/video clip per call
//! (cu_seqlens degenerates to one span). Weights are BORROWED from the
//! caller-owned map (like qwen38.zig borrowTensor): deinit frees only the
//! block slice, not the weight arrays.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const conv_mod = @import("mlx").conv;
const fast_mod = @import("mlx").fast;
const math_mod = @import("mlx").math;
const activations = @import("mlx").activations;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

pub const Qwen38VitConfig = struct {
    depth: usize,
    hidden: i32,
    heads: i32,
    intermediate: i32,
    patch: i32,
    temporal_patch: i32,
    merge: i32,
    out_hidden: i32,
    num_pos: i32,
    in_ch: i32,
};

const LN_EPS: f32 = 1e-6;
const ROPE_THETA: f64 = 10000.0;

fn linear(ctx: EagerContext, x: Array, w: Array, b: ?Array) !Array {
    const wt = try ops.transpose(ctx, w);
    defer wt.deinit();
    const y = try ops.matmul(ctx, x, wt);
    if (b) |bias| {
        defer y.deinit();
        return ops.add(ctx, y, bias);
    }
    return y;
}

/// Exact GELU (erf) — used by the merger (nn.GELU() default in vision.py).
fn geluExact(ctx: EagerContext, x: Array) !Array {
    const inv_sqrt2 = try ops.scalarF32(ctx, @floatCast(1.0 / @sqrt(2.0)));
    defer inv_sqrt2.deinit();
    const scaled = try ops.multiply(ctx, x, inv_sqrt2);
    defer scaled.deinit();
    const erf_v = try math_mod.erf(ctx, scaled);
    defer erf_v.deinit();
    const one = try ops.scalarF32(ctx, 1.0);
    defer one.deinit();
    const one_plus = try ops.add(ctx, one, erf_v);
    defer one_plus.deinit();
    const half = try ops.scalarF32(ctx, 0.5);
    defer half.deinit();
    const half_x = try ops.multiply(ctx, x, half);
    defer half_x.deinit();
    return ops.multiply(ctx, half_x, one_plus);
}

/// rotate_half(x) = cat(-x[d/2:], x[:d/2]) along the last axis.
fn rotateHalf(ctx: EagerContext, x: Array) !Array {
    const sh = x.shape();
    const nd = sh.len;
    const d: i32 = sh[nd - 1];
    const half: i32 = @divExact(d, 2);
    var start1 = [_]i32{0} ** 8;
    var stop1 = [_]i32{0} ** 8;
    var strides = [_]i32{1} ** 8;
    for (0..nd) |i| {
        start1[i] = 0;
        stop1[i] = sh[i];
    }
    var start2 = start1;
    var stop2 = stop1;
    stop1[nd - 1] = half;
    start2[nd - 1] = half;
    const first = try ops.slice(ctx, x, start1[0..nd], stop1[0..nd], strides[0..nd]);
    defer first.deinit();
    const second = try ops.slice(ctx, x, start2[0..nd], stop2[0..nd], strides[0..nd]);
    defer second.deinit();
    const neg = try ops.negative(ctx, second);
    defer neg.deinit();
    return shape_mod.concatenateAxis(ctx, &[_]Array{ neg, first }, @intCast(nd - 1));
}

/// apply_rotary_pos_emb_vision (vision.py:23-40): cos/sin tiled x2 along the
/// head dim, x*cos + rotate_half(x)*sin, cast back to the input dtype.
/// x: [1, N, heads, D]; freqs: [N, D/2] f32.
fn applyRotaryVision(ctx: EagerContext, x: Array, freqs: Array) !Array {
    const dt = x.dtype();
    const cos0 = try ops.cos(ctx, freqs);
    defer cos0.deinit();
    const cos1 = try ops.expandDims(ctx, cos0, 1);
    defer cos1.deinit();
    const cos2 = try ops.tile(ctx, cos1, &[_]i32{ 1, 1, 2 });
    defer cos2.deinit();
    const cos3 = try ops.expandDims(ctx, cos2, 0);
    defer cos3.deinit();

    const sin0 = try ops.sin(ctx, freqs);
    defer sin0.deinit();
    const sin1 = try ops.expandDims(ctx, sin0, 1);
    defer sin1.deinit();
    const sin2 = try ops.tile(ctx, sin1, &[_]i32{ 1, 1, 2 });
    defer sin2.deinit();
    const sin3 = try ops.expandDims(ctx, sin2, 0);
    defer sin3.deinit();

    const x_cos = try ops.multiply(ctx, x, cos3);
    defer x_cos.deinit();
    const rot = try rotateHalf(ctx, x);
    defer rot.deinit();
    const rot_sin = try ops.multiply(ctx, rot, sin3);
    defer rot_sin.deinit();
    const out = try ops.add(ctx, x_cos, rot_sin);
    defer out.deinit();
    return ops.astype(ctx, out, dt);
}

const VitBlock = struct {
    norm1_w: Array,
    norm1_b: Array,
    qkv_w: Array,
    qkv_b: Array,
    proj_w: Array,
    proj_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    fc1_w: Array,
    fc1_b: Array,
    fc2_w: Array,
    fc2_b: Array,
};

pub const Qwen38Vit = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    cfg: Qwen38VitConfig,
    patch_w: Array, // [hidden, t, kh, kw, in] (MLX conv3d layout)
    patch_b: Array,
    pos_embed: Array, // [num_pos, hidden]
    blocks: []VitBlock,
    merger_norm_w: Array,
    merger_norm_b: Array,
    merger_fc1_w: Array,
    merger_fc1_b: Array,
    merger_fc2_w: Array,
    merger_fc2_b: Array,

    fn lookup(weights: *const std.StringHashMap(Array), allocator: std.mem.Allocator, prefix: []const u8, name: []const u8) !Array {
        const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, name });
        defer allocator.free(full);
        return weights.get(full) orelse {
            std.log.err("qwen38_vit: missing weight '{s}'", .{full});
            return error.MissingWeight;
        };
    }

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: EagerContext,
        cfg: Qwen38VitConfig,
        weights: *const std.StringHashMap(Array),
    ) !Qwen38Vit {
        const prefix: []const u8 = if (weights.get("vision_tower.patch_embed.proj.weight") != null)
            "vision_tower."
        else
            "";

        const patch_w = try lookup(weights, allocator, prefix, "patch_embed.proj.weight");
        const patch_b = try lookup(weights, allocator, prefix, "patch_embed.proj.bias");
        const pos_embed = try lookup(weights, allocator, prefix, "pos_embed.weight");

        const blocks = try allocator.alloc(VitBlock, cfg.depth);
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*blk, i| {
            var name_buf: [96]u8 = undefined;
            const base = std.fmt.bufPrint(&name_buf, "blocks.{d}.", .{i}) catch unreachable;
            const names = [_][]const u8{
                "norm1.weight",          "norm1.bias",          "attn.qkv.weight",       "attn.qkv.bias",
                "attn.proj.weight",      "attn.proj.bias",      "norm2.weight",          "norm2.bias",
                "mlp.linear_fc1.weight", "mlp.linear_fc1.bias", "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
            };
            var arrs: [12]Array = undefined;
            for (names, 0..) |n, j| {
                var nb: [128]u8 = undefined;
                const full_name = std.fmt.bufPrint(&nb, "{s}{s}", .{ base, n }) catch unreachable;
                arrs[j] = try lookup(weights, allocator, prefix, full_name);
            }
            blk.* = .{
                .norm1_w = arrs[0],
                .norm1_b = arrs[1],
                .qkv_w = arrs[2],
                .qkv_b = arrs[3],
                .proj_w = arrs[4],
                .proj_b = arrs[5],
                .norm2_w = arrs[6],
                .norm2_b = arrs[7],
                .fc1_w = arrs[8],
                .fc1_b = arrs[9],
                .fc2_w = arrs[10],
                .fc2_b = arrs[11],
            };
        }

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .cfg = cfg,
            .patch_w = patch_w,
            .patch_b = patch_b,
            .pos_embed = pos_embed,
            .blocks = blocks,
            .merger_norm_w = try lookup(weights, allocator, prefix, "merger.norm.weight"),
            .merger_norm_b = try lookup(weights, allocator, prefix, "merger.norm.bias"),
            .merger_fc1_w = try lookup(weights, allocator, prefix, "merger.linear_fc1.weight"),
            .merger_fc1_b = try lookup(weights, allocator, prefix, "merger.linear_fc1.bias"),
            .merger_fc2_w = try lookup(weights, allocator, prefix, "merger.linear_fc2.weight"),
            .merger_fc2_b = try lookup(weights, allocator, prefix, "merger.linear_fc2.bias"),
        };
    }

    /// Borrows weights from the caller-owned map; only the block slice is ours.
    pub fn deinit(self: *Qwen38Vit) void {
        self.allocator.free(self.blocks);
    }

    /// conv3d patch embed: pixels [N, in_ch*t*p*p] -> [N, hidden]
    /// (stride == kernel, so this is unfold + GEMM).
    fn patchEmbed(self: *const Qwen38Vit, pixels: Array, n: i32) !Array {
        const ctx = self.ctx;
        const cfg = self.cfg;
        var pix = pixels;
        var casted = false;
        if (pixels.dtype() != self.patch_w.dtype()) {
            pix = try ops.astype(ctx, pixels, self.patch_w.dtype());
            casted = true;
        }
        defer if (casted) pix.deinit();

        const r = try ops.reshape(ctx, pix, &[_]i32{ n, cfg.in_ch, cfg.temporal_patch, cfg.patch, cfg.patch });
        defer r.deinit();
        const moved = try shape_mod.moveaxis(ctx, r, 1, 4); // [N, t, p, p, in]
        defer moved.deinit();
        const conv = try conv_mod.conv3d(ctx, moved, self.patch_w, cfg.temporal_patch, cfg.patch, cfg.patch, 0, 0, 0, 1, 1, 1, 1);
        defer conv.deinit();
        const flat = try ops.reshape(ctx, conv, &[_]i32{ n, cfg.hidden });
        defer flat.deinit();
        return ops.add(ctx, flat, self.patch_b);
    }

    /// fast_pos_embed_interpolate (vision.py:295-376) for a single (t, h, w)
    /// grid: bilinear 4-corner interpolation of the learned pos_embed table on
    /// the num_grid_per_side^2 canonical grid, then the merge-order permute.
    /// Returns [t*h*w, hidden].
    fn posEmbedInterpolate(self: *const Qwen38Vit, grid_t: i32, grid_h: i32, grid_w: i32) !Array {
        const ctx = self.ctx;
        const m: usize = @intCast(self.cfg.merge);
        const t: usize = @intCast(grid_t);
        const h: usize = @intCast(grid_h);
        const w: usize = @intCast(grid_w);
        const side: usize = @intFromFloat(@sqrt(@as(f64, @floatFromInt(self.cfg.num_pos))));
        std.debug.assert(side * side == @as(usize, @intCast(self.cfg.num_pos)));
        const hidden: usize = @intCast(self.cfg.hidden);
        const n_hw = h * w;

        // Host-side f64 corner indices + weights (row-major h x w grid).
        const idx = try self.allocator.alloc(i32, 4 * n_hw);
        defer self.allocator.free(idx);
        const wgt = try self.allocator.alloc(f32, 4 * n_hw);
        defer self.allocator.free(wgt);
        const linspace = struct {
            fn at(i: usize, n: usize, hi: f64) f64 {
                return if (n > 1) hi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n - 1)) else 0.0;
            }
        }.at;
        const hi: f64 = @floatFromInt(side - 1);
        for (0..h) |i| {
            const hx = linspace(i, h, hi);
            const hf: usize = @intFromFloat(@floor(hx));
            const hc = @min(hf + 1, side - 1);
            const dh: f64 = hx - @as(f64, @floatFromInt(hf));
            for (0..w) |j| {
                const wx = linspace(j, w, hi);
                const wf: usize = @intFromFloat(@floor(wx));
                const wc = @min(wf + 1, side - 1);
                const dw: f64 = wx - @as(f64, @floatFromInt(wf));
                const p = i * w + j;
                idx[0 * n_hw + p] = @intCast(hf * side + wf);
                idx[1 * n_hw + p] = @intCast(hf * side + wc);
                idx[2 * n_hw + p] = @intCast(hc * side + wf);
                idx[3 * n_hw + p] = @intCast(hc * side + wc);
                wgt[0 * n_hw + p] = @floatCast((1.0 - dh) * (1.0 - dw));
                wgt[1 * n_hw + p] = @floatCast((1.0 - dh) * dw);
                wgt[2 * n_hw + p] = @floatCast(dh * (1.0 - dw));
                wgt[3 * n_hw + p] = @floatCast(dh * dw);
            }
        }

        const n_dim: i32 = @intCast(n_hw);
        var acc: ?Array = null;
        errdefer if (acc) |a| a.deinit();
        for (0..4) |corner| {
            const idx_arr = try Array.fromData(self.allocator, i32, idx[corner * n_hw .. (corner + 1) * n_hw], &[_]i32{n_dim});
            defer idx_arr.deinit();
            const gathered = try shape_mod.takeAxis(ctx, self.pos_embed, idx_arr, 0);
            defer gathered.deinit();
            const w_arr = try Array.fromData(self.allocator, f32, wgt[corner * n_hw .. (corner + 1) * n_hw], &[_]i32{ n_dim, 1 });
            defer w_arr.deinit();
            const w_cast = try ops.astype(ctx, w_arr, gathered.dtype());
            defer w_cast.deinit();
            const weighted = try ops.multiply(ctx, gathered, w_cast);
            defer weighted.deinit();
            if (acc) |a| {
                const s = try ops.add(ctx, a, weighted);
                a.deinit();
                acc = s;
            } else {
                acc = try ops.copy(ctx, weighted);
            }
        }
        const pos = acc.?;
        defer pos.deinit(); // [h*w, hidden]

        // Merge-order permute (vision.py:359-372): tile over t, then
        // [t, h/m, m, w/m, m, F] -> transpose (0,1,3,2,4,5) -> [t*h*w, F].
        const tiled = try ops.tile(ctx, pos, &[_]i32{ @intCast(t), 1 });
        defer tiled.deinit();
        const r5 = try ops.reshape(ctx, tiled, &[_]i32{ grid_t, grid_h, grid_w, self.cfg.hidden });
        defer r5.deinit();
        const r6 = try ops.reshape(ctx, r5, &[_]i32{
            grid_t, @intCast(h / m), @intCast(m), @intCast(w / m), @intCast(m), self.cfg.hidden,
        });
        defer r6.deinit();
        const tr = try ops.transposeAxes(ctx, r6, &[_]i32{ 0, 1, 3, 2, 4, 5 });
        defer tr.deinit();
        return ops.reshape(ctx, tr, &[_]i32{ @intCast(t * n_hw), @intCast(hidden) });
    }

    /// rot_pos_emb (vision.py:232-281) for a single grid: 2D coords in
    /// merge-block order, freq table outer(arange(max_hw), inv_freq), h and w
    /// embeddings concatenated. Returns [t*h*w, head_dim/2] f32.
    fn rotPosEmb(self: *const Qwen38Vit, grid_t: i32, grid_h: i32, grid_w: i32) !Array {
        const heads: usize = @intCast(self.cfg.heads);
        const hidden: usize = @intCast(self.cfg.hidden);
        const head_dim = hidden / heads;
        const rot_dim = head_dim / 2; // VisionRotaryEmbedding(head_dim // 2)
        const n_freq = rot_dim / 2;
        const m: usize = @intCast(self.cfg.merge);
        const t: usize = @intCast(grid_t);
        const h: usize = @intCast(grid_h);
        const w: usize = @intCast(grid_w);
        const max_hw: usize = @max(h, w);
        const n_hw = h * w;

        // Host freq table [max_hw, n_freq] (f64 -> f32).
        const table = try self.allocator.alloc(f32, max_hw * n_freq);
        defer self.allocator.free(table);
        for (0..max_hw) |p| {
            for (0..n_freq) |j| {
                const inv = std.math.pow(f64, ROPE_THETA, -2.0 * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(rot_dim)));
                table[p * n_freq + j] = @floatCast(@as(f64, @floatFromInt(p)) * inv);
            }
        }

        // Coords in merge-block order: (bh, bw, ir, ic) -> row = bh*m+ir,
        // col = bw*m+ic; emb = concat(freq[row], freq[col]); tiled over t.
        const emb = try self.allocator.alloc(f32, t * n_hw * rot_dim);
        defer self.allocator.free(emb);
        var dst: usize = 0;
        for (0..t) |_| {
            for (0..h / m) |bh| {
                for (0..w / m) |bw| {
                    for (0..m) |ir| {
                        for (0..m) |ic| {
                            const row = bh * m + ir;
                            const col = bw * m + ic;
                            @memcpy(emb[dst .. dst + n_freq], table[row * n_freq .. (row + 1) * n_freq]);
                            @memcpy(emb[dst + n_freq .. dst + rot_dim], table[col * n_freq .. (col + 1) * n_freq]);
                            dst += rot_dim;
                        }
                    }
                }
            }
        }
        return Array.fromData(self.allocator, f32, emb, &[_]i32{ @intCast(t * n_hw), @intCast(rot_dim) });
    }

    fn blockForward(self: *const Qwen38Vit, blk: *const VitBlock, x: Array, rope: Array) !Array {
        const ctx = self.ctx;
        const n: i32 = x.shape()[0];
        const heads = self.cfg.heads;
        const head_dim: i32 = @divExact(self.cfg.hidden, heads);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Attention sub-block.
        const n1 = try fast_mod.layerNorm(ctx, x, blk.norm1_w, blk.norm1_b, LN_EPS);
        defer n1.deinit();
        const qkv = try linear(ctx, n1, blk.qkv_w, blk.qkv_b); // [N, 3*hidden]
        defer qkv.deinit();
        const qkv_r = try ops.reshape(ctx, qkv, &[_]i32{ n, 3, heads, head_dim });
        defer qkv_r.deinit();
        const qkv_t = try ops.transposeAxes(ctx, qkv_r, &[_]i32{ 1, 0, 2, 3 }); // [3, N, heads, D]
        defer qkv_t.deinit();
        // slice q/k/v out of the stacked tensor (shape_mod.split segfaults in
        // this mlx-c build).
        const ones3 = &[_]i32{ 1, 1, 1, 1 };
        const q_raw = try ops.slice(ctx, qkv_t, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, n, heads, head_dim }, ones3);
        defer q_raw.deinit();
        const k_raw = try ops.slice(ctx, qkv_t, &[_]i32{ 1, 0, 0, 0 }, &[_]i32{ 2, n, heads, head_dim }, ones3);
        defer k_raw.deinit();
        const v_raw = try ops.slice(ctx, qkv_t, &[_]i32{ 2, 0, 0, 0 }, &[_]i32{ 3, n, heads, head_dim }, ones3);
        defer v_raw.deinit();
        const q = try applyRotaryVision(ctx, q_raw, rope); // [1, N, heads, D]
        defer q.deinit();
        const k = try applyRotaryVision(ctx, k_raw, rope);
        defer k.deinit();
        const q_t = try ops.transposeAxes(ctx, q, &[_]i32{ 0, 2, 1, 3 }); // [1, heads, N, D]
        defer q_t.deinit();
        const k_t = try ops.transposeAxes(ctx, k, &[_]i32{ 0, 2, 1, 3 });
        defer k_t.deinit();
        const v_t = try ops.transposeAxes(ctx, v_raw, &[_]i32{ 0, 2, 1, 3 });
        defer v_t.deinit();
        const attn = try fast_mod.scaledDotProductAttention(ctx, q_t, k_t, v_t, scale, "", null, null);
        defer attn.deinit();
        const attn_t = try ops.transposeAxes(ctx, attn, &[_]i32{ 0, 2, 1, 3 });
        defer attn_t.deinit();
        const attn_flat = try ops.reshape(ctx, attn_t, &[_]i32{ n, self.cfg.hidden });
        defer attn_flat.deinit();
        const proj = try linear(ctx, attn_flat, blk.proj_w, blk.proj_b);
        defer proj.deinit();
        const h1 = try ops.add(ctx, x, proj);
        defer h1.deinit();

        // MLP sub-block (GELU tanh).
        const n2 = try fast_mod.layerNorm(ctx, h1, blk.norm2_w, blk.norm2_b, LN_EPS);
        defer n2.deinit();
        const f1 = try linear(ctx, n2, blk.fc1_w, blk.fc1_b);
        defer f1.deinit();
        const act = try activations.gelu(ctx, f1);
        defer act.deinit();
        const f2 = try linear(ctx, act, blk.fc2_w, blk.fc2_b);
        defer f2.deinit();
        return ops.add(ctx, h1, f2);
    }

    /// pixels: [grid_t*grid_h*grid_w, in_ch*t*p*p] bf16 (preprocess output);
    /// returns [ grid_t*(grid_h/merge)*(grid_w/merge), out_hidden ] bf16.
    pub fn forward(self: *Qwen38Vit, pixels: Array, grid_t: i32, grid_h: i32, grid_w: i32) !Array {
        const ctx = self.ctx;
        const m: i32 = self.cfg.merge;
        const n: i32 = grid_t * grid_h * grid_w;
        const n_merged: i32 = grid_t * @divExact(grid_h, m) * @divExact(grid_w, m);

        var hidden = try self.patchEmbed(pixels, n);

        const pos = try self.posEmbedInterpolate(grid_t, grid_h, grid_w);
        defer pos.deinit();
        const hidden_pos = try ops.add(ctx, hidden, pos);
        hidden.deinit();
        hidden = hidden_pos;

        const rope = try self.rotPosEmb(grid_t, grid_h, grid_w);
        defer rope.deinit();

        for (self.blocks) |*blk| {
            const out = try self.blockForward(blk, hidden, rope);
            hidden.deinit();
            hidden = out;
        }

        // Merger (use_postshuffle_norm=false): LN on [N, hidden], reshape to
        // [n_merged, hidden*m*m], fc1 -> GELU(exact) -> fc2.
        const normed = try fast_mod.layerNorm(ctx, hidden, self.merger_norm_w, self.merger_norm_b, LN_EPS);
        hidden.deinit();
        defer normed.deinit();
        const resh = try ops.reshape(ctx, normed, &[_]i32{ n_merged, self.cfg.hidden * m * m });
        defer resh.deinit();
        const f1 = try linear(ctx, resh, self.merger_fc1_w, self.merger_fc1_b);
        defer f1.deinit();
        const act = try geluExact(ctx, f1);
        defer act.deinit();
        return linear(ctx, act, self.merger_fc2_w, self.merger_fc2_b);
    }
};

// ============================================================
// Tests
// ============================================================

fn toyFill(buf: []f32) void {
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 13) - 6)) * 0.02;
    }
}

fn putToy(allocator: std.mem.Allocator, map: *std.StringHashMap(Array), name: []const u8, shape_: []const i32) !void {
    var n: usize = 1;
    for (shape_) |s| n *= @intCast(s);
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    toyFill(buf);
    try map.put(name, try Array.fromData(allocator, f32, buf, shape_));
}

fn putToyNorm(allocator: std.mem.Allocator, map: *std.StringHashMap(Array), name: []const u8, dim: i32) !void {
    const buf = try allocator.alloc(f32, @intCast(dim));
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = 1.0 + 0.01 * @as(f32, @floatFromInt(i));
    try map.put(name, try Array.fromData(allocator, f32, buf, &[_]i32{dim}));
    const zeros = try allocator.alloc(f32, @intCast(dim));
    defer allocator.free(zeros);
    @memset(zeros, 0.0);
    const bias_name = try std.fmt.allocPrint(allocator, "{s}", .{name[0 .. name.len - ".weight".len]});
    defer allocator.free(bias_name);
    try map.put(try std.fmt.allocPrint(allocator, "{s}.bias", .{bias_name}), try Array.fromData(allocator, f32, zeros, &[_]i32{dim}));
}

/// Toy tower: depth 1, hidden 16, heads 2, patch 2, temporal_patch 2,
/// merge `merge`, intermediate 32, out_hidden 8, num_pos 16 (4x4), in_ch 3.
fn makeToyMap(allocator: std.mem.Allocator, merge: i32) !std.StringHashMap(Array) {
    var map = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = map.valueIterator();
        while (it.next()) |v| v.deinit();
        var kit = map.keyIterator();
        while (kit.next()) |k| allocator.free(k.*);
        map.deinit();
    }
    const H: i32 = 16;
    const m2 = merge * merge;
    try putToy(allocator, &map, try allocator.dupe(u8, "patch_embed.proj.weight"), &[_]i32{ H, 2, 2, 2, 3 });
    try putToy(allocator, &map, try allocator.dupe(u8, "patch_embed.proj.bias"), &[_]i32{H});
    try putToy(allocator, &map, try allocator.dupe(u8, "pos_embed.weight"), &[_]i32{ 16, H });
    try putToyNorm(allocator, &map, try allocator.dupe(u8, "blocks.0.norm1.weight"), H);
    try putToy(allocator, &map, try allocator.dupe(u8, "blocks.0.attn.qkv.weight"), &[_]i32{ 3 * H, H });
    try putToy(allocator, &map, try allocator.dupe(u8, "blocks.0.attn.qkv.bias"), &[_]i32{3 * H});
    try putToy(allocator, &map, try allocator.dupe(u8, "blocks.0.attn.proj.weight"), &[_]i32{ H, H });
    try putToy(allocator, &map, try allocator.dupe(u8, "blocks.0.attn.proj.bias"), &[_]i32{H});
    try putToyNorm(allocator, &map, try allocator.dupe(u8, "blocks.0.norm2.weight"), H);
    try putToy(allocator, &map, try allocator.dupe(u8, "blocks.0.mlp.linear_fc1.weight"), &[_]i32{ 32, H });
    try putToy(allocator, &map, try allocator.dupe(u8, "blocks.0.mlp.linear_fc1.bias"), &[_]i32{32});
    try putToy(allocator, &map, try allocator.dupe(u8, "blocks.0.mlp.linear_fc2.weight"), &[_]i32{ H, 32 });
    try putToy(allocator, &map, try allocator.dupe(u8, "blocks.0.mlp.linear_fc2.bias"), &[_]i32{H});
    try putToyNorm(allocator, &map, try allocator.dupe(u8, "merger.norm.weight"), H);
    try putToy(allocator, &map, try allocator.dupe(u8, "merger.linear_fc1.weight"), &[_]i32{ H * m2, H * m2 });
    try putToy(allocator, &map, try allocator.dupe(u8, "merger.linear_fc1.bias"), &[_]i32{H * m2});
    try putToy(allocator, &map, try allocator.dupe(u8, "merger.linear_fc2.weight"), &[_]i32{ 8, H * m2 });
    try putToy(allocator, &map, try allocator.dupe(u8, "merger.linear_fc2.bias"), &[_]i32{8});
    return map;
}

fn freeToyMap(allocator: std.mem.Allocator, map: *std.StringHashMap(Array)) void {
    var it = map.iterator();
    while (it.next()) |e| {
        e.value_ptr.deinit();
        allocator.free(e.key_ptr.*);
    }
    map.deinit();
}

fn toyVitTest(allocator: std.mem.Allocator, grid_t: i32, grid_h: i32, grid_w: i32, merge: i32) !void {
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    var map = try makeToyMap(allocator, merge);
    defer freeToyMap(allocator, &map);

    const cfg = Qwen38VitConfig{
        .depth = 1,
        .hidden = 16,
        .heads = 2,
        .intermediate = 32,
        .patch = 2,
        .temporal_patch = 2,
        .merge = merge,
        .out_hidden = 8,
        .num_pos = 16,
        .in_ch = 3,
    };
    var vit = try Qwen38Vit.init(allocator, ctx, cfg, &map);
    defer vit.deinit();

    const n: usize = @intCast(grid_t * grid_h * grid_w);
    const patch_dim: usize = @intCast(3 * 2 * 2 * 2);
    const buf = try allocator.alloc(f32, n * patch_dim);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    const pix_f32 = try Array.fromData(allocator, f32, buf, &[_]i32{ @intCast(n), @intCast(patch_dim) });
    defer pix_f32.deinit();
    const pixels = try ops.astype(ctx, pix_f32, .bfloat16);
    defer pixels.deinit();

    const out1 = try vit.forward(pixels, grid_t, grid_h, grid_w);
    defer out1.deinit();
    try out1.eval();
    const n_merged: i32 = grid_t * @divExact(grid_h, merge) * @divExact(grid_w, merge);
    try std.testing.expectEqual(n_merged, out1.shape()[0]);
    try std.testing.expectEqual(@as(i32, 8), out1.shape()[1]);
    try std.testing.expectEqual(dtype_mod.Dtype.float32, out1.dtype()); // toy weights are f32

    // Determinism: same input -> bitwise identical output.
    const out2 = try vit.forward(pixels, grid_t, grid_h, grid_w);
    defer out2.deinit();
    const diff = try ops.subtract(ctx, out1, out2);
    defer diff.deinit();
    const abs_diff = try ops.abs(ctx, diff);
    defer abs_diff.deinit();
    const max_diff = try reduce_mod.max(ctx, abs_diff, false);
    defer max_diff.deinit();
    try max_diff.eval();
    try std.testing.expectEqual(@as(f32, 0.0), try max_diff.item(f32));
}

const dtype_mod = @import("mlx").dtype;
const reduce_mod = @import("mlx").reduce;

test "Qwen38Vit toy tower: shape + determinism (merge=1, t=2)" {
    c.initErrorHandler();
    try toyVitTest(std.testing.allocator, 2, 2, 2, 1);
}

test "Qwen38Vit toy tower: shape + determinism (merge=2, t=1, 4x4 grid)" {
    c.initErrorHandler();
    try toyVitTest(std.testing.allocator, 1, 4, 4, 2);
}

test "Qwen38Vit oracle parity vs mlx_vlm dump" {
    // Parity artifacts are produced by Task 6; skip gracefully until then.
    return error.SkipZigTest;
}
