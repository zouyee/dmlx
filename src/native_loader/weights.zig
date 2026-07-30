/// MLX-free weight loader for DeepSeek-V4-Flash.
/// Reads safetensors directly, builds EngineWeights for engine.c.
/// No MLX dependency.
const std = @import("std");
const safetensors = @import("safetensors.zig");
const config_mod = @import("config.zig");

const TensorIndex = safetensors.TensorIndex;
const DSV4NativeConfig = config_mod.DSV4NativeConfig;

/// Quantized weight pointers (matches DSV4Model.QuantWeightPtrs in deepseek_v4.zig).
pub const QuantWeightPtrs = struct {
    packed_ptr: []const u32,
    scales: []const f32,
    biases: []const f32,
    out_dim: i32,
    in_dim: i32,
    group_size: i32,
};

/// MLA attention weight pointers (matches DSV4Model.AttnWeightPtrs).
pub const AttnWeightPtrs = struct {
    wq_a: QuantWeightPtrs,
    q_norm: []const f32,
    wq_b: QuantWeightPtrs,
    wkv: QuantWeightPtrs,
    kv_norm: []const f32,
    wo_a_dense: []const f32,
    wo_a: QuantWeightPtrs,
    wo_b: QuantWeightPtrs,
    attn_sink: []const f32,
};

/// Engine weights (matches DSV4Model.EngineWeights).
pub const EngineWeights = struct {
    embed: []const f32,
    lm_head: []const f32,
    // Quantized (native 4-bit affine) forms — kept packed by default to save
    // 3.2GB RAM; the f32 dense forms are loaded only as fallback
    // (DMLX_DENSE_HEAD=1 or packed tensors missing).
    embed_q: QuantWeightPtrs = .{ .packed_ptr = &[_]u32{}, .scales = &[_]f32{}, .biases = &[_]f32{}, .out_dim = 0, .in_dim = 0, .group_size = 64 },
    lm_head_q: QuantWeightPtrs = .{ .packed_ptr = &[_]u32{}, .scales = &[_]f32{}, .biases = &[_]f32{}, .out_dim = 0, .in_dim = 0, .group_size = 64 },
    final_norm: []const f32,
    input_norms: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    attn_norms: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    gate_projs: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    gate_biases: [64]?[]const f32 = [_]?[]const f32{null} ** 64,
    tid2eid: [64]?[]const i64 = [_]?[]const i64{null} ** 64,
    attn: [64]?AttnWeightPtrs = [_]?AttnWeightPtrs{null} ** 64,
    shared_gate: [64]QuantWeightPtrs = [_]QuantWeightPtrs{.{ .packed_ptr = &[_]u32{}, .scales = &[_]f32{}, .biases = &[_]f32{}, .out_dim = 0, .in_dim = 0, .group_size = 64 }} ** 64,
    shared_up: [64]QuantWeightPtrs = [_]QuantWeightPtrs{.{ .packed_ptr = &[_]u32{}, .scales = &[_]f32{}, .biases = &[_]f32{}, .out_dim = 0, .in_dim = 0, .group_size = 64 }} ** 64,
    shared_down: [64]QuantWeightPtrs = [_]QuantWeightPtrs{.{ .packed_ptr = &[_]u32{}, .scales = &[_]f32{}, .biases = &[_]f32{}, .out_dim = 0, .in_dim = 0, .group_size = 64 }} ** 64,
    attn_hc_fn: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    attn_hc_base: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    attn_hc_scale: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    ffn_hc_fn: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    ffn_hc_base: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    ffn_hc_scale: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    hc_head_fn: []const f32 = &[_]f32{},
    hc_head_base: []const f32 = &[_]f32{},
    hc_head_scale: []const f32 = &[_]f32{},
    // Compressor/Indexer weights (T2C.1)
    compress_ratio: [64]u32 = [_]u32{0} ** 64,
    // Compressor weights (layers with compress_ratio > 0)
    comp_wkv: [64]?QuantWeightPtrs = [_]?QuantWeightPtrs{null} ** 64,
    comp_wgate: [64]?QuantWeightPtrs = [_]?QuantWeightPtrs{null} ** 64,
    comp_ape: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    comp_norm: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    // Indexer weights (only ratio=4 layers: 2,4,6,...,42)
    idx_wq_b: [64]?QuantWeightPtrs = [_]?QuantWeightPtrs{null} ** 64,
    idx_weights_proj: [64]?QuantWeightPtrs = [_]?QuantWeightPtrs{null} ** 64,
    idx_comp_wkv: [64]?QuantWeightPtrs = [_]?QuantWeightPtrs{null} ** 64,
    idx_comp_wgate: [64]?QuantWeightPtrs = [_]?QuantWeightPtrs{null} ** 64,
    idx_comp_ape: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    idx_comp_norm: [64][]const f32 = [_][]const f32{&[_]f32{}} ** 64,
    n_layers: usize = 0,
};

/// All data buffers owned by NativeWeightStore, freed via arena.
pub const NativeWeightStore = struct {
    arena: std.heap.ArenaAllocator,
    weights: EngineWeights,
    /// Dense F32 embed lookup table [vocab_size, hidden_size] (empty when quantized)
    embed_f32: []const f32,
    /// Dense F32 lm_head table [vocab_size, hidden_size] (empty when quantized)
    lm_head_f32: []const f32,

    pub fn deinit(self: *NativeWeightStore) void {
        self.arena.deinit();
    }
};

/// Monotonic millisecond timestamp.
fn nowMs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @intCast(ts.sec * 1000 + @divTrunc(ts.nsec, 1_000_000));
}

/// Load all weights from safetensors index into engine-ready structures.
/// Returns a NativeWeightStore that owns all memory via an arena.
/// progress_fn: optional callback called each layer, fn(layer: usize, total: usize) void
/// timeout_ms: 0 = no timeout; otherwise returns error.LoadTimeout if exceeded
pub fn loadAll(
    allocator: std.mem.Allocator,
    idx: *const TensorIndex,
    cfg: DSV4NativeConfig,
    progress_fn: ?*const fn (layer: usize, total: usize) void,
    timeout_ms: u64,
) !NativeWeightStore {
    const t_start = nowMs();
    var arena = std.heap.ArenaAllocator.init(allocator);
    const a = arena.allocator();
    errdefer arena.deinit();

    var w: EngineWeights = undefined;
    // Initialize all nullable/slice fields to empty/null
    w.attn = [_]?AttnWeightPtrs{null} ** 64;
    w.tid2eid = [_]?[]const i64{null} ** 64;
    w.gate_biases = [_]?[]const f32{null} ** 64;
    w.input_norms = [_][]const f32{&.{}} ** 64;
    w.attn_norms = [_][]const f32{&.{}} ** 64;
    w.gate_projs = [_][]const f32{&.{}} ** 64;
    const empty_qw = QuantWeightPtrs{
        .packed_ptr = &.{},
        .scales = &.{},
        .biases = &.{},
        .out_dim = 0,
        .in_dim = 0,
        .group_size = 64,
    };
    w.shared_gate = [_]QuantWeightPtrs{empty_qw} ** 64;
    w.shared_up = [_]QuantWeightPtrs{empty_qw} ** 64;
    w.shared_down = [_]QuantWeightPtrs{empty_qw} ** 64;
    w.attn_hc_fn = [_][]const f32{&.{}} ** 64;
    w.attn_hc_base = [_][]const f32{&.{}} ** 64;
    w.attn_hc_scale = [_][]const f32{&.{}} ** 64;
    w.ffn_hc_fn = [_][]const f32{&.{}} ** 64;
    w.ffn_hc_base = [_][]const f32{&.{}} ** 64;
    w.ffn_hc_scale = [_][]const f32{&.{}} ** 64;
    w.n_layers = cfg.num_hidden_layers;
    // Compressor/Indexer init (T2C.1)
    w.compress_ratio = [_]u32{0} ** 64;
    w.comp_wkv = [_]?QuantWeightPtrs{null} ** 64;
    w.comp_wgate = [_]?QuantWeightPtrs{null} ** 64;
    w.comp_ape = [_][]const f32{&.{}} ** 64;
    w.comp_norm = [_][]const f32{&.{}} ** 64;
    w.idx_wq_b = [_]?QuantWeightPtrs{null} ** 64;
    w.idx_weights_proj = [_]?QuantWeightPtrs{null} ** 64;
    w.idx_comp_wkv = [_]?QuantWeightPtrs{null} ** 64;
    w.idx_comp_wgate = [_]?QuantWeightPtrs{null} ** 64;
    w.idx_comp_ape = [_][]const f32{&.{}} ** 64;
    w.idx_comp_norm = [_][]const f32{&.{}} ** 64;

    // =========================================================================
    // Embed tokens: keep native packed affine 4-bit (row dequant on read,
    // bit-exact by construction). Dense f32 (2.1GB) only as fallback.
    // =========================================================================
    const empty_q = QuantWeightPtrs{ .packed_ptr = &[_]u32{}, .scales = &[_]f32{}, .biases = &[_]f32{}, .out_dim = 0, .in_dim = 0, .group_size = 64 };
    const want_dense_head = std.c.getenv("DMLX_DENSE_HEAD") != null;
    w.embed_q = loadQuantWeight(a, idx, "model.embed_tokens", 64) catch empty_q;
    w.lm_head_q = loadQuantWeight(a, idx, "lm_head", 64) catch empty_q;
    const need_dense = want_dense_head or w.embed_q.packed_ptr.len == 0 or w.lm_head_q.packed_ptr.len == 0;
    if (need_dense) {
        const embed_f32 = try dequantAffineF32(a, idx, "model.embed_tokens", cfg.vocab_size, cfg.hidden_size, 64);
        w.embed = embed_f32;
        const lm_head_f32 = try dequantAffineF32(a, idx, "lm_head", cfg.vocab_size, cfg.hidden_size, 64);
        w.lm_head = lm_head_f32;
    } else {
        w.embed = &[_]f32{};
        w.lm_head = &[_]f32{};
    }

    // =========================================================================
    // Final norm (BF16 → F32)
    // =========================================================================
    w.final_norm = try idx.loadBF16AsF32("model.norm.weight", a);

    // =========================================================================
    // Per-layer weights
    // =========================================================================
    var name_buf: [256]u8 = undefined;

    for (0..cfg.num_hidden_layers) |i| {
        // Timeout check
        if (timeout_ms > 0) {
            const elapsed = nowMs() - t_start;
            if (elapsed > timeout_ms) return error.LoadTimeout;
        }
        // Progress callback
        if (progress_fn) |cb| cb(i, cfg.num_hidden_layers);

        // ---- RMSNorm weights ----
        w.input_norms[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "model.layers.{d}.attn_norm.weight", .{i}), a);
        w.attn_norms[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "model.layers.{d}.ffn_norm.weight", .{i}), a);

        // ---- Router gate (BF16 dense → F32) ----
        w.gate_projs[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "model.layers.{d}.ffn.gate.weight", .{i}), a);

        // e_score_correction_bias (layers 3-42, not present on hash layers 0-2)
        const bias_name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.ffn.gate.e_score_correction_bias", .{i});
        if (idx.get(bias_name) != null) {
            w.gate_biases[i] = try idx.loadBF16AsF32(bias_name, a);
        }

        // tid2eid (hash routing, layers 0-2)
        const tid2eid_name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.ffn.gate.tid2eid", .{i});
        if (idx.get(tid2eid_name) != null) {
            w.tid2eid[i] = try idx.loadI64(tid2eid_name, a);
        }

        // ---- MLA attention weights (affine quantized) ----
        var ap: AttnWeightPtrs = undefined;
        const layer_prefix = try std.fmt.allocPrint(a, "model.layers.{d}.attn", .{i});

        ap.wq_a = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.wq_a", .{layer_prefix}), 64);
        ap.q_norm = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.q_norm.weight", .{layer_prefix}), a);
        ap.wq_b = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.wq_b", .{layer_prefix}), 64);
        ap.wkv = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.wkv", .{layer_prefix}), 64);
        ap.kv_norm = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.kv_norm.weight", .{layer_prefix}), a);

        // wo_a: keep native packed affine 4-bit — the GPU dequants in-kernel.
        // The f32 dense form costs 128MB/layer of per-token GPU reads (evicted
        // and refaulted under memory pressure) vs 16MB packed, so dense is now
        // only a fallback (DMLX_WOA_F32 / DMLX_USE_Q8_WOA) or used when the
        // packed tensors are missing.
        ap.wo_a = loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.wo_a", .{layer_prefix}), 64) catch
            QuantWeightPtrs{
                .packed_ptr = &[_]u32{},
                .scales = &[_]f32{},
                .biases = &[_]f32{},
                .out_dim = 0,
                .in_dim = 0,
                .group_size = 64,
            };
        const want_dense = std.c.getenv("DMLX_WOA_F32") != null or
            std.c.getenv("DMLX_USE_Q8_WOA") != null or
            ap.wo_a.packed_ptr.len == 0;
        if (want_dense) {
            // Shape: [8192, 512] U32 = 8 groups × (N_HEADS/O_GROUPS) × HEAD_DIM × (in_dim/8)
            // in_dim = 4096, out_dim = 8192 (= O_GROUPS * O_LORA_RANK = 8 * 1024)
            // After dequant: [8192, 4096] F32
            ap.wo_a_dense = try dequantAffineF32(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.wo_a", .{layer_prefix}), 8192, 4096, 64);
        } else {
            ap.wo_a_dense = &[_]f32{};
        }

        ap.wo_b = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.wo_b", .{layer_prefix}), 64);

        // attn_sink: F32 [64]
        const sink_name = try std.fmt.bufPrint(&name_buf, "{s}.attn_sink", .{layer_prefix});
        if (idx.get(sink_name)) |info| {
            if (info.dtype == .F32) {
                ap.attn_sink = try idx.loadF32(sink_name, a);
            } else {
                ap.attn_sink = try idx.loadBF16AsF32(sink_name, a);
            }
        } else {
            ap.attn_sink = &.{};
        }
        w.attn[i] = ap;

        // ---- Shared expert (affine quantized, gs=64) ----
        const se_prefix = try std.fmt.allocPrint(a, "model.layers.{d}.ffn.shared_experts", .{i});
        w.shared_gate[i] = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.gate_proj", .{se_prefix}), 64);
        w.shared_up[i] = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.up_proj", .{se_prefix}), 64);
        w.shared_down[i] = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.down_proj", .{se_prefix}), 64);

        // ---- mHC weights (BF16 → F32) ----
        const ahc = try std.fmt.allocPrint(a, "model.layers.{d}.attn_hc", .{i});
        w.attn_hc_fn[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.fn", .{ahc}), a);
        w.attn_hc_base[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.base", .{ahc}), a);
        w.attn_hc_scale[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.scale", .{ahc}), a);

        const fhc = try std.fmt.allocPrint(a, "model.layers.{d}.ffn_hc", .{i});
        w.ffn_hc_fn[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.fn", .{fhc}), a);
        w.ffn_hc_base[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.base", .{fhc}), a);
        w.ffn_hc_scale[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.scale", .{fhc}), a);

        // ---- Compressor/Indexer weights (T2C.1) ----
        const ratio = cfg.compress_ratios[i];
        w.compress_ratio[i] = ratio;
        if (ratio > 0) {
            const comp_base = try std.fmt.allocPrint(a, "model.layers.{d}.attn.compressor", .{i});
            w.comp_wkv[i] = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.wkv", .{comp_base}), 64);
            w.comp_wgate[i] = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.wgate", .{comp_base}), 64);
            w.comp_ape[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.ape", .{comp_base}), a);
            w.comp_norm[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.norm.weight", .{comp_base}), a);

            if (ratio == 4) {
                const idx_base = try std.fmt.allocPrint(a, "model.layers.{d}.attn.indexer", .{i});
                w.idx_wq_b[i] = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.wq_b", .{idx_base}), 64);
                w.idx_weights_proj[i] = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.weights_proj", .{idx_base}), 64);
                const ic_base = try std.fmt.allocPrint(a, "{s}.compressor", .{idx_base});
                w.idx_comp_wkv[i] = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.wkv", .{ic_base}), 64);
                w.idx_comp_wgate[i] = try loadQuantWeight(a, idx, try std.fmt.bufPrint(&name_buf, "{s}.wgate", .{ic_base}), 64);
                w.idx_comp_ape[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.ape", .{ic_base}), a);
                w.idx_comp_norm[i] = try idx.loadBF16AsF32(try std.fmt.bufPrint(&name_buf, "{s}.norm.weight", .{ic_base}), a);
            }
        }
    }

    // ---- HyperHead weights (BF16 -> F32) ----
    w.hc_head_fn = try idx.loadBF16AsF32("model.hc_head.fn", a);
    w.hc_head_base = try idx.loadBF16AsF32("model.hc_head.base", a);
    w.hc_head_scale = try idx.loadBF16AsF32("model.hc_head.scale", a);

    if (cfg.num_hidden_layers > 0) {
        std.log.info("native_loader: layer 0 loaded ok, attn_sink.len={d}", .{
            if (w.attn[0]) |ap_| ap_.attn_sink.len else 0,
        });
    }

    return NativeWeightStore{
        .arena = arena,
        .weights = w,
        .embed_f32 = w.embed,
        .lm_head_f32 = w.lm_head,
    };
}

// ============================================================================
// Helpers
// ============================================================================

/// Load an affine-quantized weight as packed U32 + F32 scales + F32 biases.
/// base_name: e.g. "model.layers.0.attn.wq_a"
/// Looks for base_name.weight (U32), base_name.scales (BF16→F32), base_name.biases (BF16→F32).
fn loadQuantWeight(
    a: std.mem.Allocator,
    idx: *const TensorIndex,
    base_name: []const u8,
    group_size: i32,
) !QuantWeightPtrs {
    var buf: [512]u8 = undefined;

    const weight_name = try std.fmt.bufPrint(&buf, "{s}.weight", .{base_name});
    const info = idx.get(weight_name) orelse return error.TensorNotFound;

    // out_dim = shape[0], in_dim = shape[1] * 8 (each u32 holds 8 nibbles)
    const out_dim: i32 = @intCast(info.shape[0]);
    const in_dim: i32 = @intCast(info.shape[1] * 8);

    const packed_data = try idx.loadU32(weight_name, a);

    const scales_name = try std.fmt.bufPrint(&buf, "{s}.scales", .{base_name});
    const scales = if (idx.get(scales_name) != null)
        try idx.loadBF16AsF32(scales_name, a)
    else
        &[_]f32{};

    var buf2: [512]u8 = undefined;
    const biases_name = try std.fmt.bufPrint(&buf2, "{s}.biases", .{base_name});
    const biases = if (idx.get(biases_name) != null)
        try idx.loadBF16AsF32(biases_name, a)
    else
        &[_]f32{};

    return QuantWeightPtrs{
        .packed_ptr = packed_data,
        .scales = scales,
        .biases = biases,
        .out_dim = out_dim,
        .in_dim = in_dim,
        .group_size = group_size,
    };
}

/// Dequantize an affine-4bit weight to dense F32.
/// base_name: e.g. "model.embed_tokens"
/// Returns [out_dim * in_dim] F32 (row-major).
/// Formula: w[row, col] = scales[row, col/gs] * nibble[row, col] + biases[row, col/gs]
fn dequantAffineF32(
    a: std.mem.Allocator,
    idx: *const TensorIndex,
    base_name: []const u8,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
) ![]f32 {
    var buf: [512]u8 = undefined;

    const weight_name = try std.fmt.bufPrint(&buf, "{s}.weight", .{base_name});
    const packed_data = try idx.loadU32(weight_name, a);
    defer a.free(packed_data);

    var buf2: [512]u8 = undefined;
    const scales_name = try std.fmt.bufPrint(&buf2, "{s}.scales", .{base_name});
    const scales = try idx.loadBF16AsF32(scales_name, a);
    defer a.free(scales);

    var buf3: [512]u8 = undefined;
    const biases_name = try std.fmt.bufPrint(&buf3, "{s}.biases", .{base_name});
    const biases = if (idx.get(biases_name) != null)
        try idx.loadBF16AsF32(biases_name, a)
    else
        try a.alloc(f32, 0);
    defer a.free(biases);

    const out = try a.alloc(f32, out_dim * in_dim);
    const num_groups = in_dim / group_size;
    const packed_per_group = group_size / 8;
    const packed_cols = in_dim / 8;

    for (0..out_dim) |row| {
        const packed_row = packed_data[row * packed_cols ..][0..packed_cols];
        const scale_row = scales[row * num_groups ..][0..num_groups];
        const bias_row = if (biases.len > 0) biases[row * num_groups ..][0..num_groups] else null;

        for (0..num_groups) |g| {
            const scale = scale_row[g];
            const bias = if (bias_row) |br| br[g] else 0.0;
            const col_base = g * group_size;

            for (0..packed_per_group) |p| {
                const pw = packed_row[g * packed_per_group + p];
                for (0..8) |k| {
                    const nibble: f32 = @floatFromInt((pw >> @intCast(k * 4)) & 0xF);
                    out[row * in_dim + col_base + p * 8 + k] = scale * nibble + bias;
                }
            }
        }
    }
    return out;
}
