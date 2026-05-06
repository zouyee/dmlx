/// SFT (Supervised Fine-Tuning) Trainer.
///
/// Minimal but complete training loop for LLaMA-style models.
///
/// Usage:
///   var trainer = try SFTTrainer.init(allocator, &model, &config, &optimizer);
///   try trainer.train(dataset, epochs, batch_size);
///
/// Features:
///   - Cross-entropy loss with label masking
///   - Per-step logging
///   - Checkpoint save/resume
///   - LR scheduling (cosine with warmup)
const std = @import("std");
const c = @import("c.zig");
const array_mod = @import("array.zig");
const ops = @import("ops.zig");
const reduce_mod = @import("ops/reduce.zig");
const loss_mod = @import("ops/loss.zig");
const grad_mod = @import("grad.zig");
const optim = @import("optim.zig");
const llama = @import("models/llama.zig");
const tree_mod = @import("tree.zig");
const lora_mod = @import("lora.zig");
const io = @import("io/mlx_io.zig");
const tokenizer_mod = @import("tokenizer.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const LlamaModel = llama.LlamaModel;
const AdamW = optim.AdamW;
const LoRAModel = lora_mod.LoRAModel;

/// A single training example.
pub const TrainingExample = struct {
    input_ids: []const u32,
    labels: []const i32,
};

/// Dataset interface.
pub const Dataset = struct {
    examples: []const TrainingExample,

    pub fn len(self: Dataset) usize {
        return self.examples.len;
    }

    pub fn deinit(self: Dataset, allocator: std.mem.Allocator) void {
        for (self.examples) |ex| {
            allocator.free(ex.input_ids);
            allocator.free(ex.labels);
        }
        allocator.free(self.examples);
    }
};

/// LR schedule type.
pub const LRSchedule = union(enum) {
    constant: struct { lr: f32 },
    cosine: struct { base_lr: f32, warmup_steps: usize, total_steps: usize, min_lr: f32 },
    linear: struct { base_lr: f32, warmup_steps: usize, total_steps: usize },
};

/// Training configuration.
pub const TrainerConfig = struct {
    max_seq_len: usize = 2048,
    gradient_accumulation_steps: usize = 1,
    log_interval: usize = 10,
    save_interval: usize = 500,
    checkpoint_dir: ?[]const u8 = null,
    lr_schedule: LRSchedule = .{ .constant = .{ .lr = 1e-4 } },
    clip_grad_norm: ?f32 = 1.0,
};

// ------------------------------------------------------------------
// Forward + Loss closure payload for value_and_grad
// ------------------------------------------------------------------

const ForwardLossPayload = struct {
    model: *LlamaModel,
    lora: ?*LoRAModel,
    ctx: EagerContext,
    allocator: std.mem.Allocator,
};

fn forwardLossCallback(
    res: [*c]c.c.mlx_vector_array,
    inputs: c.c.mlx_vector_array,
    payload: ?*anyopaque,
) callconv(.c) c_int {
    const p: *ForwardLossPayload = @ptrCast(@alignCast(payload.?));
    const n = c.c.mlx_vector_array_size(inputs);
    if (n < 2) return 1;

    var input_ids_handle = c.c.mlx_array_new();
    if (c.c.mlx_vector_array_get(&input_ids_handle, inputs, 0) != 0) return 1;
    const input_ids = array_mod.Array.fromHandle(input_ids_handle);

    var labels_handle = c.c.mlx_array_new();
    if (c.c.mlx_vector_array_get(&labels_handle, inputs, 1) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        return 1;
    }
    const labels = array_mod.Array.fromHandle(labels_handle);

    const n_params = n - 2;
    const params = p.allocator.alloc(array_mod.Array, n_params) catch {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    };
    defer p.allocator.free(params);

    for (0..n_params) |i| {
        var arr = c.c.mlx_array_new();
        if (c.c.mlx_vector_array_get(&arr, inputs, @intCast(i + 2)) != 0) {
            for (0..i) |j| _ = c.c.mlx_array_free(params[j].inner);
            _ = c.c.mlx_array_free(input_ids_handle);
            _ = c.c.mlx_array_free(labels_handle);
            return 1;
        }
        params[i] = array_mod.Array.fromHandle(arr);
    }
    defer {
        for (params) |arr| _ = c.c.mlx_array_free(arr.inner);
    }

    if (p.lora) |lora| {
        lora.setParams(params);
    } else {
        p.model.setParams(params);
    }

    const logits = p.model.forward(input_ids, null, null) catch {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    };
    defer _ = c.c.mlx_array_free(logits.inner);

    const shape = logits.shape();
    const vocab_size = shape[2];
    const bs = shape[0] * shape[1];

    var logits_2d: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_reshape(&logits_2d, logits.inner, &[_]c_int{ @intCast(bs), @intCast(vocab_size) }, 2, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(logits_2d);

    var labels_1d: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_reshape(&labels_1d, labels.inner, &[_]c_int{@intCast(bs)}, 1, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(labels_1d);

    // Masked cross-entropy via mlx-c ops:
    // 1. log_probs = logits - logsumexp(logits, axis=-1, keepdims=true)
    var lse: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_logsumexp_axis(&lse, logits_2d, -1, true, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(lse);

    var log_probs: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_subtract(&log_probs, logits_2d, lse, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(log_probs);

    // 2. Gather log-probabilities at label indices
    var labels_exp: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_expand_dims(&labels_exp, labels_1d, -1, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(labels_exp);

    var gathered: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_take_along_axis(&gathered, log_probs, labels_exp, -1, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(gathered);

    // squeeze last dim
    var squeezed: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_squeeze_axes(&squeezed, gathered, &[_]c_int{-1}, 1, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(squeezed);

    // 3. negative log-likelihood
    var neg_gathered: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_negative(&neg_gathered, squeezed, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(neg_gathered);

    // 4. Mask out -100 labels
    const neg100_data: i32 = -100;
    const mask_neg100: c.c.mlx_array = c.c.mlx_array_new_data(&neg100_data, &[_]c_int{1}, 1, c.c.MLX_INT32);
    defer _ = c.c.mlx_array_free(mask_neg100);

    var not_mask: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_not_equal(&not_mask, labels_1d, mask_neg100, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(not_mask);

    const zero_loss: c.c.mlx_array = c.c.mlx_array_new_float32(0.0);
    defer _ = c.c.mlx_array_free(zero_loss);

    var masked_loss: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_where(&masked_loss, not_mask, neg_gathered, zero_loss, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(masked_loss);

    // 5. mean over non-masked positions: sum(loss) / sum(mask)
    var loss_sum: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_sum(&loss_sum, masked_loss, false, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(loss_sum);

    var mask_f: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_astype(&mask_f, not_mask, c.c.MLX_FLOAT32, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(mask_f);

    var mask_count: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_sum(&mask_count, mask_f, false, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }
    defer _ = c.c.mlx_array_free(mask_count);

    var loss_arr: c.c.mlx_array = .{ .ctx = null };
    if (c.c.mlx_divide(&loss_arr, loss_sum, mask_count, p.ctx.stream.inner) != 0) {
        _ = c.c.mlx_array_free(input_ids_handle);
        _ = c.c.mlx_array_free(labels_handle);
        return 1;
    }

    const out_vec = c.c.mlx_vector_array_new_data(&loss_arr, 1);
    res.* = out_vec;
    return 0;
}

fn forwardLossDtor(payload: ?*anyopaque) callconv(.c) void {
    const p: *ForwardLossPayload = @ptrCast(@alignCast(payload.?));
    p.allocator.destroy(p);
}

/// SFT Trainer state.
pub const SFTTrainer = struct {
    allocator: std.mem.Allocator,
    model: *LlamaModel,
    optimizer: *AdamW,
    config: TrainerConfig,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    vg: grad_mod.ValueAndGradClosure,
    closure: c.c.mlx_closure,
    lora: ?*LoRAModel,

    global_step: usize,
    epoch: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        model: *LlamaModel,
        optimizer: *AdamW,
        train_config: TrainerConfig,
        ctx: EagerContext,
        stream: c.c.mlx_stream,
        lora: ?*LoRAModel,
    ) !SFTTrainer {
        const params = if (lora) |lm|
            try lm.collectParams(allocator)
        else
            try tree_mod.treeToArrays(allocator, model.*);
        defer allocator.free(params);
        const param_count = params.len;

        var argnums = try allocator.alloc(i32, param_count);
        defer allocator.free(argnums);
        for (0..param_count) |i| {
            argnums[i] = @intCast(i + 2);
        }

        const payload = try allocator.create(ForwardLossPayload);
        payload.* = .{
            .model = model,
            .lora = lora,
            .ctx = ctx,
            .allocator = allocator,
        };
        const closure = c.c.mlx_closure_new_func_payload(forwardLossCallback, payload, forwardLossDtor);
        const closure_wrapper = grad_mod.Closure{ .inner = closure };
        const vg = try grad_mod.valueAndGrad(closure_wrapper, argnums);

        return .{
            .allocator = allocator,
            .model = model,
            .optimizer = optimizer,
            .config = train_config,
            .ctx = ctx,
            .stream = stream,
            .vg = vg,
            .closure = closure,
            .lora = lora,
            .global_step = 0,
            .epoch = 0,
        };
    }

    pub fn deinit(self: *SFTTrainer) void {
        self.vg.deinit();
        _ = c.c.mlx_closure_free(self.closure);
    }

    pub fn train(self: *SFTTrainer, dataset: Dataset, epochs: usize, batch_size: usize) !void {
        std.log.info("Starting training: {d} examples, {d} epochs, batch_size={d}", .{ dataset.len(), epochs, batch_size });

        for (0..epochs) |epoch| {
            self.epoch = epoch;
            try self.trainEpoch(dataset, batch_size);
        }

        std.log.info("Training complete. Total steps: {d}", .{self.global_step});
    }

    pub fn trainEpoch(self: *SFTTrainer, dataset: Dataset, batch_size: usize) !void {
        const num_batches = (dataset.len() + batch_size - 1) / batch_size;
        var epoch_loss: f32 = 0;
        var num_loss_steps: usize = 0;

        for (0..num_batches) |batch_idx| {
            const start = batch_idx * batch_size;
            const end = @min(start + batch_size, dataset.len());

            var batch_loss: f32 = 0;
            for (dataset.examples[start..end]) |example| {
                const loss = try self.trainStep(example);
                batch_loss += loss;
            }
            const avg_loss = batch_loss / @as(f32, @floatFromInt(end - start));
            epoch_loss += avg_loss;
            num_loss_steps += 1;

            self.global_step += 1;

            if (self.global_step % self.config.log_interval == 0) {
                const lr = self.currentLR();
                std.log.info("Epoch {d} Step {d} | loss={d:.4} | lr={e:.4}", .{ self.epoch, self.global_step, avg_loss, lr });
            }

            if (self.config.checkpoint_dir) |dir| {
                if (self.global_step % self.config.save_interval == 0) {
                    try self.saveCheckpoint(dir);
                }
            }
        }

        const avg_epoch_loss = epoch_loss / @as(f32, @floatFromInt(num_loss_steps));
        std.log.info("Epoch {d} complete | avg_loss={d:.4}", .{ self.epoch, avg_epoch_loss });
    }

    pub fn trainStep(self: *SFTTrainer, example: TrainingExample) !f32 {
        if (example.input_ids.len < 2 or example.input_ids.len != example.labels.len) {
            return error.InvalidExample;
        }
        const seq_len = example.input_ids.len;
        const input_ids_flat = try self.arrayFromU32(example.input_ids[0 .. seq_len - 1]);
        defer input_ids_flat.deinit();
        const input_ids = try ops.reshape(self.ctx, input_ids_flat, &[_]i32{ 1, @intCast(seq_len - 1) });

        const labels_flat = try self.arrayFromI32(example.labels[1..seq_len]);
        defer labels_flat.deinit();
        const labels_arr = try ops.reshape(self.ctx, labels_flat, &[_]i32{ 1, @intCast(seq_len - 1) });

        const model_params = if (self.lora) |lm|
            try lm.collectParams(self.allocator)
        else
            try tree_mod.treeToArrays(self.allocator, self.model.*);
        defer self.allocator.free(model_params);

        var inputs = try self.allocator.alloc(Array, 2 + model_params.len);
        defer {
            for (inputs) |arr| arr.deinit();
            self.allocator.free(inputs);
        }

        inputs[0] = input_ids;
        inputs[1] = labels_arr;
        for (model_params, 0..) |param, i| {
            var copy = c.c.mlx_array_new();
            try c.check(c.c.mlx_array_set(&copy, param.inner));
            inputs[2 + i] = Array.fromHandle(copy);
        }

        const result = try self.vg.apply(inputs, self.allocator);
        defer {
            for (result.value) |arr| arr.deinit();
            self.allocator.free(result.value);
            for (result.grad) |arr| arr.deinit();
            self.allocator.free(result.grad);
        }

        if (result.value.len == 0) return error.NoLoss;
        const loss_val = result.value[0];
        try loss_val.eval();
        const loss_scalar = (try loss_val.dataPtr(f32))[0];

        if (self.config.clip_grad_norm) |max_norm| {
            // Compute global gradient norm: sqrt(sum(||grad_i||^2))
            var global_norm_sq: f32 = 0;
            for (result.grad) |grad| {
                const sq = try ops.multiply(self.ctx, grad, grad);
                defer sq.deinit();
                const sq_sum = try reduce_mod.sum(self.ctx, sq, false);
                defer sq_sum.deinit();
                try sq_sum.eval();
                const val = (try sq_sum.dataPtr(f32))[0];
                global_norm_sq += val;
            }
            const global_norm = @sqrt(global_norm_sq);
            if (global_norm > max_norm) {
                const scale = max_norm / global_norm;
                const scale_arr = try ops.scalarF32(self.ctx, scale);
                defer scale_arr.deinit();
                for (result.grad) |*grad| {
                    const scaled = try ops.multiply(self.ctx, grad.*, scale_arr);
                    grad.*.deinit();
                    grad.* = scaled;
                }
            }
        }

        try self.optimizer.step(result.grad, self.stream);

        return loss_scalar;
    }

    fn computeLoss(self: *SFTTrainer, logits: Array, labels: Array) !Array {
        const shape = logits.shape();
        const vocab_size = shape[2];
        const bs = shape[0] * shape[1];

        const logits_2d = try ops.reshape(self.ctx, logits, &[_]i32{ bs, vocab_size });
        defer logits_2d.deinit();

        const labels_1d = try ops.reshape(self.ctx, labels, &[_]i32{bs});
        defer labels_1d.deinit();

        return loss_mod.crossEntropyGraph(self.ctx, logits_2d, labels_1d);
    }

    fn currentLR(self: *SFTTrainer) f32 {
        return switch (self.config.lr_schedule) {
            .constant => |cfg| cfg.lr,
            .cosine => |cfg| lrCosine(cfg.base_lr, cfg.warmup_steps, cfg.total_steps, self.global_step, cfg.min_lr),
            .linear => |cfg| lrLinear(cfg.base_lr, cfg.warmup_steps, cfg.total_steps, self.global_step),
        };
    }

    pub fn saveCheckpoint(self: *SFTTrainer, dir: []const u8) !void {
        const path = try std.fmt.allocPrint(self.allocator, "{s}/checkpoint_{d}.safetensors", .{ dir, self.global_step });
        defer self.allocator.free(path);

        std.log.info("Saving checkpoint to {s}", .{path});

        var weights = std.StringHashMap(Array).init(self.allocator);
        defer {
            var it = weights.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                entry.value_ptr.*.deinit();
            }
            weights.deinit();
        }

        const entries = try tree_mod.flattenStruct(self.allocator, self.model.*);
        defer tree_mod.freeEntries(self.allocator, entries);

        for (entries) |entry| {
            var copied = c.c.mlx_array_new();
            try c.check(c.c.mlx_array_set(&copied, entry.value.inner));
            try weights.put(entry.key, Array.fromHandle(copied));
        }

        var metadata = std.StringHashMap([]const u8).init(self.allocator);
        defer metadata.deinit();

        try io.saveSafetensors(self.allocator, path, weights, metadata);
    }

    pub fn loadCheckpoint(self: *SFTTrainer, dir: []const u8, step: usize) !void {
        const path = try std.fmt.allocPrint(self.allocator, "{s}/checkpoint_{d}.safetensors", .{ dir, step });
        defer self.allocator.free(path);

        std.log.info("Loading checkpoint from {s}", .{path});

        const loaded = try io.loadSafetensors(self.allocator, path);
        defer loaded.deinit(self.allocator);

        const entries = try tree_mod.flattenStruct(self.allocator, self.model.*);
        defer tree_mod.freeEntries(self.allocator, entries);

        for (entries) |entry| {
            if (loaded.weights.get(entry.key)) |weight| {
                _ = c.c.mlx_array_free(entry.value.inner);
                var copied = c.c.mlx_array_new();
                try c.check(c.c.mlx_array_set(&copied, weight.inner));
                entry.value.inner = copied;
            }
        }
    }

    fn arrayFromU32(_: *SFTTrainer, data: []const u32) !Array {
        const arr = c.c.mlx_array_new_data(
            data.ptr,
            &[_]c_int{@intCast(data.len)},
            1,
            c.c.MLX_UINT32,
        );
        return Array.fromHandle(arr);
    }

    fn arrayFromI32(_: *SFTTrainer, data: []const i32) !Array {
        const arr = c.c.mlx_array_new_data(
            data.ptr,
            &[_]c_int{@intCast(data.len)},
            1,
            c.c.MLX_INT32,
        );
        return Array.fromHandle(arr);
    }
};

fn lrCosine(base_lr: f32, warmup_steps: usize, total_steps: usize, current_step: usize, min_lr: f32) f32 {
    if (current_step < warmup_steps) {
        return base_lr * @as(f32, @floatFromInt(current_step)) / @as(f32, @floatFromInt(warmup_steps));
    }

    const progress = @as(f32, @floatFromInt(current_step - warmup_steps)) /
        @as(f32, @floatFromInt(total_steps - warmup_steps));
    const cosine_decay = 0.5 * (1.0 + @cos(std.math.pi * progress));
    return min_lr + (base_lr - min_lr) * cosine_decay;
}

fn lrLinear(base_lr: f32, warmup_steps: usize, total_steps: usize, current_step: usize) f32 {
    if (current_step < warmup_steps) {
        return base_lr * @as(f32, @floatFromInt(current_step)) / @as(f32, @floatFromInt(warmup_steps));
    }

    const progress = @as(f32, @floatFromInt(current_step - warmup_steps)) /
        @as(f32, @floatFromInt(total_steps - warmup_steps));
    return base_lr * (1.0 - progress);
}

pub fn loadJsonlDataset(
    allocator: std.mem.Allocator,
    io_ctx: std.Io,
    path: []const u8,
    tokenizer: tokenizer_mod.TokenizerStrategy,
) !Dataset {
    const cwd = std.Io.Dir.cwd();
    const content = try cwd.readFileAlloc(io_ctx, path, allocator, .limited(100 * 1024 * 1024));
    defer allocator.free(content);

    var examples = std.ArrayList(TrainingExample).empty;
    errdefer {
        for (examples.items) |ex| {
            allocator.free(ex.input_ids);
            allocator.free(ex.labels);
        }
        examples.deinit(allocator);
    }

    var line_it = std.mem.splitScalar(u8, content, '\n');
    while (line_it.next()) |line| {
        if (line.len == 0) continue;
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, line, .{});
        defer parsed.deinit();

        const text_val = parsed.value.object.get("text") orelse continue;
        const text = text_val.string;

        const tokens = try tokenizer.encode(text, false, allocator);
        if (tokens.len < 2) {
            allocator.free(tokens);
            continue;
        }
        errdefer allocator.free(tokens);

        const labels = try allocator.alloc(i32, tokens.len);
        errdefer allocator.free(labels);
        for (tokens, 0..) |t, i| {
            labels[i] = @intCast(t);
        }

        try examples.append(allocator, .{
            .input_ids = tokens,
            .labels = labels,
        });
    }

    const exs = try examples.toOwnedSlice(allocator);
    return Dataset{ .examples = exs };
}
