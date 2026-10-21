/// Loss functions for MLX-like operations
/// Provides cross_entropy, BCE, MSE, L1, Huber loss functions
/// All loss functions use mlx-c operator graph composition for GPU acceleration.
const std = @import("std");
const array_mod = @import("../array.zig");
const dtype_mod = @import("../dtype.zig");
const ops_mod = @import("../ops.zig");
const comparison_mod = @import("comparison.zig");
const shape_mod = @import("shape.zig");
const reduce_mod = @import("reduce.zig");
const c = @import("../c.zig");

pub const Array = array_mod.Array;
pub const Dtype = dtype_mod.Dtype;
const ShapeElem = array_mod.ShapeElem;
const EagerContext = ops_mod.EagerContext;

// ============================================================
// Cross Entropy Loss (Graph-based, differentiable)
// ============================================================

/// Graph-based cross-entropy loss suitable for autodiff.
/// logits: [B*S, V] float32
/// labels: [B*S] int32 with -100 for masked positions
pub fn crossEntropyGraph(ctx: EagerContext, logits: Array, labels: Array) !Array {
    // 1. Create mask: labels != -100
    const neg_100 = try ops_mod.scalarI32(ctx, -100);
    defer neg_100.deinit();
    const mask_arr = try comparison_mod.notEqual(ctx, labels, neg_100);
    defer mask_arr.deinit();

    // 2. Replace -100 with 0 to avoid invalid indices for take_along_axis
    const zero_label = try ops_mod.scalarI32(ctx, 0);
    defer zero_label.deinit();
    const safe_labels = try ops_mod.where(ctx, mask_arr, labels, zero_label);
    defer safe_labels.deinit();

    // 3. Stable log-softmax: logits - logsumexp(logits, axis=-1, keepdims=true)
    const log_sum_exp = try reduce_mod.logsumexpAxis(ctx, logits, -1, true);
    defer log_sum_exp.deinit();
    const log_probs = try ops_mod.subtract(ctx, logits, log_sum_exp);
    defer log_probs.deinit();

    // 4. Gather log-probabilities of correct labels: [B*S, 1] -> squeeze -> [B*S]
    const labels_expanded = try shape_mod.expandDims(ctx, safe_labels, -1);
    defer labels_expanded.deinit();
    const gathered = try shape_mod.takeAlongAxis(ctx, log_probs, labels_expanded, -1);
    defer gathered.deinit();
    const gathered_squeezed = try shape_mod.squeezeAxis(ctx, gathered, -1);
    defer gathered_squeezed.deinit();

    // 5. Mask out ignored positions
    const neg_gathered = try ops_mod.negative(ctx, gathered_squeezed);
    defer neg_gathered.deinit();
    const zero_loss = try ops_mod.scalarF32(ctx, 0.0);
    defer zero_loss.deinit();
    const masked_loss = try ops_mod.where(ctx, mask_arr, neg_gathered, zero_loss);
    defer masked_loss.deinit();

    // 6. Mean over non-masked positions: sum(loss) / sum(mask)
    const loss_sum = try reduce_mod.sum(ctx, masked_loss, false);
    defer loss_sum.deinit();
    const mask_float = try shape_mod.astype(ctx, mask_arr, .float32);
    defer mask_float.deinit();
    const mask_count = try reduce_mod.sum(ctx, mask_float, false);
    defer mask_count.deinit();

    return ops_mod.divide(ctx, loss_sum, mask_count);
}

// ============================================================
// Cross Entropy Loss (Eager CPU fallback)
// ============================================================

pub fn crossEntropy(ctx: EagerContext, logits: Array, labels: Array) !Array {
    // Graph-mode: logits (batch, num_classes), labels (batch,) int32
    // log_softmax → gather → negate → mean
    const log_sum_exp = try reduce_mod.logsumexpAxis(ctx, logits, -1, true);
    defer log_sum_exp.deinit();
    const log_probs = try ops_mod.subtract(ctx, logits, log_sum_exp);
    defer log_probs.deinit();

    const labels_exp = try shape_mod.expandDims(ctx, labels, -1);
    defer labels_exp.deinit();
    const gathered = try shape_mod.takeAlongAxis(ctx, log_probs, labels_exp, -1);
    defer gathered.deinit();
    const gathered_sq = try shape_mod.squeezeAxis(ctx, gathered, -1);
    defer gathered_sq.deinit();

    const neg = try ops_mod.negative(ctx, gathered_sq);
    defer neg.deinit();
    return reduce_mod.mean(ctx, neg, false);
}

// ============================================================
// Binary Cross Entropy Loss
// ============================================================

pub fn binaryCrossEntropy(ctx: EagerContext, logits: Array, targets: Array) !Array {
    // Graph-mode BCE: -mean(y*log(x) + (1-y)*log(1-x))
    const eps = try ops_mod.scalarF32(ctx, 1e-7);
    defer eps.deinit();
    const one = try ops_mod.scalarF32(ctx, 1.0);
    defer one.deinit();
    const one_minus_eps = try ops_mod.subtract(ctx, one, eps);
    defer one_minus_eps.deinit();

    // Clip logits to [eps, 1-eps]
    var clipped = c.c.mlx_array_new();
    try c.check(c.c.mlx_clip(&clipped, logits.inner, eps.inner, one_minus_eps.inner, ctx.stream.inner));
    const clipped_arr = Array.fromHandle(clipped);
    defer clipped_arr.deinit();

    // y * log(x)
    const log_x = try ops_mod.log(ctx, clipped_arr);
    defer log_x.deinit();
    const y_log_x = try ops_mod.multiply(ctx, targets, log_x);
    defer y_log_x.deinit();

    // (1-y) * log(1-x)
    const one_minus_y = try ops_mod.subtract(ctx, one, targets);
    defer one_minus_y.deinit();
    const one_minus_x = try ops_mod.subtract(ctx, one, clipped_arr);
    defer one_minus_x.deinit();
    const log_one_minus_x = try ops_mod.log(ctx, one_minus_x);
    defer log_one_minus_x.deinit();
    const second_term = try ops_mod.multiply(ctx, one_minus_y, log_one_minus_x);
    defer second_term.deinit();

    // -(y*log(x) + (1-y)*log(1-x))
    const sum_terms = try ops_mod.add(ctx, y_log_x, second_term);
    defer sum_terms.deinit();
    const neg_sum = try ops_mod.negative(ctx, sum_terms);
    defer neg_sum.deinit();

    return reduce_mod.mean(ctx, neg_sum, false);
}

// ============================================================
// Mean Squared Error Loss
// ============================================================

pub fn mseLoss(ctx: EagerContext, predictions: Array, targets: Array) !Array {
    // Graph-mode MSE: mean((pred - target)^2)
    const diff = try ops_mod.subtract(ctx, predictions, targets);
    defer diff.deinit();
    const sq = try ops_mod.multiply(ctx, diff, diff);
    defer sq.deinit();
    return reduce_mod.mean(ctx, sq, false);
}

// ============================================================
// L1 Loss (Mean Absolute Error)
// ============================================================

pub fn l1Loss(ctx: EagerContext, predictions: Array, targets: Array) !Array {
    // Graph-mode L1: mean(|pred - target|)
    const diff = try ops_mod.subtract(ctx, predictions, targets);
    defer diff.deinit();
    const abs_diff = try ops_mod.abs(ctx, diff);
    defer abs_diff.deinit();
    return reduce_mod.mean(ctx, abs_diff, false);
}

// ============================================================
// Huber Loss
// ============================================================

pub fn huberLoss(ctx: EagerContext, predictions: Array, targets: Array, delta: f32) !Array {
    // Graph-mode Huber: mean(where(|d| <= delta, 0.5*d^2, delta*(|d| - 0.5*delta)))
    const diff = try ops_mod.subtract(ctx, predictions, targets);
    defer diff.deinit();
    const abs_diff = try ops_mod.abs(ctx, diff);
    defer abs_diff.deinit();

    const delta_s = try ops_mod.scalarF32(ctx, delta);
    defer delta_s.deinit();
    const half = try ops_mod.scalarF32(ctx, 0.5);
    defer half.deinit();
    const half_delta = try ops_mod.scalarF32(ctx, 0.5 * delta);
    defer half_delta.deinit();

    // Quadratic branch: 0.5 * diff^2
    const sq = try ops_mod.multiply(ctx, diff, diff);
    defer sq.deinit();
    const quad = try ops_mod.multiply(ctx, half, sq);
    defer quad.deinit();

    // Linear branch: delta * (|diff| - 0.5 * delta)
    const shifted = try ops_mod.subtract(ctx, abs_diff, half_delta);
    defer shifted.deinit();
    const linear = try ops_mod.multiply(ctx, delta_s, shifted);
    defer linear.deinit();

    // Condition: |diff| <= delta
    const cond = try comparison_mod.lessEqual(ctx, abs_diff, delta_s);
    defer cond.deinit();

    const per_elem = try ops_mod.where(ctx, cond, quad, linear);
    defer per_elem.deinit();

    return reduce_mod.mean(ctx, per_elem, false);
}

// ============================================================
// KL Divergence Loss
// ============================================================

pub fn klDivLoss(ctx: EagerContext, log_preds: Array, targets: Array) !Array {
    // Graph-mode KL divergence: sum(target * (log(target) - log_pred))
    const eps = try ops_mod.scalarF32(ctx, 1e-7);
    defer eps.deinit();

    // Clamp targets to avoid log(0)
    var clamped = c.c.mlx_array_new();
    const inf_s = try ops_mod.scalarF32(ctx, std.math.inf(f32));
    defer inf_s.deinit();
    try c.check(c.c.mlx_clip(&clamped, targets.inner, eps.inner, inf_s.inner, ctx.stream.inner));
    const clamped_arr = Array.fromHandle(clamped);
    defer clamped_arr.deinit();

    const log_t = try ops_mod.log(ctx, clamped_arr);
    defer log_t.deinit();
    const diff = try ops_mod.subtract(ctx, log_t, log_preds);
    defer diff.deinit();
    const weighted = try ops_mod.multiply(ctx, clamped_arr, diff);
    defer weighted.deinit();

    return reduce_mod.sum(ctx, weighted, false);
}

// ============================================================
// NLL Loss (Negative Log Likelihood)
// ============================================================

pub fn nllLoss(ctx: EagerContext, log_preds: Array, labels: Array) !Array {
    // Graph-mode NLL: -mean(gather(log_preds, labels))
    const labels_exp = try shape_mod.expandDims(ctx, labels, -1);
    defer labels_exp.deinit();
    const gathered = try shape_mod.takeAlongAxis(ctx, log_preds, labels_exp, -1);
    defer gathered.deinit();
    const gathered_sq = try shape_mod.squeezeAxis(ctx, gathered, -1);
    defer gathered_sq.deinit();
    const neg = try ops_mod.negative(ctx, gathered_sq);
    defer neg.deinit();
    return reduce_mod.mean(ctx, neg, false);
}

// ============================================================
// Smooth L1 Loss (Huber Loss with delta=1)
// ============================================================

pub fn smoothL1Loss(ctx: EagerContext, predictions: Array, targets: Array) !Array {
    // Smooth L1 is Huber with delta=1.0
    return huberLoss(ctx, predictions, targets, 1.0);
}

// ============================================================
// Cosine Similarity Loss
// ============================================================

pub fn cosineSimilarityLoss(ctx: EagerContext, predictions: Array, targets: Array) !Array {
    // Graph-mode cosine similarity loss: 1 - (dot(p,t) / (||p|| * ||t|| + eps))
    const eps = try ops_mod.scalarF32(ctx, 1e-8);
    defer eps.deinit();
    const one = try ops_mod.scalarF32(ctx, 1.0);
    defer one.deinit();

    // dot product: sum(p * t)
    const prod = try ops_mod.multiply(ctx, predictions, targets);
    defer prod.deinit();
    const dot = try reduce_mod.sum(ctx, prod, false);
    defer dot.deinit();

    // norms: sqrt(sum(x^2))
    const p_sq = try ops_mod.multiply(ctx, predictions, predictions);
    defer p_sq.deinit();
    const p_norm_sq = try reduce_mod.sum(ctx, p_sq, false);
    defer p_norm_sq.deinit();
    const p_norm = try ops_mod.sqrt(ctx, p_norm_sq);
    defer p_norm.deinit();

    const t_sq = try ops_mod.multiply(ctx, targets, targets);
    defer t_sq.deinit();
    const t_norm_sq = try reduce_mod.sum(ctx, t_sq, false);
    defer t_norm_sq.deinit();
    const t_norm = try ops_mod.sqrt(ctx, t_norm_sq);
    defer t_norm.deinit();

    // similarity = dot / (p_norm * t_norm + eps)
    const norm_prod = try ops_mod.multiply(ctx, p_norm, t_norm);
    defer norm_prod.deinit();
    const denom = try ops_mod.add(ctx, norm_prod, eps);
    defer denom.deinit();
    const similarity = try ops_mod.divide(ctx, dot, denom);
    defer similarity.deinit();

    return ops_mod.subtract(ctx, one, similarity);
}

// ============================================================
// Triplet Loss
// ============================================================

pub fn tripletLoss(ctx: EagerContext, anchor: Array, positive: Array, negative: Array, margin: f32) !Array {
    // Graph-mode triplet loss: max(0, ||a-p||^2 - ||a-n||^2 + margin)
    const zero = try ops_mod.scalarF32(ctx, 0.0);
    defer zero.deinit();
    const margin_s = try ops_mod.scalarF32(ctx, margin);
    defer margin_s.deinit();

    // ||anchor - positive||^2
    const ap_diff = try ops_mod.subtract(ctx, anchor, positive);
    defer ap_diff.deinit();
    const ap_sq = try ops_mod.multiply(ctx, ap_diff, ap_diff);
    defer ap_sq.deinit();
    const pos_dist = try reduce_mod.sum(ctx, ap_sq, false);
    defer pos_dist.deinit();

    // ||anchor - negative||^2
    const an_diff = try ops_mod.subtract(ctx, anchor, negative);
    defer an_diff.deinit();
    const an_sq = try ops_mod.multiply(ctx, an_diff, an_diff);
    defer an_sq.deinit();
    const neg_dist = try reduce_mod.sum(ctx, an_sq, false);
    defer neg_dist.deinit();

    // max(0, pos_dist - neg_dist + margin)
    const raw_loss = try ops_mod.subtract(ctx, pos_dist, neg_dist);
    defer raw_loss.deinit();
    const with_margin = try ops_mod.add(ctx, raw_loss, margin_s);
    defer with_margin.deinit();

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_maximum(&res, with_margin.inner, zero.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}
