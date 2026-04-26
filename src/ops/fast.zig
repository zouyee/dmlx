const std = @import("std");
const c = @import("../c.zig");
const ops_mod = @import("../ops.zig");
const array_mod = @import("../array.zig");
const device_mod = @import("../device.zig");
const dtype_mod = @import("../dtype.zig");

const Array = ops_mod.Array;
const Stream = device_mod.Stream;
const Dtype = dtype_mod.Dtype;
const EagerContext = ops_mod.EagerContext;

pub fn layerNorm(_ctx: EagerContext, x: Array, weight: ?Array, bias: ?Array, eps: f32) !Array {
    var res = c.c.mlx_array_new();
    const w_ptr = if (weight) |w| w.inner else c.c.mlx_array_new();
    const b_ptr = if (bias) |b| b.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_fast_layer_norm(&res, x.inner, w_ptr, b_ptr, eps, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn rmsNorm(_ctx: EagerContext, x: Array, weight: Array, eps: f32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fast_rms_norm(&res, x.inner, weight.inner, eps, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn rope(_ctx: EagerContext, x: Array, dims: i32, traditional: bool, base: ?f32, scale: f32, offset: i32, freqs: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const opt_base: c.c.mlx_optional_float = .{ .has_value = base != null, .value = if (base) |v| v else 0 };
    const f_ptr = if (freqs) |f| f.inner else c.c.mlx_array_empty;
    try c.check(c.c.mlx_fast_rope(&res, x.inner, dims, traditional, opt_base, scale, offset, f_ptr, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn ropeDynamic(_ctx: EagerContext, x: Array, dims: i32, traditional: bool, base: ?f32, scale: f32, offset: Array, freqs: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const opt_base: c.c.mlx_optional_float = .{ .has_value = base != null, .value = if (base) |v| v else 0 };
    const f_ptr = if (freqs) |f| f.inner else c.c.mlx_array_empty;
    try c.check(c.c.mlx_fast_rope_dynamic(&res, x.inner, dims, traditional, opt_base, scale, offset.inner, f_ptr, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn scaledDotProductAttention(_ctx: EagerContext, queries: Array, keys: Array, values: Array, scale: f32, mask_mode: []const u8, mask_arr: ?Array, sinks: ?Array) !Array {
    var res = c.c.mlx_array_new();
    // mlx-c C++ binding checks mask_arr.ctx to decide whether to pass
    // std::nullopt. mlx_array_empty is zero-initialized (ctx=null) so it
    // correctly translates to std::nullopt in the C++ layer.
    const m_ptr = if (mask_arr) |m| m.inner else c.c.mlx_array_empty;
    const s_ptr = if (sinks) |s| s.inner else c.c.mlx_array_empty;
    try c.check(c.c.mlx_fast_scaled_dot_product_attention(&res, queries.inner, keys.inner, values.inner, scale, mask_mode.ptr, m_ptr, s_ptr, _ctx.stream.inner));
    return Array.fromHandle(res);
}
