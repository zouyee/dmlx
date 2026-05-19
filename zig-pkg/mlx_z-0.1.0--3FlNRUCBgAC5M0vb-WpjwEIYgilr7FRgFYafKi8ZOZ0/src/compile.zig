/// Graph compilation and compile mode control.
const std = @import("std");
const c = @import("c.zig");
const closure_mod = @import("closure.zig");

const Closure = closure_mod.Closure;

pub const CompileMode = enum(c_uint) {
    disabled = c.c.MLX_COMPILE_MODE_DISABLED,
    no_simplify = c.c.MLX_COMPILE_MODE_NO_SIMPLIFY,
    no_fuse = c.c.MLX_COMPILE_MODE_NO_FUSE,
    enabled = c.c.MLX_COMPILE_MODE_ENABLED,
};

/// Compile a closure for optimized execution.
pub fn compile(closure: Closure, shapeless: bool) !Closure {
    var res: c.c.mlx_closure = .{ .ctx = null };
    try c.check(c.c.mlx_compile(&res, closure.inner, shapeless));
    return .{ .inner = res };
}

/// Enable compilation globally.
pub fn enableCompile() !void {
    try c.check(c.c.mlx_enable_compile());
}

/// Disable compilation globally.
pub fn disableCompile() !void {
    try c.check(c.c.mlx_disable_compile());
}

/// Set the global compile mode.
pub fn setCompileMode(mode: CompileMode) !void {
    try c.check(c.c.mlx_set_compile_mode(@intFromEnum(mode)));
}
