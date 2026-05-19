/// Custom Metal kernel registration for DMLX.
///
/// Provides Zig wrappers around the mlx-c custom Metal kernel API
/// (`mlx_fast_metal_kernel`), enabling users to register and execute
/// custom Metal shaders for specialized operations (e.g., MoE expert
/// dispatch, custom attention patterns, fused kernels).
///
/// ## Usage
///
/// ```zig
/// const kernel = try CustomMetalKernel.init(
///     allocator,
///     "my_kernel",
///     &.{"input"},
///     "kernel void my_kernel(device float* input [[buffer(0)]], ...) { ... }",
///     "",
///     false,
/// );
/// defer kernel.deinit();
///
/// var config = try KernelConfig.init();
/// defer config.deinit();
/// try config.setGrid(256, 1, 1);
/// try config.setThreadGroup(256, 1, 1);
/// try config.addOutputArg(&.{1024}, .float32);
///
/// const outputs = try kernel.apply(allocator, &.{input_array}, config, stream);
/// ```
///
/// ## Requirements
///
/// - mlx-c 0.6.0+ with `mlx_fast_metal_kernel` API
/// - Apple Silicon Mac with Metal support
/// - Metal shader source as a string
///
/// ## API Reference
///
/// The underlying mlx-c API provides:
///   - `mlx_fast_metal_kernel_new` — Create a kernel from Metal source
///   - `mlx_fast_metal_kernel_apply` — Execute the kernel with inputs/config
///   - `mlx_fast_metal_kernel_config_*` — Configure grid, thread groups, outputs
///
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const dtype_mod = @import("../dtype.zig");

const Array = array_mod.Array;
const Dtype = dtype_mod.Dtype;

/// Configuration for a custom Metal kernel execution.
/// Wraps `mlx_fast_metal_kernel_config` from mlx-c.
pub const KernelConfig = struct {
    inner: c.c.mlx_fast_metal_kernel_config,

    pub fn init() !KernelConfig {
        return .{ .inner = c.c.mlx_fast_metal_kernel_config_new() };
    }

    pub fn deinit(self: *KernelConfig) void {
        c.c.mlx_fast_metal_kernel_config_free(self.inner);
    }

    /// Set the Metal compute grid dimensions (number of threadgroups).
    pub fn setGrid(self: *KernelConfig, x: i32, y: i32, z: i32) !void {
        try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(self.inner, x, y, z));
    }

    /// Set the Metal threadgroup dimensions (threads per threadgroup).
    pub fn setThreadGroup(self: *KernelConfig, x: i32, y: i32, z: i32) !void {
        try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(self.inner, x, y, z));
    }

    /// Add an output argument with the given shape and dtype.
    pub fn addOutputArg(self: *KernelConfig, shape: []const i32, dtype: c.c.mlx_dtype) !void {
        try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(
            self.inner,
            shape.ptr,
            shape.len,
            dtype,
        ));
    }

    /// Set the initial value for output buffers.
    pub fn setInitValue(self: *KernelConfig, value: f32) !void {
        try c.check(c.c.mlx_fast_metal_kernel_config_set_init_value(self.inner, value));
    }

    /// Enable verbose mode for debugging kernel compilation.
    pub fn setVerbose(self: *KernelConfig, verbose: bool) !void {
        try c.check(c.c.mlx_fast_metal_kernel_config_set_verbose(self.inner, verbose));
    }

    /// Add a dtype template argument to the kernel.
    pub fn addTemplateArgDtype(self: *KernelConfig, name: [:0]const u8, dtype: c.c.mlx_dtype) !void {
        try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_dtype(self.inner, name.ptr, dtype));
    }

    /// Add an integer template argument to the kernel.
    pub fn addTemplateArgInt(self: *KernelConfig, name: [:0]const u8, value: i32) !void {
        try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(self.inner, name.ptr, value));
    }

    /// Add a boolean template argument to the kernel.
    pub fn addTemplateArgBool(self: *KernelConfig, name: [:0]const u8, value: bool) !void {
        try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_bool(self.inner, name.ptr, value));
    }
};

/// A custom Metal kernel that can be executed on Apple Silicon GPU.
/// Wraps `mlx_fast_metal_kernel` from mlx-c.
pub const CustomMetalKernel = struct {
    inner: c.c.mlx_fast_metal_kernel,

    /// Create a new custom Metal kernel.
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for temporary strings
    ///   - name: Kernel function name (must match the Metal function name)
    ///   - input_names: Names of input arguments (bound to Metal buffers in order)
    ///   - source: Metal shader source code string
    ///   - header: Optional Metal header source (e.g., shared structs/functions)
    ///   - atomic_outputs: Whether outputs use atomic operations
    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
        input_names: []const []const u8,
        source: []const u8,
        header: []const u8,
        atomic_outputs: bool,
    ) !CustomMetalKernel {
        const name_z = try allocator.dupeZ(u8, name);
        defer allocator.free(name_z);
        const source_z = try allocator.dupeZ(u8, source);
        defer allocator.free(source_z);
        const header_z = try allocator.dupeZ(u8, header);
        defer allocator.free(header_z);

        // Build mlx_vector_string for input names
        const vec = c.c.mlx_vector_string_new();
        defer _ = c.c.mlx_vector_string_free(vec);
        for (input_names) |iname| {
            const iname_z = try allocator.dupeZ(u8, iname);
            defer allocator.free(iname_z);
            try c.check(c.c.mlx_vector_string_add_value(vec, iname_z.ptr));
        }

        return .{
            .inner = c.c.mlx_fast_metal_kernel_new(
                name_z.ptr,
                vec,
                source_z.ptr,
                header_z.ptr,
                atomic_outputs,
            ),
        };
    }

    pub fn deinit(self: *CustomMetalKernel) void {
        c.c.mlx_fast_metal_kernel_free(self.inner);
    }

    /// Execute the kernel with the given inputs and configuration.
    /// Returns a slice of output arrays. Caller owns the returned arrays.
    pub fn apply(
        self: *CustomMetalKernel,
        allocator: std.mem.Allocator,
        inputs: []const Array,
        config: KernelConfig,
        stream: c.c.mlx_stream,
    ) ![]Array {
        // Build input vector
        const input_vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(input_vec);
        for (inputs) |inp| {
            try c.check(c.c.mlx_vector_array_add_value(input_vec, inp.inner));
        }

        var output_vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(output_vec);

        try c.check(c.c.mlx_fast_metal_kernel_apply(
            &output_vec,
            self.inner,
            input_vec,
            config.inner,
            stream,
        ));

        // Extract output arrays
        const num_outputs = c.c.mlx_vector_array_size(output_vec);
        var result = try allocator.alloc(Array, num_outputs);
        for (0..num_outputs) |i| {
            var arr = c.c.mlx_array_new();
            try c.check(c.c.mlx_vector_array_get(&arr, output_vec, i));
            result[i] = Array.fromHandle(arr);
        }

        return result;
    }
};

// ============================================================
// Unit Tests
// ============================================================

test "KernelConfig: init and deinit" {
    var config = try KernelConfig.init();
    config.deinit();
}

test "KernelConfig: set grid and thread group" {
    var config = try KernelConfig.init();
    defer config.deinit();
    try config.setGrid(256, 1, 1);
    try config.setThreadGroup(32, 1, 1);
}

test "KernelConfig: set init value and verbose" {
    var config = try KernelConfig.init();
    defer config.deinit();
    try config.setInitValue(0.0);
    try config.setVerbose(false);
}
