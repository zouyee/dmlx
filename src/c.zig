/// Low-level bindings to mlx-c.
/// Re-exports the C API with minimal Zig sugar.
const std = @import("std");

pub const c = @cImport({
    @cInclude("mlx/c/mlx.h");
});

pub const Array = c.mlx_array;
pub const Dtype = c.mlx_dtype;
pub const Device = c.mlx_device;
pub const Stream = c.mlx_stream;
pub const String = c.mlx_string;

pub const DType = enum(c_int) {
    bool_ = c.MLX_BOOL,
    uint8 = c.MLX_UINT8,
    uint16 = c.MLX_UINT16,
    uint32 = c.MLX_UINT32,
    uint64 = c.MLX_UINT64,
    int8 = c.MLX_INT8,
    int16 = c.MLX_INT16,
    int32 = c.MLX_INT32,
    int64 = c.MLX_INT64,
    float16 = c.MLX_FLOAT16,
    float32 = c.MLX_FLOAT32,
    float64 = c.MLX_FLOAT64,
    bfloat16 = c.MLX_BFLOAT16,
    complex64 = c.MLX_COMPLEX64,
};

var last_error_buffer: [2048]u8 = undefined;
var last_error_len: usize = 0;

/// C-callable error handler registered with mlx_set_error_handler.
/// Stores the error message in a global buffer for retrieval.
export fn mlxErrorHandler(msg: [*c]const u8, data: ?*anyopaque) callconv(.c) void {
    _ = data;
    const len = std.mem.len(msg);
    const copy_len = @min(len, last_error_buffer.len - 1);
    @memcpy(last_error_buffer[0..copy_len], msg[0..copy_len]);
    last_error_len = copy_len;
}

/// Register the Zig error handler with mlx-c so that C++ exceptions
/// are caught and converted to error messages instead of crashing.
pub fn initErrorHandler() void {
    c.mlx_set_error_handler(mlxErrorHandler, null, null);
}

/// Retrieve the last error message captured by the error handler.
pub fn getLastError() []const u8 {
    return last_error_buffer[0..last_error_len];
}

/// Check return code from mlx-c functions.
/// Logs the return code and any captured error message.
/// Resets the error buffer after reading so stale messages don't leak
/// into subsequent error reports.
pub fn check(rc: c_int) !void {
    if (rc != 0) {
        const msg = getLastError();
        if (msg.len > 0) {
            std.log.err("MLX operation failed with rc={d}: {s}", .{ rc, msg });
            last_error_len = 0; // clear after consumption
        } else {
            std.log.err("MLX operation failed with rc={d}", .{rc});
        }
        return error.MlxError;
    }
}

/// Free an mlx_array.
pub fn arrayFree(arr: c.mlx_array) void {
    _ = c.mlx_array_free(arr);
}

/// Evaluate an array.
pub fn arrayEval(arr: c.mlx_array) !void {
    return check(c.mlx_array_eval(arr));
}

/// Get default device.
pub fn defaultDevice() !c.mlx_device {
    var dev: c.mlx_device = undefined;
    try check(c.mlx_get_default_device(&dev));
    return dev;
}

/// Set default device.
pub fn setDefaultDevice(dev: c.mlx_device) !void {
    return check(c.mlx_set_default_device(dev));
}

/// Create a new CPU/GPU device.
pub fn deviceNew(dtype: c.mlx_device_type, index: c_int) c.mlx_device {
    return c.mlx_device_new_type(dtype, index);
}

/// Create a new stream on a device.
pub fn streamNew(dev: c.mlx_device) !c.mlx_stream {
    var stream: c.mlx_stream = undefined;
    try check(c.mlx_stream_new(&stream, dev));
    return stream;
}

/// Synchronize a stream.
pub fn synchronize(stream: c.mlx_stream) !void {
    return check(c.mlx_synchronize(stream));
}

/// Get default stream for a device.
pub fn defaultStream(dev: c.mlx_device) !c.mlx_stream {
    var stream: c.mlx_stream = undefined;
    try check(c.mlx_stream_get_default(&stream, dev));
    return stream;
}
