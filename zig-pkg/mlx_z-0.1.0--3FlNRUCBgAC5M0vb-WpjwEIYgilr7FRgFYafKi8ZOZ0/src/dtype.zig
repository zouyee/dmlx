/// Data type mapping between Zig types and MLX dtypes.
const c = @import("c.zig");

pub const Dtype = enum(c_int) {
    bool_ = c.c.MLX_BOOL,
    uint8 = c.c.MLX_UINT8,
    uint16 = c.c.MLX_UINT16,
    uint32 = c.c.MLX_UINT32,
    uint64 = c.c.MLX_UINT64,
    int8 = c.c.MLX_INT8,
    int16 = c.c.MLX_INT16,
    int32 = c.c.MLX_INT32,
    int64 = c.c.MLX_INT64,
    float16 = c.c.MLX_FLOAT16,
    float32 = c.c.MLX_FLOAT32,
    float64 = c.c.MLX_FLOAT64,
    bfloat16 = c.c.MLX_BFLOAT16,
    complex64 = c.c.MLX_COMPLEX64,

    /// Size of this dtype in bytes.
    pub fn size(self: Dtype) usize {
        return c.c.mlx_dtype_size(@intFromEnum(self));
    }
};

/// Map a Zig comptime type to its MLX dtype.
pub fn dtypeOf(comptime T: type) Dtype {
    return switch (T) {
        bool => .bool_,
        u8 => .uint8,
        u16 => .uint16,
        u32 => .uint32,
        u64 => .uint64,
        i8 => .int8,
        i16 => .int16,
        i32 => .int32,
        i64 => .int64,
        f16 => .float16,
        f32 => .float32,
        f64 => .float64,
        else => @compileError("unsupported type for MLX dtype"),
    };
}

pub const float32 = Dtype.float32;
pub const float64 = Dtype.float64;
pub const int32 = Dtype.int32;
pub const int64 = Dtype.int64;
pub const uint32 = Dtype.uint32;
pub const uint64 = Dtype.uint64;
pub const bool_ = Dtype.bool_;
pub const complex64 = Dtype.complex64;
pub const bfloat16 = Dtype.bfloat16;
pub const float16 = Dtype.float16;
