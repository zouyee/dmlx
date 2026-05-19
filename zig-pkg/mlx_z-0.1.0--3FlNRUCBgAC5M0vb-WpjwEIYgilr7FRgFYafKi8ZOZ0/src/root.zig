/// MLX-Z: Apple MLX Zig bindings via mlx-c.
/// Core library providing Zig wrappers around Apple's MLX C API.
const std = @import("std");

// Foundation (FFI + types)
pub const c = @import("c.zig");
pub const dtype = @import("dtype.zig");
pub const device = @import("device.zig");

// Array layer
pub const array = @import("array.zig");
pub const array_arena = @import("array_arena.zig");

// Execution / evaluation
pub const eval = @import("eval.zig");
pub const closure = @import("closure.zig");
pub const grad = @import("grad.zig");
pub const compile = @import("compile.zig");

// Operations
pub const ops = @import("ops.zig");
pub const comparison = @import("ops/comparison.zig");
pub const math = @import("ops/math.zig");
pub const shape = @import("ops/shape.zig");
pub const reduce = @import("ops/reduce.zig");
pub const sort = @import("ops/sort.zig");
pub const creation = @import("ops/creation.zig");
pub const random = @import("ops/random.zig");
pub const linalg = @import("ops/linalg.zig");
pub const fft = @import("ops/fft.zig");
pub const conv = @import("ops/conv.zig");
pub const fast = @import("ops/fast.zig");
pub const nn = @import("ops/nn.zig");
pub const activations = @import("ops/activations.zig");
pub const loss = @import("ops/loss.zig");
pub const fused = @import("ops/fused.zig");
pub const custom_kernel = @import("ops/custom_kernel.zig");
pub const batch = @import("ops/batch.zig");

// I/O
pub const io = @import("io/mlx_io.zig");
pub const npy = @import("io/npy.zig");
pub const safetensors_reader = @import("io/safetensors_reader.zig");
pub const jang_loader = @import("io/jang_loader.zig");

// System
pub const memory = @import("memory.zig");
pub const metal = @import("metal.zig");

// Utilities
pub const quantize = @import("quantize.zig");
pub const tree = @import("tree.zig");
pub const optim = @import("optim.zig");

// Convenience re-exports
pub const Dtype = dtype.Dtype;
pub const Array = array.Array;
pub const Device = device.Device;
pub const Stream = device.Stream;
pub const EagerContext = ops.EagerContext;

// Dtype constants
pub const bool_ = dtype.bool_;
pub const uint8 = dtype.uint8;
pub const uint16 = dtype.uint16;
pub const uint32 = dtype.uint32;
pub const uint64 = dtype.uint64;
pub const int8 = dtype.int8;
pub const int16 = dtype.int16;
pub const int32 = dtype.int32;
pub const int64 = dtype.int64;
pub const float16 = dtype.float16;
pub const float32 = dtype.float32;
pub const float64 = dtype.float64;
pub const bfloat16 = dtype.bfloat16;
pub const complex64 = dtype.complex64;

pub const version = "0.1.0";
