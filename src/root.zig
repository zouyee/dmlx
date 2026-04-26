/// MLX-Zig: Apple MLX bindings via mlx-c.
const std = @import("std");

pub const dtype = @import("dtype.zig");
pub const array = @import("array.zig");
pub const device = @import("device.zig");
pub const ops = @import("ops.zig");
pub const nn = @import("ops/nn.zig");
pub const loss = @import("ops/loss.zig");
pub const activations = @import("ops/activations.zig");

pub const io = @import("io/mlx_io.zig");
pub const npy = @import("io/npy.zig");

pub const eval = @import("eval.zig");
pub const closure = @import("closure.zig");
pub const grad = @import("grad.zig");
pub const compile = @import("compile.zig");
pub const kvcache = @import("kvcache.zig");
pub const tree = @import("tree.zig");
pub const sampling = @import("sampling.zig");
pub const optim = @import("optim.zig");
pub const models = @import("models/llama.zig");
pub const model_loader = @import("models/llama_loader.zig");
pub const deepseek_v4 = @import("models/deepseek_v4.zig");
pub const deepseek_v4_loader = @import("models/deepseek_v4_loader.zig");
pub const lora = @import("lora.zig");
pub const hf_config = @import("hf_config.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const trainer = @import("trainer.zig");
pub const array_arena = @import("array_arena.zig");
pub const generation = @import("generation.zig");
pub const model_registry = @import("model_registry.zig");
pub const prompt_cache = @import("prompt_cache.zig");
pub const server = @import("server.zig");
pub const scheduler = @import("scheduler.zig");
pub const batch_builder = @import("batch_builder.zig");
pub const speculative = @import("speculative.zig");
pub const guided = @import("guided.zig");
pub const quantize = @import("quantize.zig");
pub const qlora = @import("qlora.zig");
pub const moe_router = @import("moe_router.zig");
pub const model_pool = @import("model_pool.zig");
pub const memory = @import("memory.zig");
pub const benchmark = @import("benchmark.zig");
pub const evaluate = @import("evaluate.zig");
pub const distributed = @import("distributed.zig");

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
pub const fused = @import("ops/fused.zig");
pub const custom_kernel = @import("ops/custom_kernel.zig");

// Re-exports for convenience
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

pub const version = "0.3.0-mlx-c";
