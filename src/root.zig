/// DMLX: Run Frontier LLMs on Your Mac.
/// Core MLX bindings provided by the mlx-zig library.
const mlx_z = @import("mlx");

// Core MLX library re-exports (from mlx-zig)
pub const dtype = mlx_z.dtype;
pub const array = mlx_z.array;
pub const device = mlx_z.device;
pub const ops = mlx_z.ops;
pub const nn = mlx_z.nn;
pub const loss = mlx_z.loss;
pub const activations = mlx_z.activations;
pub const io = mlx_z.io;
pub const npy = mlx_z.npy;
pub const eval = mlx_z.eval;
pub const closure = mlx_z.closure;
pub const grad = mlx_z.grad;
pub const compile = mlx_z.compile;
pub const tree = mlx_z.tree;
pub const optim = mlx_z.optim;
pub const array_arena = mlx_z.array_arena;
pub const quantize = mlx_z.quantize;
pub const comparison = mlx_z.comparison;
pub const math = mlx_z.math;
pub const shape = mlx_z.shape;
pub const reduce = mlx_z.reduce;
pub const sort = mlx_z.sort;
pub const creation = mlx_z.creation;
pub const random = mlx_z.random;
pub const linalg = mlx_z.linalg;
pub const fft = mlx_z.fft;
pub const conv = mlx_z.conv;
pub const fast = mlx_z.fast;
pub const fused = mlx_z.fused;
pub const custom_kernel = mlx_z.custom_kernel;

// Application-level modules (dmlx-specific)
pub const kvcache = @import("kvcache.zig");
pub const sampling = @import("sampling.zig");
pub const models = @import("models/llama.zig");
pub const model_loader = @import("models/llama_loader.zig");
pub const deepseek_v4 = @import("models/deepseek_v4.zig");
pub const deepseek_v4_loader = @import("models/deepseek_v4_loader.zig");
pub const expert_stream = @import("models/expert_stream.zig");
pub const layer_prefetcher = @import("models/layer_prefetcher.zig");
pub const nemotron_h = @import("models/nemotron_h.zig");
pub const nemotron_h_loader = @import("models/nemotron_h_loader.zig");
pub const llava_loader = @import("models/llava_loader.zig");
pub const lora = @import("lora.zig");
pub const hf_config = @import("hf_config.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const trainer = @import("trainer.zig");
pub const generation = @import("generation.zig");
pub const model_registry = @import("model_registry.zig");
pub const prompt_cache = @import("prompt_cache.zig");
pub const server = @import("server.zig");
pub const scheduler = @import("scheduler.zig");
pub const diffusion_scheduler = @import("diffusion/scheduler.zig");
pub const diffusion_vae = @import("diffusion/vae.zig");
pub const diffusion_flux = @import("diffusion/flux.zig");
pub const vision_preprocess = @import("vision/preprocess.zig");
pub const vision_vit = @import("vision/vit.zig");
pub const vision_llava = @import("vision/llava.zig");
pub const batch_builder = @import("batch_builder.zig");
pub const speculative = @import("speculative.zig");
pub const guided = @import("guided.zig");
pub const jang_quantizer = @import("jang_quantizer.zig");
pub const qlora = @import("qlora.zig");
pub const moe_router = @import("moe_router.zig");
pub const model_pool = @import("model_pool.zig");
pub const memory = @import("memory.zig");
pub const benchmark = @import("benchmark.zig");
pub const evaluate = @import("evaluate.zig");
pub const distributed = @import("distributed.zig");

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

pub const version = "0.4.0";
