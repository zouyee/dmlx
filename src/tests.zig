const std = @import("std");
const c = @import("c.zig");

pub const core_tests = @import("tests/core_tests.zig");
pub const comparison_tests = @import("tests/comparison_tests.zig");
pub const math_tests = @import("tests/math_tests.zig");
pub const shape_tests = @import("tests/shape_tests.zig");
pub const reduce_tests = @import("tests/reduce_tests.zig");
pub const sort_tests = @import("tests/sort_tests.zig");
pub const creation_tests = @import("tests/creation_tests.zig");
pub const random_tests = @import("tests/random_tests.zig");
pub const linalg_tests = @import("tests/linalg_tests.zig");
pub const fft_tests = @import("tests/fft_tests.zig");
pub const kvcache_tests = @import("tests/kvcache_tests.zig");
pub const tokenizer_tests = @import("tests/tokenizer_tests.zig");
pub const safetensors_tests = @import("tests/safetensors_tests.zig");
pub const trainer_tests = @import("tests/trainer_tests.zig");
pub const e2e_tests = @import("tests/e2e_tests.zig");
pub const deepseek_v4_tests = @import("tests/deepseek_v4_tests.zig");
pub const minimax_tests = @import("tests/minimax_tests.zig");
pub const nemotron_h = @import("models/nemotron_h.zig");
pub const nemotron_h_loader = @import("models/nemotron_h_loader.zig");
pub const golden_tests = @import("tests/golden_test.zig");
pub const arena_tests = @import("tests/arena_tests.zig");
pub const generation_tests = @import("tests/generation_tests.zig");
pub const model_registry_tests = @import("model_registry.zig");
pub const llava_tests = @import("vision/llava.zig");
pub const prompt_cache_tests = @import("prompt_cache.zig");
pub const fused_tests = @import("tests/fused_tests.zig");
pub const scheduler_tests = @import("tests/scheduler_tests.zig");
pub const server_tests = @import("server.zig");
pub const batch_builder_tests = @import("tests/batch_builder_tests.zig");
pub const speculative_tests = @import("speculative.zig");
pub const guided_tests = @import("guided.zig");
pub const quantize_tests = @import("quantize.zig");
pub const qlora_tests = @import("qlora.zig");
pub const moe_router_tests = @import("moe_router.zig");
pub const model_pool_tests = @import("model_pool.zig");
pub const tiered_kvcache_tests = @import("tests/tiered_kvcache_tests.zig");
pub const prefix_disk_tests = @import("kvcache/prefix_disk.zig");
pub const memory_tests = @import("memory.zig");
pub const memory_property_tests = @import("tests/memory_property_tests.zig");
pub const benchmark_tests = @import("benchmark.zig");
pub const integration_tests = @import("tests/integration_tests.zig");
pub const model_smoke_tests = @import("tests/model_smoke_tests.zig");

pub const tool_calling_tests = @import("tool_calling.zig");
pub const tool_executor_tests = @import("tool_executor.zig");
pub const jang_loader_tests = @import("io/jang_loader.zig");
pub const jang_quantizer_tests = @import("jang_quantizer.zig");


pub const vision_preprocess_tests = @import("vision/preprocess.zig");
pub const vision_vit_tests = @import("vision/vit.zig");
pub const vision_llava_tests = @import("vision/llava.zig");

test "init mlx error handler" {
    c.initErrorHandler();
}

test {
    _ = core_tests;
    _ = comparison_tests;
    _ = math_tests;
    _ = shape_tests;
    _ = reduce_tests;
    _ = sort_tests;
    _ = creation_tests;
    _ = random_tests;
    _ = linalg_tests;
    _ = fft_tests;
    _ = kvcache_tests;
    _ = tokenizer_tests;
    _ = safetensors_tests;
    _ = trainer_tests;
    _ = e2e_tests;
    _ = deepseek_v4_tests;
    _ = nemotron_h;
    _ = nemotron_h_loader;
    _ = golden_tests;
    _ = arena_tests;
    _ = generation_tests;
    _ = model_registry_tests;
    _ = llava_tests;
    _ = prompt_cache_tests;
    _ = fused_tests;
    _ = scheduler_tests;
    _ = server_tests;
    _ = batch_builder_tests;
    _ = speculative_tests;
    _ = guided_tests;
    _ = quantize_tests;
    _ = qlora_tests;
    _ = moe_router_tests;
    _ = model_pool_tests;
    _ = tiered_kvcache_tests;
    _ = prefix_disk_tests;
    _ = memory_tests;
    _ = memory_property_tests;
    _ = benchmark_tests;
    _ = integration_tests;
    _ = model_smoke_tests;

    _ = tool_calling_tests;
    _ = tool_executor_tests;
    _ = jang_loader_tests;
    _ = jang_quantizer_tests;

    _ = vision_preprocess_tests;
    _ = vision_vit_tests;
    _ = vision_llava_tests;
}
