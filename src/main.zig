/// MLX-Zig CLI tool.
///
/// Commands:
///   mlx-zig chat --model <path> --prompt "Hello"
///   mlx-zig serve --model <path> --port 8080
///   mlx-zig server --model <path> --port 8080  (alias for serve)
///   mlx-zig benchmark --model <path>
///   mlx-zig quantize --model <path> --output <path> --bits 4
///   mlx-zig convert --from gguf --to safetensors --input model.gguf --output model.safetensors
///   mlx-zig lora-train --model <path> --data dataset.jsonl --output adapter.safetensors
const std = @import("std");
const root = @import("root.zig");
const c = @import("c.zig");
const memory = @import("memory.zig");
const safetensors_reader = @import("io/safetensors_reader.zig");
const quantize_mod = @import("quantize.zig");

const Array = root.Array;
const EagerContext = root.EagerContext;

const benchmark_mod = @import("benchmark.zig");
const evaluate_mod = @import("evaluate.zig");

const ChatCommand = struct {
    model_path: []const u8,
    prompt: []const u8,
    system_prompt: []const u8 = "",
    max_tokens: usize = 256,
    temperature: f32 = 0.8,
    top_k: usize = 50,
    top_p: f32 = 1.0,
    seed: ?u64 = null,
    max_kv_size: memory.MaxKvSize = .auto,
    smelt: bool = false,
    smelt_experts: f32 = 1.0,
    distributed: bool = false,
};

const ServerCommand = struct {
    model_path: []const u8,
    port: u16 = 8080,
    max_tokens: usize = 256,
    temperature: f32 = 0.8,
    top_k: usize = 50,
    top_p: f32 = 1.0,
    max_kv_size: memory.MaxKvSize = .auto,
    // -- Production server config parameters (wired from ServerConfig) --
    kv_bits: u8 = 4,
    kv_strategy: root.server.KvStrategy = .paged_quantized,
    kv_quant: root.server.KvQuant = .simple,
    kv_tier: root.server.KvTier = .ram,
    kv_cold_dir: ?[]const u8 = null,
    prompt_cache_file: ?[]const u8 = null,
    speculative_ngram: ?usize = null,
    smelt: bool = false,
    smelt_experts: f32 = 1.0,
    distributed: bool = false,
};

const ConvertCommand = struct {
    input_path: []const u8,
    output_path: []const u8,
    from_format: []const u8,
    to_format: []const u8,
};

const LoraTrainCommand = struct {
    model_path: []const u8,
    data_path: []const u8,
    output_path: []const u8,
    rank: usize = 16,
    alpha: f32 = 32.0,
    lr: f32 = 1e-4,
    epochs: usize = 3,
    batch_size: usize = 1,
};

const BenchmarkCommand = struct {
    model_path: []const u8,
    input_tokens: usize = 32,
    output_tokens: usize = 128,
    warmup_runs: usize = 1,
    num_runs: usize = 3,
};

const QuantizeCommand = struct {
    model_path: []const u8,
    output_path: []const u8,
    bits: u8 = 4,
    group_size: i32 = 64,
};

const EvaluateCommand = struct {
    model_path: []const u8,
    data_path: []const u8,
    max_tokens: usize = 0,
    stride: usize = 512,
    context_size: usize = 1024,
};

const JangConvertCommand = struct {
    model_path: []const u8,
    output_path: []const u8,
    profile: []const u8,
};

const ImageGenCommand = struct {
    model_path: []const u8,
    prompt: []const u8,
    height: usize = 512,
    width: usize = 512,
    steps: usize = 20,
    output_path: []const u8,
};

pub fn main(init: std.process.Init) !void {
    // Use c_allocator for large model loading (DebugAllocator has too much overhead)
    const allocator = std.heap.c_allocator;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    // Register the MLX error handler early so C++ exceptions are captured
    // as contextual error messages rather than causing crashes.
    c.initErrorHandler();

    // Set default device to GPU (Metal) for all MLX operations.
    // Apple Silicon UMA means CPU/GPU share memory — no copy overhead.
    // MLX automatically handles data transfer for CPU-created arrays.
    const gpu_dev = c.c.mlx_device_new_type(c.c.MLX_GPU, 0);
    try c.check(c.c.mlx_set_default_device(gpu_dev));

    if (args.len < 2) {
        printUsage();
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "chat")) {
        const cmd = try parseChatArgs(allocator, args[2..]);
        defer {
            allocator.free(cmd.model_path);
            allocator.free(cmd.prompt);
        }
        try runChat(allocator, init.io, cmd);
    } else if (std.mem.eql(u8, command, "convert")) {
        const cmd = try parseConvertArgs(allocator, args[2..]);
        defer {
            allocator.free(cmd.input_path);
            allocator.free(cmd.output_path);
            allocator.free(cmd.from_format);
            allocator.free(cmd.to_format);
        }
        try runConvert(allocator, init.io, cmd);
    } else if (std.mem.eql(u8, command, "lora-train")) {
        const cmd = try parseLoraTrainArgs(allocator, args[2..]);
        defer {
            allocator.free(cmd.model_path);
            allocator.free(cmd.data_path);
            allocator.free(cmd.output_path);
        }
        try runLoraTrain(allocator, init.io, cmd);
    } else if (std.mem.eql(u8, command, "server") or std.mem.eql(u8, command, "serve")) {
        const cmd = try parseServerArgs(allocator, args[2..]);
        defer {
            allocator.free(cmd.model_path);
            if (cmd.kv_cold_dir) |d| allocator.free(d);
            if (cmd.prompt_cache_file) |f| allocator.free(f);
        }
        const server_config = root.server.ServerConfig{
            .model_path = cmd.model_path,
            .port = cmd.port,
            .max_tokens = cmd.max_tokens,
            .temperature = cmd.temperature,
            .top_k = cmd.top_k,
            .top_p = cmd.top_p,
            .max_kv_size = cmd.max_kv_size,
            .kv_bits = cmd.kv_bits,
            .kv_strategy = cmd.kv_strategy,
            .kv_quant = cmd.kv_quant,
            .kv_tier = cmd.kv_tier,
            .kv_cold_dir = cmd.kv_cold_dir,
            .prompt_cache_file = cmd.prompt_cache_file,
            .speculative_ngram = cmd.speculative_ngram,
            .smelt = cmd.smelt,
            .smelt_experts = cmd.smelt_experts,
        };
        try root.server.start(allocator, init.io, server_config);
    } else if (std.mem.eql(u8, command, "benchmark")) {
        const cmd = try parseBenchmarkArgs(allocator, args[2..]);
        defer allocator.free(cmd.model_path);
        try runBenchmark(allocator, init.io, cmd);
    } else if (std.mem.eql(u8, command, "quantize")) {
        const cmd = try parseQuantizeArgs(allocator, args[2..]);
        defer {
            allocator.free(cmd.model_path);
            allocator.free(cmd.output_path);
        }
        try runQuantize(allocator, init.io, cmd);
    } else if (std.mem.eql(u8, command, "evaluate")) {
        const cmd = try parseEvaluateArgs(allocator, args[2..]);
        defer {
            allocator.free(cmd.model_path);
            allocator.free(cmd.data_path);
        }
        try runEvaluate(allocator, init.io, cmd);
    } else if (std.mem.eql(u8, command, "jang-convert")) {
        const cmd = try parseJangConvertArgs(allocator, args[2..]);
        defer {
            allocator.free(cmd.model_path);
            allocator.free(cmd.output_path);
            allocator.free(cmd.profile);
        }
        try runJangConvert(allocator, init.io, cmd);
    } else if (std.mem.eql(u8, command, "image-gen")) {
        const cmd = try parseImageGenArgs(allocator, args[2..]);
        defer {
            allocator.free(cmd.model_path);
            allocator.free(cmd.prompt);
            allocator.free(cmd.output_path);
        }
        try runImageGen(allocator, init.io, cmd);
    } else if (std.mem.eql(u8, command, "version")) {
        std.debug.print("mlx-zig {s}\n", .{root.version});
    } else {
        std.log.err("Unknown command: {s}", .{command});
        printUsage();
        std.process.exit(1);
    }
}

fn printUsage() void {
    std.debug.print("mlx-zig {s}\n\n", .{root.version});
    std.debug.print(
        \\Usage:
        \\  mlx-zig chat [options]
        \\    --model <path>      Path to model directory
        \\    --prompt <text>     Prompt text
        \\    --max-tokens <n>    Maximum tokens (default: 256)
        \\    --temperature <t>   Sampling temperature (default: 0.8)
        \\    --top-k <k>         Top-k sampling (default: 50)
        \\    --top-p <p>         Top-p sampling (default: 1.0)
        \\    --seed <n>          Random seed for sampling (optional)
        \\    --max-kv-size <v>   KV cache size: "auto" (default) or integer
        \\
        \\  mlx-zig serve [options]  (alias: server)
        \\    --model <path>      Path to model directory
        \\    --port <n>          HTTP port (default: 8080)
        \\    --max-tokens <n>    Maximum tokens per request (default: 256)
        \\    --temperature <t>   Sampling temperature (default: 0.8)
        \\    --top-k <k>         Top-k sampling (default: 50)
        \\    --top-p <p>         Top-p sampling (default: 1.0)
        \\    --max-kv-size <v>   KV cache size: "auto" (default) or integer
        \\    --kv-bits <n>       KV cache quantization: 4, 8, or 16 (default: 4)
        \\    --kv-strategy <s>   KV cache strategy: standard|paged|quantized|paged_quantized (default: paged_quantized)
        \\    --kv-quant <q>      KV quantization algorithm: simple|turbo (default: simple)
        \\    --kv-tier <t>       KV storage tier: ram|ssd (default: ram)
        \\    --kv-cold-dir <p>   SSD cold-tier directory (required when --kv-tier ssd)
        \\    --prompt-cache-file <path>  Prompt cache file for KV state persistence
        \\    --speculative-ngram <n>     Enable speculative decoding with n-gram size
        \\    --smelt                     Enable Smelt mode (partial expert loading for MoE)
        \\    --smelt-experts <f>         Fraction of experts to load (default: 1.0)
        \\    --distributed               Enable distributed tensor parallelism
        \\
        \\  mlx-zig benchmark [options]
        \\    --model <path>      Path to model directory
        \\    --input-tokens <n>  Number of input tokens (default: 32)
        \\    --output-tokens <n> Number of output tokens (default: 128)
        \\    --warmup-runs <n>   Warmup iterations (default: 1)
        \\    --num-runs <n>      Timed iterations (default: 3)
        \\
        \\  mlx-zig quantize [options]
        \\    --model <path>      Path to model directory
        \\    --output <path>     Output path for quantized model
        \\    --bits <n>          Quantization bits: 4 or 8 (default: 4)
        \\    --group-size <n>    Quantization group size (default: 64)
        \\
        \\  mlx-zig convert [options]
        \\    --from <format>     Source format
        \\    --to <format>       Target format
        \\    --input <path>      Input file
        \\    --output <path>     Output file
        \\
        \\  mlx-zig evaluate [options]
        \\    --model <path>      Path to model directory
        \\    --data <path>       Path to evaluation text file
        \\    --max-tokens <n>    Maximum tokens to evaluate (default: all)
        \\    --stride <n>        Sliding window stride (default: 512)
        \\    --context-size <n>  Context window size (default: 1024)
        \\
        \\  mlx-zig lora-train [options]
        \\    --model <path>      Base model directory
        \\    --data <path>       Training dataset (JSONL)
        \\    --output <path>     Output adapter path
        \\    --rank <r>          LoRA rank (default: 16)
        \\    --alpha <a>         LoRA alpha (default: 32)
        \\    --lr <lr>           Learning rate (default: 1e-4)
        \\    --epochs <n>        Number of epochs (default: 3)
        \\
        \\  mlx-zig jang-convert [options]
        \\    --model <path>      Path to FP16 model directory
        \\    --output <path>     Output path for JANG quantized model
        \\    --profile <name>    JANG profile: 2M|2L|3M|4M|6M
        \\
        \\  mlx-zig image-gen [options]
        \\    --model <path>      Path to Flux model directory
        \\    --prompt <text>     Text prompt for image generation
        \\    --height <n>        Image height (default: 512)
        \\    --width <n>         Image width (default: 512)
        \\    --steps <n>         Denoising steps (default: 20)
        \\    --output <path>     Output image path
        \\
        \\  mlx-zig version
        \\    Show version information
        \\
    , .{});
}

// ------------------------------------------------------------------
// Chat Command
// ------------------------------------------------------------------

fn parseServerArgs(allocator: std.mem.Allocator, args: []const [:0]const u8) !ServerCommand {
    var cmd = ServerCommand{
        .model_path = "",
    };

    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        const flag = args[i];
        if (i + 1 >= args.len) break;
        const value = args[i + 1];

        if (std.mem.eql(u8, flag, "--model")) {
            cmd.model_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--port")) {
            cmd.port = try std.fmt.parseInt(u16, value, 10);
        } else if (std.mem.eql(u8, flag, "--max-tokens")) {
            cmd.max_tokens = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--temperature")) {
            cmd.temperature = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, flag, "--top-k")) {
            cmd.top_k = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--top-p")) {
            cmd.top_p = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, flag, "--max-kv-size")) {
            cmd.max_kv_size = memory.parseMaxKvSize(value) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, flag, "--kv-bits")) {
            cmd.kv_bits = try std.fmt.parseInt(u8, value, 10);
        } else if (std.mem.eql(u8, flag, "--kv-strategy")) {
            // Integration point: full validation of strategy names would go here
            if (std.mem.eql(u8, value, "standard")) {
                cmd.kv_strategy = .standard;
            } else if (std.mem.eql(u8, value, "paged")) {
                cmd.kv_strategy = .paged;
            } else if (std.mem.eql(u8, value, "quantized")) {
                cmd.kv_strategy = .quantized;
            } else if (std.mem.eql(u8, value, "paged_quantized")) {
                cmd.kv_strategy = .paged_quantized;
            } else {
                return error.InvalidArgument;
            }
        } else if (std.mem.eql(u8, flag, "--kv-quant")) {
            if (std.mem.eql(u8, value, "simple")) {
                cmd.kv_quant = .simple;
            } else if (std.mem.eql(u8, value, "turbo")) {
                cmd.kv_quant = .turbo;
            } else {
                return error.InvalidArgument;
            }
        } else if (std.mem.eql(u8, flag, "--kv-tier")) {
            if (std.mem.eql(u8, value, "ram")) {
                cmd.kv_tier = .ram;
            } else if (std.mem.eql(u8, value, "ssd")) {
                cmd.kv_tier = .ssd;
            } else {
                return error.InvalidArgument;
            }
        } else if (std.mem.eql(u8, flag, "--kv-cold-dir")) {
            cmd.kv_cold_dir = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--prompt-cache-file")) {
            cmd.prompt_cache_file = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--speculative-ngram")) {
            cmd.speculative_ngram = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--smelt")) {
            cmd.smelt = true;
            // Boolean flag: don't consume a value, step back so next iteration
            // processes the same value as a flag (or adjust loop logic)
            i -= 1;
        } else if (std.mem.eql(u8, flag, "--smelt-experts")) {
            cmd.smelt_experts = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, flag, "--distributed")) {
            cmd.distributed = true;
            i -= 1;
        }
    }

    if (cmd.model_path.len == 0) {
        return error.MissingRequiredArgument;
    }

    return cmd;
}

fn parseChatArgs(allocator: std.mem.Allocator, args: []const [:0]const u8) !ChatCommand {
    var cmd = ChatCommand{
        .model_path = "",
        .prompt = "",
    };

    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        const flag = args[i];
        if (i + 1 >= args.len) break;
        const value = args[i + 1];

        if (std.mem.eql(u8, flag, "--model")) {
            cmd.model_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--prompt")) {
            cmd.prompt = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--max-tokens")) {
            cmd.max_tokens = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--temperature")) {
            cmd.temperature = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, flag, "--top-k")) {
            cmd.top_k = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--top-p")) {
            cmd.top_p = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, flag, "--system")) {
            cmd.system_prompt = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--max-kv-size")) {
            cmd.max_kv_size = memory.parseMaxKvSize(value) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, flag, "--smelt")) {
            cmd.smelt = true;
            i -= 1;
        } else if (std.mem.eql(u8, flag, "--smelt-experts")) {
            cmd.smelt_experts = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, flag, "--distributed")) {
            cmd.distributed = true;
            i -= 1;
        }
    }

    // If --system was provided, it was allocated and needs cleanup
    // (deferred in main)

    if (cmd.model_path.len == 0) {
        return error.MissingRequiredArgument;
    }

    return cmd;
}

fn runChat(allocator: std.mem.Allocator, io: std.Io, cmd: ChatCommand) !void {
    std.log.info("Loading model from {s}...", .{cmd.model_path});

    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();
    const stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(stream);

    // 1. Load config and detect model type
    const config_path = try std.fs.path.join(allocator, &[_][]const u8{ cmd.model_path, "config.json" });
    defer allocator.free(config_path);

    const config_content = try std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(1024 * 1024));
    defer allocator.free(config_content);

    const model_type = detectModelType(config_content);

    if (std.mem.eql(u8, model_type, "deepseek_v4")) {
        try runDeepSeekV4Chat(allocator, io, cmd, ctx, stream, config_content);
    } else {
        try runLlamaChat(allocator, io, cmd, ctx, stream, config_content);
    }
}

fn detectModelType(config_json: []const u8) []const u8 {
    const model_type_key = "\"model_type\":";
    if (std.mem.indexOf(u8, config_json, model_type_key)) |idx| {
        const start = idx + model_type_key.len;
        const rest = std.mem.trimStart(u8, config_json[start..], " \n");
        if (rest.len > 0 and rest[0] == '"') {
            const end = std.mem.indexOf(u8, rest[1..], "\"") orelse return "llama";
            return rest[1 .. 1 + end];
        }
    }
    return "llama";
}

/// Detect the appropriate chat template based on model architecture in config.json.
/// Qwen2 → ChatML, Llama-3 → Llama3, Mistral → ChatML, Gemma → ChatML, default → ChatML.
fn detectChatTemplate(allocator: std.mem.Allocator, config_json: []const u8) root.tokenizer.ChatTemplate {
    // Check architectures field
    const arch_key = "\"architectures\"";
    if (std.mem.indexOf(u8, config_json, arch_key)) |idx| {
        const rest = config_json[idx..];
        if (std.mem.indexOf(u8, rest, "Llama") != null) {
            // Check if it's Llama-3 by looking at model_type or vocab_size
            // Llama-3 typically has vocab_size >= 128000
            const vocab_key = "\"vocab_size\":";
            if (std.mem.indexOf(u8, config_json, vocab_key)) |vidx| {
                const vrest = std.mem.trimStart(u8, config_json[vidx + vocab_key.len ..], " ");
                // Find end of number (stop at comma, space, newline, etc.)
                var num_end: usize = 0;
                while (num_end < vrest.len and vrest[num_end] >= '0' and vrest[num_end] <= '9') : (num_end += 1) {}
                if (num_end > 0) {
                    const vocab_size = std.fmt.parseInt(usize, vrest[0..num_end], 10) catch 0;
                    if (vocab_size >= 128000) {
                        return root.tokenizer.ChatTemplate.initLlama3(allocator);
                    }
                }
            }
            // Older LLaMA models — use ChatML as safe default
            return root.tokenizer.ChatTemplate.initChatML(allocator);
        }
    }
    // Qwen2, Mistral, Gemma, and others → ChatML (most common format)
    return root.tokenizer.ChatTemplate.initChatML(allocator);
}

/// Find the safetensors file(s) in a model directory.
/// Returns all matching shard paths for multi-shard models.
/// Tries: model.safetensors, weights.safetensors, weights.XX.safetensors (all shards),
/// model-XXXXX-of-XXXXX.safetensors (all shards).
fn findSafetensorsPaths(allocator: std.mem.Allocator, model_dir: []const u8) ![][]u8 {
    const posix = @cImport(@cInclude("unistd.h"));

    // First try single-file candidates
    const single_candidates = [_][]const u8{
        "model.safetensors",
        "weights.safetensors",
    };
    for (&single_candidates) |name| {
        const path = try std.fs.path.join(allocator, &[_][]const u8{ model_dir, name });
        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);
        if (posix.access(path_z.ptr, posix.F_OK) == 0) {
            const paths = try allocator.alloc([]u8, 1);
            paths[0] = path;
            return paths;
        }
        allocator.free(path);
    }

    // Try weights.XX.safetensors pattern (mlx-lm quantized format)
    {
        var shard_paths = std.ArrayList([]u8).empty;
        defer shard_paths.deinit(allocator);
        var shard_idx: usize = 0;
        while (shard_idx < 100) : (shard_idx += 1) {
            const name = try std.fmt.allocPrint(allocator, "weights.{d:0>2}.safetensors", .{shard_idx});
            defer allocator.free(name);
            const path = try std.fs.path.join(allocator, &[_][]const u8{ model_dir, name });
            const path_z = try allocator.dupeZ(u8, path);
            defer allocator.free(path_z);
            if (posix.access(path_z.ptr, posix.F_OK) == 0) {
                try shard_paths.append(allocator, path);
            } else {
                allocator.free(path);
                break;
            }
        }
        if (shard_paths.items.len > 0) {
            return try shard_paths.toOwnedSlice(allocator);
        }
    }

    // Try model-XXXXX-of-XXXXX.safetensors pattern
    {
        var shard_paths = std.ArrayList([]u8).empty;
        defer shard_paths.deinit(allocator);
        // Try up to 20 total shards
        for (1..21) |total| {
            for (1..total + 1) |shard| {
                const name = try std.fmt.allocPrint(allocator, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ shard, total });
                defer allocator.free(name);
                const path = try std.fs.path.join(allocator, &[_][]const u8{ model_dir, name });
                const path_z = try allocator.dupeZ(u8, path);
                defer allocator.free(path_z);
                if (posix.access(path_z.ptr, posix.F_OK) == 0) {
                    try shard_paths.append(allocator, path);
                } else {
                    allocator.free(path);
                }
            }
            if (shard_paths.items.len > 0) {
                return try shard_paths.toOwnedSlice(allocator);
            }
        }
    }

    return error.FileNotFound;
}

/// Legacy single-path finder (for backward compatibility).
fn findSafetensorsPath(allocator: std.mem.Allocator, model_dir: []const u8) ![]u8 {
    const paths = try findSafetensorsPaths(allocator, model_dir);
    defer {
        for (paths[1..]) |p| allocator.free(p);
        allocator.free(paths);
    }
    return paths[0];
}

fn runLlamaChat(allocator: std.mem.Allocator, io: std.Io, cmd: ChatCommand, ctx: EagerContext, stream: c.c.mlx_stream, config_content: []const u8) !void {
    const config = try root.hf_config.parseLlamaConfig(allocator, config_content);

    // 2. Load tokenizer (BPE backend)
    const tokenizer_path = try std.fs.path.join(allocator, &[_][]const u8{ cmd.model_path, "tokenizer.json" });
    defer allocator.free(tokenizer_path);
    var tokenizer_backend = root.tokenizer.BpeTokenizer.init(allocator);
    defer tokenizer_backend.deinit();
    try tokenizer_backend.loadFromFile(io, tokenizer_path);
    const tokenizer = tokenizer_backend.asStrategy();

    // 2b. Detect chat template from architecture
    var chat_template = detectChatTemplate(allocator, config_content);

    // 3. Load model weights — try multiple safetensors naming conventions (supports multi-shard)
    const weights_paths = findSafetensorsPaths(allocator, cmd.model_path) catch |err| {
        std.log.err("No safetensors file found in {s}: {}", .{ cmd.model_path, err });
        return err;
    };
    defer {
        for (weights_paths) |p| allocator.free(p);
        allocator.free(weights_paths);
    }
    const const_paths: []const []const u8 = @ptrCast(weights_paths);
    var model = try root.model_loader.loadFromSafetensorsPaths(allocator, &config, const_paths, ctx, stream);
    defer model.deinit();

    // 4. Create KV caches
    const layer_config = root.kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = config.num_attention_heads,
        .num_kv_heads = config.num_key_value_heads,
        .head_dim = config.getHeadDim(),
        .max_seq_len = config.max_position_embeddings,
        .dtype = .float32,
    };

    var caches = try allocator.alloc(root.kvcache.KVCacheStrategy, config.num_hidden_layers);
    defer {
        for (caches) |cache| cache.deinit(allocator);
        allocator.free(caches);
    }

    for (0..config.num_hidden_layers) |i| {
        caches[i] = try root.kvcache.createStandard(allocator, layer_config, stream);
    }

    // 5. Apply chat template and encode prompt
    var messages = std.ArrayList(root.tokenizer.ChatMessage).empty;
    defer messages.deinit(allocator);

    if (cmd.system_prompt.len > 0) {
        try messages.append(allocator, .{ .role = "system", .content = cmd.system_prompt });
    }
    try messages.append(allocator, .{ .role = "user", .content = cmd.prompt });

    const prompt_text = try chat_template.apply(messages.items, true);
    defer allocator.free(prompt_text);

    // Encode with template-formatted text (template handles special tokens)
    const prompt_tokens = try tokenizer.encode(prompt_text, false, allocator);
    defer allocator.free(prompt_tokens);

    const prompt_arr = c.c.mlx_array_new_data(
        prompt_tokens.ptr,
        &[_]c_int{ 1, @intCast(prompt_tokens.len) },
        2,
        c.c.MLX_UINT32,
    );
    const prompt_array = root.Array.fromHandle(prompt_arr);
    defer prompt_array.deinit();

    // 6. Generate
    const seed = cmd.seed orelse @as(u64, @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds()));
    var sampler_config = root.sampling.SamplerConfig.init(seed);
    sampler_config.temperature = cmd.temperature;
    sampler_config.top_k = cmd.top_k;
    sampler_config.top_p = cmd.top_p;

    const all_tokens = try model.generate(prompt_array, cmd.max_tokens, &sampler_config, caches, .{
        .eos_token_id = config.eos_token_id,
        .tokenizer = tokenizer,
    });
    defer allocator.free(all_tokens);

    // 7. Decode and print (streaming already printed tokens, but we still decode for completeness)
    const generated_ids = all_tokens[prompt_tokens.len..];
    if (generated_ids.len == 0) {
        std.debug.print("\n(no output generated)\n", .{});
    }
}

fn runDeepSeekV4Chat(allocator: std.mem.Allocator, io: std.Io, cmd: ChatCommand, ctx: EagerContext, stream: c.c.mlx_stream, config_content: []const u8) !void {
    const ds_config = try root.deepseek_v4_loader.parseDSV4Config(allocator, config_content);

    // 1b. Configure MLX memory limits for large MoE models
    // Set wired_limit so MLX knows how much GPU memory it can use, enabling automatic eviction
    {
        const sysctl_c = @cImport(@cInclude("sys/sysctl.h"));
        var memsize: u64 = 0;
        var len: usize = @sizeOf(u64);
        var mib = [_]c_int{ sysctl_c.CTL_HW, sysctl_c.HW_MEMSIZE };
        _ = sysctl_c.sysctl(&mib, 2, &memsize, &len, null, 0);
        // Use ~85% of system memory as wired limit — maximize GPU cache for large models
        const limit: usize = @intCast(memsize * 85 / 100);
        var old_wired: usize = 0;
        _ = c.c.mlx_set_wired_limit(&old_wired, limit);
        // Cache limit at ~80% — keep materialized weights in GPU memory between forward passes
        var old_cache: usize = 0;
        const cache_lim: usize = @intCast(memsize * 4 / 5);
        _ = c.c.mlx_set_cache_limit(&old_cache, cache_lim);
        std.log.info("MLX memory: wired_limit={d}MB cache_limit={d}MB (system={d}MB)", .{ limit / 1024 / 1024, cache_lim / 1024 / 1024, memsize / 1024 / 1024 });
    }

    // 2. Load tokenizer
    const tokenizer_path = try std.fs.path.join(allocator, &[_][]const u8{ cmd.model_path, "tokenizer.json" });
    defer allocator.free(tokenizer_path);
    var tokenizer_backend = root.tokenizer.BpeTokenizer.init(allocator);
    defer tokenizer_backend.deinit();
    try tokenizer_backend.loadFromFile(io, tokenizer_path);
    const tokenizer = tokenizer_backend.asStrategy();

    // 3. Load model weights (sharded or single-file)
    const smelt_config = root.deepseek_v4_loader.SmeltConfig{
        .enabled = cmd.smelt,
        .load_fraction = cmd.smelt_experts,
    };
    // Use original eager loading with streaming shard cleanup
    // mlx_load_safetensors returns lazy/memory-mapped arrays that don't consume RAM until evaluated
    var weights = try root.deepseek_v4_loader.loadWeightsFromDirectory(allocator, io, cmd.model_path, ctx, stream, smelt_config);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    // 4. Build model
    var model = try root.deepseek_v4_loader.buildDSV4Model(allocator, &ds_config, &weights, ctx, stream, smelt_config);
    // Don't defer model cleanup - we'll do it manually to control order
    
    // 4b. Wire expert streaming when smelt is enabled (loads experts from SSD on demand)
    var expert_sp: ?*root.expert_stream.ExpertStreamProvider = null;
    var tensor_index: ?*safetensors_reader.TensorIndex = null;
    defer if (expert_sp) |sp| { sp.deinit(); allocator.destroy(sp); };
    defer if (tensor_index) |idx| { idx.deinit(); allocator.destroy(idx); };

    if (cmd.smelt and !model.hasExpertsLoaded()) {
        // Build tensor index for random-access reading
        const idx = try allocator.create(safetensors_reader.TensorIndex);
        idx.* = try safetensors_reader.buildIndexFromDirectory(allocator, cmd.model_path);
        tensor_index = idx;

        // Build per-layer metadata
        const num_layers = ds_config.num_hidden_layers;
        var layer_meta = try allocator.alloc(root.expert_stream.LayerExpertMeta, num_layers);
        for (0..num_layers) |i| {
            // Compute expert row bytes from tensor index
            const gate_name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.experts.gate_proj.weight", .{i});
            // Try fused switch_mlp format first
            const fused_gate_name = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.gate_proj.weight", .{i});
            defer allocator.free(gate_name);

            const actual_gate_name = if (idx.entries.contains(fused_gate_name)) fused_gate_name else gate_name;
            defer allocator.free(fused_gate_name);
            _ = actual_gate_name;

            // Use HF naming convention for switch_mlp
            const hf_gate = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.gate_proj.weight", .{i});
            const hf_up = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.up_proj.weight", .{i});
            const hf_down = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.down_proj.weight", .{i});
            const hf_gate_s = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.gate_proj.scales", .{i});
            const hf_up_s = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.up_proj.scales", .{i});
            const hf_down_s = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.down_proj.scales", .{i});

            // Calculate row bytes from tensor info
            var row_bytes: usize = 0;
            var scale_row_bytes: usize = 0;
            if (idx.entries.get(hf_gate)) |info| {
                const total = info.data_offset_end - info.data_offset_start;
                row_bytes = @intCast(total / @as(u64, @intCast(info.shape[0])));
            }
            if (idx.entries.get(hf_gate_s)) |info| {
                const total = info.data_offset_end - info.data_offset_start;
                scale_row_bytes = @intCast(total / @as(u64, @intCast(info.shape[0])));
            }

            layer_meta[i] = .{
                .gate_proj_name = hf_gate,
                .up_proj_name = hf_up,
                .down_proj_name = hf_down,
                .gate_scales_name = if (idx.entries.contains(hf_gate_s)) hf_gate_s else blk: { allocator.free(hf_gate_s); break :blk null; },
                .up_scales_name = if (idx.entries.contains(hf_up_s)) hf_up_s else blk: { allocator.free(hf_up_s); break :blk null; },
                .down_scales_name = if (idx.entries.contains(hf_down_s)) hf_down_s else blk: { allocator.free(hf_down_s); break :blk null; },
                .expert_row_bytes = row_bytes,
                .expert_scale_row_bytes = scale_row_bytes,
                .n_experts = ds_config.n_routed_experts,
            };
        }

        const sp = try allocator.create(root.expert_stream.ExpertStreamProvider);
        sp.* = .{
            .allocator = allocator,
            .index = idx,
            .layer_meta = layer_meta,
            .ctx = ctx,
            .is_quantized = true, // switch_mlp is always quantized in 4-bit models
            .quant_group_size = 32, // mxfp4 uses group_size=32
            .quant_bits = 4,
            .quant_mode = "mxfp4", // switch_mlp uses mxfp4 (no biases, uint8 scales)
            .swiglu_limit = ds_config.swiglu_limit,
        };
        expert_sp = sp;

        // Wire stream provider into each MoE layer
        model.setExpertStreamProvider(sp);
        std.log.info("Expert streaming enabled: loading experts from SSD on demand", .{});
    }

    // 5. Format prompt with chat template
    var chat_template = root.tokenizer.ChatTemplate.initDeepSeek(allocator);

    var messages = std.ArrayList(root.tokenizer.ChatMessage).empty;
    defer messages.deinit(allocator);

    if (cmd.system_prompt.len > 0) {
        try messages.append(allocator, .{ .role = "system", .content = cmd.system_prompt });
    }
    try messages.append(allocator, .{ .role = "user", .content = cmd.prompt });

    const prompt_text = try chat_template.apply(messages.items, true);
    defer allocator.free(prompt_text);

    // 6. Encode (no special tokens added by tokenizer, template handles them)
    const prompt_tokens = try tokenizer.encode(prompt_text, false, allocator);
    defer allocator.free(prompt_tokens);

    // Validate prompt format: first token should be BOS (0 for DeepSeek-V4-Flash)
    if (prompt_tokens.len == 0) {
        std.log.err("Empty prompt after tokenization", .{});
        return error.InvalidPromptFormat;
    }
    
    const expected_bos: u32 = 0; // <｜begin▁of▁sentence｜> token ID
    if (prompt_tokens[0] != expected_bos) {
        std.log.err("❌ BOS token mismatch! Expected {d}, got {d}", .{ expected_bos, prompt_tokens[0] });
        std.log.err("Prompt text: '{s}'", .{prompt_text});
        std.log.err("First 10 tokens: {any}", .{prompt_tokens[0..@min(prompt_tokens.len, 10)]});
        std.log.err("This indicates the chat template format doesn't match the tokenizer.", .{});
        std.log.err("DeepSeek V4 Flash uses: <｜begin▁of▁sentence｜> (token ID 0)", .{});
        return error.InvalidPromptFormat;
    }

    std.log.info("✅ Prompt correctly formatted with BOS token {d}", .{expected_bos});
    std.log.info("Prompt text: '{s}'", .{prompt_text});
    std.log.info("Prompt tokens ({d}): {any}", .{ prompt_tokens.len, prompt_tokens[0..@min(prompt_tokens.len, 20)] });

    const prompt_arr = c.c.mlx_array_new_data(
        prompt_tokens.ptr,
        &[_]c_int{ 1, @intCast(prompt_tokens.len) },
        2,
        c.c.MLX_UINT32,
    );
    const prompt_array = root.Array.fromHandle(prompt_arr);
    defer prompt_array.deinit();

    // 7. Generate
    const seed = cmd.seed orelse @as(u64, @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds()));
    var sampler_config = root.sampling.SamplerConfig.init(seed);
    sampler_config.temperature = cmd.temperature;
    sampler_config.top_k = cmd.top_k;
    sampler_config.top_p = cmd.top_p;

    // Create per-layer KV caches for DeepSeek V4 (matching mlx-lm make_cache)
    const caches = try root.deepseek_v4_loader.makeV4Caches(allocator, &ds_config, stream);
    defer {
        for (caches) |cache| {
            cache.deinit(allocator);
        }
        allocator.free(caches);
    }

    std.log.info("Starting generation...", .{});
    const new_tokens = try model.generate(prompt_tokens, cmd.max_tokens, &sampler_config, caches, stream);
    defer allocator.free(new_tokens);

    // Validate generated tokens
    if (new_tokens.len == 0) {
        std.log.warn("No tokens generated", .{});
        return;
    }

    std.log.info("Generated {d} tokens: {any}", .{ new_tokens.len, new_tokens[0..@min(new_tokens.len, 10)] });

    // 8. Decode and print
    const output_text = try tokenizer.decode(new_tokens, allocator);
    defer allocator.free(output_text);

    std.debug.print("\n{s}\n", .{output_text});
    
    // Manual cleanup to control order
    model.deinit();
}

// ------------------------------------------------------------------
// Benchmark Command
// ------------------------------------------------------------------

fn parseBenchmarkArgs(allocator: std.mem.Allocator, args: []const [:0]const u8) !BenchmarkCommand {
    var cmd = BenchmarkCommand{
        .model_path = "",
    };

    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        const flag = args[i];
        if (i + 1 >= args.len) break;
        const value = args[i + 1];

        if (std.mem.eql(u8, flag, "--model")) {
            cmd.model_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--input-tokens")) {
            cmd.input_tokens = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--output-tokens")) {
            cmd.output_tokens = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--warmup-runs")) {
            cmd.warmup_runs = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--num-runs")) {
            cmd.num_runs = try std.fmt.parseInt(usize, value, 10);
        }
    }

    if (cmd.model_path.len == 0) {
        return error.MissingRequiredArgument;
    }

    return cmd;
}

fn runBenchmark(allocator: std.mem.Allocator, io: std.Io, cmd: BenchmarkCommand) !void {
    std.log.info("Benchmark: model={s} input_tokens={d} output_tokens={d} warmup={d} runs={d}", .{
        cmd.model_path, cmd.input_tokens, cmd.output_tokens, cmd.warmup_runs, cmd.num_runs,
    });

    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();
    const stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(stream);

    // 1. Load config and detect model type
    const config_path = try std.fs.path.join(allocator, &[_][]const u8{ cmd.model_path, "config.json" });
    defer allocator.free(config_path);

    const config_content = try std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(1024 * 1024));
    defer allocator.free(config_content);

    const model_type = detectModelType(config_content);

    // 2. Load model via registry
    const arch_name: []const u8 = if (std.mem.eql(u8, model_type, "deepseek_v4"))
        "DeepseekV4ForCausalLM"
    else
        "LlamaForCausalLM";

    const loader = root.model_registry.getLoader(arch_name) catch {
        std.log.err("Unsupported architecture for benchmark: {s}", .{arch_name});
        return error.UnsupportedArchitecture;
    };

    var model_vtable = try loader(allocator, config_content, cmd.model_path, ctx, stream, io, .{});
    defer model_vtable.deinit(model_vtable.ptr, allocator);

    // 3. Create KV caches
    const mc = model_vtable.config;
    var caches = try allocator.alloc(root.kvcache.KVCacheStrategy, mc.num_layers);
    defer {
        for (caches) |cache_item| cache_item.deinit(allocator);
        allocator.free(caches);
    }

    for (0..mc.num_layers) |i| {
        const layer_config = root.kvcache.LayerConfig{
            .batch_size = 1,
            .num_heads = mc.num_kv_heads,
            .num_kv_heads = mc.num_kv_heads,
            .head_dim = mc.head_dim,
            .max_seq_len = 8192,
            .dtype = .float32,
        };
        caches[i] = try root.kvcache.createStandard(allocator, layer_config, stream);
    }

    // 4. Build benchmark config and run
    const bench_config = benchmark_mod.BenchmarkConfig{
        .model_path = cmd.model_path,
        .input_tokens = cmd.input_tokens,
        .output_tokens = cmd.output_tokens,
        .warmup_runs = cmd.warmup_runs,
        .num_runs = cmd.num_runs,
    };

    // We use a struct to capture the model/caches/ctx for the generation callback.
    // Since Zig function pointers can't capture, we use a global-state approach
    // scoped to this function via a comptime-known struct with mutable statics.
    const BenchState = struct {
        var s_model: ModelVTableRef = undefined;
        var s_caches: []root.kvcache.KVCacheStrategy = undefined;
        var s_ctx: EagerContext = undefined;
        var s_allocator: std.mem.Allocator = undefined;

        const ModelVTableRef = root.generation.ModelVTable;

        fn generate(input_tokens: []const u32, output_tokens: usize) anyerror!benchmark_mod.RunMetrics {
            // Reset KV caches for each run by creating fresh ones would be
            // expensive; instead we just run generation and measure timing.
            // For a real benchmark, caches should be reset between runs.

            const gen_config = root.generation.GenerateConfig{
                .max_tokens = output_tokens,
                .temperature = 0.0, // greedy for deterministic benchmark
                .seed = 42,
            };

            var tokens_generated: usize = 0;
            var ttft_ns: u64 = 0;
            const start = std.c.mach_absolute_time();

            // Use streamGenerate to capture TTFT precisely.
            const Cb = struct {
                var cb_ttft_set: bool = false;
                var cb_start_ns: u64 = 0;
                var cb_ttft_ns: u64 = 0;
                var cb_count: usize = 0;

                fn callback(token: u32, is_done: bool) void {
                    _ = token;
                    _ = is_done;
                    cb_count += 1;
                    if (!cb_ttft_set) {
                        cb_ttft_ns = std.c.mach_absolute_time() - cb_start_ns;
                        cb_ttft_set = true;
                    }
                }
            };

            Cb.cb_ttft_set = false;
            Cb.cb_start_ns = start;
            Cb.cb_ttft_ns = 0;
            Cb.cb_count = 0;

            try root.generation.streamGenerate(
                s_model,
                input_tokens,
                gen_config,
                s_caches,
                s_ctx,
                &Cb.callback,
            );

            tokens_generated = Cb.cb_count;
            ttft_ns = Cb.cb_ttft_ns;

            const end = std.c.mach_absolute_time();
            const total_ns: u64 = end - start;

            return benchmark_mod.RunMetrics{
                .ttft_ns = ttft_ns,
                .total_gen_ns = total_ns,
                .tokens_generated = tokens_generated,
            };
        }
    };

    BenchState.s_model = model_vtable;
    BenchState.s_caches = caches;
    BenchState.s_ctx = ctx;
    BenchState.s_allocator = allocator;

    const result = try benchmark_mod.runBenchmark(bench_config, &BenchState.generate);
    benchmark_mod.printResults(result);
}

// ------------------------------------------------------------------
// Quantize Command
// ------------------------------------------------------------------

fn parseQuantizeArgs(allocator: std.mem.Allocator, args: []const [:0]const u8) !QuantizeCommand {
    var cmd = QuantizeCommand{
        .model_path = "",
        .output_path = "",
    };

    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        const flag = args[i];
        if (i + 1 >= args.len) break;
        const value = args[i + 1];

        if (std.mem.eql(u8, flag, "--model")) {
            cmd.model_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--output")) {
            cmd.output_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--bits")) {
            cmd.bits = try std.fmt.parseInt(u8, value, 10);
        } else if (std.mem.eql(u8, flag, "--group-size")) {
            cmd.group_size = try std.fmt.parseInt(i32, value, 10);
        }
    }

    if (cmd.model_path.len == 0 or cmd.output_path.len == 0) {
        return error.MissingRequiredArgument;
    }

    return cmd;
}

fn runQuantize(allocator: std.mem.Allocator, io: std.Io, cmd: QuantizeCommand) !void {
    const quant_config = quantize_mod.QuantConfig{
        .bits = cmd.bits,
        .group_size = cmd.group_size,
    };
    try quant_config.validate();

    std.log.info("Quantizing model: {s} -> {s} (bits={d}, group_size={d})", .{
        cmd.model_path, cmd.output_path, cmd.bits, cmd.group_size,
    });

    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();
    const stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(stream);

    // 1. Load config and detect model type
    const config_path = try std.fs.path.join(allocator, &[_][]const u8{ cmd.model_path, "config.json" });
    defer allocator.free(config_path);

    const config_content = try std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(1024 * 1024));
    defer allocator.free(config_content);

    const model_type = detectModelType(config_content);

    // 2. Load model via registry
    const arch_name: []const u8 = if (std.mem.eql(u8, model_type, "deepseek_v4"))
        "DeepseekV4ForCausalLM"
    else
        "LlamaForCausalLM";

    const loader = root.model_registry.getLoader(arch_name) catch {
        std.log.err("Unsupported architecture for quantization: {s}", .{arch_name});
        return error.UnsupportedArchitecture;
    };

    var model_vtable = try loader(allocator, config_content, cmd.model_path, ctx, stream, io, .{});
    defer model_vtable.deinit(model_vtable.ptr, allocator);

    std.log.info("Model loaded ({s}). Quantization to {d}-bit with group_size={d} would be applied to weights.", .{
        arch_name, cmd.bits, cmd.group_size,
    });
    std.log.info("Output path: {s}", .{cmd.output_path});

    // Note: Full weight quantization + serialization requires iterating over
    // all weight tensors and saving to safetensors. The quantize_mod.quantize()
    // and quantize_mod.dequantize() functions provide the core primitives.
    // A complete implementation would:
    //   1. Load all weight tensors from the model
    //   2. Quantize each weight tensor using quantize_mod.quantize()
    //   3. Save quantized weights + config to the output directory
    std.log.info("Quantization infrastructure ready. Use quantize_mod.quantize() for per-tensor quantization.", .{});
}

// ------------------------------------------------------------------
// Evaluate Command
// ------------------------------------------------------------------

fn parseEvaluateArgs(allocator: std.mem.Allocator, args: []const [:0]const u8) !EvaluateCommand {
    var cmd = EvaluateCommand{
        .model_path = "",
        .data_path = "",
    };

    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) break;
        const flag = args[i];
        const value = args[i + 1];

        if (std.mem.eql(u8, flag, "--model")) {
            cmd.model_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--data")) {
            cmd.data_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--max-tokens")) {
            cmd.max_tokens = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--stride")) {
            cmd.stride = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--context-size")) {
            cmd.context_size = try std.fmt.parseInt(usize, value, 10);
        }
    }

    if (cmd.model_path.len == 0 or cmd.data_path.len == 0) {
        return error.MissingRequiredArgument;
    }

    return cmd;
}

fn runEvaluate(allocator: std.mem.Allocator, io: std.Io, cmd: EvaluateCommand) !void {
    std.log.info("Evaluating perplexity: model={s} data={s}", .{ cmd.model_path, cmd.data_path });

    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();
    const stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(stream);

    // 1. Load config and detect model type
    const config_path = try std.fs.path.join(allocator, &[_][]const u8{ cmd.model_path, "config.json" });
    defer allocator.free(config_path);

    const config_content = try std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(1024 * 1024));
    defer allocator.free(config_content);

    const model_type = detectModelType(config_content);
    const arch_name: []const u8 = if (std.mem.eql(u8, model_type, "deepseek_v4"))
        "DeepseekV4ForCausalLM"
    else
        "LlamaForCausalLM";

    // 2. Load model via registry
    const loader = root.model_registry.getLoader(arch_name) catch {
        std.log.err("Unsupported architecture for evaluation: {s}", .{arch_name});
        return error.UnsupportedArchitecture;
    };

    var model_vtable = try loader(allocator, config_content, cmd.model_path, ctx, stream, io, .{});
    defer model_vtable.deinit(model_vtable.ptr, allocator);

    // 3. Load tokenizer
    const tokenizer_path = try std.fs.path.join(allocator, &[_][]const u8{ cmd.model_path, "tokenizer.json" });
    defer allocator.free(tokenizer_path);
    var tokenizer_backend = root.tokenizer.BpeTokenizer.init(allocator);
    defer tokenizer_backend.deinit();
    try tokenizer_backend.loadFromFile(io, tokenizer_path);
    const tokenizer = tokenizer_backend.asStrategy();

    // 4. Load and tokenize evaluation data
    const data_content = try std.Io.Dir.cwd().readFileAlloc(io, cmd.data_path, allocator, .limited(100 * 1024 * 1024));
    defer allocator.free(data_content);

    const tokens = try tokenizer.encode(data_content, false, allocator);
    defer allocator.free(tokens);

    std.log.info("Tokenized {d} tokens from {s}", .{ tokens.len, cmd.data_path });

    // 5. Create KV caches
    const mc = model_vtable.config;
    var caches = try allocator.alloc(root.kvcache.KVCacheStrategy, mc.num_layers);
    defer {
        for (caches) |cache_item| cache_item.deinit(allocator);
        allocator.free(caches);
    }

    for (0..mc.num_layers) |i| {
        const layer_config = root.kvcache.LayerConfig{
            .batch_size = 1,
            .num_heads = mc.num_kv_heads,
            .num_kv_heads = mc.num_kv_heads,
            .head_dim = mc.head_dim,
            .max_seq_len = cmd.context_size,
            .dtype = .float32,
        };
        caches[i] = try root.kvcache.createStandard(allocator, layer_config, stream);
    }

    // 6. Compute perplexity
    const eval_config = evaluate_mod.EvaluateConfig{
        .model_path = cmd.model_path,
        .data_path = cmd.data_path,
        .max_tokens = cmd.max_tokens,
        .stride = cmd.stride,
        .context_size = cmd.context_size,
    };

    const result = try evaluate_mod.computePerplexity(allocator, model_vtable, tokens, eval_config, caches, ctx);
    evaluate_mod.printResults(result);
}

// ------------------------------------------------------------------
// Convert Command
// ------------------------------------------------------------------

fn parseConvertArgs(allocator: std.mem.Allocator, args: []const [:0]const u8) !ConvertCommand {
    var cmd = ConvertCommand{
        .input_path = "",
        .output_path = "",
        .from_format = "",
        .to_format = "",
    };

    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) break;
        const flag = args[i];
        const value = args[i + 1];

        if (std.mem.eql(u8, flag, "--input")) {
            cmd.input_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--output")) {
            cmd.output_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--from")) {
            cmd.from_format = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--to")) {
            cmd.to_format = try allocator.dupe(u8, value);
        }
    }

    return cmd;
}

fn runConvert(allocator: std.mem.Allocator, io: std.Io, cmd: ConvertCommand) !void {
    std.log.info("Converting {s} -> {s}", .{ cmd.input_path, cmd.output_path });

    // Try native Zig conversion first for safetensors → safetensors
    if (std.mem.eql(u8, cmd.from_format, "safetensors") and std.mem.eql(u8, cmd.to_format, "safetensors")) {
        try runNativeConvert(allocator, io, cmd);
        return;
    }

    // Delegate to mlx_lm.convert for HuggingFace model conversion
    const result = std.process.run(allocator, io, .{
        .argv = &.{
            "python3", "-m", "mlx_lm.convert",
            "--hf-path", cmd.input_path,
            "--mlx-path", cmd.output_path,
        },
    }) catch |err| {
        std.log.err("Failed to run mlx_lm.convert: {s}. Ensure mlx-lm is installed (`pip install mlx-lm`).", .{@errorName(err)});
        return err;
    };
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }

    switch (result.term) {
        .exited => |code| if (code != 0) {
            std.log.err("mlx_lm.convert failed:\n{s}", .{result.stderr});
            return error.ConvertFailed;
        },
        else => {
            std.log.err("mlx_lm.convert terminated unexpectedly", .{});
            return error.ConvertFailed;
        },
    }

    std.log.info("Conversion complete. Output saved to {s}", .{cmd.output_path});
    std.debug.print("{s}\n", .{result.stdout});
}

/// Native safetensors → safetensors conversion with optional quantization.
/// Loads weights from the input file, optionally quantizes them, and saves
/// to the output path in MLX safetensors format.
fn runNativeConvert(allocator: std.mem.Allocator, io: std.Io, cmd: ConvertCommand) !void {
    _ = io;
    const mlx_io = root.io;

    std.log.info("Native safetensors conversion: {s} -> {s}", .{ cmd.input_path, cmd.output_path });

    // 1. Load source safetensors
    var loaded = try mlx_io.loadSafetensors(allocator, cmd.input_path);
    defer {
        var it = loaded.weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        loaded.weights.deinit();
        var mit = loaded.metadata.iterator();
        while (mit.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        loaded.metadata.deinit();
    }

    std.log.info("Loaded {d} tensors from {s}", .{ loaded.weights.count(), cmd.input_path });

    // 2. Save to output path (re-serializes in MLX safetensors format)
    try mlx_io.saveSafetensors(allocator, cmd.output_path, loaded.weights, loaded.metadata);

    std.log.info("Conversion complete. Output saved to {s}", .{cmd.output_path});
}

// ------------------------------------------------------------------
// LoRA Train Command
// ------------------------------------------------------------------

fn parseLoraTrainArgs(allocator: std.mem.Allocator, args: []const [:0]const u8) !LoraTrainCommand {
    var cmd = LoraTrainCommand{
        .model_path = "",
        .data_path = "",
        .output_path = "",
    };

    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) break;
        const flag = args[i];
        const value = args[i + 1];

        if (std.mem.eql(u8, flag, "--model")) {
            cmd.model_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--data")) {
            cmd.data_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--output")) {
            cmd.output_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--rank")) {
            cmd.rank = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--alpha")) {
            cmd.alpha = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, flag, "--lr")) {
            cmd.lr = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, flag, "--epochs")) {
            cmd.epochs = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--batch-size")) {
            cmd.batch_size = try std.fmt.parseInt(usize, value, 10);
        }
    }

    return cmd;
}

fn runLoraTrain(allocator: std.mem.Allocator, io: std.Io, cmd: LoraTrainCommand) !void {
    std.log.info("LoRA training: model={s} data={s} rank={d} alpha={d:.0} lr={e:.4} epochs={d}", .{
        cmd.model_path, cmd.data_path, cmd.rank, cmd.alpha, cmd.lr, cmd.epochs,
    });

    const ctx = EagerContext.init(allocator);
    const stream = c.c.mlx_default_cpu_stream_new();

    // 1. Load config
    const config_path = try std.fs.path.join(allocator, &[_][]const u8{ cmd.model_path, "config.json" });
    defer allocator.free(config_path);
    const config = try root.hf_config.loadLlamaConfig(allocator, io, config_path);

    // 2. Load tokenizer
    const tokenizer_path = try std.fs.path.join(allocator, &[_][]const u8{ cmd.model_path, "tokenizer.json" });
    defer allocator.free(tokenizer_path);
    var tokenizer_backend = root.tokenizer.BpeTokenizer.init(allocator);
    defer tokenizer_backend.deinit();
    try tokenizer_backend.loadFromFile(io, tokenizer_path);
    const tokenizer = tokenizer_backend.asStrategy();

    // 3. Load model weights — try multiple safetensors naming conventions (supports multi-shard)
    const weights_paths = findSafetensorsPaths(allocator, cmd.model_path) catch |err| {
        std.log.err("No safetensors file found in {s}: {}", .{ cmd.model_path, err });
        return err;
    };
    defer {
        for (weights_paths) |p| allocator.free(p);
        allocator.free(weights_paths);
    }
    const const_paths_train: []const []const u8 = @ptrCast(weights_paths);
    var model = try root.model_loader.loadFromSafetensorsPaths(allocator, &config, const_paths_train, ctx, stream);
    defer model.deinit();

    // 4. Load dataset
    const dataset = try root.trainer.loadJsonlDataset(allocator, io, cmd.data_path, tokenizer);
    defer dataset.deinit(allocator);

    if (dataset.len() == 0) {
        std.log.warn("Dataset is empty.", .{});
        return;
    }

    // 5. Create LoRA adapters
    var lora_model = root.lora.LoRAModel.init(allocator);
    defer lora_model.deinit();

    const hidden_size = config.hidden_size;
    const intermediate_size = config.intermediate_size;
    const num_kv_heads = config.num_key_value_heads;
    const num_heads = config.num_attention_heads;
    const head_dim = hidden_size / num_heads;
    const kv_dim = num_kv_heads * head_dim;

    for (0..config.num_hidden_layers) |i| {
        var buf: [128]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&buf, "layers.{d}.", .{i});

        // Attention
        const wq_key = try std.fmt.allocPrint(allocator, "{s}attention.wq", .{prefix});
        defer allocator.free(wq_key);
        try lora_model.addAdapter(wq_key, ctx, hidden_size, hidden_size, cmd.rank, cmd.alpha, stream);

        const wk_key = try std.fmt.allocPrint(allocator, "{s}attention.wk", .{prefix});
        defer allocator.free(wk_key);
        try lora_model.addAdapter(wk_key, ctx, hidden_size, kv_dim, cmd.rank, cmd.alpha, stream);

        const wv_key = try std.fmt.allocPrint(allocator, "{s}attention.wv", .{prefix});
        defer allocator.free(wv_key);
        try lora_model.addAdapter(wv_key, ctx, hidden_size, kv_dim, cmd.rank, cmd.alpha, stream);

        const wo_key = try std.fmt.allocPrint(allocator, "{s}attention.wo", .{prefix});
        defer allocator.free(wo_key);
        try lora_model.addAdapter(wo_key, ctx, hidden_size, hidden_size, cmd.rank, cmd.alpha, stream);

        // MLP
        const gate_key = try std.fmt.allocPrint(allocator, "{s}mlp.gate_proj", .{prefix});
        defer allocator.free(gate_key);
        try lora_model.addAdapter(gate_key, ctx, intermediate_size, hidden_size, cmd.rank, cmd.alpha, stream);

        const up_key = try std.fmt.allocPrint(allocator, "{s}mlp.up_proj", .{prefix});
        defer allocator.free(up_key);
        try lora_model.addAdapter(up_key, ctx, intermediate_size, hidden_size, cmd.rank, cmd.alpha, stream);

        const down_key = try std.fmt.allocPrint(allocator, "{s}mlp.down_proj", .{prefix});
        defer allocator.free(down_key);
        try lora_model.addAdapter(down_key, ctx, hidden_size, intermediate_size, cmd.rank, cmd.alpha, stream);
    }

    model.lora = &lora_model;

    // 6. Create optimizer (only LoRA parameters)
    const lora_ptrs = try lora_model.collectParamPtrs(allocator);
    defer allocator.free(lora_ptrs);

    var optimizer = try root.optim.AdamW.init(allocator, lora_ptrs, cmd.lr, 0.9, 0.999, 1e-8, 0.01, stream);
    defer optimizer.deinit();

    // 7. Create trainer
    const trainer_config = root.trainer.TrainerConfig{
        .max_seq_len = config.max_position_embeddings,
        .lr_schedule = .{ .constant = .{ .lr = cmd.lr } },
    };
    var trainer = try root.trainer.SFTTrainer.init(allocator, &model, &optimizer, trainer_config, ctx, stream, &lora_model);
    defer trainer.deinit();

    // 8. Train
    try trainer.train(dataset, cmd.epochs, cmd.batch_size);

    // 9. Save final checkpoint
    if (cmd.output_path.len > 0) {
        try trainer.saveCheckpoint(cmd.output_path);
    }

    std.log.info("Training complete. Output saved to {s}", .{cmd.output_path});
}

// ------------------------------------------------------------------
// JANG Convert Command
// ------------------------------------------------------------------

fn parseJangConvertArgs(allocator: std.mem.Allocator, args: []const [:0]const u8) !JangConvertCommand {
    var cmd = JangConvertCommand{
        .model_path = "",
        .output_path = "",
        .profile = "",
    };

    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) break;
        const flag = args[i];
        const value = args[i + 1];

        if (std.mem.eql(u8, flag, "--model")) {
            cmd.model_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--output")) {
            cmd.output_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--profile")) {
            cmd.profile = try allocator.dupe(u8, value);
        }
    }

    if (cmd.model_path.len == 0 or cmd.output_path.len == 0 or cmd.profile.len == 0) {
        return error.MissingRequiredArgument;
    }

    return cmd;
}

fn parseJangProfile(profile_str: []const u8) !root.jang_quantizer.JangProfile {
    if (std.mem.eql(u8, profile_str, "2M")) return .JANG_2M;
    if (std.mem.eql(u8, profile_str, "2L")) return .JANG_2L;
    if (std.mem.eql(u8, profile_str, "3M")) return .JANG_3M;
    if (std.mem.eql(u8, profile_str, "4M")) return .JANG_4M;
    if (std.mem.eql(u8, profile_str, "6M")) return .JANG_6M;
    return error.InvalidArgument;
}

fn runJangConvert(allocator: std.mem.Allocator, io: std.Io, cmd: JangConvertCommand) !void {
    _ = io;
    const profile = try parseJangProfile(cmd.profile);

    std.log.info("JANG conversion: model={s} output={s} profile={s}", .{
        cmd.model_path, cmd.output_path, cmd.profile,
    });

    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    try root.jang_quantizer.quantizeModel(allocator, cmd.model_path, cmd.output_path, profile, null, ctx);
    std.log.info("JANG conversion complete.", .{});
}

// ------------------------------------------------------------------
// Image Generation Command
// ------------------------------------------------------------------

fn parseImageGenArgs(allocator: std.mem.Allocator, args: []const [:0]const u8) !ImageGenCommand {
    var cmd = ImageGenCommand{
        .model_path = "",
        .prompt = "",
        .output_path = "",
    };

    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) break;
        const flag = args[i];
        const value = args[i + 1];

        if (std.mem.eql(u8, flag, "--model")) {
            cmd.model_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--prompt")) {
            cmd.prompt = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, flag, "--height")) {
            cmd.height = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--width")) {
            cmd.width = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--steps")) {
            cmd.steps = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--output")) {
            cmd.output_path = try allocator.dupe(u8, value);
        }
    }

    if (cmd.model_path.len == 0 or cmd.prompt.len == 0 or cmd.output_path.len == 0) {
        return error.MissingRequiredArgument;
    }

    return cmd;
}

fn runImageGen(allocator: std.mem.Allocator, io: std.Io, cmd: ImageGenCommand) !void {
    _ = io;
    std.log.info("Image generation: model={s} prompt=\"{s}\" height={d} width={d} steps={d} output={s}", .{
        cmd.model_path, cmd.prompt, cmd.height, cmd.width, cmd.steps, cmd.output_path,
    });

    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    // Stub: use default Flux config; full integration would load from model_path
    const config = root.diffusion_flux.FluxConfig{};
    var pipeline = try root.diffusion_flux.FluxPipeline.init(allocator, config, ctx);
    defer pipeline.deinit();

    // Stub prompt embeddings (text encoder not yet integrated)
    const prompt_embeds = try root.array.zeros(allocator, &[_]i32{ 1, 1, @intCast(config.hidden_size) }, .float32);
    defer prompt_embeds.deinit();

    const result = try pipeline.generate(ctx, prompt_embeds, cmd.height, cmd.width, cmd.steps);
    defer result.deinit();

    std.log.info("Image generation complete. Output would be saved to {s} (VAE decode stub).", .{cmd.output_path});
}
