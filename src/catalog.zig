/// Registry of first-party supported models: aliases, source repos, hardware
/// tiers, and install locations. Shared by `dmlx pull` (downloads),
/// `dmlx doctor` (health checks), and the `--model` alias resolution used by
/// serve/chat/benchmark.
const std = @import("std");

pub const Role = enum {
    /// Standalone chat/serve model.
    chat_model,
    /// MTP speculative-decoding drafter; not usable alone.
    mtp_drafter,
    /// Large MoE served through the native SSD-streaming path.
    moe_native,
};

pub const Entry = struct {
    /// Short name accepted by `--model` and `dmlx pull`.
    alias: []const u8,
    /// Hugging Face repo ("org/name"). These are the exact upstream ids the
    /// local reference checkpoints were taken from; do not silently retarget
    /// to third-party derivatives.
    hf_repo: []const u8,
    blurb: []const u8,
    approx_size_gb: f64,
    /// Minimum unified memory (GB) for the documented configuration.
    min_ram_gb: u32,
    role: Role,
    /// Companion model alias, e.g. the MTP drafter paired with a target.
    companion: ?[]const u8 = null,
    /// Directory name under the models cache dir.
    dir_name: []const u8,
    /// Files that must exist for an install to count as complete.
    required_files: []const []const u8,
    /// DeepSeek V4 native path requires packed_experts/ next to the weights.
    needs_packed_experts: bool = false,
};

pub const entries = [_]Entry{
    .{
        .alias = "qwen3.8-27b-4bit",
        .hf_repo = "mlx-community/Qwen3.8-27B-4bit",
        .blurb = "Qwen3.8 27B chat, MLX 4-bit, vision-capable (flagship)",
        .approx_size_gb = 15.5,
        .min_ram_gb = 32,
        .role = .chat_model,
        .companion = "qwen3.8-27b-mtp-4bit",
        .dir_name = "Qwen3.8-27B-4bit",
        .required_files = &.{ "config.json", "tokenizer.json", "model.safetensors.index.json" },
    },
    .{
        .alias = "qwen3.8-27b-mtp-4bit",
        .hf_repo = "mlx-community/Qwen3.8-27B-MTP-4bit",
        .blurb = "MTP drafter for qwen3.8-27b-4bit speculative decoding",
        .approx_size_gb = 1.0,
        .min_ram_gb = 32,
        .role = .mtp_drafter,
        .companion = "qwen3.8-27b-4bit",
        .dir_name = "Qwen3.8-27B-MTP-4bit",
        .required_files = &.{ "config.json", "model.safetensors" },
    },
    .{
        .alias = "deepseek-v4-flash-4bit",
        .hf_repo = "mlx-community/DeepSeek-V4-Flash-4bit",
        .blurb = "DeepSeek V4 Flash 284B MoE (native SSD expert streaming)",
        .approx_size_gb = 150,
        .min_ram_gb = 48,
        .role = .moe_native,
        .dir_name = "DeepSeek-V4-Flash-4bit",
        .required_files = &.{ "config.json", "tokenizer.json" },
        .needs_packed_experts = true,
    },
};

pub fn find(name: []const u8) ?*const Entry {
    for (&entries) |*e| {
        if (std.ascii.eqlIgnoreCase(e.alias, name)) return e;
        if (std.ascii.eqlIgnoreCase(e.dir_name, name)) return e;
        if (std.ascii.eqlIgnoreCase(e.hf_repo, name)) return e;
    }
    return null;
}

/// Directory where pulled models live: $DMLX_MODEL_HOME or
/// ~/.cache/dmlx/models. Caller owns the returned slice.
pub fn modelsDir(allocator: std.mem.Allocator) ![]u8 {
    if (std.c.getenv("DMLX_MODEL_HOME")) |p| {
        const sp = std.mem.span(p);
        if (sp.len > 0) return allocator.dupe(u8, sp);
    }
    const home = std.mem.span(std.c.getenv("HOME") orelse return error.NoHome);
    return std.fs.path.join(allocator, &.{ home, ".cache", "dmlx", "models" });
}

/// Absolute install path of a registry entry, whether or not it exists.
/// Caller owns the returned slice.
pub fn installPath(allocator: std.mem.Allocator, entry: *const Entry) ![]u8 {
    const dir = try modelsDir(allocator);
    defer allocator.free(dir);
    return std.fs.path.join(allocator, &.{ dir, entry.dir_name });
}

fn dirExists(io: std.Io, abs_path: []const u8) bool {
    std.Io.Dir.accessAbsolute(io, abs_path, .{}) catch return false;
    return true;
}

/// Where an installed copy of `entry` actually is: the pull cache first, then
/// ~/models/<dir_name> for developer setups. Returns null when not installed.
/// Caller owns the returned slice.
pub fn installedPath(allocator: std.mem.Allocator, io: std.Io, entry: *const Entry) !?[]u8 {
    const cache_path = try installPath(allocator, entry);
    if (dirExists(io, cache_path)) return cache_path;
    allocator.free(cache_path);

    const home = std.mem.span(std.c.getenv("HOME") orelse return null);
    const dev_path = try std.fs.path.join(allocator, &.{ home, "models", entry.dir_name });
    if (dirExists(io, dev_path)) return dev_path;
    allocator.free(dev_path);
    return null;
}

/// True when every required file of the entry exists under `dir_path`.
pub fn isCompleteInstall(io: std.Io, dir_path: []const u8, entry: *const Entry) bool {
    var buf: [4096]u8 = undefined;
    for (entry.required_files) |rel| {
        const p = std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir_path, rel }) catch return false;
        std.Io.Dir.accessAbsolute(io, p, .{}) catch return false;
    }
    return true;
}

/// Resolve a `--model` argument. Existing filesystem paths pass through
/// unchanged; registry aliases resolve to their installed location. Unknown
/// values are passed through so downstream error messages keep the original
/// text. Caller owns the returned slice.
pub fn resolveModelArg(allocator: std.mem.Allocator, io: std.Io, arg: []const u8) ![]u8 {
    if (std.mem.indexOfScalar(u8, arg, '/') != null) return allocator.dupe(u8, arg);
    // Relative bare name: only treat it as a path if a directory by that
    // name exists in the cwd.
    if (std.Io.Dir.cwd().access(io, arg, .{})) {
        return allocator.dupe(u8, arg);
    } else |_| {}

    if (find(arg)) |entry| {
        if (try installedPath(allocator, io, entry)) |p| return p;
        std.log.err(
            "model '{s}' is not installed. Run: dmlx pull {s}",
            .{ arg, entry.alias },
        );
        return error.ModelNotInstalled;
    }
    return allocator.dupe(u8, arg);
}
