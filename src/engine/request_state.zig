/// Per-request isolated state for the Server Engine V2.
///
/// Each concurrent HTTP request gets its own heap-allocated RequestState,
/// eliminating the mutable-static data race in the V1 StreamState design.
/// The HTTP handler fiber owns the RequestState lifecycle (alloc + free).
const std = @import("std");
const completion_signal = @import("completion_signal.zig");

pub const CompletionSignal = completion_signal.CompletionSignal;
pub const TokenEvent = completion_signal.TokenEvent;

/// API format for response formatting.
pub const ApiFormat = enum {
    openai,
    anthropic,
};

/// Reason generation finished.
pub const FinishReason = enum {
    stop,
    length,
    error_,
};

/// Configuration for creating a new request.
pub const RequestConfig = struct {
    prompt_tokens: []const u32,
    max_tokens: usize,
    temperature: f32 = 0.8,
    top_k: usize = 50,
    top_p: f32 = 1.0,
    seed: u64 = 0,
    stop_strings: ?[]const []const u8 = null,
    stop_tokens: []const u32 = &.{},
    streaming: bool = true,
    model_name: []const u8 = "default",
    api_format: ApiFormat = .openai,
    speculative_ngram: ?usize = null,
};

/// Per-request mutable state, heap-allocated and owned by the HTTP handler fiber.
///
/// The Engine fiber writes to `generated_tokens`, `text_buffer`, `token_count`,
/// `done`, `finish_reason`, and `error_msg` — but only through the
/// CompletionSignal mechanism, which provides synchronization.
pub const RequestState = struct {
    /// Unique request ID for correlation.
    id: u64,
    /// Allocator used for this request's lifetime.
    allocator: std.mem.Allocator,
    /// Prompt token IDs (owned copy, freed on deinit).
    prompt_tokens: []const u32,
    /// Generation configuration.
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    /// Whether this is a streaming request.
    streaming: bool,
    /// Stop strings for early termination.
    stop_strings: ?[]const []const u8,
    /// Stop token IDs.
    stop_tokens: []const u32,
    /// API format (OpenAI or Anthropic).
    api_format: ApiFormat,
    /// Model name for response formatting.
    model_name: []const u8,

    // --- Mutable state (written by Engine via signal, read by HTTP after signal) ---
    /// Accumulated generated tokens.
    generated_tokens: std.ArrayList(u32),
    /// Decoded text buffer for stop-string detection.
    text_buffer: std.ArrayList(u8),
    /// Number of tokens generated so far.
    token_count: usize,
    /// Whether generation is complete.
    done: bool,
    /// Finish reason (null while generating).
    finish_reason: ?FinishReason,
    /// Error message if generation failed.
    error_msg: ?[]const u8,

    // --- Synchronization ---
    /// Signal for engine→HTTP token delivery.
    completion: CompletionSignal,
    /// Whether cancellation has been requested (by HTTP fiber on disconnect).
    cancel_requested: std.atomic.Value(bool),

    /// Allocate and initialize a new RequestState on the heap.
    pub fn init(allocator: std.mem.Allocator, id: u64, config: RequestConfig) !*RequestState {
        const prompt_copy = try allocator.dupe(u32, config.prompt_tokens);
        errdefer allocator.free(prompt_copy);

        const self = try allocator.create(RequestState);
        self.* = .{
            .id = id,
            .allocator = allocator,
            .prompt_tokens = prompt_copy,
            .max_tokens = config.max_tokens,
            .temperature = config.temperature,
            .top_k = config.top_k,
            .top_p = config.top_p,
            .seed = config.seed,
            .streaming = config.streaming,
            .stop_strings = config.stop_strings,
            .stop_tokens = config.stop_tokens,
            .api_format = config.api_format,
            .model_name = config.model_name,
            .generated_tokens = std.ArrayList(u32).empty,
            .text_buffer = std.ArrayList(u8).empty,
            .token_count = 0,
            .done = false,
            .finish_reason = null,
            .error_msg = null,
            .completion = CompletionSignal.init(allocator),
            .cancel_requested = std.atomic.Value(bool).init(false),
        };
        return self;
    }

    /// Free all owned memory and destroy the RequestState.
    pub fn deinit(self: *RequestState) void {
        const allocator = self.allocator;
        allocator.free(self.prompt_tokens);
        self.generated_tokens.deinit(allocator);
        self.text_buffer.deinit(allocator);
        self.completion.deinit();
        allocator.destroy(self);
    }

    /// Check if this request has been cancelled by the HTTP fiber.
    pub fn isCancelled(self: *const RequestState) bool {
        return self.cancel_requested.load(.acquire);
    }

    /// Request cancellation (called by HTTP fiber on client disconnect).
    pub fn requestCancel(self: *RequestState) void {
        self.cancel_requested.store(true, .release);
    }
};
