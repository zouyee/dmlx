/// OpenAI-compatible chat completion handler.
const std = @import("std");
const root = @import("../root.zig");
const c = @import("mlx").c;
const generation_mod = @import("../generation.zig");
const kvcache = @import("../kvcache.zig");
const tool_calling_mod = @import("../tool_calling.zig");
const tool_executor_mod = @import("../tool_executor.zig");
const engine = @import("../engine/root.zig");
const config_mod = @import("config.zig");
const utils_mod = @import("utils.zig");
const state_mod = @import("state.zig");
const vision_prompt = @import("../vision_prompt.zig");
const qwen38_preprocess = @import("../vision/qwen38_preprocess.zig");
const metrics_mod = @import("metrics.zig");
const jsonEscapeInto = utils_mod.jsonEscapeInto;

const ServerConfig = config_mod.ServerConfig;
const ServerState = state_mod.ServerState;

pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []Message,
    stream: ?bool = false,
    temperature: ?f32 = null,
    /// Per-request nucleus sampling override (must be in (0, 1]).
    top_p: ?f32 = null,
    /// Per-request top-k override (must be >= 1).
    top_k: ?u32 = null,
    max_tokens: ?u32 = null,
    /// Newer OpenAI clients (and gateways like pi-ai) send
    /// `max_completion_tokens` instead of `max_tokens`; honored as a
    /// fallback when `max_tokens` is absent.
    max_completion_tokens: ?u32 = null,
    /// Optional seed for deterministic sampling.
    seed: ?u64 = null,
    /// Optional stop strings — generation stops when any of these are produced.
    stop: ?[]const []const u8 = null,
    /// Optional response format constraint for guided decoding.
    /// When `type` is "json_schema", the `schema` field contains a JSON schema string.
    /// When `type` is "regex", the `schema` field contains a regex pattern.
    /// Currently parsed but not enforced — see TODO comments in generation functions.
    response_format: ?ResponseFormat = null,
    /// Optional tool definitions for function calling.
    /// When provided, the model output is scanned for tool call patterns.
    tools: ?[]const ToolDefinition = null,
    /// Optional tool_choice: "auto" (default), "none", "required", or
    /// {"type":"function","function":{"name":...}}. v1 supports only "auto"
    /// and "none" — see resolveToolChoice.
    tool_choice: ?std.json.Value = null,
    /// stream_options.include_usage: when true, the stream ends with a
    /// usage chunk (empty choices) before [DONE], per the OpenAI streaming
    /// contract. Agent harnesses (pi-ai/dsh) read it for token accounting.
    stream_options: ?StreamOptions = null,

    pub const StreamOptions = struct {
        include_usage: ?bool = null,
    };

    pub const Message = struct {
        role: []const u8,
        /// String content, or an array of content parts
        /// ({"type":"text","text":...} / {"type":"image_url","image_url":{"url":...}}).
        content: std.json.Value,
        /// Historical assistant tool calls (OpenAI shape), re-rendered into
        /// the model's native call format when building the prompt.
        tool_calls: ?std.json.Value = null,
        /// Present on role=tool messages; accepted for schema compatibility.
        tool_call_id: ?[]const u8 = null,
        /// Present on some clients' tool messages; accepted and unused.
        name: ?[]const u8 = null,
    };

    pub const ResponseFormat = struct {
        type: []const u8, // "json_schema" or "regex"
        schema: ?[]const u8 = null, // JSON schema string or regex pattern
    };

    pub const ToolDefinition = struct {
        type: []const u8 = "function",
        function: ?ToolFunction = null,

        pub const ToolFunction = struct {
            name: []const u8,
            description: ?[]const u8 = null,
            parameters: ?std.json.Value = null,
        };
    };
};

/// Decoded image payload cap (20 MB) — plan risk register.
pub const max_decoded_image_bytes: usize = 20 * 1024 * 1024;

/// Validate per-request sampling overrides. top_p must be in (0, 1],
/// top_k must be >= 1 (0 would be meaningless / trip the sampler).
pub fn validateSampling(req: ChatCompletionRequest) !void {
    if (req.top_p) |p| {
        if (!(p > 0.0 and p <= 1.0)) return error.InvalidTopP;
    }
    if (req.top_k) |k| {
        if (k == 0) return error.InvalidTopK;
    }
}

/// Apply the request's tool_choice under the v1 support policy:
///   - "auto" (default): current behavior — tool definitions are injected
///     into the prompt and the output is scanned for tool-call patterns.
///   - "none": strip the tools entirely, behaving as if none were given.
///   - "required" / named-function ({"type":"function","function":...}):
///     forcing a specific call needs prompt- or sampler-level machinery the
///     codebase does not have (tool_calling only detects calls in free-form
///     output), so these return error.UnsupportedToolChoice rather than
///     silently behaving like "auto".
pub fn resolveToolChoice(req: *ChatCompletionRequest) !void {
    const tc = req.tool_choice orelse return;
    switch (tc) {
        .string => |s| {
            if (std.mem.eql(u8, s, "auto")) return;
            if (std.mem.eql(u8, s, "none")) {
                req.tools = null;
                return;
            }
            if (std.mem.eql(u8, s, "required")) return error.UnsupportedToolChoice;
            return error.InvalidToolChoice;
        },
        .object => return error.UnsupportedToolChoice,
        else => return error.InvalidToolChoice,
    }
}

/// Build an OpenAI-style error body for a rejected chat request.
pub fn invalidRequestJson(allocator: std.mem.Allocator, code: []const u8, message: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\{{"error":{{"message":"{s}","type":"invalid_request_error","code":"{s}"}}}}
    , .{ message, code });
}

/// Validate sampling params and tool_choice support; on failure returns
/// the error body to send back (caller frees). Shared by the HTTP dispatch
/// (400) and the non-streaming handler (defensive re-check).
pub fn validateChatRequest(allocator: std.mem.Allocator, req: ChatCompletionRequest) !?[]u8 {
    validateSampling(req) catch |err| {
        const msg: []const u8 = switch (err) {
            error.InvalidTopP => "top_p must be in the range (0, 1]",
            error.InvalidTopK => "top_k must be >= 1",
        };
        return try invalidRequestJson(allocator, @errorName(err), msg);
    };
    var mut = req;
    resolveToolChoice(&mut) catch |err| {
        const msg: []const u8 = switch (err) {
            error.UnsupportedToolChoice => "tool_choice 'required' and named-function selection are not supported by this server (use 'auto' or 'none')",
            error.InvalidToolChoice => "tool_choice must be 'auto', 'none', 'required', or an object selecting a function",
        };
        return try invalidRequestJson(allocator, @errorName(err), msg);
    };
    return null;
}

/// Extract the text of one message and collect any image inputs. Content
/// may be a plain string or an array of text/image_url parts.
fn extractMessageContent(
    allocator: std.mem.Allocator,
    content: std.json.Value,
    images: *std.ArrayList(generation_mod.VisionImage),
    decoded: *std.ArrayList([]u8),
) ![]const u8 {
    switch (content) {
        .string => |s| return allocator.dupe(u8, s),
        // Null content is legal on assistant messages that carry tool_calls.
        .null => return allocator.dupe(u8, ""),
        .array => |parts| {
            var text = std.ArrayList(u8).empty;
            errdefer text.deinit(allocator);
            for (parts.items) |part| {
                const obj = switch (part) {
                    .object => |o| o,
                    else => return error.InvalidContent,
                };
                const ty = switch (obj.get("type") orelse return error.InvalidContent) {
                    .string => |s| s,
                    else => return error.InvalidContent,
                };
                if (std.mem.eql(u8, ty, "text")) {
                    switch (obj.get("text") orelse return error.InvalidContent) {
                        .string => |s| try text.appendSlice(allocator, s),
                        else => return error.InvalidContent,
                    }
                } else if (std.mem.eql(u8, ty, "image_url")) {
                    const url = blk: {
                        const iu = obj.get("image_url") orelse return error.InvalidContent;
                        switch (iu) {
                            .string => |s| break :blk s,
                            .object => |io| break :blk switch (io.get("url") orelse return error.InvalidContent) {
                                .string => |s| s,
                                else => return error.InvalidContent,
                            },
                            else => return error.InvalidContent,
                        }
                    };
                    try loadImageRef(allocator, url, images, decoded);
                } else {
                    return error.InvalidContent;
                }
            }
            return text.toOwnedSlice(allocator);
        },
        else => return error.InvalidContent,
    }
}

/// Resolve one image_url: data URLs are base64-decoded inline (20 MB
/// decoded cap; note the 16 MB HTTP body cap in http.zig makes this
/// effectively ~12 MB of decoded image per request, since the base64 text
/// inflates the payload by ~4/3); absolute file paths are passed through
/// for local dev. Remote URLs are not fetched in v1.
fn loadImageRef(
    allocator: std.mem.Allocator,
    url: []const u8,
    images: *std.ArrayList(generation_mod.VisionImage),
    decoded: *std.ArrayList([]u8),
) !void {
    if (std.mem.startsWith(u8, url, "data:")) {
        const comma = std.mem.indexOfScalar(u8, url, ',') orelse return error.BadImageUrl;
        const meta = url[5..comma];
        if (!std.mem.endsWith(u8, meta, ";base64")) return error.BadImageUrl;
        const b64 = std.mem.trim(u8, url[comma + 1 ..], " \t\r\n");
        const size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch |e| {
            std.log.warn("image_url base64 size failed ({s}), b64 len={d} tail={s}", .{ @errorName(e), b64.len, b64[@max(b64.len, 8) - 8 ..] });
            return error.BadImageUrl;
        };
        if (size > max_decoded_image_bytes) return error.ImageTooLarge;
        const buf = try allocator.alloc(u8, size);
        errdefer allocator.free(buf);
        std.base64.standard.Decoder.decode(buf, b64) catch return error.BadImageUrl;
        try decoded.append(allocator, buf);
        try images.append(allocator, .{ .bytes = buf });
        return;
    }
    if (std.mem.startsWith(u8, url, "/")) {
        try images.append(allocator, .{ .path = url });
        return;
    }
    return error.BadImageUrl;
}

/// Prompt ready for the engine: token ids + (for image requests) the
/// tower features/blocks the vision prefill consumes.
pub const PreparedPrompt = struct {
    tokens: []u32, // owned
    vision: ?generation_mod.VisionData, // owned when set

    pub fn deinit(self: *PreparedPrompt, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        if (self.vision) |*v| v.deinit(allocator);
    }
};

/// Build the chat-template prompt from OpenAI messages, handling both
/// string content and array content parts. Image parts run through the
/// model's VisionOps (preprocess + ViT tower) inline.
pub fn preparePrompt(
    allocator: std.mem.Allocator,
    state: *ServerState,
    req: ChatCompletionRequest,
    extra_system: ?[]const u8,
) !PreparedPrompt {
    var images = std.ArrayList(generation_mod.VisionImage).empty;
    defer images.deinit(allocator);
    var decoded = std.ArrayList([]u8).empty;
    defer {
        for (decoded.items) |b| allocator.free(b);
        decoded.deinit(allocator);
    }

    const texts = try allocator.alloc([]const u8, req.messages.len);
    var n_texts: usize = 0;
    defer {
        for (texts[0..n_texts]) |t| allocator.free(t);
        allocator.free(texts);
    }
    const img_counts = try allocator.alloc(usize, req.messages.len);
    defer allocator.free(img_counts);
    for (req.messages, 0..) |msg, i| {
        const before = images.items.len;
        texts[i] = try extractMessageContent(allocator, msg.content, &images, &decoded);
        img_counts[i] = images.items.len - before;
        n_texts += 1;
    }

    if (images.items.len == 0) {
        const tokens = try tokenizeMessages(allocator, state, req.messages, texts, extra_system, null);
        return .{ .tokens = tokens, .vision = null };
    }

    const vops = state.vtable.vision orelse return error.VisionNotSupported;
    var proc = try vision_prompt.processImages(allocator, state.ctx, vops, state.vtable.ptr, images.items);
    defer allocator.free(proc.blocks_text);
    errdefer proc.data.deinit(allocator);

    // Keep each image in its OWN user message: prepending every block to the
    // last user message (the old behavior) relocates the image on every turn,
    // so an earlier turn's prompt is never a token prefix of the next one and
    // the model-side prefix snapshot can never hit.
    const inserts = try buildVisionInserts(allocator, req.messages.len, img_counts, proc.data.blocks);
    defer freeVisionInserts(allocator, inserts);

    const tokens = try tokenizeMessages(allocator, state, req.messages, texts, extra_system, inserts);
    errdefer allocator.free(tokens);
    try vision_prompt.validatePadCount(tokens, vops.image_token_id, proc.n_pads);
    return .{ .tokens = tokens, .vision = proc.data };
}

/// One "<|vision_start|><|image_pad|>*n<|vision_end|>" literal block.
fn appendImageBlock(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, n_pads: usize) !void {
    try buf.appendSlice(allocator, "<|vision_start|>");
    for (0..n_pads) |_| try buf.appendSlice(allocator, "<|image_pad|>");
    try buf.appendSlice(allocator, "<|vision_end|>");
}

/// Per-image pad count from its merge-divided grid (t * h_merge * w_merge).
fn gridPads(g: [3]i32) usize {
    return @as(usize, @intCast(g[0])) * @as(usize, @intCast(g[1])) * @as(usize, @intCast(g[2]));
}

/// Build the per-message vision block text: inserts[i] holds the blocks of
/// the images that appeared in message i (empty slice when none), in
/// in-message order. grids are the tower blocks for ALL images in the
/// request, in appearance order.
fn buildVisionInserts(
    allocator: std.mem.Allocator,
    n_messages: usize,
    img_counts: []const usize,
    grids: []const [3]i32,
) ![]const []const u8 {
    const inserts = try allocator.alloc([]const u8, n_messages);
    var built: usize = 0;
    errdefer {
        for (inserts[0..built]) |s| allocator.free(s);
        allocator.free(inserts);
    }
    var img_idx: usize = 0;
    for (0..n_messages) |i| {
        if (img_counts[i] == 0) {
            inserts[i] = "";
            built += 1;
            continue;
        }
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        for (0..img_counts[i]) |_| {
            try appendImageBlock(&buf, allocator, gridPads(grids[img_idx]));
            img_idx += 1;
        }
        inserts[i] = try buf.toOwnedSlice(allocator);
        built += 1;
    }
    return inserts;
}

fn freeVisionInserts(allocator: std.mem.Allocator, inserts: []const []const u8) void {
    for (inserts) |s| {
        if (s.len > 0) allocator.free(s);
    }
    allocator.free(inserts);
}

/// Per-message literal vision block text: inserts[i] is prepended to
/// message i's text (empty slice = no images in that message).
/// Token count for a chat request WITHOUT running generation: same chat
/// template + tokenizer as preparePrompt, so the result equals the chat
/// path's usage.prompt_tokens for the same messages.
///
/// Image semantics: image parts are preprocessed (smart_resize + patch
/// count) but do NOT run the ViT tower — each image contributes exactly its
/// n_image_pad <|image_pad|> tokens inside the standard
/// <|vision_start|>...<|vision_end|> block (identical to what the real
/// generation prompt carries). Pad counting currently uses the Qwen3.8
/// preprocessor; a request with images on a model without vision ops
/// returns error.VisionNotSupported.
pub fn countPromptTokens(
    allocator: std.mem.Allocator,
    state: *ServerState,
    req: ChatCompletionRequest,
    extra_system: ?[]const u8,
) !usize {
    var images = std.ArrayList(generation_mod.VisionImage).empty;
    defer images.deinit(allocator);
    var decoded = std.ArrayList([]u8).empty;
    defer {
        for (decoded.items) |b| allocator.free(b);
        decoded.deinit(allocator);
    }

    const texts = try allocator.alloc([]const u8, req.messages.len);
    var n_texts: usize = 0;
    defer {
        for (texts[0..n_texts]) |t| allocator.free(t);
        allocator.free(texts);
    }
    const img_counts = try allocator.alloc(usize, req.messages.len);
    defer allocator.free(img_counts);
    for (req.messages, 0..) |msg, i| {
        const before = images.items.len;
        texts[i] = try extractMessageContent(allocator, msg.content, &images, &decoded);
        img_counts[i] = images.items.len - before;
        n_texts += 1;
    }

    if (images.items.len == 0) {
        const tokens = try tokenizeMessages(allocator, state, req.messages, texts, extra_system, null);
        defer allocator.free(tokens);
        return tokens.len;
    }

    // Preprocess-only pad counting (no tower forward — token count needs
    // just the grid, and the tower would waste a GPU forward per image).
    _ = state.vtable.vision orelse return error.VisionNotSupported;
    const inserts = try allocator.alloc([]const u8, req.messages.len);
    var n_inserts: usize = 0;
    defer {
        for (inserts[0..n_inserts]) |s| {
            if (s.len > 0) allocator.free(s);
        }
        allocator.free(inserts);
    }
    var img_idx: usize = 0;
    for (0..req.messages.len) |i| {
        if (img_counts[i] == 0) {
            inserts[i] = "";
            n_inserts += 1;
            continue;
        }
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        for (0..img_counts[i]) |_| {
            var pp = switch (images.items[img_idx]) {
                .path => |p| try qwen38_preprocess.preprocessImageFile(allocator, state.ctx, p),
                .bytes => |b| try qwen38_preprocess.preprocessImageBytes(allocator, state.ctx, b),
            };
            defer pp.deinit();
            try appendImageBlock(&buf, allocator, pp.n_image_pad);
            img_idx += 1;
        }
        inserts[i] = try buf.toOwnedSlice(allocator);
        n_inserts += 1;
    }

    const tokens = try tokenizeMessages(allocator, state, req.messages, texts, extra_system, inserts);
    defer allocator.free(tokens);
    return tokens.len;
}

/// Apply the chat template to the extracted message texts (optionally with
/// the per-message vision blocks prepended to their own message's text) and
/// tokenize.
///
/// For Qwen-family models, history is rendered in the native chat-template
/// tool format: role=tool messages become user turns wrapping their content
/// in <tool_response>, and assistant tool_calls are re-rendered as native
/// <tool_call><function=...> blocks.
fn tokenizeMessages(
    allocator: std.mem.Allocator,
    state: *ServerState,
    req_messages: []const ChatCompletionRequest.Message,
    texts: []const []const u8,
    extra_system: ?[]const u8,
    vision: ?[]const []const u8,
) ![]u32 {
    const family = tool_calling_mod.detectFamily(state.model_name);
    var messages = std.ArrayList(root.tokenizer.ChatMessage).empty;
    defer messages.deinit(allocator);
    var merged: ?[]u8 = null;
    defer if (merged) |m| allocator.free(m);
    var owned = std.ArrayList([]const u8).empty;
    defer {
        for (owned.items) |s| allocator.free(s);
        owned.deinit(allocator);
    }
    var parsed_args = std.ArrayList(std.json.Parsed(std.json.Value)).empty;
    defer {
        for (parsed_args.items) |*p| p.deinit();
        parsed_args.deinit(allocator);
    }
    if (extra_system) |sys| {
        try messages.append(allocator, .{ .role = "system", .content = sys });
    }
    for (texts, 0..) |t, i| {
        var content = t;
        var role = req_messages[i].role;
        if (vision) |v| {
            if (v[i].len > 0) {
                merged = try std.fmt.allocPrint(allocator, "{s}\n{s}", .{ v[i], t });
                content = merged.?;
            }
        }
        if (family == .qwen) {
            if (std.mem.eql(u8, role, "tool")) {
                const wrapped = try std.fmt.allocPrint(allocator, "<tool_response>\n{s}\n</tool_response>", .{content});
                try owned.append(allocator, wrapped);
                content = wrapped;
                role = "user";
            } else if (std.mem.eql(u8, role, "assistant")) {
                if (req_messages[i].tool_calls) |tc| {
                    if (tc == .array and tc.array.items.len > 0) {
                        var buf = std.ArrayList(u8).empty;
                        try buf.appendSlice(allocator, content);
                        for (tc.array.items) |call| {
                            if (call != .object) continue;
                            const fn_v = call.object.get("function") orelse continue;
                            if (fn_v != .object) continue;
                            const name_v = fn_v.object.get("name") orelse continue;
                            if (name_v != .string) continue;
                            var args_val: ?std.json.Value = null;
                            if (fn_v.object.get("arguments")) |a| {
                                switch (a) {
                                    .string => |s| {
                                        const p = try std.json.parseFromSlice(std.json.Value, allocator, s, .{});
                                        try parsed_args.append(allocator, p);
                                        args_val = parsed_args.items[parsed_args.items.len - 1].value;
                                    },
                                    .object => args_val = a,
                                    else => {},
                                }
                            }
                            try buf.appendSlice(allocator, "\n");
                            try tool_calling_mod.appendQwenNativeCall(&buf, allocator, name_v.string, args_val);
                        }
                        const owned_content = try buf.toOwnedSlice(allocator);
                        try owned.append(allocator, owned_content);
                        content = owned_content;
                    }
                }
            }
        }
        try messages.append(allocator, .{ .role = role, .content = content });
    }
    const prompt_text = try state.chat_template.apply(messages.items, true);
    defer allocator.free(prompt_text);
    return tokenizeWithPrefixReuse(allocator, state, prompt_text);
}

// ------------------------------------------------------------------
// Prompt token prefix cache
// ------------------------------------------------------------------
// Agent harnesses resend the full conversation every turn, so the
// rendered chat prompt shares a long string prefix with the previous
// request. Re-encoding the whole string is NOT token-prefix stable (BPE
// can merge across the old boundary), which defeats the model-side KV
// prefix snapshot's exact token match. Cache rendered prompt → tokens
// and re-encode only the new string suffix; the cached prefix tokens are
// then byte-identical across turns. The suffix is still a valid
// tokenization of its text (decodes to the same string), it may just
// split differently at the seam.
const TokenPrefixEntry = struct {
    prompt: []u8,
    tokens: []u32,
    last_access: u64,
};
const token_prefix_cap = 4;
var token_prefix_cache: [token_prefix_cap]?TokenPrefixEntry = .{ null, null, null, null };
var token_prefix_access: u64 = 0;
var token_prefix_lock = std.atomic.Value(u32).init(0);

fn tpLock() void {
    while (token_prefix_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}
fn tpUnlock() void {
    token_prefix_lock.store(0, .release);
}

fn tokenizeWithPrefixReuse(allocator: std.mem.Allocator, state: *ServerState, prompt_text: []const u8) ![]u32 {
    // Lookup: longest cached prompt that is a prefix of this one.
    var hit: ?usize = null;
    var hit_len: usize = 0;
    tpLock();
    for (&token_prefix_cache, 0..) |*e, i| {
        if (e.*) |ent| {
            if (ent.prompt.len <= prompt_text.len and ent.prompt.len > hit_len and
                std.mem.eql(u8, ent.prompt, prompt_text[0..ent.prompt.len]))
            {
                hit = i;
                hit_len = ent.prompt.len;
            }
        }
    }
    var prefix_tokens: ?[]u32 = null;
    if (hit) |i| {
        token_prefix_access += 1;
        token_prefix_cache[i].?.last_access = token_prefix_access;
        prefix_tokens = allocator.dupe(u32, token_prefix_cache[i].?.tokens) catch null;
    }
    tpUnlock();

    var tokens: []u32 = undefined;
    if (prefix_tokens) |pt| {
        defer allocator.free(pt);
        if (hit_len == prompt_text.len) {
            tokens = try allocator.dupe(u32, pt);
        } else {
            const suffix = try state.tokenizer_strategy.encode(prompt_text[hit_len..], false, allocator);
            defer allocator.free(suffix);
            tokens = try allocator.alloc(u32, pt.len + suffix.len);
            @memcpy(tokens[0..pt.len], pt);
            @memcpy(tokens[pt.len..], suffix);
        }
        std.log.info("[openai] token prefix cache hit: {d}/{d} tokens reused", .{ pt.len, tokens.len });
    } else {
        tokens = try state.tokenizer_strategy.encode(prompt_text, false, allocator);
    }
    errdefer allocator.free(tokens);

    // Store: replace the identical prompt, else evict the oldest.
    tpLock();
    defer tpUnlock();
    token_prefix_access += 1;
    var slot: usize = 0;
    var found = false;
    for (&token_prefix_cache, 0..) |*e, i| {
        if (e.*) |ent| {
            if (std.mem.eql(u8, ent.prompt, prompt_text)) {
                slot = i;
                found = true;
                break;
            }
        }
    }
    if (!found) {
        var oldest: u64 = std.math.maxInt(u64);
        for (&token_prefix_cache, 0..) |*e, i| {
            if (e.* == null) {
                slot = i;
                oldest = 0;
                break;
            }
            if (e.*.?.last_access < oldest) {
                oldest = e.*.?.last_access;
                slot = i;
            }
        }
    }
    if (token_prefix_cache[slot]) |*ent| {
        allocator.free(ent.prompt);
        allocator.free(ent.tokens);
    }
    token_prefix_cache[slot] = .{
        .prompt = try allocator.dupe(u8, prompt_text),
        .tokens = try allocator.dupe(u32, tokens),
        .last_access = token_prefix_access,
    };
    return tokens;
}

// ------------------------------------------------------------------
// Tool-calling prompt injection (shared by streaming + non-streaming)
// ------------------------------------------------------------------

/// Build the tool-calling system prompt for a request, or null when the
/// request carries no usable tool definitions. Caller owns the string.
pub fn buildToolSystem(allocator: std.mem.Allocator, req: ChatCompletionRequest) !?[]u8 {
    const tools = req.tools orelse return null;
    const family = tool_calling_mod.detectFamily(req.model);
    var tool_defs = std.ArrayList(tool_calling_mod.ToolDefinition).empty;
    defer {
        for (tool_defs.items) |*td| {
            allocator.free(td.name);
            if (td.description) |d| allocator.free(d);
            if (td.parameters) |p| allocator.free(p);
        }
        tool_defs.deinit(allocator);
    }
    for (tools) |tool| {
        const fn_info = tool.function orelse continue;
        const params_json = if (fn_info.parameters) |params| blk: {
            const out = try std.json.Stringify.valueAlloc(allocator, params, .{});
            break :blk out;
        } else null;
        try tool_defs.append(allocator, .{
            .name = try allocator.dupe(u8, fn_info.name),
            .description = if (fn_info.description) |d| try allocator.dupe(u8, d) else null,
            .parameters = params_json,
        });
    }
    if (tool_defs.items.len == 0) return null;
    return try tool_calling_mod.buildToolSystemPrompt(allocator, tool_defs.items, family);
}

pub fn generateChatCompletion(allocator: std.mem.Allocator, io: std.Io, state: *ServerState, request_json: []const u8, server_config: ServerConfig) ![]u8 {
    const t_entry = std.c.mach_absolute_time();
    // 1. Parse request
    const parsed = try std.json.parseFromSlice(ChatCompletionRequest, allocator, request_json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var req = parsed.value;

    // 2. Streaming is handled by handleStreamingCompletion; reject here as safety net
    if (req.stream orelse false) {
        return try allocator.dupe(u8, "{\"error\":\"streaming_not_supported\"}");
    }

    // 2b. Sampling params and tool_choice support are validated at HTTP
    // dispatch (http.zig, before the streaming/non-streaming split), so by
    // the time this handler runs the request is guaranteed valid — a
    // failure here would mean the dispatch check was bypassed. assert in
    // Debug; resolveToolChoice is still applied unconditionally for its
    // side effect of stripping tools when tool_choice is "none".
    std.debug.assert((validateChatRequest(allocator, req) catch null) == null);
    resolveToolChoice(&req) catch unreachable;

    // 3-5. Build the prompt: extract text/images from the messages, run any
    // images through the vision tower inline, apply the chat template and
    // tokenize. Tool definitions inject a leading system prompt.
    var tool_system: ?[]u8 = null;
    defer if (tool_system) |ts| allocator.free(ts);
    tool_system = try buildToolSystem(allocator, req);

    var prepared = preparePrompt(allocator, state, req, tool_system) catch |err| {
        const msg: []const u8 = switch (err) {
            error.VisionNotSupported => "vision input is not supported by the loaded model",
            error.ImageTooLarge => "decoded image exceeds the 20MB limit",
            error.BadImageUrl => "unsupported image_url (data:...;base64 URLs and absolute file paths only)",
            error.InvalidContent => "invalid message content (string or text/image_url parts only)",
            else => return err,
        };
        return try std.fmt.allocPrint(allocator,
            \\{{"error":{{"message":"{s}","type":"invalid_request_error","code":"{s}"}}}}
        , .{ msg, @errorName(err) });
    };
    defer prepared.deinit(allocator);
    const prompt_tokens = prepared.tokens;
    @import("http.zig").maybeDumpPromptBody(io, request_json, prompt_tokens.len);

    // 6. Submit request to engine queue and wait for completion.
    const max_tokens = if (req.max_tokens orelse req.max_completion_tokens) |mt| @min(mt, @as(u32, @intCast(server_config.max_tokens))) else server_config.max_tokens;
    const temperature = req.temperature orelse server_config.temperature;
    const seed = req.seed orelse @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toMilliseconds()));

    const request_config = engine.RequestConfig{
        .prompt_tokens = prompt_tokens,
        .max_tokens = max_tokens,
        .temperature = temperature,
        .top_k = if (req.top_k) |k| @as(usize, k) else server_config.top_k,
        .top_p = req.top_p orelse server_config.top_p,
        .seed = seed,
        .stop_strings = req.stop,
        .stop_tokens = state.stop_tokens,
        .streaming = false,
        .model_name = req.model,
        .guided_json_schema = if (req.response_format) |rf| blk: {
            if (std.mem.eql(u8, rf.type, "json_schema")) break :blk rf.schema;
            break :blk null;
        } else null,
        .guided_regex = if (req.response_format) |rf| blk: {
            if (std.mem.eql(u8, rf.type, "regex")) break :blk rf.schema;
            break :blk null;
        } else null,
        .vision = prepared.vision,
    };

    const request_id = state.next_request_id.fetchAdd(1, .monotonic);
    // On init failure `prepared`'s defer frees the vision payload.
    const req_state = try engine.RequestState.init(allocator, request_id, request_config);
    // Pushed to the engine queue unconditionally below (no fallible call in
    // between): wait for the engine's post-generation bookkeeping before
    // freeing, not just for the final token event.
    defer {
        req_state.waitEngineDone(io);
        req_state.deinit();
    }
    // Vision ownership has moved into the RequestState.
    prepared.vision = null;

    _ = state.active_requests.fetchAdd(1, .monotonic);
    defer _ = state.active_requests.fetchSub(1, .monotonic);

    const t_queued = std.c.mach_absolute_time();
    req_state.queued_time_ns = @intCast(t_queued);
    const prep_ms = @as(f64, @floatFromInt(t_queued - t_entry)) * 125.0 / 3_000_000.0;
    var node = engine.QueueNode.init(req_state);
    state.request_queue.push(&node);
    std.log.info("[HTTP] Request {d} submitted to queue (prepare={d:.1}ms)", .{ request_id, prep_ms });

    // Wait for the engine to deliver the complete response.
    var all_text = std.ArrayList(u8).empty;
    defer all_text.deinit(allocator);

    var finish_reason: []const u8 = "stop";
    const t_wait_start = std.c.mach_absolute_time();
    var wait_call_count: usize = 0;
    while (true) {
        wait_call_count += 1;
        std.log.info("[OpenAI] Waiting for token...", .{});
        const event = req_state.completion.waitForToken(io) catch break;
        std.log.info("[OpenAI] waitForToken returned", .{});
        if (event) |e| {
            defer if (e.token_text.len > 0) allocator.free(e.token_text);
            if (e.token_text.len > 0) {
                try all_text.appendSlice(allocator, e.token_text);
            }
            std.log.info("[OpenAI] Got token, is_final={}", .{e.is_final});
            if (e.is_final) {
                if (e.finish_reason) |fr| {
                    finish_reason = switch (fr) {
                        .stop => "stop",
                        .length => "length",
                        .error_ => "stop", // error detail is surfaced via hasError below
                    };
                }
                break;
            }
        } else {
            std.log.info("[OpenAI] waitForToken returned null", .{});
            break;
        }
    }

    const t_wait_end = std.c.mach_absolute_time();
    const wait_ms = @as(f64, @floatFromInt(t_wait_end - t_wait_start)) * 125.0 / 3_000_000.0;
    std.log.info("[OpenAI] Token loop done, all_text.len={d}, wait={d:.1}ms calls={d}", .{ all_text.items.len, wait_ms, wait_call_count });

    // Engine may still be finishing after the final event; wait so the
    // token_count/error reads below and the deferred deinit are race-free.
    req_state.waitEngineDone(io);

    // Count consumed tokens on both success and generation-failure paths.
    metrics_mod.global.addTokenCounts(@intCast(prompt_tokens.len), @intCast(req_state.token_count));

    // Check if the engine signaled an error.
    if (req_state.completion.hasError(io)) {
        const err_msg = req_state.completion.getError(io) orelse "Generation failed";
        return try std.fmt.allocPrint(allocator,
            \\{{"error":{{"message":"{s}","type":"server_error","code":"generation_failed"}}}}
        , .{err_msg});
    }

    const final_text = all_text.items;
    const new_tokens_len = req_state.token_count;

    // Thinking split (qwen ChatML with thinking enabled): text before the
    // first "</think>" is reasoning — return it as message.reasoning_content
    // (DeepSeek/OpenRouter convention) so clients render collapsible
    // thinking instead of mixing it into the answer. Tool detection runs on
    // the post-think content only.
    var content_text = final_text;
    var reasoning_json: ?[]u8 = null;
    defer if (reasoning_json) |r| allocator.free(r);
    if (!state.chat_template.empty_think and tool_calling_mod.detectFamily(req.model) == .qwen) {
        if (std.mem.indexOf(u8, final_text, "</think>")) |p| {
            const reasoning = final_text[0..p];
            var rest = final_text[p + "</think>".len ..];
            if (std.mem.startsWith(u8, rest, "\n\n")) rest = rest[2..];
            content_text = rest;
            if (reasoning.len > 0) {
                var esc_r = std.ArrayList(u8).empty;
                defer esc_r.deinit(allocator);
                try jsonEscapeInto(&esc_r, allocator, reasoning);
                reasoning_json = try std.fmt.allocPrint(allocator, ",\"reasoning_content\":\"{s}\"", .{esc_r.items});
            }
        }
    }

    // 8. Tool calling loop (single round)
    if (req.tools) |tools| {
        _ = tools;
        const family = tool_calling_mod.detectFamily(req.model);
        if (try tool_calling_mod.parse(allocator, content_text, family)) |parse_result| {
            defer parse_result.deinit();

            if (parse_result.calls.len > 0) {
                // Execute tools and build tool results
                var tool_results = std.ArrayList(u8).empty;
                defer tool_results.deinit(allocator);

                const exec_config = tool_executor_mod.ExecutorConfig{
                    .allow_shell_exec = server_config.allow_unsafe_tools,
                    .max_file_size = 1024 * 1024,
                    .io = state.io,
                };

                for (parse_result.calls) |call| {
                    const result = try tool_executor_mod.execute(allocator, exec_config, call.function.name, call.function.arguments);
                    defer result.deinit();

                    const result_text = if (result.success) result.output else result.error_message orelse "error";
                    try tool_results.appendSlice(allocator, "Tool '");
                    try tool_results.appendSlice(allocator, call.function.name);
                    try tool_results.appendSlice(allocator, "' result: ");
                    try tool_results.appendSlice(allocator, result_text);
                    try tool_results.appendSlice(allocator, "\n");
                }

                // Build OpenAI-compatible response with tool_calls
                const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));
                const tool_calls_json = try formatToolCallsJson(allocator, parse_result.calls);
                defer allocator.free(tool_calls_json);

                return try std.fmt.allocPrint(allocator,
                    \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":null,"tool_calls":{s}}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
                , .{
                    created,
                    created,
                    req.model,
                    tool_calls_json,
                    prompt_tokens.len,
                    new_tokens_len,
                    prompt_tokens.len + new_tokens_len,
                });
            }
        }
    }

    // 9. Build standard OpenAI-compatible response
    const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));

    // JSON-escape the output text
    var escaped = std.ArrayList(u8).empty;
    defer escaped.deinit(allocator);
    try jsonEscapeInto(&escaped, allocator, content_text);

    return try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"{s}}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{
        created,
        created,
        req.model,
        escaped.items,
        reasoning_json orelse "",
        finish_reason,
        prompt_tokens.len,
        new_tokens_len,
        prompt_tokens.len + new_tokens_len,
    });
}

// ------------------------------------------------------------------
// SSE streaming completion handler (using Generation API streamGenerate)
// ------------------------------------------------------------------

/// Serialize parsed tool calls into the OpenAI `tool_calls` array shape.
/// Per OpenAI convention, `function.arguments` is a JSON *string* containing
/// the serialized argument object, so the raw argument JSON (and the id/name,
/// which come from model output) must be escaped before embedding — otherwise
/// arguments containing quotes/braces produce invalid response JSON.
pub fn formatToolCallsJson(allocator: std.mem.Allocator, calls: []const tool_calling_mod.ToolCall) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "[");
    for (calls, 0..) |call, i| {
        if (i > 0) try out.appendSlice(allocator, ",");
        var escaped_id = std.ArrayList(u8).empty;
        defer escaped_id.deinit(allocator);
        try jsonEscapeInto(&escaped_id, allocator, call.id);
        var escaped_name = std.ArrayList(u8).empty;
        defer escaped_name.deinit(allocator);
        try jsonEscapeInto(&escaped_name, allocator, call.function.name);
        var escaped_args = std.ArrayList(u8).empty;
        defer escaped_args.deinit(allocator);
        try jsonEscapeInto(&escaped_args, allocator, call.function.arguments);
        try out.print(allocator,
            \\{{"id":"{s}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}
        , .{ escaped_id.items, escaped_name.items, escaped_args.items });
    }
    try out.appendSlice(allocator, "]");
    return out.toOwnedSlice(allocator);
}

/// Delta-shaped `tool_calls` for streaming chunks: same serialization as
/// formatToolCallsJson plus the per-call "index" field OpenAI clients expect
/// inside streaming deltas.
pub fn formatToolCallsDeltaJson(allocator: std.mem.Allocator, calls: []const tool_calling_mod.ToolCall) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "[");
    for (calls, 0..) |call, i| {
        if (i > 0) try out.appendSlice(allocator, ",");
        var escaped_id = std.ArrayList(u8).empty;
        defer escaped_id.deinit(allocator);
        try jsonEscapeInto(&escaped_id, allocator, call.id);
        var escaped_name = std.ArrayList(u8).empty;
        defer escaped_name.deinit(allocator);
        try jsonEscapeInto(&escaped_name, allocator, call.function.name);
        var escaped_args = std.ArrayList(u8).empty;
        defer escaped_args.deinit(allocator);
        try jsonEscapeInto(&escaped_args, allocator, call.function.arguments);
        try out.print(allocator,
            \\{{"index":{d},"id":"{s}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}
        , .{ i, escaped_id.items, escaped_name.items, escaped_args.items });
    }
    try out.appendSlice(allocator, "]");
    return out.toOwnedSlice(allocator);
}

/// Keep-alive threshold in milliseconds (5 seconds).
const keep_alive_threshold_ms: i64 = 5000;

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

fn testRequest() ChatCompletionRequest {
    // `messages` is a mutable slice in the struct; a zero-length const slice
    // needs a const_cast (no element is ever written through it).
    return .{ .model = "m", .messages = @constCast(&[_]ChatCompletionRequest.Message{}) };
}

test "validateChatRequest accepts defaults and boundary values" {
    const allocator = std.testing.allocator;

    var req = testRequest();
    try std.testing.expect(try validateChatRequest(allocator, req) == null);

    req.top_p = 1.0;
    req.top_k = 1;
    try std.testing.expect(try validateChatRequest(allocator, req) == null);
}

test "validateChatRequest rejects out-of-range sampling params" {
    const allocator = std.testing.allocator;

    for ([_]f32{ 0.0, -0.5, 1.5, std.math.nan(f32) }) |p| {
        var req = testRequest();
        req.top_p = p;
        const body = (try validateChatRequest(allocator, req)) orelse
            return error.TestUnexpectedResult;
        defer allocator.free(body);
        try std.testing.expect(std.mem.indexOf(u8, body, "InvalidTopP") != null);
    }

    var req = testRequest();
    req.top_k = 0;
    const body = (try validateChatRequest(allocator, req)) orelse
        return error.TestUnexpectedResult;
    defer allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "InvalidTopK") != null);
}

fn parseToolChoiceReq(allocator: std.mem.Allocator, tc_json: []const u8) !std.json.Parsed(ChatCompletionRequest) {
    const body = try std.fmt.allocPrint(allocator,
        \\{{"model":"m","messages":[],"tool_choice":{s}}}
    , .{tc_json});
    defer allocator.free(body);
    return std.json.parseFromSlice(ChatCompletionRequest, allocator, body, .{});
}

test "resolveToolChoice: auto keeps tools, none strips them" {
    const allocator = std.testing.allocator;
    const tools = [_]ChatCompletionRequest.ToolDefinition{.{ .function = .{ .name = "f" } }};

    {
        var parsed = try parseToolChoiceReq(allocator, "\"auto\"");
        defer parsed.deinit();
        var req = parsed.value;
        req.tools = &tools;
        try resolveToolChoice(&req);
        try std.testing.expect(req.tools != null);
    }
    {
        var parsed = try parseToolChoiceReq(allocator, "\"none\"");
        defer parsed.deinit();
        var req = parsed.value;
        req.tools = &tools;
        try resolveToolChoice(&req);
        try std.testing.expect(req.tools == null);
    }
    // Absent tool_choice behaves like auto.
    var req = testRequest();
    req.tools = &tools;
    try resolveToolChoice(&req);
    try std.testing.expect(req.tools != null);
}

test "resolveToolChoice: required and named-function are rejected" {
    const allocator = std.testing.allocator;

    {
        var parsed = try parseToolChoiceReq(allocator, "\"required\"");
        defer parsed.deinit();
        var req = parsed.value;
        try std.testing.expectError(error.UnsupportedToolChoice, resolveToolChoice(&req));
    }
    {
        var parsed = try parseToolChoiceReq(allocator, "{\"type\":\"function\",\"function\":{\"name\":\"f\"}}");
        defer parsed.deinit();
        var req = parsed.value;
        try std.testing.expectError(error.UnsupportedToolChoice, resolveToolChoice(&req));
    }
    {
        var parsed = try parseToolChoiceReq(allocator, "\"bogus\"");
        defer parsed.deinit();
        var req = parsed.value;
        try std.testing.expectError(error.InvalidToolChoice, resolveToolChoice(&req));
    }
}

test "formatToolCallsJson escapes arguments with braces, quotes and unicode" {
    const allocator = std.testing.allocator;
    const calls = [_]tool_calling_mod.ToolCall{
        .{
            .id = "call_1",
            .type = "function",
            .function = .{
                .name = "get_weather",
                .arguments = "{\"city\": \"Paris\", \"note\": \"say \\\"héllo\\\" ☃\"}",
            },
        },
        .{
            .id = "call_\"2\"",
            .type = "function",
            .function = .{
                .name = "f\\n",
                .arguments = "{\"nested\": {\"a\": [1, 2]}}",
            },
        },
    };
    const json = try formatToolCallsJson(allocator, &calls);
    defer allocator.free(json);

    // The whole array must be valid JSON...
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    const arr = parsed.value.array;
    try std.testing.expectEqual(@as(usize, 2), arr.items.len);

    // ...and `arguments` must be a STRING that round-trips to the raw args.
    const f0 = arr.items[0].object.get("function").?.object;
    try std.testing.expectEqualStrings("get_weather", f0.get("name").?.string);
    const args0 = f0.get("arguments").?.string;
    try std.testing.expectEqualStrings("{\"city\": \"Paris\", \"note\": \"say \\\"héllo\\\" ☃\"}", args0);
    // The arguments string itself must parse as JSON.
    const args_parsed = try std.json.parseFromSlice(std.json.Value, allocator, args0, .{});
    defer args_parsed.deinit();
    try std.testing.expectEqualStrings("Paris", args_parsed.value.object.get("city").?.string);

    const f1 = arr.items[1].object.get("function").?.object;
    try std.testing.expectEqualStrings("f\\n", f1.get("name").?.string);
    try std.testing.expectEqualStrings("{\"nested\": {\"a\": [1, 2]}}", f1.get("arguments").?.string);
    try std.testing.expectEqualStrings("call_\"2\"", arr.items[1].object.get("id").?.string);
}

test "formatToolCallsDeltaJson adds per-call index and escapes arguments" {
    const allocator = std.testing.allocator;
    const calls = [_]tool_calling_mod.ToolCall{
        .{
            .id = "call_1",
            .type = "function",
            .function = .{
                .name = "get_weather",
                .arguments = "{\"city\": \"Paris\"}",
            },
        },
        .{
            .id = "call_2",
            .type = "function",
            .function = .{
                .name = "get_time",
                .arguments = "{\"tz\": \"UTC\"}",
            },
        },
    };
    const json = try formatToolCallsDeltaJson(allocator, &calls);
    defer allocator.free(json);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    const arr = parsed.value.array;
    try std.testing.expectEqual(@as(usize, 2), arr.items.len);
    try std.testing.expectEqual(@as(i64, 0), arr.items[0].object.get("index").?.integer);
    try std.testing.expectEqual(@as(i64, 1), arr.items[1].object.get("index").?.integer);
    const f0 = arr.items[0].object.get("function").?.object;
    try std.testing.expectEqualStrings("get_weather", f0.get("name").?.string);
    try std.testing.expectEqualStrings("{\"city\": \"Paris\"}", f0.get("arguments").?.string);
}

test "validateChatRequest reports unsupported tool_choice forms" {
    const allocator = std.testing.allocator;

    var parsed = try parseToolChoiceReq(allocator, "\"required\"");
    defer parsed.deinit();
    const body = (try validateChatRequest(allocator, parsed.value)) orelse
        return error.TestUnexpectedResult;
    defer allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "UnsupportedToolChoice") != null);
    // The error body must be valid JSON.
    const err_parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer err_parsed.deinit();
    try std.testing.expectEqualStrings(
        "invalid_request_error",
        err_parsed.value.object.get("error").?.object.get("type").?.string,
    );
}
