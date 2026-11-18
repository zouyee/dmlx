/// Responses API request translation: `POST /v1/responses` request bodies
/// are rewritten into equivalent chat-completions request bodies so the
/// engine flow stays single-pathed.
///
/// Supported request surface (everything else is a clean 400):
///   model (ignored — single-model server), input (string or item array:
///   message / function_call / function_call_output; reasoning items are
///   tolerated and dropped), instructions (prepended system message),
///   max_output_tokens, temperature, top_p, stream, store, tools (function
///   type only), tool_choice ("auto"/"none"), metadata and the Codex
///   provider fields (reasoning, include, text, prompt_cache_key,
///   client_metadata, service_tier, stream_options, parallel_tool_calls)
///   which are accepted but have no effect.
const std = @import("std");
const openai_mod = @import("../openai.zig");
const utils_mod = @import("../utils.zig");

const jsonEscapeInto = utils_mod.jsonEscapeInto;

pub const TranslateOutcome = union(enum) {
    /// Translated request, ready to feed into the chat-completions flow.
    ok: struct {
        /// Chat-completions request body (JSON).
        body: []u8,
        /// Effective model id (owned) — request's `model` or the default.
        model: []u8,
        stream: bool,
        /// OpenAI default is store=true; store:false = don't retain the
        /// response object in the in-memory store (response_store.zig).
        store: bool,
        /// Request carried function tool definitions; the streaming path
        /// buffers output so tool-call markup never leaks into text deltas.
        has_tools: bool,
    },
    /// Complete JSON error body to return with HTTP 400.
    err: []u8,

    pub fn deinit(self: TranslateOutcome, allocator: std.mem.Allocator) void {
        switch (self) {
            .ok => |o| {
                allocator.free(o.body);
                allocator.free(o.model);
            },
            .err => |b| allocator.free(b),
        }
    }
};

/// Fields that are explicitly rejected with a tailored message when present.
const RejectedField = struct { name: []const u8, reason: []const u8 };
const rejected_fields = [_]RejectedField{
    .{ .name = "previous_response_id", .reason = "chained responses are not supported (this server is stateless); send the full conversation in 'input'" },
    .{ .name = "background", .reason = "background mode is not supported" },
    .{ .name = "truncation", .reason = "the 'truncation' parameter is not supported" },
    .{ .name = "max_tool_calls", .reason = "tool-call limits are not supported" },
    .{ .name = "prompt", .reason = "prompt templates are not supported" },
    .{ .name = "modalities", .reason = "only text output is supported" },
    .{ .name = "audio", .reason = "audio output is not supported" },
    .{ .name = "top_logprobs", .reason = "logprobs are not supported" },
};

/// Fields that are accepted but have no effect on this server. These are
/// fields Codex (and OpenAI SDKs) send unconditionally; the server is
/// stateless and has no hosted-model concepts (encrypted reasoning,
/// structured outputs, cache keys, tiers), so the honest answer is "ignore".
/// `store` is NOT in this list — it gates retention in the response store.
const ignored_fields = [_][]const u8{
    "metadata",
    "reasoning",
    "include",
    "text",
    "parallel_tool_calls",
    "prompt_cache_key",
    "prompt_cache_options",
    "client_metadata",
    "service_tier",
    "stream_options",
    "safety_identifier",
};

fn invalidRequest(allocator: std.mem.Allocator, message: []const u8) !TranslateOutcome {
    return .{ .err = try openai_mod.invalidRequestJson(allocator, "unsupported_parameter", message) };
}

fn invalidRequestFmt(allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !TranslateOutcome {
    const message = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(message);
    return invalidRequest(allocator, message);
}

/// Translate a Responses API request body into an equivalent
/// chat-completions request body. `default_model` is used when the request
/// omits `model`. All validation failures come back as `.err` (400 body);
/// only OOM/parse failures propagate as errors.
pub fn translateRequest(
    allocator: std.mem.Allocator,
    body: []const u8,
    default_model: []const u8,
) !TranslateOutcome {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        return invalidRequest(allocator, "request body is not valid JSON");
    };
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return invalidRequest(allocator, "request body must be a JSON object"),
    };

    // Field allowlist / rejection pass — anything the translation cannot
    // honor fails loudly here rather than being silently dropped.
    var it = obj.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        if (std.mem.eql(u8, name, "model") or
            std.mem.eql(u8, name, "input") or
            std.mem.eql(u8, name, "instructions") or
            std.mem.eql(u8, name, "max_output_tokens") or
            std.mem.eql(u8, name, "temperature") or
            std.mem.eql(u8, name, "top_p") or
            std.mem.eql(u8, name, "store") or
            std.mem.eql(u8, name, "tools") or
            std.mem.eql(u8, name, "tool_choice") or
            std.mem.eql(u8, name, "stream")) continue;
        var is_ignored = false;
        for (ignored_fields) |f| {
            if (std.mem.eql(u8, name, f)) {
                is_ignored = true;
                break;
            }
        }
        if (!is_ignored) {
            for (rejected_fields) |rf| {
                if (std.mem.eql(u8, name, rf.name)) {
                    return invalidRequestFmt(allocator, "unsupported field '{s}': {s}", .{ name, rf.reason });
                }
            }
            return invalidRequestFmt(allocator, "unsupported field '{s}': this server implements a minimal Responses API subset", .{name});
        }
    }

    var model: []const u8 = default_model;
    if (obj.get("model")) |m| {
        switch (m) {
            .string => |s| model = s,
            .null => {},
            else => return invalidRequest(allocator, "'model' must be a string"),
        }
    }

    const input = obj.get("input") orelse
        return invalidRequest(allocator, "missing required field 'input' (string or array of message items)");

    var instructions: ?[]const u8 = null;
    if (obj.get("instructions")) |v| {
        switch (v) {
            .string => |s| instructions = s,
            .null => {},
            else => return invalidRequest(allocator, "'instructions' must be a string"),
        }
    }

    var max_output_tokens: ?u32 = null;
    if (obj.get("max_output_tokens")) |v| {
        switch (v) {
            .integer => |n| {
                if (n <= 0) return invalidRequest(allocator, "'max_output_tokens' must be a positive integer");
                if (n > std.math.maxInt(u32))
                    return invalidRequestFmt(allocator, "'max_output_tokens' must be at most {d}", .{std.math.maxInt(u32)});
                max_output_tokens = @intCast(n);
            },
            .null => {},
            else => return invalidRequest(allocator, "'max_output_tokens' must be a positive integer"),
        }
    }

    var temperature: ?f64 = null;
    if (obj.get("temperature")) |v| {
        switch (v) {
            .float => |f| temperature = f,
            .integer => |n| temperature = @floatFromInt(n),
            .null => {},
            else => return invalidRequest(allocator, "'temperature' must be a number"),
        }
    }

    var top_p: ?f64 = null;
    if (obj.get("top_p")) |v| {
        switch (v) {
            .float => |f| top_p = f,
            .integer => |n| top_p = @floatFromInt(n),
            .null => {},
            else => return invalidRequest(allocator, "'top_p' must be a number"),
        }
        if (top_p) |pp| {
            // Matches openai_mod.validateSampling so the translated request
            // passes the chat path's validation assert.
            if (!(pp > 0.0 and pp <= 1.0))
                return invalidRequest(allocator, "top_p must be in the range (0, 1]");
        }
    }

    var stream = false;
    if (obj.get("stream")) |v| {
        switch (v) {
            .bool => |b| stream = b,
            .null => {},
            else => return invalidRequest(allocator, "'stream' must be a boolean"),
        }
    }

    var store = true;
    if (obj.get("store")) |v| {
        switch (v) {
            .bool => |b| store = b,
            .null => {},
            else => return invalidRequest(allocator, "'store' must be a boolean"),
        }
    }

    // Tools: Responses function tools ({type,name,description,parameters})
    // map onto chat tools ({type,function:{name,description,parameters}}).
    var tools_json: ?[]u8 = null;
    defer if (tools_json) |t| allocator.free(t);
    var tool_choice_json: ?[]u8 = null;
    defer if (tool_choice_json) |t| allocator.free(t);
    if (obj.get("tools")) |v| {
        switch (v) {
            .null => {},
            .array => |tools| {
                var buf = std.ArrayList(u8).empty;
                errdefer buf.deinit(allocator);
                try buf.append(allocator, '[');
                var count: usize = 0;
                for (tools.items) |tool| {
                    if (try translateToolOrNamespace(allocator, &buf, tool, &count)) |outcome| return outcome;
                }
                try buf.append(allocator, ']');
                tools_json = try buf.toOwnedSlice(allocator);
            },
            else => return invalidRequest(allocator, "'tools' must be an array"),
        }
    }
    if (obj.get("tool_choice")) |v| {
        switch (v) {
            .null => {},
            .string => |s| {
                if (!std.mem.eql(u8, s, "auto") and !std.mem.eql(u8, s, "none"))
                    return invalidRequestFmt(allocator, "unsupported tool_choice '{s}' (supported: \"auto\", \"none\")", .{s});
                tool_choice_json = try std.fmt.allocPrint(allocator, "\"{s}\"", .{s});
            },
            else => return invalidRequest(allocator, "unsupported tool_choice shape (supported: \"auto\", \"none\")"),
        }
    }

    // Build the messages array JSON.
    var messages = std.ArrayList(u8).empty;
    defer messages.deinit(allocator);
    try messages.append(allocator, '[');
    var first = true;
    if (instructions) |instr| {
        try messages.appendSlice(allocator, "{\"role\":\"system\",\"content\":\"");
        try jsonEscapeInto(&messages, allocator, instr);
        try messages.appendSlice(allocator, "\"}");
        first = false;
    }
    switch (input) {
        .string => |s| {
            if (!first) try messages.append(allocator, ',');
            try messages.appendSlice(allocator, "{\"role\":\"user\",\"content\":\"");
            try jsonEscapeInto(&messages, allocator, s);
            try messages.appendSlice(allocator, "\"}");
        },
        .array => |items| {
            for (items.items) |item| {
                const r = try translateInputItem(allocator, &messages, item, &first);
                if (r) |outcome| return outcome;
            }
        },
        else => return invalidRequest(allocator, "'input' must be a string or an array of message items"),
    }
    try messages.append(allocator, ']');

    // Assemble the chat-completions body.
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":\"");
    try jsonEscapeInto(&out, allocator, model);
    try out.appendSlice(allocator, "\",\"messages\":");
    try out.appendSlice(allocator, messages.items);
    if (max_output_tokens) |mt| try out.print(allocator, ",\"max_tokens\":{d}", .{mt});
    if (temperature) |t| try out.print(allocator, ",\"temperature\":{d}", .{t});
    if (top_p) |p| try out.print(allocator, ",\"top_p\":{d}", .{p});
    if (stream) try out.appendSlice(allocator, ",\"stream\":true");
    if (tools_json) |t| {
        try out.appendSlice(allocator, ",\"tools\":");
        try out.appendSlice(allocator, t);
    }
    if (tool_choice_json) |tc| {
        try out.appendSlice(allocator, ",\"tool_choice\":");
        try out.appendSlice(allocator, tc);
    }
    try out.append(allocator, '}');
    return .{ .ok = .{
        .body = try out.toOwnedSlice(allocator),
        .model = try allocator.dupe(u8, model),
        .stream = stream,
        .store = store,
        .has_tools = tools_json != null,
    } };
}

/// Append one Responses tool entry to the chat `tools` JSON array.
/// Namespace entries (Codex groups tools under
/// {type:"namespace",name,tools:[...]}) are flattened: each nested function
/// tool becomes a top-level chat tool addressed by its leaf name; nested
/// freeform "custom" tools are skipped (the chat side cannot carry their
/// grammar constraints). Hosted/server-side tool types (web_search,
/// tool_search, ...) are skipped as well: they need OpenAI's backend, and
/// tools the model never sees are tools it never calls.
fn translateToolOrNamespace(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    tool: std.json.Value,
    count: *usize,
) !?TranslateOutcome {
    const obj = switch (tool) {
        .object => |o| o,
        else => return try invalidRequest(allocator, "each 'tools' entry must be an object"),
    };
    const ty = switch (obj.get("type") orelse
        return try invalidRequest(allocator, "tool entry is missing 'type'")) {
        .string => |s| s,
        else => return try invalidRequest(allocator, "tool entry 'type' must be a string"),
    };
    if (std.mem.eql(u8, ty, "namespace")) {
        const nested = switch (obj.get("tools") orelse .null) {
            .array => |a| a,
            .null => return null, // empty namespace: nothing to advertise
            else => return try invalidRequest(allocator, "namespace tool 'tools' must be an array"),
        };
        for (nested.items) |n| {
            if (n != .object) continue;
            const nt = switch (n.object.get("type") orelse .null) {
                .string => |s| s,
                else => continue,
            };
            if (std.mem.eql(u8, nt, "custom")) continue; // freeform tools unsupported
            if (try translateTool(allocator, buf, n, count.* > 0)) |outcome| return outcome;
            count.* += 1;
        }
        return null;
    }
    if (std.mem.eql(u8, ty, "function")) {
        if (try translateTool(allocator, buf, tool, count.* > 0)) |outcome| return outcome;
        count.* += 1;
        return null;
    }
    // Hosted/server-side tools (web_search, tool_search, image_generation,
    // code_interpreter, ...) need OpenAI's backend; this server is local.
    // Skipping them keeps the request usable — the model simply never
    // learns they exist, so it can never emit a call to them.
    return null;
}

/// Append one function tool definition to the chat `tools` JSON array.
fn translateTool(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    tool: std.json.Value,
    comma: bool,
) !?TranslateOutcome {
    const obj = switch (tool) {
        .object => |o| o,
        else => return try invalidRequest(allocator, "each 'tools' entry must be an object"),
    };
    const ty = switch (obj.get("type") orelse
        return try invalidRequest(allocator, "tool entry is missing 'type'")) {
        .string => |s| s,
        else => return try invalidRequest(allocator, "tool entry 'type' must be a string"),
    };
    if (!std.mem.eql(u8, ty, "function"))
        return try invalidRequestFmt(allocator, "unsupported tool type '{s}' (only function tools are supported)", .{ty});

    // Responses shape carries name/description/parameters at the top level
    // (or nested under "function" for SDK compat); chat expects them nested.
    const fn_obj = if (obj.get("function")) |f|
        switch (f) {
            .object => |o| o,
            else => return try invalidRequest(allocator, "tool 'function' must be an object"),
        }
    else
        obj;

    const name = switch (fn_obj.get("name") orelse
        return try invalidRequest(allocator, "function tool is missing 'name'")) {
        .string => |s| s,
        else => return try invalidRequest(allocator, "function tool 'name' must be a string"),
    };

    if (comma) try buf.append(allocator, ',');
    try buf.appendSlice(allocator, "{\"type\":\"function\",\"function\":{\"name\":\"");
    try jsonEscapeInto(buf, allocator, name);
    try buf.append(allocator, '"');
    if (fn_obj.get("description")) |d| {
        if (d == .string) {
            try buf.appendSlice(allocator, ",\"description\":\"");
            try jsonEscapeInto(buf, allocator, d.string);
            try buf.append(allocator, '"');
        }
    }
    if (fn_obj.get("parameters")) |p| {
        if (p != .null) {
            try buf.appendSlice(allocator, ",\"parameters\":");
            const raw = try std.json.Stringify.valueAlloc(allocator, p, .{});
            defer allocator.free(raw);
            try buf.appendSlice(allocator, raw);
        }
    }
    try buf.appendSlice(allocator, "}}");
    return null;
}

/// Append one Responses input item to the chat `messages` JSON array.
/// Returns an error outcome to propagate, or null on success.
fn translateInputItem(
    allocator: std.mem.Allocator,
    messages: *std.ArrayList(u8),
    item: std.json.Value,
    first: *bool,
) !?TranslateOutcome {
    const obj = switch (item) {
        .object => |o| o,
        else => return try invalidRequest(allocator, "each 'input' item must be an object"),
    };
    if (obj.get("type")) |t| {
        const ts = switch (t) {
            .string => |s| s,
            else => return try invalidRequest(allocator, "input item 'type' must be a string"),
        };
        if (std.mem.eql(u8, ts, "message")) {
            return translateMessageItem(allocator, messages, obj, first);
        }
        if (std.mem.eql(u8, ts, "function_call")) {
            return translateFunctionCallItem(allocator, messages, obj, first);
        }
        if (std.mem.eql(u8, ts, "function_call_output")) {
            return translateFunctionCallOutputItem(allocator, messages, obj, first);
        }
        // Reasoning items echo server-side thinking the local model never
        // produced (or produced as plain text); dropping them keeps the
        // conversation faithful without inventing content.
        if (std.mem.eql(u8, ts, "reasoning")) return null;
        return try invalidRequestFmt(allocator, "unsupported input item type '{s}' (supported: message, function_call, function_call_output)", .{ts});
    }
    return translateMessageItem(allocator, messages, obj, first);
}

/// Append a message item (role + content) to the chat messages array.
fn translateMessageItem(
    allocator: std.mem.Allocator,
    messages: *std.ArrayList(u8),
    obj: std.json.ObjectMap,
    first: *bool,
) !?TranslateOutcome {
    const role = switch (obj.get("role") orelse
        return try invalidRequest(allocator, "input message item is missing 'role'")) {
        .string => |s| s,
        else => return try invalidRequest(allocator, "input message 'role' must be a string"),
    };
    if (!std.mem.eql(u8, role, "user") and !std.mem.eql(u8, role, "assistant") and
        !std.mem.eql(u8, role, "system") and !std.mem.eql(u8, role, "developer"))
    {
        return try invalidRequestFmt(allocator, "unsupported role '{s}' in input message", .{role});
    }
    const content = obj.get("content") orelse
        return try invalidRequest(allocator, "input message item is missing 'content'");

    if (!first.*) try messages.append(allocator, ',');
    first.* = false;
    try messages.appendSlice(allocator, "{\"role\":\"");
    try jsonEscapeInto(messages, allocator, role);
    try messages.appendSlice(allocator, "\",\"content\":");
    switch (content) {
        .string => |s| {
            try messages.append(allocator, '"');
            try jsonEscapeInto(messages, allocator, s);
            try messages.append(allocator, '"');
        },
        .array => |parts| {
            try messages.append(allocator, '[');
            for (parts.items, 0..) |part, i| {
                if (i > 0) try messages.append(allocator, ',');
                if (try translateContentPart(allocator, messages, part)) |outcome| return outcome;
            }
            try messages.append(allocator, ']');
        },
        else => return try invalidRequest(allocator, "input message 'content' must be a string or an array of content parts"),
    }
    try messages.append(allocator, '}');
    return null;
}

/// Append a function_call item as an assistant message carrying chat-style
/// tool_calls. `arguments` arrives as a JSON string and is passed through
/// verbatim (the chat path re-renders it into the model's native call
/// format when building the prompt).
fn translateFunctionCallItem(
    allocator: std.mem.Allocator,
    messages: *std.ArrayList(u8),
    obj: std.json.ObjectMap,
    first: *bool,
) !?TranslateOutcome {
    const name = switch (obj.get("name") orelse
        return try invalidRequest(allocator, "function_call item is missing 'name'")) {
        .string => |s| s,
        else => return try invalidRequest(allocator, "function_call 'name' must be a string"),
    };
    const call_id = switch (obj.get("call_id") orelse
        return try invalidRequest(allocator, "function_call item is missing 'call_id'")) {
        .string => |s| s,
        else => return try invalidRequest(allocator, "function_call 'call_id' must be a string"),
    };
    const arguments: []const u8 = switch (obj.get("arguments") orelse .null) {
        .string => |s| s,
        .null => "{}",
        else => return try invalidRequest(allocator, "function_call 'arguments' must be a string"),
    };

    if (!first.*) try messages.append(allocator, ',');
    first.* = false;
    try messages.appendSlice(allocator,
        \\{"role":"assistant","content":null,"tool_calls":[{"id":"
    );
    try jsonEscapeInto(messages, allocator, call_id);
    try messages.appendSlice(allocator, "\",\"type\":\"function\",\"function\":{\"name\":\"");
    try jsonEscapeInto(messages, allocator, name);
    try messages.appendSlice(allocator, "\",\"arguments\":\"");
    try jsonEscapeInto(messages, allocator, arguments);
    try messages.appendSlice(allocator, "\"}}]}");
    return null;
}

/// Append a function_call_output item as a chat tool message. `output` may
/// be a plain string or an array of text content parts.
fn translateFunctionCallOutputItem(
    allocator: std.mem.Allocator,
    messages: *std.ArrayList(u8),
    obj: std.json.ObjectMap,
    first: *bool,
) !?TranslateOutcome {
    const call_id = switch (obj.get("call_id") orelse
        return try invalidRequest(allocator, "function_call_output item is missing 'call_id'")) {
        .string => |s| s,
        else => return try invalidRequest(allocator, "function_call_output 'call_id' must be a string"),
    };
    const output = obj.get("output") orelse
        return try invalidRequest(allocator, "function_call_output item is missing 'output'");

    if (!first.*) try messages.append(allocator, ',');
    first.* = false;
    try messages.appendSlice(allocator, "{\"role\":\"tool\",\"tool_call_id\":\"");
    try jsonEscapeInto(messages, allocator, call_id);
    try messages.appendSlice(allocator, "\",\"content\":\"");
    switch (output) {
        .string => |s| try jsonEscapeInto(messages, allocator, s),
        .array => |parts| {
            // Concatenate text parts; Codex sends single output_text parts.
            for (parts.items) |part| {
                if (part != .object) continue;
                const p = part.object;
                const pt = switch (p.get("type") orelse continue) {
                    .string => |s| s,
                    else => continue,
                };
                if (!std.mem.eql(u8, pt, "input_text") and !std.mem.eql(u8, pt, "output_text")) continue;
                switch (p.get("text") orelse continue) {
                    .string => |s| try jsonEscapeInto(messages, allocator, s),
                    else => continue,
                }
            }
        },
        else => return try invalidRequest(allocator, "function_call_output 'output' must be a string or an array of text parts"),
    }
    try messages.appendSlice(allocator, "\"}");
    return null;
}

/// Append one content part (input_text / output_text / input_image) as a
/// chat-completions content part (text / image_url).
fn translateContentPart(
    allocator: std.mem.Allocator,
    messages: *std.ArrayList(u8),
    part: std.json.Value,
) !?TranslateOutcome {
    const obj = switch (part) {
        .object => |o| o,
        else => return try invalidRequest(allocator, "content parts must be objects"),
    };
    const ty = switch (obj.get("type") orelse
        return try invalidRequest(allocator, "content part is missing 'type'")) {
        .string => |s| s,
        else => return try invalidRequest(allocator, "content part 'type' must be a string"),
    };
    if (std.mem.eql(u8, ty, "input_text") or std.mem.eql(u8, ty, "output_text")) {
        const text = switch (obj.get("text") orelse
            return try invalidRequestFmt(allocator, "'{s}' part is missing 'text'", .{ty})) {
            .string => |s| s,
            else => return try invalidRequestFmt(allocator, "'{s}' part 'text' must be a string", .{ty}),
        };
        try messages.appendSlice(allocator, "{\"type\":\"text\",\"text\":\"");
        try jsonEscapeInto(messages, allocator, text);
        try messages.appendSlice(allocator, "\"}");
        return null;
    }
    if (std.mem.eql(u8, ty, "input_image")) {
        const url = switch (obj.get("image_url") orelse
            return try invalidRequest(allocator, "'input_image' part is missing 'image_url'")) {
            .string => |s| s,
            else => return try invalidRequest(allocator, "'input_image' part 'image_url' must be a string (data:...;base64 URL or absolute file path)"),
        };
        try messages.appendSlice(allocator, "{\"type\":\"image_url\",\"image_url\":{\"url\":\"");
        try jsonEscapeInto(messages, allocator, url);
        try messages.appendSlice(allocator, "\"}}");
        return null;
    }
    return try invalidRequestFmt(allocator, "unsupported content part type '{s}' (supported: input_text, input_image)", .{ty});
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

fn expectTranslateErr(allocator: std.mem.Allocator, body: []const u8, needle: []const u8) !void {
    const outcome = try translateRequest(allocator, body, "m");
    defer outcome.deinit(allocator);
    const err_body = switch (outcome) {
        .err => |b| b,
        .ok => return error.TestUnexpectedResult,
    };
    try std.testing.expect(std.mem.indexOf(u8, err_body, needle) != null);
    // Error bodies must be valid JSON with the invalid_request_error shape.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, err_body, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(
        "invalid_request_error",
        parsed.value.object.get("error").?.object.get("type").?.string,
    );
}

fn expectTranslateOk(allocator: std.mem.Allocator, body: []const u8) !std.json.Parsed(std.json.Value) {
    const outcome = try translateRequest(allocator, body, "default-model");
    defer outcome.deinit(allocator);
    const chat_body = switch (outcome) {
        .ok => |o| o.body,
        .err => |b| {
            std.debug.print("unexpected translate error: {s}\n", .{b});
            return error.TestUnexpectedResult;
        },
    };
    return std.json.parseFromSlice(std.json.Value, allocator, chat_body, .{});
}

test "translateRequest: string input" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"model":"gpt-x","input":"Hello there","max_output_tokens":64,"temperature":0.5,"top_p":0.9}
    );
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("gpt-x", obj.get("model").?.string);
    const messages = obj.get("messages").?.array;
    try std.testing.expectEqual(@as(usize, 1), messages.items.len);
    const msg = messages.items[0].object;
    try std.testing.expectEqualStrings("user", msg.get("role").?.string);
    try std.testing.expectEqualStrings("Hello there", msg.get("content").?.string);
    try std.testing.expectEqual(@as(i64, 64), obj.get("max_tokens").?.integer);
    try std.testing.expectEqual(@as(f64, 0.5), obj.get("temperature").?.float);
    try std.testing.expectEqual(@as(f64, 0.9), obj.get("top_p").?.float);
    try std.testing.expect(obj.get("stream") == null);
}

test "translateRequest: default model, instructions prepended as system" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"input":"Hi","instructions":"You are terse.","store":false,"metadata":{"k":"v"}}
    );
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("default-model", obj.get("model").?.string);
    const messages = obj.get("messages").?.array;
    try std.testing.expectEqual(@as(usize, 2), messages.items.len);
    try std.testing.expectEqualStrings("system", messages.items[0].object.get("role").?.string);
    try std.testing.expectEqualStrings("You are terse.", messages.items[0].object.get("content").?.string);
    try std.testing.expectEqualStrings("user", messages.items[1].object.get("role").?.string);
}

test "translateRequest: array input with input_text parts" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"part one"},{"type":"input_text","text":"part two"}]}]}
    );
    defer parsed.deinit();
    const messages = parsed.value.object.get("messages").?.array;
    const parts = messages.items[0].object.get("content").?.array;
    try std.testing.expectEqual(@as(usize, 2), parts.items.len);
    try std.testing.expectEqualStrings("text", parts.items[0].object.get("type").?.string);
    try std.testing.expectEqualStrings("part two", parts.items[1].object.get("text").?.string);
}

test "translateRequest: input_image maps to image_url part" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,aGVsbG8="},{"type":"input_text","text":"what is this?"}]}],"stream":true}
    );
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expect(obj.get("stream").?.bool);
    const parts = obj.get("messages").?.array.items[0].object.get("content").?.array;
    const img = parts.items[0].object;
    try std.testing.expectEqualStrings("image_url", img.get("type").?.string);
    try std.testing.expectEqualStrings(
        "data:image/png;base64,aGVsbG8=",
        img.get("image_url").?.object.get("url").?.string,
    );
}

test "translateRequest parses the store flag (default true)" {
    const allocator = std.testing.allocator;
    {
        const outcome = try translateRequest(allocator, "{\"input\":\"hi\"}", "m");
        defer outcome.deinit(allocator);
        try std.testing.expect(switch (outcome) {
            .ok => |o| o.store,
            .err => return error.TestUnexpectedResult,
        });
    }
    {
        const outcome = try translateRequest(allocator, "{\"input\":\"hi\",\"store\":false}", "m");
        defer outcome.deinit(allocator);
        try std.testing.expect(!(switch (outcome) {
            .ok => |o| o.store,
            .err => return error.TestUnexpectedResult,
        }));
    }
    try expectTranslateErr(allocator, "{\"input\":\"hi\",\"store\":\"no\"}", "store");
}

test "translateRequest tolerates Codex provider fields" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"input":"hi","reasoning":{"effort":"high"},"include":["reasoning.encrypted_content"],"text":{"verbosity":"low"},"parallel_tool_calls":true,"prompt_cache_key":"abc","client_metadata":{"k":"v"},"service_tier":"default","stream_options":{"x":1}}
    );
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("reasoning") == null);
}

test "translateRequest accepts function tools and tool_choice" {
    const allocator = std.testing.allocator;
    const outcome = try translateRequest(allocator,
        \\{"input":"hi","tools":[{"type":"function","name":"shell","description":"run a command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}},"strict":false}],"tool_choice":"auto"}
    , "m");
    defer outcome.deinit();
    switch (outcome) {
        .err => |b| {
            std.debug.print("unexpected translate error: {s}\n", .{b});
            return error.TestUnexpectedResult;
        },
        .ok => |o| try std.testing.expect(o.has_tools),
    }
    var parsed = try expectTranslateOk(allocator,
        \\{"input":"hi","tools":[{"type":"function","name":"shell","description":"run a command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}},"strict":false}],"tool_choice":"auto"}
    );
    defer parsed.deinit();
    const obj = parsed.value.object;
    const tools = obj.get("tools").?.array;
    try std.testing.expectEqual(@as(usize, 1), tools.items.len);
    const f = tools.items[0].object.get("function").?.object;
    try std.testing.expectEqualStrings("shell", f.get("name").?.string);
    try std.testing.expectEqualStrings("run a command", f.get("description").?.string);
    try std.testing.expect(f.get("parameters").?.object.get("properties") != null);
    try std.testing.expectEqualStrings("auto", obj.get("tool_choice").?.string);
}

test "translateRequest flattens namespace tools" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"input":"hi","tools":[{"type":"namespace","name":"codex","description":"codex tools","tools":[{"type":"function","name":"shell","parameters":{"type":"object"}},{"type":"custom","name":"apply_patch"},{"type":"function","name":"read_file","description":"read"}]}]}
    );
    defer parsed.deinit();
    const tools = parsed.value.object.get("tools").?.array;
    // custom entry skipped; both function entries flattened.
    try std.testing.expectEqual(@as(usize, 2), tools.items.len);
    try std.testing.expectEqualStrings("shell", tools.items[0].object.get("function").?.object.get("name").?.string);
    try std.testing.expectEqualStrings("read_file", tools.items[1].object.get("function").?.object.get("name").?.string);
}

test "translateRequest skips hosted tools, rejects bad tool shapes" {
    const allocator = std.testing.allocator;
    // Hosted tools are skipped, leaving an empty tools array.
    var parsed = try expectTranslateOk(allocator,
        \\{"input":"hi","tools":[{"type":"web_search"},{"type":"image_generation"}]}
    );
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 0), parsed.value.object.get("tools").?.array.items.len);
    // Malformed function tools still 400.
    try expectTranslateErr(allocator,
        \\{"input":"hi","tools":[{"type":"function"}]}
    , "name");
    try expectTranslateErr(allocator,
        \\{"input":"hi","tool_choice":"required"}
    , "tool_choice");
    try expectTranslateErr(allocator,
        \\{"input":"hi","tool_choice":{"type":"function","name":"f"}}
    , "tool_choice");
}

test "translateRequest translates function_call and function_call_output items" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"input":[
        \\  {"type":"message","role":"user","content":[{"type":"input_text","text":"list files"}]},
        \\  {"type":"function_call","id":"fc_1","call_id":"call_1","name":"shell","arguments":"{\"cmd\":\"ls\"}"},
        \\  {"type":"function_call_output","call_id":"call_1","output":"a.txt b.txt"},
        \\  {"type":"reasoning","id":"rs_1","summary":[]}
        \\]}
    );
    defer parsed.deinit();
    const messages = parsed.value.object.get("messages").?.array;
    // reasoning item is dropped; 3 chat messages remain.
    try std.testing.expectEqual(@as(usize, 3), messages.items.len);

    const asst = messages.items[1].object;
    try std.testing.expectEqualStrings("assistant", asst.get("role").?.string);
    try std.testing.expect(std.meta.activeTag(asst.get("content").?) == .null);
    const tcs = asst.get("tool_calls").?.array;
    try std.testing.expectEqual(@as(usize, 1), tcs.items.len);
    const tc = tcs.items[0].object;
    try std.testing.expectEqualStrings("call_1", tc.get("id").?.string);
    const f = tc.get("function").?.object;
    try std.testing.expectEqualStrings("shell", f.get("name").?.string);
    try std.testing.expectEqualStrings("{\"cmd\":\"ls\"}", f.get("arguments").?.string);

    const tool = messages.items[2].object;
    try std.testing.expectEqualStrings("tool", tool.get("role").?.string);
    try std.testing.expectEqualStrings("call_1", tool.get("tool_call_id").?.string);
    try std.testing.expectEqualStrings("a.txt b.txt", tool.get("content").?.string);
}

test "translateRequest: function_call_output accepts content-part arrays" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"input":[{"type":"function_call_output","call_id":"c1","output":[{"type":"output_text","text":"hello"}]}]}
    );
    defer parsed.deinit();
    const tool = parsed.value.object.get("messages").?.array.items[0].object;
    try std.testing.expectEqualStrings("hello", tool.get("content").?.string);
}

test "translateRequest rejects unsupported fields loudly" {
    const allocator = std.testing.allocator;
    try expectTranslateErr(allocator,
        \\{"input":"hi","previous_response_id":"resp_abc"}
    , "stateless");
    try expectTranslateErr(allocator,
        \\{"input":"hi","background":true}
    , "background");
    try expectTranslateErr(allocator,
        \\{"input":"hi","truncation":"auto"}
    , "truncation");
    try expectTranslateErr(allocator,
        \\{"input":"hi","some_future_field":1}
    , "some_future_field");
}

test "translateRequest validates input shapes and top_p" {
    const allocator = std.testing.allocator;
    try expectTranslateErr(allocator,
        \\{"model":"m"}
    , "input");
    try expectTranslateErr(allocator,
        \\{"input":42}
    , "input");
    try expectTranslateErr(allocator,
        \\{"input":[{"role":"user"}]}
    , "content");
    try expectTranslateErr(allocator,
        \\{"input":[{"role":"user","content":[{"type":"input_file","file_data":"..."}]}]}
    , "input_file");
    try expectTranslateErr(allocator,
        \\{"input":[{"type":"local_shell_call","call_id":"c"}]}
    , "local_shell_call");
    try expectTranslateErr(allocator,
        \\{"input":"hi","top_p":0}
    , "top_p");
    try expectTranslateErr(allocator,
        \\{"input":"hi","top_p":1.5}
    , "top_p");
    try expectTranslateErr(allocator, "not json", "valid JSON");
}

test "translateRequest rejects max_output_tokens overflow" {
    const allocator = std.testing.allocator;
    // 5e9 exceeds maxInt(u32); must be a clean 400, never an @intCast panic.
    try expectTranslateErr(allocator,
        \\{"input":"hi","max_output_tokens":5000000000}
    , "max_output_tokens");
    try expectTranslateErr(allocator,
        \\{"input":"hi","max_output_tokens":4294967296}
    , "max_output_tokens");
    // maxInt(u32) itself is accepted and threaded through.
    var parsed = try expectTranslateOk(allocator,
        \\{"input":"hi","max_output_tokens":4294967295}
    );
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 4294967295), parsed.value.object.get("max_tokens").?.integer);
}
