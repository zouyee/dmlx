/// Chat template formatting for conversational LLMs.
///
/// Supports common formats:
/// - DeepSeek (V2/V3/V4): system + <|User|>user<|Assistant|>assistant<|end▁of▁sentence|>
/// - ChatML: <|im_start|>system/ user/ assistant<|im_end|>
/// - Llama-3: <|start_header_id|>system/ user/ assistant<|end_header_id|>\n\ncontent<|eot_id|>
const std = @import("std");

pub const ChatMessage = struct {
    role: []const u8, // "system", "user", "assistant"
    content: []const u8,
};

pub const ChatTemplateType = enum {
    deepseek,
    chatml,
    llama3,
    glm,
    raw, // No formatting, just concatenate messages

    pub fn fromString(name: []const u8) ChatTemplateType {
        if (std.mem.eql(u8, name, "deepseek")) return .deepseek;
        if (std.mem.eql(u8, name, "chatml")) return .chatml;
        if (std.mem.eql(u8, name, "llama3")) return .llama3;
        if (std.mem.eql(u8, name, "glm")) return .glm;
        return .raw;
    }
};

pub const ChatTemplate = struct {
    allocator: std.mem.Allocator,
    template_type: ChatTemplateType,
    bos_token: []const u8,
    eos_token: []const u8,
    /// DeepSeek: 'chat' (non-thinking) or 'thinking' (model reasons before answering)
    thinking_mode: []const u8 = "chat",
    /// Qwen ChatML: when true, the generation prompt appends an EMPTY think
    /// block (`<think>\n\n</think>\n\n`), matching the HF chat template's
    /// `enable_thinking=false` branch (chat_template.jinja:165) — omlx's
    /// default (`thinking_budget_enabled: false`). Without it the bare
    /// `<|im_start|>assistant\n` prompt lets the model free-think, which
    /// generates different (longer) content than engines that honor
    /// enable_thinking=false and confounds decode-rate comparisons.
    empty_think: bool = false,
    /// Qwen ChatML reasoning effort: "xhigh" (default), "medium", or "low",
    /// mirroring chat_template.jinja:47-56. Only xhigh and low emit the
    /// system-block instructions; medium emits none. Ignored when
    /// `empty_think` is set (the jinja's enable_thinking=false branch drops
    /// the reasoning system block entirely).
    reasoning_effort: []const u8 = "xhigh",

    pub fn init(allocator: std.mem.Allocator, template_type: ChatTemplateType) ChatTemplate {
        return .{
            .allocator = allocator,
            .template_type = template_type,
            .bos_token = "",
            .eos_token = "",
        };
    }

    pub fn initDeepSeek(allocator: std.mem.Allocator) ChatTemplate {
        return .{
            .allocator = allocator,
            .template_type = .deepseek,
            .bos_token = "<｜begin▁of▁sentence｜>",
            .eos_token = "<｜end▁of▁sentence｜>",
            .thinking_mode = "chat",
        };
    }

    pub fn initChatML(allocator: std.mem.Allocator) ChatTemplate {
        return .{
            .allocator = allocator,
            .template_type = .chatml,
            .bos_token = "",
            .eos_token = "<|im_end|>",
        };
    }

    pub fn initLlama3(allocator: std.mem.Allocator) ChatTemplate {
        return .{
            .allocator = allocator,
            .template_type = .llama3,
            .bos_token = "<|begin_of_text|>",
            .eos_token = "<|eot_id|>",
        };
    }

    pub fn initGlm(allocator: std.mem.Allocator) ChatTemplate {
        return .{
            .allocator = allocator,
            .template_type = .glm,
            .bos_token = "[gMASK]<sop>",
            .eos_token = "<|endoftext|>",
        };
    }

    /// Apply chat template to a list of messages.
    /// Returns allocated string that caller must free.
    pub fn apply(self: *ChatTemplate, messages: []const ChatMessage, add_generation_prompt: bool) ![]const u8 {
        var result = std.ArrayList(u8).empty;
        errdefer result.deinit(self.allocator);

        switch (self.template_type) {
            .deepseek => try self.applyDeepSeek(messages, add_generation_prompt, &result),
            .chatml => try self.applyChatML(messages, add_generation_prompt, &result),
            .llama3 => try self.applyLlama3(messages, add_generation_prompt, &result),
            .glm => try self.applyGlm(messages, add_generation_prompt, &result),
            .raw => try self.applyRaw(messages, add_generation_prompt, &result),
        }

        return result.toOwnedSlice(self.allocator);
    }

    fn applyDeepSeek(self: *ChatTemplate, messages: []const ChatMessage, add_generation_prompt: bool, result: *std.ArrayList(u8)) !void {
        // Matches official chat_template.jinja from DeepSeek-V4-Flash-4bit:
        // {%- set mode = thinking_mode|default('chat') -%}
        // User msg: <｜User｜>content<｜Assistant｜>
        //   - last+thinking: <think>
        //   - else: </think>
        // Assistant msg: reasoning(if thinking) + </think> + content + <eos>
        // add_generation_prompt: only when messages[-1].role != 'user' → <Assistant|></think>
        const is_thinking = std.mem.eql(u8, self.thinking_mode, "thinking");

        try result.appendSlice(self.allocator, self.bos_token);

        for (messages, 0..) |msg, idx| {
            const is_last = idx == messages.len - 1;

            if (std.mem.eql(u8, msg.role, "system")) {
                try result.appendSlice(self.allocator, msg.content);
            } else if (std.mem.eql(u8, msg.role, "user")) {
                try result.appendSlice(self.allocator, "<｜User｜>");
                try result.appendSlice(self.allocator, msg.content);
                try result.appendSlice(self.allocator, "<｜Assistant｜>");
                // Add thinking or non-thinking marker
                if (is_last and is_thinking) {
                    try result.appendSlice(self.allocator, "<think>");
                } else {
                    try result.appendSlice(self.allocator, "</think>");
                }
            } else if (std.mem.eql(u8, msg.role, "assistant")) {
                // Assistant: content may include thinking; we just append content + eos
                try result.appendSlice(self.allocator, msg.content);
                try result.appendSlice(self.allocator, self.eos_token);
            }
        }

        // add_generation_prompt: only when last message is NOT user
        if (add_generation_prompt and messages.len > 0 and !std.mem.eql(u8, messages[messages.len - 1].role, "user")) {
            try result.appendSlice(self.allocator, "<｜Assistant｜></think>");
        }
    }

    pub const chatml_reasoning_xhigh = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    pub const chatml_reasoning_low = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

    fn applyChatML(self: *ChatTemplate, messages: []const ChatMessage, add_generation_prompt: bool, result: *std.ArrayList(u8)) !void {
        // Qwen3.5 chat_template.jinja parity (no-tools, text-only path).
        //
        // Thinking ON (the HF default, enable_thinking undefined/true): a
        // system block carries the reasoning-effort instructions — merged
        // with the caller's leading system message when one is present —
        // and the generation prompt opens "<think>\n". empty_think maps to
        // the jinja's enable_thinking=false branch: no reasoning system
        // block and an empty think block in the generation prompt.
        //
        // Before 2026-11-02 this produced NEITHER the default system prompt
        // NOR the think block, diverging from what HF/omlx send by default
        // (and silently benching different content — see the close-decode-gap
        // plan's 2026-11-02 content-mismatch entry).
        var start: usize = 0;
        if (!self.empty_think) {
            const instructions: ?[]const u8 = if (std.mem.eql(u8, self.reasoning_effort, "xhigh"))
                chatml_reasoning_xhigh
            else if (std.mem.eql(u8, self.reasoning_effort, "low"))
                chatml_reasoning_low
            else
                null; // "medium" (and anything else) emits no instructions
            const sys_content: ?[]const u8 = blk: {
                if (messages.len == 0 or !std.mem.eql(u8, messages[0].role, "system")) break :blk null;
                const trimmed = std.mem.trim(u8, messages[0].content, " \t\n\r");
                break :blk if (trimmed.len > 0) trimmed else null;
            };
            if (sys_content != null or instructions != null) {
                try result.appendSlice(self.allocator, "<|im_start|>system\n");
                if (instructions) |ins| {
                    try result.appendSlice(self.allocator, ins);
                    if (sys_content != null) try result.appendSlice(self.allocator, "\n\n");
                }
                if (sys_content) |sc| try result.appendSlice(self.allocator, sc);
                try result.appendSlice(self.allocator, "<|im_end|>\n");
            }
            // The leading system message is consumed by the merge above even
            // when its content is empty (the jinja emits nothing for it).
            if (messages.len > 0 and std.mem.eql(u8, messages[0].role, "system")) start = 1;
        }
        for (messages[start..]) |msg| {
            try result.appendSlice(self.allocator, "<|im_start|>");
            try result.appendSlice(self.allocator, msg.role);
            try result.appendSlice(self.allocator, "\n");
            // Historical assistant turns carry an (empty, since our message
            // type has no reasoning_content field) think block before the
            // content — jinja lines 116-117, preserve_thinking default true,
            // independent of enable_thinking.
            if (std.mem.eql(u8, msg.role, "assistant")) {
                try result.appendSlice(self.allocator, "<think>\n\n</think>\n\n");
            }
            try result.appendSlice(self.allocator, msg.content);
            try result.appendSlice(self.allocator, "<|im_end|>\n");
        }

        if (add_generation_prompt) {
            try result.appendSlice(self.allocator, "<|im_start|>assistant\n");
            if (self.empty_think) {
                try result.appendSlice(self.allocator, "<think>\n\n</think>\n\n");
            } else {
                try result.appendSlice(self.allocator, "<think>\n");
            }
        }
    }

    fn applyLlama3(self: *ChatTemplate, messages: []const ChatMessage, add_generation_prompt: bool, result: *std.ArrayList(u8)) !void {
        // Llama-3 format:
        // <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
        // <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
        // <|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>
        try result.appendSlice(self.allocator, self.bos_token);

        for (messages) |msg| {
            try result.appendSlice(self.allocator, "<|start_header_id|>");
            try result.appendSlice(self.allocator, msg.role);
            try result.appendSlice(self.allocator, "<|end_header_id|>\n\n");
            try result.appendSlice(self.allocator, msg.content);
            try result.appendSlice(self.allocator, self.eos_token);
        }

        if (add_generation_prompt) {
            try result.appendSlice(self.allocator, "<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
    }

    fn applyGlm(self: *ChatTemplate, messages: []const ChatMessage, add_generation_prompt: bool, result: *std.ArrayList(u8)) !void {
        // Matches GLM-5.x chat_template.jinja (no tools, no reasoning params,
        // pure conversation path):
        //   [gMASK]<sop><|system|>Reasoning Effort: Max
        //   system:    <|system|>{content}
        //   user:      <|user|>{content}
        //   assistant: <|assistant|><think></think>{content}
        //   generation prompt: <|assistant|><think>
        // reasoning_effort undefined in jinja defaults to 'max', so the
        // "Reasoning Effort: Max" system block is unconditional.
        try result.appendSlice(self.allocator, self.bos_token);
        try result.appendSlice(self.allocator, "<|system|>Reasoning Effort: Max");

        for (messages) |msg| {
            if (std.mem.eql(u8, msg.role, "system")) {
                try result.appendSlice(self.allocator, "<|system|>");
                try result.appendSlice(self.allocator, msg.content);
            } else if (std.mem.eql(u8, msg.role, "user")) {
                try result.appendSlice(self.allocator, "<|user|>");
                try result.appendSlice(self.allocator, msg.content);
            } else if (std.mem.eql(u8, msg.role, "assistant")) {
                try result.appendSlice(self.allocator, "<|assistant|><think></think>");
                try result.appendSlice(self.allocator, std.mem.trim(u8, msg.content, " \n\t\r"));
            }
        }

        if (add_generation_prompt) {
            try result.appendSlice(self.allocator, "<|assistant|><think>");
        }
    }

    fn applyRaw(self: *ChatTemplate, messages: []const ChatMessage, add_generation_prompt: bool, result: *std.ArrayList(u8)) !void {
        for (messages) |msg| {
            try result.appendSlice(self.allocator, msg.content);
            try result.appendSlice(self.allocator, "\n");
        }
        if (add_generation_prompt) {
            try result.appendSlice(self.allocator, "Assistant: ");
        }
    }
};

test "applyGlm: system+user+generation prompt" {
    const allocator = std.testing.allocator;
    var tpl = ChatTemplate.initGlm(allocator);
    const msgs = [_]ChatMessage{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Hi" },
    };
    const out = try tpl.apply(&msgs, true);
    defer allocator.free(out);
    try std.testing.expect(std.mem.startsWith(u8, out, "[gMASK]<sop>"));
    try std.testing.expect(std.mem.indexOf(u8, out, "<|system|>Reasoning Effort: Max") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "<|system|>You are helpful.") != null);
    try std.testing.expect(std.mem.endsWith(u8, out, "<|assistant|><think>"));
}

test "applyGlm: multi-turn with assistant history" {
    const allocator = std.testing.allocator;
    var tpl = ChatTemplate.initGlm(allocator);
    const msgs = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
        .{ .role = "assistant", .content = "Hello!" },
        .{ .role = "user", .content = "How are you?" },
    };
    const out = try tpl.apply(&msgs, true);
    defer allocator.free(out);
    const expected = "[gMASK]<sop><|system|>Reasoning Effort: Max" ++
        "<|user|>Hi<|assistant|><think></think>Hello!" ++
        "<|user|>How are you?<|assistant|><think>";
    try std.testing.expectEqualStrings(expected, out);
}

test "applyGlm: no generation prompt" {
    const allocator = std.testing.allocator;
    var tpl = ChatTemplate.initGlm(allocator);
    const msgs = [_]ChatMessage{
        .{ .role = "user", .content = "Hi" },
    };
    const out = try tpl.apply(&msgs, false);
    defer allocator.free(out);
    try std.testing.expectEqualStrings("[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi", out);
}
