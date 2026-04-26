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
    raw, // No formatting, just concatenate messages

    pub fn fromString(name: []const u8) ChatTemplateType {
        if (std.mem.eql(u8, name, "deepseek")) return .deepseek;
        if (std.mem.eql(u8, name, "chatml")) return .chatml;
        if (std.mem.eql(u8, name, "llama3")) return .llama3;
        return .raw;
    }
};

pub const ChatTemplate = struct {
    allocator: std.mem.Allocator,
    template_type: ChatTemplateType,
    bos_token: []const u8,
    eos_token: []const u8,

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

    /// Apply chat template to a list of messages.
    /// Returns allocated string that caller must free.
    pub fn apply(self: *ChatTemplate, messages: []const ChatMessage, add_generation_prompt: bool) ![]const u8 {
        var result = std.ArrayList(u8).empty;
        errdefer result.deinit(self.allocator);

        switch (self.template_type) {
            .deepseek => try self.applyDeepSeek(messages, add_generation_prompt, &result),
            .chatml => try self.applyChatML(messages, add_generation_prompt, &result),
            .llama3 => try self.applyLlama3(messages, add_generation_prompt, &result),
            .raw => try self.applyRaw(messages, add_generation_prompt, &result),
        }

        return result.toOwnedSlice(self.allocator);
    }

    fn applyDeepSeek(self: *ChatTemplate, messages: []const ChatMessage, add_generation_prompt: bool, result: *std.ArrayList(u8)) !void {
        // DeepSeek-V3/V4 format:
        // <｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜>{assistant}<｜end▁of▁sentence｜>...
        try result.appendSlice(self.allocator, self.bos_token);

        for (messages) |msg| {
            if (std.mem.eql(u8, msg.role, "system")) {
                try result.appendSlice(self.allocator, msg.content);
            } else if (std.mem.eql(u8, msg.role, "user")) {
                try result.appendSlice(self.allocator, "<｜User｜>");
                try result.appendSlice(self.allocator, msg.content);
            } else if (std.mem.eql(u8, msg.role, "assistant")) {
                try result.appendSlice(self.allocator, "<｜Assistant｜>");
                try result.appendSlice(self.allocator, msg.content);
                try result.appendSlice(self.allocator, self.eos_token);
            }
        }

        if (add_generation_prompt) {
            try result.appendSlice(self.allocator, "<｜Assistant｜>");
        }
    }

    fn applyChatML(self: *ChatTemplate, messages: []const ChatMessage, add_generation_prompt: bool, result: *std.ArrayList(u8)) !void {
        for (messages) |msg| {
            try result.appendSlice(self.allocator, "<|im_start|>");
            try result.appendSlice(self.allocator, msg.role);
            try result.appendSlice(self.allocator, "\n");
            try result.appendSlice(self.allocator, msg.content);
            try result.appendSlice(self.allocator, "\n<|im_end|>\n");
        }

        if (add_generation_prompt) {
            try result.appendSlice(self.allocator, "<|im_start|>assistant\n");
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
