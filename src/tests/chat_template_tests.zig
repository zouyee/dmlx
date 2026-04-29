/// Unit tests for chat template formatting.
///
/// Validates that chat templates produce correct special token sequences
/// for different model architectures.
const std = @import("std");
const chat_template = @import("../tokenizer/chat_template.zig");

const ChatTemplate = chat_template.ChatTemplate;
const ChatMessage = chat_template.ChatMessage;

test "DeepSeek chat template uses correct special tokens" {
    const allocator = std.testing.allocator;
    var template = ChatTemplate.initDeepSeek(allocator);

    // Verify special tokens use half-width characters
    try std.testing.expectEqualStrings("<|begin_of_sentence|>", template.bos_token);
    try std.testing.expectEqualStrings("<|end_of_sentence|>", template.eos_token);

    // Ensure no full-width characters
    try std.testing.expect(std.mem.indexOf(u8, template.bos_token, "｜") == null);
    try std.testing.expect(std.mem.indexOf(u8, template.bos_token, "▁") == null);
    try std.testing.expect(std.mem.indexOf(u8, template.eos_token, "｜") == null);
    try std.testing.expect(std.mem.indexOf(u8, template.eos_token, "▁") == null);
}

test "DeepSeek chat template formats single user message correctly" {
    const allocator = std.testing.allocator;
    var template = ChatTemplate.initDeepSeek(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello" },
    };

    const result = try template.apply(&messages, true);
    defer allocator.free(result);

    // Expected format: <|begin_of_sentence|><|User|>: Hello\n\n<|Assistant|>: 
    const expected = "<|begin_of_sentence|><|User|>: Hello\n\n<|Assistant|>: ";
    try std.testing.expectEqualStrings(expected, result);
}

test "DeepSeek chat template formats system + user message correctly" {
    const allocator = std.testing.allocator;
    var template = ChatTemplate.initDeepSeek(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "system", .content = "You are a helpful assistant." },
        .{ .role = "user", .content = "Hello" },
    };

    const result = try template.apply(&messages, true);
    defer allocator.free(result);

    // Expected format: <|begin_of_sentence|>You are a helpful assistant.\n\n<|User|>: Hello\n\n<|Assistant|>: 
    const expected = "<|begin_of_sentence|>You are a helpful assistant.\n\n<|User|>: Hello\n\n<|Assistant|>: ";
    try std.testing.expectEqualStrings(expected, result);
}

test "DeepSeek chat template formats multi-turn conversation correctly" {
    const allocator = std.testing.allocator;
    var template = ChatTemplate.initDeepSeek(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "assistant", .content = "Hi there!" },
        .{ .role = "user", .content = "How are you?" },
    };

    const result = try template.apply(&messages, true);
    defer allocator.free(result);

    // Expected format includes EOS after assistant response
    const expected = "<|begin_of_sentence|><|User|>: Hello\n\n<|Assistant|>: Hi there!<|end_of_sentence|>\n\n<|User|>: How are you?\n\n<|Assistant|>: ";
    try std.testing.expectEqualStrings(expected, result);
}

test "DeepSeek chat template without generation prompt" {
    const allocator = std.testing.allocator;
    var template = ChatTemplate.initDeepSeek(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello" },
    };

    const result = try template.apply(&messages, false);
    defer allocator.free(result);

    // Should not include trailing <|Assistant|>: 
    const expected = "<|begin_of_sentence|><|User|>: Hello\n\n";
    try std.testing.expectEqualStrings(expected, result);
}

test "DeepSeek special tokens contain only ASCII characters" {
    const allocator = std.testing.allocator;
    var template = ChatTemplate.initDeepSeek(allocator);

    // All characters should be ASCII (< 128)
    for (template.bos_token) |c| {
        try std.testing.expect(c < 128);
    }
    for (template.eos_token) |c| {
        try std.testing.expect(c < 128);
    }
}

test "ChatML template still works correctly" {
    const allocator = std.testing.allocator;
    var template = ChatTemplate.initChatML(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello" },
    };

    const result = try template.apply(&messages, true);
    defer allocator.free(result);

    const expected = "<|im_start|>user\nHello\n<|im_end|>\n<|im_start|>assistant\n";
    try std.testing.expectEqualStrings(expected, result);
}

test "Llama3 template still works correctly" {
    const allocator = std.testing.allocator;
    var template = ChatTemplate.initLlama3(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello" },
    };

    const result = try template.apply(&messages, true);
    defer allocator.free(result);

    const expected = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    try std.testing.expectEqualStrings(expected, result);
}
