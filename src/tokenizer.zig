/// Tokenizer module — Runtime-polymorphic text encoding/decoding.
///
/// Backends:
///   - GreedyTokenizer: simplified longest-match (no BPE merges)
///   - BpeTokenizer: HF tokenizer.json BPE with merges
///
/// Usage:
///   var greedy = tokenizer.greedy.GreedyTokenizer.init(allocator);
///   defer greedy.deinit();
///   try greedy.loadFromFile("tokenizer.json");
///   var strategy = greedy.asStrategy();
///   const ids = try strategy.encode("Hello", true);
///   const text = try strategy.decode(ids);
const std = @import("std");

pub const interface = @import("tokenizer/interface.zig");
pub const greedy = @import("tokenizer/greedy.zig");
pub const bpe = @import("tokenizer/bpe.zig");
pub const chat_template = @import("tokenizer/chat_template.zig");

// Re-exports for convenience
pub const TokenizerStrategy = interface.TokenizerStrategy;
pub const GreedyTokenizer = greedy.GreedyTokenizer;
pub const BpeTokenizer = bpe.BpeTokenizer;
pub const ChatTemplate = chat_template.ChatTemplate;
pub const ChatMessage = chat_template.ChatMessage;
pub const ChatTemplateType = chat_template.ChatTemplateType;

// Backward compatibility alias
pub const Tokenizer = GreedyTokenizer;
