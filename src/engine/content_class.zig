/// Content classifier for the MTP depth controller's cross-request memory.
///
/// Acceptance of a speculative draft chain is a property of the CONTENT being
/// generated, and the per-request controller cannot learn it fast enough: the
/// explore early-exit decision has to be made in the first ~10 rounds, while
/// the in-request trend signal (fast vs slow evidence windows) only separates
/// math from code around round 15. Prompts of the same class, however, produce
/// the same acceptance BEHAVIOUR request after request, so a cheap prompt
/// classifier plus a per-class memory supplies the prior the in-request
/// signals lack.
///
/// The classifier only needs to be right often enough to help: a wrong class
/// seeds a weak prior that the per-request evidence (decay 0.9/round) washes
/// out within ~10 rounds, and the give-up ratio selection falls back to the
/// safe loose default whenever the memory has no confident opinion.
const std = @import("std");

pub const ContentClass = enum(u8) {
    /// Catch-all: no confident classification. Memory is neither read nor
    /// written for this class — behaviour is exactly the pre-memory default.
    general = 0,
    prose,
    code,
    math,
    count,
};

pub const n_classes = @typeInfo(ContentClass).@"enum".fields.len;

fn countOccurrences(text: []const u8, needle: []const u8) u32 {
    var n: u32 = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, text, i, needle)) |pos| {
        n += 1;
        i = pos + needle.len;
    }
    return n;
}

fn countChar(text: []const u8, c: u8) u32 {
    var n: u32 = 0;
    for (text) |ch| {
        if (ch == c) n += 1;
    }
    return n;
}

/// Classify a prompt (the rendered chat text, special tokens and all).
/// Pure keyword/character heuristics; deterministic and allocation-free.
pub fn classify(text: []const u8) ContentClass {
    if (text.len < 8) return .general;

    // Code: fences, braces/semicolons, and common syntax keywords.
    var code_score: u32 = 0;
    code_score += 6 * countOccurrences(text, "```");
    code_score += 2 * countOccurrences(text, "def ");
    code_score += 2 * countOccurrences(text, "function");
    code_score += 2 * countOccurrences(text, "import ");
    code_score += 2 * countOccurrences(text, "class ");
    code_score += 2 * countOccurrences(text, "#include");
    code_score += 2 * countOccurrences(text, "return ");
    code_score += 2 * countOccurrences(text, "python");
    code_score += 2 * countOccurrences(text, "Python");
    code_score += 2 * countOccurrences(text, "implement");
    code_score += 2 * countOccurrences(text, "algorithm");
    code_score += 2 * countOccurrences(text, "debug");
    code_score += countOccurrences(text, "->");
    code_score += countOccurrences(text, "=>");
    code_score += countOccurrences(text, ";\n");
    code_score += countOccurrences(text, "\n    "); // indented blocks
    code_score += countChar(text, '{') + countChar(text, '}');

    // Math: LaTeX markers, digit-operator mix, math verbs.
    var math_score: u32 = 0;
    math_score += 3 * countOccurrences(text, "\\frac");
    math_score += 3 * countOccurrences(text, "\\sum");
    math_score += 3 * countOccurrences(text, "\\int");
    math_score += 3 * countOccurrences(text, "\\sqrt");
    math_score += 2 * countChar(text, '$');
    math_score += 2 * countOccurrences(text, "solve");
    math_score += 2 * countOccurrences(text, "Solve");
    math_score += 2 * countOccurrences(text, "prove");
    math_score += 2 * countOccurrences(text, "Prove");
    math_score += 2 * countOccurrences(text, "equation");
    math_score += 2 * countOccurrences(text, "integral");
    math_score += 2 * countOccurrences(text, "derivative");
    math_score += 2 * countOccurrences(text, "calculate");
    math_score += 2 * countOccurrences(text, "compute");
    math_score += 2 * countOccurrences(text, "Compute");
    math_score += 2 * countOccurrences(text, "show your working");
    math_score += 2 * countOccurrences(text, "Show your working");
    math_score += countOccurrences(text, "^2");
    var digits: u32 = 0;
    var ops: u32 = 0;
    for (text) |ch| {
        if (ch >= '0' and ch <= '9') digits += 1;
        if (ch == '+' or ch == '*' or ch == '/' or ch == '=') ops += 1;
    }
    if (digits >= 2 and ops >= 1) math_score += 4;

    // Counting prompts: "how many X in Y", letter/occurrence counting.
    var count_score: u32 = 0;
    count_score += 4 * countOccurrences(text, "how many");
    count_score += 4 * countOccurrences(text, "How many");
    count_score += 4 * countOccurrences(text, "count from");
    count_score += 4 * countOccurrences(text, "Count from");
    count_score += 3 * countOccurrences(text, "count the");
    count_score += 3 * countOccurrences(text, "Count the");
    count_score += 2 * countOccurrences(text, "occurrences of");
    count_score += 2 * countOccurrences(text, "letter");

    // Prose: discursive verbs, or simply a long text with no other signal.
    var prose_score: u32 = 0;
    prose_score += 2 * countOccurrences(text, "explain");
    prose_score += 2 * countOccurrences(text, "Explain");
    prose_score += 2 * countOccurrences(text, "paragraph");
    prose_score += 2 * countOccurrences(text, "story");
    prose_score += 2 * countOccurrences(text, "essay");
    prose_score += 2 * countOccurrences(text, "describe");
    prose_score += 2 * countOccurrences(text, "Describe");

    if (code_score >= 4 and code_score >= math_score and code_score >= count_score) return .code;
    if (math_score >= 5 and digits >= 2 and math_score >= count_score) return .math;
    if (count_score >= 4) return .count;
    // Prose only when something discursive fired or the prompt is substantial
    // and clean — short ambiguous prompts stay .general so they neither seed
    // from nor pollute a class memory.
    if (prose_score >= 3) return .prose;
    if (text.len >= 200 and code_score == 0 and math_score == 0) return .prose;
    return .general;
}

// ============================================================
// Tests
// ============================================================

test "classify code prompts" {
    const py = "Write a Python function that sorts a list:\n\n```python\ndef sort_list(xs):\n    return sorted(xs)\n```\nExplain how it works.";
    try std.testing.expectEqual(ContentClass.code, classify(py));

    const c = "#include <stdio.h>\nint main() {\n    printf(\"hi\");\n    return 0;\n}\n";
    try std.testing.expectEqual(ContentClass.code, classify(c));

    const js = "Refactor this: const f = (x) => { return x + 1; }; function g() { import y; }\n\n\n\n\n\n\n\n";
    try std.testing.expectEqual(ContentClass.code, classify(js));
}

test "classify math prompts" {
    const latex = "Solve for x: $\\frac{x^2 + 3x + 2}{x + 1} = 5$. Show your steps and prove the result is unique.";
    try std.testing.expectEqual(ContentClass.math, classify(latex));

    const plain = "What is 347 * 291 + 58 / 4 = ? Compute step by step, then solve 1024 = 2^x.";
    try std.testing.expectEqual(ContentClass.math, classify(plain));
}

test "classify counting prompts" {
    const q = "How many times does the letter 'r' appear in the word strawberry?";
    try std.testing.expectEqual(ContentClass.count, classify(q));
}

test "classify prose prompts" {
    const base = "Write a short story about a lighthouse keeper who discovers something unusual on the shore one foggy morning. ";
    const text = base ++ base ++ base ++ base;
    try std.testing.expectEqual(ContentClass.prose, classify(text));
}

test "ambiguous and short prompts stay general" {
    try std.testing.expectEqual(ContentClass.general, classify("hi"));
    try std.testing.expectEqual(ContentClass.general, classify("what is the capital of France?"));
}
