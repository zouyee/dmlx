//! Tokenizer parity harness: encode fixed strings with the REAL model
//! tokenizer.json and compare against golden ids produced by HF `tokenizers`
//! (tests/golden/qwen38_tokenizer_parity.json, generated 2026-10-30).
//! Guards the Qwen2 pre-tokenizer Split matcher (qwen_word pattern).
const std = @import("std");
const c = @import("mlx").c;
const bpe = @import("tokenizer/bpe.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    c.initErrorHandler();

    const args = try init.minimal.args.toSlice(allocator);
    const model_dir = if (args.len > 1) args[1] else "/Users/zouyee/models/Qwen3.8-27B-4bit";
    const golden_path = if (args.len > 2) args[2] else "tests/golden/qwen38_tokenizer_parity.json";

    const tok_path = try std.fs.path.join(allocator, &.{ model_dir, "tokenizer.json" });
    var tokenizer = try allocator.create(bpe.BpeTokenizer);
    tokenizer.* = bpe.BpeTokenizer.init(allocator);
    try tokenizer.loadFromFile(io, tok_path);

    const golden_text = try std.Io.Dir.cwd().readFileAlloc(io, golden_path, allocator, .limited(64 * 1024 * 1024));
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, golden_text, .{});
    const obj = parsed.value.object;

    var pass: usize = 0;
    var fail: usize = 0;
    var it = obj.iterator();
    while (it.next()) |entry| {
        const text = entry.key_ptr.*;
        const expected = entry.value_ptr.array.items;
        const t0 = std.c.mach_absolute_time();
        const ids = try tokenizer.encode(text, false);
        const t1 = std.c.mach_absolute_time();
        const us = (t1 - t0) * 125 / 3 / 1000;
        if (ids.len != expected.len) {
            std.debug.print("MISMATCH len {d} != {d}: {s}\n", .{ ids.len, expected.len, text });
            fail += 1;
            continue;
        }
        var ok = true;
        for (ids, expected) |got, exp| {
            if (got != @as(u32, @intCast(exp.integer))) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            std.debug.print("MISMATCH ids: {s}\n  got {any}\n", .{ text, ids });
            fail += 1;
        } else {
            pass += 1;
            std.debug.print("ok ({d:>4} us) {s:.40}\n", .{ us, text });
        }
    }
    std.debug.print("parity: {d}/{d} pass\n", .{ pass, pass + fail });
    if (fail > 0) std.process.exit(1);
}
