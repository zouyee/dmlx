/// SSE (Server-Sent Events) streaming helpers.
const std = @import("std");
const http_mod = @import("http.zig");
const utils_mod = @import("utils.zig");
const streamWriteAll = http_mod.streamWriteAll;
const jsonEscapeInto = utils_mod.jsonEscapeInto;

// ------------------------------------------------------------------
// SSE (Server-Sent Events) streaming helpers
// ------------------------------------------------------------------

/// Write a single SSE data event. Follows the SSE protocol:
///   data: <payload>\n\n
pub fn writeSSEEvent(writer: anytype, data: []const u8) !void {
    try writer.writeAll("data: ");
    try writer.writeAll(data);
    try writer.writeAll("\n\n");
}

/// Write an SSE keep-alive comment to prevent client read timeouts.
///   : keep-alive\n\n
pub fn writeSSEKeepAlive(writer: anytype) !void {
    try writer.writeAll(": keep-alive\n\n");
}

/// SSE buffered writer that wraps a network stream for SSE output.
pub const SSEWriter = struct {
    connection: std.Io.net.Stream,
    io: std.Io,
    buf: [4096]u8 = undefined,

    pub fn sendEvent(self: *SSEWriter, data: []const u8) !void {
        var w = self.connection.writer(self.io, &self.buf);
        try writeSSEEvent(&w.interface, data);
        try w.interface.flush();
    }

    pub fn sendKeepAlive(self: *SSEWriter) !void {
        var w = self.connection.writer(self.io, &self.buf);
        try writeSSEKeepAlive(&w.interface);
        try w.interface.flush();
    }
};

/// Write SSE response headers (HTTP/1.1 200 OK with text/event-stream).
pub fn writeSSEHeaders(connection: std.Io.net.Stream, io: std.Io) !void {
    try streamWriteAll(connection, io, "HTTP/1.1 200 OK\r\n");
    try streamWriteAll(connection, io, "Content-Type: text/event-stream\r\n");
    try streamWriteAll(connection, io, "Cache-Control: no-cache\r\n");
    try streamWriteAll(connection, io, "Connection: keep-alive\r\n");
    try streamWriteAll(connection, io, "\r\n");
}

/// Format a single SSE chunk in OpenAI-compatible format.
/// Returns an allocated JSON string; caller must free.
pub fn formatSSEChunk(
    allocator: std.mem.Allocator,
    completion_id: u64,
    created: u64,
    model_name: []const u8,
    token_text: ?[]const u8,
    finish_reason: ?[]const u8,
) ![]u8 {
    // JSON-escape the token text if present
    var escaped_buf = std.ArrayList(u8).empty;
    defer escaped_buf.deinit(allocator);

    if (token_text) |text| {
        try jsonEscapeInto(&escaped_buf, allocator, text);
    }

    const content_part = if (token_text != null)
        try std.fmt.allocPrint(allocator, "\"delta\":{{\"content\":\"{s}\"}}", .{escaped_buf.items})
    else
        try std.fmt.allocPrint(allocator, "\"delta\":{{}}", .{});
    defer allocator.free(content_part);

    const finish_part = if (finish_reason) |reason|
        try std.fmt.allocPrint(allocator, ",\"finish_reason\":\"{s}\"", .{reason})
    else
        try allocator.dupe(u8, ",\"finish_reason\":null");
    defer allocator.free(finish_part);

    return try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,{s}{s}}}]}}
    , .{
        completion_id,
        created,
        model_name,
        content_part,
        finish_part,
    });
}
