/// Guided decoding for constrained LLM generation.
///
/// Uses a finite state machine (FSM) to constrain token generation to follow
/// a grammar (JSON schema or regex pattern). At each generation step, the FSM
/// determines which tokens are valid in the current state, and disallowed token
/// logits are set to negative infinity before sampling.
///
/// References:
///   - Willard & Louf, "Efficient Guided Generation for Large Language Models" (2023)
///
/// Requirements: R17.1, R17.2, R17.3
const std = @import("std");
const ops = @import("mlx").ops;
const array_mod = @import("mlx").array;
const comparison = @import("mlx").comparison;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

// ============================================================
// FSM State
// ============================================================

/// A single state in the finite state machine.
pub const State = struct {
    /// Mapping from token_id → next_state_index.
    transitions: std.AutoHashMap(u32, usize),
    /// Whether this state is an accepting (final) state.
    is_accepting: bool,
    /// Precomputed list of allowed token IDs in this state.
    allowed_tokens: []const u32,
    /// Allocator used for allowed_tokens (for cleanup).
    allocator: std.mem.Allocator,

    pub fn deinit(self: *State) void {
        self.transitions.deinit();
        if (self.allowed_tokens.len > 0) {
            self.allocator.free(self.allowed_tokens);
        }
    }
};

// ============================================================
// FiniteStateMachine
// ============================================================

/// A finite state machine for constraining token generation.
pub const FiniteStateMachine = struct {
    states: []State,
    allocator: std.mem.Allocator,
    eos_token: u32,

    pub fn allowedTokens(self: *const FiniteStateMachine, state_idx: usize) []const u32 {
        if (state_idx >= self.states.len) return &.{};
        return self.states[state_idx].allowed_tokens;
    }

    pub fn transition(self: *const FiniteStateMachine, current: usize, token: u32) usize {
        if (current >= self.states.len) return current;
        return self.states[current].transitions.get(token) orelse current;
    }

    pub fn isAccepting(self: *const FiniteStateMachine, state_idx: usize) bool {
        if (state_idx >= self.states.len) return false;
        return self.states[state_idx].is_accepting;
    }

    pub fn fromJsonSchema(allocator: std.mem.Allocator, schema: []const u8) !FiniteStateMachine {
        const schema_type = parseSchemaType(schema);
        const has_enum = std.mem.indexOf(u8, schema, "\"enum\"") != null;
        return switch (schema_type) {
            .string => if (has_enum) try buildStringEnumFSM(allocator, schema) else try buildStringFSM(allocator),
            .integer => try buildIntegerFSM(allocator),
            .boolean => try buildBooleanFSM(allocator),
            .unknown => try buildStringFSM(allocator),
        };
    }

    pub fn fromRegex(allocator: std.mem.Allocator, pattern: []const u8) !FiniteStateMachine {
        return try buildRegexFSM(allocator, pattern);
    }

    pub fn deinit(self: *FiniteStateMachine) void {
        for (self.states) |*state| {
            @constCast(state).deinit();
        }
        self.allocator.free(self.states);
    }
};

// ============================================================
// GuidedDecoder
// ============================================================

pub const GuidedDecoder = struct {
    fsm: FiniteStateMachine,
    current_state: usize,

    pub fn init(fsm: FiniteStateMachine) GuidedDecoder {
        return .{ .fsm = fsm, .current_state = 0 };
    }

    pub fn maskLogits(self: *GuidedDecoder, logits: Array, ctx: EagerContext) !Array {
        const allowed = self.fsm.allowedTokens(self.current_state);
        return applyTokenMask(logits, allowed, ctx);
    }

    pub fn advance(self: *GuidedDecoder, token: u32) void {
        self.current_state = self.fsm.transition(self.current_state, token);
    }

    pub fn reset(self: *GuidedDecoder) void {
        self.current_state = 0;
    }

    pub fn isComplete(self: *const GuidedDecoder) bool {
        return self.fsm.isAccepting(self.current_state);
    }

    pub fn deinit(self: *GuidedDecoder) void {
        self.fsm.deinit();
    }
};

// ============================================================
// Token Masking (MLX computation graph)
// ============================================================

fn applyTokenMask(logits: Array, allowed_tokens: []const u32, ctx: EagerContext) !Array {
    const logits_shape = logits.shape();
    const vocab_size: usize = @intCast(logits_shape[logits_shape.len - 1]);

    var mask_data = try ctx.allocator.alloc(bool, vocab_size);
    defer ctx.allocator.free(mask_data);
    @memset(mask_data, false);

    for (allowed_tokens) |tok| {
        if (tok < vocab_size) {
            mask_data[tok] = true;
        }
    }

    const mask = try Array.fromData(ctx.allocator, bool, mask_data, &[_]i32{@intCast(vocab_size)});
    defer mask.deinit();

    const neg_inf = try Array.scalar(ctx.allocator, f32, -std.math.inf(f32));
    defer neg_inf.deinit();

    const neg_inf_broadcast = try ops.broadcastTo(ctx, neg_inf, logits_shape);
    defer neg_inf_broadcast.deinit();

    return ops.where(ctx, mask, logits, neg_inf_broadcast);
}

// ============================================================
// Schema Type Detection
// ============================================================

const SchemaType = enum { string, integer, boolean, unknown };

fn parseSchemaType(schema: []const u8) SchemaType {
    if (std.mem.indexOf(u8, schema, "\"integer\"") != null) return .integer;
    if (std.mem.indexOf(u8, schema, "\"boolean\"") != null) return .boolean;
    if (std.mem.indexOf(u8, schema, "\"string\"") != null) return .string;
    return .unknown;
}

// ============================================================
// Helper: create a State with given transitions and allowed tokens
// ============================================================

pub const Transition = struct { char: u32, next: usize };

fn makeState(allocator: std.mem.Allocator, transitions: []const Transition, is_accepting: bool) !State {
    var trans = std.AutoHashMap(u32, usize).init(allocator);
    errdefer trans.deinit();
    var allowed = try allocator.alloc(u32, transitions.len);
    errdefer allocator.free(allowed);
    for (transitions, 0..) |t, i| {
        try trans.put(t.char, t.next);
        allowed[i] = t.char;
    }
    return .{ .transitions = trans, .is_accepting = is_accepting, .allowed_tokens = allowed, .allocator = allocator };
}

fn makeEmptyState(allocator: std.mem.Allocator, is_accepting: bool) !State {
    const trans = std.AutoHashMap(u32, usize).init(allocator);
    return .{ .transitions = trans, .is_accepting = is_accepting, .allowed_tokens = try allocator.alloc(u32, 0), .allocator = allocator };
}

// ============================================================
// FSM Builders — JSON Schema
// ============================================================

fn buildStringEnumFSM(allocator: std.mem.Allocator, schema: []const u8) !FiniteStateMachine {
    const enum_start = std.mem.indexOf(u8, schema, "[") orelse return buildStringFSM(allocator);
    const enum_end = std.mem.indexOf(u8, schema, "]") orelse return buildStringFSM(allocator);
    if (enum_start >= enum_end) return buildStringFSM(allocator);

    const enum_content = schema[enum_start + 1 .. enum_end];

    // Collect enum values (slices into schema)
    var values_buf: [64][]const u8 = undefined;
    var num_values: usize = 0;

    var iter = std.mem.splitScalar(u8, enum_content, ',');
    while (iter.next()) |raw_val| {
        const trimmed = std.mem.trim(u8, raw_val, " \t\n\r");
        if (trimmed.len >= 2 and trimmed[0] == '"' and trimmed[trimmed.len - 1] == '"') {
            if (num_values < values_buf.len) {
                values_buf[num_values] = trimmed[1 .. trimmed.len - 1];
                num_values += 1;
            }
        }
    }

    if (num_values == 0) return buildStringFSM(allocator);
    const values = values_buf[0..num_values];

    // Build states using ArrayList(State).empty pattern
    var state_list = std.ArrayList(State).empty;
    errdefer {
        for (state_list.items) |*s| s.deinit();
        state_list.deinit(allocator);
    }

    // State 0: expects '"'
    var s0_trans = std.AutoHashMap(u32, usize).init(allocator);
    try s0_trans.put('"', 1);
    var s0_allowed = try allocator.alloc(u32, 1);
    s0_allowed[0] = '"';
    try state_list.append(allocator, .{ .transitions = s0_trans, .is_accepting = false, .allowed_tokens = s0_allowed, .allocator = allocator });

    // State 1: after opening quote — branch on first char of each value
    const s1_trans = std.AutoHashMap(u32, usize).init(allocator);
    var s1_allowed_list = std.ArrayList(u32).empty;
    defer s1_allowed_list.deinit(allocator);

    try state_list.append(allocator, .{ .transitions = s1_trans, .is_accepting = false, .allowed_tokens = &.{}, .allocator = allocator });

    // Accept state index will be at the end
    var accept_state_idx: usize = 2; // placeholder, updated below

    for (values) |value| {
        if (value.len == 0) continue;
        const first_char: u32 = value[0];
        const chain_start = state_list.items.len;

        if (!state_list.items[1].transitions.contains(first_char)) {
            try state_list.items[1].transitions.put(first_char, chain_start);
            try s1_allowed_list.append(allocator, first_char);
        }

        for (value, 0..) |_, ci| {
            var trans = std.AutoHashMap(u32, usize).init(allocator);
            var allowed: []u32 = undefined;
            if (ci + 1 < value.len) {
                const next_char: u32 = value[ci + 1];
                try trans.put(next_char, chain_start + ci + 1);
                allowed = try allocator.alloc(u32, 1);
                allowed[0] = next_char;
            } else {
                // Placeholder — will fix accept_state_idx after loop
                try trans.put('"', 0); // temporary
                allowed = try allocator.alloc(u32, 1);
                allowed[0] = '"';
            }
            try state_list.append(allocator, .{ .transitions = trans, .is_accepting = false, .allowed_tokens = allowed, .allocator = allocator });
        }
    }

    // Now we know the accept state index
    accept_state_idx = state_list.items.len;

    // Fix up the last state of each value chain to point to accept_state_idx
    var chain_offset: usize = 2; // first chain starts at state 2
    for (values) |value| {
        if (value.len == 0) continue;
        const last_state_idx = chain_offset + value.len - 1;
        // Update the '"' transition to point to accept_state_idx
        try state_list.items[last_state_idx].transitions.put('"', accept_state_idx);
        chain_offset += value.len;
    }

    // Update state 1's allowed_tokens
    state_list.items[1].allowed_tokens = try s1_allowed_list.toOwnedSlice(allocator);

    // Accepting state
    try state_list.append(allocator, try makeEmptyState(allocator, true));

    return .{ .states = try state_list.toOwnedSlice(allocator), .allocator = allocator, .eos_token = 0 };
}

fn buildStringFSM(allocator: std.mem.Allocator) !FiniteStateMachine {
    var state_list = std.ArrayList(State).empty;
    errdefer {
        for (state_list.items) |*s| s.deinit();
        state_list.deinit(allocator);
    }

    // State 0: expects '"'
    var s0_trans = std.AutoHashMap(u32, usize).init(allocator);
    try s0_trans.put('"', 1);
    var s0_allowed = try allocator.alloc(u32, 1);
    s0_allowed[0] = '"';
    try state_list.append(allocator, .{ .transitions = s0_trans, .is_accepting = false, .allowed_tokens = s0_allowed, .allocator = allocator });

    // State 1: inside string
    var s1_trans = std.AutoHashMap(u32, usize).init(allocator);
    var s1_allowed = try allocator.alloc(u32, 95); // printable ASCII 32..126
    var idx: usize = 0;
    var ch: u32 = 32;
    while (ch <= 126) : (ch += 1) {
        if (ch == '"') {
            try s1_trans.put(ch, 2);
        } else {
            try s1_trans.put(ch, 1);
        }
        s1_allowed[idx] = ch;
        idx += 1;
    }
    try state_list.append(allocator, .{ .transitions = s1_trans, .is_accepting = false, .allowed_tokens = s1_allowed, .allocator = allocator });

    // State 2: accepting
    try state_list.append(allocator, try makeEmptyState(allocator, true));

    return .{ .states = try state_list.toOwnedSlice(allocator), .allocator = allocator, .eos_token = 0 };
}

fn buildIntegerFSM(allocator: std.mem.Allocator) !FiniteStateMachine {
    var state_list = std.ArrayList(State).empty;
    errdefer {
        for (state_list.items) |*s| s.deinit();
        state_list.deinit(allocator);
    }

    // State 0: '-' or digit
    var s0_trans = std.AutoHashMap(u32, usize).init(allocator);
    var s0_allowed = try allocator.alloc(u32, 11);
    try s0_trans.put('-', 1);
    s0_allowed[0] = '-';
    var d: u32 = '0';
    var i: usize = 1;
    while (d <= '9') : (d += 1) {
        try s0_trans.put(d, 2);
        s0_allowed[i] = d;
        i += 1;
    }
    try state_list.append(allocator, .{ .transitions = s0_trans, .is_accepting = false, .allowed_tokens = s0_allowed, .allocator = allocator });

    // State 1: after '-', digits only
    var s1_trans = std.AutoHashMap(u32, usize).init(allocator);
    var s1_allowed = try allocator.alloc(u32, 10);
    d = '0';
    i = 0;
    while (d <= '9') : (d += 1) {
        try s1_trans.put(d, 2);
        s1_allowed[i] = d;
        i += 1;
    }
    try state_list.append(allocator, .{ .transitions = s1_trans, .is_accepting = false, .allowed_tokens = s1_allowed, .allocator = allocator });

    // State 2: accepting, more digits ok
    var s2_trans = std.AutoHashMap(u32, usize).init(allocator);
    var s2_allowed = try allocator.alloc(u32, 10);
    d = '0';
    i = 0;
    while (d <= '9') : (d += 1) {
        try s2_trans.put(d, 2);
        s2_allowed[i] = d;
        i += 1;
    }
    try state_list.append(allocator, .{ .transitions = s2_trans, .is_accepting = true, .allowed_tokens = s2_allowed, .allocator = allocator });

    return .{ .states = try state_list.toOwnedSlice(allocator), .allocator = allocator, .eos_token = 0 };
}

fn buildBooleanFSM(allocator: std.mem.Allocator) !FiniteStateMachine {
    var state_list = std.ArrayList(State).empty;
    errdefer {
        for (state_list.items) |*s| s.deinit();
        state_list.deinit(allocator);
    }

    // 0=start, 1=t, 2=tr, 3=tru, 4=true(accept), 5=f, 6=fa, 7=fal, 8=fals, 9=false(accept)
    // State 0
    var s0_trans = std.AutoHashMap(u32, usize).init(allocator);
    try s0_trans.put('t', 1);
    try s0_trans.put('f', 5);
    var s0_allowed = try allocator.alloc(u32, 2);
    s0_allowed[0] = 't';
    s0_allowed[1] = 'f';
    try state_list.append(allocator, .{ .transitions = s0_trans, .is_accepting = false, .allowed_tokens = s0_allowed, .allocator = allocator });

    // "true" chain: states 1,2,3
    const true_chain = [_]Transition{ .{ .char = 'r', .next = 2 }, .{ .char = 'u', .next = 3 }, .{ .char = 'e', .next = 4 } };
    for (true_chain) |tc| {
        try state_list.append(allocator, try makeState(allocator, &[_]Transition{.{ .char = tc.char, .next = tc.next }}, false));
    }
    // State 4: true accepting
    try state_list.append(allocator, try makeEmptyState(allocator, true));

    // "false" chain: states 5,6,7,8
    const false_chain = [_]Transition{ .{ .char = 'a', .next = 6 }, .{ .char = 'l', .next = 7 }, .{ .char = 's', .next = 8 }, .{ .char = 'e', .next = 9 } };
    for (false_chain) |fc| {
        try state_list.append(allocator, try makeState(allocator, &[_]Transition{.{ .char = fc.char, .next = fc.next }}, false));
    }
    // State 9: false accepting
    try state_list.append(allocator, try makeEmptyState(allocator, true));

    return .{ .states = try state_list.toOwnedSlice(allocator), .allocator = allocator, .eos_token = 0 };
}

// ============================================================
// FSM Builder — Regex
// ============================================================

fn parseCharClass(pattern: []const u8, allocator: std.mem.Allocator) ![]u32 {
    var chars = std.ArrayList(u32).empty;
    errdefer chars.deinit(allocator);

    var pi: usize = 0;
    while (pi < pattern.len) {
        if (pi + 2 < pattern.len and pattern[pi + 1] == '-') {
            const start = pattern[pi];
            const end = pattern[pi + 2];
            var ch: u32 = start;
            while (ch <= end) : (ch += 1) {
                try chars.append(allocator, ch);
            }
            pi += 3;
        } else {
            try chars.append(allocator, pattern[pi]);
            pi += 1;
        }
    }
    return try chars.toOwnedSlice(allocator);
}

fn allPrintableAscii(allocator: std.mem.Allocator) ![]u32 {
    var chars = try allocator.alloc(u32, 95);
    var idx: usize = 0;
    var ch: u32 = 32;
    while (ch <= 126) : (ch += 1) {
        chars[idx] = ch;
        idx += 1;
    }
    return chars;
}

const Quantifier = enum { one, zero_or_more, one_or_more, optional };

const Atom = struct {
    chars: []u32,
    quantifier: Quantifier,
};

fn buildRegexFSM(allocator: std.mem.Allocator, pattern: []const u8) !FiniteStateMachine {
    // Check for top-level alternation
    var alt_parts = std.ArrayList([]const u8).empty;
    defer alt_parts.deinit(allocator);

    var bracket_depth: usize = 0;
    var part_start: usize = 0;
    for (pattern, 0..) |ch, pi| {
        if (ch == '[') bracket_depth += 1;
        if (ch == ']' and bracket_depth > 0) bracket_depth -= 1;
        if (ch == '|' and bracket_depth == 0) {
            try alt_parts.append(allocator, pattern[part_start..pi]);
            part_start = pi + 1;
        }
    }
    try alt_parts.append(allocator, pattern[part_start..]);

    if (alt_parts.items.len > 1) {
        return buildAlternationFSM(allocator, alt_parts.items);
    }

    // Parse atoms
    var atoms = std.ArrayList(Atom).empty;
    defer {
        for (atoms.items) |atom| allocator.free(atom.chars);
        atoms.deinit(allocator);
    }

    var pi: usize = 0;
    while (pi < pattern.len) {
        var chars: []u32 = undefined;

        if (pattern[pi] == '[') {
            const close = std.mem.indexOfScalarPos(u8, pattern, pi + 1, ']') orelse return error.InvalidRegex;
            chars = try parseCharClass(pattern[pi + 1 .. close], allocator);
            pi = close + 1;
        } else if (pattern[pi] == '.') {
            chars = try allPrintableAscii(allocator);
            pi += 1;
        } else {
            chars = try allocator.alloc(u32, 1);
            chars[0] = pattern[pi];
            pi += 1;
        }

        var quant: Quantifier = .one;
        if (pi < pattern.len) {
            switch (pattern[pi]) {
                '*' => {
                    quant = .zero_or_more;
                    pi += 1;
                },
                '+' => {
                    quant = .one_or_more;
                    pi += 1;
                },
                '?' => {
                    quant = .optional;
                    pi += 1;
                },
                else => {},
            }
        }
        try atoms.append(allocator, .{ .chars = chars, .quantifier = quant });
    }

    return buildAtomsFSM(allocator, atoms.items);
}

fn buildAtomsFSM(allocator: std.mem.Allocator, atoms: []const Atom) !FiniteStateMachine {
    var state_list = std.ArrayList(State).empty;
    errdefer {
        for (state_list.items) |*s| s.deinit();
        state_list.deinit(allocator);
    }

    // Start state
    var current_state: usize = 0;
    const s0_trans = std.AutoHashMap(u32, usize).init(allocator);
    try state_list.append(allocator, .{ .transitions = s0_trans, .is_accepting = atoms.len == 0, .allowed_tokens = try allocator.alloc(u32, 0), .allocator = allocator });

    for (atoms, 0..) |atom, ai| {
        const is_last = (ai == atoms.len - 1);
        const next_state = state_list.items.len;

        switch (atom.quantifier) {
            .one => {
                const trans = std.AutoHashMap(u32, usize).init(allocator);
                try state_list.append(allocator, .{ .transitions = trans, .is_accepting = is_last, .allowed_tokens = try allocator.alloc(u32, 0), .allocator = allocator });

                for (atom.chars) |ch| {
                    try state_list.items[current_state].transitions.put(ch, next_state);
                }
                const old = state_list.items[current_state].allowed_tokens;
                state_list.items[current_state].allowed_tokens = try mergeAllowed(allocator, old, atom.chars);
                if (old.len > 0) allocator.free(old);

                current_state = next_state;
            },
            .zero_or_more => {
                if (is_last) state_list.items[current_state].is_accepting = true;
                for (atom.chars) |ch| {
                    try state_list.items[current_state].transitions.put(ch, current_state);
                }
                const old = state_list.items[current_state].allowed_tokens;
                state_list.items[current_state].allowed_tokens = try mergeAllowed(allocator, old, atom.chars);
                if (old.len > 0) allocator.free(old);
            },
            .one_or_more => {
                const trans = std.AutoHashMap(u32, usize).init(allocator);
                try state_list.append(allocator, .{ .transitions = trans, .is_accepting = is_last, .allowed_tokens = try allocator.alloc(u32, 0), .allocator = allocator });

                for (atom.chars) |ch| {
                    try state_list.items[current_state].transitions.put(ch, next_state);
                }
                const old = state_list.items[current_state].allowed_tokens;
                state_list.items[current_state].allowed_tokens = try mergeAllowed(allocator, old, atom.chars);
                if (old.len > 0) allocator.free(old);

                // Self-loop on next state
                for (atom.chars) |ch| {
                    try state_list.items[next_state].transitions.put(ch, next_state);
                }
                const old_next = state_list.items[next_state].allowed_tokens;
                state_list.items[next_state].allowed_tokens = try mergeAllowed(allocator, old_next, atom.chars);
                if (old_next.len > 0) allocator.free(old_next);

                current_state = next_state;
            },
            .optional => {
                const trans = std.AutoHashMap(u32, usize).init(allocator);
                try state_list.append(allocator, .{ .transitions = trans, .is_accepting = is_last, .allowed_tokens = try allocator.alloc(u32, 0), .allocator = allocator });

                if (is_last) state_list.items[current_state].is_accepting = true;

                for (atom.chars) |ch| {
                    try state_list.items[current_state].transitions.put(ch, next_state);
                }
                const old = state_list.items[current_state].allowed_tokens;
                state_list.items[current_state].allowed_tokens = try mergeAllowed(allocator, old, atom.chars);
                if (old.len > 0) allocator.free(old);

                current_state = next_state;
            },
        }
    }

    return .{ .states = try state_list.toOwnedSlice(allocator), .allocator = allocator, .eos_token = 0 };
}

fn buildAlternationFSM(allocator: std.mem.Allocator, parts: []const []const u8) !FiniteStateMachine {
    var state_list = std.ArrayList(State).empty;
    errdefer {
        for (state_list.items) |*s| s.deinit();
        state_list.deinit(allocator);
    }

    // State 0: start
    const s0_trans = std.AutoHashMap(u32, usize).init(allocator);
    var s0_allowed_list = std.ArrayList(u32).empty;
    defer s0_allowed_list.deinit(allocator);

    try state_list.append(allocator, .{ .transitions = s0_trans, .is_accepting = false, .allowed_tokens = &.{}, .allocator = allocator });

    for (parts) |part| {
        if (part.len == 0) {
            state_list.items[0].is_accepting = true;
            continue;
        }

        const chain_start = state_list.items.len;
        const first_char: u32 = part[0];

        if (!state_list.items[0].transitions.contains(first_char)) {
            try state_list.items[0].transitions.put(first_char, chain_start);
            try s0_allowed_list.append(allocator, first_char);
        }

        for (part, 0..) |_, ci| {
            const trans = std.AutoHashMap(u32, usize).init(allocator);
            if (ci + 1 < part.len) {
                const next_char: u32 = part[ci + 1];
                var t = trans;
                try t.put(next_char, chain_start + ci + 1);
                var allowed = try allocator.alloc(u32, 1);
                allowed[0] = next_char;
                try state_list.append(allocator, .{ .transitions = t, .is_accepting = false, .allowed_tokens = allowed, .allocator = allocator });
            } else {
                try state_list.append(allocator, .{ .transitions = trans, .is_accepting = true, .allowed_tokens = try allocator.alloc(u32, 0), .allocator = allocator });
            }
        }
    }

    state_list.items[0].allowed_tokens = try s0_allowed_list.toOwnedSlice(allocator);

    return .{ .states = try state_list.toOwnedSlice(allocator), .allocator = allocator, .eos_token = 0 };
}

fn mergeAllowed(allocator: std.mem.Allocator, existing: []const u32, new: []const u32) ![]u32 {
    var set = std.AutoHashMap(u32, void).init(allocator);
    defer set.deinit();
    for (existing) |t| try set.put(t, {});
    for (new) |t| try set.put(t, {});

    var result = try allocator.alloc(u32, set.count());
    var idx: usize = 0;
    var it = set.keyIterator();
    while (it.next()) |key| {
        result[idx] = key.*;
        idx += 1;
    }
    return result;
}

// ============================================================
// Unit Tests
// ============================================================

test "FiniteStateMachine: fromJsonSchema string enum builds correct FSM" {
    const allocator = std.testing.allocator;
    var fsm = try FiniteStateMachine.fromJsonSchema(allocator,
        \\{"type": "string", "enum": ["yes", "no"]}
    );
    defer fsm.deinit();

    try std.testing.expectEqual(@as(usize, 1), fsm.allowedTokens(0).len);
    try std.testing.expectEqual(@as(u32, '"'), fsm.allowedTokens(0)[0]);
    try std.testing.expect(!fsm.isAccepting(0));

    const s1 = fsm.transition(0, '"');
    try std.testing.expectEqual(@as(usize, 1), s1);
    try std.testing.expect(fsm.allowedTokens(s1).len >= 2);
}

test "FiniteStateMachine: fromJsonSchema string type builds open string FSM" {
    const allocator = std.testing.allocator;
    var fsm = try FiniteStateMachine.fromJsonSchema(allocator,
        \\{"type": "string"}
    );
    defer fsm.deinit();

    try std.testing.expectEqual(@as(usize, 3), fsm.states.len);
    try std.testing.expectEqual(@as(u32, '"'), fsm.allowedTokens(0)[0]);

    const s1 = fsm.transition(0, '"');
    try std.testing.expectEqual(@as(usize, 1), s1);
    try std.testing.expect(fsm.allowedTokens(s1).len > 50);
    try std.testing.expectEqual(@as(usize, 1), fsm.transition(s1, 'a'));

    const s2 = fsm.transition(s1, '"');
    try std.testing.expectEqual(@as(usize, 2), s2);
    try std.testing.expect(fsm.isAccepting(s2));
}

test "FiniteStateMachine: fromJsonSchema integer type" {
    const allocator = std.testing.allocator;
    var fsm = try FiniteStateMachine.fromJsonSchema(allocator,
        \\{"type": "integer"}
    );
    defer fsm.deinit();

    try std.testing.expectEqual(@as(usize, 11), fsm.allowedTokens(0).len);
    const s = fsm.transition(0, '5');
    try std.testing.expect(fsm.isAccepting(s));

    const s1 = fsm.transition(0, '-');
    try std.testing.expect(!fsm.isAccepting(s1));
    const s2 = fsm.transition(s1, '3');
    try std.testing.expect(fsm.isAccepting(s2));
}

test "FiniteStateMachine: fromJsonSchema boolean type" {
    const allocator = std.testing.allocator;
    var fsm = try FiniteStateMachine.fromJsonSchema(allocator,
        \\{"type": "boolean"}
    );
    defer fsm.deinit();

    var state: usize = 0;
    for ("true") |ch| state = fsm.transition(state, ch);
    try std.testing.expect(fsm.isAccepting(state));

    state = 0;
    for ("false") |ch| state = fsm.transition(state, ch);
    try std.testing.expect(fsm.isAccepting(state));
}

test "FiniteStateMachine: fromRegex literal pattern" {
    const allocator = std.testing.allocator;
    var fsm = try FiniteStateMachine.fromRegex(allocator, "abc");
    defer fsm.deinit();

    var state: usize = 0;
    for ("abc") |ch| state = fsm.transition(state, ch);
    try std.testing.expect(fsm.isAccepting(state));

    state = 0;
    for ("ab") |ch| state = fsm.transition(state, ch);
    try std.testing.expect(!fsm.isAccepting(state));
}

test "FiniteStateMachine: fromRegex character class" {
    const allocator = std.testing.allocator;
    var fsm = try FiniteStateMachine.fromRegex(allocator, "[a-c]");
    defer fsm.deinit();

    for ([_]u32{ 'a', 'b', 'c' }) |ch| {
        const state = fsm.transition(0, ch);
        try std.testing.expect(fsm.isAccepting(state));
    }
    try std.testing.expectEqual(@as(usize, 0), fsm.transition(0, 'd'));
}

test "FiniteStateMachine: fromRegex star quantifier" {
    const allocator = std.testing.allocator;
    var fsm = try FiniteStateMachine.fromRegex(allocator, "a*");
    defer fsm.deinit();

    try std.testing.expect(fsm.isAccepting(0));
    var state = fsm.transition(0, 'a');
    try std.testing.expect(fsm.isAccepting(state));
    state = fsm.transition(state, 'a');
    try std.testing.expect(fsm.isAccepting(state));
}

test "FiniteStateMachine: fromRegex plus quantifier" {
    const allocator = std.testing.allocator;
    var fsm = try FiniteStateMachine.fromRegex(allocator, "[0-9]+");
    defer fsm.deinit();

    try std.testing.expect(!fsm.isAccepting(0));
    var state = fsm.transition(0, '5');
    try std.testing.expect(fsm.isAccepting(state));
    state = fsm.transition(state, '3');
    try std.testing.expect(fsm.isAccepting(state));
}

test "FiniteStateMachine: fromRegex alternation" {
    const allocator = std.testing.allocator;
    var fsm = try FiniteStateMachine.fromRegex(allocator, "cat|dog");
    defer fsm.deinit();

    var state: usize = 0;
    for ("cat") |ch| state = fsm.transition(state, ch);
    try std.testing.expect(fsm.isAccepting(state));

    state = 0;
    for ("dog") |ch| state = fsm.transition(state, ch);
    try std.testing.expect(fsm.isAccepting(state));
}

test "GuidedDecoder: init and advance" {
    const allocator = std.testing.allocator;
    const fsm = try FiniteStateMachine.fromRegex(allocator, "ab");
    var decoder = GuidedDecoder.init(fsm);
    defer decoder.deinit();

    try std.testing.expectEqual(@as(usize, 0), decoder.current_state);
    try std.testing.expect(!decoder.isComplete());

    decoder.advance('a');
    try std.testing.expect(!decoder.isComplete());

    decoder.advance('b');
    try std.testing.expect(decoder.isComplete());
}

test "GuidedDecoder: reset returns to initial state" {
    const allocator = std.testing.allocator;
    const fsm = try FiniteStateMachine.fromRegex(allocator, "ab");
    var decoder = GuidedDecoder.init(fsm);
    defer decoder.deinit();

    decoder.advance('a');
    decoder.advance('b');
    try std.testing.expect(decoder.isComplete());

    decoder.reset();
    try std.testing.expectEqual(@as(usize, 0), decoder.current_state);
    try std.testing.expect(!decoder.isComplete());
}

test "GuidedDecoder: maskLogits sets disallowed tokens to -inf" {
    const c_mod = @import("mlx").c;
    c_mod.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Build a simple FSM: only tokens 2 and 5 are allowed in state 0
    var state_list = try allocator.alloc(State, 1);
    var trans = std.AutoHashMap(u32, usize).init(allocator);
    try trans.put(2, 0);
    try trans.put(5, 0);
    var allowed = try allocator.alloc(u32, 2);
    allowed[0] = 2;
    allowed[1] = 5;
    state_list[0] = .{ .transitions = trans, .is_accepting = true, .allowed_tokens = allowed, .allocator = allocator };

    const fsm = FiniteStateMachine{ .states = state_list, .allocator = allocator, .eos_token = 0 };
    var decoder = GuidedDecoder.init(fsm);
    defer decoder.deinit();

    const logits_data = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    const logits = try Array.fromData(allocator, f32, &logits_data, &[_]i32{8});
    defer logits.deinit();

    const masked = try decoder.maskLogits(logits, ctx);
    defer masked.deinit();

    try masked.eval();
    const data = try masked.dataSlice(f32);

    for (data, 0..) |val, idx| {
        if (idx == 2 or idx == 5) {
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), val, 1e-6);
        } else {
            try std.testing.expect(std.math.isNegativeInf(val));
        }
    }
}

test "FiniteStateMachine: transition with unknown token stays in current state" {
    const allocator = std.testing.allocator;
    var fsm = try FiniteStateMachine.fromRegex(allocator, "abc");
    defer fsm.deinit();

    try std.testing.expectEqual(@as(usize, 0), fsm.transition(0, 'z'));
}

test "FiniteStateMachine: out of bounds state returns empty allowed tokens" {
    const allocator = std.testing.allocator;
    var fsm = try FiniteStateMachine.fromRegex(allocator, "a");
    defer fsm.deinit();

    try std.testing.expectEqual(@as(usize, 0), fsm.allowedTokens(999).len);
}

// ============================================================
// Property 16: Guided Decoding Constraint Satisfaction
//
// Feature: production-deployment, Property 16: Guided Decoding
// Constraint Satisfaction
//
// For any grammar constraint (JSON schema or regex), the logits
// mask applied at each generation step SHALL set all disallowed
// token logits to negative infinity. The resulting generated
// token sequence SHALL satisfy the constraint.
//
// **Validates: Requirements R17.1, R17.2, R17.3**
// ============================================================

test "Property 16: Guided Decoding Constraint Satisfaction — maskLogits sets disallowed tokens to -inf and preserves allowed (100 iterations)" {
    const c_mod = @import("mlx").c;
    c_mod.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var prng = std.Random.DefaultPrng.init(0xD0DE_0016);
    const rand = prng.random();

    // Test patterns that produce FSMs with known allowed token sets
    const regex_patterns = [_][]const u8{
        "abc",
        "[a-z]+",
        "[0-9]+",
        "true|false",
        "a*",
        "[a-c]",
        "hello",
        "[A-Z][a-z]*",
    };

    const json_schemas = [_][]const u8{
        \\{"type": "boolean"}
        ,
        \\{"type": "integer"}
        ,
        \\{"type": "string"}
        ,
        \\{"type": "string", "enum": ["yes", "no"]}
        ,
    };

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Randomly pick regex or JSON schema FSM
        const use_regex = rand.boolean();
        var fsm: FiniteStateMachine = undefined;

        if (use_regex) {
            const pattern_idx = rand.intRangeAtMost(usize, 0, regex_patterns.len - 1);
            fsm = try FiniteStateMachine.fromRegex(allocator, regex_patterns[pattern_idx]);
        } else {
            const schema_idx = rand.intRangeAtMost(usize, 0, json_schemas.len - 1);
            fsm = try FiniteStateMachine.fromJsonSchema(allocator, json_schemas[schema_idx]);
        }

        var decoder = GuidedDecoder.init(fsm);
        defer decoder.deinit();

        // Random vocab size between 32 and 256
        const vocab_size: usize = rand.intRangeAtMost(usize, 32, 256);

        // Generate random logits data
        var logits_data = try allocator.alloc(f32, vocab_size);
        defer allocator.free(logits_data);
        for (0..vocab_size) |i| {
            logits_data[i] = @as(f32, @floatFromInt(rand.intRangeAtMost(i32, -1000, 1000))) / 100.0;
        }

        const logits = try Array.fromData(allocator, f32, logits_data, &[_]i32{@intCast(vocab_size)});
        defer logits.deinit();

        // Apply mask
        const masked = try decoder.maskLogits(logits, ctx);
        defer masked.deinit();

        try masked.eval();
        const masked_data = try masked.dataSlice(f32);

        // Get allowed tokens for current state
        const allowed = decoder.fsm.allowedTokens(decoder.current_state);

        // Build a set of allowed token IDs for fast lookup
        var allowed_set = std.AutoHashMap(u32, void).init(allocator);
        defer allowed_set.deinit();
        for (allowed) |tok| {
            if (tok < vocab_size) {
                try allowed_set.put(tok, {});
            }
        }

        // Property 16a: All disallowed tokens must be -inf
        // Property 16b: All allowed tokens must retain original logit values
        for (0..vocab_size) |idx| {
            if (allowed_set.contains(@intCast(idx))) {
                // Allowed token: original value preserved
                try std.testing.expectApproxEqAbs(logits_data[idx], masked_data[idx], 1e-6);
            } else {
                // Disallowed token: must be -inf
                try std.testing.expect(std.math.isNegativeInf(masked_data[idx]));
            }
        }
    }
}

test "Property 16: Guided Decoding Constraint Satisfaction — generated token sequence satisfies FSM constraint (100 iterations)" {
    const c_mod = @import("mlx").c;
    c_mod.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var prng = std.Random.DefaultPrng.init(0xD0DE_1600);
    const rand = prng.random();

    // Patterns with known valid sequences we can verify
    const TestCase = struct {
        pattern: []const u8,
        is_regex: bool,
    };

    const test_cases = [_]TestCase{
        .{ .pattern = "abc", .is_regex = true },
        .{ .pattern = "[0-9]+", .is_regex = true },
        .{ .pattern = "true|false", .is_regex = true },
        .{ .pattern = "[a-z]+", .is_regex = true },
        .{ .pattern = "[a-c]", .is_regex = true },
        .{
            .pattern =
            \\{"type": "boolean"}
            ,
            .is_regex = false,
        },
        .{
            .pattern =
            \\{"type": "integer"}
            ,
            .is_regex = false,
        },
        .{
            .pattern =
            \\{"type": "string", "enum": ["yes", "no"]}
            ,
            .is_regex = false,
        },
    };

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Pick a random test case
        const tc_idx = rand.intRangeAtMost(usize, 0, test_cases.len - 1);
        const tc = test_cases[tc_idx];

        var fsm: FiniteStateMachine = undefined;
        if (tc.is_regex) {
            fsm = try FiniteStateMachine.fromRegex(allocator, tc.pattern);
        } else {
            fsm = try FiniteStateMachine.fromJsonSchema(allocator, tc.pattern);
        }

        var decoder = GuidedDecoder.init(fsm);
        defer decoder.deinit();

        // Simulate a generation sequence: at each step, mask logits then
        // pick a random allowed token (simulating sampling from masked logits).
        const vocab_size: usize = 256;
        var generated_tokens = std.ArrayList(u32).empty;
        defer generated_tokens.deinit(allocator);

        const max_steps: usize = 20;
        var step: usize = 0;
        while (step < max_steps) : (step += 1) {
            const allowed = decoder.fsm.allowedTokens(decoder.current_state);
            if (allowed.len == 0) break; // No valid transitions — stop

            // Create random logits
            var logits_data = try allocator.alloc(f32, vocab_size);
            defer allocator.free(logits_data);
            for (0..vocab_size) |i| {
                logits_data[i] = @as(f32, @floatFromInt(rand.intRangeAtMost(i32, -500, 500))) / 100.0;
            }

            const logits = try Array.fromData(allocator, f32, logits_data, &[_]i32{@intCast(vocab_size)});
            defer logits.deinit();

            const masked = try decoder.maskLogits(logits, ctx);
            defer masked.deinit();
            try masked.eval();
            const masked_data = try masked.dataSlice(f32);

            // Verify masking is correct at this step
            for (0..vocab_size) |idx| {
                var is_allowed = false;
                for (allowed) |tok| {
                    if (tok == @as(u32, @intCast(idx))) {
                        is_allowed = true;
                        break;
                    }
                }
                if (!is_allowed) {
                    try std.testing.expect(std.math.isNegativeInf(masked_data[idx]));
                }
            }

            // Pick a random allowed token (simulating argmax on masked logits)
            const pick_idx = rand.intRangeAtMost(usize, 0, allowed.len - 1);
            const chosen_token = allowed[pick_idx];

            try generated_tokens.append(allocator, chosen_token);
            decoder.advance(chosen_token);

            // If we reached an accepting state, we can stop
            if (decoder.isComplete()) break;
        }

        // Property 16c: The generated token sequence must be valid
        // according to the FSM — replaying the tokens through a fresh
        // FSM traversal must follow valid transitions and (if we stopped
        // at an accepting state) end in an accepting state.
        var verify_state: usize = 0;
        for (generated_tokens.items) |token| {
            // Token must be in the allowed set for the current state
            const verify_allowed = decoder.fsm.allowedTokens(verify_state);
            var token_is_allowed = false;
            for (verify_allowed) |a| {
                if (a == token) {
                    token_is_allowed = true;
                    break;
                }
            }
            try std.testing.expect(token_is_allowed);

            verify_state = decoder.fsm.transition(verify_state, token);
        }

        // If we stopped because we reached an accepting state, verify it
        if (decoder.isComplete()) {
            try std.testing.expect(decoder.fsm.isAccepting(verify_state));
        }
    }
}
