//! TF_DSV41_PROFILE's exchanges on one rank: each prefill all-gather's arrival to completion, by segment and prompt.

const std = @import("std");
const calls = @import("block_calls.zig");

/// How the profile counts a call: a timed exchange, an exchange it cannot time alone, or neither.
pub const Kind = enum { none, timed, untimed };

/// Timed: a plain `glue.exchange` on the main stream, nothing waited first: its two events bracket the all-gather.
pub fn kindOf(c: *const calls.Call) Kind {
    if (!calls.isExchange(c.*)) return .none;
    const plain = std.mem.eql(u8, c.name, "glue.exchange") and !c.side and !c.fork and !c.join and c.wait == 0 and !c.defer_side and !c.defer_join;
    return if (plain) .timed else .untimed;
}

/// The bytes an exchange sends from each rank: its first argument's elements, gathered as bf16.
pub fn bytesOf(c: *const calls.Call) u64 {
    if (c.args.len == 0 or c.args[0].arg != .t) return 0;
    var n: u64 = 2;
    for (c.args[0].arg.t.shape) |d| n *= @intCast(@max(d, 0));
    return n;
}

/// Timed exchanges a rank keeps (32 bytes each); segments that start past it are counted only, on every rank alike.
pub const max_exchanges: usize = 1 << 20;

/// A profiled prefill window: its prompt, its rows, the bytes its timed exchanges [first, first + count) send.
pub const Segment = struct { prompt: u32, first: u32, count: u32 = 0, untimed: u32 = 0, rows: u64 = 0, tail: bool = false, bytes: u64 = 0 };

/// One timed exchange, from this rank's events.
pub const Exchange = struct {
    /// this rank's arrival (its start event) to its completion (its end event)
    in_ns: u64,
    /// the previous exchange's completion, or the segment's first call, to this arrival
    gap_ns: u64,
    /// the part of the gap between one call's end and the next call's start: the GPU waiting for the host
    idle_ns: u64,
    bytes: u64,
    /// its layer's place in the window (the calls that begin a layer counted) and its order in that layer
    layer: u32,
    site: u32,
};

pub const Rank = struct {
    rank: u32 = 0,
    world: u32 = 1,
    in_ns: std.ArrayList(u64) = .empty,
    gap_ns: std.ArrayList(u64) = .empty,
    idle_ns: std.ArrayList(u64) = .empty,
    layer: std.ArrayList(u32) = .empty,
    site: std.ArrayList(u32) = .empty,
    segments: std.ArrayList(Segment) = .empty,
    /// each segment: host time from the previous segment's end to its start, in one prompt (a prompt's first: 0)
    boundary_ns: std.ArrayList(u64) = .empty,
    prompts: u32 = 0,
    /// prompt() was called: the next segment opens a new prompt
    fresh: bool = true,
    /// the segment the last segment() opened (null: past max_exchanges, counted only)
    last: ?usize = null,
    dropped_segments: u64 = 0,

    pub fn deinit(x: *Rank, gpa: std.mem.Allocator) void {
        x.in_ns.deinit(gpa);
        x.gap_ns.deinit(gpa);
        x.idle_ns.deinit(gpa);
        x.layer.deinit(gpa);
        x.site.deinit(gpa);
        x.segments.deinit(gpa);
        x.boundary_ns.deinit(gpa);
    }

    /// A prompt's prefill starts (one prefill call on every rank): its segments follow.
    pub fn prompt(x: *Rank) void {
        x.fresh = true;
    }

    /// A profiled prefill window ended: its segment, which exchange() and skip() then fill.
    pub fn segment(x: *Rank, gpa: std.mem.Allocator, boundary_ns: u64) !void {
        if (x.fresh) x.prompts += 1;
        x.fresh = false;
        x.last = null;
        if (x.in_ns.items.len >= max_exchanges) {
            x.dropped_segments += 1;
            return;
        }
        try x.boundary_ns.ensureUnusedCapacity(gpa, 1);
        try x.segments.append(gpa, .{ .prompt = x.prompts - 1, .first = @intCast(x.in_ns.items.len) });
        x.boundary_ns.appendAssumeCapacity(boundary_ns);
        x.last = x.segments.items.len - 1;
    }

    /// A timed exchange of the open segment.
    pub fn exchange(x: *Rank, gpa: std.mem.Allocator, e: Exchange) !void {
        const i = x.last orelse return;
        try x.in_ns.ensureUnusedCapacity(gpa, 1);
        try x.gap_ns.ensureUnusedCapacity(gpa, 1);
        try x.idle_ns.ensureUnusedCapacity(gpa, 1);
        try x.layer.ensureUnusedCapacity(gpa, 1);
        try x.site.ensureUnusedCapacity(gpa, 1);
        x.in_ns.appendAssumeCapacity(e.in_ns);
        x.gap_ns.appendAssumeCapacity(e.gap_ns);
        x.idle_ns.appendAssumeCapacity(e.idle_ns);
        x.layer.appendAssumeCapacity(e.layer);
        x.site.appendAssumeCapacity(e.site);
        x.segments.items[i].count += 1;
        x.segments.items[i].bytes += e.bytes;
    }

    /// An exchange of the open segment the profile does not time (Kind.untimed).
    pub fn skip(x: *Rank) void {
        const i = x.last orelse return;
        x.segments.items[i].untimed += 1;
    }

    /// The rows the last segment covered (`tail`: a CED decoder replay's).
    pub fn rows(x: *Rank, n: u64, tail: bool) void {
        const i = x.last orelse return;
        x.segments.items[i].rows += n;
        x.segments.items[i].tail = tail;
    }

    /// Segments [lo, hi) summed.
    fn total(x: *const Rank, lo: usize, hi: usize) Total {
        var t: Total = .{};
        for (x.segments.items[lo..hi], x.boundary_ns.items[lo..hi]) |s, b| {
            t.exchanges += s.count;
            t.untimed += s.untimed;
            t.rows += s.rows;
            t.bytes += s.bytes;
            t.boundary_ns += b;
            for (s.first..s.first + s.count) |k| {
                t.in_ns += x.in_ns.items[k];
                t.gap_ns += x.gap_ns.items[k];
                t.idle_ns += x.idle_ns.items[k];
            }
        }
        return t;
    }

    /// The profile's "exchanges" object: totals, each prompt, each segment, then every timed exchange.
    pub fn write(x: *const Rank, w: *std.Io.Writer) !void {
        const n = x.segments.items.len;
        const t = x.total(0, n);
        try w.print("{{\"rank\": {d}, \"world\": {d}, \"timed\": {d}, \"untimed\": {d}, \"dropped_segments\": {d}, ", .{ x.rank, x.world, t.exchanges, t.untimed, x.dropped_segments });
        try t.write(w);
        try w.writeAll(", \"prompts\": [");
        var i: usize = 0;
        while (i < n) {
            const p = x.segments.items[i].prompt;
            var j = i;
            while (j < n and x.segments.items[j].prompt == p) j += 1;
            const t_p = x.total(i, j);
            try w.print("{s}{{\"prompt\": {d}, \"segments\": {d}, \"exchanges\": {d}, ", .{ if (i == 0) "" else ", ", p, j - i, t_p.exchanges });
            try t_p.write(w);
            try w.writeAll("}");
            i = j;
        }
        try w.writeAll("], \"segments\": [");
        for (x.segments.items, 0..) |s, k| {
            try w.print("{s}{{\"prompt\": {d}, \"tail\": {}, \"exchanges\": {d}, \"untimed\": {d}, ", .{ if (k == 0) "" else ", ", s.prompt, s.tail, s.count, s.untimed });
            try x.total(k, k + 1).write(w);
            try w.writeAll("}");
        }
        try w.writeAll("], \"in_ns\": ");
        try list(u64, w, x.in_ns.items);
        try w.writeAll(", \"gap_ns\": ");
        try list(u64, w, x.gap_ns.items);
        try w.writeAll(", \"idle_ns\": ");
        try list(u64, w, x.idle_ns.items);
        try w.writeAll(", \"layer\": ");
        try list(u32, w, x.layer.items);
        try w.writeAll(", \"site\": ");
        try list(u32, w, x.site.items);
        try w.writeAll(", \"boundary_ns\": ");
        try list(u64, w, x.boundary_ns.items);
        try w.writeAll("}");
    }
};

/// Some segments' exchanges summed; the kernel time is the gap less its idle.
const Total = struct {
    exchanges: u64 = 0,
    untimed: u64 = 0,
    rows: u64 = 0,
    bytes: u64 = 0,
    in_ns: u64 = 0,
    gap_ns: u64 = 0,
    idle_ns: u64 = 0,
    boundary_ns: u64 = 0,

    fn write(t: Total, w: *std.Io.Writer) !void {
        try w.print("\"rows\": {d}, \"bytes\": {d}, \"in_ms\": {d:.3}, \"gap_ms\": {d:.3}, \"kernel_ms\": {d:.3}, \"idle_ms\": {d:.3}, \"boundary_ms\": {d:.3}", .{ t.rows, t.bytes, ms(t.in_ns), ms(t.gap_ns), ms(t.gap_ns -| t.idle_ns), ms(t.idle_ns), ms(t.boundary_ns) });
    }
};

pub fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn list(comptime T: type, w: *std.Io.Writer, v: []const T) !void {
    try w.writeAll("[");
    for (v, 0..) |y, i| try w.print("{s}{d}", .{ if (i == 0) "" else ", ", y });
    try w.writeAll("]");
}
