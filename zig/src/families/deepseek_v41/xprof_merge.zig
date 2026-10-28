//! `tf-dsv41 exchanges`: every rank's profiled exchanges side by side: who arrived last, how long the others waited.

const std = @import("std");
const xprof = @import("xprof.zig");
const Segment = xprof.Segment;

/// One prefill call name's totals on one rank (the profile's "prefill"."calls").
pub const Named = struct { name: []const u8, calls: u64, ms: f64 };

/// One rank's "exchanges" object (xprof.Rank.write) and its prefill calls, read back.
pub const View = struct {
    rank: u32,
    world: u32,
    in_ns: []const u64,
    gap_ns: []const u64,
    idle_ns: []const u64,
    layer: []const u32,
    site: []const u32,
    segments: []const Segment,
    boundary_ns: []const u64,
    calls: []const Named = &.{},
};

/// One rank's sums: its late count (a tie shared), wait, in-time, gap, the gap's idle and the segments' boundaries.
pub const Per = struct {
    late: f64 = 0,
    wait_ns: u64 = 0,
    in_ns: u64 = 0,
    gap_ns: u64 = 0,
    idle_ns: u64 = 0,
    boundary_ns: u64 = 0,

    /// The gap's GPU time in calls: the gap less its idle.
    pub fn kernel(x: Per) u64 {
        return x.gap_ns -| x.idle_ns;
    }
};

/// Exchanges summed: each rank's Per and the transfers, with the rows, bytes and place they cover.
pub const Sums = struct {
    exchanges: u64 = 0,
    transfer_ns: u64 = 0,
    ranks: []Per,
    rows: u64 = 0,
    bytes: u64 = 0,
    segments: u64 = 0,
    prompt: u32 = 0,
    tail: bool = false,
    layer: u32 = 0,
    site: u32 = 0,

    fn init(a: std.mem.Allocator, world: usize) !Sums {
        const r = try a.alloc(Per, world);
        @memset(r, .{});
        return .{ .ranks = r };
    }

    fn add(s: *Sums, o: *const Sums) void {
        s.exchanges += o.exchanges;
        s.transfer_ns += o.transfer_ns;
        s.rows += o.rows;
        s.bytes += o.bytes;
        s.segments += o.segments;
        for (s.ranks, o.ranks) |*x, y| {
            x.late += y.late;
            x.wait_ns += y.wait_ns;
            x.in_ns += y.in_ns;
            x.gap_ns += y.gap_ns;
            x.idle_ns += y.idle_ns;
            x.boundary_ns += y.boundary_ns;
        }
    }

    /// Exchange k: the last rank (the smallest in-time; a tie shared), the transfer (its in-time) and each wait.
    fn take(s: *Sums, views: []const View, k: usize, t: u64, tied: f64) void {
        s.exchanges += 1;
        s.transfer_ns += t;
        for (views, s.ranks) |v, *x| {
            if (v.in_ns[k] == t) x.late += 1 / tied;
            x.wait_ns += v.in_ns[k] - t;
            x.in_ns += v.in_ns[k];
            x.gap_ns += v.gap_ns[k];
            x.idle_ns += v.idle_ns[k];
        }
    }

    /// Rank r's share of these exchanges where it arrived last.
    pub fn share(s: *const Sums, r: usize) f64 {
        return if (s.exchanges == 0) 0 else s.ranks[r].late / @as(f64, @floatFromInt(s.exchanges));
    }

    /// The bytes each rank received (its peers' shards) over the transfer time: bytes per ns, so GB/s.
    pub fn gbytesPerS(s: *const Sums, world: u32) f64 {
        if (s.transfer_ns == 0) return 0;
        return @as(f64, @floatFromInt(s.bytes * (world - 1))) / @as(f64, @floatFromInt(s.transfer_ns));
    }
};

/// A prefill call name's GPU time on every rank, and the spread between the slowest and fastest rank.
pub const CallRow = struct { name: []const u8, calls: []u64, ms: []f64, spread_ms: f64 = 0 };

/// Call names the report keeps, the widest spread first.
pub const top_calls: usize = 12;

pub const Report = struct {
    world: u32,
    total: Sums,
    prompts: []Sums,
    segments: []Sums,
    /// each segment's first exchange, the one after the profile's per-window sync, and every other exchange
    starts: Sums,
    rest: Sums,
    /// full (not tail) segments' exchanges by layer and site
    layers: []Sums,
    calls: []CallRow,
    /// each exchange: the rank that arrived last (the lowest of a tie), and the transfer (that rank's in-time)
    last: []u32,
    transfer_ns: []u64,
    ties: u64 = 0,
    /// each rank's wait at the median and 90th percentile exchange
    wait_p50_ns: []u64,
    wait_p90_ns: []u64,
};

/// The ranks' exchanges matched in order (every rank runs the same prefill programs), refused where they differ.
pub fn merge(a: std.mem.Allocator, views: []const View) !Report {
    const W = views.len;
    if (W == 0) return error.NoRanks;
    const n = views[0].in_ns.len;
    for (views, 0..) |v, i| {
        if (v.rank != i or v.world != W) return error.RanksNotOneWorld;
        if (v.in_ns.len != n or v.gap_ns.len != n or v.idle_ns.len != n) return error.ExchangesDiffer;
        if (!std.mem.eql(u32, v.layer, views[0].layer) or !std.mem.eql(u32, v.site, views[0].site) or v.layer.len != n) return error.ExchangesDiffer;
        if (v.segments.len != views[0].segments.len or v.boundary_ns.len != v.segments.len) return error.SegmentsDiffer;
        for (v.segments, views[0].segments) |s, z| if (!std.meta.eql(s, z)) return error.SegmentsDiffer;
    }
    var rep: Report = .{ .world = @intCast(W), .total = try Sums.init(a, W), .prompts = &.{}, .segments = try a.alloc(Sums, views[0].segments.len), .starts = try Sums.init(a, W), .rest = try Sums.init(a, W), .layers = &.{}, .calls = try calls(a, views), .last = try a.alloc(u32, n), .transfer_ns = try a.alloc(u64, n), .wait_p50_ns = try a.alloc(u64, W), .wait_p90_ns = try a.alloc(u64, W) };
    const waits = try a.alloc([]u64, W);
    for (waits) |*x| x.* = try a.alloc(u64, n);
    var prompts: std.ArrayList(Sums) = .empty;
    var layers: std.AutoArrayHashMapUnmanaged([2]u32, Sums) = .empty;
    for (views[0].segments, rep.segments, 0..) |sg, *ss, si| {
        ss.* = try Sums.init(a, W);
        ss.rows = sg.rows;
        ss.bytes = sg.bytes;
        ss.segments = 1;
        ss.prompt = sg.prompt;
        ss.tail = sg.tail;
        for (views, ss.ranks) |v, *x| x.boundary_ns = v.boundary_ns[si];
        if (sg.first != rep.total.exchanges or sg.first + sg.count > n) return error.ExchangesDiffer;
        for (sg.first..sg.first + sg.count) |k| {
            var t: u64 = std.math.maxInt(u64);
            for (views) |v| t = @min(t, v.in_ns[k]);
            var tied: f64 = 0;
            for (views) |v| tied += if (v.in_ns[k] == t) 1 else 0;
            rep.last[k] = for (views, 0..) |v, r| {
                if (v.in_ns[k] == t) break @intCast(r);
            } else unreachable;
            if (tied > 1) rep.ties += 1;
            rep.transfer_ns[k] = t;
            for (views, waits) |v, wr| wr[k] = v.in_ns[k] - t;
            ss.take(views, k, t, tied);
            (if (k == sg.first) &rep.starts else &rep.rest).take(views, k, t, tied);
            if (sg.tail) continue;
            const g = try layers.getOrPut(a, .{ views[0].layer[k], views[0].site[k] });
            if (!g.found_existing) {
                g.value_ptr.* = try Sums.init(a, W);
                g.value_ptr.layer = views[0].layer[k];
                g.value_ptr.site = views[0].site[k];
            }
            g.value_ptr.take(views, k, t, tied);
        }
        if (prompts.items.len == 0 or prompts.items[prompts.items.len - 1].prompt != sg.prompt) {
            var p = try Sums.init(a, W);
            p.prompt = sg.prompt;
            try prompts.append(a, p);
        }
        prompts.items[prompts.items.len - 1].add(ss);
        rep.total.add(ss);
    }
    if (rep.total.exchanges != n) return error.ExchangesDiffer;
    rep.prompts = try prompts.toOwnedSlice(a);
    rep.layers = layers.values();
    for (waits, 0..) |x, r| {
        std.mem.sort(u64, x, {}, std.sort.asc(u64));
        rep.wait_p50_ns[r] = if (n == 0) 0 else x[n / 2];
        rep.wait_p90_ns[r] = if (n == 0) 0 else x[@min(n - 1, n * 9 / 10)];
    }
    return rep;
}

/// Every prefill call name on any rank, its GPU time on each, the widest spread first, top_calls of them.
fn calls(a: std.mem.Allocator, views: []const View) ![]CallRow {
    var rows: std.StringArrayHashMapUnmanaged(CallRow) = .empty;
    for (views, 0..) |v, r| {
        for (v.calls) |c| {
            const g = try rows.getOrPut(a, c.name);
            if (!g.found_existing) {
                g.value_ptr.* = .{ .name = c.name, .calls = try a.alloc(u64, views.len), .ms = try a.alloc(f64, views.len) };
                @memset(g.value_ptr.calls, 0);
                @memset(g.value_ptr.ms, 0);
            }
            g.value_ptr.calls[r] = c.calls;
            g.value_ptr.ms[r] = c.ms;
        }
    }
    const out = rows.values();
    for (out) |*c| c.spread_ms = std.mem.max(f64, c.ms) - std.mem.min(f64, c.ms);
    std.mem.sort(CallRow, out, {}, struct {
        fn gt(_: void, x: CallRow, y: CallRow) bool {
            return x.spread_ms > y.spread_ms;
        }
    }.gt);
    return out[0..@min(out.len, top_calls)];
}

/// The report: ranks, segment starts apart, the calls that differ most, then prompts, segments, layers, exchanges.
pub fn write(rep: *const Report, w: *std.Io.Writer) !void {
    const t = &rep.total;
    try w.print("{{\"world\": {d}, \"prompts\": {d}, \"segments\": {d}, \"exchanges\": {d}, \"ties\": {d}, \"transfer_ms\": {d:.3}, \"ranks\": [", .{ rep.world, rep.prompts.len, rep.segments.len, t.exchanges, rep.ties, xprof.ms(t.transfer_ns) });
    for (t.ranks, 0..) |x, r| {
        try w.print("{s}{{\"rank\": {d}, \"late\": {d:.4}, \"wait_ms\": {d:.3}, \"wait_p50_us\": {d:.1}, \"wait_p90_us\": {d:.1}, \"in_ms\": {d:.3}, \"gap_ms\": {d:.3}, \"kernel_ms\": {d:.3}, \"idle_ms\": {d:.3}, \"boundary_ms\": {d:.3}}}", .{ if (r == 0) "" else ", ", r, t.share(r), xprof.ms(x.wait_ns), us(rep.wait_p50_ns[r]), us(rep.wait_p90_ns[r]), xprof.ms(x.in_ns), xprof.ms(x.gap_ns), xprof.ms(x.kernel()), xprof.ms(x.idle_ns), xprof.ms(x.boundary_ns) });
    }
    try w.writeAll("], \"segment_starts\": {");
    try part(w, &rep.starts);
    try w.writeAll("}, \"after_starts\": {");
    try part(w, &rep.rest);
    try w.writeAll("}, \"calls\": [");
    for (rep.calls, 0..) |c, i| {
        try w.print("{s}{{\"name\": \"{s}\", \"spread_ms\": {d:.3}, \"ms\": [", .{ if (i == 0) "" else ", ", c.name, c.spread_ms });
        for (c.ms, 0..) |x, r| try w.print("{s}{d:.3}", .{ if (r == 0) "" else ", ", x });
        try w.writeAll("], \"calls\": [");
        for (c.calls, 0..) |x, r| try w.print("{s}{d}", .{ if (r == 0) "" else ", ", x });
        try w.writeAll("]}");
    }
    try w.writeAll("], \"by_prompt\": [");
    for (rep.prompts, 0..) |*p, i| {
        try w.print("{s}{{\"prompt\": {d}, \"segments\": {d}, ", .{ if (i == 0) "" else ", ", p.prompt, p.segments });
        try sized(w, p, rep.world);
    }
    try w.writeAll("], \"by_segment\": [");
    for (rep.segments, 0..) |*s, i| {
        try w.print("{s}{{\"prompt\": {d}, \"tail\": {}, ", .{ if (i == 0) "" else ", ", s.prompt, s.tail });
        try sized(w, s, rep.world);
    }
    try w.writeAll("], \"by_layer\": [");
    for (rep.layers, 0..) |*s, i| {
        try w.print("{s}{{\"layer\": {d}, \"site\": {d}, ", .{ if (i == 0) "" else ", ", s.layer, s.site });
        try part(w, s);
        try w.writeAll("}");
    }
    try w.writeAll("], \"last\": [");
    for (rep.last, 0..) |r, i| try w.print("{s}{d}", .{ if (i == 0) "" else ", ", r });
    try w.writeAll("], \"transfer_ns\": [");
    for (rep.transfer_ns, 0..) |x, i| try w.print("{s}{d}", .{ if (i == 0) "" else ", ", x });
    try w.writeAll("]}\n");
}

/// A prompt or segment: its rows, the bytes each rank sent and the receive rate, its exchanges, its boundaries.
fn sized(w: *std.Io.Writer, s: *const Sums, world: u32) !void {
    try w.print("\"rows\": {d}, \"bytes\": {d}, \"recv_gbytes_s\": {d:.2}, ", .{ s.rows, s.bytes, s.gbytesPerS(world) });
    try part(w, s);
    try w.writeAll(", \"boundary_ms\": [");
    for (s.ranks, 0..) |x, r| try w.print("{s}{d:.3}", .{ if (r == 0) "" else ", ", xprof.ms(x.boundary_ns) });
    try w.writeAll("]}");
}

fn part(w: *std.Io.Writer, s: *const Sums) !void {
    try w.print("\"exchanges\": {d}, \"transfer_ms\": {d:.3}, \"late\": [", .{ s.exchanges, xprof.ms(s.transfer_ns) });
    for (0..s.ranks.len) |r| try w.print("{s}{d:.4}", .{ if (r == 0) "" else ", ", s.share(r) });
    try ranks(w, "wait_ms", s, .wait_ns);
    try ranks(w, "kernel_ms", s, null);
    try ranks(w, "idle_ms", s, .idle_ns);
    try w.writeAll("]");
}

/// `], "<name>": [` and each rank's field in ms (null: its kernel time).
fn ranks(w: *std.Io.Writer, name: []const u8, s: *const Sums, comptime f: ?std.meta.FieldEnum(Per)) !void {
    try w.print("], \"{s}\": [", .{name});
    for (s.ranks, 0..) |x, r| try w.print("{s}{d:.3}", .{ if (r == 0) "" else ", ", xprof.ms(if (f) |g| @field(x, @tagName(g)) else x.kernel()) });
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3;
}

/// A profile file's "prefill" exchanges and calls (TF_DSV41_PROFILE's JSON) as a View; `a` owns the slices.
pub fn viewOf(a: std.mem.Allocator, root: std.json.Value) !View {
    const pf = try field(root, "prefill");
    const x = try field(pf, "exchanges");
    const segs = (try field(x, "segments")).array.items;
    const out = try a.alloc(Segment, segs.len);
    var first: u32 = 0;
    for (segs, out) |s, *o| {
        o.* = .{ .prompt = try int(u32, s, "prompt"), .first = first, .count = try int(u32, s, "exchanges"), .untimed = try int(u32, s, "untimed"), .rows = try int(u64, s, "rows"), .tail = try boolean(s, "tail"), .bytes = try int(u64, s, "bytes") };
        first += o.count;
    }
    const cs = try field(pf, "calls");
    if (cs != .object) return error.NotAProfile;
    const named = try a.alloc(Named, cs.object.count());
    for (cs.object.keys(), cs.object.values(), named) |k, v, *o| o.* = .{ .name = k, .calls = try int(u64, v, "calls"), .ms = try float(v, "ms") };
    return .{ .rank = try int(u32, x, "rank"), .world = try int(u32, x, "world"), .in_ns = try ints(u64, a, try field(x, "in_ns")), .gap_ns = try ints(u64, a, try field(x, "gap_ns")), .idle_ns = try ints(u64, a, try field(x, "idle_ns")), .layer = try ints(u32, a, try field(x, "layer")), .site = try ints(u32, a, try field(x, "site")), .segments = out, .boundary_ns = try ints(u64, a, try field(x, "boundary_ns")), .calls = named };
}

fn field(v: std.json.Value, name: []const u8) !std.json.Value {
    if (v != .object) return error.NotAProfile;
    return v.object.get(name) orelse error.NotAProfile;
}

fn int(comptime T: type, v: std.json.Value, name: []const u8) !T {
    const x = try field(v, name);
    if (x != .integer) return error.NotAProfile;
    return std.math.cast(T, x.integer) orelse error.NotAProfile;
}

fn float(v: std.json.Value, name: []const u8) !f64 {
    const x = try field(v, name);
    return switch (x) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.NotAProfile,
    };
}

fn boolean(v: std.json.Value, name: []const u8) !bool {
    const x = try field(v, name);
    return if (x == .bool) x.bool else error.NotAProfile;
}

fn ints(comptime T: type, a: std.mem.Allocator, v: std.json.Value) ![]T {
    if (v != .array) return error.NotAProfile;
    const out = try a.alloc(T, v.array.items.len);
    for (v.array.items, out) |x, *o| {
        if (x != .integer) return error.NotAProfile;
        o.* = std.math.cast(T, x.integer) orelse return error.NotAProfile;
    }
    return out;
}

/// `tf-dsv41 exchanges PROFILE...`: each rank's TF_DSV41_PROFILE file, ranks in any order; the report on `out`.
pub fn main(gpa: std.mem.Allocator, io: std.Io, paths: []const []const u8, out: *std.Io.Writer) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const views = try a.alloc(View, paths.len);
    for (paths, views) |p, *v| {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(1 << 30));
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        v.* = viewOf(a, root) catch |e| {
            std.log.err("exchanges: {s} has no prefill exchanges ({t}); was TF_DSV41_PROFILE set on that rank?", .{ p, e });
            return e;
        };
    }
    std.mem.sort(View, views, {}, struct {
        fn lt(_: void, x: View, y: View) bool {
            return x.rank < y.rank;
        }
    }.lt);
    const rep = merge(a, views) catch |e| {
        std.log.err("exchanges: the {d} profiles are not one run's ranks ({t}): same build, same prompts, every rank's file", .{ paths.len, e });
        return e;
    };
    try write(&rep, out);
}
