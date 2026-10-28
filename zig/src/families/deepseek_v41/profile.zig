//! TF_DSV41_PROFILE=<file>: eager windows' calls timed on the GPU plus host wall, summed by name, written as JSON.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const xprof = @import("xprof.zig");

pub const Profile = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    path: []const u8,
    events: std.ArrayList(cuda.Event) = .empty,
    /// this window's calls: name, start event, exchange kind and bytes, and whether the call begins a layer
    open: std.ArrayList(struct { name: []const u8, at: usize, ex: xprof.Kind = .none, bytes: u64 = 0, layer: bool = false }) = .empty,
    used: usize = 0,
    totals: std.StringArrayHashMapUnmanaged(Total) = .empty,
    windows: u64 = 0,
    wall_ns: u64 = 0,
    gpu_ns: u64 = 0,
    t0: u64 = 0,
    /// the open window is a prefill segment (beginPrefill): its times go to `pf`
    prefill: bool = false,
    pf: Totals = .{},
    /// each prefill segment's exchanges, from the same events (xprof.zig; every rank writes its own)
    x: xprof.Rank = .{},
    /// host clock at the last prefill window's end, and the open window's boundary (Rank.boundary_ns)
    pf_end: u64 = 0,
    bound: u64 = 0,

    pub const Totals = struct {
        calls: std.StringArrayHashMapUnmanaged(Total) = .empty,
        windows: u64 = 0,
        wall_ns: u64 = 0,
        gpu_ns: u64 = 0,
        /// prompt rows the encoder / full segments covered, and the decoder replays'
        rows: u64 = 0,
        tail_rows: u64 = 0,
    };

    pub const Total = struct { calls: u64 = 0, ns: u64 = 0 };

    /// The runner's view of `p` (Runner.prof).
    pub fn timer(p: *Profile) run.Timer {
        return .{ .ptr = p, .vtable = &timer_vtable };
    }

    const timer_vtable: run.Timer.VTable = .{
        .begin = struct {
            fn f(ptr: *anyopaque) void {
                self(ptr).begin();
            }
        }.f,
        .beginPrefill = struct {
            fn f(ptr: *anyopaque) void {
                self(ptr).beginPrefill();
            }
        }.f,
        .prefillRows = struct {
            fn f(ptr: *anyopaque, rows: u64, tail: bool) void {
                self(ptr).prefillRows(rows, tail);
            }
        }.f,
        .prompt = struct {
            fn f(ptr: *anyopaque) void {
                self(ptr).x.prompt();
            }
        }.f,
        .before = struct {
            fn f(ptr: *anyopaque, s: cuda.Stream, c: *const calls.Call) anyerror!void {
                return self(ptr).before(s, c);
            }
        }.f,
        .after = struct {
            fn f(ptr: *anyopaque, s: cuda.Stream) anyerror!void {
                return self(ptr).after(s);
            }
        }.f,
        .end = struct {
            fn f(ptr: *anyopaque, s: cuda.Stream) anyerror!void {
                return self(ptr).end(s);
            }
        }.f,
        .close = struct {
            fn f(ptr: *anyopaque, io: std.Io) void {
                self(ptr).close(io);
            }
        }.f,
    };

    fn self(ptr: *anyopaque) *Profile {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn fromEnv(gpa: std.mem.Allocator, d: *const cuda.Driver, rank: u32, world: u32) !?*Profile {
        const p = std.c.getenv("TF_DSV41_PROFILE") orelse return null;
        const x = try gpa.create(Profile);
        x.* = .{ .gpa = gpa, .d = d, .path = std.mem.span(p), .x = .{ .rank = rank, .world = world } };
        return x;
    }

    fn event(p: *Profile) !*cuda.Event {
        if (p.used == p.events.items.len) try p.events.append(p.gpa, try cuda.Event.init(p.d, true));
        p.used += 1;
        return &p.events.items[p.used - 1];
    }

    pub fn begin(p: *Profile) void {
        p.prefill = false;
        p.used = 0;
        p.open.clearRetainingCapacity();
        p.t0 = nowNs();
    }

    /// A prefill segment's window (its end adds to the prefill totals).
    pub fn beginPrefill(p: *Profile) void {
        p.begin();
        p.prefill = true;
        p.bound = if (p.x.fresh or p.pf_end == 0) 0 else p.t0 -| p.pf_end;
    }

    /// The rows the last prefill window covered (`tail`: a CED decoder replay's).
    pub fn prefillRows(p: *Profile, rows: u64, tail: bool) void {
        if (tail) p.pf.tail_rows += rows else p.pf.rows += rows;
        p.x.rows(rows, tail);
    }

    /// Before a call: its start event on the stream.
    pub fn before(p: *Profile, s: cuda.Stream, c: *const calls.Call) !void {
        const at = p.used;
        try (try p.event()).record(s);
        const ex = xprof.kindOf(c);
        try p.open.append(p.gpa, .{ .name = c.name, .at = at, .ex = ex, .bytes = if (ex == .timed) xprof.bytesOf(c) else 0, .layer = c.begin == .layer });
    }

    /// After a call: its end event.
    pub fn after(p: *Profile, s: cuda.Stream) !void {
        try (try p.event()).record(s);
    }

    /// The window's end: wait for the stream, add every call's time and the window's wall time.
    pub fn end(p: *Profile, s: cuda.Stream) !void {
        try s.synchronize();
        const wall = nowNs() -| p.t0;
        const map = if (p.prefill) &p.pf.calls else &p.totals;
        if (p.prefill) {
            p.pf.wall_ns += wall;
            p.pf.windows += 1;
            try p.x.segment(p.gpa, p.bound);
            try p.exchanges();
        } else {
            p.wall_ns += wall;
            p.windows += 1;
        }
        for (p.open.items) |o| {
            const ns = try p.between(o.at, o.at + 1);
            const g = try map.getOrPut(p.gpa, o.name);
            if (!g.found_existing) {
                g.key_ptr.* = try p.gpa.dupe(u8, o.name);
                g.value_ptr.* = .{};
            }
            g.value_ptr.calls += 1;
            g.value_ptr.ns += ns;
            if (p.prefill) p.pf.gpu_ns += ns else p.gpu_ns += ns;
        }
        if (p.prefill) p.pf_end = nowNs();
        p.prefill = false;
    }

    /// The prefill window's timed exchanges: in, gap, the gap's idle between calls, layer and site (xprof.zig).
    fn exchanges(p: *Profile) !void {
        // the event the next exchange's gap starts from: the last timed exchange's end, else the window's first
        var from: usize = 0;
        var idle: u64 = 0;
        var layer: u32 = 0;
        var site: u32 = 0;
        for (p.open.items, 0..) |o, i| {
            if (i > 0) idle += try p.between(p.open.items[i - 1].at + 1, o.at);
            if (o.layer) {
                layer += 1;
                site = 0;
            }
            switch (o.ex) {
                .timed => {
                    try p.x.exchange(p.gpa, .{ .in_ns = try p.between(o.at, o.at + 1), .gap_ns = try p.between(from, o.at), .idle_ns = idle, .bytes = o.bytes, .layer = layer, .site = site });
                    from = o.at + 1;
                    idle = 0;
                    site += 1;
                },
                .untimed => p.x.skip(),
                .none => {},
            }
        }
    }

    /// GPU time from event `a` to event `b` of this window.
    fn between(p: *const Profile, a: usize, b: usize) !u64 {
        const ms = try p.events.items[a].elapsedMs(p.events.items[b]);
        return @intFromFloat(@max(0.0, ms) * 1e6);
    }

    /// Writes the totals (JSON) and frees everything.
    pub fn close(p: *Profile, io: std.Io) void {
        var w: std.Io.Writer.Allocating = .init(p.gpa);
        defer w.deinit();
        w.writer.print("{{\"windows\": {d}, \"wall_ms\": {d:.3}, \"gpu_ms\": {d:.3}, \"calls\": {{", .{ p.windows, @as(f64, @floatFromInt(p.wall_ns)) / 1e6, @as(f64, @floatFromInt(p.gpu_ns)) / 1e6 }) catch {};
        for (p.totals.keys(), p.totals.values(), 0..) |k, v, i| w.writer.print("{s}\"{s}\": {{\"calls\": {d}, \"ms\": {d:.4}}}", .{ if (i == 0) "" else ", ", k, v.calls, @as(f64, @floatFromInt(v.ns)) / 1e6 }) catch {};
        w.writer.print("}}, \"prefill\": {{\"segments\": {d}, \"rows\": {d}, \"tail_rows\": {d}, \"wall_ms\": {d:.3}, \"gpu_ms\": {d:.3}, \"calls\": {{", .{ p.pf.windows, p.pf.rows, p.pf.tail_rows, @as(f64, @floatFromInt(p.pf.wall_ns)) / 1e6, @as(f64, @floatFromInt(p.pf.gpu_ns)) / 1e6 }) catch {};
        for (p.pf.calls.keys(), p.pf.calls.values(), 0..) |k, v, i| w.writer.print("{s}\"{s}\": {{\"calls\": {d}, \"ms\": {d:.4}}}", .{ if (i == 0) "" else ", ", k, v.calls, @as(f64, @floatFromInt(v.ns)) / 1e6 }) catch {};
        w.writer.writeAll("}, \"exchanges\": ") catch {};
        p.x.write(&w.writer) catch {};
        w.writer.writeAll("}}\n") catch {};
        if (std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p.path, .data = w.written() })) |_| {
            std.log.info("profile: {s} written ({d} windows, {d} prefill segments)", .{ p.path, p.windows, p.pf.windows });
        } else |e| std.log.err("profile: {s} not written ({t})", .{ p.path, e });
        for (p.events.items) |*e| e.deinit();
        p.events.deinit(p.gpa);
        p.open.deinit(p.gpa);
        for (p.totals.keys()) |k| p.gpa.free(k);
        p.totals.deinit(p.gpa);
        for (p.pf.calls.keys()) |k| p.gpa.free(k);
        p.pf.calls.deinit(p.gpa);
        p.x.deinit(p.gpa);
        p.gpa.destroy(p);
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
