//! Host tests of TF_DSV41_PROFILE's prefill exchanges: the profile on fake events, the prefill programs, the merge.

const std = @import("std");
const testing = std.testing;
const cuda = @import("cuda");
const abi = cuda.abi;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const check = @import("calls_check.zig");
const run = @import("run.zig");
const Config = @import("config.zig").Config;
const Profile = @import("profile.zig").Profile;
const xprof = @import("xprof.zig");
const merge = @import("xprof_merge.zig");

/// Events stamped with a clock the test moves (`now`, ns); records, creations and stream syncs counted.
const Fake = struct {
    var now: u64 = 0;
    var stamps: [256]u64 = undefined;
    var made: usize = 0;
    var records: usize = 0;
    var syncs: usize = 0;

    fn index(e: abi.Event) usize {
        return @intFromPtr(e) - 1;
    }

    fn create(e: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        made += 1;
        e.* = @ptrFromInt(made);
        return abi.success;
    }

    fn destroy(_: abi.Event) callconv(.c) abi.Result {
        return abi.success;
    }

    fn record(e: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        records += 1;
        stamps[index(e)] = now;
        return abi.success;
    }

    fn elapsed(ms: *f32, a: abi.Event, b: abi.Event) callconv(.c) abi.Result {
        ms.* = @floatCast(@as(f64, @floatFromInt(stamps[index(b)])) / 1e6 - @as(f64, @floatFromInt(stamps[index(a)])) / 1e6);
        return abi.success;
    }

    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        syncs += 1;
        return abi.success;
    }

    fn driver() cuda.Driver {
        now = 0;
        made = 0;
        records = 0;
        syncs = 0;
        var d: cuda.Driver = undefined;
        d.api.cuEventCreate = create;
        d.api.cuEventDestroy_v2 = destroy;
        d.api.cuEventRecord = record;
        d.api.cuEventElapsedTime = elapsed;
        d.api.cuStreamSynchronize = sync;
        return d;
    }
};

fn glue(name: []const u8) calls.Call {
    return .{ .triton = false, .glue = true, .name = name, .args = &.{} };
}

const embed = glue("glue.embed");
const exchange = glue("glue.exchange");
const kernel: calls.Call = .{ .triton = true, .name = "_fused", .args = &.{} };
const side_rows: calls.Call = .{ .triton = false, .glue = true, .name = "glue.exchange_rows", .args = &.{}, .side = true };

/// One call as longpf.window issues it under the profile: start event, `ms` of GPU time, end event.
fn issue(t: run.Timer, s: cuda.Stream, c: *const calls.Call, ms: u64) !void {
    try t.before(s, c);
    Fake.now += ms * std.time.ns_per_ms;
    try t.after(s);
}

/// The profile's JSON as close() wrote it to `path`.
fn closed(a: std.mem.Allocator, t: run.Timer, tmp: *std.testing.TmpDir, name: []const u8) !std.json.Value {
    t.close(testing.io);
    const text = try tmp.dir.readFileAlloc(testing.io, name, a, .limited(1 << 20));
    return std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
}

fn get(v: std.json.Value, name: []const u8) !std.json.Value {
    return v.object.get(name) orelse error.MissingMember;
}

fn nums(v: std.json.Value) ![]const i64 {
    var out: std.ArrayList(i64) = .empty;
    for (v.array.items) |x| try out.append(testing.allocator, x.integer);
    return out.toOwnedSlice(testing.allocator);
}

test "profile: a prefill segment's exchanges come from the events it already records: no event or sync added" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var d = Fake.driver();
    const s = cuda.Stream.borrowed(&d, null);
    const p = try testing.allocator.create(Profile);
    p.* = .{ .gpa = testing.allocator, .d = &d, .path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/p.json", .{&tmp.sub_path}) };
    const t = p.timer();
    // a segment: the embedding, a kernel, the attention exchange, a kernel, the MoE exchange
    t.beginPrefill();
    try issue(t, s, &embed, 1);
    try issue(t, s, &kernel, 3);
    try issue(t, s, &exchange, 2);
    try issue(t, s, &kernel, 5);
    try issue(t, s, &exchange, 1);
    try t.end(s);
    t.prefillRows(2048, false);
    // two events a call and one stream sync a window: keeping the exchanges adds no event and no sync
    try testing.expectEqual(@as(usize, 10), Fake.records);
    try testing.expectEqual(@as(usize, 10), Fake.made);
    try testing.expectEqual(@as(usize, 1), Fake.syncs);
    const o = try closed(a, t, &tmp, "p.json");
    const pf = try get(o, "prefill");
    try testing.expectEqual(@as(i64, 2), (try get((try get(pf, "calls")).object.get("glue.exchange").?, "calls")).integer);
    const x = try get(pf, "exchanges");
    const in_ns = try nums(try get(x, "in_ns"));
    defer testing.allocator.free(in_ns);
    const gap_ns = try nums(try get(x, "gap_ns"));
    defer testing.allocator.free(gap_ns);
    // in: each exchange's own start to end; gap: the segment's first call (then the last exchange's end) to its start
    try testing.expectEqualSlices(i64, &.{ 2_000_000, 1_000_000 }, in_ns);
    try testing.expectEqualSlices(i64, &.{ 4_000_000, 5_000_000 }, gap_ns);
    const seg = (try get(x, "segments")).array.items[0];
    try testing.expectEqual(@as(i64, 2), (try get(seg, "exchanges")).integer);
    try testing.expectEqual(@as(i64, 2048), (try get(seg, "rows")).integer);
}

test "profile: prompts group segments, a side-stream exchange is counted untimed, decode windows add none" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var d = Fake.driver();
    const s = cuda.Stream.borrowed(&d, null);
    const p = try testing.allocator.create(Profile);
    p.* = .{ .gpa = testing.allocator, .d = &d, .path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/r1.json", .{&tmp.sub_path}), .x = .{ .rank = 1, .world = 2 } };
    const t = p.timer();
    t.prompt();
    for (0..2) |_| {
        t.beginPrefill();
        try issue(t, s, &kernel, 2);
        try issue(t, s, &exchange, 1);
        try issue(t, s, &side_rows, 1);
        try t.end(s);
        t.prefillRows(1024, false);
    }
    // an eager decode window's exchange is not a prefill exchange
    t.begin();
    try issue(t, s, &exchange, 7);
    try t.end(s);
    t.prompt();
    t.beginPrefill();
    try issue(t, s, &exchange, 3);
    try t.end(s);
    t.prefillRows(16, true);
    const o = try closed(a, t, &tmp, "r1.json");
    const x = try get(try get(o, "prefill"), "exchanges");
    try testing.expectEqual(@as(i64, 1), (try get(x, "rank")).integer);
    try testing.expectEqual(@as(i64, 3), (try get(x, "timed")).integer);
    try testing.expectEqual(@as(i64, 2), (try get(x, "untimed")).integer);
    const prompts = (try get(x, "prompts")).array.items;
    try testing.expectEqual(@as(usize, 2), prompts.len);
    try testing.expectEqual(@as(i64, 2), (try get(prompts[0], "segments")).integer);
    try testing.expectEqual(@as(i64, 2048), (try get(prompts[0], "rows")).integer);
    try testing.expectEqual(@as(i64, 2), (try get(prompts[0], "exchanges")).integer);
    try testing.expectEqual(@as(i64, 1), (try get(prompts[1], "exchanges")).integer);
    const segs = (try get(x, "segments")).array.items;
    try testing.expectEqual(@as(i64, 1), (try get(segs[1], "untimed")).integer);
    try testing.expect((try get(segs[2], "tail")).bool);
    // read back as the merge reads it: the same rows, counts and times
    const v = try merge.viewOf(a, o);
    try testing.expectEqual(@as(usize, 3), v.segments.len);
    try testing.expectEqualSlices(u64, &.{ 1_000_000, 1_000_000, 3_000_000 }, v.in_ns);
    try testing.expectEqual(@as(u32, 2), v.segments[2].first);
}

const shard: calls.Tensor = .{ .role = .{ .buf = "L.part" }, .dt = .bf16, .shape = &.{ 16, 5120 }, .stride = &.{ 5120, 1 } };
const sized: calls.Call = .{ .triton = false, .glue = true, .name = "glue.exchange", .args = &.{.{ .arg = .{ .t = shard } }} };
const layer_start: calls.Call = .{ .triton = true, .name = "_fused", .args = &.{}, .begin = .layer };

test "profile: a gap is its calls' GPU time plus the idle between calls; each exchange's layer, site, bytes, boundary" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var d = Fake.driver();
    const s = cuda.Stream.borrowed(&d, null);
    const p = try testing.allocator.create(Profile);
    p.* = .{ .gpa = testing.allocator, .d = &d, .path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/idle.json", .{&tmp.sub_path}) };
    const t = p.timer();
    t.prompt();
    t.beginPrefill();
    try issue(t, s, &kernel, 3);
    try issue(t, s, &sized, 2);
    // the host is 4 ms late with the next call: the GPU idles between the exchange and that call
    Fake.now += 4 * std.time.ns_per_ms;
    try issue(t, s, &kernel, 5);
    try issue(t, s, &sized, 1);
    Fake.now += 1 * std.time.ns_per_ms;
    try issue(t, s, &layer_start, 2);
    try issue(t, s, &sized, 1);
    try t.end(s);
    t.prefillRows(16, false);
    // the prompt's next segment starts 5 ms of host time after this one ended; a new prompt's first starts at 0
    p.pf_end -|= 5 * std.time.ns_per_ms;
    t.beginPrefill();
    try issue(t, s, &sized, 2);
    try t.end(s);
    t.prefillRows(16, false);
    t.prompt();
    t.beginPrefill();
    try issue(t, s, &sized, 1);
    try t.end(s);
    t.prefillRows(16, false);
    try testing.expectEqual(@as(usize, 3), Fake.syncs);
    const o = try closed(a, t, &tmp, "idle.json");
    const v = try merge.viewOf(a, o);
    try testing.expectEqualSlices(u64, &.{ 2_000_000, 1_000_000, 1_000_000, 2_000_000, 1_000_000 }, v.in_ns);
    try testing.expectEqualSlices(u64, &.{ 3_000_000, 9_000_000, 3_000_000, 0, 0 }, v.gap_ns);
    try testing.expectEqualSlices(u64, &.{ 0, 4_000_000, 1_000_000, 0, 0 }, v.idle_ns);
    try testing.expectEqualSlices(u32, &.{ 0, 0, 1, 0, 0 }, v.layer);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 0, 0, 0 }, v.site);
    try testing.expectEqual(@as(u64, 3 * 16 * 5120 * 2), v.segments[0].bytes);
    try testing.expectEqual(@as(u64, 0), v.boundary_ns[0]);
    try testing.expect(v.boundary_ns[1] >= 5_000_000);
    try testing.expectEqual(@as(u64, 0), v.boundary_ns[2]);
    const seg = (try get(try get(try get(o, "prefill"), "exchanges"), "segments")).array.items[0];
    try testing.expectEqual(@as(f64, 10), (try get(seg, "kernel_ms")).float);
    try testing.expectEqual(@as(f64, 5), (try get(seg, "idle_ms")).float);
    try testing.expectEqual(@as(usize, 2), v.calls.len);
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

fn kinds(cs: []const calls.Call) [3]usize {
    var n: [3]usize = @splat(0);
    for (cs) |*c| n[@backingInt(xprof.kindOf(c))] += 1;
    return n;
}

test "profile: a prefill segment's two exchanges a layer are timed, the overlap's pieces counted untimed" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(a);
    const layers = [_]u32{ 1, 2, 3 };
    const base: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 };
    var served = base;
    served.branches = true;
    served.r1 = true;
    served.mhc_defer = true;
    for ([_]block.Options{ base, served }) |o| {
        const n = kinds(try bp.emitPrefill(a, &cfg, &w, o, &layers, 2048, 0, false));
        try testing.expectEqual(@as(usize, 2 * layers.len), n[@backingInt(xprof.Kind.timed)]);
        try testing.expectEqual(@as(usize, 0), n[@backingInt(xprof.Kind.untimed)]);
    }
    // TF_DSV41_PF_OVERLAP: the exchanges go in row pieces on the side stream, which the profile's events cannot bracket
    var ovl = base;
    ovl.branches = true;
    ovl.pf_overlap = 2;
    const n = kinds(try bp.emitPrefill(a, &cfg, &w, ovl, &layers, 2048, 0, false));
    try testing.expectEqual(@as(usize, 0), n[@backingInt(xprof.Kind.timed)]);
    try testing.expectEqual(@as(usize, 2 * 2 * layers.len), n[@backingInt(xprof.Kind.untimed)]);
}

/// Rank r's view: each exchange's in and gap, no idle, all at layer 0 in site order, no boundaries, no calls.
fn view(a: std.mem.Allocator, r: u32, world: u32, in_ns: []const u64, gap_ns: []const u64, segs: []const xprof.Segment) !merge.View {
    const idle = try a.alloc(u64, in_ns.len);
    @memset(idle, 0);
    const layer = try a.alloc(u32, in_ns.len);
    @memset(layer, 0);
    const site = try a.alloc(u32, in_ns.len);
    for (site, 0..) |*x, i| x.* = @intCast(i);
    const bound = try a.alloc(u64, segs.len);
    @memset(bound, 0);
    return .{ .rank = r, .world = world, .in_ns = in_ns, .gap_ns = gap_ns, .idle_ns = idle, .layer = layer, .site = site, .segments = segs, .boundary_ns = bound };
}

test "exchanges: two ranks side by side: the last to arrive, the other's wait, a tie shared, the transfer" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const segs = [_]xprof.Segment{ .{ .prompt = 0, .first = 0, .count = 2, .rows = 2048 }, .{ .prompt = 0, .first = 2, .count = 2, .rows = 1024 } };
    const r0 = try view(a, 0, 2, &.{ 100, 300, 100, 250 }, &.{ 900, 700, 900, 750 }, &segs);
    const r1 = try view(a, 1, 2, &.{ 400, 100, 100, 600 }, &.{ 600, 900, 900, 400 }, &segs);
    const rep = try merge.merge(a, &.{ r0, r1 });
    try testing.expectEqualSlices(u32, &.{ 0, 1, 0, 0 }, rep.last);
    try testing.expectEqualSlices(u64, &.{ 100, 100, 100, 250 }, rep.transfer_ns);
    try testing.expectEqual(@as(u64, 1), rep.ties);
    try testing.expectEqual(@as(f64, 0.625), rep.total.share(0));
    try testing.expectEqual(@as(f64, 0.375), rep.total.share(1));
    try testing.expectEqual(@as(u64, 200), rep.total.ranks[0].wait_ns);
    try testing.expectEqual(@as(u64, 650), rep.total.ranks[1].wait_ns);
    try testing.expectEqual(@as(u64, 550), rep.total.transfer_ns);
    try testing.expectEqualSlices(u64, &.{ 0, 300 }, rep.wait_p50_ns);
    try testing.expectEqualSlices(u64, &.{ 200, 350 }, rep.wait_p90_ns);
    // every rank's in-time is its wait plus the transfer
    for (rep.total.ranks) |x| try testing.expectEqual(x.in_ns, x.wait_ns + rep.total.transfer_ns);
    try testing.expectEqual(@as(usize, 1), rep.prompts.len);
    try testing.expectEqual(@as(u64, 3072), rep.prompts[0].rows);
    try testing.expectEqual(@as(f64, 1), rep.segments[0].share(0) + rep.segments[0].share(1));
    try testing.expectEqual(@as(f64, 0.75), rep.segments[1].share(0));
    var out: std.Io.Writer.Allocating = .init(a);
    try merge.write(&rep, &out.writer);
    const o = try std.json.parseFromSliceLeaky(std.json.Value, a, out.written(), .{});
    const ranks = (try get(o, "ranks")).array.items;
    try testing.expectEqual(@as(f64, 0.625), (try get(ranks[0], "late")).float);
    try testing.expectEqual(@as(i64, 4), (try get(o, "exchanges")).integer);
}

test "exchanges: four ranks, a three-way tie shares the late count; profiles from different runs are refused" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const segs = [_]xprof.Segment{.{ .prompt = 0, .first = 0, .count = 1, .rows = 16 }};
    var views: [4]merge.View = undefined;
    for (&views, 0..) |*v, r| v.* = try view(a, @intCast(r), 4, if (r == 3) &.{40} else &.{10}, &.{0}, &segs);
    const rep = try merge.merge(a, &views);
    for (0..3) |r| try testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), rep.total.share(r), 1e-12);
    try testing.expectEqual(@as(f64, 0), rep.total.share(3));
    try testing.expectEqual(@as(u64, 30), rep.total.ranks[3].wait_ns);
    // another rank count, a rank twice, or another prefill shape: not one run's ranks
    try testing.expectError(error.RanksNotOneWorld, merge.merge(a, views[0..3]));
    var twice = views;
    twice[2].rank = 1;
    try testing.expectError(error.RanksNotOneWorld, merge.merge(a, &twice));
    const other = [_]xprof.Segment{.{ .prompt = 0, .first = 0, .count = 1, .rows = 32 }};
    var shape = views;
    shape[1].segments = &other;
    try testing.expectError(error.SegmentsDiffer, merge.merge(a, &shape));
}

test "exchanges: the tool reads each rank's profile file, in any order" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_]u32{ 0, 1 }) |r| {
        var x: xprof.Rank = .{ .rank = r, .world = 2 };
        defer x.deinit(testing.allocator);
        x.prompt();
        try x.segment(testing.allocator, 0);
        try x.exchange(testing.allocator, .{ .in_ns = if (r == 0) 50_000 else 80_000, .gap_ns = 10_000, .idle_ns = 0, .bytes = 64 * 5120 * 2, .layer = 0, .site = 0 });
        x.rows(64, false);
        var w: std.Io.Writer.Allocating = .init(a);
        try w.writer.writeAll("{\"prefill\": {\"calls\": {\"_fused\": {\"calls\": 1, \"ms\": 0.5}}, \"exchanges\": ");
        try x.write(&w.writer);
        try w.writer.writeAll("}}");
        try tmp.dir.writeFile(testing.io, .{ .sub_path = if (r == 0) "r0.json" else "r1.json", .data = w.written() });
    }
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    const paths = [_][]const u8{ try std.fs.path.join(a, &.{ dir, "r1.json" }), try std.fs.path.join(a, &.{ dir, "r0.json" }) };
    var out: std.Io.Writer.Allocating = .init(a);
    try merge.main(testing.allocator, testing.io, &paths, &out.writer);
    const o = try std.json.parseFromSliceLeaky(std.json.Value, a, out.written(), .{});
    try testing.expectEqual(@as(i64, 0), (try get(o, "last")).array.items[0].integer);
    try testing.expectEqual(@as(f64, 0.03), (try get((try get(o, "ranks")).array.items[1], "wait_ms")).float);
}

test "exchanges: kernel and idle time, segment starts apart, layers without tail segments, the calls that differ most" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const segs = [_]xprof.Segment{ .{ .prompt = 0, .first = 0, .count = 2, .rows = 2048, .bytes = 1000 }, .{ .prompt = 0, .first = 2, .count = 2, .rows = 16, .tail = true, .bytes = 10 } };
    var r0 = try view(a, 0, 2, &.{ 100, 300, 100, 250 }, &.{ 900, 700, 900, 750 }, &segs);
    r0.idle_ns = &.{ 0, 100, 0, 50 };
    r0.site = &.{ 0, 1, 0, 1 };
    r0.boundary_ns = &.{ 0, 30 };
    r0.calls = &.{ .{ .name = "_fused", .calls = 10, .ms = 5.0 }, .{ .name = "glue.exchange", .calls = 4, .ms = 0.75 } };
    var r1 = try view(a, 1, 2, &.{ 400, 100, 100, 600 }, &.{ 600, 900, 900, 400 }, &segs);
    r1.idle_ns = &.{ 300, 0, 0, 200 };
    r1.site = r0.site;
    r1.boundary_ns = &.{ 0, 20 };
    r1.calls = &.{ .{ .name = "glue.embed", .calls = 2, .ms = 0.125 }, .{ .name = "glue.exchange", .calls = 4, .ms = 1.25 }, .{ .name = "_fused", .calls = 10, .ms = 6.5 } };
    const rep = try merge.merge(a, &.{ r0, r1 });
    // kernel time is the gap less its idle
    try testing.expectEqual(@as(u64, 3100), rep.total.ranks[0].kernel());
    try testing.expectEqual(@as(u64, 2300), rep.total.ranks[1].kernel());
    try testing.expectEqual(@as(u64, 500), rep.total.ranks[1].idle_ns);
    try testing.expectEqual(@as(u64, 30), rep.total.ranks[0].boundary_ns);
    // exchanges 0 and 2 open their segments; 1 and 3 do not
    try testing.expectEqual(@as(u64, 200), rep.starts.transfer_ns);
    try testing.expectEqual(@as(f64, 0.75), rep.starts.share(0));
    try testing.expectEqual(@as(u64, 300), rep.starts.ranks[1].wait_ns);
    try testing.expectEqual(@as(f64, 0.5), rep.rest.share(0));
    try testing.expectEqual(@as(u64, 350), rep.rest.transfer_ns);
    // the tail segment is left out of the layers: one exchange at each site
    try testing.expectEqual(@as(usize, 2), rep.layers.len);
    try testing.expectEqual(@as(u32, 1), rep.layers[1].site);
    try testing.expectEqual(@as(u64, 1), rep.layers[1].exchanges);
    try testing.expectEqual(@as(f64, 1), rep.layers[1].share(1));
    // 1000 bytes from the other rank in 200 ns of transfer: 5 GB/s
    try testing.expectEqual(@as(f64, 5), rep.segments[0].gbytesPerS(2));
    try testing.expectEqual(@as(usize, 3), rep.calls.len);
    try testing.expectEqualStrings("_fused", rep.calls[0].name);
    try testing.expectEqual(@as(f64, 1.5), rep.calls[0].spread_ms);
    try testing.expectEqualStrings("glue.embed", rep.calls[2].name);
    try testing.expectEqualSlices(f64, &.{ 0, 0.125 }, rep.calls[2].ms);
    var out: std.Io.Writer.Allocating = .init(a);
    try merge.write(&rep, &out.writer);
    const o = try std.json.parseFromSliceLeaky(std.json.Value, a, out.written(), .{});
    try testing.expectEqual(@as(f64, 0.002), (try get((try get(o, "ranks")).array.items[1], "kernel_ms")).float);
    try testing.expectEqual(@as(i64, 2), (try get(try get(o, "segment_starts"), "exchanges")).integer);
    try testing.expectEqual(@as(usize, 2), (try get(o, "by_layer")).array.items.len);
}
