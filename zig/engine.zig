const std = @import("std");
const zml = @import("zml");

const model = @import("model.zig");

const log = std.log.scoped(.engine);

var verbose: bool = false;
var last_error_buf: [512]u8 = [_]u8{0} ** 512;
var platform_name: [*:0]const u8 = "uninitialized";

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = logFn,
};

fn logFn(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    if (level == .debug and !verbose) return;
    if ((level == .info) and !verbose and !std.mem.eql(u8, @tagName(scope), "engine")) return;
    std.log.defaultLog(level, scope, format, args);
}

fn setError(message: []const u8) void {
    const n = @min(message.len, last_error_buf.len - 1);
    @memcpy(last_error_buf[0..n], message[0..n]);
    last_error_buf[n] = 0;
}

const Engine = struct {
    threaded: std.Io.Threaded,
    platform: *zml.Platform,

    fn create() !*Engine {
        const allocator = std.heap.smp_allocator;
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);

        self.threaded = .init(allocator, .{});
        errdefer self.threaded.deinit();

        const io = self.threaded.io();
        // Unified memory on the Orin Nano is the system RAM. XLA's default
        // reserves 90% of the device up front, which leaves the desktop
        // nothing. Grow the allocator as the matmul needs it.
        self.platform = try zml.Platform.auto(allocator, io, .{
            .cuda = .{
                .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.45 } },
            },
        });
        errdefer self.platform.deinit(allocator, io);

        if (self.platform.target != .cuda) {
            setError("cuda platform was not selected");
            return error.NotCuda;
        }
        platform_name = "cuda";
        log.info("\n{f}", .{self.platform.fmtVerbose()});
        return self;
    }

    fn destroy(self: *Engine) void {
        const allocator = std.heap.smp_allocator;
        const io = self.threaded.io();
        self.platform.deinit(allocator, io);
        self.threaded.deinit();
        allocator.destroy(self);
    }
};

export fn engine_init() ?*Engine {
    return Engine.create() catch |err| {
        setError(@errorName(err));
        log.err("engine_init failed: {s}", .{@errorName(err)});
        return null;
    };
}

export fn engine_deinit(ctx: ?*Engine) void {
    const engine = ctx orelse return;
    engine.destroy();
}

export fn engine_matmul(
    ctx: ?*Engine,
    m: u32,
    k: u32,
    n: u32,
    a: [*]const f32,
    b: [*]const f32,
    out: [*]f32,
) i32 {
    const engine = ctx orelse {
        setError("null engine");
        return -1;
    };
    const a_len = std.math.mul(u32, m, k) catch {
        setError("shape overflow");
        return -1;
    };
    const b_len = std.math.mul(u32, k, n) catch {
        setError("shape overflow");
        return -1;
    };
    const out_len = std.math.mul(u32, m, n) catch {
        setError("shape overflow");
        return -1;
    };

    const allocator = std.heap.smp_allocator;
    model.run(
        allocator,
        engine.threaded.io(),
        engine.platform,
        m,
        k,
        n,
        a[0..a_len],
        b[0..b_len],
        out[0..out_len],
    ) catch |err| {
        setError(@errorName(err));
        log.err("engine_matmul failed: {s}", .{@errorName(err)});
        return -1;
    };
    return 0;
}

export fn engine_last_error() [*:0]const u8 {
    return @ptrCast(&last_error_buf);
}

export fn engine_platform_name() [*:0]const u8 {
    return platform_name;
}

export fn engine_set_verbose(flag: bool) void {
    verbose = flag;
}
