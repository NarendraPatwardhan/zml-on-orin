//! One matrix multiply. Row-major f32, contracting the shared k axis.
//! (m, k) dot (k, n) -> (m, n).

const std = @import("std");
const zml = @import("zml");

const log = std.log.scoped(.engine);

pub fn multiply(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k);
}

pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    m: u32,
    k: u32,
    n: u32,
    a: []const f32,
    b: []const f32,
    out: []f32,
) !void {
    if (m == 0 or k == 0 or n == 0) return error.EmptyShape;
    const a_len = std.math.mul(u32, m, k) catch return error.ShapeOverflow;
    const b_len = std.math.mul(u32, k, n) catch return error.ShapeOverflow;
    const out_len = std.math.mul(u32, m, n) catch return error.ShapeOverflow;
    // The Orin Nano has 8 GB of unified memory. Refuse a shape that would
    // allocate a large XLA executable on the device.
    if (m > 4096 or k > 4096 or n > 4096) return error.ShapeTooLarge;
    if (a.len != a_len or b.len != b_len or out.len != out_len) return error.LengthMismatch;

    const a_spec: zml.Tensor = .init(.{ .m = m, .k = k }, .f32);
    const b_spec: zml.Tensor = .init(.{ .k = k, .n = n }, .f32);

    log.info("compiling matmul ({d}, {d}) x ({d}, {d})", .{ m, k, k, n });
    var executable = try platform.compileFn(allocator, io, multiply, .{ a_spec, b_spec }, .{});
    defer executable.deinit();

    const a_slice: zml.Slice = .init(a_spec.shape(), std.mem.sliceAsBytes(a));
    const b_slice: zml.Slice = .init(b_spec.shape(), std.mem.sliceAsBytes(b));
    var a_buf: zml.Buffer = try .fromSlice(io, platform, a_slice, .replicated);
    defer a_buf.deinit();
    var b_buf: zml.Buffer = try .fromSlice(io, platform, b_slice, .replicated);
    defer b_buf.deinit();

    var args = try executable.args(allocator);
    defer args.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);

    args.set(.{ a_buf, b_buf });
    executable.callOpts(io, args, &results, .{ .wait = true });

    var result: zml.Buffer = results.get(zml.Buffer);
    defer result.deinit();
    const result_slice = try result.toSliceAlloc(allocator, io);
    defer result_slice.free(allocator);

    const got = result_slice.items(f32);
    if (got.len != out.len) return error.BadResult;
    @memcpy(out, got);
}
