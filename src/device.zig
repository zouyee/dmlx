/// Device and Stream abstractions backed by mlx-c.
const std = @import("std");
const c = @import("c.zig");

pub const DeviceType = enum(c_uint) {
    cpu = c.c.MLX_CPU,
    gpu = c.c.MLX_GPU,
};

pub const Device = struct {
    inner: c.c.mlx_device,

    pub fn new(dtype: DeviceType, index: c_int) Device {
        return .{ .inner = c.c.mlx_device_new_type(@intCast(@intFromEnum(dtype)), index) };
    }

    pub fn default() !Device {
        var dev: c.c.mlx_device = undefined;
        try c.check(c.c.mlx_get_default_device(&dev));
        return .{ .inner = dev };
    }

    pub fn setDefault(self: Device) !void {
        return c.check(c.c.mlx_set_default_device(self.inner));
    }

    pub fn cpu() Device {
        return new(.cpu, 0);
    }

    pub fn gpu() Device {
        return new(.gpu, 0);
    }

    pub fn isCpu(self: Device) bool {
        var dtype: c.c.mlx_device_type = undefined;
        _ = c.c.mlx_device_get_type(&dtype, self.inner);
        return dtype == c.c.MLX_CPU;
    }

    pub fn isGpu(self: Device) bool {
        return !self.isCpu();
    }

    pub fn deinit(self: Device) void {
        _ = c.c.mlx_device_free(self.inner);
    }
};

pub const Stream = struct {
    inner: c.c.mlx_stream,

    pub fn new(device: Device) Stream {
        return .{ .inner = c.c.mlx_stream_new_device(device.inner) };
    }

    pub fn defaultStream(device: Device) !Stream {
        var stream: c.c.mlx_stream = undefined;
        try c.check(c.c.mlx_get_default_stream(&stream, device.inner));
        return .{ .inner = stream };
    }

    pub fn synchronize(self: Stream) !void {
        return c.check(c.c.mlx_synchronize(self.inner));
    }

    pub fn deinit(self: Stream) void {
        _ = c.c.mlx_stream_free(self.inner);
    }
};

pub fn defaultStream(device: Device) !Stream {
    return Stream.defaultStream(device);
}

pub fn newStream(device: Device) Stream {
    return Stream.new(device);
}

pub fn defaultDevice() !Device {
    return Device.default();
}

pub fn setDefaultDevice(device: Device) !void {
    return device.setDefault();
}
