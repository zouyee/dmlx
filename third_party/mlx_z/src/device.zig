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

/// Check if a device is available.
pub fn isAvailable(device: Device) !bool {
    var avail: bool = false;
    try c.check(c.c.mlx_device_is_available(&avail, device.inner));
    return avail;
}

/// Get the number of available devices of a given type.
pub fn deviceCount(dtype: DeviceType) !i32 {
    var count: c_int = 0;
    try c.check(c.c.mlx_device_count(&count, @intCast(@intFromEnum(dtype))));
    return count;
}

/// Device info key-value structure.
pub const DeviceInfo = struct {
    inner: c.c.mlx_device_info,

    pub fn new(device: Device) !DeviceInfo {
        var info = c.c.mlx_device_info_new();
        errdefer _ = c.c.mlx_device_info_free(info);
        try c.check(c.c.mlx_device_info_get(&info, device.inner));
        return .{ .inner = info };
    }

    pub fn deinit(self: DeviceInfo) void {
        _ = c.c.mlx_device_info_free(self.inner);
    }

    /// Check if a key exists in the device info.
    pub fn hasKey(self: DeviceInfo, key: []const u8) !bool {
        var exists: bool = false;
        try c.check(c.c.mlx_device_info_has_key(&exists, self.inner, key.ptr));
        return exists;
    }

    /// Check if a value is a string type.
    pub fn isString(self: DeviceInfo, key: []const u8) !bool {
        var is_str: bool = false;
        try c.check(c.c.mlx_device_info_is_string(&is_str, self.inner, key.ptr));
        return is_str;
    }

    /// Get a string value from device info.
    /// Returns error if key not found or wrong type.
    pub fn getString(self: DeviceInfo, key: []const u8) ![:0]const u8 {
        var value: [*c]const u8 = undefined;
        try c.check(c.c.mlx_device_info_get_string(@ptrCast(&value), self.inner, key.ptr));
        return std.mem.span(value);
    }

    /// Get a size_t value from device info.
    /// Returns error if key not found or wrong type.
    pub fn getSize(self: DeviceInfo, key: []const u8) !usize {
        var value: usize = 0;
        try c.check(c.c.mlx_device_info_get_size(&value, self.inner, key.ptr));
        return value;
    }
};
