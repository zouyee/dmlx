/// Engine module — Server Engine V2 core infrastructure.
///
/// Provides per-request state isolation, MPSC request queue, and
/// completion signal mechanism for safe concurrent inference serving.
pub const request_state = @import("request_state.zig");
pub const completion_signal = @import("completion_signal.zig");
pub const request_queue = @import("request_queue.zig");
pub const engine_loop = @import("engine_loop.zig");

pub const RequestState = request_state.RequestState;
pub const RequestConfig = request_state.RequestConfig;
pub const ApiFormat = request_state.ApiFormat;
pub const FinishReason = request_state.FinishReason;

pub const CompletionSignal = completion_signal.CompletionSignal;
pub const TokenEvent = completion_signal.TokenEvent;

pub const RequestQueue = request_queue.RequestQueue;
pub const QueueNode = request_queue.Node;

pub const EngineLoop = engine_loop.EngineLoop;
pub const EngineConfig = engine_loop.EngineConfig;
pub const requestShutdown = engine_loop.requestShutdown;
pub const isShutdownRequested = engine_loop.isShutdownRequested;
pub const threadSleepMs = engine_loop.threadSleepMs;
pub const generateGuided = engine_loop.generateGuided;
