# Requirements Document

## Introduction

本文档定义了 mlx-zig 项目中 DEEP_VERIFICATION_REPORT.md 审计报告剩余未修复问题的修复需求。经过代码级逐行验证，大部分 P0/P1 问题已在之前的 commits 中修复。剩余工作包括：修复 `io/mlx_io.zig` 和 `grad.zig` 中的 stream 泄漏、清理测试代码中的 stream 泄漏、整理未提交的文件变更（删除二进制文件、更新文档），以及运行完整的测试套件和性能基准验证。

## Glossary

- **MLX_Stream**: MLX C API 中的计算流对象，通过 `mlx_default_cpu_stream_new()` 创建，必须通过 `mlx_stream_free()` 释放
- **Stream_Leak**: 创建了 MLX_Stream 但未调用 `mlx_stream_free()` 释放的代码路径
- **defaultStream_Helper**: `grad.zig` 和 `io/mlx_io.zig` 中的 `inline fn defaultStream()` 辅助函数，每次调用创建新的 MLX_Stream 但不负责释放
- **CoW**: Copy-on-Write，MLX 的内存共享机制，`dataSliceMut` 通过 `@constCast` 绕过此机制
- **EagerContext**: `ops.zig` 中的执行上下文，持有 allocator 和 stream，`deinit()` 会释放 stream
- **KVCacheStrategy**: KV Cache 的运行时多态接口，通过 VTable 实现策略切换
- **best_test_sh**: `scripts/best_test.sh`，7-Prompt 端到端正确性测试脚本
- **Performance_Baseline**: PERFORMANCE_BENCHMARK.md 中记录的性能基线数据（Prefill 170ms, 稳态 ~41ms/token, ~23 tok/s）
- **Binary_Artifacts**: 仓库中意外提交的编译产物（`check_types`, `main`, `test_arr`, `test_fs`, `test_fs2`, `test_mlx_c`, `test_try`）

## Requirements

### Requirement 1: 修复 io/mlx_io.zig 中的 Stream 泄漏

**User Story:** As a developer running mlx-zig in long-lived server mode, I want all MLX stream objects to be properly freed after use, so that the process does not accumulate leaked stream handles over time.

#### Acceptance Criteria

1. WHEN `loadSafetensors` is called, THE IO_Module SHALL create a local MLX_Stream, pass it to `mlx_load_safetensors`, and free it via `mlx_stream_free` before the function returns
2. WHEN `load` is called, THE IO_Module SHALL create a local MLX_Stream, pass it to `mlx_load`, and free it via `mlx_stream_free` before the function returns
3. THE IO_Module SHALL NOT use the `defaultStream` inline helper pattern that creates streams without a corresponding free

### Requirement 2: 修复 grad.zig 中的 Stream 泄漏

**User Story:** As a developer using mlx-zig's automatic differentiation features, I want stream objects to be properly managed, so that gradient computations do not leak resources.

#### Acceptance Criteria

1. THE Grad_Module SHALL NOT contain a `defaultStream` inline helper that creates MLX_Stream objects without a corresponding free
2. IF the `defaultStream` helper is unused in production code paths, THEN THE Grad_Module SHALL remove the dead code entirely
3. IF the `defaultStream` helper is used, THEN THE Grad_Module SHALL replace each call site with a locally-scoped stream that is freed via `defer`

### Requirement 3: 修复测试代码中的 Stream 泄漏

**User Story:** As a developer running the test suite, I want test code to properly free stream objects, so that memory leak detectors do not report false positives from test infrastructure.

#### Acceptance Criteria

1. WHEN `vision/llava.zig` test creates a stream via `mlx_default_cpu_stream_new()`, THE Test_Code SHALL free the stream after use via `mlx_stream_free`

### Requirement 4: 清理仓库中的 Binary Artifacts 和未提交变更

**User Story:** As a project maintainer, I want the repository to be clean of compiled binary artifacts and have all documentation updates properly committed, so that the git history is tidy and the working tree is clean.

#### Acceptance Criteria

1. THE Repository SHALL NOT contain the following Binary_Artifacts in the tracked files: `check_types`, `main`, `test_arr`, `test_fs`, `test_fs2`, `test_mlx_c`, `test_try`
2. THE `.gitignore` file SHALL include patterns to prevent future accidental commits of compiled binaries
3. THE Repository SHALL have all modified documentation files (`DEEP_VERIFICATION_REPORT.md`, `PERFORMANCE_BENCHMARK.md`, `troubleshooting/deepseek-v4.md`) committed
4. THE Repository SHALL have all new analysis report files and scripts committed or explicitly excluded via `.gitignore`

### Requirement 5: 单元测试全部通过

**User Story:** As a developer, I want all unit tests to pass after the fixes are applied, so that I can be confident the fixes do not introduce regressions.

#### Acceptance Criteria

1. WHEN `zig build test` is executed, THE Build_System SHALL exit with code 0 and report no test failures
2. THE Test_Suite SHALL include the existing 430+ tests plus any new tests added for the stream leak fixes

### Requirement 6: 7-Prompt 端到端测试全部通过

**User Story:** As a developer, I want the 7-Prompt end-to-end correctness test to pass after all fixes, so that model inference quality is verified.

#### Acceptance Criteria

1. WHEN `scripts/best_test.sh` is executed with the DeepSeek-V4-Flash-4bit model, THE Test_Script SHALL report 7/7 PASS and 0 FAIL
2. IF any prompt test fails, THEN THE Test_Script SHALL exit with a non-zero exit code

### Requirement 7: 性能无下降验证

**User Story:** As a developer, I want to verify that the fixes do not cause performance degradation compared to the Performance_Baseline, so that inference speed remains acceptable.

#### Acceptance Criteria

1. WHEN a performance benchmark is run after all fixes, THE System SHALL achieve Prefill latency within 20% of the 170ms baseline (i.e., no worse than 204ms)
2. WHEN a performance benchmark is run after all fixes, THE System SHALL achieve steady-state token latency within 20% of the 41ms baseline (i.e., no worse than 49ms)
3. WHEN a performance benchmark is run after all fixes, THE System SHALL achieve throughput within 20% of the 23 tok/s baseline (i.e., no worse than 18.4 tok/s)
4. IF performance degrades beyond the 20% tolerance, THEN THE Developer SHALL investigate and document the cause before committing

### Requirement 8: Server Batch Forward — 实现 Engine Loop 的实际 Forward Pass

**User Story:** As a server operator, I want the engine loop to perform actual batched forward passes using the existing BatchBuilder and Scheduler infrastructure, so that concurrent requests can be processed efficiently through the model.

#### Acceptance Criteria

1. WHEN the engine loop receives decode requests from the Scheduler, THE Engine_Loop SHALL call `batch_builder.build()` to construct a batched input tensor from all scheduled requests
2. WHEN a BatchResult is constructed, THE Engine_Loop SHALL call `model.forward()` with the batched tokens, attention mask, and KV caches
3. WHEN the forward pass produces logits, THE Engine_Loop SHALL sample one token per request from the output logits
4. WHEN tokens are sampled, THE Engine_Loop SHALL call `scheduler.postprocess()` with the TokenOutput array to append tokens and check stop conditions
5. WHEN a request reaches its max_tokens limit or generates a stop token, THE Engine_Loop SHALL mark the request as done and free its KV cache blocks
6. WHEN prefill requests are scheduled, THE Engine_Loop SHALL process the prefill chunk through the model forward pass and advance the prefill offset
7. THE Engine_Loop SHALL handle the case where the batch is empty by sleeping briefly to avoid busy-waiting (existing behavior preserved)
8. IF the forward pass fails for any request, THEN THE Engine_Loop SHALL log the error and mark the affected request as done with an error state

### Requirement 9: 完善 Server 端到端测试脚本

**User Story:** As a developer, I want the server-mode end-to-end test script (`e2e_server.sh`) to cover the same 7 core prompts as `best_test.sh`, so that server-mode inference correctness is verified with the same rigor as CLI-mode.

#### Acceptance Criteria

1. THE `e2e_server.sh` SHALL include all 7 prompts from `best_test.sh`: P1(2+2=), P2(Capital of France completion), P3(Water freezes), P4(Earth round), P5(3*3=), P6(10-5=), P7(Capital of France question)
2. THE `e2e_server.sh` SHALL use the same expected output matching patterns as `best_test.sh` for the 7 core prompts
3. THE `e2e_server.sh` MAY include additional test cases beyond the 7 core prompts (geography, science, language, code, history)
4. WHEN all 7 core prompts pass in server mode, THE Test_Script SHALL report the core prompt pass rate separately from extended tests
5. IF any core prompt fails, THEN THE Test_Script SHALL exit with a non-zero exit code

### Requirement 10: 提交干净的 Git Commit

**User Story:** As a project maintainer, I want all fixes and cleanups committed as a well-organized git commit on the tuning branch, so that the change history is clear and traceable.

#### Acceptance Criteria

1. THE Commit SHALL be created on the `tuning` branch
2. THE Commit message SHALL summarize all fixes applied (stream leaks, binary cleanup, server batch forward, test alignment, documentation updates)
3. THE Commit SHALL NOT include any unrelated changes or temporary files
4. WHEN `git status` is run after the commit, THE Repository SHALL report a clean working tree (no modified or untracked files that should be tracked)