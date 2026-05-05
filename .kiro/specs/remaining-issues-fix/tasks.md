# Implementation Plan: Remaining Issues Fix

## Overview

本计划按照设计文档的修复顺序，依次修复 stream 泄漏、实现 server engine loop batch forward、对齐 e2e 测试脚本、清理仓库、运行验证测试，最后提交 git commit。所有任务使用 Zig 语言实现。

## Tasks

- [x] 1. 修复 Stream 泄漏 (R1, R2, R3)
  - [x] 1.1 修复 `src/io/mlx_io.zig` 中的 stream 泄漏
    - 删除 `defaultStream()` inline helper 函数（第 8-10 行）
    - 在 `loadSafetensors` 函数体内创建局部 `const stream = c.c.mlx_default_cpu_stream_new();`，紧接 `defer _ = c.c.mlx_stream_free(stream);`，将 `defaultStream()` 调用替换为 `stream`
    - 在 `load` 函数体内同样创建局部 stream 并 defer 释放，将 `defaultStream()` 调用替换为 `stream`
    - _Requirements: R1.1, R1.2, R1.3_

  - [x] 1.2 清理 `src/grad.zig` 中的 dead code
    - 删除 `defaultStream()` inline helper 函数定义（第 10-12 行）
    - 该函数在当前代码中未被任何函数调用，属于纯 dead code 清理
    - _Requirements: R2.1, R2.2_

  - [x] 1.3 修复 `src/vision/llava.zig` 测试代码中的 stream 泄漏
    - 在 `test "LlavaModel init and deinit"` 中，将第 158 行 `c.c.mlx_default_cpu_stream_new()` 直接传参改为局部变量 + `defer` 释放
    - 具体：在 `buildModel` 调用前添加 `const stream = c.c.mlx_default_cpu_stream_new(); defer _ = c.c.mlx_stream_free(stream);`，然后将 `c.c.mlx_default_cpu_stream_new()` 替换为 `stream`
    - _Requirements: R3.1_

- [x] 2. Checkpoint — 确保 stream 泄漏修复后编译通过
  - 运行 `zig build test` 确保所有测试通过，ask the user if questions arise.

- [x] 3. 实现 Server Engine Loop Batch Forward (R8)
  - [x] 3.1 实现 `engineLoop` 中的 batch forward 循环
    - 替换 `src/server.zig` 中 `engineLoop` 函数的 stub 实现
    - 实现完整的 schedule → `batch_builder_mod.build()` → `state.vtable.forward()` → `sampleFromLogits()` → `sched.postprocess()` 循环
    - 对 decode requests：构建 batch input，执行 forward pass，从 logits 中采样 token
    - 对 prefill requests：通过 model forward pass 处理 prefill chunk，推进 prefill offset
    - 保留空 batch 时的 sleep 行为避免 busy-wait
    - _Requirements: R8.1, R8.2, R8.3, R8.4, R8.5, R8.6, R8.7_

  - [x] 3.2 实现辅助函数 `sampleFromLogits` 和 `markScheduledRequestsDone`
    - `sampleFromLogits(state, logits, scheduled)`: 从 logits tensor 中为每个 request 提取最后一个 token 位置的 logits，执行 argmax（temperature=0）或采样，返回 `[]TokenOutput`
    - `markScheduledRequestsDone(scheduled, state)`: 错误恢复辅助函数，将所有 scheduled requests 标记为 done
    - 每个阶段独立错误处理：batch build 失败、forward 失败、sampling 失败均 log 错误并标记请求完成，不 panic
    - _Requirements: R8.3, R8.8_

- [x] 4. 对齐 `scripts/e2e_server.sh` 测试脚本 (R9)
  - 在脚本测试用例部分开头添加 7 个核心 prompt 测试（与 `best_test.sh` 完全一致的 prompt 措辞、max_tokens 和 expected 匹配模式）：P1(2+2=, 30, "4"), P2(The capital of France is, 20, "Paris"), P3(Water freezes Celsius, 30, "0"), P4(Earth round yes/no, 30, "yes"), P5(3*3=, 30, "9"), P6(10-5=, 50, "answer is 5"), P7(Capital of France question, 30, "Paris")
  - 将核心 prompt 通过率与扩展测试通过率分别统计
  - 核心 prompt 任一失败则脚本返回非零退出码
  - 保留现有扩展测试用例
  - _Requirements: R9.1, R9.2, R9.3, R9.4, R9.5_

- [x] 5. Checkpoint — 确保 server 修改编译通过
  - 运行 `zig build test` 确保所有测试通过，ask the user if questions arise.

- [x] 6. 仓库清理 (R4)
  - [x] 6.1 更新 `.gitignore` 添加二进制排除模式
    - 在 `.gitignore` 中添加以下模式：`check_types`, `main`, `test_arr`, `test_fs`, `test_fs2`, `test_mlx_c`, `test_try`, `test_remap_standalone`, `test_mlx_take_zero`
    - _Requirements: R4.2_

  - [x] 6.2 删除 git 跟踪的二进制文件
    - 使用 `git rm --cached` 删除仓库中跟踪的二进制文件（如果存在）：`check_types`, `main`, `test_arr`, `test_fs`, `test_fs2`, `test_mlx_c`, `test_try`
    - 确认 `git status` 中无意外的 tracked binary artifacts
    - _Requirements: R4.1_

  - [x] 6.3 暂存文档变更
    - 确保 `analysis-report/DEEP_VERIFICATION_REPORT.md`, `analysis-report/PERFORMANCE_BENCHMARK.md`, `docs/` 下的文档变更已暂存
    - 确保新增的脚本文件已暂存
    - _Requirements: R4.3, R4.4_

- [x] 7. 单元测试验证 (R5)
  - 运行 `zig build test`，确认 exit code 0，430+ 测试全部通过，无回归
  - _Requirements: R5.1, R5.2_

- [x] 8. 7-Prompt 端到端测试 (R6)
  - 构建优化版本 `zig build -Doptimize=ReleaseFast`
  - 运行 `scripts/best_test.sh`，确认 7/7 PASS, 0 FAIL
  - 如有失败，排查原因并修复后重新运行
  - _Requirements: R6.1, R6.2_

- [x] 9. 性能基准验证 (R7)
  - 运行性能基准测试，对比 PERFORMANCE_BENCHMARK.md 中的基线数据
  - 验证 Prefill ≤ 204ms, 稳态 ≤ 49ms/token, 吞吐量 ≥ 18.4 tok/s（±20% 容忍范围）
  - 如性能下降超过容忍范围，排查原因并记录
  - _Requirements: R7.1, R7.2, R7.3, R7.4_

- [x] 10. Git Commit (R10)
  - 在 `tuning` 分支上创建 commit，commit message 概述所有修复内容：stream 泄漏修复、server batch forward 实现、e2e 测试对齐、二进制清理、文档更新
  - 确认 `git status` 报告 clean working tree
  - 不包含无关变更或临时文件
  - _Requirements: R10.1, R10.2, R10.3, R10.4_

## Notes

- 设计文档明确说明 PBT 不适用于本特性，因此不包含 property-based test 任务
- 验证策略依赖现有的单元测试（`zig build test`）、端到端测试（`best_test.sh`、`e2e_server.sh`）和性能基准
- Task 7-9 需要模型文件和 GPU 硬件，如环境不可用则记录并跳过
- 每个 checkpoint 确保增量修复不引入回归

## Post-Audit Fixes (2026-05-04)

### Additional Issues Found & Fixed

- [x] **A1. Server KV Cache Type for DeepSeek V4** — `server.zig:loadModel()` created `StandardKVCache` for ALL architectures. DSv4 requires per-layer specialized caches (`DeepseekV4Cache` for MLA-compressed layers, `RotatingWithWindow` for sliding-window layers). Fixed by calling `makeV4Caches()` when `arch_name == "DeepseekV4ForCausalLM"`.

- [x] **A2. Dead code in eval.zig** — `defaultStream()` inline helper defined but never called (same pattern fixed in R1/R2 for `io/mlx_io.zig` and `grad.zig`). Removed.

- [x] **A3. e2e_server.sh stale limitation comment** — Updated to reflect that specialized KV caches are now properly created.

### Verification

| Check | Result |
|-------|--------|
| `zig build` | ✅ Compiles |
| Server startup (DSv4 + SMELT 10%) | ✅ Loads, listens, no KV cache errors |
| `nativeGenerate` path | ✅ Already correct (creates own caches) |
| Binary artifacts cleaned | ✅ All removed |
| All 10 original tasks re-verified | ✅ Complete |
