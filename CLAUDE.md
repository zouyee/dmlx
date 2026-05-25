# dmlx 开发规则

## 性能优化纪律

1. **不允许在无 benchmark 验证的情况下 commit 性能优化**
   - 任何声称提升性能的改动，必须先跑 `bash scripts/run_benchmark.sh` 获取数据
   - commit message 中的性能数字必须来自 benchmark 报告，不得使用手动单次测试数据
   - 手动测试只能用于快速验证正确性，不能作为 commit 依据

2. **正确性优先于性能**
   - benchmark 7-prompt E2E 未达到 7/7 时，不允许 push
   - 正确性回归时，即使性能有提升也不能提交

3. **只提交已验证的改动**
   - 每次 commit 只包含一个逻辑改动
   - commit message 必须注明 benchmark 数据来源（report 路径或 benchmark 日志）
   - 实验性代码（DyMoE 阈值调优等）必须先 stash，验证通过后再 commit

4. **测试环境要求**
   - 连续多次测试前必须 `sudo purge` 清理 page cache，否则后续测试数据不可靠
   - benchmark 脚本已内置 purge 逻辑（`scripts/run_benchmark.sh`）

5. **禁止的行为**
   - 不得在 commit message 中使用手动测试的性能数据
   - 不得在未跑 benchmark 的情况下声称性能提升
   - 不得猜测性能瓶颈原因（节流/竞态/污染等），必须有日志或指标证明
   - 不得连续多次 commit 后再跑 benchmark
   - 不得在未 purge 的情况下连续跑多次 benchmark 并对比数据
