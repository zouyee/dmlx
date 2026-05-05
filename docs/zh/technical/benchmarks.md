# 性能基准测试与仪表盘

本文档介绍了 mlx-zig 的性能基准测试基础设施。

## 概述

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  本地 Mac   │────▶│ benchmark-   │────▶│ GitHub Pages        │
│  (make      │     │ results      │     │ (自动部署)           │
│   benchmark)│     │ 分支         │     │                     │
└─────────────┘     └──────────────┘     └─────────────────────┘
```

- **本地**：在你的机器上运行基准测试（需要 DeepSeek V4 模型）
- **上传**：将结果推送到 `benchmark-results` 分支
- **仪表盘**：自动生成的 HTML，包含趋势图表，部署到 GitHub Pages

## 快速开始

### 1. 本地运行基准测试

```bash
cd /Users/zouyee/work/code/mlx-zig
make benchmark
```

该命令会根据绝对阈值和基线历史记录测量 TTFT、ITL 和 TPS。

### 2. 上传结果

```bash
make upload-benchmark
```

该命令将最新基线追加到 `benchmark-results/history.jsonl` 并推送到远端仓库。

### 3. 查看仪表盘

推送后，GitHub Actions 会自动生成并部署仪表盘：

```
https://dmlx.ai/
```

## 命令

| 命令 | 说明 |
|---------|-------------|
| `make benchmark` | 运行性能基准测试 |
| `make upload-benchmark` | 上传结果到 `benchmark-results` 分支 |
| `make setup-benchmark-branch` | 初始化 `benchmark-results` 分支（仅需运行一次） |
| `make check` | 运行所有本地测试（构建 + 测试 + 验证 + 端到端 + 基准测试） |

## 配置

`make benchmark` 的环境变量：

```bash
MAX_TTFT_MS=300       # TTFT 阈值（默认：500）
MAX_ITL_MS=80         # ITL 阈值（默认：150）
MIN_TPS=10            # 最低吞吐量（默认：5）
REGRESSION_PCT=10     # 回归告警阈值百分比（默认：20）
OUTPUT_TOKENS=50      # 生成 token 数（默认：20）
```

示例：

```bash
MAX_TTFT_MS=300 MAX_ITL_MS=80 make benchmark
```

## 基准测试结果格式

每次基准测试生成一条 JSON 记录：

```json
{
  "hostname": "MacBook-Pro",
  "model": "DeepSeek-V4-Flash-4bit",
  "git_commit": "a18bc24",
  "git_branch": "main",
  "timestamp": "2026-05-01T10:00:00Z",
  "ttft_ms": 215.0,
  "itl_ms": 35.0,
  "tps": 28.5,
  "output_tokens": 20,
  "thresholds": {
    "max_ttft_ms": 500,
    "max_itl_ms": 150,
    "min_tps": 5,
    "regression_pct": 20
  }
}
```

## 仪表盘功能

自动生成的仪表盘包括：

- **趋势图表**：TTFT、ITL、TPS 随时间变化的趋势（Plotly.js）
- **历史表格**：包含提交哈希和日期的全部运行记录
- **回归高亮**：当指标退化超过阈值时，单元格变为红色
- **按机器分组**：结果按 `(hostname, model)` 分组

## CI 集成

### CI（GitHub Actions）中运行的内容

每个 PR 都会触发 `.github/workflows/ci.yml`：
- 构建检查（`zig build`）
- 单元测试（`zig build test`）
- 格式检查（`zig fmt --check`）

### CI 中不运行的内容

- 性能基准测试（需要 141GB 模型，GitHub Runner 不可用）
- 端到端正确性测试（需要 141GB 模型）

这些测试在本地运行，结果手动上传。

### 仪表盘部署

`.github/workflows/dashboard.yml` 在以下情况下运行：
- 有人推送到 `benchmark-results` 分支
- 通过 `workflow_dispatch` 手动触发

它从 `history.jsonl` 生成 HTML 并部署到 GitHub Pages。

## 添加新机器

每台机器自动维护自己的基线：

```bash
# 在新机器上首次运行——创建基线
make benchmark
make upload-benchmark

# 后续运行将与该机器的基线进行对比
make benchmark
```

基线文件按 `hostname` 和模型名称作为键进行区分。

## 故障排除

### "No baseline found" 错误

先运行 `make benchmark` 生成基线，再运行 `make upload-benchmark`。

### 仪表盘未更新

查看 GitHub Actions 标签页中的 "Deploy Performance Dashboard" 工作流。推送后可能需要 1-2 分钟。

### Git 推送失败

确保你有仓库的推送权限，且 `benchmark-results` 分支存在：

```bash
git branch benchmark-results
make upload-benchmark
```
