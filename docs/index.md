# dmlx 文档导航 / Documentation

> **核心叙事：小规格 Mac 运行大模型** — 48GB MacBook Pro 运行 284B DeepSeek V4 MoE
> 
> **Core Narrative: Large Models on Small Macs** — 284B DeepSeek V4 MoE on a 48GB MacBook Pro

---

## ⭐ DeepSeek MoE 专题 / DeepSeek MoE Deep Dive

| English | 中文 |
|---------|------|
| [Technical Deep Dive: How It Works](en/deepseek-moe/README.md) | [技术深度：如何实现](zh/deepseek-moe/README.md) |
| [Application Scenarios](en/scenarios/README.md) | [应用场景](zh/scenarios/README.md) |

### Core Technology Stack / 核心技术栈

| Layer | Technology | Memory Impact |
|-------|-----------|---------------|
| Expert Streaming | SMELT + on-demand loading | 138GB → 10GB |
| 4-bit Quantization | Affine INT4/MXFP4/TurboQuant | 40GB → 6-12GB (SMELT) |
| MLA Compression | Multi-head Latent Attention | KV: 2×heads×dim → 2×latent_dim |
| KV Cache | 6 strategies (Paged/Tiered/Quantized) | Configurable per workload |
| Zero-Copy | mmap direct loading | 7GB memcpy → 0 |

---

## English

### 🚀 Getting Started
- [Quick Start](../README.md#quick-start) — 5-minute setup
- [Installation](../README.md#installation) — Requirements & build
- [DeepSeek V4 Quick Fix](en/user-guide/deepseek-v4-quickfix.md) — Fix garbled output
- [Troubleshooting: DeepSeek V4](en/user-guide/troubleshooting/deepseek-v4.md)
- [Troubleshooting: Scheduler](en/user-guide/troubleshooting/scheduler.md)

### 🧠 DeepSeek MoE Architecture
- [Technical Deep Dive](en/deepseek-moe/README.md) — How 150GB fits in 48GB
- [Application Scenarios](en/scenarios/README.md) — Local, privacy, edge, offline
- [SMELT Expert Streaming](en/technical/smelt-flow.md) — Expert loading flow
- [4-bit + SMELT](en/technical/4bit-smelt.md) — Quantization + partial loading
- [TTFT Optimization](en/technical/ttft-optimization.md) — Model loading optimization
- [Inference Speed Optimization](en/technical/inference-optimization.md) — Per-token latency reduction plan
- [Auto-Detect Experts](en/technical/auto-detect-experts.md) — Partial expert detection

### 📊 DeepSeek V4 Fix History
- [Consolidated Fixes](en/deepseek-v4/DEEPSEEK-V4-FIXES-AND-DETAILS.md)
- [Chat Analysis & Troubleshooting](en/deepseek-v4/DEEPSEEK-V4-CHAT-ANALYSIS-AND-TROUBLESHOOTING.md)
- [Optimization Roadmap](en/deepseek-v4/DEEPSEEK-V4-OPTIMIZATION-AND-ROADMAP.md)

### ⚙️ Technical
- [Continuous Batching Design](en/technical/continuous-batching-design.md) — Scheduler + BatchBuilder + Paged KV cache architecture
- [Benchmarks](en/technical/benchmarks.md) — Performance data
- [Inference Speed Optimization](en/technical/inference-optimization.md) — Per-token latency reduction plan (82ms → 24ms)
- [Project Roadmap](../ROADMAP.md) — Full project plan
- [Stream Mode Status](en/technical/stream-mode-status.md)
- [Debug: argmax16](en/troubleshooting/debug-argmax16.md)

### 📈 Analysis Reports
- [Executive Summary](en/analysis/00-executive-summary.md)
- [Architecture](en/analysis/01-architecture-overview.md) | [Infrastructure](en/analysis/02-core-infrastructure.md)
- [Models & Inference](en/analysis/03-models-and-inference.md) | [KV Cache](en/analysis/04-kv-cache-subsystem.md)
- [Server](en/analysis/05-server-and-service.md) | [Quantization](en/analysis/06-quantization-training.md)
- [Testing](en/analysis/07-testing-quality.md) | [Security](en/analysis/08-security-audit.md)
- [Issues](en/analysis/09-issue-verification-matrix.md) | [Technical Debt](en/analysis/10-technical-debt.md)
- [Performance Benchmark](en/analysis/performance-benchmark.md)
- [Python vs Zig](en/analysis/python-zig-comparison.md)
- [Competitive Advantages](en/analysis/competitive-advantages.md)
- [Full Report](en/analysis/full-report.md) | [Analysis Report](en/analysis/analysis-report.md)

### 🛠 Developer
- [Contributing](../CONTRIBUTING.md)
- [Architecture](../README.md#architecture)

---

## 中文

### 🚀 快速开始
- [快速开始](../README.md#quick-start)
- [安装说明](../README.md#installation)
- [DeepSeek V4 快速修复](zh/user-guide/deepseek-v4-quickfix.md)
- [故障排查: DeepSeek V4](zh/user-guide/troubleshooting/deepseek-v4.md)
- [故障排查: 调度器](zh/user-guide/troubleshooting/scheduler.md)

### 🧠 DeepSeek MoE 架构
- [技术深度分析](zh/deepseek-moe/README.md) — 150GB 如何装入 48GB
- [应用场景](zh/scenarios/README.md) — 本地推理、隐私、边缘部署
- [SMELT 专家流式加载](zh/technical/smelt-flow.md)
- [4-bit + SMELT](zh/technical/4bit-smelt.md)
- [TTFT 优化](zh/technical/ttft-optimization.md)
- [推理速度优化](zh/technical/inference-optimization.md) — 逐 token 延迟优化方案
- [自动检测专家](zh/technical/auto-detect-experts.md)

### 📊 DeepSeek V4 修复历史
- [综合修复文档](zh/deepseek-v4/DEEPSEEK-V4-FIXES-AND-DETAILS.md)
- [Chat 分析与故障排查](zh/deepseek-v4/DEEPSEEK-V4-CHAT-ANALYSIS-AND-TROUBLESHOOTING.md)
- [优化路线图](zh/deepseek-v4/DEEPSEEK-V4-OPTIMIZATION-AND-ROADMAP.md)

### ⚙️ 技术文档
- [Continuous Batching 设计](zh/technical/continuous-batching-design.md) — Scheduler + BatchBuilder + Paged KV cache 架构
- [性能基准](zh/technical/benchmarks.md)
- [推理速度优化](zh/technical/inference-optimization.md) — 逐 token 延迟优化方案（82ms → 24ms）
- [项目路线图](../ROADMAP.md)
- [流式模式状态](zh/technical/stream-mode-status.md)
- [调试: argmax16](zh/troubleshooting/debug-argmax16.md)

### 📈 分析报告
- [执行摘要](zh/analysis/00-executive-summary.md)
- [架构](zh/analysis/01-architecture-overview.md) | [基础设施](zh/analysis/02-core-infrastructure.md)
- [模型与推理](zh/analysis/03-models-and-inference.md) | [KV Cache](zh/analysis/04-kv-cache-subsystem.md)
- [服务器](zh/analysis/05-server-and-service.md) | [量化](zh/analysis/06-quantization-training.md)
- [测试](zh/analysis/07-testing-quality.md) | [安全](zh/analysis/08-security-audit.md)
- [问题](zh/analysis/09-issue-verification-matrix.md) | [技术债务](zh/analysis/10-technical-debt.md)
- [性能基准](zh/analysis/performance-benchmark.md)
- [Python vs Zig](zh/analysis/python-zig-comparison.md)
- [竞争优势](zh/analysis/competitive-advantages.md)
- [完整报告](zh/analysis/full-report.md) | [分析报告](zh/analysis/analysis-report.md)

### 🛠 开发者
- [贡献指南](../CONTRIBUTING.md)
- [项目架构](../README.md#architecture)

---

## Project Meta
- [README](../README.md) | [Changelog](../CHANGELOG.md)
- [Contributing](../CONTRIBUTING.md) | [Code of Conduct](../CODE_OF_CONDUCT.md)
- [License](../LICENSE)

## Internal Specs
- [Production Deployment](.kiro/specs/production-deployment/)
- [Stream Mode Correctness](.kiro/specs/stream-mode-correctness/)
- [Stream Mode Performance](.kiro/specs/stream-mode-performance/)
- [Remaining Issues Fix](.kiro/specs/remaining-issues-fix/)
- [TTFT Optimization](.kiro/specs/ttft-optimization/)
