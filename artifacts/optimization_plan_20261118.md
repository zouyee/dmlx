# Client tok/s 优化方案 — 由 2026-11-18 全天实测数据决定

## 测量结论（方案选择依据）

| 测量 | 结果 | 含义 |
|---|---|---|
| client/server gap（非 MTP，3 类请求×3 + burst） | **0.95-1.0×**，每 token 差 ≤1ms | 流式管线健康，client tok/s = 引擎计算速度，优化全在计算侧 |
| prefill 吞吐（4 引擎对照 + 16k chunk A/B） | 全体一致：~110 tok/s（swap 污染）/ **153-159 tok/s（净机）** | 计算天花板，所有引擎打平；kernel 层面无差异化空间 |
| 裸 decode | 14.5 tok/s（污染）/ 15-16（净机） | 基线 |
| DFlash2 采样 decode（accept GPU 化 + draft 块物化修复后） | 2k: 34.6 / 8k: 34.6-60 tok/s | 投机收益 2.3-4×，采样路径已无系统性 sync 浪费 |
| verify 相位 | 68-90ms/round，sg8/mma 两种形态收敛 157-198 GB/s | DRAM 流式天花板（2026-11-06 verdict 复核成立），**已结案** |
| agent 实测（pi/codex） | 每轮 system prompt ~11k token；进程内 prefix cache 命中时 TTFT 0.18s，**冷启/新会话 70-100s** | agent 体感的最大痛点是重复 prefill，不是 decode |

## 方案取舍（对照候选清单）

**P0 — Prompt/KV cache 磁盘持久化 + 跨会话复用** ← 采纳，client 体感收益最大
- 证据：agent harness 每轮带 ~11k 稳定 token；进程内快照已把重复 TTFT 打到 0.18s，但重启/新会话归零，一次 70-100s。落盘后等于归零。
- 已有基础：kvcache/ 量化 KV、prefix_snaps 快照机制、df2_cache_rows（drafter 投影行，进程内已跨请求复用）。落盘 = 快照序列化 + 启动时加载 + token 前缀校验。
- 预期：agent 场景 TTFT 70-100s → <1s（含冷启动）。这就是"persistent agent memory / KV 复用跨会话"项，单机版。

**P1 — 采样路径 predraft 重叠** ← 采纳，小改动
- 残余：采样轮 round ≈ draft 46-67 + verify 68-80 + accept 22-25ms；accept 残余主要是 bonus-row pull 的 GPU drain 等待。
- predraft 目前只对 greedy（argmax 块）；对采样路径 predraft 块 logits 可在 all-accepted（~60-73% 接受率）时把 draft 相位藏进 verify drain，round -20-30%。
- 注意此前实测 greedy predraft 默认开反而 -1 tok/s（采用率低）；采样路径采用率更高（bonus token 稳定），值得单独 A/B。

**不采纳 / 已做：**
- PhaseGate 分相并发：单机串行引擎，并发上限不敏感。搁置。
- GDN prefill 并行化：收益被 4 引擎一致的计算天花板封顶（净机 153-159），工程量大，性价比低。搁置。
- MoE 权重调度：qwen38-27B 是 dense 模型，不适用。
- 动态投机深度：**已在线实现**（MtpDepthController 按 cost(n)/E[tokens] 在线选深，p_i 跨深度池化）。
- ANE/CoreML、P/D 分离：按既定结论不采纳。

## 执行顺序

1. P1（小改动，直接提升 decode 2k/8k 各 ~20-30%）
2. P0（落盘格式 + 加载 + 校验；以 pi/codex 11k system prompt 为验收场景：重启 server 后首个 agent 请求 TTFT <2s）
3. P0 完成后复测全天对照矩阵，更新 README/HANDOFF 数字。
