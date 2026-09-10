# 2026-08-19 mapproc 向量化：单步 552 → 417ms，分数回归无损

## 概述

性能优化第二轮：`map_process._interpolate_points` 从 shapely 逐点插值改为 numpy 弧长参数化向量化（前一轮拆解定位的最大单块，[summary/2026-08-19 性能拆解](2026-08-19_perf-breakdown-stage-timing.md)）。等价性 900 用例对照 + 5 场景端到端回归双双通过。

## 事实结果（全部实测）

### 等价性（`test_interpolate_equiv.py`，容器内）

- 300 条随机折线（含注入重复点/零长段）× 3 档 num_point = **900 用例，最差绝对差 7.1e-15**（纯浮点求和顺序差异），阈值 1e-9 通过

### 端到端对照（Run C vs Run D，同配置：单线程独占 device 5、jit off、DP_LIMIT=5、one_continuous_log）

| 指标（warm 中位 / ms） | Run C 基线 | Run D 向量化后 | 变化 |
|---|---|---|---|
| **mapproc** | 221.6 | **73.2** | **-67%** |
| mapquery（未动，对照组） | 53.3 | 54.9 | 持平 ✓ |
| adapt | 333.1 | 185.1 | -44% |
| fwd | 202.1 | 212.7 | +10（run 间波动，见推理） |
| **total** | 552.4 | **417.1** | **-24.5%** |
| **final_score** | 0.9942 | **0.9941** | 分项全一致 |

Run D 完整瀑布（warm 中位）：adapt 185.1（mapproc 73.2 + mapquery 54.9 + agents 38.3 + route 8.4 + 杂项 ~10）+ fwd 212.7 + norm 9.4 + post 8.8 ≈ total 417.1。

### 改动

- `map_process.py::_interpolate_points`：`LineString` + 逐点 `interpolate()` → `cumsum` 弧长 + `searchsorted` 定段 + 线性插值，全 numpy 向量化；新增 <2 点/零长线两个退化分支（旧实现会 raise，属行为超集）
- 新工具：`test_interpolate_equiv.py`（新旧实现等价对照）、`read_scores.py`（读实验目录 aggregator parquet 出分）

### 累计战果

单步中位：**615ms（run4 原始）→ 552（jit 窗口关）→ 417（mapproc 向量化），累计 -32%**，分数无回归。

## 可信推理

- 向量化收益 -148ms 与前一轮"shapely 逐点 Python 调用主导 mapproc"的代码结构推断吻合，因果链闭合
- fwd +10ms 为 run 间波动：Run D 时段 device 2/3 有邻居进程 35–44% 负载（发车前 npu-smi 记录），且 fwd 在 replay 纯循环下 192–203ms（[replay nojit](../run_prints/2026-08-19_bench_replay_nojit.log)），Run D 的 212.7 仍在波动包络内；两轮 adapt 数据（mapquery 53→55）同时段稳定，说明波动源在 NPU 侧不在 CPU 侧
- final_score 第 4 位小数差异（0.994160→0.994098）由 1e-15 级输入差经闭环混沌放大——两轮分项指标逐项相等（collision/TTC/drivable/speed 全 1.0，progress 0.98131→0.98111）
- **各阶段中位数近似可加但数学上不保证**（median 不满足线性）；本轮数据各阶段分布窄恰好对齐（416.0 vs 417.1），跨 Run 对比以同口径中位为准

## 下轮方向（用户已拍板：TorchAir）

- fwd 213ms 已占总耗时 51%，decoder（扩散采样循环）占 fwd ~85%——TorchAir 图模式整图编译 decoder 是下一个大池子
- 前置调研：容器内 torchair 包版本/310P 支持；DPM-solver 多步循环 + randn 的整图可行性；动态 shape 风险（输入全定长是利好）
- 次级遗留：mapquery p90 554ms 长尾（偶发慢查询，中位 55ms 本身不大）

## 原始 stdout 索引

- [2026-08-19_bench_runD_vec.log](../run_prints/2026-08-19_bench_runD_vec.log)（Run D 计时，服务器原件 /data/syx_dp/bench_runD_vec.log）
- Run C 基线与对照日志见 [summary/2026-08-19 性能拆解](2026-08-19_perf-breakdown-stage-timing.md)末尾索引
- Run C/D 实验目录：服务器 `/data/syx_dp/exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/one_continuous_log/diffusion_planner_release/model_2026-08-19-02-44-59`（C）与 `model_2026-08-19-03-*`（D）
