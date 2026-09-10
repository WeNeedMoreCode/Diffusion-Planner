# Diffusion-Planner NPU 移植 — 文档与日志索引

非原生项目（上游 github.com/WeNeedMoreCode/Diffusion-Planner），文档隔离在 `syx_docs/`，不放进上游 repo 根。

## 核心文档

- [环境配置](setup.md) — 容器 / conda / 数据 / 同步，怎么搭
- [推理管线结构](designs/inference-pipeline.md) — 调用链 / 图模式组件 / 数据流 / 设计取舍（立体）
- [Diffusion-Planner 设计说明书](Diffusion-Planner_设计说明书.md) — 公开交付物（自包含）：上游推理架构（第一部分）+ 本仓 NPU 适配设计（第二部分）；编号引用式图表、术语表与接口速查表

## 目录用途

| 目录 | 用途 | 写入时机 |
|------|------|---------|
| `decisions/` | 关键决策记录（ADR） | 做决策时 |
| `dev_logs/summary/` | 已发生归档 | compact 时 |
| `dev_logs/handoff.md` | 当前活跃任务 | /compact 前 |
| `dev_logs/run_prints/` | 仿真运行 stdout 原始归档 | tee 存 |

## 索引

### Decisions
- [001 — nuplan-devkit 依赖选择性安装](decisions/001-nuplan-devkit-dependency-deviation.md)
- [002 — 环境兼容补丁收编为猴子补丁](decisions/002-env-patches-as-monkeypatch.md)
- [003 — MHA fast path 踩 310P 缺失算子：float mask 强制分解路径](decisions/003-mha-fastpath-float-mask.md)

### Summary
- [2026-08-18 NPU 闭环首跑成功（50/50，0.9170）](dev_logs/summary/2026-08-18_npu-closed-loop-first-success.md)
- [2026-08-19 单步 615ms 性能拆解（阶段/子块定位 + jit 窗口实测有害）](dev_logs/summary/2026-08-19_perf-breakdown-stage-timing.md)
- [2026-08-19 mapproc 向量化（552→417ms，累计 -32%，分数无回归）](dev_logs/summary/2026-08-19_mapproc-vectorization.md)
- [2026-08-19 TorchAir 图模式（408→268ms，累计 -56.5%，分数无回归）](dev_logs/summary/2026-08-19_torchair-graph-mode.md)
- [2026-08-20 adapt 缓存+向量化（267.6→141.1ms，累计 -77.1%，分数无回归；四轮全链路对比表）](dev_logs/summary/2026-08-20_adapt-vectorization-cache.md)
- [2026-08-21 fwd 侧图化：fast 采样器 + encoder 静态化 + cache_compile（141.1→86.6ms，累计 -85.9%，分数无回归；五轮全链路对比表）](dev_logs/summary/2026-08-21_fwd-graph-mode.md)
- [2026-08-28 收口轮：50 场景转默认 + sim log 修复 + RC 移植攻坚 + Syncthing 事故恢复 + sequential 基准](dev_logs/summary/2026-08-28_rc-port-and-tooling.md)
- [2026-08-31 R7：OM 离线化全链（DUO 80.6ms 反超 torchair、RC 50 场景 0.9171 首过；fold+simplify 31.6→5.55ms；stream 泄漏/pyACL context/条件 restore 三 bug 定位链）](dev_logs/summary/2026-08-31_r7-om-offline.md)
- [2026-09-02 RC 诊断轮：bench 三工具族（om/step/launch）与发射邮费证据链（RC 算力与 DUO 逐位同、邮费 2×、CPU 1.34×）；下轮路线 1 削发射](dev_logs/summary/2026-09-02_rc-diagnosis-bench-tools.md)
- [2026-09-03 R8：削发射轮 encoder.om v3（norm/pos/route 并图；TASK_QUEUE=2 与 route 搬 CPU 双否决；ATC slice 写回静默 miscompile）；RC 121.3→102.5、DUO 80.6→77.5，双板 0.9171](dev_logs/summary/2026-09-03_r8-launch-cut.md)
- [2026-09-04 R9：CPU 段削减——host 侧直通 + 地图不变量缓存（RC ①-6.2/②-6.3 分步实测；dispatch/launch/搬运概念澄清；SoC 误判教训）；RC 102.5→90.0、DUO 77.5→70.5，双板 0.9171；尾声 skill 三合一为 inference-delivery](dev_logs/summary/2026-09-04_r9-cpu-side.md)

### Handoff
- [dev_logs/handoff.md](dev_logs/handoff.md) — /compact 三段式 prompt 生成器（① Compact 参数 / ② Post-compact 首句 / ③ Export 标题）
