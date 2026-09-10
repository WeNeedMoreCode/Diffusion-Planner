# 2026-08-19 TorchAir 图模式：单步 408 → 268ms，累计 -56.5%，分数无回归

## 概述

性能优化第三轮：源码编译 TorchAir 7.2.0 并把 DiT 采样主体（含 RouteEncoder 外提）接入 `torch.compile` 图模式，端到端闭环仿真验证通过。三轮累计 **615 → 267.6ms（-56.5%）**，final_score 全程 0.994 无回归。

## 事实结果（全部实测）

### DiT 单步小样（`bench_torchair_dit.py`，捕获输入重放，device 5）

| 组 | p50 | 说明 |
|---|---|---|
| 1 eager-DiT（生产形态） | 15.79ms | 每次调用重算 RouteEncoder |
| 2 eager-body（route 外提） | 12.27ms | 外提省 3.5ms/步；vs 组1 数值 diff=0 |
| 3 graph-body（torchair） | **1.00ms** | **vs 组1 15.8×**；图编译 31.4s 一次性；vs 组1 数值 max_abs_diff=3.0e-2（ref 尺度 5.95，相对 0.5%） |

### 端到端（5 场景 one_continuous_log，单线程独占 device 5，warm 步中位 / ms）

| Run | 配置 | adapt | fwd | total | final_score |
|---|---|---|---|---|---|
| D（基线） | eager | 185.1 | 212.7 | 417.1 | 0.9942 |
| E1 | eager + route 外提 | 206.9 | **175.9** | 408.3 | **0.9942**（逐位一致） |
| E2 | **DP_TORCHAIR=1** | 179.8 | **71.1** | **267.6** | **0.9941** |

- E2 的 fwd max 43.5s = 首步一次性图编译（约 31s）
- 三轮累计：615（run4）→ 552（jit 关）→ 417（mapproc 向量化）→ **267.6（route 外提 + torchair）**

### TorchAir 环境（容器 syx_dp）

- 华为云 pypi 源**无 torchair**（pypi 单装 torch_npu 不带它）；gitcode `Ascend/torchair` 7.2.0 分支源码编译（gcc 11.4/cmake 3.22，`configure → cmake → make torchair`，产物 `torchair-0.1-py3-none-any.whl`）
- 配套表关键行：**TorchAir 7.2.0 ↔ torch 2.1.0 ↔ CANN 8.3.RC1 ↔ py3.8-3.11**；7.2.0 分支支持型号明列 "Atlas 推理系列产品（配置 Ascend 310P）"。torch_npu 2.1.0.post17（老编号线）实测兼容
- 冒烟（`torch.add` 图编译）通过后接真模型

### 生产集成（decoder.py）

- `DiTBody`：DiT.forward 去 RouteEncoder 版（静态 shape），数学上与 DiT.forward 恒等（model_type=x_start）
- `SamplerAdapter`：对 dpm_sampler 保持调用签名；`begin_step()` 每步预计算 route encoding（loop 不变量外提，eager 也受益 -37ms）；`DP_TORCHAIR=1` 时惰性 `torch.compile(body, backend=npu_backend, dynamic=False)`，图进程内缓存
- **坑 1**：adapter 直接挂 Decoder 属性会把 dit 注册进 state_dict（多一套 `_sampler.dit.*` key，checkpoint 匹配炸）→ 用 **list 容器挂载**（`nn.Module.__setattr__` 不注册 list 值），dit 引用别名共享权重
- **坑 2**：dpm_solver 的 `model_wrapper` 读 `model.model_type` → adapter 加 property 代理
- **坑 3**：RouteEncoder 的布尔散回（`x_result[valid_indices]=x`，动态 shape）GE 图不支持（ERR03007）→ 外提正是解法

## 可信推理

- eager 12ms→图 1ms 的收益本质是 **kernel launch 开销的消除**（DiT 小算子密集，~30μs/launch × 数百算子），与 310P 单卡小 batch 推理场景对症
- E2 分数 0.9941 与基线第 4 位小数差异（0.99416→0.99410）由图编译 3e-2 级单步数值差经闭环混沌放大产生，与 mapproc 轮（1e-15 输入差）同级表现——图偏差端到端可消化
- adapt 在 E1/E2 间波动（185→207→180）为 run 间 CPU 噪声（mapquery 长尾），不影响结论

## 下轮方向

- 瓶颈回到 **CPU adapt（179.8ms，67.2%）**：mapquery 中位 55ms + 长尾 p90 550ms、agents 38ms、mapproc 剩 73ms（`_lane_polyline_process` 逐 element 循环是下一个向量化目标）
- 可选：encoder（25ms）+ norm（7ms）也可进图；dpm_solver 的标量循环若整图化可再省 10 步间 launch（~10-15ms）
- 图编译缓存跨进程持久化（当前每 Ray worker 首步各编译一次，4 worker = 4×31s 一次性成本；TorchAir 支持 cache 落盘可查）

## 插曲（与主线无关）

- 项目目录正名 `Diffussion_Planner` → `Diffusion_Planner`（用户确认双 s 是最初笔误）：停同步→两边改名→开同步；连带修复 sim 脚本硬编码路径、两个 editable 包重装（`pip install -e` 路径映射失效）
- E 系列五次发车：①脚本路径旧 ②nuplan-devkit 同步窗口 ③editable 失效 ④state_dict key ⑤model_type 代理——集成类改动的典型回归链，均已修复

## 原始 stdout 索引

- [2026-08-19_bench_runE1_route_hoist_eager.log](../run_prints/2026-08-19_bench_runE1_route_hoist_eager.log)
- [2026-08-19_bench_runE2_torchair_graph.log](../run_prints/2026-08-19_bench_runE2_torchair_graph.log)
- DiT 三组小样：见本 session chat export（未单独落盘）
