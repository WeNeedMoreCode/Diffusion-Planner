# 2026-08-19 单步 615ms 性能拆解：分阶段计时 + jit 窗口实测有害 + mapproc 定位

## 概述

性能优化第一轮：给 planner 调用链加分阶段计时（`DP_STAGE_TIMING` 环境开关），跑 3 组对照仿真 + 捕获输入重放（replay）A/B，把 run4 的单步中位 615ms 拆解到阶段和子块，落地一项免费优化（jit 窗口默认关，-10%），并定位下一轮两大优化目标的精确位置。

## 事实结果（全部实测）

### 阶段拆分（单步，中位 / ms，warm 步 = 剔除 >2s 首步预热）

| Run | 条件 | n(warm) | adapt | norm | fwd | post | total |
|---|---|---|---|---|---|---|---|
| A | 4 worker 共卡，device 5，jit=1 | 88 | 375.8 | 8.0 | 251.8 | 9.2 | **676.4** |
| B | 单线程独占，device 5，jit=1 | 737 | 331.7 | 9.1 | 250.4 | 9.2 | **608.7** |
| C | 单线程独占，device 5，jit=0 | 744 | 333.1 | 7.6 | 202.1 | 8.8 | **552.4** |

- adapt = `planner_input_to_model_inputs`（CPU 数据组装：agent 处理 + 地图查询/处理 + 转张量）；fwd = 模型前向（含边界 sync）；norm/post 各 <10ms
- 各 Run 详见 [run_prints](../run_prints/)（runA_4worker / runB_1thread_jit / runC_1thread_nojit）

### replay 纯前向 A/B（捕获真实输入离线重放，30 iters × 4 输入，device 5）

| jit 窗口 | full p50 | encoder p50 | decoder p50 |
|---|---|---|---|
| =1（旧基线条件） | 243.8–246.6 | ~25 | **~219** |
| =0（纯预编译内核） | 192.4–202.7 | ~25 | ~170 |

- 4 份捕获输入（4 个 Ray worker 各存一份，均 122KB，shape 逐字节一致）数字高度一致
- replay fwd（245）与 in-situ fwd（250.4）吻合 → 捕获重放法可信

### adapt 子块拆分（Run C，warm 中位 / ms，占 adapt 333ms 比）

| 子块 | 中位 | 占 adapt | 占单步 552ms |
|---|---|---|---|
| **mapproc（`map_process`）** | **221.6** | **67.0%** | **40.1%** |
| mapquery（`get_neighbor_vector_set_map`） | 53.3 | 16.1% | 9.7% |
| agents（`agent_past_process` 等） | 38.2 | 11.6% | 6.9% |
| route（`route_roadblock_correction`） | 8.2 | 2.5% | 1.5% |
| to_tensor / ego | 1.7 | ~0 | ~0 |

### 落地的代码改动

- `planner.py`：`DP_JIT_WINDOW` 默认 `"0"`（原 `"1"`）——jit 窗口默认关闭，恢复旧基线行为需显式 `DP_JIT_WINDOW=1`
- 新工具：`bench_dp_forward.py`（捕获输入重放压测）、`analyze_stage_log.py`（计时日志统计）、`planner.py`/`data_processor.py` 的 `DP_STAGE_TIMING` 分阶段/子块计时、`DP_CAPTURE_DIR` 输入捕获、runner 脚本 `DP_DEVICE/DP_LIMIT/DP_THREADS` 覆盖

### 插曲（与主线无关，已闭环）

- 本地 Syncthing 发送端停摆（连接正常但改动不推送）→ 两边重启治愈；根因推测 watcher 静默失效，排障记录已入 remote-ssh-windows skill
- MSYS ssh 中文 HOME 打不开 known_hosts → `ssh -F /c/sshkeys/config 119` 固化，已入 skill + memory

## 可信推理

- **jit 窗口真实代价 ~48ms/步**：replay A/B 差值（245-197）与 in-situ Run B/C fwd 差值（250.4-202.1）两个独立测量一致。decisions/003 旧表述"无效但无害"的"无害"不成立，已在原文勘误
- **4-worker 共卡代价 ~10%**（608.7→676.4 中位）：修正 [summary/2026-08-18](2026-08-18_npu-closed-loop-first-success.md) 对场景间波动（std 284ms）"主要由共卡争抢解释"的归因——争抢只解释一小部分，波动主体在 adapt 的 p90 长尾（856ms，mapquery/mapproc 的偶发慢查询）
- **fwd 是纯计算瓶颈**：p10–p90 = 244–270（jit=1）/199–208（jit=0），极窄且与并发数无关——shape 全定长、FLOPs 恒定的推断由此坐实
- **decoder 占前向 ~89%**（219/245）：encoder 仅 ~25ms，将来图模式/算子级优化应聚焦扩散采样循环
- mapproc 222ms 的构成由代码结构推出主导嫌疑：`_interpolate_points` 对每条 polyline 逐点调用 shapely `line.interpolate`（每 element 3 条线 × max_points 次 Python 级调用）+ `_lane_polyline_process` 逐 element numpy 循环——**未逐函数 profile，属强推理**

## 推测未严格证明

- mapproc 向量化（numpy 弧长参数化替代 shapely 逐点插值）可把 222ms 压到几十 ms——方向明确，未实施
- TorchAir 图模式对 decoder（~170ms）的加速幅度——未试验
- Run A 首步最大 116s（run4 为 19s）来自 4 worker 并发 JIT 编译竞争——时间点吻合，未逐项拆分
- mapquery 的 53ms 有多少可被缓存消除（ego 每 0.5s 移动 5–8m，tile 级增量缓存命中率未知）

## 下轮优先级（按 池子大小 / 风险 排序）

1. **mapproc 向量化**（池子 222ms，纯 numpy 重写不动语义，风险低）
2. **decoder 图模式**（池子 ~170ms，需 TorchAir/310P 支持调研）
3. mapquery 增量缓存（池子 53ms，需评估命中率）

## 原始 stdout 索引

- [2026-08-19_bench_runA_4worker.log](../run_prints/2026-08-19_bench_runA_4worker.log)（4 worker 计时，服务器原件 /data/syx_dp/bench_4w.log）
- [2026-08-19_bench_runB_1thread_jit.log](../run_prints/2026-08-19_bench_runB_1thread_jit.log)（单线程 jit=1）
- [2026-08-19_bench_runC_1thread_nojit.log](../run_prints/2026-08-19_bench_runC_1thread_nojit.log)（单线程 jit=0，含 adapt 子块）
- [2026-08-19_bench_replay_jit.log](../run_prints/2026-08-19_bench_replay_jit.log) / [.._nojit.log](../run_prints/2026-08-19_bench_replay_nojit.log)（replay A/B）
- 分析命令：`python analyze_stage_log.py <log>`（容器内，脚本在仓库根）
