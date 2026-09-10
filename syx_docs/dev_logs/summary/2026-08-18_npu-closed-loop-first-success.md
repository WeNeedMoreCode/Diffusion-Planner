# 2026-08-18 NPU 闭环仿真首跑成功（50/50，final_score 0.9170）

## 概述

Diffusion-Planner 的 nuPlan 闭环仿真在 Ascend 310P（300I-DUO）上完整跑通并出分。本 session 覆盖：环境全链路搭建 → 数据/ckpt 落位 → 四个运行期故障击穿 → 成功验证 → 版本管理收尾 → 同步体系三次大坑修复。

## 完成的事

- **环境**：容器 `syx_dp` + py3.9 conda 环境 + torch 2.1.0/torch_npu 2.1.0.post17 + nuplan-devkit 依赖闭包（8 个"幽灵依赖"靠深度 import 逐个揪出，全记录在 [decisions/001](../../decisions/001-nuplan-devkit-dependency-deviation.md)）
- **数据**：nuPlan mini（64 db）+ nuplan-maps-v1.0 + HF ckpt 全部落位 `/data/syx_dp/`（根盘满 → 数据走 /data 的决策也是本期定的）
- **四个运行期故障**（按击穿顺序）：
  1. `mmengine` 缺失（normalizer→train_utils 顶层 import）
  2. bokeh 2.4.3 `np.bool8` × numpy 1.26.4（3.x 升级被 nuplan 的 `Figure` import 挡死 → 回退 2.4.3 + sed 补丁，见 [decisions/002](../../decisions/002-env-patches-as-monkeypatch.md)）
  3. **Ray 清空 NPU 可见性**：`number_of_gpus_allocated_per_simulation=0` 触发 Ray 把 worker 的 `ASCEND_RT_VISIBLE_DEVICES` 抹空 → `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` 解
  4. **MHA fast path 踩 310P 缺失算子**：`aclnnTransformBiasRescaleQkv` 无 310P 二进制（EZ1001）→ bool mask 改 float 加性 mask，用 fast path 条件列表第一条（float mask 直接否决）强制走分解路径
- **版本管理**：嵌套仓 `npu_port` 分支提交移植改动（464104fd）；ModelZoo 本地/fork/服务器三方对齐 `npu_adpat @ a3c42931e`
- **同步体系**（详见 remote-ssh-windows skill 踩坑记录）：磁盘满导致拉取器静默卡死（`df` 应第一步查）、Windows↔Linux 往返产生 7.5 万幻影改动 + 486 软链接损毁、单边切分支引发文件复活——最终确立"分支纪律"（停同步→双边切→开同步）

## 实测结果

### 事实结果（全部实测，来源 run4 = model_2026-08-18-03-58-43）

| 指标 | 值 |
|---|---|
| 场景成功 | 50 / 50（closed_loop_nonreactive_agents，one_continuous_log ×50） |
| **final_score（加权）** | **0.9170** |
| 分项 | drivable_area 1.0 / direction 1.0 / comfort 0.98 / progress 1.0 / route_progress 0.9806 / **no_collision 0.94** / speed_limit 1.0 / **TTC 0.88** |
| 单步规划时延（场景中位的中位） | 614.9 ms（≈1.6 Hz） |
| 场景内中位均值 / 最大 | 623.5 / 953.5 ms |
| 场景内 std 中位 / 最大 | 284 ms / 19 s（首步 JIT 预热） |
| 单场景墙钟 | 均值 179.8 s（4 worker 共享 device 4） |
| 总墙钟 | 41 min 52 s |

参照：论文 val14 NR 官方 89.87（1100+ 场景全集，不同场景集不可直接比）。

### 可信推理

- 单步 615ms 远离 10Hz 实时要求：由"输入 shape 全定长（FLOPs 恒定）+ MHA 走分解路径（非融合）+ CPU 数据组装在同链路"推出，瓶颈必在这三者的串行组合中——具体各占比未测。
- 场景间步时波动（std 284ms）：4 个 Ray worker 共享单卡 + CPU 争抢可解释全部波动量级（shape 恒定排除计算量因素）。

### 推测未严格证明

- 独占单卡 + 剔除 CPU 预处理后，纯模型前向可能显著低于 615ms（未做隔离压测）。
- TorchAir 图模式 / OM 离线对分解路径 MHA 的加速幅度（未试验）。
- 首步 19s 尖峰全部归因 JIT 算子预热（日志时间吻合但未逐项拆分）。

## 待办（残项，主方向见 handoff ②性能优化）

- 两边 ModelZoo 的 `.gitignore` 工作区 ignore 规则未提交（逻辑上属 `npu_adpat` 分支）
- bokeh `np.bool8` sed 补丁待收编 compat.py 猴子补丁（[decisions/002](../../decisions/002-env-patches-as-monkeypatch.md)）
- `GenPosePlus/om_wrappers.sync-conflict-*.py` 未比对去留
- 嵌套仓 `npu_port`（464104fd）仅存本地，未推远程

## 原始 stdout 索引

- 成功跑完整日志：[run_prints/2026-08-18_sim_run4_first_npu_success.log](../run_prints/2026-08-18_sim_run4_first_npu_success.log)（34K，服务器原件 `/data/syx_dp/sim_run4.log`）
- 结构化结果：服务器 `.../model_2026-08-18-03-58-43/runner_report.parquet` + `aggregator_metric/*.parquet`
- 失败跑日志（run1-3，排障过程）：服务器 `/data/syx_dp/sim_run{1,2,3}.log`
