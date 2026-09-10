# 2026-09-02 RC 诊断轮：bench 三工具族与"发射邮费"证据链（R7 尾声）

> R7 收口后用户追问"RC 比 DUO 慢的 40ms 到底在哪"。本轮造了三件测量工具把这个问题拆到微秒级并完全闭合：**RC 的 NPU 算力与 DUO 逐位相同，慢在 kernel 投递邮费（2 倍）与 CPU（1.34 倍）**；OM 图不受影响（一次投递）。据此定下轮路线 1（减少发射次数）。

## 事实：三件 bench 工具（全部进外层仓库）

| 工具 | 测什么 | 口径 |
|---|---|---|
| `bench_om.py` | 三 OM 图单图时延；`--mode mixed` 复刻闭环数据路径（NPU 源张量→每输入 D2H→infer→结果 H2D→sync），d2h/infer/h2d/sync 分段报告 | 1000 次均值；host=纯执行参照 |
| `bench_step.py` | 脱离 nuplan 重放一个 planning step：norm/pos/enc_om/route/prep/dit_om/invnorm/post_sim 逐段计时 + `cpu_probe`（numpy+对象访问固定负载 = CPU 算力标尺，外推 adapt） | capture 真实输入 200 次 median |
| `bench_launch.py` | 发射邮费探针五项：`cpu_micro`（host 派发）/`npu_micro_stream`（派发+投递，流水）/`npu_small_mm`（小 kernel 连发）/`npu_big_mm`（1024³ matmul = AI Core 算力标尺）/`npu_micro_sync`（单 op 全往返） | us/op，块均摊 |

概念锚点（本轮教学沉淀）：一次 `x+1`（x 在 NPU）= ①Python 解释 + ②torch 派发层（校验/推 shape/选 kernel）合称 **host 派发（dispatch）** + ③**投递（launch）**：走驱动把工单写进 NPU 命令队列 + ④NPU 执行。①② 与目标设备无关；③ 只在目标为 NPU 时存在且与任务大小无关（固定 ~9-35us/次）；小算子串的性能 = 邮费 × 个数。

## 事实：DUO 基线（卡 2，共享卡有长尾，median 口径）

- bench_om 1000 次：encoder 2.35 / dit_body 1.18 / dit_loop 5.65 ms；mixed 分段（d2h/infer/h2d/sync）：encoder 0.97/2.41/0.20/0.11、dit_loop 1.11/5.77/0.32/0.16——搬运大头在 d2h（每输入 ~0.2ms 固定成本）
- bench_step 200 次：norm 5.95 / pos 1.81 / enc_om 5.92 / route 5.07 / prep 0.99 / dit_om 12.81 / invnorm 0.16 / post_sim 0.19 / sum 33.6；cpu_probe 6.47
- bench_launch：cpu 10.3 / stream 19.1 / small_mm 27.9 / big_mm 223 / sync 78.9 us

## 事实：RC 数据与三锁证据链

```
                DUO      RC      比值
npu_big_mm     223.00   221.75   1.00×   ← 锁1：AI Core 算力逐位相同
cpu_micro       10.31    17.10   1.66×   （host 派发）
npu_micro_stream 19.08   34.55   1.81×
npu_small_mm    27.91    49.01   1.76×   ← 锁2：发射类全部 ~1.8×
npu_micro_sync  78.87   114.94   1.46×
纯投递邮费（stream−cpu）：8.8 → 17.5 us = 2.0×
```

锁 3：bench_step 受害段比值 norm 1.93× / pos 1.85× / route 2.11×，全部落在 host 1.66× 与邮费 2.0× 之间（段内 host/NPU 发射配比决定落点）；`enc_om`/`dit_om` 两板同量级（图一次投递，邮费无关）。

闭环账完全对上：fwd 由段值拼出 DUO 15.5(实测 17)/RC 24.8(实测 26)；adapt 51-57×1.34(cpu_probe 比)=68-76(实测 67)。**RC +40ms 最终归因：CPU 1.34×（adapt 大头 + prep/post）+ 投递邮费 2×（norm/pos/route 小算子串，每步几百次发射 × ~9us 差价）+ sync 路径 1.46×；NPU 算力零贡献。**

## 事实：过程中修掉的工具 bug 与新坑

- `bench_om` free_resource 拆 torch context（第一个 InferSession 复用 PTA context，free 即毁）→ 不主动 free（skill checklist 既有条款的又一实证）
- **protobuf 混链**：`import torch_npu`/`import acl` 必须都在第一个 InferSession 之前——任一反序会在 RC 上双 libprotobuf 混链 abort（`CHECK failed: file != nullptr`）；DUO 无感。闭环/ctx_probe 的顺序天然正确所以从未触发
- **RC 上 jit 在线编译通道不可用**：bench_step 漏了 `torch.npu.set_compile_mode(jit_compile=False)`，normalizer 的 NonZero→Cast 触发 tbe 在线编译，tbe 的 multiprocessing 队列在 RC 上 `qsize→NotImplementedError` → 算子编译 500002。planner.py 第一天就设了 False 所以闭环从未踩到——**RC 上任何"新形状触发在线编译"都是雷**（比 randn V2 kernel 问题面更宽）
- `bench_launch` 输出口径自纠：50-op 块值应直接输出 us/op（工具输出必须原样可贴，不做事后换算）

## 决定：下轮路线 1（减少发射次数）

| 方案 | 做法 | RC 预期 | 工作量 |
|---|---|---|---|
| RouteEncoder 搬 CPU | `_om_sample` 里 route_lanes 在 CPU 上 eager 跑（~100 op × 17us ≈ 2ms vs NPU 10.7ms） | -8ms | 小，先做 |
| norm 并进 encoder.om | 归一化 mean/std 常量烤进图，encoder 直吃 adapt 原始输出；mask 全量算+置零（StaticEncoderBody 同款） | -12ms | 中 |
| pos 搬 CPU | norm 并图后数据流经 CPU，pos 小 op 顺势 CPU | -3ms | 小 |

合计 RC 121 → ~97ms（DUO 80.6 → ~73ms 同赚）。每步对拍 + 5 场景。前置零成本试探：`TASK_QUEUE_ENABLE=2` 重跑 bench_launch 看邮费。路线 2（RC 驱动/固件升级，邮费或减半）由用户权衡验收机风险后定。

## commit（外层 npu_adpat，本轮 10 个）

`6ed3552e3` bench_om → `5d62f64f2` mixed 模式 → `97e189938` 分段计时 → `b5c1c33f8` 免 free_resource → `f54cb0be5` torch_npu 前置 → `db1faac87` import 顺序 → `90cc723da` bench_step → `fe40a175f` jit_compile=False → `ca20b3397` bench_launch → `9b7d14756` us/op 输出。另有 R7 收口尾：`ee69f926c`（README RC 行）。

## 待办沉淀（未做，待用户点头）

- skill 增补：rc_device_toolkit 加"RC jit 在线编译不可用（tbe qsize 崩）"；CONTEXT_MANAGEMENT 加"torch_npu/acl import 必须先于第一个 InferSession（protobuf 混链）"与"restore 不得按输入设备做条件"（R7 案例）
- RC torchair 100min wall 大头验证（低优先，无数据）
