# 2026-09-03 R8：削发射轮——encoder.om v3（norm/pos/route 并图），RC 121.3→102.5ms

> R7 诊断定了路线 1（削发射次数）。本轮执行：方案表第 2 项"RouteEncoder 搬 CPU"被实测**否决**（CPU 算力瓶颈，11× 负优化），改并图路线一次做完三项——**encoder.om v3 直吃 adapt 原始输出，norm+pos+route 全部烤进一张图**。RC 121.3→102.5ms（-15.5%），DUO 80.6→77.5ms，双板 50 场景 score 0.9171 逐位持平。途中抓到 ATC 对 slice 写回的**静默 miscompile**（编译成功、ort 正确、OM 输出错）。

## 否决速览（两个失败的尝试 + 最终路线）

| | 本来想做什么 | 实测结果 | 为什么不行 |
|---|---|---|---|
| **TASK_QUEUE_ENABLE=2** | CANN 任务队列把投递异步化，削 RC 的 2× 发射邮费 | micro 邮费 -58%，但真实负载（分段重放）**全线 +19%** | 异步化的代价是**排空变贵**。闭环节奏是"发射一小段→OM 调用强制排空→再发射"，发射收益没攒够就要排空、伤害每次全付。micro 探针纯流水计时恰好只测到收益测不到代价——micro 与真实负载必须连跑对照 |
| **route 搬 CPU**（原方案表第 2 项） | route 段在 NPU 上 10.7ms 全是邮费；按"RC host 派发单价 17us × 100 op ≈ 2ms"估搬 CPU 省 8ms | RC 空机 CPU eager **42-50ms，比 NPU 4.5ms 慢 11 倍**（线程 1~8 全一样；空机排除争用） | 估算拿错单价：把 **NPU host 派发单价**当成 **CPU 执行单价**。这条链 NPU 上瓶颈是邮费、CPU 上瓶颈是算力（aarch64 跑 GELU/LN/小 GEMM ~1 GFLOP/s，每 op ~1ms）——两头瓶颈类型不同，不能外推 |
| **→ 并图吸收（最终）** | 两头"单价"都贵，就把**发射次数**砍到零 | norm/pos/route 三段进 encoder.om，一次投递 | （成功：RC 121.3→102.5、DUO 80.6→77.5，score 持平） |

## 事实：前置试探 TASK_QUEUE_ENABLE=2 → 否决

DUO 卡 2（同卡对照，micro 与真实负载连跑）：

```
bench_launch (us/op)   baseline   TQ=2      bench_step (median ms)  baseline   TQ=2
npu_micro_stream         21.35    14.46     norm                     5.065     6.940
cpu_micro                 9.85     9.63     pos                      0.746     0.855
npu_big_mm              265.82   267.03     route                    3.615     3.815
npu_micro_sync           91.09    82.07     sum                     20.661    24.520   (+19%)
```

- micro 层面投递邮费 stream−cpu = 11.5→4.8us（-58%），两个"理应无关"标尺（cpu_micro/big_mm）纹丝不动——确实只动了投递路径
- 但 bench_step（每段尾 synchronize，与闭环 fwd 的 sync 结构同构：OmBody infer 前强制排空）**全线变慢 19%**
- 结论：任务队列把投递异步化但排空变贵；闭环是"发射一小段→OM 排空"节奏，发射收益兑现不了、排空伤害全吃。RC 不再试（sync 结构相同；路线 1 落地后小算子串消失，作用对象也没了）

## 事实：RouteEncoder 搬 CPU 被否决（bench_route，RC 空机）

| 项 | RC 空机实测 |
|---|---|
| NPU eager + d2h | **4.547 ms** |
| CPU eager + d2h | **50.069 ms**（11× 负优化） |
| 线程扫描 t=1/2/4/8 | 42.1~45.1（线程无关） |
| cpu_probe | 8.637（119 为 6.4，RC CPU 慢 1.34× 与 R7 诊断一致） |
| parity | max_abs 1.5e-3（0.039%，数值本无问题） |

119 上同款 CPU 路径 174ms = RC 底数 42ms × vllm 争用放大。**方案表错在哪**：按"NPU host 派发单价 17us × 100 op ≈ 2ms"估 CPU，错把派发单价当 CPU 执行单价；该 op 链（~40 MFLOPs）在 RC aarch64 上是算力瓶颈（~1 GFLOP/s 吞吐）。教训：**小算子串的两头——NPU 上是邮费、CPU 上是算力——要实测，不能拿另一头的单价外推**。

## 事实：encoder.om v3 演进（两步）

**v2（route 并图）**：`_encode_route` = RouteEncoder 静态版（全量算 + **batch 级** mask_b 置零——注意与 fusion encoder 的 per-row 置零语义不同，route 上游是整 batch 过滤、invalid P 行保留 bias 垃圾即训练语义）。encoder.om 9 输入 2 输出。DUO：ort 对拍 1.2e-06（静态化数学等价实锤）、bench_step sum 20.66→18.87、5 场景 0.9941 与改前逐位一致。

**v3（norm+pos 并图）**：EncoderRawExportWrapper 直吃 **7 个 raw adapt 键**；图内 ①per-key norm（json mean/std 烤广播常量，全零行判据在原始数据上=上游语义，不在 json 的 key 透传）→ ②pos 三提取（函数式）→ ③StaticEncoderBody+route → ④current_states 组装（norm 后 ego[:4]+neighbors 末帧[:4]）；**3 输出**（encoding/route_encoding/current_states），decoder 采样锚点也零 norm 依赖。planner 侧 DP_OM 跳过 ObservationNormalizer（norm=0.0），capture 移到 norm 前（**raw 语义，旧 norm 后 capture 全部作废**）。

对拍（DUO，raw capture）：ort 侧 encoding 4e-5 / route 1.9e-6 / **current_states 逐位 0**；om 侧 encoding 0.632%（v2 同量级）/ route 0.078% / cs 0.032%，全 PASS。encoder.om 6.1MB、**2.44ms/次**（v1 2.28ms 吃 norm 后输入 → v3 2.44ms 全包，norm+pos+route 只花 0.16ms 图内代价）。

## 事实：ATC 静默 miscompile（本轮最重要的坑）

v3 首次导出：**ort 三路全对但 om:encoder.encoding 32.5% FAIL**（同图 route/cs 两路过）。mini 图隔离（pos 三函数单独导 onnx→atc→ort vs om，两分钟）：

```
ort apos/spos/lpos: 0 / 0 / 2.4e-07   （数学全对）
om  apos/spos/lpos: worst 全在 type one-hot 位——ref 1.0 got 0.0
```

根因：`clone() + pos[..., -3] = 1.0` 这类 **slice 写回**被 trace 成写回型节点，ATC 310P 上静默算错（不报错）。修复：pos 三函数**函数式重写**（切片只读 + cat 拼接，值恒等，eager/torchair/OM 全路径安全）。修复后 om:encoding 0.632% 回到融合噪声水平。

通用教训（已进 skill FAQ 11 + validation-gates）：**"ATC 成功 + ort 正确"不保证 OM 正确**；ATC 引入的错只有 OM-vs-eager 对拍能抓到，判据要含结构性检查（zero-viol）。判别信号：偏差 >10% of |ref|max、错误集中在特定构造模式位置、ort-vs-eager 全对 → mini 图隔离，别按精度问题调参。

## 事实：双板 50 场景全量（v3）

| | R7 OM loop | R8 v3 | 变化 |
|---|---|---|---|
| DUO 单步中位 | 80.6ms | **77.5ms** | -3.1 |
| RC 单步中位 | 121.3ms | **102.5ms** | **-18.8（-15.5%）** |
| DUO score (n=50) | 0.9171 | **0.9171** | 逐位持平 |
| RC score (n=50) | 0.9171 | **0.9171** | 逐位持平 |
| RC wall | ~40min | concurrent 9.3min | |

bench_step 演进（median，DUO）：R7 sum 33.6（norm 5.95/pos 1.81/route 5.07 三段在列）→ v2 18.87（route 段没了）→ **v3 11.68**（norm/pos 段也没了；enc_om 4.05 含全部）。**RC v3 sum 10.72 反超 DUO 11.68**（RC 空机 vs DUO vllm 共享卡的噪声差）——R7 诊断结论完全兑现：RC 慢在发射次数不在算力，砍掉发射后两板 fwd 同价。两板剩余 25ms 差距全在 CPU 段（adapt 1.34× + post），RC 的新大头是 adapt。

## 决定：v3 交付形态与边界

- capture 语义变更（norm 后 → raw）：export/bench_step 全走 raw capture；`bench_route.py` 标注为历史工具（期望 norm 后输入）
- 旧 encoder.om 与新代码不兼容（输入数变了）→ 报错即重导，行为显式；dit_body/dit_loop 两图不受影响无需重导
- torchair 路径不受 v3 影响（StaticEncoderBody 与 cache 均未动；pos 函数式化值恒等）

## commit（R8 全程）

小 DP npu_port：`4abd6cd`（route CPU 副本，后被否决）→ `f3d9050`（v2 route 并图+_encode_route）→ `fcb6ed5`（v3 raw 直进）→ `630b00b`（pos 函数式修复 ATC miscompile）。外层 npu_adpat：`6fdc1428b`（bench_route）→ `5d4ff323e`（v2 工具族）→ `178c351bb`（v3 工具族）→ `1db720d0e`（bench_route 历史标注）→ `b16e95fe9`（prepare 三 bug 补 commit——**调试期修复忘了推，RC 撞上第一个，教训：跑通后立刻 commit，不留未提交的工作树**）。

## 待办沉淀

- skill 已增补：om/FAQ 11（ATC slice 写回 miscompile）+ validation-gates 诊断流程第 2/5 条（非良性偏差判别 + 函数式重写修复手段）
- MSprof/onnxsim 案例进 skill（低优先，FAQ 11 已覆盖本轮最险的坑）
