# 推理管线结构（NPU 图模式形态）

> **职责**：闭环仿真一步推理的组件结构、数据流、性能形态与设计取舍。
> 时间线上的实测数字与优化过程见 summary/2026-08-19 三篇 + 0820/0821 两篇；本文只讲"现在长什么样、为什么"。

## 1. 调用链总览

```
nuplan simulation step (0.5s)
 └─ DiffusionPlanner.compute_planner_trajectory        [planner/planner.py]
     ├─ ① adapt   planner_input_to_model_inputs        [data_process/data_processor.py]
     │     ego/agents/route/mapquery/mapproc/to_tensor 子块计时（DP_STAGE_TIMING）
     ├─ ② norm    observation_normalizer（DP_OM 时跳过：v3 已并图，见 2c）
     ├─ ③ fwd     Diffusion_Planner(inputs)            [model/diffusion_planner.py]
     │     ├─ Encoder → StaticEncoderBody（DP_TORCHAIR=1 时整图编译，pos 段图外）
     │     │    └─ DP_OM 时：encoder.om v3 直吃 raw 7 输入，norm/pos/route 全图内
     │     └─ Decoder → fast_dpm_sample（DP_FASTDPM=1）10 步循环，每步 DiTBody 图
     └─ ④ post    outputs_to_trajectory（.cpu() 同步点）
```

性能形态（2026-09-03 R8 后）：

| 路径 | 口径 | 单步 | score(50) | 说明 |
|---|---|---|---|---|
| torchair（默认） | Ray 4-worker warm mean | **90.6ms**（中位 86.6/p90 97） | 0.9170 | DUO，wall 13min |
| torchair | sequential 中位 | 83.2 | 0.9364* | DUO，*闭环采样样本波动 |
| OM loop（R7） | sequential 中位 | 80.6ms（fwd 17） | 0.9171 | DUO；RC 同路径 121.3ms |
| **OM loop v3（R8）** | sequential 中位 | **77.5ms**（fwd ~10.8） | **0.9171** | DUO，norm/pos/route 并图 |
| **OM loop v3（RC 310P1）** | sequential 中位 | **102.5ms** | **0.9171** | RC，concurrent 9.3min |
| **OM loop v3+R9（DUO）** | sequential 中位 | **70.5ms**（fwd ~9.4） | **0.9171** | host 侧直通 + roadblock 缓存 |
| **OM loop v3+R9（RC）** | sequential 中位 | **90.0ms** | **0.9171** | 分步：①96.3→②90.0（对 615 基线 -85.4%） |

DUO 构成（R9 sequential）：adapt ~45.5（ego 0.1/agents 15.4/route 5.8/mapquery 13.3/mapproc 9.6/to_tensor 0.5）+ norm 0（图内）+ fwd 9.4 + post 12.5。fwd 内：encoder.om 2.44（含 norm/pos/route 三段）+ dit_loop.om 5.5 + 编排 ~1.5（host 侧直通后 d2h 消失）。两板差距 19.5ms 全在 CPU 段（CPU 1.34× 作用于 ~60ms host 代码 + 框架杂项），NPU 侧三项（算力/图执行/发射邮费）全部排除或消除——R7 诊断结论完全兑现。

adapt 数据通路（第四轮后形态，本轮未动）：`_get_lane_polylines` 从 devkit map 对象取 `discrete_path`，经 per-lane 缓存（`(map_name, lane_id, role) → [N,2] ndarray`）产出折线数组 → `map_process` 直接消费 ndarray（不再走 devkit `to_vector()` 拆包）→ `_interpolate_points_batch` ragged 批量定长重采样 → `_lane_polyline_process` 全向量化拼 12 维特征。

## 2. 图模式子系统（encoder + decoder 两个编译单元 + 采样循环；OM 离线变体）

```
Encoder.forward (DP_TORCHAIR=1)                   [model/module/encoder.py]
 ├─ _agent_pos/_static_pos/_lane_pos(neighbors,...)  # 图外 eager：pos 构造
 │    └─ lane heading: atan2 无 GE converter → 整段外提
 └─ StaticEncoderBody(...8 张量+3 pos)               # torchair GE 图
      ├─ 三个子编码器：全量静态 shape 计算，无 x[valid] 散回
      │    └─ 无效行：全零输入 → 有限垃圾 → ×valid_mask 置零（= 上游 scatter 回零）
      ├─ lane 限速分支：torch.where（两侧都算），替代 if sum()>0 masked-fill
      └─ fusion：float additive mask，ego 行 out-of-place 解除屏蔽

Decoder.forward (推理分支)                        [model/module/decoder.py]
 ├─ xT = cat(current_states, randn*0.5)           # 噪声起点（图外）
 ├─ sampler = self._sampler_holder[0]             # SamplerAdapter，list 挂载
 ├─ sampler.begin_step(route_lanes, cur_mask)     # RouteEncoder + attn_mask 各算一次
 ├─ DP_FASTDPM=1: fast_dpm_sample(...)            # 预计算系数的 10 步循环
 │    ├─ 系数：进程首次在 CPU 用上游 NoiseScheduleVP 同公式链预计算（Python float）
 │    ├─ 每 NFE：model_fn ≡ DiTBody 输出（x_start+dpmsolver++ 包装数学互逆，直接消掉）
 │    └─ 每步：线性组合(3 op) → constrain(1 op) → body
 └─ 默认: dpm_sampler(...)                        # 上游通用路径（A/B 基线）

DiTBody = DiT.forward 去掉 RouteEncoder 的静态 shape 主体
编译入口 = torchair.inference.cache_compile(body.forward, cache_dir=...)
           （编译产物落盘 $DP_DATA/torchair_cache，跨 worker 复用；
            无 dynamo guard，比 torch.compile 每步更轻）
pickle 安全 = SamplerAdapter/Encoder 各有 __getstate__，sim-log 序列化时
             剔除编译产物（运行时缓存，读档 lazy 重建）
RC 分流   = xT 噪声按 npu_utils.is_rc_device() 条件走 CPU randn
             （310P1 老固件无 StatelesslessRandomNormalV2 aicpu kernel）
```

### 2b. OM 离线变体（DP_OM=1 / loop，R7）

```
导出（export_om.py --stage export,atc,val）
 ├─ 图划分与 torchair 一致：encoder.om=StaticEncoderBody、dit_body.om=DiTBody
 ├─ dit_loop.om = DiTBody × 10 步 fast-DPM 整循环展开
 │    系数/t 序列烤成常量（schedule 标量与输入无关，R5 红利）
 │    首帧约束用函数式 cat 进图（等价原地写，免 trace scatter）
 ├─ 零 bool 图输入：has_speed_limit float+图内比较；attn_mask float 0/-inf
 └─ 导出期图简化（性能关键）：do_constant_folding + onnxsim shape 折叠
      torch 导出器不折 shape 链 → ATC 编成 4140 微 kernel/次
      （70% launch 空隙 + TransData 1057 + 66 Mod 掉 AI_CPU）= 31.6ms/次
      折叠后 nodes 12.7k→3.6k = 5.55ms/次，OM 183MB→18.4MB

运行（om_runtime.py OmBody，外层）
 ├─ InferSession 进程级缓存（nuplan 每场景重建 planner；不缓存则
 │  ~40 次新建耗尽 driver stream 池 EL0009，50 场景 42 起连锁挂）
 ├─ pyACL context 管理（skill: CONTEXT_MANAGEMENT.md）：
 │  set_device → get_context(save，必须在首个 InferSession 前) →
 │  每次 infer 后无条件 set_context 恢复——aclruntime 换走 context
 │  与输入设备无关（dit_loop 全 CPU 输入也必须 restore，DUO 的
 │  torch_npu 每调用自设 context 会掩盖漏 restore）
 ├─ 全局锁串行化 infer + 进 torch 前 synchronize
 └─ 采样循环留 eager CPU：微张量 host 侧零成本，只余 session 内部搬运

模式：DP_OM=loop（整循环图，每 planning step 1 次 OM 调用，推荐）
      DP_OM=1（逐次 dit_body，11 次调用，A/B 诊断用）
优先级：DP_OM > DP_TORCHAIR > eager；pickle 安全（holder 惰性重建）
```

### 2c. encoder.om v3（R8 削发射）：raw 直进 + 三输出

R7 → R8 的 fwd 内部数据流对比（动的是一张图和它的两个消费方，图外三段小算子串整体消失）：

```
R7 的 fwd 内部                                R8 v3 的 fwd 内部
─────────────────────────────                ─────────────────────────────
norm  ObservationNormalizer (NPU eager)  ─┐
pos   三个 pos 提取，含 atan2 (NPU eager) ─┼─→  encoder.om 直吃 adapt 原始输入
route RouteEncoder (NPU eager)+.cpu()   ─┘    （norm 常量烤图 / pos 含 atan2 的
                                               ONNX 分解 / RouteEncoder 静态化，
enc   encoder.om (吃 norm 后+pos)             全是图内 kernel 组）
dit   dit_loop.om  ←─ 不动                    ↓ 三输出：encoding +
                                               route_encoding + current_states
```

- 被砍的三段是几十~几百个小 op 的串：单个 op 的 NPU 执行微秒级，成本全在**发射邮费**（RC 是 DUO 的 2 倍）。RC 账：norm 5.7 + pos 1.1 + route 10.7 ≈ 17.5ms 清零，换图内 +0.16ms；DUO 邮费便宜只赚 3.1ms。副作用：两板 fwd 同价（~11ms）
- `planner.py`：DP_OM 时跳过 ObservationNormalizer（dp-stage 的 norm=0.0）；`decoder.py`：route_encoding/current_states 从图输出拿

```
EncoderRawExportWrapper（export_om.py，v3 导出单元）
 ├─ 输入：7 个 RAW adapt 键（neighbor_agents_past/static_objects/lanes/
 │        lanes_speed_limit/lanes_has_speed_limit/route_lanes/ego_current_state）
 ├─ 图内顺序：
 │   ① per-key norm：json mean/std 烤广播常量；全零行判据在原始数据上
 │      （上游语义），norm 后乘 (~mask).unsqueeze(-1) 置零；
 │      不在 json 的 key（lanes_has_speed_limit）原样透传
 │   ② pos 三提取（_agent/_static/_lane_pos 函数式版）：
 │      lane heading 走 ONNX atan2 分解——ATC 有 kernel，
 │      与 torchair GE 无 converter 是两回事
 │   ③ StaticEncoderBody（2b 同款，含静态 RouteEncoder，R8 第一步）
 │   ④ current_states 组装：norm 后 ego[:4] + neighbors 末帧[:4]
 └─ 输出：encoding [1,107,192] + route_encoding [1,192] +
          current_states [1,11,4]（decoder 采样锚点，零 norm 依赖）

闭环侧（planner.py / encoder.py / decoder.py）
 ├─ planner：DP_OM 时跳过 observation_normalizer 整段（norm=0.0）
 ├─ capture 移到 norm 前 → 语义 raw（旧 norm 后 capture 作废，bench_route
 │   为历史工具注明期望 norm 后输入）
 └─ decoder._om_sample：cs/route_encoding 全从 encoder 输出拿

战果：R7 fwd 33.6ms（norm 5.95+pos 1.81+route 5.07 三段邮费受害者）
     → v3 单图全包 10.7-11.7ms；RC 单步 121.3→102.5，DUO 80.6→77.5，
     score 双板 0.9171 逐位持平
```

## 3. 关键设计取舍

| 决策 | 为什么 | 备选与放弃原因 |
|---|---|---|
| RouteEncoder 外提（`begin_step`） | route_lanes 是 DPM 循环不变量，原实现每步重算 10 次；且其 `x[valid_indices]` 布尔散回 GE 图不支持（ERR03007） | 改写 RouteEncoder 为静态 mask 版：语义等价性要重证，先不动 |
| DiTBody 与 DiT 分离（而非改 DiT） | DiT.forward 保持上游原样（CUDA 兼容、训练路径零改动）；Body 是纯推理视图，数学恒等已验证（单步 diff=0） | 直接改 DiT.forward 签名：动上游代码，回归面大 |
| SamplerAdapter 用 **list 挂载** | `nn.Module.__setattr__` 会把裸 module 属性注册进 state_dict → 多一套 `_sampler.dit.*` key，checkpoint 匹配炸；list 值不注册，dit 引用别名共享权重（load 一份生效两处） | load_state_dict(strict=False)：掩盖真实加载错误，违反项目显式暴露原则 |
| `model_type` property 代理 | dpm_solver 的 `model_wrapper` 从 model 对象读 `model_type` | — |
| `DP_TORCHAIR` 默认 0（eager） | 图编译有 3e-2 级单步数值差（端到端已消化，5 场景无回归），更大场景集验证前保守默认 | 默认 1：收益已实测，但只验过 one_continuous_log 5 场景 |
| fast 采样器 = 系数预计算（非整图 trace） | schedule 标量与输入无关，预计算成 Python float 即消掉全部标量 op，不必把含 list 滚动的循环体 trace 进图；包装互逆（x_start+dpmsolver++）代数恒等，消掉反而更精确（上游往返被 1/α≈240 放大） | 整图 dpm_solver：理论还能省每步 11 次图 launch 的间隙，收益 ~ms 级，风险（multistep list 控制流）不值 |
| encoder 散回 → 全量计算 + valid mask 置零 | 三个子编码器行独立 → 有效行 bit-exact；无效行全零输入产有限垃圾 ×0 精确归零；零向量参与 cross-attn softmax 是训练语义，必须精确零 | gather/scatter 等价替代：保留索引语义但治标；量化剔除无效 token：引入近似语义 |
| pos 构造段图外（含 atan2） | atan2 无 GE converter（第三类图障碍：eager 正常、图不支持）；pos 是输入→位置的纯函数，与矩阵主体无交织，最小独立段外提代价 ~1ms | atan2 手写分解（atan(y/x)+象限）：除零分支与精度风险，不值 |
| 编译入口 = cache_compile（非 torch.compile） | 编译产物落盘跨 worker 复用（74s→35s 首步）；无 dynamo guard，warm 每步再省 ~4ms；坑：cache key 无源码 hash，改代码须清缓存 | torch.compile：每 worker 重复编译，guard 开销在 |
| OM = torchair 的离线孪生（非替代） | ATC 编译期锁死算子选择 → RC 免疫在线编译劣化（RC 上 torchair 同图劣化，OM 稳定 121ms）；图划分/静态化代码完全复用，两路径互为 A/B | 只留一条路径：丢对照与回退 |
| OM 整循环图（dit_loop）而非逐次调用 | 逐次 = 11 次 aclruntime 同步往返（host 编排 ~13ms）；整循环 = 1 次（系数烤常量后无控制流，cat 式约束可 trace）。R5 时"循环外置不值"的判断在 aclruntime 语境翻面：launch 间隙从可忽略变成大头 | 逐次变体保留为 DP_OM=1：调问题时能定位"第几步开始偏" |
| OM 导出必须过 onnxsim | torch 导出器从不折叠 shape 计算链；ATC 把幸存的 Shape/Gather/Mod 编成微 kernel 群 + aicpu Mod（msprof 实测 31.6ms 中 70% 是 launch 空隙）。onnxsim 静态 shape 推断删整条链 → 5.55ms | 只靠 do_constant_folding：折不掉 shape 链（Mod 66 个幸存） |
| RouteEncoder 并图（非搬 CPU） | 原方案表按"NPU host 派发单价 17us × 100 op ≈ 2ms"估 CPU，RC 空机实测 CPU eager 42-50ms vs NPU 4.5ms——该 op 链在 RC aarch64 上是**算力**瓶颈（~1 GFLOP/s），错把派发单价当 CPU 单价；并图 = 零额外投递，StaticEncoderBody 全量算+置零手法复用（注意 route 是 batch 级 mask_b 置零，非 per-row） | CPU 权重副本：11× 负优化；留 eager NPU：10.7ms 邮费白付 |
| encoder.om v3 直吃 raw（norm+pos+route 一批并图） | 三段全是小算子串（邮费受害），一次导出对拍成本与分步相同；atan2 走 ONNX 分解（ATC 有 kernel），当年 torchair 无 converter 的外提理由在 ATC 路径不成立；三输出含 current_states 让 decoder 也零 norm 依赖 | 分步并（先 norm 后 pos）：多一轮全链验证无额外收益；pos 搬 CPU：重蹈 route 覆辙 |
| pos 函数式重写（cat 拼接） | clone+slice 写回 trace 成写回型节点，ATC 310P **静默 miscompile**（one-hot 1 变 0；ort 逐位对/om 错，mini 图两分钟定位，skill FAQ 11）；cat 值恒等、全路径安全 | 保留 slice 写回：编译成功但输出错——"ATC 成功 + ort 正确"不保证 OM 正确 |
| DP_OM 下 adapt 产出留 host（R9①） | 计算全在图内（aclruntime 交换 host buffer），卡上中转是纯 H2D+d2h；mask 小 op 顺势转 CPU 削 RC 发射。实测 RC 收益更大（-6.2 vs DUO ~-2.3，50 场景口径）——RC 搬运/sync 类更贵（1.46×，R7 佐证） | 保留卡上中转：两板各白付 2-6ms。教训：RC 首测"没反应"是内层未 pull 的误测，据此编的"SoC 共享内存搬运便宜"论已推翻——对"没反应"先查测量再造理论 |
| roadblock 地图不变量缓存（R9②） | candidates 误差循环每步逐 lane 重转 np.array、相同 route ids 每步重建 dict——全是地图不变量；严格等价（无量化），单槽 map_name 换图整体清空防撞 key；route 段 10.5→5.8 | 整函数结果缓存：key 需含 ego 连续 pose（candidates 依赖），构造 key 的成本就在函数内部，不值 |
| mapquery 保持现状 | 大头 devkit `get_proximal_map_objects` 依赖 ego 连续位置，严格等价空间已尽；量化格子缓存是近似语义（半径查询随 ego 精确位置变），需验收口径拍板 | 量化缓存：~13ms 收益 vs 规划语义近似，未拍板不动 |
| InferSession 进程级缓存 + 无条件 context restore | 每场景新建 session → stream 池耗尽（EL0009）；aclruntime 换 context 与输入设备无关，按设备做条件 restore 会漏 dit_loop（全 CPU 输入）——DUO 的 torch_npu 每调用自设 context 掩盖此漏，RC 现形 | 每场景独立 session：泄漏；条件 restore：看似省事实则埋雷 |
| per-lane 折线缓存（`_POLYLINE_CACHE`） | `discrete_path` 是地图不变量（devkit cached_property 挂常驻对象），转换是纯函数 → 缓存严格等价；闭环每步重查同一批 lane，命中后 mapquery 60→14.6ms | 量化格子缓存空间查询：半径查询结果依赖 ego 精确位置，量化是近似语义，未做 |
| 坐标链路直通 ndarray（绕过 devkit `to_vector()`） | 旧链路 Point2D 列表 → 嵌套 list → np.array 三次逐点 Python 操作（~13% 步时）；消费方全在我方 `map_process`，返回类型语义可控 | 改 devkit `MapObjectPolylines` 类型标注：动上游，无必要 |
| ragged 批量插值（span 偏移） | 90×3 次小数组调用 → 3 次大数组调用，摊薄 numpy 固定调度开销；每线弧长加 `i×span` 保持全局单调，一次 searchsorted 服务全部线 | 逐条插值保留：`_interpolate_points` 单线版仍服务退化分支 |
| `_lane_polyline_process` 全量向量化（不按 avails 分支） | 无效 element 输入全零（上游 `coords[~avails]=0`）→ 全量算出全零，与"跳过"等价；flip 判断只需首点，`[E,1]` 切片即可 | 保留 avails 参数：调用方签名不变 |
| agents 侧只向量化 filter/pad/extract 三处 | 复查火焰图：剩余 18.8ms 是 devkit TrackedObject `__slots__` 属性访问 floor（11 帧×~50 agent×7 字段），不重写 devkit 对象层压不动 | 重写 devkit Observation 对象层：动上游，收益/风险比差 |

## 4. 计时/开关体系（横切结构）

| 层 | 位置 | 开关 |
|---|---|---|
| 阶段计时 | planner.py 四阶段 + data_processor.py 六子块 | `DP_STAGE_TIMING=1` |
| jit 窗口 | planner.py 前向窗口 | `DP_JIT_WINDOW`（默认 0，历史对照用） |
| 图模式 | decoder.py DiTBody + encoder.py StaticEncoderBody | `DP_TORCHAIR`（**默认 1**，50 场景全量 0.9170 验证后转正；`=0` 回退 eager） |
| OM 离线 | 同上双图 + dit_loop 整循环 | `DP_OM`（`loop` 整循环图推荐 / `1` 逐次变体；默认关。优先级高于 DP_TORCHAIR。产物 `$DP_DATA/om/om_models`，`DP_OM_DIR` 覆盖；R9：DUO 70.5ms / RC 90.0ms，均 0.9171） |
| fast 采样器 | decoder.py 推理分支 | `DP_FASTDPM`（**默认 1**，同上） |
| 图缓存目录 | 两处 cache_compile | `DP_TORCHAIR_CACHE`（runner 默认 `$DP_DATA/torchair_cache`；**改 forward 源码后必须换目录或清空**——cache key 无源码 hash，会静默加载旧图） |
| worker 模式 | runner | `DP_WORKER=sequential`（单进程排障模式，无 Ray，threads 参数跳过） |
| 数据目录 | runner | `DP_DATA`（默认项目内；119 用 `/data/syx_dp` 覆盖） |
| 输入捕获 | planner.py（norm 前 = raw，R8 起语义） | `DP_CAPTURE_DIR=<dir>` |
| runner 覆盖 | sim 脚本 | `DP_DEVICE`（默认 0）/ `DP_LIMIT` / `DP_THREADS` |

工具：`read_scores.py`（实验目录出分）；等价性对拍四件（插值/adapt 向量化/fast 采样器/静态编码器）、计时分布统计 `analyze_stage_log.py`、火焰图聚合器 `agg_speedscope.py`（py-spy speedscope JSON 按 self/total 出占比）与过时基准两件（bench_dp_forward、bench_torchair_dit）已归档至 [../scripts/](../scripts/README.md)——复跑时拷回仓库根；外层：`export_om.py`（三段式导出/编译/校验，v3 图）、`ctx_probe.py`（ACL context 链隔离探针，A/B/C 判读）、`om_runtime.py`（运行时 shim，多输出支持）、`bench_om.py`（OM 单图时延，mixed 模式复刻闭环数据路径并分段 d2h/infer/h2d/sync）、`bench_step.py`（闭环单步分段重放 + cpu_probe 标尺；v3 起段=enc_om/prep/dit_om/invnorm/post_sim）、`bench_launch.py`（发射邮费五项探针：host 派发/投递/小 kernel/算力标尺/往返）、`bench_route.py`（历史工具：route CPU-vs-NPU 对拍+线程扫描，裁定 CPU 方案否决）、`npu_utils.py` / `read_results.py`。

## 5. 扩展点

- **R8 已完成**：norm/pos/route 三段并进 encoder.om v3，RC 121.3→102.5、DUO 80.6→77.5。`TASK_QUEUE_ENABLE=2` 否决（micro 邮费 -58% 但真实负载 +19%：排空变贵）
- **R9 已完成**：①to_tensor 免上卡（DP_OM 下 adapt 留 host；分步实测 RC -6.2 > DUO ~-2.3——RC 搬运/sync 更贵 1.46×，勿跨板外推收益）②roadblock 地图不变量缓存（严格等价，DUO route 10.5→5.8；RC -6.3——段总时长同价但构成不同，被砍的 np 转换在 RC 被 CPU 1.34× 放大）。RC 102.5→90.0、DUO 77.5→70.5，双板 0.9171 三轮逐位一致
- **RC 剩余差距的本质**：两板 fwd 同价（~9-11ms），~26ms 差距全在 CPU 段。严格等价空间已收完；再往下全部需拍板：mapquery 量化格子缓存（~13ms，**近似语义**，验收口径）、agents devkit 对象层重写（~15ms，`__slots__` floor，动上游）、post 轨迹插值（~12.5ms，小）
- **RC 搬运形态待坐实**：RC 跑 `bench_om.py --mode mixed` 对比 DUO d2h 段（encoder 0.97 / dit_loop 1.11ms）——验证 davinci-mini 是否 SoC 统一内存（①的板间不对称解释）
- **运维注意**：119 根盘长期高压（现 98%），Syncthing index 要求 ≥1% 余量——盘满会静默假同步（本地仍显示 idle）；权重库软链在 `/data/weights/`
- **post 12.4ms**：`.cpu()` 同步点 + 轨迹插值（numpy），收益空间有限
- **MSprof 定位法 + onnxsim 案例进 skill**：om/references 补"导出必过 onnxsim（31.6→5.55ms 案例与微 kernel 判别特征）"（低优先，FAQ 11 的 ATC miscompile 已先进 skill）
- **adapt 侧剩余**（~50ms，floor 附近）：agents ~16-19 是 devkit 对象属性访问 floor（不动 devkit 压不掉）；shapely 距离排序（量化缓存是近似语义，未做）；route_roadblock_correction
- **运维注意**：119 根盘长期高压（现 98%），Syncthing index 要求 ≥1% 余量——盘满会静默假同步（本地仍显示 idle）；权重库软链在 `/data/weights/`

## 相关

- decisions/003（float mask）、003 勘误（jit 有害）
- summary/2026-08-19 三篇（R1~R3）、summary/2026-08-20（R4 adapt）、summary/2026-08-21（R5 fwd 图化）、summary/2026-08-28（收口轮）、summary/2026-08-31（R7 OM 离线化）、summary/2026-09-02（RC 诊断轮）、summary/2026-09-03（R8 削发射：encoder.om v3）
- setup.md §8（开关用法、OM 依赖、bench 工具、py-spy、torchair cache 注意事项、运维注意）、§torchair 编译
- cuda-to-npu skill：om/references/CONTEXT_MANAGEMENT.md（pyACL save/restore 规范，R7 的 RC 修复即按此实施）
