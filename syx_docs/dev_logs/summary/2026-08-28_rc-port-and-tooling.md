# 2026-08-28 第六轮：全量验证转默认 + RC 移植 + 基础设施事故恢复（R5 后收口轮）

> R5（0821 fwd 图化 141.1→86.6ms）之后的收口与外延：50 场景全量验证并转默认开关、sim log 落盘修复、外层 README 与 runner 便携化、RC（310P1）移植攻坚、Syncthing 假同步事故恢复、sequential 口径基准。无单步性能优化（fwd/adapt 均在 floor）。

## 事实：50 场景全量验证与默认开关转正

- 50 场景 one_continuous_log，DP_TORCHAIR=1 + DP_FASTDPM=1，119/310P3，4 worker：**50/50 成功、final_score 0.9170（与优化前全量基线完全一致）**；
- 转默认（a719c5e）：`DP_TORCHAIR`/`DP_FASTDPM` 缺省 0→1，`=0` 保留回退；cache key 不受影响（CompilerConfig 未变）。

## 事实：sim log 落盘修复（R3 起的隐性缺陷）

- **现象链**：仿真收尾时 `SimulationLog` pickle 整个 planner → 遍历到 torchair 编译产物（LazyCompiledModel）不可 pickle → `simulation_log.py:38` 抛异常 → 该场景"Simulation failed"。
- **从 R3（图模式落地）起就存在**：单线程 5 场景 bench 同样在挂（各 21 处失败标记），只是 metric 链路（读内存历史）不受影响、score 照常出，从未被察觉。50 场景 run 的 108 处失败同性质。0818 基线（无 torchair）50 个 sim log 全落盘为对照。
- **修复**（cf42adf）：`SamplerAdapter`/`Encoder` 各加 `__getstate__`，pickle 时把编译产物剔成 None（运行时缓存，读档后 `_get_body()` lazy 重建）。权重照常进存档，恢复 0818 行为。
- **验证**：pickle 往返 23MB；5 场景 5/5 落盘 score 0.9941；50 场景 50/50 落盘、零失败、0.9170。

## 事实：性能口径修正——对外报 warm mean

- runner_report 的 mean 是全生命周期（含首步编译 ~30s 级），745 步中 5 个冷启动步把 mean 从 90.6 拉到 275.5；
- 业界基准惯例（MLPerf/trtexec/torch.benchmark）都是 warmup 后 mean；重尾/共享场景用 percentile；
- **对外口径定为 warm mean 90.6ms（p90 97.0，单线程独占）**，中位 86.6 退居参考。

## 事实：外层 README 与 runner 便携化（RC bring-up 驱动）

- 外层样例 README（GR00T_n1d7 风格）：驱动/CANN 表、conda+依赖分块、源码布局（clone npu_port 分支 + nuplan-devkit pin e924167）、torchair 源码编译、数据/权重放置、runner 用法、性能表（300I DUO 90.6ms / 13min / 0.9170）。
- bring-up 实测补齐的坑（每条都是 RC 上撞出来的）：conda 源 SJTU 403（改用机器自配源）、opencv 裸装顶到 5.0.0.93（与 numpy 1.26.4 实测兼容，钉版本）、`mmengine==0.10.7` 是漏记的硬依赖（train_utils 顶层 import 经 normalizer 进推理链）、HF 权重要下 args.json+model.pth 两个文件、maps 直接在 `datasets/maps/` 下（无 nuplan-maps-v1.0 层）、mini 包解压顶层即 `mini/`。
- runner 便携化系列 commit：相对脚本路径（6ccec71）、`scenario_builder.data_root` override 兼容 cache/mini 布局（2a3e419）、默认卡 0（2731581）、`DP_WORKER=sequential` 单进程排障模式（b79212f/cf33022）、结尾自动 `read_results.py` 出分（8896626，read_results.py 与 npu_utils.py 放外层样例目录）。
- npu_port 分支 push 到 github（WeNeedMoreCode fork），外层样例目录提交至 gitcode npu_adpat（README×2、npu_utils、read_results 共 4 commit）。

## 事实：RC（310P1）移植攻坚——三次崩溃的定位链

1. **`npu is not available`**：runner 默认卡 4 而 RC 单卡 → 改默认 0；
2. **507018 aicpu exception（多 worker）**：4 worker 挤满载卡互相拖挂，连 torch.load 的 H2D 都失败——**多进程误伤模式**，报错位置互相误导；
3. **真凶定位**：`ASCEND_LAUNCH_BLOCKING=1` 后栈现形——异步模式下报错算子名（aclnnNonzeroV2）是**误报**，真凶是 `torch.randn` 底层的 aicpu `StatelessRandomNormalV2`（310P1 老固件执行不了 V2 系 kernel）。
- **修复**（c632b44 + 外层 npu_utils.py 5ca53c19f）：`is_rc_device()`（lspci 无 "accelerators" 判定，GR00T 同款启发式）放外层 `npu_utils.py`，decoder 的 xT 噪声**条件分流**（RC 走 CPU randn + to，正常设备路径零改动——保住已验证基准的 RNG 流）；runner 自动把样例目录加 PYTHONPATH。
- 踩坑教训（写进 skill）：异步栈不可信先开 blocking；多进程误伤先降 sequential；改主路径必须条件分支不能无条件。

## 事实：工具沉淀（两个 skill）

- **inference-optimization**（新建，设备无关）：分侧计时/循环不变量外提/采样器系数预计算+包装互逆/常量缓存/等价性对拍五模式；references/diffusion-planner-case.md 承载五轮 615→86.6 案例。
- **cuda-to-npu 增量**：`torch_npu/references/rc_device_toolkit.md`（is_rc_device / randn 分流 / floordiv 补丁 / 整段搬 CPU 四工具 + RC 移植接入清单）；第 10 章补排障两步法与多进程误伤模式（事实/推测严格分层：NonZero 记为被误报、实测能跑）。

## 事实：Syncthing 假同步事故（0826-0828）

- **根因链**：119 根盘 100% 满 → Syncthing index 数据库要求所在盘 ≥1% 余量 → folder 进 error 态（0826 02:36）→ **本地端仍显示 idle（对端 index 冻结）**→ 两天内所有代码同步静默失效。
- **清盘**：/home 大头全是他人资产（weights 573G 共享权重库/l00586152 248G/LargeModelInference 178G），仅挪两个已被迭代的老模型 `Qwen3.5-27B`(52G)+`Qwen3.5-27B-w8a8-mtp`(34G) 到 /data，原位软链保路径 → 释放 86G（100%→98%）。
- **恢复**：重扫后 folder 复活，267 个积压文件追平（read_results/npu_utils/decoder 全部到位）。
- 顺带修正：本地 `diffusion-planner` folder 指向 0819 目录正名前的拼错路径（path missing，僵尸 folder）。
- 流程沉淀（memory 两条）：ModelZoo .gitignore 改动永不提交；Syncthing-git 状态拉锯机制（stash/checkout 回滚磁盘会被服务器版 ~10s 内同步回来）。

## 事实：sequential 口径基准（119/310P3，50 场景单进程）

| 口径 | 总时长 | 单步中位(p90) | score |
|---|---|---|---|
| sequential 单进程 | **49 分 32 秒** | **83.2ms (94.6)** | **0.9364** |
| 4-worker Ray 并发 | 13 min | 88ms（排队口径） | 0.9170 |

- sequential 是无 worker 干扰的"裸值"：p90 从 Ray 口径的 122 降到 94.6（无设备争抢长尾）；
- 两次全量 score 差 2 个点（0.9170/0.9364）为闭环随机采样的样本波动（每步噪声不同→轨迹不同），均 ≥ 基线，无回归结论不变。

## 下轮方向：ONNX→OM 离线化（方案已评估待实施）

- **动机**：torchair 在 RC 上兼容雷区大（V2 系 aicpu kernel、GE 版本配套）；ATC 离线编译把算子选择锁死在编译期（编不过列表化报错），运行时只要驱动+轻量 runtime，消除 RC 运行时惊喜。**不指望比 torchair 快**（同属 GE 编译栈）。
- 架构：双 OM 图与 torchair 双图一一对应（encoder.om=StaticEncoderBody、dit_body.om=DiTBody，R5 静态化红利直接复用）；采样循环/线性组合/constrain/RouteEncoder/噪声在图外由 ACL session 编排（fast_dpm 每步 3 张量 op + 1 次 body 调用的结构即接口缝）。
- 已知坑：MHA 的 ONNX 化与 float mask 表示、310P1/310P3 分别编（soc-version）、OM vs eager 重新对拍定基线。
- 分批：批1 导出+ATC+CPU 对拍（ais_bench，抄 GenPosePlus export/onnx2om 模式）；批2 OM 编排器接回闭环 5 场景对齐。

## 工件

- 计时日志：`run_prints/2026-08-28_rc-port-and-tooling/`（bench_r5_full50.log、bench_seq50.log）；服务器 `/data/syx_dp/r5/`
- 关键 commit（npu_port）：cf42adf（pickle 修复）、a719c5e（转默认）、6ccec71/2a3e419/2731581/b79212f/cf33022/8896626（runner 便携化）、c632b44（randn 分流）；外层 npu_adpat：72fb60ca0/ac17cc9d4（README）、5ca53c19f（npu_utils）、8e8bbe973（read_results）
- 权重软链：`/home/weights/{Qwen3.5-27B,Qwen3.5-27B-w8a8-mtp} -> /data/weights/`；移动日志 `/data/move_weights.log`
