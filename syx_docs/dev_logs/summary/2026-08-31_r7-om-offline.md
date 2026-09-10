# 2026-08-31 第七轮：ONNX→OM 离线化全链打通（DUO 反超 torchair，RC 首次 50 场景全过）

> 动机承接 R6 结论：torchair 在线编译在 RC 上兼容雷区大，ATC 离线编译把算子选择锁死在编译期。本轮把 OM 做成了不只是"RC 能跑"的路径：fold+simplify 导出优化后 DUO 单步 80.6ms（torchair 83.2）反超，RC 50 场景 0.9171 全过、121.3ms。过程中修掉三类真 bug（stream 泄漏、pyACL context 管理、条件 restore 漏洞），全部由最小复现定位。

## 事实：批1 导出/编译/对拍工具链

- `export_om.py`（外层新文件）三段式：`--stage export,atc,val`。图划分与 torchair 双图一致（encoder.om=StaticEncoderBody、dit_body.om=DiTBody），后追加 dit_loop.om（整循环）。
- 图友好 dtype 设计：零 bool 图输入——`lanes_has_speed_limit` 以 float 0/1 进图（`>0.5` 在图内）；DiT attn_mask 以 float 0/-inf 加性 mask 进图（与 DiTBlock 内部 bool→float 转换逐位同值）。
- ATC：input_shape 从导出的 onnx 读回（免二次维护）；soc 自动判 RC（is_rc_device→310P1）/常规（310P3）；新版 CANN 输出带 `_linux_aarch64` 后缀已归一化。
- 对拍两层：onnxruntime-vs-eager（CPU，隔离导出错）+ OM-vs-eager（ais_bench，覆盖 ATC）。**判据迭代**：初版 rel@floor(1e-3)——近零元素噪声主导无意义；终版 = zero-viol（ref 精确零行输出保持精确 0）+ max_abs ≤ 量级 1%（GE LN/Softmax/MatMul 数值族口径；torchair 同输入实测 encoder rel 0.69 同族，佐证非导出错）。
- 数值（310P3）：ORT 层 encoder 2.9e-5 / dit 4.3e-6；OM 层 encoder 0.65% 量级比 / dit 0.11%，zero-viol 全 0。
- 环境件：ais_bench/aclruntime 从 gitee 页面 README 指向的华为云 OBS 直链装（仓库已不放 whl，raw 路径 404）；onnx/onnxruntime pip。

## 事实：批2 运行时接回 + 三次性能跃迁

路径形态：`om_runtime.py`（外层）OmBody 把 InferSession 包装成 torch body 签名（host float32 交换、拍平输出 reshape、ais_bench 惰性导入）；模型侧 `DP_OM` 分支（优先级 DP_OM > DP_TORCHAIR > eager）。fast_dpm 循环、约束、randn 留 eager CPU——循环 op 是 [1,11,324] 微张量，host 侧零成本，只剩 session 内部 H2D/D2H。

| 版本 | dit 图单次 | 单步(DUO,seq) | 事件 |
|---|---|---|---|
| body 逐次（DP_OM=1） | 3.4ms×11 次 | 126ms（fwd 55） | 5 场景 0.9941 ✓ |
| dit_loop 整循环 | 31.6ms | 117.5ms（fwd 44） | 50 场景 42/50 ✗（stream 泄漏，见下） |
| +do_constant_folding | （未单测） | — | Sin/Cos/频率表折常量；Mod 仍在 shape 链 |
| +onnxsim shape 折叠 | **5.55ms** | **80.6ms（fwd 17）** | 50 场景 0.9171 ✓（== 基线 0.9170） |

三图终值（fold+simplify 后）：encoder.om 2.28ms/5.3MB、dit_body.om 0.92ms/9.7MB（比 torchair 单体步快）、dit_loop.om 5.55ms/18.4MB；onnx 节点 dit_loop 12.7k→3.6k；OM 体积从 86.5/94/183MB → 5.3/9.7/18.4MB（shape 链消除后 ATC 重新保住权重共享）。

**性能定位方法论（本轮最大可迁移资产）**：msprof op 级 profile（`msprof --application=... --output=...`；aclruntime 的 acl_json profiler 与该版 aclruntime 不兼容，session_options 仅 3 属性）揭开 31.6ms 构成：**70% 是 launch/调度空隙**——4140 个微 kernel/次（p50 时长 1.1us，每 kernel launch+间隙 ~5.4us）、TransData 1057 个/次（19.8%，MHA Transpose 链的格式转换风暴）、66 个 Mod 全部落 AI_CPU（2.98ms，W11001 警告即此）。mac_time 每 iter 仅 0.04ms——纯计算量极小，慢在发射。torchair 快是因为 aten 路径融合激进 kernel 数少一个量级；implmode 两档（high_precision/high_performance）实测仅差 1.2ms，证实问题在图结构不在算子实现。

**导出侧修复**：`do_constant_folding=True`（沿抄 GenPose 设了 False）+ onnxsim（torch 导出器从不折叠 shape 计算链，Shape/Gather/Mod 全部幸存，ATC 把它们编成微 kernel + aicpu Mod）。已固化进 stage_export；onnxsim 未装时打印警告照常导出但性能不可用。

## 事实：三个真 bug 的定位链

1. **stream 泄漏（EL0009）**：nuplan 每场景重建 planner → 每场景新开 InferSession 且不释放 → ~40 次后 driver stream 池耗尽（`Insufficient_Stream_Resources`，rtStreamCreate 拒绝），50 场景首跑从第 ~42 个起连锁挂。修复：进程级 `_SESSION_CACHE`（name+device 键）。5 场景测不出累积型泄漏——50 场景就是验证场。
2. **pyACL context 管理**：RC 首跑 OM 全挂 107003（stream not in current context，norm 挂点 = OM infer 后第一个 torch kernel）。初版修复用 `torch.zeros(1)` 间接逼 context 重设——**用户质询后对照 skill（om/references/CONTEXT_MANAGEMENT.md）发现不符合规范**：正解是 pyACL 显式 save/restore，且 save 必须在第一个 InferSession 之前（ais_bench 源码：首个 session 复用 PTA context、后续新建并切走 current）。
3. **条件 restore 漏洞（真凶）**：`ctx_probe.py` 隔离探针（A=null save/B=restore 无效/C=链存活 三判）在 RC 全绿 VERDICT C，闭环仍挂。对比发现 probe 手写每次 infer 后**无条件** restore，而运行时代码 `if mixed_npu`（首输入在 NPU 才 restore）——**dit_loop 的输入全在 CPU**（循环状态 host 侧），它的每次 infer 都不 restore → 下一步 norm 必挂。DUO 无感是因 torch_npu 每次调用自设 context 兜住了。修复：restore 无条件化（换 context 的是 aclruntime，与输入设备无关）。probe 首版自身还踩了 device randn（StatelessRandomNormalV2 在 RC 必炸，R6 同族），torch_op 改 arange。

## 事实：RC 数据与 +40ms 归因

RC（310P1，davinci-mini）50 场景 DP_OM=loop sequential：**0.9171（== DUO 基线）、50/50、单步中位 121.3ms、wall ~40min**。torchair 在 RC 同为 ~120ms（wall ~100min，多出的 ~60min 推测为编译类开销——未验证，无 runner_report 数据）。

RC vs DUO 的 +40ms 逐段账（dp-stage 实测）：

| 段 | DUO | RC | 放大 | 负载类型 |
|---|---|---|---|---|
| adapt | ~51-57 | 67 | 1.3× | 纯 CPU |
| norm | 4.3 | 12.4 | **2.9×** | NPU 小算子密集（逐 key mask） |
| fwd | 17 | 26 | 1.5× | OM 图 + host 编排 |
| post | 8.9 | 12 | 1.4× | CPU + 同步点 |

结论：不是单点 40ms，是**四段按负载类型各自放大**——CPU 密集段按 CPU 差 ~1.3×，launch 密集段（norm 最典型）被老驱动固件的 kernel launch 延迟放大 ~3×。两条推理路径在 RC 上同为 120ms 也由此解释：fwd 的 launch/编排开销掩盖了图执行差异。若将来专攻 RC：方向是削 launch 次数（norm/post 小算子批处理化或小张量搬 CPU），不是换推理路径。

## 事实：运维事件与基础设施

- **根盘二次 100%**（0829）：lhh 的 Qwen3.5-27B 量化系列 24h +29G（v4 visionfloat）至 312G，叠加 docker 停止容器层 226G 吃光 R6 清出的余量 → Syncthing 假同步复发（export_om.py 卡 20min 未达）。按用户指示清 10 个久停容器（大头 `v0230-310p-oe` 单个 111G 可写层），100%→93%。
- **`spill_watch.sh` 常驻守护**（容器 /root，用户要求通用化）：规则表驱动（`<watch_dir>|<limit_gb>|<move_glob>|<spill_dir>` 一行一个对象），当前盯 /home/l00586152@300G、搬 mtime 最老的 `Qwen3.5-27B-*` 到 /data+原位软链；首搬 w8a8s 35G（311→276G，78s）；`--loop` 300s 一轮，每规则 flock，每轮最多一份，软链永不再选。容器重启需手动重启守护。
- **pip 依赖互踩**：`pip install onnx`/`onnxsim` 会把 protobuf 顶到 6.x → tensorboard 2.11.2 pb2 链炸（run_simulation 入口挂，报错在 nuplan import 链，与本体无关极难定位）；钉 `protobuf==3.20.3`（onnx 3.20 可用实测）。装一次顶一次，onnxsim 装完必回钉。
- **PYTHONPATH 覆盖式 export 之坑**：`export PYTHONPATH=<dir>`（非 `:<dir>` 追加）冲掉 set_env.sh 注入的 CANN python 路径 → `No module named 'tbe'` → npu_init 500001。用户一眼定位。runner 本来就是追加式，临时脚本容易写错。

## 事实：文档与交付物

- 外层 README 补 OM 章节：三段式命令逐步注释（onnxsim 在 export 内自动跑、val 是可选体检）、三图表、DP 开关、性能表加 DUO/RC OM 行（RC 行待 commit）。表达白话化（"对拍"→数值校验的完整描述）。
- 默认路径对齐：export_om 数据根默认 `./Diffusion-Planner`（原 HERE 差一层）；om_runtime.om_dir 未设 DP_DATA 时脚本相对定位（原 cwd 相对路径静默错误）。
- ctx_probe.py 进外层仓库（RC 排障探针，A/B/C 判读）。

## commit

- 小 DP（github npu_port）：`ab4ee6c`（DP_OM body 路径）、`46b54f7`（DP_OM=loop 整循环模式）
- 外层（gitcode npu_adpat）：`75b942a37`（export_om+om_runtime）、`a560ea2f8`（fold+simplify 固化）、`af577be43`（session 缓存修 EL0009）、`89f248ac4`（README OM 章节）、`7fec7c906`（README 白话化+默认路径）、`aad1c1c41`（pyACL save/restore）、`a8d0b7195`（构造后 restore）、`d59e762c2`（ctx_probe）、`53047b619`（probe arange）、`1382eecc9`（无条件 restore）

## 工件

- 日志归档：`run_prints/2026-08-31_r7-om-offline/`（reexport_fold_simplify、sim50_om_loop_duo、sim50_om_body_first_fail=EL0009 现场）
- OM 产物（119 `/data/syx_dp/om/`、RC `Diffusion-Planner/om/`）：onnx_models/{encoder,dit_body,dit_loop}.onnx + val_inputs.npz；om_models/{...}.om + val_ref.npz
- RC 实验目录：model_2026-08-31-18-20-56（50 场景 0.9171）

## 下轮候选（未排期）

- RC 专项：削 launch 次数（norm 12ms/post 12ms 的小算子批处理化）——RC 单步 121→~100 的空间
- skill 增补：CONTEXT_MANAGEMENT.md 加"restore 不得按输入设备做条件"案例；om 路径加 onnxsim 性能案例（31.6→5.55ms）与 msprof 定位法
- RC torchair 100min wall 大头验证（runner_report 对比，低优先）
