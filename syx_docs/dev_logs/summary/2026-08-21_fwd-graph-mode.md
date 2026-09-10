# 2026-08-21 第五轮：fwd 侧图化（141.1→86.6ms，累计 -85.9%）

> 主攻 handoff ②：fwd 70.2ms/50%（dpm_solver 标量开销 ~11ms、encoder ~25ms 未进图、torchair 图缓存多 worker 重复编译）。adapt 侧 54ms 按锁定决策未动（agents 18.8 是 devkit 对象属性访问 floor）。

## 事实：改动详账（4 处，三个环境开关 + 1 处默认行为）

### A. fast_dpm_sampler：采样器系数预计算 + 包装互逆消除（fwd 70.2→43.0）

**实现**：新文件 `diffusion_planner/model/diffusion_utils/fast_dpm_sampler.py`，`DP_FASTDPM=1` 启用（默认 0 走上游 dpm_sampler 作 A/B 基线）。只支持本项目固定超参（multistep, order=2, logSNR, linear schedule, denoise_to_zero, uncond）。`SamplerAdapter.begin_step(route_lanes, neighbor_current_mask)` 同时预构造 attn_mask（循环不变量，上游 DiT.forward 每个 NFE 重建）。

**原理**（两个独立优化点）：

1. **schedule 标量与输入数据零关系**。通用 dpm_solver 每步在 NPU 上用小张量重算 `marginal_lambda / marginal_std / expm1` 链（每次 solver update ~25 个小算子）+ 每次 NFE 外面包 noise_pred→data_pred 变换（~15 个小算子）。这些量只依赖 schedule 常量（linear schedule 闭式 `-0.25t²(β₁-β₀)-0.5tβ₀`）和步号。fast 版进程首次在 **CPU 上用上游 NoiseScheduleVP 类**跑同公式链（保证公式逐条同源，不是重抄）预计算 timesteps 和每步线性组合系数存 Python float；运行时循环只剩：线性组合（3 个大张量 op）→ constrain（1 op）→ body 调用。
2. **包装数学互逆**。`model_type=x_start` + `dpmsolver++` 下 model_fn 返回 `(x - σ·((x - α·body_out)/σ))/α`，代数上恒等于 `body_out`——直接用 body 输出，省掉每 NFE 的包装 op。浮点上该往返不精确抵消：两次除法/乘法的舍入被 `1/α(t)` 放大（t=1 时 α≈0.004，放大约 240 倍），fast 版反而更精确。

**效果**：fwd 70.2→43.0（**-27.2ms**，超火焰图预估 ~13ms——见"推测"节归因）。norm 段 7.1→4.8（改动 D 同批落地）。

### B. encoder 静态化进图：全量计算 + valid mask 置零（fwd 43.0→22.4）

**实现**：`encoder.py` 新增 `StaticEncoderBody`（`DP_TORCHAIR=1` 时编译进图，与 DiT body 复用同一开关；上游 `Encoder.forward` 不动）。四个翻译：
1. 三个子编码器的 `x[valid_indices]` 剔除 + scatter 回零 → **全量静态 shape 计算 + 输出乘 valid mask 置零**；
2. lane 编码器 `if has_speed_limit.sum()>0` 两段 masked-fill 分支 → `torch.where`（两侧都算按位选）；
3. fusion 的 `mask[:, 0] = False` in-place 参数修改 → out-of-place `cat`；bool key_padding_mask → float additive mask（decisions/003 同款）；
4. pos 构造段（含 `atan2`）整体外提 eager——`NotImplementedError: aten.atan2 ge_converter is not implemented`（图障碍第三类：硬件有算子、eager 正常、GE 后端没写转换器），按数据流切最小独立段外提，矩阵主体照常进图。

**原理**（bit-exact 三条论证，对拍前先纸上证明）：
1. 三个子编码器**行独立**（LayerNorm/Mlp 逐行，MixerBlock 只在行内 token 间混合）→ 有效行的 op 序列与剔除版完全相同 → bit-exact；
2. 无效行输入全零（padded 数据定义）→ 过 MLP 产出**有限**垃圾（GELU(bias) 有限、LayerNorm 零方差由 eps 兜住不 NaN）→ 乘 0 精确归零 = 上游零初始化 scatter 的结果；
3. **零必须精确**：DiT cross-attn 对 context 无 mask，无效 token 以零向量参与 softmax（q·0=0 → e⁰=1 权重）是训练语义，垃圾向量或 NaN 会改变 softmax 分布。

**效果**：fwd 43.0→22.4（**-20.6ms**）；对拍 4 个真实 capture 输入**全部 bit-exact**（20544/20544 逐位相同，含 float mask 转换在内整条链）。

### C. torchair cache_compile：编译产物跨进程持久化（fwd 22.4→18.5，首步 74→35s）

**实现**：两处编译入口从 `torch.compile(body, backend=npu_backend)` 换成 `torchair.inference.cache_compile(body.forward, config=config, dynamic=False, cache_dir=...)`（`DP_TORCHAIR_CACHE` 可覆盖，默认 `/data/syx_dp/torchair_cache`）。

**原理**：`torch.compile` 图缓存是进程内的，每个 Ray worker 首次各编译一次（DiT ~31s + encoder ~40s）。cache_compile 把编译产物（marshal 的 wrapper code + GE 图）序列化落盘，后续进程直接加载。两个源码级确认的坑（torchair 7.2）：只接受 bound method（`body.forward`，传实例 ValueError）；**cache key = str(module) + config 的 md5，不含 forward 源码 hash**——改代码后必须清 cache 目录或换 `DP_TORCHAIR_CACHE`，否则静默加载旧图（已写进代码注释与 setup.md）。

**效果**：首步 74s→35s（cache 命中后仍剩 GE 侧首次加载）；**warm 每步 fwd 22.4→18.5（-3.9ms）**——归因见"推测"节。

### D. normalizer 设备缓存（norm 7.1→4.8）

**实现**：`StateNormalizer` / `ObservationNormalizer` 加 per-device 缓存，mean/std 首次 `.to(device)` 后存住。

**原理**：归一化统计量是 CPU 张量，旧实现每次调用 `.to(device)` × 每 key × 2 个张量 ≈ 14 次/步 H2D 同步拷贝，纯调度开销。缓存后值 bit 不变（同一 CPU 值的一次性搬运）。

## 事实：验证

- **对拍**（真 ckpt + DP_CAPTURE_DIR 捕获的真实输入，服务器 device 5）：

| 对拍项 | 用例 | 结果 |
|---|---|---|
| fast_dpm vs 上游 dpm_sampler（固定 seed 同 xT） | 4 输入 | max_abs_diff 8.2e-4~6.1e-3，**相对 1.7e-4~9.0e-4** |
| StaticEncoderBody vs 上游 Encoder（eager） | 4 输入 | **全部 bit-exact 20544/20544，diff=0** |

- fast_dpm 的 diff 判定：若公式错（timesteps/系数/r0），两条路径解不同的 ODE，diff 会是 O(1) 轨迹级；实测 1e-4 相对级 = 同一解的不同舍入路径。参照系：torchair 图模式单步数值差 3e-2（R3 实测）端到端无回归，本差值小两个量级。
- **E2E**（每批独立 5 场景 one_continuous_log）：批次1 0.994213 / 批次2 0.9942 / 批次3 0.9941——全程波动 ≤0.0002，无回归。
- **计时**（745 步，warm 中位 ms）：

| 阶段 | R4 后 | +A/D | +B | +C | 五轮累计 |
|---|---|---|---|---|---|
| adapt | 54.2 | 54.5 | 54.5 | 54.3 | 持平（未动 ✓） |
| norm | 7.1 | **4.8** | 4.8 | 4.8 | -2.3 |
| fwd | 70.2 | **43.0** | **22.4** | **18.5** | -51.7 |
| post | 8.8 | 8.9 | 8.9 | 8.7 | 持平 |
| **total** | **141.1** | **112.0** | **90.6** | **86.6** | **615→86.6 = -85.9%** |

## 可信推理（等价性）

- **全量计算 + 置零的三条论证**（见 B 原理）是数学性质：行独立算子下每行不感知其他行；IEEE 有限值 × 0 = +0；LayerNorm 的 eps 保证零输入行输出有限。对拍 bit-exact 实证。
- **包装互逆**：`(x - σ·(x - α·out)/σ)/α` 代数化简恒等 `out`，纯符号推导（数学必然）；浮点差来自舍入顺序，方向是 fast 版更少运算次数。
- **cache_compile 无数值影响**：加载的是同一份编译产物，运行时与首次编译产物执行路径相同（源码结构事实）；分数实测无回归。
- **fast_dpm 系数与上游同源**：预计算调用上游 `NoiseScheduleVP` 类的同名方法，公式逐条相同，仅计算设备（CPU vs NPU）与求值时机不同——系数差异只剩 transcendental 函数的设备实现 ulp 差。

## 推测未严格证明

- **cache_compile 的 -3.9ms/步 归因于 dynamo guard 消失**：机制上成立（cache_compile 重放 marshal 的 code object，跳过 `torch.compile` 每步的 Python 帧捕获 + guard 匹配；源码结构可见），但未做直接消融（如 profile guard 耗时）验证 4ms 全部来自 guard。
- **fast_dpm 实测 -27.2ms 超火焰图预估（~13ms）的归因**：推测火焰图只计入 dpm_solver 帧的 self time，而包装 op 的开销分两部分——host 侧 launch（在火焰图可见）+ device 侧排队执行（异步，摊到 fwd 段总时长，火焰图不可见）——后者未被预估计入。未做逐项计时验证。
- **首步仍剩 35s 的构成**：推测是 GE 侧首次执行（算子加载/子图编译）不走 cache_compile 的 pickle 通道。未深究。

## 事实：五轮全链路对比（终版）

**性能（5 场景 one_continuous_log，单线程独占 310P，warm 中位）**：

| 轮次 | 改动 | total | 单轮收益 | 累计 |
|---|---|---|---|---|
| 起点 | 闭环跑通形态（jit 窗口还在） | 615 | — | — |
| R1 (0819) | jit 窗口关闭 | 552 | -63 | -10.2% |
| R2 (0819) | mapproc 插值向量化（单线版） | 417 | -135 | -32.2% |
| R3 (0819) | route 外提 + TorchAir 图模式 | 267.6 | -149 | -56.5% |
| R4 (0820) | adapt 缓存+向量化 | 141.1 | -126.5 | -77.1% |
| **R5 (0821)** | **fwd：fast 采样器 + encoder 图化 + cache_compile + norm 缓存** | **86.6** | **-54.5** | **-85.9%** |

**子块明细**（warm 中位）：

| 子块 | R4 后 | R5 后 | 说明 |
|---|---|---|---|
| adapt | 54.2 | 54.3 | agents 18.8（devkit floor）/ mapquery 14.5 / mapproc 9.6 / route 8.1 / to_tensor 1.6 |
| norm | 7.1 | 4.8 | 常量设备缓存 |
| fwd | 70.2 | **18.5** | encoder ~2ms（图）+ pos ~1ms（图外）+ 采样循环 ~15ms |
| post | 8.8 | 8.7 | .cpu() 同步点 + 轨迹插值 |
| total | 141.1 | **86.6** | |

**精度（final_score）**：

| 检查点 | 分数 |
|---|---|
| 闭环首跑 50 场景全集（优化前，唯一一次全量） | 0.9170 |
| 5 场景基线锚点 | 0.9942 |
| R4 后 | 0.994266 |
| R5 批次1 / 2 / 3 | 0.994213 / 0.9942 / 0.9941 |
| R5 对拍 | fast_dpm rel ≤9e-4；encoder bit-exact |

## 沉淀

本轮的通用模式已提炼进两个 skill（`embodied_ai/.claude/skills/`，随 Syncthing 同步）：
- **cuda-to-npu**（NPU 特有）：compiler_constraints.md 新增第 3 章（布尔散回根治：全量+mask 置零）、第 4 章（数据依赖 if→where）、第 5 章（三类图障碍速查：EZ1001 硬件缺算子 / ERR03007 动态 shape / ge_converter 无转换器）；torchair README 新增第 10 条 cache_compile（含两个坑）
- **inference-optimization**（新建，设备无关）：分侧计时 / 循环不变量外提 / 采样器系数预计算+包装互逆 / 常量缓存 / 等价性对拍；`references/diffusion-planner-case.md` 承载五轮全链路案例

## 工件

- 计时日志：`/data/syx_dp/r5/bench_r5_fastdpm.log`（批次1）、`bench_r5_staticenc.log`（批次2/3，末次为 cache 命中版）；归档 `dev_logs/run_prints/2026-08-21_fwd-graph-mode/`
- 对拍：`test_fast_dpm_equiv.py`、`test_static_encoder_equiv.py`（仓库根，已随 f43161f 提交）
- torchair cache：`/data/syx_dp/torchair_cache/{DiTBody,StaticEncoderBody}_static_<md5>/`
- 代码 commit：`f43161f`（npu_port 分支，R1~R5 全部代码 + 工具）
