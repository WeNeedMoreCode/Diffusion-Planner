# scripts 档案：开发期工具与等价性对拍

> 私人离线档案（随 syx_docs 保存，**不上库**）。主线仓 clean code 时已把这里的六个文件清出仓库根——四个等价性对拍是"随时可拷回复跑"的活证据，两个过时基准是历史存档。R1–R9 的"严格等价"主张全部由四个对拍支撑；删除的两个工具的证据链在各轮 summary。

## 等价性对拍四件（active，可复跑）

对拍模式统一：同输入喂新旧两版实现，报差值，判据对应改动的数学性质。复跑方法：拷回仓库根 `python test_xxx.py`。

| 文件 | 守的改动 | 参考方 | 判据 | 实测 |
|---|---|---|---|---|
| `test_interpolate_equiv.py` | R2 插值向量化（shapely→numpy 弧长参数化） | 内嵌旧 shapely 版 | worst < 1e-9 | 7.1e-15（900 用例，注入零长段/叠点） |
| `test_adapt_equiv_r4.py` | R4 数据管线向量化（四段） | 内嵌 R4 改写前 git HEAD 旧版 | 逐位（插值批量层 <1e-10） | bit-exact（900+300+300+200 用例，插值 4.7e-13） |
| `test_fast_dpm_equiv.py` | R5 预计算系数采样器 | 上游 `dpm_sampler` 现行代码 | rel > 1e-3 判公式错 | rel ≤ 9e-4（真 ckpt＋capture，同种子 xT） |
| `test_static_encoder_equiv.py` | R5/R6 静态 shape 化编码器 | 上游 `Encoder.forward` 现行代码 | 逐位 | bit-exact 100%（真 ckpt＋capture） |

要点：

- **参考方分两类**：内嵌旧码（前两个——旧实现已从主线删，冻结在对拍里即设计）；上游现行代码（后两个——上游不许动，无需内嵌）。
- **判据分级对应数学性质**：纯重排（浮点序不变）必须逐位；结合顺序变了给 ulp 容差；代数变形（包装互逆消去）给 1e-3 相对容差。
- **退化注入**是故意的：零长折线、单点线、空帧、叠点——生产里真实出现（占位槽、短路线）。
- 后两个要求 eager body（脚本内 assert `DP_TORCHAIR` 未开），把图编译浮点噪声隔离在外；需要 NPU 容器＋真 ckpt＋capture 输入。前两个 CPU＋numpy 即可（第一个还要 shapely，第二个 import devkit 的两个枚举常量）。

## 过时基准两件（archived，勿再当工具用）

### `bench_dp_forward.py`（R1，2026-08-19）

forward 基准：捕获输入脱离 nuPlan 重放，测全 forward＋encoder/decoder 拆分（每段同步读钟）。战功：拆出 fwd 70.2ms 中 decoder 占 ~85%，把优化火力定到扩散采样；R2 演进表 70.2→43.0→22.4→18.5 每步是它测的。

**下线原因**：①capture 语义漂移——它假设归一化后捕获，R8 v3 起捕获改归一化前 raw，重放新 capture 时 eager/torchair 路径缺归一化、数值语义已破；②默认 `DP_JIT_WINDOW=1` 的基线口径已废（jit 窗口被证伪永久关闭）；③继任者 `bench_step.py`（raw 语义、五段拆分）全面接管。

### `bench_torchair_dit.py`（R3，2026-08-19）

torchair 可行性验证：DiT 采样主体三组对照（生产 eager DiT / RouteEncoder 外提的 eager body / torchair 图编译 body），报图构建耗时、三组时延与加速比、数值对拍。**`DiTBody` 类在这个文件里首次诞生**，验证通过后搬进 `decoder.py` 成为主线代码。

**下线原因**：①使命（可行性实验）完成，结论固化在 R3 summary 与设计文档；②文件内的 DiTBody 是 `decoder.py` 正式版的逐字复制品，正式版演化后成僵尸影子代码；③torchair 路径时延对账走闭环 dp-stage，不走它。真要单独压 torchair DiT，用正式 DiTBody 重写 30 行即可。
