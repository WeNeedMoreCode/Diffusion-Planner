# 2026-09-14：设计说明书 Part 2 核查收口＋clean code 清档启动

> 9-10 起 session 的三段：①设计说明书 Part 2（本仓 NPU 适配设计）按七条写作纪律逐节核查改造并收口——**两部分全部定稿**；②py-spy 方法论整合进 inference-delivery skill；③Syncthing 服务器删除事件拦截＋clean code 清档启动（六工具处置、syx_docs/scripts 档案建立）。

## 事实：设计说明书 Part 2 核查改造

- **取证**：§5 十三小节＋§6/§7/§8 全部断言对当前源码（planner/encoder/decoder/dit/export_om/om_runtime/npu_utils/fast_dpm_sampler/map_process/roadblock_utils）与 R7–R9 summary 逐条核验。抓到两处失准并修正：StaticEncoderBody 输入"8 张量＋3 pos"→实为 **5 键＋3 pos＝8 输入**（route_lanes 不进编码器）；"三段 ~17.5ms 清零"补标 **RC 账**（norm 5.7＋pos 1.1＋route 10.7，出处 designs/inference-pipeline.md L115；DUO 同三段 ≈12.8ms、单步仅 -3.1）
- **图改造**：图 2.4.1/2.4.2 类名别名法编号 (1)–(13)/续接 (14)–(17)；新增 `om_runtime <<module>>` 框（`_SESSION_CACHE`/`_PTA_CONTEXT` 是模块级全局，原画在 OmBody 类成员里失实）；图 2.8.1 编号引用式重画（(1)–(8) 节点、(A)–(F) 变换、外部输入 (9)(10)(11) 虚线垫底、图下 17 条按数据流顺序逐条带前后对照），**顺带修掉一个结构错误**：原图 encoding/route_encoding 流经"采样准备"框，实际直连 dit_loop.om
- **§4.1 逐类速查补全**：新增/改写 11 条（DataProcessor、map_process、roadblock_utils、fast_dpm_sampler、om_runtime、npu_utils、DiTExportWrapper、export_om 等）——图上成员在解释里全有落点；来源对照表类名括注编号
- **§3 功能点分解** 17 条全部挂链到 §5 对号小节＋条目内术语首现挂链（MHA fast path/布尔散回/调度标量）
- **三处"非人话"重写**：数据管线小节的向量化/不变量缓存两条（拆成"先因果后分项"＋编号子条目）；float-mask 小节一段拆四段（报错原文/JIT 无米下锅/q-k-v 同张量判定/等价性数学）；OM 运行时会话泄漏条展开机制（泄漏链→模块级 dict→总数封顶 2×1）
- **排版**：长条目列表空行纪律铺四处（§4.1、§8 图下、Part 1 §3.1/§3.2）
- **代码路径链接**：终态格式 `[路径:行号](/工作区根相对#行号)`（VSCode 预览 Ctrl+click 直达），§4.1 十三条＋§5 一处共 18 链接，验证行号精确落在 class/def 定义行
- **终态校验**：9 图围栏完整、锚点 96/96 零死链零孤儿

## 事实：skill 变化

- **design-spec** 四条新纪律：清单式章节条目挂链对号小节＋条目内术语挂链；长条目列表空行隔开；"名词（括号塞限定）；名词（…）"分号长链病灶拆引导句＋编号子条目；代码实体路径链接规范（格式/挂载点/验证/两个已知代价）
- **inference-delivery**：optimization 分支新增 `references/profiling-pyspy.md`（采样命令含 `--format speedscope` 必带、找 PID 两法、speedscope.app 三视图、离线聚合、self/total 读数、R4 战例）＋`scripts/agg_speedscope.py` 快照收编；README 第 3 步改链
- 删除 `Diffusion_Planner/.claude/skills/design-spec`（Part 1 沉淀前的旧基础版副本，防止同名 skill 两级漂移）
- memory 删 `py_spy_flamegraph_command.md`（内容归 skill，消除双份）

## 事实：git（两仓四笔）

- 内层 `0089444`：syx_docs 摘出 .gitignore 纳入跟踪＋Part 2 改动（48 文件）
- 外层 `7ec2ed598`：.gitignore 本地 dev 块挪 `.git/info/exclude`（原生本地忽略、永不进库），.claude skills＋vla/GR00T_n1d7 notes 纳入；用户已 push（origin 为私人 fork gitcode/frost_mourne）
- 内层 `868b9a5`＋外层 `6944c0448`：clean code 清档（见下）
- 遗留警示：chat_exports 对话原文在内层 git 历史里，**push 前需处理**

## 事实：Syncthing 删除事件（9-10）

- 用户要求开同步，启动后 **20 秒内**发现服务器端积压的 `Diffusion_Planner` 整目录删除指令正对本机执行（被 shell 占用拦住、每分钟重试）——立即停进程止血
- 服务器实证：reflog **无分支切换**（最后一次 8-18）；磁盘目录内容完整；Syncthing **双进程**（8-19 与 9-08 各一）；服务器 .stignore 排除 `.git/`（git 状态走 push/pull、从不同步——两仓 HEAD 分叉是设计如此）
- 删除指令源头未完全定论（服务器 db 有 8.4 万条删除记录），**同步保持停用**
- 恢复方案（定而未执行）：本机 modelzoo-pytorch folder 临时改 sendonly 覆盖对齐→needFiles=0→改回 sendreceive

## 事实：clean code 清档（R10 前置）

| 文件 | 处置 | 依据 |
|---|---|---|
| analyze_stage_log.py | 留（仓根） | 发布"warm 中位"数字的统计尺子 |
| agg_speedscope.py | 留（仓根） | 火焰图方法论工具（skill 另有快照） |
| bench_dp_forward.py | 删→归档 | capture 语义漂移（post-norm 时代）＋jit 口径废＋bench_step 继任 |
| bench_torchair_dit.py | 删→归档 | R3 可行性实验使命完成＋DiTBody 影子代码 |
| test_{interpolate,adapt_r4,fast_dpm,static_encoder}_equiv.py | 归档 | 等价性证据活体：R 系列全部"严格等价"数字的出处，复跑拷回仓库根即可 |

- 归档地 `syx_docs/scripts/`（README 档案含对拍总表/判据分级/依赖/下线原因）；git 识别四 test 为 100% rename
- 待审：read_scores.py、data_process.sh、外层 read_results.py

## 下一轮

- 继续 clean code 三件待审；之后 R10 上库收拢（7 处 syx_docs/skill 引用、9 处 /data/syx_dp 默认路径）
- Syncthing 恢复（sendonly 对齐流程，方案已定）
