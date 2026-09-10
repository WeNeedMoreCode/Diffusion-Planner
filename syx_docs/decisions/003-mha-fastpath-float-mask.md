# 003 — MHA fast path 踩 310P 缺失算子：float mask 强制分解路径

**日期**：2026-08-18
**状态**：accepted（已验证：50/50 场景、final_score 0.9170）

## 背景（Context）

闭环仿真首个 NPU 前向即炸：

```
RuntimeError: _transform_bias_rescale_qkv ... aclnnTransformBiasRescaleQkv failed, error code 161001
AclNN_Parameter_Error(EZ1001): socVersion [ascend310p] does not support opType [TransformBiasRescaleQkv]
```

排查链条（每个假说都做了实验，记录以免重走）：

| # | 假说 | 验证方式 | 结果 |
|---|------|---------|------|
| 1 | `set_compile_mode(jit_compile=False)` 导致只用预编译库 | 隔离脚本（同参数 MHA，不设 False）单进程/4 进程并发 × 300 迭代 | **全通过**——且后来证明隔离脚本根本没走到出事路径（假阴性复现） |
| 2 | JIT 模式能现编该算子 → planner.py 加 jit 窗口（前向 True / 后 False） | 真实仿真重跑 | **仍炸同错**——算子在整个 310P binary_info_config.json 里不存在，JIT 也无米下锅 |
| 3 | 4 个 Ray worker 共享单卡引发调度错乱（日志有 RTS_SCHED 报错） | 4 进程并发压测 | 干净，排除 |
| 4 | 从源码定位 | clone `gitcode.com/Ascend/pytorch`（torch_npu 本体）+ 读服务器上 torch 2.1 的 `MultiheadAttention.forward` 源码 | **命中**，见下 |

根因（源码实证）：
- torch core 的 `nn.MultiheadAttention.forward` 在 **eval + 自注意力（q/k/v 同张量）+ batch_first** 等条件满足时走 **fast path**（`_native_multi_head_attention`，traceback 里 activation.py:1196 即此处）
- torch_npu 把该路径映射到融合算子 `aclnnTransformBiasRescaleQkv`——**310P 没有这个算子的二进制**（EZ1001 直说了）。算子功能本身很朴素（torch_npu 的 test_native_mha.py 有纯 Python 参考实现：拆 QKV + 加偏置 + Q 预乘 1/√d），但没有 Python 回退
- 本模型只有 `dit.py` DiTBlock 自注意力一处满足 fast path 条件（encoder 的 q=`norm1(x)`≠k=`x`、cross-attn 同理，被"non-self attention"条件天然挡住）

## 决策（Decision）

**在 dit.py 把 bool `key_padding_mask` 转为 float 加性 mask（`0 / -inf`）**。原理：fast path 条件列表的**第一条**就是 `floating-point masks are not supported for fast path`——float mask 直接否决 fast path，强制走分解路径（linear + bmm + softmax，NPU 全支持）。语义完全等价（key_padding_mask 的 float 形式本来就是加性 mask 官方语义），CUDA 上同样正确，一行改动。

## 备选方案（Alternatives）

| 方案 | 优点 | 缺点 | 为啥不选 |
|------|------|------|---------|
| float mask（选定） | 一行、语义等价、跨平台正确、零维护 | 放弃融合路径的潜在性能 | — |
| jit_compile 窗口 | 不改模型代码 | **实测无效**（算子不存在，非编译模式问题）；且**有害**——2026-08-19 replay A/B 实测 jit_compile=True 前向窗口每步多耗 ~48ms（245 vs 197ms），已在 planner.py 默认关闭（`DP_JIT_WINDOW` 默认 `"0"`） | 实测否决 |
| 猴子补丁 torch_npu 的 MHA 分发逻辑 | 保留 fast path 可能性 | torch_npu 分发在 C++/闭源 aclnn 层，Python 侧补丁点不明确；升级即碎 | 风险大收益不明 |
| 手写 attention 替换 nn.MultiheadAttention | 完全可控 | 改动大、要自证数值等价、丢官方实现的长尾正确性 | 过度工程 |
| AscendC 自实现该算子 | 根治 | 工作量最大（算子开发+注册+对接），为一次性推理路径不值 | 严重过度 |

## 后果（Consequences）

- 正向：单行修复即全线跑通（50/50 验证）；CUDA 环境跑同一份代码不受影响
- 负向：放弃融合 attention 路径，单步时延含此代价（615ms 中占比未拆分）；若未来 CANN 补齐 310P 的该算子，可考虑改回 bool mask 以换性能
- 教训入 skill 候选：**"隔离复现通过 ≠ 没问题"**——本例隔离脚本因走了不同分支而假阴性，真模型前向复现才是金标准

## 相关

- [setup.md §7](../setup.md)（运行期补丁清单）
- [decisions/002](002-env-patches-as-monkeypatch.md)（其"MHA 分发补丁占位"已被本决策取代——见该文更正）
- [summary/2026-08-18](../dev_logs/summary/2026-08-18_npu-closed-loop-first-success.md)
