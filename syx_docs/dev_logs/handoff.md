# Handoff

> 给 /compact 工作流的三段复制内容。每次更新时重写①②③。

## ① Compact 参数（贴到 `/compact ` 后面）

```
保留：设计说明书两部分均已完成（Part 1 上游架构 + Part 2 本仓 NPU 适配设计均已按七条写作纪律核查定稿，路径链接 18 处验证通过）；七条写作纪律与 py-spy profiling 已沉淀进 skill（design-spec 终态、inference-delivery optimization 分支）；clean code 进展（已删 bench_dp_forward/bench_torchair_dit、六个工具与对拍归档至 syx_docs/scripts 并附 README 档案）；工作流约束（本机 Windows 改文档、内层 git 实证、Syncthing 当前停用——服务器端有待处理的删除事件，恢复同步前须本机 folder 改 sendonly 对齐，勿直接恢复 sendreceive）。丢弃：Part 2 核查与链接格式试错的全部过程、逐文件审读对话。
```

## ② Post-compact 首句（贴到压缩后 session 第一句）

```
继续 clean code 清档（R10 前置）：syx_docs/scripts 归档已完成（README 在 syx_docs/scripts/README.md）。剩 read_scores.py、data_process.sh、外层 read_results.py 未审；审完后进入 R10 上库收拢（清洗 7 处 syx_docs/skill 引用、9 处 /data/syx_dp 默认路径默认值）。commit 已做完内层与外层两笔。
```

## ③ Export 标题建议（/export 时复制）

```
D:\compass\modelzoo\ModelZoo-PyTorch\ACL_PyTorch\built-in\embodied_ai\Diffusion_Planner\Diffusion-Planner\syx_docs\dev_logs\chat_exports\20260914-Part2核查与clean-code清档启动
```
