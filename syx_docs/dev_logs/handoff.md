# Handoff

> 给 /compact 工作流的三段复制内容。每次更新时重写①②③。

## ① Compact 参数（贴到 `/compact ` 后面）

```
保留：设计说明书第一部分已定稿（syx_docs/Diffusion-Planner_设计说明书.md，约 850 行/9 图/锚点验证干净）与七条写作纪律（编号引用式、图文一致、前后对照禁堆叠、图号 N.M.K、类图类名别名语法、设计文档口吻、断言须有源码或日志支撑——全文见 summary/2026-09-10 与 design-spec skill 终态）；skill 更新落盘位置（design-spec 注意事项与附录章、syx_skills 记终态不记过程）；工作流约束（本机 Windows 改文档、基线 git 实证用内层 .git、服务器 args.json 已核对超参、mermaid classDiagram 用类名别名法且成员行禁 Unicode 符号）。丢弃：本轮全部编辑过程与用户纠偏对话、mermaid 语法试错历史（终态已进 skill）、Part 1 各节的中间版本。
```

## ② Post-compact 首句（贴到压缩后 session 第一句）

```
继续设计说明书：第一部分（上游架构）已按七条写作纪律定稿，本轮核查第二部分（本仓 NPU 适配设计，§1 Story–§8 数据流）。逐节审：图 2.4.1/2.4.2/2.6.1/2.8.1 是否需编号引用式改造（图下逐条解释、数据流排序）、图文是否一致（图上成员在解释有落点）、术语与接口链接是否补齐（对照 Part 1 的术语表/接口速查表机制）、§5 实现思路各小节是否满足"是什么/为什么/怎么做"且断言有源码或日志支撑。标准全文在 summary/2026-09-10_design-spec-part1.md 的"写作纪律"节。
```

## ③ Export 标题建议（/export 时复制）

```
D:\compass\modelzoo\ModelZoo-PyTorch\ACL_PyTorch\built-in\embodied_ai\Diffusion_Planner\Diffusion-Planner\syx_docs\dev_logs\chat_exports\20260910-设计说明书Part1定稿与写作纪律沉淀
```
