# 002 — 环境兼容补丁收编为猴子补丁（进 Diffusion-Planner 代码库）

**日期**：2026-08-18
**状态**：accepted（方向已定，实施待做——见 TODO）

## 背景（Context）

在 NPU 容器里跑通闭环仿真的过程中，对**已安装的第三方包**做过一处直接修改（sed 改 site-packages 源文件）：

| 补丁 | 内容 | 当时为什么这么打 |
|---|---|---|
| bokeh 2.4.3 `np.bool8` | 容器 `syx_dp` 内 `/usr/local/miniconda/envs/diffusion_planner/lib/python3.9/site-packages/bokeh/` 所有 `np.bool8` → `np.bool_` | bokeh 2.4.3（nuplan-devkit 硬 pin，nuBoard 依赖）用 `np.bool8`，该别名 numpy 1.24 已删（环境为 1.26.4）。bokeh 3.x 又被 nuplan 的 `from bokeh.plotting import Figure`（3.0 已删）挡在 import 链上。numpy 不能降（pyarrow/torch 锁死）。sed 是当时唯一活路 |

**问题**：这类补丁打在容器 overlay 层，**容器删除 / `pip --force-reinstall` 即失效**，且不留任何痕迹，下次环境重建会神秘复发。

## 决策（Decision）

**所有环境兼容类补丁一律收编为猴子补丁，放进 Diffusion-Planner 代码库**（随 Syncthing 同步、随代码走，环境重建后自动生效），不再手改 site-packages。

TODO（跑通仿真后实施）：
1. 建 `diffusion_planner/compat.py`，在 `planner.py` import 链最前面 import 它。首个补丁一行即可恢复别名：`numpy.bool8 = numpy.bool_`（在 bokeh 被 import 之前执行，等效于 sed 且零侵入）
2. ~~torch_npu 的 MHA 分发路径问题（310P 无 `TransformBiasRescaleQkv` 实现）如最终也是 patch 分发逻辑解决，同样进 compat.py（尚未验证可行性，先占位）~~（2026-08-18 更正：最终**未走补丁路线**，以 dit.py 的 float mask 强制分解路径解决，属模型代码而非环境补丁，见 [decisions/003](003-mha-fastpath-float-mask.md)。此条占位作废）

## 备选方案（Alternatives）

| 方案 | 优点 | 缺点 | 为啥不选 |
|------|------|------|---------|
| 猴子补丁进代码库（选定） | 随代码走、可版本管理、环境重建自愈 | 进程内生效（对本项目够用） | — |
| sed 改 site-packages（现状） | 立即生效 | 容器重建即丢、无痕迹、难维护 | 长期不可持续 |
| 锁定环境镜像 | 最干净 | 换镜像/升级 numpy 又要重做 | 当前阶段过重 |

## 后果（Consequences）

- 正向：环境补丁有单一权威来源（代码库），重建环境零手工步骤
- 负向：compat.py 里的补丁和包版本强耦合（bokeh 升级后 `np.bool8` 补丁应移除）——每个补丁注明针对的包版本与移除条件

## 相关

- [decisions/001](001-nuplan-devkit-dependency-deviation.md)（依赖偏差总账）
- [setup.md](../setup.md)（环境搭建，补丁实施后同步更新）
