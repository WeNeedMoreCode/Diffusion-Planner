# 001 — nuplan-devkit 依赖选择性安装（不整体 pip install -r requirements.txt）

**日期**：2026-08-17
**状态**：accepted（已验证跑通：run_simulation 深度 import OK、torch↔numpy 桥 OK、12 关键模块全 OK）

## 背景（Context）

nuplan-devkit 1.2.2 的 `setup.py` 不声明 install_requires，`pip install -e .` 不拉任何依赖。仓库 `requirements.txt` 是唯一清单，但它是 2022 年代完整开发环境清单，直接 `-r` 安装在 **aarch64 + py3.9 + torch 2.1** 容器里有硬伤：numpy 被降到 1.23.4、setuptools 被 pin 老、grpcio/opencv/Fiona/guppy3 等 pin 无 aarch64 py39 wheel（源码编译要 GDAL 等系统库）、外加大量仿真用不到的重包。

## 决策（Decision，最终生效方案）

**选择性安装 + 深度 import 迭代补齐**，完整命令见 [setup.md §3](../setup.md)。要点：

1. **保留官方 pin**（nuplan 真依赖）：`hydra-core==1.1.0rc1`、`SQLAlchemy==1.4.27`、`bokeh==2.4.3`
2. **numpy 全程钉 1.26.4**（torch 2.1 的 numpy 桥只认 1.x；且必须**最后再钉一次**——后续装包会把它顶到 2.x，`from_numpy` 直接 `RuntimeError: Numpy is not available`，实测中招）
3. **pip 降 <24.1**：`hydra-core==1.1.0rc1` 的依赖 `omegaconf==2.1.0.rc1` 用旧式元数据（`PyYAML>=5.1.*`），pip ≥24.1 拒收
4. **`rasterio` 走 conda**（cp39+aarch64 零 pip wheel，`--only-binary` 探测实锤"from versions: none"）：SJTU 镜像 + `repodata_use_zst false`（conda 先请求 `.zst`，SJTU 的 S3 对缺失文件回 403，conda 不回退）+ `--override-channels`（`-c` 只是追加，defaults 源会卡死索引收集 11min+）
5. **`Fiona` 不装**：全库 grep 零直接 import（读 .gpkg 的是 `pyogrio`，`gpkg_mapsdb.py:14`）；Fiona 无 wheel 且编译要 GDAL
6. **import 闭包补齐**（`run_simulation` 深度 import 逐个暴露，全是顶层 import 躲不开）：`pytorch_lightning==2.1.0`（配 torch 2.1；devkit 的 requirements 竟未声明）、`aioboto3 aiofiles boto3 s3fs`（s3_utils 顶层）、`opencv-python-headless`（image.py）、`pytest`（lidar.py 生产代码导入）、`tensorboard==2.11.2`、`pyinstrument`、`mmengine`（diffusion_planner 自己的 train_utils 顶层 import，requirements_torch.txt 有但属训练侧，首装漏了）
7. **bokeh 2.4.3 的 `np.bool8` sed 补丁**（实测生效）：该别名 numpy 1.24 已删而环境钉 1.26.4；bokeh 3.x 升级路线被 nuplan 的 `from bokeh.plotting import Figure`（3.0 已删该符号）挡死，且 tab 配置在 MetricSummaryCallback 的 import 链上、仿真启动就炸。numpy 降级会连坐 pyarrow/torch。遂 sed site-packages 的 `np.bool8`→`np.bool_`（类型等价）；按 [decisions/002](002-env-patches-as-monkeypatch.md) 后续收编猴子补丁
7. **跳过**（import 闭包确实不碰）：`grpcio`（提交容器通信）、`casadi`/`control`（PDM planner 专用，跑 diffusion_planner 不触发）、`selenium`/`jupyter*`（nuBoard/notebook）、`moto`/`mock`/`coverage`/`hypothesis`/`pre-commit`/`docker`、`guppy3`

## 备选方案（Alternatives）

| 方案 | 优点 | 缺点 | 为啥不选 |
|------|------|------|---------|
| 选择性安装 + 迭代补齐（选定） | 装得上、不破坏 torch 环境 | 前期迭代多轮 | — |
| `pip install -r requirements.txt` | 与官方一致 | numpy 降级 / 无 wheel 源码编译必炸 / 大量重包 | 坑致命 |
| 官方 nuplan docker 镜像 | 环境一致 | 无 NPU/torch_npu | 平台不符 |

## 后果（Consequences）

- 正向：环境完整可用（验证状态见"状态"行）；体积可控
- 负向：若运行期再触发缺包（深层惰性导入），补装对应新版后**必须复查 numpy 是否被顶到 2.x**
- 镜像可用性（2026-08-17 实测本服务器）：pip 华为云 ✅；conda-forge 清华 403、华为云/中科大跳转只给 HTML、官方 ~690KB/s 慢、**SJTU 可用**（禁 zst 后）

## 相关

- [setup.md §3](../setup.md)（完整安装命令）
