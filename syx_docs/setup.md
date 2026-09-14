# Diffusion-Planner NPU 移植 — 开发环境配置

**用途**：新机器 / 新 session 配置环境的参考。

**最后更新**：2026-09-03

## 1. 工作区关系

- 本地 Windows（改代码）：`D:\compass\modelzoo\ModelZoo-PyTorch\ACL_PyTorch\built-in\embodied_ai\Diffusion_Planner\Diffusion-Planner`（2026-08-19 由拼错的 `Diffussion_Planner` 正名，单 s）
- 服务器（跑代码）：`root@192.168.13.119`，容器 `syx_dp`
- 服务器项目根（下文 `$DP_ROOT`）：`/home/syx/ModelZoo-PyTorch/ACL_PyTorch/built-in/embodied_ai/Diffusion_Planner/Diffusion-Planner`
- 同步：Syncthing 文件夹 `modelzoo-pytorch` 整树双向实时（本地 `D:\compass\modelzoo\ModelZoo-PyTorch` ↔ 服务器 `/home/syx/ModelZoo-PyTorch`）。本地改完约 10s 内服务器可见，无需手动传
- SSH 统一用 `ssh -F /c/sshkeys/config 119`（key/config/known_hosts 全在 ASCII 路径；中文 HOME 会让裸 ssh 报 Host key verification failed，详见 remote-ssh-windows skill）；**必须先连 VPN**（不通时 ping 都超时）

## 2. Git 托管

- 上游 `https://github.com/WeNeedMoreCode/Diffusion-Planner.git`（本地有 `.git`；`.git` 被 `.stignore` 排除不同步，服务器无 git 历史）
- nuplan-devkit：上游 motional/nuplan-devkit，clone 在同级 `../nuplan-devkit`（commit `e924167`，随同步树到服务器）

## 3. 语言 / 包管理器

服务器容器 `syx_dp`（镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-onnx-inference:cann8.3.rc1_torch2.1.0-300I-DUO-ubuntu22.04-py3.11-aarch64`，aarch64）。

conda 环境 `diffusion_planner`（**Python 3.9**，路径 `/usr/local/miniconda/envs/diffusion_planner`）。py3.9 原因：nuplan-devkit 1.2.2 官方按 py3.9 开发，镜像自带 py3.11 有兼容风险；torch/torch_npu 按镜像同版本在 py3.9 里重装。

容器内安装命令（依赖取舍理由见 [decisions/001](decisions/001-nuplan-devkit-dependency-deviation.md)）：

```bash
P=/usr/local/miniconda/envs/diffusion_planner/bin
CONDA=/usr/local/miniconda/bin/conda

# 1) pip 降 <24.1（新版拒收 omegaconf 2.1.0rc1 的旧式元数据）
$P/pip install "pip<24.1"
# 2) torch 对齐镜像版本
$P/pip install torch==2.1.0 torch_npu==2.1.0.post17 "numpy<2" timm==1.0.10
# 3) 两个 editable 包（setup.py 均不声明依赖）
cd ../nuplan-devkit && $P/pip install -e .
cd ../Diffusion-Planner  && $P/pip install -e .
# 4) core 依赖
$P/pip install "hydra-core==1.1.0rc1" "SQLAlchemy==1.4.27" "bokeh==2.4.3" \
    ray pandas pyarrow scipy shapely matplotlib \
    cachetools pyquaternion ujson joblib psutil requests retry
# 5) import 闭包补齐（run_simulation 深度 import 实测逐个暴露，均顶层导入，见 decisions/001）
$P/pip install pytorch_lightning==2.1.0 aioboto3 aiofiles boto3 s3fs \
    opencv-python-headless pytest tensorboard==2.11.2 pyinstrument
# 6) rasterio 走 conda（cp39+aarch64 零 pip wheel；SJTU 镜像 + 禁 zst + 踢掉 defaults，见 decisions/001）
$CONDA config --system --set repodata_use_zst false
$CONDA install -n diffusion_planner -y --override-channels \
    -c https://mirror.sjtu.edu.cn/anaconda/cloud/conda-forge \
    rasterio "numpy=1.26.4" "python=3.9"
$P/pip install geopandas pyogrio
# 7) 最后钉回 numpy（步骤 5 的包会把它顶到 2.x，torch 2.1 的 numpy 桥会断：
#    from_numpy 报 "Numpy is not available"，planner 的 .numpy() 主路径会炸）
$P/pip install "numpy==1.26.4"
```

pip 源：镜像已配华为云（torch_npu 可直装）。跑任何东西先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，且用 `$P/python`（容器自带 py3.11 不是我们的环境）。

## 4. 数据 / 模型

| 内容 | 服务器路径 | 来源 |
|---|---|---|
| mini db | `$DP_ROOT/datasets/data/nuplan-v1.1/splits/mini/` | https://www.nuplan.org/nuplan 注册下载（S3 匿名直链 2026-08 实测 403 已关闭） |
| 地图 | `$DP_ROOT/datasets/maps/nuplan-maps-v1.0/` | 同上；**目录名 = `nuplan_mini.yaml` 的 `map_version`，不能改** |
| ckpt | `$DP_ROOT/checkpoints/{args.json, model.pth}` | HuggingFace `ZhengYinan2001/Diffusion-Planner` |

`.stignore`（ModelZoo 根，两边各一份、不自动同步）已排除 `datasets/`、`*.pth`、`exp/`——服务器大文件不倒灌本地。

## 5. 硬件

- NPU 300I-DUO（310P3），共享服务器：device 0-3 常被 vllm/mindie 占，4-7 常空闲；**跑前 `npu-smi info` 实时查**，挑 <2GB 占用的卡设 `ASCEND_RT_VISIBLE_DEVICES`
- 验证过的状态（2026-08-18）：`torch.npu.is_available()`=True；`run_simulation` 深度 import OK；`torch.from_numpy` 桥 OK（numpy 1.26.4）；ray/hydra/shapely/pandas/pyarrow/matplotlib/bokeh/scipy/sqlalchemy/geopandas/pyogrio/rasterio 全 OK；**闭环仿真已跑通：50/50 场景成功，final_score 0.9170**（one_continuous_log × closed_loop_nonreactive_agents，单步中位 615ms，详见 [dev_logs/summary](dev_logs/summary/2026-08-18_npu-closed-loop-first-success.md)）

## 7. 跑通闭环的三个运行期补丁（2026-08-18 实测）

1. **`RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`**（sim 脚本已 export）：Ray 对 `num_gpus=0` 的任务会清空 worker 的 `ASCEND_RT_VISIBLE_DEVICES`（空串 = 零卡可见）→ 每个 Ray worker 里 `torch.npu.is_available()=False`，`torch.load(map_location='npu')` 直接炸。此环境变量让 Ray 不动加速卡可见性变量
2. **`dit.py` DiTBlock：bool key_padding_mask → float 加性 mask**：torch 的 MHA fast path（eval + 自注意力 + q/k/v 同张量时自动启用，即 `nn.MultiheadAttention.forward` 里的 `_native_multi_head_attention`）在 NPU 上映射到 `aclnnTransformBiasRescaleQkv`，**310P 无此算子二进制**（EZ1001，JIT 也救不了——算子根本不存在）。float mask 是 fast path 条件列表的**第一条否决项**（"floating-point masks are not supported for fast path"），强制走分解路径（linear+bmm+softmax），语义等价、CUDA 上同样正确。全模型仅 `dit.py` 自注意力一处踩雷（encoder/cross-attn 因 q≠k 天然免疫）。~~`planner.py` 里的 jit 窗口（前向前后 True/False 切换）对本案无效但无害，留着~~（2026-08-19 更正：实测**有害**——jit 窗口每步多耗 ~48ms，已在 planner.py 默认关闭，恢复旧行为需 `DP_JIT_WINDOW=1`，见 [summary/2026-08-19](dev_logs/summary/2026-08-19_perf-breakdown-stage-timing.md)）
3. **bokeh 2.4.3 的 `np.bool8`**：该别名 numpy 1.24 已删；已 sed 补丁 site-packages（`np.bool8`→`np.bool_`），待按 [decisions/002](decisions/002-env-patches-as-monkeypatch.md) 收编为 compat.py 猴子补丁

## 6. 常见踩坑

- **S3 匿名下载 403**：老文档的 `aws s3 sync --no-sign-request s3://nuplan/...` 已失效，走 nuplan.org 注册
- **镜像别选错**：ascendhub-test 有同名 tag 的残缺 stub（content 31kB），用 south-1 完整版
- **跑仿真用 conda env 的 python**：`PATH` 前置 `/usr/local/miniconda/envs/diffusion_planner/bin` 再 `bash sim_diffusion_planner_runner.sh`
- nuPlan Ray 在 NPU 上的调度：`number_of_gpus_allocated_per_simulation=0`（Ray 看不到 NPU），并发靠 `worker.threads_per_node`

## 8. 性能计时工具（2026-08-19 加，详见 summary/2026-08-19）

仿真与压测都从这些环境变量控制（不改代码）：

| 变量 | 作用 | 默认 |
|---|---|---|
| `DP_STAGE_TIMING=1` | 每步打印 `[dp-stage]`（adapt/norm/fwd/post/total）+ `[dp-adapt]`（ego/agents/route/mapquery/mapproc/to_tensor）分阶段毫秒数 | 关 |
| `DP_JIT_WINDOW=1` | 恢复旧 jit 窗口行为（实测每步慢 ~48ms，仅作对照用） | `0`（关） |
| `DP_TORCHAIR=1` | DiT 采样主体 + encoder（StaticEncoderBody）走 torchair 图编译（fwd 70→18.5ms） | **`1`（默认开，50 场景全量 0.9170 已验证；`=0` 回退 eager）** |
| `DP_FASTDPM=1` | 预计算系数的 10 步 DPM-Solver++ 替代上游 dpm_sampler（fwd -27ms，对拍 rel ≤9e-4） | **`1`（默认开，同上；`=0` 回退上游路径）** |
| `DP_OM=loop` | OM 离线路径（`export_om.py` 产三图 → aclruntime）：`loop`=整循环图每步 1 次 OM 调用（推荐，R9：DUO 70.5ms / RC 90.0ms，双板 0.9171）；`1`=逐次 dit_body（A/B 诊断用）。优先级高于 DP_TORCHAIR。R9 起 DP_OM 下 adapt 产出留 host（免上卡） | 关 |
| `DP_OM_DIR` / `DP_OM_DEVICE` | OM 产物目录（默认 `$DP_DATA/om/om_models`，未设 DP_DATA 时 `./Diffusion-Planner/om/om_models`）/ InferSession 逻辑卡号 | 见左 |
| `DP_TORCHAIR_CACHE=<dir>` | cache_compile 编译产物目录（跨 worker 复用，首步 74→35s）。**改了 decoder.py/encoder.py 的 forward 后必须清空或换目录**——cache key 无源码 hash，会静默加载旧图 | `$DP_DATA/torchair_cache` |
| `DP_WORKER=sequential` | 单进程串行（无 Ray）排障模式——多 worker 的 aicpu 异常会互相拖挂设备、报错位置误导 | `ray_distributed` |
| `DP_DATA=<dir>` | 数据根（datasets/checkpoints/exp/torchair_cache 跟随）；mini 包解压到 `datasets/data/cache/mini/` | 脚本所在目录（119 用 `/data/syx_dp` 覆盖） |
| `DP_CAPTURE_DIR=<dir>` | 每 worker 首步把模型输入存 `<dir>/inputs_pid*.pt`（**R8 起 raw 语义**：存 norm 之前——encoder.om v3 直吃 raw，export/bench_step 都要 raw capture；旧 norm 后 capture 作废） | 不捕获 |
| `DP_DEVICE` / `DP_LIMIT` / `DP_THREADS` | runner 脚本覆盖：NPU 卡号 / 场景数 / Ray 线程数 | 0 / 50 / 4 |

runner 结尾自动调外层样例目录的 `read_results.py`，打印本次 final_score、单步中位/均值、wall time、sim log/metric 产物计数。

### 运维注意（2026-08-28 事故沉淀；08-31 增补）

- **Syncthing 假同步**：119 根盘 ≥99% 时 Syncthing index 拒写（要求所在盘 ≥1% 余量），folder 进 error 态且**本地端仍显示 idle**——代码同步静默失效。验证：服务器端 `curl 127.0.0.1:8384/rest/db/status?folder=modelzoo-pytorch` 看 state/needFiles（API key 在 `/root/.config/syncthing/config.xml`）。盘满止血：挪大权重到 /data + 原位软链。
- **git 与 Syncthing 拉锯**：工作区文件的服务器版本 ≠ git HEAD 时，任何 stash/checkout 回滚都会被 Syncthing ~10s 内同步回来。切分支被卡用 `checkout -f`（修改在 stash/同步里不会丢）；切分支前后暂停/恢复两个 folder（本地 GUI 127.0.0.1:18384）。
- **根盘增长源**（08-29 二次撞满）：`/home/l00586152` 的 Qwen3.5-27B 量化系列持续膨胀（9 份 ≈264G，24h +29G）+ docker 停止容器层（单 `v0230-310p-oe` 111G）。已部署容器内 `/root/spill_watch.sh`（规则表驱动溢出搬迁守护，300s 一轮，首搬 35G 311→276G）；**容器重启后需手动重启守护**：`docker exec -d syx_dp bash -c 'nohup bash /root/spill_watch.sh --loop >> /root/spill_watch.log 2>&1 &'`。
- **pip 依赖互踩（08-29/08-31 两次）**：`pip install onnx` / `onnxsim` 会把 protobuf 顶到 6.x → tensorboard 2.11.2 的 pb2 导入链炸，**run_simulation 入口直接挂**（栈在 nuplan import 链，与被装包无关，极难定位）。装完必钉回 `pip install "protobuf==3.20.3"`。
- **PYTHONPATH 必须追加式**：`export PYTHONPATH=<dir>`（覆盖式）会冲掉 `set_env.sh` 注入的 CANN python 路径 → `No module named 'tbe'` → npu_init 500001。正确写法 `export PYTHONPATH=<dir>:$PYTHONPATH`（runner 本来就对）。
- **OM 路径环境件**：ais_bench/aclruntime 从 gitee 页面 README 指向的 OBS 直链装（`aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/`，仓库已不放 whl）；onnxsim 装完回钉 protobuf。

torchair 环境：容器内已装（配套 torch 2.1.0 / CANN 8.3.RC1 / 310P，源码树在服务器 `/data/syx_dp/torchair`）。华为云 pypi 源**没有** torchair 包，容器重建后需重新源码编译：

### OM/bench 工具速查（外层样例目录，CANN 环境已 source）

```bash
python export_om.py --stage export,atc,val   # 导出+ATC+数值校验（encoder.om v3：raw 7 输入/3 输出；需 raw capture，缺省用合成输入）
python bench_om.py [--mode mixed]            # 三 OM 图时延；mixed=闭环数据路径分段（d2h/infer/h2d/sync）
python bench_step.py <raw_capture_dir>       # 闭环单步分段重放（v3 段=enc_om/prep/dit_om/invnorm/post_sim + cpu_probe 标尺）
python bench_launch.py                       # 发射邮费五项（host 派发/投递/小 kernel/算力标尺/往返），us/op
python bench_route.py <norm_capture_dir>     # 历史工具：route CPU-vs-NPU 对拍+线程扫描（期望 norm 后 capture）
python ctx_probe.py                          # ACL context 链隔离探针（A=null save/B=restore 无效/C=链存活）
```

bench 工具纪律（0902 轮教训）：**不主动 `free_resource()`**（第一个 InferSession 复用 torch context，free 即拆）；**`import torch_npu`/`import acl` 必须先于第一个 InferSession**（反序双 libprotobuf 混链 abort，DUO 无感 RC 现形）；脱离闭环的脚本要带 `torch.npu.set_compile_mode(jit_compile=False)`（RC 上 tbe 在线编译崩：qsize→NotImplementedError→500002）。

```bash
# 容器内（gcc 11.4 / cmake 3.22 镜像自带）
git clone --depth 1 -b 7.2.0 https://gitcode.com/Ascend/torchair.git /data/syx_dp/torchair
cd /data/syx_dp/torchair && git submodule update --init --recursive
NO_ASCEND_SDK=1 TARGET_PYTHON_PATH=/usr/local/miniconda/envs/diffusion_planner/bin/python bash ./configure
mkdir build && cd build && cmake .. && make torchair -j8
pip install dist/dist/torchair-0.1-py3-none-any.whl
# 冒烟：python /data/syx_dp/torchair_smoke.py（torch.add 图编译，应打印 SMOKE_OK）
```

用法：

```bash
# 分阶段计时仿真（容器内）
DP_DEVICE=5 DP_THREADS=1 DP_LIMIT=5 DP_STAGE_TIMING=1 bash sim_diffusion_planner_runner.sh 2>&1 | tee bench.log
python analyze_stage_log.py bench.log          # 统计分布（仓库根）

# 捕获输入（给外层 bench_step.py 重放与 export_om.py 导出对拍用；R8 起为归一化前 raw 语义）
DP_CAPTURE_DIR=/data/syx_dp/capture DP_STAGE_TIMING=1 bash sim_diffusion_planner_runner.sh ...

# CPU 侧热点定位：py-spy 火焰图（容器内需先 pip install py-spy）
# 仿真后台跑起来后，worker pid 从 bench.log 的 (wrapped_fn pid=NNNN) 抓
P=/usr/local/miniconda/envs/diffusion_planner/bin
$P/py-spy record --pid <PID> --rate 50 --format speedscope --duration 240 --output /data/syx_dp/pyspy.json
$P/python agg_speedscope.py /data/syx_dp/pyspy.json map_process   # self/total 占比（仓库根）
```
