# 2026-08-20 第四轮：CPU adapt 优化（267.6→141.1ms，累计 -77.1%）

> 主攻 handoff ②：adapt 179.8ms/67%（mapquery 长尾 p90 550ms、agents 38ms、_lane_polyline_process 循环）。

## 事实：诊断（py-spy 火焰图）

5 场景仿真 + py-spy record 50Hz speedscope（9788 样本 / 195.8s），`agg_speedscope.py` 聚合：

| 热点 | 样本占比 | 折算（1% ≈ 3.4ms/步） |
|---|---|---|
| `map_process.py:70/74/75` 三行 `[Point2D(node.x,node.y) ...]`（mid/left/right 逐点构造） | 17.3% | ~58ms |
| `MapObjectPolylines.to_vector()` 嵌套 listcomp（devkit）+ 我方 `np.array` 二次打包 | 13.1% | ~44ms |
| agents 三层循环（`_filter_agents_array` O(T·N²)、`_pad_agent_states` 逐行 dict、`_extract_agent_array` 逐字段 setitem） | 4.3% | ~14ms |
| 90×3 次小数组 `_interpolate_points` 调用链 + `_lane_polyline_process` 逐 element 循环 | ~4% | ~13ms |

mapquery p90 550ms 长尾与 17.3% 同源：长尾步 = 周边 lane 多/点多 → 逐点构造线性放大。空间查询 `get_proximal_map_objects` 本身未冒头。

## 改动详账（8 处，全在 `diffusion_planner/data_process/`，devkit 不动）

### A. mapquery：per-lane 折线缓存 + 消 Point2D 逐点构造

**实现**：模块级 `_POLYLINE_CACHE: Dict[(map_name, lane_id, role), ndarray]`。`_get_lane_polylines` 循环体三行逐点构造：

```python
# 旧（每步每 lane 每条折线都执行）
lanes_mid.append([Point2D(node.x, node.y) for node in map_obj.baseline_path.discrete_path])
# 新（首次 miss 转数组，此后字典直取）
lanes_mid.append(_polyline_array(map_name, map_obj.id, "mid", map_obj.baseline_path.discrete_path))
# _polyline_array miss 分支：np.array([(s.x, s.y) for s in path], dtype=np.float64)
```

**原理**：闭环每步重查同一地图同一批 lane（ego 移动 ~5m/步 vs 100m 查询半径，相邻步结果高度重叠）；`discrete_path` 是地图不变量（devkit cached_property 挂常驻 map 对象），StateSE2 列表 → `[N,2]` 数组是纯函数 → 同 key 返回逐位相同，缓存严格等价、无近似。旧路径每点一次 `Point2D.__init__` + 两次属性读 × mid/left/right 三条折线 × 每 lane，长尾步（lane 多/点多）线性放大——即 p90 550ms 的来源（火焰图 17.3% 样本）。内存代价 ~1KB/lane，只缓存查过的，几 MB 级。

**效果**：mapquery 中位 60→**14.6ms**，p90 **550→19.1**（长尾消灭）。剩余是 shapely 距离排序 + `get_proximal_map_objects`（ego 每步在动，需实时算，未动）。

### B. mapproc 坐标链路：绕过 devkit `to_vector()` 二次转换

**实现**：`map_process()` 打包循环：

```python
# 旧：ndarray 化的数据被拆回对象再转回数组
for element_coords in feature_coords.to_vector():          # [[coord.x, coord.y]...] 嵌套 listcomp
    list_feature_coords.append(np.array(element_coords, dtype=np.float64))
# 新：A 产出已是 [N,2] ndarray，直接消费
for element_coords in feature_coords.polylines:
    list_feature_coords.append(np.asarray(element_coords, dtype=np.float64))  # 零拷贝视图
```

**原理**：旧链路三遍逐点 Python 操作（Point2D 列表 → to_vector 逐点读 .x/.y 拆嵌套 list → np.array 从嵌套 list 重建）；`to_vector` 内层 listcomp 是全场最大单项（9.44% self），加我方 np.array 打包 3.65%。A 之后数据已是数组，这两遍纯属浪费。lane 系是唯一请求的 feature 层，无 Point2D 型 polygon 层混入（已核对 map_features 配置）。

**效果**：与 A 同次落地，无单独计量；体现在 mapproc/mapquery 段合并降幅。

### C. mapproc 插值：ragged 批量 + `_lane_polyline_process` 全向量化

**批量插值实现**（`_interpolate_points_batch`，替代 90×3 次逐条 `_interpolate_points`）：所有线 concat 成一根 `[Ntot,2]`，`line_of_point` 标记归属，跨线接缝段长置 0；`within = cum - cum[线起点]` 得每线从 0 起的累计弧长；查询点 `s = i×span + linspace(0,1,P)×len_i`，被查数组 `within + i×span`（span > 最长线长）——同一偏移系下全局单调 → **一次 searchsorted 定位所有线所有查询点**。退化线（<2 点、零长）走与单线版一致的 repeat 分支。

**批量插值原理**：开销本质是 numpy 固定调度成本（Python→C 切换 + 输出分配，~几 μs/算子，与数组大小无关）。旧形态 270 次小调用 × 每次 ~15 个算子 ≈ 4000 次调度，真正算数每条线只有几千 flop；批量后同样算子链只跑 3 次（mid/left/right 各一），调度次数 4000→54，数据量增大摊薄单次开销。等价性：每线内部数学同单线版（cumsum 沿线内、插值同式），唯一浮点差是查询点由 `linspace(0,1,P)×len` 而非 `linspace(0,len,P)` 生成——1ulp 级，对拍实测 worst 4.7e-13。

**polyline_process 实现与原理**：per-element 循环（2 次 norm 判 flip + diff + `np.insert` 补零行 + concat）→ 全量 `[E,P,2]` 向量化：`polyline_vector[:, :-1] = polylines[:, 1:] - polylines[:, :-1]`（末行保持 0，等价原 insert 语义）；flip 判断只需首点 `polylines[:, 0]`，一次算 `[E]` 布尔，fancy 索引赋回实现整行翻转。无效 element 输入全零（上游 `coords[~avails]=0` 保证）→ 全量算出全零，与原循环"跳过无效"等价；原 in-place flip 改的是 `array_output` 里 LEFT/RIGHT_BOUNDARY 的同一数组——已核对后续无读者（ROUTE_LANES 只用 lanes 系）。

**效果**：mapproc 中位 82→**9.6ms**（B+C 合并口径）。

### D. agents：三处循环向量化

**filter**（`_filter_agents_array`）：旧每 agent 对 target 全列比较 `(id == target_ids).max()` → O(T·N²)；新 `np.isin(frame_ids, target_ids)` 一次算全帧布尔 + 保序索引。

**pad**（`_pad_agent_states`）：旧逐行 dict 查 `id_row_mapping[int(...)]` 逐行赋值；新利用 token 是连续小整数（`_extract_agent_array` 从 0 顺序分配）建稠密数组 `row_of_id`，`current_state[row_of_id[frame_ids]] = frame` fancy scatter。等价性：帧内 token 唯一 → 目标行无碰撞 → 赋值顺序无关（数学必然）。

**extract**（`_extract_agent_array`）：旧 8 字段 × N agent 逐 numpy 标量 setitem（每次 ~1μs 调度）；新属性读进 Python tuple 列表 → 一次 `np.array` + 7 次列赋值。`agent.velocity.x` 等 `__slots__` 属性访问不可避免（devkit 对象层），只消灭 setitem 调度。

**效果**：filter/pad 火焰图占比 1.73%+0.41% → **0.49%+0.04%**；agents 段中位 24→18.8ms——低于预估 -14ms，偏差归因见下节（推测），剩余为 devkit 对象属性访问 floor。

## 事实：验证

- 对拍（`test_adapt_equiv_r4.py`，内嵌旧版参考实现）：批量插值 900 用例 worst 4.7e-13（linspace 生成方式 1ulp 级差，预期内）；lane_polyline 300 用例 bit-exact；filter+pad 300 用例 bit-exact；extract 200 用例 bit-exact
- 端到端 5 场景（one_continuous_log，DP_TORCHAIR=1）：**final_score 0.994266**（基线 0.9942，无回归）
- warm 中位（n=740）：

| 阶段 | 优化前 | 优化后 |
|---|---|---|
| total | 267.6 | **141.1** |
| adapt | 179.8 | **54.2** |
| ├ mapquery | 60（p90 550） | **14.6（p90 19.1）** |
| ├ mapproc | 82 | **9.6** |
| ├ agents | 24 | 18.8 |
| ├ route / to_tensor | 8~19 / 1.5 | 8.0 / 1.4 |
| fwd / norm / post | 70 / 7 / 9（不变） | 70.2 / 7.1 / 8.8 |

## 事实：agents 复查（第三张火焰图，5283 样本 / 105.7s 优化后代码）

`_filter_agents_array` 0.49% + `_pad_agent_states` 0.04%（优化前 1.73%+0.41%，压干净）。剩余 `sampled_tracked_objects_to_array_list` 2.88% 几乎全是 `_extract_agent_array` 的对象属性访问（11 帧 × ~50 agent × 7 个 `__slots__` 属性读取）——**devkit TrackedObject 数据结构决定的 floor**，不重写 devkit 对象层压不动。

新 top 热点：dpm_solver 标量数学 ~8%（10 步 eager 循环 Python 开销 ≈ 11ms/步）、shapely 距离/多边形判断 ~5%（mapquery 排序 + route_roadblock_correction）、DB 查询 3.6%（场景加载，planner 外）。

## 推测未严格证明（预估偏差归因）

agents 预估 -14ms 只兑现 -5ms。推测原因：第一张火焰图（优化前）的样本窗口包含 Ray worker 空闲等待（`_recv` 3.96% 样本），把 agents 段的"样本占比"稀释到 4.3%，据此折算的 wall 时间低估了属性访问的真实占比。未做对照实验严格验证（如剔除 idle 样本后重算占比）。

## 事实：四轮全链路对比

**性能（5 场景 one_continuous_log，单线程独占 310P，warm 中位）**：

| 轮次 | 改动 | total | 单轮收益 | 累计 |
|---|---|---|---|---|
| 起点 | 闭环跑通形态（jit 窗口还在） | 615 | — | — |
| R1 (0819) | jit 窗口关闭 | 552 | -63 | -10.2% |
| R2 (0819) | mapproc 插值向量化（单线版） | 417 | -135 | -32.2% |
| R3 (0819) | route 外提 + TorchAir 图模式 | 267.6 | -149 | -56.5% |
| R4 (0820) | 本轮：adapt 缓存+向量化 | **141.1** | -126.5 | **-77.1%** |

**子块明细**（有计时装的粒度，warm 中位）：

| 子块 | 起点 | R2 后 | R3 后 | R4 后 |
|---|---|---|---|---|
| mapproc | 221.6 | 73.2 | 82* | **9.6** |
| fwd | — | — | 175.9 → 71.1（图模式） | 70.2 |
| mapquery | — | — | 60（p90 550） | **14.6（p90 19.1）** |
| agents | — | — | 24 | 18.8 |
| route | — | — | 8~19 | 8.0 |
| adapt 合计 | — | — | 179.8 | **54.2** |
| norm + post | — | — | 17 | 15.9 |

\* R3 后 baseline log 实测 82.4（本轮 bench_r4_base），与 R2 轮 summary 的 73.2 有差——不同 Run 的场景/机器负载差异，未深究。起点行的子块计时 0819 拆解轮才加入，615 时代无此粒度。

**精度（final_score）**：

| 检查点 | 分数 | 说明 |
|---|---|---|
| 闭环首跑，50 场景全集 | 0.9170 | 优化前，唯一一次全量 |
| 5 场景基线 | 0.9942 | R1~R4 所有轮次的同口径锚点 |
| R2 后 | 0.9941 | ±0.0001 波动 |
| R3 后（eager 外提 / 图模式） | 0.9942 / 0.9941 | 图编译 3e-2 单步偏差端到端消化 |
| **R4 后** | **0.994266** | 无回归 |
| R4 对拍 | 900+300+300+200 用例 | 插值 worst 4.7e-13，其余 bit-exact |

## 可信推理（缓存与向量化的等价性）

- per-lane 折线缓存：`discrete_path` 由地图文件决定（devkit cached_property 挂在常驻 map 对象上），`[(s.x, s.y) for s in path]` 是纯函数 → 同 key 缓存返回逐位相同的结果（数学必然）
- `_lane_polyline_process` 无效 element 等价：输入端 coords/avails 由 `vector_set_coordinates_to_local_frame` 的 `coords[~avails] = 0.0` 保证全零 → 全量计算的全零输入产生全零输出（逐元素运算的数学性质）
- `_pad_agent_states` scatter 无序等价：帧内 track token 唯一（每 agent 每帧一行）→ fancy index 目标行无碰撞 → 赋值顺序不影响结果（数学必然）

## 工件

- 火焰图：`/data/syx_dp/pyspy_adapt_r4.json`（优化前）、`pyspy_adapt_r4b.json`（优化后）；聚合脚本 `agg_speedscope.py`（仓库根，用法 `python agg_speedscope.py <file.json> [filter]`）
- 计时日志：`/data/syx_dp/bench_r4_base.log`（基线）、`bench_r4_opt.log`（优化后）、`bench_r4_opt3.log`（profile 伴随跑）；归档副本 `dev_logs/run_prints/2026-08-20_adapt-optimization/`
- 对拍：`test_adapt_equiv_r4.py`（仓库根）
