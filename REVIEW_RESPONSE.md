# 对《双网络球形爆炸 PINN：代码审查与修复清单》的逐项回应

审查日期 2026-09-21，本回应日期 2026-09-22。审查清单里的 11 项**全部**做了改动，
每项都附一条直接对应那个错误的回归测试（`tests/test_review_regressions.py`）。

本机测试：`PYTHONPATH=src pytest -q` → **92 passed**（修复前 52 passed / 1 skipped）。
测试不计入物理验收，见文末第 4 节。

---

## 1. P1 初始流场把拉格朗日初始坐标误当成当前物理坐标 —— 已修

**审查结论成立。** 确认过：`solve_reference_field` 用 `solver.mesh.centres`（**初始**
单元中心）当空间坐标，而 `ReferenceField.sample` 拿当前物理半径去上面插值。

改动：

* `physics/precursor.py`
  * `ReferenceField.radii` → `ReferenceField.cell_radii`，形状 `(n_t, n_cells)`，
    每个快照带自己那一帧的**当前**单元中心；
  * 查询重写为**分材料**插值：先按该帧的接触面判定查询点属于哪种材料，再只在
    该材料的单元块里插值（材料已排序，产物是前缀、空气是后缀），接触面两侧各取
    单侧状态；
  * 时间插值：接触面在两帧之间扫过查询点时，两帧的状态属于不同的场，**不做**
    线性混合，退回到时间上更近的那一帧，并把该比例记进 `query_diagnostics`；
  * `__post_init__` 增加形状、材料单调、单元中心单调的校验；
  * 新增 `final_cell_radii`、`n_product_cells`。
* `cli/_common.py`
  * `REFERENCE_FIELD_VERSION = 2` 进入缓存签名；读到旧格式缓存（只有单个
    `radii` 数组）时**拒绝**并说明原因，不静默重建也不静默复用。
* `cli/reference_field.py`、`tests/test_similitude_data_and_run.py`
  * 尺度映射核查改用 `final_cell_radii`（两个算例各求各的拉格朗日网格）。

回归证据（全部实际执行）：

* `test_reference_field_queries_the_current_lagrangian_position`：在每一帧的
  当前单元中心查询，密度/比内能相对误差 **≤1.5e-16**、速度绝对误差 **≤1.6e-27**。
* `test_reference_field_coordinates_actually_move`：最后一帧相对第一帧的最大单元
  位移 > `1e-3 R_0`（前提检查，确保上一条不是恒等式）。
* `test_reference_field_never_interpolates_across_the_contact`：紧贴接触面内/外的
  查询分别精确复现最后一个产物单元与第一个空气单元的状态（相对 1e-9）。
* `test_reference_field_does_not_blend_materials_in_time`：被接触面扫过的固定半径上，
  两帧中点的查询结果**等于**其中一帧，不是加权平均，且
  `last_query_crossing_material_fraction` 记录在案。
* 交付给 PINN 的查询接口重做了初值质量/能量/界面检查：`reference_field` 命令
  退出码 0，质量相对漂移 **0.000e+00**，外边界最大相对压力偏离 **1.436e-16**。

旧缓存已失效：`outputs/run`、`outputs/audit`、`outputs/reference_smoke` 已删除重建。

## 2. P1 80/100 mm 的 PDE 时间导数用了 50 mm 的尺度 —— 已修

**审查结论成立，倍率也复现了。** `CoupledModel.chain_factors` 把 `self.scales`
（来自第一个训练半径）传给链式法则。

改动：

* `models/coupled_model.py`：`chain_factors(medium, fronts, radius, *, scales)`，
  `scales` 改为**必填关键字参数**，没有默认值。一个可以省略的参数迟早会被省略。
* `losses/terms.py`：`medium_pde_residual` 传入 `batch.scales`。

回归证据：

* `test_chain_time_factor_follows_the_case_radius`：三个半径的时间项系数与
  `U_*/(R_0 hat T_*)` 相对误差 1e-12，且
  `d(0.08)/d(0.05) = 0.05/0.08`、`d(0.10)/d(0.05) = 0.05/0.10`。
  （旧实现在三个半径上都给 50 mm 的值，三个比值都是 1。）
* `test_normalised_pde_residual_is_scale_invariant_across_radii`：同一网络、同一
  `(s, xi)`，三个半径的归一化质量/动量/能量残差两两相对误差 **≤1e-9**。
  这是审查给的验收判据。
* `test_chain_rule_in_physical_coordinates_agrees_at_every_radius`：每个半径上
  显式链式法则与物理坐标自动求导一致（rel 1e-9）。单半径测试确实不够——偏差
  正比于 `R_0`，在 50 mm 上恰好为零。

## 3. P1 有效帧采样返回了列表位置，未映射回真实帧号 —— 已修

**审查结论成立。** `pick_frames` 返回 `0..len(frames)-1` 内的位置，被当成原始帧号用。

改动：`sampling/sampler.py` 的 `pick_frames` 返回 `frames[positions]`；抽样帧号写入
`batch.diagnostics["sampled_frames"]`；在 `_data_points` 里加一条硬检查——被抽中的
帧的物理时间不得早于该半径的启动时刻。

回归证据：

* `test_sampler_returns_real_frame_numbers`：夹具刻意把前 4 帧做成无效（用"位置"
  的实现必然落到它们身上），断言所有被抽中的帧号都属于有效集合，且时间不早于
  启动时刻。
* `test_front_supervision_uses_the_same_frame_mapping`：前沿标签的时间必须落在有效
  帧的真实时间上。

## 4. P1 Torch 的 JWL 声速公式少了一个密度因子 —— 已修

**审查结论成立。** Torch 版写的是 `dp_drho + dp_de * p / rho`，少了 `p/rho²` 里的
一个 `rho`。NumPy 版一直是对的。

改动：

* `physics/eos.py`：新增 `jwl_sound_speed_squared` / `..._numpy`（**未裁剪**），
  `jwl_sound_speed{,_numpy}` 改为它的 `sqrt(clamp_min(0))`；公式统一为
  `c² = (∂p/∂ρ)|_e + (p/ρ²)(∂p/∂e)|_ρ`。
* `models/product_net.py`：新增 `sound_speed_squared(fields)`。
* `losses/terms.py`：物理解条件改用未裁剪的 `c²`；新增
  `adm_product_invalid_sound_speed_rate` / `adm_air_invalid_sound_speed_rate`
  监控量（先把负值裁成零再判断等于没判断）。

回归证据（数值与审查给的对照值一致）：

* `test_jwl_sound_speed_matches_between_torch_and_numpy`：产物初值 `rho=1630`、
  `e=E0/rho0` 下，Torch 与 NumPy 都给出 **4059.1204 m/s**（相对 1e-12）。
  （修复前 Torch 给约 49810 m/s。）
* `test_sound_speed_squared_is_not_clamped`：构造一个 `c² < 0` 的状态，未裁剪的
  函数返回负值，裁剪过的返回 0——用来判断有效性的必须是前者。

## 5. P1 预测和对照的时间零点处理不一致 —— 已修

**审查结论成立，三个问题都复现了。**

改动：

* `cli/predict.py`：`predict_grid` 的入参改为**物理时间** `physical_times`，
  `t_phys → t_PINN → s` 的换算在函数内**只做一次**；一次查询早于
  `epsilon(R_0)` 直接报错，不静默钳位。输出同时给 `physical_time` 与 `pinn_time`
  并附带 `time_origin` 说明。`--time-end` 是物理秒，默认值由训练覆盖反推。
* `cli/compare.py`：`compare_trajectories` 直接使用 `prediction["physical_time"]`，
  **不再**加第二次 `epsilon`；场量对照的时间网格同理；`_time_bracket` 先判定
  覆盖再算权重，域外返回 `nan`，权重的 `[0,1]` 由钳位保证，不再出现权重 >1 的
  静默外推。

回归证据：

* `test_compare_does_not_shift_the_prediction_a_second_time`：同一组轨迹自比较，
  `contact_relative_rms` 与 `shock_relative_rms` 都是 **0.0（abs 1e-12）**。
  （修复前该函数报出 0.06/0.08。）
* `test_predict_grid_sets_both_clocks_consistently`：
  `physical_time - pinn_time == epsilon(R_0)` 逐点成立，`pinn_time[0] == 0`。
* `test_predict_grid_refuses_queries_before_the_time_origin`：早于启动时刻的查询
  抛 `SetupError`。
* `test_field_lookup_never_extrapolates_in_time`：域外查询返回 `nan`；权重恒在
  `[0,1]`。

## 6. P2 新半径的外推判断没有按统一尺度化时间进行 —— 已修

**审查结论成立。** 原实现取训练半径的**最大有量纲 PINN 秒数**去比目标半径的秒数。

改动：

* `physics/similitude.py`：新增 `scaled_time_coverage(...)`，由冻结的 `hat T_*` 与
  各训练算例的最后有效 `t_PINN` 给出**无量纲**上界 `s_max` 及其 provenance；
  新增 `physical_time_from_scaled(s, R_0)` 供"默认预测到覆盖末端"使用。
* `cli/predict.py`：外推标记改为 `s_query > s_max`；默认 `--time-end` 由 `s_max`
  经目标半径还原成物理秒，并在 JSON 里写明来源。

回归证据：

* `test_scaled_time_coverage_is_the_same_for_every_radius`：同一 `s` 在 0.05/0.08/0.10
  三个半径上，`0.99 s_max` 一律判为覆盖内、`1.01 s_max` 一律判为外推；并显式
  展示旧的秒数判据会把 100 mm 的覆盖内查询误判。
* 实机：`predict --radius 0.08` 输出 `s_max = 1`（治理半径给出）且"所有查询时间
  都在覆盖内"。

## 7. P1 数据审查报告没有成为训练入口的强制条件 —— 已修

**审查结论成立。** 这正是最该有闸门的地方：审查报告说"不兼容"，训练入口只看
文件是否存在，两者可以同时成立。

改动：

* 新增 `data/compatibility.py`：状态 `compatible / incompatible / unknown /
  synthetic`，报告格式 `spherical_blast_pinn.data_compatibility` v1，
  `enforce_training_gate(...)`。
* `cli/audit_dyna.py`：除人读的表以外，写 `data_compatibility.json`。五项判据：
  爆轰模型、时间零点、单位制、JWL 系数、材料分区。报告里没有的算例是 `unknown`。
* `cli/_common.py`：`resolve_case_compatibility`，报告位置 `data.compatibility_report`
  或默认 `<output_dir>/dyna_audit/`；算例自身记录的分区来源会覆盖报告里更乐观的
  快照。
* `training/trainer.py`：`Trainer.__init__` 调用闸门；`cli/train.py` 不通过时以
  退出码 3 结束。越过闸门（`data.allow_unverified_cases: true`）后，检查点与运行
  清单被打上 `data_compatibility: unverified`。
* `configs/base.yaml`（默认关闭）、`configs/experiments/smoke.yaml`（显式打开并说明理由）。

回归证据（实机）：

```
$ python -m spherical_blast_pinn.cli.train --config configs/experiments/full.yaml --max-steps 2
** 训练未开始：数据兼容性闸门没有放行 **     (exit 3)
  tnt_50mm : detonation_model 不一致; time_origin 不一致; material_partition 不可靠
  tnt_80mm : detonation_model 不一致; time_origin 缺证据; material_partition 不可靠
  tnt_100mm: 报告里没有它的条目
```

单元测试：`test_gate_allows_compatible_cases`、
`test_gate_refuses_incompatible_or_unknown_cases`（含 `unknown`）、
`test_gate_treats_a_missing_entry_as_unknown`、`test_gate_records_an_explicit_override`、
`test_synthetic_mode_refuses_real_cases`、`test_report_round_trips_through_disk`、
`test_heuristic_material_partition_is_not_reported_as_verified`。

**仍需你决定的那一件事没有变**：方案 A 与有限速度爆轰数据的冲突。把监督窗口
平移到燃尽之后**不足以**宣称已匹配——那只是改了标签的时间区间，没有证明初值场
一致。现在这个决定有了代码强制：不解决，正式训练就进不去。

## 8. P2 独立检查和选模流程尚未按提示词落地 —— 已修

改动：

* 新增 `evaluation/checks.py`
  * `build_check_set`：**一次采样、全程复用**、独立种子（`seed + 7919`）、取
    **开发帧**、不进梯度；定义（种子、划分、点数、帧号、选模偏序）随检查点与
    运行清单一同保存。
  * `run_checks`：逐半径 × 逐介质 × 逐方程报告归一化残差的 **RMS / P95 / 最大值
    和最大值位置**；接触面（压力、两侧速度）+ 完整三条 RH；初值；非物理比例
    （负压、未裁剪 `c²<0`、非有限、`R_c/R_0 < 1` 的比例与最小值）。
  * `conservation_diagnostics`：产物质量最大相对漂移 + **固定控制体**能量平衡，
    含外边界通量积分；归一化用物理零点的初始产物能量
    `(4π/3) R_0³ E_0`，与 PINN 启动时刻的能量分开。只诊断，不进梯度。
  * `SELECTION_CRITERIA`：固定的字典序——先非物理计数，再物理/界面残差超出 §16
    容差的量，最后才是开发集误差。**不按加权训练总损失选模型。**
* `data/extracted_case.py`：`frame_split`，按**连续帧组**留出约 10%（§11）。
* `training/trainer.py`：`verify()` 每 `check_every` 步跑一次；`best.pt` 按上述偏序
  保存；`checks.json` 落盘。

回归证据：`test_check_set_is_deterministic_and_separate_from_the_sampler`（同种子
逐位相同、不同种子不同、帧集合与训练采样器不同）、
`test_frame_split_holds_out_whole_frame_groups`（留出的是连续段）。
实机：`outputs/_archive_scheme_a/smoke/checks.json` 含逐半径逐方程的完整指标与
守恒诊断。该文件是**方案 A** 的历史运行，已原样移入归档（见
`outputs/_archive_scheme_a/README.md`）；它不代表当前方案 B 的验收结果。

## 9. P2 `Rc>=R0` 是新增约束，关闭开关也不能完整退回 —— 已修

**审查结论成立，我接受。** 我把一个初值/优化初期的问题升级成了全时域物理假设，
并且即使关掉 FrontNet 的开关，PDE 路径里的硬中止仍然阻断运行。

改动：

* `models/front_net.py`：默认恢复为提示词 §7 的
  `hat_Rc = a softplus(z_c)`（`a = hat_Rc^IC > 0`，初始位置由初值损失约束）。
  额外约束改为**可选的早时段锚定**：
  `hat_Rc = a softplus(z_c) + w(s)`，`w(s) = ½(1+cos π min(s/s_anchor,1))`，
  只在 `0 ≤ s ≤ s_anchor` 上非零，两端一阶导为零；类文档里写明适用时段与依据。
* `models/coupled_model.py`：`resolve_front_anchor(config)`，从
  `networks.front.contact_anchor` 读 `enabled` 与 `until_s`；`build_model` 正常传入。
* `losses/terms.py`：**删除** `R_c < R_0` 的全时域硬失败，改为
  `front_floor_monitor` 监控量（`contact_below_charge_radius_rate` 与
  `min_contact_over_charge_radius`）。硬失败只保留"配置点半径到达轴心"。
* `cli/predict.py`：从检查点恢复时读回 `anchor_until_s`。
* 配置默认 `enabled: false`。

回归证据：`test_front_default_construction_is_the_literal_one`、
`test_anchor_is_opt_in_and_has_a_window`（窗口内带偏移、窗口外逐位等于字面构造）、
`test_anchor_is_configurable_through_build_model`、
`test_pde_path_does_not_abort_when_the_contact_dips_below_the_charge_radius`
（把接触面刻意压到 `R_0` 以内，PDE 必须照常求值）。

**我改变了主张**：README 现在说 `R_c >= R_0` 不是全时域物理假设；锚定关闭时
`R_c` 允许短暂低于 `R_0`，那是一个会被**报告**的瞬态。是否重新打开、以及窗口取
多少，等你定——它现在是一个显式配置项，不是一个默认行为。

## 10. P2 损失分组和断点恢复仍有偏差 —— 已修

**A. 分组。** `losses/terms.py` 的 `data_loss` 改为**三组等权**：产物 / 空气 /
前沿，前沿组**内部**再对 `Rc`、`Rs` 取平均。审查给的判断正确——原来的四组等权
让前沿占 1/2 而不是 1/3，没有任何理由让两条标量曲线和两个完整场平分秋色。

回归：`test_data_loss_has_three_equal_groups`（`data_n_groups == 3`，且总损失等于
三组的均值、前沿组等于 `Rc`/`Rs` 的均值）。

**B. 恢复状态。** `sampling/sampler.py` 增加 `rng_state()` / `set_rng_state()`；
`GroupedSampler` 转发；`training/trainer.py` 把 `sampler_rng` 写进检查点，恢复时
读回；检查点版本升到 3，缺少该项的检查点**拒绝**恢复而不是安静地换一条轨迹。

回归：`test_sampler_rng_state_round_trips`（JSON 往返后"恢复"给出与连续轨迹逐位
相同的采样）、`test_resuming_reproduces_a_continuous_run`（Trainer 级：连续 8 步
与"4 步 + 恢复 + 4 步"的 `state_dict` **逐元素相同**、检查指标相对 1e-12 相同）、
`test_a_checkpoint_without_sampler_rng_is_refused`。
实机：12 步连续运行与"4 步 + 恢复"在 step 8 / step 12 的指标与最终损失完全一致。

## 11. P2 场量验证会掩盖区域错误，指标名称与实际计算不符 —— 已修

`cli/compare.py` 的场量对照重写：

* **先按材料身份分区再算误差**：参考标签 0/1/2 与预测区域逐点比较，材料不一致
  的单元只进 `region_mismatch_cells`（分介质计数），**不进任何介质的误差**。
* **标签 2（未受扰动）不再当成空气**：它被排除在场误差之外并单独计数——
  那是几千个常数单元，拿它评分对所有模型一样地"好看"。
* **分介质报告**：`product_*` / `air_*` 各自给相对 L2、点数、最大绝对误差，
  某一介质没有共同区域时明确写"不可评价"，不并入另一侧。
* **固定物理尺度**：分母取检查点里的 `rho_{*,k}`、`U_*`、`U_*²`，不再用参考数据的
  局部最大值。报告里写出 `normalisation_scales` 与指标的完整数学定义。
* **禁止跨间断插值**：`_field_at` 要求参与插值的两个半径网格点与两个时间切片
  落在同一区域标签里，否则返回 `nan`。
* **删掉** `_reference_scale`（局部最大值冒充固定尺度）。
* 新增按 `r/R_0` 的主激波**到达时间**对照（线性插值，两条曲线各自首次到达），
  两侧任一未到达就报 `—`，不做时间对齐寻优。

回归：`test_compare_fields_self_consistency_is_zero`（同一份数据自比较，六个分介质
指标全为 **0.0**）、`test_compare_fields_does_not_count_ambient_as_air`、
`test_compare_fields_keeps_mismatched_material_out_of_the_error`、
`test_compare_field_scales_are_frozen_not_local`（把参考密度放大 3 倍，误差必须
随之变大而不是分母同步变大）、
`test_compare_field_lookup_does_not_interpolate_across_a_discontinuity`。

---

# 交付说明

## 1. 改动文件

| 文件 | 对应项 |
|---|---|
| `physics/eos.py` | 4 |
| `physics/precursor.py` | 1 |
| `physics/similitude.py` | 5、6 |
| `models/coupled_model.py` | 2、9 |
| `models/front_net.py` | 9 |
| `models/product_net.py` | 4 |
| `losses/terms.py` | 2、4、9、10A |
| `sampling/sampler.py` | 3、10B、§11 |
| `data/extracted_case.py` | §11 帧组分划 |
| `data/compatibility.py` | 7（新增） |
| `evaluation/checks.py` | 8（新增） |
| `training/trainer.py` | 7、8、10B |
| `cli/_common.py` | 1、7 |
| `cli/predict.py` | 5、6 |
| `cli/compare.py` | 5、11 |
| `cli/train.py` | 7 |
| `cli/audit_dyna.py` | 7 |
| `cli/reference_field.py` | 1 |
| `configs/base.yaml`、`configs/experiments/smoke.yaml` | 7、9、8 |
| `tests/test_review_regressions.py` | 全部（新增，39 条） |
| `tests/test_similitude_data_and_run.py` | 1（API 变更） |
| `README.md`、`AGENTS.md` | 文档同步 |

包内 Python：16411 行；测试：2209 行。

## 2. 回归结果

* `pytest -q` → **92 passed**，约 11 秒。其中 39 条是本次新增、逐项对应上述 11 个
  错误。
* 七个 CLI 命令实机跑通：`audit_dyna`（并写出兼容性报告）、`reference_field`、
  `audit_data`、`validate_eos`、`train`（含闸门拒绝路径与冒烟路径）、`predict`、
  `compare`。
* 三半径联合小规模更新：冒烟配置 40 步；`full` 配置的采样规模下 2 步（用于确认
  满规模下跑得动）。**两者都不是训练完成。**

## 3. 仍然缺的数据 / 未完成的检查

* **方案 A 与数据冲突尚未由你裁定**（第 7 节）。这是唯一阻塞正式训练的项。
* 80/100 mm 的原始 d3plot 不在本机（`G:` 盘不存在），材料体积分数无法重新提取；
  50 mm 的 d3plot 在 `C:/Users/work/Documents/GitHub/2/sim_data/`，可以重做。
* `D:/500mmtnt/extracted` 不存在，盲测半径没有剖面可对照。
* 100 mm 的 47 µs 输出间隔不足以看见燃尽，时间零点判据在该算例上给不出结论。
* 二次激波（域内明显间断）没有单独处理，不声称不存在——这是本版强形式方案的
  已知限制。
* 正式基线训练（20000 步）**未执行**，按用户指示等待发起。

## 4. 不要把这些当成物理验收

单元测试通过、尺度映射精确到 `1e-9`、冒烟闭环跑通，**都不是**物理验收通过。
目前唯一有物理含义的结论来自第 8 节之前各项的**数据审查**：三份训练数据都不是
方案 A 的算例。§16 的各项精度验收（PDE 残差 RMS ≤1e-2、初值相对 L2 ≤2%、
新半径轨迹 RMS ≤2% 等）**一条都没有评价过**，因为还没有在合规数据上训练出模型。
