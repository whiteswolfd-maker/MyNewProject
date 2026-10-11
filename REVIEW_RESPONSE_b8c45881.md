# 复核回应：`b8c45881820bf165f743203ee4a20454b682b562`（2026-10-11）

## 0. 版本、环境与执行规则

| 项 | 值 |
|---|---|
| 复核基线 | `b8c45881820bf165f743203ee4a20454b682b562`（复核方给出的 GitHub SHA） |
| 本机 HEAD | **无本地 git 仓库**（`git rev-parse` 报 "not a git repository"）。交付物是当前工作区，**无法给出 SHA**；下面用文件哈希标识 |
| 复核方环境 | Python 3.12、PyTorch 2.14 CPU，**364 passed / 6 skipped** |
| 本机环境 | Python **3.10.8**、torch **2.5.1+cu121**、numpy **1.26.4**、`torch.cuda.is_available() = True`（但本轮**没有**用 GPU 跑任何东西） |
| 测试命令 | `PYTHONPATH=src python -m pytest -q` |
| 测试结果 | **457 passed，0 skipped，0 failed**（本轮开始时 373 passed） |

**两条必须先说明的事**：

1. **本机版本与复核基线不是同一个提交。** 复核方在 `b8c45881` 上跑出 364 passed /
   6 skipped，本机跑出 457 passed / 0 skipped。本机没有 git，**无法比对两者的差异**；
   本轮按提示词的要求，在**本机实际版本**上逐条复验，已经修好的项只补证据。
   凡是"本机复验为不成立"的，下面都给出实测输出，不推测复核看到的是哪一份。
2. 复核方说 6 项 skipped 依赖真实数据。本机 **0 skipped**——本机的真实数据在
   `C:\Users\work\Documents\GitHub\2\`（50/100/80 mm），能读到，所以那些测试没有跳过。
   这是环境差异，不是"本机的测试更强"。

交付文件的哈希（`sha256` 前 16 位）：

| 文件 | 字节 | `sha256[:16]` |
|---|---:|---|
| `src/spherical_blast_pinn/cli/compare.py` | 55140 | `33925dc8283aa34e` |
| `src/spherical_blast_pinn/evaluation/waveform.py` | 13030 | `4b9d019821fc1704` |
| `src/spherical_blast_pinn/training/trainer.py` | 102005 | `55f19646bcb185e8` |
| `src/spherical_blast_pinn/sampling/sampler.py` | 42902 | `c34b0792ec9a50e6` |
| `src/spherical_blast_pinn/data/extracted_case.py` | 83153 | `1a9d643eba407504` |
| `src/spherical_blast_pinn/data/contract.py` | 19942 | `a2450deaf87334c4` |
| `scripts/bounded_snake_check.py` | 30391 | `23ea4a8a6ed58afe` |
| `tests/test_compare_cli.py` | 19411 | `977933acc27d1a12` |
| `tests/test_waveform_impulse.py` | 7808 | `4364a506f943b3fe` |
| `tests/test_refine_and_checkpoint.py` | 30038 | `6fed38a03296584e` |
| `tests/test_profile_layout.py` | 12032 | `3a94a3a4c576cc6d` |

复算命令（复现用，不需要第 2 列）：

```bash
PYTHONPATH=src python -c "
import hashlib, pathlib
for p in '''src/spherical_blast_pinn/cli/compare.py
src/spherical_blast_pinn/evaluation/waveform.py
src/spherical_blast_pinn/training/trainer.py
src/spherical_blast_pinn/sampling/sampler.py
src/spherical_blast_pinn/data/extracted_case.py
src/spherical_blast_pinn/data/contract.py
scripts/bounded_snake_check.py
tests/test_compare_cli.py
tests/test_waveform_impulse.py
tests/test_refine_and_checkpoint.py
tests/test_profile_layout.py'''.split():
    b = pathlib.Path(p).read_bytes()
    print(hashlib.sha256(b).hexdigest()[:16], len(b), p)"
```

---

## 10.3 逐项对应表

| 编号 | 基线反例与实际复现结果 | 根因 | 修改文件/函数 | 回归测试名称 | 实际执行结果 | 尚未覆盖的边界 |
|---|---|---|---|---|---|---|
| §1 | **复现**：`compare` 在任何输入下抛 `AttributeError: 'dict' object has no attribute 'files'`（`data` 已是 dict，又对它取 `.files`） | 命令行这一层把"字典"和"NpzFile"两种接口混用；已有单测只覆盖 `compare_fields()`，正好绕开 CLI | `cli/compare.py`：新增 `load_prediction()` / `validate_prediction()` / `PredictionError`；`main` 改为单一入口 | `test_the_compare_command_runs_end_to_end_and_writes_a_report`、`test_the_report_says_which_reasons_dropped_points`、`test_a_missing_required_field_is_refused_with_its_name`(×4)、`test_a_field_with_the_wrong_frame_count_is_refused`、`test_a_descending_radial_axis_is_refused`、`test_a_single_frame_prediction_is_refused_not_extrapolated`、`test_a_time_axis_that_does_not_match_the_start_time_is_refused`、`test_a_missing_prediction_file_is_refused`、`test_a_charge_radius_mismatch_stops_the_comparison`、`test_a_missing_ambient_pressure_is_refused_not_defaulted`、`test_an_ambient_pressure_of_101325_instead_of_101320_is_refused` | passed ×12 | 单帧预测**明确拒绝**（不提供定义）；没有为它实现"精确查询" |
| §2 | **复现**：Adam 一次正常更新后注入 NaN 梯度再 `refine(1)`，模型参数出现 NaN，写出的 `last.pt` 也含 NaN | `_update()` 的有限性守卫没有覆盖 L-BFGS；`closure` 里既不查损失也不查梯度，线搜索改动参数后也没有还原点 | `training/trainer.py:Trainer.refine`：闭包内反传前查损失、更新前查梯度；`NonFiniteUpdate` 增加 `detail` 属性；失败时 `_restore_state` 全量回滚 | `test_a_non_finite_gradient_blocks_refinement_and_leaves_the_model_alone`、`test_a_non_finite_loss_blocks_refinement_before_backward`、`test_a_failure_after_the_line_search_moved_the_parameters_is_rolled_back`、`test_a_failed_refinement_writes_a_separate_diagnostics_file`、`test_a_successful_refinement_records_who_it_really_changed` | passed ×5 | 未覆盖 CUDA 上的 L-BFGS（本机跑的是 CPU 路径） |
| §3 | **复现**：完整日程后精修 → 返回值与清单 `diagnostic`，但 `last.pt.run_status` 仍是 `completed`，清单 `metadata.diagnostic` 是 `None`，检查点不含精修历史，新实例 `refine_history == []` | `refine()` 先调用 `run()` 落盘，**之后**才改返回对象的字段，只重写了清单 | `Trainer.run` 新增 `refinement=` 参数，在**写任何产物之前**决定终态；`TrainingResult.as_dict` 顶层暴露 `diagnostic/refined/refinement_count/check_state`；新增 `Trainer.resume_refinement()`（明确拒绝） | `test_the_refinement_identity_is_the_same_in_every_artifact`、`test_a_new_instance_loading_a_refined_checkpoint_knows_it_was_refined`、`test_a_truncated_then_refined_run_is_diagnostic`、`test_an_explicit_diagnostic_run_stays_diagnostic_through_save_and_reload`、`test_resuming_an_interrupted_refinement_is_refused_explicitly` | passed ×5 | "继续中断的 L-BFGS"**不支持**（明确拒绝，见 §3.5）；未实现 L-BFGS 状态持久化 |
| §4 | **复现**：精修前 1 条检查、精修后仍是 1 条，最后一条检查的参数指纹**不是**当前指纹 | 收尾条件写的是 `self.step > start_step`，而精修不改 `step`；且 `refine` 在进 `run()` **之前**就改完了参数，进函数时的指纹已经是精修后的 | `Trainer.run`：改用"参数是否变动"（进入时指纹比较 + `refinement is not None`）；`verify()` 记录 `scope` 与 `check_set_identity`；`_check_stale()` 增加 scope/检查集两项 | `test_refinement_triggers_a_full_recheck_on_the_new_parameters`、`test_a_zero_budget_call_does_not_recheck`、`test_checks_disabled_is_reported_as_unchecked_not_as_passed`、`test_a_windowed_check_does_not_substitute_for_a_full_one`、`test_a_check_from_another_check_set_is_stale`、`test_best_may_keep_an_earlier_model_and_the_new_one_still_competes` | passed ×6 | best 的"参与比较"是**逐次核对指纹**证明的，没有构造"新模型确实更优"的物理场景 |
| §5 | **复现**：`*_eligible` 置 None / 删除 / 全 False / 未知 `level` 四种改动，基线 `ready=True` 的对象**全部仍 `ready=True`** | 校验只在数组**存在时**才查（`continue` 跳过缺失），档位字符串从不核对；派生掩码与资格证据是两份从不相交的记录 | `data/extracted_case.py`：新增封闭枚举 `ELIGIBILITY_LEVELS`、`ELIGIBILITY_ORDER`、`geometric_front_validity()`、`_eligibility_side_evidence()`、`validate_front_eligibility()`；`validate_case_object()` 调用它。读取端的 `LEVEL_*` 改为引用同一份枚举（以前是本地再定义一份） | `test_a_self_contradictory_eligibility_record_is_refused`(×7 参数)、`test_an_all_false_eligibility_array_cannot_keep_true_front_labels`、`test_the_derived_labels_can_be_recomputed_from_the_evidence`、`test_an_explicit_false_never_becomes_a_front_label`、`test_the_two_sides_keep_their_own_evidence_levels`、`test_an_order_flag_that_contradicts_the_radii_is_refused`、`test_the_overall_level_is_derived_from_the_sides_not_asserted`、`test_the_contract_reports_where_the_problem_is` | passed ×14 | 契约只判"记录自洽"；"资格数组本身算得对不对"不在它的能力范围内 |
| §6 | **复现四类**：① 最后一次 Adam 损失没进检查点，加载后零更新得到 `final_loss=nan, final_loss_step=0`；② 同名阶段换预算/学习率/PDE 权重但窗口相同 → 照常加载；③ 删掉全局 `rng` → 照常加载；④ 提交中途失败回滚后 `_window_history` 从 `[1.0, 0.25]` 变成 `[1.0, 0.25, 1.0]` | 检查点缺字段（损失、日程签名、检查集、数据身份）；`load` 只比窗口；`_restore_rng(None)` 静默返回；`restore_time_window` 不还原历史 | `trainer.py`：`CHECKPOINT_VERSION` 4→5；`_save` 写入 `last_update_loss`/`last_update_loss_step`/`refinement`/`schedule`/`check_set`/`data_identity`；`load` 逐项核对并要求 `rng` 完整；`_restore_state` 逐步兜错、先还原采样器、失败即标记 `_unusable`；新增 `_schedule_signature`/`_signatures_differ`/`_data_identity`/`_batch_fingerprint`/`_assert_usable`。`sampling/sampler.py`：新增 `window_history` 属性，`restore_time_window(..., history=)` | `test_the_last_measured_loss_survives_a_checkpoint_round_trip`、`test_a_checkpoint_records_model_data_and_check_set_identity`、`test_a_different_schedule_is_refused_even_when_the_window_matches`(×3)、`test_a_checkpoint_without_the_global_rng_record_is_refused`、`test_a_partial_global_rng_record_is_refused`、`test_a_checkpoint_without_the_last_update_loss_is_refused`、`test_an_old_checkpoint_version_is_refused_with_a_migration_note`、`test_a_rollback_does_not_invent_a_window_history_event`、`test_a_failed_rollback_marks_the_instance_unusable`、`test_resuming_matches_a_continuous_run_across_a_stage_boundary`、`test_a_blocked_stage_can_be_resumed_and_re_evaluated`、`test_moving_the_output_directory_does_not_change_data_identity`、`test_changing_the_data_behind_the_same_path_is_refused` | passed ×16 | 版本 4 及更早的检查点**一律拒绝**（无迁移路径，如实写明）；未实现权重导入模式 |
| §7 | **复现三项**：① 同一半径前后帧分属空气/产物，中间时刻插出 **51**；② `_region_at` 取时间区间**左端**那一帧的前沿；③ 域外半径仍被贴上产物/空气/环境标签；④ 精确节点自身有限、零权重邻点 NaN → 查询得到 NaN | 场量插值与测点压力是**三套**实现（`_field_at` / `_field_corner_average` / `_region_at`），规则各不相同；`nan * 0` 让零权重邻点污染结果 | `cli/compare.py`：删掉 `_field_at`/`_field_corner_average`，新增统一核心 `sample_field()`（返回**逐点原因码**）、`sample_field_values()`、`_region_from_fronts()`、`_fronts_at()`；`_pressure_at` 改为走同一核心；`_region_at` 按时**插值后**的前沿判定并对域外返回 `-1`；`compare_fields` 按原因码做互斥计数并新增 `sample_reason_counts`/`prediction_time_coverage_s`/`reference_frames_outside_prediction_coverage` | `test_a_smooth_same_material_field_is_interpolated_exactly`、`test_the_last_frame_uses_the_last_frame_s_fronts`、`test_a_radius_outside_the_prediction_grid_is_unavailable_not_labelled`、`test_a_query_time_outside_the_prediction_coverage_says_so`、`test_an_unusable_front_record_makes_the_region_unavailable`、`test_the_reason_counts_and_the_mutually_exclusive_classes_close`、`test_a_field_with_no_comparable_point_is_unavailable_not_zero`、`test_the_coverage_of_the_prediction_is_reported_separately`、`test_the_midpoint_between_product_and_air_is_not_fabricated`、`test_an_exact_grid_node_is_not_poisoned_by_a_zero_weight_nan_neighbour`、`test_a_front_order_violation_makes_the_region_unavailable_not_a_label` | passed ×11 | 前沿按时间**线性**插值；这是明确写下的选择，未做更高阶或基于 RH 的插值 |
| §8 | **复现**：`[0,1,2,3] / [102,102,98,98] / ambient 100` 修前返回 **3.0**，解析值 **2.5** | 首个负压点被留在积分节点里、过零时刻被**追加到数组末尾**（节点倒序 `[0,1,2,1.5]`，`trapezoid` 不排序也不报错）；且 `times >= arrival` 丢掉到达时刻到首个采样点之间那一段 | `evaluation/waveform.py:gauge_metrics`：节点改为"过零前的全部非负采样点 + 真正的过零点"；补入到达插值点；新增 `arrival_observed`；非有限压力与非有限时间轴明确拒绝 | `test_the_review_counterexample_is_2_point_5`、`test_the_integration_nodes_are_strictly_increasing_on_the_covered_interval`、`test_a_triangle_has_half_base_times_height`、`test_a_trapezoid_has_the_analytic_area`、`test_a_zero_crossing_exactly_on_a_node_is_not_duplicated`、`test_arrival_between_two_nodes_keeps_the_first_segment`、`test_a_waveform_that_starts_positive_does_not_claim_an_observed_arrival`、`test_a_waveform_still_positive_at_the_end_is_marked_incomplete`、`test_a_waveform_without_any_positive_overpressure_reports_that`、`test_a_consecutive_zero_stretch_is_an_exact_zero_crossing`、`test_gaps_and_invalid_time_axes_are_refused`(×5)、`test_too_few_samples_is_refused`、`test_mismatched_lengths_are_refused` | passed ×17 | 到达阈值定义**没有改**；零点恰在两个采样点之间的解析积分误差来自线性插值，未做更高阶重构 |
| §9 | 1. **本机存在** `scripts/bounded_snake_check.py`，可独立运行（`exit 0`）。无法判断基线提交里为什么没有它。2. **复现**：`README.md` 的 `7.86e-2` 与"三个半径相同 ⇒ 与分辨率无关"。3. `0.41 %` **可复现**。4. 三项研究事项已分别记录 | 旧文本把"三个**装药半径**相同"读成"与**分辨率**无关"——两者不是一回事 | `README.md` 第八节、`AGENTS.md` 物理选择一节、`PHYSICS_BASIS.md` 覆盖表、`DISCRETE_BUDGET.md` 新增"复现记录" | 无（文档；其数字由 `test_scheme_b_migration.py` 的两条不变量测试守着） | 见 §9 详述 | `7.86e-2` 的**出处配置与版本已不可考**，标为"据报告、待复验" |
| §10 | — | — | `scripts/bounded_snake_check.py` 重写为 v2（14 项逐项检查）；本文件 | `pytest -q` → 457 passed / 0 skipped / 0 failed；联检 `exit 0` | passed | 见"仍未解决"一节 |

---

## §9 详述：文档更正

### 9.1 `scripts/bounded_snake_check.py`

本机存在且可独立运行：

```
PYTHONPATH=src python scripts/bounded_snake_check.py F:/2/snake_check/final3
exit code = 0
version = 2，14 项检查全部 ok
```

**无法判断**基线 `b8c45881` 里为什么没有它：本机没有 git，看不到提交内容。
这里只报告本机的实情，不推测。

### 9.2 撤回"与分辨率无关"

`README.md` 与 `AGENTS.md` 都写过：能量积分不变量偏差 `7.86e-2`，"三个半径上五位
数字完全相同，**所以**与分辨率无关，不能靠加密网格消掉"。

**这条推断是错的，而且方向相反。** 本机实测（`detonation_energy_budget`，
`charge_radius = 0.05 m`，`D = 6930 m/s`）：

| 单元数 | `energy_residual` |
|---|---|
| 300 | +4.894e-03 |
| 500 | +4.088e-03 |
| 1000 | +3.364e-03 |
| 2000 | +2.949e-03 |

同一半径、换分辨率，偏差从 4.89e-3 变到 2.95e-3——**它就是随分辨率变的**。
按装药半径不变只说明它与 `R_0` 无关（另测 0.025/0.040/0.080/0.100 m：同一分辨率下
数字完全相同）。

**当前值不是 `7.86e-2`**：修好之后落在 0.29 %~0.41 %（容差 1.0e-2）。根因是前端
单元的压力功按**本单元压力**扣（本单元一边被扫过一边膨胀，压力低于波前压力），
改成按**波前压力**扣之后降到这个量级。钉住它的是
`tests/test_scheme_b_migration.py::test_product_energy_invariant_is_met_at_the_default_resolution`。

`7.86e-2` 现在标为**历史值 / 据报告、待复验**：它没有随版本留下可核验的产物，
**不要**把它当成当前值。

### 9.3 `0.41 %` 是可复现的

`DISCRETE_BUDGET.md` 里那个 0.41 %（内能与动能之和与总能量的差）**不是只有报告
来源**：它由公开函数 `detonation_energy_budget()` 现算得出。已在文档里写出复现命令
与本机实测表。**来源仍未定论**，这一点没有变。

### 9.4 三项研究事项分别记录

已经写在 `README.md` 第八节、`DISCRETE_BUDGET.md` 与 `PHYSICS_BASIS.md` 里，并且
**明确标出它们不是彼此推出的结论**：

1. **离散能量残差 0.41 % 的来源**——未定论，只有否定结果。**可复现**，且随分辨率变化。
2. **相容离散 / 网格收敛**——未实施 / 未通过。这是**关于一族解**的陈述，单次求解
   生不出它。注意：`0.41 %` 随分辨率变化**不等于**网格收敛通过；两者是不同的命题。
3. **物理适用性 `physics_basis_status`**——未核实。它与上面两项**互补推出**。

数值上 `0.41 %` 与产物区能量不变量偏差的 0.29 %~0.41 % 很接近，**这是巧合**，
两处已分别注明，不得互相引用为证据。

---

## §10.2 独立有界合成联检

```bash
PYTHONPATH=src python scripts/bounded_snake_check.py F:/2/snake_check/final3
```

```
script  = scripts/bounded_snake_check.py
version = 2
budget  = {max_adam_updates: 18, max_lbfgs_iterations_requested: 5}
device  = cpu
duration = 2.32 s
ok      = true
```

逐项结果（一个总的 `ok` **不替代**这一张表）：

| # | 检查 | 结果 | 实测证据 |
|---|---|---|---|
| 1 | `continuous_run_reaches_the_end_of_the_schedule` | PASS | updates=6、status=`completed`、0.35 s、检查点存在 |
| 2 | `every_shortened_stage_boundary_is_crossed` | PASS | 进入 `initialise / strengthen_pde / joint` |
| 3 | `resume_at_a_stage_boundary_matches_a_continuous_run` | PASS | 断在第 2 步（窗口 0.65），权重**逐位相同**、采样器 RNG 相同、窗口 1.0 |
| 4 | `a_resumed_run_reports_the_measured_loss_of_its_last_update` | PASS | `last_update_loss = 51.4865`、`step = 6`（不再是 `nan` / `0`） |
| 5 | `a_failed_stage_stays_blocked_after_a_resume` | PASS | 创建时被拦；`blocked=True` 与 `pending='initialise'` 都经得起加载；恢复后**仍被拦下** |
| 6 | `a_completed_refinement_reports_the_same_identity_everywhere` | PASS | 返回值 / 清单 / 检查点三处都是 `diagnostic`；检查点含 1 条精修记录 |
| 7 | `a_refinement_rechecks_the_parameters_it_changed` | PASS | 检查数 9 → 11；其中**恰好 1 条**是落在当前参数上的 `full` 检查（另一条是收尾的阶段窗口检查） |
| 8 | `the_refinement_records_what_it_did` | PASS | 请求 5 次迭代、实际 **6** 次 closure（预算 40）；`parameters_changed=true`；在被接受的参数上求值 `27.7836`；0.14 s |
| 9 | `a_refinement_after_a_truncated_run_is_still_diagnostic` | PASS | updates=1、status=`diagnostic`、复检指纹 = 当前指纹 |
| 10 | `an_injected_refinement_failure_leaves_the_state_alone` | PASS | `NonFiniteUpdate`（含 `refinement #1, closure #1` 与参数名）；参数指纹不变；`last.pt` 字节不变；诊断文件写 `parameters_restored=true` |
| 11 | `the_impulse_counterexample_is_2_point_5` | PASS | impulse=2.5（期望 2.5）、duration=1.5（期望 1.5） |
| 12 | `a_contact_sweeping_between_frames_is_not_interpolated_across` | PASS | 原因码 `cross_discontinuity_interpolation`、值 NaN |
| 13 | `outside_the_prediction_grid_or_time_coverage_says_why` | PASS | `radius_outside_prediction_grid` / `time_outside_prediction_coverage` |
| 14 | `compare_cli_runs_end_to_end_and_writes_a_readable_report` | PASS | `exit 0`；`compare_synthetic-compare.json`；`comparable_points = 1344` |

输出目录：`F:/2/snake_check/final3/`（含 `snake_check.json` 与各步的检查点）。

**这份联检证明的是控制流，不是物理。** 它在测试范围内把阶段判据换成空集，
所以它**不可能**证明任何物理判据通过；正式入口不借用这个替身。判据本身由
`tests/test_stage_machine.py` 与 `tests/test_run_status_and_guards.py` 守着。

**联检自己踩到的两处**（都已修，属于**联检夹具**问题，不是产品缺陷）：

* 联检的合成算例是为**训练**那条链写的：网格粗、介质带窄，启发式前沿检测器在上面
  给出的 `R_c` 会落到 `R_s` 之外（实测两者几乎相等），材料分区退化成一整片"未受扰动"，
  于是对照一个点都比不了。修法不是放宽对照，而是给对照**另写一份**夹具
  （`scripts/bounded_snake_check.py:_compare_case`，走 `material.npz` 配套文件那条
  "分区是读数"的路径），并把断言从"报告存在"收紧为"`comparable_points > 0`"。
* 第 5 项的判据替换原本只罩住"创建阻塞状态"那一半，恢复那一半用的是空判据集合——
  那样一跑就过，"仍然被拦下"测的其实是"判据被我撤掉了"。现在替换罩住两个阶段。

---

## 10.1 分层验证

```bash
PYTHONPATH=src python -m pytest -q
# 457 passed, 0 skipped, 0 failed in 199.20s
# Python 3.10.8 / torch 2.5.1+cu121 / numpy 1.26.4
```

**0 skipped**——本机的真实数据（50/100/80 mm）在 `C:\Users\work\Documents\GitHub\2\`，
能读到，所以依赖真实数据的测试没有跳过。这与复核方的"6 skipped"是**环境差异**。

本轮新增/扩充的测试（共 84 条新测试）：

| 文件 | 条数 | 覆盖 |
|---|---:|---|
| `tests/test_compare_cli.py` | 22 | §1 命令行端到端、§7 插值与原因码 |
| `tests/test_waveform_impulse.py` | 17 | §8 过零积分 |
| `tests/test_refine_and_checkpoint.py` | 31 | §2/§3/§4/§6 |
| `tests/test_data_contract_fields.py`（扩充） | +14 | §5 资格自洽 |
| `tests/test_structural_validation.py`（扩充 + 改 1 条无效断言） | +2 | §5/§7 端到端 |

**没有**用改预期值、放宽容差、跳过检查或 `xfail` 消除任何失败。本轮修掉的
一条**无效断言**：`tests/test_structural_validation.py` 里原来的
`assert evaluate_case_contract(case).ready or True`——永真断言，基线本来就坏着也通过。
已改成对 `structural_problems` 的具体断言。

**测试替身只在 `tests/` 与 `scripts/bounded_snake_check.py` 里，不进正式入口。**

---

## 已修复的软件问题（本轮）

1. `compare` 命令行必现 `AttributeError`（§1）——并新增整条 CLI 的端到端测试。
2. 预测文件缺字段/错形状/单帧/时间语义不符**不再靠默认常数顶替**（§1.2/§1.3）。
3. `ambient_pressure` 从"有默认值"改为**必需**，并核对它是 101320 而非 101325 Pa（§1.3）。
4. L-BFGS 精修的非有限损失/梯度拦截与**全量回滚**（§2）。
5. 精修终态在返回值、清单、检查点、CLI 产物里**一致**，且身份随模型来源持久化（§3）。
6. 精修后**必定**在当前参数上重查（§4）。
7. best 允许保留更早的模型；检查记录带 `scope` 与检查集身份，窗口检查不再顶替完整检查（§4.3/§4.5）。
8. 前沿资格：档位封闭、证据与档位一致、派生掩码可重算、两侧独立（§5）。
9. 检查点补齐损失、日程、检查集与数据身份；全局 RNG 缺失即拒绝；回滚不制造历史事件；
   回滚失败即标记实例不可用（§6）。
10. 场量插值与测点压力**统一到一个核心**；跨间断不插值、零权重邻点不污染、域外与
    缺前沿各有原因码、计数互斥闭合（§7）。
11. 正相冲量过零积分的节点顺序与首段面积（§8）。
12. `compare_fields` 的结果字典在零样本时补齐 `_n_points`/`_max_abs_error`，并让
    `_render` 按**值**判"不可评价"——这两个缺陷叠在一起会让命令在渲染最后一步崩掉。
13. `_jsonable(dict)` 会把整份记录压成一行字符串——精修失败诊断文件因此读不回 dict。
14. 精修快照从"取批次之后"挪到"取批次之前"：`sampler.batch()` 会推进采样器 RNG，
    晚一步取快照，回滚之后参数还原了、采样轨迹没有。
15. Python 3.12 / NumPy 2.x 兼容性：`np.savez` 把标量读成 0 维数组，`isinstance(x, float)`
    会误判——`scaled_time_coverage` 改用 `np.ndim` 判定。

## 仍未解决的软件问题

* **"继续中断的 L-BFGS"不支持**（§3.5）。L-BFGS 的优化器状态与线搜索状态都不进
  检查点，而且线搜索本身不能在半途停下。现在由 `Trainer.resume_refinement()` **明确
  拒绝**，而不是只在报告里写一行 `resume_supported: false`。要做完整恢复就需要把
  线搜索状态一起持久化——本轮没有做。
* **版本 4 及更早的检查点一律拒绝**，没有迁移路径。缺的字段（最后一次损失、日程、
  检查集、数据身份）无法从旧文件重建，所以只能重训或用权重显式导入。
* **single-frame 一维场布局**已按版本化规则落地（`profile_layout`），但三份真数据
  是旧提取器产出的，文件里没有这一项，读出来是 `declared=False`（推断的缺省，
  不是声明）。补上要重提。
* **80 mm 缺配套 deck**，无法重提（提取器需要它做 part id → ALE 材料序号映射）。

## 缺失资源 / 真实数据验证

实查结果（不是照抄历史报告）：

| 数据 | 位置 | 帧数 | 读取 | 契约 |
|---|---|---:|---|---|
| 50 mm | `C:\Users\work\Documents\GitHub\2\extracted` | 195 | 成功 | 未就绪（缺 companion） |
| 100 mm | `C:\Users\work\Documents\GitHub\2\extracted_100mm` | 30 | 成功 | 未就绪（缺 companion） |
| 80 mm | `C:\Users\work\Documents\GitHub\2\extracted_80mm` | 483 | 成功 | 未就绪（缺 companion） |
| 500 mm | `D:/500mmntt/_ProjectScratch/Scr59F2/` | — | — | **未提取** |

三份都是 `profile_layout = frames_by_radius`、`declared=False`（旧提取器没写声明）。
**没有**为它们跑过任何提取或训练。

## 独立保留的研究性验收项

1. **离散能量残差 0.41 % 的来源**——未定论。**可复现**（见 `DISCRETE_BUDGET.md`），
   随分辨率变化，但**来源不明**。
2. **相容离散未实施 / 网格收敛未通过**——关于一族解的陈述。
3. **`physics_basis_status` 未核实**。

三者**并列**，不是彼此推出的结论，也不能用来说明上面任何一条软件问题。

## 正式入口当前的实际原因

`build_reference` → `assert_reference_accepted` 在**任何训练开始之前**
抛 `ReferenceNotAccepted`：缺 `grid_convergence`、`timestep_sensitivity`、
`scale_mapping`，且物理适用性未核实。实测：

```bash
PYTHONPATH=src python -m spherical_blast_pinn.cli.train --config configs/experiments/geom_smoke.yaml
# -> ReferenceNotAccepted
```

这是**物理闸门**，与本轮的软件项无关。软件项已按上表逐条闭合，但**这不改变物理
闸门的状态**，也**不构成**"正式训练已就绪"。

---

## 一句话总结

本轮列出的 §1–§9 软件问题，**逐条在本机实际版本上复现、修复并验证**；
对应路径的回归测试与独立联检都通过（457 passed / 0 skipped；联检 14/14）。
**但**：精度恢复不支持、旧检查点无迁移、真数据缺声明、80 mm 缺 deck，
以及三项研究性验收项，都仍然开着——它们与本轮软件修复无关，也**不能**
用"软件修好了"来替代。
