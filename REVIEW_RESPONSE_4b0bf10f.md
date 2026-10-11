# 复核回应：`4b0bf10f`（2026-10-11）

## 0. 版本、环境与执行规则

| 项 | 值 |
|---|---|
| 复核基线 | `4b0bf10f`（复核方给出的提交号） |
| 本机 HEAD | **无本地 git 仓库**（`git rev-parse` 报 "not a git repository"）。交付物是当前工作区，**无法给出提交号**；下面用文件哈希标识 |
| 复核方环境 | —— 实跑 **451 passed / 6 skipped** |
| 本机环境 | Python **3.10.8**、torch **2.5.1+cu121**、numpy **1.26.4** |
| 测试命令 | `PYTHONPATH=src python -m pytest -q` |
| 测试结果 | **490 passed，0 skipped，0 failed**（本轮开始时 457 passed） |
| 联检命令 | `PYTHONPATH=src python scripts/bounded_snake_check.py <输出目录>` |
| 联检结果 | **21/21 通过，exit 0**（脚本版本 **3**） |

**本机与复核基线不是同一个提交。** 复核方在 `4b0bf10f` 上跑出 451 passed / 6 skipped，
本机 490 passed / 0 skipped。本机没有 git，**无法比对差异**；本轮按提示词要求在本机
实际版本上逐条复验，**§1–§7 全部复现**（下表给出实测输出）。

交付文件的 `sha256` 前 16 位：

| 文件 | 字节 | `sha256[:16]` |
|---|---:|---|
| `src/spherical_blast_pinn/cli/compare.py` | 55923 | `a9625f531185da65` |
| `src/spherical_blast_pinn/evaluation/waveform.py` | 16434 | `8d6b04cf57351220` |
| `src/spherical_blast_pinn/training/trainer.py` | 114085 | `19a80d6dc324f0eb` |
| `src/spherical_blast_pinn/sampling/sampler.py` | 42902 | `c34b0792ec9a50e6` |
| `src/spherical_blast_pinn/data/extracted_case.py` | 83153 | `1a9d643eba407504` |
| `src/spherical_blast_pinn/data/contract.py` | 19942 | `a2450deaf87334c4` |
| `scripts/bounded_snake_check.py` | 39933 | `620f3cf7ddcaf208` |
| `tests/test_compare_cli.py` | 22572 | `0d83d90d046f8766` |
| `tests/test_waveform_impulse.py` | 14156 | `13aaa25f5be5a298` |
| `tests/test_refine_and_checkpoint.py` | 43811 | `53bcba892e1bda7b` |
| `tests/test_profile_layout.py` | 12032 | `3a94a3a4c576cc6d` |

---

## 逐项对应表

| 编号 | 基线反例与实际复现结果 | 根因 | 修改文件/函数 | 回归测试名称 | 实际执行结果 | 尚未覆盖的边界 |
|---|---|---|---|---|---|---|
| §1 | **复现**：精修改参数后让收尾 `verify()` 抛错 → 参数**没恢复**（指纹比较 = False）、`refine_history` 0→**1**、**没有** `refinement_failures.json`、内存模型已变而 `last.pt` 未变 | `try/except` 只罩住"线搜索 + 最终损失求值"；采样、历史追加、复检、写盘都在事务之外；也没有像 `load()` 那样处理"回滚本身失败" | `trainer.py:Trainer.refine`：把采样 → 优化 → 最终求值 → 指纹 → **历史提交** → **复检与写盘**全部纳入一个 `try`；新增 `_capture_artifacts()`/`_restore_artifacts()`（检查点**字节**级还原，当时不存在的文件要删掉）；失败记录带 `stage`；回滚失败即 `_unusable` 并保留两个异常 | `test_a_refinement_failure_at_any_stage_rolls_the_whole_flow_back`(×4 参数：sampling/optimisation/recheck/save)、`test_a_failed_rollback_during_refinement_marks_the_instance_unusable` | passed ×5 | 失败注入是**人为的**（monkeypatch）；没有在真实磁盘满/权限错误下测过 |
| §2 | **复现**：`best.pt` 的 `run_status` = **`completed`**，`last.pt` = `diagnostic`，清单 = `diagnostic`；两份参数相同，且 `best.pt` 已含 1 条精修记录 | `verify()` 直接 `_save(best.pt)`，而 `run()` 到后面才设本次终态——新参数配上了**上一次**的状态 | `trainer.py`：`verify()` 只登记候选并置 `_best_dirty`；`run()` 算完终态之后才写 `best.pt`（先 best 后 last）；保留旧最佳时**不重写** `best.pt`；`best` 记录带上 `run_origin` 与 `refinement_count` | `test_best_pt_carries_the_status_of_the_run_that_selected_it`、`test_a_kept_earlier_best_keeps_its_own_status` | passed ×2 | 只在单算例小网络上验证过；best 的 `score` 偏序本身没有改动 |
| §3 | **复现**：`diagnostic=True` 完成 → 存盘 → 新实例加载得到 `diagnostic` → **零更新再 `run()` → `completed`，`formally_accepted=True`** | 只继承了精修历史，没有持久继承"这条运行**最初被明确指定为诊断**"的来源；`_final_status` 也不是可继承的东西（阶段失败/截断必须重新评价） | `trainer.py`：新增 `_run_origin = {explicit_diagnostic, refined}`，随检查点存取；`run()` **合并**而不是覆盖来源；终态由来源决定；检查点缺 `run_origin` 即拒绝（在 `load()` 的**只读**阶段核对类型） | `test_an_explicit_diagnostic_identity_survives_save_load_and_further_runs`、`test_a_plain_run_keeps_its_normal_status`、`test_a_checkpoint_without_a_run_origin_is_refused` | passed ×3 | 来源一旦是诊断就**永久**是诊断（要正式结果需另起一个不加载的 trainer）；这是刻意的，但没有提供"显式清除来源"的入口 |
| §4 | **复现**：保持坐标与前沿不变，密度 ×2、速度 +123 → **哈希不变**，旧检查点照常加载 | `_data_identity()` 只哈希时间轴、径向坐标、两条前沿与材料码的**计数**；计相同也证明不了逐点分布相同 | `trainer.py`：`_data_identity()` 覆盖四个场量、`material`、`valid`、`contact_valid`/`shock_valid`、**逐字段有效性**、**前沿监督资格**；新增 `_hash_field()`（名字 + 形状 + dtype + 字节，`None` 写显式 `missing`）；检查点版本 5→**6** | `test_a_changed_field_is_refused_by_the_data_identity`(×6 参数)、`test_missing_and_empty_are_different_in_the_data_identity`、`test_the_data_identity_ignores_the_case_id_label`、`test_moving_the_output_directory_does_not_change_data_identity`、`test_changing_the_data_behind_the_same_path_is_refused` | passed ×10 | 版本 5 及更早的检查点**一律拒绝**（哈希格式变了，旧的不完整哈希不能当新格式用）；未做迁移 |
| §5 | **复现**：查询第一帧本身有效的节点，只把**第二帧**的接触面改成 NaN → 修前 `NaN` / `front_record_unavailable_at_query_time` | `_fronts_at()` 无条件检查两个时间端点，即使其中一个是零权重；场量插值早就按"零权重不参与"改了，前沿这一侧漏掉 | `cli/compare.py:_fronts_at`：只累加权重 `> 0` 的那一帧；非零权重的帧缺测才返回不可用 | `test_an_exact_hit_on_a_frame_ignores_a_missing_neighbour_frame`、`test_an_exact_hit_on_the_last_frame_ignores_a_missing_neighbour_frame`、`test_a_query_genuinely_between_frames_needs_both_frames`、`test_the_front_rules_agree_between_exact_nodes_and_between_frames` | passed ×4 | 帧间前沿仍是**线性**插值；这是写下来的选择 |
| §6 | **复现**：模型正相未结束、参考已结束 → 仍输出正相时长误差 `0.400` 与冲量误差 `0.0163`；两边 `arrival_observed=False` → 仍输出到达误差 `0.0` | 到达误差无条件计算；正相误差只检查参考一侧，没有同时检查模型 | `evaluation/waveform.py:compare_gauge`：到达误差要求**两边都** `arrival_observed`；正相时长与冲量要求**两边都** `positive_phase_complete`；不满足时把原因写进 `notes`，已覆盖区间的积分值仍留在各自指标里 | `test_an_incomplete_model_phase_removes_the_phase_errors`、`test_both_truncated_is_reported_as_such`、`test_an_unobserved_arrival_removes_the_arrival_error`、`test_a_fully_observed_pair_still_gets_all_the_errors` | passed ×4 | 峰值超压仍然**无条件**比较（两端都有定义）；没有为它引入可用性标志 |
| §7 | **复现**：`[0,1,2,3]` / 超压 `[0,2,0,-1]` / 阈值 0 → 面积 **0**（正确 **2**）、时长 **0**（正确 **2**）、`positive_phase_complete=True` | 到达插值点恰好为零，找"第一个 `<= 0`"时直接选中了**起点**（上穿点），没有区分上穿与下穿；`stop == 0` 分支把整段判成退化 | `evaluation/waveform.py:gauge_metrics`：先定位**严格正**的区间（`strictly_positive`），再往后找第一次下穿或到零；起点那个 0 **保留**为积分节点；新增阈值合法性校验（**零阈值仍然合法**，只拒绝负值与非有限值） | `test_a_zero_threshold_does_not_confuse_the_start_with_the_end`、`test_the_zero_threshold_case_still_reports_the_original_regression`、`test_a_zero_threshold_with_a_leading_zero_plateau`、`test_a_zero_threshold_that_never_goes_positive`、`test_an_invalid_threshold_is_refused`(×3) | passed ×7 | 峰值**恰好为 0** 时走"没有正超压"那条路（冲量 `nan` + 说明），不是 0——这是规定下来的行为，已写进测试 |
| §8 | **本机存在** `scripts/bounded_snake_check.py`（无盘符硬编码，输出目录由参数指定）。**干净检出验证**：把仓库复制到 `C:/Users/work/AppData/Local/Temp/clean_checkout/repo`（排除 `__pycache__`/`.pytest_cache`/`_legacy_v1_backup`/`outputs`），在**那份副本**里运行，输出到另一个目录 → **exit 0，21/21** | —— | `scripts/bounded_snake_check.py` 升到 v3：新增 7 项检查（精修事务、best 终态、诊断身份、数据身份、零阈值三角波、零权重邻帧、不可用指标不判分） | —— | 见下 | **无法判断**基线提交里为什么没有它（本机无 git）。真实数据相关的检查在联检里**不运行**，与合成联检分开报告 |

---

## §10.2 独立有界合成联检（v3）

```bash
PYTHONPATH=src python scripts/bounded_snake_check.py <输出目录>
```

```
script  = scripts/bounded_snake_check.py
version = 3
budget  = {max_adam_updates: 18, max_lbfgs_iterations_requested: 5}
device  = cpu
duration = 3.6 s
ok      = true     （21 项，exit 0）
```

**干净检出复跑**（排除 `__pycache__`、`.pytest_cache`、`_legacy_v1_backup`、`outputs`，
输出目录与仓库无关）：

```
copied -> C:\Users\work\AppData\Local\Temp\clean_checkout\repo
PYTHONPATH=src python scripts/bounded_snake_check.py C:/Users/work/AppData/Local/Temp/clean_checkout/out
exit = 0，21/21 通过
```

逐项（`ok` 的总值不替代这一张表）：

| # | 检查 | 结果 | 关键证据 |
|---|---|---|---|
| 1 | `continuous_run_reaches_the_end_of_the_schedule` | PASS | updates=6、`completed` |
| 2 | `every_shortened_stage_boundary_is_crossed` | PASS | `initialise / strengthen_pde / joint` |
| 3 | `resume_at_a_stage_boundary_matches_a_continuous_run` | PASS | 权重逐位相同、采样器 RNG 相同 |
| 4 | `a_resumed_run_reports_the_measured_loss_of_its_last_update` | PASS | `51.4865` @ step 6 |
| 5 | `a_failed_stage_stays_blocked_after_a_resume` | PASS | `blocked` 标志与 `pending` 经得起加载，恢复后仍被拦下 |
| 6 | `a_completed_refinement_reports_the_same_identity_everywhere` | PASS | 三处都是 `diagnostic`，检查点含 1 条精修记录 |
| 7 | `a_refinement_rechecks_the_parameters_it_changed` | PASS | 恰好 1 条 `full` 检查落在当前指纹上 |
| 8 | `the_refinement_records_what_it_did` | PASS | 请求 5、实际 closure ≤ 40；`parameters_changed=true` |
| 9 | `a_refinement_after_a_truncated_run_is_still_diagnostic` | PASS | updates=1、`diagnostic` |
| 10 | `an_injected_refinement_failure_leaves_the_state_alone` | PASS | 参数指纹与检查点字节都不变 |
| 11 | `the_impulse_counterexample_is_2_point_5` | PASS | 2.5 / 1.5 |
| 12 | `a_contact_sweeping_between_frames_is_not_interpolated_across` | PASS | `cross_discontinuity_interpolation` |
| 13 | `outside_the_prediction_grid_or_time_coverage_says_why` | PASS | 两个原因码各就其位 |
| 14 | `compare_cli_runs_end_to_end_and_writes_a_readable_report` | PASS | `exit 0`，`comparable_points = 1344` |
| 15 | `a_refinement_failure_after_the_update_leaves_no_half_changed_state` | PASS | **本轮 §1**：参数 / RNG / 历史 / 两份检查点**四项全不变** |
| 16 | `an_explicit_diagnostic_identity_survives_a_reload` | PASS | **本轮 §3**：加载后零更新 `run()` 仍是 `diagnostic` |
| 17 | `best_pt_carries_the_status_of_the_run_that_selected_it` | PASS | **本轮 §2**：`best` 与 `last` 都是 `diagnostic` |
| 18 | `a_changed_field_is_refused_by_the_data_identity` | PASS | **本轮 §4**：四个变体全拒绝，**未改动的对照能加载** |
| 19 | `a_zero_threshold_triangle_reports_area_2_and_duration_2` | PASS | **本轮 §7**：2.0 / 2.0 |
| 20 | `an_exact_frame_ignores_a_missing_neighbour_frame` | PASS | **本轮 §5**：精确命中 `ok`，帧间 `front_record_unavailable` |
| 21 | `an_incomplete_phase_removes_the_phase_and_arrival_errors` | PASS | **本轮 §6**：三项误差都不判分，部分积分值仍在 |

**这份联检证明的是控制流，不是物理。** 它在测试范围内把阶段判据换成空集，
**不可能**证明任何物理判据通过；正式入口不借用这个替身。

**真实数据相关的检查在联检里没有运行**，不与合成联检合并成"全部通过"。

### 联检 v3 自己踩到的两处（**夹具**问题，不是产品缺陷，都已修）

* `material` 的变异写成"把空气改成环境"，在这份粗网格合成算例上是**空操作**
  （启发式检测器给出的 `material` 全是 2），于是"未拒绝"看起来像产品漏了。改成
  整体轮转 `0→1→2→0`，每个单元都变且仍在合法取值域里。
* 多个变体的 trainer **共享同一个算例对象**（`Trainer.cases = list(cases)` 只复制
  列表），第一处变异把**对照组**也改坏了。改成每个变体按来源目录重新读一份算例。
  这是夹具的问题，但值得记住：**共享可变对象会让"对照组"失去意义。**

---

## 交付验证

```bash
PYTHONPATH=src python -m pytest -q
# 490 passed, 0 skipped, 0 failed in 259.39s
# Python 3.10.8 / torch 2.5.1+cu121 / numpy 1.26.4
```

本轮新增 **33** 条测试（457 → 490）：

| 文件 | 新增 | 覆盖 |
|---|---:|---|
| `tests/test_refine_and_checkpoint.py` | 16 | §1 事务（×4 阶段 + 回滚失败）、§2 best 终态（×2）、§3 诊断身份（×3）、§4 数据身份（×6 + 2） |
| `tests/test_compare_cli.py` | 4 | §5 前沿插值的零权重规则 |
| `tests/test_waveform_impulse.py` | 11 | §6 可用性标志（×4）、§7 零阈值（×7） |

新增测试的清单：

* `test_a_refinement_failure_at_any_stage_rolls_the_whole_flow_back`（4 参数）
* `test_a_failed_rollback_during_refinement_marks_the_instance_unusable`
* `test_best_pt_carries_the_status_of_the_run_that_selected_it`
* `test_a_kept_earlier_best_keeps_its_own_status`
* `test_an_explicit_diagnostic_identity_survives_save_load_and_further_runs`
* `test_a_plain_run_keeps_its_normal_status`
* `test_a_checkpoint_without_a_run_origin_is_refused`
* `test_a_changed_field_is_refused_by_the_data_identity`（6 参数）
* `test_missing_and_empty_are_different_in_the_data_identity`
* `test_the_data_identity_ignores_the_case_id_label`
* `test_an_exact_hit_on_a_frame_ignores_a_missing_neighbour_frame`
* `test_an_exact_hit_on_the_last_frame_ignores_a_missing_neighbour_frame`
* `test_a_query_genuinely_between_frames_needs_both_frames`
* `test_the_front_rules_agree_between_exact_nodes_and_between_frames`
* `test_an_incomplete_model_phase_removes_the_phase_errors`
* `test_both_truncated_is_reported_as_such`
* `test_an_unobserved_arrival_removes_the_arrival_error`
* `test_a_fully_observed_pair_still_gets_all_the_errors`
* `test_a_zero_threshold_does_not_confuse_the_start_with_the_end`
* `test_the_zero_threshold_case_still_reports_the_original_regression`
* `test_a_zero_threshold_with_a_leading_zero_plateau`
* `test_a_zero_threshold_that_never_goes_positive`
* `test_an_invalid_threshold_is_refused`（3 参数）

**没有**用改预期值、放宽容差、跳过检查或 `xfail` 消除任何失败。

---

## 已修复的软件问题（本轮）

1. 精修的**整条流程**（采样 → 优化 → 求值 → 指纹 → 历史 → 复检 → 写盘）纳入一个事务；
   失败时内存与**盘上**检查点一起还原；回滚失败即标记实例不可用（§1）。
2. `best.pt` 不再沿用上一次运行的终态：候选登记与写盘分开，保留旧最佳时不重写（§2）。
3. 显式诊断身份随检查点持久化，加载与后续运行都不会把它算成正式完成（§3）。
4. 数据身份覆盖四个场量、逐点材料分布、联合掩码、逐字段有效性与前沿资格；
   `missing` 与"空"分开（§4）。
5. `_fronts_at` 只在权重非零的帧上要求有效（§5）。
6. `compare_gauge` 按可用性标志决定是否判分（§6）。
7. 零阈值下区分上穿起点与下穿终点；零阈值本身仍然合法（§7）。
8. 联检脚本随交付、无盘符硬编码、干净检出可复跑，逐项报告（§8）。

## 仍未解决的软件 / 交付问题

* **中断的 L-BFGS 无法继续**：`resume_refinement()` 明确拒绝。**保留**，属于已声明的
  功能限制。要支持它需要把优化器与线搜索状态完整恢复，不能把"重新开始"叫"继续"。
* **版本 5 及更早的检查点拒绝恢复**：本轮的 `run_origin` 与新的数据身份格式都改变了
  检查点结构，版本升到 **6**，**没有迁移路径**。缺失信息无法伪造成精确恢复。
* **`_run_origin` 一旦为诊断就不可清除**：这是刻意的（身份跟着参数走），但没有提供
  一个"显式开一条新血统"的入口。要正式结果目前得新建 trainer 而不加载检查点。
* **峰值超压没有可用性标志**：它两端都有定义，所以无条件比较；如果以后引入"某侧
  根本没有正压"的情形，需要一并处理。

## 缺失资源 / 真实数据验证

实查（不是照抄历史报告）：

| 数据 | 位置 | 帧数 | 读取 | 契约 |
|---|---|---:|---|---|
| 50 mm | `C:\Users\work\Documents\GitHub\2\extracted` | 195 | 成功 | 未就绪（缺 companion） |
| 100 mm | `C:\Users\work\Documents\GitHub\2\extracted_100mm` | 30 | 成功 | 未就绪（缺 companion） |
| 80 mm | `C:\Users\work\Documents\GitHub\2\extracted_80mm` | 483 | 成功 | 未就绪（缺 companion） |
| 500 mm | `D:/500mmntt/_ProjectScratch/Scr59F2/` | — | — | **未提取** |

三份都是 `profile_layout = frames_by_radius`、`declared=False`（旧提取器没写声明）。
**没有**把 `declared` 手动改成真。**80 mm 缺配套 deck**，无法重提。
本机 **0 skipped**（真实数据可读）；复核方的 6 项跳过是**环境差异**，需要有数据的
环境单独给出同一提交下的执行证据。

## 独立保留的研究性验收项

1. **离散能量残差 0.41 % 的来源**——未定论。**可复现**（`detonation_energy_budget()`），
   实测随分辨率变化（300/500/1000/2000 单元：4.894e-3 / 4.088e-3 / 3.364e-3 / 2.949e-3）。
2. **相容离散未实施 / 网格收敛未通过**。
3. **`physics_basis_status` 未核实**。

三者**并列**，不是彼此推出的结论。数值上 `0.41 %` 与产物区能量不变量偏差的
0.29 %~0.41 % 接近，**是巧合**，两处已分别注明。

## 正式入口当前的实际原因

`assert_reference_accepted()` 保留 `ReferenceNotAccepted`：缺 `grid_convergence`、
`timestep_sensitivity`、`scale_mapping`，且物理适用性未核实。

```bash
PYTHONPATH=src python -m spherical_blast_pinn.cli.train --config configs/experiments/geom_smoke.yaml
# -> ReferenceNotAccepted
```

软件回归通过**不能**自动补齐缺失的验收证据。

---

## 一句话总结

本轮列出的 §1–§8 问题**逐条在本机实际版本上复现、修复并验证**：490 passed / 0 skipped，
联检 v3 **21/21**，并在**干净检出副本**里独立复跑通过。**但**：中断的 L-BFGS 仍不支持、
版本 5 及更早的检查点无迁移、诊断来源不可清除、真数据缺布局声明、80 mm 缺 deck，
以及三项研究性验收项——全部照旧开着，不能用"软件修好了"替代。
