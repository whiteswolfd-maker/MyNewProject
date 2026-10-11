# 复核回应：`3e4e1c61b351fb4665c35f4dfc7e832c12e80e70`（2026-10-11）

## 0. 提交、环境与执行规则

| 项 | 值 |
|---|---|
| 复核基线提交 | `3e4e1c61b351fb4665c35f4dfc7e832c12e80e70` |
| 本机实际克隆到的基线 | `3e4e1c61b351fb4665c35f4dfc7e832c12e80e70`（`git clone` 实测，与复核一致） |
| 本轮修改提交 | `8d7fda6303ece357bd9819e450b15cdab08522ef`（已推送到 `origin/main`） |
| 远端复验提交 | `8d7fda6303ece357bd9819e450b15cdab08522ef`（**全新 `git clone`** 后在该目录内执行） |
| 分支 | `main` |
| 本机环境 | Python **3.10.8**、torch **2.5.1+cu121**、numpy **1.26.4** |
| 复核方环境 | 实跑 **484 passed / 6 skipped / 0 failed** |
| 本机（原工作区） | `pytest -q -rs` → **520 passed / 0 skipped / 0 failed** |
| 本机（**远端新克隆**） | 同上：**520 passed / 0 skipped / 0 failed** |

**关于 6 skipped**：复核环境缺真实提取数据，所以依赖它的测试跳过；本机在
`C:\Users\work\Documents\GitHub\2\` 下**有**那些数据，因此不跳过。这是**环境差异**，
不是产品差异，也不把跳过写成通过——两侧的源码提交是同一个。

上一轮报告说"干净检出"用的是**复制本地目录**。这一步只能证明目录迁移后能跑，
不能证明远端交付完整——而且它正是 `scripts/` 缺席两轮没被发现的原因。本轮改为
真正的 `git clone`，见 §5。

---

## 1. 五项状态一览

| 项 | 状态 | 一句话 |
|---|---|---|
| §1 最佳候选候选快照 | **完成** | 入选时对整个状态取独立快照，发布用候选那一份；指纹对照见 §3 |
| §2 事务覆盖整套产物 | **完成** | 四份产物按实际调用链保护，内存状态一并入快照；故障矩阵见 §4 |
| §3 内存/磁盘恢复独立 | **完成** | 分别受保护、逐文件尝试；不可恢复的产物被标记且加载端拒绝 |
| §4 左端缺失不判完整 | **完成** | `positive_phase_complete` 要求两端都覆盖；组合矩阵见 §5 |
| §5 脚本进入远端提交 | **完成** | 已提交并推送，已用**全新 clone** 复验 |

---

## 2. 逐项：修前复现 / 根因 / 修改 / 测试

### §1 最佳候选必须保存入选时的状态

**修前复现**（真实小 trainer、真实参数更新、真实写盘，只控制评分）：

```
best_check.step                     = 1
best.pt 里的 step                    = 6
best_check 指纹 == best.pt 实际参数  = False
best.pt 参数 == 最后一步参数         = True
```

**根因**：`verify()` 只登记 `self.best = record` 与 `_best_dirty`；发布时通用
`_save()` 读的是**当前** `self.model.state_dict()`。中途选出的最佳因此被写成末步
参数，而 `best_check.parameter_fingerprint` 与文件里的参数不是同一组。

**修改**（`src/spherical_blast_pinn/training/trainer.py`）：

* 新增 `_checkpoint_state()`：取那一刻的**独立副本**（模型张量 `clone()`、
  优化器/平衡器/历史 `deepcopy`）。
* 保存拆成三层：`_checkpoint_state()` 取状态 → `_build_payload()` 拼载荷 →
  `_write_payload()` 原子写出。`_save()` = 用当前状态走这三层。
* `verify()` 入选时 `self._best_candidate = self._checkpoint_state()`。
* `run()` 发布 `best.pt` 时用**候选**那份状态，并补 `candidate` 块；
  `final_stage_verdict` 置 `None`（判据是在整次运行自己最后那组参数上评出来的）。
* 候选与 `_best_dirty` 随检查点保存（`pending_best_candidate` / `best_dirty`），
  并进入事务快照。
* `CHECKPOINT_VERSION` 6 → **7**，缺这两项的检查点明确拒绝。

**回归测试**（`tests/test_best_candidate_and_transaction.py`）：
`test_a_mid_run_best_is_published_with_its_own_parameters`、
`test_the_last_step_becoming_best_keeps_everything_self_consistent`、
`test_a_candidate_snapshot_does_not_follow_later_parameter_updates`、
`test_an_interrupted_run_keeps_the_unpublished_candidate`、
`test_resuming_from_best_matches_resuming_from_an_independent_checkpoint`、
`test_a_kept_earlier_best_is_not_rewritten`、
`test_refined_candidates_keep_or_yield_the_diagnostic_identity`（2 参数）

### §2 事务必须覆盖整套发布产物

**修前复现**（`checks.json` 已经写出、`run_manifest.json` 才失败）：

```
last.pt             恢复 = True
best.pt             恢复 = True
checks.json         恢复 = False      <-- 
run_manifest.json   恢复 = True
_final_check_state  = 'checked_on_current_parameters'（原本 = 'current_parameters_already_carry_a_conclusion'）
```

**根因**：`_PROTECTED_ARTIFACTS` 只有 `last.pt` / `best.pt`。`run()` 实际会写四份
产物，清单与调用链不符；`_final_check_state` 也没进 `_snapshot_state()`。

**修改**：

* `_PROTECTED_ARTIFACTS` 按 `run()` 的**实际调用链**扩到四份；
  `cli/train.py` 的 `train_result.json` 明确记为外层责任（它在 `run()` 返回之后写）。
* `_snapshot_state()` / `_restore_state()` 补 `_final_check_state`、`_final_status`、
  `_final_stage_verdict`、`_best_candidate`、`_best_dirty`。
* `_restore_artifacts()` 返回**逐文件结果**，不再是一个总布尔值。
* `run()` 开头清掉上一次的终态（一次正在进行的运行没有终态）。
* 多文件发布前写 `.publish_in_progress.json`，成功后删除；`load()` 在**被加载
  检查点旁边**找这个标记，命中即拒绝。

### §3 内存恢复与磁盘恢复必须独立尝试

**修前复现**（写出 `last.pt` 后失败 + 内存恢复失败）：

```
实例被标为不可用    = True
last.pt 恢复旧文件  = False
last.pt 带本次精修记录 = 1
新实例加载这份 last.pt = 成功 **
```

**根因**：`try: _restore_state(); _restore_artifacts()` 写在一个 `try` 里——第一步
抛错，第二步根本不执行。一个对象不可用**并不能**阻止另一个对象读盘。

**修改**：两路分别受保护；磁盘恢复逐文件独立尝试并返回逐项结果；诊断记录带
`memory_rollback_error` 与 `artifact_rollback`；只要有一项没恢复就保持 `_unusable`；
诊断文件写不进去时原始错误与恢复错误都在最终异常里。标记只在**磁盘**全部还原时
撤掉（内存失败不影响目录里是什么）。

### §4 左端缺失的时程不能判为完整正相

**修前复现**（`times=[0,1,2,3]`、超压 `[2,2,-2,-2]`）：

```
arrival_observed = False
positive_phase_complete = True          <-- 首点已为正，却在判分
positive_duration_relative 仍然输出 = 0.0
positive_impulse_relative  仍然输出 = 0.0
```

**根因**：`arrival_observed` 的判据是 `arrival > times[0]`，而
`positive_phase_complete` 只表示"右端观测到"。两个端点被混成了一个。

**修改**（`src/spherical_blast_pinn/evaluation/waveform.py`）：

* `arrival_observed = (首点 <= 阈值)`——阈值为 0 时首点恰在零点上，那一刻**是**
  观测到的起点，与"首点已经为正"分开。
* 新增 `phase_end_observed` 与 `positive_phase_coverage`
  （`full` / `start_missing` / `end_truncated` / `start_missing_and_end_truncated` /
  `no_positive_phase`）。
* `positive_phase_complete` = **两端都覆盖**。
* `compare_gauge()`：完整正相时长与冲量要求**两侧各自**完整；原因分三种写进
  `notes`；已覆盖区间的积分值保留并由 `positive_impulse_is_partial` 标为部分积分。

**已有一条测试的语义断言被更新**（复核要求先说明冲突）：`test_waveform_impulse.py`
原来把 `[2,2,-2,-2]` 断言为 `positive_phase_complete is True`。它与"整段正相都被
记录覆盖"这一要求**冲突**——该曲线首点已严格高于阈值，左端未观测到。断言改为
`is False`，**解析面积 2.5 一个字没动**。

### §5 联检脚本进入远端提交

**证据**（全部在本机实跑）：

```
$ git clone https://github.com/whiteswolfd-maker/MyNewProject.git mynewproject_clone
$ cd mynewproject_clone && git ls-tree -r --name-only HEAD scripts
                    <-- 空：基线提交里没有 scripts/
$ git show HEAD:src/spherical_blast_pinn/training/trainer.py | sha256sum
a263eba9903881f574a22a61e5bd96d0904a1b7c41f37f88298dfe666b57d429
```

**原因**：`scripts/` 从来没有被 `git add` 过；前两轮报告引用了它，但交付里没有。
本机也没有 git 仓库，所以我此前无从发现，只能用"复制目录"自证，而那**不能**证明
远端交付完整。

**修改**：脚本随本轮提交进入 `main`，并升至 **v4**（27 项检查，含本轮四项）。
远端复验见 §6。

---

## 3. 最佳候选与文件内实际参数的指纹对照

`best.pt` 里的参数指纹是**独立重算**的（对文件里的 `model` 状态逐张量哈希），
不是读 `best_check` 文本。

| 项 | 值 |
|---|---|
| 每步独立记录的指纹（键为步号） | 6 步，互不相同 = `True` |
| `best_check['step']` | **1** |
| `best.pt['step']` | **1** |
| `best.pt` 参数指纹（独立重算） | `7349e27aab0e66e26aee7802…` |
| `best_check['parameter_fingerprint']` | `7349e27aab0e66e26aee7802…` |
| 两者相符 | **True** |
| `best.pt` 参数 == 第 1 步独立指纹 | **True** |
| 总步数 | 6 |
| `last.pt['step']` | 6 |
| `last.pt` 参数 == 末步独立指纹 | **True** |
| `best.pt` 参数 != `last.pt` 参数 | **True** |
| `best.pt['candidate']` | `{"step": 1, "parameter_fingerprint": "7349e27a…", "selected_in_run_with_status": "completed", …}` |
| `best.pt['final_stage_verdict']` | `None`（判据属于整次运行，不套在中途候选上） |

**候选快照独立性**：快照步号在后续 5 步更新前后都是 `1`，指纹不变（`True`），
而当前模型指纹已变（`True`）。

---

## 4. 事务故障矩阵

| 故障位置 | 已完成的副作用 | 内存恢复 | 文件恢复 | 失败产物能否被新实例加载 |
|---|---|---|---|---|
| A：`run_manifest.json` 写出失败（`last.pt`/`best.pt`/`checks.json` 都已真实替换） | 三份已替换 | **是** | **是**（四份全部还原） | 可加载；回滚完整，标记已撤 |
| B：内存恢复失败（磁盘恢复仍执行） | `last.pt` 已替换 | **否**（实例标为不可用） | **是** | 可加载，盘上是**旧**检查点（精修记录 0 条） |
| C：`last.pt` 无法恢复 | 四份都被替换过 | 是 | **否** | **拒绝加载**（未提交标记在场） |

补充：

* `_final_check_state` 在 A 中恢复为原值（修前为 `checked_on_current_parameters`）。
* 原先不存在的产物，失败后**不留下**本次创建的版本。
* 成功事务：四份产物互相一致（清单 `status` == `last.pt.run_status` ==
  `manifest.metadata.check_state` == `last.pt.check_state`），且不留下标记。
* 失败诊断写在**独立**文件 `refinement_failures.json`，不冒充成功检查历史。
* 注入都让**真实写盘先发生**再失败，不是把保存函数整个换成"一进来就抛错"。

---

## 5. 波形左端/右端覆盖组合

`times = [0, 0.5, 1, 1.5, 2, 2.5, 3]`；四类曲线分别代表四种覆盖；16 个组合逐格核对：

| 模型 | 参考 | 模型覆盖 | 参考覆盖 | 被判分的正相误差 | 模型已覆盖面积 |
|---|---|---|---|---|---|
| full | full | full | full | `positive_duration_relative`, `positive_impulse_relative` | 3.375 |
| full | start_missing | full | start_missing | — | 3.375 |
| full | end_truncated | full | end_truncated | — | 3.375 |
| full | start_missing_and_end_truncated | full | 同左 | — | 3.375 |
| start_missing | full | start_missing | full | — | 4.625 |
| start_missing | start_missing | start_missing | start_missing | — | 4.625 |
| start_missing | end_truncated | start_missing | end_truncated | — | 4.625 |
| start_missing | 两端都缺 | start_missing | 两端都缺 | — | 4.625 |
| end_truncated | full | end_truncated | full | — | 6.25 |
| end_truncated | start_missing | end_truncated | start_missing | — | 6.25 |
| end_truncated | end_truncated | end_truncated | end_truncated | — | 6.25 |
| end_truncated | 两端都缺 | end_truncated | 两端都缺 | — | 6.25 |
| 两端都缺 | full | 两端都缺 | full | — | 6.5 |
| 两端都缺 | start_missing | 两端都缺 | start_missing | — | 6.5 |
| 两端都缺 | end_truncated | 两端都缺 | end_truncated | — | 6.5 |
| 两端都缺 | 两端都缺 | 两端都缺 | 两端都缺 | — | 6.5 |

**只有 `full / full` 一格判分**。其余 15 格都不判分，并在 `notes` 里分别写出是
左端缺失、右端截断、还是两者都有——不统一写成"被终止时间截断"。

**解析积分值保持不变**：

| 曲线 | 解析面积 | 实测 | 解析时长 | 实测 | 覆盖分类 |
|---|---|---|---|---|---|
| `[2, 2, -2, -2]` | 2.5 | **2.5** | 1.5 | **1.5** | `start_missing` |
| `[0, 2, 0, -1]`，阈值 0 | 2 | **2.0** | 2 | **2.0** | `full` |

---

## 6. 远端新克隆复验

```bash
git clone https://github.com/whiteswolfd-maker/MyNewProject.git verify_clone
cd verify_clone

git rev-parse HEAD
# 8d7fda6303ece357bd9819e450b15cdab08522ef

git ls-files --error-unmatch scripts/bounded_snake_check.py
# scripts/bounded_snake_check.py          <- 存在且被跟踪
git ls-tree -r --name-only HEAD scripts
# scripts/bounded_snake_check.py
git status --short
# （空）

PYTHONPATH=src python -m pytest -q -rs
# 520 passed, 47 warnings in 283.46s      <- 0 skipped（本机有真实数据）

PYTHONPATH=src python scripts/bounded_snake_check.py <仓库外目录>
# exit 0
```

联检报告（读自 `<仓库外目录>/snake_check.json`）：

```
version  : 4
ok       : True
checks   : 27 | passed: 27
duration : 4.35 s
output   : C:\Users\work\AppData\Local\Temp\verify_out
```

**输出目录在仓库之外**，脚本无盘符硬编码。

### 逐项联检结果（27/27）

| # | 检查 | 结果 |
|---|---|---|
| 1 | `continuous_run_reaches_the_end_of_the_schedule` | PASS |
| 2 | `every_shortened_stage_boundary_is_crossed` | PASS |
| 3 | `resume_at_a_stage_boundary_matches_a_continuous_run` | PASS |
| 4 | `a_resumed_run_reports_the_measured_loss_of_its_last_update` | PASS |
| 5 | `a_failed_stage_stays_blocked_after_a_resume` | PASS |
| 6 | `a_completed_refinement_reports_the_same_identity_everywhere` | PASS |
| 7 | `a_refinement_rechecks_the_parameters_it_changed` | PASS |
| 8 | `the_refinement_records_what_it_did` | PASS |
| 9 | `a_refinement_after_a_truncated_run_is_still_diagnostic` | PASS |
| 10 | `an_injected_refinement_failure_leaves_the_state_alone` | PASS |
| 11 | `the_impulse_counterexample_is_2_point_5` | PASS |
| 12 | `a_contact_sweeping_between_frames_is_not_interpolated_across` | PASS |
| 13 | `outside_the_prediction_grid_or_time_coverage_says_why` | PASS |
| 14 | `compare_cli_runs_end_to_end_and_writes_a_readable_report` | PASS |
| 15 | `a_refinement_failure_after_the_update_leaves_no_half_changed_state` | PASS |
| 16 | `an_explicit_diagnostic_identity_survives_a_reload` | PASS |
| 17 | `best_pt_carries_the_status_of_the_run_that_selected_it` | PASS |
| 18 | `a_changed_field_is_refused_by_the_data_identity` | PASS |
| 19 | `a_zero_threshold_triangle_reports_area_2_and_duration_2` | PASS |
| 20 | `an_exact_frame_ignores_a_missing_neighbour_frame` | PASS |
| 21 | `an_incomplete_phase_removes_the_phase_and_arrival_errors` | PASS |
| 22 | `a_mid_run_best_is_published_with_its_own_parameters` | PASS | **本轮 §1** |
| 23 | `best_pt_can_be_resumed` | PASS | **本轮 §1** |
| 24 | `a_failure_after_earlier_artifacts_were_published_rolls_all_of_them_back` | PASS | **本轮 §2** |
| 25 | `a_memory_restore_failure_still_restores_the_disk` | PASS | **本轮 §3** |
| 26 | `an_unrestorable_publish_cannot_be_loaded` | PASS | **本轮 §3** |
| 27 | `a_missing_left_edge_is_not_a_complete_positive_phase` | PASS | **本轮 §4** |

联检在测试范围内把阶段判据换成空集，**不可能**证明任何物理判据通过；
**真实数据相关的检查不在联检里运行**，不与合成联检合并成"全部通过"。

---

## 7. 交付清单

**修改提交**：`8d7fda6303ece357bd9819e450b15cdab08522ef`（`main`，已推送）

`sha256` 前 16 位（**换行符统一为 LF 后**计算；仓库 `.gitattributes` 是
`* text=auto`，工作区为 CRLF）：

| 文件 | 字节（LF） | sha256[:16] |
|---|---:|---|
| `src/spherical_blast_pinn/training/trainer.py` | 124129 | `a08c57a8ea1ecd00` |
| `src/spherical_blast_pinn/evaluation/waveform.py` | 19968 | `9216ecf6aa645f06` |
| `src/spherical_blast_pinn/cli/compare.py` | 54828 | `a54fa8746798ac1b` |
| `src/spherical_blast_pinn/data/extracted_case.py` | 81407 | `2539ded5ea5c58c0` |
| `scripts/bounded_snake_check.py` | 49749 | `1cd8005b47f24fa7` |
| `tests/test_best_candidate_and_transaction.py` | 24312 | `1011427756cea1c6` |
| `tests/test_waveform_impulse.py` | 20599 | `6b2151411c999c5e` |
| `tests/test_refine_and_checkpoint.py` | 42793 | `154182feb11ff895` |
| `tests/test_compare_cli.py` | 22064 | `8ccca85d4be03198` |
| `AGENTS.md` | 32425 | `0a35dde92feeadc9` |

**测试**：520 passed / 0 skipped / 0 failed（本机与远端新克隆各跑一次）。
本机 `pytest -q -rs` 在**新克隆目录**里执行，与复核命令一致。

---

## 8. 仍未完成的事项

**与本轮软件修复无关，独立列出，不写成"全部完成"：**

1. **中断的 L-BFGS 无法继续**——`Trainer.resume_refinement()` 明确拒绝（本机无 git
   仓库时确认的既定功能限制，本轮**未实现**，也不打算把"重新开始"叫"继续"）。
2. **检查点兼容性**——本轮结构再次改变（`pending_best_candidate` / `best_dirty`），
   版本 6 → **7**；版本 6 及更早**一律拒绝**，**没有迁移路径**，也不把缺字段默认为通过。
3. **`_run_origin` 一旦为诊断不可清除**——刻意的（身份跟着参数走），但**没有**提供
   "显式开一条新血统"的入口；要正式结果目前得新建 trainer 而不加载检查点。
4. **真实数据资源**——50 / 100 / 80 mm 读到但契约未就绪（缺 companion、
   `profile_layout.declared=False`）；**80 mm 缺配套 deck**；500 mm **未提取**。
   `declared` 没有被手工改成真。
5. **研究性验收（三项并列）**——离散能量残差来源未定论（**可复现**，随分辨率变化）、
   相容离散未实施 / 网格收敛未通过、`physics_basis_status` 未核实。三者互补推出。
6. **研究性验收闸门**——`ReferenceNotAccepted` 保留；软件回归通过**不能**补齐缺失的
   验收证据。
7. **`test_real_data_chain.py` 等依赖真实数据的测试**——本机 **0 skipped**（有数据）；
   复核环境 6 skipped 是**环境差异**，需要有对应数据的环境单独给出同一提交下的证据。

---

## 9. 一句话总结

本轮五项**逐条在本机实际版本上复现、修复、验证**，并已**推送**；
远端**全新 `git clone`** 后在其目录内复跑：`pytest -q -rs` **520 passed / 0 skipped**，
联检 v4 **27/27、exit 0**，`scripts/bounded_snake_check.py` 在 Git 中**存在且被跟踪**。
第 8 节那七项照旧开着，不能与"软件修好"混为一谈。
