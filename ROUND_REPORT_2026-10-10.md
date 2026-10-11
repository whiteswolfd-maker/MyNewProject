# 修改提示词执行报告（2026-10-10）

## 0. 环境与交付清单

| 项 | 值 |
|---|---|
| 复核基线 | `e855e31756390eb8e23b2d4246e46269ddb2f94d`（复核方提供的 GitHub SHA） |
| 本机 HEAD | **无本地 git 仓库**（`git rev-parse` 报 "not a git repository"）。交付物就是当前工作区，无法给出 SHA。 |
| 工作区差异 | 无版本控制，无法列出 diff。改动文件见下表。 |
| Python | 3.10.8 |
| torch | 2.5.1+cu121 |
| numpy | 1.26.4 |
| 测试命令 | `PYTHONPATH=src python -m pytest -q` |
| 测试结果 | **370 passed**（本轮开始前 321 passed；上一轮报告时 300） |

**先说明一件事**：上一轮我只回应了可见的复核正文，**没有执行随附的修改提示词**。
本轮提示词已从会话记录中取回并逐节执行。这是上一轮报告"全部完成"与复核结论不符的
直接原因。

改动文件：

| 文件 | 改动 |
|---|---|
| `training/trainer.py` | 四档终态、参数指纹、事务性恢复、非有限更新拦截、精修语义、检查点版本 4 |
| `cli/train.py` | 退出码分档、`--max-steps` 的 `None`/`0` 区分、`--diagnostic` |
| `cli/compare.py` | 逐字段有效性、全量计数、非有限预测不再静默消失 |
| `data/extracted_case.py` | 原始结构先校验后规范化、材料码/掩码 dtype、统一校验函数；**布局声明**（4.1 节） |
| `data/contract.py` | 结构化来源、对象级结构检查、档位一致性；对象检查改为交给 `validate_case_object`（4.2 节） |
| `data/provenance.py` | **新增**：来源记录模式与校验 |
| `sampling/sampler.py` | 采样器 RNG 状态校验、窗口回滚 |
| `cli/extract_profiles.py` | 写出结构化来源记录与输入标识；写出 `profile_layout` 声明 |
| `AGENTS.md` | 新增一节说明运行终态与检查点约定 |
| `scripts/bounded_snake_check.py` | **新增**：有界合成联检 |
| 测试 | 新增 `tests/test_run_status_and_guards.py`、`tests/test_profile_layout.py`；扩充 5 个已有文件 |

---

## 1. 验收失败却返回成功 —— 已修，验证通过

**修前反例（实际复现）**：单阶段日程、预算 2 步、最终判据必然不过 →
`TrainingResult.ok` 为 `True`，`result.as_dict()["final_stage_verdict"]["passed"]`
为 `False`，CLI 返回 0。两者并存。

**修后**：`TrainingResult.ok` / `formally_accepted` 只在 `status == "completed"`
时为真；四档终态 `completed` / `criteria_failed` / `incomplete` / `diagnostic`
各自有独立的退出码（0 / 2 / 1 / 0）。

**真实调用路径**：`trainer.run()` 控制流 → 结果对象 → `run_manifest.json` →
`train_checkpoint`。收尾结论在**写最终检查点之前**算出，`last.pt` 里带
`run_status` 与 `final_stage_verdict`。

| 测试 | 状态 |
|---|---|
| `test_a_full_run_with_passing_criteria_is_completed` | 通过 |
| `test_a_full_run_with_failing_criteria_is_not_a_success` | 通过 |
| `test_a_truncated_run_is_incomplete_and_makes_no_claim` | 通过 |
| `test_a_run_without_any_check_cannot_be_completed` | 通过 |
| `test_an_explicit_diagnostic_run_claims_nothing` | 通过 |
| `test_the_final_loss_is_labelled_with_the_step_it_came_from` | 通过 |
| `test_the_cli_exit_codes_distinguish_the_four_end_states` | 通过 |

`final_loss` 现在带 `final_loss_step`，取自**最后一次更新实测**，不是最后一条日志。

---

## 2. 评估规则与逐字段有效性脱节 —— 已修，验证通过

**修前反例（实际复现）**：联合掩码为假、密度自身有效 → 密度标签整片丢失；
三个候选点中一个预测为 NaN → 该点从分子分母同时消失，仍报零误差、`complete=True`。

**修后**：每个字段用自己的有效性掩码；计数按介质与字段分开记录且**互斥闭合**
（候选 = 缺测 + 域外 + 错配 + 保护区 + 比较）；`model_non_finite` 单独计数并进入
`unavailable_fields`；无可比点时该字段为 `nan` 且 `available=False`，不是 0。

| 测试 | 状态 |
|---|---|
| `test_compare_uses_per_field_validity` | 通过 |
| `test_a_non_finite_prediction_is_counted_not_silently_dropped` | 通过 |
| `test_every_dropped_point_is_accounted_for` | 通过 |
| `test_a_field_with_no_comparable_point_is_unavailable_not_zero_error` | 通过 |

修这条时发现并修掉了一个**自己引入的**计数错误：`reference_missing` 一度把域外、
错配、保护区一起算进去（重复计数）。`test_every_dropped_point_is_accounted_for`
就是为此加的。

---

## 3. 恢复的事务性 —— 已修，验证通过

**修前反例（实际复现，三个）**：非法采样器 RNG 内容在模型/优化器/步数/全局 RNG
已改掉之后才报错；缺 `window_fraction` 时 `float(None)` 在改对象之后才抛；
`sampler_rng={}` 被接受且**没有恢复任何采样序列**。

**修后**：`load()` 分两段。第一段只读检查——格式、版本、`hat T_*`、步数类型、
模型键与形状、优化器、平衡器、采样器 RNG（版本 + 每算例键 + bit-generator 结构）、
`stage` 块的四个键必须存在、窗口与阶段的一致性（**阻塞与否决定用哪条等式**，不再
"接受任何一个记录过的阶段窗口"）。第二段提交，**全过程快照**，任何异常完整回滚，
覆盖模型、优化器、步数、历史、best、阶段/窗口/待决/阻塞、平衡器、各采样器（含窗口
与 RNG）与全局 RNG。

检查点版本升到 **4**：缺 `pending`/`blocked` 直接拒绝，不默认通过。

| 测试 | 状态 |
|---|---|
| `test_a_broken_checkpoint_fails_without_touching_the_target`（7 个参数化反例） | 通过 |
| `test_a_failure_inside_the_commit_rolls_everything_back` | 通过 |
| `test_a_checkpoint_saved_after_a_failed_stage_check_can_be_resumed` | 通过 |
| `test_resuming_across_a_stage_boundary_matches_a_continuous_run` | 通过 |

未支持并**明确拒绝**的：精修中途恢复（L-BFGS 优化器状态不进检查点），精修历史里
记 `resume_supported: false`。

---

## 4. 统一结构校验接到原始输入与契约两端 —— 已修，验证通过

**修前反例（实际复现，五个）**：`times=[[1,2]]` 与 `radius=[[1,2,3]]` 读取成功且
`ready=True`；二维状态码被展平；`material` 全 999 仍通过；联合 `valid` 全 NaN 被读成
全 True；读取后把 `velocity` 改成错误形状再调契约仍 `ready=True`。

**根因**：读取器先 `reshape(-1)` / `dtype=bool`，再把结果交给一个检查原始维度的
函数——检查看到的已经是规范化后的数组。

**修后**：`validate_case_arrays()` 一处核完所有数组与两条坐标轴；两条读取路径共用，
且**没有提前 return 绕过**；规范化只在校验之后。`material` 按 `MATERIAL_CODES` 检查
（未知码是格式错误，不是"另一个区域"）；`material`/`valid` 的 dtype 先查再转。
`evaluate_case_contract` 对**任何**对象执行同一套结构检查。

| 测试 | 状态 |
|---|---|
| `test_a_two_dimensional_axis_is_refused_not_flattened`（2 个参数） | 通过 |
| `test_an_unknown_material_code_is_refused` | 通过 |
| `test_an_all_nan_joint_valid_mask_is_refused` | 通过 |
| `test_the_contract_does_not_trust_that_the_case_was_read` | 通过 |
| `test_both_reading_paths_apply_the_same_rule` | 通过 |

单帧一维场格式：**第一版报告在这里写错了两处**，2026-10-11 已按版本化规则重做，
见下面新增的第 4.1 节。原话是"未支持，也没有用『任意展平』去猜布局"——两句都不对：
那种格式**读得进去**，而读取器**正是在猜布局**。

---

## 4.1 场量布局：从"猜"改成"声明"（2026-10-11 补做）

**这一节修正的是本报告自己的一句话。** 第 4 节原先写"单帧一维场格式：未支持，
也没有用『任意展平』去猜布局"。实际上：

**修前反例（实际复现，2026-10-11）**：

| 探针 | 结果 |
|---|---|
| 一份只有 1 帧、``rho``/``u``/``P`` 写成 ``(n_x,)`` 的导出 | **读取成功**，``density.shape`` 变成 ``(1, n_x)`` |
| 同一对象上找记录布局的字段 | ``source`` 里**没有任何** layout/ndim 相关键 |
| 给 ``validate_case_arrays`` 打桩，看它收到什么 | 每个剖面 ``ndim == 2``——校验看到的已经是读取器造出来的形状 |
| 直接调 ``validate_case_arrays`` 传一维剖面 | **接受**，归一成 ``(1, n_x)`` |
| 一维剖面 + 两帧时间轴 | 报 ``density has shape (1, 4), expected (2, 4)``——但文件里的 ``rho`` 是 ``(4,)`` |

**根因**：``np.atleast_2d`` 在读取器里出现 **10 处**（两处剖面各 ×3、``material``/
``valid`` 各 ×1、``e`` ×1、逐字段掩码 ×1），都在 ``validate_case_arrays`` **之前**
执行。于是：

* 布局是**猜**出来的（从形状推），文件里没有声明；
* 读完之后对象上看不出它来自一维场——``(1, n)`` 与"只有一帧的二维导出"逐字相同；
* 校验函数里那句 ``ndim == 1 and size == radius.size and times.size == 1`` 的顺手
  展平，在两条真实读取路径上**从来没有被走到过**：它是一句死代码，掩盖了真正
  生效的机制是上游的 ``atleast_2d``；
* "一维场配多帧时间轴"这种真正的格式错误被报成一个文件里根本不存在的形状，
  把读取器的猜测说成文件里的内容。

**修后**（提示词 §4 要求的版本化规则）：

* ``metadata.json`` 新增 ``profile_layout = {"version": 1, "layout": ...}``，布局是
  **封闭枚举**：``frames_by_radius``（逐帧二维，缺省，与既有导出一致）/
  ``single_frame_1d``（单帧一维）；
* ``declared_profile_layout()`` 核结构、核枚举、核版本（``True`` 是 ``int`` 的子类，
  显式排除，否则 ``"version": true`` 会被当成版本 1）；不认识的布局、不支持的版本、
  多余的键一律**拒绝**，不是"退回默认值"；
* 10 处 ``np.atleast_2d`` 全部去掉，改成 ``normalise_to_layout()``：按**声明**核对原始
  形状，只在核对通过之后归一；
* 规范形状**只有一种** ``(n_t, n_x)``——声明描述的是文件，不是对象。``single_frame_1d``
  读进来也是 ``(1, n_x)``，这正是必须记 ``source["profile_layout"]`` 的原因；
* 记录 ``{"version", "layout", "declared"}``。``declared`` 区分"文件里写过"和
  "读取器推断的缺省"——既有导出没有这一项，它们记 ``False``；
* 提取器同步写出声明（``frames_by_radius``）。
* ``single_frame_1d`` 配多帧时间轴是**文件自相矛盾**，在看任何场量之前就报出来。

| 测试 | 状态 |
|---|---|
| `test_a_one_dimensional_profile_without_a_declaration_is_refused` | 通过 |
| `test_a_declared_single_frame_one_dimensional_export_reads_and_says_so` | 通过 |
| `test_the_default_layout_is_recorded_as_inferred_not_declared` | 通过 |
| `test_a_declaration_that_contradicts_the_file_is_refused` | 通过 |
| `test_the_single_frame_layout_says_nothing_about_several_frames` | 通过 |
| `test_an_unrecognised_declaration_is_refused`（7 个参数） | 通过 |
| `test_the_error_does_not_blame_a_shape_the_reader_invented` | 通过 |
| `test_the_companion_masks_follow_the_same_layout_rule` | 通过 |

**真实数据未受影响（实测）**：三份导出都是二维，仍按缺省档读出，``declared=False``
（它们由旧提取器产出，没有这一项）：

```
50mm : density(195, 2500) layout={'version': 1, 'layout': 'frames_by_radius', 'declared': False}
100mm: density( 30, 2500) layout={'version': 1, 'layout': 'frames_by_radius', 'declared': False}
80mm : density(483, 2500) layout={'version': 1, 'layout': 'frames_by_radius', 'declared': False}
```

**界限**：``declared=False`` 是**推断**，不等于这三份文件声明过。要让它们带上声明，
需要重提——而那件事仍然卡在资源上（见第 10 节）。这里没有把它写成"已声明"。

---

## 4.2 复核"读取路径仍先 reshape(-1)" —— 机制不成立，但结论对了一半（2026-10-11）

复核说：两条读取路径仍先 ``times = ...reshape(-1)``、``radius = ...reshape(-1)``，
再调用 ``validate_case_arrays()``，并给出六条仍可复现的反例。

**先说机制**：``reshape(-1)`` 在这个仓库里只出现在注释和字符串里，没有任何一处是
活代码。实测（同一台机器、同一份代码）：

```
grep -rn "np.atleast_2d(" src/     -> 无
grep -rn "reshape(-1)"  src/       -> 命中全部在注释/说明文字，或与本议题无关的模块
```

**六条反例逐条实测（2026-10-11，本机当前工作区）**：

| 复核的输入 | 复核的说法 | 实测 |
|---|---|---|
| 时间轴 ``[[1, 2]]`` | 展平后读取成功，``ready=True`` | **拒绝**：``the time axis must be 1-D, got shape (1, 6)`` |
| 半径轴 ``[[1, 2, 3]]`` | 展平后读取成功，``ready=True`` | **拒绝**：``the radial axis must be 1-D, got shape (1, 3)`` |
| 二维状态码数组 | 校验内部展平后接受 | **拒绝**：``state code 'contact_state' has shape (1, 2); it must be 1-D, one code per frame`` |
| 材料编码全部 999 | 读取成功，``ready=True`` | **拒绝**：``'material' contains [999], which this reader does not define`` |
| 联合 valid 全 NaN | 转成全 True，``ready=True`` | **拒绝**：dtype 是 float64，不是 bool |
| 读取后改 velocity 形状再调契约 | ``ready=True`` | 基线 ``ready=True`` → 改后 ``ready=False``，且 ``structural_problems`` 报出形状不符 |

六条里没有一条复现。上表两条"时间轴/半径轴"我第一遍还写错过探针（把 ``[[1,2]]``
构造成了 ``(2,1)``），第二遍按字面形状重做，结论不变。另外这个工作区**只有一份**
``extracted_case.py``（全盘搜索确认），不存在"改了一份、复核看的是另一份"。最可能的
原因是复核看的是 GitHub 上的 ``e855e317``，**早于**这两轮修复——本机没有 git，我无法
比对，所以这一条只是推测，不作为结论。

**但复核的结论对了一半，而且那一半是真的。** 顺着"读取校验和契约校验还没有完整
衔接"这句去查覆盖面，发现契约的对象级检查把要查的数组**又列了一遍，只列了七个名字**，
与读取端没有共用清单：

| 读取后改坏的对象 | 修前契约的结论 |
|---|---|
| ``velocity`` / ``density`` / ``pressure`` / ``times`` / ``radius`` | 拒绝 |
| ``specific_energy`` 形状不对 | **放行（``ready=True``）** |
| ``material`` 全改成 999 | **放行** |
| ``material`` 改成 float | **放行** |
| ``front_supervision_eligibility`` 里的数组形状不对 | **放行** |
| ``valid`` / ``contact_valid`` / 逐字段掩码 | 判了不就绪，但**结构检查一条问题都没报**——是别的原因顺带拦住的 |

最后一行和前三行同样严重：**"碰巧拦住了"不是覆盖率**。同一类错误，有的拒绝、有的放行、
有的靠别的原因顺带拒绝——那说明覆盖面是个偶然。

**修后**：新增 ``validate_case_object()``，对对象上**每一个**数组做与读取端同一套检查
（``times``/``radius``/四个剖面/两条前沿/两条资格掩码/``material``/``valid``/逐字段
有效性/前沿资格数组），``contract._structural_problems`` 直接调它——**对象上有什么就
查什么，清单不再由契约维护**。``material`` 的取值域与 dtype、``valid`` 与逐字段掩码的
布尔 dtype 在这里**也**核一遍：读取时核过不等于对象上没被改过。

| 测试 | 状态 |
|---|---|
| `test_the_contract_rechecks_every_array_the_object_carries`（15 个参数） | 通过 |
| `test_a_two_dimensional_state_code_is_refused_end_to_end` | 通过 |
| `test_a_two_dimensional_material_table_is_refused_end_to_end` | 通过 |

这组测试**只断言 ``ready`` 不够**：每一条都要求 ``structural_problems`` 非空，否则
"碰巧被别的原因拦下"会冒充覆盖率。另有一处测试替身（``_contract_case``）因为少了
``specific_energy`` / ``material`` / ``contact_valid`` 而被新检查拦下——**修的是替身**
（补齐 ``ExtractedCase`` 的必填字段），不是放宽检查。

---

## 5. 来源记录的结构、一致性与版本 —— 已修，验证通过

**修前反例（实际复现）**：`material_partition_source` 为 `123` / `{"unrelated": "value"}`
/ `"heuristic — not from volume fraction"` 时，只要布尔标志为真就 `ready=True`。

**根因**：判据是 `str(record or "").strip()`——只是"必须有一个非空值"。

**修后**：`data/provenance.py` 定义模式（方法封闭枚举、生产端与版本、输入字段、
输入标识、明细），契约检查结构、类型、必填字段、支持版本，以及**与布尔声明的一致性**。
生产端（`cli/extract_profiles.py`）与读取端一起迁移，输入标识是 deck 内容与 d3plot
家族清单的哈希；输入不可读时写 `unavailable: <原因>`，**不编指纹**。
`arrays` / `legacy_state_code` / `none` 三档保留，并检查档位与实际数组一致。

旧导出（字符串来源）**按规则拒绝**，没有为了维持旧数据 ready 而削弱规则——
50/100 mm 已用新提取器重新导出。

| 测试 | 状态 |
|---|---|
| `test_an_unstructured_or_inconsistent_source_is_refused`（6 个参数） | 通过 |
| `test_a_structured_source_that_agrees_with_the_flag_is_accepted` | 通过 |
| `test_an_input_identifier_must_not_be_faked_when_the_input_is_unavailable` | 通过 |
| `test_the_reader_does_not_stamp_the_partition_flag_itself` | 通过 |
| `test_a_legacy_three_state_export_is_diagnosable_but_not_ready` | 通过 |

**界限**：这份校验只证明"记录符合约定"，**不能**独立证明物理算法正确。它不说明体积
分数读对了列，也不说明提取器没有 bug。

---

## 6. 零预算与参数边界 —— 已修，验证通过

**修前**：`int(max_updates) if max_updates else None`——`0` 是假值，于是"跑零步"被
解释成"跑完整日程"。

**修后**：`None` 与 `0` 分开；`0` 表示**零次更新**（运行立刻结束并判 `incomplete`）；
负数在**构造上下文之前**拒绝（退出码 7）；显示预算与传入 `run()` 的预算相同。
恢复时预算是**累计上限**（`run(max_updates=N)` 表示"累计到 N 步"）。

| 测试 | 状态 |
|---|---|
| `test_zero_updates_means_zero_updates_not_the_whole_schedule` | 通过 |
| `test_a_negative_budget_is_refused_before_the_expensive_context` | 通过 |

---

## 7. 精修后不能用旧模型的检查结果 —— 已修，验证通过

**修前**：精修改参数但不改 `step`；收尾检查按"同一步数已检查"跳过；精修调
`total_loss` 时**没传 `weights`**，静默从自适应权重退回基础权重。

**修后**：检查结论绑定**参数指纹**（参数张量内容哈希），不只绑步数；精修使用并记录
当前自适应权重；精修过的运行一律记 `diagnostic`（它中途换过目标函数）；阶段被阻塞
时**拒绝精修**；精修历史记录请求迭代数、**实际 closure 次数**、权重、`resume_supported`。

| 测试 | 状态 |
|---|---|
| `test_refinement_invalidates_the_previous_check` | 通过 |
| `test_refinement_records_the_weights_it_used` | 通过 |
| `test_refinement_is_refused_while_a_stage_is_blocked` | 通过 |

**未完成**：精修的完整恢复（含 L-BFGS 优化器状态）**不支持**，且明确拒绝而非假装。

---

## 8. 非有限损失/梯度的更新前检查 —— 已修，验证通过

**修前**：`_update()` 里没有任何有限性检查。一次 NaN 梯度会把模型污染成 NaN，
训练"照常"跑完，检查点覆盖掉上一个已知有效的快照。

**修后**：损失非有限在**反传之前**拦下；梯度非有限在 `optimizer.step()` **之前**
拦下，报出步骤与参数名，参数不被改动；不用 `nan_to_num`、不裁剪、不跳过该组。
正式入口与保护行为一致，没有通用绕过开关。

| 测试 | 状态 |
|---|---|
| `test_a_non_finite_gradient_blocks_the_update` | 通过 |
| `test_the_divergence_stops_the_run_instead_of_being_absorbed` | 通过 |

---

## 9. 测试、文档与短程联检

**逐项矩阵**（本提示词第 0 节要求的格式）

| 编号 | 实际消费函数 | 反例 | 修复 | 回归测试 | 执行状态 |
|---|---|---|---|---|---|
| §1 | `Trainer.run` / `TrainingResult.ok` / `train.main` | 判据失败仍 `ok=True`、退出 0 | 四档终态 + 退出码 | 7 条 | 验证通过 |
| §2 | `cli.compare.compare_fields` | 联合掩码丢字段；NaN 预测消失 | 逐字段 + 全量计数 | 4 条 | 验证通过 |
| §3 | `Trainer.load` | 非法 RNG / 缺窗口 / 空 RNG 后对象已改 | 两段式 + 快照回滚 | 10 条 | 验证通过 |
| §4 | `load_extracted_case` / `evaluate_case_contract` | 二维轴、999 材料、NaN 掩码、改后对象 | 先校验后规范化 | 9 条 | 验证通过 |
| §4.1 | `declared_profile_layout` / `normalise_to_layout` / `validate_case_arrays` | 一维场无声明即读入、形状无记录、报错引用读取器造的形状 | 版本化布局声明（10 处 `atleast_2d` 去掉） | 14 条（其中 7 个参数） | 验证通过 |
| §4.2 | `validate_case_object` / `contract._structural_problems` | 契约只查七个数组：`specific_energy`/`material`/`valid`/掩码/资格数组改坏仍 `ready=True` | 对象上有什么就查什么 | 17 条（其中 15 个参数） | 验证通过 |
| §5 | `evaluate_case_contract` | 123 / 任意字典 / 缺字段 / 版本 / 矛盾 | 结构化模式 | 10 条 | 验证通过 |
| §6 | `cli.train` | `--max-steps 0` → 完整日程 | `None`/`0` 分离 + 前置拒绝 | 2 条 | 验证通过 |
| §7 | `Trainer.refine` | 精修沿用旧结论；权重静默回退 | 参数指纹 + 权重记录 | 3 条 | 验证通过（完整恢复**未完成**） |
| §8 | `Trainer._update` | 无任何有限性检查 | 更新前拦截 | 2 条 | 验证通过 |

没有用修改预期值、放宽容差、跳过检查或 `xfail` 消除任何失败。测试替身只在
`tests/` 与 `scripts/bounded_snake_check.py` 里，**不进正式入口**。

**有界合成联检**（`scripts/bounded_snake_check.py`，独立输出目录 `F:/2/snake_check`）

```
budget                     = 18 updates（写死的常量）
实际更新数                 = 6（三阶段各 2 步，完整日程跑满）
终态                       = completed
进入的阶段                 = initialise / strengthen_pde / joint
断点                       = 第 2 步（阶段边界，窗口 0.65）
续跑 vs 连续跑权重逐位相同 = true
续跑 vs 连续跑采样器 RNG   = 相同
精修                       = 请求 3 次迭代，实际 3 次 closure
精修终态                   = diagnostic
被改坏的算例是否拒绝       = true
ok                         = true
```

**正式入口为什么仍被拦截**：`build_reference` → `assert_reference_accepted` 在
任何训练开始之前抛出 `ReferenceNotAccepted`（缺 `grid_convergence`、
`timestep_sensitivity`、`scale_mapping`，且物理适用性未核实）。这是**物理闸门**，
与本提示词的软件项无关；软件项已全部闭合，但这**不**改变物理闸门的状态。

已同步 `AGENTS.md`（新增一节说明运行终态、检查点版本与来源记录约定）。README 与
AGENTS 中**没有**发现已撤回的旧数字或因果归因需要修正——上一轮的因果链表述写在
回合报告里，已在 2026-10-10 的更正说明中改掉。

---

## 10. 仍未完成

**软件**

* 精修的**完整恢复**（L-BFGS 优化器状态）不支持；已明确拒绝，不是"看起来支持"。
* 单帧一维场格式的版本化规则已做（第 4.1 节）。**仍差一步**：三份真数据是由旧提取器
  产出的，文件里没有 ``profile_layout`` 这一项，所以读出来是 ``declared=False``
  （推断的缺省，不是声明）。要补上得重提，重提卡在资源上。

**一条更正**：上一版把 `_legacy_v1_backup/` 列成"未清理"，方向是错的。它不是遗漏，
是**约定保留**：`CLAUDE.md` 写着"第一版顺序耦合实现整体保留在 `_legacy_v1_backup/`，
仅供追溯；不要恢复它，也不要在新路径里引用它"，`AGENTS.md`、`README.md` 同义，
`training/balancing.py` 还把它当作权重方案出处的引文。删掉它会同时破坏这四处可追溯性，
所以**不该**清理。

**资源 / 数据**

* 80 mm 缺配套 deck（d3plot 在 `C:/Users/work/Documents/GitHub/2/sim_data/`）；
* 50 / 100 mm 只有**代表状态**（6 / 3 帧）；500 mm 未提取；
* 全量重提、§16.2 compare 闭环、§16.3 时程诊断输出未做。

**研究性（三项，并列，未核实）**

* 离散能量残差 0.41 % 的来源未定论（只有否定结果）；
* 相容离散未实施、网格收敛未通过；
* `physics_basis_status` 未核实。

这三项**不是**彼此推出的结论，也**不能**用来说明上面任何一条软件问题。
