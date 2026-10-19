# 复核回应（第三项）：未发布最佳候选的完整性与参数指纹一致性

**只改了第三项。** 发布标记、冻结爆速、算例关联校验三处的代码一个字没动。

## 0. 基线确认 —— 有一件我做不到的事

复核给的仓库是 `whiteswolfd-maker/deskbook`，基线提交
`699d7af11ddc18de6e3efa9780e3052fc1addf80`。

**本机这个工作树没有 `.git`**（`git rev-parse HEAD` → `fatal: not a git repository`），
所以**我无法确认当前提交号**，也无法证明"基线之后已有的正确改动都保留了"。
`checkpoint.py` 里已经有 `require_case_scales_association` / 发布标记契约
（`PUBLISH_ARTIFACTS_KEY`、`owner` 校验）等本项之前的工作，说明这棵树**确实晚于**
我上一轮看到的版本；但我只能用**行为**说明它没被本项破坏（见 §4），
不能用提交号说明。这一条请你或复核方用有 git 的环境核对。

**执行环境**：Python 3.12.12 / torch 2.9.1+cu130 / NumPy 2.3.5（CPU）。
复核环境若是 Python 3.12 + torch 2.14，版本不同，但下面每条都在本机实跑过。
仓库按 LF 计算哈希。

## 1. 修前复现（走真实控制流，不手工拼坏检查点）

脚本：`%TEMP%\scan_19246a0\repro_candidate_item3.py`（在仓库外，不入版本库）。
做法：真实 `verify()` 登记候选 → 真实 `_save()` 写中间检查点 → 保留**未修改的
合法对照** → 对独立副本分别施加三种变异，并**断言变异确实生效**（比对字节 / 键集合）。

| 变异 | 共享校验 | `Trainer.load()` | 错误信息 |
|---|---|---|---|
| A 候选 `model` 改成空字典 | **通过（未拦下）** | **通过（未拦下）** | — |
| B 改一个候选参数、指纹不动 | **通过（未拦下）** | **通过（未拦下）** | — |
| C 删除候选 `optimizer` | **通过（未拦下）** | **通过（未拦下）** | — |

**修前结论：3/3 个坏候选被接受**，与复核报告逐条一致。

## 2. 原因（读代码确认）

`require_best_candidate_consistency()` 当时只查了四件事：候选是 `Mapping`、步号与
`best_check` 相同、`model` 是 `Mapping`、`parameter_fingerprint` 是非空字符串。
它**没有**：

* 拿候选的 `model` 与当前模型的结构对（键/类型/形状/有限性）；
* 从候选里的**实际参数**重算指纹再与记录比；
* 检查候选快照的其他必需状态（优化器、阶段、采样器与全局 RNG、平衡器、历史、来源）；
* 核对候选内部的 `best_check` 与顶层最佳记录是否描述同一次选择。

## 3. 修后复现（同一脚本）

| 变异 | 共享校验 | `Trainer.load()` | 拒绝理由 |
|---|---|---|---|
| 合法对照（未修改） | **接受** | **接受** | — |
| A 空候选 `model` | **拒绝** | **拒绝** | `model state is empty. An empty state dict loads without error into any model…` |
| B 改参数留旧指纹 | **拒绝** | **拒绝** | `recorded parameter fingerprint is ec48003b…, but the parameters it actually carries hash to …` |
| C 删候选 `optimizer` | **拒绝** | **拒绝** | `the optimizer state is NoneType, not a mapping` |

**修后结论：3/3 全部被拒绝，合法对照仍被接受。**

## 4. 改了什么

### `src/spherical_blast_pinn/training/checkpoint.py`

| 新增 / 修改 | 作用 |
|---|---|
| `parameter_fingerprint_of(state_dict)` | **指纹的唯一实现**。`Trainer.parameter_fingerprint()` 改为调它——"重算"必须用同一个算法，另写一套会在某个 dtype 或遍历顺序的细节上分道扬镳 |
| `validate_model_state(model, expected, …)` | 候选/顶层的模型结构校验：非空、键集合、张量类型、形状、dtype、**有限性** |
| `validate_state_snapshot(state, …)` | 一份"训练状态快照"的只读校验，**顶层载荷与候选共用这一份** |
| `same_selection_value(a, b)` | 结构相等且 **NaN 当作等于自己**（见下） |
| `require_best_candidate_consistency(...)` | 重写：候选走同一份状态校验 + 指纹重算 + 步号三处一致 + 选择依据不矛盾 |

`expected_model_state` 与 `sampler_rng_validator` 是**必填**关键字参数：给默认值就等于
"调用方忘了传时这几项静默不查"，而这一项要修的正是那种静默。

### `src/spherical_blast_pinn/training/trainer.py`

| 位置 | 改动 |
|---|---|
| `Trainer.parameter_fingerprint()` | 委托给 `parameter_fingerprint_of` |
| `Trainer.load()` | 把自查模型/优化器/平衡器/采样器 RNG/阶段的那一大段**换成一次** `validate_state_snapshot()`；`require_best_candidate_consistency()` 补上两个上下文参数 |
| `Trainer._commit_checkpoint()` | 同一个调用补上上下文参数（纵深防御：绕过 `load` 的调用路径也拦得住） |
| `Trainer.run()` | **在 `_begin_publish()` 之前**校验待发布候选——写标记是这次发布的第一个副作用 |

**位置说明**：`validate_state_snapshot()` 放在原来那批逐条检查**相同的位置**
（日程 / 检查集 / 数据身份 / 算例关联之后），因为这些"更具体"的诊断仍应先说出口，
既有测试依赖这个次序（见 §6 的两条）。

### `AGENTS.md`

第二项规则整段扩写：四件事、候选参数**可以**与当前参数不同、必需字段的依据、
合法初始状态要保留、发布前校验。

## 5. 候选必需字段及其依据

依据是**两端的实际行为**：写入端 `Trainer._checkpoint_state()` 产生它们，
恢复端 `Trainer._commit_checkpoint()` 消费它们。

| 字段 | 为什么必需（恢复端怎么用它） |
|---|---|
| `step` | `_commit_checkpoint` → `self.step` |
| `model` | `load_state_dict`（并先与当前模型比键/形状/dtype/有限性） |
| `optimizer` | `optimizer.load_state_dict`；**`state` 为空是合法的**（一次更新都没做过） |
| `last_update_loss` / `last_update_loss_step` | 零更新的恢复运行会报 `final_loss=nan, final_loss_step=0`；**`nan` 是合法的**（未测） |
| `refinement` | `self.refine_history` |
| `rng` | `_restore_rng`；必须带 `numpy` 与 `torch` 两项 |
| `sampler_rng` | `sampler.set_rng_state`，并过 `validate_rng_state` |
| `history` / `checks` | `self.history` / `self.check_history` |
| `best_check` | `self.best`（候选内部那一份还要与顶层记录比对） |
| `balancing` | `balancer.load_state_dict` |
| `run_origin` | `_run_origin`（`explicit_diagnostic` / `refined` 两个布尔） |
| `stage` | `current` / `window_fraction` / `pending` / `blocked` / `history` |

**不在必需之列**：`learning_rates`。它被 `_checkpoint_state()` 写入但**全仓库无人读取**
（`grep -rn learning_rates src/ tests/ scripts/` 只有写入那一处）。把"写过"当成"必需"
会让规则比实际契约更严。这一条我**没有动它**（不属本项范围），只如实报告。

**明确不要求的**：候选参数与当前参数**数值相同**（候选本来就可能来自更早的一步），
以及"所有字段非空 / 所有数值有限"。

## 6. 我自己改出来的两个错，都当场修掉了

诚实记录，因为它们恰好是这一项要防的同一类东西：

1. **`score` 里带 NaN，被我判成"选择依据矛盾"。** `best_check.score` 是
   `[0.0, 9639311.5, 6.42, nan]`——`nan != nan`，于是**合法对照被我自己新增的检查
   拒掉**，两边打印出来的 score 一模一样。修法：`same_selection_value()` 把 NaN 视作
   等于自己。这正是"不要简单要求所有数值都有限"那一条的现场版。
2. **把共享校验放早了一格**，于是"算例改名"这类场景先报出的是采样器 RNG 缺键，
   而不是原来那条更具体的算例关联诊断。修法：把调用移回原来那批检查的位置。

## 7. 验收测试（`tests/test_checkpoint_prediction_chain.py` 新增 13 个函数 / 22 个用例）

| 复核要求 | 测试 |
|---|---|
| 空候选模型被拒绝 | `test_a_candidate_with_an_empty_model_is_refused` |
| 改参数保留原指纹 → 以**指纹不一致**拒绝 | `test_a_candidate_whose_parameters_do_not_match_its_fingerprint_is_refused`（并断言**记录没被改写**） |
| 删除候选 optimizer 被拒绝 | `test_a_candidate_without_its_optimizer_is_refused` |
| 缺键 / 错误形状 / NaN 被拒绝 | `..._missing_key_or_wrong_shape_...` / `..._wrong_shape_...` / `..._non_finite_parameter_...` |
| 其他必需候选状态损坏被拒绝 | `test_a_candidate_missing_required_state_is_refused`（10 个字段参数化） |
| 阶段名不与日程相容被拒绝 | `test_a_candidate_naming_an_unknown_stage_is_refused`（**复查时补的**，见 §10） |
| 拒绝后接收实例与已有文件保持不变 | `test_refusing_a_bad_candidate_leaves_the_receiver_and_the_files_untouched`（比对模型张量/步号/标志/候选/最佳/历史/lr + 文件字节） |
| 合法候选正常恢复、`best_dirty` 保留 | `test_an_honest_pending_candidate_still_loads_and_keeps_its_flag`（原有） |
| 恢复后走**真实 `run()`** 发布，`best.pt` 指纹/步号/最佳记录一致 | `test_a_loaded_candidate_publishes_a_consistent_best_checkpoint` |
| 发布的 `best.pt` 能被新实例恢复**并**过预测入口 | `test_a_published_best_checkpoint_is_loadable_and_predictable` |
| 候选来自较早一步、当前参数已继续更新仍通过 | `test_a_candidate_from_an_earlier_step_still_loads`（断言二者参数**确实不同**） |
| 坏候选不会被后续新候选顶掉而掩盖 | `test_a_corrupt_in_memory_candidate_never_reaches_best_pt`（**控制评分顺序**让新候选选不上，再断言发布前被拦、无标记、无 `best.pt`） |

## 8. 实跑命令与结果

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
PY=/e/Anaconda3/envs/pytorch/python.exe

# 全量测试
$PY -m pytest -q -rs
# → 676 passed, 6 skipped, 0 failed (15:13)

# 本项最相关的文件
$PY -m pytest -q tests/test_checkpoint_prediction_chain.py
# → 37 passed

# 发布标记 / 冻结爆速 / 算例关联（本项**未修改**的三处）
$PY -m pytest -q tests/test_publish_marker_contract.py
# → 46 passed

# 有界合成联检
$PY scripts/bounded_snake_check.py <out>
# → version 5, ok True, 30/30, exit 0

# 修前/修后同一份复现脚本（仓库外）
$PY %TEMP%/scan_19246a0/repro_candidate_item3.py
# → 修前 3/3 被接受；修后 3/3 被拒绝、合法对照被接受
```

6 项跳过全部因为真实提取数据不在本机（`F:\2\extracted_geom\50mm_d` 不存在）——
**跳过不是通过**，这里也不当作通过。

## 9. 未做的（明确标注）

* **没有升检查点版本**：结构没变（只是校验变严），坏数据不靠版本号回避。
* **没有运行真实数据链路**：`test_real_data_chain.py` 的 5 项与
  `test_similitude_data_and_run.py` 的 1 项在本机跳过（数据不在本机）。本项的改动
  不涉及数据读取路径。
* **没有做第四项，也没有开始第四项**——按提示词要求，第三项完成即停，等复核。
* **没有验证提交号**（本机无 git，见 §0）。

## 10. 完成后的复查（提示词要求"完成后复查，还有问题就重头改"）

复查走的是"逐条对着提示词的修复要求问：这条真的落实了吗"，结果**发现并补掉了一个缺口**：

* 提示词要求候选的必需字段要查"必要的类型、**结构和相容性**"。我第一版查了
  `stage` 块的键存在、`window_fraction` 有限、`blocked` 是布尔，但 **`stage.current`
  与 `stage.pending` 只查了"键在不在"**——`current = 123` 或一个日程里没有的阶段名
  都能过，而这两个值到了下游正是"这一份是哪个阶段的结论"的依据。
* 补法：`validate_state_snapshot()` 增加必填的 `known_stage_names`，两个值要么是
  `None`、要么必须在这套日程里；训练器三个调用点都传 `self._known_stage_names()`。
  同时把 `load()` 里原先那段**重复的**阶段名检查删掉（同一份规则不该有两处）。
* 配套测试 `test_a_candidate_naming_an_unknown_stage_is_refused`。

复查同时确认的两件事：

* **共享校验没有把顶层的检查"统一掉"**：逐段比对过 `load()` 改前的只读阶段，
  格式/版本/`time_scale`/冻结爆速与算例关联/日程/检查集/数据身份**全部原样保留**，
  被替换掉的只是模型/优化器/平衡器/采样器 RNG/阶段那一批（共享函数覆盖得更严）。
* **本项没有碰那三处**：发布标记契约 46 项测试全过；冻结爆速与算例关联的检查与
  测试未改动一行。

补完之后重跑：全量 676 passed / 6 skipped、本文件 37 passed、联检 30/30、修前修后
复现脚本 3/3 拒绝且合法对照被接受——四项一致。

## 11. 一处**我知道但这一版查不了**的边界（如实说明）

校验能证明的是"**候选自证一致**"：指纹与它自己带的参数对得上、步号三处一致、
选择依据（`score`/`check_set_identity`）与顶层最佳记录相同。它**不能**证明"这组
参数真的是某次真实评分选出来的"——一份把这些字段**成套**改成互相一致的检查点，
在文件层面与真品不可区分。要挡住那一类需要训练侧的可信签名，不是本项的范畴。
把边界写在这里，免得"校验通过"被读成比它实际更强的话。
