# 复核回应（第五项）：步号字段的存在性、类型与范围

**只处理这一项。** 优化器非负检查与探针更新后有限性检查**原样保留**（它们的测试
全部继续通过，见 §7）。没动物理模型、损失、日程、优化器规则、检查点版本与联检
脚本，没推进别的项。

## 0. 基线确认 —— 我做不到的事

复核给的基线提交是 `5115e750287daab37ff5440d30576083b68ab8fa`。

**本机这个工作树没有 `.git`**（`git rev-parse HEAD` → `fatal: not a git
repository`）。所以我**无法确认当前提交号**、无法读 `HEAD`、无法看工作区差异，
**没有**提交、**没有**推送、**没有**核对远端。把目录复制一份**不算** Git 检出
证明，我没有那样做过。下面所有"交付"都只是**工作树里的文件**。

同理，"若当前版本更晚，保留后续正确修改"我只能用**行为**保证：上一轮已通过独立
复核的优化器检查及其全部测试本轮继续通过（§7），没有放宽或重写。

**执行环境**：Python 3.12.12 / torch 2.9.1+cu130 / NumPy 2.3.5（CPU）。
哈希按 **LF** 规范化（工作区是 CRLF）。

## 1. 修前复现：两条反例都被接受

脚本：`%TEMP%\scan_item5_r1\repro_step_gap.py`（仓库外）。

真实流程：`verify()` 登记最佳候选 → `_save()` 写出带待发布候选的检查点。
合法对照保留不动；对**独立深拷贝**分别施加：

* **A** 删除 `pending_best_candidate.best_check.step`
* **B** 把它改成 `"broken"`

断言变异真实生效（读回来逐项确认），并断言对照未被改动。

| 步骤 | 合法对照 | A 缺失 | B `"broken"` |
|---|---|---|---|
| 共享候选校验 | 通过 | **通过** | **通过** |
| `Trainer.load()` | 通过 | **通过** | **通过** |
| `run()` 发布 `best.pt` | 写出 | **写出** | **写出** |
| 发布文件里的 `best_check.step` | `1` | **`<缺失>`** | **`"broken"`** |
| 新实例再次加载该文件 | 通过 | **通过** | **通过** |

**与复核给的结果逐条一致。**

另外复现了第三档（复核 §3 提到、但 §2 没列）：**已发布**的 `best.pt`
（`best_dirty=False`、`pending_best_candidate=None`）把顶层 `best_check.step`
改成 `"broken"` —— 修前 `Trainer.load()` **通过（未拦下）**。

## 2. 根因

`_require_one_moment()` 是唯一比较三处步号的地方，但它：

```python
step = candidate.get("step")
if not isinstance(step, (int, np.integer)) or isinstance(step, bool):
    return                      # ← 静默跳过
...
    if not isinstance(other, (int, np.integer)) or isinstance(other, bool):
        continue                # ← 静默跳过
```

注释声称"类型不对时保持沉默——那是状态校验的活"。**但后续的状态校验
（`validate_state_snapshot`）只查了快照自身的 `step`**，从未看过
`best_check.step`；对候选来说，"快照自身"是 `candidate["step"]`，候选内部的
`best_check.step` 与顶层 `best_check.step` 都没有第二道防线。而
`best_dirty=False` 时 `require_best_candidate_consistency()` 在第一道判据就
`return`，已发布文件的自身记录于是完全没人看。

## 3. 修复

### 3.1 一个小的共享校验函数

`checkpoint.require_step_value(value, *, field, where, error)` —— **步号规则的唯一
实现**：

* 接受 Python `int` 与 NumPy 整数（`np.integer`）；
* 拒绝缺失（`None`）、`None`、字符串、浮点、`bool`、`np.bool_`；
* **不先 `int()`**：`int("1")`、`int(1.0)`、`int(True)` 都得到 `1`，会把"合法
  记录"与"损坏记录"的差别抹掉；
* 范围沿用既有契约：非负。

报错点名**字段路径**与**实际值与类型**，例如：

```
...: pending_best_candidate.best_check.step is 'broken' (str), not an integer.
Step numbers are plain Python/NumPy integers — a missing key, None, a string, a
float or a boolean is not coerced with int(): int('1'), int(1.0) and int(True) all
yield 1 and would hide the difference between a valid record and a corrupt one
```

### 3.2 三处步号分别由哪个共享规则检查

| 字段 | 何时 | 由谁 |
|---|---|---|
| `pending_best_candidate.step` | 有候选时 | `_require_one_moment()`（发布前与加载前），**并且** `validate_state_snapshot(candidate)` 自身也查 |
| `pending_best_candidate.best_check.step` | 有候选时 | `_require_one_moment()` **并且** `validate_state_snapshot(candidate)` 通过它自己的 `best_check` 分支查 |
| 顶层 `best_check.step` | **总是** | `validate_state_snapshot(payload)`（`load()` 的第一道共享校验）；有候选时 `_require_one_moment()` 再比一次数值 |

`_require_one_moment()` 现在对三处**各调用一次** `require_step_value()`，**类型/范围
全过之后**才比较数值；三者不一致时报错列出三处的值与字段名，并保留既有测试依赖的
"different moments" 措辞。

### 3.3 无待发布候选的已发布检查点怎么查

`validate_state_snapshot()` 里新增（顶层载荷与候选**共用**同一份）：

```python
best_record = state.get("best_check")
if best_record is not None:
    if not isinstance(best_record, Mapping): raise ...
    require_step_value(best_record.get("step"), field="best_check.step", ...)
```

因此 `best.pt`（`best_dirty=False`、没有候选）也会查它**自身**的最佳记录。
**合法初始状态保留**：还没产生过最佳记录时 `best_check is None`，写入端
（`Trainer._checkpoint_state`）在这种情况写的就是 `None` 而不是缺键，不判损坏。

`best_check` **整个缺键**仍按"没有记录"处理（与 `None` 同义）——写入端从不省略
这个键，所以现实生活中只可能是 `None`。

### 3.4 没有修补坏记录

没有用顶层步号补齐缺失值、没有用当前训练步号覆盖候选步号、没有清空最佳记录、
没有清 `best_dirty`、没有丢弃候选、没有捕获错误后继续发布。候选来自较早一步是
合法情形：要求的是"**候选与它自己的最佳记录一致**"，不是"等于当前训练步号"。

## 4. 修后复现（同一脚本）

| 步骤 | 合法对照 | A 缺失 | B `"broken"` |
|---|---|---|---|
| 共享候选校验 | 通过 | **拒绝** | **拒绝** |
| `Trainer.load()` | 通过 | **拒绝** | **拒绝** |
| `run()` 发布 | 正常写出 | 未执行 | 未执行 |
| 新实例重新加载 | 通过 | — | — |

第三档（已发布 `best.pt` 的 `best_check.step`）修后 **拒绝**，报错含
`best_check.step`。

## 5. 回归测试（新增 35 个用例，本文件 64 → 99）

| 复核要求 | 测试 |
|---|---|
| 三处步号**各自**缺失/类型错误都被拒绝，且类型有区别 | `test_every_step_field_must_be_a_valid_integer` — **3 字段 × 7 种坏值** = 21 例：`missing` / `none` / `string` / `float` / `bool` / `numpy_bool` / `negative` |
| 三处都是合法整数但数值不一致 | `test_three_legal_but_disagreeing_steps_are_refused` — 3 字段各 1 例 |
| 已发布检查点自身 `best_check.step` 损坏、无候选也要拒绝 | `test_a_published_checkpoint_with_a_corrupt_best_record_step_is_refused` — 5 种坏值 |
| 合法初始状态：尚无最佳记录 | `test_a_checkpoint_without_a_best_record_is_still_legal`（`best_check=None`） |
| 合法整数步号 + 较早候选（参数确实不同） | `test_legal_integer_steps_and_an_earlier_candidate_still_load` |
| 合法候选发布后新实例可恢复、预测入口可用 | `test_a_legal_candidate_publishes_and_the_published_steps_are_intact` |
| 拒绝发生在接收实例被修改之前 | `test_refusing_a_bad_step_leaves_the_receiver_and_the_file_untouched`（模型、**完整优化器状态**、步号、最佳记录、候选、标志、历史、阶段、采样器 RNG、全局 RNG + 文件字节） |
| 内存里破坏候选步号，发布路径拒绝 | `test_a_corrupt_in_memory_step_is_refused_before_begin_publish` — **直接监测 `_begin_publish()` 未被调用**（不是靠"没有标记文件"推断） |
| 校验不得修改输入载荷 | `test_validation_does_not_modify_the_input_payload`（递归比较，张量逐元素；**NaN 当作等于自己**） |

三处字段的合法性断言都检查**报错里出现字段路径**（`candidate.step` /
`candidate.best_check.step` / `best_check.step`）。

## 6. 我自己踩到的一个坑（记录）

`_deep_equal` 第一版把 `nan != nan` 判成了"载荷被修改"。`best_check.score` 里带
NaN 是**合法**的（某个分量没有可测值）——这正是上一轮 `same_selection_value()`
处理过的那件事。修法：数值分支里把 NaN 当作等于自己。**这是比较器的错，不是校验
真的改了输入**：加上之后同一条测试通过。

## 7. 实跑命令与结果

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
PY=/e/Anaconda3/envs/pytorch/python.exe
```

| 命令 | 结果 |
|---|---|
| 直接相关回归（预测链路 + 最佳候选事务 + 阶段机 + 精修检查点） | **164 passed** |
| 本轮新增 35 例（含 4 条既有的 "step" 测试） | **39 passed** |
| 全量 `pytest -q -rs` | **739 passed, 6 skipped, 0 failed**（17:58，exit 0） |
| 修前/修后复现脚本 | 见 §1 / §4 |
| 死代码扫描 | **A=0 / B=0 / C=0** |

### 文件哈希（LF 规范化，按工作区实际内容计算）

| 文件 | 字节（LF） | sha256[:16] |
|---|---:|---|
| `src/spherical_blast_pinn/training/checkpoint.py` | 78733 | `74b428df69df6bab` |
| `tests/test_checkpoint_prediction_chain.py` | 92895 | `51cb8d81004a64aa` |
| `AGENTS.md` | 46096 | `d6f80d4b343ffd0a` |
| `src/spherical_blast_pinn/training/trainer.py`（本轮**未改**） | 137681 | `623bac99eb960909` |
| `scripts/bounded_snake_check.py`（本轮**未改**） | 56618 | `2ba72422244ea48b` |

> 本报告自身（`REVIEW_RESPONSE_5115e750.md`）的哈希不列在这里：写进去之后它就变了。
> 复现脚本与产物都在仓库外：`%TEMP%\scan_item5_r1\`。
> **`trainer.py` 一个字节没动**——本轮修复全部在 `checkpoint.py`。

**上一轮的优化器检查没有回归**：单独跑负二阶动量 / 探针结果 / 合法负一阶矩 /
合法零二阶矩 / 合法恢复那 11 个用例 → **11 passed**；它们也在上面的全量里。
`validate_optimizer_state()` 与 `_probe_optimizer_step()` 的**代码一行未动**。

6 项跳过**全部**因为真实提取数据不在本机（`F:\2\extracted_geom\50mm_d`）——
**跳过不是通过**，这里不当作通过。

### 7.1 一条**在本机通过、在远端已知失败**的测试（单独列出，未动它）

`tests/test_checkpoint_prediction_chain.py::test_the_delivered_snake_script_declares_v5_and_carries_the_three_added_checks`

* 复核说：远端失败，因为交付的联检脚本仍是 v4。
* 本机：`scripts/bounded_snake_check.py` 是 **v5 / 30 项**，该测试**通过**。
* 差异来自 v5 的交付没落地（本机无 `.git`）。**按复核要求：本轮不修改脚本、
  不跳过该测试、不降低断言，也不据此宣称全绿。**

## 8. 未验证 / 未提交（明确标注）

* **未验证提交身份**：无 `.git`，无法确认当前提交、无法比对基线
  `5115e750…`、无法确认"更晚的版本里哪些改动被保留"（§0）。
* **未提交、未推送、未核对远端。**
* **未运行真实数据链路**（数据不在本机）。
* **预测入口（`cli.predict`）不运行状态快照校验**：它只查发布标记、爆速与算例
  关联，不验 `best_check.step`。本轮的范围是 `Trainer.load()` 与发布路径，没有
  扩大；如实标出这个边界。
* **未修改**：物理模型、损失函数、阶段日程、优化器规则、检查点版本、联检脚本；
  未做旧检查点迁移，未推进第四项。
* **边界**：这些校验证明"记录自证一致"，不证明"这组参数真是一次真实训练的产物"。
