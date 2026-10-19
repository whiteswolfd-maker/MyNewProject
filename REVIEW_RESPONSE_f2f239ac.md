# 复核回应（第四项）：损坏的优化器状态通过校验、恢复后污染模型参数

**只处理这一项。** 没动手号校验，没动 v5 联检脚本，没推进第四项。物理模型、损失
函数、阶段日程、正式验收门槛、检查点版本**一行未改**。

## 0. 基线确认 —— 我做不到的两件事

复核给的基线提交是 `f2f239ac3ac0d747cda92e1dc4f3c807ab332670`。

1. **本机这个工作树没有 `.git`**（`git rev-parse HEAD` → `fatal: not a git
   repository`）。我**无法确认当前提交号**、无法读出 `HEAD`、无法看工作区差异，
   **没有**提交、**没有**推送、**没有**核对远端。把目录复制一份**不算** Git 检出
   证明，我没有那样做过。下面所有"交付"都只是**工作树里的文件**。
2. 因此**"如果当前提交更晚，保留后续正确修改"这句话我无法执行**：没有基线可
   比对，"后续修改"是哪几处、有没有被本项覆盖，我判断不了。我实际保证的是**行为
   层面**的保留：上一轮那批已通过的测试（发布标记、冻结爆速、算例关联、
   候选完整性、非模型状态）本轮**全部继续通过**，见 §7。

**执行环境**：Python 3.12.12 / torch 2.9.1+cu130 / NumPy 2.3.5（CPU）。
哈希按 **LF** 规范化（工作区是 CRLF）。

## 1. 修前复现：损坏的状态通过校验并污染模型

脚本：`%TEMP%\scan_item4_r1\repro_negative_exp_avg_sq.py`（仓库外）。

夹具：`_checkpoint_with_a_candidate_that_has_momentum()` 同一条真实控制流
（`_update()` 一步 → `verify()` 登记候选 → 再走一步 → `_save()`），候选在第 1 步、
顶层在第 2 步，候选带着真实的 Adam 动量。

变异：对**独立深拷贝**只改一处——
`pending_best_candidate.optimizer.state[0].exp_avg_sq` 全部填成 `-1`。
断言变异生效（读回来逐元素等于 -1），并断言**合法对照未被改动**。

修前实测：

| 步骤 | 合法对照 | 负二阶动量 |
|---|---|---|
| 共享候选校验 | 通过 | **通过** |
| `Trainer.load()` | 通过 | **通过** |
| `run()` 发布 `best.pt` | 写出 | **写出** |
| 新实例重新加载 `best.pt` | 通过 | **通过** |
| 一次真实 `_update()` 之后 | 0/12 非有限 | **1/12 非有限**（`product_net.mlp.network.0.weight`） |
| 再一次 `_update()` | 0/12 非有限 | `NonFiniteUpdate: update 3 was blocked: the total loss is not finite` |

**控制流与复核描述逐条一致。**

### 一处数字对不上，如实说明

复核写的是"**16 个**参数变为非有限值"，我实测是 **1 个**（我的夹具一共 12 个
参数）。我把 12 个参数号**逐个试了一遍**：无论污染哪一个，结果都是**正好那一个**
参数变非有限（`1/12`）。这是因为 `_update()` 里 `optimizer.step()` 是**最后**一步，
一次更新只会用坏动量写坏它自己那一个参数；污染扩散要等到**下一次**迭代，而那时
`_update()` 会在 `optimizer.step()` **之前**先抛 `NonFiniteUpdate`（梯度里已经
有 NaN），所以不会继续扩散。

也就是说：**"坏状态通过校验 → 发布 → 恢复 → 下一次更新把参数写坏"这条链我完全
复现了；"一次更新污染 16 个参数"我复现不出来**。如果复核的 16 是那个夹具的参数
总数，那只是夹具规模不同；如果那 16 是"全部参数一起变坏"，那在我的夹具里不成立，
我没有为了让数字对上而改夹具。这一点请复核方确认。

## 2. A/B：两条新检查各自是不是真的在拦

只改**内存里一个探针实例**，不动源码、不改文件：

| 变异 | 两条都在 | 停用非负语义 | 停用探针结果 | 两条都停用 |
|---|---|---|---|---|
| 负二阶动量 `-1` | 拒绝（非负语义） | 拒绝（**探针结果**） | 拒绝（非负语义） | **被放行** |
| `exp_avg=1e308, exp_avg_sq=0` | 拒绝（探针结果） | 拒绝（探针结果） | **被放行** | **被放行** |

两件事因此是确定的：

* 负二阶动量**被两条检查各自独立拦一次**（纵深防御）；
* "有限且非负、但更新结果非有限"这一档**只有探针结果检查能拦**——这正是复核
  提醒的"只靠负二阶动量测试，可能只测到了前置的非负检查"。

## 3. 修了什么

### `src/spherical_blast_pinn/training/checkpoint.py`

| 新增 / 修改 | 作用 |
|---|---|
| `OPTIMIZER_NON_NEGATIVE_FIELDS = ("exp_avg_sq", "max_exp_avg_sq")` | 必须**逐元素非负**的字段表 |
| `validate_optimizer_state()` 的逐项循环 | 在原有"有限性 + 形状"之后，对表内字段再加一条**非负**检查；报错点名**状态字段**与**参数编号**，并给出负值个数与最小值 |

按**字段名**判断而不是先去读参数组里的 `amsgrad` 开关：`max_exp_avg_sq` 只在
`amsgrad=True` 时存在，按名判断天然就只在它出现时才要求；而"开关与字段不一致时
该信哪个"没有依据，"这个字段是平方量"则是确定的。

**`exp_avg` 不在表内**：它是一阶矩，本来就可以为负。把它一起要求非负会把合法的
训练历史判成损坏。

**没有**用取绝对值、截断到零、清空动量、重建状态或 `nan_to_num`——这几种都是把
损坏的记录改成能过，而不是把它认出来。

### `src/spherical_blast_pinn/training/trainer.py`

| 位置 | 改动 |
|---|---|
| `_probe_optimizer_step()` | 探针 `step()` 之后**新增结果检查**；同时把克隆参数按名字记下来（报错能指名参数）；docstring 写明"跑完了 ≠ 结果是好的"，以及校验不碰全局随机状态 |
| `_require_probe_results_finite()`（新，`@staticmethod`） | 两类检查：**探针参数**逐元素有限；**探针优化器浮点状态**（`state[pid][key]`）逐元素有限。分别给出各自的报错 |

**隔离保持不变**：参数克隆、优化器同类新建、状态字典 `_copy.deepcopy`。深拷贝
不是可选项——`load_state_dict` 在同 dtype/同设备时会把张量**原样接过去**，不深拷贝
的话探针那一步的原地更新就改到了候选自己的动量上。这一条现在**有测试钉住**（§6）。

没有扩展成训练事务系统重构。

## 4. 负二阶动量被拒绝的位置

两处，按执行顺序：

1. **`validate_optimizer_state()`**（共享的只读结构/语义校验）——先命中。实跑原文：

```
<文件路径>: pending_best_candidate: optimizer: the optimizer state 'exp_avg_sq' for
parameter 3 holds 3 negative value(s), down to -12345. This field estimates a
**squared** gradient and Adam takes its square root before dividing by it, so the
first update after restoring yields NaN — for this parameter and then for the whole
model, since the loss built from it is not finite either. Refusing rather than
clamping to zero, taking the absolute value or rebuilding the moments: each of those
would silently substitute a different training history
```

2. **`_probe_optimizer_step()` → `_require_probe_results_finite()`** ——如果第 1 条被
   绕过（A/B 里停用过），探针那一步会把参数写坏，由这一条拦下：

```
... pending_best_candidate: optimizer: the restored optimiser would drive 1
parameter(s) to a non-finite value on the very next update
(['product_net.mlp.network.0.weight']). The step ran without raising, which is not
the same as the update being valid; publishing this candidate would put a best.pt
on disk whose first resumed update poisons the model
```

**顶层与候选走的是同一条**：`_snapshot_rules()` 是校验上下文的唯一构造点，
`load()` 与 `run()` 的调用点都展开它，没有"只在某一个入口追加特判"。

## 5. 探针结果检查的独立证据

两个层次，都不是"负二阶动量测试"能替代的：

* **真实触发（非注入）**：`exp_avg = 1e308` 配 `exp_avg_sq = 0`——两个值都**有限**
  且二阶矩**非负**，非负语义那条放它过去；Adam 的分母是
  `sqrt(0)/sqrt(bias_correction2) + eps`，更新量放大约 `1e6` 倍，参数直接变 `inf`，
  **`step()` 全程不抛异常**。这一条由探针结果检查拦下（A/B 里停用它就**被放行**）。
* **受控故障注入（两条，明确标注为探针分支测试）**：
  `test_the_probe_branch_refuses_an_injected_non_finite_step` —— 把
  `torch.optim.Adam.step` 换成"调用真实现之后再 `fill_(nan)`"，状态本身完全合法，
  断言被拒，并断言**真模型参数、真优化器状态、全局随机状态都没被碰过**；
  `test_the_probe_branch_refuses_an_injected_non_finite_optimizer_state` —— 同样注入
  但只写坏优化器状态，验证"状态侧"那一半检查。

## 6. 保留正常恢复行为（新增测试）

| 复核要求 | 测试 |
|---|---|
| 有参数组、`state == {}` 的初始状态 | `test_legal_candidates_resume_with_finite_parameters_and_moments`（并断言候选确实是空的） |
| 较早一步、带正常动量的候选 | 同上（候选第 1 步、顶层第 2 步，走完发布→恢复→更新） |
| **合法的负值 `exp_avg`** | `test_a_legal_negative_first_moment_is_accepted` |
| **合法的零值二阶动量** | `test_a_legal_zero_second_moment_is_accepted` |
| 候选依据**自己的**快照校验 | 动量夹具本身就是"候选 vs 当前状态不同时刻"；`_same_optimizer` 分别比对两者 |
| 顶层与候选同一套规则 | `test_the_top_level_optimizer_with_a_negative_second_moment_is_refused` |

补充：`_checkpoint_with_a_candidate_that_has_momentum` 让候选在第 1 步、顶层在第 2
步，且在同一阶段内（日程每阶段 8 步）——手工推进不会撞上阶段边界。

## 7. 测试与验证

新增 **12 个用例**（`tests/test_checkpoint_prediction_chain.py`，本文件 52 → 64）：
**5 条损坏/拒绝**（候选、顶层、端到端不落盘、内存内发布前拦截、拒绝后只读）、
**3 条合法放行**（负一阶矩、零二阶矩、空状态+动量的完整恢复）、
**3 条探针分支**（1 条真实触发 + 2 条受控注入，其中一条覆盖探针**状态侧**）、
**1 条只读性与隔离性**。

| 命令 | 结果 |
|---|---|
| `pytest -q -rs`（修后、**未含**新测试） | **692 passed, 6 skipped, 0 failed**（exit 0）——修没碰坏既有回归 |
| `pytest -q -rs`（修后、**含**新测试） | **704 passed, 6 skipped, 0 failed**（15:38，exit 0） |
| `pytest -q tests/test_checkpoint_prediction_chain.py` | **64 passed**（52 → +12） |
| 修前/修后复现脚本 | 控制流见 §1 |
| A/B 停用检查 | 矩阵见 §2 |
| 死代码扫描 | A=0 / B=0 / C=0 |

### 文件哈希（LF 规范化，按工作区实际内容计算）

| 文件 | 字节（LF） | sha256[:16] |
|---|---:|---|
| `src/spherical_blast_pinn/training/checkpoint.py` | 74995 | `6ed9a563e318b8a0` |
| `src/spherical_blast_pinn/training/trainer.py` | 137681 | `623bac99eb960909` |
| `tests/test_checkpoint_prediction_chain.py` | 76548 | `46e8828ac43167d4` |
| `AGENTS.md` | 45048 | `7ffd019fb0ad0670` |
| `scripts/bounded_snake_check.py`（本轮**未改**） | 56618 | `2ba72422244ea48b` |

> 本报告自身（`REVIEW_RESPONSE_f2f239ac.md`）的哈希**不列在这里**：写进去之后它
> 就变了。请以工作区里的实际文件为准。脚本与产物目录都在仓库外：
> 复现脚本 `%TEMP%\scan_item4_r1\`。

**只读性与隔离性**由
`test_validating_a_candidate_does_not_touch_its_tensors_or_the_global_rng` 覆盖：
校验一份**合法**候选之后，输入候选的**优化器状态张量**与**模型张量**逐元素不变、
全局随机状态不变；再加上注入测试里对**真模型 / 真优化器 / 全局随机**的三重断言，
以及 `test_refusing_a_negative_second_moment_leaves_everything_alone` 对
接收实例（模型、完整优化器状态、阶段、历史、采样器 RNG、全局 RNG）与**输入文件
字节**的比对。

### 7.1 一条**在本机通过、在远端已知失败**的测试（单独列出，未动它）

`tests/test_checkpoint_prediction_chain.py::test_the_delivered_snake_script_declares_v5_and_carries_the_three_added_checks`

* 复核说：远端失败，因为**交付的联检脚本仍是 v4**。
* 本机实测：**通过**（`1 passed in 5.19s`）。本机的
  `scripts/bounded_snake_check.py` 声明 `SNAKE_CHECK_VERSION = 5`、实际记录 30 项、
  30/30、exit 0，与那条测试的断言一致。
* 差异的来源是**交付没落地**（本机无 `.git`，我无法提交/推送），不是断言写错。
* **按复核要求：我没有修改它、没有跳过它、没有降低它的要求，也没有据此宣称全绿。**
  它在远端失败这件事本身是真的，修复它需要把 v5 那份脚本真正交付出去，那是另一
  件事，不属于本轮。

### 7.2 真实数据链路

6 项跳过**全部**因为真实提取数据不在本机（`F:\2\extracted_geom\50mm_d`）——
**跳过不是通过**，这里也不当作通过。

## 8. 未验证 / 未提交（明确标注）

* **未验证提交身份**：无 `.git`，无法确认当前提交、无法比对基线
  `f2f239ac3ac0…`、无法确认"更晚的提交里哪些改动被保留"（§0）。
* **未提交、未推送、未核对远端**。
* **未运行真实数据链路**（数据不在本机）。
* **未修改**：物理模型、损失函数、阶段日程、正式验收门槛、检查点版本；
  也没动步号校验与 v5 联检脚本，没推进第四项。
* **一处边界**：这套校验证明的是"这份状态**自证相容**"。它不能证明这组动量真是
  某次真实训练跑出来的——把动量、参数、指纹**成套**改成互相一致的检查点，在文件
  层面与真品不可区分。
* **上一轮遗留、仍待你决定的一件事**：检查点 7→8 会让所有旧检查点失效、必须重训；
  是否有旧检查点必须保留？
