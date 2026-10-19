# 交付回应（v5 有界合成联检脚本）

**只处理这一项。** 训练器、检查点校验、预测/对照产品逻辑、物理模型、正式验收
门槛、检查点版本**一行未改**。唯一修改的文件是
`scripts/bounded_snake_check.py`。

## 0. 基线与能力边界 —— 先说做不到的

复核给的基线提交：`3d5924130a11e7efff499c5e5ed224da0e22612e`。

* **本机工作树没有 `.git`**（`git rev-parse HEAD` → `fatal: not a git
  repository`）。无法确认 HEAD、无法看工作区差异、无法确认基线。
* **没有远端访问能力**：
  * `git ls-remote https://github.com/whiteswolfd-maker/deskbook` →
    `Failed to connect to github.com port 443`（并且会提示输入用户名，
    说明不是匿名可达）；
  * `gh` 未安装；`GH_TOKEN`/`GITHUB_TOKEN`/`GIT_ASKPASS` 均未设置；
    `credential.helper` 只配了 `whiteswolfd-maker`（无凭据可用）。
* 因此 **提交、推送、远端重新克隆、在克隆中复验，全部未完成**。按复核 §6 的
  要求，本报告明确标注为：**本地完成，远端交付及重新克隆复验未完成**。
  我没有把复制目录当成远端克隆，也没有用旧提交的测试结果为新提交背书。

## 1. v5 是找到原文件，还是本轮重建？

**找到了原文件，但交付的是"找到 + 修正"后的版本。**

* 本轮修改**之前**，`scripts/bounded_snake_check.py` 已经是
  `SNAKE_CHECK_VERSION = 5`、30 项检查、本机 30/30、exit 0；
  上一轮报告记录的哈希是 **56618 字节（LF）/ `2ba72422244ea48b`（sha256[:16]）**。
  它就是此前报告里说的那一份 v5。
* **但它有两项达不到本轮复核的要求**，所以我没有原样交付：
  1. `the_model_prediction_can_be_compared_against_the_reference` 只要求"报告
     存在、`case_id` 对得上"。上一轮的实际输出里 **`comparable_points: 0`**、
     `available: false`，检查照样 `ok: true`——正是复核点名要避免的"空报告冒充
     成功"。
  2. `the_same_uncommitted_checkpoint_is_refused_by_both_entries` **没有合法
     对照**：只验证了"有标记时两个入口都拒绝"，没有先证明"同一份检查点在没有
     标记时确实能读"；拒绝也可能来自配置错误或无关异常。
* 因此本轮**修正**了这两项的行为（并给第一项补了"有效查询点上的有限性"），
  其余 27 项与它们的实现未改。

**所以：不是重建，也不是原样找回。** 原文件存在；交付版本是在它上面按复核要求
把新增三项真正落实之后的结果。

### 与"远端 v4"的差异核对

无法取得远端 v4 的文件文本（无网络），因此**无法做代码级 diff**。我做了能做
的两件事：

* 以 `REVIEW_RESPONSE_3e4e1c61.md` 中**逐条列出的 v4 27 项名字**为底稿，
  确认这 27 个名字在 v5 中**全部存在**；
* 这 27 项清单已写进
  `tests/test_checkpoint_prediction_chain.py::V4_CHECK_NAMES`，新增三项写进
  `V5_ADDED_CHECKS`，由
  `test_the_delivered_snake_script_declares_v5_and_carries_the_three_added_checks`
  强制校验（缺任何一项即失败）。

运行时实测：**30 个条目、30 个唯一名字、无遗漏、无重复**（源码里有 31 处
`rec.check(...)` 调用，其中 `the_model_prediction_can_be_compared_against_the_reference`
在"预测没有产出文件"的提前返回分支里还有一次——两条分支互斥，一次运行只记一条）。

## 2. 三项新增检查分别调用了什么真实入口

### ① `prediction_entry_runs_from_a_training_checkpoint`

* 用 `_trainer()` 真实训练 → `trainer.run()` → **真实检查点** `last.pt`；
* 调 **`spherical_blast_pinn.cli.predict.main([...])`**（不是手写 NPZ）；
* 读回它写出的 NPZ，核对：
  * `density.shape == (n_times, n_radii)`；
  * `charge_radius` 与训练算例一致；
  * `start_time` 与用**检查点里冻结的 `D`** 独立重算的
    `T_h = R_0/D + ε` **逐位相等**（容差 `1e-18`）；
  * `contact_radius` 全有限；
  * **新增**：预测自己标出的可用区域（`region >= 0`）里，`density`/`velocity`/
    `specific_energy` **全部有限**，且可用点数 > 0。

  实测证据：`valid_query_points: 240`、`fields_finite_at_valid_points: true`。

### ② `the_model_prediction_can_be_compared_against_the_reference`

* 取**上一步真实产出的那份预测文件**；
* 调 **`spherical_blast_pinn.cli.compare.main([...])`**（不是直接调内部函数、
  也不是把参考数组塞进预测文件）；
* 要求：退出码 0、报告文件存在、`case_id` 匹配、**`fields.available is True`**、
  **`fields.comparable_points > 0`**（整数）。
* **不**对误差大小设阈值：冒烟模型的精度不在本项考核范围内；但"一个可比点都
  没有"不再算成功。

  实测证据：`available: true`、`comparable_points: 609`、逐字段
  `product_density/velocity/specific_energy = 203/203/203`。

  **为什么换用 `_compare_case`**：原来链路用训练那条链的 `_synthetic_case`。
  它的网格粗、介质带窄，启发式前沿检测器把材料分区退化成一整片"未受扰动"，
  对照的候选点因此是 **0**。`_compare_case` 写了一份 `material.npz`（材料分区是
  **读数**而不是启发式判据），训练、预测、对照现在共用**同一份**这样的合成
  算例，链路自洽。这是脚本内的合成夹具，不碰正式数据。

### ③ `the_same_uncommitted_checkpoint_is_refused_by_both_entries`

* 在检查点**所在目录**写下真实的 `.publish_in_progress.json`，`artifacts` 点名
  被检查的那一份（`last.pt`），并带 `owner`；
* **合法对照（新增）**：同一份检查点、**没有标记**时，
  * `predict.load_model_from_checkpoint(...)` 接受；
  * 新 `Trainer.load(...)` 接受；
* **阻断组**：有标记时，
  * 预测入口抛 `SetupError`；
  * 训练恢复入口抛 `TrainingError`；
* 两个入口的报错都必须包含 **`left uncommitted`**——证明拒绝来自发布完整性
  检查，而不是配置错误、缺文件或无关异常；
* 标记在 `finally` 里删除。

  实测证据：`control_prediction_entry_accepts: true`、
  `control_training_entry_accepts: true`、`prediction_entry_refused: true`、
  `training_entry_refused: true`，两条报错原文均为
  `... was left uncommitted by an interrupted publish ...`。

## 3. 有界性与独立运行

* 预算常量未变：`MAX_UPDATES = 18`、`MAX_REFINE_ITERATIONS = 5`；CPU、小网络
  （width 8 / depth 1）、小样本、固定种子。
* 脚本**没有**盘符硬编码、没有 `/tmp`、没有 `%TEMP%`、没有对旧输出或仓库外辅助
  文件的引用（`grep` 实测为空）。模块级只 `import` 标准库与 NumPy；其余都是
  仓库内模块的惰性导入。
* 产物写到命令行给定的目录（本次在仓库外）。
* **没有新增辅助文件**，所以不需要额外交付。
* 失败时 `main()` 返回 1：实测把其中一项强制为失败 →
  `report.ok=False`、30 个条目里 1 条 FAILED、**`main()` 返回 1**。

## 4. 实跑命令与结果

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
PY=/e/Anaconda3/envs/pytorch/python.exe
```

| 命令 | 结果 |
|---|---|
| `$PY -m pytest -q "tests/test_checkpoint_prediction_chain.py::test_the_delivered_snake_script_declares_v5_and_carries_the_three_added_checks"` | **1 passed**（此前在远端失败的那条） |
| `$PY -m pytest -q -rs tests/test_checkpoint_prediction_chain.py` | **99 passed** |
| `PYTHONPATH=src $PY scripts/bounded_snake_check.py <仓库外目录>` | **version 5 / 30 项 / 30 PASS / exit 0** |
| 强制一项失败后重跑（验证退出码） | `ok=False`，30 项中 1 FAIL，**`main()` 返回 1** |
| `$PY -m pytest -q -rs`（全量） | **739 passed, 6 skipped, 0 failed**（18:16，exit 0） |

**联检输出 JSON**（仓库外）：`version: 5`、`checks` 长度 **30**、唯一名字
**30**、`ok: true`、逐项 `ok` 全为 `true`；进程退出码 **0**，与
`report.ok` 一致。

### 4.1 全量与尚未通过的项

* 全量（含上述脚本测试）：**739 passed / 6 skipped / 0 failed**（exit 0）。
  与上一轮同为 739：本轮**没有新增测试**，只改了联检脚本。
* 死代码扫描：A=0 / B=0 / C=0。

**已知会在远端失败、本轮未修改的项**：
`tests/test_checkpoint_prediction_chain.py::test_the_delivered_snake_script_declares_v5_and_carries_the_three_added_checks`
在**远端**失败，因为远端 `scripts/bounded_snake_check.py` 仍是 v4。本轮**本机**
它通过（本机脚本是 v5）。让它真正通过的唯一办法是**把本文件交付到远端**——
这一步没有完成（§0）。**没有跳过该测试、没有修改检查项清单、没有降低断言。**

6 项跳过**全部**因为真实提取数据不在本机（`F:\2\extracted_geom\50mm_d`）——
**跳过不是通过**。**有界合成联检通过不等于真实数据或物理验收通过。**

## 5. 交付物与哈希

| 文件 | 字节 | sha256 |
|---|---:|---|
| `scripts/bounded_snake_check.py`（工作树） | 60741 | `32cc4487d1cdcffb66809219d6f68a5c4ab6c5f0aaaf986507cfc65d12199cbc` |
| 仓库外副本 `%TEMP%\delivery_3d592413\bounded_snake_check.py` | 60741 | 同上（逐字节相同） |

（本工作区无 CRLF，`sha256(原始)` 与 `sha256(LF 规范化)` 相同。）
修改前那一版：56618 字节 / `2ba72422244ea48b`（sha256[:16]）。

**修改文件（本轮唯一）**：`scripts/bounded_snake_check.py`。

## 6. 未完成的交付步骤（明确标注）

1. **没有提交**（无 `.git`）。
2. **没有推送**（无凭据、无网络）。
3. **没有远端提交 SHA**——无提交可引用。
4. **没有从远端重新克隆**，因此**没有克隆目录的 HEAD，也没有克隆内复验结果**。
5. 因此"远端交付的脚本与 v4 的代码级 diff"也**无法提供**：只能给出与 v4
   27 项**清单**的比对（§1）。

如果复核环境具备 Git 与远端访问，落地方式是：把上表那个文件放到
`scripts/bounded_snake_check.py`（整文件替换，无其他改动），提交并推送，然后
在全新克隆中运行 §4 的三条命令；预期 `version 5 / 30 项 / exit 0`，且
`test_the_delivered_snake_script_declares_v5_and_carries_the_three_added_checks`
由失败转为通过。
