# 复核回应（第三项补完）：最佳候选的**非模型**状态校验，与联检交付一致性

**只补第三项。** 发布标记、冻结爆速、算例关联三处的代码一个字没动（回归见 §7）。
上一轮已通过的六条（空候选模型、改参数留旧指纹、删候选 optimizer、合法候选可
加载/发布/恢复、候选参数可与当前不同、三处受保护校验）**没有放宽**，本轮另加一条
测试专门钉住它们（§6）。

## 0. 基线确认 —— 两件我做不到的事

复核给的仓库是 `whiteswolfd-maker/deskbook`，基线提交
`746c319ab423d5777db754be78902d6373350892`。

1. **本机这个工作树没有 `.git`**（`git rev-parse HEAD` → `fatal: not a git
   repository`）。所以我**无法确认当前提交号**，**没有**提交、**没有**推送、
   **没有**验证远端版本。下面所有"交付"都只是**工作树里的文件**，请在有 git 的
   环境里核对并提交。
2. **我无法读到远端那份 `scripts/bounded_snake_check.py`**（无 git、无网络）。
   复核说它是 v4（27 项），我只能用本机文件 + 历史报告交叉验证，见 §8。

**执行环境**：Python 3.12.12 / torch 2.9.1+cu130 / NumPy 2.3.5（CPU）。
哈希按 **LF** 规范化后计算（工作区是 CRLF）。

## 1. 修前复现：三条反例 3/3 被接受

脚本：`%TEMP%\scan_item3_r2\repro_nonmodel_item3.py`（仓库外）。

做法（与上一轮一致，不手工拼检查点）：真实 `verify()` 登记候选 → 真实 `_save()`
写出中间检查点 → 保留**未修改的合法对照** → 对三个独立副本分别修改 → **断言变异
确实生效** → 分别过共享校验与 `Trainer.load()` → 被接受的再走一遍**真实 `run()`
发布**。

修前实测：

| 变异 | 共享校验 | `Trainer.load()` | 真实 `run()` | 新实例恢复 `best.pt` 时 |
|---|---|---|---|---|
| A 候选 `optimizer = {}` | **通过** | **通过** | **写出 `best.pt`** | `KeyError: 'param_groups'` |
| B 候选 `rng['torch'] = 0` | **通过** | **通过** | **写出 `best.pt`** | `TypeError: expected a torch.ByteTensor, but got int` |
| C 候选 `stage.window_fraction = 0.123` | **通过** | **通过** | **写出 `best.pt`** | 窗口 0.123 与候选步号 0（要求 0.25）不符 |

**与复核给的报错逐条一致。** 合法对照（未修改）修前修后都被接受。

## 2. 逐条停用新检查（A/B：证明拦下它的就是那一条）

只把修前结果和修后结果并排，说明不了"是哪一条在拦"。所以脚本在**内存里**逐条
停用本次新增的检查（不改文件），看那条变异会不会被重新放行：

```
A optimizer={}     ：停用该条检查后 → **被放行**（这一条检查就是拦住它的原因）
B rng['torch']=0   ：停用该条检查后 → **被放行**（这一条检查就是拦住它的原因）
C window=0.123     ：停用该条检查后 → **被放行**（这一条检查就是拦住它的原因）
```

## 3. 原因（读代码确认）

那一版的 `validate_state_snapshot` 对这三项只问到"键在不在、是不是 Mapping"：

* **优化器**只 `isinstance(..., Mapping)`：`{"state": {}, "param_groups": [...]}`
  与 `{}` 都通过；
* **全局随机**只查 `numpy`/`torch` **不是 `None`**：整数 `0` 不是 `None`；
* **窗口**只查"是不是有限数"：`0.123` 是有限数。

三个缺口都在**恢复端**，于是错误发生在"接收实例已经被改到一半"之后。

## 4. 修后复现（同一脚本）

| 变异 | 共享校验 | `Trainer.load()` | 真实 `run()` |
|---|---|---|---|
| 合法对照 | 接受 | 接受 | 正常发布 |
| A | **拒绝** | **拒绝** | `load` 阶段就拒绝，**未产生任何发布副作用** |
| B | **拒绝** | **拒绝** | 同上 |
| C | **拒绝** | **拒绝** | 同上 |

拒绝理由（`where` 里带着**发布目标或文件路径**、`pending_best_candidate` 下的
**具体字段**与原因）：

```
A  <path>: pending_best_candidate: optimizer: the optimizer state carries no
   'param_groups' key. Restoring it fails with KeyError('param_groups') — but only
   after the receiving instance has already been modified ...
B  <path>: pending_best_candidate: the global RNG state is not usable:
   rng['torch']: expected a torch.ByteTensor, but got int
C  <path>: pending_best_candidate: stage.window_fraction is 0.123, but step 0 of
   this schedule requires 0.25; this is not the same schedule
```

`<path>` 在 `load()` 里是文件路径，在 `run()` 里是**发布目标** `.../best.pt`。

**禁止的几种"修法"都没有采用**：不自动修补字段、不清 `best_dirty`、不丢弃候选、
不改用当前状态、不用当前优化器补齐、不清空已有动量。

## 5. 改了什么

### `src/spherical_blast_pinn/training/checkpoint.py`

| 新增 / 修改 | 作用 |
|---|---|
| `validate_optimizer_state(...)` | 优化器的**结构**校验（纯结构，不 import torch）：`param_groups`/`state` 必须都在；参数组数量、每个组的参数标识与当前优化器**逐组相等**；`state` 的键必须是这套标识之一；逐参数状态必须是张量、形状要与该参数相容、数值要有限 |
| `_is_int_like` / `STATE_OPTIMIZER_KEYS` | `True` 不算参数标识；必需键的次序照**真实读取路径**排（`load_state_dict` 先读 `param_groups`，所以 `{}` 报的就是它） |
| `_require_one_moment(...)` | 把"步号三处一致"抽出来，**提前**到完整状态校验之前执行（理由见 §6.1） |
| `validate_state_snapshot(...)` | 签名扩展：`known_stage_names` → **`stage_windows`**（键就是"这套日程里真有的阶段名"，严格更强）；新增必填的 `optimizer_validator` / `rng_state_validator` / `window_at_step`；新增**全局随机可用性**校验与**阶段/窗口/步号一致性**（从 `load()` 搬来，见下） |
| `require_best_candidate_consistency(...)` | 上下文参数原样转给 `validate_state_snapshot`，候选与顶层载荷仍然**共用同一条**规则 |

### `src/spherical_blast_pinn/training/trainer.py`

| 位置 | 改动 |
|---|---|
| `_validate_global_rng_state()`（模块级，新） | 用**独立的临时生成器**（`np.random.RandomState()` / `torch.Generator()` / `random.Random()`）验证状态真能被接受。**不改全局再恢复**——失败路径正是最需要全局序列保持原样的时候。`python` 按契约可选，给了就要能用 |
| `Trainer._optimizer_reference()` | 从**当前优化器**取参数组标识与逐参数形状（`state_dict()['param_groups'][i]['params']` 的序号与扁平参数列表一一对应，实测确认） |
| `Trainer._probe_optimizer_step()` | 在一份**隔离的**探针优化器上**真走一步**。三层隔离：参数**克隆**、优化器**新建**、状态字典**深拷贝**（`load_state_dict` 在同 dtype/同设备时会把张量**原样接过去**，不深拷贝的话探针那一步的原地更新就改到了候选自己的动量上） |
| `Trainer._validate_optimizer_state()` | 把上两者串成共享校验要的那个回调 |
| `Trainer._snapshot_rules()` | 校验上下文的**唯一构造点**：模型结构、四个验证器、`stage_windows`、`window_at_step`。三个调用点都用 `**self._snapshot_rules()` 展开 |
| `Trainer.load()` | 内联的**阶段/窗口一致性规则整段删除**（搬进共享校验）；`known_stage_names` 参数不再需要，随之删掉 `_known_stage_names()`；顺手删掉只在那里用过的 `stored_step` |
| `Trainer._commit_checkpoint()` / `Trainer.run()` | 同上改为传上下文。`run()` 的调用仍在 `_begin_publish()` **之前** |

### `AGENTS.md`

第二项规则里加了"非模型状态也要真查"一段（三条反例的表格、优化器/全局随机/
阶段窗口三小节、探针与独立生成器的理由）。

### `tests/test_checkpoint_prediction_chain.py`

新增 15 个用例（§6），并把"接收实例不该被改动"的快照从"模型+步号+**学习率**"
扩到**完整优化器状态**（参数组逐键 + 动量逐参数逐张量）、阶段、采样器 RNG、
**全局随机状态**。

## 5.1 `load()` 里那条规则搬到了哪里

搬进 `validate_state_snapshot`，**内容一字未改**（两条精确等值分支：未阻塞按
步号所属阶段、已阻塞按未切走的那个阶段），并且**用被检查快照自己的步号**。
`load()` 里那份**已删除** —— 同一份规则不留两处，这正是候选一路漏掉整整一套
规则的原因。

搬完两条老测试仍要过：`tests/test_stage_machine.py` 里 `match="not the same
schedule"` 的措辞**原样保留**。

## 6. 我自己改出来的四个错，都当场修掉了

诚实记录（其中 6.1 是**既有测试当场抓住的真回归**）：

1. **诊断次序回归。** 我把"步号三处一致"留在快照校验之后，于是既有测试
   `test_a_candidate_that_does_not_match_the_best_record_is_refused`（把候选步号
   抬高 5）报出来的不再是 `different moments`，而是"窗口与日程对不上"——那个被抬高的
   步号**同时**违反两条，而后者只是同一个改动的**下游症状**。修法：把步号一致性
   抽成 `_require_one_moment()` 并**提前**执行；类型不对时它保持沉默，仍由状态校验
   按类型报。现在 `require_best_candidate_consistency` 的执行次序是 **3 → 1 → 2 → 4**，
   已写进函数 docstring。
2. **报错指向的键不对。** 我把必需键排成 `("state", "param_groups")`，于是
   `{}` 报的是 `KeyError('state')`；而真实的 `load_state_dict` **先读
   `param_groups`**（复核实测的报错就是它）。修法：次序照真实读取路径排。
3. **我把"候选的动量"和"恢复出来的动量"混为一谈**（两条新测试的断言写错了）。
   `load()` 恢复的是**顶层**状态（当前参数，第 2 步），候选只是作为"待发布最佳"
   被带着走——两者本来就是两个时刻。修法：分别断言
   `fresh.optimizer.state_dict() == payload["optimizer"]` 与
   `fresh._best_candidate["optimizer"] == candidate["optimizer"]`。
4. **测试里的字典推导写错**：`write_bytes(...) or read_bytes()` —— `write_bytes`
   返回写入字节数（真值），`or` 短路，于是"基准值"变成了 `16`。
5. **报错没把字段名说全。** 复核要求"错误信息包含文件或发布目标、
   `pending_best_candidate` 下的**具体字段**及原因"，而我第一版只说
   "the global RNG state is not usable: torch: ..." 与 "the snapshot's stage
   window is ..."。改成 `rng['torch']` 与 `stage.window_fraction` 两个**点号路径**，
   这样从报错行本身就能定位到字段，不用去读代码。

## 7. 验收测试（`tests/test_checkpoint_prediction_chain.py`，新增 15 个用例）

| 复核要求 | 测试 |
|---|---|
| 三条反例在**共享校验**与 `Trainer.load()` 中都被拒绝 | `test_a_candidate_with_a_corrupt_non_model_field_is_refused`（3 个参数化） |
| A 单独一条 | `test_a_candidate_with_an_empty_optimizer_is_refused` |
| B 单独一条 | `test_a_candidate_whose_global_rng_is_not_usable_is_refused` |
| C 单独一条（并断言夹具的候选确实在第 0 步） | `test_a_candidate_whose_stage_window_does_not_match_its_step_is_refused` |
| 正常加载后**在内存中**施加同样变异，发布前也被拒绝 | `test_a_corrupt_non_model_field_in_memory_is_refused_before_publishing`（3 个参数化；**控制评分**让坏候选不被新候选顶掉） |
| 拒绝发布时：**不创建**标记或 `best.pt`，**不覆盖**已有发布产物 | 同上（`last.pt`/`checks.json`/`run_manifest.json` 预先写入并比对字节） |
| 拒绝加载后：模型、**完整优化器状态**、阶段、候选、历史、采样器及**全局随机状态**不变 | `test_refusing_a_corrupt_candidate_leaves_every_piece_of_the_receiver_alone` + 原有 `..._leaves_the_receiver_and_the_files_untouched` |
| **合法初始空优化器 `state` 仍可用** | `test_a_candidate_with_an_empty_optimizer_state_is_still_legal`（恢复后还要真能走一步） |
| 已执行更新、**带动量**的合法候选可恢复并继续更新 | `test_a_candidate_with_momentum_can_be_restored_and_keeps_updating` |
| 合法**较早**候选仍可发布，`best.pt` 能恢复**并**预测 | `test_an_earlier_candidate_with_momentum_is_published_and_predictable` |
| 原来的三条反例**继续**被拒绝 | `test_the_original_three_counterexamples_are_still_refused` |
| 联检脚本版本/项数与报告一致 | `test_the_delivered_snake_script_declares_v5_and_carries_the_three_added_checks`（见 §8） |

**正例是从实际构造生成的**（真实 `verify()`/`_save()`/`run()`），不是改坏文件
造出来的；每个变体与对照各用独立副本。

## 8. 联检脚本与报告的一致性（复核第五条）

### 先核实"v5 原文件是否存在"

**存在。** 本机 `scripts/bounded_snake_check.py`：

* 声明 `SNAKE_CHECK_VERSION = 5`；
* 实际记录 **30** 项检查，`ok: true`，**exit 0**。

### 新增的那三项，逐项核对

`REVIEW_RESPONSE_3e4e1c61.md` 的"逐项联检结果（27/27）"一节**逐条列出了 v4 的
27 个检查名**，可以直接比对，不用猜：

| v4（27 项，来自 `REVIEW_RESPONSE_3e4e1c61.md`） | v5（30 项，本机实测） |
|---|---|
| 27 项全部 | **逐名全在**，且**相对次序不变** |
| — | **新增 3 项**：<br>`prediction_entry_runs_from_a_training_checkpoint`<br>`the_model_prediction_can_be_compared_against_the_reference`<br>`the_same_uncommitted_checkpoint_is_refused_by_both_entries` |

27 + 3 = 30，与实测一致；新增的三项正好对应上一轮那一节（§1 训练检查点进预测
入口、预测可对照、§2 未提交检查点两个入口都拒绝）。文件体积也从报告里记的 v4
`49749`（LF）长到现在的 `56618`（LF）。

这条比对现在被**钉进测试**（`test_the_delivered_snake_script_declares_v5_and_carries_the_three_added_checks`）：
v4 的 27 个名字写死在测试里，版本号写死为 5——脚本被改回 v4、或任何一项被删掉，
测试立刻失败，不再靠人去比对报告。

### 关于"远端仍是 v4"

复核看到的远端基线是 v4 / 27 项，本机是 v5 / 30 项。**这只能说明 v5 的那次改动
从没被交付出去**——这个工作树没有 `.git`，我无法提交，也无法推送（见 §0）。
**我不改版本号或数量去迎合报告**：本机文件确实是 v5、确实跑 30 项、确实 30/30。
请在有 git 的环境里把 §9 列出的文件提交上去，两端才会一致。

### 本次实跑（输出在仓库之外）

```
PYTHONPATH=src <py> scripts/bounded_snake_check.py C:/.../snake_item3_delivered
→ exit 0；snake_check.json：version 5，ok true，30 项全部 passed
```

## 9. 交付文件、命令与结果

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
PY=/e/Anaconda3/envs/pytorch/python.exe
```

| 命令 | 结果 |
|---|---|
| `$PY -m pytest -q -rs` | **692 passed, 6 skipped, 0 failed**（15:02，exit 0） |
| `$PY -m pytest -q tests/test_checkpoint_prediction_chain.py` | **52 passed**（上一轮 37 → **+15**） |
| `$PY -m pytest -q tests/test_publish_marker_contract.py tests/test_frozen_detonation_velocity.py tests/test_case_scales_association.py` | **126 passed**（本项**未修改**的三处） |
| `PYTHONPATH=src $PY scripts/bounded_snake_check.py <out>` | version 5，ok true，**30/30**，**exit 0** |
| `$PY %TEMP%/scan_item3_r2/repro_nonmodel_item3.py` | 修前 **3/3 被接受**；修后 **3/3 被拒绝**、合法对照被接受；逐条停用新检查 3/3 被重新放行 |

**关于总数**：上一轮本机同样命令是 `676 passed / 6 skipped`，本机现在 `692 / 6`。
其中 **+15 我可以逐条指名**（§7 那 15 个用例）。**剩下的 +1 我无法核实**：这个工作
树没有 `.git`，没法把当前的测试收集结果与上一轮那棵树逐文件比对。如实写在这里，
不把它算进"新增"。

6 项跳过**全部**因为真实提取数据不在本机（`F:\2\extracted_geom\50mm_d`）——
**跳过不是通过**，这里也不当作通过。

### 文件哈希（LF 规范化，按工作区实际内容计算）

| 文件 | 字节（LF） | sha256[:16] |
|---|---:|---|
| `src/spherical_blast_pinn/training/checkpoint.py` | 72614 | `0a8e2c21c662994e` |
| `src/spherical_blast_pinn/training/trainer.py` | 134468 | `3e112f9c8f0254f4` |
| `tests/test_checkpoint_prediction_chain.py` | 55953 | `0ca32b0de025c3de` |
| `scripts/bounded_snake_check.py` | 56618 | `2ba72422244ea48b` |
| `AGENTS.md` | 43822 | `0cc626f15bb8e641` |

> 本报告自身（`REVIEW_RESPONSE_746c319a.md`）的哈希**不列在这里**：写进去之后它
> 就变了。请以工作区里的实际文件为准。

`snake_check.json`（仓库外，9067 字节，LF 规范化）：
`dc46706ff2b9fd8ebf83691bc9fa5068d1b9b6ad291c43b7522e5cf1ad548b51`
输出目录：`%TEMP%\snake_item3_delivered\`（**不在仓库内**）。

## 10. 未做的 / 边界（明确标注）

* **没有升检查点版本**：结构没变，只是校验变严；坏数据不靠版本号回避。
* **没有改物理方案**，**没有重写已通过的模型指纹逻辑**（`parameter_fingerprint_of`
  一行未动），**没有重写发布事务恢复策略**，**没有放宽任何既有判据**。
* **没有运行真实数据链路**：`F:\2\extracted_geom\50mm_d` 不在本机，相关测试
  在本机**跳过**——**跳过不是通过**，这里也不当作通过。
* **没有验证提交号、没有提交/推送/核对远端**（本机无 `.git`，见 §0）。
* **没有进入第四项**。
* **一处仍然存在的边界**（与上一轮相同，如实说明）：这些校验证明的是"**这份快照
  自证一致**"——指纹与自带参数对得上、优化器状态与自带参数相容、随机状态能被
  生成器接受、窗口与自带步号属于同一套日程。它**不能**证明"这组参数真的是一次
  真实训练的产物"：一份把 `model`/`optimizer`/`rng`/`stage` **成套**改成互相
  一致的检查点，在文件层面与真品不可区分。要挡住那一类需要训练侧可信签名，
  不属本项范围。
* **写而不读的字段仍在必需清单之外**：`learning_rates`，以及 `rng` 里的 `seed`
  两项全仓库只有写入、没有读取（`grep` 确认）。把"写过"当成"必需"会让规则比
  实际契约更严，这一条与上一轮的口径一致。
