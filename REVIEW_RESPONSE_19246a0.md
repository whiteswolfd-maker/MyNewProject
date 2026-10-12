# 复核回应：检查点字段、预测入口与发布标记的归属

**本轮基线**：复核给出的 `19246a0`（上一轮 `adf1214` 到它只改复核报告）。

**本机实跑**：改动前 **514 passed / 6 skipped / 0 failed**——与复核报的
"514 passed、6 skipped" 逐项相同，确认本机工作树就是被复核的那份快照。

改动后 **529 passed / 6 skipped / 0 failed**（新增 15 条回归），
有界合成联检 **v5，30/30，exit 0**（升版：v4 是 27 项）。

**本机没有 git 仓库**，所以无法比对提交号；"本机工作树 == `19246a0`" 的依据是
测试总数与跳过原因逐项吻合。

**执行环境**：Python 3.12.12、torch 2.9.1+cu130、NumPy 2.3.5、SciPy 1.16.3（CPU）。
复核用的是 Python 3.12 + torch 2.14 CPU——版本不同，但下面每一条都在本机复现过。
换行符：仓库按 LF 计算哈希。

---

## 复核那句话，成立

> **为什么 27 项联检通过了，预测仍然有问题？** 当前联检的 `compare` 用例直接把
> 参考数组写成预测 NPZ，没有经过模型预测入口。

成立，而且是本轮四条的**共同根因**。上面那条 `compare` 检查把 `density` /
`velocity` / `pressure` 直接抄参考数组写进 NPZ，于是"对照命令能跑"被当成了
"整条预测链路能跑"——而这条链路第一次真的走一遍，第一步（`predict` 入口）就断。

---

## 1. 训练检查点不能直接用于预测 —— 高优先级

**位置**：`Trainer._build_payload()`、`predict.load_model_from_checkpoint()`。

**复核的判断成立，且比报告的还多一层。**

保存端确实**从来不写** `detonation_velocity`：改动前 `_build_payload()` 返回的
字典里有 `time_scale`、`scales`（逐算例）、`contact_pressure_scale_pa`，没有这个
顶层键。读取端却有硬要求，于是每一份**方案 B** 训练出来的检查点都被归成方案 A 的
旧文件。

改法（按提示词"统一检查点字段定义，从实际冻结的参数保存、读取并校验"）：

* 新增 `training/checkpoint.py`，**格式标识、版本号、字段名与读取规则只有这一份
  定义**，保存端（`trainer`）与读取端（`predict`）共用。以前两边各维护一份清单，
  这种脱节就没有藏身之处。
* 保存：`_build_payload()` 写入 `detonation_velocity`，取值走新的
  `_frozen_detonation_velocity()` ——**从冻结的尺度集合里取**，不取 `cases[0]`、
  不填默认值。四个算例给出不同的 `D` 时**硬失败**（共用一套尺度化网络时"一个
  `D`"才成立），不是悄悄挑一个。
* 读取：`require_detonation_velocity()` 分三个拒绝理由——**没有这个字段**
  （旧写入端没保存）、**不是数**、**不是有限正数**。三者都不回落到默认常量。
* 报错文本改掉了"was written by the scheme-A pipeline"那句错误归类。

**顺带发现一个真正的断链，它比复核列出的更早发生。** 复核说"调用预测入口报
`SetupError`"。实际在默认 `--time-end` 那条路上，**连这个错都到不了**：

```python
end_physical = physical_time_from_scaled(s_max, args.radius, time_scale=time_scale)
# TypeError: physical_time_from_scaled() missing 1 required keyword-only
#            argument: 'detonation_velocity'
```

`physical_time_from_scaled()` 的 `detonation_velocity` 是**必填**的（映射里含
`T_h = R_0/D + epsilon`），而这里从没传过。这条默认路径**从来没被测过**——正因为
没有任何一条测试真的从训练检查点走到预测。已修（`predict.py`，两处：`main()` 与
`load_model_from_checkpoint()` 都经同一条规则取值）。

**验收**：新增的"训练保存 → 预测加载 → 生成预测文件"测试走**完整的
`predict.main(argv)`**（真实 npz、真实配置、真实检查点），并核对预测文件里的
`start_time` 与**独立算出来的** `T_h = R_0/D + epsilon` 逐位相同（`8.215007215007216e-06`）
——这一点才证明 `D` 是**从检查点读的**，不是默认常量补的。

## 2. 预测入口没有检查"未提交"标记 —— 高优先级

**位置**：`predict.load_model_from_checkpoint()`。

成立：改动前只有 `Trainer.load()` 调 `_refuse_if_uncommitted`，预测入口直接
`torch.load()`。

改法：把这条规则**提取成共享读取规则** `training/checkpoint.py::refuse_if_uncommitted`，
两个入口都调它（`Trainer._refuse_if_uncommitted` 变成它的薄包装）。
标记存在却读不出来时**保守拒绝**——"读不懂标记"不能当成"没有标记"。

**验收**：新增测试对**同一个**未提交检查点断言两个入口都拒绝，且两边都死在
"uncommitted"这条理由上；另有一条覆盖"标记不可读"。联检里也加了同一项
（`the_same_uncommitted_checkpoint_is_refused_by_both_entries`）。

## 3. 后续失败操作会误删以前的未提交标记 —— 高优先级

**位置**：`refine()` 的回滚分支、`_begin_publish()`、`_end_publish()`。

成立。改动前的 `_end_publish()` 是**无条件** `unlink()`，而事务快照里没有标记
——复核的四步实测（四份文件字节未变、旧标记被删、新实例由拒绝变成功）逐个对应
这条路径。

改法（按提示词"进入新操作前识别未完成事务；标记需要明确归属"）：

* **标记记下归属**（`owner`），`_begin_publish()` 每次铸造一个新的归属标识；
  `_end_publish()` **只删自己写的那一个**，别人的一律留着。归属标识刻意**不进**
  `_snapshot_state()`：它是事务机制、不是训练状态，回滚时还原它反而让回滚方认不出
  "刚才是自己写的标记"。
* **精修开始之前先看目录**：有未完成的发布就**拒绝开始**。理由写进错误里——精修的
  事务语义建立在"开始那一刻的盘上状态"上，若那个状态本身就是新旧混合的，留底-还原
  只是把不一致**原样恢复**。恢复路径是 `run()`（它重写全部四份产物）；零预算精修
  走的正是 `run()`，所以在检查之前先返回，不受影响。
* `run()` 覆盖一次未完成的发布是**合法**的，但**不假装它没发生过**：被覆盖的那份
  记进标记的 `supersedes`，并写进运行清单的 `superseded_uncommitted_publish`。

**验收**：新增测试按复核的四步原样走——先让一次 `run()` 在写清单时失败留下标记，
再执行另一项失败操作，然后断言**标记仍在、四份文件字节一字未变、新实例仍然拒绝**。
另有三条分别钉住归属规则本身、`run()` 的覆盖记录、以及"干净目录不受影响"。

## 4. 候选缺失的损坏检查点被静默接受 —— 中优先级

**位置**：`Trainer.load()`、`_commit_checkpoint()`。

成立，机制就是这一行：

```python
self._best_dirty = bool(payload.get("best_dirty")) and (self._best_candidate is not None)
```

"候选缺失"被折成 `False`，于是复核实测的三件事同时发生：加载成功、`best_dirty`
被自动改成 `False`、最佳记录的分数还在而参数没有、再次运行也不再产出 `best.pt`。

改法：新增 `require_best_candidate_consistency()`——`best_dirty=True` 必须对应一份
**完整可用**的候选：候选是映射、带整数 `step`、带 `model` 状态；`step` 与最佳记录
的 `step` 相同；最佳记录带参数指纹。不成立就**明确拒绝**，`_commit_checkpoint()`
里那行静默清除**删掉**（一致性由被调用方负责，`load` 的只读阶段与提交阶段各查一次，
任何绕过 `load` 的路径也拦得住）。

**验收**：四条——坏输入拒绝（且**拒绝之前不改对象**）、候选步号与最佳记录对不上
时拒绝、**合法的待发布候选不被误伤且标志原样保住**、以及"拒绝发生在只读阶段因此
`fresh._best_dirty` 仍是 `False`"。

---

## 联检：补通那条实际调用链

按提示词"下一轮应优先补通这条实际调用链"，联检升到 **v5（30 项）**，新增三项：

| 项 | 断言 |
|---|---|
| `prediction_entry_runs_from_a_training_checkpoint` | 真实训练落盘 → 真实 `predict.main` → 预测文件真产生；`start_time` 与检查点里冻结的 `D` 独立重建的 `T_h` 逐位相同 |
| `the_model_prediction_can_be_compared_against_the_reference` | 拿**模型产出的那一份**喂给真实 `compare.main`，报告真写出来 |
| `the_same_uncommitted_checkpoint_is_refused_by_both_entries` | 两个入口对同一份未提交检查点都拒绝 |

**`comparable_points` 如实记为 0 并写进证据，不作为通过条件。** 这是一条冒烟尺度的
小模型（预算 18 次更新），它的前沿与参考剖面本来就不该对上；把"比不了"写成"通过"
正是这份脚本最该避免的事。上面那条新增项证明的是**链路通了**，不是**物理对上了**。

---

## 明确不改的东西

* 不宣称物理验收通过、不宣称模型可以正式训练。本轮全部是软件实现与流程问题，
  双网络物理设计一个字没动。
* 60 项里有 6 项跳过，全部因为**真实提取数据不在本机**
  （`F:\2\extracted_geom\50mm_d` 不存在）。**跳过不是通过**，这里也不当作通过。
* `_legacy_v1_backup/` 按约定保留，未清理。

## 一个需要你决定的点

**检查点版本 7 → 8**，按本仓库既有约定（"旧版本明确拒绝、无迁移路径"）执行：
新增字段是结构变化，旧检查点没有它，方案 B 的时间轴无法重建——**旧的检查点会全部
作废，需要重训**。若你希望保留某几份旧检查点，需要单独说，我不会自动加迁移。

---

# 附：死代码清理与本轮的独立复核

## 一、死代码清理

用自写的 AST 扫描器（本机没有 ruff / flake8 / vulture，也不往环境里装）扫全部
78 个 `.py`。分四类，**只删能确证的**：

| 类 | 判据 | 删除 |
|---|---|---|
| A 未使用的 import | 名字在本文件里除 import 行外零出现 | **33** 条（16 处 `src`、17 处 `tests`） |
| B 模块级私有定义零引用 | 全仓库（含 `.md`）除定义处外零出现 | **8** 个 |
| C 不可达语句 | `return`/`raise` 之后的语句 | 0（本来就没有） |
| D 单目标局部变量赋值后从不读取 | 见下 | **20** 条 |

**B 类删掉的 8 个**，其中三个的 docstring 在骗人：

* `eos.py::_cj_energy` 写着 "used only by `cj_state_estimate`"——**那个方法在仓库里
  根本不存在**，全仓库只有这句注释提到它。
* `constants.py::_PRESSURE_DIAGNOSTIC_TNT` 写着 "Used only by the `auto` detector"——
  `auto` 检测器用的是 `units.py` 里另一组窗口常量。
* `kfile_parser.py::_MATERIAL_KEYS` 与**活代码里那个字面量元组**逐字重复
  （`_parse_material_block` 用的是字面量，不是这个常量）。

其余 5 个：`_crossing_radius`（`_crossing_radius_and_gap` 的无人调用包装）、
`_weighted_shell_mean`（改成稀疏算子之前的实现）、`_softplus_inverse`、
`_valid_volume_ratio_mask`、`_fingerprint_of`（测试辅助函数）。

最离谱的一条在 `cli/audit_dyna.py`：

```python
unit_time = 1.0 if "SI" in str(audit.detail.get("unit_system_inferred")) else 1.0
```

**两个分支都是 `1.0`**，而且结果从不读取。

**D 类里"是死代码"和"只是惯用写法"要分开。** 初版扫描器报了 91 条，逐条看下来
大部分是解包后丢掉一个（`trainer, tiny = _tiny(...)`、`inner, outer = candidate.band`、
`for name, mask in ...`）——那是 Python 的惯用写法，不是死代码。加了两道过滤后剩 26 条，
再排除"删掉会抹掉线索"的，实删 20 条。

三个值得点名的：

* `extract_profiles.py` 里 `deck_ids` / `lookup` / `element_part_ids` 四行——其中
  `deck_ids` 后面还跟着一句 `del deck_ids`，**作者当年就知道它没用**。
* `trainer.py::run()` 的 `start_step = int(self.step)`——正是第八轮 §4 把判据从
  "步数涨了吗"换成"参数指纹变了吗"时留下的残骸；紧挨着的新代码的注释还写着
  "这跟'步数涨了吗'不是同一个问题"。
* `detonation.py::_advance` 的 `front_old` / `inner_old` / `u_cell_old`——几何收支
  搬到事后审计（`_step_budget` 从快照重算）之后留下的化石。

### 有 5 处我**故意没删**，理由统一

**删掉会抹掉"本该有但不在"的线索**，那比留着更糟：

| 位置 | 留着的原因 |
|---|---|
| `detonation.py::_cell_pressure` + `_timestep` 里那次死调用 | 它的 docstring 是一句**物理选择的记录**："前端单元与内部单元一样由 EOS 给出"。函数体是纯计算、全仓库只被那次死调用引用——但删了就把那句话一起删了。 |
| `extract_profiles.py::_part_index_to_deck_id` + `part_map` | docstring 记录 ALE 映射的约定："按卡片里 part 的出现顺序，不是 `index + 1`"。活路径已经改从 `deck.product_ale_indices` 取，但那是**硬争取来的**约定，不留记录不合适。 |
| `tests/test_review_regressions.py:385 usable` | 测试叫 `test_front_supervision_uses_the_same_frame_mapping`，算了 `usable` 却**从没拿它断言**。删掉这行 = 把一个"测试比名字弱"的证据抹平。 |
| `tests/test_review_regressions.py:1881 fields`（含它上面 5 行脚手架） | 测试叫 `..._uses_pointwise_eos_not_eos_of_the_mean`，算了逐点场 `fields` 却没用，实际只比较 `shell_average_of_model` 的输出与 `EOS(平均)`。同上。 |
| `tests/test_best_candidate_and_transaction.py:142 per_step` | `per_step = _record_every_step(trainer)` 有**副作用**（给 `trainer._update` 挂记录器），结果从不使用——测试名声称逐步骤核对，实际没有。 |

这五处是**测试/文档强度问题**，不是"少写了几行"；我把它们留在原位并写在这里，
交给你和复核方判断要不要补断言。

### 清理之后

| 指标 | 值 |
|---|---|
| 扫描器 | A=0，B=0，C=0；D 只剩上面那 5 处**故意保留**的 |
| 全量测试 | **529 passed / 6 skipped / 0 failed**（清理前同值） |
| 联检 | **v5，30/30，exit 0** |

## 二、这几条修复到底成没成——独立复核，不是重跑我自己的测试

`tests/` 里那些测试是我写的，它们通过只说明"代码符合我的预期"。所以另写了一份
**A/B 复核**（放在仓库外，不进版本库），把修改前的行为**逐条还原回去**，看复核
报告里的症状会不会重现：

| 项 | 把旧行为还原回去 | 换回当前代码 |
|---|---|---|
| §1 | 保存端不写该字段 → 旧读取端给出**复核报告里那句** `SetupError ... scheme-A pipeline` | 载荷带 `detonation_velocity = 6930.0`（冻结值）；`predict.main` exit 0 并写出 `predict_R0050mm.npz` |
| §2 | 旧预测入口（无标记检查）→ **接受了**未提交的检查点 | 预测入口拒绝、训练恢复入口拒绝（同一条规则） |
| §3 | 旧 `_end_publish`（无条件 `unlink`）→ 别人的标记被删掉，**目录由"拒绝加载"变成"加载成功"** | 复核四步原样走：`run()` 失败留下标记 → 后续失败操作被拒绝开始 → 标记仍在、四份文件**字节未变**、新实例仍拒绝；`_end_publish` 对别人的标记返回 `False` 并保留 |
| §4 | 旧提交逻辑 → 标志被静默改成 `False`，最佳记录的分数还在、参数却没有 | 明确拒绝、且**拒绝发生在只读阶段**（对象没被改）；合法的待发布候选不被误伤 |

**结果：17/17 通过，0 失败。** 症状在旧行为下全部重现、在当前代码下全部消失——
说明修复打在病灶上，测试也不是空转。

复现方式：临时把 `Trainer._build_payload` / `_end_publish` 换回旧实现、用旧读取端
逐字重放。脚本在仓库外（`%TEMP%\scan_19246a0\verify_round_19246a0.py`），不污染
交付物。**中途它自己错过两次**（一次把 `_tiny_trainer` 当元组解包；一次探针实例
用了长日程、于是 `load` 先在日程上拒绝而不是在标记上）——两次都是**复核脚本的
夹具问题**，不是产品缺陷，已修正后重跑。

