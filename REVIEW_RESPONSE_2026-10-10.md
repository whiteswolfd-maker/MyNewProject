# 对 2026-10-10 复核意见的逐条回应

复核针对提交 `9bdbcaf3`，列出 6 项可复现的软件错误。**六项全部复现，全部已修**，
每项配一条"旧实现会通过、新实现会失败"的回归测试。测试 254 → **271 passed**。

复核的基本判断是对的：上一轮把"能改的改完，只剩研究问题"说成结论，不成立——
新字段读进来了但没有在取数的地方生效，闸门也只堵住了"属性不存在"那一种缺失。
下面逐条写清楚**根因**，而不只是"改了哪里"。

---

## 1. 逐字段掩码没有被采样消费 —— 确认，已修

**根因**：`sampler.py` 的取数掩码是
`(case.material[frame] == medium) & case.valid[frame]`。`case.valid` 是"四项场量
联合可用"的掩码，于是 §11.1 的承诺——一个字段缺失不丢掉其他字段——在**取数的地方**
根本没有兑现。新加的 `field_validity` 只被 `contract.py` 读，等于让契约检查变绿。

**修法**：`_DATA_FIELDS` 把"批次键 / 逻辑字段名 / `field_validity` 键"绑到同一个
来源；取点时按字段各自求掩码，位置取并集，某个字段在某个位置不可用就**只把那个字段
置 NaN**，位置与其余字段保留。使用的掩码档位记进 `batch.diagnostics["data_validity_source"]`
（`per-field` / `joint`），退化与等价在报表上不再长得一样。半张 `field_validity`
映射**抛错**，不退回联合掩码。

**如实说明**：在这批重提数据上，逐字段掩码与联合掩码**逐点相同**
（50 mm：0/15000 个位置不同），所以修完之后这些数值**没有变化**。这是一个正确性
修复，不是一次数值改进；把它说成后者会是另一种假完成。

回归测试 `tests/test_validity_consumption.py`：两个方向各一条——联合掩码说不可用而
字段掩码说可用时必须**保留**，反之必须**丢掉**。旧实现在这两条上都会失败。

---

## 2. 前沿资格没有接进读取逻辑，两条前沿仍然耦合 —— 确认，已修

**根因**有三层，逐层修：

* **耦合**：`contact_valid` 里带着 `isfinite(shock_radius) & (shock_radius > contact_radius)`。
  "接触面在激波内侧"是**空气层**的要求，不是接触面标签的前提。现在
  `contact_valid` 只由接触面自己决定，`front_ordering_ok` 单独算、单独给空气组用。
* **资格数组没生效**：读取器只看 `contact_state`/`shock_state`。现在优先用
  `contact_has_supervision`/`shock_has_supervision`，其次三态码，两者都没有才是
  最弱的一档——而那一档被标成 `UNVERIFIED` 并由契约拒绝。用了哪一档写进
  `source["front_eligibility_source"]`。
* **长度错误时退回**：以前"忽略数组、退回更弱的判据"。退回等于把一份**对不上**的
  导出当成一份"证据更少"的导出，而后者仍会被当同一批数据用。现在直接抛
  `DataError`。

回归测试：`test_the_eligibility_arrays_win_over_the_detection_code`（三态码说 2、
资格数组说不够格 ⇒ 以资格为准）、`test_a_missing_shock_does_not_strike_out_a_usable_contact`、
`test_a_length_mismatch_in_the_companion_is_refused_not_downgraded`。

---

## 3. 契约会把不完整或非法的记录判成 ready —— 确认，已修

逐项对应复核的表：

| 输入 | 以前 | 现在 |
|---|---|---|
| `field_validity={}` | ready | 拒绝：缺 `density/velocity/specific_energy` |
| `symmetry_factor=-8` | ready | 拒绝：不是有限正数 |
| `symmetry_factor=NaN` | ready | 拒绝：不是有限正数 |
| 因子与份额互相矛盾 | ready | 拒绝：互为倒数的两个数应当相乘得 1 |
| 来源写成 `"not from volume fraction"` | ready | 看**布尔标志**，不看字符串 |

最后一条的根因值得单说：判据是 `"volume fraction" in source.lower()`，
`"not from volume fraction"` 里也含 `"volume fraction"`——一句明确的**否认**会被读成
**肯定**，而且读起来完全正常。现在提取器写出机器可判的
`material_partition_is_volume_fraction_based`，契约只认这个布尔量；说明文字怎么写
都不影响结论（两个方向都有测试）。

**NaN 掩码读成全 True**：`np.asarray(x, dtype=bool)` 会把 `nan` 变成 `True`。现在先验
dtype 必须是 `bool`，否则拒绝。

回归测试：`tests/test_data_contract_fields.py` 新增 4 组（含 3 个参数化用例）。

---

## 4. 阶段边界上的检查点被自己的恢复函数拒绝 —— 确认，已修

**根因**是状态管理，不是数值：更新完成后先 `self.step += 1`、再保存，而切阶段发生在
**下一轮循环的开头**。于是第 2000 步的检查点里存的是阶段 1 的窗口（0.25），而恢复
函数按步数推算出阶段 2（0.65），直接拒绝。默认每 500 步保存一次，2000 与 5000 恰好
是两个边界——**默认配置下必然发生**。

**修法**：把切换挪到"步数一增加"之后，使 `步数 ↔ 阶段窗口` 在任何时刻保存的检查点里
都成立。同时用**阶段结束步**（而不是步数区间）判断该不该推进：`--max-steps` 截断的
联检停在半路，按区间会算出"已经进入下一阶段"，拿没训完的模型去审上一阶段的判据，
把"没跑完"报成"没通过"。

回归测试：`test_a_checkpoint_written_exactly_on_a_stage_boundary_can_be_resumed`、
`test_the_saved_window_always_matches_the_step`（逐步存、逐步读、逐步核对窗口）。

---

## 5. "恢复可复现"测试没有测试恢复后的继续执行 —— 确认，已修

复核说得准确：原测试里 C 加载后**没有再跑一步**，所以它验证的是保存与加载，不是
恢复。第 4 条的边界错误正是从它下面漏过去的。

现在改成：A 连跑 4 步；B 跑 2 步、存盘；C 从盘恢复、**再跑 2 步**；然后比较 C 与 A
的权重**和**最终检查指标。

---

## 6. 严格盲测允许空保护清单 —— 确认，已修

`strict_blind_test: true` + `final_validation_radii_mm: []` 会让 `is_protected` 恒为假，
整套检查落空并返回 `ok=True`——**即使训练列表里就有本该隔离的算例**。现在空清单
（无论严格与否）都判不通过，非正/非有限的半径值也拒绝。

另外补上复核提到的收尾项：`configs/base.yaml` 的 `data.cases` 仍指向旧导出目录，
已在文件里写明**它们尚未迁移**、契约会拒绝它们（训练入口返回码 4）、以及重提命令
的位置——"路径还指着旧的"不会变成"拿旧数据训练了"，但也不该被读成"迁移已完成"。

回归测试：`test_an_empty_protected_list_makes_the_whole_check_vacuous`（两个参数）、
`test_a_non_positive_protected_radius_is_refused`。

---

## 关于"研究性阻塞"的表述

复核指出上一份报告把"残差来源未定论 → 相容离散未实施 → 网格收敛不通过 → 参考场
不验收"写成了一条必然的因果链，而现有报告尚未证明这条链。**这个批评成立，已更正**：
这三项是**并列的待核实项**，不是彼此推出的结论。已知的只有否定结果——残差不在内部
单元、不在动量功恒等式、不在前胞功写法——以及参考场当前判为 NOT ACCEPTED。它们各自
的状态由各自的那份文档记录（`DISCRETE_BUDGET.md`、`PHYSICS_BASIS.md`）。

---

## 仍未解决

* 研究性的三项（同上），无新证据不推进；
* 80 mm 缺 deck；50/100 mm 只重提了代表状态；500 mm 未提取；
* 全量重提、§16.2 compare 闭环、§16.3 时程诊断、L-BFGS 实跑未做。
