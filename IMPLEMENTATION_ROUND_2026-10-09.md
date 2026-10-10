# 实现回合报告：§9 / §14 / §15 / §16

日期 2026-10-09。本轮把"能解决的"逐项做完，**没有**动研究性的那个阻塞。
本机测试：`PYTHONPATH=src pytest -q` → **254 passed**（本轮开始时 186 passed）。

全文的结论按三档分开写，因为它们混在一起时最容易读成一句含糊的"差不多了"：

* **已解决**：有代码、有回归测试、有在真实数据上跑出来的数；
* **未解决（工程）**：还没做，但知道怎么做；
* **未解决（研究）**：没有新证据之前做不了，也不该绕过。

---

## 1. §14 三阶段训练机

### 1.1 之前缺什么

`training/schedule.py` 只有一张三行的表（2000 / 3000 / 15000 次更新，`lambda_PDE`
与学习率的爬坡），它的文档还明确写着"每个阶段都覆盖完整的目标时间区间"——而那正是
§14 要求**扩大**的东西。`trainer.py` 里没有任何时间窗口、没有阶段判据、也没有
"预算耗尽仍不通过就明确停止"的行为。

### 1.2 现在是什么

**时间窗口（§14.2）。** `Stage` 增加 `time_window_fraction`，三个阶段分别是
0.25 / 0.65 / 1.0。窗口区间恒为 `[0, W]`，扩大只把右端往外推、**不动左端**，
所以早期采样点一直留在池子里——这就是"保留早期采样"的可执行含义。

三条守护：

* 日程构造时校验窗口**单调不减**（`_assert_window_is_non_decreasing`）；
* `RadiusSampler.set_time_window` 拒绝收窄（第一次是"设定"，之后才是"推进"）；
* `window_frames` 把"保留早期采样"变成可以逐帧核对的集合。

**不把预测冻结成真初值（§14.2）。** 初值标签始终取自参考 B 初场。回归测试从三个
方向钉住它：同一批点上换窗口标签不变、等于参考场的独立重算、**不等于**网络自己的
输出。

**阶段判据（§14）。** `STAGE_CRITERIA` 在训练开始之前就固定，限额直接引用 §16 的
容差表（`initial_relative_l2` / `initial_gap_relative` / `pde_rms` /
`field_relative_l2`），不另立第二套数。未通过时 `StageNotPassed` **明确停止**，
不静默跳阶段。

判据在**本阶段的窗口内**评价：检查集只建一次、全程复用（§13），阶段验收时按时间
坐标裁剪同一批固定点（`restrict_to_time_window`），不重采。选的模型仍然只在**完整**
检查集上选（`update_best=False`）。

### 1.3 证据

* `tests/test_stage_machine.py` 共 26 条，覆盖窗口单调性、早期帧保留、逐字段时间
  坐标裁剪（含 `s_product` 与 `s_air` 同长不同序的陷阱）、初值不变性、判据的
  保守失败、阶段拒绝推进、以及一次小预算的三阶段全穿越。
* 小预算联检（§14"只做小预算验证阶段机"）：三个阶段的窗口 0.25 → 0.65 → 1.0
  逐级扩大，每一步 batch 自报的时间上端与阶段窗口一致，平衡器在预热后确实产出了
  权重。**这不是训练**，配置里也不含任何精度主张。

---

## 2. §14 损失处理：有界梯度平衡

原方案（`_legacy_v1_backup/losses/adaptive_weights.py`）的分母是 `norm + 1e-12`：
一个梯度范数 1e-13 的项会拿到约 5e11 的目标权重，虽然随后被 `max_weight` 截住，
但"截住"意味着它长期贴在**上限**上——等于把一个没有信息的项钉成最大权重。

`training/balancing.py` 按当前接口重写，补上 §14 明确要求、原方案没有的三件事：

1. **不拿近零梯度做除数**：范数低于"最大范数 × `floor_ratio`"的组判为"梯度证据
   不足"，保持上一轮权重，不做除法、不更新移动平均；
2. **零样本组不参与、也不更新统计**：缺标签/零样本的组，其未加权张量是
   `requires_grad=False` 的常数，跳过它，不把 0 当成一次真实测量；
3. **监测权重上下限与未加权残差**：逐项记录触限次数、被保持的原因、以及未加权残差。

`lambda_PDE` **不**由平衡器接管：它的 0.1 → 1 爬坡是 §13 规定的，两个机制写同一个
数时结果取决于更新顺序。测试 `test_the_balancer_never_takes_over_the_pde_weight`
把这条固定下来。

---

## 3. §16 验证隔离

### 3.1 之前的问题

`configs/base.yaml` 里写着 `splits.strict_blind_test: true`，而**没有任何代码读
它**。500 mm 的 `split: extrapolation_test` 也只是一个字符串——把它改成 `train`，
什么都不会报错，500 mm 就悄悄进了训练集，而所有报表仍然写着"盲测半径已隔离"。

### 3.2 现在是什么

`data/isolation.py` 把这句话变成四件具体的检查：

1. 被声明为最终验证半径的算例，`split` 不能是 `train`；
2. 训练算例的半径集合与最终验证半径集合**不相交**（不看 split 字符串，独立核一遍）；
3. 参考 B 初场建立在**训练半径**上（初值标定也是一次"用过了"）；
4. 声明了 `strict_blind_test` 就必须真的存在一个非训练的最终验证算例。

半径清单写进配置（`splits.final_validation_radii_mm: [500.0]`），训练入口在
`train.py` 硬失败（返回码 5）。`predict` 那一侧的隔离由命名空间检查守着：算例加载
的名字不允许出现在 `cli/predict.py` 里。

---

## 4. §9 接触面逐帧几何重构

### 4.1 之前是什么

接触面位置来自 `alpha = 0.5` 的壳平均穿越——§9 明确说它"不能继续充当最终几何重构"。

### 4.2 现在是什么

`data/interface_reconstruction.py`：球对称 PLIC 的闭式切分。单元是球壳 `[a, b]`，
产物在内侧、占体积分数 `alpha`，界面半径
`r = (a³ + alpha (b³ − a³))^{1/3}`。

边界处理逐条对应 §9：纯相相邻（界面恰在公共面，状态 `on_shared_face`）、缺邻域、
混合带不连续、邻域的相不是预期的相、空壳层（体积分数是 `nan` 的层**没有相信息**，
不是空气）。定位不了就返回状态 `unresolved` + 区间 + 原因，**不给半径**。

报告分开给：`detection` / `position_uncertainty` / `supervision_eligible`，与激波
是否可用无关。`alpha = 0.5` 穿越降级为 `shell_crossing_radius`，只作辅助对照。

三个诚实的限制写在模块文档里：

* 逐单元的体积分数核对在闭式解下**必然是浮点尾数量级**（`r` 是由 `alpha` 反解的）。
  它守的是公式实现，不是"数据与网格自洽"——后者从一个量里根本看不出来。这一点在
  文档里写清楚了，没有让它读起来像一项物理校验。
* 重构出来的是**亚单元**位置时 `sub_cell_position` 置位，并写明"没有让原始数据多出
  任何分辨率"。
* 径向平均过的数据没有方向信息，`direction_spread` 报 `nan` 加原因，**不填 0**
  （填 0 会读成"界面完美球对称"，一个从未被测过的结论）。

### 4.3 真实数据上的验证

提取器已改为由重构给出接触面（`cli/extract_profiles.py`），并输出
`contact_reconstruction_status` / `contact_position_uncertainty_m` /
`contact_mixing_band_*` / `contact_mixing_cell_count` /
`contact_shell_crossing_radius_m` / `contact_has_supervision`。

50 mm、6 个代表状态（t = 0 / 20 / 100 / 300 / 502 / 1000 µs）实际跑出来的结果：

| 读数 | 几何重构 | 0.5 壳平均穿越 |
|---|---|---|
| t = 0 相对 `R_0 = 50 mm` 的偏差 | **−3.32e-7 m（−0.00033 mm）** | +7.73e-6 m（+0.0077 mm） |

`t = 0` 时接触面按定义就在 `R_0`，所以这两个数都是**不用任何模型**的直接误差读数。
几何重构比壳平均穿越好 **23 倍**。这不是一句主张，元数据里两个数并排存放
（`contact_initial_offset_m` 与 `contact_initial_offset_shell_crossing_m`），可以自己复算。

六帧的定位状态全部 `reconstructed`、全部有监督资格、全部是亚单元位置（不确定性
如实报为混合带的一半，不写成 0）。

---

## 5. §15 数据契约：现在在真实数据上就绪

### 5.1 之前

真实数据上契约判 `not ready`，五条理由。其中两条是**提取器没写**，三条是**读的
一侧没接**。

### 5.2 现在

提取器补齐并接线：球对称份额（`symmetry_factor`，来自它实际用的壳层网格）、逐字段
有效性（`field_valid_*`，先在"位置可用"上分开再各自与场量的有限性相交）、前沿监督
资格（两条前沿各一份，检测状态与资格分开）。

读的一侧新增 `ExtractedCase.field_validity` 与 `.front_supervision_eligibility`，
契约的判据从 `hasattr` 改成 `is None`——这两个属性是带默认值的字段，`hasattr` 恒为
真，用它做判据会让"这份导出没记录"和"这份导出都有效"变成同一件事。

**重新提取后的实际结果**：

```
50 mm  (F:/2/extracted_geom/50mm_c)   ready = True
   symmetry_factor = 8.0,  detonation_velocity = 6930.0
   material_masks_from = ALE material volume fraction (d3plot history variables 1..7)
   field_validity_fields = [density, pressure, specific_energy, velocity]
   front_supervision_eligibility = contact{reconstructed:6}, shock{confirmed:2, doubtful:3, not_detected:1}
   可监督帧 5 / 6

100 mm (F:/2/extracted_geom/100mm)    ready = True
   可监督帧 2 / 3
```

§16.1 第 1–3 步（先改导出语义 → 小量代表状态提取 → 读取—导出—重读—采样—观测
闭环）已完成，闭环在真实数据上跑通并由 `tests/test_real_data_chain.py` 守着。

---

## 6. 仍未解决

### 6.1 研究性（没有新证据之前做不了）

**这三项是并列的待核实项，不是彼此推出的结论。** 上一版把它们写成了一条因果链，
那是一个未经证明的断言，已于 2026-10-10 更正（见 `REVIEW_RESPONSE_2026-10-10.md`
最后一节）。

* **离散能量残差 0.41 % 的来源尚未定论**。已知的只有**否定结果**：不在内部单元、
  不在动量功恒等式、不在前胞功写法。但那几个排除实验**只排除了那几个特定改动**，
  不证明基准里对应项是对的。没有新证据之前不去凑一个能过的数字。
* **相容离散未实施、网格收敛不通过**：各自独立记录在 `DISCRETE_BUDGET.md`。
* **`physics_basis_status` 未核实**：恒定爆速强支与整个球形流动的相容性论证没有
  文献支撑（`PHYSICS_BASIS.md`）。

三者的共同后果是：**正式初值与正式训练仍不能放行**，`assert_reference_accepted`
依然拒绝。这一条被 `test_the_reference_gate_still_refuses_on_real_data` 钉住——
任何一天有人把它放宽到能过，测试会立刻失败。

### 6.2 工程性（还没做）

| 项 | 状态 |
|---|---|
| 全量重新提取 50/80/100 mm | 只做了代表状态（6 / 3 帧）。全量命令见第 7 节 |
| 500 mm 提取 | d3plot 在 `D:/500mmtnt/_ProjectScratch/Scr59F2/`，**已确认可访问**；本轮未提取，因为它只做最终独立验证，不属于"规则还变动时启动的长任务" |
| §16.2 predict/compare 闭环 | predict 隔离已有（命名空间 + 接口检查）；compare 的超压时程、峰值衰减、分项误差表尚未本轮复核 |
| §16.3 独立物理诊断的均值/最大值/时间曲线输出 | 诊断本身在 `evaluation/checks.py`，输出的时程曲线未本轮复核 |
| §17.2 定向回归的剩余行 | 本轮新增 5 个测试文件；逐行对照表未重做 |
| 完整训练 | **按规范不做**。三阶段机只做了小预算验证 |
| L-BFGS 精修 | 未在真实数据上跑过 |

---

## 7. 可运行的命令

```bash
# 重新提取（规则已冻结；--states 为早期/中期/末期的代表状态）
PYTHONPATH=src python -m spherical_blast_pinn.cli.extract_profiles \
  --config configs/base.yaml --case tnt_50mm \
  --deck  F:/tnt/tnt_files/dp0/SYS/MECH/input.k \
  --d3plot-dir F:/tnt/tnt_files/dp0/SYS/MECH \
  --out F:/2/extracted_geom/50mm_full --charge-radius 0.05 --stride 10

PYTHONPATH=src python -m spherical_blast_pinn.cli.extract_profiles \
  --config configs/base.yaml --case tnt_100mm \
  --deck F:/2/decks/tnt_spherical_100mm.k \
  --d3plot-dir F:/2/runs/100mm_full \
  --out F:/2/extracted_geom/100mm_full --charge-radius 0.10 --stride 10

# 小预算联检（不是训练）
PYTHONPATH=src python -m spherical_blast_pinn.cli.train \
  --config configs/experiments/geom_smoke.yaml

# 全部回归
PYTHONPATH=src python -m pytest -q
```

**80 mm 缺的是 deck，不是 d3plot。** `C:/Users/work/Documents/GitHub/2/sim_data/` 下有
一族 5.8 GB 的 d3plot（尺寸与旧导出 `extracted_80mm` 记的 `R_0 = 0.08`、`n_t = 483`
相符，但**没有 deck 就无法确认**它是不是那个算例）。提取器需要 `--deck` 才能把卡片里的
part id 映射到 d3plot 声明的 ALE 材料序号——`extract_profiles.py` 里有一条硬检查，
数量对不上就拒绝继续（按错的列组合体积分数会得到一份看起来正常的数据）。
已找过的位置：`F:/2/decks`、`F:/2/runs/*`、`C:/Users/work/Documents/GitHub/2`、
`D:/500mmtnt`——只有 50 / 100 / 500 mm 的 deck。**这是未执行项，不是失败**；
拿到 80 mm 的 `input.k` 之后，命令同上、把 `--case/--deck/--d3plot-dir` 换掉即可。

---

## 8. 本轮新增的测试

| 文件 | 条数 | 守什么 |
|---|---|---|
| `tests/test_stage_machine.py` | 26 | §14 窗口、判据、平衡器 |
| `tests/test_interface_reconstruction.py` | 19 | §9 重构的每一条边界 |
| `tests/test_isolation.py` | 9 | §16 四种破坏方式 + predict 隔离 |
| `tests/test_data_contract_fields.py` | 9 | §15 契约的每个分支 |
| `tests/test_real_data_chain.py` | 5 | 真实数据整链 + 研究闸门仍然关着 |
