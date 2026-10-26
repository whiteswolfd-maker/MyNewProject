# 初值决策：均匀高压产物球 + 静止空气（2026-10-21）

**状态**：已由用户确认。零时刻初值仍适用于
`src/spherical_blast_pinn/physics/two_medium_initial.py`。

**2026-10-25 更新**：小正时间启动已从旧 Lagrangian `TwoMediumStartup.run_to`
改为**局部平面 JWL—空气黎曼近似**（见下文“2026-10-25”一节）。

**2026-10-26 更新（当前）**：采样器与接触压力尺度已经直接接到新启动场：

* `RadiusSampler(field, scales, tau_end)` 由 `PlanarStartupField` 生成
  IC / PDE / 接触面 / 激波点；`tau_end` 显式给出，不再由数据最后帧决定；
  真实数据监督尚未迁移，传入算例会明确报未支持。
* `contact_pressure_scale(field)` 取场自己的 `p_*`，只作为损失归一化尺度。
* `cli.check_startup_chain` 完成“磁盘 handoff → 完整批次 → total_loss →
  一次 backward”的装配检查；不训练、不写检查点、不改变 `training_qualified`。
* `Trainer` 构造即抛 `TrainingNotConnected`，正式训练/预测仍未接入。
* 检查点迁移到 `spherical_blast_pinn.startup_checkpoint` v1：只绑定启动场身份与
  `PlanarStartupScales`，保存模型/优化器/步数/随机状态；旧格式（带 `D` 与 `T_h`）
  明确拒绝。独立接口 `save_startup_checkpoint` / `load_startup_checkpoint` 已通过
  落盘往返与短程恢复一致性测试。
* 检查点同时保存**采样配置与动态窗口**（`tau_end`、点数、分层比例、
  `r_min_fraction`、当前窗口分数与历史）。恢复前先核对静态配置与接收采样器的
  启动场/尺度；不同 `tau_end` 或采样设置、或换了启动场，都在修改模型、优化器
  与采样器之前拒绝。窗口按保存值无条件恢复（允许相对接收端默认值收窄），
  因此非默认阶段窗口续训后的下一批采样与连续运行逐位一致。

旧的方案 B 初始化/前置求解链路已整体删除：
`physics/detonation.py`、`physics/precursor.py`（含 `ReferenceField` 与验收缓存）、
`physics/two_medium_startup.py`（旧 Lagrangian 推进）、`cli/reference_field.py`、
`cli/validate_startup.py`、`cli/diagnose_startup.py` 及其专属测试与配置节都已
移除。下面“与现有文档的关系”“仍未接入正式训练”两处旧表述以本节为准。

## 决策

物理时间 `t_phys = 0` 的初值改为**分段常数两介质球对称场**：

- `0 <= r < R_0`：爆轰产物。常数密度 `rho_p0`、零速度、常数比内能
  `e_p0 = E0/rho_ref`；压力由一般 JWL EOS 从 `(rho_p0, e_p0)` 计算。
- `r > R_0`：静止空气。常数密度与常数比内能；比内能由空气 EOS 从给定的
  环境压力反解，`e_air = p_amb/((gamma-1) rho_air)`。
- `r = R_0` 是一个**单元界面**：左侧单元属于产物，右侧单元属于空气。
  不做左右状态平均，不加人为过渡层，允许初始压力跳跃。

不采用 Taylor 初场，不模拟炸药内部的爆轰波传播。短时两介质数值演化与 PINN
交接属于后续工作；本轮**不定义** `t_handoff`，也不伪造交接场。

## 为什么不是旧的方案 A / 方案 B

- 旧的**方案 A** 是整球均匀产物、没有外部空气，且已被删除。
- 旧的**方案 B** 要求初场由中心起爆、有限爆速的产物场构造
  （`physics/detonation.py`），并在 `T_h = T_b + epsilon` 交接。
- 新决策是**第三种**初值：均匀产物球 + 静止空气，物理时刻固定为 `t = 0`；
  它有自己的 `model_id = "uniform_product_sphere_air_v1"`，不写旧参考场缓存，
  不被旧的验收闸门接受。

## 与现有文档的关系

旧的“B 初场由 `physics/detonation.py` 构造”的要求已随旧方案一起退出，不再是
任何代码的规范。`physics/two_medium_initial.py` 不导入、不调用
`ProductsDetonation`、`SphericalPrecursor` 或 `detonation_product_profile`，也不
写任何参考场文件。旧求解器与旧参考场代码**已删除**，不是“保留不修改”。

## 参数与单位

- `material.PRODUCT_VOLUME_ENERGY = 6.0e9 J/m^3` 是 `*EOS_JWL` 的 `E0`，按
  **参考体积**定义；因此 `e_p0 = E0/rho_ref`（J/kg），**不再**除一次密度。
- 产物压力由 `(rho_p0, e_p0)` 经 JWL 得到，不是独立输入；不使用 CJ 条件或
  固定爆速反推能量。
- 空气：`rho_air = 1.225 kg/m^3`、`p_amb = 101320 Pa`（卡片值，
  不是标准大气压 101325 Pa）、`gamma = 1.4`；`e_air` 由 `p_amb` 反解。
- 质量与内能按球体积分 `(r_{i+1}^3-r_i^3)/3` 计算，单位立体角。

## 2026-10-25：局部平面黎曼启动（当前）

- 零时刻仍是 `two_medium_initial.py` 的均匀静止产物球 + 静止空气；EOS、质量、
  能量与几何校验未改。新增完整球口径字段：单位立体角数组与
  `full_sphere_*` 明确分开。
- `physics/two_medium_riemann.py`：局部平面双介质黎曼问题。产物侧用实际 JWL
  等熵线 `de/dV=-p/rho0`、`du/dV=c/V` 的标量 RK4 积分；空气侧用理想气体激波
  三条 RH；压力匹配求 `p_star, u_star`。没有把常 gamma 产物稀疏波公式替代
  JWL，也没有用线性声学近似。
- `physics/two_medium_startup_planar.py`：把自相似扇嵌入球面，分区为
  未扰动产物、产物稀疏扇、产物星区、空气星区、环境空气。`epsilon =
  delta*R0/(D_s-u_star)`，候选 `delta=1/200`；`Rc=R0+u_star*epsilon`、
  `Rs=R0+D_s*epsilon`。没有网格时间推进，没有调用 `ProductsDetonation`、
  `SphericalPrecursor` 或旧 `TwoMediumStartup.run_to`。
- 输出 `PlanarStartupField`：同一物理时刻 `t_phys=epsilon`、`tau=0` 的
  `rho,u,e,p,c, material`、`Rc/Rs`、`contact_speed/shock_speed`、稀疏波头尾
  位置/速度、初始参数、完整 EOS、`epsilon/delta`、方法版本、单位/积分口径、
  质量/能量与残差。`training_qualified=False`，不是旧 `ReferenceField`。
- 局部平面场嵌入球面会产生几何近似误差：球对称质量与总能量残差由独立高分辨率
  积分报告，不做全局缩放或阈值放行。平面质量/能量残差用来单独检查黎曼扇本身。
- 能量误差保留原有 `spherical_energy_residual`（分母为全域初始能量，包含环境
  空气，随远场外边界变化）。另报有符号的球面总能量变化 `ΔE`、`|ΔE|` 和
  `|ΔE|/初始产物内能`；后者只作补充诊断，不赋予训练资格。不同 `delta` 对应
  不同物理时刻，不能称为同一物理时刻的时间收敛。
- 入口：`python -m spherical_blast_pinn.cli.startup`；`delta` 敏感性诊断：
  `python -m spherical_blast_pinn.cli.validate_startup_planar`；测试：
  `tests/test_two_medium_riemann.py`、`tests/test_two_medium_startup_planar.py`。
- **仍未接入正式训练**；旧 `ReferenceField`/`detonation` 链路已在本轮删除，新 handoff 尚未接到采样器、损失尺度、训练器与检查点。

## 2026-10-21 第二轮（历史）：旧 Lagrangian 短时推进

- `physics/two_medium_startup.py`：球对称拉格朗日 + **相容能量更新**（力—功配对、
  时间中心速度、实际边界功）。第一步不会出现「内能不变、动能增加」；逐步恒等式
  `Σ m e + K + W_boundary = const` 由测试核对（实测相对漂移 ~1e-16）。
- `build_handoff()` 输出 `TwoMediumHandoff`：同一物理时刻的两侧 `rho, u, e, p, c`、
  材料身份、`Rc, Rs, \dot Rc, \dot Rs`、能量账与来源信息。它不是旧
  `ReferenceField`，不进入旧验收闸门。
- 入口：`python -m spherical_blast_pinn.cli.startup --handoff-time ...`。
- 时间口径：`t_phys = 0` 是初值；交接为 `t_h > 0`；网络时间 `tau = t_phys - t_h`。
  不再使用 `R_0/D + epsilon`，也不写死 1 μs。

### 2026-10-21 旧 Lagrangian 推进：已被取代

上述 `two_medium_startup.py` 的实现已在 2026-10-25 被局部平面黎曼启动取代，
并在 2026-10-26 随旧方案一起删除。它的相容能量更新、时间中心速度与边界功实现
不再存在于本仓库；需要时可从版本历史追溯。

### 尚未完成（阻塞项）

- **正式训练接入**：`PlanarStartupField` / `handoff` 还没有接到采样器、损失尺度、
  训练器与检查点。`cli._common.build_context` 现在**明确失败**并报告
  “新启动场的训练接口尚未接入”，不会回退旧方案。
- 数据时间轴还没有迁移到 `tau = t_phys - epsilon`；这属于同一轮接线工作。
- 新链路的物理适用性（`delta` 选择、平面近似误差、RH 一致性）只做了小规模数值
  检查，不构成验收结论。

**零时刻初值、局部平面黎曼启动、handoff 与初值批次已实现；正式训练接入尚未完成。**
