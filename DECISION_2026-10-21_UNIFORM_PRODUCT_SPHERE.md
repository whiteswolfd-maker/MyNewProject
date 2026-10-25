# 初值决策：均匀高压产物球 + 静止空气（2026-10-21）

**状态**：已由用户确认。适用于新的初值模块
`src/spherical_blast_pinn/physics/two_medium_initial.py`。

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

`AGENTS.md` 中"B 初场由 `physics/detonation.py` 构造"的要求继续约束**旧参考场
构造**；它不再定义新的初值模块。`physics/two_medium_initial.py` 不导入、不调用
`ProductsDetonation`、`SphericalPrecursor` 或 `detonation_product_profile`，也不
写任何参考场文件。旧求解器与旧参考场代码保留，不删除、不修改。

## 参数与单位

- `material.PRODUCT_VOLUME_ENERGY = 6.0e9 J/m^3` 是 `*EOS_JWL` 的 `E0`，按
  **参考体积**定义；因此 `e_p0 = E0/rho_ref`（J/kg），**不再**除一次密度。
- 产物压力由 `(rho_p0, e_p0)` 经 JWL 得到，不是独立输入；不使用 CJ 条件或
  固定爆速反推能量。
- 空气：`rho_air = 1.225 kg/m^3`、`p_amb = 101320 Pa`（卡片值，
  不是标准大气压 101325 Pa）、`gamma = 1.4`；`e_air` 由 `p_amb` 反解。
- 质量与内能按球体积分 `(r_{i+1}^3-r_i^3)/3` 计算，单位立体角。

## 2026-10-21 第二轮：短时推进与交接场（已实现）

- `physics/two_medium_startup.py`：球对称拉格朗日 + **相容能量更新**（力—功配对、
  时间中心速度、实际边界功）。第一步不会出现「内能不变、动能增加」；逐步恒等式
  `Σ m e + K + W_boundary = const` 由测试核对（实测相对漂移 ~1e-16）。
- `build_handoff()` 输出 `TwoMediumHandoff`：同一物理时刻的两侧 `rho, u, e, p, c`、
  材料身份、`Rc, Rs, \dot Rc, \dot Rs`、能量账与来源信息。它不是旧
  `ReferenceField`，不进入旧验收闸门。
- 入口：`python -m spherical_blast_pinn.cli.startup --handoff-time ...`。
- 时间口径：`t_phys = 0` 是初值；交接为 `t_h > 0`；网络时间 `tau = t_phys - t_h`。
  不再使用 `R_0/D + epsilon`，也不写死 1 μs。

### 尚未完成（阻塞项）

- **正式训练接入**：`cli._common.build_context` 仍用旧 `ReferenceField` 与
  `T_h = R_0/D + epsilon` 构造模型和尺度。把 `TwoMediumHandoff` 接进去之前，
  旧链路必须保留。
- 因此本轮**没有删除** `physics/detonation.py`、`cli/reference_field.py` 与旧参考场
  验收：它们仍被 `build_context`、`evaluation/checks.py` 和 4 个测试模块导入，
  直接删除会产生悬空导入。旧诊断脚本 `scripts/diagnose_front_momentum.py` 已删除。
- 新链路的物理适用性（交接时刻、激波展宽、RH 一致性）只做了小规模数值检查，
  不构成验收结论。

**分段常数初值、短时两介质推进、交接对象已实现；正式训练接入尚未完成。**
