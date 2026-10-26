# 自由空气中球形 TNT 爆炸的 1D 球对称双场 PINN

一个连续耦合的一维球对称 Euler PINN：一个网络解爆轰产物，一个网络解受扰动
空气，另有一个小网络直接预测两条前沿。全部内部计算采用 SI。

本仓库是**第二版**。第一版（顺序耦合、`tau = t - t_b`、以数据帧作初值）已整体
移到 `_legacy_v1_backup/` 保留，新路径不再引用它。

---

## 一、物理方案

**唯一初始化方案（2026-10-21 决策，2026-10-25 定稿）。**

`t_phys = 0` 时是**均匀静止的 JWL 产物球 + 静止空气**：

* 产物用现行 JWL 与 `E0`，初始密度 `rho_p0`、比内能 `e_p0 = E0/rho_ref`；
* 空气用现行理想气体 EOS，环境态 101320 Pa；
* `R_0` 是**材料界面**，不是爆轰波前。

它**不是** CJ 表面突破，也不引入 Taylor 内场；**不求解化学反应**，两个网络
同时启动、全程保留。

小正时间启动由**局部平面黎曼问题**给出：

1. 产物侧用实际 JWL 等熵线积分（`de/dV = -p/rho0`、`du/dV = c/V`）；
2. 空气侧用理想气体激波的三条 RH 关系；
3. 压力/速度匹配得到 `p_*`、`u_*`，完整自相似稀疏波与星区都保留；
4. 把该自相似扇嵌入球面，得到 `t_phys = epsilon` 的 `PlanarStartupField`。

三个域：

| 区域 | 谁在解 |
|---|---|
| `0 <= r < R_c` | `ProductNet`，压力由一般 JWL `p(rho, e)` |
| `R_c < r < R_s` | `AirNet`，压力由 `p = (gamma_a - 1) rho e` |
| `r >= R_s` | 解析环境态，**无网络、无标签** |

几何：`R_c` 与 `R_s` 都是移动的数据驱动前沿，逐帧重建。接触面直接施加
`p_p = p_a`、`u_p = u_a = dR_c/dt`，**不要求**密度或比内能相等。主激波由完整
三条 Rankine–Hugoniot 条件约束，波前取解析环境态。

2026-09-22 至 2026-10-21 使用过的“方案 B”（中心起爆 + 有限爆速 + 非均匀爆后
产物场 + 拉格朗日双材料前置 + `T_h = R_0/D + epsilon`）已随旧代码整体删除；
它只作为历史记录保留，不再是操作规范。

## 二、时间与尺度化

新链路只有一条时间轴：

```
epsilon      = delta R_0 / (D_s - u_*)     来自实际启动场
tau          = t_phys - epsilon
hat_t        = U_* tau / R_0
s            = hat_t / hat_T_*
s = 0  <=>  tau = 0  <=>  t_phys = epsilon_field
```

由 `physics.similitude.PlanarStartupScales` 显式表达：`start_time()` 返回该场
自己的 `epsilon`，`detonation_velocity=None`（新原点不经过 `R_0/D`）。它只能由
`PlanarStartupScales.for_startup_field(field, time_scale=...)` 构造；把旧的
`SimilitudeScales`（`start_time() = T_h = R_0/D + epsilon(R_0)`）传给初值批次
会被明确拒绝——两条轴都只有秒的量纲，用错不会报错。

默认 `delta = 1/200`；50 mm、默认参数下 `epsilon = 3.096678562615235e-07 s`。
不同 `delta` 对应不同物理时刻，不能相减后称为时间收敛。

尺度：

```
e_* = E_0/rho_0 = 3.680981595e6 J/kg      U_* = sqrt(e_*) = 1918.588 m/s
hat_t = U_* tau / R_0                     s = hat_t / hat_T_*
```

`rho_*,p = 1630`、`rho_*,a = 1.225`，速度尺度同为 `U_*`，比内能尺度同为
`U_*^2`，压力尺度分别为 `rho_*,k U_*^2`。

`hat_T_*` 仍由**训练半径**有效时间范围内最大的 `hat_t` 冻结得到；验证与盲测
半径不参与它的估计。空间坐标、速度/密度/能量尺度与时间导数尺度系数都没有因为
更换初值方案而改变。

移动坐标与链式法则：

```
xi  = r/R_c          d_r f  = f_xi / R_c
                     d_t f  = [U_*/(R_0 hat_T_*)] f_s - xi (Rc_dot/R_c) f_xi
eta = (r-R_c)/h      d_r f  = f_eta / h
                     d_t f  = [U_*/(R_0 hat_T_*)] f_s
                              - [Rc_dot + eta (Rs_dot - Rc_dot)] / h f_eta
```

`h = R_s - R_c`，`Rc_dot = (U_*/hat_T_*) d(hat_Rc)/ds`，`Rs_dot` 同理。

## 三、网络

| 网络 | 输入 | 输出构造 | 结构 |
|---|---|---|---|
| `ProductNet` | `(s, xi^2)` | `rho = rho_*,p softplus(z)`，`u = U_* xi z_u`，`e = e_* softplus(z)` | 4×128 tanh |
| `AirNet` | `(s, eta)` | `rho = rho_*,a softplus(z)`，`u = U_* z_u`，`e = e_* softplus(z)` | 4×128 tanh |
| `FrontNet` | `s` | `hat_Rc = a softplus(z_c)`，`hat_h = b softplus(z_h)` | 2×64 tanh |

三处硬结构，都不是惩罚项：

* `softplus` 使密度与比内能恒为正；
* 显式的 `xi` 因子给出 `u_p(0) = 0`，即球心条件；
* `hat_Rc = a softplus(z_c)` 中 `a = hat_Rc^IC > 0`，保证 `hat_Rc > 0`。

前沿的**初始位置由初值损失约束**，不写进输出构造——这是提示词 §7 的原话，也是
默认行为。

可选的早时段锚定（默认关闭）：`networks.front.contact_anchor` 打开后，

```
hat_Rc = a softplus(z_c) + w(s),   w(s) = ½(1 + cos π min(s/s_anchor, 1))
```

`w` 只在 `0 <= s <= s_anchor` 上非零，两端都有一阶零导数；`until_s` 省略即全时段。
依据：接触面就是原装药表面所在的那个界面，参考前置求解给出的 `R_c(t)` 在整个
训练区间上严格大于 `R_0`，而初值损失只是 `s = 0` 上的一个点约束——实测它的量级
远小于同一时刻的接触面条件，`hat_Rc(0)` 会被其它项的梯度推着下滑。锚定只用来
让**早期几步**的候选轨迹不越出那条已知下界，**不强制单调**，也不是关于整个
演化的物理假设。把"候选轨迹还没满足初值"升级成"全时域物理假设"是把自己实现
的数值困难伪装成物理，所以它必须显式打开、并且可以带适用时段。

半径不进入任何网络。三个训练半径共用同一套网络；物理半径只在还原时乘回去。

## 四、损失

```
L_i = lambda_PDE L_PDE + 10 L_IC + 10 L_c + 10 L_s + 1 L_d + 1 L_adm
L   = (L_50 + L_80 + L_100) / 3
```

| 项 | 内容 |
|---|---|
| `L_PDE` | 两材料各自的三个强形式残差；各自均方后等权平均，再对两材料等权平均 |
| `L_IC` | 参考初值场的 `rho/u/e` 分布 + `R_c,IC`、`R_s,IC`、`h_IC = R_s,IC - R_c,IC` |
| `L_c` | `p_p - p_a`（除以 `Pc_*`）、`u_p - Rc_dot`、`u_a - Rc_dot`（除以 `U_*`） |
| `L_s` | 完整三条 RH，波后取 `eta = 1` |
| `L_d` | 数据监督的 `rho/u/e` 与两条前沿；产物、空气、前沿**三组**分别取均值后等权（前沿组内部再平均 `Rc`/`Rs`） |
| `L_adm` | 两材料压力与声速平方为正、激波压缩、`Ds >= c_a0`、`Ds - u_b > 0`、熵不减 |

约化规则：先在组内对自己的样本取均值，再在组间等权平均。采样数量不改变隐含
权重。

**没有**弱形式、没有全局积分守恒训练项、没有经验曲线损失。整体质量与能量
平衡只作为诊断报告，不进梯度。

数值保护全部记录在案：`softplus`、`clamp_min`、对数与除法都有显式下限，并且
非物理状态的比例单独回报，不用保护后的值冒充有效状态。

## 五、训练日程

| 阶段 | 更新次数 | `lambda_PDE` | 学习率 |
|---|---:|---:|---:|
| 建立初值、基本场和前沿 | 2000 | 0.1 | 1e-3 |
| 增强 PDE | 3000 | 0.1 → 1（线性） | 1e-3 → 1e-4（余弦） |
| 完整联合训练 | 15000 | 1 | 1e-4 → 1e-5（余弦） |

所有物理方程从一开始就保留；三个模块全程联合优化，不冻结点络，不做阶段切换。
`lambda_PDE` 是唯一随阶段变化的权重。可选 L-BFGS 精修最多 500 次迭代，且
**固定采样点**。

每次 Adam 更新，三个半径都参与并等权：一次采样、一个联合损失、一次优化器
更新。

**独立检查与选模型。** 每 500 步（`training.check_every`）在一套**固定**的检查
点上跑一次完整物理验收：一次采样、全程复用、独立的随机种子、取**开发帧**、
不进梯度。报告逐半径、逐介质、逐方程给出归一化残差的 RMS / P95 / 最大值与位置，
加上接触面、RH、初值、非物理比例，以及产物质量漂移与控制体能量平衡。定义随
检查点与运行清单一同保存。

选模型的偏序在看最终测试结果之前固定：先看非物理状态计数，再看物理与界面残差
超出 §16 容差多少，最后才比较开发集误差。不按加权训练总损失选模型。`best.pt`
是这条偏序下的最优检查点。

数据在三个训练半径内按**连续帧组**留出约 10% 作开发检查（`data.development_holdout_fraction`），
而不是把同一帧的相邻网格点随机拆开——随机拆分会让两个集合互相看得见对方，
"开发集误差"于是只是训练误差的一个更贵的写法。

## 六、启动初值：均匀产物球 + 局部平面黎曼

链路（全部在 `physics/` 与 `sampling/`）：

* `physics/two_medium_initial.py`：`t_phys=0` 的均匀静止 JWL 产物球 + 静止空气，
  `R0` 是材料界面；压力由 EOS 从 `(rho_p0, e_p0=E0/rho_ref)` 计算，不用整球
  CJ 压力反推内能。质量/能量同时给出单位立体角与 `full_sphere_*` 两种口径。
* `physics/two_medium_riemann.py`：局部平面 JWL—空气黎曼问题。产物侧用实际
  JWL 等熵线积分（`de/dV=-p/rho0`、`du/dV=c/V`），空气侧用理想气体激波三条 RH；
  求 `p_star,u_star`，不把常 gamma 产物公式替代 JWL。
* `physics/two_medium_startup_planar.py`：把自相似扇嵌入球面，候选
  `delta=(Rs-Rc)/R0=1/200`，`epsilon=delta R0/(D_s-u_star)`。**没有网格时间推进**，
  不做有限体积/差分/拉格朗日逐步推进。输出 `PlanarStartupField`，
  `training_qualified=False`。
* `physics/two_medium_startup_handoff.py`：`cli.startup` 写出带版本号的
  `handoff.npz` + `handoff.json`；`load_planar_startup_handoff` 校验格式版本、
  必需字段、JSON/NPZ 标量、重建的黎曼解与保存状态、保存数组与公共解析采样的
  一致性，随后独立重算质量/能量诊断并核对保存的 `checks`、`integrals` 及顶层
  副本。不跨间断面做普通插值。
* `sampling/planar_startup_initial.py`：把一个通过校验的 `PlanarStartupField`
  转成现有 `SampleBatch` 的初值字段（`xi/eta/rho/u/e` 两组、`contact_ic/shock_ic`），
  产物用 `sample_product`、空气压缩层用 `sample_air`，自动加入球心、接触面两侧
  和激波波后锚点。只做一次坐标无量纲化，`rho/u/e` 保持 SI，返回 `(N,1)` 张量。
  批次必须携带 `PlanarStartupScales`（`s=0 ⇔ t_phys=epsilon_field`），并在
  `diagnostics` 里记录由此换算出的物理时间、`tau` 与无量纲 `s`；旧尺度或与场
  不一致的 `epsilon` 会被拒绝。

局部平面场嵌入球面会产生几何近似误差；球对称质量/总能量残差由独立高分辨率
积分报告，不用全局缩放或验收阈值掩盖。不同 `delta` 对应不同物理时刻，不是同一
物理时刻的时间收敛。

**采样与损失已接入新启动场**：

* `sampling.sampler.RadiusSampler(field, scales, tau_end)`：初值复用上面的适配器，
  PDE/接触面/激波点按移动坐标契约生成；`tau_end` 是从网络启动时刻算起的时长，
  不再由数据最后帧决定。无数据监督时数据与几何批次为空，不构造假算例；传入
  真实算例会明确报未支持（数据时间轴尚未迁移）。
* `losses.objective.contact_pressure_scale(field)`：取该启动场黎曼解的 `p_*`，
  只作为接触压力残差的归一化尺度；接触条件与损失权重不变。
* `evaluation.checks.build_check_set(field, scales, tau_end)`：用同一场、同一
  `p_*` 建立固定检查集。

`python -m spherical_blast_pinn.cli.check_startup_chain --handoff-json ...` 会从
磁盘加载 handoff，装配完整批次、调用 `total_loss` 并执行**一次**反向传播，
报告点数、分项损失与梯度。它不是训练：不建优化器、不更新参数、不写检查点，
`training_qualified` 保持 false。

**检查点（新契约）**：`training.checkpoint` 只读写
`spherical_blast_pinn.startup_checkpoint` v1。它绑定启动场的 `model_id` /
`method_version` / `cache_key`、完整的 `PlanarStartupScales` 参数（半径、
`epsilon`、`time_scale`、各材料尺度、时间约定）、模型结构/EOS/dtype、模型参数、
可选的优化器状态、步数、全局与采样器随机状态，以及**采样配置与动态时间窗口**
（`tau_end`、各类点数、分层比例、`r_min_fraction`、当前窗口分数与历史）。
保存/读取的公开接口是
`save_startup_checkpoint(..., sampler=...)` 与
`load_startup_checkpoint(..., handoff_json=..., sampler=...)`；加载时可用显式提供
的 `handoff.json` 核对身份，检查点本身不复制初场数组。恢复前逐项核对：`tau_end`、
点数或分层不同，或接收采样器不属于同一个启动场/尺度，都会在修改模型、优化器和
采样器之前被拒绝；窗口按保存值恢复（允许相对接收端默认值收窄），因此续训的下
一批采样与连续运行一致。旧格式（`spherical_blast_pinn.checkpoint`，version ≤ 8，
带爆速与 `T_h`）会被明确拒绝并提示用新链路重新生成。

**尚未接入**：`train`/`predict`/`compare` 会明确报错；`Trainer` 构造即抛
`TrainingNotConnected`（见第七节）。

## 七、命令

干净环境（独立 venv，不复用任何 editable 安装或 `PYTHONPATH`）：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install .        # 依赖在 pyproject.toml
.venv/Scripts/python -c "import spherical_blast_pinn; print(spherical_blast_pinn.__file__)"
```

从原始 d3plot 重新提取剖面（保留材料体积分数）：

```bash
python -m spherical_blast_pinn.cli.extract_profiles --config configs/base.yaml \
  --case tnt_50mm --deck F:/tnt/tnt_files/dp0/SYS/MECH/input.k \
  --d3plot-dir F:/tnt/tnt_files/dp0/SYS/MECH --out F:/2/extracted_vf/50mm \
  --charge-radius 0.05 --shells 2500 --stride 1 --outer-radius-factor 40
```

启动链路与数据审查：

```bash
# 1. DYNA 真实物理初始化审查（必须先做）
python -m spherical_blast_pinn.cli.audit_dyna --config configs/base.yaml \
  --deck tnt_50mm=F:/tnt/tnt_files/dp0/SYS/MECH/input.k \
  --data tnt_50mm=F:/2/extracted

# 2. 启动场：均匀产物球 + 局部平面黎曼，写出 handoff.npz/json
python -m spherical_blast_pinn.cli.startup --charge-radius 0.05 --delta 0.005

# 3. delta 敏感性/自相似缩放诊断（不同 delta 是不同物理时刻，不是时间收敛）
python -m spherical_blast_pinn.cli.validate_startup_planar \
  --charge-radius 0.05 --deltas 0.005,0.0025,0.00125

# 3b. 装配检查：磁盘加载 handoff -> 完整物理批次 -> total_loss -> 一次 backward
python -m spherical_blast_pinn.cli.check_startup_chain \
  --handoff-json outputs/planar_startup/<model>/<key>/handoff.json \
  --tau-end 1e-6 --output-json outputs/startup_chain_check.json

# 4. 数据质量审查 / 状态方程一致性
python -m spherical_blast_pinn.cli.audit_data --config configs/experiments/full.yaml
python -m spherical_blast_pinn.cli.validate_eos --config configs/experiments/full.yaml
```

**尚未接入**：`train`、`predict`、`compare` 目前只打印
“新启动场的训练接口尚未接入”并以退出码 3 结束；它们不会回退到已删除的旧参考场。

测试：

```bash
python -m pytest -q
```

## 八、数据状态与已知限制

这一节必须如实读。**数据侧的结论仍然有效**，但它们限制了这个模型能宣称什么；
训练入口尚未接入新启动场（见第六、七节）。

### 8.0 这些结论现在是训练入口的一道闸门，不是一段说明

`audit_dyna` 除了写人读的审查表，还写一份机器可读的
`data_compatibility.json`。训练入口（接线后）读它，并对每个训练算例要求
状态为 `compatible`：

| 状态 | 含义 | 训练是否放行 |
|---|---|---|
| `compatible` | 初始化、单位、EOS、时间定义、材料分区都审过且一致 | 放行 |
| `incompatible` | 审过，**发现**与项目当前约定冲突 | **拒绝** |
| `unknown` | 没有审查证据，或审查里是"缺证据" | **拒绝** |
| `synthetic` | 明确标注的合成样例 | 只在合成路径放行 |

本机三份数据实测的状态（`outputs/dyna_audit/data_compatibility.json`）：

```
tnt_50mm : incompatible（detonation_model 不一致；time_origin 不一致；material_partition 不可靠）
tnt_80mm : incompatible（detonation_model 不一致；time_origin 缺证据；material_partition 不可靠）
tnt_100mm: 报告里没有它的条目 —— unknown
```

要越过闸门必须显式写 `data.allow_unverified_cases: true`，并且运行清单与检查点
会被打上 `data_compatibility: unverified`。冒烟配置就是这么做的，因为它的用途是
验证链路能跑通，不是物理训练。

### 8.1 时间零点：数据是有限速度爆轰；旧方案 B 已退出

`audit_dyna` 用数据本身判定时间零点：反应前沿在 `t <= R_0/D` 的帧上跟随
`r = D t`，50 mm 算例上拟合斜率 7029 m/s 对卡片爆速 6930 m/s，偏差 1.4%；
d3plot 里 t = 0 时刻装药单元的 `p` 恰好为零，只有爆轰波扫过才升压。这说明真实
算例用的是**有限速度爆轰**，不是“零点整球同时转化”。

2026-09-22 至 2026-10-21 使用的**方案 B** 曾据此构造中心起爆产物场，再做短时
拉格朗日双材料前置，并把启动时刻定为 `T_h = T_b + epsilon`。该方案及其代码
（`physics/detonation.py`、`physics/precursor.py`、
`physics/two_medium_startup.py`、`cli/reference_field.py`、
`cli/validate_startup.py`、`cli/diagnose_startup.py`）已于本轮删除，旧缓存也不再
被任何代码读取。

**当前唯一初始化方案**是“`t=0` 均匀静止产物球 + 静止空气 + 局部平面黎曼启动”。
它不把均匀球解释为 CJ 表面突破，也不引入 Taylor 内场。**数据时间轴还没有迁移**：
真实数据仍按 LS-DYNA 原始时间记录；把新启动场与实验帧对齐属于下一轮训练接线的
工作，本轮不做，也不能用简单平移宣称二者等价。

### 8.2 材料分区：旧的启发式判据已废弃，改用体积分数

**旧的那份 `raw_line.npz` 只写了 `t/x/rho/u/P`，丢掉了材料体积分数。** 下游只能
用密度最陡下降去猜接触面，实测结果是 50 mm 的 101 个"有效"帧里有 86 个给出
`R_c < R_0`——物理上不可能。

数据本身没有这个问题：四个算例都是 `elform = 11` 的 **ALE 多物质**，每个单元都带
材料体积分数。`cli.extract_profiles` 把它读出来，材料分区因此从**启发式**变成
**读数**：

| 检查项 | 旧提取 | 新提取 |
|---|---:|---:|
| 材料分区来源 | 密度最陡下降 | d3plot 体积分数 |
| `R_c < R_0` 的帧 | 92 | **0** |
| 空气壳层有有限场量 | — | **全部** |

`material_partition` 是兼容性闸门里独立的一项，而且判定的是**提取结果**本身——
体积分数范围、标签与阈值自洽、每个被标为有效的壳层都带有限的 `rho/u/e/p`、以及
从磁盘重读一遍逐元素一致。一份"来源写着体积分数、空气壳层却全是 `nan`"的导出会
被直接拒绝：那样的空气监督是空的，采样器会把 `nan` 标签逐个丢掉，数据损失里的
空气组随之静默消失。

**"标签必须蕴含状态"现在是精确判据，不是比例门槛。** 一个标了材料却没有有限场量的
壳层必须落在提取器写出的 `contaminated_shell_fraction > 0` 里；**没法解释的缺口必须
是零个**。原来那条"已标壳层里至少 99% 要有有限场量"既抓不住"剩下 1% 里藏着一整个
静默消失的空气组"，又会误伤真实的物理：装药里尚未释放化学能的单元**没有**产物状态
（实测 50 mm 第 0 帧 218328 个 `alpha >= 0.95` 的单元全部如此，`p` 恰为 0），把它们
算成"标签漏洞"等于要求提取器编一个数出来。

这一项通过**也不代表**起爆方式与时间零点的检查通过——那是另外的项，各自独立判定。

### 8.2.1 比内能是**重建量**，而且反解从来不是"解不出来"

这套 d3plot 里实体单元**没有**比内能结果（LASSO 的 `element_solid_*` 系列只有应力、
应变和 history 变量），所以纸上只有 `(rho, p)`，比内能只能按各材料的 EOS 反解：

```
e = (p - p_cold(V)) / (omega rho)        产物（一般 JWL）
e = p / ((gamma - 1) rho)                空气（理想气体）
```

两条都是**除法，不是求根**：有限输入、`rho > 0`、`omega > 0` 时结果必然是有限实数。
实测 50 mm 算例 40 个状态 × 1220736 个单元，反解出非有限值的**一个都没有**。所以
不存在"压力转负之后没有实数解"这一档——会出现的只是反解出**负**值，含义是该单元的
`p` 落在 JWL 冷压曲线之下，即**化学能还没放出来**，套产物 EOS 本来就不成立。

提取器因此把这种单元标成"没有状态"：密度、速度、压力照留（它们是真测量），比内能
留空，并逐帧计数写进元数据。壳层平均会静默跳过 `nan`，所以污染按**体积**单独算一遍
（`material.npz` 的 `contaminated_shell_fraction`），任何非零比例让整个壳层作废。

### 8.2.2 前沿位置的分辨率是**网格的**，不是壳层的

壳层网格加密到 0.12 mm 不会让一个 10 mm 厚的 ALE 单元里的梯度变得可见。实测 50 mm
算例的**单元径向厚度**（节点半径极差）：

| 半径 | 单元径向厚度（中位数） |
|---|---:|
| 0–50 mm（装药内） | 0.85 mm |
| 50–55 mm（装药外） | 9.7 mm |
| 55–60 mm | 12.7 mm |
| 70–90 mm | 17.8 mm |
| 90–120 mm | 30.9 mm |
| 3 m 以上 | 105 mm |

两个结论分开报：

* **壳层网格收敛**（2500 vs 5000 层，在双方都分辨得出的 11 帧上）：`max|dR_c| =
  0.72 mm = 2.4e-3 R_c`，`max|dR_s| = 4.8 mm = 2.2e-2 R_s`，`max|dh/h| = 4.5e-2`。
  这说明**提取对分箱不敏感**。
* **网格下限**：`h = R_s - R_c` 小于一个单元厚度时，`h` 不是一个被测量出来的量。
  提取器把每一帧标成 `shock_state`：`0` 测不到、`1` 测到了但亚单元、`2` 分辨得出，
  并同时写出 `shock_cell_thickness_m` 与 `h / 单元厚度`。50 mm 算例第 7 状态
  （t = 6.95 µs）就是 `h = 5.24 mm` 对 `10.95 mm` 的单元，比值 0.478。

一条不经过任何模型的直接读数：**`t = 0` 时接触面按定义就在 `R_0`**，所以它的实测值
是一次免费的全链路标定。实测 `R_c(t=0)`：

| 算例 | 装药半径 | 实测 `R_c(t=0)` | 偏差 | 装药外第一个空气单元厚度 |
|---|---:|---:|---:|---:|
| 50 mm | 50 mm | 52.12 mm | +2.12 mm | 8.9–14.1 mm |
| 80 mm | 80 mm | **131.97 mm** | **+52.0 mm** | **208–361 mm** |
| 100 mm | 100 mm | **164.96 mm** | **+65.0 mm** | **260–452 mm** |
| 500 mm | 500 mm | **824.80 mm** | **+324.8 mm** | **1303–2258 mm** |

**后三个算例的近场空气只有一层单元**，径向厚度就是上表最后一列。它们的体积分数从 1
直接跳到 0，0.5 穿越被线性插值到"两个壳层中间某处"——所以接触面偏高 65%，激波定不出来
（`shock_state = 0` 每一帧），空气侧按 8.2.1 的规则整体撤标（`air: 0`）。这三个算例因此
**只能提供产物侧监督**（它们装药内的单元是 1.3–2.8 / 1.7–3.5 / 8.3–17.3 mm，是分辨得出的），
近场空气与两条前沿的位置都不可用。这是数据本身的限制，不是提取器的问题：把壳层从
2500 加到 5000 不会改变一个 300 mm 厚的单元。

提取器把每一帧的接触面也标成三态（`contact_state`：`0` 没找到、`1` 跨着空壳层插出来的、
`2` 分辨得出），并报出穿越区间里的**空壳层个数**（`contact_empty_shells_in_gap`）。判据
刻意**不用**"跨距 vs 单元厚度"比大小：单个巨型单元内部处处是空的，跨距（104 mm）反而
小于单元厚度（128 mm），比大小会把这一档整个漏掉。

### 8.2.3 激波定位：连通段的外缘，不是最外侧的超压壳层

激波以外的远场压力是数值噪声，实测在 `p/p_amb = 1.000` 上下摆动、幅度可达 1.5%，
而阈值是 1%。于是"最外侧超压壳层"这个判据测到的是**噪声尖峰**：50 mm 第 10 状态给出
95.2 mm（2500 层）/ 101.0 mm（5000 层），而真实的超压区外缘在 58.2 mm——两套网格各
测噪声的不同尖峰，`R_s` 差 5.8 mm（`R_s` 的 5.8%），看起来像"分辨率不够"。

改成取**与接触面相连的连续超压区**的外缘（外缘线性插值）之后，同一帧两套网格只差
0.1 mm。求不到时返回诊断而不是一个数：峰值过压说明**为什么**求不到。

### 8.3 100 mm 的旧导出只有 30 帧——那是提取步长，不是数据限制

旧导出里 100 mm 只有 30 帧、间隔 47 µs，看起来"燃尽前没有采样点"。实际上该算例
有 **782** 个原生状态（`G:/500mmtnt/100mmtnt_files/dp0/SYS-3/MECH`），30 帧是当初
提取时自选的步长。全量重新提取之后这个限制不再存在。

（顺带一条：**d3plot 文件数不等于状态数。** 实测 50 mm 是 780 个文件 / **779** 个
状态，其余三个是 783 / **782**，每个都差一个。提取器用一个"请求不可能存在的状态
号、从 LASSO 的报错里读出真实上限"的探针拿到权威值，不再拿文件数当状态数。）

### 8.4 其他限制

* 域内的明显间断（二次激波）是本版强形式方案的已知限制，没有单独处理，也
  不声称不存在。
* 数据里的比内能不是输出字段，由 `(rho, p)` 按材料各自的 EOS 反解（见 8.2.1）。
  反解永远有实数解，闭合误差在机器精度上（最大 3.2e-16），但它是**重建量**，
  不是测量量；而且它在单元的 `p` 低于 JWL 冷压曲线时是**负**的，那种单元没有
  比内能标签。
* 主激波之外的环境态不参与监督。这是有意的：那里是几千个常数单元，拿它评分
  对所有模型一样地"好看"，衡量不出任何东西。`R_s` 定不出来的帧（`shock_state`
  为 0）整个空气侧都撤标，因为那条带子没有边界——否则均匀大气会被当成空气监督。
* 前沿位置与 `h = R_s - R_c` 的绝对精度受 **ALE 单元径向厚度**限制（见 8.2.2）：
  早期几帧的 `h` 小于一个单元，那些帧里 `h` 不是被测量出来的量。
* **80 / 100 / 500 mm 三个算例只有产物侧可用**（见 8.2.2 的表格）：近场空气是一层
  208–2258 mm 厚的单元，接触面与激波都定不出来。
* **lasso-python 2.0.3 读不动那三个算例的单元 history 变量**（50 mm 正常）。它们的
  头块 `neiph = 22` 且两个应变标志为真，LASSO 据此预计 10 列、实际写 16 列，读取直接抛
  `ValueError: could not broadcast ...`。`raw_reader._install_solid_history_allocation_fix`
  把容器按数据的宽度加宽（只增不减、不动数值），正确性由 `_charge_radius_from_volume`
  独立验证：四个算例分别给出 49.9979 / 79.9943 / 99.9929 / 499.9610 mm。**升级 LASSO
  之后要重新验这一条**，`VALIDATED_LASSO_VERSION` 记的是 2.0.3。
* 半径变化在当前模型里主要检验尺度一致性，不构成对非相似起爆条件泛化能力的
  主张。

## 九、仓库布局

```
src/spherical_blast_pinn/
  physics/
    material.py       材料参数与物理零点（唯一的一份数值来源）
    eos.py            一般 JWL 与理想气体，及其导数与声速
    similitude.py     时间尺度、hat T_*、PlanarStartupScales（epsilon 原点）
    moving_frame.py   xi/eta 与链式法则
    pde.py            强形式残差
    two_medium_initial.py    t=0 均匀静止产物球 + 静止空气
    two_medium_riemann.py    JWL—空气局部平面黎曼解
    two_medium_startup_planar.py  自相似扇嵌入球面（t_phys=epsilon）
    two_medium_startup_handoff.py 版本化 handoff 保存/加载与重建核对
  models/
    product_net.py    (s, xi^2) -> (rho, u, e)
    air_net.py        (s, eta)  -> (rho, u, e)
    front_net.py      s -> (hat_Rc, hat_h)
    coupled_model.py  三者的组合与分区
  losses/
    terms.py          各项损失
    objective.py      总损失与权重
  sampling/
    batch.py          一个优化步的全部样本
    planar_startup_initial.py  启动场 -> SampleBatch 初值字段
    sampler.py        §12 的采样规则（旧参考场参数已解除，接线待做）
  training/
    schedule.py       §13 的三阶段日程
    trainer.py        训练循环与检查点
  data/
    extracted_case.py 时空剖面读取、材料分区、质量报告、训练/开发帧划分
    compatibility.py  数据与项目约定的兼容性状态，以及训练入口的闸门
    raw_reader.py     直接读 d3plot
    shell_average.py  按材料体积分数做球壳平均
    kfile_parser.py   .k 卡片解析（含 10 字符定宽格式）
  evaluation/
    waveform.py       超压、到达时间、正压时长、正相冲量
    checks.py         固定独立检查集、逐方程指标、守恒诊断、选模型偏序
  cli/                数据与启动场命令；train/predict/compare 尚未接入
tests/                §15 要求的测试 + 审查清单的逐项回归
_legacy_v1_backup/    第一版实现，仅供追溯
```
