# 一维球对称 TNT 空气爆炸：ContactNet + AirNet PINN

本仓库只有一个当前方案：直接消费一维球对称径向 ALE 数据，用共享的
ContactNet 与 AirNet 交替训练。旧 ProductNet/FrontNet、旧三维球壳提取、
旧平面黎曼启动、旧联合总损失训练器、旧 `R_b` 环境边界和固定 500 mm 规则均已
删除，不在本仓库保留可执行副本。

所有内部物理量采用 SI：米、秒、kg/m³、m/s、Pa。压力一律为绝对压力；超压只在
输出诊断时形成，不进入 EOS/PDE/RH。

---

## 1. 几何与网络

```text
ContactNet: (t, R0) -> R_c = R_s(t) * sigmoid(raw(t, R0))
AirNet:     (zeta, t, R0) -> (rho, u, p),  e = p / ((gamma_a-1) rho)
域:         R_c(t) <= r <= R_s(t)
            zeta = (r - R_c) / (R_s - R_c)
            L = R_s - R_c
            L_dot = D_s - R_c_dot
            u_hat = (u - R_c_dot - zeta * L_dot) / L
```

- `R_s` 是**外传空气激波**，由原始径向压力剖面的 `-dp/dr` 识别；不是旧
  `R_b`，也不是仿真外边界 `R_outer`。
- ContactNet 默认 5 个隐藏层 × 50 神经元；AirNet 默认 10 × 32。`R0` 是米。
  两个网络的时间输入是 `tau = t / networks.time_reference_s`：默认
  `time_reference_s: 1.0`，即物理秒；可设为一个由训练组确定的固定值（例如
  训练组最大 `t_end`），所有算例共用。只有送进 MLP 的数字被缩放，激波轨迹、
  接触速度、PDE、边界条件和自动求导仍按秒计算，速度仍是 m/s。
- `R_c = R_s sigmoid(raw)` 保证 `0 < R_c < R_s`。完整乘积导数：
  `R_c_dot = D_s s + R_s s (1-s) raw_t`。有限精度下 `L <= 0` 明确报错，不截断
  厚度制造可用几何。
- 产物压力闭合沿用前置释放分支：
  `V = V_sep (R_c/R_sep)^3`，
  `p_c = A exp(-R1 V) + B exp(-R2 V) + C / V^(1+omega)`。
  `V` 是无量纲相对比容，不与有量纲 `1/rho` 混用。ContactNet 没有产物全场 PDE。

## 2. 一维数据入口

主实现：[one_d.py](src/spherical_blast_pinn/contact_air/one_d.py)。

标准中间格式 `spherical_blast_pinn.standard_1d_source`，支持：

- 单个 `.npz`，含 `metadata`（JSON 文本）、`frame_offsets` 和拼接数组；
- 目录，含 `metadata.json` + `data.npz`。

每帧必须提供：

```text
case_id, charge_radius_m, 原始帧号, 原始时间 t_data
cell_id（显式提供或由读取器生成）, r_left, r_right, 代表位置 radius
alpha_product, alpha_air（可选）, material（可选）
rho [kg/m^3], u [m/s 径向分量], p [Pa 绝对压力]
valid_density, valid_velocity, valid_pressure（逐字段）
position_type = cell / cell_center / node
```

`frame_offsets` 必须是一维整数数组，首值为 0、严格递增、末值等于逐单元数组
总长度；`cells_per_frame` 必须与它逐帧一致。所有逐单元数组（含 `cell_id`）
必须等长，不允许截断切片后静默接受。

metadata 必须显式给出：

```text
units: {length, time, density, velocity, pressure}
coordinate = radial_r            # 缺键拒绝，不静默默认
velocity_component = radial_u    # 缺键拒绝，不接受速度模长
absolute_pressure = true         # 否则拒绝
position_type
frame_times, cells_per_frame
```

读取后统一换算到 SI。长度支持 m/mm/cm，时间 s/ms/us，密度 kg/m³、g/cm³、
t/mm³，速度 m/s、mm/us、km/s、cm/us，压力 Pa/kPa/MPa/GPa/bar/kbar。

校验（全部在读取阶段给出字段、帧或索引）：

- `frame_times` 必须全部有限且严格递增；
- `r_left/r_right/radius` 必须有限；径向坐标非负、单元宽度为正、代表位置在
  单元内、单元不重叠；真实正缺口允许并作为分段；
- `cell_id` 必须为整数且每帧唯一；缺省生成的 ID 在帧和 case 的
  `source["cell_id_source"]` / `source["generated_cell_id_frames"]` 中明确记录，
  不伪称原始 ID；
- 体积分数有限且在 `[0,1]`。当前两介质模型只支持 product + air：显式给出
  `alpha_air` 时，`alpha_product + alpha_air` 必须在 `1 ± 1e-4` 内；空腔/第三
  介质不支持，读取器不会归一化、填充或修正。未提供 `alpha_air` 时，剩余分数
  按 `1 - alpha_product` 解释为空气，并在 source 中记录；
- 显式 `alpha_air` 是空气判定的依据；仅凭 `alpha_product` 接近零不会把单元
  定为纯空气。接触区间、空气掩码、激波检测、事件检测和观测筛选共用
  `RadialFrame.material_masks` / `air_mask` 的同一套语义；
- 某字段 `valid=True` 但数值为 NaN 时拒绝；缺失字段不填零、不填环境常数；
- 节点源必须已按支持约定转换为单元代表值：
  `node_to_cell_convention = "preconverted_cell_representatives"`；任意字符串
  或未转换的节点数据拒绝。

**LS-DYNA 二进制 ALE1D 槽位尚未核实，因此没有实现猜测式二进制读取。** 真实
接入需要文件样例、版本和字段映射；在此之前只能使用上述中间格式。

### LS-DYNA ASCII profile 导入

`scripts/import_lsdyna_profiles_1d.py` 把 `*DATABASE_PROFILE` 生成的 `.xy`
文本剖面转换为上述标准中间格式。它只读取 ASCII profile，不解析
`d3plot`/`d3dump` 二进制槽位。

- 压力、密度、径向速度分别来自 pressure、density、x-velocity profile；
- 产物体积分数和空气质量分数必须来自两条**按 ALE 材料组**的体积分数请求：
  `MMG=1`（TNT/产物）和 `MMG=2`（空气）。`MMG=-10`（多材料组列表）输出的是
  指定组的体积平均，所有单元恒为 1，不能当作 `alpha_product`；
- 单元边界取速度剖面中的真实节点坐标 `node[i]`、`node[i+1]`，不假设均匀网格；
- 节点速度按相邻两节点平均到单元代表位置；长度、时间、密度、速度、压力统一
  换算为 SI；
- 体积分数不做归一化、填充，也不由密度/压力反推；`alpha_product +
  alpha_air` 不在 `1 ± 1e-4` 内直接拒绝；若输入只有 `MMG=1` 的产物分数（例如
  R150–R350 的 deck），用 `--alpha-air-pattern` 省略空气文件，读取器按
  `1-alpha_product` 隐式空气并记录来源；
- `.xy` 必须以 `endcurve` 结束；没有结束标记视为求解器尚未写完，整帧拒绝，
  不能把截断文件当完整数据；
- 所有选中帧的单元数、节点数必须一致；固定 ALE1D 网格还要求首末径向边界
  一致，否则拒绝。移动网格必须显式传 `--allow-moving-mesh`；
- profile 文件只按单元集顺序给值，不含 LS-DYNA 单元 ID；导入器不生成 ID，
  由读取器生成并在 `generated_cell_id_frames` 中记录为 generated；
- 剖面文件名只把输出时刻写到 3 位有效数字，文件和 `d3hsp` 中没有该
  `*DATABASE_PROFILE` 写入的精确时刻。导入器把文件名时间写入 `frame_times`，
  并在 `source["time_source"] = "profile_filename_3sigfig"` 记录这一限制；
  不得把它宣称为精确求解器时刻；
- 默认保留 `--dense-until-s` 之前的全部帧（当前默认 500 µs，覆盖 R1000 的
  `t_sep=370.6 µs`），之后每 `--frame-stride` 帧取一帧；实际保留的原始文件
  键写入 source。
- 缩放算例的原始输出间隔是 `T/1000`（R200 为 4 µs、R1000 为 20 µs），
  粗 stride 可能把 `t_sep` 之后唯一的帧删掉。显式传 `--event-time-s`
  （理论 `t_sep`）时，窗口 `[t_sep-50 µs, t_sep+50 µs]` 内的全部帧以及
  两侧各一个邻近帧无论如何都会保留，其余时段仍按 stride 降采样。

示例：

```bash
PYTHONPATH=src python scripts/import_lsdyna_profiles_1d.py \
  --source-dir G:/R50_R100_reference_v2/R50_full_alpha_profiles \
  --case r50_full --radius-m 0.05 \
  --alpha-product-pattern 'profile4_volume_fraction_x_time*.xy' \
  --alpha-air-pattern 'profile5_volume_fraction_x_time*.xy' \
  --dense-until-s 2.5e-5 --frame-stride 10 \
  --out outputs/contact_air_r50/standard_1d.npz
```

## 3. 接触区间、激波与事件对齐

### 接触区间

只识别与**爆心主产物区外缘相连**的混合段：

- `alpha_product >= pure_product_threshold` 为主产物；
- `alpha_product <= 1 - pure_air_threshold` 为纯空气；
- 中间为混合段，输出 `[R_c,low, R_c,up]`，取**实际单元边界**；
- 直接纯产物—纯空气转换使用共享单元边界，区间宽度为零；
- 外侧孤立产物碎片不参与；
- 混合段路径必须几何连通；空间缺口/重叠中断时标记无效并记录原因。

阈值在 `contact_interval` 配置中显式给出，尚未由真实数据确认。不使用
`alpha=0.5`；**不计算、不返回也不缓存区间中点**，训练对象只看到
`[R_c,low, R_c,up]`。数据坐标映射只用预测的 `R_c` 和测得的 `R_s`。

### 激波识别

在有效、几何连续的纯空气段上计算 `g(r) = -dp/dr`，用
`scipy.signal.find_peaks` 找显著正峰。要求：

- 真实径向坐标，支持非均匀网格；
- 不跨缺失点、混合带、数组缺号或**真实空间缺口**；
- 压缩方向：内侧压力高于外侧；
- 候选距接触带足够远；
- 相邻有效帧中外传；位置完全重复是 plateau，不是向内运动；
- 位置严格回退的候选拒绝；不自动跨失败帧插值；
- 每帧记录识别位置 `R_s(t_i)`、候选、`run_bounds`、所选候选所在段的
  prominence、压缩/RH 诊断和失败原因。

不做压力峰法对照，不使用“恢复环境的第一个点”，不在失败时切换位置定义，
不暗中平滑。阈值集中在 `shock_detection` 配置。

### 事件对齐

用理论 `t_sep` 指导原始帧搜索。当某帧同时具有可信接触区间、外层激波和两者
之间几何连续的纯空气区间时，取**不早于 `t_sep`** 的首个可辨认分离帧
`t_event_data`：

```text
t_model = t0 + (t_data - t_event_data)
```

若搜索窗口内没有不早于 `t_sep` 的合格帧，明确报错，**不退回** `t_sep` 之前的
帧。原始帧时间不平移、不缩放；事件对齐只作用于模型时间 `t_model`，不是取消
偏移。空气观测、接触标签和激波轨迹共用同一平移。保存原始时间、原始帧号、
对齐时间、事件帧和偏移。若事件帧是首个可用帧，或前一帧无法验证，明确记录
`event_in_first_frame` / `preceding_frame_confirmed`，不宣称找到了全数据“首次
分离”。不平移半径。

### ShockTrajectory

在连续有效段内用 PCHIP 插值 `R_s(t)`，`D_s = dR_s/dt`：

- 单点只有位置，没有速度；两点只有该区间平均速度；
- 不跨无效帧、不外推；
- 预处理用 SciPy 生成分段系数，运行时用 PyTorch 分段多项式求值和解析导数；
  不把 `t` detach 后转 NumPy 查询；
- 位置重复形成 plateau：位置保留，速度记为 0/unresolved；`D_s <= c_inf` 的
  采样时刻不进入 RH 边界损失；
- 真实回退段拒绝并在诊断中记录。

## 4. 前置与初场

保留 `contact_air/precursor.py` 的 CJ + JWL 等熵释放 + 空气 Hugoniot 匹配、
`R_sep`、`t_sep` 和 JWL `C` 的来源。初始薄层尺度：

```text
epsilon = R_sep / [200 * (D_s_sep - u_c_sep)]
t0      = t_sep + epsilon
Rc0     = R_sep + u_c_sep * epsilon
Rs0     = R_sep + D_s_sep * epsilon
```

`epsilon_denominator_factor` 参与计算；固定 `epsilon_s` 输入被明确拒绝。
硬检查 `factor > 0`、`D_s_sep > u_c_sep`、`epsilon` 有限为正、初始厚度为正。

AirNet 初值在 `t=t0`、**整个** `0 < zeta < 1` 使用分离态
`(rho_a_sep, u_a_sep, p_a_sep)` 的缩放平方误差；没有未扰动空气子区，也没有
旧 `zeta_s0` 分段初场。

ContactNet 初值只约束位置：`[(R_c(t0) - Rc0)/s_R]^2`。独立初始速度损失已
删除。

## 5. 损失

### AirNet

```text
L_a = lambda_PDE L_PDE + lambda_IC L_a_IC
    + lambda_left L_interface + lambda_right L_a_right
    + lambda_data L_a_data
```

- 左边界：
  `L_interface = mean[((u(0,t)-Rc_dot)/s_u)^2 + ((p(0,t)-p_c(t))/s_p)^2]`。
- 右边界：有限马赫数 RH 波后极限（`zeta=1-`）：
  `c_inf = sqrt(gamma_a p_inf/rho_inf)`，`M_s = D_s/c_inf`，
  `rho_RH = rho_inf (gamma+1)M_s^2 / [(gamma-1)M_s^2+2]`，
  `u_RH = 2 c_inf/(gamma+1) (M_s^2-1)/M_s`，
  `p_RH = p_inf [2 gamma M_s^2-(gamma-1)]/(gamma+1)`。
  只有 `M_s > 1` 的采样进入该项；不再是静止环境 rho_inf/0/p_inf。
- 数据损失：缓存物理 `r,t` 与逐字段标签/掩码；训练时用**当前预测几何**重算
  `zeta=(r-R_c)/(R_s-R_c)`，每字段独立有效点归一；域外点计数后排除，不 clamp；
  整批全被筛掉硬失败。预测/目标/掩码先规范形状再按掩码选择，禁止 `[N,1]` 与
  `[N]` 广播成 `[N,N]`。

### ContactNet

```text
L_c = lambda_IC L_c_IC + lambda_data * w_c L_c_data
    + lambda_PDE L_PDE + lambda_interface L_interface

L_c_IC   = [(R_c(t0)-Rc0)/s_R]^2
L_c_data = mean[(ReLU(R_c,low-R_c)/s_R)^2 + (ReLU(R_c-R_c,up)/s_R)^2]
```

- `w_c = training.contact_data_weight`（默认 1.0）**只乘接触区间监督项**，
  不动初值、界面、空气数据和 PDE 权重；命令行 `--contact-data-weight`
  可覆盖。每条 loop 记录未加权的 `contact_parts`，并记录实际 `w_c`，避免
  用乘权后的总损失做跨实验比较。

区间内零罚、区间外距离平方；接触标签在自己的 `t_contact` 上重新计算网络。两个
网络复用同一组映射空气残差，不复制“接触面 PDE”。

映射 PDE 与单位体积总能量：

```text
E = p/(gamma_a-1) + rho u^2/2
f_rho = rho_t + d_zeta(u_hat rho) + 2 rho u/r + (L_dot/L) rho
f_m   = (rho u)_t + d_zeta(u_hat rho u + p/L)
        + 2 rho u^2/r + (L_dot/L) rho u
f_E   = E_t + d_zeta(u_hat E + u p/L)
        + 2 u(E+p)/r + (L_dot/L) E

s_E  = s_p/(gamma_a-1) + 0.5 s_rho s_u^2
s_fE = s_E / T_star
```

`tests/test_contact_air_mapping.py` 用光滑构造场对照物理坐标 Euler 残差。

## 6. 交替训练与检查点

- 两个持久 Adam；空气阶段只更新 AirNet，接触面阶段只更新 ContactNet。
- 空气阶段 ContactNet 几何 detached；接触面阶段 AirNet 参数冻结但
  `zeta/t` 导数与计算图保留（`AirNet.forward_functional`，detached 参数）。
  不用 `no_grad`/整体 detach 包住物理残差。
- 每次反传检查目标网络有非零有限梯度、非目标网络无梯度、参数未移动。
- 每条 loop 日志在原有字段后追加空气阶段三个**归一化** PDE 分项
  （`pde_mass_normalised` / `pde_momentum_normalised` /
  `pde_energy_normalised`）；它们只是同一 `pde_loss` 总和的三个分量，用于
  记录，不改变损失、采样或梯度，且不消耗训练随机数。历史记录保留原有未加权
  分项。
- 多半径均衡采样；空气数据先抽帧再抽帧内点；接触标签用自己的时刻；只在有效
  轨迹段采样；先不加激波附近额外权重。
- 缓存 schema **version 7**；检查点 **version 4**。旧格式明确拒绝并要求重建。
- 检查点保存两个网络、两个优化器、阶段状态、RNG、配置、数据身份、固定验证
  样本；恢复先核对再应用。网络元数据中的 `time_reference_s` 也参与核对：
  旧检查点缺少该字段时按 1.0 处理，整个描述块缺失则拒绝，与当前网络不一致就
  拒绝续训（预测路径同样检查）。检查点保存的 `contact_data_weight` 必须与
  当前配置一致（旧检查点缺少该字段按 1.0），否则拒绝续训，避免把 ×3 的实验
  悄悄退回 ×1。
- 训练/验证/测试由配置中的显式 `role` 决定；没有固定半径禁令。`role: test`
  不能进入训练。验证有两个来源：训练算例末尾的连续帧组
  （`validation.holdout_fraction > 0`），以及配置中整例 `role: validation`
  的算例（例如一条不重叠的半径带）。后者每 `validate_every` 步评分并可决定
  `best.pt`，但绝不进入训练采样，且不能与训练算例共用 `case_id`。两者可同时
  使用；要把验证完全交给半径带，就把 `holdout_fraction` 设为 0。

### 参考数据与实现审计（只读）

`configs/contact_air/reference_audit.yaml` +
`scripts/audit_reference_data.py` + `contact_air/reference_audit.py` 提供不依赖
训练链路的核验：

- **A 原始仿真**：20 个算例的文件身份、求解器版本、正常结束、原始时间范围、
  均匀球装药质量 `W=4/3 pi rho R^3`、固定半径压力历程；
- **B 导入链路**：用独立的最小 `.xy` 解析器和配置中的单位因子，逐值与
  `standard_1d.npz` 比较，状态按容差判定而不是只要等长就写 `ok`；峰值比较
  先提取**初次入射脉冲**（到达阈值、严格回零、脉冲内最大值），再分别对原始
  完整时序和导入帧求同一量，并记录峰值帧是否被抽帧选中；
- **C 经验关系与相容性**：Henrych（球形自由空气、入射超压，系数状态
  `provisional`）逐点比较；Kingery-Bulmash 因缺少可核实系数表仍标记
  `unavailable`，不猜测、不替换；RH 用实测激波位置差分和两侧实测状态独立
  检查，能量通量必须是 `u(E+p) - D_s E`；接触区间与 JWL/空气压力在 t0 之后
  的早、中、晚多个有效帧上采样，报告多个纯空气单元及压力随距离的变化；
  接触速度只称为两条区间边界轨迹的差分斜率，不称为物理速度上下界；
- **D PINN**：复用 `contact_air_diagnostics_*` 的输出。

审计只写 `outputs/reference_audit/`（manifest、CSV、JSON、图和 `report.md`），
不改缓存、检查点或训练配置。经验公式的数值样例测试在
`tests/test_contact_air_reference_audit.py`；Henrych 系数尚未与原书逐项核对，
报告必须保留这一限制。

本轮审计工具修正记录：旧版 RH 能量通量写成 `(E+p)(u-D_s)`，每侧多减
`D_s p`，会伪造能量残差；旧版“整段最大值 + 1% 阈值”会把脉冲内振荡误标为
二次脉冲。`one_d.py` 的缓存期 RH 诊断也从“比内能 + 相对动能”改为激波坐标系
总焓 `h + (u-D_s)^2/2`，`h = gamma p / ((gamma-1) rho)`。旧报告中的 −54%～−64% 能量残差、`partial_contaminated` 状态和单一
中间帧的 0.19～0.33 JWL/空气比均已撤回，以本轮重算结果为准。现有缓存中由
旧公式写入的 `event_frame_rh_diagnostic` 不重写；审计用缓存状态重新计算，
并把旧值单独保存为 `event_frame_rh_diagnostic_cache_recorded`。Kingery-Bulmash
仍未完成，不能把 `unavailable` 当作已完成。

## 7. 命令

本机验证使用的解释器是
`C:/Users/work/AppData/Local/Programs/Python/Python310/python.exe`（带 PyTorch）。
下面用 `python` 代指，并设置 `PYTHONPATH=src`。

```bash
# 0. 前置量表；base.yaml 没有配置算例时明确报告待接入项
PYTHONPATH=src python -m spherical_blast_pinn.cli.contact_air precheck \
  --config configs/contact_air/base.yaml --all

# 1. 生成合成标准一维夹具（工程检查，不是 TNT 数据）
PYTHONPATH=src python scripts/make_synthetic_standard_1d.py \
  --out outputs/contact_air_smoke/standard_1d.npz \
  --case tnt_50mm --radius 0.05 --frames 21 --cells 2400

# 2. 标准一维文件 -> 缓存
PYTHONPATH=src python -m spherical_blast_pinn.cli.contact_air build-cache \
  --config configs/contact_air/smoke_cached.yaml --case tnt_50mm \
  --radius 0.05 --source outputs/contact_air_smoke/standard_1d.npz \
  --out outputs/contact_air_smoke/cache.npz --role train

# 3. 训练 / 续训
PYTHONPATH=src python -m spherical_blast_pinn.cli.contact_air train \
  --config configs/contact_air/smoke_cached.yaml --cases all \
  --max-loops 2 --output-dir outputs/contact_air_smoke
PYTHONPATH=src python -m spherical_blast_pinn.cli.contact_air train \
  --config configs/contact_air/smoke_cached.yaml --cases all \
  --max-loops 3 --output-dir outputs/contact_air_smoke \
  --resume outputs/contact_air_smoke/last.pt

# 4. 预测（给定测得的 R_s 轨迹）
PYTHONPATH=src python -m spherical_blast_pinn.cli.contact_air predict \
  --config configs/contact_air/smoke_cached.yaml --case tnt_50mm \
  --checkpoint outputs/contact_air_smoke/last.pt \
  --output outputs/contact_air_smoke/prediction.npz --n-times 16 --n-zeta 16

# 5. 评估（单独读取参考内部场）
PYTHONPATH=src python -m spherical_blast_pinn.cli.contact_air evaluate \
  --config configs/contact_air/smoke_cached.yaml --case tnt_50mm \
  --prediction outputs/contact_air_smoke/prediction.npz \
  --output outputs/contact_air_smoke/evaluation.json

# 6. 测试
python -m pytest -q
```

也可使用等价的 `python -m spherical_blast_pinn.contact_air.cli ...`。`cli/contact_air.py`
只是转发入口，没有第二套参数解析。

`configs/contact_air/smoke.yaml` 是纯内存合成检查；`smoke_cached.yaml` 走真实
文件读取、缓存和预测加载链路。两者都是工程检查，不是数据划分或精度结果。

## 8. 数据状态与未解决项

- **没有真实 50 mm 一维 ALE1D 数据。** 未执行真实 50 mm 试跑。
- **二进制 ALE1D 读取未实现。** 不猜测槽位；需要真实文件样例、LS-DYNA 版本、
  字段/材料组映射、单位、时间轴、节点/单元约定。
- 中间格式可用不等于二进制读取已验证。
- 接触阈值、激波 prominence/距离参数尚未由真实数据确认。
- 材料卡 JWL/detonation/air 数值仍是配置占位值，
  `material_source_verified: false`；使用前必须对照实际卡片。
- 训练/验证/测试半径尚未确定；`base.yaml` 的 `cases` 为空是刻意的，不预设
  50/80/100 训练、500 测试。
- `outputs/` 下可能存在历史结果；它们不是本次输入，也不会被自动读取。

## 9. 仓库布局

```text
src/spherical_blast_pinn/
  config.py
  contact_air/
    one_d.py       标准一维入口、接触区间、激波检测、事件对齐
    precursor.py   CJ + JWL 等熵释放 + 空气 Hugoniot，epsilon/t0/Rc0/Rs0
    geometry.py    ShockTrajectory（PCHIP）、ContactIntervalTable、初始几何
    data.py        标准源 -> CaseData/Cache（v7）
    models.py      ContactNet / AirNet / JWL 闭合
    mapping.py     zeta 映射与三条空气残差
    losses.py      AirNet/ContactNet 损失、有限马赫数 RH、数据坐标
    sampling.py    帧优先采样、标签时刻、右边界资格
    trainer.py     两个持久 Adam 的交替训练、隔离检查、检查点
    cli.py         precheck/build-cache/train/predict/evaluate
    testing.py     小型工程夹具默认值
  cli/contact_air.py   唯一 CLI 的转发入口
scripts/make_synthetic_standard_1d.py   合成标准一维夹具生成器
tests/             当前方案测试与端到端流水线
```

## 10. 测试

当前测试覆盖：单位与必填语义、节点约定、逐字段掩码、空间缺口/重叠、接触区间
实际边界、非均匀压力梯度、重复峰 plateau、真正回退拒绝、PCHIP/Torch 导数与
`R_c` 完整乘积导数、事件平移与帧距、epsilon 因子、有限马赫数 RH 三条守恒、
初值范围、数据坐标跟随预测 Rc、域外计数、映射/物理 PDE 一致、两阶段梯度隔离、
检查点恢复与不匹配拒绝、删除内部参考数组后的预测、任意半径按 role 隔离，以及
标准 `.npz → build-cache → train → resume → predict → evaluate` 新进程全链路。

测试数量不是完成标准；以实际工作树、实际解释器和实际命令的输出为准。

## 11. 已知相容性差异

- 理论 `R_s0` 与事件帧测得 `R_s(t0)` 不一定相等；差值、`zeta_s0`、事件帧激波
  状态和 RH 诊断保存在 `CaseData.initial_compatibility()`。
- `p_c(t0)`（JWL）与 `p_a_sep` 的差值按实际大小报告，不强行相等。
- 若理论 `R_c0 >= R_s_data(t0)`，硬几何无法容纳初始接触位置，明确报错。
- 这些差异不会被“加入虚构轨迹点”或“静默改理论参数”消除。
