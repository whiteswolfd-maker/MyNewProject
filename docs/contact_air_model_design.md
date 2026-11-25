# 一维球对称 TNT 空气爆炸：ContactNet + AirNet 模型设计

本文描述当前仓库实际执行的方案，对应
`src/spherical_blast_pinn/contact_air/` 和
`configs/contact_air/split_train_r50_500_fullframes_similarity.yaml`。
旧 ProductNet/FrontNet、旧三维球壳提取、旧 `R_b` 环境边界和固定 500 mm 规则
不在当前方案中。

所有内部物理量使用 SI：m、s、kg/m³、m/s、Pa。压力为绝对压力；超压只在输出
诊断时形成，不进入 EOS/PDE/RH。

---

## 0. 一句话概括

用一个共享的 **ContactNet** 预测产物-空气接触面半径 `R_c(t)`，用另一个共享的
**AirNet** 在移动的归一化坐标 `zeta` 上预测空气场 `(rho,u,p)`。两个网络交替训练，
共享同一套：

- 移动域 `R_c(t) <= r <= R_s(t)`；
- 坐标映射 `zeta = (r-R_c)/(R_s-R_c)`；
- 球对称 Euler PDE；
- 接触界面条件、初始条件、有限马赫激波 Rankine-Hugoniot 条件；
- 来自 ALE1D 数据的接触区间和空气观测。

`R_s(t)` 和 `D_s(t)=dR_s/dt` 是外部数据，不是网络输出。

---

## 1. 物理问题与关键假设

### 1.1 物理图像

TNT 在空气中爆炸，分离后形成：

- 内部：爆轰产物，占据 `r < R_c(t)`；
- 外部：空气，占据 `R_c(t) < r < R_s(t)`；
- 外部边界 `R_s(t)`：向外传播的透射空气激波；
- 接触面 `R_c(t)`：产物和空气的材料界面。

只解空气区。产物区不求解 PDE，只通过 JWL 压力-比容关系提供接触面压力。

### 1.2 显式假设

1. 一维球对称；所有场只依赖 `(r,t)`。
2. 空气为理想气体，`gamma_a = 1.4`，无粘、无热传导、无化学反应。
3. 产物为 JWL 爆轰产物；只使用 JWL 压力闭合，不求解产物内部场。
4. 产物和空气之间无质量、动量和能量交换；接触面是材料界面。
5. 接触面压力连续、法向速度连续；无表面张力。
6. 激波 `R_s(t)` 和激波速度 `D_s(t)` 由实验/数值数据的压力梯度识别，作为
   固定外部数据；模型不预测激波轨迹。
7. 初始时刻 `t0` 在分离之后；`t0` 由前置解计算，不对应起爆瞬间。
8. `R_s` 只在连续、非下降的识别段内使用；平台会被记录，右边界不使用
   `D_s<=c_inf` 的样本；禁止外推、补 gap 或复制上一帧；回归段不建 PCHIP。
9. 数据是 ALE1D 单元平均值；接触面只能定位到零个或一个单元宽的混合带。

---

## 2. 符号与几何

| 符号 | 含义 |
|---|---|
| `R0` | 装药半径 |
| `R_c(t)` | 产物-空气接触面半径 |
| `R_s(t)` | 外传空气激波半径 |
| `D_s(t)` | `dR_s/dt`，激波速度 |
| `R_c_dot(t)` | `dR_c/dt`，接触面速度 |
| `L(t)` | `R_s(t)-R_c(t)`，空气层厚度 |
| `L_dot(t)` | `D_s(t)-R_c_dot(t)` |
| `zeta` | `(r-R_c)/L`，`0<=zeta<=1` |
| `r` | `R_c + zeta L` |
| `v_g` | `R_c_dot + zeta L_dot`，网格/坐标速度 |
| `T_star` | `R_sep/u_a_sep`，相似时间尺度 |
| `theta` | `(t-t0)/T_star` |
| `x_t` | `theta / 64`，送进 MLP 的无量纲时间 |

几何：

```text
r = R_c + zeta * L
L = R_s - R_c
v_g = R_c_dot + zeta * L_dot
```

硬约束：

- `L > 0`；
- `0 < R_c < R_s`；
- 所有时间查询必须落在同一个连续 `R_s` 段内；
- 不允许用截断厚度、外推或 gap filling 制造可计算几何。

---

## 3. 数据与标签

### 3.1 数据源

当前生产数据是 LS-DYNA ALE1D 的 d3plot state 数据库，经
`scripts/import_lsdyna_d3plot_1d.py` 转成标准中间格式，再由
`spherical_blast_pinn.contact_air.one_d.read_standard_1d` 读取。

每个 state 提供逐单元：

```text
r_left, r_right, radius (cell representative)
alpha_product, alpha_air
rho [kg/m^3], u [m/s 径向], p [Pa 绝对]
valid_density, valid_velocity, valid_pressure
```

标准格式还保存 `frame_offsets`、`frame_times`、`units`、`position_type` 等
metadata。读取阶段统一换算到 SI；体积分数不归一化、不填充，两介质
`alpha_product + alpha_air = 1` 必须在容差内。

### 3.2 接触区间

对每一帧：

1. `alpha_product >= 0.95` 的单元为纯产物；
2. `alpha_air >= 0.95` 的单元为纯空气；
3. 其余为混合单元；
4. 只取与中央产物区外缘相连的连续混合段；
5. `low` = 第一个混合单元的 `r_left`；
6. `up` = 最后一个混合单元的 `r_right`；
7. 没有混合单元时，`low = up` = 最后一个纯产物单元和第一个纯空气单元之间的
   共享边界；
8. 每一步都必须几何连续；有 gap/overlap 则整帧无效。

因此 `[low, up]` 的宽度是 **0 或 1 个 ALE 单元**，不是任意区间。训练/验证的
默认接触数据项是：

```text
d = max(low - R_c, 0) + max(R_c - up, 0)
L_contact_data = mean( (d / R_sep)^2 )
```

它只在模型跑到区间外时产生惩罚；区间内为零。`alpha=0.5` 等值面和区间中点
都不是默认标签。

可选的点目标（当前实验代码支持，默认不启用）：

```text
alpha_p = 混合单元内的产物体积分数
r_i = (r_l^3 + alpha_p (r_r^3 - r_l^3))^(1/3)
L_point = mean( ((R_c - r_i) / R_sep)^2 )
```

这是球壳体积反演，不是 `alpha=0.5` 插值；零宽帧取共享边界。点目标仍受数据
单元精度限制。

### 3.3 激波识别

`R_s(t)` 来自每一帧原始径向压力剖面的 **最外层显著正 `-dp/dr` 峰**：

- 只在有效、几何连续的空气单元上找；
- `relative_prominence=8`；
- 最小压力跳变 `1e-3`；
- 至少离接触面 1 个单元；
- 不加平滑；
- 只保留连续、非下降的段；
- 平台（`R_s` 相等）会记录；该段内 `D_s` 可为零，右边界采样会拒绝；
- 回归（`R_s` 下降）不修复，直接拒绝该段，不建 PCHIP。

`R_s(t)` 用 PCHIP 插值，`D_s` 用 PCHIP 的解析导数。torch 路径直接在计算图中
求值，保证 `dR_s/dt` 参与接触速度的乘积法则。

### 3.4 空气观测

对每个有效帧，在 `contact_up <= r <= R_s` 内取纯空气单元：

- 位置资格只看空气体积分数；
- 密度、速度、压力各自有独立 `valid_*` 掩码；
- 缺失压力不丢弃有效密度/速度；
- 观测存储**物理半径**，不存 `zeta`；
- 训练时用预测的 `R_c` 重新计算
  `zeta = (r - R_c(pred)) / (R_s - R_c(pred))`；
- `zeta` 不在 `[0,1]` 的点被计数并排除，不 clamp 到边界。

### 3.5 事件对齐

原始帧时间与模型时间的关系：

```text
t_model = t0 + (t_raw - t_event_raw)
```

`t_event_raw` 是满足以下条件的第一帧（且不早于理论 `t_sep`）：

- 接触区间可定位；
- 外激波可识别；
- 接触面和激波之间有至少一个连续纯空气单元。

`t0` 来自前置解。事件对齐只把数据平移到模型时间原点，**不声称**原始时间与
`t0` 是同一绝对物理时刻。

### 3.6 数据 split

当前全帧配置：

```text
train      : R50, R100, ..., R500      (10 cases)
validation : (removed 2026-11-23 together with R150-R1000)
```

每个 case 有 1002 个 d3plot state；接触可用帧约 983 个（去掉 t0 前和
`D_s` 不可用/无斜率帧）。验证集不采样训练，只用于评分和 best checkpoint。

---

## 4. 前置解：分离状态、初始几何和尺度

前置解 `compute_precursor` 从 JWL、爆轰和空气参数计算：

### 4.1 CJ 状态

```text
V_cj = rho_TNT / rho_CJ
u_cj = D_CJ (1 - rho_TNT/rho_CJ)
```

JWL：

```text
p(V) = A exp(-R1 V) + B exp(-R2 V) + C / V^(1+omega)
```

若 `C` 未给出，由 `p(V_cj)=p_CJ` 闭合：

```text
C = (p_CJ - A exp(-R1 V_cj) - B exp(-R2 V_cj)) V_cj^(1+omega)
```

### 4.2 释放到分离状态

沿 JWL 释放等熵线积分声速：

```text
u_c(V) = u_cj + integral_{V_cj}^{V} c(eta)/eta deta
```

与空气 Hugoniot 的粒子速度匹配，求 `V_sep`：

```text
u_c(V_sep) = u_a_sep(V_sep)
```

空气 Hugoniot（`strong` 模型，当前默认）：

```text
D_s = sqrt((gamma+1) p / (2 rho_inf))
u_a = sqrt(2 p / ((gamma+1) rho_inf))
rho_a = rho_inf (gamma+1)/(gamma-1)
e_a = p / ((gamma-1) rho_a)
```

`exact` 模型只是交叉检查选项，不改变主设计。

### 4.3 分离半径、时间和初始几何

```text
R_sep = R0 V_sep^(1/3)
t_sep = t_cj + integral_{R_cj}^{R_sep} 1/u_c(r) dr
epsilon = R_sep / (factor * (D_s_sep - u_c_sep))
t0 = t_sep + epsilon
Rc0 = R_sep + u_c_sep epsilon
Rs0 = R_sep + D_s_sep epsilon
```

当前 `epsilon_denominator_factor=200`；它是配置选择，不是实验验证常数。

### 4.4 初始压力兼容性

`p_c(t0)` 由 JWL 在 `R_c(t0)` 处计算，`p_a_sep` 是空气分离压力。二者差异
**只报告，不强制为零**；初始条件仍按下面的形式施加。

### 4.5 网络尺度

前置解给出：

```text
rho_scale = rho_a_sep
u_scale   = u_a_sep
p_scale   = p_a_sep
e_scale   = e_a_sep
length_scale = R_sep
time_scale   = R_sep / u_a_sep
gamma_air
```

这些尺度用于网络输出缩放、PDE 残差归一化和误差报告。

---

## 5. 网络结构

### 5.1 ContactNet

**输入：**

```text
x_t = scaled_time(t)       # theta/64 或 t/time_reference_s
R0  = 装药半径 [m]
```

注意：当前 ContactNet 的 MLP 只接收 `(x_t, R0)`。`R_s` 和 `D_s` 不进入 MLP，
只用于输出乘法。

**结构：**

- 5 个隐藏层 × 50 个神经元；
- `tanh` 激活；
- 输出 1 个标量 `raw`；
- 所有层 float64。

**输出：**

```text
s      = sigmoid(raw)
R_c    = R_s(t) * s
R_c_dot = D_s(t) * s + R_s(t) * s * (1-s) * d(raw)/dt
```

`sigmoid` 保证 `0 < R_c < R_s`。速度使用完整乘积法则，包含 `dR_s/dt` 和
`d(raw)/dt`。

**产物压力：**

```text
V   = V_sep * (R_c / R_sep)^3
p_c = A exp(-R1 V) + B exp(-R2 V) + C / V^(1+omega)
```

`V` 是无量纲相对比容，不与有量纲 `1/rho` 混用。

### 5.2 AirNet

**输入：**

```text
zeta
x_t = scaled_time(t)
R0  = 装药半径 [m]（可选 gate）
```

**结构：**

- 10 个隐藏层 × 32 个神经元；
- `tanh` 激活；
- 三个输出头：`rho_raw, u_raw, p_raw`；
- float64。

**输出：**

```text
rho = rho_scale * softplus(rho_raw)
u   = u_scale   * u_raw
p   = p_scale   * softplus(p_raw)
e   = p / ((gamma_a - 1) rho)
```

`softplus` 保证密度和压力为正；速度不加正性约束。

### 5.3 时间输入

两种显式模式：

```text
legacy:
    x_t = t / time_reference_s

similarity:
    T_star = R_sep / u_a_sep
    theta  = (t - t0) / T_star
    x_t    = theta / similarity_input_divisor
```

当前全帧配置使用 `similarity`，`divisor=64.0`。`divisor` 只是固定数值缩放，
不是物理常数，不能按各 case 的 `t_end` 或验证误差选择。

只有送进 MLP 的时间被缩放；PDE、界面、激波 RH、接触速度和自动求导仍按物理
秒计算。因此 `d x_t/dt = 1/(divisor*T_star)` 会进入链式法则。

可选的 `r0_input_gate`：

```text
none:
    R0_input = R0
similarity_time:
    gate = 1 - exp(-theta)
    R0_input = R0 * gate
```

当前默认 `none`。

---

## 6. 坐标映射、PDE 和边界条件

### 6.1 映射

```text
r      = R_c + zeta L
L      = R_s - R_c
L_dot  = D_s - R_c_dot
v_g    = R_c_dot + zeta L_dot
```

`zeta=0` 是接触面，`zeta=1` 是激波内侧。

### 6.2 球对称 Euler 方程

连续形式：

```text
mass:     rho_t + (rho u)_r + 2 rho u / r = 0
momentum: (rho u)_t + (rho u^2 + p)_r + 2 rho u^2 / r = 0
energy:   E_t + (u(E+p))_r + 2 u(E+p)/r = 0
E = p/(gamma-1) + 0.5 rho u^2
```

映射到 `(zeta,t)` 后，代码组装的残差是：

```text
mass:
  rho_t + [ (rho u)_zeta - v_g rho_zeta ] / L + 2 rho u / r

momentum:
  (rho u)_t + [ (rho u^2 + p)_zeta - v_g (rho u)_zeta ] / L
            + 2 rho u^2 / r

energy:
  E_t + [ (u(E+p))_zeta - v_g E_zeta ] / L
      + 2 u(E+p) / r
```

PDE 损失按三个残差分别用以下尺度归一化：

```text
mass_scale     = rho_scale / T_star
momentum_scale = rho_scale * u_scale / T_star
energy_scale   = E_scale / T_star
```

```text
L_pde = mean((mass/mass_scale)^2)
      + mean((momentum/momentum_scale)^2)
      + mean((energy/energy_scale)^2)
```

### 6.3 接触界面条件

在 `zeta=0`：

```text
u_air(R_c,t) = R_c_dot(t)
p_air(R_c,t) = p_c(R_c)
```

```text
L_interface = mean(((u_air - R_c_dot)/u_scale)^2)
            + mean(((p_air - p_c)/p_scale)^2)
```

### 6.4 激波边界条件

在 `zeta=1` 使用有限马赫 Rankine-Hugoniot 后激波状态：

```text
c_inf = sqrt(gamma_a p_inf / rho_inf)
M     = D_s / c_inf
```

必须 `M>1`；否则报错，不截断。

```text
rho_rh = rho_inf (gamma+1) M^2 / ((gamma-1) M^2 + 2)
u_rh   = 2 c_inf/(gamma+1) * (M^2-1)/M
p_rh   = p_inf (2 gamma M^2 - (gamma-1))/(gamma+1)
```

```text
L_RH = mean(((rho_air - rho_rh)/rho_scale)^2)
     + mean(((u_air - u_rh)/u_scale)^2)
     + mean(((p_air - p_rh)/p_scale)^2)
```

右边界只在 `D_s>c_inf` 的时间采样；不会把亚声速或外推激波静默用作边界。

### 6.5 初始条件

在 `t=t0`：

```text
R_c(t0) = Rc0
```

```text
L_contact_IC = ((R_c(t0)-Rc0)/R_sep)^2
```

空气在 `0<zeta<1` 上取分离态：

```text
rho = rho_a_sep
u   = u_a_sep
p   = p_a_sep
```

```text
L_air_IC = mean(((rho-rho_scale)/rho_scale)^2)
         + mean(((u-u_scale)/u_scale)^2)
         + mean(((p-p_scale)/p_scale)^2)
```

没有额外的环境子区间；`zeta=1` 是激波面，不是环境边界。

### 6.6 空气数据映射

空气观测原始坐标是 `(t,r)`。训练时：

```text
zeta = (r - R_c(pred)) / (R_s(t) - R_c(pred))
```

- `R_c(pred)` 是 ContactNet 的预测；
- `R_s(t)` 是识别激波；
- 只用 `0<=zeta<=1` 的点；
- 域外点计数并排除；
- 不使用接触区间中点替代 `R_c`；
- 不把域外点 clamp 到边界。

空气数据损失：

```text
L_air_data =
    mean_valid_rho(((rho-rho_data)/rho_scale)^2)
  + mean_valid_u(((u-u_data)/u_scale)^2)
  + mean_valid_p(((p-p_data)/p_scale)^2)
```

每个字段使用独立 mask；没有有效点时该项为零并计数。

---

## 7. 损失函数与交替训练

### 7.1 AirNet 阶段

固定 ContactNet 参数，用其预测几何但 detach ContactNet 的输出；AirNet 的
场和 `t/zeta` 导数保持可微。

```text
L_a = lambda_pde * L_pde
    + lambda_data * L_air_data
    + L_air_IC
    + L_interface
    + L_RH
```

### 7.2 ContactNet 阶段

固定 AirNet 参数，用 **functional** AirNet（冻结权重、保留输入导数）评估
`L_pde` 和 `L_interface`；ContactNet 的几何和输出保持可微。

```text
L_c = lambda_pde * L_pde
    + lambda_data * contact_data_weight * L_contact_data
    + L_contact_IC
    + L_interface
```

其中 `L_contact_data` 默认是区间投影损失；可选 point 损失。

### 7.3 权重 schedule

```text
progress = min(max(loop/n_schedule,0),1)
lambda_data = 0.3 + 0.3 * (1 + cos(pi * progress))
lambda_pde  = 1 - lambda_data
```

即：

- 起点：`lambda_data=0.9`，`lambda_pde=0.1`；
- 终点：`lambda_data=0.3`，`lambda_pde=0.7`；
- 当前配置 `n_schedule=2000`。

### 7.4 优化器和隔离

- 两个独立 Adam：`opt_contact`、`opt_air`，默认 `lr=1e-4`；
- 每个 stage 只更新目标网络；
- 反向传播后检查非目标网络没有梯度、参数没有移动；
- 可选全局梯度裁剪；
-  checkpoint 保存网络、优化器、RNG、history、validation batch 和 case
  signature；
-  resume 时检查 case signature、数据 split、采样数和损失约定；不匹配直接
  拒绝。

---

## 8. 采样、等权和验证

### 8.1 每 loop 的采样预算

配置给的是**总预算**，按 case 数等分：

```text
n_pde, n_ic, n_right, n_data, n_contact
```

当前全帧配置：

```text
n_pde=2048, n_ic=512, n_right=512, n_data=512, n_contact=512
```

10 个训练 case 每个约 205/51/51/51/51 个点。

### 8.2 等权

- 每一帧是采样单元；
- 每个抽到的点/标签权重相同；
- 没有 `theta`-cell 权重、没有早/晚帧加权；
- 同一帧被抽两次就贡献两个独立点；
- contact 标签从有效接触帧中均匀抽；
- air 数据是 frame-first，再在帧内抽点。

### 8.3 采样域

- PDE：`t` 在 `[t0,t_end]` 均匀，`zeta` 在 `(zeta_floor,1-zeta_floor)` 均匀；
- IC：`t=t0`，`zeta` 均匀；
- 右边界：只接受 `D_s>c_inf` 的 `t`，最多重抽 8 次；
- 接触：只接受 `t0<=t<=t_end` 且 `shock.has_slope(t)` 的有效帧；
- 空气数据：只用 `t0<=t<=t_end` 的帧。

### 8.4 验证

验证 batch 用固定 seed 只构建一次；验证集不采样训练。

单 case 分数：

```text
air_rmse = mean(rho_rmse, u_rmse, p_rmse)     # 各自归一化后
score    = air_rmse + contact_violation_Rsep
```

总分数是 case 平均。`contact_violation_Rsep` 是接触区间外距离的 RMSE，
单位 `R_sep`。验证分数用于 best checkpoint；不参与梯度。

当前全帧配置：

- `validation.holdout_fraction=0`；
- 外部验证集 R550–R1000 已于 2026-11-23 删除；
- 每个验证 case 固定 256 个空气点和 256 个接触点。

---

## 9. 当前训练配置

```text
data          : outputs/fullframes/cache/*.npz
train         : R50–R500 (10 cases)
validation    : (removed 2026-11-23)
frames        : 1002 states/case; ~983 usable contact frames
time input    : similarity, divisor=64
contact net   : 5 × 50, tanh, float64
air net       : 10 × 32, tanh, float64
loops         : 4000 (base default 2000)
learning rate : 1e-4 (Adam, both networks)
n_schedule    : 2000
contact data  : interval projection, normalization=R_sep
contact weight: 3.0
seed          : 20261110
batch totals  : 2048 / 512 / 512 / 512 / 512
```

`time_reference_s=0.01` 只在 legacy 时间输入下使用；similarity 模式不使用它。

---

## 10. 硬失败与不变量

以下情况必须报错，不能静默修复：

- `R_s` 查询在连续段外、gap 内或孤立帧上；
- `R_s` 回归段被强行建 PCHIP；
- `D_s <= c_inf` 的右边界样本；
- `L=R_s-R_c <= 0`；
- 网络输出非有限；
- 接触区间 `low>up`、非正、非有限；
- 数据标签 `valid=True` 但数值 NaN；
- 所有空气数据点都在 `(R_c,R_s)` 外；
- 两个网络的梯度隔离失败；
- checkpoint 的 case signature、采样数、损失约定不匹配；
- 体积分数和不为 1、出现空洞/第三介质。

---

## 11. 已知限制与当前设计问题

### 11.1 接触标签精度

- `[low,up]` 是 0.95 纯度阈值下的混合段，不是严格包住 `alpha=0.5` 界面的
  不确定带；
- 零宽帧的真实界面可能落在共享边界外最多约 0.05 个单元；
- 数据只能给出 0 或 1 个单元宽的接触带；
- 球壳点反演是子单元估计，零宽帧用边界，不是精确界面。

### 11.2 ContactNet 输入

ContactNet 的 MLP 只看到 `(x_t,R0)`；`R_s`、`D_s` 只用于输出乘法，不作为
网络输入。若数据中的接触轨迹还依赖激波强度/衰减历史，这是设计上的信息缺口。

### 11.3 共享 `theta` 的相似性

当前方案把所有 case 的 `theta=(t-t0)/(R_sep/u_a_sep)` 当作共同进度坐标。
近期全帧检查显示：

- 固定 `theta` 下，不同 case 的 `R_c/R_sep` 仍有明显离散；
- 只用 `theta` 的训练均值曲线在 validation 上约 55 cells RMSE；
- 当前模型在该阶段约 98 cells，甚至差于这个均值基线；
- 点目标与区间目标在当前几十 cell 误差下梯度几乎相同（cosine≈0.9999998）。

因此当前误差主要不是“区间 vs 点”的标签选择，而是：

1. 共享 `theta` 不是严格的单变量相似坐标；
2. ContactNet 没有学好 `(theta,R0)` 映射；
3. 多目标（PDE/IC/interface/contact）梯度方向存在冲突。

这些是当前研究的开放问题，不属于已解决的设计。

### 11.4 其他

- JWL/爆轰/空气参数在 `base.yaml` 是配置占位值，除非
  `material_source_verified=true`；不是材料卡验证值；
- 产物区没有 PDE，只有 JWL 压力闭合；不提供产物温度/能量场；
- 空气是理想气体、无粘、无反应；
- 激波识别依赖压力梯度阈值，阈值是配置选择；
- 事件对齐是时间平移，不声称绝对时间一致；
- 1D 球对称，不处理非球对称/射流/界面失稳。

---

## 12. 代码地图

| 模块 | 职责 |
|---|---|
| `models.py` | ContactNet、AirNet、JWL、时间输入、场定义 |
| `mapping.py` | `zeta` 映射、球对称 Euler 残差 |
| `losses.py` | PDE/IC/interface/RH/data/contact 损失、schedule |
| `trainer.py` | 交替训练、验证、checkpoint、梯度隔离 |
| `sampling.py` | 每 loop 的 PDE/IC/right/data/contact 采样 |
| `geometry.py` | 激波 PCHIP、接触区间表、初始几何 |
| `data.py` | CaseData、cache 读写、观测、事件对齐 |
| `one_d.py` | 标准 1D 格式、接触区间、激波检测 |
| `precursor.py` | JWL 释放、空气 Hugoniot、分离态、t0/Rc0/Rs0 |
| `cli.py` | precheck/build-cache/train/predict/evaluate |
| `reference_audit.py` | 数据/reference 审计 |

---

## 13. 一句话总结

这套模型是：

> 以实验/数值数据识别的 `R_s(t)` 为移动外边界，用共享 ContactNet 预测接触面
> `R_c(t)`、共享 AirNet 在 `zeta` 坐标上预测空气场；用球对称 Euler 残差、
> 接触界面条件、有限马赫激波 RH、初始条件和 ALE 接触/空气数据交替训练。

它的核心精度限制不在 PDE 形式，而在：

1. 接触面数据只有 0/1 单元宽；
2. 共同 `theta` 的相似性只是近似；
3. ContactNet 对 `(theta,R0)` 的拟合和它与物理项的耦合仍是开放问题。
