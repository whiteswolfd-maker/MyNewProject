# 单半径 R50：ContactNet + AirNet 完整设计（从头到尾）

本文记录当前实际跑通的**单半径**方案：只用 R50 数据训练，R100 只做零样本测试。
对应代码在 `src/spherical_blast_pinn/contact_air/`，最终检查点在
`outputs/single_r50_bounded_cheb_logp_6600/last.pt`。

这不是旧的交替训练版本。当前版本的关键约定是：

```text
τ = (t - t0) / (t_end - t0) ∈ [0, 1]
R_c = R0 + (R_s - R0) * sigmoid(raw)
联合训练：一次 backward，两个 Adam 同时 step
接触损失：区间 ReLU，按 R0 归一化
空气初值：用 t0 时刻的实测数据，不用均匀分离态
没有 cap，没有 holdout，没有多半径
```

---

## 0. 一句话概括

用 LS-DYNA ALE1D 的 R50 球对称算例，提取接触面区间 `[R_c,low, R_c,up]`、
外激波 `R_s(t)` 和空气场 `(rho,u,p)`；用前置解给出 `t0, Rc0, R_sep` 和
归一化尺度；用 ContactNet 预测 `R_c(t)`，用 AirNet 在移动坐标
`ζ=(r-R_c)/(R_s-R_c)` 上预测空气场；用球对称 Euler 残差、接触界面条件、
有限马赫 RH 激波条件、空气数据和接触区间数据联合训练。`R_s(t)` 和 `D_s(t)`
始终是数据，不是网络输出。

---

## 1. 数据生成

### 1.1 LS-DYNA 算例

- 一维球对称 ALE1D，TNT 在空气中爆炸。
- 20 个几何相似算例：R50, R100, ..., R1000（装药半径 0.05–1.0 m）。
- 每个算例 50,000 个 ALE 单元，外边界 100 R0；R50 外边界 5 m，单元宽
  `Δr = R0/500 = 1.0e-4 m`；R100 单元宽 `2.0e-4 m`。
- 本轮数据使用二阶格式 `METH=2`（strict van Leer）和时间步安全系数
  `TSSFAC=0.2`，用于消除旧数据在接触面附近的数值抹平。
- 材料卡来自 `G:/50-500/50/R50_meth2_tssfac02.k`，`material_source_verified=true`；
  所有半径共用同一 JWL/爆轰/空气参数，只缩放几何。
- 每个算例输出 1002 个 d3plot state。

### 1.2 d3plot 导入

`scripts/import_lsdyna_d3plot_1d.py` 读取 d3plot 第一个体积平均 ALE 层：

```text
axial stress      -> 体积平均压力 p
rs-shear          -> 体积平均密度 rho
tr-shear          -> 压缩比
plastic strain    -> 单位体积内能
axial strain      -> 体积分数加权 ALE group ID
alpha_air         = gid - 1
alpha_product     = 2 - gid
单元速度           = 梁单元两节点 x 速度平均
单元边界           = 该 state 的节点坐标
```

导出为标准一维 npz：

```text
frame_times, frame_offsets, cells_per_frame
r_left, r_right, radius
alpha_product, alpha_air
rho, u, p
valid_density, valid_velocity, valid_pressure
```

压力是绝对压力；超压只在后处理中形成，不进入 EOS/PDE/RH。

### 1.3 接触区间标签

`contact_air/one_d.py` 对每一帧按以下规则定位接触带：

1. `alpha_product >= 0.95` 的单元是纯产物；
2. `alpha_air >= 0.95` 的单元是纯空气；
3. 其余是混合单元；
4. 只取与中央产物区外缘相连的连续混合段；
5. `low` = 第一个混合单元的 `r_left`；
6. `up` = 最后一个混合单元的 `r_right`；
7. 没有混合单元时，`low = up` = 最后一个纯产物单元与第一个纯空气单元的共享边界；
8. 几何必须连续；出现 gap/overlap 整帧无效。

因此标签是**实际单元边界包住的区间**，不是 `alpha=0.5` 等值面，也不是区间
中点。当前 ALE 界面很锐，绝大多数帧的区间宽度正好是 0 或 1 个单元。

R50 在 `t0` 的第一帧（事件帧 19）：

```text
[R_c,low, R_c,up] = [0.1072, 0.1073] m
R_s(t0)          = 0.11445 m
```

### 1.4 激波识别

`R_s(t)` 是每一帧原始径向压力剖面的**最外层显著正 `-dp/dr` 峰**：

- 只在有效、几何连续的空气单元上找；
- `relative_prominence = 8`；
- 最小压力跳变 `1.0e-3`；
- 至少离接触面 1 个单元；
- 不加平滑；
- 只保留连续、非下降的段；平台记录但不当作可用的 `D_s`；
- 回归段不修、不补、不外推。

`R_s(t)` 用 PCHIP 插值，`D_s(t)` 用 PCHIP 的解析导数。torch 路径直接在
计算图中求值，保证乘积法则里的 `dR_s/dt` 正确。

### 1.5 空气观测

对每个有效帧，在 `contact_up <= r <= R_s` 内取纯空气单元：

- 位置资格只看空气体积分数；
- 密度、速度、压力各有独立 `valid_*` 掩码；
- 观测存**物理半径和物理时间**，不存 `ζ`；
- 训练时用当前预测的 `R_c` 重新算
  `ζ = (r - R_c(pred)) / (R_s - R_c(pred))`；
- `ζ` 不在 `[0,1]` 的点计数后排除，不 clamp。

R50 缓存中共有 2,504,713 个观测点；在 `[t0,t_end]` 和预测域内的有效点
2,503,964 个。

### 1.6 事件对齐

原始 d3plot 时间和模型时间不是同一个绝对时钟。对齐规则是：

```text
t_model = t0 + (t_raw - t_event_raw)
```

`t_event_raw` 是满足以下条件的第一帧，且不早于理论分离时间 `t_sep`：

- 接触区间可定位；
- 外激波可识别；
- 接触面和激波之间至少有一个连续纯空气单元。

R50 的结果：

```text
event_frame      = 19
t_event_raw      = 1.899910688996108e-05 s
t0               = 1.891385486802314e-05 s
offset           = -8.525202193794264e-08 s
```

事件帧映射到模型时间 `t0`。这只是时间平移，不声称 d3plot 的绝对时钟等于
前置解时钟。

---

## 2. 前置解

`contact_air/precursor.py` 从 JWL、爆轰和空气参数计算分离状态、`t0` 和
初始几何。全部为 SI。

### 2.1 CJ 状态

```text
V_cj = rho_TNT / rho_CJ
u_cj = D_CJ * (1 - rho_TNT / rho_CJ)
```

JWL：

```text
p(V) = A exp(-R1 V) + B exp(-R2 V) + C / V^(1+omega)
```

`C` 未给定时，由 `p(V_cj)=p_CJ` 闭合：

```text
C = (p_CJ - A exp(-R1 V_cj) - B exp(-R2 V_cj)) * V_cj^(1+omega)
```

### 2.2 等熵释放与空气 Hugoniot 匹配

沿 JWL 释放等熵线积分声速：

```text
u_c(V) = u_cj + ∫_{V_cj}^{V} c(η)/η dη
```

空气 Hugoniot（当前 `strong` 模型）：

```text
D_s = sqrt((gamma+1) p / (2 rho_inf))
u_a = sqrt(2 p / ((gamma+1) rho_inf))
rho_a = rho_inf (gamma+1)/(gamma-1)
e_a = p / ((gamma-1) rho_a)
```

求根：

```text
u_c(V_sep) = u_a_sep(V_sep)
```

### 2.3 分离半径、时间和初始几何

```text
R_sep = R0 * V_sep^(1/3)
t_sep = t_cj + ∫_{R_cj}^{R_sep} 1/u_c(r) dr
epsilon = R_sep / [200 * (D_s_sep - u_c_sep)]
t0 = t_sep + epsilon
Rc0 = R_sep + u_c_sep * epsilon
Rs0 = R_sep + D_s_sep * epsilon
```

`epsilon` 的 200 是配置选择，不是实验验证常数。它的作用是把初始时刻放在
分离之后一个很薄的时间层，避免从奇异的分离点直接开始。

R50 数值：

```text
V_cj        = 0.7409090909
V_sep       = 9.6652993991
C           = 1.3522530832e9
u_cj        = 1795.5 m/s
c_cj        = 5073.6152 m/s
p_cj        = 2.1e10 Pa
p_sep       = 7.1225390215e7 Pa
u_c_sep     = 6960.7935183 m/s
u_a_sep     = 6960.7935183 m/s
rho_a_sep   = 7.35 kg/m^3
e_a_sep     = 2.4226323202e7 J/kg
D_s_sep     = 8352.9522220 m/s
R_cj        = 0.0452437206 m
R_sep       = 0.1065062535 m
t_cj        = 6.5286754178e-6 s
t_sep       = 1.8531332914e-5 s
epsilon     = 3.8252195404e-7 s
t0          = 1.8913854868e-5 s
Rc0         = 0.1091689099 m
Rs0         = 0.1097014411 m
```

数据在 `t0` 的激波半径是 `R_s(t0)=0.11445 m`，与理论 `Rs0` 不完全相同；
这个差异只报告，不强行抹平。

### 2.4 归一化尺度

```text
rho_scale    = rho_a_sep      = 7.35 kg/m^3
u_scale      = u_a_sep        = 6960.7935 m/s
p_scale      = p_a_sep        = 7.1225390215e7 Pa
e_scale      = e_a_sep        = 2.4226323202e7 J/kg
length_scale = R_sep          = 0.1065062535 m
time_scale   = R_sep/u_a_sep  = 1.5300878162e-5 s
```

这些尺度用于网络输出缩放、PDE 残差归一化和误差报告。

---

## 3. 单半径数据集

```text
case_id      : r50_full
role         : train
R0           : 0.05 m
cache        : outputs/fullframes/cache/r50_full.npz
t0           : 1.8913854868e-5 s
t_end        : 9.9992645811e-4 s
window       : t_end - t0 = 9.8101260324e-4 s
cell width   : 1.0e-4 m
contact frames in [t0,t_end] : 983
air observations             : 2,504,713
```

没有验证集，没有 holdout。983 帧全部用于训练；这是单半径过拟合/可行性实验，
不是泛化实验。泛化由后面的 R100 零样本测试单独检查。

---

## 4. 时间输入

两个网络都使用同一个无量纲时间：

```text
τ = (t - t0) / (t_end - t0) ∈ [0,1]
```

代码位置：`models.py::TimeInput` 与 `TimeInput.window`。`data.py` 中：

```python
TimeInput(self.t0, self.time_scale, window=self.t_end - self.t0)
```

`window` 一旦设置，`scaled_time()` 使用

```text
(t - origin) / window
```

而不是旧的 `theta/64`。旧的 `similarity_input_divisor=64` 仍保存在
`run_metadata.json` 里，但在当前路径中不参与计算。

关键点：只有送进 MLP 的时间被归一化；PDE、界面、RH、接触速度和自动求导
仍然对**物理秒**求导。因此链式法则里自然出现 `dτ/dt = 1/window`，不需要
手工乘任何因子。

---

## 5. 网络结构

### 5.1 ContactNet

输入特征：

```text
x = [T_1(2τ-1), T_2(2τ-1), ..., T_30(2τ-1), R0]
```

- `T_k` 是 Chebyshev 多项式，`T_1(x)=x`，递推 `T_k=2xT_{k-1}-T_{k-2}`；
- 输入维度 `30 + 1 = 31`；
- MLP：6 个隐藏层 × 100 神经元，`tanh`，`float64`；
- 另有一条同输入的线性 skip `poly`，初始化为零；
- 实际输出 `raw = MLP(features) + poly(features)`。

Chebyshev 的作用：接触面 `R_c(τ)` 是一条很光滑、但早期变化很快的曲线。
普通 tanh MLP 在单半径上会停在 5–7 个单元的误差平台；30 阶 Chebyshev 加
零初始化线性 skip 给网络一条直接的多项式路径，能把接触误差压到 1 个单元
左右。

输出：

```text
s      = sigmoid(raw)
R_c    = R0 + (R_s(t) - R0) * s
R_c_dot = D_s(t) * s + (R_s(t) - R0) * s * (1-s) * d(raw)/dt
```

这是 `bounded_rs` 形式：

- 下界是 `R0`，对应论文里的装药半径下界；
- 上界是数据激波 `R_s(t)`，保证 `L=R_s-R_c>0`；
- 没有硬 cap，没有分段截断，没有零梯度分支；
- 速度用完整乘积法则，`D_s` 和 `d(raw)/dt` 都进入。

产物压力由 JWL 闭合：

```text
V   = V_sep * (R_c / R_sep)^3
p_c = A exp(-R1 V) + B exp(-R2 V) + C / V^(1+omega)
```

`V` 是无量纲相对体积，不与质量比容 `1/rho` 混用。

参数量：

```text
MLP:   31*100+100 + 5*(100*100+100) + 100+1 = 63,801
skip:  31+1 = 32
合计:  63,833
```

### 5.2 AirNet

输入：

```text
[ζ, τ, R0]
```

- `ζ = (r - R_c)/(R_s - R_c) ∈ [0,1]`；
- 10 个隐藏层 × 64 神经元，`tanh`，`float64`；
- 三个输出头：`rho_raw, u_raw, p_raw`。

输出：

```text
rho = rho_scale * softplus(rho_raw)
u   = u_scale   * u_raw
p   = p_scale   * softplus(p_raw)
e   = p / ((gamma_a - 1) * rho)
```

`softplus` 保证密度和压力为正；速度不约束符号。

参数量：

```text
3*64+64 + 9*(64*64+64) + 3*(64+1) = 37,891
```

---

## 6. 坐标映射与 PDE

### 6.1 移动域

```text
L      = R_s - R_c
L_dot  = D_s - R_c_dot
r      = R_c + ζ L
v_g    = R_c_dot + ζ L_dot
```

`ζ=0` 是接触面，`ζ=1` 是激波内侧。

### 6.2 球对称 Euler 方程

```text
mass:     rho_t + (rho u)_r + 2 rho u / r = 0
momentum: (rho u)_t + (rho u^2 + p)_r + 2 rho u^2 / r = 0
energy:   E_t + (u(E+p))_r + 2 u(E+p)/r = 0
E = p/(gamma-1) + 0.5 rho u^2
```

映射到 `(ζ,t)` 后，代码组装的残差是：

```text
mass:
  rho_t + [(rho u)_ζ - v_g rho_ζ]/L + 2 rho u/r

momentum:
  (rho u)_t + [(rho u^2 + p)_ζ - v_g (rho u)_ζ]/L
            + 2 rho u^2/r

energy:
  E_t + [(u(E+p))_ζ - v_g E_ζ]/L + 2 u(E+p)/r
```

PDE 损失按三个残差分别归一化：

```text
mass_scale     = rho_scale / T_star
momentum_scale = rho_scale * u_scale / T_star
energy_scale   = E_scale / T_star
T_star         = R_sep / u_a_sep
```

```text
L_pde = mean((mass/mass_scale)^2)
      + mean((momentum/momentum_scale)^2)
      + mean((energy/energy_scale)^2)
```

---

## 7. 损失函数

### 7.1 联合总损失

当前 `training_mode=joint`，一次 backward：

```text
L_total = L_flow + L_contact
```

其中：

```text
L_flow = λ_pde * L_pde
       + λ_data * L_air_data
       + L_interface
       + L_RH
```

```text
L_contact = w_c * mean_i [
    (ReLU(low_i - R_c,i)/R0)^2
  + (ReLU(R_c,i - up_i)/R0)^2
]
```

当前：

```text
w_c = contact_data_weight = 1.0
normalization = charge_radius (R0)
loss_shape = quadratic
beta = 1.0
```

### 7.2 权重 schedule

```text
progress = min(max(loop / n_schedule, 0), 1)
lambda_data = 0.3 + 0.3 * (1 + cos(pi * progress))
lambda_pde  = 1 - lambda_data
```

起点：

```text
lambda_data = 0.9, lambda_pde = 0.1
```

终点：

```text
lambda_data = 0.3, lambda_pde = 0.7
```

`n_schedule=3300`。前 3300 圈走完 schedule；从 3300 resume 到 6600 时，
`progress` 已经饱和，所以后 3300 圈固定为 `lambda_pde=0.7, lambda_data=0.3`。
这不是把 `n_schedule` 改成 6600；两者是不同的实验。

### 7.3 界面条件

在 `ζ=0`：

```text
u_air(R_c,t) = R_c_dot(t)
p_air(R_c,t) = p_c(R_c)
```

```text
L_interface = mean(((u_air - R_c_dot)/u_scale)^2)
            + mean(((p_air - p_c)/p_scale)^2)
```

### 7.4 激波条件

在 `ζ=1` 使用有限马赫 RH 后激波状态：

```text
c_inf = sqrt(gamma_a p_inf / rho_inf)
M     = D_s / c_inf
```

必须 `M>1`；否则硬失败，不截断。

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

右边界只采样 `D_s > c_inf` 的时间；不会把亚声速或外推激波静默用作边界。

### 7.5 空气数据损失

空气观测的坐标用**当前预测的接触面**重算：

```text
ζ = (r - R_c(pred)) / (R_s - R_c(pred))
```

- `R_c(pred)` 在 flow 分支里被 detach，作为固定几何；
- 只用 `0<=ζ<=1` 的点；
- 域外点计数后排除，不 clamp；
- 整批都在域外则硬失败。

损失：

```text
rho: mean_valid(((rho-rho_data)/rho_scale)^2)
u:   mean_valid(((u-u_data)/u_scale)^2)
p:   mean_valid((ln(max(p,p_amb)) - ln(max(p_data,p_amb)))^2)
```

压力用 log 损失的原因：线性损失按 `p_scale=7.12e7 Pa` 归一化，而晚期数据
压力只有 `1e5–1e6 Pa`；模型把压力塌到 `1e2 Pa` 时线性损失仍然很小。改成
以环境压力 `p_amb=101320 Pa` 为下限的 log 损失后，压力误差从 0.013 降到
约 0.001，同时没有改变 pressure 的正性约束。

### 7.6 接触损失

```text
d_i = max(low_i - R_c,i, 0) + max(R_c,i - up_i, 0)
L_contact = mean_i (d_i / R0)^2
```

区间内为零，区间外才有惩罚。R0 是装药半径，当前为 0.05 m；接触单元宽
`Δr=R0/500=1e-4 m`，所以 `(d/R0)^2 = (d/500Δr)^2`，损失天然以 500 个
“半单元”为单位。训练日志里的 `contact_data_cells_rmse` 是换算后的单元数，
仅用于诊断。

---

## 8. 采样

单 case 每个 loop 的预算：

```text
n_pde     = 2048
n_ic      = 64
n_right   = 512
n_data    = 512
n_contact = 983
```

具体含义：

- **PDE**：`t` 在 `[t0,t_end]` 均匀，`ζ` 在 `[1e-4, 1-1e-4]` 均匀；
- **右边界**：只接受 `D_s>c_inf` 的时间，最多重抽 8 次；
- **空气数据**：先均匀抽帧，再在帧内抽一个纯空气单元；共 512 个；
- **t0 数据**：额外抽 64 个 `t=t0` 的观测，追加到空气数据，合计 576 个。
  这就是“初始条件”的来源：不是均匀分离态，而是第一帧实测剖面；
- **接触**：983 个有效帧全部使用（`n_contact=983`），并强制包含窗口内第一帧。

等权：每帧是采样单元，抽中的点和标签权重相同；没有 `τ` 单元权重，没有
早/晚帧加权，同一帧抽两次就是两个独立点。

---

## 9. 优化与训练流程

```text
dtype          : float64
optimizer      : Adam, ContactNet 和 AirNet 各一个
learning rate  : 1e-4 (both)
gradient_clip  : 0
seed           : 20261110
loops          : 6600 total
n_schedule     : 3300
log_every      : 50
checkpoint_every: 100
validate_every : 0
```

每个 joint step：

1. 采样一个 `CaseBatch`；
2. ContactNet 在 PDE 时刻算 `R_c, R_c_dot`，**detach** 后送入几何；
3. AirNet 在 `(ζ,t,R0)` 上算场和 `t/ζ` 导数；
4. 组装 `L_pde`、`L_interface`、`L_RH`、`L_air_data`，得到 `L_flow`；
5. ContactNet 在接触标签时刻重新算 `R_c`（**不 detach**），算区间损失；
6. `L_total = L_flow + L_contact`，一次 `backward()`；
7. 检查两个网络都有有限梯度；
8. `opt_air.step()` 和 `opt_contact.step()` 同时更新。

因此：

- AirNet 只从 flow 项拿梯度；
- ContactNet 只从接触区间项拿梯度；
- 接触项的 `R_c` 是 live 的，flow 项的 `R_c` 是 detached 的；
- 没有交替 stage，没有 contact IC 项，没有 uniform air IC 项。

训练分两段：

```text
第一段：0 -> 3300，从零开始，n_schedule=3300
第二段：3300 -> 6600，从 last_3300.pt resume
```

resume 恢复网络、Adam 状态、RNG、history 和 case signature；`n_schedule`
仍是 3300，所以第二段固定使用 schedule 终点权重。

---

## 10. 预测与评估

### 10.1 预测

`predict` 生成：

```text
times  = linspace(t0, t_end, n_t)
zeta   = linspace(0, 1, n_z)
```

最终评估使用 `983 × 64` 网格。输出包括：

```text
radius, contact_radius, contact_speed
shock_radius, shock_speed
density, velocity, pressure, specific_energy
```

### 10.2 空气误差

`evaluate` 把每个观测点用预测的 `R_c` 映射到 `ζ`，再在预测网格上插值：

```text
rho_rmse = sqrt(mean(((rho_pred-rho_data)/rho_scale)^2))
u_rmse   = sqrt(mean(((u_pred-u_data)/u_scale)^2))
p_rmse   = sqrt(mean(((p_pred-p_data)/p_scale)^2))
```

### 10.3 接触误差

```text
d_i = max(low_i - R_c,i, 0) + max(R_c,i - up_i, 0)
mean_violation_over_Rsep = mean(d_i / R_sep)
coverage = 区间内帧数 / 983
```

另外把 `d_i/Δr` 换算成“接触单元”数，报告 mean、RMSE、median、p90、max。

---

## 11. 最终结果

### 11.1 接触面

```text
mean violation       = 0.692 cell
RMSE                 = 1.304 cell
median               = 0.392 cell
p90                  = 1.790 cell
max                  = 19.93 cell（早期单帧离群）
coverage             = 27.4% (官方 evaluate，精确帧)
mean(d)/R_sep        = 6.466e-4
```

分 τ 区间（预测均匀网格，单位 cell）：

| τ | n | mean | RMSE |
|---|---:|---:|---:|
| 0.00–0.05 | 50 | 2.176 | 4.109 |
| 0.05–0.10 | 49 | 1.574 | 1.980 |
| 0.10–0.20 | 98 | 1.194 | 1.414 |
| 0.20–0.40 | 196 | 0.627 | 0.897 |
| 0.40–0.60 | 197 | 0.345 | 0.516 |
| 0.60–0.80 | 196 | 0.303 | 0.455 |
| 0.80–1.00 | 197 | 0.644 | 0.975 |

早期误差大是因为接触面在几微秒内快速加速，一帧的相位差就会变成几个单元；
中后期已经稳定在半个单元以内。

### 11.2 空气场

```text
rho_rmse = 0.019728   -> 0.145 kg/m^3
u_rmse   = 0.002837   -> 19.75 m/s
p_rmse   = 0.000963   -> 6.86e4 Pa
```

归一化误差都在 2% 以内；物理上压力误差约 0.07 MPa，相对当地压力随位置变化。

### 11.3 训练损失

最后一圈：

```text
air_loss       = 4.516e-3
contact_loss   = 6.899e-6
```

`contact_loss=6.899e-6` 对应

```text
sqrt(6.899e-6) * R0 / Δr = 1.313 cell
```

与评估 RMSE 1.304 cell 一致。

---

## 12. 与经验公式对比

脚本：`scripts/compare_empirical.py`。

比较对象是自由空气 TNT 的**峰值入射超压**：

```text
Z = R_s / W^(1/3)
W = (4/3) pi R0^3 rho_TNT
```

R50：`W=0.853466 kg`，`W^(1/3)=0.948554`，`Z=0.1207–1.3376`。
R100：`W=6.827728 kg`，`W^(1/3)=1.897108`，几何缩放后 Z 范围相同。

Henrych (1979)，MPa：

```text
0.05 <= Z <= 0.3:
  dp = 1.40717/Z + 0.55397/Z^2 - 0.03572/Z^3 + 0.000625/Z^4
0.3 < Z <= 1:
  dp = 0.61938/Z - 0.03262/Z^2 + 0.21324/Z^3
1 < Z <= 10:
  dp = 0.0662/Z + 0.405/Z^2 + 0.3288/Z^3
```

Kinney–Graham (1985)，Pa：

```text
dp = P0 * 808 * (1+(Z/4.5)^2)
     / [sqrt(1+(Z/0.048)^2)
        * sqrt(1+(Z/0.32)^2)
        * sqrt(1+(Z/1.35)^2)]
```

ALE 峰值取激波后 5 个纯空气单元内的最大压力；模型取 `ζ=1`。

结果：

| 对比 | R50 | R100 零样本 |
|---|---:|---:|
| PINN vs ALE 数据 | 13.82% | 13.51% |
| ALE 数据 vs Henrych | 18.38% | 18.38% |
| ALE 数据 vs Kinney–Graham | 34.21% | 34.21% |

结论：模型与数据的差距（约 14%）小于数据与经验公式的差距（18%/34%）。
也就是说，当前主要的“模型 vs 公式”差异来自 ALE 数据本身与经验公式的差异，
不是 PINN 没拟合数据。

图：

```text
outputs/empirical_comparison_r50/peak_overpressure_vs_Z.png
outputs/empirical_comparison_r100_zero_shot/peak_overpressure_vs_Z.png
```

---

## 13. R100 零样本测试与“等比例采样”

R100 用同一个 R50 checkpoint，不重训、不微调：

```text
contact mean violation = 0.017533 over R_sep
                       = 18.67 cell
coverage               = 0.71% (7/983)
rho_rmse               = 0.02554
u_rmse                 = 0.00309
p_rmse                 = 0.00132
```

分 τ 区间的接触误差（单位 cell）：

```text
τ 0.00–0.05   +0.17
τ 0.05–0.10   +3.08
τ 0.10–0.20   +7.08
τ 0.20–0.40   +10.00
τ 0.40–0.60   +21.75
τ 0.60–0.80   +27.99
τ 0.80–1.00   +30.76
```

误差全为正，且随 τ 增大。原因是 `bounded_rs` 里的 `s`：

```text
R_c = R0 + (R_s - R0) * s
```

网络对 `R0` 的输入是原始米制。训练只见过 `R0=0.05`，R100 输入 `R0=0.10`。
同一个 τ 下，网络输出的 `s` 有约 `+0.0021 ~ +0.0028` 的正偏差；R100 的
`R_s-R0` 晚期约 2.4 m，于是

```text
0.0025 * 2.4 / 2e-4 ≈ 30 cell
```

正好对上晚期 28–31 cell 的误差。

所以：

- 如果“等比例采样”指**时间采样**：时间已经是 `τ∈[0,1]`，不是问题；
- 如果指**多半径按相同权重采样训练**：这会帮助网络学到 `R0` 依赖，但它不是
  零样本失败的唯一原因，因为 R100 根本没进训练；
- 直接原因是 **`R0` 输入没有无量纲化**，小 `s` 偏差被 R100 的大半径放大。

最干净的修法是：

```text
把输入 R0 改成 R0/R_sep（或 R0/R_s(t)）
```

R50 和 R100 的 `R0/R_sep` 都是 `0.4695`，网络看到的输入完全一致，`s(τ)`
可以直接迁移；然后重训 R50，再零样本跑 R100。多半径等比例采样是另一条路线，
用于让网络显式学习半径依赖。

---

## 14. 已知限制

1. 接触标签只有 0 或 1 个单元宽；`alpha=0.5` 界面不是标签，区间中点也不是。
2. 单半径训练没有泛化验证；R100 零样本是分布外测试。
3. `R0` 原始米制输入是跨半径迁移的主要障碍。
4. 早期接触误差受帧相位影响大；单帧离群可达 20 cell。
5. 压力数据用 log 损失是为了避免压力塌缩；它改变了损失的数值形状，但没有
   改变压力正性约束。
6. 激波 `R_s` 和 `D_s` 是数据，不是网络预测；模型是“给定实测激波轨迹求
   空气场和接触面”，不是自主激波预报。
7. `epsilon` 的 200、接触阈值 0.95、激波 prominence 8 都是配置选择，不是
   独立实验验证常数。

---

## 15. 代码地图与命令

| 模块 | 职责 |
|---|---|
| `one_d.py` | 标准一维格式、接触区间、激波检测、事件对齐 |
| `precursor.py` | CJ、JWL 等熵释放、空气 Hugoniot、`t0/Rc0/Rs0` |
| `geometry.py` | `ShockTrajectory`（PCHIP）、接触区间表、初始几何 |
| `data.py` | 标准源 -> `CaseData`/cache v7、观测、时间输入 |
| `models.py` | ContactNet、AirNet、JWL 闭合、`TimeInput` |
| `mapping.py` | `ζ` 映射、三条球对称 Euler 残差 |
| `losses.py` | PDE、界面、RH、空气数据、接触区间损失 |
| `sampling.py` | 帧优先采样、t0 数据、右边界资格、接触标签 |
| `trainer.py` | joint step、优化器、检查点、梯度检查 |
| `cli.py` | `precheck/build-cache/train/predict/evaluate` |

关键配置链：

```text
_tmp_train_50_500_meth2.yaml
  -> _tmp_single_r50_paper_joint.yaml
    -> _tmp_single_r50_bounded_wide.yaml
      -> _tmp_single_r50_bounded_cheb.yaml
        -> _tmp_single_r50_bounded_cheb_air64.yaml
          -> _tmp_single_r50_bounded_cheb_logp.yaml
```

最终训练：

```bash
PYTHONPATH=src python -m spherical_blast_pinn.contact_air.cli train \
  --config configs/contact_air/_tmp_single_r50_bounded_cheb_logp.yaml
```

从 3300 续到 6600：

```bash
PYTHONPATH=src python -m spherical_blast_pinn.contact_air.cli train \
  --config configs/contact_air/_tmp_single_r50_bounded_cheb_logp.yaml \
  --resume outputs/single_r50_bounded_cheb_logp/last_3300.pt
```

预测与评估：

```bash
PYTHONPATH=src python -m spherical_blast_pinn.contact_air.cli predict \
  --config configs/contact_air/_tmp_single_r50_bounded_cheb_logp.yaml \
  --checkpoint outputs/single_r50_bounded_cheb_logp_6600/last.pt \
  --case r50_full --output outputs/single_r50_bounded_cheb_logp_6600/pred_r50_full.npz \
  --n-times 983 --n-zeta 64

PYTHONPATH=src python -m spherical_blast_pinn.contact_air.cli evaluate \
  --config configs/contact_air/_tmp_single_r50_bounded_cheb_logp.yaml \
  --case r50_full \
  --prediction outputs/single_r50_bounded_cheb_logp_6600/pred_r50_full.npz \
  --output outputs/single_r50_bounded_cheb_logp_6600/eval_r50_full.json
```
