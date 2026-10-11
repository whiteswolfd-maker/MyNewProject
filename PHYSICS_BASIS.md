# 恒定爆速强支模型的物理依据核实（提示词 §5）

版本：2026-09-23。对象：`physics/detonation.py` 的产物场构造。

**核实方式**：三份原始 PDF 由 `curl` 下载后用 `pdftotext -layout` 提取全文，逐段
阅读原文；不是从记忆或二手综述转述。下面每条都标注是**原文直接陈述**、**本项目
推断**，还是**未核实**。

---

## 0. 被核实的构造

* 只解产物区；未反应装药静止、零压、零内能（能量零点）。
* 波前位置给定为 `r_f(t) = D t`，`D` 取卡片值 6930 m/s 且**恒定**。
* 波后状态由 Rayleigh–Hugoniot（质量/动量/能量 + 放热 `Q = E0/rho_ref`）确定，
  代入 JWL 后解 `D_RH(V) = D`。
* 本卡片 `D_CJ = 6883.14 < 6930`，所以有两个根；取**声速比
  `c/(D-u) > 1`**（波后相对波前亚声速）的那一支。

---

## 1. Taylor (1950), *Proc. R. Soc. A* **200**, 235–247

原文获取成功（有文本层）。

**(1a) 假设 —— 与我们的模型一致的部分（原文直接陈述）**

* 未反应炸药静止、反应在薄层内：引言，p.235 —— "…the whole chemical reaction in the
  explosive takes place within a narrow region, the explosive being at rest before this
  thin detonation front reaches it"。
* 波前恒速、`R = Ut`：p.238 与 p.243（式(26) 前后）—— "a detonation front is assumed
  to move with uniform velocity U into a stationary explosive"；"If R is the radius of
  the wave at any time R = Ut"。
* 球面产物流可以自相似：p.242 —— "…a steadily expanding regime would be set up…
  the radial velocity u and also p, ρ and c would be functions of r/t only"。

**(1b) 波前被固定为 CJ —— 与我们**不一致**的部分（原文直接陈述）**

* p.243 §Boundary conditions："The Chapman-Jouguet condition, together with the
  Rankine-Hugoniot equation, determines the detonation velocity U and also the value of
  `u/U = 1 - 1/μ` at the wave front."
* 式(31) 前："The Chapman-Jouguet condition, namely `u/U + c/U = 1`, gives…"
* Table 2 在 `r/R = 1.0` 处 `u/U = 0.2424`、`c/U = 0.7575`，与 `(U-u₁)/U = 0.7576`
  在浮点内相等，即波前处于 `c = U - u` 的声速点。

**结论：Taylor 的球面解不允许过驱动（强支）波前。他用的是自由度最高的那个
解——`D = D_CJ`。我们用的 `D = 6930 > 6883.14`，Taylor 1950 不覆盖这一情形。**

**(1c) 恒速球面爆轰是否需要外部驱动（原文直接陈述）**

* p.246："The fact that the velocity drops to zero at some point between the detonation
  surface and the centre shows that **a spherical detonation wave can maintain itself**
  in the particular case investigated."
* 同页 §Previous work：他把这当作对 Jouguet (1907) 的反驳 —— "The view here put
  forward is **contrary to** that expressed by Jouguet that spherical detonation waves
  moving with constant speed probably cannot exist."

**所以 Taylor 的立场是：CJ 速度下的恒速球面爆轰**可以自持**，不需要外部活塞、
不需要额外驱动。**但这是 CJ（非过驱动）情形下的结论，不能直接搬到我们的强支上。

* 全文检索 "piston / driver / external / overdriven / strong / weak" 在 Taylor 文中
  **零命中**（除 "tube strong enough" 一类无关用词）。他从不使用强支/弱支概念。

**(1d) 自相似的代价（原文直接陈述）**

* p.247："the equations for the flow behind a spherical wave do not admit of a simple
  solution of the type obtained by Riemann for a plane wave"。
* p.241："**Progressive spherical waves of finite amplitude analogous to progressive
  Riemann waves do not exist.**"
* 摘要："The radial rate of change of the variables, velocity, pressure and density
  become infinite at the detonation front, but it is unlikely that this result would be
  true in a real explosive where the time of reaction is not zero."

**(1e) 未核实项**：p.247 有一段讨论 `U > u₁ + c₁`。该处的**精确不等号方向在提取
文本里不可靠**（OCR 噪声），且与 Table 2 的数值存在张力。**标记为未核实**，
不要以它为依据。这不影响 (1b)：他的边界条件式(31) 明确是 CJ。

---

## 2. Caltech SDToolbox / Shepherd 组手册

来源身份更正：`ShockDetonation.pdf` 实际是 **SDToolbox 完整手册** ——
*Numerical Tools for Shock and Detonation Wave Modeling*, Explosion Dynamics
Laboratory, GALCIT Report **FM2018.001**（Rev. 2024-08-01），300+ 页。
以下按该手册章节引用。

**(2a) 定义（原文直接陈述，§6.5 p.73–76；§C.1 p.299–301）**

* CJ：§6.5 "Definition I: The Chapman-Jouguet detonation velocity is the minimum wave
  speed for which there exists a solution to the jump conditions from reactants to
  equilibrium products…"；"Definition II: …occurs when the flow in the products is
  **sonic** relative to the wave."
* 过驱动：§7.6 p.93 "…often observed to have a velocity in excess of the
  Chapman-Jouguet speed for that mixture, `U > U_CJ`. **Such waves are referred to as
  overdriven**…"
* 强/弱支：§6.5 p.74 "Above this minimum wave speed `w₁ > U_CJ`, the Rayleigh line and
  equilibrium Hugoniot intersect at two points, usually known as the **strong (S)** and
  **weak (W)** solutions."

**(2b) 选支判据的方向 —— 本节的裁定**

* §6.5 p.73："according to **Jouguet's rule**… these solutions have **subsonic flow
  behind the wave `w₂ < a₂`** and satisfy the condition of causality…"
* §C.1 p.301："**Jouguet's rule: The flow downstream is subsonic or supersonic if the
  slope of the Hugoniot is smaller or larger than the slope of the Rayleigh line.**
  … **The physically reasonable solutions for both detonations and deflagrations have
  subsonic states (strong solution) downstream of the wave.**"

**(2c) 过驱动波实际实现哪一支（原文直接陈述，§6.5 p.76）**

"Consider a detonation wave traveling **faster than the CJ velocity** such that the
state behind the wave is **the upper intersection (S - the strong solution)** and the
flow behind the wave is **subsonic relative to the wave front**. In this situation,
perturbations from behind the detonation wave can propagate through the flow and
interact with the leading shock. In particular, if the perturbations are expansion
waves, these perturbations will **eventually slow the lead shock to the CJ speed**."

紧接着："A similar argument cannot be made for the lower (W) or **weak solution which
has supersonic flow behind the wave** relative to the wave front."

**(2d) 三处独立确认过驱动波后相对波前是亚声速**

* §6.5 p.76（上引）
* §11.10 p.202："**Overdriven detonations** and shock waves with purely endothermic
  reaction zones have values of `M < 1` throughout the reaction zone, i.e., **the flow
  is always subsonic relative to the shock**."（该手册 `M ≡ 1 - (U-u)^2/a²`，故
  `M > 0 ⟺ U-u < a`，即亚声速）
* §11.11 p.225："Since the flow is **subsonic behind an overdriven detonation wave or
  shock**… **the wave will decay**, `dP/dt < 0`… As the wave approaches the CJ
  condition, `M → 0`… the wave will tend to a steady wave `dP/dt = 0` with `U - u = a_e`
  at large times."

### 对代码的裁定

代码取 `c/(D-u) > 1`，等价于 `(D-u) < c`，等价于该手册的 `M > 0`，即
**亚声速 = 强支 = 过驱动爆轰物理上实际实现的那一支**。

> **代码的分支选择方向是正确的，有三处独立原文支持，应予保留。**
> 代码里的命名（"subsonic/strong"）与来源一致。

**但必须记录两条限定：**

1. **原文支持**：`D > D_CJ` 的过驱动波**不是自持的**；没有外部驱动（活塞/强激波/
   约束）时，波后膨胀波会把它减速回 CJ（§6.5 p.76 与 §11.11 p.225 两处）。
   所以"恒定 `D > D_CJ`"在物理上**隐含一个持续驱动器**。
2. **本项目推断**：**"球面 + 过驱动 + 恒定 D"这个组合，三篇来源都没有处理。**
   Taylor 是球面但固定 CJ；SDToolbox 的强/弱判据是**一维稳态平面跳跃条件**
   （P–v 图几何，无球面曲率项）。把这个判据直接外推到球面是**项目自创**，需要
   自行论证（例如球面发散项是否改变分支可实现的判据）。

---

## 3. Caramana, Burton, Shashkov, Whalen (1998), *J. Comput. Phys.* **146**, 227–262

原文获取成功。

**(3a) 相容性的机制（原文直接陈述，§2.1.2–2.1.3）**

* corner force `f^p_z`（式(7) 下方）：常压 `P_z` 时 `f^p_z = P_z(S₂ - S₁)`。
* 动量（式(7)）：`M_p dv_p/dt = -∫P dV = -∮P dS = Σ_z f^p_z ≡ F_p`。
* **相容内能更新（式(12)）：`M_z de_z/dt = -Σ_p f^z_p · v_p`**。
* p.235："a corner force `f^z_p`, whatever its origin, produces a momentum change by
  acting on its associated point p, and **does work at the rate `-f^z_p · v_p`** with
  respect to its associated zone z."

**(3b) 力功恒等式（原文直接陈述）**

* **节点质量 = 该节点周围所有 corner mass 之和**（式(5)：`M_z = Σ_p m^z_p`，
  `M_p = Σ_z m^p_z`），**不是**相邻两格质量平均。
* 体积增量必须由 corner 向量分解给出：`dV_i = Σ_j C^j_i · dx_j`（式(27)）。
* 最简精确形式（式(25)）：`Δe_z = -(Σ_p m^z_p v_p·Δv_p)/M_z`；对所有 zone 求和并
  交换次序得 **`Σ_p M_p v_p·Δv_p + Σ_z M_z Δe_z = 0`**。

**1D 球面特化：论文没有印出显式公式**（其显式算例是 2D 柱坐标，§3；Sedov 球面
爆轰算例也在 2D 柱坐标里跑）。1D 球面是同一个构造的退化 —— **这是本项目推断，
需要自行验证。** 我们已经在 `DISCRETE_BUDGET.md` §3.2 里对自己的格式做了这一步
验证（分部求和恒等式，实测 1.7e-8）。

**(3c) 给定速度边界必须计边界功（原文直接陈述，§4.2 p.254–256）**

* 总能量（式(54)）：`E_T(t) = E_T(0) + Σ_m Σ_i W^m_{bd,i}`，边界功率
  `W^m_{bd,i} = f^m_{bd,i} · v^{m+1/2}_{bd,i} Δt`。
* **给定速度边界（式(55)）**："Suppose the velocity at the boundary is specified.
  Then in order to compute total energy balance from Eq.(54) we must find the work…
  **Since the velocity is known the momentum equation, given as Eq.(17), can be used
  to find f_bd,i**; thus the boundary work is given by
  `W = v_bd · (M_i Δv_bd,i - Σ_z f^{i,z} Δt)`"。
* 边界节点质量只来自内侧相邻单元（§2.1.1 p.233：外部 corner mass 假定为 0）。

**对我们的含义**：波前节点的位置**是给定的**（`r_f = Dt`），所以它就是一个
给定速度边界，**边界功这一项不能默认置为 0**。我们在 `DISCRETE_BUDGET.md` §4.3
里实测过引入它的效果（"节点 k 加来流动量"那一行），结论是单独加会把偏差放大
4 倍——因为波前节点的**速度**（`u_f`）与它的**位置变化率**（`D`）本来就不一致
（这是非物质的波前节点），这一项的相容处理必须与波前节点的整体重写一起做。

**(3d) 框架**不覆盖**变质量单元（原文直接、反复陈述）**

* §2.1.1 p.233："We consider both `M_z` and `M_p` on a totally equal footing. Thus,
  in the rest of our development we assume that **both `M_z` and `M_p` are constant**,
  Lagrangian objects."
* §2.1.4 p.238："**An important part of the derivations just completed is that the nodal
  mass as well as the usual zonal mass is considered constant.** Often in Lagrangian
  algorithms only the zonal mass is constant, and the nodal mass is recomputed on each
  timestep as a different number. **This leads to a flux of momentum and kinetic energy
  from the nodes and results in an extra term `v_p(dM_p/dt)` that should appear on the
  LHS of the momentum equation, but which is often neglected in practice.** By keeping
  `M_p` constant in time we have employed a stronger form of the Lagrangian assumption
  than usually given."
* §5 结论 p.259 重述同一条。

**对我们的含义**：我们的节点质量**必须**随时间变化（前端在吸积），所以整个
Caramana 框架**不覆盖**我们的情形——`Σ M_j u_j du_j = Σ P_i dV_i` 那一族恒等式必须
为质量通量项做修正。**论文对 accretive / growing-mass cell 没有任何处理。**
这是本项目必须自行补齐的部分，不能引用它当作已经推导过。

**(3e) 人工粘性的相容性要求（原文直接陈述）**

* p.244："**When forces are specified in a heuristic manner, as is the case for an
  artificial viscosity, the conservation of energy approach enables one to construct
  the work equation in a simple, unique, and generic manner…** In the staggered grid
  case this information is already given by the assumption that a corner force `f^p_z`,
  whatever its origin, acts on a point p and does work with respect to zone z.
  **There is no arbitrariness in this arrangement**…"
* p.239（式(24) 前后）：粘性 "**each term of Eq.(24) is required to be positive
  definite**"（单调耗散，熵不减）。

**对我们的含义**：粘性力与粘性耗散必须由**同一个** `q_i` 导出，不允许给粘性力
单独配一个"看起来合理"的做功公式。我们的实现满足这一条（见
`DISCRETE_BUDGET.md` §2.7）。

---

## 4. 状态汇总

### 资料直接支持

| 项 | 来源 |
|---|---|
| 只解产物区、未反应炸药静止零压 | Taylor 1950 引言 p.235 |
| 波前恒速、`r_f = D t` | Taylor 1950 p.238/p.243 |
| 波后状态由 RH（质量/动量/能量 + 放热 Q）确定 | Taylor 式(9)–(13)；SDToolbox §6.1–6.3 |
| 球面产物流可自相似（点起爆 + 恒定 U） | Taylor 1950 p.242 |
| **`c/(D-u) > 1` 取强支/亚声速是过驱动爆轰实际实现的支** | SDToolbox §6.5 p.73/76、§C.1 p.301、§11.10/11.11 |
| 交错拉氏的力/功/内能需由总能量守恒耦合 | Caramana 式(7)(12)(25) |
| 给定速度边界必须计边界功 | Caramana 式(54)(55) |
| 粘性力与耗散须同源、且正定 | Caramana p.239/p.244 |

### 本项目推导 / 工程选择（无来源背书）

| 项 | 说明 |
|---|---|
| 用 JWL 代替 Taylor 的 `p ρ^-γ` 律 / Jones 绝热线 | 三篇来源都不支持 JWL 用于此模型；这是按真实卡片装填的工程选择 |
| 取 `D = D_card = 6930 > D_CJ = 6883.14`（过驱动） | Taylor **不覆盖**；SDToolbox 只在一维稳态平面层面覆盖 |
| **"球面 + 过驱动 + 恒定 D"三者同时** | **三篇来源都没有处理**。强/弱判据是平面 P–v 几何，把它外推到球面是本项目自创 |
| 恒 D 过驱动隐含外部驱动，但代码里没有对应的边界功项 | 与 Caramana 式(55) 冲突，是一个**已知的账目缺口** |
| 1D 球面 Caramana 恒等式的显式形式 | 原文未印；已由本项目自行推导并在 `DISCRETE_BUDGET.md` §3.2 验证 |
| 吸积前端的离散质量/动量/能量处理 | Caramana 明确建立在质量恒定之上，**不覆盖**；本项目自行处理，**尚未闭合** |

### 未核实

| 项 | 原因 |
|---|---|
| Taylor p.247 关于 `U > u₁ + c₁` 的精确不等号方向 | 提取文本 OCR 噪声大，且与 Table 2 数值有张力。不影响边界条件是 CJ 这一结论 |
| 卡片七参数 `(A, B, R1, R2, ω, E0, D)` 是否互相自洽 | `D_CJ(D=6883.14) ≠ D_card(6930)`，差 0.68%，说明这组参数不是自洽的 CJ 解。**"两个 RH 根"这一现象本身可能就是参数不自洽的产物**，不是物理必然。建议核查卡片来源与拟合条件 |

---

## 5. 结论

1. **分支选择方向正确，保留 `c/(D-u) > 1`。** 这是唯一一个"我们靠猜测实现、
   但文献明确支持"的环节。
2. **"球面 + 过驱动 + 恒定 D"没有来源背书。** 它隐含一个持续驱动器，而代码里
   没有对应的边界功项。这一点**不是数值误差，是模型层面的假设**，必须写在使用
   说明里：产物场是**工程构造**，不是对 DYNA 结果的独立预测。
3. **Caramana 的框架不覆盖变质量前端。** 相容离散要闭合，必须自行推导——这正是
   `DISCRETE_BUDGET.md` §5 指出仍未完成的部分。
4. **没有核实卡片参数的自洽性之前，不要声称整体模型已验证。** 卡片的 `D_CJ` 与
   `D_card` 差 0.68%，"两个根"是这件事的直接后果。
