"""参考前置求解：拉格朗日球对称双材料解，给出参考初始场 ``q_ref(r, t)``。

提示词 §5 要求参考场来自一个**独立的双材料球对称前置求解**，而不是把某个
仿真帧改名当初始条件。本模块就是那个求解器。

方法的选取
----------

采用**一维球对称拉格朗日有限体积**格式，交错网格：

* 节点 ``r_j`` 带着速度 ``u_j``，单元 ``i`` 带着质量 ``m_i``、比容 ``v_i``、
  比内能 ``e_i``；单元的材料身份是**固定的**，因此接触面永远是单元边界，
  两种材料永远不会被数值扩散混进同一个单元；
* 球对称拉格朗日形式的动量方程是 ``du/dt = -r^2 \\partial p/\\partial m``
  （``m`` 为单位立体角的质量坐标），配上前向欧拉时间推进；
* 激波用 von Neumann–Richtmyer 人工粘性捕捉，系数取 ``q1=1.5``、``q2=0.06``，
  与算例卡片里的 ``*CONTROL_BULK_VISCOSITY`` 一致。

这个选择直接满足 §5 的几条硬要求：质量逐单元精确守恒；接触面位置精确等于
一个节点，不需要界面重构；两种材料各自使用自己的状态方程；守恒性与正值性
可以逐时刻检查。

被排除的做法
------------

* 不用 ``R_c(t)`` 的解析经验式代替真实径向分布；
* 不把接触面压成单个界面状态；
* 不做 RK 前沿推进——那是提示词对 **PINN 训练**的禁令，但这里也不需要：
  前沿位置是解的一部分，不是外部输入。

初始条件（方案 B）
------------------

本求解器**不再自带初值**。它要求调用方传入一个
``DetonationProfile``，也就是波前恰好到达装药表面那一刻（``T_b = R_0/D``）的产物场。
产物场由 ``physics/detonation.py`` 解中心起爆的有限爆速球对称爆轰得到；那里给出了
它的方程、波前关系、选支判据与两条积分不变量。

装药内取该产物场在**当前**半径处的插值，外部取静止环境空气：

.. math::

    0 \\le r < R_0 &: \\rho, u, e \\;\\text{来自产物场},\\quad p = p_{\\rm JWL}(\\rho, e) \\\\
    r \\ge R_0 &: \\rho = 1.225,\\; u = 0,\\; p = p_{\\rm amb}

**方案 A 的那套整球均匀初值（``u = 0``、``e = E_0/\\rho_0``，因而 ``p = 8.220 GPa``）
已经删除，而且没有回退分支。** 一个可以省略的初值参数迟早会被省略，而省略之后
算出来的场仍然定常、为正、量纲正确——只是属于另一个方案。

本阶段积分的是 **epsilon 段**：从 ``T_b`` 到交接时刻 ``T_h = T_b + epsilon``。
不能再从 ``T_b`` 起积分一个 ``T_h``，那会把 ``T_b`` 重复计入。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import material as mat
from .detonation import DetonationProfile
from .eos import (
    JWLParameters,
    ideal_gas_pressure_numpy,
    jwl_pressure_numpy,
    jwl_sound_speed_numpy,
)

PRODUCT = 0
AIR = 1

MATERIAL_NAMES = {PRODUCT: "product", AIR: "air"}


class PrecursorError(RuntimeError):
    """前置求解失败。"""


# --------------------------------------------------------------------------
# 网格
# --------------------------------------------------------------------------


@dataclass
class LagrangianMesh:
    """一个球对称拉格朗日网格。

    ``radii`` 是 ``n_cells + 1`` 个节点半径，``mass`` 是 ``n_cells`` 个单元的
    单位立体角质量。两者一旦建立就不再改变：拉格朗日网格的质量是它的定义，
    不是它的解。
    """

    radii: np.ndarray
    mass: np.ndarray
    material: np.ndarray
    charge_radius: float

    def __post_init__(self) -> None:
        if self.radii.size != self.mass.size + 1:
            raise PrecursorError("radii must have exactly one more entry than mass")
        if self.material.size != self.mass.size:
            raise PrecursorError("material must have one entry per cell")
        if np.any(np.diff(self.radii) <= 0.0):
            raise PrecursorError("node radii must be strictly increasing")
        if np.any(self.mass <= 0.0):
            raise PrecursorError("cell masses must be positive")

    @property
    def n_cells(self) -> int:
        return int(self.mass.size)

    @property
    def centres(self) -> np.ndarray:
        """单元中心半径，``(r_{i+1/2}^3 + r_{i-1/2}^3)/2`` 的立方根。"""
        left = self.radii[:-1]
        right = self.radii[1:]
        return (0.5 * (right**3 + left**3)) ** (1.0 / 3.0)

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.radii)

    def contact_cell_index(self) -> int:
        """最后一个产物单元的序号；其**右侧节点**就是接触面。"""
        product = np.flatnonzero(self.material == PRODUCT)
        if product.size == 0:
            raise PrecursorError("the mesh has no product cells")
        return int(product[-1])

    def describe(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "charge_radius": self.charge_radius,
            "outer_radius": float(self.radii[-1]),
            "min_width": float(self.widths.min()),
            "max_width": float(self.widths.max()),
            "product_cells": int(np.count_nonzero(self.material == PRODUCT)),
            "air_cells": int(np.count_nonzero(self.material == AIR)),
        }


def _node_masses(cell_mass: np.ndarray) -> np.ndarray:
    """节点质量 ``M_j = (m_{j-1} + m_j)/2``。"""
    node = np.zeros(cell_mass.size + 1, dtype=np.float64)
    node[0] = 0.5 * cell_mass[0]
    node[1:-1] = 0.5 * (cell_mass[:-1] + cell_mass[1:])
    node[-1] = 0.5 * cell_mass[-1]
    return node


def _node_weighted_kinetic_energy(mass: np.ndarray, velocity: np.ndarray) -> float:
    """``sum_j M_j u_j^2 / 2``，``M_j = (m_{j-1} + m_j)/2``。

    这是拉格朗日格式**实际守恒**的动能：动量方程作用在节点上，它的能量恒等式配的
    就是节点质量。用单元中心约定 ``sum_i m_i [(u_i+u_{i+1})/2]^2/2`` 去读同一个解，
    两者差 ``O(du^2)``——用错口径去核对一个精确不变量，会读出根本不存在的偏差。
    """
    node_mass = np.zeros(velocity.size, dtype=np.float64)
    node_mass[0] = 0.5 * mass[0]
    node_mass[1:-1] = 0.5 * (mass[:-1] + mass[1:])
    node_mass[-1] = 0.5 * mass[-1]
    return 0.5 * float(np.sum(node_mass * velocity**2))


def conservative_overlap_remap(
    source_edges: np.ndarray,
    source_density: np.ndarray,
    source_values: np.ndarray,
    target_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把 ``(rho, q)`` 从一套球壳网格**守恒地**搬到一个径向区间上。

    返回 ``(目标区间的质量, 质量加权的 q 平均值)``，单位立体角。

    为什么不能点值插值
    ------------------

    产物场的拉格朗日网格在波前附近**聚**（那里的物质被压缩，同样的质量占更小的
    体积），而前置网格在装药内是**均匀**的。在一个陡的地方从聚的网格点采样到
    均匀网格的单元中心，会漏掉峰值：实测在 2000→600 的搬移上丢掉约 2% 的能量，
    而那不是"守恒律被破坏"，是**重采样误差**——可是对下游来说它和破坏守恒没有
    区别。

    重叠体积按圆台公式 ``(b^3 - a^3)/3`` 算（``dV/dr ~ r^2``，对固定立体角精确）。
    搬移之后 ``Σ m_i = ∫rho r^2 dr`` 是**构造性**成立的，交接这一环于是不再是
    误差来源，剩下的只有源场自身的离散误差。
    """
    source_edges = np.asarray(source_edges, dtype=np.float64)
    target_edges = np.asarray(target_edges, dtype=np.float64)
    density = np.asarray(source_density, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    n_target = target_edges.size - 1
    mass = np.zeros(n_target)
    weighted = np.zeros(n_target)

    for i in range(n_target):
        a, b = target_edges[i], target_edges[i + 1]
        # 与目标区间相交的源单元
        lo = int(np.searchsorted(source_edges, a, side="right")) - 1
        hi = int(np.searchsorted(source_edges, b, side="left"))
        lo = max(lo, 0)
        hi = min(hi, density.size)
        for j in range(lo, hi):
            left = max(a, source_edges[j])
            right = min(b, source_edges[j + 1])
            if right <= left:
                continue
            volume = (right**3 - left**3) / 3.0
            contribution = density[j] * volume
            mass[i] += contribution
            weighted[i] += contribution * values[j]
    return mass, np.divide(weighted, mass, out=np.zeros_like(mass), where=mass > 0.0)


def build_lagrangian_mesh(
    *,
    charge_radius: float,
    outer_radius_factor: float = 40.0,
    cells_inside: int = 2000,
    growth: float = 1.005,
    product_density: float = mat.PRODUCT_REFERENCE_DENSITY,
    air_density: float = mat.AIR_DENSITY,
) -> LagrangianMesh:
    """按参考问题建网格：装药内均匀，外部按固定比 ``growth`` 逐渐放宽。

    内层用均匀细网格，是因为接触面、初始激波和整个早期产物分布都发生在
    ``R_0`` 附近；外层用几何放宽，是因为激波很快跑到几十个 ``R_0`` 之外，
    在那么远的地方用和内层一样细的网格只是浪费。

    ``growth`` 是**固定的**，不随半径变，这样 ``R_0 = 50`` 与 ``R_0 = 80``
    的网格在无量纲坐标下是同一套（差一个离散化常数），尺度映射检验才有意义。
    """
    if charge_radius <= 0.0:
        raise PrecursorError(f"charge radius must be positive, got {charge_radius}")
    if cells_inside < 8:
        raise PrecursorError(f"cells_inside must be at least 8, got {cells_inside}")
    if growth <= 1.0:
        raise PrecursorError(f"growth must exceed 1, got {growth}")

    inner_width = charge_radius / cells_inside
    outer_radius = outer_radius_factor * charge_radius

    edges = [0.0]
    # 装药内：均匀，共 cells_inside 个单元。
    for _ in range(cells_inside):
        edges.append(edges[-1] + inner_width)
    # 装药外：几何放宽。
    width = inner_width
    while edges[-1] < outer_radius:
        width *= growth
        edges.append(edges[-1] + width)
    radii = np.asarray(edges, dtype=np.float64)

    # 每个单元的几何体积（单位立体角）与质量。
    volume = (radii[1:] ** 3 - radii[:-1] ** 3) / 3.0
    centres = (0.5 * (radii[1:] ** 3 + radii[:-1] ** 3)) ** (1.0 / 3.0)
    material = np.where(centres < charge_radius, PRODUCT, AIR).astype(np.int64)

    density = np.where(material == PRODUCT, product_density, air_density)
    mass = density * volume
    return LagrangianMesh(
        radii=radii, mass=mass, material=material, charge_radius=charge_radius
    )


# --------------------------------------------------------------------------
# 状态
# --------------------------------------------------------------------------


@dataclass
class PrecursorState:
    """一个时刻的完整解。"""

    time: float
    radii: np.ndarray
    radii_initial: np.ndarray
    velocity: np.ndarray
    specific_volume: np.ndarray
    density: np.ndarray
    specific_energy: np.ndarray
    pressure: np.ndarray
    material: np.ndarray
    mass: np.ndarray
    #: 声速。§7.1 要求把"正声速平方"并进放行条件，而它必须来自**实际用的那条
    #: EOS 公式**，不能事后再猜一个。以前状态里没有这个字段，于是检查只能退化成
    #: "拿默认值比一比"——一个查不出东西的检查。
    sound_speed: np.ndarray
    contact_radius: float
    shock_radius: float
    #: 环境压力。外边界"未被扰动"是相对**它**说的，而它来自卡片（101320 Pa），
    #: 不是标准大气压。
    air_pressure: float = mat.AIR_PRESSURE
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def centres(self) -> np.ndarray:
        return (0.5 * (self.radii[1:] ** 3 + self.radii[:-1] ** 3)) ** (1.0 / 3.0)

    @property
    def node_velocity(self) -> np.ndarray:
        """节点速度，与 ``radii`` 同长。"""
        return self.velocity

    def cell_velocity(self) -> np.ndarray:
        return 0.5 * (self.velocity[:-1] + self.velocity[1:])

    def product_mask(self) -> np.ndarray:
        return self.material == PRODUCT

    def air_mask(self) -> np.ndarray:
        return self.material == AIR

    def total_mass(self) -> float:
        return float(self.mass.sum())

    def total_energy(self) -> float:
        """单位立体角的总能量：内能 + 动能。

        动能生活在节点上，把相邻两个单元质量的一半归给每个节点。
        """
        internal = float(np.sum(self.mass * self.specific_energy))
        node_mass = np.zeros_like(self.velocity)
        node_mass[:-1] += 0.5 * self.mass
        node_mass[1:] += 0.5 * self.mass
        kinetic = float(np.sum(0.5 * node_mass * self.velocity**2))
        return internal + kinetic

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "contact_radius": self.contact_radius,
            "shock_radius": self.shock_radius,
            "n_cells": int(self.mass.size),
            "total_mass": self.total_mass(),
            "total_energy": self.total_energy(),
            "min_density": float(self.density.min()),
            "min_pressure": float(self.pressure.min()),
            "max_pressure": float(self.pressure.max()),
            "max_speed": float(np.abs(self.velocity).max()),
            **self.diagnostics,
        }


# --------------------------------------------------------------------------
# 求解器
# --------------------------------------------------------------------------


@dataclass
class PrecursorConfig:
    """求解参数。默认值与算例卡片一致的地方在注释里说明。"""

    cells_inside: int = 2000
    outer_radius_factor: float = 40.0
    growth: float = 1.005
    #: CFL 数。显式格式，取值明显小于 1。
    cfl: float = 0.2
    #: von Neumann–Richtmyer 人工粘性系数，与 *CONTROL_BULK_VISCOSITY 一致。
    q1: float = 1.5
    q2: float = 0.06
    #: 最大步数保护，防止参数错误时无限循环。
    max_steps: int = 4_000_000
    #: 判为零压缩的速度差阈值，m/s。
    compression_floor: float = 1.0e-8

    def validate(self) -> None:
        if self.cells_inside < 8:
            raise PrecursorError(f"cells_inside must be at least 8, got {self.cells_inside}")
        if self.growth <= 1.0:
            raise PrecursorError(f"growth must exceed 1, got {self.growth}")
        if not (0.0 < self.cfl < 1.0):
            raise PrecursorError(f"cfl must lie in (0, 1), got {self.cfl}")
        if self.q1 < 0.0 or self.q2 < 0.0:
            raise PrecursorError("artificial viscosity coefficients must be non-negative")


class SphericalPrecursor:
    """拉格朗日球对称双材料前置求解器。"""

    def __init__(
        self,
        *,
        charge_radius: float,
        config: PrecursorConfig | None = None,
        jwl: JWLParameters | None = None,
        gamma_air: float = mat.AIR_GAMMA,
        ambient_density: float = mat.AIR_DENSITY,
        ambient_pressure: float = mat.AIR_PRESSURE,
        products: "DetonationProfile | None" = None,
    ) -> None:
        self.config = config or PrecursorConfig()
        self.config.validate()
        self.charge_radius = float(charge_radius)
        self.jwl = jwl or mat.product_jwl()
        self.gamma_air = float(gamma_air)
        self.ambient_density = float(ambient_density)
        self.ambient_pressure = float(ambient_pressure)

        self.mesh = build_lagrangian_mesh(
            charge_radius=self.charge_radius,
            outer_radius_factor=self.config.outer_radius_factor,
            cells_inside=self.config.cells_inside,
            growth=self.config.growth,
            air_density=self.ambient_density,
        )
        self.radii_initial = self.mesh.radii.copy()
        self._contact_cell = self.mesh.contact_cell_index()

        # 初始条件：**必须来自方案 B 的产物场构造**（``physics/detonation.py``）。
        #
        # 方案 A 的那套整球均匀初值（u = 0、全装药 e = E0/rho0、因此 p = 8.220 GPa）
        # 已经删除，而且**不留回退分支**：一个可以省略的初值参数，迟早会被省略，
        # 而省略之后算出来的场看起来仍然"像那么回事"——定常的、正的、量纲正确的，
        # 只是物理上属于另一个方案。所以这里没有默认值，只有显式传入。
        if products is not None:
            self._initialise_from_products(products)
        else:
            raise PrecursorError(
                "SphericalPrecursor 需要方案 B 的产物场作为初值（products=DetonationProfile）；"
                "方案 A 的整球均匀初值已删除，没有回退路径。"
                "先用 physics.detonation.detonation_product_profile() 构造波前到达装药表面"
                "那一刻的产物场，再把它交进来。"
            )
        self.steps = 0
        self.artificial_viscosity = np.zeros(self.mesh.n_cells, dtype=np.float64)

    # -- 方案 B 初值 -------------------------------------------------------
    def _initialise_from_products(self, products) -> None:
        """把爆轰产物场铺到装药内的拉格朗日单元上，外部装静止空气。

        产物场的拉格朗日网格（在波前附近**聚**）与本求解器的网格（装药内**均匀**）
        不是同一套，所以这里是**守恒的重叠搬移**（圆台体积重叠），不是点值插值：
        点值采样在陡处会漏掉峰值，实测 2000→600 丢掉约 2% 的能量。

        搬移之后还要做一件事：**把离散功能量的口径统一到节点加权**。搬移守恒的是
        场量积分，而格式演化的是节点加权功能量，两者差 O(du²)——不统一的话，"交接
        守恒"只是口径造出来的读数。见下面比内能那一段。
        """
        n = self.mesh.n_cells
        self.velocity = np.zeros(n + 1, dtype=np.float64)
        self.radii = self.mesh.radii.copy()
        self.specific_volume = (self.radii[1:] ** 3 - self.radii[:-1] ** 3) / 3.0 / self.mesh.mass
        self.density = 1.0 / self.specific_volume

        product = self.mesh.material == PRODUCT
        if not np.any(product):
            raise PrecursorError("网格里没有产物单元，无法铺方案 B 的初值")

        # **守恒的重叠搬移**，不是点值插值。
        #
        # 传出去的不只是 rho，还有每单位质量的 ``e + u^2/2``：能量要按质量加权，
        # 而质量按体积（圆台）加权。两者用**同一套**重叠体积，交接因此是守恒的。
        product_index = np.flatnonzero(product)
        self.air_specific_energy = self.ambient_pressure / (
            (self.gamma_air - 1.0) * self.ambient_density
        )
        self.specific_energy = np.full(n, self.air_specific_energy, dtype=np.float64)
        edges = self.mesh.radii[: product_index[-1] + 2]
        source_edges = np.asarray(products.edges, dtype=np.float64)
        density_source = np.asarray(products.density, dtype=np.float64)
        energy_source = np.asarray(products.specific_energy, dtype=np.float64)

        # 搬移的是 **(rho, u, e) 三个场量**，不是 ``e + u²/2`` 一个总量。
        #
        # 曾经搬的是总比能量，然后为了配平总能量把整场内能乘一个全局比例。
        # 那会**改变局部热力学状态**：源网格与目标网格相同时，输入均匀的
        # e = 3.000 MJ/kg，交接后变成 2.138~3.556 MJ/kg——总量确实是 0 变化，
        # 但剖面被搅了。均匀输入必须得到均匀输出，这是交接的底线。
        mass, _ = conservative_overlap_remap(
            source_edges, density_source, np.zeros_like(energy_source), edges
        )
        _, internal = conservative_overlap_remap(
            source_edges, density_source, energy_source, edges
        )
        # 速度**不在这里搬**：它是**节点量**，见下面按节点插值那一段。

        # 密度换了，**单元质量与比容必须一起换**。三者是同一个状态的三种写法：
        #   v = 体积/质量,  rho = 1/v,  m = rho*体积
        # 只改 `density` 而把 `mesh.mass` 与 `specific_volume` 留在原处，会让它们
        # 互相矛盾——而 `_advance` 是用 `mesh.mass` 组节点质量、用几何除以质量算
        # 比容的，于是**第一步就把 B 的密度覆盖回 1630**（原始装药/空气密度）。
        # 这不是精度问题：交接后的场根本活不过一步。
        volume = (self.radii[1:] ** 3 - self.radii[:-1] ** 3) / 3.0
        new_mass = np.empty(product_index.size)
        for slot, cell in enumerate(product_index):
            if mass[cell] <= 0.0:
                raise PrecursorError(
                    f"第 {cell} 个产物单元在守恒搬移后质量为零；产物场的覆盖与前置"
                    "网格的装药区间不一致"
                )
            self.mesh.mass[cell] = mass[cell]
            self.density[cell] = mass[cell] / volume[cell]
            self.specific_volume[cell] = volume[cell] / mass[cell]
            new_mass[slot] = mass[cell]
        # 速度是**节点量**，必须按节点搬移。
        #
        # 产物场给的是节点速度（``node_velocity``，定义在 ``edges`` 上），前置网格的
        # 节点在 ``mesh.radii``。所以这里是在**节点表**之间插值。
        #
        # 旧做法把"单元平均速度"赋给节点，那是把速度场**换了一个量**，不是搬移：
        # 网格完全相同时，线性速度剖面的节点值与相邻平均相差半个节点间距，实测
        # 100/200/400 单元的节点速度变化是 10/5/2.5 m/s，总能量差
        # +0.3442%/+0.1753%/+0.0885%——**随网格间距线性下降**，是一阶误差。
        # 那既不是口径差（口径差是 O(du²)），也不是重采样残差，是一个纯粹的错。
        product_nodes = np.arange(product_index.size + 1)
        self.velocity[product_nodes] = np.interp(
            self.mesh.radii[: product_index.size + 1],
            np.asarray(products.edges, dtype=np.float64),
            np.asarray(products.node_velocity, dtype=np.float64),
        )
        # 球心的正则性条件是**精确**的：``u(0) = 0``。产物场在中心处的速度是数值
        # 解出来的、符号不定的小量（实测 ~1e-4 m/s）。``_advance`` 里球心节点的
        # 加速度恒为零，所以那个小速度会一直保留、节点匀速穿过轴心报"网格翻转"。
        # 方案 A 的初值是 ``u = 0``，所以这个坑只在 B 下出现，而且失败是**非单调**的
        # （400/500/2000 通过、600/1000 失败），看起来像分辨率问题。
        self.velocity[0] = 0.0

        # 内能**照搬**，不做任何全局重分配。
        self.specific_energy[product_index] = internal

        # **不做任何配平。** 搬移的是 (rho, u, e) 三个场量，交接的职责就是把它们
        # 忠实地表示到新网格上——均匀输入必须得到均匀输出。
        #
        # 走过的两条弯路都记在这里，因为它们各自"看起来"都对：
        #   1) 把内能乘一个全局比例去补总能差 -> 总量配平到舍入，但**搅乱了局部
        #      热力学状态**（源/目标网格相同时，均匀的 e = 3 MJ/kg 变成 2.14~3.56）。
        #   2) 把差额吸收到速度的整体缩放上 -> 局部状态保住了，但**功能量反而不守恒**
        #      （实测 −4.2e-3）。
        #
        # 真正的道理是：**功能量本身是依赖离散的**。源求解器守恒的是它自己的
        # ``sum m_j e_j + sum M_j u_j^2/2``；只要网格与节点速度逐点对应，搬移之后
        # 目标网格上的同一个式子自动相等，**根本不需要配平**。网格不同时剩下的差额
        # 是 O(du^2) 的**重采样**效应，它是被测量、被报告的量，不是被抹掉的量。
        #
        # 所以这里不配平，只在 ``staged_budget`` 里如实报出交接前后的功能量之差。
        self.pressure = self._pressure(self.density, self.specific_energy)
        self.sound_speed = self._sound_speed(self.density, self.specific_energy)
        self.time = float(products.time)
        self.products_source = products.as_dict()

    # -- 状态方程 ---------------------------------------------------------
    def _pressure(self, density, specific_energy):
        pressure = np.empty_like(density)
        product = self.mesh.material == PRODUCT
        if np.any(product):
            pressure[product] = jwl_pressure_numpy(
                density[product], specific_energy[product], self.jwl
            )
        if np.any(~product):
            pressure[~product] = ideal_gas_pressure_numpy(
                density[~product], specific_energy[~product], self.gamma_air
            )
        return pressure

    def _sound_speed(self, density, specific_energy):
        speed = np.empty_like(density)
        product = self.mesh.material == PRODUCT
        if np.any(product):
            speed[product] = jwl_sound_speed_numpy(
                density[product], specific_energy[product], self.jwl
            )
        if np.any(~product):
            p = ideal_gas_pressure_numpy(density[~product], specific_energy[~product], self.gamma_air)
            speed[~product] = np.sqrt(
                np.maximum(self.gamma_air * p / np.maximum(density[~product], 1e-30), 0.0)
            )
        return speed

    # -- 人工粘性 ---------------------------------------------------------
    def _artificial_viscosity(self, velocity, density, sound_speed):
        """von Neumann–Richtmyer：只在压缩（``du < 0``）时作用。"""
        du = np.diff(velocity)  # 单元 i 的两端节点速度差
        compression = np.where(du < 0.0, du, 0.0)
        q = (
            self.config.q1 * density * sound_speed * np.abs(compression)
            + self.config.q2 * density * compression**2
        )
        return np.where(compression < -self.config.compression_floor, q, 0.0)

    # -- 时间步 -----------------------------------------------------------
    def _timestep(self) -> float:
        width = self.radii[1:] - self.radii[:-1]
        speed = self.sound_speed + np.abs(0.5 * (self.velocity[:-1] + self.velocity[1:]))
        dt = self.config.cfl * float(np.min(width / np.maximum(speed, 1e-30)))
        if not math.isfinite(dt) or dt <= 0.0:
            raise PrecursorError(f"non-positive timestep {dt!r}")
        return dt

    # -- 一步推进 ---------------------------------------------------------
    def _advance(self, dt: float) -> None:
        n = self.mesh.n_cells

        # 1) 节点移动
        self.radii = self.radii + dt * self.velocity
        # 轴心节点（``r = 0``）在球对称下恒为 0，所以判据是"半径非负"而不是
        # "半径为正"：把 0 也当成翻转会把每一个合法解都拒掉。
        if self.radii[0] < 0.0 or np.any(self.radii < 0.0):
            raise PrecursorError(
                "a node crossed the axis; the mesh has inverted. Reduce the timestep."
            )
        if np.any(np.diff(self.radii) <= 0.0):
            raise PrecursorError(
                "the mesh has tangled (non-monotone node radii). Reduce the timestep."
            )

        # 2) 比容
        new_volume = (self.radii[1:] ** 3 - self.radii[:-1] ** 3) / 3.0
        new_specific_volume = new_volume / self.mesh.mass

        # 3) 压力与声速（用旧内能，能量随后更新）
        density = 1.0 / new_specific_volume
        pressure = self._pressure(density, self.specific_energy)
        sound_speed = self._sound_speed(density, self.specific_energy)

        # 4) 人工粘性，需要新几何下的速度差
        q = self._artificial_viscosity(self.velocity, density, sound_speed)

        # 5) 能量：de/dt = -(p + q) dv/dt
        self.specific_energy = self.specific_energy - (pressure + q) * (
            new_specific_volume - self.specific_volume
        )

        # 6) 节点加速度：du/dt = -r^2 dp/dm，节点质量取两侧单元质量的一半
        node_mass = np.zeros(n + 1)
        node_mass[:-1] += 0.5 * self.mesh.mass
        node_mass[1:] += 0.5 * self.mesh.mass
        node_mass[0] = 0.5 * self.mesh.mass[0]
        node_mass[-1] = 0.5 * self.mesh.mass[-1]

        total_pressure = pressure + q
        pressure_jump = np.zeros(n + 1)
        pressure_jump[1:-1] = total_pressure[1:] - total_pressure[:-1]
        pressure_jump[0] = total_pressure[0] - self.ambient_pressure
        pressure_jump[-1] = self.ambient_pressure - total_pressure[-1]

        acceleration = -(self.radii**2) * pressure_jump / node_mass
        acceleration[0] = 0.0  # 球心对称条件

        # 7) 速度
        self.velocity = self.velocity + dt * acceleration

        # 8) 接受新状态
        self.specific_volume = new_specific_volume
        self.density = density
        self.pressure = self._pressure(density, self.specific_energy)
        self.sound_speed = self._sound_speed(density, self.specific_energy)
        self.artificial_viscosity = q
        self.time += dt
        self.steps += 1

        if not np.all(np.isfinite(self.density)) or not np.all(np.isfinite(self.pressure)):
            raise PrecursorError("the solution went non-finite")

    # -- 前沿检测 ---------------------------------------------------------
    def _contact_radius(self) -> float:
        """接触面半径：产物单元与空气单元之间的那个节点。"""
        return float(self.radii[self._contact_cell + 1])

    def _shock_radius(self, threshold_ratio: float = 1.0e-3) -> float:
        """主激波半径：压力仍明显高于环境的最外侧位置。

        人工粘性把激波展宽到几个单元，所以这里取的是"压力扰动的前沿"，
        这就是 §5.7 要求标记的激波展宽区的外边界。
        """
        excess = self.pressure - self.ambient_pressure
        reference = max(float(np.max(np.abs(excess))), 1.0)
        active = np.flatnonzero(excess > threshold_ratio * reference)
        if active.size == 0:
            return self._contact_radius()
        index = int(active[-1])
        return float(self.radii[index + 1])

    # -- 快照 -------------------------------------------------------------
    def state(self) -> PrecursorState:
        contact = self._contact_radius()
        shock = self._shock_radius()
        return PrecursorState(
            time=self.time,
            radii=self.radii.copy(),
            radii_initial=self.radii_initial.copy(),
            velocity=self.velocity.copy(),
            specific_volume=self.specific_volume.copy(),
            density=self.density.copy(),
            specific_energy=self.specific_energy.copy(),
            pressure=self.pressure.copy(),
            material=self.mesh.material.copy(),
            mass=self.mesh.mass.copy(),
            sound_speed=self.sound_speed.copy(),
            contact_radius=contact,
            shock_radius=shock,
            diagnostics={"steps": self.steps},
        )

    # -- 主循环 -----------------------------------------------------------
    def run_to(
        self,
        end_time: float,
        *,
        stop_when_shock_reaches: float | None = None,
        sample_times: Sequence[float] | None = None,
        on_sample: Callable[[PrecursorState], None] | None = None,
    ) -> PrecursorState:
        """推进到 ``end_time``，可选在给定时刻回调快照。"""
        if end_time <= 0.0:
            raise PrecursorError(f"end_time must be positive, got {end_time}")

        supplied = () if sample_times is None else sample_times
        targets = sorted(float(t) for t in supplied if 0.0 < float(t) <= end_time)
        pending = list(targets)

        while self.time < end_time:
            dt = self._timestep()
            if self.time + dt > end_time:
                dt = end_time - self.time
            self._advance(dt)

            if stop_when_shock_reaches is not None:
                shock = self._shock_radius()
                outer_radius = float(self.radii[-1])
                if shock > stop_when_shock_reaches * outer_radius:
                    raise PrecursorError(
                        f"the shock reached {shock:.6g} m, beyond "
                        f"{stop_when_shock_reaches:.3g} of the outer boundary "
                        f"{outer_radius:.6g} m; the outer boundary is no longer undisturbed"
                    )

            while pending and self.time >= pending[0]:
                if on_sample is not None:
                    on_sample(self.state())
                pending.pop(0)

            if self.steps > self.config.max_steps:
                raise PrecursorError(
                    f"exceeded max_steps = {self.config.max_steps} before reaching "
                    f"t = {end_time}; the timestep is probably collapsing"
                )
        return self.state()


# --------------------------------------------------------------------------
# 参考场：在 (r, t) 上的可查询表
# --------------------------------------------------------------------------


@dataclass
class ReferenceField:
    """参考场 ``q_ref(r, t)``：一套 (半径, 时间) 上的样本。

    只保存一次 ``R_0 = R_ref`` 的求解结果。其它半径通过

    .. math::  q_{R_0}(r, t_{\\rm phys}) = q_{\\rm ref}(r/k, t_{\\rm phys}/k)

    得到：只缩放长度与时间，状态量不乘 ``k``。

    空间坐标是**每个时刻各自的**单元中心
    --------------------------------------

    前置求解用的是**拉格朗日**网格：单元带着质量一起向外飞，同一时刻的单元中心
    和初始时刻完全不是一组半径。所以 ``cell_radii`` 的形状是 ``(n_t, n_cells)``
    ——每一帧一套当前坐标——而不是一个与时间无关的 ``radii`` 向量。

    这不是形式问题。把初始单元中心当成固定的空间坐标、再拿当前物理半径去上面
    插值，会让查询点落在"这个单元初始在哪"的位置，而读出来的却是"这个单元现在
    是什么状态"。在 50 mm、1 µs 的算例上，最后一个产物单元的初始中心是
    49.938 mm、当前中心是 53.117 mm，接触面在 53.791 mm：同一个查询半径会得到
    131.95 或 7.24 kg/m³ 两个完全不同的密度，而两者看起来都"像那么回事"。
    尺度映射精确也证明不了坐标是对的——它只说明缩放实现一致。

    材料身份也是固定的
    ------------------

    ``material`` 有 ``n_cells`` 个条目，每个单元的归属在整个求解过程中不变
    （拉格朗日网格的接触面永远落在同一个单元边界上）。因此查询时先按**该帧的**
    接触面判断查询点属于哪种材料，再**只在该材料的单元里**插值：接触面两侧各自
    取单侧状态，永远不会把两种材料插成一个既不属此也不属彼的平均值。

    时间插值同样如此：同一固定半径在不同时刻可能先后属于两种材料（接触面扫过
    它）。这时两个时间样本之间不是同一个场的两个采样，线性混合没有意义——查询
    退回到时间上更近的那一帧，并把这件事记在 ``crossing_fraction`` 诊断里。
    """

    times: np.ndarray
    cell_radii: np.ndarray      # (n_t, n_cells) 当前单元中心半径
    density: np.ndarray         # (n_t, n_cells)
    velocity: np.ndarray
    specific_energy: np.ndarray
    material: np.ndarray        # (n_cells,) 固定材料身份
    contact_radius: np.ndarray  # (n_t,)
    shock_radius: np.ndarray
    reference_radius: float
    metadata: dict[str, Any] = field(default_factory=dict)
    #: 最近一次 ``sample`` 的诊断：查询点中落在"两帧材料不同"的比例。这是
    #: 查询行为的记录，不是物理量，所以和 ``metadata`` 分开存。
    query_diagnostics: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        n_t = self.times.size
        n_cells = self.material.size
        for name in ("density", "velocity", "specific_energy"):
            array = getattr(self, name)
            if array.shape != (n_t, n_cells):
                raise PrecursorError(
                    f"{name} has shape {array.shape}, expected {(n_t, n_cells)}"
                )
        if self.cell_radii.shape != (n_t, n_cells):
            raise PrecursorError(
                f"cell_radii has shape {self.cell_radii.shape}, expected {(n_t, n_cells)}; "
                "every frame carries its own current cell centres"
            )
        if self.contact_radius.size != n_t or self.shock_radius.size != n_t:
            raise PrecursorError("front arrays must have one entry per time")
        if np.any(np.diff(self.material) < 0):
            raise PrecursorError(
                "material must be sorted (product cells first, then air); the material-"
                "aware query relies on each medium occupying one contiguous block"
            )
        if not np.any(self.material == PRODUCT):
            raise PrecursorError("the reference field has no product cells")
        if np.any(np.diff(self.cell_radii, axis=1) <= 0.0):
            raise PrecursorError("cell centres must be strictly increasing within every frame")

    # -- 材料分块 ---------------------------------------------------------
    @property
    def n_product_cells(self) -> int:
        """产物单元数。材料已排序，所以产物是 ``[0, n)``、空气是 ``[n, end)``。"""
        return int(np.count_nonzero(self.material == PRODUCT))

    @property
    def final_cell_radii(self) -> np.ndarray:
        """最后一帧的单元中心半径。"""
        return self.cell_radii[-1]

    # -- 压力始终由 EOS 重算 ---------------------------------------------
    def pressure(self, gamma_air: float = mat.AIR_GAMMA, jwl: JWLParameters | None = None):
        """由 ``(rho, e)`` 重算压力。

        §5.6 要求插值 ``rho/u/e`` 之后**重算**压力，而不是独立插值出一个
        可能与 EOS 不一致的压力。这张表因此从不存储压力。
        """
        params = jwl or mat.product_jwl()
        product = self.material == PRODUCT
        out = np.empty_like(self.density)
        if np.any(product):
            out[:, product] = jwl_pressure_numpy(
                self.density[:, product], self.specific_energy[:, product], params
            )
        if np.any(~product):
            out[:, ~product] = ideal_gas_pressure_numpy(
                self.density[:, ~product], self.specific_energy[:, ~product], gamma_air
            )
        return out

    def scale_to(self, charge_radius: float) -> "ReferenceField":
        """返回同一个物理场在另一个半径下的视图（``k = R_0/R_ref``）。

        注意：这不是"重采样到新网格"，而是记录缩放因子。查询时再按 ``k``
        读取，避免为同一个场存两份数据。
        """
        return _ScaledReferenceField(base=self, charge_radius=float(charge_radius))

    def sample(self, radius, physical_time):
        """在 ``(r, t_phys)`` 处求 ``q_ref``，返回 ``(rho, u, e)``。"""
        return self._sample(radius, physical_time, k=1.0)

    # -- 按帧、按材料插值 -------------------------------------------------
    def _interp_frame(self, frame: int, flat_r: np.ndarray, is_product: np.ndarray):
        """在第 ``frame`` 帧上插值：每个查询点只在自己的材料块里取值。

        ``is_product`` 由调用方按**该帧**的接触面判定，因此产物点永远不会读到
        空气单元。块边界处自然取到单侧状态，不跨材料平滑。
        """
        cut = self.n_product_cells
        out = {}
        windows = {True: (0, cut), False: (cut, self.material.size)}
        for name in ("density", "velocity", "specific_energy"):
            row = getattr(self, name)[frame]
            values = np.empty(flat_r.size, dtype=np.float64)
            for want_product, (lo, hi) in windows.items():
                selection = is_product == want_product
                if not np.any(selection):
                    continue
                values[selection] = _interp_segment(
                    self.cell_radii[frame][lo:hi], row[lo:hi], flat_r[selection]
                )
            out[name] = values
        return out

    def _interp_frames(self, frames: np.ndarray, flat_r: np.ndarray, is_product: np.ndarray):
        """``frames`` 是**逐查询点**的帧号，所以按帧分组后各自插值。

        查询点可以带各自不同的时间，因此"哪一帧"是一个与点同长的数组，而不是
        一个标量。按帧分组的做法让每个材料块内的插值仍然是一次向量化调用。
        """
        names = ("density", "velocity", "specific_energy")
        out = {name: np.empty(flat_r.size, dtype=np.float64) for name in names}
        for frame in np.unique(frames):
            selection = frames == frame
            block = self._interp_frame(int(frame), flat_r[selection], is_product[selection])
            for name in names:
                out[name][selection] = block[name]
        return out

    def _sample(self, radius, physical_time, *, k: float):
        radius = np.asarray(radius, dtype=np.float64)
        physical_time = np.asarray(physical_time, dtype=np.float64)
        shape = np.broadcast(radius, physical_time).shape
        radius = np.broadcast_to(radius, shape).astype(np.float64)
        physical_time = np.broadcast_to(physical_time, shape).astype(np.float64)

        # 查询换到参考半径的坐标系里；存下来的场始终是参考半径的场。
        scaled_radius = radius / k
        scaled_time = physical_time / k

        if np.any(scaled_time < 0.0):
            raise PrecursorError("reference field is not defined before t = 0")
        if np.any(scaled_time > self.times[-1]):
            raise PrecursorError(
                f"query time {float(scaled_time.max()):.6g} s exceeds the reference "
                f"field's coverage ({float(self.times[-1]):.6g} s)"
            )

        flat_r = scaled_radius.reshape(-1)
        flat_t = scaled_time.reshape(-1)

        if self.times.size == 1:
            lower = np.zeros(flat_t.size, dtype=np.int64)
            upper = lower
        else:
            lower = np.clip(
                np.searchsorted(self.times, flat_t, side="right") - 1,
                0,
                self.times.size - 2,
            )
            upper = np.minimum(lower + 1, self.times.size - 1)
        t0 = self.times[lower]
        t1 = self.times[upper]
        weight_t = np.where(t1 > t0, (flat_t - t0) / np.maximum(t1 - t0, 1e-300), 0.0)

        # 材料归属按**每一帧各自的**接触面判定，而不是用一个与时间无关的分界。
        product_lower = flat_r < self.contact_radius[lower]
        product_upper = flat_r < self.contact_radius[upper]
        same_medium = product_lower == product_upper

        # 接触面在两帧之间扫过查询点时，两帧的状态属于不同的场。这时不做线性
        # 混合，退回到时间上更近的那一帧——混合会造出一个两种材料都不对应的状态。
        weight_t = np.where(same_medium, weight_t, (weight_t >= 0.5).astype(np.float64))

        low = self._interp_frames(lower, flat_r, product_lower)
        if self.times.size == 1:
            high = low
        else:
            high = self._interp_frames(upper, flat_r, product_upper)

        out = {}
        for name in ("density", "velocity", "specific_energy"):
            left = low[name]
            right = high[name]
            out[name] = (left + weight_t * (right - left)).reshape(shape)

        self.query_diagnostics["last_query_crossing_material_fraction"] = float(
            np.mean(~same_medium)
        )
        self.query_diagnostics["last_query_count"] = int(flat_r.size)
        return out["density"], out["velocity"], out["specific_energy"]

    def contact_and_shock_at(self, physical_time, *, k: float = 1.0):
        scaled_time = np.asarray(physical_time, dtype=np.float64) / k
        contact = np.interp(scaled_time, self.times, self.contact_radius) * k
        shock = np.interp(scaled_time, self.times, self.shock_radius) * k
        return contact, shock

    def contact_pressure_at(
        self,
        physical_time,
        *,
        k: float = 1.0,
        gamma_air: float = mat.AIR_GAMMA,
        jwl: JWLParameters | None = None,
    ) -> float:
        """接触面上的压力，由**产物侧最后一个单元**的状态经 EOS 得到。

        拉格朗日网格的材料身份是固定的，所以接触面永远落在同一个单元边界上，
        产物侧最后一个单元的下标是一个常数。压力在接触面上连续，因此从哪一侧
        取都一样；这里取产物侧，因为它的 EOS 在高压区数值上更稳。

        插值到接触半径再取值是**错的**：``(rho, e)`` 在接触面上间断，线性插值
        会把两种材料混成一个既不属此也不属彼的状态。
        """
        product_cells = np.flatnonzero(self.material == PRODUCT)
        if product_cells.size == 0:
            raise PrecursorError("the reference field has no product cells")
        index = int(product_cells[-1])
        scaled_time = float(np.asarray(physical_time).reshape(-1)[0]) / k
        frame = int(np.argmin(np.abs(self.times - scaled_time)))
        density = np.array([self.density[frame, index]])
        energy = np.array([self.specific_energy[frame, index]])
        params = jwl or mat.product_jwl()
        return float(jwl_pressure_numpy(density, energy, params)[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_radius": self.reference_radius,
            "n_times": int(self.times.size),
            "n_cells": int(self.material.size),
            "n_product_cells": self.n_product_cells,
            "time_min": float(self.times[0]),
            "time_max": float(self.times[-1]),
            # 单元中心在动，所以这是一个跨帧的包络，不是某一帧的网格。
            "cell_radius_min": float(self.cell_radii.min()),
            "cell_radius_max": float(self.cell_radii.max()),
            "spatial_coordinates": (
                "per-frame current Lagrangian cell centres, shape (n_t, n_cells); "
                "material-aware interpolation, one-sided at the contact"
            ),
            **self.metadata,
        }


class _ScaledReferenceField:
    """按 ``k`` 缩放后的参考场视图。"""

    def __init__(self, *, base: ReferenceField, charge_radius: float) -> None:
        self.base = base
        self.charge_radius = float(charge_radius)
        self.k = self.charge_radius / base.reference_radius

    def sample(self, radius, physical_time):
        return self.base._sample(radius, physical_time, k=self.k)

    def contact_and_shock_at(self, physical_time):
        return self.base.contact_and_shock_at(physical_time, k=self.k)

    def contact_pressure_at(self, physical_time, **kwargs):
        return self.base.contact_pressure_at(physical_time, k=self.k, **kwargs)

    @property
    def reference_radius(self) -> float:
        return self.charge_radius

    @property
    def final_cell_radii(self) -> np.ndarray:
        """最后一帧的单元中心半径，按 ``k`` 缩放到目标半径。"""
        return self.base.final_cell_radii * self.k

    @property
    def base_reference_radius(self) -> float:
        """原始参考场的半径。``reference_radius`` 在这个视图里是目标半径。"""
        return self.base.reference_radius

    def as_dict(self) -> dict[str, Any]:
        return {**self.base.as_dict(), "scaled_to_charge_radius": self.charge_radius}

    def __getattr__(self, item):
        return getattr(self.base, item)


def _interp_segment(abscissa: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """在一个**单一材料块**内按半径线性插值。

    ``abscissa`` 与 ``values`` 是该材料在这帧里的单元中心与状态，形状 ``(n,)``；
    ``query`` 形状 ``(m,)``。返回 ``(m,)``。

    超出该材料块范围的查询钳到块的端点。钳位是有意的：落在产物块之外的产物
    查询在物理上不存在，而 ``nan`` 在损失里会被均值悄悄吞掉。真正跨材料的情况
    由调用方在**进入本函数之前**按接触面判定并换块处理，所以这里的钳位绝不会
    把两种材料混在一起。
    """
    abscissa = np.asarray(abscissa, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    if abscissa.size != values.size:
        raise PrecursorError(
            f"_interp_segment got {abscissa.size} abscissae for {values.size} values"
        )
    if abscissa.size == 0:
        return np.full(query.size, np.nan)
    if abscissa.size == 1:
        return np.full(query.size, float(values[0]))

    clipped = np.clip(query, abscissa[0], abscissa[-1])
    index = np.clip(np.searchsorted(abscissa, clipped, side="right") - 1, 0, abscissa.size - 2)
    left = abscissa[index]
    right = abscissa[index + 1]
    weight = np.where(right > left, (clipped - left) / np.maximum(right - left, 1e-300), 0.0)
    return values[index] * (1.0 - weight) + values[index + 1] * weight


# --------------------------------------------------------------------------
# 顶层入口
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# §5 验收：状态由**实测数字**决定，不是由写死的字符串决定
# --------------------------------------------------------------------------

#: 验收容差。每一条都要说得出来源，否则它就只是一个让运行通过的数字。
#:
#: 分两类：
#:
#: * **守恒/精度类** —— 有一个物理上应当为零（或很小）的量，容差是数值精度声明；
#: * **违例计数类** —— 容差恒为 0。它们数的是"物理上非法的状态出现了几个单元"。
#:   这类量**不能只写在报告里**：以前的版本把正值、外边界、网格次序打印出来，
#:   但最终放行条件不看它们，于是"报告里全是红的、状态是 ACCEPTED"。§7.1 明确
#:   要求把它们并进放行条件。
ACCEPTANCE_TOLERANCES: dict[str, float] = {
    # §5.4「质量」：产物构造的质量不变量是闭式精确的（波前扫过的装药全在波后），
    # 所以只允许浮点舍入。
    "detonation_mass_invariant_relative": 1.0e-12,
    # §5.4「能量」：同一条推导给出 E = E0 r_f^3/3，解析上精确，所以容差是一个
    # **数值精度声明**，不是物理不确定性。取值依据：前端单元是一层一单元厚的
    # 边界层，格式一阶；实测 2000 单元 2.94e-3、1000 单元 3.35e-3、500 单元
    # 4.06e-3，且随 CFL 减小而变差。**为什么随 CFL 变差尚未定论**（见
    # DISCRETE_BUDGET.md §4.4）——不写成"每步一个不随 dt 消失的量已被证明"，
    # 那是一条已经撤回的错误归因。
    # 1e-2 是"默认分辨率下必须达到"的门槛，实测有 3 倍余量。
    "detonation_energy_invariant_relative": 1.0e-2,
    # 交接是**守恒的重叠搬移**（不是点值插值）：同网格下精确、跨网格有 O(du²) 的
    # 重采样残差。容差按"远小于上游偏差"定。
    "handover_mass_defect_relative": 1.0e-3,
    "handover_energy_defect_relative": 1.0e-2,
    # 球心正则性 u(0)=0 是**精确**约束，不是近似。
    "centre_velocity_absolute": 0.0,
    # §5.4「前置阶段」：整域的**节点加权**总能量在 epsilon 段里的漂移。外边界取在
    # 40 R_0，扰动只到约 1.15 R_0，外边界通量为零，所以这一项应当只是舍入。
    # 容差按"比上游偏差小一个量级"定：产物构造 ~3e-3，这里留 1e-3。
    "precursor_energy_drift_relative": 1.0e-3,
    # 拉格朗日格式的这一项恒等于零，是**构造性**的；记它是因为"零"本身要能被
    # 看见，否则没人知道它被查过。
    "precursor_mass_drift_relative": 1.0e-12,
    # --- 违例计数类（§7.1）：容差恒为 0 --------------------------------
    #
    # 这些都是"物理上非法状态的单元个数"。它们必须是**放行条件**的一部分，不能
    # 只出现在报告里。§2.1 同时禁止用裁剪伪造有效（"不得将负超压裁成零"），
    # 所以这里数的是**真实的非正压力**，不是被 clip 过的。
    "pressure_nonpositive_cells": 0.0,
    "density_nonpositive_cells": 0.0,
    "specific_energy_nonpositive_cells": 0.0,
    "sound_speed_squared_nonpositive_cells": 0.0,
    "nonfinite_field_cells": 0.0,
    # 网格次序：节点半径必须严格递增，单元体积必须为正。拉格朗日网格翻转在这里
    # 被抓住，而不是等到某个单元体积变成负的把后面全污染掉。**NaN 也算违例**——
    # 它与 ``<= 0`` 的比较结果是 False，不单独数就会被放过。
    "mesh_ordering_violations": 0.0,
    # 前沿次序：产物单元必须是从球心开始的**一串连续单元**。中间夹一个空气单元
    # 说明材料块被打乱了。
    "material_block_violations": 0.0,
    # 前沿次序：Rs 必须 >= Rc（接触面在内、激波在外）。Rs < Rc 是物理非法的
    # 前沿配置，会被"其他指标都合格"的假阳性放过。
    "front_ordering_violations": 0.0,
    # 材料编码必须合法（只能是 PRODUCT / AIR）。一个非法编码会被"块连续"检查
    # 漏掉（它既不是产物也不是空气，但可能不打断产物的连续性）。
    "illegal_material_cells": 0.0,
    # 外边界必须**未被扰动**（扰动只到约 1.15 R_0，边界在 40 R_0）。用相对环境态
    # 的偏差度量；它同时是"外边界通量为零"这个假设的实测依据。
    # 依据：实测 1e-15 量级，1e-10 是宽松但仍有意义的门槛。
    "outer_boundary_relative_disturbance": 1.0e-10,
    # 球心正则性：不光是 u(0)=0（那是精确硬约束），还有第一个单元不能有反常的
    # 密度/压力尖峰。用它与相邻单元的相对差度量。
    "centre_cell_relative_jump": 0.5,
}

ACCEPTED_STATUS = "ACCEPTED"
NOT_ACCEPTED_STATUS = "NOT ACCEPTED"

#: 每条容差对应的中文名，报告里直接读。
_METRIC_LABELS = {
    "detonation_mass_invariant_relative": "产物构造·质量不变量相对偏差",
    "detonation_energy_invariant_relative": "产物构造·能量不变量相对偏差（节点加权）",
    "handover_mass_defect_relative": "交接·质量相对缺陷",
    "handover_energy_defect_relative": "交接·能量相对缺陷",
    "centre_velocity_absolute": "初值·球心速度 |u(0)|",
    "precursor_energy_drift_relative": "前置演化·整域总能量漂移（节点加权）",
    "precursor_mass_drift_relative": "前置演化·整域质量漂移",
    "pressure_nonpositive_cells": "状态·非正压力单元数",
    "density_nonpositive_cells": "状态·非正密度单元数",
    "specific_energy_nonpositive_cells": "状态·非正比内能单元数",
    "sound_speed_squared_nonpositive_cells": "状态·非正声速平方单元数",
    "nonfinite_field_cells": "状态·非有限值单元数",
    "mesh_ordering_violations": "网格·次序/体积违例数",
    "material_block_violations": "网格·材料块不连续处数",
    "front_ordering_violations": "前沿·Rs<Rc 违例数",
    "illegal_material_cells": "状态·非法材料编码单元数",
    "outer_boundary_relative_disturbance": "外边界·相对环境态扰动",
    "centre_cell_relative_jump": "球心单元·与相邻单元相对跳变",
}


#: 仅凭**一次求解**无法给出收敛证据的有哪些 §5 验收项。它们必须由
#: ``cli.reference_field`` 的加密/时间步/尺度核查提供，缺了就只能记"未验收"。
EVIDENCE_DEPENDENT_CHECKS = (
    "grid_convergence",
    "timestep_sensitivity",
    "scale_mapping",
)


def field_violation_metrics(state: Any) -> dict[str, float]:
    """数出**物理上非法状态**出现了多少单元（§7.1）。

    这些都是"报告里全红、状态却是 ACCEPTED"的来源：以前它们只被打印出来，
    没有并进放行条件。现在它们和守恒指标一起进 ``ACCEPTANCE_TOLERANCES``，
    容差恒为 0。

    **不做任何裁剪**。§2.1 明令不得把负超压裁成零；这里数的是真实的非正值，
    不是被 clip 之后的。发现非正值是把它们报出来，不是把它们修掉。
    """
    density = np.asarray(state.density, dtype=np.float64)
    energy = np.asarray(state.specific_energy, dtype=np.float64)
    pressure = np.asarray(state.pressure, dtype=np.float64)
    sound = np.asarray(getattr(state, "sound_speed", np.zeros_like(density)), dtype=np.float64)
    mass = np.asarray(state.mass, dtype=np.float64)
    radii = np.asarray(state.radii, dtype=np.float64)
    # 速度是**节点**量（n+1 个），密度/内能/压力/声速是**单元**量（n 个）。
    # 不能把它们塞进同一个 ``&`` 掩码——形状不匹配。这里把两类分开处理：
    # 单元量按单元查，节点速度单独查（只要有一个节点非有限就算违例）。
    velocity = np.asarray(state.velocity, dtype=np.float64)
    material = np.asarray(state.material)

    # 单元中心量的有限性：单元口径。
    finite_cells = (
        np.isfinite(density)
        & np.isfinite(energy)
        & np.isfinite(pressure)
        & np.isfinite(sound)
    )
    velocity_finite = bool(np.all(np.isfinite(velocity)))

    # 网格次序与体积：半径严格递增，单元质量为正。
    # **NaN 与 ``<= 0`` 的比较结果是 False，所以必须单独数**——否则半径或质量里
    # 混进一个 NaN，``np.diff(radii) <= 0`` 与 ``mass <= 0`` 都数不到它，违例被
    # 放过。
    order = int(np.count_nonzero(np.diff(radii) <= 0.0)) + int(
        np.count_nonzero(mass <= 0.0)
    )
    order += int(np.count_nonzero(~np.isfinite(radii))) + int(
        np.count_nonzero(~np.isfinite(mass))
    )

    # 材料块：产物必须是从球心开始的一串连续单元。
    product_index = np.flatnonzero(material == PRODUCT)
    if product_index.size == 0:
        block = 1  # 一个产物单元都没有：这本身就是违例
    else:
        expected = np.arange(product_index.size)
        block = int(np.count_nonzero(product_index != expected))

    # 非法材料编码：只能是 PRODUCT / AIR。
    illegal_material = int(
        np.count_nonzero(~np.isin(material, (PRODUCT, AIR)))
    )

    # 前沿次序：Rs 必须 >= Rc。state 的接触/激波半径可以是标量（单帧）或数组。
    contact = np.asarray(getattr(state, "contact_radius", 0.0), dtype=np.float64)
    shock = np.asarray(getattr(state, "shock_radius", 0.0), dtype=np.float64)
    front_order = int(np.count_nonzero(~np.isfinite(shock) | ~np.isfinite(contact)))
    front_order += int(np.count_nonzero(np.isfinite(shock) & np.isfinite(contact) & (shock < contact)))

    # 外边界未被扰动：与环境态比较（空气那一侧的最外一个单元）。
    outer_cell = int(mass.size - 1)
    ambient_pressure = float(getattr(state, "air_pressure", mat.AIR_PRESSURE))
    pressure_scale = max(abs(ambient_pressure), 1.0)
    outer_disturbance = max(
        abs(float(pressure[outer_cell]) - ambient_pressure) / pressure_scale,
        # 最外**节点**的速度（外边界相对球心静止，扰动应接近 0）。
        abs(float(velocity[-1]))
        / max(float(np.sqrt(ambient_pressure / mat.AIR_DENSITY)), 1.0),
    )

    # 球心单元与相邻单元的相对跳变（球心正则性的一个可测表现）。
    if mass.size >= 2 and density[1] > 0.0 and pressure[1] > 0.0:
        centre_jump = max(
            abs(float(density[0]) - float(density[1])) / float(density[1]),
            abs(float(pressure[0]) - float(pressure[1])) / float(pressure[1]),
        )
    else:
        centre_jump = float("nan")

    return {
        "pressure_nonpositive_cells": float(np.count_nonzero(finite_cells & (pressure <= 0.0))),
        "density_nonpositive_cells": float(np.count_nonzero(finite_cells & (density <= 0.0))),
        "specific_energy_nonpositive_cells": float(
            np.count_nonzero(finite_cells & (energy <= 0.0))
        ),
        # 声速平方：``c`` 非有限或非正都算。EOS 给出 nan 时**不**先裁剪再宣称有效。
        "sound_speed_squared_nonpositive_cells": float(
            np.count_nonzero(~np.isfinite(sound) | (sound <= 0.0))
        ),
        # 非有限：单元量里非有限的个数，加上节点速度是否非有限（一个节点非有限
        # 记为 1，因为一整根速度表都不可信了）。
        "nonfinite_field_cells": float(np.count_nonzero(~finite_cells)) + float(
            0 if velocity_finite else 1
        ),
        "mesh_ordering_violations": float(order),
        "material_block_violations": float(block),
        "front_ordering_violations": float(front_order),
        "illegal_material_cells": float(illegal_material),
        "outer_boundary_relative_disturbance": float(outer_disturbance),
        "centre_cell_relative_jump": float(centre_jump),
    }


#: 物理适用性状态。**数值代码不得自动把它填成"已核实"**（§7.1）。
PHYSICS_BASIS_UNVERIFIED = "未核实"
PHYSICS_BASIS_VERIFIED = "已核实"


#: 场的**内容标识**必须覆盖的数组。缺任何一个都不成标识——见
#: :func:`field_content_hash`。
FIELD_CONTENT_ARRAYS = (
    "times",
    "cell_radii",
    "density",
    "velocity",
    "specific_energy",
    "material",
    "contact_radius",
    "shock_radius",
)


def field_content_hash(reference: Any) -> str:
    """场内容的**内容标识**（Trusty URIs 意义上的内容寻址）。

    一个数字对象由它的内容标识。内容变了，它就是**另一个对象**，不能继续继承
    原对象的身份与证据。这里把这个原则落到具体的字节上：

    * 覆盖 :data:`FIELD_CONTENT_ARRAYS` 里**每一个**数组，缺一个就报错——不是
      跳过。以前用 ``getattr(...) is None: continue``，删掉一个字段哈希照样不变,
      于是"少了 ``specific_energy`` 的场"和完整的场被判成同一个对象。
    * 每个数组连同**字段名、形状、dtype**一起进摘要。只哈希字节的话，
      ``float64`` 改成 ``float32``、或者形状从 ``(2, 3)`` 变成 ``(3, 2)`` 而字节
      数相同，都会得出同一个标识——而它们是不同的数据。

    **这个标识只说明"是不是同一份数据"，不说明数据在科学上是否正确。**
    后者要靠验收指标与物理依据，那是另外两条独立的检查。
    """
    digest = hashlib.sha256()
    for name in FIELD_CONTENT_ARRAYS:
        array = getattr(reference, name, None)
        if array is None:
            raise PrecursorError(
                f"场缺少必需数组 {name!r}，无法形成内容标识；"
                "不能因为少了一个字段就把它当成同一份场"
            )
        arr = np.ascontiguousarray(np.asarray(array))
        digest.update(name.encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(arr.dtype.str.encode("utf-8"))
        digest.update(arr.tobytes())
    digest.update(
        np.float64(float(getattr(reference, "reference_radius", 0.0))).tobytes()
    )
    return digest.hexdigest()


def assess_acceptance(
    metrics: Mapping[str, float],
    *,
    convergence_evidence: Mapping[str, Any] | None = None,
    physics_basis: str = PHYSICS_BASIS_UNVERIFIED,
    data_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """由实测指标判定验收状态。**没有一项是通过还是失败的默认值。**

    旧版本把 ``acceptance_status`` 与一段写死的说明放进元数据。两件事一起出错：
    说明里的数字会与代码漂移（一个 7.86e-2 被写进文本之后再没人测过），而且
    失败的**理由**可能是一句推错的结论——文字不会因为结论错了就停止流传。
    现在状态由这张表算出来，理由由**哪几条超限**生成。

    **三类状态分开记**（§7.1 末条）：数值、物理适用性、数据就绪。它们各自是
    独立条件，正式使用必须全部满足。特别地：**物理依据未核实不能因为三个数值
    布尔值为真就变成可正式使用**——所以 ``physics_basis`` 默认是"未核实"，
    而且它**只接受显式传入**，数值代码不会自己填"通过"。
    """
    failures = []
    for key, tolerance in ACCEPTANCE_TOLERANCES.items():
        value = float(metrics.get(key, float("nan")))
        if not math.isfinite(value) or value > tolerance:
            failures.append(
                f"{_METRIC_LABELS.get(key, key)} = {value:.4e} > 容差 {tolerance:.1e}"
            )
    # 单次求解**永远不足以**给出验收状态：§5 里有一半的项（网格加密、时间步、
    # 尺度映射）是**关于一族解**的陈述，一次求解生不出它们。缺证据就必须记未验收
    # ——否则"这个场通过了它自己的那几项检查"会被读成"§5 全部通过"。
    evidence = dict(convergence_evidence or {})
    numeric_failures = list(failures)
    missing = [name for name in EVIDENCE_DEPENDENT_CHECKS if name not in evidence]
    if missing:
        failures.append(
            "缺少证据：" + "、".join(missing) + "（单次求解无法给出，须由 cli.reference_field 提供）"
        )
    else:
        for name, ok in evidence.items():
            if not ok:
                failures.append(f"{name}：核查未通过")

    numerical_ok = not failures

    # 物理适用性与数据就绪各自独立成条。它们**不是**数值检查的附属品。
    if physics_basis != PHYSICS_BASIS_VERIFIED:
        failures.append(
            f"物理适用性：{physics_basis}（恒定爆速强支与整个球形流动的相容性论证；"
            "数值收敛不能代替它，见 PHYSICS_BASIS.md）"
        )
    # ``data_contract=None`` 表示**这一层不评估数据**（§7.1：数值 / 物理适用性 /
    # 数据就绪三类状态分开记录）。参考场自己的验收与"数据有没有重提"无关——它们
    # 阻塞的是不同的入口。只有在调用方明确传进来时才计入。
    if data_contract is not None:
        contract = dict(data_contract)
        if not contract.get("ready", False):
            failures.append("数据契约：" + str(contract.get("reason", "未就绪")))

    status = ACCEPTED_STATUS if not failures else NOT_ACCEPTED_STATUS
    return {
        "acceptance_status": status,
        # 三个**分开**的状态，对外可读。数值那一项是上面那批指标 + 证据；
        # 物理与数据各自独立，且都不能由数值代码代填。
        "numerical_status": ACCEPTED_STATUS if numerical_ok else NOT_ACCEPTED_STATUS,
        "physics_basis_status": physics_basis,
        "data_contract_status": (
            "not evaluated"
            if data_contract is None
            else ("ready" if dict(data_contract).get("ready", False) else "not ready")
        ),
        "acceptance_evidence_supplied": sorted(evidence),
        "acceptance_metrics": {k: float(v) for k, v in metrics.items()},
        "acceptance_tolerances": dict(ACCEPTANCE_TOLERANCES),
        "acceptance_reason": (
            "全部指标在容差内" if not failures else "；".join(failures)
        ),
        # 数值那一步单独失败的原因，便于区分"数值还没过"和"物理/数据挡住"。
        "numerical_failure_reason": (
            "数值核查全部在容差内" if numerical_ok else "；".join(numeric_failures or failures)
        ),
    }


def solve_reference_field(
    *,
    charge_radius: float = 0.05,
    sample_times: Sequence[float] | None = None,
    config: PrecursorConfig | None = None,
    gamma_air: float = mat.AIR_GAMMA,
    check_outer_boundary: float = 0.5,
    detonation_velocity: float | None = None,
    detonation_config: "DetonationConfig | None" = None,
) -> ReferenceField:
    """在参考半径上求解方案 B 的两介质前置，返回 ``q_ref``。

    流程就是 §2 的完整链路：中心起爆的有限爆速球对称爆轰 → 波前到达装药表面那一刻
    的产物场 → 接上外部静止空气 → **只积分 epsilon 段**到交接时刻 ``T_h``。

    ``sample_times`` 给的是**相对交接时刻的时间**。``tau = T - T_h = 0`` 必须在其中，
    因为 PINN 的初值就是那一刻的两侧场。
    """
    if detonation_velocity is None:
        raise PrecursorError(
            "solve_reference_field 需要卡片爆速 detonation_velocity：T_b = R_0/D 与"
            "波前状态都由它决定，省略它没有物理上正确的默认值"
        )
    from .detonation import DetonationConfig, detonation_product_profile

    products = detonation_product_profile(
        charge_radius=charge_radius,
        detonation_velocity=detonation_velocity,
        config=detonation_config,
    )
    solver = SphericalPrecursor(
        charge_radius=charge_radius,
        config=config,
        gamma_air=gamma_air,
        products=products,
    )
    # ``sample_times`` 是**相对爆轰渡越时间** ``T_b = R_0/D`` 的时间。参考场对外
    # 一律用 PHYSICAL 时间（LS-DYNA 的原始时间）登记，所以这里加上 T_b。两者都只有
    # 秒的量纲，混用不会报错，只会把整条曲线平移到另一个时刻。
    relative = np.asarray(
        sorted(set(float(t) for t in (sample_times or default_sample_times()))),
        dtype=np.float64,
    )
    if relative.size == 0:
        raise PrecursorError("sample_times 不能为空")
    if relative[0] < 0.0:
        raise PrecursorError(
            f"sample_times 是相对 T_b 的时间，不能为负（收到 {relative[0]!r}）："
            "T_b 之前是爆轰阶段，由 physics/detonation.py 描述，不是本求解器的定义域"
        )
    times = relative + float(products.time)

    snapshots: list[PrecursorState] = []

    def collect(state: PrecursorState) -> None:
        snapshots.append(state)

    # **交接态在这里冻结。** 放到 run_to 之后去取，读到的是已经演化过的状态，
    # 于是"交接缺陷"里混进了前置演化，"前置演化"那一栏恒等于零——三段分成两段，
    # 而且分界还画错了地方。
    _product_cells = solver.mesh.material == PRODUCT
    _handover_mass = float(np.sum(solver.mesh.mass[_product_cells]))
    # 用**节点加权**动能——与格式守恒的量一致，也和 detonation 侧的
    # ``energy_from_density`` 口径统一（那一侧也是节点估计）。用单元左端节点速度
    # 读出来的数会小看动能的差异，"交接守恒"于是只是口径造出来的。
    _n_product = int(np.count_nonzero(_product_cells))
    _handover_energy = float(
        np.sum(solver.mesh.mass[_product_cells] * solver.specific_energy[_product_cells])
    ) + _node_weighted_kinetic_energy(
        solver.mesh.mass[_product_cells], solver.velocity[: _n_product + 1]
    )
    # **节点加权**的总能量——这才是拉格朗日格式守恒的那一个（动量方程作用在
    # 节点上，它的能量恒等式配的就是节点质量）。用单元中心约定去读，会读出一个
    # 根本不在格式里的"损失"：实测前置阶段 −8.8e-3（单元约定）对 +3.7e-4（节点
    # 约定），而外边界完全未受扰动。
    total_domain_energy_handover = float(solver.state().total_energy())

    final = solver.run_to(
        float(times[-1]),
        stop_when_shock_reaches=check_outer_boundary,
        sample_times=times[:-1],
        on_sample=collect,
    )
    snapshots.append(final)

    # 每个快照带**自己那一帧**的当前单元中心：拉格朗日网格在动，同一时刻的单元
    # 中心和初始时刻完全不是一组半径。用初始中心当空间坐标、再拿当前物理半径去
    # 插值，读出来的状态属于另一个位置——这一段历史见 ReferenceField 的类文档。
    cell_radii = np.stack([state.centres for state in snapshots])
    density = np.stack([state.density for state in snapshots])
    velocity = np.stack([state.cell_velocity() for state in snapshots])
    energy = np.stack([state.specific_energy for state in snapshots])

    # 质量与能量守恒的独立核查：拉格朗日网格的单元质量是常量，所以守恒在构造上
    # 成立；这里报告的是内外边界通量与动能，用来确认没有别的损失通道。
    # -- 分阶段收支：产物构造 / 交接 / 前置演化，三段各自记账 --------------
    #
    # 不能拿"前置阶段的质量恒定"当作全链路守恒的证据：拉格朗日格式的单元质量是
    # 常量，那一条**在构造上就成立**，它证明不了交接（守恒的重叠搬移到另一套网格）
    # 没有损失。所以三段分开测，并把每一段的绝对值都记下来。
    product_mask = solver.mesh.material == PRODUCT
    exact_mass = mat.PRODUCT_REFERENCE_DENSITY * charge_radius**3 / 3.0
    exact_energy = mat.PRODUCT_VOLUME_ENERGY * charge_radius**3 / 3.0
    detonation_mass = float(products.diagnostics.get("mass_from_density", float("nan")))
    # 源侧也要用**节点加权**，否则量的是口径差而不是交接缺陷。
    detonation_energy = float(
        products.diagnostics.get("energy_node_weighted", float("nan"))
    )
    handover_mass = _handover_mass
    handover_energy = _handover_energy
    final_energy = float(
        np.sum(
            final.mass[product_mask]
            * (
                final.specific_energy[product_mask]
                + 0.5 * final.velocity[:-1][product_mask] ** 2
            )
        )
    )
    staged = {
        "stage1_product_construction": {
            "mass": detonation_mass,
            "energy": detonation_energy,
            "mass_relative": (detonation_mass - exact_mass) / exact_mass,
            "energy_relative": (detonation_energy - exact_energy) / exact_energy,
        },
        "stage2_handover": {
            "mass": handover_mass,
            "energy": handover_energy,
            "mass_relative": (handover_mass - detonation_mass) / exact_mass,
            "energy_relative": (handover_energy - detonation_energy) / exact_energy,
        },
        "stage3_precursor_evolution": {
            # **只取产物单元**。整张网格一直铺到 40 R_0 的空气，把它一起求和会
            # 得到比精确值大 48 倍的质量，读起来像"丢了一大截"，其实只是把空气
            # 算进了装药。
            "mass": float(np.sum(final.mass[product_mask])),
            "energy": final_energy,
            # 拉格朗日格式这一项恒等于零，是**构造性**的，不构成独立证据——
            # 它能证明的只有"前置阶段没有质量源"，证明不了交接没有损失。
            # 全域质量：拉格朗日格式没有质量源，这一项**恒等于零**。
            "mass_relative": (
                float(np.sum(final.mass)) - float(np.sum(solver.mesh.mass))
            )
            / float(np.sum(solver.mesh.mass)),
            # 产物区自己的能量变化**不是**守恒量：产物推动外面的空气，能量是
            # 转出去的，不是丢掉的。把它当守恒读数会得出"前置阶段丢 2%"这种
            # 既错又吓人的结论。
            "energy_relative_product_only": (final_energy - handover_energy) / exact_energy,
            # 守恒的读数是**整个计算域**（产物 + 空气）的总能量。外边界取在
            # 40 R_0，ε 段里扰动只到约 1.15 R_0，所以外边界的通量应当是零。
            # 守恒的读数是**整个计算域**（产物 + 空气）的**节点加权**总能量。
            # 外边界取在 40 R_0，ε 段里扰动只到约 1.15 R_0，实测外边界压力偏离
            # 与速度都是 1e-15 量级，所以这一项的变化应当只是舍入。
            "energy_total_domain": float(final.total_energy()),
            "energy_relative_total_domain": (
                float(final.total_energy()) - total_domain_energy_handover
            )
            / exact_energy,
            "energy_relative_product_only_cell_convention": (
                final_energy - handover_energy
            )
            / exact_energy,
        },
        "exact": {"mass": exact_mass, "energy": exact_energy},
    }
    metrics = {
        "detonation_mass_invariant_relative": abs(
            float(products.diagnostics.get("mass_relative_error", float("nan")))
        ),
        # **权威口径是节点加权的那一个**（§7.1）。格式守恒的是
        # ``sum_i m_i e_i + sum_j M_j u_j^2/2``；``energy_relative_error`` 是单元
        # 中心口径，它是辅助诊断，**不能**用来判定。以前这里读的正是单元中心那一个，
        # 而 stage1 报告里印的是节点加权那一个——报告和判定检查的不是同一个量。
        "detonation_energy_invariant_relative": abs(
            float(
                products.diagnostics.get(
                    "energy_relative_error_node_weighted", float("nan")
                )
            )
        ),
        "handover_mass_defect_relative": abs(staged["stage2_handover"]["mass_relative"]),
        "handover_energy_defect_relative": abs(staged["stage2_handover"]["energy_relative"]),
        "centre_velocity_absolute": abs(float(solver.velocity[0])),
        "precursor_energy_drift_relative": abs(
            staged["stage3_precursor_evolution"]["energy_relative_total_domain"]
        ),
        "precursor_mass_drift_relative": abs(
            staged["stage3_precursor_evolution"]["mass_relative"]
        ),
    }
    # §7.1：正值、外边界、网格次序、球心这些**必须进放行条件**，不能只打印。
    metrics.update(field_violation_metrics(final))
    acceptance = assess_acceptance(metrics)

    metadata = {
        "method": (
            "scheme B: the product field at T_b = R_0/D from physics/detonation.py "
            "(centre initiation, finite detonation speed, Rayleigh-Hugoniot front "
            "state), then a 1D spherical Lagrangian two-material finite-volume "
            "precursor advanced over the epsilon interval to T_h = T_b + epsilon"
        ),
        "q1": solver.config.q1,
        "q2": solver.config.q2,
        "cfl": solver.config.cfl,
        "cells_inside": solver.config.cells_inside,
        "growth": solver.config.growth,
        "outer_radius_factor": solver.config.outer_radius_factor,
        "mesh": solver.mesh.describe(),
        "steps": int(final.diagnostics.get("steps", 0)),
        "total_mass_initial": float(solver.mesh.mass.sum()),
        "total_mass_final": final.total_mass(),
        "initial_condition": (
            "scheme B: the product field at T_b = R_0/D from physics/detonation.py "
            "(centre initiation, finite detonation speed, Rayleigh-Hugoniot front state); "
            "the charge is NOT uniform and u is NOT zero"
        ),
        # §5 的验收状态随产物一起走，而且由**实测指标**算出来（见
        # ``assess_acceptance``）。状态、指标、容差、理由四件一起记。
        "staged_budget": staged,
        **acceptance,
        "spatial_coordinates": (
            "per-frame current Lagrangian cell centres; queries are interpolated within "
            "the material block each query point belongs to at that frame"
        ),
        "contact_displacement_over_R0": float(
            snapshots[-1].contact_radius / charge_radius
        ),
        "max_cell_displacement_over_R0": float(
            np.max(np.abs(cell_radii[-1] - cell_radii[0])) / charge_radius
        ),
    }
    field = ReferenceField(
        times=np.asarray([state.time for state in snapshots], dtype=np.float64),
        cell_radii=cell_radii,
        density=density,
        velocity=velocity,
        specific_energy=energy,
        material=solver.mesh.material.copy(),
        contact_radius=np.asarray([state.contact_radius for state in snapshots]),
        shock_radius=np.asarray([state.shock_radius for state in snapshots]),
        reference_radius=float(charge_radius),
        metadata=metadata,
    )
    # **场一出生就带内容标识**（identity），**指标在算出来的那一刻就绑定它所属的
    # 场内容**（Trusty URIs：内容变了就是另一个对象，不能继承原对象的证据）。
    # 少了后一条，"重新保存"会刷新场标识、却让旧指标继续有效——旧指标于是成了
    # 新数据的合格证明。
    identity = field_content_hash(field)
    field.metadata["field_content_hash"] = identity
    field.metadata["metrics_field_hash"] = identity
    return field


def default_sample_times(*, reference_start: float = 1.0e-6, factor: float = 4.0, count: int = 41):
    """默认采样时刻：相对 ``T_b`` 的 ``[0, factor * epsilon_ref]`` 上均匀取点。

    ``epsilon_ref = 1 us`` 是交接延迟的参考值，而交接时刻 ``T_h = T_b + epsilon``
    落在相对时间 ``epsilon_ref`` 处，所以这个窗口同时覆盖交接前的一小段和交接后。
    """
    return list(np.linspace(0.0, factor * reference_start, count))


__all__ = [
    "AIR",
    "MATERIAL_NAMES",
    "PRODUCT",
    "LagrangianMesh",
    "PrecursorConfig",
    "PrecursorError",
    "PrecursorState",
    "ReferenceField",
    "ACCEPTANCE_TOLERANCES",
    "ACCEPTED_STATUS",
    "NOT_ACCEPTED_STATUS",
    "SphericalPrecursor",
    "assess_acceptance",
    "build_lagrangian_mesh",
    "default_sample_times",
    "solve_reference_field",
]
