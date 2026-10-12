"""§15.3、§15.5：解析光滑函数检验移动坐标链式法则与前沿物理时间导数；
静止均匀状态的球对称 PDE 残差，尤其是动量几何项。"""
from __future__ import annotations

import pytest

from spherical_blast_pinn.physics.moving_frame import (
    AIR,
    PRODUCT,
    FrontData,
    air_chain_factors,
    differentiate_scalar,
    product_chain_factors,
)
from spherical_blast_pinn.physics.pde import residuals_from_physical_derivatives
from spherical_blast_pinn.physics.similitude import SimilitudeScales

import torch


R0 = 0.05
SCALES = SimilitudeScales.for_radius(R0, time_scale=2.0, detonation_velocity=6930.0)


def _fronts_at(t):
    """一组解析前沿，不带任何网络。"""
    a, b, c, d = 3800.0, 2.0e9, 1200.0, 5.0e8
    contact = R0 + a * t + b * t * t
    gap = c * t + d * t * t
    return FrontData(
        contact_radius=contact,
        shock_radius=contact + gap,
        contact_speed=a + 2.0 * b * t,
        shock_speed=a + 2.0 * b * t + c + 2.0 * d * t,
    )


def _coordinate_map(medium, radius, physical_time):
    """把 ``(r, t)`` 映射到网络输入 ``(s, xi 或 eta)`` 的**可微**包装器。"""
    fronts = _fronts_at(physical_time)
    s = SCALES.reference_velocity * physical_time / (R0 * SCALES.time_scale)
    if medium == PRODUCT:
        return s, radius / fronts.contact_radius, fronts
    return s, (radius - fronts.contact_radius) / fronts.gap, fronts


def _smooth_field(s, coordinate):
    """一个纯 ``(s, coord)`` 的解析光滑场，不隐含任何 ``r`` 依赖。"""
    return torch.sin(1.7 * s) * torch.exp(0.9 * coordinate) + 0.3 * s * coordinate


@pytest.mark.parametrize("medium", [PRODUCT, AIR])
def test_chain_rule_matches_autograd(medium):
    """§15.3 的核心：显式链式法则与自动求导必须给同一个数。

    两条路径互相独立：一条把网络对 ``(s, coord)`` 的偏导乘上解析系数，另一条
    直接对 ``(r, t)`` 求导。用其中一条验证另一条，比任何一条自己和自己比都可信。
    """
    physical_time = torch.tensor([[0.4e-6]], dtype=torch.float64, requires_grad=True)
    fronts = _fronts_at(physical_time)
    if medium == PRODUCT:
        radius = (0.6 * fronts.contact_radius).detach().requires_grad_(True)
    else:
        radius = (fronts.contact_radius + 0.5 * fronts.gap).detach().requires_grad_(True)

    # 显式路径
    s, coordinate, fronts = _coordinate_map(medium, radius, physical_time)
    s = s.detach().requires_grad_(True)
    coordinate = coordinate.detach().requires_grad_(True)
    factors = (
        product_chain_factors(fronts, SCALES, radius)
        if medium == PRODUCT
        else air_chain_factors(fronts, SCALES, radius)
    )
    value = _smooth_field(s, coordinate)
    d_r_explicit, d_t_explicit = factors.physical_derivatives(
        differentiate_scalar(value, s), differentiate_scalar(value, coordinate)
    )

    # 自动求导路径
    s2, coordinate2, _ = _coordinate_map(medium, radius, physical_time)
    value2 = _smooth_field(s2, coordinate2)
    d_r_auto = differentiate_scalar(value2, radius)
    d_t_auto = differentiate_scalar(value2, physical_time)

    assert float(d_r_explicit) == pytest.approx(float(d_r_auto), rel=1e-10, abs=1e-12)
    assert float(d_t_explicit) == pytest.approx(float(d_t_auto), rel=1e-10, abs=1e-12)


def test_missing_the_moving_time_term_changes_the_answer():
    """把移动坐标的时间项漏掉会得到不同的数——链式检验不能是恒等式。"""
    physical_time = torch.tensor([[0.4e-6]], dtype=torch.float64, requires_grad=True)
    fronts = _fronts_at(physical_time)
    radius = (0.6 * fronts.contact_radius).detach().requires_grad_(True)
    s, coordinate, fronts = _coordinate_map(PRODUCT, radius, physical_time)
    s = s.detach().requires_grad_(True)
    coordinate = coordinate.detach().requires_grad_(True)
    factors = product_chain_factors(fronts, SCALES, radius)
    value = _smooth_field(s, coordinate)
    f_s = differentiate_scalar(value, s)
    f_coordinate = differentiate_scalar(value, coordinate)

    full = factors.d_time_ds * f_s + f_coordinate * factors.d_coord_dt
    without_moving_term = factors.d_time_ds * f_s
    relative = abs(float(full) - float(without_moving_term)) / max(abs(float(full)), 1e-30)
    assert relative > 1.0e-3, "移动坐标时间项在这一组参数下应当是可分辨的"


def test_front_speed_is_not_the_parametric_slope():
    """``dR_c/dt = v_c + tau dv_c/dtau``：前沿速度不是参数化里那个斜率。"""
    from spherical_blast_pinn.models.front_net import FrontNet

    net = FrontNet(scales=SCALES, initial_contact_hat=1.08, initial_gap_hat=0.02, seed=0)
    # 输出层权重初值很小（见 FRONT_OUTPUT_WEIGHT_SCALE），这里放大它，让前沿
    # 明显随 s 变化，差分才不是在两个几乎相等的数上做减法。
    with torch.no_grad():
        net.mlp.network[-1].weight.mul_(2000.0)
    s = torch.linspace(0.0, 1.0, 5, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    state = net.state(s, charge_radius=R0, create_graph=False)

    # Rc = R0 hat_Rc(s) 且 ds/dt = U_*/(R0 hat T_*)，所以
    # dRc/dt = (U_*/hat T_*) d(hat_Rc)/ds。
    eps = 1e-7
    plus = net.positions(s.detach() + eps).contact_hat
    minus = net.positions(s.detach() - eps).contact_hat
    ds_dt = SCALES.reference_velocity / (R0 * SCALES.time_scale)
    numeric = R0 * (plus - minus) / (2.0 * eps) * ds_dt
    assert torch.allclose(state.contact_speed, numeric, rtol=1e-4, atol=1e-9)
    # 前沿确实在动，否则这个检验是空的。差分的量级本身就是 2 eps * dhat/ds，
    # 所以判据要加在导数上，不是加在差分上。
    slope = float(((plus - minus) / (2.0 * eps)).abs().max())
    assert slope > 1.0e-3, f"hat_Rc 几乎不随 s 变化（斜率 {slope:.3e}），检验是空的"


# --------------------------------------------------------------------------
# §15.5 静止均匀状态的球对称 PDE 残差
# --------------------------------------------------------------------------


def test_uniform_static_state_has_zero_residual_including_the_geometric_term():
    """均匀静止场：``u = 0``，所有导数非零的只有零，几何项也必须为零。

    动量方程右端的几何源项是 ``2 rho u^2 / r``。在 ``u = 0`` 时它为零；把系数
    写成 ``rho u / r`` 或漏掉一个 ``u`` 都能通过"常数场"测试，所以下面的
    ``test_geometric_term_is_quadratic_in_velocity`` 另外用一个匀速膨胀场来钉它。
    """
    radius = torch.tensor([[0.01], [0.03], [0.08]], dtype=torch.float64)
    rho = torch.full_like(radius, 1.225)
    u = torch.zeros_like(radius)
    e = torch.full_like(radius, 2.0e5)
    p = torch.full_like(radius, 101325.0)
    zeros = torch.zeros_like(radius)

    mass, momentum, energy = residuals_from_physical_derivatives(
        rho=rho, u=u, e=e, p=p,
        drho_dr=zeros, du_dr=zeros, de_dr=zeros, dp_dr=zeros,
        drho_dt=zeros, du_dt=zeros, de_dt=zeros, dp_dt=zeros,
        r=radius,
    )
    assert torch.allclose(mass, zeros, atol=1e-12)
    assert torch.allclose(momentum, zeros, atol=1e-12)
    assert torch.allclose(energy, zeros, atol=1e-12)


def test_geometric_term_is_quadratic_in_velocity():
    """动量几何项 = ``2 rho u^2 / r``。

    用一个只有径向速度、其余均匀的场：连续方程的几何项是 ``2 rho u / r``，
    动量的是 ``2 rho u^2 / r``。两者对 ``u`` 的幂次不同，所以这一步同时钉住了
    "动量里是二次、连续里是一次"，也就钉住了"没有把两个几何项写成一个"。
    """
    radius = torch.tensor([[0.02], [0.05]], dtype=torch.float64)
    rho = torch.full_like(radius, 1.225)
    u = torch.tensor([[100.0], [100.0]], dtype=torch.float64)
    e = torch.full_like(radius, 2.0e5)
    p = torch.full_like(radius, 101325.0)
    zeros = torch.zeros_like(radius)

    mass, momentum, _ = residuals_from_physical_derivatives(
        rho=rho, u=u, e=e, p=p,
        drho_dr=zeros, du_dr=zeros, de_dr=zeros, dp_dr=zeros,
        drho_dt=zeros, du_dt=zeros, de_dt=zeros, dp_dt=zeros,
        r=radius,
    )
    assert torch.allclose(mass, 2.0 * rho * u / radius, rtol=1e-12)
    assert torch.allclose(momentum, 2.0 * rho * u * u / radius, rtol=1e-12)
    # 二次与一次之比就是 u
    ratio = momentum / mass
    assert torch.allclose(ratio, u, rtol=1e-12)
