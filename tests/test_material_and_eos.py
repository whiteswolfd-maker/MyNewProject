"""§15.1、§15.2：单位换算、E0 到比内能、JWL 初值压力、空气 EOS、
一般 JWL 的偏导与声速平方、物理有效域。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spherical_blast_pinn.constants import UNIT_SYSTEMS
from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.eos import (
    EOSError,
    JWLParameters,
    check_linear_polynomial_is_ideal_gas,
    ideal_gas_pressure_numpy,
    ideal_gas_sound_speed_numpy,
    jwl_pressure_derivatives,
    jwl_pressure_numpy,
    jwl_sound_speed_numpy,
)

import torch


# --------------------------------------------------------------------------
# §15.1 单位换算与派生量
# --------------------------------------------------------------------------


def test_e_star_is_E0_over_rho0():
    assert mat.PRODUCT_SPECIFIC_ENERGY == pytest.approx(6.0e9 / 1630.0, rel=1e-15)
    assert mat.PRODUCT_SPECIFIC_ENERGY == pytest.approx(3.680981595e6, rel=1e-9)


def test_u_star_is_sqrt_of_e_star():
    assert mat.REFERENCE_VELOCITY == pytest.approx(math.sqrt(mat.PRODUCT_SPECIFIC_ENERGY), rel=1e-15)
    assert mat.REFERENCE_VELOCITY == pytest.approx(1919.0, rel=1e-3)


def test_product_pressure_at_the_reference_state():
    """参考状态 (rho_ref, e_*) 的 EOS 压力。**不是初值压力**——方案 B 里没有
    整球均匀初压这个量。"""
    """规范 §4 的 P_init；它必须由 EOS 得到，不能是一个独立常量。"""
    assert mat.product_pressure_at_reference_state() == pytest.approx(8.220261897e9, rel=1e-8)


def test_product_pressure_scale_is_exactly_E0():
    """``rho_*,p U_*^2 = E0``。这不是巧合，是 ``U_* = sqrt(E0/rho0)`` 的定义。"""
    scale = mat.PRODUCT_REFERENCE_DENSITY * mat.REFERENCE_VELOCITY**2
    assert scale == pytest.approx(mat.PRODUCT_VOLUME_ENERGY, rel=1e-15)


def test_air_energy_comes_from_the_card_not_from_standard_atmosphere():
    """环境内能从**卡片** ``*EOS_LINEAR_POLYNOMIAL`` 的 E0 推出，方向不能反。

    卡片 ``E0 = 0.2533 MPa``（按单位参考体积）给出 ``e0 = E0/rho0 = 206775.510 J/kg``，
    再给出 ``p0 = (gamma-1) rho0 e0 = 101320.0 Pa``。

    反过来做——先取标准大气压 101325 Pa 再反推 e——会得到 206785.714 J/kg，与卡片
    的 E0 差 5 Pa 对应的一点点。DYNA 实际初始化出来的环境压力是 **101320 Pa**（在
    d3plot 里可以直接读到），所以与数据比较的量必须用卡片值。两种环境压力不能混用。
    """
    assert mat.AIR_VOLUME_ENERGY == pytest.approx(0.2533e6, rel=1e-12)
    assert mat.AIR_SPECIFIC_ENERGY == pytest.approx(mat.AIR_VOLUME_ENERGY / mat.AIR_DENSITY, rel=1e-15)
    assert mat.AIR_SPECIFIC_ENERGY == pytest.approx(206775.510, rel=1e-8)
    assert mat.AIR_PRESSURE == pytest.approx(
        (mat.AIR_GAMMA - 1.0) * mat.AIR_DENSITY * mat.AIR_SPECIFIC_ENERGY, rel=1e-15
    )
    assert mat.AIR_PRESSURE == pytest.approx(101320.0, rel=1e-12)
    # 标准大气压只是参考，两者确实不同，而且差别就是那 5 Pa。
    assert mat.STANDARD_ATMOSPHERE == pytest.approx(101325.0, rel=1e-12)
    assert mat.STANDARD_ATMOSPHERE - mat.AIR_PRESSURE == pytest.approx(5.0, abs=1e-9)
    # 环境态由卡片推出，四个量自洽。
    ambient = mat.ambient_state_from_card()
    assert ambient.pressure == mat.AIR_PRESSURE
    assert ambient.specific_energy == mat.AIR_SPECIFIC_ENERGY
    assert ambient.velocity == 0.0


def test_mm_tonne_s_density_factor_is_1e12_not_1e9():
    """mm-tonne-s 的密度因子。

    吨每立方毫米到 kg/m³ 是 ``1e3 / (1e-3)^3 = 1e12``。写成 ``1e9`` 会让每一
    个从 mm-tonne 卡片读来的材料参数差三个数量级，而密度那一项"看起来差不多"
    正好能把错误藏起来。
    """
    system = UNIT_SYSTEMS["mm_s_t"]
    assert system.density_to_si == pytest.approx(1.0e12, rel=1e-12)
    assert system.pressure_to_si == pytest.approx(1.0e6, rel=1e-12)
    assert system.velocity_to_si == pytest.approx(1.0e-3, rel=1e-12)
    # 卡片值 1.63e-9 t/mm³ 应当换回 1630 kg/m³。
    assert 1.63e-9 * system.density_to_si == pytest.approx(1630.0, rel=1e-9)


def test_air_linear_polynomial_card_is_ideal_gas():
    coefficients = {"C0": 0.0, "C1": 0.0, "C2": 0.0, "C3": 0.0, "C4": 0.4, "C5": 0.4, "C6": 0.0}
    check_linear_polynomial_is_ideal_gas(coefficients, 1.4)


def test_a_wrong_air_card_is_rejected():
    """C4 错了就整个空气模型错了，必须是硬失败。"""
    bad = {"C0": 0.0, "C1": 0.0, "C2": 0.0, "C3": 0.0, "C4": 0.5, "C5": 0.4, "C6": 0.0}
    with pytest.raises(EOSError):
        check_linear_polynomial_is_ideal_gas(bad, 1.4)


# --------------------------------------------------------------------------
# §15.2 JWL 偏导与声速
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rho,e", [(1630.0, mat.PRODUCT_SPECIFIC_ENERGY), (2206.0, 6.0e6), (400.0, 3.0e6)])
def test_jwl_derivatives_match_finite_differences(rho, e):
    params = mat.product_jwl()
    density = torch.tensor([[rho]], dtype=torch.float64, requires_grad=True)
    energy = torch.tensor([[e]], dtype=torch.float64, requires_grad=True)
    pressure = jwl_pressure_numpy(np.array([rho]), np.array([e]), params)[0]

    analytic_rho, analytic_e = jwl_pressure_derivatives(
        density.detach(), energy.detach(), params
    )

    step_rho = rho * 1.0e-6
    numeric_rho = (
        jwl_pressure_numpy(np.array([rho + step_rho]), np.array([e]), params)[0]
        - jwl_pressure_numpy(np.array([rho - step_rho]), np.array([e]), params)[0]
    ) / (2.0 * step_rho)
    step_e = e * 1.0e-6
    numeric_e = (
        jwl_pressure_numpy(np.array([rho]), np.array([e + step_e]), params)[0]
        - jwl_pressure_numpy(np.array([rho]), np.array([e - step_e]), params)[0]
    ) / (2.0 * step_e)

    assert float(analytic_rho) == pytest.approx(numeric_rho, rel=1e-6)
    assert float(analytic_e) == pytest.approx(numeric_e, rel=1e-6)


def test_jwl_and_numpy_paths_agree():
    """torch 与 numpy 两条通路必须给同一个数。"""
    params = mat.product_jwl()
    density = torch.tensor([[1400.0]], dtype=torch.float64)
    energy = torch.tensor([[3.0e6]], dtype=torch.float64)
    from spherical_blast_pinn.physics.eos import jwl_pressure

    torch_value = float(jwl_pressure(density, energy, params))
    numpy_value = float(jwl_pressure_numpy(np.array([1400.0]), np.array([3.0e6]), params)[0])
    assert torch_value == pytest.approx(numpy_value, rel=1e-15)


def test_jwl_sound_speed_squared_is_positive_in_the_physical_domain():
    params = mat.product_jwl()
    density = np.array([1630.0, 1800.0, 2206.0, 1200.0])
    energy = np.array([mat.PRODUCT_SPECIFIC_ENERGY] * 4)
    c2 = jwl_sound_speed_numpy(density, energy, params) ** 2
    assert np.all(np.isfinite(c2))
    assert np.all(c2 > 0.0)


def test_jwl_sound_speed_matches_the_thermodynamic_definition():
    """``c^2 = (dp/drho)|_e + (p/rho^2)(dp/de)|_rho``。"""
    params = mat.product_jwl()
    rho, e = 1500.0, 3.2e6
    p = float(jwl_pressure_numpy(np.array([rho]), np.array([e]), params)[0])
    dp_drho, dp_de = jwl_pressure_derivatives(
        torch.tensor([[rho]]), torch.tensor([[e]]), params
    )
    expected = float(dp_drho) + p / rho**2 * float(dp_de)
    actual = float(jwl_sound_speed_numpy(np.array([rho]), np.array([e]), params)[0]) ** 2
    assert actual == pytest.approx(expected, rel=1e-12)


def test_air_sound_speed_is_gamma_p_over_rho():
    rho, e = 1.225, mat.AIR_SPECIFIC_ENERGY
    p = float(ideal_gas_pressure_numpy(np.array([rho]), np.array([e]))[0])
    expected = math.sqrt(1.4 * p / rho)
    actual = float(ideal_gas_sound_speed_numpy(np.array([rho]), np.array([e]))[0])
    assert actual == pytest.approx(expected, rel=1e-12)


def test_ambient_air_sound_speed_is_about_340():
    from spherical_blast_pinn.physics import ambient_air

    air = ambient_air()
    p = ideal_gas_pressure_numpy(np.array([air.density]), np.array([air.specific_energy]))[0]
    assert math.sqrt(1.4 * float(p) / air.density) == pytest.approx(340.3, rel=2e-3)


def test_jwl_rejects_a_non_positive_reference_density():
    with pytest.raises(EOSError):
        JWLParameters(A=1.0, B=1.0, R1=1.0, R2=1.0, omega=0.3, rho0=0.0).validate()
