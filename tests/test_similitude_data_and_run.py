"""§15.8–§15.11：尺度变换、数据材料分区与验证数据泄漏。

旧的参考场/三半径联合更新/检查点恢复已经不在这条链路里：初值来源改为
``PlanarStartupField``，训练接口尚未接入。这里只保留与初值来源无关的尺度与
数据测试。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spherical_blast_pinn.data.extracted_case import (
    ExtractedCase,
    specific_energy_from_pressure,
)
from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.similitude import (
    REFERENCE_RADIUS,
    SimilitudeScales,
    freeze_time_scale,
    physical_to_pinn_time,
    radius_ratio,
    detonation_transit_time,
    handover_time,
    pinn_to_physical_time,
    start_delay,
    surface_arrival_relative_time,
)

import torch


# --------------------------------------------------------------------------
# §15.8 尺度变换
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "radius,delay,transit,handover",
    [
        (0.05, 1.0e-6, 7.215007215007215e-6, 8.215007215007215e-6),
        (0.08, 1.6e-6, 11.544011544011545e-6, 13.144011544011545e-6),
        (0.10, 2.0e-6, 14.430014430014431e-6, 16.430014430014433e-6),
    ],
)
def test_scheme_b_clock_matches_the_specification_table(radius, delay, transit, handover):
    """**§3 的时间表。** 交接延迟按半径等比；时间原点还要加一个理论爆轰渡越时间。

    方案 A 的原点只有 ``epsilon``（50 mm 时 1 µs），那等于宣称整球在 1 µs 时已经是
    均匀产物；方案 B 的原点是 ``T_h = T_b + epsilon``，因为真实算例中心起爆、有限爆速。
    两个数都只有秒的量纲，用错了不会报错，只会把整条轨迹平移一个 ``T_b``。
    """
    assert radius_ratio(radius) == pytest.approx(radius / REFERENCE_RADIUS, rel=1e-15)
    assert start_delay(radius) == pytest.approx(delay, rel=1e-12)
    assert detonation_transit_time(radius, detonation_velocity=6930.0) == pytest.approx(
        transit, rel=1e-12
    )
    assert handover_time(radius, detonation_velocity=6930.0) == pytest.approx(
        handover, rel=1e-12
    )
    # 原点必须**严格晚于**方案 A 的那个值，差值正好是 T_b。
    assert handover_time(radius, detonation_velocity=6930.0) - start_delay(radius) == pytest.approx(
        transit, rel=1e-12
    )


def test_the_handover_instant_maps_to_zero_on_the_pinn_clock():
    """**§3：tau = T - T_h，只转换一次。**

    交接时刻本身必须映到 ``tau = 0``。转换两次或者漏掉 ``T_b`` 都会让这个断言失败，
    而两种错误都只有秒的量纲，不会自己报错——只会把整条轨迹整体平移。
    """
    radius = 0.08
    handover = handover_time(radius, detonation_velocity=6930.0)
    assert float(physical_to_pinn_time(handover, radius, detonation_velocity=6930.0)) == pytest.approx(
        0.0, abs=1e-18
    )
    # 往返必须回到原值。
    assert float(
        pinn_to_physical_time(0.0, radius, detonation_velocity=6930.0)
    ) == pytest.approx(handover, rel=1e-15)
    # 表面到达后时间的定义：t_surface = tau + epsilon。
    assert float(
        surface_arrival_relative_time(0.0, radius, detonation_velocity=6930.0)
    ) == pytest.approx(start_delay(radius), rel=1e-15)






def test_time_scale_is_frozen_from_training_radii_only():
    scale, provenance = freeze_time_scale({0.05: 1.0e-4, 0.08: 1.0e-4, 0.10: 1.0e-4})
    expected = mat.REFERENCE_VELOCITY * 1.0e-4 / 0.05
    assert scale == pytest.approx(expected, rel=1e-12)
    assert provenance["training_radii_only"] is True
    assert provenance["governing_radius"] == pytest.approx(0.05)


def test_time_scale_refuses_an_empty_or_non_positive_set():
    from spherical_blast_pinn.physics.similitude import SimilitudeError

    with pytest.raises(SimilitudeError):
        freeze_time_scale({})
    with pytest.raises(SimilitudeError):
        freeze_time_scale({0.05: 0.0})


def test_network_time_is_hat_t_over_hat_T():
    scales = SimilitudeScales.for_radius(0.08, time_scale=3.0, detonation_velocity=6930.0)
    pinn_time = 5.0e-5
    assert scales.hat_time(pinn_time) == pytest.approx(
        mat.REFERENCE_VELOCITY * pinn_time / 0.08, rel=1e-15
    )
    assert scales.network_time(pinn_time) == pytest.approx(
        scales.hat_time(pinn_time) / 3.0, rel=1e-15
    )
    assert scales.physical_time_from_network(scales.network_time(pinn_time)) == pytest.approx(
        pinn_time + handover_time(0.08, detonation_velocity=6930.0), rel=1e-12
    )


def test_pressure_scales_are_medium_specific():
    scales = SimilitudeScales.for_radius(0.05, time_scale=1.0, detonation_velocity=6930.0)
    assert scales.product_pressure == pytest.approx(mat.PRODUCT_VOLUME_ENERGY, rel=1e-15)
    assert scales.air_pressure == pytest.approx(
        mat.AIR_DENSITY * mat.REFERENCE_VELOCITY**2, rel=1e-15
    )
    # 两者相差三个数量级，这正是接触面压力残差必须共用一个尺度的原因。
    assert scales.product_pressure / scales.air_pressure > 100.0


def test_residual_scales_follow_the_specification():
    scales = SimilitudeScales.for_radius(0.05, time_scale=1.0, detonation_velocity=6930.0)
    u = mat.REFERENCE_VELOCITY
    rho = mat.PRODUCT_REFERENCE_DENSITY
    assert scales.mass_residual_scale("product") == pytest.approx(rho * u / 0.05, rel=1e-15)
    assert scales.momentum_residual_scale("product") == pytest.approx(rho * u**2 / 0.05, rel=1e-15)
    assert scales.energy_residual_scale("product") == pytest.approx(rho * u**3 / 0.05, rel=1e-15)
    assert scales.density_scale("air") == pytest.approx(mat.AIR_DENSITY, rel=1e-15)


# --------------------------------------------------------------------------
# §15.9 数据材料分区与能量转换
# --------------------------------------------------------------------------


def _synthetic_case(tmp_path: Path, *, charge_radius: float = 0.05) -> ExtractedCase:
    """造一份最小但**物理上完整**的数据集。

    三段结构缺一不可：产物球、激波后受压缩的空气层、以及未被扰动的环境态。
    没有中间那一段，主激波就会和接触面重合，探测器会（正确地）判定没有空气
    层，于是没有任何标签可用——一个缺了这段的"最小算例"检验不到任何东西。
    """
    radius = np.linspace(0.002, 0.4, 800)
    times = np.linspace(0.0, 2.0e-4, 9)
    density = np.empty((times.size, radius.size))
    pressure = np.empty_like(density)
    velocity = np.zeros_like(density)
    for i in range(times.size):
        expansion = 1.0 + 0.05 * i
        contact = charge_radius * expansion
        shock = contact * 1.9
        product = radius < contact
        shocked = (radius >= contact) & (radius < shock)
        density[i] = np.where(
            product, 1600.0 / expansion, np.where(shocked, 6.0 * mat.AIR_DENSITY, mat.AIR_DENSITY)
        )
        pressure[i] = np.where(
            product, 5.0e8 / expansion**3, np.where(shocked, 25.0 * mat.AIR_PRESSURE, mat.AIR_PRESSURE)
        )
    directory = tmp_path / "case"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "raw_line.npz", t=times, x=radius, rho=density, u=velocity, P=pressure)
    (directory / "metadata.json").write_text(
        json.dumps({"R_0": charge_radius, "x_end": float(radius[-1]), "detonation_velocity_m_s": 6930.0}), encoding="utf-8"
    )
    from spherical_blast_pinn.data.extracted_case import load_extracted_case

    return load_extracted_case(directory, case_id="synthetic")


def test_material_partition_never_crosses_the_discontinuity(tmp_path):
    case = _synthetic_case(tmp_path)
    for frame in range(case.times.size):
        contact = case.contact_radius[frame]
        if not np.isfinite(contact) or not case.contact_valid[frame]:
            continue
        widths = float(np.median(np.diff(case.radius)))
        # 接触面两侧各留了保护带；保护带内必须是无效的。
        near = np.abs(case.radius - contact) <= widths
        assert not np.any(case.valid[frame] & near)


def test_specific_energy_round_trips_through_the_eos(tmp_path):
    case = _synthetic_case(tmp_path)
    good = np.isfinite(case.specific_energy)
    assert np.any(good)
    density = case.density[good]
    pressure = case.pressure[good]
    material = case.material[good]
    from spherical_blast_pinn.physics.eos import ideal_gas_pressure_numpy, jwl_pressure_numpy

    energy = case.specific_energy[good]
    back = np.where(
        material == 0,
        jwl_pressure_numpy(density, energy, mat.product_jwl()),
        ideal_gas_pressure_numpy(density, energy, mat.AIR_GAMMA),
    )
    assert np.allclose(back, pressure, rtol=1e-9)


def test_contact_labels_inside_the_charge_radius_are_rejected(tmp_path):
    """接触面不可能跑到装药半径以内；这样的帧必须被剔除并计数。"""
    case = _synthetic_case(tmp_path)
    assert case.source["frames_rejected_contact_below_charge_radius"] >= 0
    valid = np.flatnonzero(case.contact_valid)
    if valid.size:
        assert np.all(case.contact_radius[valid] >= case.charge_radius - 1e-12)


def test_energy_reconstruction_leaves_nonphysical_points_as_nan():
    density = np.array([1.225, 1630.0])
    pressure = np.array([mat.AIR_PRESSURE, 0.0])
    material = np.array([1, 0])
    energy = specific_energy_from_pressure(density, pressure, material)
    assert np.isfinite(energy[0])
    # 产物在 p = 0 处反解不出正的内能，必须保持 nan 而不是被截断成一个"保护"值。
    assert not np.isfinite(energy[1])


# --------------------------------------------------------------------------
# §15.10 三半径联合前向/反向/优化器更新
# --------------------------------------------------------------------------


DATA_AVAILABLE = all(
    Path(p).joinpath("raw_line.npz").exists()
    for p in (
        "F:/2/extracted",
        "C:/Users/work/Documents/GitHub/2/extracted_80mm",
        "C:/Users/work/Documents/GitHub/2/extracted_100mm",
    )
)




def test_hat_time_epsilon_is_radius_independent():
    """``hat t_epsilon = U_* epsilon(R0)/R0`` 与半径无关：``k`` 约掉了。

    这个量是早期时间对数采样的下界，半径相关性会直接改变三个半径各自采到
    多少早期点。
    """
    values = []
    for radius in (0.05, 0.08, 0.15):
        scale = SimilitudeScales.for_radius(radius, time_scale=2.0, detonation_velocity=6930.0)
        values.append(
            scale.reference_velocity * scale.start_delay() / scale.charge_radius
        )
    assert values[0] == pytest.approx(values[1], rel=1e-12)
    assert values[0] == pytest.approx(values[2], rel=1e-12)
