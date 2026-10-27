"""代码审查清单（2026-09-21）逐项的回归测试。

每一条都直接对应审查里指出的一个**可复现差异**，而不是"能跑通"或"有梯度"。
一条测试只有在该差异存在时会失败，才算数——所以这里的断言大多对着具体数值，
并且刻意用了会让错误实现给出**另一个合法数字**的构造（位置 vs 帧号、初始坐标
vs 当前坐标、50 mm 尺度 vs 本算例尺度）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import torch

from spherical_blast_pinn.data import compatibility as compat
from spherical_blast_pinn.data.compatibility import (
    COMPATIBLE,
    INCOMPATIBLE,
    SYNTHETIC,
    UNKNOWN,
    CaseCompatibility,
    CompatibilityError,
    enforce_training_gate,
)
from spherical_blast_pinn.data.extracted_case import ExtractedCase, load_extracted_case
from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.eos import (
    jwl_sound_speed,
    jwl_sound_speed_numpy,
    jwl_sound_speed_squared,
    jwl_sound_speed_squared_numpy,
)
from spherical_blast_pinn.physics.moving_frame import PRODUCT, FrontData
from spherical_blast_pinn.physics.pde import medium_residual
from spherical_blast_pinn.physics.similitude import (
    SimilitudeScales,
    physical_time_from_scaled,
    scaled_time_coverage,
    handover_time,
)

DTYPE = torch.float64
TIME_SCALE = 2.0
RADII = (0.05, 0.08, 0.10)


def _scales(radius: float) -> SimilitudeScales:
    return SimilitudeScales.for_radius(radius, time_scale=TIME_SCALE, detonation_velocity=6930.0)


# ==========================================================================
# 1. 参考初始流场的空间坐标
# ==========================================================================










# ==========================================================================
# 2. 多半径的 PDE 时间导数
# ==========================================================================


def test_chain_time_factor_follows_the_case_radius():
    """时间项系数是 ``U_*/(R_0 hat T_*)``，``R_0`` 随算例变。

    共享模型的默认尺度来自第一个训练半径；照它算，80/100 mm 的时间导数分别差
    1.6 倍与 2 倍。
    """
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet
    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.product_net import ProductNet

    base = _scales(0.05)
    model = CoupledModel(
        product=ProductNet(scales=base, width=8, depth=1, seed=0),
        air=AirNet(scales=base, width=8, depth=1, seed=1),
        front=FrontNet(scales=base, width=8, depth=1, initial_contact_hat=1.05, seed=2),
        scales=base,
    )
    s = torch.tensor([[0.3]], dtype=DTYPE, requires_grad=True)
    factors = {}
    for radius in RADII:
        fronts = model.front_data(s, charge_radius=radius)
        factors[radius] = model.chain_factors(
            PRODUCT, fronts, torch.full_like(s, 0.5), scales=_scales(radius)
        ).d_time_ds

    expected = {
        radius: mat.REFERENCE_VELOCITY / (radius * TIME_SCALE) for radius in RADII
    }
    for radius in RADII:
        assert factors[radius] == pytest.approx(expected[radius], rel=1e-12)
    # 旧实现在三个半径上都用 50 mm 的尺度：比值会是 1、1、1。
    assert factors[0.08] / factors[0.05] == pytest.approx(0.05 / 0.08, rel=1e-12)
    assert factors[0.10] / factors[0.05] == pytest.approx(0.05 / 0.10, rel=1e-12)


def test_normalised_pde_residual_is_scale_invariant_across_radii():
    """同一网络、同一 ``(s, xi)``，三个半径的无量纲残差必须一致。

    这是审查给的验收判据：尺度相似意味着同一个无量纲时空点上的无量纲方程
    残差与 ``R_0`` 无关。任何一个链式项混进了某一个算例的半径，这个等式就破了。
    """
    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet
    from spherical_blast_pinn.models.product_net import ProductNet

    base = _scales(0.05)
    model = CoupledModel(
        product=ProductNet(scales=base, width=12, depth=2, seed=3),
        air=AirNet(scales=base, width=12, depth=2, seed=4),
        front=FrontNet(scales=base, width=12, depth=2, initial_contact_hat=1.05, seed=5),
        scales=base,
    )
    s_value = 0.4
    coordinate_value = 0.6

    residuals = {}
    for radius in RADII:
        scales = _scales(radius)
        s = torch.tensor([[s_value]], dtype=DTYPE, requires_grad=True)
        coordinate = torch.tensor([[coordinate_value]], dtype=DTYPE, requires_grad=True)
        fronts = model.front_data(s, charge_radius=radius)
        physical_radius = fronts.contact_radius * coordinate
        fields = model.product_fields(s, coordinate)
        factors = model.chain_factors(PRODUCT, fronts, physical_radius, scales=scales)
        residual = medium_residual(
            fields=fields,
            s=s,
            coordinate=coordinate,
            factors=factors,
            radius=physical_radius,
            scales=scales,
            medium=PRODUCT,
            pressure_derivatives=model.product_net.pressure_derivatives,
        )
        mass, momentum, energy = residual.scaled()
        residuals[radius] = (
            float(mass.detach()),
            float(momentum.detach()),
            float(energy.detach()),
        )

    reference = residuals[0.05]
    for radius in RADII[1:]:
        for index, name in enumerate(("mass", "momentum", "energy")):
            assert residuals[radius][index] == pytest.approx(
                reference[index], rel=1.0e-9, abs=1.0e-14
            ), f"{name} residual at R_0 = {radius}"


def test_chain_rule_in_physical_coordinates_agrees_at_every_radius():
    """每个半径上，显式链式法则与物理坐标自动求导必须给同一个数。

    只在一个半径上验证是不够的：偏差正比于 ``R_0``，在 50 mm 上恰好等于零。
    """

    def fronts_at(t, radius_0):
        a, b = 3800.0, 2.0e9
        contact = radius_0 + a * t + b * t * t
        gap = 1200.0 * t + 5.0e8 * t * t
        return FrontData(
            contact_radius=contact,
            shock_radius=contact + gap,
            contact_speed=a + 2.0 * b * t,
            shock_speed=a + 2.0 * b * t + 1200.0 + 1.0e9 * t,
        )

    def field(s, coordinate):
        return torch.sin(1.7 * s) * torch.exp(0.9 * coordinate) + 0.3 * s * coordinate

    for radius_0 in RADII:
        scales = _scales(radius_0)
        t = torch.tensor([[0.4e-6]], dtype=DTYPE, requires_grad=True)
        fronts = fronts_at(t, radius_0)
        radius = (0.6 * fronts.contact_radius).detach().requires_grad_(True)

        s = (scales.reference_velocity * t / (radius_0 * TIME_SCALE))
        coordinate = radius / fronts.contact_radius
        s = s.detach().requires_grad_(True)
        coordinate = coordinate.detach().requires_grad_(True)
        own = fronts_at(t, radius_0)
        factors = scales  # 只为可读性；下面显式构造
        from spherical_blast_pinn.physics.moving_frame import product_chain_factors

        factors = product_chain_factors(own, scales, radius)
        value = field(s, coordinate)
        from spherical_blast_pinn.physics.moving_frame import differentiate_scalar

        d_r_explicit, d_t_explicit = factors.physical_derivatives(
            differentiate_scalar(value, s), differentiate_scalar(value, coordinate)
        )

        s2 = scales.reference_velocity * t / (radius_0 * TIME_SCALE)
        value2 = field(s2, radius / fronts_at(t, radius_0).contact_radius)
        d_r_auto = differentiate_scalar(value2, radius)
        d_t_auto = differentiate_scalar(value2, t)

        assert float(d_r_explicit) == pytest.approx(float(d_r_auto), rel=1e-9, abs=1e-12)
        assert float(d_t_explicit) == pytest.approx(float(d_t_auto), rel=1e-9, abs=1e-12)


# ==========================================================================
# 3. 有效帧抽样返回真实帧号
# ==========================================================================


def _write_case(tmp_path: Path, *, radius: float = 0.05, n_frames: int = 24) -> ExtractedCase:
    """一份只有一个后期窗口可监督的最小数据：前若干帧被刻意做成无效。

    这不是装饰。用"位置"代替"帧号"时，被选中的永远是前几帧——而那正是这里
    被剔除的帧。没有前期无效帧的构造，两种实现在这份数据上恰好给出同一批帧，
    测试也就测不到东西。
    """
    # 接触面要有一段**被网格分辨出来的过渡**，而不是一个理想的数学台阶。
    #
    # 这不是为了让探测器好过：真实算例的接触面本来就被数值扩散展宽到几个单元，
    # 而探测器的可靠判据是"接触面两侧相邻单元的密度对比度"，它天然假设过渡是
    # 被分辨的。给一个零宽度的台阶，最陡下降会落到台阶内侧、两侧密度相同，
    # 于是每一帧都被（正确地）判为不可靠——那份夹具就什么都测不到了。
    grid = np.linspace(0.002, 0.30, 3000)
    spacing = float(grid[1] - grid[0])
    transition = 3.0 * spacing
    times = np.linspace(0.0, 3.0e-4, n_frames)
    density = np.empty((n_frames, grid.size))
    pressure = np.empty_like(density)
    velocity = np.zeros_like(density)
    params = mat.product_jwl()
    from spherical_blast_pinn.physics.eos import jwl_pressure_numpy

    for i in range(n_frames):
        expansion = 1.0 + 0.02 * i
        contact = radius * expansion
        shock = contact * 2.0
        # 1 表示产物、0 表示空气的平滑体积分数。
        fraction = 0.5 * (1.0 - np.tanh((grid - contact) / transition))
        air = np.where(grid < shock, 6.0 * mat.AIR_DENSITY, mat.AIR_DENSITY)
        air_p = np.where(grid < shock, 25.0 * mat.AIR_PRESSURE, mat.AIR_PRESSURE)
        density[i] = air + (1600.0 / expansion - air) * fraction
        # 产物压力必须由 JWL 从 (rho, e) 算出。随手写一个"看起来像爆炸"的压力会
        # 让比内能的反解得到负值，于是所有产物标签都变成 nan——而数据读取器
        # （正确地）不会把 nan 当作可用标签，测试就会在"没有产物标签"上失败。
        product_density = 1600.0 / expansion
        product_energy = mat.PRODUCT_SPECIFIC_ENERGY / expansion**3
        product_pressure = float(
            jwl_pressure_numpy(
                np.array([product_density]), np.array([product_energy]), params
            )[0]
        )
        pressure[i] = air_p + (product_pressure - air_p) * fraction
    directory = tmp_path / "case"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "raw_line.npz", t=times, x=grid, rho=density, u=velocity, P=pressure)
    (directory / "metadata.json").write_text(
        json.dumps({"R_0": radius, "x_end": float(grid[-1]), "detonation_velocity_m_s": 6930.0}), encoding="utf-8"
    )
    case = load_extracted_case(directory, case_id="frame-check")
    # 人为把前几帧标为无效——用"位置"的实现会落到它们身上。
    #
    # **两侧前沿都要标无效**：§11.1 之后 ``supervisable_frames`` 是"至少一侧
    # 可靠"，只标 contact 的话这些帧会经由 shock 继续可用，夹具的前提就不成立了
    # （而那不是缺陷，是解耦本身在起作用）。
    case.contact_valid[:4] = False
    case.shock_valid[:4] = False
    return case






# ==========================================================================
# 4. JWL 声速
# ==========================================================================


def test_jwl_sound_speed_matches_between_torch_and_numpy():
    """两条 EOS 路径必须给同一个声速。

    少一个 ``/rho`` 因子时，Torch 版本在产物初值上给出约 50 km/s，而正确值是
    约 4 km/s——一个"看起来像个速度"的数。
    """
    params = mat.product_jwl()
    rho = torch.tensor([[mat.PRODUCT_REFERENCE_DENSITY]], dtype=DTYPE)
    e = torch.tensor([[mat.PRODUCT_SPECIFIC_ENERGY]], dtype=DTYPE)
    torch_value = float(jwl_sound_speed(rho, e, params))
    numpy_value = float(
        jwl_sound_speed_numpy(
            np.array([mat.PRODUCT_REFERENCE_DENSITY]),
            np.array([mat.PRODUCT_SPECIFIC_ENERGY]),
            params,
        )[0]
    )
    assert torch_value == pytest.approx(numpy_value, rel=1e-12)
    # 与热力学关系 c^2 = (dp/drho)_e + p/rho^2 (dp/de)_rho 一致的量级。
    assert 3000.0 < numpy_value < 6000.0
    assert float(jwl_sound_speed_squared(rho, e, params)) == pytest.approx(
        float(jwl_sound_speed_squared_numpy(
            np.array([mat.PRODUCT_REFERENCE_DENSITY]),
            np.array([mat.PRODUCT_SPECIFIC_ENERGY]),
            params,
        )[0]),
        rel=1e-12,
    )
    assert float(jwl_sound_speed_squared(rho, e, params)) == pytest.approx(
        torch_value**2, rel=1e-12
    )


def test_sound_speed_squared_is_not_clamped():
    """无效状态必须能被**检测**出来：裁剪过的 ``c^2`` 永远非负。"""
    params = mat.product_jwl()
    # 极低密度 + 负内能：构造一个 c^2 < 0 的状态。
    rho = torch.tensor([[1.0]], dtype=DTYPE)
    e = torch.tensor([[-1.0e9]], dtype=DTYPE)
    assert float(jwl_sound_speed_squared(rho, e, params)) < 0.0
    assert float(jwl_sound_speed(rho, e, params)) == 0.0


# ==========================================================================
# 5. 物理时间轴
# ==========================================================================


def test_predict_grid_sets_both_clocks_consistently():
    """``physical_time = pinn_time + epsilon(R_0)``，且只加一次。"""
    from spherical_blast_pinn.cli.predict import predict_grid
    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet
    from spherical_blast_pinn.models.product_net import ProductNet

    radius = 0.05
    scales = _scales(radius)
    model = CoupledModel(
        product=ProductNet(scales=scales, width=8, depth=1, seed=0),
        air=AirNet(scales=scales, width=8, depth=1, seed=1),
        front=FrontNet(scales=scales, width=8, depth=1, initial_contact_hat=1.05, seed=2),
        scales=scales,
    )
    physical = np.linspace(scales.start_time(), scales.start_time() + 1.0e-5, 5)
    result = predict_grid(
        model,
        physical_times=physical,
        radii=np.linspace(0.001 * radius, 6.0 * radius, 40),
        scaled_time_coverage=None,
    )
    assert np.allclose(
        result["physical_time"], result["pinn_time"] + handover_time(radius, detonation_velocity=6930.0), rtol=0, atol=1e-18
    )
    assert result["pinn_time"][0] == pytest.approx(0.0, abs=1e-18)


def test_predict_grid_refuses_queries_before_the_time_origin():
    from spherical_blast_pinn.cli.predict import predict_grid
    from spherical_blast_pinn.cli._common import SetupError
    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet
    from spherical_blast_pinn.models.product_net import ProductNet

    scales = _scales(0.05)
    model = CoupledModel(
        product=ProductNet(scales=scales, width=8, depth=1, seed=0),
        air=AirNet(scales=scales, width=8, depth=1, seed=1),
        front=FrontNet(scales=scales, width=8, depth=1, initial_contact_hat=1.05, seed=2),
        scales=scales,
    )
    with pytest.raises(SetupError):
        predict_grid(
            model,
            physical_times=np.array([0.5 * scales.start_time()]),
            radii=np.array([0.1]),
        )








# ==========================================================================
# 6. 新半径的外推判据
# ==========================================================================


def test_scaled_time_coverage_is_the_same_for_every_radius():
    """同样的 ``s``，改变 ``R_0`` 不改变"是否在覆盖内"的判断。

    拿秒数比大小就会违反它：大半径的相似时间更长，50 mm 的秒数上限会把 100 mm
    的覆盖内查询判成外推。
    """
    time_scale = 3.0
    valid = {0.05: 1.0e-4, 0.08: 2.0e-4, 0.10: 3.0e-4}
    s_max, provenance = scaled_time_coverage(valid, time_scale=time_scale)
    assert provenance["radius_independent"] is True

    for radius in RADII:
        inside = physical_time_from_scaled(
                s_max * 0.99, radius, time_scale=time_scale, detonation_velocity=6930.0
            )
        outside = physical_time_from_scaled(
                s_max * 1.01, radius, time_scale=time_scale, detonation_velocity=6930.0
            )
        scales = SimilitudeScales.for_radius(radius, time_scale=time_scale, detonation_velocity=6930.0)
        s_inside = float(scales.network_time(inside - scales.start_time()))
        s_outside = float(scales.network_time(outside - scales.start_time()))
        assert s_inside < s_max
        assert s_outside > s_max

    # 50 mm 的秒数上限不足以覆盖 100 mm 的同等相似时刻：这正是旧判据会误判的点。
    naive_limit = physical_time_from_scaled(
        s_max, 0.05, time_scale=time_scale, detonation_velocity=6930.0
    )
    deep_inside = physical_time_from_scaled(
        0.5 * s_max, 0.10, time_scale=time_scale, detonation_velocity=6930.0
    )
    # 方案 B 下这里是 T_h = R_0/D + epsilon，不是 A 的 epsilon。
    assert deep_inside > naive_limit - handover_time(0.05, detonation_velocity=6930.0)

    # 缺少 hat T_* 或时间范围时给不出覆盖。
    from spherical_blast_pinn.physics.similitude import SimilitudeError

    with pytest.raises(SimilitudeError):
        scaled_time_coverage({}, time_scale=time_scale)


# ==========================================================================
# 7. 数据兼容性闸门
# ==========================================================================


def test_gate_allows_compatible_cases():
    statuses = {"a": CaseCompatibility("a", status=COMPATIBLE)}
    decision = enforce_training_gate(["a"], statuses)
    assert decision.allowed and not decision.unverified


@pytest.mark.parametrize("status", [INCOMPATIBLE, UNKNOWN])
def test_gate_refuses_incompatible_or_unknown_cases(status):
    statuses = {"a": CaseCompatibility("a", status=status, reasons=[f"审查结论：{status}"])}
    with pytest.raises(CompatibilityError):
        enforce_training_gate(["a"], statuses)


def test_gate_treats_a_missing_entry_as_unknown():
    """报告里没有的算例不是"默认通过"。"""
    with pytest.raises(CompatibilityError):
        enforce_training_gate(["never-audited"], {})


def test_gate_records_an_explicit_override():
    statuses = {"a": CaseCompatibility("a", status=INCOMPATIBLE)}
    decision = enforce_training_gate(["a"], statuses, allow_unverified=True)
    assert decision.allowed
    assert decision.unverified
    assert decision.as_dict()["data_compatibility"] == "unverified"


def test_synthetic_mode_refuses_real_cases():
    """合成冒烟路径只能吃标注为合成的数据，不能顺手把真实算例跑掉。"""
    statuses = {"a": CaseCompatibility("a", status=COMPATIBLE)}
    with pytest.raises(CompatibilityError):
        enforce_training_gate(["a"], statuses, require_compatible=False)
    synthetic = {"a": CaseCompatibility("a", status=SYNTHETIC)}
    decision = enforce_training_gate(["a"], synthetic, require_compatible=False)
    assert decision.allowed and decision.mode == "synthetic_smoke"


def test_report_round_trips_through_disk(tmp_path):
    payload = compat.build_report(
        {"a": CaseCompatibility("a", status=INCOMPATIBLE, checks={"time_origin": INCOMPATIBLE})},
        generated_by="test",
    )
    path = compat.write_report(tmp_path / "data_compatibility.json", payload)
    loaded = compat.load_report(path)
    assert loaded["a"].status == INCOMPATIBLE
    assert loaded["a"].checks["time_origin"] == INCOMPATIBLE


def test_heuristic_material_partition_is_not_reported_as_verified():
    """没有体积分数就是没有体积分数：这一项必须是"不一致"，不能被近似成通过。"""

    class _Case:
        source = {"material_masks_from": "density-steepest-descent contact detector (heuristic)"}
        detector = None

    status, evidence = compat.material_partition_verdict(_Case())
    assert status == INCOMPATIBLE
    assert evidence["has_volume_fraction_field"] is False


# ==========================================================================
# 7b. 提取结果的可用性（审查第二轮：标成空气、场量却为空）
# ==========================================================================


def _write_volume_fraction_case(
    tmp_path: Path,
    *,
    air_fields_missing: bool,
    charge_radius: float = 0.05,
) -> Path:
    """写一份带 ``material.npz`` 的提取结果。

    ``air_fields_missing=True`` 复现审查指出的那个失效：壳层被标成空气、空气标签
    计入统计，但空气那一段的 ``rho/u/e/P`` 全是 ``nan``。
    """
    directory = Path(tmp_path) / ("bad" if air_fields_missing else "good")
    directory.mkdir(parents=True, exist_ok=True)
    n_t, n_x = 6, 400
    times = np.linspace(0.0, 1.0e-4, n_t)
    centres = np.linspace(charge_radius / 500.0, 20.0 * charge_radius, n_x)
    rho = np.full((n_t, n_x), np.nan)
    vel = np.full((n_t, n_x), np.nan)
    energy = np.full((n_t, n_x), np.nan)
    pressure = np.full((n_t, n_x), np.nan)
    material = np.full((n_t, n_x), 2, dtype=np.int64)
    alpha = np.zeros((n_t, n_x))
    contact = np.zeros(n_t)
    shock = np.zeros(n_t)
    counts = np.full((n_t, n_x), 1, dtype=np.int64)
    for i in range(n_t):
        contact[i] = charge_radius * (1.0 + 0.5 * i)
        shock[i] = contact[i] * 1.6
        product = centres < contact[i]
        air = (centres >= contact[i]) & (centres < shock[i])
        material[i, product] = 0
        material[i, air] = 1
        alpha[i, product] = 1.0
        rho[i, product] = 1630.0
        vel[i, product] = 100.0
        energy[i, product] = mat.PRODUCT_SPECIFIC_ENERGY
        pressure[i, product] = 5.0e9
        if not air_fields_missing:
            rho[i, air] = 3.0
            vel[i, air] = 200.0
            energy[i, air] = 1.0e5
            pressure[i, air] = 5.0e5
    # 有效性由**标签**推出，而不是由场量推出——这正是审查复现出来的那一档：
    # 空气壳层带着 ``valid=True``，四项场量却全是 ``nan``。
    valid = material != 2
    np.savez(
        directory / "raw_line.npz",
        t=times, x=centres, rho=rho, u=vel, P=pressure, e=energy,
    )
    np.savez(
        directory / "material.npz",
        alpha_product=alpha, material=material, valid=valid, shell_count=counts,
        contact_radius=contact, shock_radius=shock,
        mixed_cell_fraction=np.zeros(n_t), state_index=np.arange(n_t),
    )
    (directory / "metadata.json").write_text(
        json.dumps({"case_id": directory.name, "R_0": charge_radius, "n_t": n_t, "n_x": n_x, "detonation_velocity_m_s": 6930.0}),
        encoding="utf-8",
    )
    return directory


def test_air_labelled_shells_must_carry_finite_fields(tmp_path):
    """**审查复现的那一条。** 空气标签存在、空气状态为空，必须判为不兼容。

    只看来源说明的实现会在这里放行：来源写着"体积分数"，而空气监督其实是空的。
    采样器会把那些 ``nan`` 逐个丢掉，数据损失里的空气组随之静默消失。
    """
    good = load_extracted_case(
        _write_volume_fraction_case(tmp_path, air_fields_missing=False), case_id="good"
    )
    status, evidence = compat.material_partition_verdict(good)
    assert status == COMPATIBLE, evidence["problems"]
    assert evidence["air_shells"] > 0
    assert evidence["air_shells_with_finite_fields"] == evidence["air_shells"]
    assert evidence["valid_shells_without_finite_fields"] == 0

    bad = load_extracted_case(
        _write_volume_fraction_case(tmp_path, air_fields_missing=True), case_id="bad"
    )
    status, evidence = compat.material_partition_verdict(bad)
    assert status == INCOMPATIBLE
    assert evidence["air_shells"] > 0, "夹具没有造出空气壳层"
    assert evidence["air_shells_with_finite_fields"] == 0
    assert evidence["valid_shells_without_finite_fields"] > 0, (
        "夹具没有复现出‘标成有效却没有状态’这一档"
    )
    assert evidence["problems"], "空气标签没有状态却没有任何一条问题"


def test_partition_verdict_checks_the_result_not_the_description(tmp_path):
    """两份完全不同质量的导出，来源说明一字不差；结论必须不同。"""
    good = load_extracted_case(
        _write_volume_fraction_case(tmp_path, air_fields_missing=False), case_id="good"
    )
    bad = load_extracted_case(
        _write_volume_fraction_case(tmp_path, air_fields_missing=True), case_id="bad"
    )
    assert good.source["material_masks_from"] == bad.source["material_masks_from"]
    assert compat.material_partition_verdict(good)[0] == COMPATIBLE
    assert compat.material_partition_verdict(bad)[0] == INCOMPATIBLE


def test_volume_fraction_alpha_is_read_not_inverted(tmp_path):
    """产物侧体积分数必须直接来自 ``alpha``，不能取补。

    取补会把产物与空气整个对调，而结果看起来只是"分区反了"，很像探测器不准。
    """
    case = load_extracted_case(
        _write_volume_fraction_case(tmp_path, air_fields_missing=False), case_id="good"
    )
    product = case.material == 0
    assert np.all(case.density[product] > 1000.0)
    assert np.all(case.density[case.material == 1] < 100.0)


def test_mixture_fraction_profile_is_defined_where_pure_product_is_absent():
    """分类必须用**全部单元**的加权体积分数。

    只对纯产物单元求平均的那个 alpha 在没有产物的壳层上返回 ``nan``，而不是 0；
    ``nan`` 既不 ``>= 0.95`` 也不 ``<= 0.05``，于是空气壳层会整个掉进"未知"。
    """
    from spherical_blast_pinn.data.raw_reader import RawFrame
    from spherical_blast_pinn.data.shell_average import (
        ShellGrid,
        mixture_fraction_profile,
        shell_average,
    )

    n = 60
    radius = np.linspace(0.001, 0.30, n)
    alpha = np.where(radius < 0.05, 1.0, 0.0)
    frame = RawFrame(
        case_id="unit",
        time=0.0,
        radius=radius,
        cell_volume=np.full(n, 1.0e-9),
        density=np.where(alpha > 0.5, 1630.0, 1.225),
        velocity_x=np.zeros(n),
        velocity_y=np.zeros(n),
        velocity_z=np.zeros(n),
        pressure=np.where(alpha > 0.5, 5.0e9, 101325.0),
        tnt_volume_fraction=alpha,
        specific_internal_energy=np.where(alpha > 0.5, 2.0e6, 2.0e5),
        material_id=np.zeros(n, dtype=np.int64),
        cell_centroid=np.column_stack([radius, np.zeros(n), np.zeros(n)]),
    ).with_radial_velocity()

    grid = ShellGrid.linear(0.001, 0.30, 40)
    mixture = mixture_fraction_profile(frame, grid)
    product_only = shell_average(frame, grid, "product").volume_fraction

    air_shells = np.asarray(mixture.alpha) <= 0.05
    assert np.any(air_shells), "夹具没有造出空气壳层"
    assert np.all(np.isfinite(mixture.alpha)), "全部单元加权的结果必须处处有定义"
    assert np.all(np.isnan(product_only[air_shells])), (
        "只对纯产物单元求平均时，空气壳层上应当是 nan——正因为如此它不能用来分类"
    )
    # 接触面必须能从 mixture alpha 上定出来。
    crossing = np.flatnonzero(np.asarray(mixture.alpha) < 0.5)
    assert crossing.size and grid.centres[crossing[0]] > 0.05


def test_fixed_mesh_geometry_hoists_only_when_the_mesh_really_is_fixed():
    """ALE 网格固定时单元几何是常量，可以先算一次。

    每帧重算 122 万个六面体的体积是整条提取里最贵的一步，而且算的是同一个数组。
    但把它提出来**只有**在网格真的不动时才成立：一份拉格朗日算例被这样处理，
    每个半径都会冻在初始值上，而剖面看起来仍然像那么回事。
    """
    from spherical_blast_pinn.data.raw_reader import (
        POSITION_IS_ABSOLUTE,
        POSITION_IS_DISPLACEMENT,
        D3plotCase,
        D3plotLayout,
        RawReadError,
        fixed_mesh_geometry,
    )

    # 两个单位立方体，手工可算。
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        ]
    )
    connectivity = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])

    def _case(convention):
        return D3plotCase(
            directory=Path("."),
            coordinates=coordinates,
            connectivity=connectivity,
            part_indexes=np.array([0, 0]),
            element_ids=np.array([1, 2]),
            part_titles=["a"],
            ale_material_ids=np.array([1]),
            n_states=2,
            layout=D3plotLayout(),
            product_ale_indices=(1,),
            air_ale_indices=(2,),
            position_convention=convention,
        )

    geometry = fixed_mesh_geometry(_case(POSITION_IS_ABSOLUTE))
    assert np.allclose(geometry.volume, 1.0, rtol=1e-12)
    assert np.allclose(geometry.centroid[0], [0.5, 0.5, 0.5], rtol=1e-12)
    assert np.allclose(geometry.centroid[1], [1.5, 0.5, 0.5], rtol=1e-12)
    assert np.allclose(geometry.radius, np.linalg.norm(geometry.centroid, axis=1), rtol=1e-12)

    with pytest.raises(RawReadError):
        fixed_mesh_geometry(_case(POSITION_IS_DISPLACEMENT))


# ==========================================================================
# 8. 固定独立检查集
# ==========================================================================




def test_frame_split_holds_out_whole_frame_groups(tmp_path):
    case = _write_case(tmp_path)
    train, dev = case.frame_split(holdout_fraction=0.1, seed=0)
    assert dev.size > 0
    assert set(train.tolist()) & set(dev.tolist()) == set()
    assert set(train.tolist()) | set(dev.tolist()) == set(
        np.flatnonzero(case.supervisable_frames()).tolist()
    )
    # 留出的必须是**连续**的一段，而不是零散的点。
    ordered = np.sort(dev)
    assert np.all(np.diff(ordered) == 1)


# ==========================================================================
# 9. 前沿构造与 ``R_c < R_0``
# ==========================================================================


def test_front_default_construction_is_the_literal_one():
    """默认必须是提示词 §7 的 ``hat_Rc = a softplus(z_c)``，不是 ``1 + ...``。"""
    from spherical_blast_pinn.models.front_net import FrontNet

    net = FrontNet(
        scales=_scales(0.05), width=8, depth=1, initial_contact_hat=1.08,
        initial_gap_hat=0.02, seed=0,
    )
    assert net.anchor_contact_at_charge_surface is False
    assert net.contact_scale == pytest.approx(1.08, rel=1e-15)
    value = float(net.positions(torch.zeros((1, 1), dtype=DTYPE)).contact_hat)
    assert value == pytest.approx(1.08, rel=1e-9)
    assert value != pytest.approx(2.08, rel=1e-9)


def test_anchor_is_opt_in_and_has_a_window():
    """开启锚定时，偏移只在 ``[0, until_s]`` 上存在，之后回到字面构造。"""
    from spherical_blast_pinn.models.front_net import FrontNet

    net = FrontNet(
        scales=_scales(0.05), width=8, depth=1, initial_contact_hat=1.08,
        initial_gap_hat=0.02, anchor_contact_at_charge_surface=True,
        anchor_until_s=0.5, seed=0,
    )
    assert net.contact_scale == pytest.approx(0.08, rel=1e-15)
    at_zero = float(net.positions(torch.zeros((1, 1), dtype=DTYPE)).contact_hat)
    assert at_zero == pytest.approx(1.08, rel=1e-9)

    # 窗口之外必须回到字面的 ``a softplus(z_c)``：把网络原始输出取出来，按定义
    # 自己算一遍。两档的参数化不同（``contact_scale`` 一个是 0.08、一个是 1.08），
    # 所以不能拿另一个网络来对照。
    beyond = torch.full((4, 1), 1.5, dtype=DTYPE)
    raw = net.mlp(beyond)
    expected = 0.08 * torch.nn.functional.softplus(raw[:, 0:1])
    assert torch.allclose(net.positions(beyond).contact_hat, expected, rtol=1e-12)
    # 窗口之内则带有那个偏移。
    inside = torch.zeros((1, 1), dtype=DTYPE)
    raw_inside = net.mlp(inside)
    assert float(net.positions(inside).contact_hat) == pytest.approx(
        1.0 + 0.08 * float(torch.nn.functional.softplus(raw_inside[0, 0])), rel=1e-12
    )


def test_anchor_is_configurable_through_build_model():
    from spherical_blast_pinn.models.coupled_model import resolve_front_anchor

    class _Config:
        def __init__(self, payload):
            self.payload = payload
            self.sections = {"networks.front": payload}

        def section(self, name):
            return self.sections.get(name, {})

        def get(self, key, default=None):
            return self.payload.get(key, default)

        def list_of(self, key, default=None):
            return default or []

    assert resolve_front_anchor(_Config({})) == (False, None)
    assert resolve_front_anchor(_Config({"contact_anchor": True})) == (True, None)
    enabled, until = resolve_front_anchor(
        _Config({"contact_anchor": {"enabled": True, "until_s": 0.7}})
    )
    assert enabled and until == pytest.approx(0.7)




# ==========================================================================
# 10. 损失分组与随机状态
# ==========================================================================












# ==========================================================================
# 11. 对照指标
# ==========================================================================






















# ==========================================================================
# §11.1 逐字段独立性：一个字段缺失不丢掉其他可靠字段
# ==========================================================================






# ==========================================================================
# §11.2 L_geometry：位置约束
# ==========================================================================














# ==========================================================================
# §12 平均观测算子
# ==========================================================================




