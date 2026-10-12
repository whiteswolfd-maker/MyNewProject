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
from spherical_blast_pinn.physics.detonation import DetonationConfig
from spherical_blast_pinn.physics.precursor import PrecursorConfig, solve_reference_field
from spherical_blast_pinn.physics.similitude import (
    SimilitudeScales,
    physical_time_from_scaled,
    scaled_time_coverage,
    handover_time,
)
from spherical_blast_pinn.sampling.sampler import RadiusSampler, SamplingConfig

DTYPE = torch.float64
TIME_SCALE = 2.0
RADII = (0.05, 0.08, 0.10)


def _scales(radius: float) -> SimilitudeScales:
    return SimilitudeScales.for_radius(radius, time_scale=TIME_SCALE, detonation_velocity=6930.0)


def _tiny_reference(radius: float = 0.05):
    return solve_reference_field(
        charge_radius=radius,
        config=PrecursorConfig(cells_inside=120, growth=1.05, cfl=0.3),
        # 方案 B：参考场从 T_b = R_0/D 起算，只积分 epsilon 段。sample_times 是
        # **相对交接时刻**的时间，tau = 0（交接时刻）必须在其中。
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
        detonation_config=DetonationConfig(cells=120),
        sample_times=[0.0, 1.0e-6, 4.0e-6],
    )


# ==========================================================================
# 1. 参考初始流场的空间坐标
# ==========================================================================


def test_reference_field_queries_the_current_lagrangian_position():
    """在**当前**单元中心查询必须返回该单元自己的状态。

    用初始单元中心当空间坐标时，这个检查会给出另一个数，而那个数看起来同样
    合理：50 mm、1 µs 时最后一个产物单元的初始中心是 49.94 mm、当前中心是
    53.12 mm，两者读出来的密度相差一个数量级。
    """
    field = _tiny_reference()
    frame = field.times.size - 1
    centres = field.cell_radii[frame]
    density, velocity, energy = field.sample(
        centres, np.full_like(centres, field.times[frame])
    )
    assert np.max(np.abs(density - field.density[frame]) / field.density[frame]) < 1.0e-12
    assert np.max(np.abs(energy - field.specific_energy[frame]) / field.specific_energy[frame]) < 1.0e-12
    assert np.max(np.abs(velocity - field.velocity[frame])) < 1.0e-9


def test_reference_field_coordinates_actually_move():
    """前提检查：拉格朗日网格真的在动，所以"用初始坐标"确实是个错误。"""
    field = _tiny_reference()
    displacement = np.max(np.abs(field.cell_radii[-1] - field.cell_radii[0]))
    assert displacement > 1.0e-3 * field.reference_radius


def test_reference_field_never_interpolates_across_the_contact():
    """接触面两侧的状态必须各自取单侧值，不能插成一个中间状态。

    判据直接对着两侧的**单元**：紧贴接触面内侧的查询必须复现最后一个产物单元的
    状态，外侧的必须复现第一个空气单元的状态。跨材料插值会给出两者的加权平均，
    既不是产物也不是空气。
    """
    field = _tiny_reference()
    frame = field.times.size - 1
    contact = float(field.contact_radius[frame])
    cut = field.n_product_cells
    last_product = float(field.density[frame, cut - 1])
    first_air = float(field.density[frame, cut])
    assert last_product > 3.0 * first_air  # 台阶确实存在

    inside = float(
        field.sample(np.array([contact * (1.0 - 1.0e-12)]), np.array([field.times[frame]]))[0][0]
    )
    outside = float(
        field.sample(np.array([contact * (1.0 + 1.0e-12)]), np.array([field.times[frame]]))[0][0]
    )
    assert inside == pytest.approx(last_product, rel=1e-9)
    assert outside == pytest.approx(first_air, rel=1e-9)
    # 混合值会落在两者之间，与任一侧都不接近。
    assert abs(outside - first_air) < 0.05 * abs(outside - last_product)


def test_reference_field_does_not_blend_materials_in_time():
    """固定半径被接触面扫过时，两个时间样本不是同一个场的采样，不得线性混合。

    取一个会被接触面扫过的半径，在两帧之间查询，结果必须**等于**其中某一帧的
    值，而不是它们的加权平均。
    """
    field = _tiny_reference()
    radius = np.array([float(field.contact_radius[0]) * 1.05])
    t0, t1 = float(field.times[0]), float(field.times[1])
    midpoint = 0.5 * (t0 + t1)
    blended = float(field.sample(radius, np.array([midpoint]))[0][0])
    at_lower = float(field.sample(radius, np.array([t0]))[0][0])
    at_upper = float(field.sample(radius, np.array([t1]))[0][0])
    assert min(abs(blended - at_lower), abs(blended - at_upper)) < 1.0e-9 * max(
        abs(at_lower), abs(at_upper)
    )
    assert field.query_diagnostics["last_query_crossing_material_fraction"] == pytest.approx(0.0)


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


def test_sampler_returns_real_frame_numbers(tmp_path):
    case = _write_case(tmp_path)
    usable = np.flatnonzero(case.supervisable_frames())
    assert usable.size > 4
    assert not np.any(np.isin(usable, np.arange(4)))

    sampler = RadiusSampler(
        case=case,
        scales=_scales(case.charge_radius),
        reference=_tiny_reference(),
        config=SamplingConfig(
            n_pde_product=8, n_pde_air=8, n_ic_product=4, n_ic_air=4,
            n_contact=2, n_shock=2, n_data_product=12, n_data_air=12, n_data_front=6,
        ),
        seed=0,
    )
    batch = sampler.batch()
    sampled = set(int(f) for f in batch.diagnostics["sampled_frames"])
    assert sampled, "没有抽到任何帧"
    assert sampled <= set(int(f) for f in usable), (
        "抽样返回了不在有效集合里的帧号——位置被当成了帧号"
    )
    for frame in sampled:
        assert case.times[frame] >= case.start_time - 1e-15


def test_front_supervision_uses_the_same_frame_mapping(tmp_path):
    case = _write_case(tmp_path)
    usable = set(int(f) for f in np.flatnonzero(case.supervisable_frames()))
    sampler = RadiusSampler(
        case=case,
        scales=_scales(case.charge_radius),
        reference=_tiny_reference(),
        config=SamplingConfig(
            n_pde_product=4, n_pde_air=4, n_ic_product=2, n_ic_air=2,
            n_contact=2, n_shock=2, n_data_product=4, n_data_air=4, n_data_front=6,
        ),
        seed=1,
    )
    batch = sampler.batch()
    # §11.1：contact 与 shock **独立**，不再有共用的 ``s_data_front``。
    assert batch.s_data_contact is not None
    assert batch.s_data_shock is not None
    # 前沿标签的时间必须落在**各自**有效帧的真实时间上：接触面落在
    # contact_valid 帧，激波落在 shock_valid 帧，互不要求对方。
    def physical_time(value):
        return (
            float(value) * TIME_SCALE * case.charge_radius / mat.REFERENCE_VELOCITY
            + case.start_time
        )

    contact_frames = set(int(f) for f in np.flatnonzero(case.contact_valid))
    contact_times = {float(case.times[f]) for f in contact_frames}
    for value in batch.s_data_contact.reshape(-1).tolist():
        assert min(abs(physical_time(value) - t) for t in contact_times) < 1.0e-12

    shock_frames = set(int(f) for f in np.flatnonzero(case.shock_valid))
    shock_times = {float(case.times[f]) for f in shock_frames}
    for value in batch.s_data_shock.reshape(-1).tolist():
        assert min(abs(physical_time(value) - t) for t in shock_times) < 1.0e-12


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


class _TrajectoryCase:
    """只提供 ``compare_trajectories`` 需要的字段。

    写成模块级类而不是函数内的类：类体**不**参与外层函数的闭包，在类体里写
    ``times = times`` 会得到一个 ``NameError``。
    """

    def __init__(self, *, charge_radius, times, contact_radius, shock_radius, start):
        self.charge_radius = charge_radius
        self.times = times
        self.contact_radius = contact_radius
        self.shock_radius = shock_radius
        self.start_time = start

    def supervisable_frames(self):
        return np.ones(self.times.size, dtype=bool)


def test_compare_does_not_shift_the_prediction_a_second_time():
    """同一组轨迹自比较必须精确为零。

    预测已经带过一次 ``epsilon``；对照里再加一次，会报出约 6%/8% 的归一化 RMS
    ——那不是模型误差，是符号约定错误。
    """
    from spherical_blast_pinn.cli.compare import compare_trajectories

    radius = 0.05
    start = handover_time(radius, detonation_velocity=6930.0)
    times = np.linspace(start + 1.0e-6, start + 2.0e-4, 60)
    contact = radius + 3000.0 * (times - start)
    shock = radius + 4000.0 * (times - start)

    case = _TrajectoryCase(
        charge_radius=radius,
        times=times,
        contact_radius=contact,
        shock_radius=shock,
        start=start,
    )
    prediction = {
        "start_time": start,
        "physical_time": times,
        "pinn_time": times - start,
        "contact_radius": contact,
        "shock_radius": shock,
    }
    report = compare_trajectories(prediction, case, ambient=mat.AIR_PRESSURE)
    assert report["available"]
    assert report["contact_relative_rms"] == pytest.approx(0.0, abs=1e-12)
    assert report["shock_relative_rms"] == pytest.approx(0.0, abs=1e-12)


def test_field_lookup_never_extrapolates_in_time():
    from spherical_blast_pinn.cli.compare import (
        SAMPLE_TIME_OUTSIDE,
        _time_bracket,
        sample_field,
    )

    times = np.array([0.0, 1.0, 2.0])
    assert _time_bracket(times, 5.0) is None
    assert _time_bracket(times, -1.0) is None
    bracket = _time_bracket(times, 1.5)
    assert bracket is not None and 0.0 <= bracket[1] <= 1.0

    prediction = {
        "pinn_time": times,
        "physical_time": times,
        "start_time": 0.0,
        "radius": np.array([0.1, 0.2, 0.3]),
        "density": np.full((3, 3), 1.0),
        "region": np.zeros((3, 3), dtype=np.int64),
        # 区域由**前沿**给出（§7.4），不再读 ``region`` 数组本身。
        "contact_radius": np.array([0.15, 0.15, 0.15]),
        "shock_radius": np.array([0.25, 0.25, 0.25]),
    }
    out, why = sample_field(prediction, "density", np.array([0.15]), physical_time=9.0)
    assert not np.isfinite(out[0])
    # 原因要**说清楚是域外**，不能只是"给了一个 nan"。
    assert why[0] == SAMPLE_TIME_OUTSIDE


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


def test_check_set_is_deterministic_and_separate_from_the_training_sampler(tmp_path):
    from spherical_blast_pinn.evaluation.checks import build_check_set

    case = _write_case(tmp_path)
    radius = case.charge_radius
    reference = _tiny_reference()
    scales_by_radius = {radius: _scales(radius)}
    config = SamplingConfig(
        n_pde_product=16, n_pde_air=16, n_ic_product=8, n_ic_air=8,
        n_contact=4, n_shock=4, n_data_product=8, n_data_air=8, n_data_front=4,
    )

    first = build_check_set(
        cases=[case], scales_by_radius=scales_by_radius, reference=reference,
        config=config, seed=11,
    )
    second = build_check_set(
        cases=[case], scales_by_radius=scales_by_radius, reference=reference,
        config=config, seed=11,
    )
    third = build_check_set(
        cases=[case], scales_by_radius=scales_by_radius, reference=reference,
        config=config, seed=12,
    )
    a = first.batches[0].s_product.reshape(-1)
    assert torch.equal(a, second.batches[0].s_product.reshape(-1))
    assert not torch.equal(a, third.batches[0].s_product.reshape(-1))

    # 用的是**开发帧**：与训练采样器的帧集合不同，否则"开发集误差"只是同一批
    # 点的另一次采样。
    train = RadiusSampler(
        case=case, scales=_scales(radius), reference=reference, config=config, seed=11
    )
    assert set(train.usable_frames.tolist()) != set(
        first.batches[0].diagnostics["sampled_frames"]
    )
    assert first.batches[0].diagnostics["data_split"] == "development"
    assert first.definition["selection_criteria"]


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


def test_pde_path_does_not_abort_when_the_contact_dips_below_the_charge_radius(tmp_path):
    """字面构造下 ``R_c`` 可以暂时低于 ``R_0``；那不是一个非法移动域。

    全时段硬失败会把"候选轨迹还没满足初值"当成"移动域不合法"，并且即使关掉
    前沿锚定也照样阻断运行。
    """
    from spherical_blast_pinn.losses.terms import medium_pde_residual
    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet

    case = _write_case(tmp_path)
    radius = case.charge_radius
    scales = _scales(radius)

    from spherical_blast_pinn.models.product_net import ProductNet

    model = CoupledModel(
        product=ProductNet(scales=scales, width=8, depth=1, seed=0),
        air=AirNet(scales=scales, width=8, depth=1, seed=1),
        # 把接触面刻意压到装药半径以内。
        front=FrontNet(scales=scales, width=8, depth=1, initial_contact_hat=0.8, seed=2),
        scales=scales,
    )
    sampler = RadiusSampler(
        case=case,
        scales=scales,
        reference=_tiny_reference(),
        config=SamplingConfig(
            n_pde_product=8, n_pde_air=8, n_ic_product=4, n_ic_air=4,
            n_contact=2, n_shock=2, n_data_product=4, n_data_air=4, n_data_front=2,
        ),
        seed=0,
    )
    batch = sampler.batch()
    residual = medium_pde_residual(batch, model, PRODUCT)
    assert residual is not None
    assert residual.n_points > 0
    mass, _, _ = residual.scaled()
    assert bool(torch.isfinite(mass).all())


# ==========================================================================
# 10. 损失分组与随机状态
# ==========================================================================


def test_data_loss_has_three_equal_groups(tmp_path):
    """产物、空气、前沿三组等权；前沿组内再平均 Rc/Rs。

    把 Rc/Rs 拆成两组参与最外层平均，前沿就占 1/2 而不是 1/3。
    """
    from spherical_blast_pinn.losses.terms import data_loss
    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet
    from spherical_blast_pinn.models.product_net import ProductNet

    case = _write_case(tmp_path)
    radius = case.charge_radius
    scales = _scales(radius)
    model = CoupledModel(
        product=ProductNet(scales=scales, width=8, depth=1, seed=0),
        air=AirNet(scales=scales, width=8, depth=1, seed=1),
        front=FrontNet(scales=scales, width=8, depth=1, initial_contact_hat=1.1, seed=2),
        scales=scales,
    )
    sampler = RadiusSampler(
        case=case, scales=scales, reference=_tiny_reference(),
        config=SamplingConfig(
            n_pde_product=4, n_pde_air=4, n_ic_product=2, n_ic_air=2,
            n_contact=2, n_shock=2, n_data_product=12, n_data_air=12, n_data_front=6,
        ),
        seed=0,
    )
    breakdown = data_loss(sampler.batch(), model)
    assert breakdown.monitors["data_n_groups"] == 3
    manual = (
        float(breakdown.parts["data_product"])
        + float(breakdown.parts["data_air"])
        + float(breakdown.parts["data_front"])
    ) / 3.0
    assert float(breakdown.total) == pytest.approx(manual, rel=1e-12)
    assert float(breakdown.parts["data_front"]) == pytest.approx(
        (
            float(breakdown.parts["data_front_contact"])
            + float(breakdown.parts["data_front_shock"])
        )
        / 2.0,
        rel=1e-12,
    )


def test_sampler_rng_state_round_trips(tmp_path):
    """中断后恢复到第 N 步必须与连续跑 N 步给出同一批样本。"""
    case = _write_case(tmp_path)
    config = SamplingConfig(
        n_pde_product=8, n_pde_air=8, n_ic_product=4, n_ic_air=4,
        n_contact=2, n_shock=2, n_data_product=8, n_data_air=8, n_data_front=4,
    )

    def _make():
        return RadiusSampler(
            case=case, scales=_scales(case.charge_radius),
            reference=_tiny_reference(), config=config, seed=7,
        )

    continuous = _make()
    sequence = [
        continuous.batch().s_product.reshape(-1).clone() for _ in range(4)
    ]

    interrupted = _make()
    for _ in range(2):
        interrupted.batch()
    state = interrupted.rng_state()
    payload = json.loads(json.dumps({"seed": state["seed"], "bit_generator": None}))
    # ``bit_generator.state`` 里含 numpy int，走一趟 JSON 确认它可序列化。
    payload["bit_generator"] = json.loads(json.dumps(state["bit_generator"], default=int))
    resumed = _make()
    resumed.set_rng_state(payload)
    assert torch.equal(
        resumed.batch().s_product.reshape(-1), sequence[2].reshape(-1)
    ), "恢复后的采样轨迹与连续轨迹不一致"


def _tiny_trainer(tmp_path: Path, output: Path):
    """一个两三个半径都嫌大、一个半径够用的最小 Trainer。"""
    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet
    from spherical_blast_pinn.models.product_net import ProductNet
    from spherical_blast_pinn.training.trainer import Trainer

    case = _write_case(tmp_path)
    radius = case.charge_radius
    scales = _scales(radius)
    reference = _tiny_reference()
    model = CoupledModel(
        product=ProductNet(scales=scales, width=8, depth=1, seed=0),
        air=AirNet(scales=scales, width=8, depth=1, seed=1),
        front=FrontNet(scales=scales, width=8, depth=1, initial_contact_hat=1.1, seed=2),
        scales=scales,
    )

    class _Config:
        def section(self, name):
            if name == "sampling":
                return {
                    "n_pde_product": 8, "n_pde_air": 8, "n_ic_product": 4,
                    "n_ic_air": 4, "n_contact": 2, "n_shock": 2,
                    "n_data_product": 8, "n_data_air": 8, "n_data_front": 4,
                }
            return {}

        def get(self, key, default=None):
            return default

        def path_of(self, key, default=None):
            return default

    config = _Config()
    return Trainer(
        config=config,
        cases=[case],
        reference=reference,
        model=model,
        scales_by_radius={radius: scales},
        time_scale=TIME_SCALE,
        seed=3,
        output_dir=output,
        log_every=1000,
        checkpoint_every=1000,
        check_every=4,
        compatibility={
            case.case_id: CaseCompatibility(case.case_id, status=INCOMPATIBLE)
        },
        allow_unverified_data=True,
    )


def test_resuming_reproduces_a_continuous_run(tmp_path):
    """中断后恢复到第 N 步必须与连续跑 N 步给出**同一个**模型。

    只保存全局 NumPy 状态是不够的：采样器用的是自己建的 ``default_rng()``，
    它不在 ``np.random.get_state()`` 里。
    """
    continuous = _tiny_trainer(tmp_path / "a", tmp_path / "out_a")
    continuous.run(max_updates=8)
    resumed = _tiny_trainer(tmp_path / "b", tmp_path / "out_b")
    resumed.run(max_updates=4)
    resumed.load(tmp_path / "out_b" / "last.pt")
    resumed.run(max_updates=8)

    assert continuous.step == resumed.step == 8
    for (name_a, param_a), (name_b, param_b) in zip(
        continuous.model.state_dict().items(), resumed.model.state_dict().items()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b), name_a
    assert continuous.check_history[-1]["step"] == resumed.check_history[-1]["step"]
    assert continuous.check_history[-1]["metrics"]["pde_rms_max"] == pytest.approx(
        resumed.check_history[-1]["metrics"]["pde_rms_max"], rel=1e-12
    )


def test_a_checkpoint_without_sampler_rng_is_refused(tmp_path):
    """旧检查点（只有全局 RNG）必须拒绝恢复，而不是安静地换一条采样轨迹。"""
    from spherical_blast_pinn.training.trainer import TrainingError

    trainer = _tiny_trainer(tmp_path / "c", tmp_path / "out_c")
    trainer.run(max_updates=2)
    path = tmp_path / "out_c" / "last.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("sampler_rng")
    torch.save(payload, path)
    with pytest.raises(TrainingError):
        trainer.load(path)


# ==========================================================================
# 11. 对照指标
# ==========================================================================


def _synthetic_prediction(radius: float = 0.05, n_t: int = 12, n_r: int = 120):
    """一份自洽的预测字典：前沿是解析的，场量按区域给常数。"""
    start = handover_time(radius, detonation_velocity=6930.0)
    pinn_times = np.linspace(0.0, 8.0e-5, n_t)
    radii = np.linspace(0.002, 0.35, n_r)
    contact = radius + 3000.0 * pinn_times
    shock = radius * 1.2 + 4000.0 * pinn_times
    region = np.where(
        radii[None, :] < contact[:, None],
        0,
        np.where(radii[None, :] < shock[:, None], 1, 2),
    ).astype(np.int64)
    return {
        "start_time": start,
        "physical_time": pinn_times + start,
        "pinn_time": pinn_times,
        "radius": radii,
        "density": np.where(region == 0, 1600.0, np.where(region == 1, 6.0, 1.225)),
        "velocity": np.zeros((n_t, n_r)),
        "specific_energy": np.where(region == 0, 2.0e6, 2.0e5),
        "pressure": np.where(region == 0, 5.0e9, np.where(region == 1, 2.0e6, 101325.0)),
        "region": region,
        "contact_radius": contact,
        "shock_radius": shock,
        "scale_product_density": 1630.0,
        "scale_air_density": 1.225,
        "scale_velocity": mat.REFERENCE_VELOCITY,
        "scale_specific_energy": mat.PRODUCT_SPECIFIC_ENERGY,
        "ambient_pressure": mat.AIR_PRESSURE,
    }


def _synthetic_reference_case(radius: float = 0.05, *, ambient_label: bool = True):
    """与上面那份预测一致的参考数据；``ambient_label`` 决定是否带标签 2。"""
    prediction = _synthetic_prediction(radius)
    times = prediction["physical_time"].copy()
    radii = prediction["radius"].copy()
    n_t, n_r = times.size, radii.size
    material = np.zeros((n_t, n_r), dtype=np.int64)
    density = np.empty((n_t, n_r))
    pressure = np.empty((n_t, n_r))
    for i in range(n_t):
        contact = prediction["contact_radius"][i]
        shock = prediction["shock_radius"][i]
        product = radii < contact
        air = (radii >= contact) & (radii < shock)
        material[i] = np.where(product, 0, np.where(air, 1, 2))
        density[i] = np.where(product, 1600.0, np.where(air, 6.0, 1.225))
        pressure[i] = np.where(product, 5.0e9, np.where(air, 2.0e6, 101325.0))
    energy = np.where(
        material == 0, 2.0e6, np.where(material == 1, 2.0e5, mat.AIR_SPECIFIC_ENERGY)
    )
    contact_radius = prediction["contact_radius"].copy()
    shock_radius = prediction["shock_radius"].copy()
    valid = np.ones((n_t, n_r), dtype=bool)
    if not ambient_label:
        # 把所有环境单元改成"空气"，检验对照不会把标签 2 当成空气。
        material = np.where(material == 2, 1, material)
        density = np.where(material == 1, 6.0, density)
        pressure = np.where(material == 1, 2.0e6, pressure)
        energy = np.where(material == 1, 2.0e5, energy)
    return ExtractedCase(
        case_id="synthetic-compare",
        charge_radius=radius,
        times=times,
        radius=radii,
        density=density,
        velocity=np.zeros((n_t, n_r)),
        pressure=pressure,
        specific_energy=energy,
        contact_radius=contact_radius,
        shock_radius=shock_radius,
        contact_valid=np.ones(n_t, dtype=bool),
        shock_valid=np.ones(n_t, dtype=bool),
        material=material,
        valid=valid,
        # 方案 B 的时间原点需要卡片爆速；合成算例也要给出，否则读取端会按设计拒绝。
        source={"detonation_velocity_m_s": 6930.0},
    )


def test_compare_fields_self_consistency_is_zero():
    """预测与参考完全一致时，每个介质的相对 L2 都应当是零。"""
    from spherical_blast_pinn.cli.compare import compare_fields

    prediction = _synthetic_prediction()
    case = _synthetic_reference_case()
    report = compare_fields(prediction, case)
    assert report["available"]
    for medium in ("product", "air"):
        for name in ("density", "velocity", "specific_energy"):
            key = f"{medium}_{name}_relative_l2"
            assert key in report, f"{key} 没有评价"
            assert report[key] == pytest.approx(0.0, abs=1e-12)


def test_compare_fields_does_not_count_ambient_as_air():
    """标签 2（未受扰动）不是空气；把它当空气会制造出纯粹的假错配。"""
    from spherical_blast_pinn.cli.compare import compare_fields

    prediction = _synthetic_prediction()
    case = _synthetic_reference_case()
    report = compare_fields(prediction, case)
    assert report["region_mismatch_cells"]["ambient"] == 0
    assert report["ambient_cells_excluded"] is True


def test_compare_fields_keeps_mismatched_material_out_of_the_error():
    """预测前沿错位时，材料不匹配的单元必须单独报，不能混进光滑区误差。

    **错位必须是物理上合法的错位。** 以前这里把预测的激波乘 0.6 推到接触面
    以内，于是 ``R_s < R_c``——一个预测不可能给出的前沿组合。区域判定现在对这种
    组合明确返回"不可用"（§7.4），而不是照着一个不可能的前沿去贴标签，所以这个
    反例改成把**接触面**往里推：``R_c`` 变小、``R_s`` 不变，前沿顺序仍然合法，
    参考里的一段产物落在预测的空气区里。
    """
    from spherical_blast_pinn.cli.compare import compare_fields

    prediction = _synthetic_prediction()
    prediction["contact_radius"] = prediction["contact_radius"] * 0.6
    region = np.where(
        prediction["radius"][None, :] < prediction["contact_radius"][:, None],
        0,
        np.where(
            prediction["radius"][None, :] < prediction["shock_radius"][:, None], 1, 2
        ),
    ).astype(np.int64)
    prediction["region"] = region

    case = _synthetic_reference_case()
    report = compare_fields(prediction, case)
    assert report["region_mismatch_total"] > 0, "构造的错位没有产生任何不匹配单元"
    # 产物区里那些"参考说是产物、预测说是空气"的点不应该进入任何一个介质的误差。
    assert report["compared_cells"] > 0
    assert report["region_mismatch_cells"]["product"] > 0


def test_a_front_order_violation_makes_the_region_unavailable_not_a_label():
    """``R_s < R_c`` 是预测不可能给出的前沿组合。照它贴标签会把一个荒唐的预测
    读成"材料错配很多"，而那听起来像一个可比较的结果。"""
    from spherical_blast_pinn.cli.compare import _region_at, compare_fields

    prediction = _synthetic_prediction()
    prediction["shock_radius"] = prediction["contact_radius"] * 0.5
    case = _synthetic_reference_case()

    region = _region_at(prediction, case.radius, float(case.times[4]))
    assert np.all(region == -1)
    report = compare_fields(prediction, case)
    assert report["region_mismatch_total"] == 0
    assert report["compared_cells"] == 0


def test_compare_field_scales_are_frozen_not_local():
    """归一化尺度来自检查点，不来自被比较的那份数据。"""
    from spherical_blast_pinn.cli.compare import compare_fields

    prediction = _synthetic_prediction()
    case = _synthetic_reference_case()
    first = compare_fields(prediction, case)
    # 把参考密度整体放大：固定尺度下误差必须随之改变，而不是因为分母也跟着
    # 变大而保持不变。
    case.density = case.density * 3.0
    second = compare_fields(prediction, case)
    assert second["product_density_relative_l2"] > first["product_density_relative_l2"] * 1.5
    assert second["normalisation_scales"] == first["normalisation_scales"]


def test_compare_field_lookup_does_not_interpolate_across_a_discontinuity():
    """跨区域的插值会造出两侧都不属于的中间状态。

    比"结果不是中间值"更明确的断言是**结果码**：落在接触面上的查询要么被直接
    拒绝（``cross_discontinuity_interpolation``），要么给出**某一侧**的实数。
    """
    from spherical_blast_pinn.cli.compare import (
        SAMPLE_CROSS_DISCONTINUITY,
        SAMPLE_OK,
        sample_field,
    )

    prediction = _synthetic_prediction()
    i = 4
    contact = float(prediction["contact_radius"][i])
    # 落在一个网格区间跨过接触面的位置上。
    radius = np.array([contact])
    value, why = sample_field(
        prediction, "density", radius, float(prediction["physical_time"][i])
    )
    assert why[0] in (SAMPLE_OK, SAMPLE_CROSS_DISCONTINUITY)
    if why[0] == SAMPLE_OK:
        # 给出来的必须是**某一侧**的实数，不是 1600 与 6 之间的任何中间值。
        assert value[0] == pytest.approx(1600.0, abs=1e-6) or value[0] == pytest.approx(
            6.0, abs=1e-6
        )
    else:
        assert not np.isfinite(value[0])


def test_the_midpoint_between_product_and_air_is_not_fabricated():
    """**复核给的原反例。** 同一半径上前一帧是空气（密度 2）、后一帧是产物
    （密度 100），两帧内部左右节点各自同材料；中间时刻线性插值给出 51——一个
    两侧都不属于的中间态。

    修后：两个时间层的支撑点分属不同区域，插值被拒绝，并给出明确原因。
    """
    from spherical_blast_pinn.cli.compare import (
        SAMPLE_CROSS_DISCONTINUITY,
        sample_field,
    )

    prediction = {
        "pinn_time": np.array([0.0, 1.0]),
        "physical_time": np.array([0.0, 1.0]),
        "start_time": 0.0,
        "radius": np.array([0.10, 0.11]),
        # 前一帧全是空气、后一帧全是产物：前沿在两帧之间扫过了整个网格。
        "density": np.array([[2.0, 2.0], [100.0, 100.0]]),
        "region": np.array([[1, 1], [0, 0]], dtype=np.int64),
        "contact_radius": np.array([0.09, 0.12]),
        "shock_radius": np.array([0.20, 0.20]),
    }
    value, why = sample_field(prediction, "density", np.array([0.105]), physical_time=0.5)
    assert why[0] == SAMPLE_CROSS_DISCONTINUITY, why[0]
    assert not np.isfinite(value[0]), "跨材料被插出了一个中间值"


def test_an_exact_grid_node_is_not_poisoned_by_a_zero_weight_nan_neighbour():
    """**复核给的原反例。** 精确落在节点上、自身有限、**零权重的邻点**是 NaN，
    查询仍然得到 NaN——因为 ``nan * 0`` 是 ``nan``。

    修后：零权重的支撑点不参与，值直接来自那个节点。
    """
    from spherical_blast_pinn.cli.compare import SAMPLE_OK, sample_field

    prediction = {
        "pinn_time": np.array([0.0, 1.0, 2.0]),
        "physical_time": np.array([0.0, 1.0, 2.0]),
        "start_time": 0.0,
        "radius": np.array([0.10, 0.11, 0.12]),
        # 中间节点在三个时刻上都是 NaN，两侧节点是有限值。
        "density": np.array(
            [[1.0, np.nan, 3.0], [1.0, np.nan, 3.0], [1.0, np.nan, 3.0]]
        ),
        "region": np.zeros((3, 3), dtype=np.int64),
        "contact_radius": np.full(3, 0.05),
        "shock_radius": np.full(3, 0.30),
    }
    # 精确落在 (t = 1, r = 0.10)：时间权重为 0 的那一层、以及径向零权重的
    # 右邻点（0.11，NaN）都不该参与。
    value, why = sample_field(
        prediction, "density", np.array([0.10]), physical_time=1.0
    )
    assert why[0] == SAMPLE_OK, why[0]
    assert value[0] == pytest.approx(1.0)


# ==========================================================================
# §11.1 逐字段独立性：一个字段缺失不丢掉其他可靠字段
# ==========================================================================


def test_a_missing_energy_field_does_not_drop_the_other_fields(tmp_path):
    """一个字段（e）缺失时，同一位置的 rho/u 仍应被保留（§11.1）。

    旧采样器 ``keep &= isfinite`` 对 rho/u/e **全部**做与：只要 e 是 NaN，整点
    被删，rho/u 也跟着丢。这里把产物单元的 ``specific_energy`` 全部置 NaN，验证
    rho/u 标签仍被采出来，而 e 带着 NaN 被损失按字段掩码跳过。
    """
    case = _write_case(tmp_path)
    # material 编码是整数：0 产物、1 空气、2 未受扰动（不是 moving_frame 的
    # "product"/"air" 字符串）。这里按整数 0 定位产物单元。
    for frame in range(case.times.size):
        product_cells = np.flatnonzero(case.material[frame] == 0)
        case.specific_energy[frame, product_cells] = np.nan

    sampler = RadiusSampler(
        case=case,
        scales=_scales(case.charge_radius),
        reference=_tiny_reference(),
        config=SamplingConfig(
            n_pde_product=4, n_pde_air=4, n_ic_product=2, n_ic_air=2,
            n_contact=2, n_shock=2, n_data_product=12, n_data_air=8, n_data_front=6,
        ),
        seed=3,
    )
    batch = sampler.batch()

    # 关键回归：e 全 NaN 时，rho/u 仍非空——旧实现会把产物点删光。
    assert batch.rho_data_product.numel() > 0
    assert batch.u_data_product.numel() > 0
    assert batch.rho_data_product.shape == batch.e_data_product.shape

    # rho/u 仍有限，e 在该批点上是 NaN（不是靠整点删除避开的）。
    assert bool(torch.isfinite(batch.rho_data_product).all())
    assert bool(torch.isfinite(batch.u_data_product).all())
    assert bool(torch.isnan(batch.e_data_product).all())


def test_a_single_valid_front_still_yields_supervisable_frames(tmp_path):
    """§11.1：一侧前沿有效就够，不再被另一侧的缺失否决。

    以前 ``supervisable_frames`` 是 ``contact_valid & shock_valid``：激波那一天
    没识别出来，接触面明明可靠也一起丢掉，而且丢得无声——表现为有效时间范围莫名
    其妙地短，看的人只会以为"数据就这样"。
    """
    import numpy as np

    case = _write_case(tmp_path)
    # 全程只有接触面可靠，激波全部标无效。
    case.shock_valid[:] = False

    frames = case.supervisable_frames()
    assert np.any(frames), "只有接触面可靠时，整帧不应被全部否决"

    # 接触面与激波各自的可监督集合也要分开给出。
    assert np.any(case.contact_supervisable_frames())
    assert not np.any(case.shock_supervisable_frames())
    # 空气间隙仍要求两侧都可靠——这一项**必须**被否掉，不能拿单侧凑。
    assert not np.any(case.air_supervisable_frames())


# ==========================================================================
# §11.2 L_geometry：位置约束
# ==========================================================================


def _geometry_model_and_batch(tmp_path, *, contact_hat: float):
    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet
    from spherical_blast_pinn.models.product_net import ProductNet

    case = _write_case(tmp_path)
    radius = case.charge_radius
    scales = _scales(radius)
    model = CoupledModel(
        product=ProductNet(scales=scales, width=8, depth=1, seed=0),
        air=AirNet(scales=scales, width=8, depth=1, seed=1),
        front=FrontNet(
            scales=scales, width=8, depth=1, initial_contact_hat=contact_hat, seed=2
        ),
        scales=scales,
    )
    sampler = RadiusSampler(
        case=case, scales=scales, reference=_tiny_reference(),
        config=SamplingConfig(
            n_pde_product=4, n_pde_air=4, n_ic_product=2, n_ic_air=2,
            n_contact=2, n_shock=2, n_data_product=12, n_data_air=12, n_data_front=6,
        ),
        seed=0,
    )
    return model, sampler.batch(), case


def test_geometry_loss_fires_when_the_support_conflicts_with_the_front(tmp_path):
    """L_geometry 必须真的产生惩罚，而且**不依赖 target 是否有限**。

    §11.2：预测域不包容可靠支持时，场量监督可以暂不启用，但 L_geometry 必须保留
    该可靠观测、产生指向正确位置的梯度。把几何冲突的样本按"场量是否有效"一起删掉，
    模型只要缩小可比较区域就能降低误差——这正是要防的。

    这里把产物单元的 ``specific_energy`` 全部置 NaN（场量监督会跳过它们），几何
    约束仍必须给出非零惩罚。
    """
    import numpy as np

    from spherical_blast_pinn.losses.terms import geometry_loss

    # 把前沿压到很小（contact_hat=0.4），于是可靠产物观测的外端必然越界。
    model, batch, case = _geometry_model_and_batch(tmp_path, contact_hat=0.4)
    assert np.isfinite(batch.geometry_delta_contact)
    assert np.isfinite(batch.geometry_delta_shock)

    breakdown = geometry_loss(batch, model)
    assert float(breakdown.total) > 0.0
    assert breakdown.monitors["geometry_product_outer_violating_fraction"] > 0.0
    assert breakdown.monitors["geometry_product_outer_max_violation_over_R0"] > 0.0

    # 关键：把产物场量全部置 NaN，几何惩罚**必须还在**。
    batch_without_fields = batch
    batch_without_fields.e_data_product = torch.full_like(
        batch.e_data_product, float("nan")
    )
    still = geometry_loss(batch_without_fields, model)
    assert float(still.total) > 0.0, (
        "场量缺失时几何约束被一起关掉了——§11.2 禁止把几何损失挂在场量掩码下面"
    )


def test_geometry_loss_delta_is_frozen_from_the_data(tmp_path):
    """delta 来自数据与探测器分辨率，训练中不更新。"""
    from spherical_blast_pinn.losses.terms import geometry_loss

    model, batch, case = _geometry_model_and_batch(tmp_path, contact_hat=1.1)
    breakdown = geometry_loss(batch, model)

    assert batch.geometry_delta_contact == pytest.approx(
        case.contact_position_uncertainty()
    )
    assert batch.geometry_delta_shock == pytest.approx(case.shock_position_uncertainty())
    # 探测器用 contact_smoothing_cells 个单元的平滑窗口定位接触面，
    # 所以它的不确定性就是那条窗口；激波是一个单元。
    assert case.contact_position_uncertainty() == pytest.approx(
        case.detector.contact_smoothing_cells * case.grid_spacing
    )
    assert case.shock_position_uncertainty() == pytest.approx(case.grid_spacing)
    # 报告里给出无量纲化的 delta，便于审查
    assert breakdown.monitors["geometry_delta_contact_over_R0"] == pytest.approx(
        case.contact_position_uncertainty() / case.charge_radius
    )


def test_geometry_loss_refuses_to_vanish_when_delta_is_undefined(tmp_path):
    """没有 delta 时**不能**把 L_geometry 降级成 0。

    ``geometry_loss`` 拿不到有限 delta 时唯一"顺手"的做法是返回零——但这一项在
    总损失里是单独加权的，返回零意味着一个本该生效的约束从损失里静默消失。它没有
    报错，只往 monitors 里塞一个 ``geometry_available = 0``，没人会去看一个不报错
    的人。所以：有几何约束该管的点却缺 delta ⇒ 抛错。
    """
    import torch

    from spherical_blast_pinn.losses.terms import LossError, geometry_loss

    model, batch, _ = _geometry_model_and_batch(tmp_path, contact_hat=1.1)
    assert batch.s_data_product is not None and batch.s_data_product.numel() > 0

    batch.geometry_delta_contact = float("nan")
    with pytest.raises(LossError, match="L_geometry"):
        geometry_loss(batch, model)

    # 换成激波一侧缺失，也必须抛
    batch.geometry_delta_contact = 1.0e-3
    batch.geometry_delta_shock = float("nan")
    with pytest.raises(LossError, match="L_geometry"):
        geometry_loss(batch, model)

    # 连非有限值都不是、只是 0 或负——同样没有定义，一样拒绝
    batch.geometry_delta_shock = 0.0
    with pytest.raises(LossError, match="L_geometry"):
        geometry_loss(batch, model)


def test_geometry_loss_still_returns_zero_when_there_is_nothing_to_constrain(tmp_path):
    """反过来：**没有**几何点可约束时，缺 delta 不是错误，应当安静地返回 0。

    否则这条守护会变成"任何不走数据监督的运行都跑不起来"，那是过度收紧。
    """
    import torch

    from spherical_blast_pinn.losses.terms import geometry_loss

    model, batch, _ = _geometry_model_and_batch(tmp_path, contact_hat=1.1)
    batch.s_data_product = torch.zeros((0, 1), dtype=torch.float64)
    batch.s_data_air = torch.zeros((0, 1), dtype=torch.float64)
    batch.geometry_delta_contact = float("nan")
    batch.geometry_delta_shock = float("nan")

    breakdown = geometry_loss(batch, model)
    assert float(breakdown.total) == 0.0
    assert breakdown.monitors["geometry_available"] == 0.0


def test_the_sampler_refuses_a_case_that_cannot_define_a_position_tolerance(tmp_path):
    """源头校验：算例定义不出位置不确定性时，采样阶段就硬失败。

    这条与上一条是**两层**：损失层的守护保证"约束不会被静默吞掉"，采样层的守护
    保证错误在离源头最近的地方报出来，并带上算例名与原因。``grid_spacing`` 在可用
    半径少于两个时是 NaN，此时 delta 无从定义——这里直接把
    ``contact_position_uncertainty`` 换成一个返回 NaN 的实现来复现那种算例，
    不去动半径数组本身（那会先撞上"没有可监督帧"的另一条闸门，测不到这一条）。
    """
    import numpy as np

    from spherical_blast_pinn.sampling.sampler import RadiusSampler, SamplingConfig, SamplingError

    case = _write_case(tmp_path)
    case.contact_position_uncertainty = lambda: float(
        case.detector.contact_smoothing_cells * case.grid_spacing * np.nan
    )
    assert not np.isfinite(case.contact_position_uncertainty())

    scales = _scales(case.charge_radius)
    sampler = RadiusSampler(
        case=case, scales=scales, reference=_tiny_reference(),
        config=SamplingConfig(
            n_pde_product=4, n_pde_air=4, n_ic_product=2, n_ic_air=2,
            n_contact=2, n_shock=2, n_data_product=4, n_data_air=4, n_data_front=2,
        ),
        seed=0,
    )
    with pytest.raises(SamplingError, match="position uncertainty"):
        sampler.batch()


# ==========================================================================
# §12 平均观测算子
# ==========================================================================


def test_the_model_observation_is_a_shell_average_not_a_point_value(tmp_path):
    """模型侧必须在**同一支持区**上做几何加权平均，不能拿中心点值去比。

    标签是数据单元的平均值（支持区就是那个单元）；拿中心点值去比壳平均，比的是
    两个不同的观测，误差里混进了"观测定义之差"。

    这条测试**独立重算**一遍：自己按
    ``rho_bar = sum(v_j rho_j)/sum(v_j)``、``u_bar = sum(v_j rho_j u_j)/sum(v_j rho_j)``、
    ``p_bar = sum(v_j p_j)/sum(v_j)`` 在支持区上求积，和函数给出的结果逐点比较。
    只调用一次函数再和它自己比是恒等式，什么都测不到。
    """
    import torch

    from spherical_blast_pinn.losses.terms import (
        N_QUADRATURE,
        data_loss,
        shell_average_of_model,
    )

    model, batch, case = _geometry_model_and_batch(tmp_path, contact_hat=1.1)
    assert float(data_loss(batch, model).total) > 0.0

    lo = batch.lo_data_product
    hi = batch.hi_data_product
    n = lo.shape[0]
    # 支持区确实是一段，不是退化成一个点
    assert float((hi - lo).min()) > 0.0

    s_leaf = batch.s_data_product.detach().requires_grad_(True)
    got = shell_average_of_model(model, PRODUCT, batch, s_leaf, lo, hi)

    # --- 独立重算 ---------------------------------------------------------
    edges = torch.linspace(0.0, 1.0, N_QUADRATURE + 1, dtype=lo.dtype).reshape(1, -1)
    edges = lo.reshape(n, 1) + (hi - lo).reshape(n, 1) * edges          # (n, NQ+1)
    centres = 0.5 * (edges[:, :-1] + edges[:, 1:])                      # (n, NQ)
    volumes = (edges[:, 1:] ** 3 - edges[:, :-1] ** 3) / 3.0            # (n, NQ)
    radius_pts = centres.reshape(-1, 1)
    s_pts = s_leaf.repeat_interleave(N_QUADRATURE, dim=0)
    fronts = model.front_data(s_pts, charge_radius=batch.charge_radius, create_graph=False)
    fields = model.product_fields(s_pts, radius_pts / fronts.contact_radius)

    rho_pts = fields.density.reshape(n, N_QUADRATURE)
    u_pts = fields.velocity.reshape(n, N_QUADRATURE)
    p_pts = fields.pressure.reshape(n, N_QUADRATURE)
    v_sum = volumes.sum(dim=1, keepdim=True)
    mass = volumes * rho_pts
    mass_sum = mass.sum(dim=1, keepdim=True)

    assert torch.allclose(got[0], (volumes * rho_pts).sum(1, keepdim=True) / v_sum, rtol=1e-12)
    assert torch.allclose(got[1], (mass * u_pts).sum(1, keepdim=True) / mass_sum, rtol=1e-12)
    assert torch.allclose(got[3], (volumes * p_pts).sum(1, keepdim=True) / v_sum, rtol=1e-12)

    # --- 它**不是**中心点值 -------------------------------------------------
    point = model.product_fields(
        s_leaf, batch.r_data_product.detach() / fronts.contact_radius[: n]
    )
    assert not torch.allclose(got[0], point.density, rtol=1e-9, atol=1e-9), (
        "模型观测等于中心点值——平均没有生效，又退回了'壳中心点代替平均标签'"
    )


def test_shell_average_uses_pointwise_eos_not_eos_of_the_mean(tmp_path):
    """压力必须**逐点**由 EOS 求再平均，不能用 ``EOS(平均 rho, 平均 e)``。

    JWL 是非线性的，两者不相等。这条测试直接比较这两个量：在支持区有宽度、
    场内变化非零时，它们必须不同。
    """
    import torch

    from spherical_blast_pinn.losses.terms import (
        N_QUADRATURE,
        shell_average_of_model,
    )

    model, batch, case = _geometry_model_and_batch(tmp_path, contact_hat=1.1)
    lo, hi = batch.lo_data_product, batch.hi_data_product
    n = lo.shape[0]
    s_leaf = batch.s_data_product.detach().requires_grad_(True)
    got = shell_average_of_model(model, PRODUCT, batch, s_leaf, lo, hi)
    rho_bar, _, e_bar, p_bar = got

    edges = torch.linspace(0.0, 1.0, N_QUADRATURE + 1, dtype=lo.dtype).reshape(1, -1)
    edges = lo.reshape(n, 1) + (hi - lo).reshape(n, 1) * edges
    centres = 0.5 * (edges[:, :-1] + edges[:, 1:])
    radius_pts = centres.reshape(-1, 1)
    s_pts = s_leaf.repeat_interleave(N_QUADRATURE, dim=0)
    fronts = model.front_data(s_pts, charge_radius=batch.charge_radius, create_graph=False)
    fields = model.product_fields(s_pts, radius_pts / fronts.contact_radius)

    from spherical_blast_pinn.physics.eos import jwl_pressure
    from spherical_blast_pinn.physics.material import product_jwl

    eos_of_mean = jwl_pressure(rho_bar, e_bar, product_jwl())
    assert not torch.allclose(p_bar, eos_of_mean, rtol=1e-9, atol=1e-9), (
        "平均压力等于用平均状态算的 EOS 压力——说明用了 EOS(平均) 而不是逐点平均"
    )
