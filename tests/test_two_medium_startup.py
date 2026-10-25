"""短时两介质启动求解器与交接对象。

验证顺序按交付要求：网格与初值装配 → 等压静止 → 单步/多步相容能量 →
质量与边界功 → 网格/时间步敏感性 → 交接对象与材料侧插值。
"""
from __future__ import annotations

import dataclasses
import math
from collections import deque

import numpy as np
import pytest

from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.eos import jwl_pressure_numpy
from spherical_blast_pinn.physics.two_medium_initial import (
    AIR,
    PRODUCT,
    TwoMediumInitialField,
    build_two_medium_initial_field,
)
from spherical_blast_pinn.physics.two_medium_startup import (
    METHOD_VERSION,
    STARTUP_MODEL_ID,
    StartupConfig,
    StartupError,
    TwoMediumHandoff,
    TwoMediumStartup,
    build_handoff,
    integrate_positions,
    locate_shock,
    shock_rh_residuals,
)


def _field(**overrides):
    kwargs = dict(charge_radius=0.05)
    kwargs.update(overrides)
    return build_two_medium_initial_field(**kwargs)


def _config(**overrides):
    kwargs = dict(
        cells_inside=200,
        outside_growth=1.01,
        outer_radius_factor=3.0,
        cfl=0.2,
    )
    kwargs.update(overrides)
    return StartupConfig(**kwargs)


# --------------------------------------------------------------------------
# 1. 网格与初值装配
# --------------------------------------------------------------------------


def test_mesh_puts_a_node_at_the_interface_and_uses_the_field_parameters():
    field = _field()  # 默认物性
    solver = TwoMediumStartup(field, config=_config())
    mesh = solver.mesh

    assert mesh.radii[mesh.n_inside] == pytest.approx(field.charge_radius, abs=1.0e-15)
    assert np.all(mesh.material[: mesh.n_inside] == PRODUCT)
    assert np.all(mesh.material[mesh.n_inside :] == AIR)
    # 初值在两种材料内部是常数：换网格不需要重采样。
    product = mesh.material == PRODUCT
    assert np.allclose(
        solver.density[product], field.product_state["density"], rtol=1.0e-13
    )
    assert np.allclose(
        solver.specific_energy[product],
        field.product_state["specific_energy"],
        rtol=1.0e-13,
    )
    assert np.allclose(solver.density[~product], field.air_state["density"], rtol=1.0e-13)
    assert np.allclose(
        solver.specific_energy[~product], field.air_state["specific_energy"], rtol=1.0e-13
    )
    # 质量 = 密度 × 球对称体积。
    volume = (mesh.radii[1:] ** 3 - mesh.radii[:-1] ** 3) / 3.0
    assert np.array_equal(mesh.mass, solver.density * volume)
    # 启动网格必须比存储网格细得多。
    assert np.diff(mesh.radii).min() < np.diff(field.cell_edges).min() / 2.0


def test_custom_jwl_reference_density_is_taken_from_the_field():
    import dataclasses

    custom = dataclasses.replace(mat.product_jwl(), rho0=2000.0)
    field = _field(jwl=custom)
    solver = TwoMediumStartup(field, config=_config())

    assert solver.product_eos.rho0 == 2000.0
    assert np.allclose(
        solver.density[solver.material == PRODUCT], 2000.0, rtol=1.0e-13
    )


def test_non_field_input_is_rejected():
    with pytest.raises(StartupError, match="TwoMediumInitialField"):
        TwoMediumStartup(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 2. 等压静止对照：两侧材料不同、压力相同、速度为零
# --------------------------------------------------------------------------


def test_isobaric_two_material_state_stays_exactly_static():
    control = _field()
    isobaric = _field(air_pressure=control.product_state["pressure"])
    solver = TwoMediumStartup(isobaric, config=_config(cells_inside=50, outer_radius_factor=2.0))

    radii0 = solver.radii.copy()
    energy0 = solver.specific_energy.copy()
    solver.run_to(2.0e-7)

    assert np.max(np.abs(solver.velocity)) < 1.0e-6
    assert np.allclose(solver.radii, radii0, rtol=0.0, atol=1.0e-15)
    assert np.allclose(solver.specific_energy, energy0, rtol=1.0e-14, atol=0.0)
    assert solver._energy_drift() < 1.0e-14


# --------------------------------------------------------------------------
# 3. 单步与连续数步的相容能量
# --------------------------------------------------------------------------


def test_first_step_turns_kinetic_energy_into_internal_energy_exactly():
    solver = TwoMediumStartup(_field(), config=_config())
    mass = solver.mass
    kinetic0 = solver._kinetic_energy(solver.velocity)
    internal0 = float(np.sum(mass * solver.specific_energy))
    assert kinetic0 == 0.0

    solver._advance(solver._timestep())

    kinetic1 = solver._kinetic_energy(solver.velocity)
    internal1 = float(np.sum(mass * solver.specific_energy))
    assert kinetic1 > 0.0, "the pressure jump must launch the interface"
    # 第一步外边界仍静止，边界功为零；内能减少量必须正好等于获得的动能。
    assert abs(solver._boundary_work) < 1.0e-15 * abs(internal0)
    assert internal1 - internal0 == pytest.approx(-kinetic1, rel=1.0e-11)
    assert solver._energy_drift() < 1.0e-13


def test_energy_budget_includes_the_actual_boundary_work_over_many_steps():
    solver = TwoMediumStartup(_field(), config=_config())
    initial = float(np.sum(solver.mass * solver.specific_energy))
    solver.run_to(5.0e-7)

    state = solver.state()
    budget = state.internal_energy + state.kinetic_energy + state.boundary_work
    assert math_isclose(budget, initial, rel=1.0e-11)
    assert state.energy_drift < 1.0e-11
    # 不能靠整体缩放内能「配平」：内能与动能都必须真的变化。
    assert state.internal_energy < initial
    assert state.kinetic_energy > 0.0


def math_isclose(a: float, b: float, *, rel: float) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0e-300)


def test_material_masses_are_conserved_exactly():
    solver = TwoMediumStartup(_field(), config=_config())
    product_before = float(np.sum(solver.mass[solver.material == PRODUCT]))
    air_before = float(np.sum(solver.mass[solver.material == AIR]))
    solver.run_to(2.0e-7)

    assert float(np.sum(solver.mass[solver.material == PRODUCT])) == product_before
    assert float(np.sum(solver.mass[solver.material == AIR])) == air_before
    analytic = mat.PRODUCT_REFERENCE_DENSITY * 0.05**3 / 3.0
    assert product_before == pytest.approx(analytic, rel=1.0e-12)


# --------------------------------------------------------------------------
# 4. 网格与时间步敏感性
# --------------------------------------------------------------------------


def test_front_positions_are_only_weakly_sensitive_to_grid_and_timestep():
    coarse = TwoMediumStartup(_field(), config=_config(cells_inside=200)).run_to(5.0e-7)
    fine = TwoMediumStartup(_field(), config=_config(cells_inside=400)).run_to(5.0e-7)
    small_cfl = TwoMediumStartup(
        _field(), config=_config(cells_inside=400, cfl=0.1)
    ).run_to(5.0e-7)

    # 网格加密：前沿位置 2%，接触速度在早期较粗的网格上放宽到 10%。
    assert fine.contact_radius == pytest.approx(coarse.contact_radius, rel=0.02)
    assert fine.shock_radius == pytest.approx(coarse.shock_radius, rel=0.02)
    assert fine.contact_speed == pytest.approx(coarse.contact_speed, rel=0.10)
    # 时间步减半：必须几乎不变。
    assert small_cfl.contact_radius == pytest.approx(fine.contact_radius, rel=0.01)
    assert small_cfl.shock_radius == pytest.approx(fine.shock_radius, rel=0.01)
    assert small_cfl.contact_speed == pytest.approx(fine.contact_speed, rel=0.01)
    assert fine.shock_radius > fine.contact_radius


def test_shock_is_in_the_air_and_resolved_by_several_cells():
    solver = TwoMediumStartup(_field(), config=_config(cells_inside=400))
    state = solver.run_to(5.0e-7)

    assert state.contact_radius < state.shock_radius
    assert state.shock_radius < state.radii[-1]
    # 激波定位必须在空气单元里，产物区的高压不能把阈值带偏。
    assert np.any(state.material == AIR)
    gap = state.shock_radius - state.contact_radius
    assert gap > 4.0 * np.diff(state.radii).min()


def test_rankine_hugoniot_and_trajectory_speeds_remain_separate_estimates():
    state = TwoMediumStartup(_field(), config=_config(cells_inside=800)).run_to(1.0e-6)
    diagnostics = state.diagnostics

    # 对外 shock_speed 明确定义为质量 RH 反算值。
    assert state.shock_speed == diagnostics["shock_speed_rh"]
    assert diagnostics["shock_speed_trajectory"] > 0.0
    assert diagnostics["shock_speed_rh"] != diagnostics["shock_speed_trajectory"]
    assert diagnostics["shock_speed_deviation"] < 0.05
    # RH 残差用**轨迹速度**算，不是用反算速度算；否则质量方程会恒等于零。
    assert 0.0 < diagnostics["shock_rh_mass_rel"] < 0.05
    assert diagnostics["shock_rh_momentum_rel"] < 0.05
    assert diagnostics["shock_rh_energy_rel"] < 0.05
    assert diagnostics["shock_diagnostics_valid"] is True


# --------------------------------------------------------------------------
# 5. 交接对象
# --------------------------------------------------------------------------


def _handoff():
    solver = TwoMediumStartup(_field(), config=_config(cells_inside=400))
    solver.run_to(5.0e-7)
    return solver, build_handoff(solver)


def test_handoff_carries_one_physical_time_and_both_fronts():
    solver, handoff = _handoff()

    assert handoff.model_id == STARTUP_MODEL_ID
    assert handoff.method_version == METHOD_VERSION
    assert handoff.time == pytest.approx(5.0e-7, rel=1.0e-14)
    assert handoff.contact_radius < handoff.shock_radius
    assert handoff.n_inside == solver.mesh.n_inside
    assert handoff.provenance["time_origin"].startswith("t_phys = 0")
    assert handoff.provenance["pin_model_time"] == "tau = t_phys - t_h"
    assert handoff.provenance["energy"]["drift"] < 1.0e-11
    assert handoff.fronts_at(handoff.time) == pytest.approx(
        (
            handoff.contact_radius,
            handoff.shock_radius,
            handoff.contact_speed,
            handoff.shock_speed,
        )
    )


def test_handoff_interpolation_never_crosses_the_material_interface():
    _, handoff = _handoff()
    rc = handoff.contact_radius
    product_centres = handoff.centres[handoff.material == PRODUCT]
    air_centres = handoff.centres[handoff.material == AIR]

    near_product = handoff.sample_product(np.array([product_centres[-1]]))
    near_air = handoff.sample_air(np.array([air_centres[0]]))
    assert near_product["density"][0] == pytest.approx(
        handoff.density[handoff.material == PRODUCT][-1]
    )
    assert near_air["density"][0] == pytest.approx(
        handoff.density[handoff.material == AIR][0]
    )
    assert near_product["density"][0] != near_air["density"][0]
    # 越过界面查询必须报错，而不是悄悄插值成混合态。
    with pytest.raises(StartupError, match="outside the product"):
        handoff.sample_product(np.array([air_centres[0]]))
    with pytest.raises(StartupError, match="outside the air"):
        handoff.sample_air(np.array([product_centres[-1]]))
    assert rc < air_centres[0]


def test_handoff_rejects_zero_time_and_other_times():
    solver = TwoMediumStartup(_field(), config=_config())
    with pytest.raises(StartupError, match="t > 0"):
        build_handoff(solver)

    _, handoff = _handoff()
    with pytest.raises(StartupError, match="represents t"):
        handoff.fronts_at(handoff.time * 2.0)


def test_handoff_cache_key_depends_on_time_and_parameters():
    _, first = _handoff()
    _, second = _handoff()
    assert first.cache_key() == second.cache_key()

    solver = TwoMediumStartup(_field(), config=_config(cells_inside=400))
    solver.run_to(6.0e-7)
    later = build_handoff(solver)
    assert later.cache_key() != first.cache_key()


def test_handoff_rejects_unordered_fronts():
    _, handoff = _handoff()
    with pytest.raises(StartupError, match="Rc < Rs"):
        TwoMediumHandoff(
            **{
                **handoff.__dict__,
                "contact_radius": handoff.shock_radius * 1.1,
            }
        )


# --------------------------------------------------------------------------
# 6. 不做静默修复
# --------------------------------------------------------------------------


def test_non_positive_or_non_finite_thermodynamics_fail_loudly():
    solver = TwoMediumStartup(_field(), config=_config())

    with pytest.raises(StartupError, match="specific energy"):
        solver._thermodynamics(
            solver.density.copy(), np.full_like(solver.specific_energy, -1.0)
        )
    with pytest.raises(StartupError, match="density"):
        solver._thermodynamics(np.full_like(solver.density, np.nan), solver.specific_energy)

    # 不能把负的 c^2 夹成 0：注入一个负值，必须报错而不是被 sqrt 变成 0。
    import spherical_blast_pinn.physics.two_medium_startup as module

    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        module,
        "jwl_sound_speed_squared_numpy",
        lambda rho, e, params: np.full(np.shape(rho), -1.0),
    )
    try:
        with pytest.raises(StartupError, match="sound speed squared"):
            solver._thermodynamics(solver.density, solver.specific_energy)
    finally:
        monkey.undo()


def test_startup_module_does_not_depend_on_the_old_detonation_solver():
    import spherical_blast_pinn.physics.two_medium_startup as module

    assert "ProductsDetonation" not in vars(module)
    assert "DetonationProfile" not in vars(module)
    assert "SphericalPrecursor" not in vars(module)
    assert not isinstance(TwoMediumStartup(_field(), config=_config()), TwoMediumInitialField)


# --------------------------------------------------------------------------
# 7. 激波速度：局部窗口，不是整段历史平均
# --------------------------------------------------------------------------


def test_trajectory_window_is_local_not_full_history():
    solver = TwoMediumStartup(_field(), config=_config())
    solver.radii = np.linspace(0.0, 1.0, 2001)
    times = np.linspace(0.0, 10.0, 1001)
    # 先快后慢的非匀速轨迹：全程平均 0.08，最后一段只有 0.02。
    radii = np.where(
        times <= 5.0, 0.1 + 0.14 * times, 0.8 + 0.02 * (times - 5.0)
    )
    solver._shock_samples = deque(zip(times, radii), maxlen=8192)

    speed = solver._trajectory_shock_speed(window_cells=10.0, min_cells=3.0)
    full_history = (radii[-1] - radii[0]) / (times[-1] - times[0])

    assert speed == pytest.approx(0.02, rel=1.0e-6)
    assert abs(speed - full_history) > 0.05
    assert full_history == pytest.approx(0.08, rel=1.0e-12)


def test_trajectory_speed_is_invalid_with_insufficient_history():
    solver = TwoMediumStartup(_field(), config=_config())
    solver.radii = np.linspace(0.0, 1.0, 2001)
    solver._shock_samples = deque([(0.0, 0.5), (1.0e-9, 0.50001)], maxlen=8192)
    assert math.isnan(solver._trajectory_shock_speed(window_cells=10.0, min_cells=3.0))


# --------------------------------------------------------------------------
# 8. Rankine–Hugoniot 残差的量纲一致归一化与失败对照
# --------------------------------------------------------------------------


def test_rh_residuals_use_per_equation_scales_and_detect_a_broken_state():
    rho_0 = mat.AIR_DENSITY
    p_0 = mat.AIR_PRESSURE
    e_0 = mat.AIR_SPECIFIC_ENERGY
    speed = 3000.0
    rho_b = 1.8 * rho_0
    u_b = speed * (1.0 - rho_0 / rho_b)
    p_b = p_0 + rho_0 * speed**2 - rho_b * (speed - u_b) ** 2
    e_b = (
        e_0
        + p_0 / rho_0
        + 0.5 * speed**2
        - p_b / rho_b
        - 0.5 * (speed - u_b) ** 2
    )
    assert p_b > 0.0 and e_b > 0.0

    def evaluate(**overrides):
        kwargs = dict(
            post_density=rho_b,
            post_velocity=u_b,
            post_pressure=p_b,
            post_specific_energy=e_b,
            ambient_density=rho_0,
            ambient_pressure=p_0,
            ambient_specific_energy=e_0,
            shock_speed=speed,
        )
        kwargs.update(overrides)
        return shock_rh_residuals(**kwargs)

    good = evaluate()
    assert good["shock_rh_mass_rel"] < 1.0e-12
    assert good["shock_rh_momentum_rel"] < 1.0e-12
    assert good["shock_rh_energy_rel"] < 1.0e-12

    broken = evaluate(post_pressure=p_b * 1.1)
    expected = 0.1 * p_b / (rho_0 * speed**2)
    assert broken["shock_rh_momentum_rel"] == pytest.approx(expected, rel=1.0e-12)
    assert broken["shock_rh_momentum"] < 0.0
    # 三个方程各自用自己的尺度：同一个 10% 压力扰动不会在三项里给出同一个数。
    assert broken["shock_rh_momentum_rel"] != pytest.approx(
        broken["shock_rh_energy_rel"], rel=0.5
    )


# --------------------------------------------------------------------------
# 9. 材料侧插值：重构 (rho, u, e) 再算 EOS
# --------------------------------------------------------------------------


def _evolved_handoff(cells_inside: int = 800, end_time: float = 1.0e-6):
    solver = TwoMediumStartup(_field(), config=_config(cells_inside=cells_inside))
    solver.run_to(end_time)
    return build_handoff(solver)


def test_material_side_interpolation_recomputes_pressure_from_the_object_eos():
    handoff = _evolved_handoff()
    rc = handoff.contact_radius
    product_centres = handoff.centres[handoff.material == PRODUCT]
    air_centres = handoff.centres[handoff.material == AIR]

    # 非单元中心的查询点。
    product_query = np.array(
        [0.0, 0.5 * product_centres[0], product_centres[len(product_centres) // 2], rc]
    )
    product = handoff.sample_product(product_query)
    expected = jwl_pressure_numpy(
        product["density"], product["specific_energy"], handoff.product_eos
    )
    assert np.allclose(product["pressure"], expected, rtol=1.0e-12, atol=0.0)
    # 场确实是演化的非常值场，不是常数延拓蒙混过关。
    assert np.ptp(product["density"]) > 0.0

    air_query = np.array(
        [rc, air_centres[0], 0.5 * (rc + handoff.shock_radius), handoff.shock_radius]
    )
    air = handoff.sample_air(air_query)
    expected_air = (handoff.air_gamma - 1.0) * air["density"] * air["specific_energy"]
    assert np.allclose(air["pressure"], expected_air, rtol=1.0e-12, atol=0.0)
    assert np.ptp(air["density"]) > 0.0
    assert not np.any(air["undisturbed"])


def test_interpolation_domains_cover_the_physical_regions_and_not_beyond():
    handoff = _evolved_handoff()
    rc = handoff.contact_radius
    outer = float(handoff.radii[-1])

    # 球心与接触面都在合法物理域内；接触面两侧各自取自己的极限，不混合。
    product_at_centre = handoff.sample_product(np.array([0.0]))
    product_at_face = handoff.sample_product(np.array([rc]))
    air_at_face = handoff.sample_air(np.array([rc]))
    assert product_at_centre["density"][0] > 0.0
    assert product_at_face["density"][0] != air_at_face["density"][0]

    # 激波之外是已知的精确环境态，而不是插值延拓。
    beyond = handoff.sample_air(np.array([0.5 * (handoff.shock_radius + outer), outer]))
    assert np.all(beyond["undisturbed"])
    ambient = handoff.provenance["initial_state"]
    assert np.allclose(beyond["density"], ambient["air_density"], rtol=1.0e-14)
    assert np.allclose(beyond["specific_energy"], ambient["air_specific_energy"], rtol=1.0e-14)
    assert np.allclose(beyond["velocity"], 0.0, atol=0.0)

    # 越过材料界面报错，而不是返回一个跨材料平均。
    with pytest.raises(StartupError, match="product physical domain"):
        handoff.sample_product(np.array([rc * 1.001]))
    with pytest.raises(StartupError, match="air physical domain"):
        handoff.sample_air(np.array([rc * 0.999]))


# --------------------------------------------------------------------------
# 10. 时间层与几何/功体积增量
# --------------------------------------------------------------------------


def test_constant_acceleration_is_exact_for_a_variable_dt_sequence():
    radii = np.array([0.0, 0.5, 1.0])
    velocity = np.array([1.0, -2.0, 0.5])
    acceleration = np.array([2.0, 2.0, 2.0])

    r1, u1 = integrate_positions(radii, velocity, acceleration, 0.1)
    assert np.allclose(u1, velocity + 2.0 * 0.1, rtol=0.0, atol=1.0e-15)
    assert np.allclose(
        r1, radii + velocity * 0.1 + 0.5 * 2.0 * 0.1**2, rtol=0.0, atol=1.0e-15
    )

    # 第二步用不同的 dt：恒加速度下位置仍精确等于 0.5 a (dt1+dt2)^2。
    r2, u2 = integrate_positions(r1, u1, acceleration, 0.3)
    total = 0.4
    assert np.allclose(u2, velocity + 2.0 * total, rtol=0.0, atol=1.0e-14)
    assert np.allclose(
        r2, radii + velocity * total + 0.5 * 2.0 * total**2, rtol=0.0, atol=1.0e-14
    )


def test_state_carries_all_quantities_at_one_integer_time():
    solver = TwoMediumStartup(_field(), config=_config())
    state = solver.run_to(5.0e-7)

    assert state.time == pytest.approx(5.0e-7, rel=0.0, abs=0.0)
    assert state.last_dt > 0.0
    # 没有半时间层速度字段，也没有把速度标到另一个时刻的偏移。
    assert not hasattr(state, "velocity_time")
    assert state.last_dt < state.time


def test_geometric_and_work_volume_increments_refine_together():
    base = TwoMediumStartup(_field(), config=_config())
    dt = base._timestep()
    base._advance(dt)
    half = TwoMediumStartup(_field(), config=_config())
    half._advance(0.5 * dt)

    assert 0.0 < base._last_volume_mismatch < 1.0e-2
    assert half._last_volume_mismatch < 0.5 * base._last_volume_mismatch
    # 能量恒等式同时成立，但它不能替代上面这项几何/功的检查。
    assert base._energy_drift() < 1.0e-13
    assert half._energy_drift() < 1.0e-13


# --------------------------------------------------------------------------
# 11. 交接资格：诊断快照 vs 训练交接
# --------------------------------------------------------------------------


def test_qualification_separates_a_diagnostic_snapshot_from_a_training_handoff():
    coarse = _evolved_handoff(cells_inside=400)
    fine = _evolved_handoff(cells_inside=800)

    # 内部检查失败：仍然是可导出的诊断快照，但不具备训练资格。
    assert coarse.internal_checks_passed is False
    assert "shock_diagnostics_valid" in coarse.qualification["internal_checks"]["failed"]
    assert coarse.contact_radius < coarse.shock_radius
    assert coarse.training_qualified is False

    # 内部检查通过，但**没有外部收敛证据**，训练资格仍然是 false。
    assert fine.internal_checks_passed is True
    assert fine.qualification["internal_checks"]["failed"] == []
    assert fine.external_validation["provided"] is False
    assert fine.training_qualified is False


# --------------------------------------------------------------------------
# 12. 速度重构：节点量，不是单元平均
# --------------------------------------------------------------------------


def test_velocity_reconstruction_uses_nodes_not_cell_averages():
    handoff = _evolved_handoff()
    rc = handoff.contact_radius

    centre = handoff.sample_product(np.array([0.0]))
    assert centre["velocity"][0] == 0.0

    product_face = handoff.sample_product(np.array([rc]))
    air_face = handoff.sample_air(np.array([rc]))
    assert product_face["velocity"][0] == handoff.contact_speed
    assert air_face["velocity"][0] == handoff.contact_speed
    assert product_face["velocity"][0] == air_face["velocity"][0]


def test_linear_node_velocity_is_reconstructed_exactly():
    handoff = _evolved_handoff()
    slope, offset = 1234.5, -7.0
    linear = slope * handoff.radii + offset
    twin = dataclasses.replace(handoff, velocity=linear)

    rc = twin.contact_radius
    product_query = np.array([0.0, 0.31 * rc, 0.72 * rc, rc])
    assert np.allclose(
        twin.sample_product(product_query)["velocity"],
        slope * product_query + offset,
        rtol=0.0,
        atol=1.0e-9,
    )
    air_query = np.array([rc, 0.5 * (rc + twin.shock_radius), twin.shock_radius])
    assert np.allclose(
        twin.sample_air(air_query)["velocity"],
        slope * air_query + offset,
        rtol=0.0,
        atol=1.0e-9,
    )


# --------------------------------------------------------------------------
# 13. 激波提取：可控压力剖面
# --------------------------------------------------------------------------


def _synthetic_profile(*, contact_peak: float = 2.0e7, ramp_overshoot=None):
    centres = np.linspace(0.0, 1.0, 401)
    ambient = 1.0e5
    pressure = np.full(401, ambient)
    for index in range(0, 298):
        pressure[index] = 2.0e7 * (1.0 - 0.002 * (297 - index))
    pressure[297] = 2.0e7
    pressure[298] = 1.4e7
    pressure[299] = 6.0e6
    pressure[300] = 1.5e5
    # 远离激波窗口的接触面附近峰值；改变它不应移动 Rs。
    pressure[:250] = contact_peak
    if ramp_overshoot is not None:
        pressure[299] = ramp_overshoot
    return centres, pressure, np.ones(401, dtype=bool), ambient


def test_local_shock_location_is_independent_of_the_contact_region_peak():
    base = _synthetic_profile(contact_peak=2.0e7)
    high = _synthetic_profile(contact_peak=4.0e7)

    first = locate_shock(*base)
    second = locate_shock(*high)

    assert first.valid and second.valid
    assert first.radius == pytest.approx(second.radius, rel=0.0, abs=1.0e-15)
    assert first.plateau_cell == second.plateau_cell
    # 旧做法用空气区全局最大值：接触面峰值一变，半跳变阈值就变了。
    old_target_base = 0.5 * (np.max(base[1]) + base[3])
    old_target_high = 0.5 * (np.max(high[1]) + high[3])
    assert old_target_high > 1.5 * old_target_base


def test_monotone_post_shock_profile_still_locates_the_shock():
    centres, pressure, air, ambient = _synthetic_profile()
    location = locate_shock(centres, pressure, air, ambient)

    assert location.valid
    assert centres[298] <= location.radius <= centres[299]


def test_overshoot_inside_the_ramp_is_rejected_not_taken_as_the_plateau():
    centres, pressure, air, ambient = _synthetic_profile(ramp_overshoot=2.5e7)
    location = locate_shock(centres, pressure, air, ambient)

    assert location.valid is False
    assert location.reason == "non-monotone shock ramp"


def test_flat_profile_and_boundary_shock_are_reported_as_unreliable():
    centres = np.linspace(0.0, 1.0, 401)
    flat = np.full(401, 1.0e5)
    location = locate_shock(centres, flat, np.ones(401, dtype=bool), 1.0e5)
    assert location.valid is False
    assert location.reason == "no disturbed air cell"

    _, pressure, air, ambient = _synthetic_profile()
    shifted = np.full(401, ambient)
    shifted[396:] = pressure[296:301][:5]
    location = locate_shock(centres, shifted, air, ambient)
    assert location.valid is False
    assert location.reason == "shock too close to the outer boundary"


def test_unbounded_peak_is_reported_with_a_specific_reason():
    centres = np.linspace(0.0, 1.0, 401)
    ambient = 1.0e5
    pressure = np.full(401, ambient)
    # 展宽层跨 20 个单元：12 个单元的窗口里看不到峰值。
    for index in range(280, 301):
        pressure[index] = ambient + (2.0e7 - ambient) * (300 - index) / 20.0
    pressure[300] = 1.5e5
    pressure[:280] = 2.0e7
    location = locate_shock(centres, pressure, np.ones(401, dtype=bool), ambient)
    assert location.valid is False
    assert location.reason == "local peak at the search-window edge"


# --------------------------------------------------------------------------
# 14. 训练资格需要绑定到本次运行的外部证据
# --------------------------------------------------------------------------


def _evidence(handoff):
    config = dict(handoff.provenance["config"])
    return {
        "source": "tests/test_two_medium_startup.py",
        "method_version": handoff.method_version,
        "charge_radius": handoff.charge_radius,
        "handoff_time_s": handoff.time,
        "initial_state": dict(handoff.provenance["initial_state"]),
        "eos": {key: float(value) for key, value in handoff.product_eos.as_dict().items()},
        "artificial_viscosity": {"q1": config["q1"], "q2": config["q2"]},
        "runs": [
            {
                "id": "n800_cfl02",
                "cells_inside": 800,
                "outside_growth": 1.01,
                "outer_radius_factor": 3.0,
                "cfl": 0.2,
                "charge_radius": handoff.charge_radius,
                "handoff_time_s": handoff.time,
            },
            {
                "id": "n1600_cfl02",
                "cells_inside": 1600,
                "outside_growth": 1.01,
                "outer_radius_factor": 3.0,
                "cfl": 0.2,
                "charge_radius": handoff.charge_radius,
                "handoff_time_s": handoff.time,
            },
            {
                "id": "n1600_cfl01",
                "cells_inside": 1600,
                "outside_growth": 1.01,
                "outer_radius_factor": 3.0,
                "cfl": 0.1,
                "charge_radius": handoff.charge_radius,
                "handoff_time_s": handoff.time,
            },
        ],
        "handoff_run": "n800_cfl02",
        "grid_convergence": {
            "coarse_run": "n800_cfl02",
            "fine_run": "n1600_cfl02",
            "front_relative": 0.004,
            "field_relative": 0.01,
        },
        "timestep_sensitivity": {
            "coarse_run": "n1600_cfl02",
            "fine_run": "n1600_cfl01",
            "front_relative": 0.001,
            "field_relative": 0.002,
        },
        "contact_consistency": {"pressure_relative": 0.002, "velocity_relative": 0.0},
    }


def _replace_run(evidence, run_id, **changes):
    runs = [dict(run) for run in evidence["runs"]]
    for run in runs:
        if run["id"] == run_id:
            run.update(changes)
    return {**evidence, "runs": runs}


def test_training_qualification_requires_external_evidence():
    solver = TwoMediumStartup(_field(), config=_config(cells_inside=800))
    solver.run_to(1.0e-6)

    without = build_handoff(solver)
    assert without.internal_checks_passed is True
    assert without.training_qualified is False

    with_evidence = build_handoff(solver, validation_evidence=_evidence(without))
    assert with_evidence.external_validation["accepted"] is True
    assert with_evidence.training_qualified is True


def test_external_evidence_must_be_bound_and_numeric():
    solver = TwoMediumStartup(_field(), config=_config(cells_inside=800))
    solver.run_to(1.0e-6)
    base = _evidence(build_handoff(solver))

    def verdict(evidence):
        return build_handoff(solver, validation_evidence=evidence).training_qualified

    assert verdict(base) is True
    assert verdict({**base, "method_version": "other"}) is False
    assert verdict({**base, "handoff_time_s": base["handoff_time_s"] * 2.0}) is False

    # 负误差不再被接受。
    assert verdict(
        {
            **base,
            "grid_convergence": {**base["grid_convergence"], "field_relative": -1.0},
        }
    ) is False

    # 与交接运行无关的网格（8/16）不能当证据。
    unrelated = {**base, "runs": [dict(run) for run in base["runs"]]}
    unrelated["runs"] = [
        dict(run, cells_inside=8) if run["id"] == "n800_cfl02" else dict(run, cells_inside=16)
        if run["id"] == "n1600_cfl02" else run
        for run in unrelated["runs"]
    ]
    assert verdict(unrelated) is False

    # 单元数不允许静默截断小数。
    assert verdict(_replace_run(base, "n1600_cfl02", cells_inside=1600.7)) is False

    # CFL 对照关系必须合法：细网格的 CFL 不能反而更大。
    assert verdict(
        _replace_run(base, "n1600_cfl01", cfl=0.3)
    ) is False

    # 材料参数必须与本次运行一致。
    assert verdict(
        {
            **base,
            "initial_state": {**base["initial_state"], "air_density": 1.0},
        }
    ) is False
    assert verdict(
        {**base, "eos": {**base["eos"], "omega": 0.31}}
    ) is False
    assert verdict(
        {
            **base,
            "artificial_viscosity": {**base["artificial_viscosity"], "q2": 0.07},
        }
    ) is False

    # 布尔值不能冒充误差数值。
    assert verdict(
        {
            **base,
            "contact_consistency": {"pressure_relative": True, "velocity_relative": 0.0},
        }
    ) is False


def test_dip_then_rise_ramp_is_rejected():
    centres = np.linspace(0.0, 1.0, 401)
    pressure = np.full(401, 1.0e5)
    pressure[:296] = 2.0e7
    pressure[296] = 4.0e6
    pressure[297] = 6.0e6
    pressure[298] = 1.5e5
    location = locate_shock(centres, pressure, np.ones(401, dtype=bool), 1.0e5)
    assert location.valid is False
    assert location.reason == "non-monotone shock ramp"


def test_monotone_ramp_after_the_peak_is_accepted():
    centres = np.linspace(0.0, 1.0, 401)
    pressure = np.full(401, 1.0e5)
    pressure[:295] = 2.0e7
    pressure[295] = 2.0e7
    pressure[296] = 4.0e6
    pressure[297] = 1.5e5
    location = locate_shock(centres, pressure, np.ones(401, dtype=bool), 1.0e5)
    assert location.valid is True
    assert centres[295] <= location.radius <= centres[296]
