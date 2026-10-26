"""新初值：均匀高压产物球 + 静止空气（2026-10-21 决策）。

只验证初值构造本身：材料分区、常量状态、EOS 一致性、球体积分、界面归属、
非法输入、与旧求解器的独立性，以及不触碰旧参考场文件。不涉及时间推进、
PINN 交接或正式训练。
"""
from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.eos import (
    JWLParameters,
    ideal_gas_pressure_numpy,
    ideal_gas_sound_speed_numpy,
    ideal_gas_specific_energy_numpy,
    jwl_pressure_numpy,
    jwl_sound_speed_numpy,
)
from spherical_blast_pinn.physics.two_medium_initial import (
    AIR,
    PRODUCT,
    TWO_MEDIUM_MODEL_ID,
    TwoMediumInitialError,
    TwoMediumInitialField,
    build_two_medium_initial_field,
)


def _field(**overrides):
    kwargs = dict(
        charge_radius=0.05,
        cells_inside=16,
        cells_outside=16,
        outside_growth=1.005,
    )
    kwargs.update(overrides)
    return build_two_medium_initial_field(**kwargs)


# --------------------------------------------------------------------------
# 1. 材料分区、常量状态、零速度
# --------------------------------------------------------------------------


def test_piecewise_constant_states_and_zero_velocity():
    field = _field()
    n = field.n_inside

    assert field.n_cells == 32
    assert np.all(field.material[:n] == PRODUCT)
    assert np.all(field.material[n:] == AIR)
    assert np.all(field.density[:n] == mat.PRODUCT_REFERENCE_DENSITY)
    assert np.all(field.density[n:] == mat.AIR_DENSITY)
    assert np.all(field.specific_energy[:n] == field.product_state["specific_energy"])
    assert np.all(field.specific_energy[n:] == field.air_state["specific_energy"])
    assert np.all(field.pressure[:n] == field.product_state["pressure"])
    assert np.allclose(field.pressure[n:], mat.AIR_PRESSURE, rtol=1.0e-14, atol=0.0)
    assert np.all(field.velocity == 0.0)
    assert np.all(field.node_velocity == 0.0)


# --------------------------------------------------------------------------
# 2. 压力、密度、内能满足各自的 EOS
# --------------------------------------------------------------------------


def test_pressures_and_energies_satisfy_their_eos():
    field = _field()
    n = field.n_inside
    jwl = mat.product_jwl()

    p_product = jwl_pressure_numpy(
        field.density[:n], field.specific_energy[:n], jwl
    )
    assert np.allclose(p_product, field.pressure[:n], rtol=1.0e-15, atol=0.0)

    p_air = ideal_gas_pressure_numpy(
        field.density[n:], field.specific_energy[n:], mat.AIR_GAMMA
    )
    assert np.allclose(p_air, field.pressure[n:], rtol=1.0e-15, atol=0.0)

    e_air = float(
        ideal_gas_specific_energy_numpy(
            np.array([mat.AIR_DENSITY]),
            np.array([mat.AIR_PRESSURE]),
            mat.AIR_GAMMA,
        )[0]
    )
    assert field.air_state["specific_energy"] == pytest.approx(e_air, rel=1.0e-14)
    assert field.air_state["pressure"] == pytest.approx(mat.AIR_PRESSURE, rel=1.0e-14)


# --------------------------------------------------------------------------
# 3. 球内总质量 = 均匀球解析质量（真实球体积权重）
# --------------------------------------------------------------------------


def test_product_mass_matches_the_analytic_uniform_sphere():
    field = _field()
    analytic = mat.PRODUCT_REFERENCE_DENSITY * 0.05**3 / 3.0

    assert field.product_mass == pytest.approx(analytic, rel=1.0e-14)
    assert np.allclose(field.mass, field.density * field.cell_volume, rtol=0.0, atol=0.0)
    assert np.sum(field.cell_volume[field.product_mask]) == pytest.approx(
        0.05**3 / 3.0, rel=1.0e-14
    )


# --------------------------------------------------------------------------
# 4. 球内总内能 = 质量 × 比内能；初始动能为零
# --------------------------------------------------------------------------


def test_product_internal_energy_and_zero_kinetic_energy():
    field = _field()
    e_product = field.product_state["specific_energy"]

    assert field.product_internal_energy == pytest.approx(
        field.product_mass * e_product, rel=1.0e-14
    )
    assert field.kinetic_energy == 0.0
    assert field.total_energy == pytest.approx(field.internal_energy, rel=1.0e-15)


# --------------------------------------------------------------------------
# 5. 改变 R0 时，产物质量与内能按 R0^3 缩放
# --------------------------------------------------------------------------


def test_product_mass_and_energy_scale_as_radius_cubed():
    small = _field(charge_radius=0.05)
    large = _field(charge_radius=0.10)

    assert large.product_mass / small.product_mass == pytest.approx(8.0, rel=1.0e-12)
    assert large.product_internal_energy / small.product_internal_energy == pytest.approx(
        8.0, rel=1.0e-12
    )
    # 材料状态不随 R0 变。
    assert large.product_state["density"] == small.product_state["density"]
    assert large.product_state["specific_energy"] == small.product_state["specific_energy"]
    assert large.product_state["pressure"] == small.product_state["pressure"]
    # 只比较强度量；体积、质量、内能是广延量，按 R0^3 缩放。
    for key in ("density", "specific_energy", "pressure", "sound_speed", "velocity"):
        assert large.air_state[key] == small.air_state[key]


# --------------------------------------------------------------------------
# 6. 界面是单元面，左右状态不混合，允许压力跳跃
# --------------------------------------------------------------------------


def test_interface_is_a_cell_face_with_explicit_left_and_right_states():
    field = _field()
    n = field.n_inside

    assert field.cell_edges[n] == pytest.approx(0.05, abs=1.0e-15)
    assert field.material[n - 1] == PRODUCT
    assert field.material[n] == AIR
    assert field.density[n - 1] == field.product_state["density"]
    assert field.density[n] == field.air_state["density"]
    assert field.pressure[n - 1] == field.product_state["pressure"]
    assert field.pressure[n] == field.air_state["pressure"]
    # 初始压力跳跃是这个问题的一部分，不能被强制抹平。
    assert field.pressure[n - 1] != field.pressure[n]
    assert field.metadata["pressure_jump_pa"] == pytest.approx(
        field.pressure[n - 1] - field.pressure[n], rel=1.0e-14
    )
    # 没有任何单元同时拥有两种材料的状态：material 是单值，界面是面。
    assert set(np.unique(field.material).tolist()) == {PRODUCT, AIR}


# --------------------------------------------------------------------------
# 7. 非法半径与非法物性输入被明确拒绝
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"charge_radius": 0.0},
        {"charge_radius": -1.0},
        {"charge_radius": float("nan")},
        {"charge_radius": float("inf")},
        {"product_density": 0.0},
        {"product_density": -1.0},
        {"product_density": float("nan")},
        {"product_specific_energy": 0.0},
        {"product_specific_energy": -1.0},
        {"air_density": 0.0},
        {"air_pressure": 0.0},
        {"air_gamma": 1.0},
        {"air_gamma": float("nan")},
        {"outer_radius": 0.05},
        {"outer_radius": 0.01},
        {"cells_inside": 0},
        {"cells_outside": 0},
        {"outside_growth": 0.9},
        {"outside_growth": float("nan")},
    ],
)
def test_invalid_inputs_are_rejected(overrides):
    with pytest.raises(TwoMediumInitialError):
        _field(**overrides)


# --------------------------------------------------------------------------
# 8. 不依赖旧爆轰求解器
# --------------------------------------------------------------------------


def test_the_old_detonation_and_precursor_modules_are_gone():
    """旧爆轰/前置求解器已删除：新初值不可能再调用它们。"""
    import importlib

    import spherical_blast_pinn.physics.two_medium_initial as module

    assert "ProductsDetonation" not in vars(module)
    assert "DetonationProfile" not in vars(module)
    assert "SphericalPrecursor" not in vars(module)
    for name in (
        "spherical_blast_pinn.physics.detonation",
        "spherical_blast_pinn.physics.precursor",
        "spherical_blast_pinn.physics.two_medium_startup",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)

    field = _field()
    assert field.product_mass > 0.0
    assert field.n_cells == 32


def test_build_writes_no_files_and_cannot_target_the_reference_cache(tmp_path):
    reference = tmp_path / "reference_field.npz"
    reference.write_bytes(b"do-not-touch")
    before = reference.read_bytes()

    _field()

    assert reference.read_bytes() == before
    parameters = inspect.signature(build_two_medium_initial_field).parameters
    for forbidden in ("path", "output", "output_dir", "cache", "cache_dir"):
        assert forbidden not in parameters


# --------------------------------------------------------------------------
# 10. 模型标识：不是方案 B 的已验收参考场
# --------------------------------------------------------------------------


def test_public_exports_include_the_new_initial_field():
    from spherical_blast_pinn import physics

    assert physics.TWO_MEDIUM_MODEL_ID == TWO_MEDIUM_MODEL_ID
    assert physics.TWO_MEDIUM_CONSTRUCTION == TWO_MEDIUM_MODEL_ID
    assert physics.build_two_medium_initial_field is build_two_medium_initial_field
    assert physics.TwoMediumInitialField is not None


def test_model_id_is_not_the_scheme_b_reference_and_has_no_handoff_time():
    field = _field()

    assert TWO_MEDIUM_MODEL_ID == "uniform_product_sphere_air_v1"
    assert TWO_MEDIUM_MODEL_ID != "scheme_b_products_only_detonation"
    assert field.metadata["model_id"] == TWO_MEDIUM_MODEL_ID
    assert field.metadata["construction"] == TWO_MEDIUM_MODEL_ID
    assert field.metadata["reference_field"] is False
    assert "NOT APPLICABLE" in field.metadata["acceptance_status"]
    assert field.metadata["handoff_time_s"] is None
    assert field.time == 0.0


# --------------------------------------------------------------------------
# 11. E0 的口径：参考体积能量密度 -> 比内能只除一次密度
# --------------------------------------------------------------------------


def test_e0_is_per_reference_volume_and_converted_once():
    field = _field()

    assert field.metadata["E0_volume_energy_j_m3"] == mat.PRODUCT_VOLUME_ENERGY
    assert field.metadata["E0_definition"].startswith("per reference volume")
    assert field.product_state["specific_energy"] == pytest.approx(
        mat.PRODUCT_VOLUME_ENERGY / mat.PRODUCT_REFERENCE_DENSITY, rel=1.0e-14
    )
    # 参考状态的压力由 EOS 得到，与历史报告的硬编码值无关。
    assert field.product_state["pressure"] == pytest.approx(
        mat.product_pressure_at_reference_state(), rel=1.0e-12
    )


# --------------------------------------------------------------------------
# 12. 时间与几何契约：反例、合法非均匀网格与只读性
# --------------------------------------------------------------------------


def test_legal_noop_replace_is_accepted():
    control = _field()
    twin = dataclasses.replace(control, time=0.0)
    assert twin.time == control.time == 0.0
    assert np.array_equal(twin.cell_edges, control.cell_edges)
    assert twin.product_mass == control.product_mass


def test_mutant_nonzero_time_is_rejected():
    control = _field()
    assert 1.0 != control.time
    with pytest.raises(TwoMediumInitialError, match="time must be exactly 0.0"):
        dataclasses.replace(control, time=1.0)


def test_mutant_charge_radius_without_moving_the_interface_is_rejected():
    control = _field()
    assert control.charge_radius == pytest.approx(0.05)
    assert control.cell_edges[control.n_inside] == pytest.approx(0.05)
    with pytest.raises(TwoMediumInitialError, match="charge_radius"):
        dataclasses.replace(control, charge_radius=0.04)


def test_mutant_nan_cell_volume_is_rejected_before_nan_comparisons():
    control = _field()
    replacement = np.full_like(control.cell_volume, np.nan)
    assert np.isnan(replacement).all()
    with pytest.raises(TwoMediumInitialError, match="cell_volume must be finite"):
        dataclasses.replace(control, cell_volume=replacement)


def test_mutant_infinite_cell_volume_is_rejected():
    control = _field()
    replacement = np.full_like(control.cell_volume, np.inf)
    assert np.isinf(replacement).all()
    with pytest.raises(TwoMediumInitialError, match="cell_volume must be finite"):
        dataclasses.replace(control, cell_volume=replacement)


def test_mutant_doubled_volume_and_mass_is_rejected_despite_mass_volume_consistency():
    control = _field()
    volume = control.cell_volume * 2.0
    mass = control.mass * 2.0
    # 这个变异仍满足 mass = density * cell_volume……
    assert np.allclose(mass, control.density * volume, rtol=0.0, atol=0.0)
    # ……但 cell_volume 已经不符合网格几何。
    with pytest.raises(TwoMediumInitialError, match="cell_volume must equal"):
        dataclasses.replace(control, cell_volume=volume, mass=mass)


def test_mutant_mass_only_is_rejected():
    control = _field()
    replacement = control.mass * 2.0
    assert not np.array_equal(replacement, control.mass)
    with pytest.raises(TwoMediumInitialError, match="mass must equal"):
        dataclasses.replace(control, mass=replacement)


def test_mutant_cell_centre_off_the_declared_definition_is_rejected():
    control = _field()
    replacement = control.cell_centres * 1.001
    assert not np.allclose(replacement, control.cell_centres)
    with pytest.raises(TwoMediumInitialError, match="half-volume centre"):
        dataclasses.replace(control, cell_centres=replacement)


def test_mutant_outer_edge_mismatch_is_rejected():
    control = _field()
    edges = control.cell_edges.copy()
    edges[-1] *= 1.01
    with pytest.raises(TwoMediumInitialError, match="outer_radius"):
        dataclasses.replace(control, cell_edges=edges)


def test_mutant_interface_face_mismatch_is_rejected():
    control = _field()
    material = control.material.copy()
    material[control.n_inside - 1] = AIR
    with pytest.raises(TwoMediumInitialError, match="product/air interface"):
        dataclasses.replace(control, material=material)


def test_legal_non_uniform_air_grid_is_accepted():
    field = _field(cells_inside=10, cells_outside=12, outside_growth=1.25)
    air_widths = np.diff(field.cell_edges[field.n_inside :])
    assert np.all(air_widths > 0.0)
    assert np.all(np.diff(air_widths) > 0.0)  # geometric growth
    assert field.product_mass == pytest.approx(
        mat.PRODUCT_REFERENCE_DENSITY * field.charge_radius**3 / 3.0, rel=1.0e-14
    )


def test_validation_does_not_modify_caller_owned_arrays():
    control = _field()
    arrays = {
        "edges": control.cell_edges.copy(),
        "centres": control.cell_centres.copy(),
        "volume": control.cell_volume.copy(),
        "density": control.density.copy(),
        "mass": control.mass.copy(),
        "specific_energy": control.specific_energy.copy(),
        "pressure": control.pressure.copy(),
        "sound_speed": control.sound_speed.copy(),
        "velocity": control.velocity.copy(),
        "node_velocity": control.node_velocity.copy(),
        "material": control.material.copy(),
    }
    before = {name: value.copy() for name, value in arrays.items()}

    TwoMediumInitialField(
        time=control.time,
        charge_radius=control.charge_radius,
        outer_radius=control.outer_radius,
        cell_edges=arrays["edges"],
        cell_centres=arrays["centres"],
        cell_volume=arrays["volume"],
        density=arrays["density"],
        velocity=arrays["velocity"],
        specific_energy=arrays["specific_energy"],
        pressure=arrays["pressure"],
        sound_speed=arrays["sound_speed"],
        material=arrays["material"],
        mass=arrays["mass"],
        node_velocity=arrays["node_velocity"],
        product_eos=control.product_eos,
        air_gamma=control.air_gamma,
        product_state=dict(control.product_state),
        air_state=dict(control.air_state),
        metadata=dict(control.metadata),
    )

    for name, value in arrays.items():
        assert np.array_equal(value, before[name]), name


# --------------------------------------------------------------------------
# 13. EOS 一致性：压力与声速必须由本对象自带的 EOS 参数重算得到
# --------------------------------------------------------------------------


def _scaled(array, mask, factor):
    out = array.copy()
    out[mask] *= factor
    return out


def _custom_jwl() -> JWLParameters:
    """一套明确非默认、但在参考态仍给出正压力与正声速的 JWL 参数。"""
    base = mat.product_jwl()
    return dataclasses.replace(base, A=base.A * 1.10, B=base.B * 0.90, omega=base.omega * 1.05)


def test_eos_parameters_used_by_the_factory_are_stored_on_the_object():
    field = _field()

    assert isinstance(field.product_eos, JWLParameters)
    assert field.product_eos == mat.product_jwl()
    assert field.air_gamma == mat.AIR_GAMMA
    # 存的参数就是重算压力/声速时用的参数。
    n = field.n_inside
    assert np.allclose(
        jwl_pressure_numpy(field.density[:n], field.specific_energy[:n], field.product_eos),
        field.pressure[:n],
        rtol=1.0e-15,
        atol=0.0,
    )


def test_mutant_doubled_product_pressure_is_rejected():
    control = _field()
    replacement = _scaled(control.pressure, control.product_mask, 2.0)
    assert replacement[control.product_mask][0] == 2.0 * control.pressure[0]
    with pytest.raises(TwoMediumInitialError, match="product pressure does not match the general JWL"):
        dataclasses.replace(control, pressure=replacement)


def test_mutant_doubled_air_pressure_is_rejected():
    control = _field()
    replacement = _scaled(control.pressure, control.air_mask, 2.0)
    assert replacement[control.air_mask][0] == 2.0 * control.pressure[-1]
    with pytest.raises(
        TwoMediumInitialError, match="air pressure does not match the ideal gas"
    ):
        dataclasses.replace(control, pressure=replacement)


def test_mutant_doubled_product_sound_speed_is_rejected():
    control = _field()
    replacement = _scaled(control.sound_speed, control.product_mask, 2.0)
    assert replacement[control.product_mask][0] == 2.0 * control.sound_speed[0]
    with pytest.raises(
        TwoMediumInitialError,
        match="product sound_speed does not match the general JWL",
    ):
        dataclasses.replace(control, sound_speed=replacement)


def test_mutant_doubled_air_sound_speed_is_rejected():
    control = _field()
    replacement = _scaled(control.sound_speed, control.air_mask, 2.0)
    assert replacement[control.air_mask][0] == 2.0 * control.sound_speed[-1]
    with pytest.raises(
        TwoMediumInitialError, match="air sound_speed does not match the ideal gas"
    ):
        dataclasses.replace(control, sound_speed=replacement)


def test_legal_custom_jwl_parameters_are_accepted_and_checked_against_themselves():
    jwl = _custom_jwl()
    assert jwl != mat.product_jwl()
    field = _field(jwl=jwl)

    assert field.product_eos == jwl
    n = field.n_inside
    expected = jwl_pressure_numpy(
        field.density[:n], field.specific_energy[:n], jwl
    )
    # 与自定义参数一致，但与全局默认参数不一致。
    assert np.allclose(field.pressure[:n], expected, rtol=1.0e-15, atol=0.0)
    with_default = jwl_pressure_numpy(
        field.density[:n], field.specific_energy[:n], mat.product_jwl()
    )
    assert not np.allclose(field.pressure[:n], with_default, rtol=1.0e-6, atol=0.0)
    # 用默认参数冒充该对象的 EOS 必须被拒绝。
    with pytest.raises(TwoMediumInitialError, match="general JWL"):
        dataclasses.replace(field, product_eos=mat.product_jwl())


def test_mutant_swapping_in_a_different_jwl_parameter_set_is_rejected():
    jwl = _custom_jwl()
    control = _field()
    assert control.product_eos == mat.product_jwl()
    with pytest.raises(TwoMediumInitialError, match="general JWL"):
        dataclasses.replace(control, product_eos=jwl)


def test_legal_custom_air_gamma_is_accepted_and_checked_against_itself():
    field = _field(air_gamma=1.35)

    assert field.air_gamma == 1.35
    n = field.n_inside
    expected_p = ideal_gas_pressure_numpy(
        field.density[n:], field.specific_energy[n:], 1.35
    )
    expected_c = ideal_gas_sound_speed_numpy(
        field.density[n:], field.specific_energy[n:], 1.35
    )
    assert np.allclose(field.pressure[n:], expected_p, rtol=1.0e-15, atol=0.0)
    assert np.allclose(field.sound_speed[n:], expected_c, rtol=1.0e-15, atol=0.0)
    # 空气状态仍与给定的环境压力一致。
    assert field.air_state["pressure"] == pytest.approx(mat.AIR_PRESSURE, rel=1.0e-14)
    with pytest.raises(TwoMediumInitialError, match="ideal gas with gamma = 1.4"):
        dataclasses.replace(field, air_gamma=1.4)


def test_explicit_product_specific_energy_is_used_as_j_per_kg_without_another_division():
    energy = 1.0e6
    field = _field(product_specific_energy=energy)
    n = field.n_inside

    assert np.all(field.specific_energy[:n] == energy)
    assert np.all(field.specific_energy[:n] != energy / mat.PRODUCT_REFERENCE_DENSITY)
    assert field.product_state["specific_energy"] == energy
    assert np.allclose(
        field.pressure[:n],
        jwl_pressure_numpy(field.density[:n], field.specific_energy[:n], field.product_eos),
        rtol=1.0e-15,
        atol=0.0,
    )
    assert np.allclose(
        field.sound_speed[:n],
        jwl_sound_speed_numpy(field.density[:n], field.specific_energy[:n], field.product_eos),
        rtol=1.0e-15,
        atol=0.0,
    )


def test_legal_noop_replace_still_passes_the_eos_checks():
    control = _field()
    twin = dataclasses.replace(
        control,
        pressure=control.pressure.copy(),
        sound_speed=control.sound_speed.copy(),
        product_eos=control.product_eos,
        air_gamma=control.air_gamma,
    )
    assert np.array_equal(twin.pressure, control.pressure)
    assert np.array_equal(twin.sound_speed, control.sound_speed)


# --------------------------------------------------------------------------
# 14. 自定义 JWL 时，默认初始密度来自实际 EOS 参数集的 rho0
# --------------------------------------------------------------------------


def _custom_rho0(rho0: float = 2000.0) -> JWLParameters:
    params = dataclasses.replace(mat.product_jwl(), rho0=rho0)
    params.validate()
    return params


def test_default_material_without_density_keeps_the_reference_state():
    field = _field()

    assert field.density[field.product_mask][0] == mat.PRODUCT_REFERENCE_DENSITY
    assert field.product_eos.rho0 == mat.PRODUCT_REFERENCE_DENSITY
    assert field.specific_energy[0] == pytest.approx(
        mat.PRODUCT_VOLUME_ENERGY / mat.PRODUCT_REFERENCE_DENSITY, rel=1.0e-14
    )
    assert field.metadata["product_density_source"].startswith("product_eos.rho0")


def test_custom_jwl_rho0_becomes_the_default_initial_density():
    jwl = _custom_rho0(2000.0)
    assert jwl.rho0 != mat.PRODUCT_REFERENCE_DENSITY
    field = _field(jwl=jwl)
    n = field.n_inside

    # 修前这里会是全局默认 1630，而比内能却按 E0/2000 换算。
    assert np.all(field.density[:n] == 2000.0)
    assert field.product_state["density"] == 2000.0
    assert field.metadata["product_density_kg_m3"] == 2000.0
    assert field.metadata["eos_reference_density_kg_m3"] == 2000.0
    assert field.metadata["product_density_source"] == (
        "product_eos.rho0 (EOS reference density)"
    )
    # 比内能仍由同一个 rho0 换算，二者不再自相矛盾。
    assert np.allclose(
        field.specific_energy[:n], mat.PRODUCT_VOLUME_ENERGY / 2000.0, rtol=1.0e-14
    )


def test_custom_jwl_rho0_with_explicit_density_keeps_both_roles_separate():
    jwl = _custom_rho0(2000.0)
    field = _field(jwl=jwl, product_density=1800.0)
    n = field.n_inside

    # 初始密度用调用者的值……
    assert np.all(field.density[:n] == 1800.0)
    assert field.metadata["product_density_kg_m3"] == 1800.0
    assert field.metadata["product_density_source"].startswith("user-provided")
    # ……但 e_p0 = E0/rho_ref 的 rho_ref 仍是 EOS 参数集的 rho0。
    assert field.metadata["eos_reference_density_kg_m3"] == 2000.0
    assert np.allclose(
        field.specific_energy[:n], mat.PRODUCT_VOLUME_ENERGY / 2000.0, rtol=1.0e-14
    )
    assert not np.allclose(
        field.specific_energy[:n], mat.PRODUCT_VOLUME_ENERGY / 1800.0, rtol=1.0e-6
    )


def test_explicit_density_with_default_jwl_does_not_rescale_the_reference_energy():
    field = _field(product_density=1500.0)
    n = field.n_inside

    assert np.all(field.density[:n] == 1500.0)
    assert np.allclose(
        field.specific_energy[:n],
        mat.PRODUCT_VOLUME_ENERGY / mat.PRODUCT_REFERENCE_DENSITY,
        rtol=1.0e-14,
    )
    assert field.metadata["eos_reference_density_kg_m3"] == mat.PRODUCT_REFERENCE_DENSITY


@pytest.mark.parametrize(
    "jwl_rho0, density",
    [(1630.0, 1630.0), (2000.0, 2000.0), (2000.0, 1800.0), (1630.0, 1500.0)],
)
def test_density_choices_still_satisfy_the_eos_and_the_mass_relation(jwl_rho0, density):
    jwl = _custom_rho0(jwl_rho0)
    field = _field(jwl=jwl, product_density=density)
    mask = field.product_mask

    assert np.all(field.density[mask] == density)
    assert np.allclose(
        field.pressure[mask],
        jwl_pressure_numpy(field.density[mask], field.specific_energy[mask], jwl),
        rtol=1.0e-15,
        atol=0.0,
    )
    assert np.allclose(
        field.sound_speed[mask],
        jwl_sound_speed_numpy(field.density[mask], field.specific_energy[mask], jwl),
        rtol=1.0e-15,
        atol=0.0,
    )
    assert np.allclose(
        field.mass[mask], density * field.cell_volume[mask], rtol=0.0, atol=0.0
    )
    assert field.product_mass == pytest.approx(
        density * field.charge_radius**3 / 3.0, rel=1.0e-14
    )


def test_explicit_specific_energy_still_wins_over_the_reference_conversion():
    jwl = _custom_rho0(2000.0)
    field = _field(jwl=jwl, product_density=1800.0, product_specific_energy=5.0e5)
    n = field.n_inside

    assert np.all(field.specific_energy[:n] == 5.0e5)
    assert np.any(field.specific_energy[:n] != mat.PRODUCT_VOLUME_ENERGY / 2000.0)
    assert np.allclose(
        field.pressure[:n],
        jwl_pressure_numpy(field.density[:n], field.specific_energy[:n], jwl),
        rtol=1.0e-15,
        atol=0.0,
    )
    assert field.metadata["product_density_kg_m3"] == 1800.0
