import json
import math
from dataclasses import replace

import numpy as np
import pytest

from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.eos import (
    JWLParameters,
    ideal_gas_pressure_numpy,
    ideal_gas_sound_speed_numpy,
    jwl_pressure_numpy,
)
from spherical_blast_pinn.physics.two_medium_initial import (
    AIR,
    PRODUCT,
    build_two_medium_initial_field,
)
from spherical_blast_pinn.physics.two_medium_startup_planar import (
    PLANAR_STARTUP_MODEL_ID,
    PlanarStartupConfig,
    PlanarStartupError,
    build_planar_startup_field,
)

R0 = 0.05
DELTA = 1.0 / 200.0


def _initial(**kwargs):
    return build_two_medium_initial_field(
        charge_radius=R0,
        outer_radius=40.0 * R0,
        **kwargs,
    )


def _config(**kwargs):
    base = dict(delta=DELTA, dense_samples=2048)
    base.update(kwargs)
    return PlanarStartupConfig(**base)


def _field(**kwargs):
    return build_planar_startup_field(_initial(), config=_config(**kwargs))


def test_zero_time_defaults_and_full_sphere_integrals():
    field = _initial()
    assert field.time == 0.0
    # The prompt gives these to 10/4 significant figures; compare at 1e-6 so
    # the test checks the formula, not the number of printed digits.
    assert field.product_state["specific_energy"] == pytest.approx(
        3.680981595e6, rel=1.0e-6
    )
    assert field.product_state["pressure"] == pytest.approx(
        8.220261897e9, rel=1.0e-6
    )
    assert field.product_state["sound_speed"] == pytest.approx(
        4059.1204, rel=1.0e-6
    )
    assert field.air_state["specific_energy"] == pytest.approx(
        206775.5102, rel=1.0e-6
    )
    assert field.air_state["sound_speed"] == pytest.approx(
        340.2856, rel=1.0e-6
    )
    assert field.full_sphere_product_mass == pytest.approx(
        0.8534660042, rel=1.0e-9
    )
    assert field.full_sphere_product_internal_energy == pytest.approx(
        3.141592654e6, rel=1.0e-9
    )
    assert np.all(field.velocity == 0.0)
    assert np.all(field.node_velocity == 0.0)
    assert field.cell_edges[field.n_inside] == R0
    assert np.all(field.material[: field.n_inside] == PRODUCT)
    assert np.all(field.material[field.n_inside :] == AIR)
    # Mass and geometry are per unit solid angle; the full-sphere numbers are
    # exactly 4*pi times the per-steradian ones.
    assert field.full_sphere_product_mass == pytest.approx(
        4.0 * math.pi * field.product_mass, rel=1.0e-15
    )


def test_zero_time_r0_cubed_scaling():
    small = build_two_medium_initial_field(
        charge_radius=0.05, outer_radius=2.0
    )
    large = build_two_medium_initial_field(
        charge_radius=0.10, outer_radius=4.0
    )
    assert large.full_sphere_product_mass == pytest.approx(
        8.0 * small.full_sphere_product_mass, rel=1.0e-12
    )
    assert large.full_sphere_product_internal_energy == pytest.approx(
        8.0 * small.full_sphere_product_internal_energy, rel=1.0e-12
    )
    assert large.product_state["pressure"] == pytest.approx(
        small.product_state["pressure"], rel=1.0e-12
    )
    assert large.product_state["sound_speed"] == pytest.approx(
        small.product_state["sound_speed"], rel=1.0e-12
    )
    assert large.cell_edges[large.n_inside] == 0.10


def test_epsilon_fronts_and_wave_positions_follow_the_planar_solution():
    field = _field()
    sol = field.solution
    epsilon = DELTA * R0 / (sol.shock_speed - sol.u_star)
    assert field.epsilon == pytest.approx(epsilon, rel=1.0e-14)
    assert field.time == field.epsilon
    assert field.pin_time == 0.0
    assert field.contact_radius == pytest.approx(
        R0 + sol.u_star * epsilon, rel=1.0e-14
    )
    assert field.shock_radius == pytest.approx(
        R0 + sol.shock_speed * epsilon, rel=1.0e-14
    )
    assert field.rarefaction_head_radius == pytest.approx(
        R0 + sol.rarefaction_head_speed * epsilon, rel=1.0e-14
    )
    assert field.rarefaction_tail_radius == pytest.approx(
        R0 + sol.rarefaction_tail_speed * epsilon, rel=1.0e-14
    )
    assert field.rarefaction_head_radius > 0.0
    assert field.shock_speed > field.contact_speed > 0.0
    assert field.air_layer_width / R0 == pytest.approx(DELTA, rel=1.0e-12)
    assert field.product_disturbed_length_over_r0 == pytest.approx(
        (sol.u_star - sol.rarefaction_head_speed) * epsilon / R0,
        rel=1.0e-12,
    )
    assert field.checks["passed"] is True
    assert field.reference_field is False
    assert field.training_qualified is False


def test_material_regions_and_one_sided_sampling():
    field = _field()
    assert np.all(np.diff(field.region) >= 0)
    assert np.all(field.material[: field.n_inside] == PRODUCT)
    assert np.all(field.material[field.n_inside :] == AIR)
    assert np.any(np.isclose(field.radii, field.contact_radius, rtol=0.0, atol=1.0e-15))
    assert np.any(np.isclose(field.radii, field.shock_radius, rtol=0.0, atol=1.0e-15))

    left = field.sample([field.contact_radius], side="left")
    right = field.sample([field.contact_radius], side="right")
    assert left["material"][0] == PRODUCT
    assert right["material"][0] == AIR
    assert left["density"][0] != right["density"][0]
    assert left["pressure"][0] == pytest.approx(right["pressure"][0], rel=1.0e-8)
    assert left["velocity"][0] == pytest.approx(right["velocity"][0], rel=0.0, abs=1.0e-10)

    post = field.sample([field.shock_radius], side="left")
    ambient = field.sample([field.shock_radius], side="right")
    assert post["material"][0] == AIR
    assert ambient["material"][0] == AIR
    assert post["pressure"][0] > ambient["pressure"][0]
    assert post["density"][0] > ambient["density"][0]
    assert post["velocity"][0] > 0.0
    assert ambient["velocity"][0] == 0.0

    product_contact = field.sample_product([field.contact_radius])
    air_contact = field.sample_air([field.contact_radius])
    assert product_contact["material"][0] == PRODUCT
    assert air_contact["material"][0] == AIR
    assert product_contact["pressure"][0] == pytest.approx(
        air_contact["pressure"][0], rel=1.0e-8
    )

    centre = field.sample([0.0])
    assert centre["velocity"][0] == 0.0
    assert centre["material"][0] == PRODUCT
    assert field.node_velocity[0] == 0.0
    assert field.velocity[0] == 0.0

    outer = field.sample([field.radii[-1]])
    assert outer["material"][0] == AIR
    assert outer["velocity"][0] == 0.0
    assert outer["pressure"][0] == pytest.approx(mat.AIR_PRESSURE, rel=1.0e-12)


def test_sampled_pressure_and_sound_speed_use_the_actual_eos():
    field = _field()
    radii = np.linspace(0.0, field.radii[-1], 200)
    states = field.sample(radii)
    product = states["material"] == PRODUCT
    air = ~product
    p_product = jwl_pressure_numpy(
        states["density"][product],
        states["specific_energy"][product],
        field.product_eos,
    )
    assert np.allclose(
        states["pressure"][product], p_product, rtol=1.0e-12, atol=0.0
    )
    p_air = ideal_gas_pressure_numpy(
        states["density"][air], states["specific_energy"][air], field.air_gamma
    )
    assert np.allclose(
        states["pressure"][air], p_air, rtol=1.0e-12, atol=0.0
    )
    c_air = ideal_gas_sound_speed_numpy(
        states["density"][air], states["specific_energy"][air], field.air_gamma
    )
    assert np.allclose(
        states["sound_speed"][air], c_air, rtol=1.0e-12, atol=0.0
    )
    assert np.all(states["density"] > 0.0)
    assert np.all(states["specific_energy"] > 0.0)
    assert np.all(states["sound_speed"] > 0.0)


def test_planar_and_spherical_integrals_and_residuals():
    field = _field()
    grid = field.grid_integrals()
    dense = field.dense_integrals()
    checks = field.checks
    assert checks["planar_mass_residual"] < 1.0e-6
    assert checks["planar_energy_residual"] < 1.0e-6
    assert checks["spherical_mass_residual"] < 1.0e-2
    assert checks["spherical_energy_residual"] < 1.0e-2
    assert checks["grid_vs_dense_spherical_mass_residual"] < 1.0e-4
    assert checks["grid_vs_dense_spherical_energy_residual"] < 1.0e-4
    assert grid["mass_product"] > 0.0 and grid["mass_air"] > 0.0
    assert dense["mass_product"] > 0.0 and dense["mass_air"] > 0.0
    # The planar solution conserves planar mass and energy much more tightly
    # than the spherical embedding; this is the expected geometric residual.
    assert checks["planar_mass_residual"] < checks["spherical_mass_residual"]
    assert checks["planar_energy_residual"] < checks["spherical_energy_residual"]


def test_delta_sensitivity_scales_epsilon_and_wave_lengths():
    ratios = [1.0, 0.5, 0.25]
    fields = []
    for ratio in ratios:
        delta = DELTA * ratio
        fields.append(
            _field(
                delta=delta,
                inner_cells=16,
                fan_cells=32,
                plateau_cells=2,
                air_cells=4,
                ambient_cells=8,
                dense_samples=1024,
            )
        )
    base = fields[0]
    for ratio, field in zip(ratios, fields):
        assert field.epsilon == pytest.approx(base.epsilon * ratio, rel=1.0e-9)
        assert (field.contact_radius - R0) == pytest.approx(
            (base.contact_radius - R0) * ratio, rel=1.0e-9
        )
        assert (field.shock_radius - R0) == pytest.approx(
            (base.shock_radius - R0) * ratio, rel=1.0e-9
        )
        assert field.air_layer_width / R0 == pytest.approx(
            base.air_layer_width / R0 * ratio, rel=1.0e-9
        )
        assert field.rarefaction_head_radius > 0.0
        assert field.checks["passed"] is True
        # The star state and shock speed are properties of the pressure jump,
        # not of the chosen epsilon; they must not drift with delta.
        assert field.solution.p_star == pytest.approx(
            base.solution.p_star, rel=1.0e-10
        )
        assert field.solution.u_star == pytest.approx(
            base.solution.u_star, rel=1.0e-10
        )
        assert field.solution.shock_speed == pytest.approx(
            base.solution.shock_speed, rel=1.0e-10
        )


def _assert_replace_rejected(field, **changes):
    with pytest.raises(PlanarStartupError):
        replace(field, **changes)


def test_consistency_validation_rejects_inconsistent_replacements():
    field = _field(dense_samples=1024)
    _assert_replace_rejected(field, delta=float("nan"))
    _assert_replace_rejected(field, delta=float("inf"))
    _assert_replace_rejected(field, delta=0.0)
    _assert_replace_rejected(field, delta=-DELTA)
    _assert_replace_rejected(field, charge_radius=0.06)
    _assert_replace_rejected(field, epsilon=field.epsilon * 1.01)
    _assert_replace_rejected(field, time=field.time + 1.0e-9)
    _assert_replace_rejected(field, pin_time=1.0e-9)
    _assert_replace_rejected(field, contact_radius=field.contact_radius + 1.0e-6)
    _assert_replace_rejected(field, shock_radius=field.shock_radius + 1.0e-6)
    _assert_replace_rejected(
        field, rarefaction_head_radius=field.rarefaction_head_radius + 1.0e-6
    )
    _assert_replace_rejected(
        field, rarefaction_tail_radius=field.rarefaction_tail_radius + 1.0e-6
    )
    _assert_replace_rejected(field, contact_speed=field.contact_speed + 1.0)
    _assert_replace_rejected(field, shock_speed=field.shock_speed + 1.0)
    _assert_replace_rejected(
        field, rarefaction_head_speed=field.rarefaction_head_speed + 1.0
    )
    _assert_replace_rejected(
        field, rarefaction_tail_speed=field.rarefaction_tail_speed + 1.0
    )
    _assert_replace_rejected(field, velocity=field.velocity + 100.0)

    node_velocity = field.node_velocity.copy()
    node_velocity[1:] += 100.0
    _assert_replace_rejected(field, node_velocity=node_velocity)

    _assert_replace_rejected(
        field, material=np.where(field.material == PRODUCT, AIR, PRODUCT).astype(np.int64)
    )
    region = field.region.copy()
    region[0] = 1
    _assert_replace_rejected(field, region=region)

    _assert_replace_rejected(field, density=field.density * 1.01)
    _assert_replace_rejected(field, specific_energy=field.specific_energy * 1.01)
    _assert_replace_rejected(field, pressure=field.pressure * 1.01)
    _assert_replace_rejected(field, sound_speed=field.sound_speed * 1.01)
    _assert_replace_rejected(field, cell_volume=field.cell_volume * 1.01)
    _assert_replace_rejected(field, mass=field.mass * 1.01)

    bad_initial = dict(field.initial_state)
    bad_initial["product_density"] = bad_initial["product_density"] + 1.0
    _assert_replace_rejected(field, initial_state=bad_initial)
    bad_initial = dict(field.initial_state)
    bad_initial["outer_radius"] = bad_initial["outer_radius"] + 0.1
    _assert_replace_rejected(field, initial_state=bad_initial)
    bad_initial = dict(field.initial_state)
    bad_initial.pop("air_pressure")
    _assert_replace_rejected(field, initial_state=bad_initial)

    changed_eos = replace(field.product_eos, A=field.product_eos.A * 1.01)
    _assert_replace_rejected(field, product_eos=changed_eos)
    _assert_replace_rejected(field, air_gamma=1.5)
    _assert_replace_rejected(field, solution=object())
    _assert_replace_rejected(
        field, solution=replace(field.solution, u_star=field.solution.u_star + 1.0)
    )


def test_noop_replace_and_custom_material_field_still_pass():
    field = _field(dense_samples=1024)
    noop = replace(field)
    for name in (
        "radii",
        "cell_volume",
        "density",
        "velocity",
        "specific_energy",
        "pressure",
        "sound_speed",
        "material",
        "region",
        "mass",
        "node_velocity",
    ):
        assert np.array_equal(getattr(noop, name), getattr(field, name))
    assert noop.epsilon == field.epsilon
    assert noop.contact_radius == field.contact_radius
    assert noop.shock_radius == field.shock_radius
    assert replace(field, delta=field.delta).checks["passed"] is True

    base = mat.product_jwl()
    custom = JWLParameters(
        A=base.A * 1.01,
        B=base.B,
        R1=base.R1,
        R2=base.R2,
        omega=base.omega,
        rho0=base.rho0,
        energy_shift=base.energy_shift,
    )
    initial = build_two_medium_initial_field(
        charge_radius=R0, outer_radius=2.0, jwl=custom
    )
    custom_field = build_planar_startup_field(initial, config=_config(dense_samples=1024))
    assert custom_field.checks["passed"] is True
    assert custom_field.product_eos.as_dict() == custom.as_dict()


def test_energy_normalization_keeps_global_and_adds_product_metric():
    field = _field(dense_samples=8192)
    checks = field.checks
    dense = field.dense_integrals()
    initial = field.integrals["initial"]
    delta_e = dense["total_energy"] - initial["initial_total_energy"]
    product_u = initial["initial_product_internal_energy"]

    assert product_u == pytest.approx(250000.0, rel=1.0e-9)
    assert checks["initial_product_internal_energy_per_sr"] == pytest.approx(product_u)
    assert checks["spherical_energy_change_signed_j_per_sr"] == pytest.approx(delta_e)
    assert checks["spherical_energy_change_abs_j_per_sr"] == pytest.approx(abs(delta_e))
    assert checks[
        "spherical_energy_change_abs_over_initial_product_internal"
    ] == pytest.approx(abs(delta_e) / product_u)
    # The prompt's rounded reference values.
    assert abs(delta_e) == pytest.approx(230.10, rel=1.0e-2)
    assert checks[
        "spherical_energy_change_abs_over_initial_product_internal"
    ] == pytest.approx(9.20e-4, rel=1.0e-2)
    # The original global metric and its definition are retained.
    assert checks["spherical_energy_residual"] == pytest.approx(
        abs(delta_e) / initial["initial_total_energy"]
    )
    report = field.as_dict()
    for key in (
        "spherical_energy_residual",
        "initial_product_internal_energy_per_sr",
        "spherical_energy_change_signed_j_per_sr",
        "spherical_energy_change_abs_j_per_sr",
        "spherical_energy_change_abs_over_initial_product_internal",
    ):
        assert key in report
    # Product mass is still reported separately.
    assert "spherical_product_mass_residual" in checks


def test_energy_metric_is_stable_to_far_outer_boundary():
    def build(outer):
        initial = build_two_medium_initial_field(
            charge_radius=R0, outer_radius=outer
        )
        return build_planar_startup_field(
            initial, config=_config(dense_samples=4096)
        )

    near = build(2.0)
    far = build(4.0)
    near_metric = near.checks[
        "spherical_energy_change_abs_over_initial_product_internal"
    ]
    far_metric = far.checks[
        "spherical_energy_change_abs_over_initial_product_internal"
    ]
    assert near.checks["initial_product_internal_energy_per_sr"] == pytest.approx(
        far.checks["initial_product_internal_energy_per_sr"], rel=1.0e-12
    )
    assert near_metric == pytest.approx(far_metric, rel=1.0e-8)
    # The original global metric is allowed to change with the outer boundary.
    assert not math.isclose(
        near.checks["spherical_energy_residual"],
        far.checks["spherical_energy_residual"],
        rel_tol=1.0e-2,
    )


def test_node_velocity_one_sided_rule():
    field = _field(dense_samples=1024)
    expected = np.empty_like(field.node_velocity)
    for index, radius in enumerate(field.radii):
        if radius <= field.contact_radius:
            expected[index] = field.sample_product(np.array([radius]))["velocity"][0]
        elif radius <= field.shock_radius:
            expected[index] = field.solution.u_star
        else:
            expected[index] = 0.0
    expected[0] = 0.0
    assert np.array_equal(field.node_velocity, expected)
    assert field.node_velocity[0] == 0.0
    assert np.all(field.node_velocity[field.radii > field.shock_radius] == 0.0)
    assert np.any(field.node_velocity > 0.0)


def test_handoff_report_cache_and_absent_legacy_fields():
    field = _field(dense_samples=1024)
    report = field.as_dict()
    assert report["model_id"] == PLANAR_STARTUP_MODEL_ID
    assert report["reference_field"] is False
    assert report["training_qualified"] is False
    assert "NOT APPLICABLE" in report["acceptance_status"]
    assert report["physical_time_s"] == field.epsilon
    assert report["pin_time_s"] == 0.0
    assert "start_time" not in report
    assert "T_h" not in report
    assert report["initial_state"]["product_density"] == pytest.approx(1630.0)
    assert report["initial_state"]["air_gamma"] == pytest.approx(1.4)
    assert report["riemann"]["p_star"] == pytest.approx(field.solution.p_star)
    assert report["cache_key"] == field.cache_key()
    same = _field(dense_samples=1024)
    assert same.cache_key() == field.cache_key()
    different = _field(delta=DELTA * 0.5, dense_samples=1024)
    assert different.cache_key() != field.cache_key()


def test_invalid_configuration_and_domain_fail_explicitly():
    with pytest.raises(PlanarStartupError):
        PlanarStartupConfig(delta=0.0).validate()
    with pytest.raises(PlanarStartupError):
        PlanarStartupConfig(dense_samples=100).validate()
    with pytest.raises(PlanarStartupError):
        PlanarStartupConfig(fan_cells=0).validate()
    tiny = build_two_medium_initial_field(
        charge_radius=R0, outer_radius=R0 * 1.001
    )
    with pytest.raises(PlanarStartupError):
        build_planar_startup_field(tiny, config=_config())


def test_cli_writes_independent_handoff(tmp_path):
    from spherical_blast_pinn.cli.startup import main

    code = main(
        [
            "--charge-radius",
            str(R0),
            "--delta",
            str(DELTA),
            "--outer-radius-factor",
            "2",
            "--inner-cells",
            "8",
            "--fan-cells",
            "16",
            "--plateau-cells",
            "2",
            "--air-cells",
            "4",
            "--ambient-cells",
            "8",
            "--dense-samples",
            "512",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    files = list(tmp_path.rglob("handoff.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text(encoding="utf-8"))
    assert report["model_id"] == PLANAR_STARTUP_MODEL_ID
    assert report["training_qualified"] is False
    assert report["checks"]["passed"] is True
    npz = files[0].with_name("handoff.npz")
    assert npz.is_file()
    data = np.load(npz)
    for key in (
        "radii",
        "density",
        "velocity",
        "specific_energy",
        "pressure",
        "sound_speed",
        "material",
        "region",
        "mass",
        "node_velocity",
    ):
        assert key in data
