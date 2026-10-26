"""Time-origin consistency between the planar-startup field and its scales.

The new chain must satisfy

    s = 0  <=>  tau = 0  <=>  t_phys = field.epsilon

with ``tau = t_phys - field.epsilon`` and no additional ``R_0/D``.  The old
scheme-B objects keep their own origin ``T_h = R_0/D + epsilon(R_0)``; the two
conventions are told apart by type, never by a value that happens to be zero.
"""

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")

from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.similitude import (
    TIME_CONVENTION_PLANAR_STARTUP,
    TIME_CONVENTION_SCHEME_B,
    PlanarStartupScales,
    SimilitudeError,
    SimilitudeScales,
    handover_time,
    start_delay,
)
from spherical_blast_pinn.physics.two_medium_initial import (
    build_two_medium_initial_field,
)
from spherical_blast_pinn.physics.two_medium_startup_planar import (
    PlanarStartupConfig,
    build_planar_startup_field,
)
from spherical_blast_pinn.sampling.batch import SampleBatch
from spherical_blast_pinn.sampling.planar_startup_initial import (
    build_planar_startup_initial_batch,
)

R0 = 0.05
DELTA = 1.0 / 200.0
TIME_SCALE = 2.0


def _field(*, radius=R0, delta=DELTA):
    initial = build_two_medium_initial_field(
        charge_radius=radius, outer_radius=2.0 * radius
    )
    config = PlanarStartupConfig(
        delta=delta,
        inner_cells=8,
        fan_cells=16,
        plateau_cells=2,
        air_cells=4,
        ambient_cells=8,
        dense_samples=512,
    )
    return build_planar_startup_field(initial, config=config)


@pytest.fixture(scope="module")
def field():
    return _field()


@pytest.fixture(scope="module")
def scales(field):
    return PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)


@pytest.fixture(scope="module")
def scheme_b():
    return SimilitudeScales.for_radius(
        R0, time_scale=TIME_SCALE, detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE
    )


def test_s_zero_is_the_field_epsilon_in_both_directions(field, scales):
    assert scales.startup_epsilon == field.epsilon
    assert scales.start_time() == field.epsilon
    # forward: s=0 -> t_phys=epsilon, tau=0
    assert float(scales.physical_time_from_network(0.0)) == field.epsilon
    assert float(scales.pinn_time_from_physical(field.epsilon)) == 0.0
    # inverse: t_phys=epsilon -> tau=0 -> s=0
    assert float(scales.network_time_from_physical(field.epsilon)) == 0.0
    assert float(scales.network_time(0.0)) == 0.0


def test_positive_times_round_trip_and_shift_only_once(scales):
    seconds_per_unit_s = TIME_SCALE * scales.charge_radius / mat.REFERENCE_VELOCITY
    previous = None
    for s_value in (0.0, 0.1, 0.5, 1.0, 2.5):
        physical = float(scales.physical_time_from_network(s_value))
        assert float(scales.network_time_from_physical(physical)) == pytest.approx(
            s_value, rel=1.0e-12, abs=1.0e-15
        )
        assert physical - scales.start_time() == pytest.approx(
            float(scales.pinn_time_from_network(s_value)), rel=1.0e-12, abs=1.0e-18
        )
        if previous is not None:
            ds, previous_physical = previous
            assert physical - previous_physical == pytest.approx(
                (s_value - ds) * seconds_per_unit_s, rel=1.0e-12
            )
        previous = (s_value, physical)
    # the shift is epsilon, not the scheme-B handover time
    assert scales.start_time() < handover_time(
        R0, detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE
    )


def test_a_smaller_delta_moves_the_origin_to_the_new_epsilon():
    fine = _field(delta=DELTA)
    thick = _field(delta=2.0 * DELTA)
    fine_scales = PlanarStartupScales.for_startup_field(fine, time_scale=TIME_SCALE)
    thick_scales = PlanarStartupScales.for_startup_field(thick, time_scale=TIME_SCALE)
    assert thick.epsilon > fine.epsilon
    assert fine_scales.start_time() == fine.epsilon
    assert thick_scales.start_time() == thick.epsilon
    assert thick_scales.start_time() != fine_scales.start_time()
    # the old fixed delay epsilon(R_0)=k*epsilon_ref is not used by either
    assert fine_scales.start_time() != pytest.approx(start_delay(R0))
    assert thick_scales.start_time() != pytest.approx(start_delay(R0))


def test_each_charge_radius_uses_its_own_epsilon():
    small = _field(radius=0.05)
    large = _field(radius=0.08)
    small_scales = PlanarStartupScales.for_startup_field(small, time_scale=TIME_SCALE)
    large_scales = PlanarStartupScales.for_startup_field(large, time_scale=TIME_SCALE)
    assert small_scales.start_time() == small.epsilon
    assert large_scales.start_time() == large.epsilon
    assert large_scales.start_time() != small_scales.start_time()
    # no reference-radius time is reused: neither origin is R_0/0.05 * epsilon_ref
    for radius, scales in ((0.05, small_scales), (0.08, large_scales)):
        reference_origin = (
            radius / mat.DETONATION_VELOCITY_REFERENCE + start_delay(radius)
        )
        assert scales.start_time() != pytest.approx(reference_origin)


def test_batch_diagnostics_equal_the_scale_conversions(field, scales):
    batch = build_planar_startup_initial_batch(
        field, scales=scales, n_ic_product=8, n_ic_air=8
    )
    expected_physical = float(scales.physical_time_from_network(0.0))
    expected_tau = float(scales.pinn_time_from_physical(expected_physical))
    expected_s = float(scales.network_time_from_physical(expected_physical))
    for prefix in ("", "_product", "_air"):
        assert batch.diagnostics[f"initial_physical_time{prefix}_s"] == expected_physical
        assert batch.diagnostics[f"initial_tau{prefix}_s"] == expected_tau
        # the network time is dimensionless: no seconds suffix
        assert batch.diagnostics[f"initial_network_time{prefix}"] == expected_s
    assert expected_physical == field.epsilon
    assert expected_tau == 0.0
    assert expected_s == 0.0
    assert batch.diagnostics["time_convention"] == TIME_CONVENTION_PLANAR_STARTUP
    assert batch.diagnostics["similitude_start_time_s"] == field.epsilon


def test_planar_scales_reject_missing_or_fake_origins(field):
    with pytest.raises(SimilitudeError, match="for_startup_field"):
        PlanarStartupScales.for_radius(
            R0, time_scale=TIME_SCALE, detonation_velocity=None
        )
    with pytest.raises(SimilitudeError, match="charge_radius"):
        PlanarStartupScales.for_startup_field(object(), time_scale=TIME_SCALE)
    with pytest.raises(SimilitudeError, match="epsilon"):
        PlanarStartupScales.for_startup_field(
            type("NoEpsilon", (), {"charge_radius": R0}), time_scale=TIME_SCALE
        )
    good = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    for bad_epsilon in (float("nan"), 0.0, -1.0e-6, float("inf")):
        with pytest.raises(SimilitudeError, match="startup_epsilon"):
            dataclasses.replace(good, startup_epsilon=bad_epsilon)
    with pytest.raises(SimilitudeError, match="detonation velocity"):
        dataclasses.replace(
            good, detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE
        )


def test_new_origin_does_not_depend_on_the_detonation_velocity(field, scales):
    assert scales.detonation_velocity is None
    # the origin is unchanged while the card velocity varies in the old path
    assert scales.start_time() == field.epsilon
    with pytest.raises(SimilitudeError, match="detonation velocity"):
        scales.transit_time()
    with pytest.raises(SimilitudeError, match="start_delay"):
        scales.start_delay()
    # old path still refuses an inadmissible velocity
    with pytest.raises(SimilitudeError, match="detonation velocity"):
        SimilitudeScales.for_radius(
            R0, time_scale=TIME_SCALE, detonation_velocity=0.0
        ).start_time()


def test_scheme_b_still_uses_its_own_origin(scheme_b):
    expected = handover_time(
        R0, detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE
    )
    assert scheme_b.time_convention == TIME_CONVENTION_SCHEME_B
    assert scheme_b.start_time() == pytest.approx(expected, rel=0, abs=0.0)
    assert float(scheme_b.physical_time_from_network(0.0)) == pytest.approx(
        expected, rel=0, abs=0.0
    )
    assert float(scheme_b.network_time_from_physical(expected)) == 0.0
    assert float(scheme_b.pinn_time_from_physical(expected)) == 0.0
    # the two conventions are not interchangeable for the same radius
    planar = PlanarStartupScales.for_startup_field(_field(), time_scale=TIME_SCALE)
    assert planar.start_time() != pytest.approx(scheme_b.start_time())


def test_sample_batch_binds_the_velocity_to_the_convention(scheme_b):
    with pytest.raises(ValueError, match="planar-startup time convention"):
        SampleBatch(
            case_id="x",
            charge_radius=R0,
            detonation_velocity=None,
            scales=scheme_b,
            pinn_time_min=0.0,
            pinn_time_max=1.0,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        SampleBatch(
            case_id="x",
            charge_radius=R0,
            detonation_velocity=-1.0,
            scales=scheme_b,
            pinn_time_min=0.0,
            pinn_time_max=1.0,
        )
    planar = PlanarStartupScales.for_startup_field(_field(), time_scale=TIME_SCALE)
    with pytest.raises(ValueError, match="must not carry a detonation velocity"):
        SampleBatch(
            case_id="x",
            charge_radius=R0,
            detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
            scales=planar,
            pinn_time_min=0.0,
            pinn_time_max=1.0,
        )
