"""New-startup sampler: explicit inputs, time convention, no data supervision."""

import dataclasses
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spherical_blast_pinn.losses.objective import contact_pressure_scale
from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.similitude import (
    TIME_CONVENTION_PLANAR_STARTUP,
    PlanarStartupScales,
    SimilitudeScales,
)
from spherical_blast_pinn.physics.two_medium_initial import (
    AIR,
    PRODUCT,
    build_two_medium_initial_field,
)
from spherical_blast_pinn.physics.two_medium_startup_planar import (
    PlanarStartupConfig,
    PlanarStartupField,
    build_planar_startup_field,
)
from spherical_blast_pinn.sampling.batch import restrict_to_time_window
from spherical_blast_pinn.sampling.sampler import (
    RadiusSampler,
    SamplingConfig,
    SamplingError,
)

R0 = 0.05
TAU_END = 1.0e-6
TIME_SCALE = 2.0


@pytest.fixture(scope="module")
def field() -> PlanarStartupField:
    initial = build_two_medium_initial_field(charge_radius=R0, outer_radius=2.0 * R0)
    return build_planar_startup_field(
        initial,
        config=PlanarStartupConfig(
            delta=1.0 / 200.0,
            inner_cells=8,
            fan_cells=16,
            plateau_cells=2,
            air_cells=4,
            ambient_cells=8,
            dense_samples=512,
        ),
    )


@pytest.fixture(scope="module")
def scales(field) -> PlanarStartupScales:
    return PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)


def _config(**overrides) -> SamplingConfig:
    base = dict(
        n_pde_product=12, n_pde_air=10, n_ic_product=6, n_ic_air=5,
        n_contact=7, n_shock=4,
    )
    base.update(overrides)
    return SamplingConfig(**base)


def _sampler(field, scales, *, seed=0, config=None) -> RadiusSampler:
    return RadiusSampler(
        field=field,
        scales=scales,
        tau_end=TAU_END,
        config=config or _config(),
        seed=seed,
    )


def test_batch_point_counts_and_tau_range(field, scales):
    sampler = _sampler(field, scales)
    batch = sampler.batch()
    assert batch.case_id.startswith("planar_startup:")
    assert batch.scales is scales
    assert batch.pinn_time_min == 0.0
    assert batch.pinn_time_max == TAU_END
    assert batch.detonation_velocity is None
    assert batch.scales.time_convention == TIME_CONVENTION_PLANAR_STARTUP

    assert batch.s_product.shape == (12, 1)
    assert batch.s_air.shape == (10, 1)
    assert batch.s_contact.shape == (7, 1)
    assert batch.s_shock.shape == (4, 1)
    # IC groups carry the requested draws plus the adapter's anchors.
    assert batch.xi_ic_product.shape[0] >= 6
    assert batch.eta_ic_air.shape[0] >= 5

    s_max = float(scales.network_time(TAU_END))
    for values in (batch.s_product, batch.s_air, batch.s_contact, batch.s_shock):
        assert float(values.min()) >= 0.0
        assert float(values.max()) <= s_max + 1.0e-14
    assert float(scales.physical_time_from_network(0.0)) == field.epsilon
    assert batch.diagnostics["initial_physical_time_s"] == field.epsilon
    assert batch.diagnostics["tau_end_s"] == TAU_END
    assert batch.diagnostics["window_tau_max_s"] == TAU_END


def test_initial_condition_is_the_adapter_sampling(field, scales):
    batch = _sampler(field, scales).batch()
    contact = float(field.contact_radius)
    shock = float(field.shock_radius)

    product_radii = contact * batch.xi_ic_product.detach().reshape(-1).numpy()
    air_radii = contact + (shock - contact) * batch.eta_ic_air.detach().reshape(-1).numpy()
    product = field.sample_product(product_radii)
    air = field.sample_air(air_radii)
    assert np.all(product["material"] == PRODUCT)
    assert np.all(air["material"] == AIR)

    for name, expected in (
        ("rho_ic_product", product["density"]),
        ("u_ic_product", product["velocity"]),
        ("e_ic_product", product["specific_energy"]),
    ):
        stored = getattr(batch, name).detach().reshape(-1).numpy()
        np.testing.assert_allclose(stored, np.asarray(expected, dtype=float), rtol=1.0e-14, atol=0.0)
    for name, expected in (
        ("rho_ic_air", air["density"]),
        ("u_ic_air", air["velocity"]),
        ("e_ic_air", air["specific_energy"]),
    ):
        stored = getattr(batch, name).detach().reshape(-1).numpy()
        np.testing.assert_allclose(stored, np.asarray(expected, dtype=float), rtol=1.0e-14, atol=0.0)
    # The adapter keeps the one-sided contact/shock anchors.
    assert float(product_radii[0]) == 0.0
    assert float(product_radii[-1]) == contact
    assert float(air_radii[0]) == contact
    assert float(air_radii[-1]) == shock


def test_epsilon_is_the_early_time_scale_and_s_zero_is_the_origin(field, scales):
    sampler = _sampler(field, scales)
    expected = scales.reference_velocity * field.epsilon / scales.charge_radius
    assert sampler.hat_time_epsilon == pytest.approx(expected, rel=1.0e-15)
    # hat_time is a duration: using epsilon does not add the time origin again.
    assert sampler.hat_time_epsilon == pytest.approx(scales.hat_time(field.epsilon))
    assert float(scales.physical_time_from_network(0.0)) == field.epsilon
    assert float(scales.network_time_from_physical(field.epsilon)) == 0.0


def test_no_data_supervision_is_an_empty_batch_not_a_fake_case(field, scales):
    batch = _sampler(field, scales).batch()
    for name in (
        "s_data_product", "r_data_product", "rho_data_product", "u_data_product",
        "e_data_product", "lo_data_product", "hi_data_product", "coverage_data_product",
        "s_data_air", "r_data_air", "rho_data_air", "u_data_air", "e_data_air",
        "lo_data_air", "hi_data_air", "coverage_data_air",
        "s_data_contact", "contact_data", "s_data_shock", "shock_data",
    ):
        assert getattr(batch, name) is None, name
    assert math.isnan(batch.geometry_delta_contact)
    assert math.isnan(batch.geometry_delta_shock)
    assert "disabled" in batch.diagnostics["data_supervision"]


def test_real_data_supervision_is_explicitly_unsupported(field, scales):
    with pytest.raises(SamplingError, match="real-data supervision is not supported"):
        RadiusSampler(
            field=field, scales=scales, tau_end=TAU_END, case=object()
        )


def test_construction_rejects_wrong_or_incomplete_inputs(field, scales):
    with pytest.raises(SamplingError, match="tau_end"):
        RadiusSampler(field=field, scales=scales, tau_end=0.0)
    with pytest.raises(SamplingError, match="tau_end"):
        RadiusSampler(field=field, scales=scales, tau_end=float("nan"))
    with pytest.raises(SamplingError, match="PlanarStartupField"):
        RadiusSampler(field=object(), scales=scales, tau_end=TAU_END)
    scheme_b = SimilitudeScales.for_radius(
        R0, time_scale=TIME_SCALE, detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE
    )
    with pytest.raises(SamplingError, match="PlanarStartupScales"):
        RadiusSampler(field=field, scales=scheme_b, tau_end=TAU_END)
    with pytest.raises(SamplingError, match="R_0"):
        RadiusSampler(
            field=field,
            scales=dataclasses.replace(scales, charge_radius=0.08),
            tau_end=TAU_END,
        )
    with pytest.raises(SamplingError, match="epsilon"):
        RadiusSampler(
            field=field,
            scales=dataclasses.replace(scales, startup_epsilon=2.0 * field.epsilon),
            tau_end=TAU_END,
        )
    with pytest.raises(SamplingError, match="n_ic_product and n_ic_air"):
        _sampler(field, scales, config=_config(n_ic_product=0))


def test_time_window_expands_and_never_shrinks(field, scales):
    sampler = _sampler(field, scales)
    first = sampler.set_time_window(0.25)
    assert first["pinn_time_max"] == pytest.approx(0.25 * TAU_END)
    assert sampler.window_history == [1.0, 0.25]
    sampler.set_time_window(0.75)
    assert sampler.window_pinn_time_max == pytest.approx(0.75 * TAU_END)
    with pytest.raises(SamplingError, match="shrink"):
        sampler.set_time_window(0.5)
    # rollback may restore an earlier value, including the history
    sampler.restore_time_window(1.0, history=[1.0, 0.25])
    assert sampler.window_fraction == 1.0
    assert sampler.window_history == [1.0, 0.25]


def test_rng_state_round_trip_reproduces_the_batch(field, scales):
    first = _sampler(field, scales, seed=11)
    batch_a = first.batch()
    state = first.rng_state()
    batch_b = first.batch()

    second = _sampler(field, scales, seed=999)
    second.set_rng_state(state)
    batch_c = second.batch()

    names = (
        "s_product", "xi_product", "s_air", "eta_air", "s_contact", "s_shock",
        "xi_ic_product", "rho_ic_product", "u_ic_product", "e_ic_product",
        "eta_ic_air", "rho_ic_air", "u_ic_air", "e_ic_air",
    )
    for name in names:
        assert torch.equal(getattr(batch_b, name), getattr(batch_c, name)), name
    assert not torch.equal(batch_a.s_product, batch_b.s_product)


def test_restrict_to_time_window_keeps_the_initial_condition(field, scales):
    batch = _sampler(
        field, scales, config=_config(n_pde_product=200, n_contact=200)
    ).batch()
    # restrict_to_time_window takes the dimensionless s bound, not seconds.
    cutoff = 0.5 * float(scales.network_time(batch.pinn_time_max))
    cut = restrict_to_time_window(batch, cutoff)
    assert 0 < cut.s_product.shape[0] < batch.s_product.shape[0]
    assert float(cut.s_product.max()) <= cutoff + 1.0e-15
    # the batch's tau bound follows the same s -> tau conversion, not seconds
    assert cut.pinn_time_max == pytest.approx(
        float(scales.pinn_time_from_network(cutoff)), rel=1.0e-14
    )
    assert cut.xi_ic_product.shape == batch.xi_ic_product.shape
    assert torch.equal(cut.rho_ic_air, batch.rho_ic_air)


def test_contact_pressure_scale_comes_from_the_validated_field(field):
    value = contact_pressure_scale(field)
    assert value == field.solution.p_star
    assert math.isfinite(value) and value > 0.0
    with pytest.raises(TypeError, match="planar startup field"):
        contact_pressure_scale(None)
