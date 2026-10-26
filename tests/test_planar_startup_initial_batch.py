"""Contract tests for PlanarStartupField -> SampleBatch initial conditions."""

import dataclasses
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spherical_blast_pinn.losses.terms import initial_condition_loss
from spherical_blast_pinn.models.air_net import AirNet
from spherical_blast_pinn.models.coupled_model import CoupledModel
from spherical_blast_pinn.models.front_net import FrontNet
from spherical_blast_pinn.models.product_net import ProductNet
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
    build_planar_startup_field,
)
from spherical_blast_pinn.sampling.batch import SampleBatch
from spherical_blast_pinn.sampling.planar_startup_initial import (
    PlanarStartupBatchError,
    build_planar_startup_initial_batch,
    initial_radii_from_batch,
)

R0 = 0.05
DELTA = 1.0 / 200.0


@pytest.fixture(scope="module")
def field():
    initial = build_two_medium_initial_field(charge_radius=R0, outer_radius=2.0 * R0)
    config = PlanarStartupConfig(
        delta=DELTA,
        inner_cells=8,
        fan_cells=16,
        plateau_cells=2,
        air_cells=4,
        ambient_cells=8,
        dense_samples=512,
    )
    return build_planar_startup_field(initial, config=config)


def _scales(field):
    """The batch carries the startup field's own time origin, not scheme B."""
    return PlanarStartupScales.for_startup_field(field, time_scale=2.0)


def _sample_points(field):
    head = field.rarefaction_head_radius
    tail = field.rarefaction_tail_radius
    contact = field.contact_radius
    shock = field.shock_radius
    product = np.array([0.0, 0.5 * (head + tail), 0.5 * (tail + contact)])
    air = np.array([contact, 0.5 * (contact + shock), shock])
    return product, air


def _expected_product_radii(field, values):
    return np.unique(np.concatenate([np.asarray(values, dtype=float), [0.0, field.contact_radius]]))


def _expected_air_radii(field, values):
    return np.unique(
        np.concatenate(
            [np.asarray(values, dtype=float), [field.contact_radius, field.shock_radius]]
        )
    )


def test_batch_targets_and_inverse_radii_match_direct_sampling(field):
    product, air = _sample_points(field)
    batch = build_planar_startup_initial_batch(
        field,
        scales=_scales(field),
        product_radii=product,
        air_radii=air,
    )
    assert isinstance(batch, SampleBatch)
    assert batch.case_id
    assert batch.charge_radius == field.charge_radius
    assert batch.contact_ic == field.contact_radius
    assert batch.shock_ic == field.shock_radius
    assert batch.pinn_time_min == 0.0
    assert batch.pinn_time_max == 0.0

    expected_product = _expected_product_radii(field, product)
    expected_air = _expected_air_radii(field, air)
    product_radii, air_radii = initial_radii_from_batch(batch)
    np.testing.assert_array_equal(product_radii, expected_product)
    np.testing.assert_array_equal(air_radii, expected_air)

    direct_product = field.sample_product(expected_product)
    direct_air = field.sample_air(expected_air)
    for batch_name, direct_name in (
        ("xi_ic_product", "density"),
        ("rho_ic_product", "density"),
        ("u_ic_product", "velocity"),
        ("e_ic_product", "specific_energy"),
    ):
        stored = getattr(batch, batch_name).detach().reshape(-1).numpy()
        if batch_name == "xi_ic_product":
            expected = expected_product / field.contact_radius
        else:
            expected = np.asarray(direct_product[direct_name], dtype=float)
        assert np.allclose(stored, expected, rtol=1.0e-15, atol=0.0)
    for batch_name, direct_name in (
        ("eta_ic_air", None),
        ("rho_ic_air", "density"),
        ("u_ic_air", "velocity"),
        ("e_ic_air", "specific_energy"),
    ):
        stored = getattr(batch, batch_name).detach().reshape(-1).numpy()
        if direct_name is None:
            expected = (expected_air - field.contact_radius) / (
                field.shock_radius - field.contact_radius
            )
        else:
            expected = np.asarray(direct_air[direct_name], dtype=float)
        assert np.allclose(stored, expected, rtol=1.0e-15, atol=0.0)

    assert np.all(np.asarray(direct_product["material"]) == PRODUCT)
    assert np.all(np.asarray(direct_air["material"]) == AIR)
    assert batch.diagnostics["material_labels"] == {
        "product": int(PRODUCT),
        "air": int(AIR),
    }
    for name in (
        "xi_ic_product",
        "rho_ic_product",
        "u_ic_product",
        "e_ic_product",
        "eta_ic_air",
        "rho_ic_air",
        "u_ic_air",
        "e_ic_air",
    ):
        value = getattr(batch, name)
        assert value.dtype == torch.float64
        assert value.ndim == 2 and value.shape[1] == 1
        assert value.shape[0] == (expected_product.size if "product" in name else expected_air.size)


def test_product_core_fan_and_star_points_are_correct(field):
    head = field.rarefaction_head_radius
    tail = field.rarefaction_tail_radius
    contact = field.contact_radius
    product = np.array([0.0, 0.5 * (head + tail), 0.5 * (tail + contact)])
    air = np.array([0.5 * (contact + field.shock_radius)])
    batch = build_planar_startup_initial_batch(
        field,
        scales=_scales(field),
        product_radii=product,
        air_radii=air,
    )
    radii, _ = initial_radii_from_batch(batch)
    rho = batch.rho_ic_product.detach().reshape(-1).numpy()
    u = batch.u_ic_product.detach().reshape(-1).numpy()
    e = batch.e_ic_product.detach().reshape(-1).numpy()

    core = np.flatnonzero(radii == 0.0)
    fan = np.flatnonzero((radii > head) & (radii < tail))
    star = np.flatnonzero((radii > tail) & (radii < contact))
    assert core.size == 1 and fan.size == 1 and star.size == 1
    assert u[core[0]] == 0.0
    assert rho[core[0]] == field.initial_state["product_density"]
    assert e[core[0]] == field.initial_state["product_specific_energy"]

    xi_fan = (radii[fan[0]] - field.charge_radius) / field.epsilon
    fan_state = field.solution.isentrope.state_at_xi(xi_fan)
    assert rho[fan[0]] == pytest.approx(fan_state["rho"], rel=1.0e-15)
    assert u[fan[0]] == pytest.approx(fan_state["u"], rel=1.0e-15)
    assert e[fan[0]] == pytest.approx(fan_state["e"], rel=1.0e-15)

    assert rho[star[0]] == pytest.approx(field.solution.product_rho_star, rel=1.0e-15)
    assert u[star[0]] == pytest.approx(field.solution.u_star, rel=1.0e-15)
    assert e[star[0]] == pytest.approx(field.solution.product_e_star, rel=1.0e-15)

    _, air_radii = initial_radii_from_batch(batch)
    air_star = np.flatnonzero(
        (air_radii > contact) & (air_radii < field.shock_radius)
    )
    assert air_star.size == 1
    assert float(batch.rho_ic_air.detach().reshape(-1)[air_star[0]]) == pytest.approx(
        field.solution.air_rho_star, rel=1.0e-15
    )
    assert float(batch.u_ic_air.detach().reshape(-1)[air_star[0]]) == pytest.approx(
        field.solution.u_star, rel=1.0e-15
    )
    assert float(batch.e_ic_air.detach().reshape(-1)[air_star[0]]) == pytest.approx(
        field.solution.air_e_star, rel=1.0e-15
    )
    assert field.sample_air(air_radii[air_star])["material"][0] == AIR


def test_contact_sides_and_shock_are_one_sided(field):
    # Only interior points are supplied; the adapter must add the four anchors.
    batch = build_planar_startup_initial_batch(
        field,
        scales=_scales(field),
        product_radii=[0.5 * (field.rarefaction_head_radius + field.rarefaction_tail_radius)],
        air_radii=[0.5 * (field.contact_radius + field.shock_radius)],
    )
    product_radii, air_radii = initial_radii_from_batch(batch)
    assert product_radii[0] == 0.0
    assert product_radii[-1] == field.contact_radius
    assert air_radii[0] == field.contact_radius
    assert air_radii[-1] == field.shock_radius

    product_contact = field.sample_product(np.array([field.contact_radius]))
    air_contact = field.sample_air(np.array([field.contact_radius]))
    assert product_contact["material"][0] == PRODUCT
    assert air_contact["material"][0] == AIR
    assert float(product_contact["pressure"][0]) == pytest.approx(
        float(air_contact["pressure"][0]), rel=1.0e-8
    )
    assert float(product_contact["velocity"][0]) == pytest.approx(
        float(air_contact["velocity"][0]), rel=1.0e-10, abs=1.0e-10
    )
    assert float(product_contact["density"][0]) != pytest.approx(
        float(air_contact["density"][0]), rel=1.0e-3
    )

    post = field.sample_air(np.array([field.shock_radius]))
    ambient = field.sample(np.array([field.shock_radius]), side="right")
    batch_post = (
        float(batch.rho_ic_air.detach().reshape(-1)[-1]),
        float(batch.u_ic_air.detach().reshape(-1)[-1]),
        float(batch.e_ic_air.detach().reshape(-1)[-1]),
    )
    assert batch_post == pytest.approx(
        (
            float(post["density"][0]),
            float(post["velocity"][0]),
            float(post["specific_energy"][0]),
        ),
        rel=1.0e-15,
    )
    assert float(post["pressure"][0]) > float(ambient["pressure"][0])
    assert float(post["density"][0]) > float(ambient["density"][0])


def test_both_groups_use_the_same_epsilon_and_zero_network_time(field):
    batch = build_planar_startup_initial_batch(
        field, scales=_scales(field), n_ic_product=8, n_ic_air=8
    )
    assert field.time == field.epsilon
    assert field.pin_time == 0.0
    assert batch.pinn_time_min == 0.0
    assert batch.pinn_time_max == 0.0
    assert batch.diagnostics["initial_physical_time_s"] == field.epsilon
    assert batch.diagnostics["initial_physical_time_product_s"] == field.epsilon
    assert batch.diagnostics["initial_physical_time_air_s"] == field.epsilon
    assert batch.diagnostics["planar_startup_epsilon_s"] == field.epsilon
    assert batch.diagnostics["initial_tau_s"] == 0.0
    assert batch.diagnostics["initial_tau_product_s"] == 0.0
    assert batch.diagnostics["initial_tau_air_s"] == 0.0
    assert batch.diagnostics["initial_network_time"] == 0.0
    assert batch.diagnostics["initial_network_time_product"] == 0.0
    assert batch.diagnostics["initial_network_time_air"] == 0.0
    assert batch.diagnostics["initial_network_time_is_dimensionless"] is True
    # The batch carries the planar-startup convention, so the scales' own
    # conversion of s=0 is the field's epsilon -- the scheme-B T_h is not here.
    assert batch.diagnostics["time_convention"] == TIME_CONVENTION_PLANAR_STARTUP
    assert batch.scales.time_convention == TIME_CONVENTION_PLANAR_STARTUP
    assert batch.scales.startup_epsilon == field.epsilon
    assert batch.diagnostics["similitude_startup_epsilon_s"] == field.epsilon
    assert batch.diagnostics["similitude_start_time_s"] == field.epsilon
    assert batch.scales.start_time() == field.epsilon
    assert float(batch.scales.physical_time_from_network(0.0)) == pytest.approx(
        field.epsilon, rel=1.0e-15, abs=0.0
    )
    assert batch.detonation_velocity is None
    assert batch.scales.detonation_velocity is None

    model = CoupledModel(
        product=ProductNet(scales=batch.scales, width=8, depth=1, seed=0),
        air=AirNet(scales=batch.scales, width=8, depth=1, seed=1),
        front=FrontNet(
            scales=batch.scales,
            width=8,
            depth=1,
            initial_contact_hat=field.contact_radius / field.charge_radius,
            initial_gap_hat=(field.shock_radius - field.contact_radius) / field.charge_radius,
            seed=2,
        ),
        scales=batch.scales,
    )
    seen = {}
    product_fields = model.product_fields
    air_fields = model.air_fields

    def spy_product(s, xi):
        seen["product_s"] = s.detach().clone()
        return product_fields(s, xi)

    def spy_air(s, eta):
        seen["air_s"] = s.detach().clone()
        return air_fields(s, eta)

    model.product_fields = spy_product
    model.air_fields = spy_air
    breakdown = initial_condition_loss(batch, model)
    assert math.isfinite(float(breakdown.total))
    assert int(breakdown.monitors["ic_n_product"]) == batch.xi_ic_product.shape[0]
    assert int(breakdown.monitors["ic_n_air"]) == batch.eta_ic_air.shape[0]
    assert torch.equal(seen["product_s"], torch.zeros_like(batch.xi_ic_product))
    assert torch.equal(seen["air_s"], torch.zeros_like(batch.eta_ic_air))


def test_default_and_explicit_float64_batches_are_identical(field):
    product, air = _sample_points(field)
    default = build_planar_startup_initial_batch(
        field, scales=_scales(field), product_radii=product, air_radii=air
    )
    explicit = build_planar_startup_initial_batch(
        field,
        scales=_scales(field),
        product_radii=product,
        air_radii=air,
        dtype=torch.float64,
    )
    for name in (
        "xi_ic_product",
        "rho_ic_product",
        "u_ic_product",
        "e_ic_product",
        "eta_ic_air",
        "rho_ic_air",
        "u_ic_air",
        "e_ic_air",
    ):
        assert torch.equal(getattr(default, name), getattr(explicit, name))
    assert default.contact_ic == explicit.contact_ic
    assert default.shock_ic == explicit.shock_ic
    assert default.diagnostics == explicit.diagnostics


def test_float32_is_supported_finite_and_close(field):
    product, air = _sample_points(field)
    default = build_planar_startup_initial_batch(
        field, scales=_scales(field), product_radii=product, air_radii=air
    )
    half = build_planar_startup_initial_batch(
        field,
        scales=_scales(field),
        product_radii=product,
        air_radii=air,
        dtype=torch.float32,
    )
    for name in (
        "xi_ic_product",
        "rho_ic_product",
        "u_ic_product",
        "e_ic_product",
        "eta_ic_air",
        "rho_ic_air",
        "u_ic_air",
        "e_ic_air",
    ):
        actual = getattr(half, name)
        expected = getattr(default, name)
        assert actual.dtype == torch.float32
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(
            actual.double(), expected, rtol=1.0e-6, atol=1.0e-6
        )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.int64,
        torch.bool,
        torch.complex128,
        torch.float16,
        torch.bfloat16,
        np.float64,
    ),
)
def test_unsupported_dtypes_are_rejected_with_supported_list(field, dtype):
    with pytest.raises(PlanarStartupBatchError, match="unsupported dtype") as exc:
        build_planar_startup_initial_batch(field, scales=_scales(field), dtype=dtype)
    message = str(exc.value)
    assert "torch.float64" in message
    assert "torch.float32" in message


def test_unsupported_dtype_is_rejected_before_any_tensor_is_built(field, monkeypatch):
    calls = []
    original = torch.as_tensor

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", spy)
    with pytest.raises(PlanarStartupBatchError, match="unsupported dtype"):
        build_planar_startup_initial_batch(
            field, scales=_scales(field), dtype=torch.int64
        )
    assert calls == []


def test_float32_conversion_overflow_is_rejected_without_clipping(field, monkeypatch):
    original = type(field).sample_product

    def huge_product(self, radius):
        states = dict(original(self, radius))
        states["density"] = np.full_like(states["density"], 1.0e300)
        return states

    monkeypatch.setattr(type(field), "sample_product", huge_product)
    with pytest.raises(
        PlanarStartupBatchError,
        match=r"rho_ic_product contains non-finite values after conversion to torch.float32",
    ):
        build_planar_startup_initial_batch(
            field, scales=_scales(field), dtype=torch.float32
        )


def test_default_uniform_points_stay_inside_and_include_anchors(field):
    batch = build_planar_startup_initial_batch(
        field, scales=_scales(field), n_ic_product=11, n_ic_air=7
    )
    product, air = initial_radii_from_batch(batch)
    assert product.size == 11
    assert air.size == 7
    assert product[0] == 0.0 and product[-1] == field.contact_radius
    assert air[0] == field.contact_radius and air[-1] == field.shock_radius
    assert np.all((product >= 0.0) & (product <= field.contact_radius))
    assert np.all((air >= field.contact_radius) & (air <= field.shock_radius))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"product_radii": [[0.01]]}, "one-dimensional"),
        ({"product_radii": [-1.0e-6]}, "closed interval"),
        ({"air_radii": [0.99 * R0]}, "closed interval"),
        ({"product_radii": [float("nan")]}, "non-finite"),
        ({"n_ic_product": 0}, "n_ic_product"),
        ({"dtype": np.float64}, "unsupported dtype"),
    ),
)
def test_bad_dimensions_ranges_and_counts_fail_explicitly(field, kwargs, message):
    with pytest.raises(PlanarStartupBatchError, match=message):
        build_planar_startup_initial_batch(field, scales=_scales(field), **kwargs)


def test_wrong_field_type_or_wrong_radius_scales_fail_explicitly(field):
    with pytest.raises(PlanarStartupBatchError, match="PlanarStartupField"):
        build_planar_startup_initial_batch(object(), scales=_scales(field))
    scheme_b = SimilitudeScales.for_radius(
        0.08,
        time_scale=2.0,
        detonation_velocity=mat.DETONATION_VELOCITY_REFERENCE,
    )
    with pytest.raises(PlanarStartupBatchError, match="planar-startup time convention"):
        build_planar_startup_initial_batch(field, scales=scheme_b)
    wrong_radius = dataclasses.replace(_scales(field), charge_radius=0.08)
    with pytest.raises(PlanarStartupBatchError, match="R_0"):
        build_planar_startup_initial_batch(field, scales=wrong_radius)
    wrong_epsilon = dataclasses.replace(
        _scales(field), startup_epsilon=2.0 * field.epsilon
    )
    with pytest.raises(PlanarStartupBatchError, match="epsilon"):
        build_planar_startup_initial_batch(field, scales=wrong_epsilon)
    with pytest.raises(PlanarStartupBatchError, match="closed interval"):
        build_planar_startup_initial_batch(
            field,
            scales=_scales(field),
            product_radii=[field.contact_radius * 1.0001],
        )
    with pytest.raises(PlanarStartupBatchError, match="closed interval"):
        build_planar_startup_initial_batch(
            field,
            scales=_scales(field),
            air_radii=[field.shock_radius * 1.0001],
        )
