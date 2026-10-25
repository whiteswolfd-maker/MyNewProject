"""Round-trip tests for the planar-startup handoff pair."""

import json

import numpy as np
import pytest

from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.eos import JWLParameters
from spherical_blast_pinn.physics.two_medium_initial import build_two_medium_initial_field
from spherical_blast_pinn.physics.two_medium_startup_handoff import (
    HANDOFF_JSON_NAME,
    HANDOFF_NPZ_NAME,
    PlanarStartupHandoffError,
    load_planar_startup_handoff,
    write_planar_startup_handoff,
)
from spherical_blast_pinn.physics.two_medium_startup_planar import (
    PlanarStartupConfig,
    build_planar_startup_field,
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


def _write(field, directory):
    return write_planar_startup_handoff(field, directory)


def _read_npz(path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.array(payload[key], copy=True) for key in payload.files}


def _write_npz(path, arrays):
    np.savez_compressed(path, **arrays)


def _mutate_json(path, mutate):
    report = json.loads(path.read_text(encoding="utf-8"))
    mutate(report)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )


def test_round_trip_preserves_parameters_arrays_and_solution(field, tmp_path):
    json_path, npz_path = _write(field, tmp_path)
    assert json_path.name == HANDOFF_JSON_NAME
    assert npz_path.name == HANDOFF_NPZ_NAME

    for target in (tmp_path, json_path, npz_path):
        loaded = load_planar_startup_handoff(target)
        assert loaded.model_id == field.model_id
        assert loaded.method_version == field.method_version
        assert loaded.product_eos == field.product_eos
        assert loaded.air_gamma == field.air_gamma
        assert loaded.initial_state == field.initial_state
        assert loaded.epsilon == field.epsilon
        assert loaded.delta == field.delta
        assert loaded.time == field.time == loaded.epsilon
        assert loaded.pin_time == 0.0
        for name in (
            "charge_radius",
            "contact_radius",
            "shock_radius",
            "contact_speed",
            "shock_speed",
            "rarefaction_head_radius",
            "rarefaction_tail_radius",
            "rarefaction_head_speed",
            "rarefaction_tail_speed",
        ):
            assert getattr(loaded, name) == getattr(field, name)
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
            assert np.array_equal(getattr(loaded, name), getattr(field, name))
        assert loaded.solution.as_dict() == field.solution.as_dict()
        assert loaded.checks == field.checks
        assert loaded.integrals == field.integrals
        assert loaded.metadata == field.metadata
        assert loaded.cache_key() == field.cache_key()


def test_round_trip_preserves_all_region_sampling_and_one_sided_faces(field, tmp_path):
    _write(field, tmp_path)
    loaded = load_planar_startup_handoff(tmp_path)

    head = field.rarefaction_head_radius
    tail = field.rarefaction_tail_radius
    contact = field.contact_radius
    shock = field.shock_radius
    radii = np.array(
        [
            0.25 * head,
            0.5 * (head + tail),
            0.5 * (tail + contact),
            0.5 * (contact + shock),
            0.5 * (shock + field.radii[-1]),
        ]
    )
    for side in ("left", "right"):
        before = field.sample(radii, side=side)
        after = loaded.sample(radii, side=side)
        for key in before:
            assert np.array_equal(before[key], after[key]), (side, key)

    pairs = (
        ("contact product side", field.sample_product([contact]), loaded.sample_product([contact])),
        ("contact air side", field.sample_air([contact]), loaded.sample_air([contact])),
        ("shock post side", field.sample_air([shock]), loaded.sample_air([shock])),
        ("shock ambient side", field.sample([shock], side="right"), loaded.sample([shock], side="right")),
    )
    for label, before, after in pairs:
        for key in before:
            assert np.array_equal(before[key], after[key]), (label, key)

    # The pair is genuinely one-sided: the contact separates product and air,
    # and the shock separates post-shock from ambient air.
    left = field.sample_product([contact])
    right = field.sample_air([contact])
    assert left["material"][0] != right["material"][0]
    assert left["density"][0] != right["density"][0]
    post = field.sample_air([shock])
    ambient = field.sample([shock], side="right")
    assert post["pressure"][0] > ambient["pressure"][0]
    assert post["density"][0] > ambient["density"][0]


def _zero_check_energy(report):
    report["checks"]["spherical_energy_residual"] = 0.0


def _zero_integral_energy_change(report):
    report["integrals"]["spherical_energy_change_signed_j_per_sr"] = 0.0


def _zero_mass_residual(report):
    report["checks"]["spherical_mass_residual"] = 0.0


def _delete_energy_check(report):
    del report["checks"]["spherical_energy_residual"]


def _zero_top_level_energy(report):
    report["spherical_energy_residual"] = 0.0


def _wrong_dense_total_energy(report):
    report["integrals"]["dense"]["total_energy"] = 1.0


def _nan_mass_residual(report):
    report["checks"]["spherical_mass_residual"] = float("nan")


def _add_unexpected_check(report):
    report["checks"]["spherical_mass_residual_unchecked"] = 0.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (_zero_check_energy, r"checks\.spherical_energy_residual"),
        (_zero_integral_energy_change, r"integrals\.spherical_energy_change_signed_j_per_sr"),
        (_zero_mass_residual, r"checks\.spherical_mass_residual"),
        (_delete_energy_check, r"missing required diagnostic field 'spherical_energy_residual'"),
        (_zero_top_level_energy, r"handoff\.json\.spherical_energy_residual"),
        (_wrong_dense_total_energy, r"integrals\.dense\.total_energy"),
        (_nan_mass_residual, "must be finite"),
        (_add_unexpected_check, r"unexpected diagnostic field.*spherical_mass_residual_unchecked"),
    ),
)
def test_loading_rejects_tampered_diagnostics(field, tmp_path, mutate, message):
    json_path, _ = _write(field, tmp_path)
    _mutate_json(json_path, mutate)
    with pytest.raises(PlanarStartupHandoffError, match=message):
        load_planar_startup_handoff(json_path)


def test_custom_material_and_solver_parameters_round_trip(tmp_path):
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
        charge_radius=R0, outer_radius=2.0 * R0, jwl=custom
    )
    config = PlanarStartupConfig(
        delta=DELTA,
        inner_cells=8,
        fan_cells=16,
        plateau_cells=2,
        air_cells=4,
        ambient_cells=8,
        dense_samples=512,
        integration_step_factor=0.004,
        root_relative_tolerance=1.0e-12,
    )
    original = build_planar_startup_field(initial, config=config)
    _write(original, tmp_path)
    loaded = load_planar_startup_handoff(tmp_path)

    assert loaded.product_eos == custom
    assert loaded.metadata["grid_config"]["integration_step_factor"] == 0.004
    assert loaded.metadata["grid_config"]["root_relative_tolerance"] == 1.0e-12
    assert loaded.checks == original.checks
    assert loaded.integrals == original.integrals
    assert loaded.cache_key() == original.cache_key()
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
        assert np.array_equal(getattr(loaded, name), getattr(original, name))


def test_loading_rejects_a_modified_saved_velocity(field, tmp_path):
    json_path, npz_path = _write(field, tmp_path)
    arrays = _read_npz(npz_path)
    arrays["velocity"][0] += 100.0
    _write_npz(npz_path, arrays)

    with pytest.raises(PlanarStartupHandoffError, match="velocity"):
        load_planar_startup_handoff(json_path)


def test_loading_rejects_a_modified_epsilon(field, tmp_path):
    json_path, _ = _write(field, tmp_path)

    def mutate(report):
        report["epsilon_s"] *= 1.01

    _mutate_json(json_path, mutate)
    with pytest.raises(PlanarStartupHandoffError, match="epsilon"):
        load_planar_startup_handoff(json_path)


def test_loading_rejects_a_reconstruction_mismatch(field, tmp_path):
    json_path, _ = _write(field, tmp_path)

    def mutate(report):
        report["riemann"]["u_star"] += 1.0

    _mutate_json(json_path, mutate)
    with pytest.raises(PlanarStartupHandoffError, match="u_star"):
        load_planar_startup_handoff(json_path)


def test_loading_rejects_missing_fields_and_unsupported_versions(field, tmp_path):
    json_path, npz_path = _write(field, tmp_path)
    original_report = json_path.read_text(encoding="utf-8")

    def drop_product_eos(report):
        del report["product_eos"]

    _mutate_json(json_path, drop_product_eos)
    with pytest.raises(PlanarStartupHandoffError, match="product_eos"):
        load_planar_startup_handoff(json_path)

    json_path.write_text(original_report, encoding="utf-8")

    def unsupported(report):
        report["handoff_format_version"] = 999

    _mutate_json(json_path, unsupported)
    with pytest.raises(PlanarStartupHandoffError, match="unsupported handoff format version"):
        load_planar_startup_handoff(json_path)

    json_path.write_text(original_report, encoding="utf-8")
    arrays = _read_npz(npz_path)
    del arrays["density"]
    _write_npz(npz_path, arrays)
    with pytest.raises(PlanarStartupHandoffError, match="density"):
        load_planar_startup_handoff(json_path)
