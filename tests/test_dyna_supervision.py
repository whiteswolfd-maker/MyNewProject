"""DYNA extracted-profile to supervision-contract conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spherical_blast_pinn.cli import compare as compare_cli

from spherical_blast_pinn.data.dyna_supervision import (
    DynaSupervisionError,
    origin_parameters_from_metadata,
    pressure_gauge_series,
    supervision_from_extracted_case,
)
from spherical_blast_pinn.data.extracted_case import ExtractedCase, load_extracted_case
from spherical_blast_pinn.data.raw_reader import (
    D3plotLayout,
    RawFrame,
    radial_velocity_from_centroid,
    resolve_d3plot_layout,
)
from spherical_blast_pinn.data.shell_average import (
    ShellGrid,
    overlap_assignment_from_arrays,
    shell_average,
)
from spherical_blast_pinn.data.startup_supervision import (
    AIR_CODE,
    NO_MEDIUM,
    PRODUCT_CODE,
    SupervisionDataset,
    SupervisionError,
    TimeOrigin,
)
from spherical_blast_pinn.cli.extract_profiles import _part_index_to_deck_id
from spherical_blast_pinn.physics.similitude import PlanarStartupScales
from spherical_blast_pinn.physics.two_medium_startup_handoff import load_planar_startup_handoff
from spherical_blast_pinn.sampling.sampler import RadiusSampler, SamplingConfig

R0 = 0.05
D = 6930.0


def _synthetic_case(*, field_validity=True) -> ExtractedCase:
    times = np.array([0.0, 1.0e-5, 3.0e-5])
    radius = np.array([0.01, 0.02, 0.03, 0.04])
    shape = (times.size, radius.size)
    density = np.array(
        [
            [1630.0, 1500.0, 1.225, 1.225],
            [1600.0, 1400.0, 1.300, 1.225],
            [1500.0, 1200.0, 1.500, 1.225],
        ]
    )
    velocity = np.array(
        [
            [0.0, -100.0, -20.0, 0.0],
            [0.0, -120.0, -30.0, 0.0],
            [0.0, -140.0, -40.0, 0.0],
        ]
    )
    energy = np.full(shape, 3.0e6)
    energy[:, 2:] = 2.0e5
    pressure = np.full(shape, 1.0e9)
    pressure[:, 2:] = 2.0e5
    pressure[:, 3] = 101320.0
    material = np.array(
        [
            [0, 0, 1, 2],
            [0, 0, 1, 2],
            [0, 0, 1, 2],
        ]
    )
    valid = np.ones(shape, dtype=bool)
    masks = {}
    if field_validity:
        for name in ("density", "velocity", "specific_energy", "pressure"):
            masks[name] = valid.copy()
        masks["density"][1, 1] = False
        masks["pressure"][2, 2] = False
    source = {
        "directory": "/synthetic/extracted",
        "detonation_velocity_m_s": D,
        "metadata_raw": {
            "R_0": R0,
            "detonation_velocity_m_s": D,
            "deck": {
                "initial_detonation": [0.0, 0.0, 0.0],
                "initial_detonation_time": 2.0e-6,
                "unit_system": "mm_s_t",
            },
            "units": {"source": "mm_s_t"},
            "field_provenance": {"rho": "synthetic", "u": "synthetic"},
        },
    }
    return ExtractedCase(
        case_id="synthetic",
        charge_radius=R0,
        times=times,
        radius=radius,
        density=density,
        velocity=velocity,
        pressure=pressure,
        specific_energy=energy,
        contact_radius=np.array([0.025, 0.025, 0.025]),
        shock_radius=np.array([0.035, 0.035, 0.035]),
        contact_valid=np.array([True, False, True]),
        shock_valid=np.array([False, True, True]),
        material=material,
        valid=valid,
        source=source,
        field_validity=masks if field_validity else None,
    )


def test_non_contiguous_part_ids_are_mapped_by_card_order_not_as_indices():
    class _Part:
        pass

    class _Summary:
        parts = {2: _Part(), 7: _Part(), 11: _Part()}

    assert _part_index_to_deck_id(_Summary()) == {0: 2, 1: 7, 2: 11}


def test_compact_history_layout_is_resolved_from_group_count_not_hardcoded_16():
    compact = resolve_d3plot_layout(
        D3plotLayout(), history_width=6, n_ale_groups=2
    )
    assert compact.volume_fraction_columns == (1, 2)
    assert compact.material_index_column == 3
    assert compact.per_material_mass_columns == (4, 5)
    seven = resolve_d3plot_layout(
        D3plotLayout(), history_width=16, n_ale_groups=7
    )
    assert seven.per_material_mass_columns == (9, 10, 11, 12, 13, 14, 15)


def test_radial_velocity_preserves_sign_and_is_zero_at_the_centre():
    centroid = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    velocity = radial_velocity_from_centroid(
        centroid,
        np.array([100.0, -10.0, 20.0]),
        np.zeros(3),
        np.zeros(3),
    )
    np.testing.assert_allclose(velocity, [0.0, -10.0, 20.0])


def test_shell_average_uses_material_mass_over_occupied_volume():
    frame = RawFrame(
        case_id="synthetic-density",
        time=1.0,
        radius=np.array([0.5, 1.5]),
        cell_volume=np.array([1.0, 1.0]),
        density=np.array([999.0, 999.0]),
        velocity_x=np.zeros(2),
        velocity_y=np.zeros(2),
        velocity_z=np.zeros(2),
        pressure=np.array([1.0e6, 1.0e5]),
        tnt_volume_fraction=np.array([1.0, 0.0]),
        radial_velocity=np.array([100.0, -10.0]),
        specific_internal_energy=np.array([1.0e5, 2.0e5]),
        density_product=np.array([12.0, np.nan]),
        density_air=np.array([np.nan, 3.0]),
        cell_radius_inner=np.array([0.0, 1.0]),
        cell_radius_outer=np.array([1.0, 2.0]),
    )
    grid = ShellGrid(np.array([0.0, 2.0]))
    assignment = overlap_assignment_from_arrays(
        frame.cell_radius_inner, frame.cell_radius_outer, frame.cell_volume, grid
    )
    product = shell_average(frame, grid, "product", assignment=assignment)
    air = shell_average(frame, grid, "air", assignment=assignment)
    assert product.density[0] == pytest.approx(12.0)
    assert air.density[0] == pytest.approx(3.0)
    # The mixture density is deliberately different; it must not leak through.
    assert product.density[0] != pytest.approx(frame.density[0])
    assert air.density[0] != pytest.approx(frame.density[1])


def test_origin_is_lt_plus_r0_over_d_and_records_all_inputs():
    case = _synthetic_case()
    params = origin_parameters_from_metadata(case, epsilon_s=1.0e-7)
    assert params.detonation_velocity_m_s == pytest.approx(D)
    assert params.lighting_time_s == pytest.approx(2.0e-6)
    expected = 2.0e-6 + R0 / D
    assert params.origin_time_s == pytest.approx(expected)
    origin = params.as_time_origin()
    assert origin.ready
    assert origin.origin_time_s == pytest.approx(expected)
    assert "LT" in origin.evidence and "R0" in origin.evidence and "D" in origin.evidence
    payload = params.as_dict()
    assert payload["lighting_time_s"] == pytest.approx(2.0e-6)
    assert payload["epsilon_case_s"] == pytest.approx(1.0e-7)


def test_converter_preserves_values_validity_medium_and_fronts():
    case = _synthetic_case()
    params = origin_parameters_from_metadata(case, epsilon_s=1.0e-7)
    dataset = supervision_from_extracted_case(
        case,
        time_origin=params.as_time_origin(),
        epsilon_s=1.0e-7,
        synthetic=True,
    )
    np.testing.assert_array_equal(dataset.times_dyna, case.times)
    np.testing.assert_array_equal(dataset.radius, np.broadcast_to(case.radius, case.density.shape))
    np.testing.assert_array_equal(dataset.density, case.density)
    np.testing.assert_array_equal(dataset.velocity, case.velocity)
    np.testing.assert_array_equal(dataset.pressure, case.pressure)
    np.testing.assert_array_equal(dataset.specific_energy, case.specific_energy)

    assert dataset.medium[0, 0] == PRODUCT_CODE
    assert dataset.medium[0, 2] == AIR_CODE
    assert dataset.medium[0, 3] == NO_MEDIUM
    assert not dataset.valid_density[1, 1]
    assert not dataset.valid_pressure[2, 2]
    # A missing field stays missing; the other fields at the same point remain valid.
    assert dataset.valid_velocity[1, 1]
    assert dataset.valid_specific_energy[2, 2]
    # Invalid values are not replaced by zero.
    assert dataset.density[1, 1] == pytest.approx(1400.0)
    np.testing.assert_array_equal(dataset.contact_valid, case.contact_valid)
    np.testing.assert_array_equal(dataset.shock_valid, case.shock_valid)
    assert dataset.contact_delta_m > 0.0 and dataset.shock_delta_m > 0.0
    assert dataset.synthetic is True
    assert dataset.source_kind == "synthetic"
    assert dataset.time_origin.ready
    np.testing.assert_allclose(
        dataset.absolute_physical_time(), case.times - dataset.time_origin.origin_time_s
    )
    np.testing.assert_allclose(
        dataset.tau_time(1.0e-7),
        case.times - dataset.time_origin.origin_time_s - 1.0e-7,
    )


def test_real_conversion_refuses_missing_per_field_validity():
    case = _synthetic_case(field_validity=False)
    params = origin_parameters_from_metadata(case, epsilon_s=1.0e-7)
    with pytest.raises(DynaSupervisionError, match="per-field validity"):
        supervision_from_extracted_case(
            case,
            time_origin=params.as_time_origin(),
            epsilon_s=1.0e-7,
            synthetic=False,
        )


def test_unknown_origin_can_be_stored_for_review_but_blocks_supervision():
    case = _synthetic_case()
    dataset = supervision_from_extracted_case(
        case,
        time_origin=TimeOrigin(
            status="unknown",
            origin_time_s=None,
            method="not determined for this review copy",
            evidence="no confirmed DYNA origin",
        ),
        epsilon_s=1.0e-7,
        synthetic=True,
    )
    assert dataset.time_origin.status == "unknown"
    assert not dataset.time_origin.ready
    assert dataset.supervision_blockers()
    with pytest.raises(SupervisionError, match="time origin"):
        dataset.require_supervision_ready()


def test_fixed_gauge_does_not_cross_material_or_extrapolate():
    case = _synthetic_case()
    params = origin_parameters_from_metadata(case, epsilon_s=1.0e-7)
    dataset = supervision_from_extracted_case(
        case,
        time_origin=params.as_time_origin(),
        epsilon_s=1.0e-7,
        synthetic=True,
    )
    exact = pressure_gauge_series(
        dataset, radius_m=0.03, ambient_pressure_pa=101320.0, epsilon_s=1.0e-7
    )
    assert exact["n_valid"] == 2
    assert exact["reason"][2] == "invalid_pressure"
    assert exact["interpolation"] and "no material-interface crossing" in exact["interpolation"]
    finite = [v for v in exact["overpressure_pa"] if v is not None]
    assert len(finite) == 2
    np.testing.assert_allclose(finite, 200000.0 - 101320.0)

    mixed_bracket = pressure_gauge_series(
        dataset, radius_m=0.025, ambient_pressure_pa=101320.0, epsilon_s=1.0e-7
    )
    assert mixed_bracket["n_valid"] == 0
    assert all(reason == "bracket_not_all_air" for reason in mixed_bracket["reason"])

    outside = pressure_gauge_series(
        dataset, radius_m=0.05, ambient_pressure_pa=101320.0, epsilon_s=1.0e-7
    )
    assert outside["n_valid"] == 0
    assert all(reason == "outside_radial_grid" for reason in outside["reason"])


REAL_PROFILE = Path("F:/2/extracted_geom/50mm_d")
REAL_HANDOFF = Path(
    "outputs/planar_startup_diagnostic/"
    "uniform_product_sphere_air_planar_riemann_v1/8257f2b650cd2b1b/handoff.json"
)


@pytest.mark.skipif(
    not (REAL_PROFILE.is_dir() and REAL_HANDOFF.is_file()),
    reason="the local current 50 mm extraction/handoff is not present",
)
def test_real_50mm_conversion_enters_existing_sampler(tmp_path):
    case = load_extracted_case(REAL_PROFILE)
    field = load_planar_startup_handoff(REAL_HANDOFF)
    params = origin_parameters_from_metadata(case, epsilon_s=field.epsilon)
    dataset = supervision_from_extracted_case(
        case,
        time_origin=params.as_time_origin(),
        epsilon_s=field.epsilon,
    )
    assert dataset.source_kind == "real"
    assert dataset.n_frames == 6
    tau = dataset.tau_time(field.epsilon)
    assert tau[0] < 0.0 and np.any(tau > 0.0)
    assert dataset.contact_valid.sum() > 0
    assert dataset.shock_valid.sum() > 0
    assert dataset.valid_density.sum() > 0
    assert dataset.medium[dataset.medium == NO_MEDIUM].size > 0
    assert dataset.supervision_blockers() == []

    path = dataset.save(tmp_path / "tnt_50mm.npz")
    loaded = SupervisionDataset.load(path)
    assert loaded.identity()["content_sha256"] == dataset.identity()["content_sha256"]

    scales = PlanarStartupScales.for_startup_field(field, time_scale=2.0)
    sampler = RadiusSampler(
        field=field,
        scales=scales,
        tau_end=1.2e-3,
        case=loaded,
        config=SamplingConfig(
            n_pde_product=8,
            n_pde_air=8,
            n_ic_product=4,
            n_ic_air=4,
            n_contact=4,
            n_shock=4,
            n_data_product=8,
            n_data_air=2,
            n_data_contact=2,
            n_data_shock=2,
        ),
        seed=1,
    )
    batch = sampler.batch()
    assert batch.diagnostics["data_time_origin_status"] == "confirmed"
    assert batch.diagnostics["data_counts"]["product"] > 0
    assert batch.s_data_product is not None
    assert float(batch.diagnostics["data_tau_range_s"][0]) > 0.0
    # The first DYNA state has tau < 0 and is not present in the sampled targets.
    assert float(batch.diagnostics["data_tau_range_s"][0]) > float(tau[0])


@pytest.mark.skipif(
    not (REAL_PROFILE.is_dir() and REAL_HANDOFF.is_file()),
    reason="the local current 50 mm extraction/handoff is not present",
)
def test_real_50mm_conversion_enters_compare_with_synthetic_prediction(tmp_path):
    case = load_extracted_case(REAL_PROFILE)
    field = load_planar_startup_handoff(REAL_HANDOFF)
    params = origin_parameters_from_metadata(case, epsilon_s=field.epsilon)
    dataset = supervision_from_extracted_case(
        case,
        time_origin=params.as_time_origin(),
        epsilon_s=field.epsilon,
    )
    reference = dataset.save(tmp_path / "reference.npz")

    frames = [4, 5]
    profiles = []
    for frame in frames:
        usable = (
            (dataset.medium[frame] == PRODUCT_CODE)
            | (dataset.medium[frame] == AIR_CODE)
        ) & dataset.valid_density[frame] & dataset.valid_pressure[frame]
        radii = dataset.radius[frame][usable]
        region = dataset.medium[frame][usable].astype(np.int64)
        pressure = dataset.pressure[frame][usable]
        profiles.append(
            {
                "radius": radii,
                "region": region,
                "density": dataset.density[frame][usable],
                "velocity": dataset.velocity[frame][usable],
                "specific_energy": dataset.specific_energy[frame][usable],
                "pressure": pressure,
                "overpressure": pressure - 101320.0,
            }
        )
    prediction = tmp_path / "synthetic_prediction.npz"
    payload = {
        "time_index": np.concatenate(
            [np.full(profile["radius"].size, i, dtype=np.int64) for i, profile in enumerate(profiles)]
        ),
        "region": np.concatenate([profile["region"] for profile in profiles]),
        "radius": np.concatenate([profile["radius"] for profile in profiles]),
        "density": np.concatenate([profile["density"] for profile in profiles]),
        "velocity": np.concatenate([profile["velocity"] for profile in profiles]),
        "specific_energy": np.concatenate([profile["specific_energy"] for profile in profiles]),
        "pressure": np.concatenate([profile["pressure"] for profile in profiles]),
        "overpressure": np.concatenate([profile["overpressure"] for profile in profiles]),
        "times_phys": dataset.absolute_physical_time()[frames],
        "contact_radius": dataset.contact_radius[frames],
        "shock_radius": dataset.shock_radius[frames],
    }
    metadata = {
        "format": "spherical_blast_pinn.startup_prediction",
        "version": 1,
        "charge_radius_m": float(dataset.charge_radius_m),
        "epsilon_s": float(field.epsilon),
        "ambient_pressure": 101320.0,
        "time_convention": "synthetic exact-profile prediction for interface testing",
        "extrapolated": [False, False],
    }
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with open(prediction, "wb") as handle:
        np.savez_compressed(handle, **payload)

    report_path = tmp_path / "compare.json"
    assert compare_cli.main([
        "--prediction", str(prediction),
        "--reference", str(reference),
        "--output", str(report_path),
        "--gauge-radius-m", "0.88",
    ]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["matched_frames"] == 2
    assert report["reference_synthetic"] is False
    assert report["reference_time_origin"]["status"] == "confirmed"
    assert report["frames"][0]["physical_time_s"] == pytest.approx(
        dataset.absolute_physical_time()[4]
    )
    product_error = report["frames"][0]["fields"]["product_density"]["relative_l2"]
    assert product_error is not None and product_error < 1.0e-12
    assert len(report["gauges"]) == 1
    assert report["gauges"][0]["tracking"] == "fixed_spatial_eulerian_shell"
    assert report["gauges"][0]["frames"][0]["reference_overpressure_pa"] is not None
    assert report["gauges"][0]["frames"][0]["predicted_overpressure_pa"] is not None
