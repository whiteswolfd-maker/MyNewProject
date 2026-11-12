"""Tests for the LS-DYNA ASCII profile importer.

The importer is a data-format adapter, not a physics model.  These tests use a
tiny 5-cell, 3-frame fixture so the unit conversion, node->cell averaging,
actual cell boundaries and material-fraction checks are all explicit.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spherical_blast_pinn.contact_air.one_d import OneDDataError, read_standard_1d

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "import_lsdyna_profiles_1d", ROOT / "scripts" / "import_lsdyna_profiles_1d.py"
)
assert _SPEC is not None and _SPEC.loader is not None
importer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(importer)

NODES_MM = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
CELLS_MM = 0.5 * (NODES_MM[:-1] + NODES_MM[1:])
N_CELLS = CELLS_MM.size


def _write_xy(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    lines = ["Curveplot", "fixture", "x", "y", " ", " #pts=1"]
    lines += [f"  {float(a):.15E}   {float(b):.15E}" for a, b in zip(x, y)]
    lines.append("endcurve")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_run(
    directory: Path,
    *,
    alpha_product: np.ndarray | None = None,
    alpha_air: np.ndarray | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    times = ["1.00E-06", "2.00E-06", "3.00E-06"]
    for frame, time in enumerate(times):
        velocity = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0]) + frame
        pressure = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + frame
        density = np.array([1.0e-9, 2.0e-9, 3.0e-9, 2.0e-12, 1.0e-12])
        a_p = (
            np.array([1.0, 0.8, 0.2, 0.0, 0.0])
            if alpha_product is None else np.asarray(alpha_product)
        )
        a_a = (
            np.array([0.0, 0.2, 0.8, 1.0, 1.0])
            if alpha_air is None else np.asarray(alpha_air)
        )
        _write_xy(
            directory / f"profile1_x-velocity_x_time{time}.xy",
            NODES_MM, velocity,
        )
        _write_xy(
            directory / f"profile2_pressure_x_time{time}.xy",
            CELLS_MM, pressure,
        )
        _write_xy(
            directory / f"profile3_density_x_time{time}.xy",
            CELLS_MM, density,
        )
        _write_xy(
            directory / f"profile4_volume_fraction_x_time{time}.xy",
            CELLS_MM, a_p,
        )
        _write_xy(
            directory / f"profile5_volume_fraction_x_time{time}.xy",
            CELLS_MM, a_a,
        )
    return directory


def _convert(source: Path, out: Path) -> dict:
    return importer.convert(
        source_dir=source, case_id="fixture", radius_m=0.05, out=out,
        velocity_pattern="profile1_x-velocity_x_time*.xy",
        pressure_pattern="profile2_pressure_x_time*.xy",
        density_pattern="profile3_density_x_time*.xy",
        alpha_product_pattern="profile4_volume_fraction_x_time*.xy",
        alpha_air_pattern="profile5_volume_fraction_x_time*.xy",
        dense_until_s=1.0, frame_stride=1,
    )


def test_profiles_are_converted_with_actual_boundaries_and_si_units(tmp_path):
    source = _write_run(tmp_path / "run")
    out = tmp_path / "standard.npz"
    _convert(source, out)

    radial = read_standard_1d(out)
    assert len(radial.frames) == 3
    assert radial.position_type == "cell_center"
    frame = radial.frames[0]

    # Actual node boundaries, not an assumed uniform grid invented downstream.
    assert np.allclose(frame.r_left, NODES_MM[:-1] * 1.0e-3, atol=1.0e-15)
    assert np.allclose(frame.r_right, NODES_MM[1:] * 1.0e-3, atol=1.0e-15)
    assert np.allclose(frame.radius, CELLS_MM * 1.0e-3, atol=1.0e-15)

    # Node velocity -> cell-centre average, mm/s -> m/s (frame 0 is unshifted).
    expected_velocity = 0.5 * (
        np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        + np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ) * 1.0e-3
    assert np.allclose(frame.velocity, expected_velocity)

    # MPa -> Pa and tonne/mm^3 -> kg/m^3.
    assert np.allclose(frame.pressure, np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 1.0e6)
    assert np.allclose(
        frame.density,
        np.array([1.0e-9, 2.0e-9, 3.0e-9, 2.0e-12, 1.0e-12]) * 1.0e12,
    )
    assert np.allclose(frame.alpha_product, [1.0, 0.8, 0.2, 0.0, 0.0])
    assert np.allclose(frame.alpha_air, [0.0, 0.2, 0.8, 1.0, 1.0])
    provenance = radial.source["metadata"]
    assert provenance["time_source"] == "profile_filename_3sigfig"
    assert provenance["no_normalization_or_filling"] is True


def test_fractions_that_do_not_sum_to_one_are_refused(tmp_path):
    source = _write_run(
        tmp_path / "run", alpha_product=np.full(N_CELLS, 0.5),
        alpha_air=np.full(N_CELLS, 0.2),
    )
    with pytest.raises(OneDDataError, match="not 1 within 1e-4"):
        _convert(source, tmp_path / "bad.npz")


def test_mismatched_profile_time_sets_are_refused(tmp_path):
    source = _write_run(tmp_path / "run")
    (source / "profile5_volume_fraction_x_time3.00E-06.xy").unlink()
    with pytest.raises(OneDDataError, match="do not match pressure"):
        _convert(source, tmp_path / "bad.npz")
