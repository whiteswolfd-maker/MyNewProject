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
    times: list[str] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    times = list(times or ["1.00E-06", "2.00E-06", "3.00E-06"])
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
    # The profile files carry values in set order, not element IDs.  The
    # importer must not present invented 1..N IDs as source-provided.
    assert radial.source["cell_id_sources"] == ["generated"] * 3
    assert radial.source["generated_cell_id_frames"] == [0, 1, 2]


def _truncate_xy(path: Path, keep_rows: int, *, keep_endcurve: bool) -> None:
    lines = path.read_text(encoding="ascii").splitlines()
    out: list[str] = []
    started = False
    kept = 0
    for line in lines:
        text = line.strip()
        if not started:
            out.append(line)
            if text.startswith("#pts"):
                started = True
            continue
        if text.lower() == "endcurve":
            continue
        if kept < keep_rows:
            out.append(line)
            kept += 1
    if keep_endcurve:
        out.append("endcurve")
    path.write_text("\n".join(out) + "\n", encoding="ascii")


def test_a_profile_without_endcurve_is_refused(tmp_path):
    source = _write_run(tmp_path / "run")
    for path in source.glob("*_time3.00E-06.xy"):
        _truncate_xy(path, 1000, keep_endcurve=False)
    with pytest.raises(OneDDataError, match="no 'endcurve' marker"):
        _convert(source, tmp_path / "bad.npz")


def test_a_globally_truncated_set_is_caught_by_the_expected_cell_count(tmp_path):
    source = _write_run(tmp_path / "run")
    for path in source.glob("*.xy"):
        keep = 4 if path.name.startswith("profile1_") else 3
        _truncate_xy(path, keep, keep_endcurve=True)
    with pytest.raises(OneDDataError, match="expected 5 cells"):
        importer.convert(
            source_dir=source, case_id="fixture", radius_m=0.05,
            out=tmp_path / "bad.npz",
            velocity_pattern="profile1_x-velocity_x_time*.xy",
            pressure_pattern="profile2_pressure_x_time*.xy",
            density_pattern="profile3_density_x_time*.xy",
            alpha_product_pattern="profile4_volume_fraction_x_time*.xy",
            alpha_air_pattern="profile5_volume_fraction_x_time*.xy",
            expected_cells=5,
        )


def test_a_consistently_truncated_frame_is_refused(tmp_path):
    source = _write_run(tmp_path / "run")
    key = "3.00E-06"
    for path in source.glob(f"*_time{key}.xy"):
        keep = 4 if path.name.startswith("profile1_") else 3
        _truncate_xy(path, keep, keep_endcurve=True)
    with pytest.raises(OneDDataError, match="differing cell counts"):
        _convert(source, tmp_path / "bad.npz")


def test_product_only_profile_uses_implicit_air(tmp_path):
    source = _write_run(tmp_path / "run")
    for path in source.glob("profile5_volume_fraction_x_time*.xy"):
        path.unlink()
    out = tmp_path / "standard_product_only.npz"
    importer.convert(
        source_dir=source, case_id="fixture", radius_m=0.05, out=out,
        velocity_pattern="profile1_x-velocity_x_time*.xy",
        pressure_pattern="profile2_pressure_x_time*.xy",
        density_pattern="profile3_density_x_time*.xy",
        alpha_product_pattern="profile4_volume_fraction_x_time*.xy",
        alpha_air_pattern=None,
    )
    radial = read_standard_1d(out)
    assert radial.source["alpha_air_source"] == "implicit_1_minus_alpha_product"
    assert radial.frames[0].alpha_air is None
    # The implicit rule makes only alpha_product = 0 cells pure air.
    assert np.array_equal(
        radial.frames[0].air_mask(0.95), np.array([False, False, False, True, True])
    )


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


def test_every_raw_frame_is_imported_including_first_and_last(tmp_path):
    # Five frames with a cadence where an old 10-frame stride would keep only
    # the first and last.  The formal path must keep all five.
    times = ["1.00E-06", "2.00E-06", "3.00E-06", "4.00E-06", "5.00E-06"]
    source = _write_run(tmp_path / "run", times=times)
    out = tmp_path / "standard.npz"
    summary = _convert(source, out)
    radial = read_standard_1d(out)
    assert len(radial.frames) == 5
    assert summary["frame_selection_policy"] == "all_frames"
    assert summary["raw_frames"] == 5
    assert summary["imported_frames"] == 5
    assert summary["dropped_frames"] == 0
    assert np.allclose(
        [frame.time_raw_s for frame in radial.frames],
        [1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6, 5.0e-6],
    )
    provenance = radial.source["metadata"]
    assert provenance["frame_selection_policy"] == "all_frames"
    assert provenance["raw_frame_count"] == 5
    assert provenance["imported_frame_count"] == 5
    assert provenance["dropped_frame_count"] == 0
    assert provenance["selected_raw_frame_keys"] == times
    assert provenance["first_raw_time_s"] == pytest.approx(1.0e-6)
    assert provenance["last_raw_time_s"] == pytest.approx(5.0e-6)
    assert provenance["raw_input_fingerprint"]["file_count"] == 25


def test_removed_frame_selection_options_are_explicit_errors(tmp_path, capsys):
    source = _write_run(tmp_path / "run")
    out = tmp_path / "standard.npz"
    for flag in ("--frame-stride", "--dense-until-s", "--event-time-s"):
        value = "1" if flag == "--frame-stride" else "1.0e-6"
        code = importer.main([
            "--source-dir", str(source), "--case", "fixture",
            "--radius-m", "0.05", "--out", str(out), flag, value,
        ])
        assert code == 2
        assert "removed" in capsys.readouterr().err.lower()
