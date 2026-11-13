"""Regression tests for audit integrity failures and the exit code."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from spherical_blast_pinn.contact_air.one_d import write_standard_1d
from spherical_blast_pinn.contact_air.reference_audit import case_manifest

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "audit_reference_data", ROOT / "scripts" / "audit_reference_data.py"
)
assert _SPEC is not None and _SPEC.loader is not None
audit_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_script)

UNITS = {
    "to_si": {"length": 1.0e-3, "density": 1.0e12,
              "pressure": 1.0e6, "velocity": 1.0e-3},
}
AMBIENT = {"pressure_mpa": 0.10132, "density_kg_m3": 1.225, "gamma": 1.4}
CHARGE = {"density_kg_m3": 1630.0}


def _write_xy(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    lines = ["Curveplot", "fixture", "x", "y", " ", " #pts=1"]
    lines += [f"  {float(a):.12E}  {float(b):.12E}" for a, b in zip(x, y)]
    lines.append("endcurve")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _raw_run(directory: Path, times: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    nodes = np.array([0.0, 0.1, 0.2])
    cells = 0.5 * (nodes[:-1] + nodes[1:])
    for index, time in enumerate(times):
        _write_xy(directory / f"profile1_x-velocity_x_time{time}.xy",
                  nodes, np.array([0.0, 1.0, 2.0]) + index)
        _write_xy(directory / f"profile2_pressure_x_time{time}.xy",
                  cells, np.array([1.0, 2.0]) + index)
        _write_xy(directory / f"profile3_density_x_time{time}.xy",
                  cells, np.array([1.0e-9, 2.0e-9]))
        _write_xy(directory / f"profile4_volume_fraction_x_time{time}.xy",
                  cells, np.array([1.0, 0.0]))


def _standard(path: Path, times: list[float], *,
              pressure_scale: float = 1.0, source_extra: dict | None = None) -> None:
    frames = []
    for _ in times:
        frames.append({
            "r_left": np.array([0.0, 0.1]),
            "r_right": np.array([0.1, 0.2]),
            "radius": np.array([0.05, 0.15]),
            "alpha_product": np.array([1.0, 0.0]),
            "density": np.array([1.0e3, 2.0e3]),
            "velocity": np.array([0.5, 1.5]),
            "pressure": np.array([1.0e6, 2.0e6]) * pressure_scale,
            "valid_density": np.ones(2, bool),
            "valid_velocity": np.ones(2, bool),
            "valid_pressure": np.ones(2, bool),
        })
    source = {
        "frame_selection_policy": "all_frames",
        "raw_frame_count": len(times),
        "imported_frame_count": len(times),
        "dropped_frame_count": 0,
        "selected_raw_frame_keys": [f"{value:.2E}" for value in times],
        "raw_frame_keys": [f"{value:.2E}" for value in times],
    }
    source.update(source_extra or {})
    write_standard_1d(
        path, case_id="fixture", charge_radius_m=0.05,
        frames=frames, frame_times=times, source=source,
    )


def _manifest_case(tmp_path: Path) -> dict:
    return {
        "case_id": "fixture",
        "raw_dir": str(tmp_path / "raw"),
        "deck": "deck.k",
        "standard": "standard.npz",
        "cache": "cache.npz",
        "radius_m": 0.05,
    }


def test_manifest_compares_imported_keys_with_raw_disk_keys(tmp_path):
    times = ["1.00E-06", "2.00E-06", "3.00E-06", "4.00E-06"]
    _raw_run(tmp_path / "raw", times)
    # A standard that claims all_frames but actually kept only three frames.
    _standard(
        tmp_path / "standard.npz", [1.0e-6, 2.0e-6, 3.0e-6],
        source_extra={
            "raw_frame_count": 4,
            "imported_frame_count": 4,
            "selected_raw_frame_keys": times[:3],
            "raw_frame_keys": times[:3],
        },
    )
    manifest = case_manifest(
        _manifest_case(tmp_path), tmp_path, UNITS, AMBIENT, CHARGE,
        require_all_frames=True,
    )
    assert manifest["raw_frame_count"] == 4
    assert manifest["standard"]["imported_equals_raw"] is False
    assert "4.00E-06" not in manifest["standard"]["selected_keys"]
    assert any("imported frames do not match raw" in item
               for item in manifest["integrity_failures"])


def test_manifest_detects_stale_frame_times_and_metadata_counts(tmp_path):
    times = ["1.00E-06", "2.00E-06", "3.00E-06"]
    _raw_run(tmp_path / "raw", times)
    _standard(
        tmp_path / "standard.npz", [1.0e-6, 2.0e-6, 3.0e-6],
        source_extra={"raw_frame_count": 4, "imported_frame_count": 4},
    )
    manifest = case_manifest(
        _manifest_case(tmp_path), tmp_path, UNITS, AMBIENT, CHARGE,
        require_all_frames=True,
    )
    assert manifest["standard"]["metadata_count_matches_disk"] is False
    assert manifest["standard"]["imported_equals_raw"] is False


def test_audit_exit_code_is_nonzero_for_a_value_mismatch(tmp_path):
    times = ["1.00E-06", "2.00E-06", "3.00E-06", "4.00E-06"]
    raw = tmp_path / "raw"
    _raw_run(raw, times)
    # Imported pressure is 10% high; all other fields match.
    _standard(tmp_path / "standard.npz", [1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6],
              pressure_scale=1.1)
    config = {
        "audit": {
            "version": 1,
            "repo_root": str(tmp_path),
            "output_dir": "audit_out",
            "require_all_frames": True,
            "units": UNITS,
            "ambient": AMBIENT,
            "charge": CHARGE,
            "measurement_scaled_distances_m_kg13": [0.5],
            "focus_cases": ["fixture"],
            "cases": [{
                "case_id": "fixture", "radius_m": 0.05,
                "raw_dir": str(raw), "deck": "deck.k",
                "standard": "standard.npz", "cache": "cache.npz",
            }],
        }
    }
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    code = audit_script.main([
        "--config", str(config_path), "--stage", "imports",
        "--cases", "fixture", "--import-sample-frames", "0",
    ])
    assert code == 1
    with (tmp_path / "audit_out" / "raw_import_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(audit_script.csv.DictReader(handle))
    assert any(row["status"] != "ok" for row in rows)
