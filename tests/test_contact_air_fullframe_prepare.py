"""Cache reuse must be tied to the standard data, not just frame times.

The importer and cache builder are exercised through the real
``prepare_fullframe_data`` orchestrator.  Only ``command_build_cache`` is
replaced by a small fixture writer: it records the standard fingerprint the
same way the real builder does, so the reuse logic is tested end to end without
requiring physically complete shock/contact data.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from spherical_blast_pinn.contact_air.one_d import (
    npz_content_digest,
    write_standard_1d,
)

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "prepare_fullframe_data", ROOT / "scripts" / "prepare_fullframe_data.py"
)
assert _SPEC is not None and _SPEC.loader is not None
prepare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare)

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


def _raw_run(directory: Path, pressure_offset: float = 0.0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    nodes = np.array([0.0, 0.1, 0.2])
    cells = 0.5 * (nodes[:-1] + nodes[1:])
    for index, time in enumerate(["1.00E-06", "2.00E-06", "3.00E-06"]):
        _write_xy(directory / f"profile1_x-velocity_x_time{time}.xy",
                  nodes, np.array([0.0, 1.0, 2.0]) + index)
        _write_xy(directory / f"profile2_pressure_x_time{time}.xy",
                  cells, np.array([1.0, 2.0]) + index + pressure_offset)
        _write_xy(directory / f"profile3_density_x_time{time}.xy",
                  cells, np.array([1.0e-9, 2.0e-9]))
        _write_xy(directory / f"profile4_volume_fraction_x_time{time}.xy",
                  cells, np.array([1.0, 0.0]))


def _config(tmp_path: Path, raw: Path) -> Path:
    payload = {
        "audit": {
            "version": 1,
            "repo_root": str(tmp_path),
            "output_dir": "audit_out",
            "require_all_frames": True,
            "units": UNITS,
            "ambient": AMBIENT,
            "charge": CHARGE,
            "cases": [{
                "case_id": "fixture", "radius_m": 0.05,
                "raw_dir": str(raw), "deck": "deck.k",
                "standard": "standard.npz", "cache": "cache.npz",
            }],
        }
    }
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _fixture_cache(source: Path, out: Path) -> None:
    """Write a cache-shaped file that records the source digest."""
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    times = np.asarray(metadata["frame_times"], dtype=np.float64)
    payload = {
        "format": "spherical_blast_pinn.contact_air_case",
        "version": 7,
        "case": {
            "case_id": "fixture",
            "source": {
                "case_id": "fixture",
                "source_path": str(source),
                "standard_data_digest": npz_content_digest(source),
            },
        },
    }
    np.savez(
        out, metadata=np.asarray(json.dumps(payload)),
        raw_times=times, contact_times=times, shock_times=times,
    )


def _run(config: Path, tmp_path: Path, *, cache_only: bool = False) -> list[dict]:
    args = [
        "--config", str(config), "--cases", "fixture",
        "--summary", str(tmp_path / "summary.json"),
    ]
    if cache_only:
        args.append("--cache-only")
    prepare.main(args)
    return json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))


def test_rebuilding_the_standard_invalidates_a_time_identical_cache(
    tmp_path, monkeypatch,
):
    raw = tmp_path / "raw"
    _raw_run(raw)
    config = _config(tmp_path, raw)
    calls: list[Path] = []

    def fake_build_cache(args):
        calls.append(Path(args.source))
        _fixture_cache(Path(args.source), Path(args.out))
        return 0

    monkeypatch.setattr(prepare, "command_build_cache", fake_build_cache)

    first = _run(config, tmp_path)
    assert first[0]["import"] != "skipped_existing"
    assert first[0]["cache"] == "built"

    second = _run(config, tmp_path)
    assert second[0]["import"] == "skipped_existing"
    assert second[0]["cache"] == "skipped_existing"
    assert second[0]["standard_rebuilt"] is False

    standard = tmp_path / "standard.npz"
    cache = tmp_path / "cache.npz"
    old_digest = npz_content_digest(standard)
    with np.load(cache, allow_pickle=False) as archive:
        old_meta = json.loads(str(archive["metadata"]))
    assert old_meta["case"]["source"]["standard_data_digest"]["sha256"] == (
        old_digest["sha256"]
    )

    # Same frame times, different pressure values.  The raw fingerprint changes,
    # so the standard is rebuilt; the cache must not survive.
    _raw_run(raw, pressure_offset=10.0)
    third = _run(config, tmp_path)
    assert third[0]["standard_rebuilt"] is True
    assert third[0]["cache"] == "built"
    assert "standard was rebuilt in this run" in third[0]["cache_validation"]
    assert len(calls) == 2

    new_digest = npz_content_digest(standard)
    assert new_digest["sha256"] != old_digest["sha256"]
    ok, reason = prepare._cache_is_current(standard, cache)
    assert ok, reason
    with np.load(cache, allow_pickle=False) as archive:
        new_meta = json.loads(str(archive["metadata"]))
    assert new_meta["case"]["source"]["standard_data_digest"]["sha256"] == (
        new_digest["sha256"]
    )


def _standard(path: Path, pressure: np.ndarray, times: list[float]) -> None:
    frames = [{
        "r_left": np.array([0.0, 0.1]),
        "r_right": np.array([0.1, 0.2]),
        "radius": np.array([0.05, 0.15]),
        "alpha_product": np.array([1.0, 0.0]),
        "density": np.array([1.0e3, 2.0e3]),
        "velocity": np.array([0.5, 1.5]),
        "pressure": np.array(pressure, dtype=np.float64),
        "valid_density": np.ones(2, bool),
        "valid_velocity": np.ones(2, bool),
        "valid_pressure": np.ones(2, bool),
    } for _ in times]
    write_standard_1d(
        path, case_id="fixture", charge_radius_m=0.05,
        frames=frames, frame_times=times,
        source={"frame_selection_policy": "all_frames",
                "raw_frame_count": len(times),
                "imported_frame_count": len(times),
                "dropped_frame_count": 0,
                "raw_frame_keys": [f"{value:.2E}" for value in times],
                "selected_raw_frame_keys": [f"{value:.2E}" for value in times]},
    )


def _cache(path: Path, times: list[float], digest: dict | None) -> None:
    source = {"case_id": "fixture", "source_path": "standard.npz"}
    if digest is not None:
        source["standard_data_digest"] = digest
    payload = {
        "format": "spherical_blast_pinn.contact_air_case",
        "version": 7,
        "case": {"case_id": "fixture", "source": source},
    }
    times = np.asarray(times, dtype=np.float64)
    np.savez(
        path, metadata=np.asarray(json.dumps(payload)),
        raw_times=times, contact_times=times, shock_times=times,
    )


def test_cache_only_rebuilds_a_cache_without_fingerprint(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _raw_run(raw)
    config = _config(tmp_path, raw)
    calls: list[Path] = []

    def fake_build_cache(args):
        calls.append(Path(args.source))
        _fixture_cache(Path(args.source), Path(args.out))
        return 0

    monkeypatch.setattr(prepare, "command_build_cache", fake_build_cache)
    _run(config, tmp_path)

    cache = tmp_path / "cache.npz"
    with np.load(cache, allow_pickle=False) as archive:
        payload = {
            key: np.asarray(archive[key]) for key in archive.files
        }
    metadata = json.loads(str(payload["metadata"]))
    metadata["case"]["source"].pop("standard_data_digest", None)
    payload["metadata"] = np.asarray(json.dumps(metadata))
    np.savez(cache, **payload)

    summary = _run(config, tmp_path, cache_only=True)
    assert summary[0]["cache"] == "built"
    assert "no standard data fingerprint" in summary[0]["cache_validation"]
    assert len(calls) == 2
    standard = tmp_path / "standard.npz"
    assert prepare._cache_is_current(standard, cache)[0] is True


def test_cache_check_requires_the_recorded_standard_digest(tmp_path):
    standard = tmp_path / "standard.npz"
    cache = tmp_path / "cache.npz"
    times = [1.0e-6, 2.0e-6, 3.0e-6]
    _standard(standard, [1.0e6, 2.0e6], times)
    _cache(cache, times, npz_content_digest(standard))
    assert prepare._cache_is_current(standard, cache) == (True, "current")

    # Same frame count and times, changed pressure: digest must catch it.
    _standard(standard, [1.1e6, 2.2e6], times)
    ok, reason = prepare._cache_is_current(standard, cache)
    assert ok is False
    assert "fingerprint" in reason

    # A cache written before fingerprinting is not accepted as current.
    _cache(cache, times, None)
    ok, reason = prepare._cache_is_current(standard, cache)
    assert ok is False
    assert "no standard data fingerprint" in reason
