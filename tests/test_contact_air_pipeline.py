"""End-to-end chain through the real file reader and CLI entry points.

The chain is exercised in separate processes:

standard 1-D source -> build-cache -> train -> resume -> predict -> evaluate

The source is a clearly labelled synthetic engineering fixture.  This proves
the reader/CLI/cache/checkpoint plumbing runs; it is not TNT validation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.contact_air.data import load_case, synthetic_radial_case
from spherical_blast_pinn.contact_air.one_d import write_standard_1d
from spherical_blast_pinn.contact_air.testing import default_precursor

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "configs" / "contact_air" / "base.yaml"
_FIELDS = (
    "cell_id", "r_left", "r_right", "radius", "alpha_product", "alpha_air",
    "material", "density", "velocity", "pressure", "valid_density",
    "valid_velocity", "valid_pressure",
)


def _run(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "spherical_blast_pinn.cli.contact_air", *args],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )


def _config(tmp_path: Path, cache: Path, output: Path, name: str = "config.yaml") -> Path:
    section = dict(load_config(BASE).section("contact_air"))
    section["cases"] = [{
        "case_id": "tnt_50mm", "role": "train", "charge_radius_m": 0.05,
        "cache": str(cache), "synthetic": True,
    }]
    section["networks"] = {
        "dtype": "float64",
        "contact": {"width": 8, "depth": 1},
        "air": {"width": 8, "depth": 1},
    }
    section["sampling"] = {
        "n_pde": 16, "n_ic": 8, "n_right": 8, "n_data": 16, "n_contact": 8,
    }
    section["training"] = {
        "loops": 1, "learning_rate": 1.0e-3, "n_schedule": 4,
        "log_every": 0, "checkpoint_every": 1, "validate_every": 2, "seed": 7,
    }
    section["validation"] = {
        "holdout_fraction": 0.3, "data_points": 16, "contact_points": 16,
    }
    section["output_dir"] = str(output)
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"contact_air": section}), encoding="utf-8")
    return path


def test_full_file_to_prediction_chain_in_new_processes(tmp_path):
    source = tmp_path / "standard_1d.npz"
    cache = tmp_path / "cache.npz"
    output = tmp_path / "run"
    config = _config(tmp_path, cache, output)

    precursor = default_precursor(0.05)
    radial = synthetic_radial_case(
        case_id="tnt_50mm", charge_radius_m=0.05, precursor=precursor,
        n_frames=21, n_radial=1600,
    )
    write_standard_1d(
        source, case_id="tnt_50mm", charge_radius_m=0.05,
        frames=[{field: getattr(frame, field) for field in _FIELDS}
                for frame in radial.frames],
        frame_times=[frame.time_raw_s for frame in radial.frames],
        source={"kind": "synthetic_engineering_fixture", "synthetic": True},
    )

    result = _run(
        "build-cache", "--config", str(config), "--case", "tnt_50mm",
        "--radius", "0.05", "--source", str(source), "--out", str(cache),
        "--role", "train",
    )
    assert result.returncode == 0, result.stderr
    assert cache.is_file()
    case = load_case(cache)
    assert case.shock.valid.sum() >= 2
    assert case.contact.n_valid > 0
    assert case.observations.size > 0

    result = _run(
        "train", "--config", str(config), "--cases", "all",
        "--max-loops", "1", "--output-dir", str(output),
    )
    assert result.returncode == 0, result.stderr
    checkpoint = output / "last.pt"
    assert checkpoint.is_file()

    result = _run(
        "train", "--config", str(config), "--cases", "all",
        "--max-loops", "2", "--output-dir", str(output),
        "--resume", str(checkpoint),
    )
    assert result.returncode == 0, result.stderr

    prediction = tmp_path / "prediction.npz"
    result = _run(
        "predict", "--config", str(config), "--case", "tnt_50mm",
        "--checkpoint", str(checkpoint), "--output", str(prediction),
        "--n-times", "8", "--n-zeta", "8",
    )
    assert result.returncode == 0, result.stderr
    with np.load(prediction) as archive:
        assert archive["density"].shape == (8, 8)
        assert archive["contact_radius"].shape == (8, 8)
        assert archive["shock_radius"].shape == (8, 8)

    report = tmp_path / "evaluation.json"
    result = _run(
        "evaluate", "--config", str(config), "--case", "tnt_50mm",
        "--prediction", str(prediction), "--output", str(report),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["synthetic"] is True
    assert payload["air"]["n_observations"] > 0
    assert payload["contact"]["n_contact_frames"] > 0

    # Prediction must work when only metadata + shock trajectory remain; the
    # reference observations and contact interval are intentionally absent.
    stripped = tmp_path / "prediction_only.npz"
    with np.load(cache, allow_pickle=False) as archive:
        prediction_only = {
            key: np.asarray(archive[key])
            for key in ("metadata", "shock_times", "shock_radius", "shock_valid")
        }
    np.savez(stripped, **prediction_only)
    prediction_config = _config(tmp_path, stripped, output, name="prediction_config.yaml")
    prediction2 = tmp_path / "prediction_only_output.npz"
    result = _run(
        "predict", "--config", str(prediction_config), "--case", "tnt_50mm",
        "--checkpoint", str(checkpoint), "--output", str(prediction2),
        "--n-times", "6", "--n-zeta", "6",
    )
    assert result.returncode == 0, result.stderr
    with np.load(prediction2) as archive:
        assert archive["density"].shape == (6, 6)
