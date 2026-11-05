"""Smoke-level CLI checks; synthetic fixtures only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spherical_blast_pinn.contact_air import cli

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "configs" / "contact_air" / "smoke.yaml"


def test_precheck_reports_the_precursor_and_the_missing_rb(tmp_path):
    output = tmp_path / "precheck.json"
    code = cli.main([
        "precheck", "--config", str(SMOKE), "--case", "tnt_50mm",
        "--output", str(output),
    ])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["precursor_checks"]["V_sep_above_V_cj"] is True
    assert payload["initial_condition_verified"] is False
    assert any("R_b(t0)" in item for item in payload["missing"])


def test_train_predict_and_evaluate_500_with_synthetic_fixtures(tmp_path):
    train_dir = tmp_path / "train"
    code = cli.main([
        "train", "--config", str(SMOKE), "--cases", "50",
        "--max-loops", "1", "--output-dir", str(train_dir),
    ])
    assert code == 0
    checkpoint = train_dir / "last.pt"
    assert checkpoint.is_file()

    prediction = tmp_path / "pred500.npz"
    code = cli.main([
        "predict-500", "--config", str(SMOKE), "--checkpoint", str(checkpoint),
        "--output", str(prediction), "--n-times", "12", "--n-zeta", "12",
    ])
    assert code == 0
    with np.load(prediction) as archive:
        assert archive["density"].shape == (12, 12)
        assert archive["contact_radius"].shape == (12, 12)

    report = tmp_path / "eval500.json"
    code = cli.main([
        "evaluate-500", "--config", str(SMOKE), "--prediction", str(prediction),
        "--output", str(report),
    ])
    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["synthetic"] is True
    assert payload["air"]["n_observations"] > 0
    assert payload["contact"]["n_contact_frames"] > 0
