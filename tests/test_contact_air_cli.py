"""CLI checks; synthetic fixtures only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spherical_blast_pinn.contact_air import cli

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "configs" / "contact_air" / "smoke.yaml"


def test_precheck_reports_the_precursor_and_the_missing_one_d_source(tmp_path):
    output = tmp_path / "precheck.json"
    code = cli.main([
        "precheck", "--config", str(SMOKE), "--case", "tnt_50mm",
        "--output", str(output),
    ])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["precursor_checks"]["V_sep_above_V_cj"] is True
    assert row["initial_condition_verified"] is False
    assert any("standard 1-D source" in item for item in row["missing"])
    assert row["precursor"]["epsilon_formula"].startswith("R_sep /")


def test_train_predict_and_evaluate_with_synthetic_fixtures(tmp_path):
    train_dir = tmp_path / "train"
    code = cli.main([
        "train", "--config", str(SMOKE), "--cases", "all",
        "--max-loops", "1", "--output-dir", str(train_dir),
    ])
    assert code == 0
    checkpoint = train_dir / "last.pt"
    assert checkpoint.is_file()

    prediction = tmp_path / "pred.npz"
    code = cli.main([
        "predict", "--config", str(SMOKE), "--case", "tnt_50mm",
        "--checkpoint", str(checkpoint), "--output", str(prediction),
        "--n-times", "12", "--n-zeta", "12",
    ])
    assert code == 0
    with np.load(prediction) as archive:
        assert archive["density"].shape == (12, 12)
        assert archive["contact_radius"].shape == (12, 12)
        assert archive["shock_radius"].shape == (12, 12)

    report = tmp_path / "eval.json"
    code = cli.main([
        "evaluate", "--config", str(SMOKE), "--case", "tnt_50mm",
        "--prediction", str(prediction), "--output", str(report),
    ])
    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["synthetic"] is True
    assert payload["air"]["n_observations"] > 0
    assert payload["contact"]["n_contact_frames"] > 0


def test_contact_learning_rate_priority_and_resume_restore(tmp_path):
    train_dir = tmp_path / "train"
    code = cli.main([
        "train", "--config", str(SMOKE), "--cases", "all",
        "--max-loops", "1", "--output-dir", str(train_dir),
        "--learning-rate", "2.0e-3", "--contact-learning-rate", "5.0e-4",
    ])
    assert code == 0
    metadata = json.loads(
        (train_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["original_config_learning_rate"] == 1.0e-4
    assert metadata["cli_learning_rate"] == 2.0e-3
    assert metadata["cli_contact_learning_rate"] == 5.0e-4
    assert metadata["effective_learning_rates"] == {
        "contact": 5.0e-4, "air": 2.0e-3,
    }
    result = json.loads(
        (train_dir / "train_result.json").read_text(encoding="utf-8")
    )
    assert result["history"]["loops"][0]["contact_learning_rate"] == 5.0e-4
    assert result["history"]["loops"][0]["air_learning_rate"] == 2.0e-3

    # Resume with no learning-rate flags: both restored optimizer values stay.
    code = cli.main([
        "train", "--config", str(SMOKE), "--cases", "all",
        "--max-loops", "2", "--output-dir", str(train_dir),
        "--resume", str(train_dir / "last.pt"),
    ])
    assert code == 0
    metadata = json.loads(
        (train_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["cli_learning_rate"] is None
    assert metadata["cli_contact_learning_rate"] is None
    assert metadata["effective_learning_rates"] == {
        "contact": 5.0e-4, "air": 2.0e-3,
    }


def test_the_smoke_config_has_only_one_50mm_program_check_case():
    from spherical_blast_pinn.config import load_config

    section = load_config(SMOKE).section("contact_air")
    assert [case["case_id"] for case in section["cases"]] == ["tnt_50mm"]
    assert section["cases"][0]["synthetic"] is True
    assert section["cases"][0]["role"] == "train"


def test_base_config_does_not_encode_a_default_radius_split():
    from spherical_blast_pinn.config import load_config

    section = load_config(ROOT / "configs" / "contact_air" / "base.yaml").section("contact_air")
    assert section["cases"] == []


def test_checkpoint_describe_treats_a_missing_time_reference_as_seconds():
    from spherical_blast_pinn.contact_air.trainer import describe_matches

    current_scaled = {"name": "ContactNet", "time_reference_s": 0.01}
    current_seconds = {"name": "ContactNet", "time_reference_s": 1.0}
    old_saved = {"name": "ContactNet"}  # written before the scaling existed
    assert describe_matches(old_saved, current_seconds) is True
    assert describe_matches(old_saved, current_scaled) is False
    assert describe_matches({"name": "ContactNet", "time_reference_s": 0.01},
                            current_scaled) is True
    # A missing description block cannot be verified, so it is not accepted.
    assert describe_matches(None, current_scaled) is False
