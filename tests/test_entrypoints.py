"""Entry-point and configuration consistency.

The cleanup this file guards:

* ``configs/base.yaml`` trains from ``startup.cases`` (50/80/100 mm) and keeps
  500 mm in ``splits.final_validation_cases``;
* it no longer carries the legacy ``data.cases`` / ``allow_unverified_cases``
  / ``compatibility_report`` keys that the old gate read;
* ``validate_eos`` runs without a legacy extracted-profile inventory;
* ``audit_data`` refuses a training config with a clear message instead of a
  traceback, because it audits legacy extracted profiles only.
"""
from __future__ import annotations

from pathlib import Path

from spherical_blast_pinn.cli import audit_data, validate_eos
from spherical_blast_pinn.config import load_config

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "configs" / "base.yaml"


def test_base_config_uses_only_startup_cases_for_training():
    config = load_config(BASE)
    startup_cases = config.get("startup.cases", [])
    assert [c["case_id"] for c in startup_cases] == [
        "tnt_50mm", "tnt_80mm", "tnt_100mm",
    ]
    assert all(str(c["split"]) == "train" for c in startup_cases)
    assert 0.5 not in {float(c["charge_radius_m"]) for c in startup_cases}


def test_base_config_keeps_500mm_as_final_validation_only():
    config = load_config(BASE)
    validation = config.get("splits.final_validation_cases", [])
    assert [c["case_id"] for c in validation] == ["tnt_500mm"]
    assert float(validation[0]["charge_radius_m"]) == 0.5
    assert str(validation[0]["split"]) != "train"
    assert config.get("splits.final_validation_radii_mm") == [500.0]


def test_base_config_has_no_legacy_training_gate_keys():
    config = load_config(BASE)
    for key in (
        "data.cases",
        "data.allow_unverified_cases",
        "data.compatibility_report",
        "data.synthetic_smoke",
    ):
        assert config.get(key, None) in (None, []), (
            f"{key!r} is still present; the new training entry does not read it"
        )


def test_validate_eos_runs_without_legacy_extracted_profiles(tmp_path):
    code = validate_eos.main([
        "--config", str(BASE), "--output-dir", str(tmp_path / "eos"),
    ])
    assert code == 0
    report = (tmp_path / "eos" / "eos_validation.json").read_text(encoding="utf-8")
    assert "reference_pressure" in report
    assert "computed_pa" in report


def test_audit_data_reports_missing_legacy_inventory_without_traceback(
    tmp_path, capsys
):
    code = audit_data.main(["--config", str(BASE)])
    captured = capsys.readouterr()
    assert code == 6
    assert "data.cases" in captured.err
    assert "training uses startup.cases" in captured.err


def test_compatibility_module_no_longer_exposes_a_training_gate():
    """The old gate was a separate code path with no production caller."""
    from spherical_blast_pinn.data import compatibility as compat

    assert not hasattr(compat, "enforce_training_gate")
    assert not hasattr(compat, "GateDecision")
