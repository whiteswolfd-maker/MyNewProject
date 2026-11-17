"""Regression tests for the read-only similarity-progress diagnostic."""
from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "check_similarity_progress",
    ROOT / "scripts" / "check_similarity_progress.py",
)
assert _SPEC is not None and _SPEC.loader is not None
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)


def _curve(case_id, low, up, shock):
    theta = [0.0, 1.0, 2.0]
    return {
        "case_id": case_id,
        "theta": theta,
        "contact_low_over_Rsep": low,
        "contact_up_over_Rsep": up,
        "shock_over_Rsep": shock,
    }


def test_identical_curves_have_zero_lower_and_upper_spread_but_nonzero_envelope():
    low = [1.0, 1.1, 1.2]
    up = [1.002, 1.102, 1.202]
    shock = [1.2, 1.3, 1.4]
    metrics = sim._curve_metrics([
        _curve("a", low, up, shock),
        _curve("b", low, up, shock),
    ])
    assert metrics["contact_low_over_Rsep"]["spread_median"] == 0.0
    assert metrics["contact_up_over_Rsep"]["spread_median"] == 0.0
    assert math.isclose(
        metrics["contact_envelope"]["median"], 0.002, rel_tol=0.0, abs_tol=1e-12
    )


def test_bracket_rejects_extrapolation_and_unchecked_large_gap():
    outside = sim._bracket(60.0, [10.0, 20.0])
    assert outside["supported"] is False
    assert outside["reason"] == "outside_range"
    gap = sim._bracket(30.0, [10.0, 11.0, 12.0, 50.0])
    assert gap["supported"] is False
    assert gap["reason"] == "large_gap"


def test_repeated_large_gap_is_only_a_candidate():
    # Median interval 1, three repeated intervals of 3.  The old code called
    # this "sparse_output_cadence" without checking the raw frame manifest.
    times = np.array([0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0, 15.0])
    bracket = sim._bracket(3.5, times)
    assert bracket["supported"] is False
    assert bracket["reason"] == "large_gap"
    assert bracket["gap_kind"] == "repeated_large_gap_candidate"
    assert bracket["gap_kind_is_candidate"] is True
    assert bracket["gap_repeat_count"] >= 3
    assert bracket["gap_repeat_run_max"] == 1
    assert "sparse_output_cadence" not in bracket["gap_kind"]


def test_plan_gap_interpretation_needs_verified_plan_and_records():
    unverified = {"standard_available": False}
    assert sim._plan_gap_interpretation(unverified, []) == \
        "unverified_candidate"
    verified = {
        "standard_available": True,
        "all_frames_imported": True,
        "time_source": "profile_filename_3sigfig",
    }
    assert sim._plan_gap_interpretation(verified, []) == "no_large_gaps"
    isolated = [{"gap_repeat_run_max": 1, "n_raw_between": 0,
                 "n_imported_between": 0, "n_observed_between": 0}]
    assert sim._plan_gap_interpretation(verified, isolated) == \
        "isolated_repeated_gap_time_key_quantization_candidate"
    sustained = [{"gap_repeat_run_max": 12, "n_raw_between": 0,
                  "n_imported_between": 0, "n_observed_between": 0}]
    assert sim._plan_gap_interpretation(verified, sustained) == \
        "sustained_raw_output_cadence"
    dropped = [{"gap_repeat_run_max": 1, "n_raw_between": 1,
                "n_imported_between": 0, "n_observed_between": 0}]
    assert sim._plan_gap_interpretation(verified, dropped) == \
        "raw_frame_not_imported"
    filtered = {
        "standard_available": True,
        "all_frames_imported": False,
        "time_source": "profile_filename_3sigfig",
    }
    assert sim._plan_gap_interpretation(filtered, isolated) == \
        "filter_or_import_gap"


def test_contact_gap_classification_uses_label_evidence_not_air_cadence():
    assert sim._classify_contact_gap(
        raw_intermediate_count=0,
        imported_intermediate_count=0,
        invalid_contact_count=0,
        filter_excluded_count=0,
    ) == "raw_no_intermediate_frame"
    assert sim._classify_contact_gap(
        raw_intermediate_count=2,
        imported_intermediate_count=1,
        invalid_contact_count=0,
        filter_excluded_count=0,
    ) == "import_missing"
    assert sim._classify_contact_gap(
        raw_intermediate_count=2,
        imported_intermediate_count=2,
        invalid_contact_count=2,
        filter_excluded_count=0,
    ) == "label_invalid"
    assert sim._classify_contact_gap(
        raw_intermediate_count=2,
        imported_intermediate_count=2,
        invalid_contact_count=0,
        filter_excluded_count=1,
    ) == "filter_excluded"
    assert sim._classify_contact_gap(
        raw_intermediate_count=2,
        imported_intermediate_count=2,
        invalid_contact_count=0,
        filter_excluded_count=0,
    ) == "insufficient_evidence"


def test_curve_windows_count_each_theta_once_and_do_not_use_q():
    theta = [0.5, 1.0, 1.5, 30.0, 30.5]
    curves = [
        {
            "case_id": "a",
            "theta": theta,
            "shock_over_Rsep": [2.0, 2.0, 2.0, 3.0, 3.0],
            "contact_low_over_Rsep": [1.0, 1.0, 1.0, 1.2, 1.2],
            "contact_up_over_Rsep": [1.1, 1.1, 1.1, 1.3, 1.3],
        },
        {
            "case_id": "b",
            "theta": theta,
            "shock_over_Rsep": [2.5, None, 2.5, 3.1, 3.1],
            "contact_low_over_Rsep": [1.0, 1.0, 1.0, 1.2, 1.2],
            "contact_up_over_Rsep": [1.1, 1.1, 1.1, 1.3, 1.3],
        },
    ]
    windows = sim._curve_windows(curves)
    early = windows["0.5-10"]
    assert early["n_theta"] == 3
    assert early["shock_spread"]["n_theta"] == 3
    assert early["shock_spread"]["n_theta_complete"] == 2
    assert early["contact_low_spread"]["n_theta_complete"] == 3
    assert early["contact_low_spread"]["median"] == 0.0
    late = windows["30-60"]
    assert late["n_theta"] == 2
    # Each theta contributes one spread value, never one per q fraction.
    assert late["shock_spread"]["n"] == 2


def test_interp_series_never_uses_endpoint_outside_range():
    result = sim._interp_series(3.0, [1.0, 2.0], [5.0, 6.0])
    assert result["supported"] is False
    assert result["reason"] == "outside_range"
    assert result["value"] is None


def test_invalid_field_is_removed_per_field_and_adjacency_is_kept():
    radius = [1.0, 2.0, 3.0]
    density = [1.0, 999.0, 3.0]
    velocity = [1.0, 2.0, 3.0]
    density_valid = [True, False, True]
    velocity_valid = [True, True, True]
    density_at_invalid = sim._interp_field_at_radius(
        radius, density, density_valid, 2.0
    )
    velocity_at_same = sim._interp_field_at_radius(
        radius, velocity, velocity_valid, 2.0
    )
    assert density_at_invalid["supported"] is False
    assert density_at_invalid["reason"] == "query_on_invalid_point"
    assert velocity_at_same["supported"]
    assert math.isclose(velocity_at_same["value"], 2.0)
    across_invalid = sim._interp_field_at_radius(
        radius, density, density_valid, 1.2
    )
    assert across_invalid["supported"] is False
    assert across_invalid["reason"] == "spans_invalid_point"
    exact_valid = sim._interp_field_at_radius(
        radius, density, density_valid, 1.0
    )
    assert exact_valid["supported"]
    assert math.isclose(exact_valid["value"], 1.0)

def test_grid_spacing_divides_by_scale_once():
    assert math.isclose(
        sim._grid_spacing([0.0, 0.4, 0.8], 2.0), 0.2,
        rel_tol=0.0, abs_tol=1e-15,
    )


def test_event_progress_uses_model_offset_not_raw_time():
    assert math.isclose(
        sim._event_progress(1.1, 1.0, 1.0, -0.1), 0.0,
        rel_tol=0.0, abs_tol=1e-15,
    )
    assert math.isclose(
        sim._event_progress(1.1, 1.0, 1.0, 0.0), 0.1,
        rel_tol=0.0, abs_tol=1e-15,
    )


def test_common_layer_uses_upper_contact_bound():
    low = np.array([1.0, 1.1])
    up = np.array([1.2, 1.3])
    shock = np.array([2.0, 2.1])
    lower, upper, width = sim._common_layer(low, up, shock)
    assert math.isclose(lower, 1.3)
    assert math.isclose(upper, 2.0)
    assert math.isclose(width, 0.7)


import json
from types import SimpleNamespace

import pytest


class _FakeObservations:
    def __init__(self):
        times = np.arange(0.0, 65.0, 1.0)
        self.frame_index = np.arange(times.size)
        self.time = times
        self.radius = np.tile(np.arange(1.0, 10.0, 1.0), (times.size, 1)).reshape(-1)
        self.frame_index = np.repeat(np.arange(times.size), 9)
        self.time = np.repeat(times, 9)
        self.density = np.full(self.radius.size, 1.0)
        self.velocity = np.full(self.radius.size, 0.5)
        self.pressure = np.full(self.radius.size, 0.2)
        self.valid_density = np.ones(self.radius.size, dtype=bool)
        self.valid_velocity = np.ones(self.radius.size, dtype=bool)
        self.valid_pressure = np.ones(self.radius.size, dtype=bool)


class _FakeContact:
    def __init__(self):
        self.times = np.arange(0.0, 65.0, 1.0)
        self.low = 1.0 + 0.05 * self.times
        self.up = self.low + 0.001
        self.valid = np.ones(self.times.size, dtype=bool)
        self.reasons = [""] * self.times.size

    def indices(self):
        return np.flatnonzero(self.valid)


class _FakeShock:
    def has_slope(self, _time):
        return True

    def shock_tensors(self, times):
        return np.asarray(times) * 0.1 + 2.0, np.ones_like(times)


class _FakeCase:
    def __init__(self, case_id, role):
        self.case_id = case_id
        self.role = role
        self.t0 = 0.0
        self.t_end = 64.0
        self.time_scale = 1.0
        self.length_scale = 1.0
        self.charge_radius_m = 0.5
        self.rho_scale = 1.0
        self.u_scale = 1.0
        self.p_scale = 1.0
        self.observations = _FakeObservations()
        self.contact = _FakeContact()
        self.shock = _FakeShock()
        self.initial = SimpleNamespace(Rs0=1.03, Rc0=1.025, Rs_data_at_t0=1.09)
        self.event_alignment = {
            "event_frame": 19,
            "event_time_raw_s": 0.0,
            "offset_s": 0.0,
            "event_in_first_frame": False,
            "preceding_frame_confirmed": False,
        }
        self.source = {
            "data_entry": {
                "metadata": {"source_dir": "synthetic", "cell_count": 50000}
            }
        }

    def shock_tensors(self, times):
        return self.shock.shock_tensors(times)


class _FakeTrainer:
    def __init__(self):
        self.cases = [_FakeCase("synthetic_train", "train")]
        self.validation_cases = [_FakeCase("synthetic_val", "validation")]


def test_main_pipeline_with_synthetic_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sim.diag, "_build_trainer",
        lambda _config: (_FakeTrainer(), {}),
    )
    rc = sim.main(["--output", str(tmp_path)])
    assert rc == 0
    payload = json.loads(
        (tmp_path / "similarity_check.json").read_text(encoding="utf-8")
    )
    assert payload["completeness"]["complete"] is True
    assert payload["assessment"]["status"] == "not_assessed"
    assert payload["groups"]["train"]["theta_counts"]["shock_over_Rsep"][
        "n_complete"
    ] > 0
    assert payload["groups"]["train"]["query_counts"]["field_complete"][
        "rho"
    ]["n_complete"] > 0
    assert (tmp_path / "query_evidence.csv").exists()
    assert (tmp_path / "curve_windows.csv").exists()
    assert payload["query_evidence_count"] > 0
    assert payload["groups"]["train"]["curve_windows"]["30-60"]["n_theta"] == 60
    assert payload["groups"]["train"]["air_windows"]["30-60"][
        "n_theta_q_queries"
    ] == 60 * len(sim.Q_FRACTIONS)
    assert payload["groups"]["train"]["frame_plan_summary"][
        "interpretation_counts"
    ] == {"unverified_candidate": 1}
    with (tmp_path / "query_evidence.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        header = next(csv.reader(handle))
    assert "lower_frame_radius_min" in header
    assert "upper_frame_radius_max" in header


def test_fields_records_front_and_back_frame_support_separately():
    data = sim.CaseData1D(_FakeCase("synthetic_train", "train"))
    result = data.fields(10.5, 5.0)["rho"]
    assert result["supported"] is True
    lower = result["lower_frame_support"]
    upper = result["upper_frame_support"]
    assert lower["frame_id"] == 10
    assert upper["frame_id"] == 11
    assert math.isclose(lower["theta"], 10.0)
    assert math.isclose(upper["theta"], 11.0)
    assert lower["radius_min"] == 1.0
    assert lower["radius_max"] == 9.0
    assert upper["radius_min"] == 1.0
    assert upper["radius_max"] == 9.0
    assert lower["n_valid"] == 9
    assert upper["n_valid"] == 9


def test_fields_keeps_both_frame_failures_when_both_are_invalid():
    case = _BadDensityCase("synthetic_train", "train")
    data = sim.CaseData1D(case)
    result = data.fields(10.5, 5.0)["rho"]
    assert result["supported"] is False
    assert result["reason"] == "insufficient_valid_points"
    lower = result["lower_frame_support"]
    upper = result["upper_frame_support"]
    assert lower["supported"] is False
    assert upper["supported"] is False
    assert lower["reason"] == "insufficient_valid_points"
    assert upper["reason"] == "insufficient_valid_points"
    assert lower["frame_id"] == 10
    assert upper["frame_id"] == 11


class _BadDensityCase(_FakeCase):
    def __init__(self, case_id, role):
        super().__init__(case_id, role)
        self.observations.valid_density = np.zeros(
            self.observations.radius.size, dtype=bool
        )


class _BadDensityTrainer(_FakeTrainer):
    def __init__(self):
        self.cases = [_BadDensityCase("synthetic_train", "train")]
        self.validation_cases = [_BadDensityCase("synthetic_val", "validation")]


class _ContactExcludedCase(_FakeCase):
    """Air frames are continuous, but two middle contact labels are invalid."""

    def __init__(self, case_id, role):
        super().__init__(case_id, role)
        self.contact.valid[29:31] = False


class _ContactExcludedTrainer(_FakeTrainer):
    def __init__(self):
        self.cases = [_ContactExcludedCase("synthetic_train", "train")]
        self.validation_cases = [
            _ContactExcludedCase("synthetic_val", "validation")
        ]


def test_contact_label_gap_is_not_explained_by_continuous_air_frames(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        sim.diag, "_build_trainer",
        lambda _config: (_ContactExcludedTrainer(), {}),
    )
    rc = sim.main(["--output", str(tmp_path)])
    assert rc == 1
    payload = json.loads(
        (tmp_path / "similarity_check.json").read_text(encoding="utf-8")
    )
    plan = payload["groups"]["train"]["cases"][0]["frame_plan"]
    # The air-frame audit sees a continuous cadence and cannot explain this.
    assert plan["large_gap_count"] == 0
    assert plan["interpretation"] == "unverified_candidate"
    evidence = plan["contact_gap_evidence"]
    assert evidence["interval_count"] == 1
    interval = evidence["intervals"][0]
    assert interval["classification"] == "label_invalid"
    assert interval["classification"] != "raw_no_intermediate_frame"
    assert interval["raw_manifest_source"] == "air_frames_proxy"
    assert interval["raw_intermediate_count"] == 2
    assert interval["invalid_contact_intermediate_count"] == 2
    assert interval["failed_contact_low"] == 5
    assert interval["failed_contact_up"] == 5
    summary = payload["groups"]["train"]["contact_gap_summary"]
    assert summary["classification_counts"] == {"label_invalid": 1}
    assert summary["failed_field_counts"] == {
        "contact_low": 5, "contact_up": 5,
    }
    assert summary["solver_deck_output_plan_read"] is False
    assert (tmp_path / "contact_gap_evidence.csv").exists()


def test_main_flow_keeps_other_fields_when_density_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sim.diag, "_build_trainer",
        lambda _config: (_BadDensityTrainer(), {}),
    )
    rc = sim.main(["--output", str(tmp_path)])
    assert rc == 1
    payload = json.loads(
        (tmp_path / "similarity_check.json").read_text(encoding="utf-8")
    )
    assert payload["completeness"]["complete"] is False
    counts = payload["groups"]["train"]["query_counts"]["field_complete"]
    assert counts["rho"]["n_complete"] == 0
    assert counts["u"]["n_complete"] > 0
    assert counts["p"]["n_complete"] > 0
    assert payload["completeness"]["failed_query_field_by_field"]["rho"] > 0
    assert payload["completeness"]["failed_query_field_by_field"].get("u", 0) == 0
