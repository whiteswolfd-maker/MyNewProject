"""Aggregation and stage-boundary tests for the C/D side-effect localiser."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "localize_gate_side_effects",
    ROOT / "scripts" / "localize_gate_side_effects.py",
)
assert _SPEC is not None and _SPEC.loader is not None
localise = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(localise)


def test_stage_boundaries_are_non_overlapping_and_cover_the_interval():
    fraction = np.array([
        0.0, 0.009999, 0.01, 0.049999, 0.05, 0.099999, 0.10, 1.0,
    ])
    masks = [
        localise._stage_mask(fraction, stage)
        for stage in localise.STAGES
    ]
    assert sum(mask.astype(int) for mask in masks).tolist() == [1] * 8
    assert masks[0].tolist() == [True, True, False, False, False, False, False, False]
    assert masks[1].tolist() == [False, False, True, True, False, False, False, False]
    assert masks[2].tolist() == [False, False, False, False, True, True, False, False]
    assert masks[3].tolist() == [False, False, False, False, False, False, True, True]


def test_pooled_aggregate_adds_squared_sums_not_rms():
    rows = [
        {"case_id": "a", "n": 1, "ss_C": 1.0, "ss_D": 4.0},
        {"case_id": "b", "n": 3, "ss_C": 9.0, "ss_D": 9.0},
    ]
    aggregate = localise._aggregate(rows, "x")
    assert aggregate["n"] == 4
    assert aggregate["ss_C"] == 10.0
    assert aggregate["ss_D"] == 13.0
    assert aggregate["mse_C"] == 2.5
    assert aggregate["mse_D"] == 3.25
    assert aggregate["rmse_C"] == math.sqrt(2.5)
    assert aggregate["rmse_D"] == math.sqrt(3.25)
    assert aggregate["diff_ss"] == 3.0
    assert aggregate["n_cases"] == 2


def test_empty_stage_has_no_none_arithmetic():
    aggregate = localise._aggregate([], "x")
    assert aggregate["n"] == 0
    assert aggregate["ss_C"] == 0.0
    assert aggregate["mse_C"] is None
    assert aggregate["rmse_C"] is None
    assert aggregate["diff_ss"] is None
    rows = [
        {"case_id": "a", "n": 0, "ss_C": 0.0, "ss_D": 0.0},
        {"case_id": "b", "n": 2, "ss_C": 2.0, "ss_D": 8.0},
    ]
    aggregate = localise._aggregate(rows, "x")
    assert aggregate["n"] == 2
    assert aggregate["ss_C"] == 2.0
    assert aggregate["ss_D"] == 8.0


def test_delta_stats_separates_positive_negative_and_top_points():
    delta = np.array([-1.0, -1.0, -1.0, 100.0])
    result = localise._delta_stats(delta)
    assert result["n_points"] == 4
    assert result["n_positive"] == 1
    assert result["n_negative"] == 3
    assert result["positive_sum"] == 100.0
    assert result["negative_sum"] == -3.0
    assert result["net_sum"] == 97.0
    assert result["top_1pct_all_n"] == 1
    assert result["top_1pct_all_positive_points"] == 1
    assert result["top_1pct_all_share_of_positive"] == 1.0
    assert result["top_5pct_all_n"] == 1


def test_nonfinite_status_separates_same_different_one_side_and_all():
    same = localise._nonfinite_status(
        3, np.array([True, False, True]), np.array([True, False, True])
    )
    assert same["status"] == "same_position"
    assert same["n_both_nonfinite"] == 1
    assert same["complete"] is False
    different = localise._nonfinite_status(
        3, np.array([False, True, True]), np.array([True, False, True])
    )
    assert different["status"] == "different_positions"
    one_side = localise._nonfinite_status(
        3, np.array([True, False, True]), np.array([True, True, True])
    )
    assert one_side["status"] == "one_side_nan"
    one_side_all = localise._nonfinite_status(
        3, np.array([False, False, False]), np.array([True, True, True])
    )
    assert one_side_all["status"] == "one_side_all_nan"
    all_nan = localise._nonfinite_status(
        3, np.array([False, False, False]), np.array([False, False, False])
    )
    assert all_nan["status"] == "all_nan"
    assert all_nan["n_both_nonfinite"] == 3
    ok = localise._nonfinite_status(
        3, np.array([True, True, True]), np.array([True, True, True])
    )
    assert ok["status"] == "ok"
    assert ok["complete"] is True


def test_nonfinite_summary_propagates_failure():
    records = {
        "a": {
            "n_candidates": 2, "n_nonfinite_C": 0, "n_nonfinite_D": 0,
            "complete": True, "status": "ok",
        },
        "b": {
            "n_candidates": 2, "n_nonfinite_C": 2, "n_nonfinite_D": 2,
            "complete": False, "status": "all_nan",
        },
    }
    summary = localise._nonfinite_summary(records)
    assert summary["complete"] is False
    assert summary["status"] == "incomplete"
    assert summary["n_nonfinite_C"] == 2
    assert summary["n_nonfinite_D"] == 2
    assert summary["failure_cases"] == [{"case_id": "b", "status": "all_nan"}]


def test_time_structure_counts_independent_times_and_spatial_points():
    case = SimpleNamespace(
        case_id="r50_full", role="train", charge_radius_m=0.05,
        t0=0.0, t_end=1.0,
    )
    times = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0])
    radii = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    delta = np.array([1.0, -1.0, 2.0, -2.0, 4.0, 8.0])
    result = localise._time_structure(
        case, times, radii, delta, "energy"
    )
    assert result["n_points"] == 6
    assert result["n_unique_times"] == 3
    assert result["points_per_time_min"] == 1
    assert result["points_per_time_median"] == 2.0
    assert result["points_per_time_max"] == 3
    assert len(result["per_time"]) == 3
    first = result["per_time"][0]
    assert first["n_spatial"] == 3
    assert first["delta_sum"] == 2.0
    assert first["delta_positive_sum"] == 3.0
    assert first["r_min"] == 1.0
    assert first["r_max"] == 3.0
