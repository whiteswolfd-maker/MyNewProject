"""Regression tests for the read-only similarity-progress diagnostic."""
from __future__ import annotations

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


def test_interp_series_never_uses_endpoint_outside_range():
    result = sim._interp_series(3.0, [1.0, 2.0], [5.0, 6.0])
    assert result["supported"] is False
    assert result["reason"] == "outside_range"
    assert result["value"] is None


def test_invalid_field_is_removed_per_field():
    radius = [1.0, 2.0, 3.0]
    density = [1.0, 999.0, 3.0]
    velocity = [1.0, 2.0, 3.0]
    density_valid = [True, False, True]
    velocity_valid = [True, True, True]
    density_value = sim._interp_field_at_radius(
        radius, density, density_valid, 2.0
    )
    velocity_value = sim._interp_field_at_radius(
        radius, velocity, velocity_valid, 2.0
    )
    assert density_value["supported"]
    assert math.isclose(density_value["value"], 2.0)
    assert velocity_value["supported"]
    assert math.isclose(velocity_value["value"], 2.0)
    invalid_velocity = [1.0, float("nan"), float("nan")]
    missing = sim._interp_field_at_radius(
        radius, invalid_velocity, velocity_valid, 2.0
    )
    assert missing["supported"] is False
    assert missing["reason"] == "insufficient_valid_points"


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
