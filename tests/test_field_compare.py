"""材料侧场量对照的逐项不变量（新公共实现）。"""

from __future__ import annotations

import numpy as np
import pytest

from spherical_blast_pinn.evaluation.field_compare import compare_field_blocks


def _predicted(radius=(0.01, 0.02, 0.03, 0.04), values=(1.0, 2.0, 3.0, 4.0)):
    return np.asarray(radius), np.asarray(values)


def test_an_identical_profile_has_zero_error():
    radius, values = _predicted()
    result = compare_field_blocks(
        reference_radius=radius,
        reference_values=values,
        valid=np.ones(radius.size, dtype=bool),
        predicted_radius=radius,
        predicted_values=values,
        low=0.0,
        high=0.05,
    )
    assert result["relative_l2"] == pytest.approx(0.0, abs=1e-15)
    assert result["counts"]["compared"] == radius.size


def test_invalid_points_are_skipped_per_field_not_per_row():
    radius, values = _predicted()
    valid = np.array([True, False, True, True])
    result = compare_field_blocks(
        reference_radius=radius,
        reference_values=values,
        valid=valid,
        predicted_radius=radius,
        predicted_values=values,
        low=0.0,
        high=0.05,
    )
    assert result["n_points"] == 3
    assert result["counts"]["invalid_field"] == 1
    assert result["relative_l2"] == pytest.approx(0.0, abs=1e-15)


def test_a_reference_point_outside_the_predicted_block_is_not_interpolated():
    radius, values = _predicted()
    result = compare_field_blocks(
        reference_radius=np.array([0.02, 0.06]),
        reference_values=np.array([2.0, 99.0]),
        valid=np.array([True, True]),
        predicted_radius=radius,
        predicted_values=values,
        low=0.0,
        high=0.05,
    )
    assert result["counts"]["outside_predicted_block"] == 1
    assert result["n_points"] == 1


def test_non_finite_prediction_is_counted_and_never_treated_as_a_value():
    radius, values = _predicted()
    predicted = np.array([1.0, np.nan, 3.0, 4.0])
    result = compare_field_blocks(
        reference_radius=radius,
        reference_values=values,
        valid=np.ones(radius.size, dtype=bool),
        predicted_radius=radius,
        predicted_values=predicted,
        low=0.0,
        high=0.05,
    )
    assert result["counts"]["predicted_non_finite"] == 1
    assert result["n_points"] == 3


def test_a_field_with_no_comparable_point_is_missing_not_zero():
    radius, values = _predicted()
    result = compare_field_blocks(
        reference_radius=radius,
        reference_values=values,
        valid=np.zeros(radius.size, dtype=bool),
        predicted_radius=radius,
        predicted_values=values,
        low=0.0,
        high=0.05,
    )
    assert result["relative_l2"] is None
    assert result["n_points"] == 0
    assert "no valid" in result["missing_reason"]


def test_a_zero_reference_norm_has_no_defined_relative_error():
    radius, values = _predicted(values=(0.0, 0.0, 0.0, 0.0))
    result = compare_field_blocks(
        reference_radius=radius,
        reference_values=values,
        valid=np.ones(radius.size, dtype=bool),
        predicted_radius=radius,
        predicted_values=values,
        low=0.0,
        high=0.05,
    )
    assert result["relative_l2"] is None
    assert result["rms"] == pytest.approx(0.0)
