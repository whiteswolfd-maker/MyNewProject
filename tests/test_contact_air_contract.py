"""Entry-contract regressions for the standard one-dimensional source.

These tests exercise the reader, not the physics.  Invalid data must fail at
the read boundary with the field/frame/index named; no normalization, filling
or silent coercion is allowed.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from spherical_blast_pinn.contact_air.one_d import (
    OneDDataError,
    ShockDetectionConfig,
    detect_frame_shock,
    pure_air_run_between,
    read_standard_1d,
    write_standard_1d,
)

N = 4
TOTAL = 2 * N


def _frame_arrays(*, include_cell_id: bool = True) -> dict[str, np.ndarray]:
    arrays = {
        "radius": np.array([0.05, 0.15, 0.25, 0.35]),
        "r_left": np.array([0.00, 0.10, 0.20, 0.30]),
        "r_right": np.array([0.10, 0.20, 0.30, 0.40]),
        "alpha_product": np.array([1.0, 1.0, 0.0, 0.0]),
        "alpha_air": np.array([0.0, 0.0, 1.0, 1.0]),
        "density": np.array([1630.0, 1630.0, 1.225, 1.225]),
        "velocity": np.array([100.0, 100.0, 0.0, 0.0]),
        "pressure": np.array([1.0e9, 1.0e9, 1.0e5, 1.0e5]),
        "valid_density": np.ones(N, dtype=bool),
        "valid_velocity": np.ones(N, dtype=bool),
        "valid_pressure": np.ones(N, dtype=bool),
    }
    if include_cell_id:
        arrays["cell_id"] = np.arange(N, dtype=np.int64)
    return arrays


def _write_source(tmp_path, *, include_cell_id: bool = True) -> str:
    frame = _frame_arrays(include_cell_id=include_cell_id)
    return str(write_standard_1d(
        tmp_path / "source.npz", case_id="contract", charge_radius_m=0.05,
        frames=[frame, frame], frame_times=[1.0e-5, 2.0e-5],
        source={"kind": "contract_test"},
    ))


def _rewrite(source: str, target, *, metadata=None, arrays=None) -> str:
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    meta = json.loads(str(payload["metadata"]))
    if metadata is not None:
        metadata(meta)
    payload["metadata"] = np.asarray(json.dumps(meta))
    if arrays is not None:
        arrays(payload)
    np.savez(target, **payload)
    return str(target)


def test_frame_times_must_be_finite_and_strictly_increasing(tmp_path):
    source = _write_source(tmp_path)
    cases = [
        (lambda md: md.__setitem__("frame_times", [float("nan"), 2.0e-5]), "frame_times"),
        (lambda md: md.__setitem__("frame_times", [1.0e-5, float("inf")]), "frame_times"),
        (lambda md: md.__setitem__("frame_times", [1.0e-5, 1.0e-5]), "strictly increasing"),
    ]
    for index, (mutate, message) in enumerate(cases):
        broken = _rewrite(source, tmp_path / f"time_{index}.npz", metadata=mutate)
        with pytest.raises(OneDDataError, match=message):
            read_standard_1d(broken)


def test_cell_boundaries_must_be_finite_nonnegative_and_ordered(tmp_path):
    source = _write_source(tmp_path)
    cases = [
        (lambda payload: payload["r_left"].__setitem__(0, np.nan), "r_left"),
        (lambda payload: payload["r_right"].__setitem__(1, np.inf), "r_right"),
        (lambda payload: payload["radius"].__setitem__(2, np.nan), "radius"),
        (lambda payload: payload["radius"].__setitem__(1, payload["radius"][0]), "strictly increasing"),
        (lambda payload: payload["r_left"].__setitem__(0, -0.01), "negative"),
        (lambda payload: payload["r_right"].__setitem__(0, -0.01), "not positive"),
        (lambda payload: payload["radius"].__setitem__(3, 0.5), "outside"),
        (lambda payload: payload["r_left"].__setitem__(1, 0.05), "overlap"),
    ]
    for index, (mutate, message) in enumerate(cases):
        broken = _rewrite(source, tmp_path / f"bounds_{index}.npz", arrays=mutate)
        with pytest.raises(OneDDataError, match=message):
            read_standard_1d(broken)


def test_cell_id_must_be_integer_unique_and_full_length(tmp_path):
    source = _write_source(tmp_path, include_cell_id=False)
    cases = [
        (lambda payload: payload.__setitem__(
            "cell_id", np.arange(TOTAL, dtype=np.float64)), "integer"),
        (lambda payload: payload.__setitem__(
            "cell_id", np.zeros(TOTAL, dtype=np.int64)), "duplicate cell_id"),
        (lambda payload: payload.__setitem__(
            "cell_id", np.arange(TOTAL - 1, dtype=np.int64)), "cell_id"),
    ]
    for index, (mutate, message) in enumerate(cases):
        broken = _rewrite(source, tmp_path / f"cell_{index}.npz", arrays=mutate)
        with pytest.raises(OneDDataError, match=message):
            read_standard_1d(broken)


def test_generated_cell_ids_are_recorded_as_generated(tmp_path):
    source = _write_source(tmp_path, include_cell_id=False)
    radial = read_standard_1d(source)
    assert radial.source["generated_cell_id_frames"] == [0, 1]
    assert radial.source["cell_id_sources"] == ["generated", "generated"]
    assert all(frame.source["cell_id_source"] == "generated" for frame in radial.frames)
    provided = read_standard_1d(_write_source(tmp_path / "provided", include_cell_id=True))
    assert provided.source["generated_cell_id_frames"] == []
    assert provided.source["cell_id_sources"] == ["provided", "provided"]


def test_frame_offsets_and_cells_per_frame_are_strictly_checked(tmp_path):
    source = _write_source(tmp_path)
    cases = [
        (None, lambda payload: payload.__setitem__(
            "frame_offsets", np.array([0, 4, 8], dtype=np.float64)), "integer dtype"),
        (None, lambda payload: payload.__setitem__(
            "frame_offsets", np.array([0, 4, 2, 8], dtype=np.int64)), "strictly increasing"),
        (None, lambda payload: payload.__setitem__(
            "frame_offsets", np.array([1, 5, 9], dtype=np.int64)), "must be 0"),
        (None, lambda payload: payload.__setitem__(
            "frame_offsets", np.array([0, 4, 7], dtype=np.int64)), "does not equal"),
        (None, lambda payload: payload.__setitem__(
            "density", payload["density"][:-1]), "density"),
        (lambda md: md.__setitem__("cells_per_frame", [3, 5]), None, "cells_per_frame"),
        (lambda md: md.__setitem__("cells_per_frame", [4.0, 4.0]), None, "not an integer"),
    ]
    for index, (meta, mutate, message) in enumerate(cases):
        broken = _rewrite(
            source, tmp_path / f"offsets_{index}.npz",
            metadata=meta, arrays=mutate,
        )
        with pytest.raises(OneDDataError, match=message):
            read_standard_1d(broken)


def test_alpha_product_and_alpha_air_must_sum_to_one(tmp_path):
    source = _write_source(tmp_path)
    # alpha_product is ~0, but alpha_air is only 0.5: the cell is NOT pure air
    # and the void/third-material fraction is unsupported.
    broken = _rewrite(
        source, tmp_path / "void.npz",
        arrays=lambda payload: (
            payload.__setitem__("alpha_product", np.zeros(TOTAL)),
            payload.__setitem__("alpha_air", np.full(TOTAL, 0.5)),
        ),
    )
    with pytest.raises(OneDDataError, match="product \\+ air only|not 1"):
        read_standard_1d(broken)

    # Explicit alpha_air is authoritative: a small alpha_product alone would be
    # "air" by the implicit rule, but here alpha_air=0.2 marks a mixture.
    mixture = _rewrite(
        source, tmp_path / "mixture.npz",
        arrays=lambda payload: (
            payload.__setitem__("alpha_product", np.full(TOTAL, 0.8)),
            payload.__setitem__("alpha_air", np.full(TOTAL, 0.2)),
        ),
    )
    radial = read_standard_1d(mixture)
    assert not radial.frames[0].air_mask(0.95).any()


def test_alpha_air_presence_must_be_uniform_across_frames(tmp_path):
    frame = _frame_arrays()
    without_air = {k: v for k, v in frame.items() if k != "alpha_air"}
    # Build a two-frame source where only one frame has alpha_air.
    path = write_standard_1d(
        tmp_path / "mixed_presence.npz", case_id="contract", charge_radius_m=0.05,
        frames=[frame, without_air],
        frame_times=[1.0e-5, 2.0e-5],
    )
    # The concatenated exchange format cannot represent per-frame presence:
    # a frame missing alpha_air shortens the array and is rejected rather than
    # silently mixing material conventions.
    with pytest.raises(OneDDataError, match="alpha_air"):
        read_standard_1d(path)


def test_legal_source_with_a_real_gap_still_reads_and_segments(tmp_path):
    left = np.array([0.09, 0.11, 0.13, 0.25, 0.27, 0.29, 0.31])
    right = left + 0.02
    frame = {
        "radius": 0.5 * (left + right),
        "r_left": left,
        "r_right": right,
        "alpha_product": np.zeros(7),
        "alpha_air": np.ones(7),
        "density": np.full(7, 1.225),
        "velocity": np.zeros(7),
        "pressure": np.array([8.0e6, 8.0e6, 8.0e6, 1.0e5, 1.0e5, 1.0e5, 1.0e5]),
        "valid_density": np.ones(7, dtype=bool),
        "valid_velocity": np.ones(7, dtype=bool),
        "valid_pressure": np.ones(7, dtype=bool),
        "cell_id": np.arange(7, dtype=np.int64),
    }
    path = write_standard_1d(
        tmp_path / "gap.npz", case_id="gap", charge_radius_m=0.05,
        frames=[frame, frame], frame_times=[1.0e-5, 2.0e-5],
    )
    radial = read_standard_1d(path)
    gas = radial.frames[0]
    assert gas.air_mask(0.95).all()
    ok, count = pure_air_run_between(
        gas, low=0.05, high=0.40, pure_air_threshold=0.95, minimum=5,
    )
    assert not ok
    assert count == 4  # the two geometric sides are never joined
    detection = detect_frame_shock(
        gas, contact_up=0.05, ambient_pressure=1.0e5,
        pure_air_threshold=0.95,
        config=ShockDetectionConfig(relative_prominence=2.0),
    )
    assert not detection.valid
    assert detection.run_bounds is None
