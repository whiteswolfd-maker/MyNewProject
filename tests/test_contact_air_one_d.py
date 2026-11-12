"""One-dimensional entry: shock detection, event alignment, RH and the bridge."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from spherical_blast_pinn.contact_air.data import (
    ContactLabelConfig,
    build_case,
    synthetic_radial_case,
)
from spherical_blast_pinn.contact_air.losses import finite_mach_rh_state
from spherical_blast_pinn.contact_air.one_d import (
    OneDDataError,
    ShockDetectionConfig,
    align_event,
    contact_interval_from_fractions,
    detect_frame_shock,
)
from spherical_blast_pinn.contact_air.testing import (
    DEFAULT_CONTACT,
    DEFAULT_SHOCK,
    DEFAULT_TOLERANCES,
    default_precursor,
)

from tests.contact_air_fixtures import make_frame as _frame
from tests.contact_air_fixtures import make_gapped_air_frame, make_nonuniform_air_frame as _nonuniform_air_frame


def test_shock_uses_only_the_pressure_gradient_on_a_nonuniform_grid():
    radius = [0.10, 0.13, 0.17, 0.20, 0.24, 0.30, 0.38]
    pressures = [8.0e6, 8.0e6, 8.0e6, 1.0e5, 1.0e5, 1.0e5, 1.0e5]
    frame = _nonuniform_air_frame(pressures, radius=radius)
    detection = detect_frame_shock(
        frame, contact_up=0.09, ambient_pressure=1.0e5, pure_air_threshold=0.95,
        config=ShockDetectionConfig(relative_prominence=2.0,
                                    min_pressure_jump_fraction=1e-3),
    )
    assert detection.valid, detection.reasons
    assert 0.17 <= detection.radius <= 0.24
    assert detection.run_bounds == (0, 6)
    assert detection.states["p_inner_pa"] > detection.states["p_outer_pa"]


def test_a_gradient_is_not_taken_across_a_missing_cell():
    radius = [0.10, 0.13, 0.17, 0.20, 0.24, 0.30, 0.38]
    pressures = [8.0e6, 8.0e6, 8.0e6, 1.0e5, 1.0e5, 1.0e5, 1.0e5]
    frame = _nonuniform_air_frame(
        pressures, radius=radius,
        valid_pressure=[True, True, True, False, True, True, True],
    )
    detection = detect_frame_shock(
        frame, contact_up=0.10, ambient_pressure=1.0e5, pure_air_threshold=0.95,
        config=ShockDetectionConfig(relative_prominence=2.0),
    )
    # The only candidate sits at the end of the first run, with no inner/outer
    # pair inside that run.  The gap must not be bridged.
    assert not detection.valid


def test_a_candidate_that_moves_inward_relative_to_the_previous_frame_is_rejected():
    from spherical_blast_pinn.contact_air.one_d import detect_shock_trajectory
    radius = [0.10, 0.15, 0.19, 0.21, 0.25, 0.30]
    first = _nonuniform_air_frame(
        [8.0e6, 8.0e6, 8.0e6, 1.0e5, 1.0e5, 1.0e5], radius=radius
    )
    second = _nonuniform_air_frame(
        [8.0e6, 8.0e6, 1.0e5, 1.0e5, 1.0e5, 1.0e5], radius=radius
    )
    first.time_raw_s, second.time_raw_s = 1.0e-5, 2.0e-5
    radii, valid, detections = detect_shock_trajectory(
        [first, second], contact_low=np.array([0.09, 0.09]),
        contact_up=np.array([0.09, 0.09]),
        contact_valid=np.ones(2, bool), ambient_pressure=1.0e5,
        pure_air_threshold=0.95,
        config=ShockDetectionConfig(relative_prominence=2.0,
                                    min_pressure_jump_fraction=1e-3),
    )
    assert valid[0], detections[0].reasons
    assert not valid[1]
    assert "moved inward" in detections[1].reasons[0]


def test_a_real_spatial_gap_is_not_bridged_by_consecutive_indices():
    frame = make_gapped_air_frame()
    detection = detect_frame_shock(
        frame, contact_up=0.05, ambient_pressure=1.0e5,
        pure_air_threshold=0.95,
        config=ShockDetectionConfig(relative_prominence=2.0,
                                    min_pressure_jump_fraction=1e-3),
    )
    # The jump 8e6 -> 1e5 lies across the [0.15, 0.25] void and must not be
    # used as a gradient.  Neither side contains a peak.
    assert not detection.valid
    assert detection.run_bounds is None


def test_a_spatial_gap_also_breaks_the_contact_path():
    frame = make_gapped_air_frame()
    frame.alpha_product = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    frame.alpha_air = 1.0 - frame.alpha_product
    _, _, valid, reason = contact_interval_from_fractions(frame)
    assert not valid
    assert "spatial gap" in reason


def test_pure_air_between_does_not_count_across_a_gap():
    from spherical_blast_pinn.contact_air.one_d import pure_air_run_between

    frame = make_gapped_air_frame()
    ok, count = pure_air_run_between(
        frame, low=0.05, high=0.4, pure_air_threshold=0.95, minimum=5,
    )
    assert not ok
    assert count == 4  # the larger side only; the two sides are never joined
    ok, count = pure_air_run_between(
        frame, low=0.05, high=0.4, pure_air_threshold=0.95, minimum=4,
    )
    assert ok and count == 4


def test_overlapping_cells_are_rejected():
    from spherical_blast_pinn.contact_air.one_d import OneDDataError

    frame = make_gapped_air_frame()
    frame.r_left[3] = frame.r_right[2] - 0.01
    with pytest.raises(OneDDataError, match="overlap"):
        frame.require_monotone()


def test_repeated_peak_positions_are_a_plateau_not_inward_motion():
    from spherical_blast_pinn.contact_air.one_d import detect_shock_trajectory

    radius = [0.10, 0.15, 0.19, 0.21, 0.25, 0.30]
    frames = [
        _nonuniform_air_frame(
            [8.0e6, 8.0e6, 8.0e6, 1.0e5, 1.0e5, 1.0e5], radius=radius
        )
        for _ in range(3)
    ]
    for index, frame in enumerate(frames):
        frame.time_raw_s = (index + 1) * 1.0e-5
    radii, valid, detections = detect_shock_trajectory(
        frames, contact_low=np.array([0.09, 0.09, 0.09]),
        contact_up=np.array([0.09, 0.09, 0.09]),
        contact_valid=np.ones(3, bool), ambient_pressure=1.0e5,
        pure_air_threshold=0.95,
        config=ShockDetectionConfig(relative_prominence=2.0,
                                    min_pressure_jump_fraction=1e-3),
    )
    assert valid.all(), [d.reasons for d in detections]
    assert np.allclose(radii, radii[0])
    assert detections[1].speed_status == "plateau"
    assert detections[2].speed_status == "plateau"
    assert detections[1].outward_speed == 0.0
    assert "moved inward" not in " ".join(detections[1].reasons)


def test_selected_prominence_belongs_to_the_selected_run():
    frame = _frame(alpha=np.zeros(8))
    frame.radius = np.array([0.10, 0.12, 0.17, 0.20, 0.30, 0.33, 0.37, 0.40])
    frame.r_left = np.array([0.09, 0.11, 0.145, 0.185, 0.28, 0.315, 0.35, 0.385])
    frame.r_right = np.array([0.11, 0.145, 0.185, 0.22, 0.315, 0.35, 0.385, 0.415])
    frame.pressure = np.array(
        [8.0e6, 8.0e6, 1.0e5, 1.0e5, 1.0e5, 1.0e5, 1.0e5, 1.0e5]
    )
    detection = detect_frame_shock(
        frame, contact_up=0.05, ambient_pressure=1.0e5,
        pure_air_threshold=0.95,
        config=ShockDetectionConfig(
            relative_prominence=2.0, min_cells_from_contact=0,
            min_pressure_jump_fraction=1e-3,
        ),
    )
    assert detection.valid, detection.reasons
    assert detection.run_bounds == (0, 3)
    lo, hi = detection.run_bounds
    r = frame.radius[lo:hi + 1]
    p = frame.pressure[lo:hi + 1]
    g = -np.gradient(p, r)
    expected = max(2.0 * float(np.median(np.abs(g))), 0.0)
    assert detection.prominence == pytest.approx(expected, rel=1e-12)


def test_event_alignment_uses_the_first_identifiable_frame_and_keeps_spacing():
    times = np.array([1.0e-5, 1.1e-5, 1.2e-5, 1.3e-5, 1.4e-5])
    frames = [_frame(alpha=np.zeros(5), time=float(t)) for t in times]
    for frame in frames:
        frame.radius = np.array([0.12, 0.17, 0.22, 0.27, 0.32])
    shock = np.array([0.20, 0.21, 0.25, 0.30, 0.35])
    contact_up = np.array([0.19, 0.20, 0.10, 0.10, 0.10])
    contact_valid = np.ones(5, bool)
    shock_valid = np.ones(5, bool)
    alignment = align_event(
        frames, contact_up=contact_up, contact_valid=contact_valid,
        shock_radius=shock, shock_valid=shock_valid,
        t_separation=0.9e-5, t0=2.0e-5, pure_air_threshold=0.95,
    )
    assert alignment.event_frame == 2
    assert alignment.event_time_raw_s == pytest.approx(1.2e-5)
    assert alignment.offset_s == pytest.approx(2.0e-5 - 1.2e-5)
    model_times = times + alignment.offset_s
    assert np.allclose(np.diff(model_times), np.diff(times))
    assert alignment.preceding_frame_confirmed is True


def test_event_alignment_never_falls_back_to_a_pre_separation_frame():
    times = np.array([1.0e-5, 2.0e-5])
    frames = [_frame(alpha=np.zeros(5), time=float(t)) for t in times]
    for frame in frames:
        frame.radius = np.array([0.12, 0.17, 0.22, 0.27, 0.32])
    # Both frames are before the theoretical separation time.  They do have a
    # contact, an identified shock and a pure-air interval, so the old fallback
    # would have selected frame 0.  The "not earlier than t_sep" rule must make
    # this an explicit error instead.
    with pytest.raises(OneDDataError, match="at or after the theoretical t_sep"):
        align_event(
            frames,
            contact_up=np.array([0.10, 0.10]),
            contact_valid=np.ones(2, bool),
            shock_radius=np.array([0.20, 0.25]),
            shock_valid=np.ones(2, bool),
            t_separation=2.5e-5, t0=3.0e-5, pure_air_threshold=0.95,
        )


def test_event_offset_is_recorded_and_does_not_require_equal_theoretical_time():
    precursor = default_precursor(0.05)
    source = synthetic_radial_case(
        case_id="shift", charge_radius_m=0.05, precursor=precursor,
        n_frames=11, n_radial=1600,
    )
    delta = 1.0e-6
    for frame in source.frames:
        frame.time_raw_s += delta
    case = build_case(
        source, precursor=precursor, role="train", tolerances=DEFAULT_TOLERANCES,
        contact_config=DEFAULT_CONTACT, shock_config=DEFAULT_SHOCK,
    )
    alignment = case.event_alignment
    assert alignment["offset_s"] == pytest.approx(
        precursor.t0 - alignment["event_time_raw_s"]
    )
    assert alignment["offset_s"] != 0.0
    event = int(alignment["event_frame"])
    assert case.contact.times[event] == pytest.approx(precursor.t0)
    assert case.raw_times_s[event] == pytest.approx(alignment["event_time_raw_s"])
    assert case.raw_times_s[0] == pytest.approx(precursor.t0 + delta)


def test_finite_mach_rh_state_satisfies_mass_momentum_and_energy_jumps():
    gamma = 1.4
    rho1 = 1.225
    p1 = 101320.0
    c_inf = (gamma * p1 / rho1) ** 0.5
    shock_speed = torch.tensor([2.3 * c_inf], dtype=torch.float64)
    state = finite_mach_rh_state(
        shock_speed, gamma=gamma, ambient_density=rho1, ambient_pressure=p1,
    )
    mach = float(state["mach"])
    rho2 = float(state["density"])
    u2 = float(state["velocity"])
    p2 = float(state["pressure"])
    ds = float(shock_speed)
    # State 1 is ambient at rest; state 2 is the post-shock state.
    assert rho1 * ds == pytest.approx(rho2 * (ds - u2), rel=1e-12)
    assert p1 + rho1 * ds ** 2 == pytest.approx(
        p2 + rho2 * (u2 - ds) ** 2, rel=1e-12
    )
    h1 = gamma / (gamma - 1.0) * p1 / rho1
    h2 = gamma / (gamma - 1.0) * p2 / rho2
    assert h1 + 0.5 * ds ** 2 == pytest.approx(
        h2 + 0.5 * (u2 - ds) ** 2, rel=1e-12
    )
    assert mach > 1.0


def test_subsonic_shock_speed_is_rejected():
    with pytest.raises(Exception, match="M_s > 1"):
        finite_mach_rh_state(
            torch.tensor([1.0], dtype=torch.float64), gamma=1.4,
            ambient_density=1.225, ambient_pressure=101320.0,
        )


def test_build_case_reports_separation_and_contact_labels_from_fractions():
    precursor = default_precursor(0.05)
    source = synthetic_radial_case(
        case_id="sep", charge_radius_m=0.05, precursor=precursor,
        n_frames=9, n_radial=1600,
    )
    case = build_case(
        source, precursor=precursor, role="train", tolerances=DEFAULT_TOLERANCES,
        contact_config=ContactLabelConfig(), shock_config=DEFAULT_SHOCK,
    )
    assert case.contact.n_valid > 0
    assert case.shock.valid.sum() >= 2
    assert case.observations.size > 0
    assert case.separation["status_counts"]["separated"] > 0
    assert case.source["pressure_convention"].startswith("absolute pressure")
    assert case.source["synthetic"] is True
