"""Right-boundary and contact-interval geometry checks."""
from __future__ import annotations

import numpy as np
import pytest

from spherical_blast_pinn.contact_air.geometry import (
    ContactIntervalTable,
    GeometryError,
    RightBoundaryTrajectory,
    build_initial_geometry,
)


def _boundary() -> RightBoundaryTrajectory:
    times = np.linspace(1.0e-5, 2.0e-5, 11)
    radii = 0.15 + 1000.0 * (times - 1.0e-5) ** 2
    return RightBoundaryTrajectory.from_arrays(
        case_id="b", times=times, radii=radii
    )


def test_pchip_position_and_derivative_are_both_available_inside_a_segment():
    boundary = _boundary()
    value = boundary.position(1.5e-5)
    slope = boundary.derivative(1.5e-5)
    assert 0.15 < value < 0.16
    assert slope > 0.0


def test_no_extrapolation_and_no_gap_filling():
    boundary = _boundary()
    with pytest.raises(GeometryError, match="extrapolation"):
        boundary.position(1.0e-6)
    with pytest.raises(GeometryError, match="dR_b/dt"):
        boundary.derivative(1.0e-6)


def test_an_isolated_point_has_a_position_but_no_slope():
    boundary = RightBoundaryTrajectory.from_arrays(
        case_id="one", times=[1.0e-5, 2.0e-5], radii=[0.15, 0.16],
        valid=[True, False],
    )
    assert boundary.position(1.0e-5) == pytest.approx(0.15)
    with pytest.raises(GeometryError, match="isolated"):
        boundary.derivative(1.0e-5)
    with pytest.raises(GeometryError, match="isolated"):
        boundary.require_covering(1.0e-5)


def test_a_gap_splits_segments_instead_of_bridging_it():
    times = np.linspace(1.0e-5, 3.0e-5, 21)
    radii = 0.15 + 100.0 * (times - 1.0e-5)
    valid = np.ones(times.size, dtype=bool)
    valid[9:12] = False
    boundary = RightBoundaryTrajectory.from_arrays(
        case_id="gap", times=times, radii=radii, valid=valid
    )
    assert len(boundary.segments) == 2
    assert boundary.position(1.5e-5) == pytest.approx(0.15 + 100.0 * 0.5e-5)
    with pytest.raises(GeometryError):
        boundary.position(2.1e-5)


def test_initial_geometry_requires_rc_lt_rs_lt_rb():
    boundary = _boundary()
    with pytest.raises(GeometryError, match="R_c < R_s < R_b"):
        build_initial_geometry(
            t0=1.5e-5, Rc0=0.12, Rs0=0.16, R_sep=0.1,
            u_c_sep=1000.0, D_s_sep=2000.0, boundary=boundary,
        )
    good = build_initial_geometry(
        t0=1.5e-5, Rc0=0.11, Rs0=0.12, R_sep=0.1,
        u_c_sep=1000.0, D_s_sep=2000.0, boundary=boundary,
    )
    assert 0.0 < good.zeta_s0 < 1.0


def test_t0_can_be_an_exact_frame_or_between_continuous_frames_but_not_gap():
    boundary = _boundary()
    build_initial_geometry(
        t0=1.0e-5, Rc0=0.11, Rs0=0.12, R_sep=0.1,
        u_c_sep=1000.0, D_s_sep=2000.0, boundary=boundary,
    )
    build_initial_geometry(
        t0=1.5e-5, Rc0=0.11, Rs0=0.12, R_sep=0.1,
        u_c_sep=1000.0, D_s_sep=2000.0, boundary=boundary,
    )
    with pytest.raises(GeometryError, match="outside every continuous"):
        build_initial_geometry(
            t0=2.5e-5, Rc0=0.11, Rs0=0.12, R_sep=0.1,
            u_c_sep=1000.0, D_s_sep=2000.0, boundary=boundary,
        )


def test_contact_interval_is_ordered_and_brackets_raw():
    table = ContactIntervalTable(
        case_id="c", times=np.array([1.0e-5, 2.0e-5]),
        low=np.array([0.10, 0.11]), up=np.array([0.12, 0.13]),
        raw=np.array([0.11, 0.12]), valid=np.array([True, False]),
    )
    assert table.n_valid == 1
    with pytest.raises(GeometryError):
        ContactIntervalTable(
            case_id="bad", times=np.array([1.0e-5]),
            low=np.array([0.12]), up=np.array([0.11]),
            raw=np.array([0.115]), valid=np.array([True]),
        )
    with pytest.raises(GeometryError):
        ContactIntervalTable(
            case_id="bad2", times=np.array([1.0e-5]),
            low=np.array([0.10]), up=np.array([0.12]),
            raw=np.array([0.13]), valid=np.array([True]),
        )
