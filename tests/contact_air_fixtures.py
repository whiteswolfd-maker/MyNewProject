"""Shared one-dimensional test fixtures.

Kept in one module so no test imports another test module's private helper.
These are analytic frames only; they are not TNT data.
"""
from __future__ import annotations

import numpy as np

from spherical_blast_pinn.contact_air.one_d import RadialFrame


def make_frame(
    *, alpha, pressure=None, density=None, velocity=None,
    valid_pressure=None, case_id="c", time=1.0e-5,
) -> RadialFrame:
    alpha = np.asarray(alpha, dtype=float)
    boundaries = np.linspace(0.0, alpha.size * 0.1, alpha.size + 1)
    centers = 0.5 * (boundaries[:-1] + boundaries[1:])
    n = alpha.size
    p = np.full(n, 101320.0) if pressure is None else np.asarray(pressure, float)
    rho = np.full(n, 1.225) if density is None else np.asarray(density, float)
    u = np.zeros(n) if velocity is None else np.asarray(velocity, float)
    vp = np.isfinite(p) if valid_pressure is None else np.asarray(valid_pressure, bool)
    return RadialFrame(
        case_id=case_id, charge_radius_m=0.05, frame_index=0, time_raw_s=time,
        cell_id=np.arange(n), r_left=boundaries[:-1], r_right=boundaries[1:],
        radius=centers, alpha_product=alpha, alpha_air=1.0 - alpha,
        material=np.where(alpha >= 0.95, 0, np.where(alpha <= 0.05, 1, 2)),
        density=rho, velocity=u, pressure=p,
        valid_density=np.isfinite(rho), valid_velocity=np.isfinite(u),
        valid_pressure=vp, position_type="cell_center",
        units={"length": "m", "time": "s", "density": "kg/m^3",
               "velocity": "m/s", "pressure": "Pa"},
    )


def make_nonuniform_air_frame(pressures, *, radius, valid_pressure=None):
    radius = np.asarray(radius, dtype=float)
    frame = make_frame(alpha=np.zeros(len(radius)))
    boundaries = np.concatenate([
        radius[:1] - 0.01,
        0.5 * (radius[:-1] + radius[1:]),
        radius[-1:] + 0.01,
    ])
    frame.r_left = boundaries[:-1]
    frame.r_right = boundaries[1:]
    frame.radius = radius
    frame.pressure = np.asarray(pressures, dtype=float)
    frame.valid_pressure = (
        np.ones(len(radius), dtype=bool) if valid_pressure is None
        else np.asarray(valid_pressure, dtype=bool)
    )
    return frame


def make_gapped_air_frame(*, pressure=(8.0e6, 8.0e6, 8.0e6, 1.0e5, 1.0e5, 1.0e5, 1.0e5)):
    """The reproduced case: a 0.10 m void between index 2 and index 3."""
    left = np.array([0.09, 0.11, 0.13, 0.25, 0.27, 0.29, 0.31])
    right = left + 0.02
    frame = make_frame(alpha=np.zeros(left.size), pressure=pressure)
    frame.r_left = left
    frame.r_right = right
    frame.radius = 0.5 * (left + right)
    return frame
