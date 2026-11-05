"""Small deterministic fixtures shared by tests and the smoke command.

These values are engineering defaults for the ContactNet/AirNet *fixture*.
They are not a claim that the TNT material card or the Yue-thesis parameters
have been validated.  Real runs must supply their own material/deck values.
"""
from __future__ import annotations

from typing import Any

from .data import CaseData, synthetic_case
from .precursor import PrecursorConfig, PrecursorResult, compute_precursor

DEFAULT_PRECURSOR: dict[str, Any] = {
    "epsilon_s": 1.0e-6,
    "shock_model": "strong",
    "jwl": {
        "A": 3.7120e11,
        "B": 3.7471e09,
        "C": None,
        "R1": 4.15,
        "R2": 0.95,
        "omega": 0.30,
        "rho_tnt": 1630.0,
    },
    "detonation": {
        "velocity": 6930.0,
        "cj_pressure": 2.1e10,
        "cj_density": 2200.0,
    },
    "air": {"density": 1.225, "pressure": 101320.0, "gamma": 1.4},
}

DEFAULT_TOLERANCES: dict[str, float] = {
    "ambient_pressure_pa": 101320.0,
    "ambient_density_kg_m3": 1.225,
    "pressure_pa": 0.10132,
    "density_kg_m3": 1.0e-9,
    "velocity_m_s": 1.0e-6,
    "min_ambient_cells": 2,
}


def default_precursor(charge_radius_m: float = 0.05) -> PrecursorResult:
    payload = dict(DEFAULT_PRECURSOR)
    payload["charge_radius_m"] = float(charge_radius_m)
    return compute_precursor(PrecursorConfig.from_mapping(payload))


def synthetic_case_for(
    charge_radius_m: float = 0.05, *, case_id: str | None = None,
    role: str = "train", n_frames: int = 21, n_radial: int = 200,
) -> CaseData:
    precursor = default_precursor(charge_radius_m)
    return synthetic_case(
        case_id=case_id or f"synthetic_{charge_radius_m:.3f}",
        charge_radius_m=charge_radius_m, precursor=precursor, role=role,
        n_frames=n_frames, n_radial=n_radial, tolerances=DEFAULT_TOLERANCES,
    )


__all__ = [
    "DEFAULT_PRECURSOR",
    "DEFAULT_TOLERANCES",
    "default_precursor",
    "synthetic_case_for",
]
