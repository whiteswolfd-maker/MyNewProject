"""Sampling for one alternating ContactNet/AirNet loop.

Air data points are drawn **frame first, then point inside the frame**.  The
right-boundary sample is restricted to times where the identified shock is
compressive in the ambient air (``D_s > c_inf``); no sample silently uses a
subsonic or extrapolated shock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .data import CaseData


class SamplingError(RuntimeError):
    """The requested sample count or case interval is invalid."""


@dataclass
class CaseBatch:
    case_id: str
    t_pde: Any
    zeta_pde: Any
    t_ic: Any
    zeta_ic: Any
    t_right: Any
    t_data: Any
    radius_data: Any
    rho_data: Any
    u_data: Any
    p_data: Any
    valid_rho: Any
    valid_u: Any
    valid_p: Any
    t_contact: Any
    low_contact: Any
    up_contact: Any
    contact_weights: Any = None
    data_weights: Any = None

    @property
    def n_pde(self) -> int:
        return int(self.t_pde.numel())


def _cell_weights(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voronoi half-spacing weights for sorted unique times."""
    unique, inverse, counts = np.unique(theta, return_inverse=True, return_counts=True)
    cell = np.zeros(unique.size, dtype=np.float64)
    if unique.size == 1:
        # A single frame has no time extent; any positive weight is equivalent
        # after the per-case normalisation.
        cell[0] = 1.0
    else:
        cell[0] = 0.5 * (unique[1] - unique[0])
        cell[-1] = 0.5 * (unique[-1] - unique[-2])
        cell[1:-1] = 0.5 * (unique[2:] - unique[:-2])
    if not np.all(np.isfinite(cell)) or np.any(cell <= 0.0):
        raise SamplingError("contact-time cells must be finite and positive")
    return cell, inverse, counts


def phase_time_weights(case: CaseData) -> np.ndarray:
    """Similarity-time cell represented by each contact row.

    The weight is defined on the **raw output cadence**: half the theta
    distance to the previous and next raw output frame, with one-sided
    half-distance at the ends.  A training/validation frame subset keeps its
    original raw frame times (including frames whose contact row is invalid),
    so a valid label does not absorb the whole time cell of an invalid
    neighbour.  If the raw cadence is unavailable, the contact times are used
    as a fallback.  Exact repeated times share one unique-time cell equally.
    """
    times = np.asarray(getattr(case.contact, "times", []), dtype=np.float64)
    if times.size == 0:
        return np.zeros(0, dtype=np.float64)
    if not np.all(np.isfinite(times)):
        raise SamplingError("contact times must be finite")
    theta = (times - float(case.t0)) / float(case.time_scale)
    raw_times = getattr(case, "raw_times_s", None)
    offset = (getattr(case, "event_alignment", {}) or {}).get("offset_s")
    if raw_times is not None and np.asarray(raw_times).size and offset is not None:
        raw = np.asarray(raw_times, dtype=np.float64)
        if not np.all(np.isfinite(raw)) or not np.all(np.diff(raw) > 0.0):
            raise SamplingError("raw output times must be finite and strictly increasing")
        raw_theta = (raw + float(offset) - float(case.t0)) / float(case.time_scale)
        cell, _, _ = _cell_weights(raw_theta)
        positions = np.searchsorted(raw_theta, theta, side="left")
        if np.any(positions >= raw_theta.size) or not np.allclose(
            raw_theta[positions], theta, rtol=1.0e-12, atol=1.0e-15
        ):
            raise SamplingError(
                "contact time is not in the raw output frame table"
            )
        contact_counts = np.bincount(positions, minlength=raw_theta.size)
        weights = cell[positions] / contact_counts[positions].astype(np.float64)
    else:
        cell, inverse, counts = _cell_weights(theta)
        weights = cell[inverse] / counts[inverse].astype(np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise SamplingError("contact-time weights must be finite and positive")
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise SamplingError("contact-time weights must have a positive finite sum")
    return weights


def air_observation_weights(case: CaseData, times) -> np.ndarray:
    """Per-point air weights: split each frame's theta cell over its draws.

    Air points are drawn frame first, then inside the frame.  A point is matched
    to its frame by its model time (``case.contact.times``), not by a stored
    frame id: a validation subset renumbers its contact rows while the
    observation store keeps the original frame ids.  The frame's cell weight is
    divided by the number of points drawn from that frame in this batch, so
    every sampled frame contributes its theta cell once.  Frames with no points
    in the batch contribute nothing; this is the usual Monte-Carlo estimate over
    the uniformly drawn frames.
    """
    values = np.asarray(times, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    contact_times = np.asarray(case.contact.times, dtype=np.float64)
    cells = phase_time_weights(case)
    # Exact repeated contact times share one unique-time cell.  Aggregate the
    # per-row cells before matching observations; otherwise every observation
    # at that time would receive only the first duplicate row's share.
    unique_times, inverse = np.unique(contact_times, return_inverse=True)
    unique_cells = np.zeros(unique_times.size, dtype=np.float64)
    np.add.at(unique_cells, inverse, cells)
    positions = np.searchsorted(unique_times, values, side="left")
    if np.any(positions >= unique_times.size) or not np.allclose(
        unique_times[positions], values, rtol=1.0e-12, atol=1.0e-15
    ):
        raise SamplingError(
            "air observation time is not in the contact/frame time table"
        )
    counts = np.bincount(positions, minlength=unique_times.size)
    denominator = counts[positions].astype(np.float64)
    return unique_cells[positions] / denominator


def sample_case_batch(
    case: CaseData,
    *,
    n_pde: int,
    n_ic: int,
    n_right: int,
    n_data: int,
    n_contact: int,
    generator=None,
    zeta_floor: float = 1.0e-4,
    pde_time_range: tuple[float, float] | None = None,
    contact_time_range: tuple[float, float] | None = None,
) -> CaseBatch:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    if case.role == "test":
        raise SamplingError(
            f"case {case.case_id!r} is a test case and must not be sampled for training"
        )
    t0, t_end = float(case.t0), float(case.t_end)
    case.shock.require_interval(t0, t_end)
    if t_end <= t0:
        raise SamplingError(f"case {case.case_id!r}: t_end must be after t0")

    def _window(value, name: str) -> tuple[float, float]:
        if value is None:
            return t0, t_end
        lo, hi = (float(value[0]), float(value[1]))
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise SamplingError(f"{name} must be finite")
        if lo < t0 - 1.0e-15 or hi > t_end + 1.0e-15 or hi <= lo:
            raise SamplingError(
                f"{name} must satisfy {t0!r} <= lo < hi <= {t_end!r}; got {value!r}"
            )
        return max(lo, t0), min(hi, t_end)

    pde_lo, pde_hi = _window(pde_time_range, "pde_time_range")
    contact_lo, contact_hi = _window(contact_time_range, "contact_time_range")

    def uniform_t(count: int, lo: float = t0, hi: float = t_end) -> Any:
        if count <= 0:
            return torch.zeros(0, dtype=torch.float64)
        return lo + (hi - lo) * torch.rand(
            count, dtype=torch.float64, generator=generator
        )

    def uniform_zeta(count: int) -> Any:
        if count <= 0:
            return torch.zeros(0, dtype=torch.float64)
        if not (0.0 < zeta_floor < 0.5):
            raise SamplingError("zeta_floor must lie in (0, 0.5)")
        lo_eff = float(zeta_floor)
        hi_eff = 1.0 - float(zeta_floor)
        return lo_eff + (hi_eff - lo_eff) * torch.rand(
            count, dtype=torch.float64, generator=generator
        )

    t_pde = uniform_t(int(n_pde), pde_lo, pde_hi)
    zeta_pde = uniform_zeta(int(n_pde))
    t_ic = torch.full((int(n_ic),), t0, dtype=torch.float64)
    zeta_ic = uniform_zeta(int(n_ic))

    # Right boundary: only compression-shock times.  Redraw a bounded number of
    # times rather than extrapolating or accepting D_s <= c_inf.
    c_inf = math.sqrt(
        float(case.gamma_air) * float(case.ambient_pressure)
        / float(case.ambient_density)
    )
    right_parts: list[np.ndarray] = []
    collected = 0
    for _ in range(8):
        if collected >= int(n_right) or int(n_right) <= 0:
            break
        candidate = uniform_t(max(int(n_right) - collected, 1)).detach().numpy()
        speed = np.asarray(case.shock.derivative(candidate), dtype=np.float64)
        keep = candidate[speed > c_inf]
        if keep.size:
            right_parts.append(keep)
            collected += int(keep.size)
    if int(n_right) > 0 and not right_parts:
        raise SamplingError(
            f"case {case.case_id!r}: no sampled time has D_s > c_inf inside the "
            "valid shock segment; a finite-Mach shock boundary cannot be enforced"
        )
    t_right = (
        torch.as_tensor(np.concatenate(right_parts)[: int(n_right)], dtype=torch.float64)
        if right_parts else torch.zeros(0, dtype=torch.float64)
    )

    data = case.observations.sample(
        int(n_data), generator, t_min=t0, t_max=t_end
    )
    data_weights = torch.as_tensor(
        air_observation_weights(case, data["t"]),
        dtype=torch.float64,
    )
    contact_indices = case.contact.indices()
    if contact_indices.size:
        contact_times = case.contact.times[contact_indices]
        in_window = (
            (contact_times >= contact_lo)
            & (contact_times <= contact_hi)
            & np.asarray([case.shock.has_slope(float(t)) for t in contact_times])
        )
        contact_indices = contact_indices[in_window]
    contact_weights = torch.zeros(0, dtype=torch.float64)
    if contact_indices.size and int(n_contact) > 0:
        if contact_indices.size <= int(n_contact):
            chosen = contact_indices
        else:
            order = torch.randperm(
                contact_indices.size, generator=generator
            ).cpu().numpy()
            chosen = contact_indices[order[: int(n_contact)]]
        t_contact = torch.as_tensor(case.contact.times[chosen], dtype=torch.float64)
        low_contact = torch.as_tensor(case.contact.low[chosen], dtype=torch.float64)
        up_contact = torch.as_tensor(case.contact.up[chosen], dtype=torch.float64)
        all_weights = phase_time_weights(case)
        contact_weights = torch.as_tensor(
            all_weights[chosen], dtype=torch.float64
        )
    else:
        t_contact = torch.zeros(0, dtype=torch.float64)
        low_contact = torch.zeros(0, dtype=torch.float64)
        up_contact = torch.zeros(0, dtype=torch.float64)

    return CaseBatch(
        case_id=case.case_id,
        t_pde=t_pde, zeta_pde=zeta_pde,
        t_ic=t_ic, zeta_ic=zeta_ic,
        t_right=t_right,
        t_data=data["t"], radius_data=data["radius"],
        rho_data=data["rho"], u_data=data["u"], p_data=data["p"],
        valid_rho=data["valid_rho"], valid_u=data["valid_u"], valid_p=data["valid_p"],
        t_contact=t_contact, low_contact=low_contact, up_contact=up_contact,
        contact_weights=contact_weights,
        data_weights=data_weights,
    )


def split_counts(total: int, n_cases: int) -> list[int]:
    """Equal split with the remainder assigned to the first cases."""
    if n_cases <= 0:
        raise SamplingError("at least one case is required")
    base, extra = divmod(int(total), int(n_cases))
    return [base + (1 if index < extra else 0) for index in range(n_cases)]


__all__ = [
    "CaseBatch",
    "SamplingError",
    "air_observation_weights",
    "phase_time_weights",
    "sample_case_batch",
    "split_counts",
]
