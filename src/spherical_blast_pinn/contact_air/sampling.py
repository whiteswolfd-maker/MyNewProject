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

    @property
    def n_pde(self) -> int:
        return int(self.t_pde.numel())


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

    def uniform_t(count: int) -> Any:
        if count <= 0:
            return torch.zeros(0, dtype=torch.float64)
        return t0 + (t_end - t0) * torch.rand(
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

    t_pde = uniform_t(int(n_pde))
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
    contact_indices = case.contact.indices()
    if contact_indices.size:
        contact_times = case.contact.times[contact_indices]
        in_window = (
            (contact_times >= t0)
            & (contact_times <= t_end)
            & np.asarray([case.shock.has_slope(float(t)) for t in contact_times])
        )
        contact_indices = contact_indices[in_window]
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
    )


def split_counts(total: int, n_cases: int) -> list[int]:
    """Equal split with the remainder assigned to the first cases."""
    if n_cases <= 0:
        raise SamplingError("at least one case is required")
    base, extra = divmod(int(total), int(n_cases))
    return [base + (1 if index < extra else 0) for index in range(n_cases)]


__all__ = ["CaseBatch", "SamplingError", "sample_case_batch", "split_counts"]
