"""Sampling for one alternating ContactNet/AirNet loop."""
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
    zeta_ic_shock: Any
    zeta_ic_ambient: Any
    t_right: Any
    t_data: Any
    zeta_data: Any
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


def _full(value: float, like) -> Any:
    return torch.full_like(like, float(value))


def sample_case_batch(
    case: CaseData,
    *,
    n_pde: int,
    n_ic_shock: int,
    n_ic_ambient: int,
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

    def uniform_t(count: int) -> Any:
        if count <= 0:
            return torch.zeros(0, dtype=torch.float64)
        return t0 + (t_end - t0) * torch.rand(count, dtype=torch.float64, generator=generator)

    def uniform_zeta(count: int, lo: float = zeta_floor, hi: float = None) -> Any:
        if count <= 0:
            return torch.zeros(0, dtype=torch.float64)
        hi = 1.0 - zeta_floor if hi is None else float(hi)
        lo_eff = max(float(lo), zeta_floor) if float(lo) == 0.0 else float(lo)
        hi_eff = min(float(hi), 1.0 - zeta_floor) if float(hi) == 1.0 else float(hi)
        if not (0.0 < lo_eff < hi_eff < 1.0):
            raise SamplingError(f"invalid zeta interval ({lo!r}, {hi!r})")
        return lo_eff + (hi_eff - lo_eff) * torch.rand(
            count, dtype=torch.float64, generator=generator
        )

    t_pde = uniform_t(int(n_pde))
    zeta_pde = uniform_zeta(int(n_pde))
    t_ic = torch.full((int(n_ic_shock) + int(n_ic_ambient),), t0, dtype=torch.float64)
    zeta_ic_shock = uniform_zeta(int(n_ic_shock), 0.0, case.initial.zeta_s0)
    zeta_ic_ambient = uniform_zeta(
        int(n_ic_ambient), case.initial.zeta_s0, 1.0
    )
    t_right = uniform_t(int(n_right))

    data = case.observations.sample(int(n_data), generator)
    contact_indices = case.contact.indices()
    if contact_indices.size and int(n_contact) > 0:
        if contact_indices.size <= int(n_contact):
            chosen = contact_indices
        else:
            order = torch.randperm(contact_indices.size, generator=generator).cpu().numpy()
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
        t_ic=t_ic, zeta_ic_shock=zeta_ic_shock, zeta_ic_ambient=zeta_ic_ambient,
        t_right=t_right,
        t_data=data["t"], zeta_data=data["zeta"], rho_data=data["rho"],
        u_data=data["u"], p_data=data["p"],
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
