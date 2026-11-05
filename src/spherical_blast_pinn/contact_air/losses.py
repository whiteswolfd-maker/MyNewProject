"""The two losses and their real valid-sample counts.

``L_a`` updates AirNet; ``L_c`` updates ContactNet.  Both contain the same
shared PDE residual, but each stage evaluates it with its own target tensors
kept differentiable and the other side fixed.  The left boundary and the
interface compatibility are the same relation, so each stage counts it once.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .data import CaseData
from .mapping import MappedGeometry, Residuals, assemble_residuals
from .models import AirFields, AirNet, ContactFields, ContactNet, JWLClosure, Scales
from .sampling import CaseBatch


class LossError(RuntimeError):
    """A loss term is non-finite or a sample set is malformed."""


@dataclass
class TermResult:
    total: torch.Tensor
    counts: dict[str, int] = field(default_factory=dict)
    parts: dict[str, float] = field(default_factory=dict)


def _zero(like: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=like.dtype, device=like.device)


def _mse(value: torch.Tensor, scale: float) -> torch.Tensor:
    if value.numel() == 0:
        return _zero(value)
    return ((value / float(scale)) ** 2).mean()


def _flatten_checked(name: str, value: torch.Tensor, size: int) -> torch.Tensor:
    flat = value.reshape(-1)
    if flat.numel() != int(size):
        raise LossError(
            f"{name} has {flat.numel()} entries but {int(size)} were expected; "
            "shape mismatch would otherwise broadcast and compare unrelated points"
        )
    return flat


def _selected_difference(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten prediction/target/mask and select only the valid entries.

    Selection happens **after** flattening and **before** subtraction.  This is
    what prevents ``pred`` ``[N,1]`` and ``target`` ``[N]`` from broadcasting
    into an ``[N,N]`` cross-comparison, and it keeps NaN in invalid labels out
    of the arithmetic entirely.
    """
    size = pred.numel()
    if target.numel() != size or mask.numel() != size:
        raise LossError(
            f"prediction/target/mask sizes differ: {pred.numel()}, "
            f"{target.numel()}, {mask.numel()}"
        )
    pred_flat = pred.reshape(-1)
    target_flat = _flatten_checked("target", target, size)
    mask_flat = _flatten_checked("mask", mask, size).to(torch.bool)
    selected = mask_flat
    if not bool(selected.any()):
        return pred_flat[:0], target_flat[:0]
    difference = pred_flat[selected] - target_flat[selected]
    if not bool(torch.isfinite(difference).all()):
        bad = int(torch.count_nonzero(~torch.isfinite(difference)))
        raise LossError(
            f"{bad} valid data point(s) have a non-finite prediction/target; "
            "invalid labels must be marked by the mask, not by NaN under a true mask"
        )
    return difference, selected


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                scale: float) -> tuple[torch.Tensor, int]:
    difference, selected = _selected_difference(pred, target, mask)
    count = int(selected.sum())
    if count == 0:
        return _zero(pred), 0
    return ((difference / float(scale)) ** 2).mean(), count


def masked_rmse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: float
) -> tuple[float | None, int]:
    """Validation-side RMSE using exactly the training selection logic."""
    difference, selected = _selected_difference(pred, target, mask)
    count = int(selected.sum())
    if count == 0:
        return None, 0
    value = torch.sqrt(torch.mean((difference / float(scale)) ** 2))
    return float(value.detach()), count


def _air_fields_at(
    net: AirNet,
    zeta: torch.Tensor,
    t: torch.Tensor,
    r0: torch.Tensor,
    scales: Scales,
) -> AirFields:
    return net(zeta.reshape(-1, 1), t.reshape(-1, 1), r0.reshape(-1, 1), scales)


def residual_scales(case: CaseData) -> dict[str, float]:
    """Named scales for the three PDE residuals, with their units.

    ``energy_density`` is ``E = p/(gamma-1) + rho u^2/2`` [Pa], not the
    specific internal energy ``e`` [J/kg].  The energy residual is a
    unit-volume energy rate, so its scale is ``E / T_star``.
    """
    s_rho = float(case.rho_scale)
    s_u = float(case.u_scale)
    s_p = float(case.p_scale)
    s_R = float(case.length_scale)
    gamma = float(case.gamma_air)
    T_star = s_R / s_u
    s_E = s_p / max(gamma - 1.0, 1.0e-12) + 0.5 * s_rho * s_u ** 2
    return {
        "rho": s_rho,
        "u": s_u,
        "p": s_p,
        "specific_energy": float(case.e_scale),
        "energy_density": s_E,
        "length": s_R,
        "time": T_star,
        "mass_residual": s_rho / T_star,
        "momentum_residual": s_rho * s_u / T_star,
        "energy_residual": s_E / T_star,
    }


def pde_loss(
    *,
    fields: AirFields,
    derivatives: dict[str, torch.Tensor],
    geometry: MappedGeometry,
    case: CaseData,
) -> TermResult:
    gamma = torch.full_like(fields.density, float(case.gamma_air))
    residuals = assemble_residuals(fields, derivatives, geometry, gamma=gamma)
    scales = residual_scales(case)
    s_frho = scales["mass_residual"]
    s_fm = scales["momentum_residual"]
    s_fe = scales["energy_residual"]
    total = (
        _mse(residuals.mass, s_frho)
        + _mse(residuals.momentum, s_fm)
        + _mse(residuals.energy, s_fe)
    )
    return TermResult(
        total=total,
        counts={"pde_points": int(fields.density.numel())},
        parts={
            "pde_mass": float(residuals.mass.detach().pow(2).mean()) if residuals.mass.numel() else 0.0,
            "pde_momentum": float(residuals.momentum.detach().pow(2).mean()) if residuals.momentum.numel() else 0.0,
            "pde_energy": float(residuals.energy.detach().pow(2).mean()) if residuals.energy.numel() else 0.0,
        },
    )


def air_ic_loss(
    net: AirNet,
    batch: CaseBatch,
    case: CaseData,
    scales: Scales,
) -> TermResult:
    total = _zero(batch.t_ic)
    counts: dict[str, int] = {}
    parts: dict[str, float] = {}
    for label, zeta, target in (
        ("shock", batch.zeta_ic_shock, (case.precursor["rho_a_sep"], case.precursor["u_a_sep"], case.precursor["p_a_sep"])),
        ("ambient", batch.zeta_ic_ambient, (case.ambient_density, 0.0, case.ambient_pressure)),
    ):
        if zeta.numel() == 0:
            counts[f"ic_{label}"] = 0
            continue
        t = torch.full_like(zeta, float(case.t0))
        r0 = torch.full_like(zeta, float(case.charge_radius_m))
        fields = _air_fields_at(net, zeta, t, r0, scales)
        rho_target, u_target, p_target = target
        term = (
            _mse(fields.density - rho_target, case.rho_scale)
            + _mse(fields.velocity - u_target, case.u_scale)
            + _mse(fields.pressure - p_target, case.p_scale)
        )
        total = total + term
        counts[f"ic_{label}"] = int(zeta.numel())
        parts[f"ic_{label}"] = float(term.detach())
    return TermResult(total=total, counts=counts, parts=parts)


def interface_loss(
    *,
    air_at_contact: AirFields,
    contact: ContactFields,
    case: CaseData,
) -> TermResult:
    term = (
        _mse(air_at_contact.velocity - contact.speed, case.u_scale)
        + _mse(air_at_contact.pressure - contact.pressure, case.p_scale)
    )
    return TermResult(
        total=term,
        counts={"interface": int(contact.radius.numel())},
        parts={"interface": float(term.detach())},
    )


def right_loss(*, air_at_right: AirFields, case: CaseData) -> TermResult:
    term = (
        _mse(air_at_right.density - case.ambient_density, case.rho_scale)
        + _mse(air_at_right.velocity, case.u_scale)
        + _mse(air_at_right.pressure - case.ambient_pressure, case.p_scale)
    )
    return TermResult(
        total=term,
        counts={"right": int(air_at_right.density.numel())},
        parts={"right": float(term.detach())},
    )


def air_data_loss(
    net: AirNet,
    batch: CaseBatch,
    case: CaseData,
    scales: Scales,
) -> TermResult:
    if batch.t_data.numel() == 0:
        return TermResult(total=_zero(batch.t_ic), counts={
            "data_rho": 0, "data_u": 0, "data_p": 0,
        })
    fields = _air_fields_at(net, batch.zeta_data, batch.t_data, batch.t_data * 0.0 + case.charge_radius_m, scales)
    # The previous line keeps R0 explicit; using a full tensor avoids a silent
    # broadcast when the batch size is one.
    total = _zero(fields.density)
    counts: dict[str, int] = {}
    parts: dict[str, float] = {}
    for name, pred, target, mask, scale in (
        ("rho", fields.density, batch.rho_data, batch.valid_rho, case.rho_scale),
        ("u", fields.velocity, batch.u_data, batch.valid_u, case.u_scale),
        ("p", fields.pressure, batch.p_data, batch.valid_p, case.p_scale),
    ):
        term, count = _masked_mse(pred, target, mask, scale)
        total = total + term
        counts[f"data_{name}"] = count
        parts[f"data_{name}"] = float(term.detach())
    return TermResult(total=total, counts=counts, parts=parts)


def contact_ic_loss(contact: ContactFields, case: CaseData) -> TermResult:
    target_radius = float(case.precursor["R_sep"]) + float(case.u_c_sep) * float(case.epsilon)
    term = (
        _mse(contact.radius - target_radius, case.length_scale)
        + _mse(contact.speed - case.u_c_sep, case.u_scale)
    )
    return TermResult(
        total=term, counts={"contact_ic": int(contact.radius.numel())},
        parts={"contact_ic": float(term.detach())},
    )


def _contact_violation_values(
    contact: ContactFields, batch: CaseBatch, case: CaseData,
) -> tuple[torch.Tensor, int]:
    """Per-label interval violation, aligned one-to-one with ``t_contact``."""
    if batch.t_contact.numel() == 0:
        return torch.zeros(0, dtype=contact.radius.dtype, device=contact.radius.device), 0
    radius = contact.radius.reshape(-1)
    if radius.numel() != batch.t_contact.reshape(-1).numel():
        raise LossError(
            f"contact prediction has {radius.numel()} rows but the batch has "
            f"{batch.t_contact.numel()} contact labels; the prediction must be "
            "evaluated at batch.t_contact"
        )
    low = _flatten_checked("low_contact", batch.low_contact, radius.numel())
    up = _flatten_checked("up_contact", batch.up_contact, radius.numel())
    if not bool(torch.isfinite(radius).all()):
        raise LossError("contact prediction contains non-finite radii")
    if not bool((low <= up).all()):
        raise LossError("contact labels have R_c,low > R_c,up")
    below = torch.clamp(low - radius, min=0.0)
    above = torch.clamp(radius - up, min=0.0)
    return (below / float(case.length_scale)) ** 2 + (
        above / float(case.length_scale)
    ) ** 2, int(radius.numel())


def contact_data_loss(contact: ContactFields, batch: CaseBatch, case: CaseData) -> TermResult:
    values, count = _contact_violation_values(contact, batch, case)
    if count == 0:
        return TermResult(total=_zero(contact.radius), counts={"contact_data": 0})
    term = values.mean()
    return TermResult(
        total=term, counts={"contact_data": count},
        parts={"contact_data": float(term.detach())},
    )


def contact_violation_rmse(
    contact: ContactFields, batch: CaseBatch, case: CaseData
) -> tuple[float | None, int]:
    values, count = _contact_violation_values(contact, batch, case)
    if count == 0:
        return None, 0
    return float(torch.sqrt(torch.mean(values)).detach()), count


def combine_air(
    *, pde: TermResult, ic: TermResult, interface: TermResult,
    right: TermResult, data: TermResult, lambda_pde: float, lambda_data: float,
) -> TermResult:
    total = (
        float(lambda_pde) * pde.total
        + float(lambda_data) * data.total
        + 1.0 * ic.total
        + 1.0 * interface.total
        + 1.0 * right.total
    )
    counts = {**pde.counts, **ic.counts, **interface.counts, **right.counts, **data.counts}
    parts = {**pde.parts, **ic.parts, **interface.parts, **right.parts, **data.parts}
    return TermResult(total=total, counts=counts, parts=parts)


def combine_contact(
    *, pde: TermResult, ic: TermResult, interface: TermResult, data: TermResult,
    lambda_pde: float, lambda_data: float,
) -> TermResult:
    total = (
        float(lambda_pde) * pde.total
        + float(lambda_data) * data.total
        + 1.0 * ic.total
        + 1.0 * interface.total
    )
    counts = {**pde.counts, **ic.counts, **interface.counts, **data.counts}
    parts = {**pde.parts, **ic.parts, **interface.parts, **data.parts}
    return TermResult(total=total, counts=counts, parts=parts)


def schedule_weights(loop: int, n_schedule: int) -> tuple[float, float]:
    """Return ``(lambda_pde, lambda_data)`` for the 0.9->0.3 data schedule."""
    if n_schedule <= 0:
        raise LossError("n_schedule must be positive")
    progress = min(max(float(loop) / float(n_schedule), 0.0), 1.0)
    lambda_data = 0.3 + 0.3 * (1.0 + math.cos(math.pi * progress))
    return 1.0 - lambda_data, lambda_data


__all__ = [
    "LossError",
    "TermResult",
    "air_data_loss",
    "air_ic_loss",
    "combine_air",
    "combine_contact",
    "contact_data_loss",
    "contact_ic_loss",
    "contact_violation_rmse",
    "interface_loss",
    "masked_rmse",
    "pde_loss",
    "residual_scales",
    "right_loss",
    "schedule_weights",
]
