"""The two losses and their real valid-sample counts.

``L_a`` updates AirNet; ``L_c`` updates ContactNet.  Both contain the same
shared mapped PDE residual and the same interface relation, but each stage
keeps only its target network's parameters trainable.  Freezing a network's
parameters never means detaching its inputs: the contact stage evaluates
AirNet functionally (frozen weights, live ``zeta``/``t``) so the PDE and the
data-position mapping can still carry gradients to ``ContactNet``.

The air data coordinate is recomputed here as

``zeta = (r - R_c(pred)) / (R_s - R_c(pred))``

using the predicted contact radius and the identified shock.  The contact
interval midpoint is never substituted for ``R_c`` and out-of-domain points
are counted, not clamped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .data import CaseData
from .mapping import MappedGeometry, Residuals, assemble_residuals
from .models import AirFields, AirNet, ContactFields, Scales
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

    Selection happens **after** flattening and **before** subtraction, so an
    ``[N,1]`` prediction and an ``[N]`` target cannot broadcast into ``[N,N]``.
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
    difference = pred_flat[mask_flat] - target_flat[mask_flat]
    if difference.numel() and not bool(torch.isfinite(difference).all()):
        bad = int(torch.count_nonzero(~torch.isfinite(difference)))
        raise LossError(
            f"{bad} valid data point(s) have a non-finite prediction/target; "
            "invalid labels must be marked by the mask, not by NaN under a true mask"
        )
    return difference, mask_flat


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
    *,
    functional: bool = False,
) -> AirFields:
    zeta = zeta.reshape(-1, 1)
    t = t.reshape(-1, 1)
    r0 = r0.reshape(-1, 1)
    if functional:
        return net.forward_functional(zeta, t, r0, scales, detach_parameters=True)
    return net(zeta, t, r0, scales)


def residual_scales(case: CaseData) -> dict[str, float]:
    """Named scales for the three PDE residuals, with their units."""
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
    mass_norm = _mse(residuals.mass, scales["mass_residual"])
    momentum_norm = _mse(residuals.momentum, scales["momentum_residual"])
    energy_norm = _mse(residuals.energy, scales["energy_residual"])
    total = mass_norm + momentum_norm + energy_norm
    return TermResult(
        total=total,
        counts={"pde_points": int(fields.density.numel())},
        parts={
            "pde_mass": float(residuals.mass.detach().pow(2).mean()) if residuals.mass.numel() else 0.0,
            "pde_momentum": float(residuals.momentum.detach().pow(2).mean()) if residuals.momentum.numel() else 0.0,
            "pde_energy": float(residuals.energy.detach().pow(2).mean()) if residuals.energy.numel() else 0.0,
            # Normalized components of the same total, for logging only.  They
            # do not change the total, the sampling, or any gradient.
            "pde_mass_normalised": float(mass_norm.detach()),
            "pde_momentum_normalised": float(momentum_norm.detach()),
            "pde_energy_normalised": float(energy_norm.detach()),
        },
    )


def air_ic_loss(
    net: AirNet,
    batch: CaseBatch,
    case: CaseData,
    scales: Scales,
    *,
    functional: bool = False,
) -> TermResult:
    """Uniform separated-air layer on ``0 < zeta < 1`` at ``t=t0``.

    There is no ambient sub-interval: the right edge is the shock surface.
    """
    zeta = batch.zeta_ic
    if zeta.numel() == 0:
        return TermResult(total=_zero(batch.t_ic), counts={"ic": 0})
    t = torch.full_like(zeta, float(case.t0))
    r0 = torch.full_like(zeta, float(case.charge_radius_m))
    fields = _air_fields_at(net, zeta, t, r0, scales, functional=functional)
    term = (
        _mse(fields.density - case.rho_scale, case.rho_scale)
        + _mse(fields.velocity - case.u_scale, case.u_scale)
        + _mse(fields.pressure - case.p_scale, case.p_scale)
    )
    return TermResult(
        total=term, counts={"ic": int(zeta.numel())},
        parts={"ic": float(term.detach())},
    )


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


def finite_mach_rh_state(
    shock_speed: torch.Tensor,
    *,
    gamma: float,
    ambient_density: float,
    ambient_pressure: float,
) -> dict[str, torch.Tensor]:
    """Post-shock state of a finite-Mach shock into the ambient air.

    ``M_s = D_s/c_inf`` with ``c_inf = sqrt(gamma p_inf / rho_inf)``.  A sample
    with ``M_s <= 1`` is not a compression shock and is an error, not a state
    to be clamped.
    """
    c_inf = math.sqrt(float(gamma) * float(ambient_pressure) / float(ambient_density))
    mach = shock_speed / c_inf
    if not bool(torch.isfinite(mach).all()):
        raise LossError("shock Mach number is not finite")
    if not bool((mach > 1.0).all()):
        bad = mach.detach()[(mach <= 1.0)][:3]
        raise LossError(
            "the shock boundary sample has D_s <= c_inf at M_s="
            f"{bad.tolist()}; a compression shock requires M_s > 1"
        )
    g = float(gamma)
    rho_rh = float(ambient_density) * (g + 1.0) * mach ** 2 / (
        (g - 1.0) * mach ** 2 + 2.0
    )
    u_rh = 2.0 * c_inf / (g + 1.0) * (mach ** 2 - 1.0) / mach
    p_rh = float(ambient_pressure) * (
        2.0 * g * mach ** 2 - (g - 1.0)
    ) / (g + 1.0)
    return {"mach": mach, "density": rho_rh, "velocity": u_rh, "pressure": p_rh}


def right_loss(
    *,
    net: AirNet,
    batch: CaseBatch,
    case: CaseData,
    scales: Scales,
    functional: bool = False,
) -> TermResult:
    """Finite-Mach RH post-shock state at the inner side of ``zeta=1``."""
    t = batch.t_right
    if t.numel() == 0:
        return TermResult(total=_zero(batch.t_ic), counts={"right": 0})
    _, shock_speed = case.shock_tensors(t.reshape(-1))
    speed = shock_speed.reshape(-1, 1)
    state = finite_mach_rh_state(
        speed, gamma=case.gamma_air,
        ambient_density=case.ambient_density,
        ambient_pressure=case.ambient_pressure,
    )
    zeta = torch.ones_like(t)
    r0 = torch.full_like(t, float(case.charge_radius_m))
    fields = _air_fields_at(net, zeta, t, r0, scales, functional=functional)
    term = (
        _mse(fields.density - state["density"], case.rho_scale)
        + _mse(fields.velocity - state["velocity"], case.u_scale)
        + _mse(fields.pressure - state["pressure"], case.p_scale)
    )
    return TermResult(
        total=term, counts={"right": int(t.numel())},
        parts={
            "right": float(term.detach()),
            "right_mach_min": float(state["mach"].detach().min()),
            "right_mach_max": float(state["mach"].detach().max()),
        },
    )


def data_coordinates(
    *, radius: torch.Tensor, t: torch.Tensor, contact: ContactFields,
    case: CaseData,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Recompute ``zeta`` from the predicted contact and the shock.

    ``contact.radius`` may be live (contact stage) or detached (air stage).
    Points with ``L <= 0`` or ``zeta`` outside ``[0,1]`` are reported and
    excluded; they are never clamped into the domain.
    """
    radius = radius.reshape(-1, 1)
    t_flat = t.reshape(-1)
    rc = contact.radius.reshape(-1, 1)
    if rc.numel() != radius.numel():
        raise LossError(
            f"contact prediction has {rc.numel()} radii but the data batch has "
            f"{radius.numel()} points; the prediction must be evaluated at the "
            "observation times"
        )
    shock_radius, _ = case.shock_tensors(t_flat)
    rs = shock_radius.reshape(-1, 1)
    length = rs - rc
    positive = torch.isfinite(length) & (length > 0.0)
    safe_length = torch.where(positive, length, torch.ones_like(length))
    zeta = (radius - rc) / safe_length
    inside = positive & (zeta >= 0.0) & (zeta <= 1.0)
    zeta = torch.where(inside, zeta, torch.zeros_like(zeta))
    outside_count = int((~inside).detach().sum())
    return zeta, inside, outside_count


def air_data_loss(
    net: AirNet,
    batch: CaseBatch,
    case: CaseData,
    scales: Scales,
    contact: ContactFields,
    *,
    functional: bool = False,
) -> TermResult:
    """Per-field masked data loss with the position mapping from ``R_c``/``R_s``."""
    if batch.t_data.numel() == 0:
        return TermResult(total=_zero(batch.t_ic), counts={
            "data_rho": 0, "data_u": 0, "data_p": 0,
            "data_domain_valid": 0, "data_outside_domain": 0,
        })
    zeta, inside, outside_count = data_coordinates(
        radius=batch.radius_data, t=batch.t_data, contact=contact, case=case,
    )
    inside = inside.reshape(-1)
    in_domain = int(inside.detach().sum())
    if in_domain == 0:
        raise LossError(
            "all sampled air data points fall outside the current (R_c, R_s) "
            "domain; training must not report a normal data term after silently "
            "dropping the whole batch"
        )
    r0 = torch.full_like(batch.t_data, float(case.charge_radius_m))
    fields = _air_fields_at(net, zeta, batch.t_data, r0, scales, functional=functional)
    total = _zero(fields.density)
    counts: dict[str, int] = {}
    parts: dict[str, float] = {}
    for name, pred, target, mask, scale in (
        ("rho", fields.density, batch.rho_data, batch.valid_rho, case.rho_scale),
        ("u", fields.velocity, batch.u_data, batch.valid_u, case.u_scale),
        ("p", fields.pressure, batch.p_data, batch.valid_p, case.p_scale),
    ):
        term, count = _masked_mse(pred, target, mask & inside, scale)
        total = total + term
        counts[f"data_{name}"] = count
        parts[f"data_{name}"] = float(term.detach())
    counts["data_domain_valid"] = in_domain
    counts["data_outside_domain"] = outside_count
    return TermResult(total=total, counts=counts, parts=parts)


def contact_ic_loss(contact: ContactFields, case: CaseData) -> TermResult:
    """Initial condition constrains the contact position only."""
    target_radius = float(case.precursor["Rc0"])
    term = _mse(contact.radius - target_radius, case.length_scale)
    return TermResult(
        total=term, counts={"contact_ic": int(contact.radius.numel())},
        parts={"contact_ic": float(term.detach())},
    )


def _contact_violation_values(
    contact: ContactFields, batch: CaseBatch, case: CaseData,
) -> tuple[torch.Tensor, int]:
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
    lambda_pde: float, lambda_data: float, contact_data_weight: float = 1.0,
) -> TermResult:
    total = (
        float(lambda_pde) * pde.total
        + float(lambda_data) * float(contact_data_weight) * data.total
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
    "data_coordinates",
    "finite_mach_rh_state",
    "interface_loss",
    "masked_rmse",
    "pde_loss",
    "residual_scales",
    "right_loss",
    "schedule_weights",
]
