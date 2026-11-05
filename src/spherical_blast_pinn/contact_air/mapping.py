"""Mapped spherical air equations on ``zeta=(r-R_c)/(R_b-R_c)``.

The three residuals are assembled in one place and used by both training
stages.  The air stage differentiates the network fields; the contact stage
differentiates the moving geometry while treating already-evaluated air fields
and their derivatives as fixed values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .models import AirFields


class MappingError(RuntimeError):
    """The moving geometry or a residual is invalid."""


@dataclass(frozen=True)
class MappedGeometry:
    rc: torch.Tensor
    rc_dot: torch.Tensor
    rb: torch.Tensor
    rb_dot: torch.Tensor
    zeta: torch.Tensor
    length: torch.Tensor
    length_dot: torch.Tensor
    radius: torch.Tensor
    grid_speed: torch.Tensor


@dataclass(frozen=True)
class Residuals:
    mass: torch.Tensor
    momentum: torch.Tensor
    energy: torch.Tensor


def build_geometry(
    rc: torch.Tensor,
    rc_dot: torch.Tensor,
    rb: torch.Tensor,
    rb_dot: torch.Tensor,
    zeta: torch.Tensor,
) -> MappedGeometry:
    length = rb - rc
    length_dot = rb_dot - rc_dot
    radius = rc + zeta * length
    grid_speed = rc_dot + zeta * length_dot
    if not bool(torch.isfinite(length).all() and (length > 0.0).all()):
        raise MappingError("the moving air layer must have L = R_b - R_c > 0")
    if not bool(torch.isfinite(radius).all() and (radius > 0.0).all()):
        raise MappingError("the mapped physical radius must be positive")
    return MappedGeometry(
        rc=rc, rc_dot=rc_dot, rb=rb, rb_dot=rb_dot, zeta=zeta,
        length=length, length_dot=length_dot, radius=radius, grid_speed=grid_speed,
    )


def assemble_residuals(
    fields: AirFields,
    derivatives: dict[str, torch.Tensor],
    geometry: MappedGeometry,
    *,
    gamma: torch.Tensor,
) -> Residuals:
    """Assemble the mapped mass, momentum and total-energy residuals.

    ``derivatives`` must contain ``rho_t, rho_z, u_t, u_z, p_t, p_z``.  The
    caller is responsible for making the right tensors differentiable: the air
    stage keeps field derivatives live; the contact stage keeps the geometry
    live and the field derivatives fixed.
    """
    required = ("rho_t", "rho_z", "u_t", "u_z", "p_t", "p_z")
    missing = [name for name in required if name not in derivatives]
    if missing:
        raise MappingError(f"missing air derivatives: {missing}")
    rho = fields.density
    u = fields.velocity
    p = fields.pressure
    rho_t, rho_z = derivatives["rho_t"], derivatives["rho_z"]
    u_t, u_z = derivatives["u_t"], derivatives["u_z"]
    p_t, p_z = derivatives["p_t"], derivatives["p_z"]
    L = geometry.length
    vg = geometry.grid_speed
    r = geometry.radius
    inv_L = 1.0 / L
    inv_r = 1.0 / torch.clamp(r, min=1.0e-30)

    rho_u = rho * u
    rho_u_t = rho_t * u + rho * u_t
    rho_u_z = rho_z * u + rho * u_z
    mass = rho_t + (rho_u_z - vg * rho_z) * inv_L + 2.0 * rho_u * inv_r

    momentum_flux_z = rho_z * u * u + 2.0 * rho * u * u_z + p_z
    momentum = (
        rho_u_t
        + (momentum_flux_z - vg * rho_u_z) * inv_L
        + 2.0 * rho * u * u * inv_r
    )

    gamma_minus_one = torch.clamp(gamma - 1.0, min=1.0e-12)
    total_energy = p / gamma_minus_one + 0.5 * rho * u * u
    E_t = p_t / gamma_minus_one + 0.5 * (rho_t * u * u + 2.0 * rho * u * u_t)
    E_z = p_z / gamma_minus_one + 0.5 * (rho_z * u * u + 2.0 * rho * u * u_z)
    flux = u * (gamma * p / gamma_minus_one + 0.5 * rho * u * u)
    flux_z = u_z * (gamma * p / gamma_minus_one + 0.5 * rho * u * u) + u * (
        gamma * p_z / gamma_minus_one + 0.5 * (rho_z * u * u + 2.0 * rho * u * u_z)
    )
    energy = E_t + (flux_z - vg * E_z) * inv_L + 2.0 * u * (total_energy + p) * inv_r

    for name, value in (("mass", mass), ("momentum", momentum), ("energy", energy)):
        if not bool(torch.isfinite(value).all()):
            raise MappingError(f"mapped {name} residual is not finite")
    return Residuals(mass=mass, momentum=momentum, energy=energy)


def physical_residuals(
    rho: torch.Tensor,
    u: torch.Tensor,
    p: torch.Tensor,
    rho_t: torch.Tensor,
    rho_r: torch.Tensor,
    u_t: torch.Tensor,
    u_r: torch.Tensor,
    p_t: torch.Tensor,
    p_r: torch.Tensor,
    radius: torch.Tensor,
    *,
    gamma: torch.Tensor,
) -> Residuals:
    """Euler residuals in physical ``(r,t)`` coordinates, for equivalence tests."""
    inv_r = 1.0 / torch.clamp(radius, min=1.0e-30)
    rho_u = rho * u
    rho_u_t = rho_t * u + rho * u_t
    rho_u_r = rho_r * u + rho * u_r
    mass = rho_t + rho_u_r + 2.0 * rho_u * inv_r
    momentum = rho_u_t + (rho_r * u * u + 2.0 * rho * u * u_r + p_r) + 2.0 * rho * u * u * inv_r
    g1 = torch.clamp(gamma - 1.0, min=1.0e-12)
    E = p / g1 + 0.5 * rho * u * u
    E_t = p_t / g1 + 0.5 * (rho_t * u * u + 2.0 * rho * u * u_t)
    E_r = p_r / g1 + 0.5 * (rho_r * u * u + 2.0 * rho * u * u_r)
    flux = u * (gamma * p / g1 + 0.5 * rho * u * u)
    flux_r = u_r * (gamma * p / g1 + 0.5 * rho * u * u) + u * (
        gamma * p_r / g1 + 0.5 * (rho_r * u * u + 2.0 * rho * u * u_r)
    )
    energy = E_t + flux_r + 2.0 * u * (E + p) * inv_r
    return Residuals(mass=mass, momentum=momentum, energy=energy)


__all__ = [
    "MappedGeometry",
    "MappingError",
    "Residuals",
    "assemble_residuals",
    "build_geometry",
    "physical_residuals",
]
