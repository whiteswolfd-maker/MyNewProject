"""Shared ContactNet and AirNet.

``ContactNet`` maps physical ``(t, R0)`` to ``R_c``.  ``R_c`` is parameterised
as ``R_b(t) * sigmoid(raw)`` so it stays inside the measured right boundary;
``dR_c/dt`` is formed from the full product rule, including the measured
``dR_b/dt`` and the autograd derivative of ``raw``.  ``AirNet`` maps
``(zeta, t, R0)`` to ``(rho, u, p)``; ``e = p/((gamma_a-1) rho)`` is derived.

The two networks share parameters across all training radii.  No time or radius
normalisation is applied to the network inputs: ``t`` is in seconds, ``R0`` and
``zeta`` as defined by the scheme.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


class ModelError(RuntimeError):
    """A network input or physical closure is invalid."""


@dataclass(frozen=True)
class JWLClosure:
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    R1: torch.Tensor
    R2: torch.Tensor
    omega: torch.Tensor
    V_sep: torch.Tensor
    R_sep: torch.Tensor

    def pressure(self, V: torch.Tensor) -> torch.Tensor:
        return (
            self.A * torch.exp(-self.R1 * V)
            + self.B * torch.exp(-self.R2 * V)
            + self.C / torch.clamp(V, min=1.0e-12) ** (1.0 + self.omega)
        )


@dataclass(frozen=True)
class Scales:
    rho: torch.Tensor
    u: torch.Tensor
    p: torch.Tensor
    e: torch.Tensor
    gamma: torch.Tensor


@dataclass(frozen=True)
class ContactFields:
    radius: torch.Tensor
    speed: torch.Tensor
    sigmoid: torch.Tensor
    raw: torch.Tensor
    pressure: torch.Tensor
    relative_volume: torch.Tensor


@dataclass(frozen=True)
class AirFields:
    density: torch.Tensor
    velocity: torch.Tensor
    pressure: torch.Tensor
    specific_energy: torch.Tensor


def _mlp(input_dim: int, width: int, depth: int, *, dtype, device) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for _ in range(int(depth)):
        linear = nn.Linear(last, int(width), dtype=dtype, device=device)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.Tanh())
        last = int(width)
    final = nn.Linear(last, 1, dtype=dtype, device=device)
    nn.init.xavier_uniform_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class ContactNet(nn.Module):
    """``(t, R0) -> R_c`` with ``R_c = R_b sigmoid(raw)``."""

    def __init__(self, *, width: int = 50, depth: int = 5, dtype=torch.float64,
                 device="cpu") -> None:
        super().__init__()
        self.net = _mlp(2, width, depth, dtype=dtype, device=device)
        self.dtype = dtype
        self.device = device

    def forward(
        self,
        t: torch.Tensor,
        r0: torch.Tensor,
        rb: torch.Tensor,
        rb_dot: torch.Tensor,
        closure: JWLClosure,
        *,
        create_graph: bool = True,
    ) -> ContactFields:
        if t.shape != r0.shape or t.shape != rb.shape or t.shape != rb_dot.shape:
            raise ModelError("t, R0, R_b and dR_b/dt must have the same shape")
        t_leaf = (t if t.requires_grad else t.detach()).reshape(-1, 1).requires_grad_(True)
        r0_in = r0.reshape(-1, 1)
        rb_in = rb.reshape(-1, 1)
        rb_dot_in = rb_dot.reshape(-1, 1)
        raw = self.net(torch.cat([t_leaf, r0_in], dim=-1))
        raw_t = torch.autograd.grad(
            raw.sum(), t_leaf, create_graph=create_graph, retain_graph=create_graph
        )[0]
        s = torch.sigmoid(raw)
        radius = rb_in * s
        speed = rb_dot_in * s + rb_in * s * (1.0 - s) * raw_t
        V = closure.V_sep * (radius / closure.R_sep) ** 3
        pressure = closure.pressure(V)
        if not bool(torch.isfinite(radius).all() and torch.isfinite(speed).all()):
            raise ModelError("contact radius or speed is not finite")
        return ContactFields(
            radius=radius, speed=speed, sigmoid=s, raw=raw,
            pressure=pressure, relative_volume=V,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": "ContactNet",
            "inputs": ["t_s", "R0_m"],
            "output": "R_c = R_b(t) * sigmoid(raw)",
            "width": self.net[0].out_features,
            "depth": sum(isinstance(m, nn.Linear) for m in self.net) - 1,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


class AirNet(nn.Module):
    """``(zeta, t, R0) -> (rho, u, p)`` with positive density/pressure."""

    def __init__(self, *, width: int = 32, depth: int = 10, dtype=torch.float64,
                 device="cpu") -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = 3
        for _ in range(int(depth)):
            linear = nn.Linear(last, int(width), dtype=dtype, device=device)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.Tanh())
            last = int(width)
        self.trunk = nn.Sequential(*layers)
        # Three outputs share the trunk; each head is 1-wide.
        self.head_rho = nn.Linear(width, 1, dtype=dtype, device=device)
        self.head_u = nn.Linear(width, 1, dtype=dtype, device=device)
        self.head_p = nn.Linear(width, 1, dtype=dtype, device=device)
        for head in (self.head_rho, self.head_u, self.head_p):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        self.dtype = dtype
        self.device = device

    def forward(
        self, zeta: torch.Tensor, t: torch.Tensor, r0: torch.Tensor, scales: Scales
    ) -> AirFields:
        if zeta.shape != t.shape or zeta.shape != r0.shape:
            raise ModelError("zeta, t and R0 must have the same shape")
        hidden = self.trunk(torch.cat([zeta, t, r0], dim=-1))
        rho_raw = self.head_rho(hidden)
        u_raw = self.head_u(hidden)
        p_raw = self.head_p(hidden)
        density = scales.rho * F.softplus(rho_raw)
        pressure = scales.p * F.softplus(p_raw)
        velocity = scales.u * u_raw
        energy = pressure / (torch.clamp(scales.gamma - 1.0, min=1.0e-12) * torch.clamp(density, min=1.0e-30))
        if not bool(
            torch.isfinite(density).all()
            and torch.isfinite(velocity).all()
            and torch.isfinite(pressure).all()
        ):
            raise ModelError("air fields are not finite")
        return AirFields(
            density=density, velocity=velocity, pressure=pressure,
            specific_energy=energy,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": "AirNet",
            "inputs": ["zeta", "t_s", "R0_m"],
            "outputs": ["rho", "u", "p"],
            "width": self.trunk[0].out_features,
            "depth": sum(isinstance(m, nn.Linear) for m in self.trunk),
            "parameters": sum(p.numel() for p in self.parameters()),
        }


def closure_from_precursor(precursor: Mapping[str, Any], *, like: torch.Tensor) -> JWLClosure:
    jwl = precursor["jwl"]
    def scalar(value: float) -> torch.Tensor:
        return torch.as_tensor(float(value), dtype=like.dtype, device=like.device)
    return JWLClosure(
        A=scalar(jwl["A"]), B=scalar(jwl["B"]), C=scalar(jwl["C"]),
        R1=scalar(jwl["R1"]), R2=scalar(jwl["R2"]), omega=scalar(jwl["omega"]),
        V_sep=scalar(precursor["V_sep"]), R_sep=scalar(precursor["R_sep"]),
    )


def scales_from_precursor(precursor: Mapping[str, Any], *, like: torch.Tensor) -> Scales:
    def scalar(value: float) -> torch.Tensor:
        return torch.as_tensor(float(value), dtype=like.dtype, device=like.device)
    return Scales(
        rho=scalar(precursor["rho_a_sep"]), u=scalar(precursor["u_a_sep"]),
        p=scalar(precursor["p_a_sep"]), e=scalar(precursor["e_a_sep"]),
        gamma=scalar(precursor["gamma_air"]),
    )


def air_fields_with_derivatives(
    net: AirNet,
    zeta: torch.Tensor,
    t: torch.Tensor,
    r0: torch.Tensor,
    scales: Scales,
    *,
    create_graph: bool,
) -> tuple[AirFields, dict[str, torch.Tensor]]:
    """Return fields and their first derivatives at fixed ``zeta`` and ``R0``."""
    zeta_leaf = zeta.detach().reshape(-1, 1).requires_grad_(True)
    t_leaf = t.detach().reshape(-1, 1).requires_grad_(True)
    fields = net(zeta_leaf, t_leaf, r0.reshape(-1, 1), scales)
    derivatives: dict[str, torch.Tensor] = {}
    for name, value in (("rho", fields.density), ("u", fields.velocity), ("p", fields.pressure)):
        derivatives[f"{name}_t"] = torch.autograd.grad(
            value.sum(), t_leaf, create_graph=create_graph, retain_graph=True
        )[0]
        derivatives[f"{name}_z"] = torch.autograd.grad(
            value.sum(), zeta_leaf, create_graph=create_graph, retain_graph=True
        )[0]
    return fields, derivatives


__all__ = [
    "AirFields",
    "AirNet",
    "ContactFields",
    "ContactNet",
    "JWLClosure",
    "ModelError",
    "Scales",
    "air_fields_with_derivatives",
    "closure_from_precursor",
    "scales_from_precursor",
]
