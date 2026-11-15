"""Shared ContactNet and AirNet.

``ContactNet`` maps physical ``(t, R0)`` to ``R_c``.  ``R_c`` is parameterised
as ``R_s(t) * sigmoid(raw)`` so it stays inside the identified external shock;
``dR_c/dt`` is formed from the full product rule, including the measured
``dR_s/dt = D_s`` and the autograd derivative of ``raw``.  ``AirNet`` maps
``(zeta, t, R0)`` to ``(rho, u, p)``; ``e = p/((gamma_a-1) rho)`` is derived.

The two networks share parameters across all training radii.  They support two
explicit time-input conventions:

``legacy``
    ``x_t = t / time_reference_s``.  This is the historical convention; it does
    not remove the per-case origin and does not align similarity stages.

``similarity``
    ``T_star = case.time_scale = R_sep / u_a_sep`` and
    ``theta = (t - case.t0) / T_star``, then
    ``x_t = theta / similarity_input_divisor``.  The divisor is only a fixed
    numerical input scale (64.0 in this experiment); it is not a physical
    constant and is never selected from validation error.

The conversion is shared by the normal and functional AirNet paths.  ``t``
remains physical seconds in every derivative: the scaling is inside the
autograd graph, so ``d x_t / d t = 1 / (divisor * T_star)`` and the PDE/contact
speed chain rule is unchanged.  ``R0`` and ``zeta`` are never rescaled.

``AirNet`` also has an optional ``r0_input_gate``.  Its default ``none`` is the
historical behavior.  ``similarity_time`` replaces ``R0`` in the network input
by ``R0 * (1 - exp(-(t-t0)/T_star))``: at t0 the R0 input is zero for every
case, while the output value is still a soft IC target, not a hard value.

``AirNet.forward_functional(..., detach_parameters=True)`` evaluates the same
network with frozen parameter values.  The contact stage uses it so gradients
can flow through ``zeta``/``t`` into ``ContactNet`` without populating any
``AirNet`` parameter gradient; this is parameter isolation, not ``no_grad`` on
the physical residual.
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


TIME_INPUT_MODES = ("legacy", "similarity")
R0_INPUT_GATE_MODES = ("none", "similarity_time")
SIMILARITY_INPUT_DIVISOR_DEFAULT = 64.0


@dataclass(frozen=True)
class TimeInput:
    """Per-case origin and physical time scale for the similarity input.

    ``origin`` is the model time ``t0`` in seconds and ``scale`` is
    ``R_sep/u_a_sep`` in seconds.  Both are physical quantities; the network
    divisor is a separate numerical input scale stored on the network.
    """

    origin: float
    scale: float

    def __post_init__(self) -> None:
        origin = float(self.origin)
        scale = float(self.scale)
        if not math.isfinite(origin):
            raise ModelError("time-input origin must be finite")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ModelError("time-input scale must be finite and positive")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "scale", scale)


def _validate_time_input_mode(mode: str) -> str:
    value = str(mode)
    if value not in TIME_INPUT_MODES:
        raise ModelError(
            f"time input mode must be one of {TIME_INPUT_MODES}, got {value!r}"
        )
    return value


def _validate_divisor(value: float) -> float:
    divisor = float(value)
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise ModelError("similarity input divisor must be finite and positive")
    return divisor


def _validate_r0_input_gate(mode: str) -> str:
    value = str(mode)
    if value not in R0_INPUT_GATE_MODES:
        raise ModelError(
            f"R0 input gate must be one of {R0_INPUT_GATE_MODES}, got {value!r}"
        )
    return value


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
    """``(t, R0) -> R_c`` with ``R_c = R_s sigmoid(raw)``."""

    def __init__(self, *, width: int = 50, depth: int = 5, dtype=torch.float64,
                 device="cpu", time_reference_s: float = 1.0,
                 time_input_mode: str = "legacy",
                 similarity_input_divisor: float = SIMILARITY_INPUT_DIVISOR_DEFAULT) -> None:
        super().__init__()
        if not math.isfinite(float(time_reference_s)) or float(time_reference_s) <= 0.0:
            raise ModelError("time_reference_s must be finite and positive")
        self.net = _mlp(2, width, depth, dtype=dtype, device=device)
        self.dtype = dtype
        self.device = device
        self.time_reference_s = float(time_reference_s)
        self.time_input_mode = _validate_time_input_mode(time_input_mode)
        self.similarity_input_divisor = _validate_divisor(similarity_input_divisor)

    def scaled_time(
        self, t: torch.Tensor, time_input: TimeInput | None = None
    ) -> torch.Tensor:
        """Return the network time argument without breaking autograd.

        In similarity mode the origin and scale are required per case.  The
        physical ``t`` is never detached, so any later ``grad(..., t)`` remains
        a derivative with respect to seconds.
        """
        if self.time_input_mode == "legacy":
            return t / self.time_reference_s
        if time_input is None:
            raise ModelError(
                "similarity time input requires TimeInput(case.t0, case.time_scale)"
            )
        return (t - time_input.origin) / (
            time_input.scale * self.similarity_input_divisor
        )

    def _time_description(self) -> dict[str, Any]:
        if self.time_input_mode == "legacy":
            return {
                "time_input": "legacy",
                "time_origin_rule": "zero",
                "time_scale_rule": "time_reference_s",
                "similarity_input_divisor": self.similarity_input_divisor,
            }
        return {
            "time_input": "similarity",
            "time_origin_rule": "case.t0",
            "time_scale_rule": "case.time_scale=R_sep/u_a_sep",
            "similarity_input_divisor": self.similarity_input_divisor,
        }

    def forward(
        self,
        t: torch.Tensor,
        r0: torch.Tensor,
        shock_radius: torch.Tensor,
        shock_speed: torch.Tensor,
        closure: JWLClosure,
        *,
        time_input: TimeInput | None = None,
        create_graph: bool = True,
    ) -> ContactFields:
        if not (
            t.shape == r0.shape == shock_radius.shape == shock_speed.shape
        ):
            raise ModelError(
                "t, R0, R_s and dR_s/dt must have the same shape"
            )
        t_leaf = (t if t.requires_grad else t.detach()).reshape(-1, 1).requires_grad_(True)
        r0_in = r0.reshape(-1, 1)
        rs_in = shock_radius.reshape(-1, 1)
        ds_in = shock_speed.reshape(-1, 1)
        # The MLP sees a dimensionless time; the conversion stays in the
        # autograd graph, so raw_t below is still d(raw)/dt in 1/s and speed
        # stays m/s.
        t_scaled = self.scaled_time(t_leaf, time_input)
        raw = self.net(torch.cat([t_scaled, r0_in], dim=-1))
        raw_t = torch.autograd.grad(
            raw.sum(), t_leaf, create_graph=create_graph, retain_graph=create_graph
        )[0]
        s = torch.sigmoid(raw)
        radius = rs_in * s
        # Full product rule: R_s(t) sigmoid(raw(t)).
        speed = ds_in * s + rs_in * s * (1.0 - s) * raw_t
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
            "geometry": "R_c = R_s(t) * sigmoid(raw)",
            "width": self.net[0].out_features,
            "depth": sum(isinstance(m, nn.Linear) for m in self.net) - 1,
            "parameters": sum(p.numel() for p in self.parameters()),
            "time_reference_s": self.time_reference_s,
            **self._time_description(),
        }


class AirNet(nn.Module):
    """``(zeta, t, R0) -> (rho, u, p)`` with positive density/pressure."""

    def __init__(self, *, width: int = 32, depth: int = 10, dtype=torch.float64,
                 device="cpu", time_reference_s: float = 1.0,
                 time_input_mode: str = "legacy",
                 similarity_input_divisor: float = SIMILARITY_INPUT_DIVISOR_DEFAULT,
                 r0_input_gate: str = "none") -> None:
        super().__init__()
        if not math.isfinite(float(time_reference_s)) or float(time_reference_s) <= 0.0:
            raise ModelError("time_reference_s must be finite and positive")
        self.time_reference_s = float(time_reference_s)
        self.time_input_mode = _validate_time_input_mode(time_input_mode)
        self.similarity_input_divisor = _validate_divisor(similarity_input_divisor)
        self.r0_input_gate = _validate_r0_input_gate(r0_input_gate)
        if self.r0_input_gate != "none" and self.time_input_mode != "similarity":
            raise ModelError(
                "r0_input_gate='similarity_time' requires time_input='similarity'"
            )
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
        self.head_rho = nn.Linear(width, 1, dtype=dtype, device=device)
        self.head_u = nn.Linear(width, 1, dtype=dtype, device=device)
        self.head_p = nn.Linear(width, 1, dtype=dtype, device=device)
        for head in (self.head_rho, self.head_u, self.head_p):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        self.dtype = dtype
        self.device = device

    def scaled_time(
        self, t: torch.Tensor, time_input: TimeInput | None = None
    ) -> torch.Tensor:
        if self.time_input_mode == "legacy":
            return t / self.time_reference_s
        if time_input is None:
            raise ModelError(
                "similarity time input requires TimeInput(case.t0, case.time_scale)"
            )
        return (t - time_input.origin) / (
            time_input.scale * self.similarity_input_divisor
        )

    def _time_description(self) -> dict[str, Any]:
        gate_rule = (
            "none"
            if self.r0_input_gate == "none"
            else "R0*(1-exp(-(t-case.t0)/case.time_scale))"
        )
        if self.time_input_mode == "legacy":
            return {
                "time_input": "legacy",
                "time_origin_rule": "zero",
                "time_scale_rule": "time_reference_s",
                "similarity_input_divisor": self.similarity_input_divisor,
                "r0_input_gate": self.r0_input_gate,
                "r0_gate_rule": gate_rule,
            }
        return {
            "time_input": "similarity",
            "time_origin_rule": "case.t0",
            "time_scale_rule": "case.time_scale=R_sep/u_a_sep",
            "similarity_input_divisor": self.similarity_input_divisor,
            "r0_input_gate": self.r0_input_gate,
            "r0_gate_rule": gate_rule,
        }

    def _network_inputs(
        self,
        zeta: torch.Tensor,
        t: torch.Tensor,
        r0: torch.Tensor,
        time_input: TimeInput | None,
    ) -> torch.Tensor:
        """Build the shared AirNet feature vector, ordinary and functional.

        ``r0_input_gate='none'`` is the historical behavior.  The optional
        ``similarity_time`` gate is

            theta=(t-case.t0)/case.time_scale
            gate=1-exp(-theta)
            R0_input=R0*gate

        so at t0 all cases feed R0_input=0 while the output value remains
        soft-constrained only by ``air_ic_loss``.  For t>t0 the physical R0
        input returns smoothly.
        """
        if self.r0_input_gate == "none":
            r0_input = r0
        else:
            if time_input is None:
                raise ModelError(
                    "r0_input_gate='similarity_time' requires "
                    "TimeInput(case.t0, case.time_scale)"
                )
            theta = (t - time_input.origin) / time_input.scale
            gate = -torch.expm1(-theta)
            r0_input = r0 * gate
        return torch.cat(
            [zeta, self.scaled_time(t, time_input), r0_input], dim=-1
        )

    def forward(
        self, zeta: torch.Tensor, t: torch.Tensor, r0: torch.Tensor, scales: Scales,
        *, time_input: TimeInput | None = None,
    ) -> AirFields:
        if zeta.shape != t.shape or zeta.shape != r0.shape:
            raise ModelError("zeta, t and R0 must have the same shape")
        # Dimensionless time; derivatives are still taken with respect to the
        # physical t passed in, because the conversion is part of the graph.
        hidden = self.trunk(self._network_inputs(zeta, t, r0, time_input))
        return self._heads(hidden, scales)

    def _heads(self, hidden: torch.Tensor, scales: Scales) -> AirFields:
        rho_raw = self.head_rho(hidden)
        u_raw = self.head_u(hidden)
        p_raw = self.head_p(hidden)
        density = scales.rho * F.softplus(rho_raw)
        pressure = scales.p * F.softplus(p_raw)
        velocity = scales.u * u_raw
        energy = pressure / (
            torch.clamp(scales.gamma - 1.0, min=1.0e-12)
            * torch.clamp(density, min=1.0e-30)
        )
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

    def forward_functional(
        self,
        zeta: torch.Tensor,
        t: torch.Tensor,
        r0: torch.Tensor,
        scales: Scales,
        *,
        detach_parameters: bool = True,
        time_input: TimeInput | None = None,
    ) -> AirFields:
        """Evaluate with frozen parameters but live inputs.

        Every ``Linear`` uses detached weight/bias when
        ``detach_parameters=True``.  Gradients with respect to ``zeta``/``t``
        remain, so a ContactNet-dependent ``zeta`` still receives its gradient.
        The time conversion is the same as :meth:`forward`.
        """
        if zeta.shape != t.shape or zeta.shape != r0.shape:
            raise ModelError("zeta, t and R0 must have the same shape")
        hidden = self._network_inputs(zeta, t, r0, time_input)
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                weight = module.weight.detach() if detach_parameters else module.weight
                bias = module.bias.detach() if detach_parameters else module.bias
                hidden = F.linear(hidden, weight, bias)
            elif isinstance(module, nn.Tanh):
                hidden = F.tanh(hidden)
            else:  # pragma: no cover - architecture is fixed
                raise ModelError(f"unsupported trunk module {type(module).__name__}")
        if detach_parameters:
            out_rho = F.linear(hidden, self.head_rho.weight.detach(), self.head_rho.bias.detach())
            out_u = F.linear(hidden, self.head_u.weight.detach(), self.head_u.bias.detach())
            out_p = F.linear(hidden, self.head_p.weight.detach(), self.head_p.bias.detach())
        else:
            out_rho = self.head_rho(hidden)
            out_u = self.head_u(hidden)
            out_p = self.head_p(hidden)
        rho_raw, u_raw, p_raw = out_rho, out_u, out_p
        density = scales.rho * F.softplus(rho_raw)
        pressure = scales.p * F.softplus(p_raw)
        velocity = scales.u * u_raw
        energy = pressure / (
            torch.clamp(scales.gamma - 1.0, min=1.0e-12)
            * torch.clamp(density, min=1.0e-30)
        )
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
            "time_reference_s": self.time_reference_s,
            **self._time_description(),
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
    functional: bool = False,
    time_input: TimeInput | None = None,
) -> tuple[AirFields, dict[str, torch.Tensor]]:
    """Return fields and their first derivatives at fixed ``zeta`` and ``R0``.

    ``functional=True`` freezes the parameter values while keeping input
    derivatives live (contact stage).
    """
    zeta_leaf = zeta.detach().reshape(-1, 1).requires_grad_(True)
    t_leaf = t.detach().reshape(-1, 1).requires_grad_(True)
    if functional:
        fields = net.forward_functional(
            zeta_leaf, t_leaf, r0.reshape(-1, 1), scales,
            detach_parameters=True, time_input=time_input,
        )
    else:
        fields = net(
            zeta_leaf, t_leaf, r0.reshape(-1, 1), scales,
            time_input=time_input,
        )
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
    "R0_INPUT_GATE_MODES",
    "Scales",
    "SIMILARITY_INPUT_DIVISOR_DEFAULT",
    "TIME_INPUT_MODES",
    "TimeInput",
    "air_fields_with_derivatives",
    "closure_from_precursor",
    "scales_from_precursor",
]
