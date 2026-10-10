"""Equations of state for detonation products and air.

Two EOS are used, and only two:

* :class:`JWL` — the general (caloric) Jones–Wilkins–Lee equation of state for
  the TNT products.  The network predicts the *specific* internal energy
  ``e`` in J/kg and the pressure is computed from ``p(rho, e)``.  The
  isentropic/principal-isentrope form ``p(rho)`` is deliberately **not**
  implemented here: freezing ``e = e_iso(rho)`` is one of the legacy schemes
  this package must not reproduce.
* :func:`ideal_gas_pressure` — ``p = (gamma - 1) rho e`` for air.

Everything is SI.  Torch and NumPy implementations are provided side by side so
that preprocessing (NumPy) and training (Torch) evaluate *the same* formula.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - exercised only in torch-free preprocessing runs
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..constants import DEFAULT_AIR_GAMMA, TINY_DENSITY

# Guard against overflow of exp(-R V) for very large expansion volumes.
_MAX_VOLUME_RATIO = 1.0e6


class EOSError(Exception):
    """Raised when an EOS evaluation is inadmissible or misconfigured."""


# ==========================================================================
# JWL
# ==========================================================================


@dataclass(frozen=True)
class JWLParameters:
    """SI parameters of the general JWL EOS.

    Parameters
    ----------
    A, B:
        Pressure coefficients, Pa.
    R1, R2:
        Dimensionless exponential coefficients.
    omega:
        Dimensionless Gruneisen coefficient.
    rho0:
        Reference density of the *unreacted* explosive, kg/m^3.  ``V = rho0/rho``
        uses this density, exactly as LS-DYNA's ``*EOS_JWL`` does.
    energy_shift:
        Constant added to the supplied specific internal energy before the EOS
        is evaluated, J/kg.  This exists because an LS-DYNA energy-reference
        mismatch is otherwise almost impossible to diagnose after training.
    """

    A: float
    B: float
    R1: float
    R2: float
    omega: float
    rho0: float
    energy_shift: float = 0.0

    # -- construction -----------------------------------------------------
    def validate(self) -> None:
        numeric = (self.A, self.B, self.R1, self.R2, self.omega, self.rho0, self.energy_shift)
        if not all(math.isfinite(v) for v in numeric):
            raise EOSError("JWL parameters must all be finite")
        if self.rho0 <= 0.0:
            raise EOSError(f"JWL rho0 must be positive, got {self.rho0}")
        for name in ("R1", "R2"):
            if getattr(self, name) <= 0.0:
                raise EOSError(f"JWL {name} must be positive, got {getattr(self, name)}")
        if self.omega < 0.0:
            raise EOSError(f"JWL omega must be non-negative, got {self.omega}")
        if self.A < 0.0 or self.B < 0.0:
            raise EOSError("JWL A and B must be non-negative")

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "JWLParameters":
        required = ("A", "B", "R1", "R2", "omega", "rho0")
        missing = [k for k in required if k not in cfg]
        if missing:
            raise EOSError(f"JWL configuration is missing {missing}")
        params = cls(
            A=float(cfg["A"]),
            B=float(cfg["B"]),
            R1=float(cfg["R1"]),
            R2=float(cfg["R2"]),
            omega=float(cfg["omega"]),
            rho0=float(cfg["rho0"]),
            energy_shift=float(cfg.get("energy_shift", 0.0) or 0.0),
        )
        params.validate()
        return params

    # -- derived quantities ----------------------------------------------
    @property
    def rho0_safe(self) -> float:
        return max(self.rho0, TINY_DENSITY)

    @staticmethod
    def specific_energy_from_volume_energy(volume_energy: float, rho0: float) -> float:
        """Convert LS-DYNA's per-initial-volume JWL energy ``E0`` to J/kg.

        ``*EOS_JWL`` stores ``E0`` as internal energy per unit *initial*
        volume, so the specific internal energy is ``e_p0 = E0 / rho0``.
        """
        if rho0 <= 0.0:
            raise EOSError("rho0 must be positive when converting E0 to specific energy")
        return float(volume_energy) / float(rho0)

    def initial_specific_energy(self, E0_volume_energy: float) -> float:
        return self.specific_energy_from_volume_energy(E0_volume_energy, self.rho0)

    # -- Chapman-Jouguet diagnostics (checks only, never initial data) ----
    def product_hugoniot_pressure(self, relative_volume: float) -> float:
        """Pressure on the product Hugoniot at ``V = v / v0``.

        The Hugoniot through the unreacted state ``(V=1, p=0, e=0)`` satisfies
        ``e = 0.5 p (v0 - v)``.  Substituting into the caloric JWL closure and
        solving for ``p`` gives

        ``p_H(V) = p_cold(V) / (1 - 0.5 omega (1 - V) / V)``.

        The denominator vanishes for a sufficiently small ``V``; ``nan`` is
        returned there rather than a spurious value.
        """
        v = float(relative_volume)
        if not (0.0 < v <= 1.0):
            return float("nan")
        denominator = 1.0 - 0.5 * self.omega * (1.0 - v) / v
        if denominator <= 1.0e-12:
            return float("nan")
        p_cold = _cold_pressure_scalar(v, self)
        return p_cold / denominator

    def cj_state(self, *, coarse_samples: int = 2000, refine_levels: int = 6) -> dict[str, float]:
        """Chapman–Jouguet state of the JWL parameters alone.

        The CJ point is the minimum of the Rayleigh detonation velocity over
        the product Hugoniot,

        ``D(V) = sqrt( p_H(V) / (rho0 (1 - V)) )``,

        which is the tangency condition between the Rayleigh line and the
        Hugoniot.  Minimising is far more robust than root-finding on the
        Rayleigh/Hugoniot intersection, which has a spurious second root deep
        in the compressed branch.

        These values are **diagnostics only**: never an initial condition,
        never a loss term, never a network input.
        """
        v0 = 1.0 / self.rho0_safe
        lo, hi = 1.0e-3, 1.0 - 1.0e-9

        def detonation_velocity(relative_volume: float) -> float:
            p = self.product_hugoniot_pressure(relative_volume)
            if not math.isfinite(p) or p <= 0.0 or relative_volume >= 1.0:
                return float("inf")
            return math.sqrt(p / (self.rho0_safe * (1.0 - relative_volume)))

        best_v, best_d = None, float("inf")
        for level in range(refine_levels + 1):
            samples = coarse_samples if level == 0 else 200
            span = (hi - lo) / (samples - 1)
            for index in range(samples):
                v = lo + index * span
                d = detonation_velocity(v)
                if d < best_d:
                    best_v, best_d = v, d
            lo, hi = max(1.0e-6, best_v - span), min(1.0 - 1.0e-12, best_v + span)
        if best_v is None or not math.isfinite(best_d):
            raise EOSError("could not locate a Chapman-Jouguet state for these JWL parameters")

        v_cj = best_v
        rho_cj = self.rho0_safe / v_cj
        p_cj = self.product_hugoniot_pressure(v_cj)
        # Hugoniot energy: e = 0.5 p (v0 - v) with v = V * v0.
        e_cj = 0.5 * p_cj * v0 * (1.0 - v_cj)
        c_cj = float(jwl_sound_speed_numpy(np.array(rho_cj), np.array(e_cj), self))
        return {
            "V_CJ": v_cj,
            "rho_CJ": rho_cj,
            "p_CJ": p_cj,
            "e_CJ_specific": e_cj,
            "D_CJ": best_d,
            "c_CJ": c_cj,
            "particle_velocity_CJ": best_d - c_cj,
        }

    def compare_detonation_velocity(self, declared_velocity: float) -> dict[str, float]:
        """Compare the deck's ``D`` against the value the JWL parameters imply."""
        cj = self.cj_state()
        relative = abs(cj["D_CJ"] - declared_velocity) / max(declared_velocity, 1.0e-30)
        return {
            "declared_detonation_velocity": declared_velocity,
            "implied_detonation_velocity": cj["D_CJ"],
            "relative_deviation": relative,
            **cj,
        }

    def as_dict(self) -> dict[str, float]:
        return {
            "A": self.A,
            "B": self.B,
            "R1": self.R1,
            "R2": self.R2,
            "omega": self.omega,
            "rho0": self.rho0,
            "energy_shift": self.energy_shift,
        }


def _cj_energy(v: float, params: JWLParameters) -> float:
    """Order-of-magnitude specific energy used only by :meth:`cj_state_estimate`."""
    return 0.5 * (params.A * v * math.exp(-params.R1 * v) + params.B * v * math.exp(-params.R2 * v)) / max(
        params.rho0, TINY_DENSITY
    )


def _cold_pressure_scalar(v: float, params: JWLParameters) -> float:
    v = max(v, 1.0e-12)
    return params.A * (1.0 - params.omega / (params.R1 * v)) * math.exp(-params.R1 * v) + params.B * (
        1.0 - params.omega / (params.R2 * v)
    ) * math.exp(-params.R2 * v)


# -- torch implementation -------------------------------------------------


def jwl_volume_ratio(rho, params: JWLParameters):
    """``V = rho0 / rho`` with an overflow-safe clamp.  Torch version."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the torch EOS path")
    rho_safe = rho.clamp_min(TINY_DENSITY)
    return (params.rho0_safe / rho_safe).clamp_max(_MAX_VOLUME_RATIO)


def jwl_pressure(rho, e, params: JWLParameters):
    """General JWL pressure ``p(rho, e)`` in Pa.  Torch version."""
    v = jwl_volume_ratio(rho, params)
    e_eff = e + params.energy_shift
    exp1 = torch.exp(-params.R1 * v)
    exp2 = torch.exp(-params.R2 * v)
    p_cold = params.A * (1.0 - params.omega / (params.R1 * v)) * exp1 + params.B * (
        1.0 - params.omega / (params.R2 * v)
    ) * exp2
    return p_cold + params.omega * rho * e_eff


def jwl_pressure_derivatives(rho, e, params: JWLParameters):
    """Return ``(dp/drho|_e, dp/de|_rho)`` for the general JWL EOS.

    ``dp/de|_rho = omega * rho`` and ``dp/drho|_e`` follows from the chain rule
    through ``V = rho0/rho``.  Both are needed for the sound speed.
    """
    v = jwl_volume_ratio(rho, params)

    def d_cold_dv(coefficient: float, rate: float):
        exp_term = torch.exp(-rate * v)
        return coefficient * exp_term * (
            params.omega / (rate * v * v) - rate + params.omega / v
        )

    d_cold_dv_total = d_cold_dv(params.A, params.R1) + d_cold_dv(params.B, params.R2)
    # dV/drho = -rho0/rho^2 = -V/rho
    d_cold_drho = d_cold_dv_total * (-v / rho.clamp_min(TINY_DENSITY))
    dp_drho = d_cold_drho + params.omega * (e + params.energy_shift)
    dp_de = params.omega * rho
    return dp_drho, dp_de


def jwl_sound_speed_squared(rho, e, params: JWLParameters):
    """``c^2`` of the products **without** clamping, Pa m^3/kg.

    .. math::

        c^2 = \\left.\\frac{\\partial p}{\\partial\\rho}\\right|_e
            + \\frac{p}{\\rho^2}
              \\left.\\frac{\\partial p}{\\partial e}\\right|_\\rho

    The ``p/rho^2`` factor is the isentropic chain rule
    ``(de/drho)_s = p/rho^2``, which follows from ``de = -p d(1/rho)``.  It is
    **not** ``p/rho`` — writing ``p/rho`` gives a value that is wrong by a factor
    ``rho`` and rises to tens of kilometres per second in the compressed
    products, where the true sound speed is about 4 km/s.  The NumPy path has
    always used the correct form (there ``dp_de = omega rho``, so
    ``dp_de p/rho^2 = omega p/rho``); this function exists so the Torch path
    cannot drift away from it again.

    The raw -- possibly negative -- value is returned.  The admissibility loss
    must test the unclamped quantity: clamping a negative ``c^2`` to zero first
    and then checking ``c^2 >= 0`` reports every invalid state as valid.
    """
    p = jwl_pressure(rho, e, params)
    dp_drho, dp_de = jwl_pressure_derivatives(rho, e, params)
    rho_safe = rho.clamp_min(TINY_DENSITY)
    return dp_drho + dp_de * p / (rho_safe * rho_safe)


def jwl_sound_speed(rho, e, params: JWLParameters):
    """Thermodynamic sound speed of the products, m/s.

    ``sqrt(max(c^2, 0))``.  Use :func:`jwl_sound_speed_squared` wherever an
    invalid ``c^2`` has to be *detected* rather than silently repaired.
    """
    return torch.sqrt(jwl_sound_speed_squared(rho, e, params).clamp_min(0.0))


def jwl_sound_speed_squared_numpy(rho, e, params: JWLParameters) -> np.ndarray:
    """NumPy twin of :func:`jwl_sound_speed_squared`, unclamped."""
    rho = np.asarray(rho, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    rho_safe = np.maximum(rho, TINY_DENSITY)
    v = np.minimum(params.rho0_safe / rho_safe, _MAX_VOLUME_RATIO)

    def d_cold_dv(coefficient: float, rate: float) -> np.ndarray:
        return coefficient * np.exp(-rate * v) * (
            params.omega / (rate * v * v) - rate + params.omega / v
        )

    d_cold_drho = (d_cold_dv(params.A, params.R1) + d_cold_dv(params.B, params.R2)) * (-v / rho_safe)
    p = jwl_pressure_numpy(rho, e, params)
    return d_cold_drho + params.omega * (e + params.energy_shift) + params.omega * p / rho_safe


# -- NumPy implementation -------------------------------------------------


def jwl_pressure_numpy(rho, e, params: JWLParameters) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    rho_safe = np.maximum(rho, TINY_DENSITY)
    v = np.minimum(params.rho0_safe / rho_safe, _MAX_VOLUME_RATIO)
    e_eff = e + params.energy_shift
    return (
        params.A * (1.0 - params.omega / (params.R1 * v)) * np.exp(-params.R1 * v)
        + params.B * (1.0 - params.omega / (params.R2 * v)) * np.exp(-params.R2 * v)
        + params.omega * rho * e_eff
    )


def jwl_sound_speed_numpy(rho, e, params: JWLParameters) -> np.ndarray:
    return np.sqrt(np.maximum(jwl_sound_speed_squared_numpy(rho, e, params), 0.0))


# ==========================================================================
# Ideal gas (air)
# ==========================================================================


def ideal_gas_pressure(rho, e, gamma: float = DEFAULT_AIR_GAMMA):
    """``p = (gamma - 1) rho e``.  Torch version."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the torch EOS path")
    return (gamma - 1.0) * rho * e


def ideal_gas_specific_energy(rho, p, gamma: float = DEFAULT_AIR_GAMMA):
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the torch EOS path")
    return p / ((gamma - 1.0) * rho.clamp_min(TINY_DENSITY))


def ideal_gas_sound_speed(rho, e, gamma: float = DEFAULT_AIR_GAMMA):
    p = ideal_gas_pressure(rho, e, gamma)
    return torch.sqrt((gamma * p / rho.clamp_min(TINY_DENSITY)).clamp_min(0.0))


def ideal_gas_pressure_numpy(rho, e, gamma: float = DEFAULT_AIR_GAMMA) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    return (gamma - 1.0) * rho * e


def ideal_gas_specific_energy_numpy(rho, p, gamma: float = DEFAULT_AIR_GAMMA) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return p / ((gamma - 1.0) * np.maximum(rho, TINY_DENSITY))


def ideal_gas_sound_speed_numpy(rho, e, gamma: float = DEFAULT_AIR_GAMMA) -> np.ndarray:
    p = ideal_gas_pressure_numpy(rho, e, gamma)
    rho = np.asarray(rho, dtype=np.float64)
    return np.sqrt(np.maximum(gamma * p / np.maximum(rho, TINY_DENSITY), 0.0))


def ambient_air_state(
    rho0: float, p0: float, gamma: float = DEFAULT_AIR_GAMMA
) -> tuple[float, float, float, float]:
    """Return ``(rho, u, e, p)`` of the quiescent ambient air."""
    if rho0 <= 0.0:
        raise EOSError(f"ambient density must be positive, got {rho0}")
    if p0 <= 0.0:
        raise EOSError(f"ambient pressure must be positive, got {p0}")
    e0 = p0 / ((gamma - 1.0) * rho0)
    return rho0, 0.0, e0, p0


def entropy_jump_over_cv(
    rho_minus, p_minus, rho0: float, p0: float, gamma: float = DEFAULT_AIR_GAMMA
):
    """Ideal-gas entropy jump ``Delta s / c_v`` across the air shock.

    ``Delta s / c_v = ln(p^- / p0) - gamma ln(rho^- / rho0)``.  Non-negative
    for any admissible shock; the training loss only *penalises* violations and
    never forces a fixed positive value, so weak shocks remain weak.
    """
    if torch is not None and isinstance(rho_minus, torch.Tensor):
        return torch.log(p_minus / p0) - gamma * torch.log(rho_minus / rho0)
    return np.log(np.asarray(p_minus) / p0) - gamma * np.log(np.asarray(rho_minus) / rho0)


# ==========================================================================
# *EOS_LINEAR_POLYNOMIAL equivalence check
# ==========================================================================


def linear_polynomial_ideal_gas_coefficients(gamma: float = DEFAULT_AIR_GAMMA) -> dict[str, float]:
    """The ``*EOS_LINEAR_POLYNOMIAL`` coefficients equivalent to ideal gas air.

    LS-DYNA writes the polynomial EOS as

    ``p = C0 + C1 mu + C2 mu^2 + C3 mu^3 + C4 E + C5 E mu + C6 E mu^2``

    with ``mu = rho/rho0 - 1`` and ``E`` the internal energy per unit *initial*
    volume.  Substituting the ideal-gas relation ``p = (gamma-1) rho e`` and
    ``E = rho0 e`` gives ``p = (gamma-1)(1 + mu) E``, hence
    ``C4 = C5 = gamma - 1`` and all other coefficients zero.
    """
    return {"C0": 0.0, "C1": 0.0, "C2": 0.0, "C3": 0.0, "C4": gamma - 1.0, "C5": gamma - 1.0, "C6": 0.0}


def check_linear_polynomial_is_ideal_gas(
    coefficients: Mapping[str, float],
    gamma: float = DEFAULT_AIR_GAMMA,
    *,
    atol: float = 1.0e-8,
    rtol: float = 1.0e-6,
) -> None:
    """Raise :class:`EOSError` unless the polynomial EOS is ideal gas air.

    This is a *hard* failure by design.  If the deck's air EOS is not
    equivalent to ``gamma = 1.4`` ideal gas then the entire air model — RH
    conditions, entropy condition, sound speed — would be wrong, and training
    must not start.
    """
    expected = linear_polynomial_ideal_gas_coefficients(gamma)
    bad: list[str] = []
    for name, want in expected.items():
        got = coefficients.get(name)
        if got is None:
            bad.append(f"{name} is missing")
            continue
        got = float(got)
        if abs(got - want) > max(atol, rtol * max(abs(want), 1.0)):
            bad.append(f"{name} = {got!r}, expected {want!r}")
    if bad:
        raise EOSError(
            "the LS-DYNA *EOS_LINEAR_POLYNOMIAL air card is not equivalent to an "
            f"ideal gas with gamma = {gamma}: " + "; ".join(bad) + ". Refusing to continue."
        )


def linear_polynomial_pressure(
    rho, e, rho0: float, coefficients: Mapping[str, float]
) -> np.ndarray:
    """Evaluate ``*EOS_LINEAR_POLYNOMIAL`` in SI for independent checking."""
    rho = np.asarray(rho, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    mu = rho / rho0 - 1.0
    E = rho0 * e  # energy per unit initial volume
    c = coefficients
    return (
        float(c.get("C0", 0.0))
        + float(c.get("C1", 0.0)) * mu
        + float(c.get("C2", 0.0)) * mu**2
        + float(c.get("C3", 0.0)) * mu**3
        + float(c.get("C4", 0.0)) * E
        + float(c.get("C5", 0.0)) * E * mu
        + float(c.get("C6", 0.0)) * E * mu**2
    )


__all__ = [
    "EOSError",
    "JWLParameters",
    "ambient_air_state",
    "check_linear_polynomial_is_ideal_gas",
    "entropy_jump_over_cv",
    "ideal_gas_pressure",
    "ideal_gas_pressure_numpy",
    "ideal_gas_sound_speed",
    "ideal_gas_sound_speed_numpy",
    "ideal_gas_specific_energy",
    "ideal_gas_specific_energy_numpy",
    "jwl_pressure",
    "jwl_pressure_derivatives",
    "jwl_pressure_numpy",
    "jwl_sound_speed",
    "jwl_sound_speed_numpy",
    "jwl_sound_speed_squared",
    "jwl_sound_speed_squared_numpy",
    "jwl_volume_ratio",
    "linear_polynomial_ideal_gas_coefficients",
    "linear_polynomial_pressure",
]
