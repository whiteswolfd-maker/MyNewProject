"""Precursor physics for the ContactNet/AirNet scheme.

The module implements the 2026-11-04 contract:

* CJ state from the detonation velocity and the CJ density (mass jump);
* a CJ-consistent JWL ``C`` coefficient when ``C`` is not supplied;
* the isentropic release branch ``p(V)``, sound speed and
  ``U(V) = u_CJ + integral c(eta)/eta d eta``;
* matching ``U(V) = u_air(p(V))`` on the strong-shock or exact air Hugoniot;
* ``R_sep = R0 V_sep^(1/3)`` and the uniform-sphere time integral from the
  CJ radius to ``R_sep``;
* the small-positive-time initial condition ``t0 = t_sep + epsilon``.

All quantities are SI.  ``V`` is the dimensionless relative volume
``rho_TNT / rho_product``; the mass specific volume is ``1/rho`` and is never
mixed with ``V``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


class PrecursorError(RuntimeError):
    """A precursor quantity is missing, inconsistent or non-physical."""


@dataclass(frozen=True)
class JWLParameters:
    A: float
    B: float
    C: float | None
    R1: float
    R2: float
    omega: float
    rho_tnt: float

    def __post_init__(self) -> None:
        for name in ("A", "B", "R1", "R2", "omega", "rho_tnt"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise PrecursorError(f"JWL {name} must be finite and positive, got {value!r}")
        if self.C is not None and (not math.isfinite(float(self.C)) or float(self.C) <= 0.0):
            raise PrecursorError(f"JWL C must be positive when supplied, got {self.C!r}")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "JWLParameters":
        return cls(
            A=float(payload["A"]),
            B=float(payload["B"]),
            C=None if payload.get("C") is None else float(payload["C"]),
            R1=float(payload["R1"]),
            R2=float(payload["R2"]),
            omega=float(payload["omega"]),
            rho_tnt=float(payload.get("rho_tnt", payload.get("rho0"))),
        )


@dataclass(frozen=True)
class AirState:
    density: float
    pressure: float
    gamma: float

    def __post_init__(self) -> None:
        for name in ("density", "pressure", "gamma"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise PrecursorError(f"air {name} must be finite and positive, got {value!r}")


@dataclass(frozen=True)
class DetonationState:
    velocity: float
    cj_pressure: float
    cj_density: float

    def __post_init__(self) -> None:
        for name in ("velocity", "cj_pressure", "cj_density"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise PrecursorError(f"detonation {name} must be finite and positive, got {value!r}")


@dataclass(frozen=True)
class PrecursorConfig:
    charge_radius_m: float
    detonation: DetonationState
    jwl: JWLParameters
    air: AirState
    #: epsilon = R_sep / [factor * (D_s_sep - u_c_sep)].  The factor is an
    #: explicit configuration value; 200 is the current round's choice, not a
    #: validated constant.  A fixed microsecond is no longer accepted.
    epsilon_denominator_factor: float = 200.0
    shock_model: str = "strong"
    v_max: float = 100.0
    integration_limit: int = 200
    root_tolerance: float = 1.0e-12
    #: Where the JWL/detonation/air numbers came from.  The default is a
    #: configuration placeholder, not a verified material card.
    material_source: str = "unverified_config_defaults"
    material_source_verified: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.charge_radius_m)) or self.charge_radius_m <= 0.0:
            raise PrecursorError("charge_radius_m must be finite and positive")
        if (
            not math.isfinite(float(self.epsilon_denominator_factor))
            or self.epsilon_denominator_factor <= 0.0
        ):
            raise PrecursorError("epsilon_denominator_factor must be finite and positive")
        if self.shock_model not in ("strong", "exact"):
            raise PrecursorError("shock_model must be 'strong' or 'exact'")
        if self.v_max <= 1.0:
            raise PrecursorError("v_max must exceed 1")
        if not str(self.material_source).strip():
            raise PrecursorError("material_source must not be empty")
        if self.material_source_verified and (
            str(self.material_source).strip().lower()
            in ("", "unverified_config_defaults", "default", "defaults")
        ):
            raise PrecursorError(
                "material_source_verified=True needs a real material-card source, "
                "not the unverified default placeholder"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PrecursorConfig":
        if "epsilon_s" in payload:
            raise PrecursorError(
                "fixed epsilon_s is not accepted: this round uses "
                "epsilon_s = R_sep / (epsilon_denominator_factor * "
                "(D_s_sep - u_c_sep)); remove epsilon_s from the configuration"
            )
        return cls(
            charge_radius_m=float(payload["charge_radius_m"]),
            detonation=DetonationState(
                velocity=float(payload["detonation"]["velocity"]),
                cj_pressure=float(payload["detonation"]["cj_pressure"]),
                cj_density=float(payload["detonation"]["cj_density"]),
            ),
            jwl=JWLParameters.from_mapping(payload["jwl"]),
            air=AirState(
                density=float(payload["air"]["density"]),
                pressure=float(payload["air"]["pressure"]),
                gamma=float(payload["air"]["gamma"]),
            ),
            epsilon_denominator_factor=float(
                payload.get("epsilon_denominator_factor", 200.0)
            ),
            shock_model=str(payload.get("shock_model", "strong")),
            v_max=float(payload.get("v_max", 100.0)),
            integration_limit=int(payload.get("integration_limit", 200)),
            root_tolerance=float(payload.get("root_tolerance", 1.0e-12)),
            material_source=str(
                payload.get("material_source", "unverified_config_defaults")
            ),
            material_source_verified=bool(
                payload.get("material_source_verified", False)
            ),
        )


@dataclass
class PrecursorResult:
    config: PrecursorConfig
    V_cj: float
    V_sep: float
    C: float
    c_source: str
    u_cj: float
    c_cj: float
    p_cj: float
    p_sep: float
    p_c_t0: float
    initial_pressure_compatibility: dict[str, Any]
    u_c_sep: float
    u_a_sep: float
    rho_a_sep: float
    e_a_sep: float
    D_s_sep: float
    R_cj: float
    R_sep: float
    t_cj: float
    t_surface_ref: float
    t_sep: float
    epsilon_s: float
    t0: float
    Rc0: float
    Rs0: float
    shock_model: str
    checks: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    clock: dict[str, Any] = field(default_factory=dict)

    @property
    def charge_radius(self) -> float:
        return float(self.config.charge_radius_m)

    @property
    def epsilon(self) -> float:
        return float(self.epsilon_s)

    @property
    def air_density(self) -> float:
        return float(self.config.air.density)

    @property
    def air_pressure(self) -> float:
        return float(self.config.air.pressure)

    @property
    def gamma_air(self) -> float:
        return float(self.config.air.gamma)

    @property
    def rho_scale(self) -> float:
        return float(self.rho_a_sep)

    @property
    def u_scale(self) -> float:
        return float(self.u_a_sep)

    @property
    def p_scale(self) -> float:
        return float(self.p_a_sep)

    @property
    def e_scale(self) -> float:
        return float(self.e_a_sep)

    @property
    def length_scale(self) -> float:
        return float(self.R_sep)

    def as_dict(self) -> dict[str, Any]:
        return {
            "charge_radius_m": self.charge_radius,
            "epsilon_s": self.epsilon,
            "epsilon_denominator_factor": self.config.epsilon_denominator_factor,
            "epsilon_formula": (
                "R_sep / (epsilon_denominator_factor * (D_s_sep - u_c_sep))"
            ),
            "shock_model": self.shock_model,
            "air_density": self.air_density,
            "air_pressure": self.air_pressure,
            "gamma_air": self.gamma_air,
            "jwl": {
                "A": self.config.jwl.A,
                "B": self.config.jwl.B,
                "C": self.C,
                "R1": self.config.jwl.R1,
                "R2": self.config.jwl.R2,
                "omega": self.config.jwl.omega,
                "rho_tnt": self.config.jwl.rho_tnt,
            },
            "detonation": {
                "velocity": self.config.detonation.velocity,
                "cj_pressure": self.config.detonation.cj_pressure,
                "cj_density": self.config.detonation.cj_density,
            },
            "material_source": self.config.material_source,
            "material_source_verified": bool(self.config.material_source_verified),
            "V_cj": self.V_cj,
            "V_sep": self.V_sep,
            "C": self.C,
            "c_source": self.c_source,
            "u_cj": self.u_cj,
            "c_cj": self.c_cj,
            "p_cj": self.p_cj,
            "p_sep": self.p_sep,
            "p_a_sep": self.p_sep,
            "p_c_t0": self.p_c_t0,
            "initial_pressure_compatibility": dict(
                self.initial_pressure_compatibility
            ),
            "u_c_sep": self.u_c_sep,
            "u_a_sep": self.u_a_sep,
            "rho_a_sep": self.rho_a_sep,
            "e_a_sep": self.e_a_sep,
            "D_s_sep": self.D_s_sep,
            "R_cj": self.R_cj,
            "R_sep": self.R_sep,
            "t_cj": self.t_cj,
            "t_surface_ref": self.t_surface_ref,
            "t_sep": self.t_sep,
            "t0": self.t0,
            "Rc0": self.Rc0,
            "Rs0": self.Rs0,
            "checks": dict(self.checks),
            "provenance": dict(self.provenance),
            "notes": list(self.notes),
            "clock": dict(self.clock),
        }


def _p(V: float, jwl: JWLParameters, C: float) -> float:
    return (
        jwl.A * math.exp(-jwl.R1 * V)
        + jwl.B * math.exp(-jwl.R2 * V)
        + C / V ** (1.0 + jwl.omega)
    )


def _dp_dV(V: float, jwl: JWLParameters, C: float) -> float:
    return (
        -jwl.A * jwl.R1 * math.exp(-jwl.R1 * V)
        - jwl.B * jwl.R2 * math.exp(-jwl.R2 * V)
        - (1.0 + jwl.omega) * C / V ** (2.0 + jwl.omega)
    )


def _sound_speed(V: float, jwl: JWLParameters, C: float) -> float:
    value = -V * V * _dp_dV(V, jwl, C) / jwl.rho_tnt
    if not math.isfinite(value) or value <= 0.0:
        raise PrecursorError(f"JWL sound speed squared is not positive at V={V!r}")
    return math.sqrt(value)


def _release_velocity(
    V: float, jwl: JWLParameters, C: float, V_cj: float, u_cj: float, limit: int
) -> float:
    integral, _ = quad(
        lambda eta: _sound_speed(eta, jwl, C) / eta,
        V_cj,
        V,
        epsabs=1.0e-10,
        epsrel=1.0e-10,
        limit=limit,
    )
    return u_cj + integral


def air_hugoniot(
    pressure: float, air: AirState, model: str
) -> tuple[float, float, float, float]:
    """Return ``(D_s, u_2, rho_2, e_2)`` for a shock into the given air state."""
    p2 = float(pressure)
    rho1 = float(air.density)
    p1 = float(air.pressure)
    gamma = float(air.gamma)
    if not math.isfinite(p2) or p2 <= p1:
        raise PrecursorError(f"shock pressure {p2!r} is not above ambient {p1!r}")
    if model == "strong":
        D_s = math.sqrt((gamma + 1.0) * p2 / (2.0 * rho1))
        u2 = math.sqrt(2.0 * p2 / ((gamma + 1.0) * rho1))
        rho2 = rho1 * (gamma + 1.0) / (gamma - 1.0)
    elif model == "exact":
        D_s = math.sqrt(((gamma + 1.0) * p2 + (gamma - 1.0) * p1) / (2.0 * rho1))
        u2 = (p2 - p1) / (rho1 * D_s)
        rho2 = rho1 * D_s / (D_s - u2)
    else:  # pragma: no cover - validated by config
        raise PrecursorError(f"unknown shock model {model!r}")
    if not (D_s > 0.0 and u2 > 0.0 and rho2 > 0.0 and D_s > u2):
        raise PrecursorError(
            f"non-physical air shock state: D_s={D_s!r}, u2={u2!r}, rho2={rho2!r}"
        )
    e2 = p2 / ((gamma - 1.0) * rho2)
    return D_s, u2, rho2, e2


def compute_precursor(config: PrecursorConfig) -> PrecursorResult:
    """Compute the separation state and the model initial condition time."""
    jwl = config.jwl
    det = config.detonation
    air = config.air
    R0 = float(config.charge_radius_m)

    V_cj = jwl.rho_tnt / det.cj_density
    if not (0.0 < V_cj < 1.0):
        raise PrecursorError(
            f"CJ relative volume V_cj = rho_TNT/rho_CJ = {V_cj!r} is not in (0, 1); "
            "check the TNT and CJ densities"
        )
    p_cj_from_jwl_base = jwl.A * math.exp(-jwl.R1 * V_cj) + jwl.B * math.exp(
        -jwl.R2 * V_cj
    )
    if jwl.C is None:
        C = (det.cj_pressure - p_cj_from_jwl_base) * V_cj ** (1.0 + jwl.omega)
        c_source = "closed from p(V_cj)=p_CJ"
    else:
        C = float(jwl.C)
        c_source = "supplied by material card"
    if not math.isfinite(C) or C <= 0.0:
        raise PrecursorError(
            f"JWL C = {C!r} is not positive; the CJ pressure and A/B/R1/R2/omega "
            "cannot define a physical release branch"
        )
    p_cj = _p(V_cj, jwl, C)
    if abs(p_cj - det.cj_pressure) > 1.0e-6 * det.cj_pressure:
        raise PrecursorError(
            f"JWL pressure at V_cj is {p_cj!r} Pa, deck CJ pressure is "
            f"{det.cj_pressure!r} Pa"
        )

    u_cj = det.velocity * (1.0 - jwl.rho_tnt / det.cj_density)
    if not math.isfinite(u_cj) or u_cj <= 0.0:
        raise PrecursorError(f"CJ particle velocity {u_cj!r} is not positive")
    c_cj = _sound_speed(V_cj, jwl, C)

    def mismatch(V: float) -> float:
        p = _p(V, jwl, C)
        _, u_air, _, _ = air_hugoniot(p, air, config.shock_model)
        return _release_velocity(V, jwl, C, V_cj, u_cj, config.integration_limit) - u_air

    # The release branch is V > V_cj.  Scan a wide bracket; report the bracket
    # instead of pretending a root exists when it does not.
    samples = np.linspace(V_cj * (1.0 + 1.0e-9), config.v_max, 400)
    values = [mismatch(float(v)) for v in samples]
    bracket: tuple[float, float] | None = None
    for left, right, f_left, f_right in zip(samples[:-1], samples[1:], values[:-1], values[1:]):
        if f_left == 0.0:
            bracket = (float(left), float(left))
            break
        if f_left * f_right < 0.0:
            bracket = (float(left), float(right))
            break
    if bracket is None:
        raise PrecursorError(
            "the JWL release branch does not cross the air Hugoniot within "
            f"V in [{V_cj!r}, {config.v_max!r}]; mismatch at the upper end is "
            f"{values[-1]!r} m/s"
        )
    if bracket[0] == bracket[1]:
        V_sep = bracket[0]
    else:
        V_sep = brentq(
            mismatch,
            bracket[0],
            bracket[1],
            xtol=config.root_tolerance,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=200,
        )
    p_sep = _p(V_sep, jwl, C)
    u_c_sep = _release_velocity(V_sep, jwl, C, V_cj, u_cj, config.integration_limit)
    D_s, u_a_sep, rho_a_sep, e_a_sep = air_hugoniot(p_sep, air, config.shock_model)
    if abs(u_c_sep - u_a_sep) > 1.0e-6 * max(abs(u_c_sep), abs(u_a_sep), 1.0):
        raise PrecursorError(
            f"release velocity {u_c_sep!r} and air shock velocity {u_a_sep!r} "
            "do not match after the root solve"
        )

    R_cj = R0 * V_cj ** (1.0 / 3.0)
    R_sep = R0 * V_sep ** (1.0 / 3.0)
    t_cj = R_cj / det.velocity
    t_surface_ref = R0 / det.velocity

    def inverse_contact_speed(radius: float) -> float:
        V = (radius / R0) ** 3
        if V <= V_cj:
            raise PrecursorError(
                f"radius {radius!r} maps to V={V!r} below the CJ branch; the uniform "
                "release integral starts at R_cj"
            )
        return _release_velocity(V, jwl, C, V_cj, u_cj, config.integration_limit)

    # A single quadrature over [R_cj, R_sep] is sufficient; it is the same
    # uniform-sphere expansion integral used by the separation model.
    dt, _ = quad(
        lambda radius: 1.0 / inverse_contact_speed(radius),
        R_cj,
        R_sep,
        epsabs=1.0e-14,
        epsrel=1.0e-10,
        limit=config.integration_limit,
    )
    t_sep = t_cj + dt
    thickness_speed = float(D_s) - float(u_c_sep)
    if not math.isfinite(thickness_speed) or thickness_speed <= 0.0:
        raise PrecursorError(
            f"D_s_sep - u_c_sep = {thickness_speed!r} m/s is not positive; the "
            "initial thin-layer scale is undefined"
        )
    epsilon = float(R_sep) / (
        float(config.epsilon_denominator_factor) * thickness_speed
    )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise PrecursorError(f"epsilon = {epsilon!r} s is not finite and positive")
    t0 = t_sep + epsilon
    Rc0 = R_sep + u_c_sep * epsilon
    Rs0 = R_sep + D_s * epsilon
    V_c0 = V_sep * (Rc0 / R_sep) ** 3
    p_c_t0 = _p(V_c0, jwl, C)
    initial_pressure_compatibility = {
        "p_c_t0_pa": float(p_c_t0),
        "p_a_sep_pa": float(p_sep),
        "absolute_difference_pa": float(p_c_t0 - p_sep),
        "relative_difference": float(abs(p_c_t0 - p_sep) / p_sep),
        "assessment": (
            "non-negligible (>=1% of p_a_sep); reported, not forced to zero"
            if abs(p_c_t0 - p_sep) / p_sep >= 1.0e-2
            else "small (<1% of p_a_sep) but still reported"
        ),
        "forced_equal": False,
        "note": (
            "p_c(t0) is JWL at R_c(t0)=R_sep+u_c,sep*epsilon; p_a,sep is the "
            "matched air pressure. They are different states at slightly "
            "different positions/times."
        ),
    }
    clock = {
        "reference_state": "CJ detonation state at V_cj = rho_TNT/rho_CJ",
        "reference_position_m": float(R_cj),
        "reference_time_s": float(t_cj),
        "time_formula": "t_cj = R_cj / D",
        "time_formula_basis": (
            "uniform CJ detonation from the charge centre at constant D; the "
            "CJ radius R_cj=R0 V_cj^(1/3) is reached at that time. This is an "
            "approximation whose physical basis still needs confirmation against "
            "the deck/initiation model"
        ),
        "time_formula_verified": False,
        "surface_arrival_reference_s": float(t_surface_ref),
        "surface_arrival_role": (
            "R0/D is recorded as a separate surface-arrival reference only; it is "
            "not used as t_cj and not used as the model origin"
        ),
        "t_sep_definition_s": float(t_sep),
        "t0_definition": (
            "t0 = t_sep + epsilon; epsilon = R_sep / "
            "(epsilon_denominator_factor * (D_s_sep - u_c_sep))"
        ),
        "d3plot_clock_relation": (
            "the raw frame clock is data-side. The one-dimensional data contract "
            "aligns the first identifiable separation frame to model t0 via "
            "t = t0 + (t_data - t_event_data) and records the offset; this module "
            "does not apply that shift and does not claim the absolute clocks agree"
        ),
        "d3plot_clock_verified": False,
        "material_source": config.material_source,
        "material_source_verified": bool(config.material_source_verified),
    }

    checks = {
        "W_cj_in_unit_interval": bool(0.0 < V_cj < 1.0),
        "C_positive": bool(C > 0.0),
        "p_cj_consistent": bool(abs(p_cj - det.cj_pressure) <= 1.0e-6 * det.cj_pressure),
        "V_sep_above_V_cj": bool(V_sep > V_cj),
        "p_sep_above_ambient": bool(p_sep > air.pressure),
        "release_and_air_velocity_match": bool(
            abs(u_c_sep - u_a_sep) <= 1.0e-8 * max(abs(u_c_sep), abs(u_a_sep), 1.0)
        ),
        "sound_speed_positive": bool(c_cj > 0.0),
        "t0_after_t_sep": bool(t0 > t_sep),
        "thin_layer_speed_positive": bool(D_s > u_c_sep),
        "epsilon_positive": bool(epsilon > 0.0),
        "Rc0_below_Rs0": bool(Rc0 < Rs0),
        "R_sep_above_R0": bool(R_sep > R0),
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise PrecursorError(f"precursor consistency checks failed: {failed}")

    provenance = {
        "jwl": {
            "A": "config input (material-card provenance unverified)",
            "B": "config input (material-card provenance unverified)",
            "R1": "config input (material-card provenance unverified)",
            "R2": "config input (material-card provenance unverified)",
            "omega": "config input (material-card provenance unverified)",
            "rho_tnt": "config input (material-card provenance unverified)",
            "C": c_source,
        },
        "detonation": {
            "velocity": "config input (deck provenance unverified)",
            "cj_pressure": "config input (deck provenance unverified)",
            "cj_density": "config input (deck provenance unverified)",
        },
        "air": {
            "density": "config input (ambient-state provenance unverified)",
            "pressure": "config input (ambient-state provenance unverified)",
            "gamma": "config input (material-card provenance unverified)",
        },
        "material_source": {
            "value": config.material_source,
            "verified": bool(config.material_source_verified),
        },
        "clock": dict(clock),
        "shock_model": (
            "strong-shock approximation (documented; exact Hugoniot is a "
            "separate cross-check)"
            if config.shock_model == "strong"
            else "exact finite-ambient Hugoniot"
        ),
        "R_sep": "calculated from uniform-sphere mass conservation, R0 V_sep^(1/3)",
        "t_sep": "calculated from the uniform-sphere CJ-radius expansion integral; "
                 "R0/D is only a separate surface-arrival reference",
        "Rc0_Rs0": "first-order displacement approximation, not a numerical evolution",
        "missing": [
            "R_s(t_event) and the contact interval are data-side quantities and are "
            "not supplied by this module",
            "the material card's C/JWL provenance must be checked against the deck",
        ],
    }
    notes = [
        "V is the dimensionless relative volume rho_TNT/rho_product; the mass "
        "specific volume is 1/rho and is not the same quantity",
        "the uniform-sphere expansion integral starts at the CJ radius "
        "R_cj=R0 V_cj^(1/3), at t_cj=R_cj/D; the surface-arrival time R0/D is "
        "reported separately as a reference, not used as t_ref",
        f"air Hugoniot model: {config.shock_model}; the other model is available "
        "for a documented mass/velocity cross-check",
        "the same initial state is used over the shocked layer; the difference "
        "between p_c(t0) and the thin-layer p_a,sep is reported with its actual "
        "relative size by CaseData.initial_compatibility(), not dismissed as "
        "O(epsilon) and not forced to zero",
        "t_cj=R_cj/D is a uniform-CJ approximation and is marked unverified; the "
        "shared origin of this model clock and the d3plot time axis is not assumed",
        f"material source = {config.material_source!r}; verified = "
        f"{bool(config.material_source_verified)}",
    ]
    return PrecursorResult(
        config=config,
        V_cj=V_cj,
        V_sep=V_sep,
        C=C,
        c_source=c_source,
        u_cj=u_cj,
        c_cj=c_cj,
        p_cj=p_cj,
        p_sep=p_sep,
        p_c_t0=p_c_t0,
        initial_pressure_compatibility=initial_pressure_compatibility,
        u_c_sep=u_c_sep,
        u_a_sep=u_a_sep,
        rho_a_sep=rho_a_sep,
        e_a_sep=e_a_sep,
        D_s_sep=D_s,
        R_cj=R_cj,
        R_sep=R_sep,
        t_cj=t_cj,
        t_surface_ref=t_surface_ref,
        t_sep=t_sep,
        epsilon_s=epsilon,
        t0=t0,
        Rc0=Rc0,
        Rs0=Rs0,
        shock_model=config.shock_model,
        checks=checks,
        provenance=provenance,
        notes=notes,
        clock=clock,
    )


def compare_shock_models(config: PrecursorConfig) -> dict[str, Any]:
    """Run both air Hugoniot models from the same JWL release root."""
    strong_config = PrecursorConfig(
        charge_radius_m=config.charge_radius_m,
        detonation=config.detonation,
        jwl=config.jwl,
        air=config.air,
        epsilon_denominator_factor=config.epsilon_denominator_factor,
        shock_model="strong",
        v_max=config.v_max,
        integration_limit=config.integration_limit,
        root_tolerance=config.root_tolerance,
        material_source=config.material_source,
        material_source_verified=config.material_source_verified,
    )
    exact_config = PrecursorConfig(
        charge_radius_m=config.charge_radius_m,
        detonation=config.detonation,
        jwl=config.jwl,
        air=config.air,
        epsilon_denominator_factor=config.epsilon_denominator_factor,
        shock_model="exact",
        v_max=config.v_max,
        integration_limit=config.integration_limit,
        root_tolerance=config.root_tolerance,
        material_source=config.material_source,
        material_source_verified=config.material_source_verified,
    )
    strong = compute_precursor(strong_config)
    exact = compute_precursor(exact_config)
    return {
        "strong": strong.as_dict(),
        "exact": exact.as_dict(),
        "relative_difference": {
            "V_sep": abs(strong.V_sep - exact.V_sep) / exact.V_sep,
            "p_sep": abs(strong.p_sep - exact.p_sep) / exact.p_sep,
            "D_s_sep": abs(strong.D_s_sep - exact.D_s_sep) / exact.D_s_sep,
            "u_a_sep": abs(strong.u_a_sep - exact.u_a_sep) / exact.u_a_sep,
        },
    }


__all__ = [
    "AirState",
    "DetonationState",
    "JWLParameters",
    "PrecursorConfig",
    "PrecursorError",
    "PrecursorResult",
    "air_hugoniot",
    "compare_shock_models",
    "compute_precursor",
]
