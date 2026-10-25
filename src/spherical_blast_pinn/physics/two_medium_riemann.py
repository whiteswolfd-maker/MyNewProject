"""Local planar JWL-product / ideal-gas-air Riemann problem.

This module is deliberately **not** a finite-volume or Lagrangian step.  It
solves the initial discontinuity between the uniform, stationary JWL product
sphere and the uniform, stationary air at ``r = R0`` as a local planar
two-material Riemann problem.  The result is the self-similar wave fan used by
:mod:`spherical_blast_pinn.physics.two_medium_startup_planar` to build a small
positive-time startup field.

Wave pattern
------------

For the project's state (``p_product >> p_air``, both fluids at rest) the
admissible pattern is fixed by the pressure ordering:

* a left-going rarefaction in the JWL products;
* a contact discontinuity moving with the interface velocity ``u_star``;
* a right-going shock in the air.

The solver **verifies** this pattern (``p_air < p_star < p_product_initial``)
and raises :class:`RiemannError` if the admissible pattern is different.  It
does not force the expected pattern onto a state that does not support it.

JWL product rarefaction
-----------------------

The product side is isentropic.  With ``V = rho_ref / rho`` the JWL isentrope
obeys

.. math::

    \\frac{de}{dV} = -\\frac{p(V,e)}{\\rho_0}, \\qquad
    \\frac{du}{dV} = \\frac{c(V,e)}{V}.

The second relation is the Riemann invariant of a left-going simple wave:
``u_star = integral c dV / V`` from the initial state to the star state.  The
ODE is integrated once with scalar RK4 (the same closed-form JWL expressions as
the NumPy EOS) and tabulated.  The star state is obtained by matching the
pressure with the air shock.  This is the general-EOS construction; the
constant-gamma rarefaction formula is **not** substituted for JWL.

Air shock
---------

The air is a calorically perfect gas.  For a right-going shock into stationary
air the Rankine-Hugoniot relations give, for a candidate star pressure ``p``,

.. math::

    D_s = \\sqrt{\\frac{(\\gamma+1)p+(\\gamma-1)p_a}{2\\rho_a}}, \\qquad
    u_a = \\frac{p-p_a}{\\rho_a D_s}.

The star density and specific energy follow from the same three RH relations.
The solver checks mass, momentum and energy residuals independently.

Literature
----------

The pressure-matching construction and the ideal-gas shock/rarefaction
relations are the standard exact Riemann solver of Toro, *Riemann Solvers and
Numerical Methods for Fluid Dynamics*, 3rd ed., chapters 4 and 6.  The JWL
isentrope ODE is the standard caloric-EOS relation ``de = -p dv`` at constant
entropy.  The JWL--air combination and the tabulated scalar integration are
project-specific; they are not a formula taken from a paper on underwater
explosion PINNs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator

from .eos import (
    JWLParameters,
    jwl_pressure_scalar,
    jwl_sound_speed_squared_scalar,
)

#: Method version.  Any change to the isentrope integration or the matching
#: condition must bump this string; it enters the startup cache key.
RIEMANN_METHOD_VERSION = "jwl_air_planar_riemann_v1"

#: Relative tolerance used for the independent RH/contact checks.  The
#: formulas are algebraic; a mismatch larger than this is a real bug, not a
#: discretisation error.
_CHECK_REL_TOL = 1.0e-8

#: The product isentrope is integrated until its pressure is this fraction of
#: the ambient air pressure.  The star pressure is always well above p_air, so
#: the extra margin only makes the root bracket safe.
_P_FLOOR_FACTOR = 0.99


class RiemannError(ValueError):
    """The local two-material Riemann problem is inadmissible."""


@dataclass(frozen=True)
class ProductIsentrope:
    """Tabulated JWL product isentrope and left-going rarefaction invariant.

    Arrays are ordered by increasing ``V``.  ``p`` decreases and ``xi = u - c``
    increases.  The table starts at the initial product state ``V=1, u=0`` and
    extends below the ambient pressure so that the star state is always inside
    the table.
    """

    V: np.ndarray
    e: np.ndarray
    u: np.ndarray
    p: np.ndarray
    rho: np.ndarray
    c: np.ndarray
    xi: np.ndarray
    rho_ref: float
    e_ref: float
    p_ref: float
    c_ref: float
    integration_step_factor: float
    integration_steps: int
    eos_params: JWLParameters
    method_version: str = RIEMANN_METHOD_VERSION

    def __post_init__(self) -> None:
        for name in ("V", "e", "u", "p", "rho", "c", "xi"):
            object.__setattr__(
                self, name, np.asarray(getattr(self, name), dtype=np.float64)
            )
        arrays = (self.V, self.e, self.u, self.p, self.rho, self.c, self.xi)
        if any(array.ndim != 1 for array in arrays):
            raise RiemannError("isentrope arrays must be one-dimensional")
        if not all(array.size == self.V.size for array in arrays):
            raise RiemannError("isentrope arrays must have the same length")
        if self.V.size < 3:
            raise RiemannError("isentrope table is too short")
        if np.any(np.diff(self.V) <= 0.0):
            raise RiemannError("isentrope V must be strictly increasing")
        if np.any(np.diff(self.p) >= 0.0):
            raise RiemannError("isentrope pressure must be strictly decreasing")
        if np.any(np.diff(self.xi) <= 0.0):
            raise RiemannError("isentrope u-c must be strictly increasing")
        if not np.all(np.isfinite(self.e)) or np.any(self.e <= 0.0):
            raise RiemannError("isentrope specific energy must be finite and positive")
        if not np.all(np.isfinite(self.rho)) or np.any(self.rho <= 0.0):
            raise RiemannError("isentrope density must be finite and positive")
        if not np.all(np.isfinite(self.c)) or np.any(self.c <= 0.0):
            raise RiemannError("isentrope sound speed must be finite and positive")
        if not np.all(np.isfinite(self.p)) or np.any(self.p <= 0.0):
            raise RiemannError("isentrope pressure must be finite and positive")
        if not math.isfinite(self.rho_ref) or self.rho_ref <= 0.0:
            raise RiemannError("rho_ref must be positive")
        if not math.isfinite(self.e_ref) or self.e_ref <= 0.0:
            raise RiemannError("e_ref must be positive")
        if not math.isfinite(self.p_ref) or self.p_ref <= 0.0:
            raise RiemannError("p_ref must be positive")
        if not math.isfinite(self.c_ref) or self.c_ref <= 0.0:
            raise RiemannError("c_ref must be positive")
        if not isinstance(self.eos_params, JWLParameters):
            raise RiemannError("eos_params must be a JWLParameters instance")
        try:
            self.eos_params.validate()
        except Exception as exc:  # EOSError is not imported to avoid a cycle
            raise RiemannError(f"eos_params is invalid: {exc}") from exc

        # PCHIP is shape preserving and monotone on these one-dimensional
        # tables.  log(p) is used as the abscissa because the table spans
        # several decades and the star pressure can lie anywhere in it.
        # Build the interpolators once; the star-pressure bisection evaluates
        # them O(100) times.
        logp = np.log(self.p[::-1])
        object.__setattr__(
            self, "_V_of_logp", PchipInterpolator(logp, self.V[::-1], extrapolate=False)
        )
        object.__setattr__(
            self, "_e_of_logp", PchipInterpolator(logp, self.e[::-1], extrapolate=False)
        )
        object.__setattr__(
            self, "_u_of_logp", PchipInterpolator(logp, self.u[::-1], extrapolate=False)
        )
        object.__setattr__(
            self, "_V_of_xi", PchipInterpolator(self.xi, self.V, extrapolate=False)
        )
        object.__setattr__(
            self, "_e_of_V", PchipInterpolator(self.V, self.e, extrapolate=False)
        )
        object.__setattr__(
            self, "_u_of_V", PchipInterpolator(self.V, self.u, extrapolate=False)
        )

    def state_at_pressure(self, pressure: float) -> dict[str, float]:
        """Interpolate the isentrope at a star pressure."""
        p = float(pressure)
        if not math.isfinite(p) or p <= 0.0:
            raise RiemannError(f"pressure must be finite and positive, got {p!r}")
        if p > self.p[0] * (1.0 + 1.0e-12) or p < self.p[-1]:
            raise RiemannError(
                f"pressure {p!r} is outside the isentrope table "
                f"[{self.p[-1]!r}, {self.p[0]!r}]"
            )
        x = math.log(p)
        V = float(self._V_of_logp(x))
        e = float(self._e_of_logp(x))
        u = float(self._u_of_logp(x))
        rho = self.rho_ref / V
        c2 = jwl_sound_speed_squared_scalar(rho, e, self._params)
        if not math.isfinite(c2) or c2 <= 0.0:
            raise RiemannError(f"isentrope star state has inadmissible c^2={c2!r}")
        return {
            "V": V,
            "rho": rho,
            "e": e,
            "u": u,
            "p": p,
            "c": math.sqrt(c2),
        }

    def state_at_xi(self, xi: float) -> dict[str, float]:
        """State inside the rarefaction fan, parameterised by ``xi = u-c``.

        ``xi`` is the lab-frame similarity coordinate relative to the initial
        interface, in m/s.  The head is ``-c_ref`` (u=0) and the tail is
        ``u_star-c_star``.
        """
        value = float(xi)
        if not math.isfinite(value):
            raise RiemannError(f"xi must be finite, got {value!r}")
        if value < self.xi[0] - 1.0e-12 or value > self.xi[-1] + 1.0e-12:
            raise RiemannError(
                f"xi={value!r} is outside the rarefaction fan "
                f"[{self.xi[0]!r}, {self.xi[-1]!r}]"
            )
        V = float(self._V_of_xi(min(max(value, self.xi[0]), self.xi[-1])))
        e = float(self._e_of_V(V))
        u = float(self._u_of_V(V))
        p = jwl_pressure_scalar(self.rho_ref / V, e, self._params)
        c2 = jwl_sound_speed_squared_scalar(self.rho_ref / V, e, self._params)
        if not math.isfinite(p) or p <= 0.0:
            raise RiemannError(f"fan state has inadmissible pressure {p!r}")
        if not math.isfinite(c2) or c2 <= 0.0:
            raise RiemannError(f"fan state has inadmissible c^2={c2!r}")
        return {
            "V": V,
            "rho": self.rho_ref / V,
            "e": e,
            "u": u,
            "p": p,
            "c": math.sqrt(c2),
        }

    @property
    def _params(self) -> JWLParameters:
        return self.eos_params

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_version": self.method_version,
            "rho_ref": self.rho_ref,
            "e_ref": self.e_ref,
            "p_ref": self.p_ref,
            "c_ref": self.c_ref,
            "V_range": [float(self.V[0]), float(self.V[-1])],
            "p_range": [float(self.p[-1]), float(self.p[0])],
            "xi_range": [float(self.xi[0]), float(self.xi[-1])],
            "integration_step_factor": self.integration_step_factor,
            "integration_steps": self.integration_steps,
        }


def _build_product_isentrope(
    params: JWLParameters,
    *,
    rho_ref: float,
    e_ref: float,
    p_floor: float,
    integration_step_factor: float,
    max_volume_ratio: float = 1.0e6,
) -> ProductIsentrope:
    """Integrate the JWL isentrope with scalar RK4 and tabulate it."""
    params.validate()
    rho_ref = float(rho_ref)
    e_ref = float(e_ref)
    p_floor = float(p_floor)
    if not math.isfinite(rho_ref) or rho_ref <= 0.0:
        raise RiemannError("rho_ref must be finite and positive")
    if not math.isfinite(e_ref) or e_ref <= 0.0:
        raise RiemannError("e_ref must be finite and positive")
    if not math.isfinite(p_floor) or p_floor <= 0.0:
        raise RiemannError("p_floor must be finite and positive")
    if not math.isfinite(integration_step_factor) or integration_step_factor <= 0.0:
        raise RiemannError("integration_step_factor must be finite and positive")

    p_ref = jwl_pressure_scalar(rho_ref, e_ref, params)
    c2_ref = jwl_sound_speed_squared_scalar(rho_ref, e_ref, params)
    if not math.isfinite(p_ref) or p_ref <= 0.0:
        raise RiemannError(f"initial product pressure is inadmissible: {p_ref!r}")
    if not math.isfinite(c2_ref) or c2_ref <= 0.0:
        raise RiemannError(f"initial product c^2 is inadmissible: {c2_ref!r}")
    if p_ref <= p_floor:
        raise RiemannError(
            f"initial product pressure {p_ref!r} is below the isentrope floor "
            f"{p_floor!r}; no rarefaction table can be built"
        )
    c_ref = math.sqrt(c2_ref)

    def rhs(V: float, e: float) -> tuple[float, float, float, float]:
        rho = rho_ref / V
        p = jwl_pressure_scalar(rho, e, params)
        c2 = jwl_sound_speed_squared_scalar(rho, e, params)
        if not math.isfinite(p) or p <= 0.0:
            raise RiemannError(f"isentrope reached inadmissible pressure {p!r} at V={V!r}")
        if not math.isfinite(c2) or c2 <= 0.0:
            raise RiemannError(f"isentrope reached inadmissible c^2={c2!r} at V={V!r}")
        return -p / rho_ref, math.sqrt(c2) / V, p, math.sqrt(c2)

    V_values = [1.0]
    e_values = [e_ref]
    u_values = [0.0]
    p_values = [p_ref]
    c_values = [c_ref]
    V = 1.0
    e = e_ref
    u = 0.0
    while p_values[-1] > p_floor:
        h = min(max(integration_step_factor * V, 1.0e-8), 2.0)
        if V + h > max_volume_ratio:
            raise RiemannError(
                f"isentrope did not reach p_floor before V={max_volume_ratio!r}; "
                "the product EOS or the ambient pressure is inconsistent"
            )
        k1e, k1u, _, _ = rhs(V, e)
        k2e, k2u, _, _ = rhs(V + 0.5 * h, e + 0.5 * h * k1e)
        k3e, k3u, _, _ = rhs(V + 0.5 * h, e + 0.5 * h * k2e)
        k4e, k4u, _, _ = rhs(V + h, e + h * k3e)
        e += h * (k1e + 2.0 * k2e + 2.0 * k3e + k4e) / 6.0
        u += h * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0
        V += h
        rho = rho_ref / V
        p = jwl_pressure_scalar(rho, e, params)
        c2 = jwl_sound_speed_squared_scalar(rho, e, params)
        if not math.isfinite(p) or p <= 0.0:
            raise RiemannError(f"isentrope reached inadmissible pressure {p!r} at V={V!r}")
        if not math.isfinite(c2) or c2 <= 0.0:
            raise RiemannError(f"isentrope reached inadmissible c^2={c2!r} at V={V!r}")
        V_values.append(V)
        e_values.append(e)
        u_values.append(u)
        p_values.append(p)
        c_values.append(math.sqrt(c2))

    V_array = np.asarray(V_values, dtype=np.float64)
    e_array = np.asarray(e_values, dtype=np.float64)
    u_array = np.asarray(u_values, dtype=np.float64)
    p_array = np.asarray(p_values, dtype=np.float64)
    c_array = np.asarray(c_values, dtype=np.float64)
    rho_array = rho_ref / V_array
    xi_array = u_array - c_array

    isentrope = ProductIsentrope(
        V=V_array,
        e=e_array,
        u=u_array,
        p=p_array,
        rho=rho_array,
        c=c_array,
        xi=xi_array,
        rho_ref=rho_ref,
        e_ref=e_ref,
        p_ref=p_ref,
        c_ref=c_ref,
        integration_step_factor=float(integration_step_factor),
        integration_steps=int(V_array.size - 1),
        eos_params=params,
    )
    return isentrope


@dataclass(frozen=True)
class RiemannSolution:
    """Result of the local JWL--air planar Riemann problem."""

    p_star: float
    u_star: float
    product_p_star: float
    product_rho_star: float
    product_e_star: float
    product_c_star: float
    air_p_star: float
    air_rho_star: float
    air_e_star: float
    air_c_star: float
    shock_speed: float
    contact_speed: float
    rarefaction_head_speed: float
    rarefaction_tail_speed: float
    product_initial_pressure: float
    product_initial_density: float
    product_initial_specific_energy: float
    product_initial_sound_speed: float
    air_initial_pressure: float
    air_initial_density: float
    air_initial_specific_energy: float
    air_initial_sound_speed: float
    isentrope: ProductIsentrope
    checks: Mapping[str, float | bool | str]
    wave_pattern: str = "product_rarefaction-contact-air_shock"
    method_version: str = RIEMANN_METHOD_VERSION

    def __post_init__(self) -> None:
        for name in (
            "p_star",
            "u_star",
            "product_p_star",
            "product_rho_star",
            "product_e_star",
            "product_c_star",
            "air_p_star",
            "air_rho_star",
            "air_e_star",
            "air_c_star",
            "shock_speed",
            "contact_speed",
            "rarefaction_head_speed",
            "rarefaction_tail_speed",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise RiemannError(f"{name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)
        if not (self.product_rho_star > 0.0 and self.air_rho_star > 0.0):
            raise RiemannError("star densities must be positive")
        if not (self.product_e_star > 0.0 and self.air_e_star > 0.0):
            raise RiemannError("star specific energies must be positive")
        if not (self.product_c_star > 0.0 and self.air_c_star > 0.0):
            raise RiemannError("star sound speeds must be positive")
        if not (self.product_p_star > 0.0 and self.air_p_star > 0.0):
            raise RiemannError("star pressures must be positive")
        if not (self.shock_speed > self.u_star):
            raise RiemannError(
                f"air shock speed {self.shock_speed!r} must exceed u_star "
                f"{self.u_star!r}"
            )
        if not (self.p_star > self.air_initial_pressure):
            raise RiemannError("the air-side wave is not a shock")
        if not (self.p_star < self.product_initial_pressure):
            raise RiemannError("the product-side wave is not a rarefaction")
        if not (self.rarefaction_head_speed < 0.0):
            raise RiemannError("the product rarefaction head must move left")
        if not (self.rarefaction_tail_speed < self.u_star):
            raise RiemannError("the rarefaction tail must lie behind the contact")

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_version": self.method_version,
            "wave_pattern": self.wave_pattern,
            "p_star": self.p_star,
            "u_star": self.u_star,
            "product_p_star": self.product_p_star,
            "product_rho_star": self.product_rho_star,
            "product_e_star": self.product_e_star,
            "product_c_star": self.product_c_star,
            "air_p_star": self.air_p_star,
            "air_rho_star": self.air_rho_star,
            "air_e_star": self.air_e_star,
            "air_c_star": self.air_c_star,
            "shock_speed": self.shock_speed,
            "contact_speed": self.contact_speed,
            "rarefaction_head_speed": self.rarefaction_head_speed,
            "rarefaction_tail_speed": self.rarefaction_tail_speed,
            "product_initial_pressure": self.product_initial_pressure,
            "product_initial_density": self.product_initial_density,
            "product_initial_specific_energy": self.product_initial_specific_energy,
            "product_initial_sound_speed": self.product_initial_sound_speed,
            "air_initial_pressure": self.air_initial_pressure,
            "air_initial_density": self.air_initial_density,
            "air_initial_specific_energy": self.air_initial_specific_energy,
            "air_initial_sound_speed": self.air_initial_sound_speed,
            "isentrope": self.isentrope.as_dict(),
            "checks": {key: value for key, value in self.checks.items()},
        }


def _air_shock_state(
    pressure: float,
    *,
    air_density: float,
    air_pressure: float,
    air_gamma: float,
) -> dict[str, float]:
    """Ideal-gas right-going shock state for a candidate star pressure."""
    p = float(pressure)
    rho0 = float(air_density)
    p0 = float(air_pressure)
    gamma = float(air_gamma)
    if not (math.isfinite(p) and p > 0.0):
        raise RiemannError(f"air star pressure must be positive, got {p!r}")
    if not (math.isfinite(rho0) and rho0 > 0.0):
        raise RiemannError("air density must be positive")
    if not (math.isfinite(p0) and p0 > 0.0):
        raise RiemannError("air pressure must be positive")
    if not (math.isfinite(gamma) and gamma > 1.0):
        raise RiemannError("air gamma must be finite and > 1")
    D = math.sqrt(((gamma + 1.0) * p + (gamma - 1.0) * p0) / (2.0 * rho0))
    if not math.isfinite(D) or D <= 0.0:
        raise RiemannError(f"air shock speed is inadmissible: {D!r}")
    u = (p - p0) / (rho0 * D)
    rho = rho0 * ((gamma + 1.0) * p + (gamma - 1.0) * p0) / (
        (gamma - 1.0) * p + (gamma + 1.0) * p0
    )
    e = p / ((gamma - 1.0) * rho)
    c2 = gamma * p / rho
    if not (math.isfinite(rho) and rho > 0.0):
        raise RiemannError(f"air star density is inadmissible: {rho!r}")
    if not (math.isfinite(e) and e > 0.0):
        raise RiemannError(f"air star specific energy is inadmissible: {e!r}")
    if not (math.isfinite(c2) and c2 > 0.0):
        raise RiemannError(f"air star c^2 is inadmissible: {c2!r}")
    return {"p": p, "u": u, "rho": rho, "e": e, "c": math.sqrt(c2), "D": D}


def _require_exact_order(*, p_star: float, p_air: float, p_product: float) -> None:
    if not (p_air < p_star < p_product):
        raise RiemannError(
            "the local problem does not admit the expected "
            "product-rarefaction/contact/air-shock pattern: "
            f"p_air={p_air!r}, p_star={p_star!r}, p_product={p_product!r}"
        )


def solve_jwl_air_riemann(
    *,
    product_eos: JWLParameters,
    product_density: float,
    product_specific_energy: float,
    air_density: float,
    air_pressure: float,
    air_gamma: float,
    integration_step_factor: float = 0.005,
    root_relative_tolerance: float = 1.0e-11,
    max_root_iterations: int = 120,
) -> RiemannSolution:
    """Solve the local planar JWL-product/air Riemann problem.

    Parameters are SI.  The products and air are both initially at rest; the
    pressure ordering is checked, not assumed.  The product rarefaction is
    integrated with the actual JWL EOS and the air shock uses the exact
    ideal-gas Rankine-Hugoniot relations.
    """
    product_eos.validate()
    rho_p = float(product_density)
    e_p = float(product_specific_energy)
    rho_a = float(air_density)
    p_a = float(air_pressure)
    gamma_a = float(air_gamma)
    for name, value in (
        ("product_density", rho_p),
        ("product_specific_energy", e_p),
        ("air_density", rho_a),
        ("air_pressure", p_a),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise RiemannError(f"{name} must be finite and positive, got {value!r}")
    if not math.isfinite(gamma_a) or gamma_a <= 1.0:
        raise RiemannError(f"air_gamma must be finite and > 1, got {gamma_a!r}")
    if not math.isfinite(integration_step_factor) or integration_step_factor <= 0.0:
        raise RiemannError("integration_step_factor must be finite and positive")
    if not math.isfinite(root_relative_tolerance) or root_relative_tolerance <= 0.0:
        raise RiemannError("root_relative_tolerance must be finite and positive")
    if int(max_root_iterations) < 10:
        raise RiemannError("max_root_iterations must be at least 10")

    p_p = jwl_pressure_scalar(rho_p, e_p, product_eos)
    c2_p = jwl_sound_speed_squared_scalar(rho_p, e_p, product_eos)
    p_a_calc = (gamma_a - 1.0) * rho_a * (p_a / ((gamma_a - 1.0) * rho_a))
    # p_a_calc is just p_a; kept explicit to make the ideal-gas closure clear.
    if not math.isfinite(p_p) or p_p <= 0.0:
        raise RiemannError(f"initial product pressure is inadmissible: {p_p!r}")
    if not math.isfinite(c2_p) or c2_p <= 0.0:
        raise RiemannError(f"initial product c^2 is inadmissible: {c2_p!r}")
    if p_p <= p_a_calc * (1.0 + 1.0e-12):
        raise RiemannError(
            "the product initial pressure must exceed the air pressure for the "
            f"expected wave pattern; got p_product={p_p!r}, p_air={p_a_calc!r}"
        )

    p_floor = _P_FLOOR_FACTOR * p_a
    isentrope = _build_product_isentrope(
        product_eos,
        rho_ref=rho_p,
        e_ref=e_p,
        p_floor=p_floor,
        integration_step_factor=float(integration_step_factor),
    )

    def u_product(pressure: float) -> float:
        return float(isentrope.state_at_pressure(pressure)["u"])

    def u_air(pressure: float) -> float:
        return float(
            _air_shock_state(
                pressure,
                air_density=rho_a,
                air_pressure=p_a,
                air_gamma=gamma_a,
            )["u"]
        )

    def mismatch(pressure: float) -> float:
        return u_product(pressure) - u_air(pressure)

    lo = p_a * (1.0 + 1.0e-14)
    hi = p_p * (1.0 - 1.0e-14)
    f_lo = mismatch(lo)
    f_hi = mismatch(hi)
    if not (f_lo > 0.0 and f_hi < 0.0):
        raise RiemannError(
            "could not bracket the star pressure: "
            f"f(p_air+)={f_lo!r}, f(p_product-)={f_hi!r}"
        )
    for _ in range(int(max_root_iterations)):
        mid = 0.5 * (lo + hi)
        f_mid = mismatch(mid)
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) <= root_relative_tolerance * max(abs(mid), 1.0):
            break
    else:
        raise RiemannError("star-pressure bisection did not converge")
    p_star = 0.5 * (lo + hi)
    _require_exact_order(p_star=p_star, p_air=p_a, p_product=p_p)

    product_star = isentrope.state_at_pressure(p_star)
    air_star = _air_shock_state(
        p_star, air_density=rho_a, air_pressure=p_a, air_gamma=gamma_a
    )
    u_p = float(product_star["u"])
    u_a = float(air_star["u"])
    u_star = 0.5 * (u_p + u_a)
    contact_velocity_residual = abs(u_p - u_a) / max(abs(u_star), 1.0e-300)
    if contact_velocity_residual > _CHECK_REL_TOL:
        raise RiemannError(
            "product and air star velocities do not match: "
            f"u_product={u_p!r}, u_air={u_a!r}"
        )

    # The interpolated pressure and the EOS pressure at the interpolated
    # (rho, e) must agree.  This catches an interpolation or table bug.
    p_product_eos = jwl_pressure_scalar(
        product_star["rho"], product_star["e"], product_eos
    )
    product_pressure_residual = abs(p_product_eos - p_star) / max(abs(p_star), 1.0e-300)
    if product_pressure_residual > 1.0e-6:
        raise RiemannError(
            "product star pressure does not match the JWL EOS at the "
            f"interpolated state: p_star={p_star!r}, p_eos={p_product_eos!r}, "
            f"relative residual={product_pressure_residual!r}"
        )

    # Independent air RH residuals.
    D_s = float(air_star["D"])
    mass_residual = abs(rho_a * D_s - air_star["rho"] * (D_s - u_star)) / max(
        abs(rho_a * D_s), 1.0e-300
    )
    momentum_residual = abs(
        (p_star - p_a) - rho_a * D_s * u_star
    ) / max(abs(p_star - p_a), 1.0e-300)
    energy_residual = abs(
        (air_star["e"] - p_a / ((gamma_a - 1.0) * rho_a))
        - 0.5
        * (p_star + p_a)
        * (1.0 / rho_a - 1.0 / air_star["rho"])
    ) / max(abs(air_star["e"]), 1.0e-300)
    for name, residual in (
        ("air_mass_rh", mass_residual),
        ("air_momentum_rh", momentum_residual),
        ("air_energy_rh", energy_residual),
    ):
        if residual > _CHECK_REL_TOL:
            raise RiemannError(f"{name} residual is too large: {residual!r}")

    # The Riemann invariant is the integral of c/V dV along the isentrope.
    # Recompute it independently from the table up to the star V.
    V_star = float(product_star["V"])
    mask = isentrope.V <= V_star
    V_table = np.concatenate([isentrope.V[mask], [V_star]])
    c_table = np.concatenate([isentrope.c[mask], [product_star["c"]]])
    integrand = c_table / V_table
    invariant = float(
        np.sum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(V_table))
    )
    invariant_residual = abs(invariant - u_star) / max(abs(u_star), 1.0e-300)

    head_speed = -float(isentrope.c_ref)
    tail_speed = u_star - float(product_star["c"])
    if not (head_speed < 0.0 and tail_speed < u_star):
        raise RiemannError("rarefaction fan orientation is inconsistent")

    checks: dict[str, float | bool | str] = {
        "wave_pattern": "product_rarefaction-contact-air_shock",
        "p_air_lt_p_star_lt_p_product": True,
        "contact_pressure_residual": product_pressure_residual,
        "contact_velocity_residual": contact_velocity_residual,
        "product_pressure_eos_residual": product_pressure_residual,
        "product_riemann_invariant_residual": invariant_residual,
        "air_mass_rh_residual": mass_residual,
        "air_momentum_rh_residual": momentum_residual,
        "air_energy_rh_residual": energy_residual,
        "shock_speed_minus_u_star": D_s - u_star,
        "isentrope_min_pressure": float(isentrope.p[-1]),
        "isentrope_max_pressure": float(isentrope.p[0]),
        "isentrope_steps": int(isentrope.integration_steps),
        "positive_density": True,
        "positive_internal_energy": True,
        "positive_sound_speed_squared": True,
    }
    if invariant_residual > 1.0e-4:
        raise RiemannError(
            "product Riemann invariant does not match the integrated rarefaction: "
            f"integral={invariant!r}, u_star={u_star!r}"
        )

    return RiemannSolution(
        p_star=float(p_star),
        u_star=float(u_star),
        product_p_star=float(p_product_eos),
        product_rho_star=float(product_star["rho"]),
        product_e_star=float(product_star["e"]),
        product_c_star=float(product_star["c"]),
        air_p_star=float(p_star),
        air_rho_star=float(air_star["rho"]),
        air_e_star=float(air_star["e"]),
        air_c_star=float(air_star["c"]),
        shock_speed=float(D_s),
        contact_speed=float(u_star),
        rarefaction_head_speed=float(head_speed),
        rarefaction_tail_speed=float(tail_speed),
        product_initial_pressure=float(p_p),
        product_initial_density=float(rho_p),
        product_initial_specific_energy=float(e_p),
        product_initial_sound_speed=float(math.sqrt(c2_p)),
        air_initial_pressure=float(p_a),
        air_initial_density=float(rho_a),
        air_initial_specific_energy=float(p_a / ((gamma_a - 1.0) * rho_a)),
        air_initial_sound_speed=float(math.sqrt(gamma_a * p_a / rho_a)),
        isentrope=isentrope,
        checks=checks,
    )


# ==========================================================================
# Standard ideal-gas exact Riemann solver (validation only)
# ==========================================================================


@dataclass(frozen=True)
class IdealGasRiemannSolution:
    """Exact ideal-gas Riemann solution for the standard Sod-type test."""

    p_star: float
    u_star: float
    left_rho_star: float
    left_e_star: float
    left_wave: str
    right_rho_star: float
    right_e_star: float
    right_wave: str
    left_shock_speed: float | None
    right_shock_speed: float | None
    checks: Mapping[str, float | str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "p_star": self.p_star,
            "u_star": self.u_star,
            "left_rho_star": self.left_rho_star,
            "left_e_star": self.left_e_star,
            "left_wave": self.left_wave,
            "right_rho_star": self.right_rho_star,
            "right_e_star": self.right_e_star,
            "right_wave": self.right_wave,
            "left_shock_speed": self.left_shock_speed,
            "right_shock_speed": self.right_shock_speed,
            "checks": dict(self.checks),
        }


def _ideal_gas_f(p: float, rho: float, p_side: float, gamma: float) -> tuple[float, str]:
    """Velocity change from a side state to pressure ``p`` (positive for p>p_side)."""
    if p > p_side:
        A = 2.0 / ((gamma + 1.0) * rho)
        B = (gamma - 1.0) / (gamma + 1.0) * p_side
        return (p - p_side) * math.sqrt(A / (p + B)), "shock"
    c = math.sqrt(gamma * p_side / rho)
    return (
        2.0 * c / (gamma - 1.0) * ((p / p_side) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0),
        "rarefaction",
    )


def solve_ideal_gas_riemann(
    *,
    left_density: float,
    left_velocity: float,
    left_pressure: float,
    left_gamma: float,
    right_density: float,
    right_velocity: float,
    right_pressure: float,
    right_gamma: float,
    root_relative_tolerance: float = 1.0e-12,
    max_root_iterations: int = 160,
) -> IdealGasRiemannSolution:
    """Exact two-ideal-gas Riemann solver, used only for validation.

    This is the standard Toro construction.  It is intentionally separate from
    :func:`solve_jwl_air_riemann`; the JWL product side is never replaced by a
    constant-gamma approximation.
    """
    for name, value in (
        ("left_density", left_density),
        ("left_pressure", left_pressure),
        ("right_density", right_density),
        ("right_pressure", right_pressure),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise RiemannError(f"{name} must be finite and positive, got {value!r}")
    for name, value in (("left_gamma", left_gamma), ("right_gamma", right_gamma)):
        if not math.isfinite(value) or value <= 1.0:
            raise RiemannError(f"{name} must be finite and > 1, got {value!r}")
    for name, value in (("left_velocity", left_velocity), ("right_velocity", right_velocity)):
        if not math.isfinite(value):
            raise RiemannError(f"{name} must be finite, got {value!r}")

    c_left = math.sqrt(left_gamma * left_pressure / left_density)
    c_right = math.sqrt(right_gamma * right_pressure / right_density)

    # Equal pressure and velocity: no wave.  This avoids a spurious tiny
    # shock in the bracket when p_left == p_right to round-off.
    if abs(left_pressure - right_pressure) <= 1.0e-14 * max(
        left_pressure, right_pressure
    ) and abs(left_velocity - right_velocity) <= 1.0e-14 * max(c_left, c_right):
        p_star = 0.5 * (left_pressure + right_pressure)
        u_star = 0.5 * (left_velocity + right_velocity)
        return IdealGasRiemannSolution(
            p_star=float(p_star),
            u_star=float(u_star),
            left_rho_star=float(left_density),
            left_e_star=float(left_pressure / ((left_gamma - 1.0) * left_density)),
            left_wave="none",
            right_rho_star=float(right_density),
            right_e_star=float(right_pressure / ((right_gamma - 1.0) * right_density)),
            right_wave="none",
            left_shock_speed=None,
            right_shock_speed=None,
            checks={
                "contact_velocity_residual": 0.0,
                "star_pressure": float(p_star),
                "star_velocity": float(u_star),
            },
        )

    def residual(p: float) -> float:
        f_left, _ = _ideal_gas_f(p, left_density, left_pressure, left_gamma)
        f_right, _ = _ideal_gas_f(p, right_density, right_pressure, right_gamma)
        # u* = u_L - f_L = u_R + f_R.  f is positive for compression and
        # negative for expansion on either side.
        return (left_velocity - f_left) - (right_velocity + f_right)

    # Bracket the star pressure.  The residual is increasing in p for the
    # standard admissible case; if it is not, the test state is outside this
    # validation solver's intended range.
    lo = max(1.0e-12 * min(left_pressure, right_pressure), 1.0e-12)
    hi = 10.0 * max(left_pressure, right_pressure)
    f_lo = residual(lo)
    f_hi = residual(hi)
    while f_lo < 0.0 and lo > 1.0e-300:
        lo *= 0.1
        f_lo = residual(lo)
    while f_hi > 0.0 and hi < 1.0e300:
        hi *= 10.0
        f_hi = residual(hi)
    if not (f_lo > 0.0 and f_hi < 0.0):
        raise RiemannError("could not bracket the ideal-gas star pressure")
    for _ in range(int(max_root_iterations)):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) <= root_relative_tolerance * max(abs(mid), 1.0):
            break
    else:
        raise RiemannError("ideal-gas star-pressure bisection did not converge")
    p_star = 0.5 * (lo + hi)
    f_left, left_wave = _ideal_gas_f(p_star, left_density, left_pressure, left_gamma)
    f_right, right_wave = _ideal_gas_f(p_star, right_density, right_pressure, right_gamma)
    u_star = 0.5 * ((left_velocity - f_left) + (right_velocity + f_right))

    if left_wave == "shock":
        left_rho_star = left_density * (
            (p_star / left_pressure + (left_gamma - 1.0) / (left_gamma + 1.0))
            / ((left_gamma - 1.0) / (left_gamma + 1.0) * p_star / left_pressure + 1.0)
        )
        left_shock_speed = left_velocity - c_left * math.sqrt(
            (left_gamma + 1.0) / (2.0 * left_gamma) * p_star / left_pressure
            + (left_gamma - 1.0) / (2.0 * left_gamma)
        )
    else:
        left_rho_star = left_density * (p_star / left_pressure) ** (1.0 / left_gamma)
        left_shock_speed = None
    if right_wave == "shock":
        right_rho_star = right_density * (
            (p_star / right_pressure + (right_gamma - 1.0) / (right_gamma + 1.0))
            / ((right_gamma - 1.0) / (right_gamma + 1.0) * p_star / right_pressure + 1.0)
        )
        right_shock_speed = right_velocity + c_right * math.sqrt(
            (right_gamma + 1.0) / (2.0 * right_gamma) * p_star / right_pressure
            + (right_gamma - 1.0) / (2.0 * right_gamma)
        )
    else:
        right_rho_star = right_density * (p_star / right_pressure) ** (1.0 / right_gamma)
        right_shock_speed = None

    left_e_star = p_star / ((left_gamma - 1.0) * left_rho_star)
    right_e_star = p_star / ((right_gamma - 1.0) * right_rho_star)
    if not (
        p_star > 0.0
        and left_rho_star > 0.0
        and right_rho_star > 0.0
        and left_e_star > 0.0
        and right_e_star > 0.0
    ):
        raise RiemannError("ideal-gas star state is inadmissible")
    return IdealGasRiemannSolution(
        p_star=float(p_star),
        u_star=float(u_star),
        left_rho_star=float(left_rho_star),
        left_e_star=float(left_e_star),
        left_wave=left_wave,
        right_rho_star=float(right_rho_star),
        right_e_star=float(right_e_star),
        right_wave=right_wave,
        left_shock_speed=None if left_shock_speed is None else float(left_shock_speed),
        right_shock_speed=None if right_shock_speed is None else float(right_shock_speed),
        checks={
            "contact_velocity_residual": abs(
                (left_velocity - f_left) - (right_velocity + f_right)
            )
            / max(c_left, c_right, 1.0e-300),
            "star_pressure": float(p_star),
            "star_velocity": float(u_star),
        },
    )


__all__ = [
    "RIEMANN_METHOD_VERSION",
    "IdealGasRiemannSolution",
    "ProductIsentrope",
    "RiemannError",
    "RiemannSolution",
    "solve_ideal_gas_riemann",
    "solve_jwl_air_riemann",
]
