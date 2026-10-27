"""Small-positive-time startup from a local planar Riemann solution.

The zero-time field is constructed by
:mod:`spherical_blast_pinn.physics.two_medium_initial`.  This module turns it
into a field at a small positive physical time ``epsilon`` **without any
finite-volume, finite-difference or Lagrangian time stepping**.  It solves the
local planar JWL-product/air Riemann problem once, then embeds the resulting
self-similar wave fan in the spherical domain.

Geometry
--------

The embedding is a local planar approximation:

* the unperturbed product region is ``0 <= r < r_head``;
* the product rarefaction fan is ``r_head <= r < r_tail``;
* the product star plateau is ``r_tail <= r < Rc``;
* the air star plateau is ``Rc <= r < Rs``;
* the ambient air is ``r >= Rs``.

The wave positions are

.. math::

    R_c = R_0 + u_* \\epsilon, \\qquad
    R_s = R_0 + D_s \\epsilon, \\qquad
    r_{\\rm head} = R_0 - c_{p0}\\epsilon, \\qquad
    r_{\\rm tail} = R_0 + (u_* - c_{p*})\\epsilon.

The candidate air-layer ratio is ``delta = (R_s-R_c)/R_0``, so

.. math::

    \\epsilon = \\frac{\\delta R_0}{D_s-u_*}.

The local planar solution is **not** an exact spherical similarity solution.
The spherical mass and total-energy integrals therefore carry a geometric
approximation residual.  That residual is computed independently and reported;
it is not hidden by rescaling the field or by inventing an acceptance
threshold.

Wave/material sampling
----------------------

Sampling is analytic.  A cell boundary is placed at every wave position, and
the material interface is a face at ``R_c``.  There are no mixed cells.  At a
discontinuity the caller chooses the side explicitly:

* ``side="left"`` at ``R_c`` returns the product star state;
* ``side="right"`` at ``R_c`` returns the air star state;
* ``side="left"`` at ``R_s`` returns the post-shock air state;
* ``side="right"`` at ``R_s`` returns the ambient air state.

The pressure and sound speed stored in the field are recomputed from its
``(density, specific_energy)`` and the actual EOS, not copied from the Riemann
solution; the two are compared in the field checks.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .eos import (
    JWLParameters,
    ideal_gas_pressure_numpy,
    ideal_gas_sound_speed_numpy,
    jwl_pressure_numpy,
)
from .two_medium_initial import AIR, PRODUCT, TwoMediumInitialField
from .two_medium_riemann import (
    ProductIsentrope,
    RiemannError,
    RiemannSolution,
    solve_jwl_air_riemann,
)

#: Model identifier for the planar startup / handoff.  It is deliberately
#: different from the old ReferenceField construction and from the old
#: Lagrangian startup.
PLANAR_STARTUP_MODEL_ID = "uniform_product_sphere_air_planar_riemann_v1"
PLANAR_STARTUP_METHOD_VERSION = "jwl_air_planar_riemann_startup_v1"

#: Region tags used by the sampler.
REGION_INNER_PRODUCT = 0
REGION_PRODUCT_FAN = 1
REGION_PRODUCT_STAR = 2
REGION_AIR_STAR = 3
REGION_AMBIENT_AIR = 4
REGION_NAMES = {
    REGION_INNER_PRODUCT: "inner_product",
    REGION_PRODUCT_FAN: "product_rarefaction_fan",
    REGION_PRODUCT_STAR: "product_star_plateau",
    REGION_AIR_STAR: "air_star_plateau",
    REGION_AMBIENT_AIR: "ambient_air",
}

_GEOMETRY_REL_TOL = 1.0e-12
_EOS_REL_TOL = 1.0e-10
#: Saved physical arrays must equal the analytic sampler.  The builder uses
#: the same functions, so a legitimate field has bitwise-equal values; this
#: tolerance only absorbs a JSON/float round trip.
_STATE_REL_TOL = 1.0e-10
#: Scalar motion relations (Rc, Rs, head/tail, delta, epsilon).
_MOTION_REL_TOL = 1.0e-10
_MOTION_ABS_TOL = 1.0e-15


class PlanarStartupError(ValueError):
    """The planar startup field or its parameters are inadmissible."""


@dataclass(frozen=True)
class PlanarStartupConfig:
    """Grid and Riemann-solver choices for the planar startup."""

    #: Candidate air-layer ratio ``(Rs-Rc)/R0``.  It is not a validated
    #: physical constant.
    delta: float = 1.0 / 200.0
    inner_cells: int = 64
    fan_cells: int = 96
    plateau_cells: int = 8
    air_cells: int = 16
    ambient_cells: int = 64
    dense_samples: int = 8192
    integration_step_factor: float = 0.005
    root_relative_tolerance: float = 1.0e-11

    def validate(self) -> None:
        if not math.isfinite(self.delta) or self.delta <= 0.0:
            raise PlanarStartupError(f"delta must be finite and positive, got {self.delta!r}")
        for name in (
            "inner_cells",
            "fan_cells",
            "plateau_cells",
            "air_cells",
            "ambient_cells",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise PlanarStartupError(f"{name} must be an integer >= 1, got {value!r}")
        if isinstance(self.dense_samples, bool) or int(self.dense_samples) != self.dense_samples:
            raise PlanarStartupError("dense_samples must be an integer")
        if int(self.dense_samples) < 512:
            raise PlanarStartupError("dense_samples must be >= 512")
        if not math.isfinite(self.integration_step_factor) or self.integration_step_factor <= 0.0:
            raise PlanarStartupError("integration_step_factor must be finite and positive")
        if not math.isfinite(self.root_relative_tolerance) or self.root_relative_tolerance <= 0.0:
            raise PlanarStartupError("root_relative_tolerance must be finite and positive")


def _finite_positive(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise PlanarStartupError(f"{name} must be finite and positive, got {value!r}")
    return number


def _build_grid_edges(
    *,
    r0: float,
    r_head: float,
    r_tail: float,
    contact: float,
    shock: float,
    outer: float,
    config: PlanarStartupConfig,
) -> np.ndarray:
    """Build a grid with a face exactly at every wave/material boundary."""
    if not (0.0 < r_head < r_tail < contact < shock < outer):
        raise PlanarStartupError(
            "wave positions must satisfy 0 < r_head < r_tail < Rc < Rs < outer; got "
            f"{r_head!r}, {r_tail!r}, {contact!r}, {shock!r}, {outer!r}"
        )

    def segment(start: float, end: float, count: int) -> np.ndarray:
        return np.linspace(start, end, count + 1, dtype=np.float64)[1:]

    pieces = [
        np.array([0.0], dtype=np.float64),
        segment(0.0, r_head, int(config.inner_cells)),
        segment(r_head, r_tail, int(config.fan_cells)),
        segment(r_tail, contact, int(config.plateau_cells)),
        segment(contact, shock, int(config.air_cells)),
        np.geomspace(shock, outer, int(config.ambient_cells) + 1, dtype=np.float64)[1:],
    ]
    edges = np.concatenate(pieces)
    edges[-1] = outer
    if not np.all(np.isfinite(edges)):
        raise PlanarStartupError("generated grid contains non-finite radii")
    if np.any(np.diff(edges) <= 0.0):
        raise PlanarStartupError("generated grid is not strictly increasing")
    # The key positions must be present bit-for-bit; the segment concatenation
    # above puts them there, but verify rather than assume.
    for name, value in (
        ("r_head", r_head),
        ("r_tail", r_tail),
        ("contact_radius", contact),
        ("shock_radius", shock),
    ):
        if not np.any(np.isclose(edges, value, rtol=0.0, atol=1.0e-15 * outer)):
            raise PlanarStartupError(f"grid is missing the exact {name} face")
    return edges


@dataclass(frozen=True)
class PlanarStartupField:
    """A small-positive-time two-medium field built from a local Riemann fan.

    All arrays are per unit solid angle.  ``velocity`` is cell centred;
    ``node_velocity`` is the one-sided node representation: the centre node is
    zero, nodes with ``r <= Rc`` take the product-side analytic velocity,
    nodes with ``Rc < r <= Rs`` take ``u_star`` (post-shock side at ``Rs``), and
    nodes with ``r > Rs`` are zero.

    Supported branch: the product and air start at rest with
    ``p_product > p_air``, giving a product rarefaction, a contact and an air
    shock.  The fan tail speed is *not* assumed positive in general; the
    default project state happens to have a positive tail.

    The object is a **startup/handoff** object, not an accepted ReferenceField:
    it sets ``reference_field=False`` and ``training_qualified=False``.
    """

    model_id: str
    method_version: str
    time: float
    pin_time: float
    charge_radius: float
    epsilon: float
    delta: float
    contact_radius: float
    shock_radius: float
    contact_speed: float
    shock_speed: float
    rarefaction_head_radius: float
    rarefaction_tail_radius: float
    rarefaction_head_speed: float
    rarefaction_tail_speed: float
    radii: np.ndarray
    cell_volume: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    specific_energy: np.ndarray
    pressure: np.ndarray
    sound_speed: np.ndarray
    material: np.ndarray
    region: np.ndarray
    mass: np.ndarray
    node_velocity: np.ndarray
    product_eos: JWLParameters
    air_gamma: float
    initial_state: Mapping[str, float]
    solution: RiemannSolution
    checks: Mapping[str, Any]
    integrals: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reference_field: bool = False
    training_qualified: bool = False
    acceptance_status: str = (
        "NOT APPLICABLE: local planar startup approximation; not a ReferenceField"
    )

    def __post_init__(self) -> None:
        arrays = {}
        for name in (
            "radii",
            "cell_volume",
            "density",
            "velocity",
            "specific_energy",
            "pressure",
            "sound_speed",
            "mass",
            "node_velocity",
        ):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            object.__setattr__(self, name, array)
            arrays[name] = array
        material = np.asarray(self.material, dtype=np.int64)
        region = np.asarray(self.region, dtype=np.int64)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "region", region)

        if self.reference_field:
            raise PlanarStartupError(
                "this local planar startup field is not a ReferenceField"
            )
        if self.training_qualified:
            raise PlanarStartupError(
                "the local planar startup field cannot claim training_qualified=True: "
                "that flag is provenance only, not a scientific acceptance certificate"
            )
        if not isinstance(self.product_eos, JWLParameters):
            raise PlanarStartupError("product_eos must be a JWLParameters instance")
        self.product_eos.validate()
        if not isinstance(self.solution, RiemannSolution):
            raise PlanarStartupError(
                "solution must be a RiemannSolution instance"
            )
        if not isinstance(self.solution.isentrope, ProductIsentrope):
            raise PlanarStartupError(
                "solution.isentrope must be a ProductIsentrope instance"
            )
        if not isinstance(self.initial_state, Mapping):
            raise PlanarStartupError("initial_state must be a mapping")
        gamma = float(self.air_gamma)
        if not math.isfinite(gamma) or gamma <= 1.0:
            raise PlanarStartupError(f"air_gamma must be finite and > 1, got {gamma!r}")
        object.__setattr__(self, "air_gamma", gamma)

        n = material.size
        if n < 3:
            raise PlanarStartupError("the startup field must contain at least 3 cells")
        if material.ndim != 1 or region.ndim != 1 or region.size != n:
            raise PlanarStartupError("material and region must be one-dimensional with one entry per cell")
        if arrays["radii"].ndim != 1 or arrays["radii"].size != n + 1:
            raise PlanarStartupError("radii must be one-dimensional with n_cells+1 entries")
        if arrays["node_velocity"].ndim != 1 or arrays["node_velocity"].size != n + 1:
            raise PlanarStartupError("node_velocity must have n_cells+1 entries")
        for name in ("cell_volume", "density", "velocity", "specific_energy", "pressure", "sound_speed", "mass"):
            if arrays[name].ndim != 1 or arrays[name].size != n:
                raise PlanarStartupError(f"{name} must have n_cells entries")
        if np.any(~np.isfinite(arrays["radii"])) or np.any(np.diff(arrays["radii"]) <= 0.0):
            raise PlanarStartupError("radii must be finite and strictly increasing")
        if abs(float(arrays["radii"][0])) > 0.0:
            raise PlanarStartupError("radii[0] must be exactly zero")
        if not math.isfinite(float(arrays["radii"][-1])):
            raise PlanarStartupError("outer radius must be finite")
        if "outer_radius" in self.initial_state and not math.isclose(
            float(arrays["radii"][-1]),
            float(self.initial_state["outer_radius"]),
            rel_tol=_GEOMETRY_REL_TOL,
            abs_tol=1.0e-15,
        ):
            raise PlanarStartupError(
                "radii[-1] must equal the initial field's outer_radius"
            )
        for name in ("cell_volume", "density", "specific_energy", "pressure", "sound_speed", "mass"):
            if np.any(~np.isfinite(arrays[name])) or np.any(arrays[name] <= 0.0):
                raise PlanarStartupError(f"{name} must be finite and positive")
        if np.any(~np.isfinite(arrays["velocity"])) or np.any(~np.isfinite(arrays["node_velocity"])):
            raise PlanarStartupError("velocity and node_velocity must be finite")
        if abs(float(arrays["node_velocity"][0])) > 0.0:
            raise PlanarStartupError("the centre node velocity must be exactly zero")
        if not np.all(np.isin(material, (PRODUCT, AIR))):
            raise PlanarStartupError("material tags must be PRODUCT or AIR")
        n_inside = int(np.count_nonzero(material == PRODUCT))
        if not (n_inside > 0 and n_inside < n):
            raise PlanarStartupError("the field must contain both product and air cells")
        if not (np.all(material[:n_inside] == PRODUCT) and np.all(material[n_inside:] == AIR)):
            raise PlanarStartupError("material must be sorted: product cells first, then air")
        if np.any(np.diff(region) < 0):
            raise PlanarStartupError("region tags must be non-decreasing")
        if not np.all(np.isin(region, tuple(REGION_NAMES))):
            raise PlanarStartupError("region tags are invalid")

        # Exact wave/material faces.
        for name, value in (
            ("contact_radius", float(self.contact_radius)),
            ("shock_radius", float(self.shock_radius)),
            ("rarefaction_head_radius", float(self.rarefaction_head_radius)),
            ("rarefaction_tail_radius", float(self.rarefaction_tail_radius)),
        ):
            if not math.isfinite(value):
                raise PlanarStartupError(f"{name} must be finite")
            if not np.any(np.isclose(arrays["radii"], value, rtol=0.0, atol=1.0e-15 * max(value, 1.0))):
                raise PlanarStartupError(f"grid does not contain the exact {name} face")
        if not (
            0.0 < self.rarefaction_head_radius < self.rarefaction_tail_radius
            < self.contact_radius < self.shock_radius < arrays["radii"][-1]
        ):
            raise PlanarStartupError("wave positions are not strictly ordered inside the domain")
        r0 = _require_positive_scalar("charge_radius", self.charge_radius)
        epsilon = _require_positive_scalar("epsilon", self.epsilon)
        delta = _require_positive_scalar("delta", self.delta)
        contact_speed = _require_finite_scalar("contact_speed", self.contact_speed)
        shock_speed = _require_finite_scalar("shock_speed", self.shock_speed)
        head_speed = _require_finite_scalar(
            "rarefaction_head_speed", self.rarefaction_head_speed
        )
        tail_speed = _require_finite_scalar(
            "rarefaction_tail_speed", self.rarefaction_tail_speed
        )
        contact = _require_positive_scalar("contact_radius", self.contact_radius)
        shock = _require_positive_scalar("shock_radius", self.shock_radius)
        head = _require_positive_scalar(
            "rarefaction_head_radius", self.rarefaction_head_radius
        )
        tail = _require_positive_scalar(
            "rarefaction_tail_radius", self.rarefaction_tail_radius
        )
        if not math.isclose(self.time, epsilon, rel_tol=0.0, abs_tol=0.0):
            raise PlanarStartupError("time must equal epsilon for this startup field")
        if self.pin_time != 0.0:
            raise PlanarStartupError("pin_time must be exactly zero at the startup handoff")
        # Supported branch: a product rarefaction, a contact, and an air shock.
        # The tail speed is not assumed positive in general; the default state
        # happens to have a positive tail.  Ordering is required for the
        # sampler's fan/plateau partition.
        if not (shock_speed > contact_speed > 0.0):
            raise PlanarStartupError("front speeds must satisfy D_s > u_star > 0")
        if not (head_speed < tail_speed < contact_speed):
            raise PlanarStartupError("rarefaction fan speeds are not ordered")

        # Mass and geometry.
        for name in ("cell_volume", "mass"):
            if not np.all(np.isfinite(arrays[name])):
                raise PlanarStartupError(f"{name} must be finite")
        expected_volume = (arrays["radii"][1:] ** 3 - arrays["radii"][:-1] ** 3) / 3.0
        if np.any(np.abs(arrays["cell_volume"] - expected_volume) > _GEOMETRY_REL_TOL * np.maximum(expected_volume, 1.0e-300)):
            raise PlanarStartupError("cell_volume must equal (r_right^3-r_left^3)/3")
        expected_mass = arrays["density"] * expected_volume
        if np.any(np.abs(arrays["mass"] - expected_mass) > _GEOMETRY_REL_TOL * np.maximum(expected_mass, 1.0e-300)):
            raise PlanarStartupError("mass must equal density * cell_volume")

        # EOS consistency, independently of the Riemann solution.
        product = material == PRODUCT
        air = ~product
        p_product = jwl_pressure_numpy(
            arrays["density"][product], arrays["specific_energy"][product], self.product_eos
        )
        c_product = np.sqrt(
            np.maximum(
                _jwl_sound_speed_squared_array(
                    arrays["density"][product],
                    arrays["specific_energy"][product],
                    self.product_eos,
                ),
                0.0,
            )
        )
        p_air = ideal_gas_pressure_numpy(
            arrays["density"][air], arrays["specific_energy"][air], gamma
        )
        c_air = ideal_gas_sound_speed_numpy(
            arrays["density"][air], arrays["specific_energy"][air], gamma
        )
        _require_rel_match("product pressure", arrays["pressure"][product], p_product)
        _require_rel_match("product sound speed", arrays["sound_speed"][product], c_product)
        _require_rel_match("air pressure", arrays["pressure"][air], p_air)
        _require_rel_match("air sound speed", arrays["sound_speed"][air], c_air)

        # -- source consistency: the Riemann solution is the reference ----
        solution = self.solution
        _require_close_scalar("contact_speed", contact_speed, solution.u_star)
        _require_close_scalar("shock_speed", shock_speed, solution.shock_speed)
        _require_close_scalar(
            "rarefaction_head_speed", head_speed, solution.rarefaction_head_speed
        )
        _require_close_scalar(
            "rarefaction_tail_speed", tail_speed, solution.rarefaction_tail_speed
        )
        _require_close_scalar(
            "contact_radius motion", contact, r0 + solution.u_star * epsilon
        )
        _require_close_scalar(
            "shock_radius motion", shock, r0 + solution.shock_speed * epsilon
        )
        _require_close_scalar(
            "rarefaction_head_radius motion",
            head,
            r0 + solution.rarefaction_head_speed * epsilon,
        )
        _require_close_scalar(
            "rarefaction_tail_radius motion",
            tail,
            r0 + solution.rarefaction_tail_speed * epsilon,
        )
        _require_close_scalar("delta definition", delta, (shock - contact) / r0)
        _require_close_scalar(
            "epsilon definition",
            epsilon,
            delta * r0 / (solution.shock_speed - solution.u_star),
        )

        if self.product_eos.as_dict() != solution.isentrope.eos_params.as_dict():
            raise PlanarStartupError(
                "product_eos does not match the EOS stored with the Riemann solution"
            )

        def state_value(key: str) -> float:
            if key not in self.initial_state:
                raise PlanarStartupError(f"initial_state is missing {key!r}")
            return _require_finite_scalar(f"initial_state[{key!r}]", self.initial_state[key])

        _require_close_scalar(
            "initial_state charge_radius", state_value("charge_radius"), r0
        )
        _require_close_scalar(
            "initial_state outer_radius", state_value("outer_radius"), float(arrays["radii"][-1])
        )
        _require_close_scalar(
            "initial_state product_density",
            state_value("product_density"),
            solution.product_initial_density,
        )
        _require_close_scalar(
            "initial_state product_specific_energy",
            state_value("product_specific_energy"),
            solution.product_initial_specific_energy,
        )
        _require_close_scalar(
            "initial_state product_pressure",
            state_value("product_pressure"),
            solution.product_initial_pressure,
        )
        _require_close_scalar(
            "initial_state product_sound_speed",
            state_value("product_sound_speed"),
            solution.product_initial_sound_speed,
        )
        _require_close_scalar(
            "initial_state air_density",
            state_value("air_density"),
            solution.air_initial_density,
        )
        _require_close_scalar(
            "initial_state air_specific_energy",
            state_value("air_specific_energy"),
            solution.air_initial_specific_energy,
        )
        _require_close_scalar(
            "initial_state air_pressure",
            state_value("air_pressure"),
            solution.air_initial_pressure,
        )
        _require_close_scalar(
            "initial_state air_sound_speed",
            state_value("air_sound_speed"),
            solution.air_initial_sound_speed,
        )
        _require_close_scalar(
            "initial_state air_gamma", state_value("air_gamma"), gamma
        )
        _require_close_scalar(
            "isentrope rho_ref", solution.isentrope.rho_ref, solution.product_initial_density
        )
        _require_close_scalar(
            "isentrope e_ref", solution.isentrope.e_ref, solution.product_initial_specific_energy
        )
        _require_close_scalar(
            "isentrope p_ref", solution.isentrope.p_ref, solution.product_initial_pressure
        )
        _require_close_scalar(
            "isentrope c_ref", solution.isentrope.c_ref, solution.product_initial_sound_speed
        )
        p_air_from_gamma = (gamma - 1.0) * state_value("air_density") * state_value(
            "air_specific_energy"
        )
        _require_close_scalar(
            "air pressure from initial (rho, e, gamma)",
            state_value("air_pressure"),
            p_air_from_gamma,
        )

        # -- saved arrays must match the common analytic sampler -----------
        # The sampler below reads only scalar geometry, solution and initial
        # state; it does not read density/velocity/e/p/c/material/region.
        # Thus this check cannot certify itself from the arrays under test.
        view = _PlanarSampler(
            {
                "charge_radius": r0,
                "epsilon": epsilon,
                "contact_radius": contact,
                "shock_radius": shock,
                "rarefaction_head_radius": head,
                "rarefaction_tail_radius": tail,
                "radii": arrays["radii"],
                "initial_state": self.initial_state,
                "solution": solution,
                "product_eos": self.product_eos,
                "air_gamma": gamma,
            }
        )
        centres = (0.5 * (arrays["radii"][1:] ** 3 + arrays["radii"][:-1] ** 3)) ** (
            1.0 / 3.0
        )
        expected = view.sample(centres, side="right")
        for key in ("density", "velocity", "specific_energy", "pressure", "sound_speed"):
            _require_array_close(key, arrays[key], expected[key])
        if not np.array_equal(material, expected["material"]):
            raise PlanarStartupError(
                "material tags do not match the analytic wave-region sampler"
            )
        if not np.array_equal(region, expected["region"]):
            raise PlanarStartupError(
                "region tags do not match the analytic wave-region sampler"
            )

        expected_node_velocity = _expected_node_velocity(
            arrays["radii"],
            contact_radius=contact,
            shock_radius=shock,
            u_star=solution.u_star,
            view=view,
        )
        _require_array_close(
            "node_velocity",
            arrays["node_velocity"],
            expected_node_velocity,
            reference="declared one-sided node-velocity rule",
        )

        contact_faces = np.flatnonzero(
            np.isclose(arrays["radii"], contact, rtol=0.0, atol=1.0e-15 * max(contact, 1.0))
        )
        if contact_faces.size != 1:
            raise PlanarStartupError(
                "contact_radius must be exactly one grid face"
            )
        n_inside = int(np.count_nonzero(material == PRODUCT))
        if n_inside != int(contact_faces[0]):
            raise PlanarStartupError(
                "the product/air material boundary is not the contact face"
            )
        shock_faces = np.flatnonzero(
            np.isclose(arrays["radii"], shock, rtol=0.0, atol=1.0e-15 * max(shock, 1.0))
        )
        if shock_faces.size != 1:
            raise PlanarStartupError(
                "shock_radius must be exactly one grid face"
            )

    @property
    def n_cells(self) -> int:
        return int(self.material.size)

    @property
    def n_inside(self) -> int:
        return int(np.count_nonzero(self.material == PRODUCT))

    @property
    def centres(self) -> np.ndarray:
        return (0.5 * (self.radii[1:] ** 3 + self.radii[:-1] ** 3)) ** (1.0 / 3.0)

    @property
    def product_disturbed_length(self) -> float:
        return float(self.contact_radius - self.rarefaction_head_radius)

    @property
    def product_disturbed_length_over_r0(self) -> float:
        return float(self.product_disturbed_length / self.charge_radius)

    @property
    def fan_length(self) -> float:
        return float(self.rarefaction_tail_radius - self.rarefaction_head_radius)

    @property
    def air_layer_width(self) -> float:
        return float(self.shock_radius - self.contact_radius)

    def _sample_region(self, radius: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(radius, dtype=np.float64)
        if r.ndim == 0:
            r = r.reshape(1)
        if np.any(~np.isfinite(r)):
            raise PlanarStartupError("query radii must be finite")
        if np.any(r < 0.0) or np.any(r > self.radii[-1]):
            raise PlanarStartupError(
                f"query radius outside [0, {self.radii[-1]:.6g}] m"
            )
        if side not in ("left", "right"):
            raise PlanarStartupError("side must be 'left' or 'right'")

        # Half-open intervals for side='right'; closed upper boundaries for
        # side='left'.  At the contact and shock this chooses the one-sided
        # material/wave state explicitly.
        if side == "right":
            region = np.full(r.shape, REGION_AMBIENT_AIR, dtype=np.int64)
            region[r < self.rarefaction_head_radius] = REGION_INNER_PRODUCT
            region[(r >= self.rarefaction_head_radius) & (r < self.rarefaction_tail_radius)] = REGION_PRODUCT_FAN
            region[(r >= self.rarefaction_tail_radius) & (r < self.contact_radius)] = REGION_PRODUCT_STAR
            region[(r >= self.contact_radius) & (r < self.shock_radius)] = REGION_AIR_STAR
        else:
            region = np.full(r.shape, REGION_AMBIENT_AIR, dtype=np.int64)
            region[r <= self.rarefaction_head_radius] = REGION_INNER_PRODUCT
            region[(r > self.rarefaction_head_radius) & (r <= self.rarefaction_tail_radius)] = REGION_PRODUCT_FAN
            region[(r > self.rarefaction_tail_radius) & (r <= self.contact_radius)] = REGION_PRODUCT_STAR
            region[(r > self.contact_radius) & (r <= self.shock_radius)] = REGION_AIR_STAR
        return r, region

    def sample(self, radius, side: str = "right") -> dict[str, np.ndarray]:
        """Analytic sample of the startup field.

        Returns arrays of ``density, velocity, specific_energy, pressure,
        sound_speed, material, region``.  ``side`` chooses the one-sided state
        at an exact wave/material face; it does not average across a face.
        """
        r, region = self._sample_region(radius, side)
        rho = np.empty_like(r)
        u = np.empty_like(r)
        e = np.empty_like(r)

        inner = region == REGION_INNER_PRODUCT
        fan = region == REGION_PRODUCT_FAN
        pstar = region == REGION_PRODUCT_STAR
        astar = region == REGION_AIR_STAR
        ambient = region == REGION_AMBIENT_AIR

        rho[inner] = self.initial_state["product_density"]
        e[inner] = self.initial_state["product_specific_energy"]
        u[inner] = 0.0

        for index in np.flatnonzero(fan):
            xi = (float(r[index]) - self.charge_radius) / self.epsilon
            state = self.solution.isentrope.state_at_xi(xi)
            rho[index] = state["rho"]
            e[index] = state["e"]
            u[index] = state["u"]

        rho[pstar] = self.solution.product_rho_star
        e[pstar] = self.solution.product_e_star
        u[pstar] = self.solution.u_star

        rho[astar] = self.solution.air_rho_star
        e[astar] = self.solution.air_e_star
        u[astar] = self.solution.u_star

        rho[ambient] = self.initial_state["air_density"]
        e[ambient] = self.initial_state["air_specific_energy"]
        u[ambient] = 0.0

        material = np.where(region <= REGION_PRODUCT_STAR, PRODUCT, AIR).astype(np.int64)
        pressure = np.empty_like(r)
        sound_speed = np.empty_like(r)
        product = material == PRODUCT
        air = ~product
        if np.any(product):
            pressure[product] = jwl_pressure_numpy(
                rho[product], e[product], self.product_eos
            )
            c2 = _jwl_sound_speed_squared_array(
                rho[product], e[product], self.product_eos
            )
            if np.any(~np.isfinite(c2)) or np.any(c2 <= 0.0):
                raise PlanarStartupError("sampled product state has inadmissible c^2")
            sound_speed[product] = np.sqrt(c2)
        if np.any(air):
            pressure[air] = ideal_gas_pressure_numpy(rho[air], e[air], self.air_gamma)
            sound_speed[air] = ideal_gas_sound_speed_numpy(
                rho[air], e[air], self.air_gamma
            )
        if (
            np.any(~np.isfinite(pressure))
            or np.any(pressure <= 0.0)
            or np.any(~np.isfinite(sound_speed))
            or np.any(sound_speed <= 0.0)
        ):
            raise PlanarStartupError("sampled pressure or sound speed is inadmissible")
        return {
            "density": rho,
            "velocity": u,
            "specific_energy": e,
            "pressure": pressure,
            "sound_speed": sound_speed,
            "material": material,
            "region": region,
        }

    def sample_product(self, radius) -> dict[str, np.ndarray]:
        r = np.asarray(radius, dtype=np.float64)
        if np.any(r < 0.0) or np.any(r > self.contact_radius):
            raise PlanarStartupError(
                f"radius outside the product domain [0, {self.contact_radius:.6g}] m"
            )
        return self.sample(r, side="left")

    def sample_air(self, radius) -> dict[str, np.ndarray]:
        r = np.asarray(radius, dtype=np.float64)
        if np.any(r < self.contact_radius) or np.any(r > self.radii[-1]):
            raise PlanarStartupError(
                f"radius outside the air domain [{self.contact_radius:.6g}, "
                f"{self.radii[-1]:.6g}] m"
            )
        # At Rc the air-side state is the right-hand state; at Rs the air-side
        # state is the post-shock (left-hand) state.
        result = self.sample(r, side="left")
        at_contact = np.isclose(r, self.contact_radius, rtol=0.0, atol=0.0)
        if np.any(at_contact):
            right = self.sample(r[at_contact], side="right")
            for key in result:
                result[key][at_contact] = right[key]
        return result

    def contact_states(self) -> dict[str, Any]:
        left = self.sample_product(np.array([self.contact_radius]))
        right = self.sample_air(np.array([self.contact_radius]))
        p_left = float(left["pressure"][0])
        p_right = float(right["pressure"][0])
        u_left = float(left["velocity"][0])
        u_right = float(right["velocity"][0])
        return {
            "product": {key: float(value[0]) if value.ndim else float(value) for key, value in left.items() if key not in ("region", "material")},
            "air": {key: float(value[0]) if value.ndim else float(value) for key, value in right.items() if key not in ("region", "material")},
            "pressure_relative": abs(p_left - p_right) / max(abs(p_left), abs(p_right), 1.0e-300),
            "velocity_relative": abs(u_left - u_right) / max(abs(u_left), abs(u_right), 1.0e-300),
        }

    def shock_states(self) -> dict[str, Any]:
        post = self.sample_air(np.array([self.shock_radius]))
        ambient = self.sample(np.array([self.shock_radius]), side="right")
        return {
            "post_shock": {key: float(value[0]) for key, value in post.items() if key not in ("region", "material")},
            "ambient": {key: float(value[0]) for key, value in ambient.items() if key not in ("region", "material")},
        }

    def grid_integrals(self) -> dict[str, float]:
        product = self.material == PRODUCT
        air = ~product
        mass = float(np.sum(self.mass))
        internal = float(np.sum(self.mass * self.specific_energy))
        kinetic = float(0.5 * np.sum(self.mass * self.velocity**2))
        return {
            "mass_total": mass,
            "mass_product": float(np.sum(self.mass[product])),
            "mass_air": float(np.sum(self.mass[air])),
            "internal_energy": internal,
            "kinetic_energy": kinetic,
            "total_energy": internal + kinetic,
            "internal_product": float(np.sum(self.mass[product] * self.specific_energy[product])),
            "internal_air": float(np.sum(self.mass[air] * self.specific_energy[air])),
        }

    def dense_integrals(self, *, samples: int | None = None) -> dict[str, float]:
        """Independent high-resolution spherical and planar integrals.

        The dense grid is built from the same wave faces, not from the handoff
        cell centres.  It is used to separate the geometric embedding residual
        from the handoff-grid quadrature error.
        """
        if samples is None:
            samples = int(
                self.metadata.get("grid_config", {}).get("dense_samples", 8192)
            )
        count = int(samples)
        if count < 512:
            raise PlanarStartupError("dense integral samples must be >= 512")
        keys = np.array(
            [
                0.0,
                self.rarefaction_head_radius,
                self.rarefaction_tail_radius,
                self.contact_radius,
                self.shock_radius,
                float(self.radii[-1]),
            ],
            dtype=np.float64,
        )
        keys = np.unique(keys)
        per_interval = max(4, count // max(1, keys.size - 1))
        edges = [keys[0]]
        for left, right in zip(keys[:-1], keys[1:]):
            if right <= left:
                continue
            piece = np.linspace(left, right, per_interval + 1, dtype=np.float64)[1:]
            edges.extend(piece.tolist())
        edges = np.asarray(edges, dtype=np.float64)
        edges[-1] = self.radii[-1]
        if np.any(np.diff(edges) <= 0.0):
            raise PlanarStartupError("dense integral grid is not increasing")
        mids = 0.5 * (edges[:-1] + edges[1:])
        states = self.sample(mids, side="right")
        dr = edges[1:] - edges[:-1]
        spherical_volume = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
        rho = states["density"]
        e = states["specific_energy"]
        u = states["velocity"]
        material = states["material"]
        product = material == PRODUCT
        air = ~product
        mass = rho * spherical_volume
        return {
            "dense_samples": int(edges.size - 1),
            "mass_total": float(np.sum(mass)),
            "mass_product": float(np.sum(mass[product])),
            "mass_air": float(np.sum(mass[air])),
            "internal_energy": float(np.sum(mass * e)),
            "kinetic_energy": float(0.5 * np.sum(mass * u**2)),
            "total_energy": float(np.sum(mass * (e + 0.5 * u**2))),
            "planar_mass_total": float(np.sum(rho * dr)),
            "planar_mass_product": float(np.sum(rho[product] * dr[product])),
            "planar_mass_air": float(np.sum(rho[air] * dr[air])),
            "planar_internal_energy": float(np.sum(rho * e * dr)),
            "planar_kinetic_energy": float(0.5 * np.sum(rho * u**2 * dr)),
            "planar_total_energy": float(np.sum(rho * (e + 0.5 * u**2) * dr)),
        }

    def cache_key(self) -> str:
        payload = {
            "model_id": self.model_id,
            "method_version": self.method_version,
            "riemann_method_version": self.solution.method_version,
            "charge_radius": float(self.charge_radius),
            "epsilon": float(self.epsilon),
            "delta": float(self.delta),
            "contact_radius": float(self.contact_radius),
            "shock_radius": float(self.shock_radius),
            "contact_speed": float(self.contact_speed),
            "shock_speed": float(self.shock_speed),
            "product_eos": self.product_eos.as_dict(),
            "air_gamma": float(self.air_gamma),
            "initial_state": dict(self.initial_state),
            "n_cells": int(self.n_cells),
            "n_inside": int(self.n_inside),
            "grid_config": dict(self.metadata.get("grid_config", {})),
            "isentrope_steps": int(self.solution.checks["isentrope_steps"]),
            "grid": {
                "rarefaction_head_radius": float(self.rarefaction_head_radius),
                "rarefaction_tail_radius": float(self.rarefaction_tail_radius),
            },
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        report = dict(self.metadata)
        report.update(
            {
                "model_id": self.model_id,
                "method_version": self.method_version,
                "riemann_method_version": self.solution.method_version,
                "physical_time_s": float(self.time),
                "pin_time_s": float(self.pin_time),
                "charge_radius_m": float(self.charge_radius),
                "epsilon_s": float(self.epsilon),
                "delta": float(self.delta),
                "contact_radius_m": float(self.contact_radius),
                "shock_radius_m": float(self.shock_radius),
                "air_layer_width_m": float(self.air_layer_width),
                "air_layer_over_r0": float(self.air_layer_width / self.charge_radius),
                "contact_speed_m_s": float(self.contact_speed),
                "shock_speed_m_s": float(self.shock_speed),
                "rarefaction_head_radius_m": float(self.rarefaction_head_radius),
                "rarefaction_tail_radius_m": float(self.rarefaction_tail_radius),
                "rarefaction_head_speed_m_s": float(self.rarefaction_head_speed),
                "rarefaction_tail_speed_m_s": float(self.rarefaction_tail_speed),
                "product_disturbed_length_m": float(self.product_disturbed_length),
                "product_disturbed_length_over_r0": float(self.product_disturbed_length_over_r0),
                "fan_length_m": float(self.fan_length),
                "spherical_energy_residual": float(
                    self.checks.get("spherical_energy_residual", float("nan"))
                ),
                "initial_product_internal_energy_per_sr": float(
                    self.integrals.get("initial_product_internal_energy_per_sr", float("nan"))
                ),
                "spherical_energy_change_signed_j_per_sr": float(
                    self.integrals.get("spherical_energy_change_signed_j_per_sr", float("nan"))
                ),
                "spherical_energy_change_abs_j_per_sr": float(
                    self.integrals.get("spherical_energy_change_abs_j_per_sr", float("nan"))
                ),
                "spherical_energy_change_abs_over_initial_product_internal": float(
                    self.integrals.get(
                        "spherical_energy_change_abs_over_initial_product_internal",
                        float("nan"),
                    )
                ),
                "n_cells": int(self.n_cells),
                "n_product_cells": int(self.n_inside),
                "grid_integrals": dict(self.integrals.get("grid", {})),
                "dense_integrals": dict(self.integrals.get("dense", {})),
                "checks": {key: value for key, value in self.checks.items()},
                "integrals": {key: value for key, value in self.integrals.items()},
                "riemann": self.solution.as_dict(),
                "initial_state": dict(self.initial_state),
                "product_eos": self.product_eos.as_dict(),
                "air_gamma": float(self.air_gamma),
                "units": "SI (m, kg, s, Pa, J/kg); mass and energy per unit solid angle",
                "reference_field": bool(self.reference_field),
                "training_qualified": bool(self.training_qualified),
                "acceptance_status": self.acceptance_status,
                "cache_key": self.cache_key(),
            }
        )
        return report


def _jwl_sound_speed_squared_array(rho, e, params: JWLParameters):
    """Vectorised JWL c^2 without clamping, using the same scalar formula."""
    rho = np.asarray(rho, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    return np.asarray(
        [
            _scalar_c2(float(r), float(ee), params)
            for r, ee in zip(rho.ravel(), e.ravel())
        ],
        dtype=np.float64,
    ).reshape(rho.shape)


def _scalar_c2(rho: float, e: float, params: JWLParameters) -> float:
    from .eos import jwl_sound_speed_squared_scalar

    return jwl_sound_speed_squared_scalar(rho, e, params)


def _require_rel_match(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if not np.all(np.isfinite(expected)) or np.any(expected <= 0.0):
        raise PlanarStartupError(f"{name} recomputed from the EOS is not finite and positive")
    if not np.all(np.isfinite(actual)):
        raise PlanarStartupError(f"{name} contains non-finite values")
    scale = np.maximum(np.abs(actual), np.abs(expected))
    residual = np.abs(actual - expected) / np.maximum(scale, 1.0e-300)
    worst = float(np.max(residual)) if residual.size else 0.0
    if worst > _EOS_REL_TOL:
        raise PlanarStartupError(
            f"{name} does not match the EOS: max relative residual {worst:.3e} > "
            f"{_EOS_REL_TOL:.1e}"
        )


def _initial_integrals(initial_state: Mapping[str, float]) -> dict[str, float]:
    """Initial per-steradian integrals from the saved initial parameters.

    This is the same formula used by the zero-time construction: the product
    occupies ``0 <= r <= R0`` and the air occupies ``R0 <= r <= outer``.  It is
    deliberately defined from the state, not read back from a saved
    ``checks``/``integrals`` block.
    """
    rho_p = float(initial_state["product_density"])
    e_p = float(initial_state["product_specific_energy"])
    rho_a = float(initial_state["air_density"])
    e_a = float(initial_state["air_specific_energy"])
    r0 = float(initial_state["charge_radius"])
    outer = float(initial_state["outer_radius"])
    product_mass = rho_p * r0**3 / 3.0
    air_mass = rho_a * (outer**3 - r0**3) / 3.0
    product_internal = product_mass * e_p
    internal = product_internal + air_mass * e_a
    return {
        "initial_mass_product": product_mass,
        "initial_mass_air": air_mass,
        "initial_mass_total": product_mass + air_mass,
        "initial_internal_energy": internal,
        "initial_product_internal_energy": product_internal,
        "initial_kinetic_energy": 0.0,
        "initial_total_energy": internal,
        "initial_planar_mass_product": rho_p * r0,
        "initial_planar_mass_air": rho_a * (outer - r0),
        "initial_planar_mass_total": rho_p * r0 + rho_a * (outer - r0),
        "initial_planar_internal_energy": rho_p * e_p * r0 + rho_a * e_a * (outer - r0),
    }


def _relative_difference(value: float, reference: float) -> float:
    return float(abs(value - reference) / max(abs(reference), 1.0e-300))


def _require_finite_scalar(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanarStartupError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(number):
        raise PlanarStartupError(f"{name} must be finite, got {value!r}")
    return number


def _require_positive_scalar(name: str, value: Any) -> float:
    number = _require_finite_scalar(name, value)
    if number <= 0.0:
        raise PlanarStartupError(f"{name} must be positive, got {value!r}")
    return number


def _require_close_scalar(
    name: str,
    actual: Any,
    expected: Any,
    *,
    rel_tol: float = _MOTION_REL_TOL,
    abs_tol: float = _MOTION_ABS_TOL,
) -> None:
    a = _require_finite_scalar(name, actual)
    b = _require_finite_scalar(f"{name} reference", expected)
    if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        raise PlanarStartupError(
            f"{name} is inconsistent: got {a!r}, expected {b!r}"
        )


def _require_array_close(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    reference: str = "analytic sampler",
) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise PlanarStartupError(
            f"{name} shape {actual.shape} does not match the {reference} "
            f"{expected.shape}"
        )
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise PlanarStartupError(f"{name} contains non-finite values")
    scale = max(
        1.0,
        float(np.max(np.abs(actual))) if actual.size else 0.0,
        float(np.max(np.abs(expected))) if expected.size else 0.0,
    )
    difference = np.abs(actual - expected)
    worst = float(np.max(difference)) if difference.size else 0.0
    if worst > _STATE_REL_TOL * scale:
        index = int(np.argmax(difference))
        raise PlanarStartupError(
            f"{name} does not match the {reference}: max |difference| "
            f"{worst:.3e} at cell {index} (stored {float(actual[index])!r}, "
            f"expected {float(expected[index])!r})"
        )


def _expected_node_velocity(
    radii: np.ndarray,
    *,
    contact_radius: float,
    shock_radius: float,
    u_star: float,
    view: Any,
) -> np.ndarray:
    """The declared one-sided node-velocity rule.

    * ``r = 0``: zero;
    * ``0 < r <= Rc``: the product-side analytic velocity (``u_star`` at Rc);
    * ``Rc < r <= Rs``: ``u_star`` (air star plateau, post-shock side at Rs);
    * ``r > Rs``: zero (ambient air).

    The node at the shock is assigned the post-shock value, not an average.
    """
    radii = np.asarray(radii, dtype=np.float64)
    result = np.empty_like(radii)
    for index, radius in enumerate(radii):
        if radius <= contact_radius:
            result[index] = float(
                view.sample_product(np.array([radius]))["velocity"][0]
            )
        elif radius <= shock_radius:
            result[index] = float(u_star)
        else:
            result[index] = 0.0
    result[0] = 0.0
    return result


def compute_planar_startup_diagnostics(
    field: PlanarStartupField,
    *,
    dense_samples: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute the startup checks and integrals from the physical field.

    Both the constructor and the handoff loader call this function.  It reads
    the cell/node arrays, the common analytic sampler, the initial parameters
    and the reconstructed Riemann solution.  It never reads ``field.checks`` or
    ``field.integrals``, so a saved diagnostic block cannot certify itself.

    ``dense_samples`` defaults to the field's own grid configuration.  The
    returned pair is exactly the data used by :meth:`PlanarStartupField.as_dict`.
    """
    if not isinstance(field, PlanarStartupField):
        raise PlanarStartupError(
            "diagnostics require a PlanarStartupField built from validated arrays"
        )
    r0 = float(field.charge_radius)
    epsilon = float(field.epsilon)
    delta = float(field.delta)
    contact = float(field.contact_radius)
    shock = float(field.shock_radius)
    head = float(field.rarefaction_head_radius)
    tail = float(field.rarefaction_tail_radius)
    outer = float(field.radii[-1])

    grid = field.grid_integrals()
    dense = field.dense_integrals(samples=dense_samples)
    initial = _initial_integrals(field.initial_state)
    contact_states = field.contact_states()
    shock_states = field.shock_states()
    product_initial_energy = float(initial["initial_product_internal_energy"])
    if not math.isfinite(product_initial_energy) or product_initial_energy <= 0.0:
        raise PlanarStartupError(
            "initial product internal energy must be finite and positive"
        )
    spherical_energy_change = float(
        dense["total_energy"] - initial["initial_total_energy"]
    )
    spherical_energy_change_abs = abs(spherical_energy_change)
    spherical_energy_change_over_product = (
        spherical_energy_change_abs / product_initial_energy
    )

    checks: dict[str, Any] = {
        "wave_pattern": field.solution.wave_pattern,
        "riemann_checks": dict(field.solution.checks),
        "epsilon_formula_residual": abs(
            epsilon
            * (field.solution.shock_speed - field.solution.u_star)
            / (delta * r0)
            - 1.0
        ),
        "d_s_gt_u_star": bool(field.solution.shock_speed > field.solution.u_star),
        "epsilon_finite_positive": bool(math.isfinite(epsilon) and epsilon > 0.0),
        "rarefaction_head_radius_positive": bool(head > 0.0),
        "release_wave_does_not_reach_centre": bool(head > 0.0),
        "fronts_ordered": bool(0.0 < head < tail < contact < shock < outer),
        "centre_node_velocity_zero": bool(field.node_velocity[0] == 0.0),
        "material_sorted": bool(
            np.all(field.material[: field.n_inside] == PRODUCT)
            and np.all(field.material[field.n_inside :] == AIR)
        ),
        "contact_face_is_grid_face": bool(
            np.any(
                np.isclose(
                    field.radii,
                    contact,
                    rtol=0.0,
                    atol=1.0e-15 * max(contact, 1.0),
                )
            )
        ),
        "shock_face_is_grid_face": bool(
            np.any(
                np.isclose(
                    field.radii,
                    shock,
                    rtol=0.0,
                    atol=1.0e-15 * max(shock, 1.0),
                )
            )
        ),
        "contact_pressure_relative": contact_states["pressure_relative"],
        "contact_velocity_relative": contact_states["velocity_relative"],
        "shock_post_to_ambient_pressure_ratio": (
            shock_states["post_shock"]["pressure"]
            / shock_states["ambient"]["pressure"]
        ),
        "product_disturbed_length_over_r0": field.product_disturbed_length_over_r0,
        "fan_length_over_r0": field.fan_length / r0,
        "air_layer_over_r0": field.air_layer_width / r0,
        "grid_integrals": grid,
        "dense_integrals": dense,
        "initial_integrals": initial,
        "spherical_mass_residual": _relative_difference(
            dense["mass_total"], initial["initial_mass_total"]
        ),
        "spherical_product_mass_residual": _relative_difference(
            dense["mass_product"], initial["initial_mass_product"]
        ),
        "spherical_air_mass_residual": _relative_difference(
            dense["mass_air"], initial["initial_mass_air"]
        ),
        "spherical_energy_residual": _relative_difference(
            dense["total_energy"], initial["initial_total_energy"]
        ),
        "initial_product_internal_energy_per_sr": product_initial_energy,
        "spherical_energy_change_signed_j_per_sr": spherical_energy_change,
        "spherical_energy_change_abs_j_per_sr": spherical_energy_change_abs,
        "spherical_energy_change_abs_over_initial_product_internal": (
            spherical_energy_change_over_product
        ),
        "planar_mass_residual": _relative_difference(
            dense["planar_mass_total"], initial["initial_planar_mass_total"]
        ),
        "planar_energy_residual": _relative_difference(
            dense["planar_total_energy"], initial["initial_planar_internal_energy"]
        ),
        "grid_vs_dense_spherical_mass_residual": _relative_difference(
            grid["mass_total"], dense["mass_total"]
        ),
        "grid_vs_dense_spherical_energy_residual": _relative_difference(
            grid["total_energy"], dense["total_energy"]
        ),
        "all_densities_positive": bool(np.all(field.density > 0.0)),
        "all_internal_energies_positive": bool(np.all(field.specific_energy > 0.0)),
        "all_sound_speed_squared_positive": bool(np.all(field.sound_speed > 0.0)),
    }
    checks["passed"] = bool(
        checks["d_s_gt_u_star"]
        and checks["epsilon_finite_positive"]
        and checks["release_wave_does_not_reach_centre"]
        and checks["fronts_ordered"]
        and checks["centre_node_velocity_zero"]
        and checks["material_sorted"]
        and checks["contact_face_is_grid_face"]
        and checks["shock_face_is_grid_face"]
        and checks["contact_pressure_relative"] < 1.0e-6
        and checks["contact_velocity_relative"] < 1.0e-8
        and checks["all_densities_positive"]
        and checks["all_internal_energies_positive"]
        and checks["all_sound_speed_squared_positive"]
    )

    integrals = {
        "grid": grid,
        "dense": dense,
        "initial": initial,
        "spherical_mass_residual": checks["spherical_mass_residual"],
        "spherical_product_mass_residual": checks["spherical_product_mass_residual"],
        "spherical_air_mass_residual": checks["spherical_air_mass_residual"],
        "spherical_energy_residual": checks["spherical_energy_residual"],
        "initial_product_internal_energy_per_sr": checks[
            "initial_product_internal_energy_per_sr"
        ],
        "spherical_energy_change_signed_j_per_sr": checks[
            "spherical_energy_change_signed_j_per_sr"
        ],
        "spherical_energy_change_abs_j_per_sr": checks[
            "spherical_energy_change_abs_j_per_sr"
        ],
        "spherical_energy_change_abs_over_initial_product_internal": checks[
            "spherical_energy_change_abs_over_initial_product_internal"
        ],
        "planar_mass_residual": checks["planar_mass_residual"],
        "planar_energy_residual": checks["planar_energy_residual"],
        "grid_vs_dense_spherical_mass_residual": checks[
            "grid_vs_dense_spherical_mass_residual"
        ],
        "grid_vs_dense_spherical_energy_residual": checks[
            "grid_vs_dense_spherical_energy_residual"
        ],
        "convention": (
            "spherical: integrate rho*r^2 dr (per steradian); "
            "planar: integrate rho*dr (per unit area); "
            "initial spherical values are per steradian"
        ),
        "energy_normalization": (
            "spherical_energy_residual uses the total initial energy (including "
            "ambient air) and therefore depends on the far outer boundary; "
            "spherical_energy_change_abs_over_initial_product_internal uses only "
            "the initial product internal energy and is the supplemental metric. "
            "Neither grants training qualification."
        ),
    }
    return checks, integrals


def build_planar_startup_field(
    initial_field: TwoMediumInitialField,
    *,
    config: PlanarStartupConfig | None = None,
) -> PlanarStartupField:
    """Build the ``epsilon``-time planar-Riemann startup field.

    The input must be the exact zero-time field
    :class:`TwoMediumInitialField`.  No old detonation, precursor or Lagrangian
    startup routine is imported or called.
    """
    if not isinstance(initial_field, TwoMediumInitialField):
        raise PlanarStartupError(
            "initial_field must be a TwoMediumInitialField built by "
            "build_two_medium_initial_field"
        )
    if initial_field.time != 0.0:
        raise PlanarStartupError("the initial field must be at exactly t_phys = 0")
    cfg = config or PlanarStartupConfig()
    cfg.validate()

    r0 = float(initial_field.charge_radius)
    outer = float(initial_field.outer_radius)
    rho_p = float(initial_field.product_state["density"])
    e_p = float(initial_field.product_state["specific_energy"])
    rho_a = float(initial_field.air_state["density"])
    p_a = float(initial_field.air_state["pressure"])
    e_a = float(initial_field.air_state["specific_energy"])
    gamma = float(initial_field.air_gamma)
    params = initial_field.product_eos

    try:
        solution = solve_jwl_air_riemann(
            product_eos=params,
            product_density=rho_p,
            product_specific_energy=e_p,
            air_density=rho_a,
            air_pressure=p_a,
            air_gamma=gamma,
            integration_step_factor=float(cfg.integration_step_factor),
            root_relative_tolerance=float(cfg.root_relative_tolerance),
        )
    except RiemannError as exc:
        raise PlanarStartupError(f"local JWL/air Riemann solve failed: {exc}") from exc

    if not (solution.shock_speed > solution.u_star > 0.0):
        raise PlanarStartupError("the Riemann solution does not give D_s > u_star > 0")
    epsilon = float(cfg.delta) * r0 / (solution.shock_speed - solution.u_star)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise PlanarStartupError(f"epsilon is inadmissible: {epsilon!r}")
    contact = r0 + solution.u_star * epsilon
    shock = r0 + solution.shock_speed * epsilon
    head = r0 + solution.rarefaction_head_speed * epsilon
    tail = r0 + solution.rarefaction_tail_speed * epsilon
    if not (0.0 < head < tail < contact < shock < outer):
        raise PlanarStartupError(
            "the local planar wave positions do not fit the domain: "
            f"head={head!r}, tail={tail!r}, Rc={contact!r}, Rs={shock!r}, outer={outer!r}"
        )

    edges = _build_grid_edges(
        r0=r0,
        r_head=head,
        r_tail=tail,
        contact=contact,
        shock=shock,
        outer=outer,
        config=cfg,
    )
    volume = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    centres = (0.5 * (edges[1:] ** 3 + edges[:-1] ** 3)) ** (1.0 / 3.0)

    # The analytic sampler needs the wave positions and epsilon.  Build the
    # object first with placeholder arrays, sample, then construct the final
    # frozen object.  This keeps one sampling implementation.
    probe_metadata = {
        "model_id": PLANAR_STARTUP_MODEL_ID,
        "construction": "local planar JWL-product/air Riemann fan embedded in a sphere",
        "method_version": PLANAR_STARTUP_METHOD_VERSION,
        "physical_time_s": epsilon,
        "pin_time_s": 0.0,
        "time_convention": "t_phys = epsilon; tau = t_phys - epsilon = 0",
        "approximation": (
            "local planar early-time approximation; not an exact spherical "
            "similarity solution and not the full JWL-air initialization of an "
            "underwater PINN paper"
        ),
        "units": "SI (m, kg, s, Pa, J/kg); mass and energy per unit solid angle",
        "cell_volume_rule": "(r_{i+1}^3 - r_i^3)/3 (spherical, per unit solid angle)",
        "grid_rule": (
            "faces at r_head, r_tail, Rc, Rs and the outer boundary; no mixed "
            "material cells; cell-centre sampling"
        ),
        "reference_field": False,
        "training_qualified": False,
        "acceptance_status": (
            "NOT APPLICABLE: local planar startup approximation; not a ReferenceField"
        ),
        "riemann_method_version": solution.method_version,
        "delta_candidate": float(cfg.delta),
        "delta_status": "candidate value, not a validated constant",
        "energy_normalization_note": (
            "spherical_energy_residual keeps the original global definition "
            "(denominator includes ambient air and depends on the outer boundary); "
            "the supplemental abs(delta E)/initial product internal energy is "
            "reported separately and does not grant training qualification"
        ),
        "grid_config": {
            "inner_cells": int(cfg.inner_cells),
            "fan_cells": int(cfg.fan_cells),
            "plateau_cells": int(cfg.plateau_cells),
            "air_cells": int(cfg.air_cells),
            "ambient_cells": int(cfg.ambient_cells),
            "dense_samples": int(cfg.dense_samples),
            "integration_step_factor": float(cfg.integration_step_factor),
            "root_relative_tolerance": float(cfg.root_relative_tolerance),
        },
    }
    initial_state = {
        "charge_radius": r0,
        "outer_radius": outer,
        "product_density": rho_p,
        "product_specific_energy": e_p,
        "air_density": rho_a,
        "air_specific_energy": e_a,
        "air_pressure": p_a,
        "air_gamma": gamma,
        "product_pressure": float(initial_field.product_state["pressure"]),
        "product_sound_speed": float(initial_field.product_state["sound_speed"]),
        "air_sound_speed": float(initial_field.air_state["sound_speed"]),
    }
    placeholder = dict(
        model_id=PLANAR_STARTUP_MODEL_ID,
        method_version=PLANAR_STARTUP_METHOD_VERSION,
        time=epsilon,
        pin_time=0.0,
        charge_radius=r0,
        epsilon=epsilon,
        delta=float(cfg.delta),
        contact_radius=contact,
        shock_radius=shock,
        contact_speed=solution.u_star,
        shock_speed=solution.shock_speed,
        rarefaction_head_radius=head,
        rarefaction_tail_radius=tail,
        rarefaction_head_speed=solution.rarefaction_head_speed,
        rarefaction_tail_speed=solution.rarefaction_tail_speed,
        radii=edges,
        cell_volume=volume,
        density=np.ones_like(volume),
        velocity=np.zeros_like(volume),
        specific_energy=np.ones_like(volume),
        pressure=np.ones_like(volume),
        sound_speed=np.ones_like(volume),
        material=np.zeros(volume.size, dtype=np.int64),
        region=np.zeros(volume.size, dtype=np.int64),
        mass=np.ones_like(volume),
        node_velocity=np.zeros(edges.size),
        product_eos=params,
        air_gamma=gamma,
        initial_state=initial_state,
        solution=solution,
        checks={},
        integrals={},
        metadata=probe_metadata,
    )
    # Build a lightweight sampler without running the heavy __post_init__
    # validation on placeholder arrays.  The final object is constructed once
    # below and validates everything.
    sampler = _PlanarSampler(placeholder)

    states = sampler.sample(centres, side="right")
    density = np.asarray(states["density"], dtype=np.float64)
    velocity = np.asarray(states["velocity"], dtype=np.float64)
    specific_energy = np.asarray(states["specific_energy"], dtype=np.float64)
    pressure = np.asarray(states["pressure"], dtype=np.float64)
    sound_speed = np.asarray(states["sound_speed"], dtype=np.float64)
    region = np.asarray(states["region"], dtype=np.int64)
    material = np.where(region <= REGION_PRODUCT_STAR, PRODUCT, AIR).astype(np.int64)
    mass = density * volume

    node_velocity = _expected_node_velocity(
        edges,
        contact_radius=contact,
        shock_radius=shock,
        u_star=solution.u_star,
        view=sampler,
    )

    # Construct the validated field first; then use it for the independent
    # integrals and checks (the sampler is analytic, so this is not circular
    # for mass/energy, only for the EOS/geometry checks).
    field = PlanarStartupField(
        model_id=PLANAR_STARTUP_MODEL_ID,
        method_version=PLANAR_STARTUP_METHOD_VERSION,
        time=epsilon,
        pin_time=0.0,
        charge_radius=r0,
        epsilon=epsilon,
        delta=float(cfg.delta),
        contact_radius=contact,
        shock_radius=shock,
        contact_speed=solution.u_star,
        shock_speed=solution.shock_speed,
        rarefaction_head_radius=head,
        rarefaction_tail_radius=tail,
        rarefaction_head_speed=solution.rarefaction_head_speed,
        rarefaction_tail_speed=solution.rarefaction_tail_speed,
        radii=edges,
        cell_volume=volume,
        density=density,
        velocity=velocity,
        specific_energy=specific_energy,
        pressure=pressure,
        sound_speed=sound_speed,
        material=material,
        region=region,
        mass=mass,
        node_velocity=node_velocity,
        product_eos=params,
        air_gamma=gamma,
        initial_state=initial_state,
        solution=solution,
        checks={},
        integrals={},
        metadata=probe_metadata,
    )
    checks, integrals = compute_planar_startup_diagnostics(
        field, dense_samples=int(cfg.dense_samples)
    )
    if not checks["passed"]:
        raise PlanarStartupError(f"planar startup checks failed: {checks}")

    # Rebuild with the final checks/integrals: the validated field is frozen,
    # so construct the final object once more with the same arrays.
    final = PlanarStartupField(
        model_id=field.model_id,
        method_version=field.method_version,
        time=field.time,
        pin_time=field.pin_time,
        charge_radius=field.charge_radius,
        epsilon=field.epsilon,
        delta=field.delta,
        contact_radius=field.contact_radius,
        shock_radius=field.shock_radius,
        contact_speed=field.contact_speed,
        shock_speed=field.shock_speed,
        rarefaction_head_radius=field.rarefaction_head_radius,
        rarefaction_tail_radius=field.rarefaction_tail_radius,
        rarefaction_head_speed=field.rarefaction_head_speed,
        rarefaction_tail_speed=field.rarefaction_tail_speed,
        radii=field.radii,
        cell_volume=field.cell_volume,
        density=field.density,
        velocity=field.velocity,
        specific_energy=field.specific_energy,
        pressure=field.pressure,
        sound_speed=field.sound_speed,
        material=field.material,
        region=field.region,
        mass=field.mass,
        node_velocity=field.node_velocity,
        product_eos=field.product_eos,
        air_gamma=field.air_gamma,
        initial_state=field.initial_state,
        solution=field.solution,
        checks=checks,
        integrals=integrals,
        metadata=field.metadata,
    )
    return final


class _PlanarSampler:
    """Small helper used before the frozen field exists.

    It deliberately implements only the analytic sampling; the final
    :class:`PlanarStartupField` owns the validation and diagnostics.
    """

    def __init__(self, values: Mapping[str, Any]) -> None:
        self.__dict__.update(values)

    def _sample_region(self, radius, side):
        return PlanarStartupField._sample_region(self, radius, side)  # type: ignore[arg-type]

    def sample(self, radius, side="right"):
        return PlanarStartupField.sample(self, radius, side)  # type: ignore[arg-type]

    def sample_product(self, radius):
        return PlanarStartupField.sample_product(self, radius)  # type: ignore[arg-type]

    def sample_air(self, radius):
        return PlanarStartupField.sample_air(self, radius)  # type: ignore[arg-type]


__all__ = [
    "PLANAR_STARTUP_METHOD_VERSION",
    "PLANAR_STARTUP_MODEL_ID",
    "REGION_AIR_STAR",
    "REGION_AMBIENT_AIR",
    "REGION_INNER_PRODUCT",
    "REGION_NAMES",
    "REGION_PRODUCT_FAN",
    "REGION_PRODUCT_STAR",
    "PlanarStartupConfig",
    "PlanarStartupError",
    "PlanarStartupField",
    "build_planar_startup_field",
    "compute_planar_startup_diagnostics",
]
