"""Per-case data contract and cache for the ContactNet/AirNet scheme.

A cache stores the **raw** frame times, physical radii, contact interval
``[R_c,low, R_c,up]``, the right-boundary samples and the air observations.
Nothing is shared across cases: each case has its own clock, boundary and
masks.  The cache is deliberately plain ``.npz`` + a JSON metadata string so it
can be audited without the training code.

Real d3plot data is not fabricated here.  ``build_case_from_profile`` consumes
an extracted profile directory (the output of ``cli.extract_profiles``) and
derives the new fields; ``synthetic_case`` is only for engineering tests and is
always labelled ``synthetic``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..data.extracted_case import DataError, load_extracted_case
from .geometry import (
    ContactIntervalTable,
    GeometryError,
    InitialGeometry,
    RightBoundaryTrajectory,
    build_initial_geometry,
)
from .precursor import PrecursorConfig, PrecursorError, PrecursorResult, compute_precursor

DTYPE = "float64"


class CaseDataError(RuntimeError):
    """A case cache is missing, inconsistent or not usable for the requested role."""


def _as_float(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CaseDataError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise CaseDataError(f"{name} must be finite{' and positive' if positive else ''}, got {result!r}")
    return result


def _tensor(values: Any, dtype: str = DTYPE):
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    return torch.as_tensor(np.asarray(values), dtype=getattr(torch, dtype))


@dataclass
class CaseArrays:
    case_id: str
    charge_radius_m: float
    times: np.ndarray
    radius: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    material: np.ndarray
    valid: np.ndarray
    valid_density: np.ndarray | None = None
    valid_velocity: np.ndarray | None = None
    valid_pressure: np.ndarray | None = None
    contact_low: np.ndarray | None = None
    contact_up: np.ndarray | None = None
    contact_raw: np.ndarray | None = None
    contact_valid: np.ndarray | None = None
    rb_times: np.ndarray | None = None
    rb_radius: np.ndarray | None = None
    rb_valid: np.ndarray | None = None
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class AirObservationStore:
    time: np.ndarray
    radius: np.ndarray
    zeta: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    valid_density: np.ndarray
    valid_velocity: np.ndarray
    valid_pressure: np.ndarray
    valid_domain: np.ndarray
    frame_index: np.ndarray

    @property
    def size(self) -> int:
        return int(self.time.size)

    def sample(
        self,
        count: int,
        generator,
        *,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> dict[str, Any]:
        """Draw air points **frame first, then point inside the frame**.

        Sampling rows uniformly would let a frame with many radial samples
        dominate the data loss.  Every eligible frame gets the same chance
        here; the point inside a selected frame is then uniform over that
        frame's eligible observations.
        """
        if self.size == 0 or count <= 0:
            empty = _tensor(np.zeros(0))
            return {
                "t": empty, "zeta": empty, "rho": empty, "u": empty, "p": empty,
                "valid_rho": empty.bool(), "valid_u": empty.bool(),
                "valid_p": empty.bool(), "count": 0,
            }
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        allowed = np.ones(self.size, dtype=bool)
        if t_min is not None:
            allowed &= self.time >= float(t_min)
        if t_max is not None:
            allowed &= self.time <= float(t_max)
        frames = np.unique(self.frame_index[allowed])
        if frames.size == 0:
            empty = _tensor(np.zeros(0))
            return {
                "t": empty, "zeta": empty, "rho": empty, "u": empty, "p": empty,
                "valid_rho": empty.bool(), "valid_u": empty.bool(),
                "valid_p": empty.bool(), "count": 0,
            }
        by_frame = {
            int(frame): np.flatnonzero(allowed & (self.frame_index == frame))
            for frame in frames
        }
        # One generator call per draw keeps the operation reproducible and
        # lets a small number of frames be represented even when another
        # frame has hundreds of radial points.
        chosen_frames = torch.randint(
            0, int(frames.size), (int(count),), generator=generator
        )
        frame_counts = torch.bincount(
            chosen_frames, minlength=int(frames.size)
        ).cpu().numpy()
        selected_parts: list[np.ndarray] = []
        for slot, frame in enumerate(frames):
            n_take = int(frame_counts[slot])
            if n_take <= 0:
                continue
            rows = by_frame[int(frame)]
            picks = torch.randint(
                0, int(rows.size), (n_take,), generator=generator
            ).cpu().numpy()
            selected_parts.append(rows[picks])
        selected = (
            np.concatenate(selected_parts)
            if selected_parts else np.zeros(0, dtype=np.int64)
        )
        return {
            "t": _tensor(self.time[selected]),
            "zeta": _tensor(self.zeta[selected]),
            "rho": _tensor(self.density[selected]),
            "u": _tensor(self.velocity[selected]),
            "p": _tensor(self.pressure[selected]),
            "valid_rho": _tensor(self.valid_density[selected]).bool(),
            "valid_u": _tensor(self.valid_velocity[selected]).bool(),
            "valid_p": _tensor(self.valid_pressure[selected]).bool(),
            "count": int(selected.size),
        }


@dataclass
class CaseData:
    case_id: str
    role: str
    charge_radius_m: float
    precursor: dict[str, Any]
    contact: ContactIntervalTable
    boundary: RightBoundaryTrajectory
    initial: InitialGeometry
    observations: AirObservationStore
    t_end: float
    tolerances: dict[str, float]
    source: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    separation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ("train", "validation", "test"):
            raise CaseDataError(f"role must be train/validation/test, got {self.role!r}")
        if not (self.initial.t0 < self.t_end):
            raise CaseDataError(
                f"case {self.case_id!r}: t_end={self.t_end!r} must be after t0="
                f"{self.initial.t0!r}"
            )
        if self.boundary.domain is None:
            raise CaseDataError(f"case {self.case_id!r}: no continuous R_b trajectory")
        # The outer bounds of R_b are not enough: [t0,t_end] must be inside one
        # connected segment, otherwise a gap in the middle would be sampled.
        try:
            self.boundary.require_interval(self.t0, self.t_end)
        except GeometryError as exc:
            raise CaseDataError(
                f"case {self.case_id!r}: the solve interval [{self.t0!r}, "
                f"{self.t_end!r}] is not a single continuous R_b segment: {exc}"
            ) from exc

    @property
    def t0(self) -> float:
        return float(self.initial.t0)

    @property
    def epsilon(self) -> float:
        return float(self.precursor["epsilon_s"])

    @property
    def rho_scale(self) -> float:
        return float(self.precursor["rho_a_sep"])

    @property
    def u_scale(self) -> float:
        return float(self.precursor["u_a_sep"])

    @property
    def p_scale(self) -> float:
        return float(self.precursor["p_a_sep"])

    @property
    def e_scale(self) -> float:
        return float(self.precursor["e_a_sep"])

    @property
    def length_scale(self) -> float:
        return float(self.precursor["R_sep"])

    @property
    def gamma_air(self) -> float:
        return float(self.precursor["gamma_air"])

    @property
    def ambient_density(self) -> float:
        return float(self.precursor["air_density"])

    @property
    def ambient_pressure(self) -> float:
        return float(self.precursor["air_pressure"])

    @property
    def u_c_sep(self) -> float:
        return float(self.precursor["u_c_sep"])

    @property
    def p_c_at_t0(self) -> float:
        """JWL contact pressure evaluated at ``R_c(t0)`` (not forced to p_a,sep)."""
        jwl = self.precursor["jwl"]
        V = float(self.precursor["V_sep"]) * (
            self.initial.Rc0 / float(self.precursor["R_sep"])
        ) ** 3
        return _jwl_pressure_numpy(V, jwl)

    @property
    def time_scale(self) -> float:
        """``T_star = s_R / s_u`` used by the PDE residual normalisation."""
        return self.length_scale / self.u_scale

    @property
    def energy_density_scale(self) -> float:
        """Unit-volume total energy scale ``E = p/(gamma-1) + rho u^2/2``.

        This is **not** the specific internal energy ``e``: ``[E] = Pa`` while
        ``[e] = J/kg``.  The energy residual is ``dE/dt + ...``, so its scale is
        ``s_E / T_star``, not ``e_a_sep / T_star``.
        """
        return self.p_scale / max(self.gamma_air - 1.0, 1.0e-12) + (
            0.5 * self.rho_scale * self.u_scale ** 2
        )

    def scale_dict(self) -> dict[str, float]:
        return {
            "rho_kg_m3": self.rho_scale,
            "u_m_s": self.u_scale,
            "p_pa": self.p_scale,
            "specific_energy_j_kg": self.e_scale,
            "energy_density_pa": self.energy_density_scale,
            "length_m": self.length_scale,
            "time_s": self.time_scale,
            "mass_residual_kg_m3_s": self.rho_scale / self.time_scale,
            "momentum_residual_kg_m2_s2": (
                self.rho_scale * self.u_scale / self.time_scale
            ),
            "energy_residual_w_m3": self.energy_density_scale / self.time_scale,
            "gamma_air": self.gamma_air,
        }

    def initial_compatibility(self) -> dict[str, Any]:
        """Report the JWL contact pressure at t0 against the air-layer value.

        The two are not forced to agree.  A large relative difference is a
        physical/initial-data issue, not a numerical tolerance to hide.
        """
        p_contact = float(self.p_c_at_t0)
        difference = p_contact - self.p_scale
        relative = abs(difference) / self.p_scale
        return {
            "p_c_at_t0_pa": p_contact,
            "p_a_sep_pa": self.p_scale,
            "absolute_difference_pa": difference,
            "relative_difference": relative,
            "assessment": (
                "non-negligible (>=1% of p_a_sep); reported, not forced to zero"
                if relative >= 1.0e-2
                else "small (<1% of p_a_sep) but still reported"
            ),
            "forced_equal": False,
            "note": (
                "p_c(t0) is the JWL pressure at the ContactNet radius R_c(t0); "
                "p_a_sep is the matched air pressure at separation. They are "
                "different states at slightly different positions/times."
            ),
        }

    def prediction_view(self) -> "PredictionCaseData":
        return PredictionCaseData.from_case(self)

    def boundary_at(self, t) -> tuple[Any, Any]:
        values = np.asarray(t, dtype=np.float64)
        return self.boundary.position(values), self.boundary.derivative(values)

    def boundary_tensors(self, t) -> tuple[Any, Any]:
        values = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        rb, rb_dot = self.boundary_at(values)
        return _tensor(rb), _tensor(rb_dot)

    def subset_frames(self, frames: Sequence[int], *, role: str | None = None) -> "CaseData":
        """Return a copy whose data and contact labels use only ``frames``.

        The boundary, precursor and initial geometry stay the same; this is a
        frame-level train/validation split, not a new case.
        """
        indices = np.asarray(sorted({int(f) for f in frames}), dtype=np.int64)
        if indices.size == 0:
            raise CaseDataError(f"case {self.case_id!r}: empty frame subset")
        n = self.contact.times.size
        if np.any(indices < 0) or np.any(indices >= n):
            raise CaseDataError(f"case {self.case_id!r}: frame subset is out of range")
        obs_mask = np.isin(self.observations.frame_index, indices)
        observations = AirObservationStore(
            time=self.observations.time[obs_mask],
            radius=self.observations.radius[obs_mask],
            zeta=self.observations.zeta[obs_mask],
            density=self.observations.density[obs_mask],
            velocity=self.observations.velocity[obs_mask],
            pressure=self.observations.pressure[obs_mask],
            valid_density=self.observations.valid_density[obs_mask],
            valid_velocity=self.observations.valid_velocity[obs_mask],
            valid_pressure=self.observations.valid_pressure[obs_mask],
            valid_domain=self.observations.valid_domain[obs_mask],
            frame_index=self.observations.frame_index[obs_mask],
        )
        contact_mask = np.zeros(n, dtype=bool)
        contact_mask[indices] = True
        contact_mask &= self.contact.valid
        contact = ContactIntervalTable(
            case_id=self.contact.case_id,
            times=self.contact.times[contact_mask],
            low=self.contact.low[contact_mask], up=self.contact.up[contact_mask],
            raw=self.contact.raw[contact_mask], valid=self.contact.valid[contact_mask],
        )
        selected_times = self.contact.times[indices]
        t_end = max(float(selected_times.max()), float(observations.time.max()) if observations.size else -np.inf)
        if not (self.t0 < t_end):
            raise CaseDataError(
                f"case {self.case_id!r}: selected frames do not extend after t0"
            )
        try:
            self.boundary.require_interval(self.t0, t_end)
        except GeometryError as exc:
            raise CaseDataError(
                f"case {self.case_id!r}: selected frames span an R_b gap: {exc}"
            ) from exc
        return CaseData(
            case_id=self.case_id, role=str(role or self.role),
            charge_radius_m=self.charge_radius_m, precursor=dict(self.precursor),
            contact=contact, boundary=self.boundary, initial=self.initial,
            observations=observations, t_end=t_end, tolerances=dict(self.tolerances),
            source=dict(self.source), notes=list(self.notes) + [
                f"frame subset {indices.tolist()}"
            ], separation=dict(self.separation),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "charge_radius_m": self.charge_radius_m,
            "t0": self.t0,
            "t_end": self.t_end,
            "initial": {
                "Rc0": self.initial.Rc0,
                "Rs0": self.initial.Rs0,
                "Rb0": self.initial.Rb0,
                "zeta_s0": self.initial.zeta_s0,
            },
            "contact": self.contact.as_dict(),
            "boundary": self.boundary.as_dict(),
            "observations": self.observations.size,
            "scales": self.scale_dict(),
            "initial_compatibility": self.initial_compatibility(),
            "p_c_at_t0_pa": self.p_c_at_t0,
            "p_a_sep_pa": self.p_scale,
            "p_c_vs_p_a_sep_relative": abs(self.p_c_at_t0 - self.p_scale) / self.p_scale,
            "tolerances": dict(self.tolerances),
            "source": dict(self.source),
            "notes": list(self.notes),
            "separation": dict(self.separation),
        }


@dataclass
class PredictionCaseData:
    """The **boundary-only** view used by ``predict-500``.

    It deliberately has no observations, no contact interval and no ``R_raw``.
    Its time range is the continuous ``R_b`` segment containing ``t0``, so it
    cannot depend on whether internal supervision happened to be valid.
    """

    case_id: str
    charge_radius_m: float
    precursor: dict[str, Any]
    boundary: RightBoundaryTrajectory
    initial: InitialGeometry
    t_end: float
    role: str = "test"
    tolerances: dict[str, float] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (self.initial.t0 < self.t_end):
            raise CaseDataError(
                f"case {self.case_id!r}: prediction t_end={self.t_end!r} must be "
                f"after t0={self.initial.t0!r}"
            )
        try:
            self.boundary.require_interval(self.initial.t0, self.t_end)
        except GeometryError as exc:
            raise CaseDataError(
                f"case {self.case_id!r}: prediction interval is not one continuous "
                f"R_b segment: {exc}"
            ) from exc

    @property
    def t0(self) -> float:
        return float(self.initial.t0)

    @property
    def epsilon(self) -> float:
        return float(self.precursor["epsilon_s"])

    @property
    def rho_scale(self) -> float:
        return float(self.precursor["rho_a_sep"])

    @property
    def u_scale(self) -> float:
        return float(self.precursor["u_a_sep"])

    @property
    def p_scale(self) -> float:
        return float(self.precursor["p_a_sep"])

    @property
    def length_scale(self) -> float:
        return float(self.precursor["R_sep"])

    @property
    def time_scale(self) -> float:
        return self.length_scale / self.u_scale

    @property
    def gamma_air(self) -> float:
        return float(self.precursor["gamma_air"])

    def boundary_at(self, t) -> tuple[Any, Any]:
        values = np.asarray(t, dtype=np.float64)
        return self.boundary.position(values), self.boundary.derivative(values)

    def boundary_tensors(self, t) -> tuple[Any, Any]:
        values = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        rb, rb_dot = self.boundary_at(values)
        return _tensor(rb), _tensor(rb_dot)

    @classmethod
    def from_case(cls, case: CaseData) -> "PredictionCaseData":
        span = case.boundary.continuous_interval_containing(case.t0)
        return cls(
            case_id=case.case_id, charge_radius_m=case.charge_radius_m,
            precursor=dict(case.precursor), boundary=case.boundary,
            initial=case.initial, t_end=float(span[1]), role=case.role,
            tolerances=dict(case.tolerances), source=dict(case.source),
            notes=list(case.notes) + [
                "boundary-only prediction view; internal observations and contact "
                "intervals are not exposed"
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "charge_radius_m": self.charge_radius_m,
            "t0": self.t0,
            "t_end": self.t_end,
            "initial": {
                "Rc0": self.initial.Rc0,
                "Rs0": self.initial.Rs0,
                "Rb0": self.initial.Rb0,
                "zeta_s0": self.initial.zeta_s0,
            },
            "boundary": self.boundary.as_dict(),
            "source": dict(self.source),
            "notes": list(self.notes),
            "prediction_fields": ["R_b(t)", "dR_b/dt", "precursor", "initial", "time range"],
        }


def _jwl_pressure_numpy(V: float, jwl: Mapping[str, Any]) -> float:
    return (
        float(jwl["A"]) * math.exp(-float(jwl["R1"]) * V)
        + float(jwl["B"]) * math.exp(-float(jwl["R2"]) * V)
        + float(jwl["C"]) / V ** (1.0 + float(jwl["omega"]))
    )


def _field_masks(arrays: CaseArrays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = arrays.density.shape
    valid = np.asarray(arrays.valid, dtype=bool)
    if arrays.valid_density is None:
        vd = valid
    else:
        vd = np.asarray(arrays.valid_density, dtype=bool)
    if arrays.valid_velocity is None:
        vu = valid
    else:
        vu = np.asarray(arrays.valid_velocity, dtype=bool)
    if arrays.valid_pressure is None:
        vp = valid
    else:
        vp = np.asarray(arrays.valid_pressure, dtype=bool)
    for name, mask in (("density", vd), ("velocity", vu), ("pressure", vp)):
        if mask.shape != shape:
            raise CaseDataError(f"valid_{name} has shape {mask.shape}, expected {shape}")
    return vd, vu, vp


@dataclass(frozen=True)
class RightBoundaryDetection:
    radius: float
    identified: bool
    evidence: str
    reasons: tuple[str, ...]
    n_ambient_cells: int
    n_disturbed_air_cells: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "radius_m": self.radius if math.isfinite(self.radius) else None,
            "identified": self.identified,
            "evidence": self.evidence,
            "reasons": list(self.reasons),
            "n_ambient_cells": self.n_ambient_cells,
            "n_disturbed_air_cells": self.n_disturbed_air_cells,
        }


def _trailing_true_run(mask: np.ndarray, *, min_cells: int) -> int:
    index = int(mask.size) - 1
    while index >= 0 and bool(mask[index]):
        index -= 1
    start = index + 1
    return start if int(mask.size) - start >= int(min_cells) else -1


def detect_right_boundary_report(
    radius: np.ndarray,
    pressure: np.ndarray,
    velocity: np.ndarray,
    density: np.ndarray,
    material: np.ndarray,
    *,
    ambient_pressure: float,
    ambient_density: float,
    tol_pressure: float,
    tol_density: float,
    tol_velocity: float,
    min_ambient_cells: int = 2,
    valid_density: np.ndarray | None = None,
    valid_velocity: np.ndarray | None = None,
    valid_pressure: np.ndarray | None = None,
    contact_radius: float | None = None,
) -> RightBoundaryDetection:
    """Locate the environmental-side right boundary with radial evidence.

    A point is not a boundary merely because it looks ambient.  The candidate
    must

    * be part of a continuous outward run ending at the outer radial edge;
    * have a radially connected shocked-air region inside it;
    * not sit inside a missing/invalid air gap.

    Two evidence tiers are separated:

    ``ambient_state``
        finite ``(rho,u,p)`` inside the ambient tolerances.
    ``unlabelled_ambient_region``
        contiguous ``material==2`` shells at the outer edge, preceded by
        shocked air.  The extractor leaves this region unlabelled and often
        without fields, so this is weaker evidence and is reported as such.
    """
    r = np.asarray(radius, dtype=np.float64).reshape(-1)
    p = np.asarray(pressure, dtype=np.float64).reshape(-1)
    u = np.asarray(velocity, dtype=np.float64).reshape(-1)
    rho = np.asarray(density, dtype=np.float64).reshape(-1)
    mat = np.asarray(material).reshape(-1)
    if not (r.size == p.size == u.size == rho.size == mat.size):
        raise CaseDataError("right-boundary inputs must have the same length")
    if r.size < 2 or np.any(np.diff(r) <= 0.0):
        raise CaseDataError("the radial axis for R_b detection must be strictly increasing")
    for name, mask in (
        ("valid_density", valid_density), ("valid_velocity", valid_velocity),
        ("valid_pressure", valid_pressure),
    ):
        if mask is not None and np.asarray(mask).reshape(-1).size != r.size:
            raise CaseDataError(f"{name} has a different length from the radial axis")
    state_ok = np.isfinite(p) & np.isfinite(u) & np.isfinite(rho)
    if valid_density is not None:
        state_ok &= np.asarray(valid_density, dtype=bool).reshape(-1)
    if valid_velocity is not None:
        state_ok &= np.asarray(valid_velocity, dtype=bool).reshape(-1)
    if valid_pressure is not None:
        state_ok &= np.asarray(valid_pressure, dtype=bool).reshape(-1)
    ambient = (
        state_ok
        & np.isin(mat, (1, 2))
        & (np.abs(p - float(ambient_pressure)) <= float(tol_pressure))
        & (np.abs(rho - float(ambient_density)) <= float(tol_density))
        & (np.abs(u) <= float(tol_velocity))
    )
    disturbed = (mat == 1) & state_ok & ~ambient
    if contact_radius is not None and math.isfinite(float(contact_radius)):
        disturbed &= r > float(contact_radius)
    n_disturbed = int(np.count_nonzero(disturbed))
    reasons: list[str] = []
    start = _trailing_true_run(ambient, min_cells=min_ambient_cells)
    evidence = "finite ambient state in a continuous run to the outer radial edge"
    if start < 0:
        # Weaker but auditable: the extractor's unlabelled ambient region.
        start = _trailing_true_run(mat == 2, min_cells=min_ambient_cells)
        evidence = (
            "material==2 unlabelled ambient region at the outer radial edge; "
            "ambient state is not available for these shells"
        )
        if start < 0:
            reasons.append(
                "ambient cells do not form a continuous run to the outer radial "
                "edge (isolated or interrupted run); neither a finite ambient "
                "state nor an unlabelled ambient region is identifiable"
            )
            return RightBoundaryDetection(
                float("nan"), False, "", tuple(reasons), 0, n_disturbed
            )
    n_ambient = int(r.size - start)
    inside = np.arange(r.size) < start
    disturbed_inside = np.flatnonzero(disturbed & inside)
    if disturbed_inside.size == 0:
        reasons.append(
            "no disturbed air cell lies inside the ambient run; an all-ambient "
            "profile is not evidence of a shock outer edge"
        )
        return RightBoundaryDetection(
            float("nan"), False, "", tuple(reasons), n_ambient, n_disturbed
        )
    last_disturbed = int(disturbed_inside[-1])
    between = np.arange(last_disturbed + 1, start)
    if between.size:
        bad_label = between[~np.isin(mat[between], (1, 2))]
        if bad_label.size:
            reasons.append(
                "product/mixed cells interrupt the radial path from shocked air "
                f"to the ambient run at r={r[bad_label[:3]].tolist()} m"
            )
        bad_air = between[(mat[between] == 1) & ~state_ok[between]]
        if bad_air.size:
            reasons.append(
                "missing or invalid air cells break radial connectivity at "
                f"r={r[bad_air[:3]].tolist()} m"
            )
    if reasons:
        return RightBoundaryDetection(
            float("nan"), False, "", tuple(reasons), n_ambient, n_disturbed
        )
    candidate = float(r[start])
    if contact_radius is not None and math.isfinite(float(contact_radius)):
        if candidate <= float(contact_radius):
            reasons.append("the ambient run is not outside the contact interval")
            return RightBoundaryDetection(
                float("nan"), False, "", tuple(reasons), n_ambient, n_disturbed
            )
    return RightBoundaryDetection(
        candidate, True, evidence, tuple(reasons), n_ambient, n_disturbed
    )


def detect_right_boundary(
    radius: np.ndarray,
    pressure: np.ndarray,
    velocity: np.ndarray,
    density: np.ndarray,
    material: np.ndarray,
    *,
    ambient_pressure: float,
    ambient_density: float,
    tol_pressure: float,
    tol_density: float,
    tol_velocity: float,
    min_ambient_cells: int = 2,
    valid_density: np.ndarray | None = None,
    valid_velocity: np.ndarray | None = None,
    valid_pressure: np.ndarray | None = None,
    contact_radius: float | None = None,
) -> tuple[float, bool, str]:
    """Backward-compatible tuple wrapper around :func:`detect_right_boundary_report`."""
    report = detect_right_boundary_report(
        radius, pressure, velocity, density, material,
        ambient_pressure=ambient_pressure, ambient_density=ambient_density,
        tol_pressure=tol_pressure, tol_density=tol_density,
        tol_velocity=tol_velocity, min_ambient_cells=min_ambient_cells,
        valid_density=valid_density, valid_velocity=valid_velocity,
        valid_pressure=valid_pressure, contact_radius=contact_radius,
    )
    reason = report.evidence if report.identified else "; ".join(report.reasons)
    return report.radius, report.identified, reason


def contact_interval_from_material(
    radius: np.ndarray,
    material: np.ndarray,
    *,
    mixed: np.ndarray | None = None,
) -> tuple[float, float, float, bool, str]:
    """Derive ``(low, up, raw, valid, reason)`` from one frame's shell labels.

    ``radius`` is the shell representative coordinate produced by
    ``extract_profiles`` (a geometric shell midpoint), not a raw ALE cell
    centroid.  ``low``/``up`` are therefore shell representatives as well.

    ``material==2`` is ambiguous in the real extractor: it is used both for
    the mixing band and for every shell outside the detected shock.  Callers
    that have the volume-fraction profile must pass an explicit ``mixed``
    mask; otherwise this function keeps the legacy ``material==2`` fallback
    for synthetic/fixture data only.
    """
    r = np.asarray(radius, dtype=np.float64)
    mat = np.asarray(material)
    mixed = mat == 2 if mixed is None else np.asarray(mixed, dtype=bool)
    if mixed.shape != r.shape:
        raise CaseDataError("the mixed mask has a different shape from the radial axis")
    if not np.any(mixed):
        return float("nan"), float("nan"), float("nan"), False, "no mixed cell"
    idx = np.flatnonzero(mixed)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    if breaks.size:
        return float("nan"), float("nan"), float("nan"), False, (
            "multiple disconnected mixed segments; contact ownership is ambiguous"
        )
    first, last = int(idx[0]), int(idx[-1])
    inner = np.flatnonzero((mat == 0) & (np.arange(r.size) < first))
    outer = np.flatnonzero((mat == 1) & (np.arange(r.size) > last))
    if inner.size == 0 or outer.size == 0:
        return float("nan"), float("nan"), float("nan"), False, (
            "mixed segment is not bracketed by a pure product and a pure air cell"
        )
    low = float(r[inner[-1]])
    up = float(r[outer[0]])
    raw = 0.5 * (float(r[first]) + float(r[last]))
    if not (low < up):
        return float("nan"), float("nan"), float("nan"), False, "R_c,low is not below R_c,up"
    if not (low <= raw <= up):
        raw = 0.5 * (low + up)
    return low, up, raw, True, "single mixed segment bracketed by pure phases"


def contact_interval_from_band(
    radius: np.ndarray,
    material: np.ndarray,
    band_inner: float,
    band_outer: float,
) -> tuple[float, float, float, bool, str]:
    """Use the extractor's mixing band and its adjacent pure shells.

    ``band_inner``/``band_outer`` are the mixing-band radii reconstructed by
    ``extract_profiles``.  The shell centre just inside the inner edge is the
    pure-product representative; the first centre at/outside the outer edge is
    the pure-air representative.  This avoids treating the extractor's
    ``material==2`` outside-shock region as part of the contact band.
    """
    r = np.asarray(radius, dtype=np.float64).reshape(-1)
    mat = np.asarray(material).reshape(-1)
    inner = float(band_inner)
    outer = float(band_outer)
    if not (math.isfinite(inner) and math.isfinite(outer) and inner < outer):
        return float("nan"), float("nan"), float("nan"), False, "mixing band is not finite and ordered"
    inner_index = int(np.searchsorted(r, inner, side="right") - 1)
    while inner_index >= 0 and mat[inner_index] != 0:
        inner_index -= 1
    outer_index = int(np.searchsorted(r, outer, side="left"))
    while outer_index < r.size and mat[outer_index] != 1:
        outer_index += 1
    if inner_index < 0 or outer_index >= r.size:
        return float("nan"), float("nan"), float("nan"), False, "mixing band extends outside the radial grid"
    if not (r[inner_index] < r[outer_index]):
        return float("nan"), float("nan"), float("nan"), False, "mixing band contains no shell representatives"
    if mat[inner_index] != 0 or mat[outer_index] != 1:
        return float("nan"), float("nan"), float("nan"), False, (
            "shells adjacent to the mixing band are not pure product/air"
        )
    low = float(r[inner_index])
    up = float(r[outer_index])
    raw = 0.5 * (inner + outer)
    if not (low <= raw <= up):
        raw = 0.5 * (low + up)
    return low, up, raw, True, "mixing band bracketed by adjacent pure shell representatives"


def build_air_observations(
    arrays: CaseArrays,
    contact: ContactIntervalTable,
    boundary: RightBoundaryTrajectory,
) -> AirObservationStore:
    vd, vu, vp = _field_masks(arrays)
    times = np.asarray(arrays.times, dtype=np.float64).reshape(-1)
    radius = np.asarray(arrays.radius, dtype=np.float64).reshape(-1)
    mat = np.asarray(arrays.material)
    obs_t, obs_r, obs_z = [], [], []
    obs_rho, obs_u, obs_p = [], [], []
    obs_vd, obs_vu, obs_vp, obs_dom, obs_frame = [], [], [], [], []
    for frame, t in enumerate(times):
        if not contact.valid[frame]:
            continue
        try:
            rb = float(boundary.position(float(t)))
        except GeometryError:
            continue
        raw = float(contact.raw[frame])
        if not (rb > raw):
            continue
        air = mat[frame] == 1
        if not np.any(air):
            continue
        indices = np.flatnonzero(air)
        zeta = (radius[indices] - raw) / (rb - raw)
        dom = (zeta >= 0.0) & (zeta <= 1.0)
        if not np.any(dom):
            continue
        indices = indices[dom]
        zeta = zeta[dom]
        obs_t.append(np.full(indices.size, t, dtype=np.float64))
        obs_r.append(radius[indices])
        obs_z.append(zeta)
        obs_rho.append(np.asarray(arrays.density[frame, indices], dtype=np.float64))
        obs_u.append(np.asarray(arrays.velocity[frame, indices], dtype=np.float64))
        obs_p.append(np.asarray(arrays.pressure[frame, indices], dtype=np.float64))
        obs_vd.append(vd[frame, indices])
        obs_vu.append(vu[frame, indices])
        obs_vp.append(vp[frame, indices])
        obs_dom.append(np.ones(indices.size, dtype=bool))
        obs_frame.append(np.full(indices.size, frame, dtype=np.int64))
    if not obs_t:
        empty = np.zeros(0, dtype=np.float64)
        return AirObservationStore(
            empty, empty, empty, empty, empty, empty,
            np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, bool),
            np.zeros(0, bool), np.zeros(0, np.int64),
        )
    return AirObservationStore(
        time=np.concatenate(obs_t), radius=np.concatenate(obs_r),
        zeta=np.concatenate(obs_z), density=np.concatenate(obs_rho),
        velocity=np.concatenate(obs_u), pressure=np.concatenate(obs_p),
        valid_density=np.concatenate(obs_vd), valid_velocity=np.concatenate(obs_vu),
        valid_pressure=np.concatenate(obs_vp), valid_domain=np.concatenate(obs_dom),
        frame_index=np.concatenate(obs_frame),
    )


def classify_separation_frames(
    *,
    case_id: str,
    times: np.ndarray,
    radius: np.ndarray,
    material: np.ndarray,
    density: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray,
    contact: ContactIntervalTable,
    boundary: RightBoundaryTrajectory,
    boundary_reports: Sequence[RightBoundaryDetection],
    valid_density: np.ndarray,
    valid_velocity: np.ndarray,
    valid_pressure: np.ndarray,
    min_shocked_air_cells: int = 2,
) -> dict[str, Any]:
    """Classify each frame as separated / not separated / unresolvable.

    The three questions are recorded separately and never collapsed into one
    flag:

    1. is the contact interval locatable?
    2. is the shocked air layer between the contact and ``R_b`` resolvable?
    3. is the environmental-side right boundary identifiable?

    The frame times are the extraction's own times.  No theoretical ``t_sep``
    is used to shift or invent an event alignment.
    """
    r = np.asarray(radius, dtype=np.float64).reshape(-1)
    mat = np.asarray(material)
    rho = np.asarray(density, dtype=np.float64)
    u = np.asarray(velocity, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    vd = np.asarray(valid_density, dtype=bool)
    vu = np.asarray(valid_velocity, dtype=bool)
    vp = np.asarray(valid_pressure, dtype=bool)
    if r.size < 2:
        raise CaseDataError(f"case {case_id!r}: radial axis is too short for separation")
    shell_width = float(np.median(np.diff(r)))
    frames: list[dict[str, Any]] = []
    counts = {"separated": 0, "not_separated": 0, "unresolvable": 0}
    for frame, t in enumerate(np.asarray(times, dtype=np.float64)):
        reasons: list[str] = []
        report = boundary_reports[frame] if frame < len(boundary_reports) else None
        contact_ok = bool(contact.valid[frame])
        rb_ok = bool(report.identified) if report is not None else False
        rb = float(report.radius) if rb_ok else float("nan")
        if not contact_ok:
            reasons.append("contact interval is not locatable in this frame")
        if not rb_ok:
            detail = "; ".join(report.reasons) if report is not None else "no report"
            reasons.append(f"environmental-side right boundary is not identifiable ({detail})")
        n_air = 0
        n_air_valid = 0
        connected = False
        layer_ok = False
        contact_up = float(contact.up[frame]) if contact_ok else float("nan")
        if contact_ok and rb_ok and rb > contact_up:
            air_mask = (
                (mat[frame] == 1) & (r >= float(contact.low[frame])) & (r < rb)
            )
            air_indices = np.flatnonzero(air_mask)
            n_air = int(air_indices.size)
            if air_indices.size:
                connected = bool(np.all(np.diff(air_indices) == 1))
                fields_ok = (
                    vd[frame, air_indices] & vu[frame, air_indices]
                    & vp[frame, air_indices]
                    & np.isfinite(rho[frame, air_indices])
                    & np.isfinite(u[frame, air_indices])
                    & np.isfinite(p[frame, air_indices])
                )
                n_air_valid = int(np.count_nonzero(fields_ok))
                if not connected:
                    reasons.append(
                        "pure-air cells between the contact and R_b are not "
                        "radially contiguous"
                    )
                if n_air_valid != n_air:
                    reasons.append(
                        f"{n_air - n_air_valid} air cell(s) in the layer have "
                        "missing/invalid fields"
                    )
                if n_air_valid < int(min_shocked_air_cells):
                    reasons.append(
                        f"only {n_air_valid} valid air cell(s); at least "
                        f"{int(min_shocked_air_cells)} are required to resolve the layer"
                    )
                layer_ok = (
                    connected and n_air_valid >= int(min_shocked_air_cells)
                    and n_air_valid == n_air
                )
            else:
                # No air cell at all between the contact bracket and R_b.  This
                # is positive evidence for "not separated" only when the
                # contact bracket actually reaches the boundary; a large
                # unexplained gap would be unresolvable instead.
                proximity = contact_up >= rb - 1.5 * shell_width
                if proximity and contact_ok and rb_ok:
                    reasons.append(
                        "no pure-air cell between the contact upper bound and the "
                        "environmental boundary; the contact interval reaches R_b"
                    )
                    status = "not_separated"
                    counts[status] += 1
                    frames.append({
                        "time_s": float(t),
                        "status": status,
                        "contact_locatable": contact_ok,
                        "shocked_air_resolvable": False,
                        "right_boundary_identified": rb_ok,
                        "n_air_cells": 0,
                        "n_valid_air_cells": 0,
                        "radially_connected": False,
                        "boundary_evidence": report.evidence if report else "",
                        "reasons": reasons,
                    })
                    continue
                reasons.append(
                    "no pure-air cell between the contact bracket and R_b, but "
                    "the bracket is not adjacent to R_b; layer state is unknown"
                )
        if contact_ok and rb_ok and layer_ok and rb > contact_up:
            status = "separated"
            evidence = (
                "contact interval locatable; a connected pure-air layer with "
                "finite fields lies between the contact and the identified "
                "environmental boundary"
            )
        elif contact_ok and rb_ok:
            status = "unresolvable"
            evidence = "contact and boundary exist but the air layer is not resolvable"
        else:
            status = "unresolvable"
            evidence = "one or more of contact/boundary is missing"
        counts[status] += 1
        frames.append({
            "time_s": float(t),
            "status": status,
            "contact_locatable": contact_ok,
            "shocked_air_resolvable": bool(layer_ok),
            "right_boundary_identified": rb_ok,
            "n_air_cells": int(n_air),
            "n_valid_air_cells": int(n_air_valid),
            "radially_connected": bool(connected),
            "boundary_evidence": report.evidence if report else "",
            "reasons": reasons if status != "separated" else [evidence],
        })
    if counts["separated"] and not counts["not_separated"] and not counts["unresolvable"]:
        status = "separated"
    elif counts["not_separated"] and not counts["separated"] and not counts["unresolvable"]:
        status = "not_separated"
    elif counts["unresolvable"] == len(frames):
        status = "unresolvable"
    else:
        # Per-frame status is always one of the three; the case-level label is
        # "mixed" when those three do not all agree.
        status = "mixed"
    separated_times = [item["time_s"] for item in frames if item["status"] == "separated"]
    not_separated_times = [
        item["time_s"] for item in frames if item["status"] == "not_separated"
    ]
    return {
        "status": status,
        "status_counts": counts,
        "n_frames": len(frames),
        "contact_locatable_count": int(np.count_nonzero(contact.valid)),
        "shocked_air_resolvable_count": int(
            np.count_nonzero([item["shocked_air_resolvable"] for item in frames])
        ),
        "right_boundary_identified_count": int(
            np.count_nonzero([item["right_boundary_identified"] for item in frames])
        ),
        "first_separated_time_s": min(separated_times) if separated_times else None,
        "first_not_separated_time_s": (
            min(not_separated_times) if not_separated_times else None
        ),
        "frame_times": "raw extraction frame times; no theoretical t_sep shift",
        "uses_theoretical_tsep_for_alignment": False,
        "basis": (
            "locatable contact interval + connected pure-air layer with finite "
            "fields + identified environmental boundary; each condition is "
            "recorded separately"
        ),
        "frames": frames,
    }


def build_case(
    arrays: CaseArrays,
    *,
    precursor: PrecursorResult,
    role: str,
    tolerances: Mapping[str, float],
    t_end: float | None = None,
    source: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
) -> CaseData:
    times = np.asarray(arrays.times, dtype=np.float64).reshape(-1)
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise CaseDataError("frame times must be finite and strictly increasing")
    if arrays.contact_low is None:
        low, up, raw, cvalid = [], [], [], []
        reasons: list[str] = []
        for frame in range(times.size):
            lo, hi, rr, ok, reason = contact_interval_from_material(
                arrays.radius, arrays.material[frame]
            )
            low.append(lo); up.append(hi); raw.append(rr); cvalid.append(ok)
            reasons.append(reason)
        contact = ContactIntervalTable(
            case_id=arrays.case_id, times=times,
            low=np.asarray(low), up=np.asarray(up), raw=np.asarray(raw),
            valid=np.asarray(cvalid, bool),
        )
        notes = list(notes or []) + [
            "contact intervals derived from shell material labels",
        ]
    else:
        contact = ContactIntervalTable(
            case_id=arrays.case_id, times=times,
            low=np.asarray(arrays.contact_low, dtype=np.float64),
            up=np.asarray(arrays.contact_up, dtype=np.float64),
            raw=np.asarray(arrays.contact_raw, dtype=np.float64),
            valid=np.asarray(arrays.contact_valid, dtype=bool),
        )
        notes = list(notes or [])

    valid_density, valid_velocity, valid_pressure = _field_masks(arrays)
    boundary_reports: list[RightBoundaryDetection] = []
    if arrays.rb_times is None:
        rb_t, rb_r, rb_v, reasons = [], [], [], []
        for frame, t in enumerate(times):
            contact_up = (
                float(contact.up[frame]) if bool(contact.valid[frame]) else None
            )
            report = detect_right_boundary_report(
                arrays.radius, arrays.pressure[frame], arrays.velocity[frame],
                arrays.density[frame], arrays.material[frame],
                ambient_pressure=float(tolerances["ambient_pressure_pa"]),
                ambient_density=float(tolerances["ambient_density_kg_m3"]),
                tol_pressure=float(tolerances["pressure_pa"]),
                tol_density=float(tolerances["density_kg_m3"]),
                tol_velocity=float(tolerances["velocity_m_s"]),
                min_ambient_cells=int(tolerances.get("min_ambient_cells", 2)),
                valid_density=valid_density[frame],
                valid_velocity=valid_velocity[frame],
                valid_pressure=valid_pressure[frame],
                contact_radius=contact_up,
            )
            boundary_reports.append(report)
            rb_t.append(t); rb_r.append(report.radius); rb_v.append(report.identified)
            reasons.append(report.evidence if report.identified else "; ".join(report.reasons))
        boundary = RightBoundaryTrajectory.from_arrays(
            case_id=arrays.case_id, times=rb_t, radii=rb_r, valid=rb_v
        )
        evidence_set = sorted({r for r in reasons if r})
        notes.append(f"right boundary evidence: {evidence_set}")
    else:
        boundary = RightBoundaryTrajectory.from_arrays(
            case_id=arrays.case_id, times=arrays.rb_times,
            radii=arrays.rb_radius, valid=arrays.rb_valid,
        )
        boundary_reports = []
        for t in times:
            if boundary.has_slope(float(t)):
                boundary_reports.append(RightBoundaryDetection(
                    radius=float(boundary.position(float(t))), identified=True,
                    evidence="supplied R_b samples (continuous segment)",
                    reasons=(), n_ambient_cells=0, n_disturbed_air_cells=0,
                ))
            else:
                boundary_reports.append(RightBoundaryDetection(
                    radius=float("nan"), identified=False,
                    evidence="",
                    reasons=("no continuous supplied R_b segment at this frame",),
                    n_ambient_cells=0, n_disturbed_air_cells=0,
                ))
    initial = build_initial_geometry(
        t0=precursor.t0, Rc0=precursor.Rc0, Rs0=precursor.Rs0,
        R_sep=precursor.R_sep, u_c_sep=precursor.u_c_sep, D_s_sep=precursor.D_s_sep,
        boundary=boundary,
    )
    obs = build_air_observations(arrays, contact, boundary)
    separation = classify_separation_frames(
        case_id=arrays.case_id, times=times, radius=arrays.radius,
        material=arrays.material, density=arrays.density,
        velocity=arrays.velocity, pressure=arrays.pressure,
        contact=contact, boundary=boundary, boundary_reports=boundary_reports,
        valid_density=valid_density, valid_velocity=valid_velocity,
        valid_pressure=valid_pressure,
        min_shocked_air_cells=int(tolerances.get("min_shocked_air_cells", 2)),
    )
    separation["t0_s"] = float(precursor.t0)
    separation["theoretical_t_sep_s"] = float(precursor.t_sep)
    # The solve interval starts at t0 and is limited to the one measured R_b
    # segment containing it.  It is never the [earliest, latest] outer bounds,
    # which could enclose a gap.
    segment = boundary.continuous_interval_containing(float(precursor.t0))
    end = float(segment[1] if t_end is None else t_end)
    if t_end is not None:
        boundary.require_interval(float(precursor.t0), end)
    if not (precursor.t0 < end):
        raise CaseDataError(
            f"case {arrays.case_id!r}: no training interval after t0; "
            f"t0={precursor.t0!r}, t_end={end!r}"
        )
    return CaseData(
        case_id=str(arrays.case_id), role=str(role),
        charge_radius_m=float(arrays.charge_radius_m),
        precursor=precursor.as_dict(), contact=contact, boundary=boundary,
        initial=initial, observations=obs, t_end=end,
        tolerances=dict(tolerances), source=dict(source or {}), notes=notes,
        separation=separation,
    )


def save_case(case: CaseData, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "spherical_blast_pinn.contact_air_case",
        "version": 2,
        "units": {
            "time": "s", "radius": "m", "density": "kg/m^3",
            "velocity": "m/s", "pressure": "Pa", "specific_energy": "J/kg",
            "energy_density": "J/m^3 (Pa)",
        },
        "case": case.as_dict(),
        "precursor": case.precursor,
        "prediction_requires_only": [
            "metadata", "rb_times", "rb_radius", "rb_valid",
        ],
        "cache_rebuild_reason": (
            "version 2: R_b gaps are break markers, separation status is per-frame "
            "evidence, the solve interval is one continuous R_b segment, and the "
            "energy residual scale is E/T_star rather than e/T_star"
        ),
    }
    np.savez(
        path,
        metadata=np.asarray(json.dumps(metadata)),
        obs_time=case.observations.time,
        obs_radius=case.observations.radius,
        obs_zeta=case.observations.zeta,
        obs_density=case.observations.density,
        obs_velocity=case.observations.velocity,
        obs_pressure=case.observations.pressure,
        valid_density=case.observations.valid_density,
        valid_velocity=case.observations.valid_velocity,
        valid_pressure=case.observations.valid_pressure,
        valid_domain=case.observations.valid_domain,
        obs_frame=case.observations.frame_index,
        contact_times=case.contact.times,
        contact_low=case.contact.low,
        contact_up=case.contact.up,
        contact_raw=case.contact.raw,
        contact_valid=case.contact.valid,
        rb_times=case.boundary.times,
        rb_radius=case.boundary.radii,
        rb_valid=case.boundary.valid,
    )
    return path


def load_case(path: Path | str, *, role: str | None = None) -> CaseData:
    path = Path(path)
    if not path.is_file():
        raise CaseDataError(f"case cache {path} does not exist")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("format") != "spherical_blast_pinn.contact_air_case":
            raise CaseDataError(f"{path}: not a contact-air case cache")
        if metadata.get("version") != 2:
            raise CaseDataError(
                f"{path}: cache version {metadata.get('version')!r} is not supported. "
                "Rebuild it with build-cache: version 1 predates the R_b gap, "
                "separation-evidence and energy-scale fixes"
            )
        case_meta = metadata["case"]
        precursor = metadata["precursor"]
        contact = ContactIntervalTable(
            case_id=case_meta["case_id"], times=np.asarray(archive["contact_times"]),
            low=np.asarray(archive["contact_low"]), up=np.asarray(archive["contact_up"]),
            raw=np.asarray(archive["contact_raw"]), valid=np.asarray(archive["contact_valid"]),
        )
        boundary = RightBoundaryTrajectory.from_arrays(
            case_id=case_meta["case_id"], times=np.asarray(archive["rb_times"]),
            radii=np.asarray(archive["rb_radius"]), valid=np.asarray(archive["rb_valid"]),
        )
        observations = AirObservationStore(
            time=np.asarray(archive["obs_time"]), radius=np.asarray(archive["obs_radius"]),
            zeta=np.asarray(archive["obs_zeta"]), density=np.asarray(archive["obs_density"]),
            velocity=np.asarray(archive["obs_velocity"]), pressure=np.asarray(archive["obs_pressure"]),
            valid_density=np.asarray(archive["valid_density"]),
            valid_velocity=np.asarray(archive["valid_velocity"]),
            valid_pressure=np.asarray(archive["valid_pressure"]),
            valid_domain=np.asarray(archive["valid_domain"]),
            frame_index=np.asarray(archive["obs_frame"]),
        )
    initial = InitialGeometry(
        t0=float(case_meta["t0"]),
        Rc0=float(case_meta["initial"]["Rc0"]),
        Rs0=float(case_meta["initial"]["Rs0"]),
        Rb0=float(case_meta["initial"]["Rb0"]),
        zeta_s0=float(case_meta["initial"]["zeta_s0"]),
        R_sep=float(precursor["R_sep"]),
        u_c_sep=float(precursor["u_c_sep"]),
        D_s_sep=float(precursor["D_s_sep"]),
    )
    return CaseData(
        case_id=str(case_meta["case_id"]),
        role=str(role if role is not None else case_meta.get("role", "train")),
        charge_radius_m=float(case_meta["charge_radius_m"]),
        precursor=dict(precursor), contact=contact, boundary=boundary,
        initial=initial, observations=observations,
        t_end=float(case_meta["t_end"]), tolerances=dict(case_meta["tolerances"]),
        source=dict(case_meta.get("source", {})), notes=list(case_meta.get("notes", [])),
        separation=dict(case_meta.get("separation", {})),
    )


def load_prediction_case(path: Path | str, *, role: str | None = None) -> PredictionCaseData:
    """Load only the fields allowed for ``predict-500``.

    This function never touches ``obs_*``, ``contact_*`` or ``R_raw`` arrays.
    It is intentionally a separate loader rather than ``load_case`` with unused
    attributes: the point is that a missing internal-supervision array cannot
    stop a boundary-only prediction.
    """
    path = Path(path)
    if not path.is_file():
        raise CaseDataError(f"case cache {path} does not exist")
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise CaseDataError(f"{path}: prediction cache has no metadata")
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("format") != "spherical_blast_pinn.contact_air_case":
            raise CaseDataError(f"{path}: not a contact-air case cache")
        if metadata.get("version") != 2:
            raise CaseDataError(
                f"{path}: cache version {metadata.get('version')!r} is not supported "
                "for boundary-only prediction; rebuild it with build-cache"
            )
        for key in ("rb_times", "rb_radius", "rb_valid"):
            if key not in archive.files:
                raise CaseDataError(f"{path}: prediction cache has no {key}")
        case_meta = metadata["case"]
        precursor = dict(metadata["precursor"])
        boundary = RightBoundaryTrajectory.from_arrays(
            case_id=case_meta["case_id"],
            times=np.asarray(archive["rb_times"]),
            radii=np.asarray(archive["rb_radius"]),
            valid=np.asarray(archive["rb_valid"]),
        )
    t0 = float(case_meta.get("t0", precursor["t0"]))
    if not math.isclose(t0, float(precursor["t0"]), rel_tol=0.0, abs_tol=1.0e-12):
        raise CaseDataError(
            f"{path}: cached t0={t0!r} disagrees with precursor t0={precursor['t0']!r}"
        )
    initial = build_initial_geometry(
        t0=t0, Rc0=float(precursor["Rc0"]), Rs0=float(precursor["Rs0"]),
        R_sep=float(precursor["R_sep"]), u_c_sep=float(precursor["u_c_sep"]),
        D_s_sep=float(precursor["D_s_sep"]), boundary=boundary,
    )
    span = boundary.continuous_interval_containing(t0)
    source = dict(case_meta.get("source", {}))
    source["prediction_cache"] = str(path)
    source["prediction_fields_read"] = ["metadata", "rb_times", "rb_radius", "rb_valid"]
    return PredictionCaseData(
        case_id=str(case_meta["case_id"]),
        charge_radius_m=float(case_meta["charge_radius_m"]),
        precursor=precursor, boundary=boundary, initial=initial,
        t_end=float(span[1]), role=str(role or case_meta.get("role", "test")),
        tolerances=dict(case_meta.get("tolerances", {})), source=source,
        notes=list(case_meta.get("notes", [])) + [
            "boundary-only prediction load: no internal observations, contact "
            "interval or R_raw array was read"
        ],
    )


def build_case_from_profile(
    profile_dir: Path | str,
    *,
    precursor: PrecursorResult,
    role: str,
    tolerances: Mapping[str, float],
    source: Mapping[str, Any] | None = None,
) -> CaseData:
    """Derive the new cache from ``cli.extract_profiles`` output.

    The extractor's ``material.npz`` is used for the auditable contact band and
    eligibility arrays; ``raw_line.npz`` supplies the shell fields.  No
    ``alpha=0.5`` or PLIC single-radius label is used.
    """
    directory = Path(profile_dir)
    try:
        case = load_extracted_case(directory)
    except DataError as exc:
        raise CaseDataError(f"cannot load extracted profile {directory}: {exc}") from exc
    companion_path = directory / "material.npz"
    if not companion_path.is_file():
        raise CaseDataError(
            "material.npz is required for the contact interval and front "
            "eligibility; a legacy raw_line-only export cannot define R_c,low/up"
        )
    with np.load(companion_path, allow_pickle=False) as archive:
        low = np.asarray(archive["contact_mixing_band_inner_m"], dtype=np.float64) \
            if "contact_mixing_band_inner_m" in archive.files else None
        up = np.asarray(archive["contact_mixing_band_outer_m"], dtype=np.float64) \
            if "contact_mixing_band_outer_m" in archive.files else None
        alpha = np.asarray(archive["alpha_product"], dtype=np.float64) \
            if "alpha_product" in archive.files else None
        eligible = np.asarray(archive["contact_has_supervision"], dtype=bool) \
            if "contact_has_supervision" in archive.files else None
    if eligible is None:
        eligible = np.asarray(case.contact_valid, dtype=bool)
    if low is None or up is None:
        low = np.full(case.times.size, np.nan, dtype=np.float64)
        up = np.full(case.times.size, np.nan, dtype=np.float64)
    if low.shape != (case.times.size,) or up.shape != (case.times.size,):
        raise CaseDataError(
            "contact mixing-band arrays must have one value per frame"
        )
    if eligible.shape != (case.times.size,):
        raise CaseDataError(
            "contact_has_supervision must have one boolean per frame"
        )
    extraction_source = case.source or {}
    # ``material==2`` is ambiguous in the real extractor (mixing band AND
    # every shell outside the detected shock).  Use the recorded volume
    # fraction when available; fall back to the legacy label only for old
    # fixtures that really encoded the mixing band as 2.
    mixed_frames = None
    if alpha is not None and alpha.shape == np.asarray(case.material).shape:
        metadata_raw = extraction_source.get("metadata_raw", {}) or {}
        partition = (metadata_raw.get("material_partition_source") or {}).get(
            "detail", {}
        ) if isinstance(metadata_raw, Mapping) else {}
        pure_air = float(partition.get("pure_air_threshold", 0.05))
        pure_product = float(partition.get("pure_product_threshold", 0.95))
        mixed_frames = (alpha > pure_air) & (alpha < pure_product)
    derived_low = np.full(case.times.size, np.nan)
    derived_up = np.full(case.times.size, np.nan)
    derived_raw = np.full(case.times.size, np.nan)
    derived_ok = np.zeros(case.times.size, dtype=bool)
    for frame in range(case.times.size):
        # Prefer the extractor's reconstructed mixing band and its adjacent
        # pure shell representatives.  Only if that is unusable fall back to
        # the explicit mixed mask / legacy material label.
        lo, hi, rr, ok, _ = contact_interval_from_band(
            np.asarray(case.radius), np.asarray(case.material)[frame],
            float(low[frame]), float(up[frame]),
        )
        if not ok:
            mixed = None if mixed_frames is None else mixed_frames[frame]
            lo, hi, rr, ok, _ = contact_interval_from_material(
                np.asarray(case.radius), np.asarray(case.material)[frame],
                mixed=mixed,
            )
        if ok:
            derived_low[frame], derived_up[frame], derived_raw[frame] = lo, hi, rr
            derived_ok[frame] = True
    low = np.where(derived_ok, derived_low, low)
    up = np.where(derived_ok, derived_up, up)
    raw = np.where(derived_ok, derived_raw, 0.5 * (low + up))
    valid = eligible & np.isfinite(low) & np.isfinite(up) & (low < up)
    field_validity = case.field_validity or {}
    radial_support = (
        "extract_profiles writes x=ShellGrid.centres (geometric shell midpoints) "
        "and shell-averaged rho/u/p/e. The values used here are therefore shell "
        "representatives, NOT raw ALE cell centroids or cell-level states. The "
        "contact bracket is made of the adjacent shell representatives."
    )
    source_payload = {
        "profile_dir": str(directory),
        "extraction": extraction_source,
        "radial_support": radial_support,
        "radial_coordinate_origin": "shell midpoint (geometric), not cell centroid",
        "fields_are_shell_averaged": True,
        "material_partition_source": extraction_source.get("material_masks_from"),
        "profile_layout": extraction_source.get("profile_layout"),
    }
    if source:
        source_payload.update(dict(source))
    arrays = CaseArrays(
        case_id=case.case_id, charge_radius_m=float(case.charge_radius),
        times=np.asarray(case.times), radius=np.asarray(case.radius),
        density=np.asarray(case.density), velocity=np.asarray(case.velocity),
        pressure=np.asarray(case.pressure), material=np.asarray(case.material),
        valid=np.asarray(case.valid),
        valid_density=field_validity.get("density"),
        valid_velocity=field_validity.get("velocity"),
        valid_pressure=field_validity.get("pressure"),
        contact_low=low, contact_up=up, contact_raw=raw, contact_valid=valid,
        source={"profile_dir": str(directory), "extraction": case.source},
    )
    return build_case(
        arrays, precursor=precursor, role=role, tolerances=tolerances,
        source=source_payload,
        notes=[
            "built by build_case_from_profile; no alpha=0.5 label was used",
            radial_support,
            "radial coordinate origin: shell midpoint (geometric), not cell centroid",
            "right-boundary and separation evidence are recorded per frame; a "
            "locatable mixed segment alone is not called 'separated'",
        ],
    )


def synthetic_case(
    *, case_id: str, charge_radius_m: float, precursor: PrecursorResult,
    role: str, n_frames: int = 41, n_radial: int = 400,
    duration_s: float = 2.0e-5, tolerances: Mapping[str, float] | None = None,
) -> CaseData:
    """Small analytic case used only for engineering tests.

    It is labelled ``synthetic`` in ``source``; it is never presented as a TNT
    validation case.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    t0 = float(precursor.t0)
    times = np.linspace(t0, t0 + float(duration_s), int(n_frames))
    dt = times - t0
    # R_c and R_b are smooth, non-parallel trajectories; R_s is not used as R_b.
    rc = precursor.Rc0 + precursor.u_c_sep * dt + 2.0e5 * dt**2
    rb = precursor.Rs0 + 1.0e-3 + precursor.D_s_sep * dt + 5.0e5 * dt**2
    if np.any(rb <= rc):
        raise CaseDataError("synthetic R_b trajectory is not outside R_c")
    radius = np.linspace(0.2 * precursor.Rc0, float(rb.max()) * 1.05, int(n_radial))
    mat = np.full((times.size, radius.size), 2, dtype=np.int8)
    pressure = np.full((times.size, radius.size), precursor.air_pressure, dtype=np.float64)
    velocity = np.zeros((times.size, radius.size), dtype=np.float64)
    density = np.full((times.size, radius.size), precursor.air_density, dtype=np.float64)
    contact_low = rc - 0.15 * (rb - rc) / max(n_frames, 1)
    contact_up = rc + 0.15 * (rb - rc) / max(n_frames, 1)
    for frame in range(times.size):
        rc_f = float(rc[frame])
        rb_f = float(rb[frame])
        r = radius
        product = r < rc_f
        air = (r >= rc_f) & (r <= rb_f)
        outside = r > rb_f
        mat[frame, product] = 0
        mat[frame, air] = 1
        mat[frame, outside] = 1
        # Decaying blast profile in the shocked air; ambient outside.
        s = np.clip((r[air] - rc_f) / max(rb_f - rc_f, 1e-30), 0.0, 1.0)
        pressure[frame, air] = precursor.air_pressure + (
            precursor.p_sep - precursor.air_pressure
        ) * np.exp(-4.0 * s)
        velocity[frame, air] = precursor.u_a_sep * np.exp(-4.0 * s)
        density[frame, air] = precursor.air_density + (
            precursor.rho_a_sep - precursor.air_density
        ) * np.exp(-4.0 * s)
        pressure[frame, product] = precursor.p_sep
        density[frame, product] = precursor.rho_a_sep
    valid = np.ones_like(pressure, dtype=bool)
    arrays = CaseArrays(
        case_id=case_id, charge_radius_m=charge_radius_m, times=times, radius=radius,
        density=density, velocity=velocity, pressure=pressure, material=mat,
        valid=valid, valid_density=valid, valid_velocity=valid, valid_pressure=valid,
        contact_low=contact_low, contact_up=contact_up, contact_raw=rc,
        contact_valid=np.ones(times.size, dtype=bool),
        rb_times=times, rb_radius=rb, rb_valid=np.ones(times.size, dtype=bool),
        source={"synthetic": True, "note": "analytic engineering fixture, not TNT data"},
    )
    tol = dict(tolerances or {
        "ambient_pressure_pa": precursor.air_pressure,
        "ambient_density_kg_m3": precursor.air_density,
        "pressure_pa": 1.0e-6 * precursor.air_pressure,
        "density_kg_m3": 1.0e-9,
        "velocity_m_s": 1.0e-6,
        "min_ambient_cells": 2,
    })
    return build_case(
        arrays, precursor=precursor, role=role, tolerances=tol,
        source={"synthetic": True, "note": "analytic engineering fixture, not TNT data"},
        notes=["synthetic engineering fixture; do not report as real TNT validation"],
    )


def precursor_from_case_mapping(payload: Mapping[str, Any]) -> PrecursorResult:
    """Convenience for configs that inline the precursor dict."""
    return compute_precursor(PrecursorConfig.from_mapping(payload))


__all__ = [
    "AirObservationStore",
    "CaseArrays",
    "CaseData",
    "CaseDataError",
    "PredictionCaseData",
    "RightBoundaryDetection",
    "build_air_observations",
    "build_case",
    "build_case_from_profile",
    "classify_separation_frames",
    "contact_interval_from_material",
    "detect_right_boundary",
    "detect_right_boundary_report",
    "load_case",
    "load_prediction_case",
    "precursor_from_case_mapping",
    "save_case",
    "synthetic_case",
]
