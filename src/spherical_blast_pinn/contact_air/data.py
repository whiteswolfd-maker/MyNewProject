"""Per-case contract and cache for the one-dimensional ContactNet/AirNet scheme.

The geometry is ``R_c(t) <= r <= R_s(t)`` where ``R_s`` is the outer shock
identified from the raw radial pressure profile.  There is no ``R_b`` and no
``R_outer`` in this contract.  Every case owns its raw frame clock, event
alignment, contact interval table, shock trajectory and air observations.

Real binary readings are not fabricated.  :func:`build_case` consumes either a
:class:`~spherical_blast_pinn.contact_air.one_d.RadialCase` from
:func:`~spherical_blast_pinn.contact_air.one_d.read_standard_1d` or the
synthetic fixture.  The synthetic path is labelled ``synthetic``.
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

from .geometry import (
    ContactIntervalTable,
    GeometryError,
    InitialGeometry,
    ShockTrajectory,
    build_initial_geometry,
)
from .one_d import (
    EventAlignment,
    OneDDataError,
    RadialCase,
    RadialFrame,
    ShockDetection,
    ShockDetectionConfig,
    align_event,
    contact_interval_from_fractions,
    detect_shock_trajectory,
)
from .precursor import PrecursorResult

DTYPE = "float64"


class CaseDataError(RuntimeError):
    """A case cache is missing, inconsistent or not usable for the requested role."""


def _as_float(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CaseDataError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise CaseDataError(
            f"{name} must be finite{' and positive' if positive else ''}, got {result!r}"
        )
    return result


def _tensor(values: Any, dtype: str = DTYPE):
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    return torch.as_tensor(np.asarray(values), dtype=getattr(torch, dtype))


@dataclass(frozen=True)
class ContactLabelConfig:
    """Explicit thresholds for the contact-mixing interval.

    These are configuration values, not thresholds confirmed against real
    50 mm data.  The interval uses actual cell boundaries, never ``alpha=0.5``
    or an interval midpoint.
    """

    pure_product_threshold: float = 0.95
    pure_air_threshold: float = 0.95
    min_pure_air_cells: int = 1
    event_search_window_s: float | None = 10.0e-6
    #: Absolute tolerance for shared cell boundaries; larger positive gaps are
    #: spatial breaks, larger negative gaps are overlaps.
    geometry_tolerance_m: float = 1.0e-9

    def __post_init__(self) -> None:
        if not (0.0 < self.pure_product_threshold <= 1.0):
            raise CaseDataError("pure_product_threshold must lie in (0, 1]")
        if not (0.0 < self.pure_air_threshold <= 1.0):
            raise CaseDataError("pure_air_threshold must lie in (0, 1]")
        if 1.0 - self.pure_air_threshold >= self.pure_product_threshold:
            raise CaseDataError(
                "pure_air_threshold and pure_product_threshold leave no mixing band"
            )
        if int(self.min_pure_air_cells) < 0:
            raise CaseDataError("min_pure_air_cells must be non-negative")
        if self.event_search_window_s is not None and self.event_search_window_s < 0.0:
            raise CaseDataError("event_search_window_s must be non-negative or null")
        if self.geometry_tolerance_m < 0.0:
            raise CaseDataError("geometry_tolerance_m must be non-negative")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ContactLabelConfig":
        payload = dict(payload or {})
        window = payload.get("event_search_window_s", 10.0e-6)
        return cls(
            pure_product_threshold=float(payload.get("pure_product_threshold", 0.95)),
            pure_air_threshold=float(payload.get("pure_air_threshold", 0.95)),
            min_pure_air_cells=int(payload.get("min_pure_air_cells", 1)),
            event_search_window_s=None if window is None else float(window),
            geometry_tolerance_m=float(payload.get("geometry_tolerance_m", 1.0e-9)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pure_product_threshold": self.pure_product_threshold,
            "pure_air_threshold": self.pure_air_threshold,
            "min_pure_air_cells": self.min_pure_air_cells,
            "event_search_window_s": self.event_search_window_s,
            "geometry_tolerance_m": self.geometry_tolerance_m,
            "label_semantics": (
                "mixing segment connected to the central product region, bounded "
                "by actual cell boundaries; midpoint is not a label"
            ),
        }


@dataclass
class AirObservationStore:
    time: np.ndarray
    radius: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    valid_density: np.ndarray
    valid_velocity: np.ndarray
    valid_pressure: np.ndarray
    frame_index: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

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
        """Draw air points **frame first, then point inside the frame**."""
        empty = _tensor(np.zeros(0))
        empty_bool = empty.bool()
        if self.size == 0 or count <= 0:
            return {
                "t": empty, "radius": empty, "rho": empty, "u": empty, "p": empty,
                "valid_rho": empty_bool, "valid_u": empty_bool,
                "valid_p": empty_bool, "count": 0,
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
            return {
                "t": empty, "radius": empty, "rho": empty, "u": empty, "p": empty,
                "valid_rho": empty_bool, "valid_u": empty_bool,
                "valid_p": empty_bool, "count": 0,
            }
        by_frame = {
            int(frame): np.flatnonzero(allowed & (self.frame_index == frame))
            for frame in frames
        }
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
            "radius": _tensor(self.radius[selected]),
            "rho": _tensor(self.density[selected]),
            "u": _tensor(self.velocity[selected]),
            "p": _tensor(self.pressure[selected]),
            "valid_rho": _tensor(self.valid_density[selected]).bool(),
            "valid_u": _tensor(self.valid_velocity[selected]).bool(),
            "valid_p": _tensor(self.valid_pressure[selected]).bool(),
            "count": int(selected.size),
        }

    def subset(self, mask: np.ndarray) -> "AirObservationStore":
        return AirObservationStore(
            time=self.time[mask], radius=self.radius[mask],
            density=self.density[mask], velocity=self.velocity[mask],
            pressure=self.pressure[mask],
            valid_density=self.valid_density[mask],
            valid_velocity=self.valid_velocity[mask],
            valid_pressure=self.valid_pressure[mask],
            frame_index=self.frame_index[mask],
            diagnostics=dict(self.diagnostics),
        )


@dataclass
class CaseData:
    case_id: str
    role: str
    charge_radius_m: float
    precursor: dict[str, Any]
    contact: ContactIntervalTable
    shock: ShockTrajectory
    initial: InitialGeometry
    observations: AirObservationStore
    t_end: float
    tolerances: dict[str, Any]
    event_alignment: dict[str, Any]
    contact_config: dict[str, Any]
    shock_config: dict[str, Any]
    source: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    separation: dict[str, Any] = field(default_factory=dict)
    raw_times_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.role not in ("train", "validation", "test"):
            raise CaseDataError(f"role must be train/validation/test, got {self.role!r}")
        if not (self.initial.t0 < self.t_end):
            raise CaseDataError(
                f"case {self.case_id!r}: t_end={self.t_end!r} must be after t0="
                f"{self.initial.t0!r}"
            )
        if self.shock.domain is None:
            raise CaseDataError(f"case {self.case_id!r}: no continuous R_s trajectory")
        try:
            self.shock.require_interval(self.t0, self.t_end)
        except GeometryError as exc:
            raise CaseDataError(
                f"case {self.case_id!r}: the solve interval [{self.t0!r}, "
                f"{self.t_end!r}] is not a single continuous R_s segment: {exc}"
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
        jwl = self.precursor["jwl"]
        V = float(self.precursor["V_sep"]) * (
            self.initial.Rc0 / float(self.precursor["R_sep"])
        ) ** 3
        return _jwl_pressure_numpy(V, jwl)

    @property
    def time_scale(self) -> float:
        return self.length_scale / self.u_scale

    @property
    def energy_density_scale(self) -> float:
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
        p_contact = float(self.p_c_at_t0)
        difference = p_contact - self.p_scale
        relative = abs(difference) / self.p_scale
        event_report = None
        event = self.event_alignment.get("event_frame")
        frames = (self.separation or {}).get("frames", [])
        if event is not None and 0 <= int(event) < len(frames):
            event_report = frames[int(event)]
        detection = (
            event_report.get("detection") if isinstance(event_report, Mapping) else None
        ) or {}
        return {
            "p_c_at_t0_pa": p_contact,
            "p_a_sep_pa": self.p_scale,
            "absolute_difference_pa": difference,
            "relative_difference": relative,
            "R_c0_theory_m": self.initial.Rc0,
            "R_s0_theory_m": self.initial.Rs0,
            "R_s_data_at_t0_m": self.initial.Rs_data_at_t0,
            "R_s_theory_minus_data_m": self.initial.Rs0 - self.initial.Rs_data_at_t0,
            "zeta_s0": self.initial.zeta_s0,
            "zeta_s0_role": "diagnostic_only; not an initial-condition split",
            "event_frame_shock_states": detection.get("states"),
            "event_frame_rh_diagnostic": detection.get("rh_diagnostic"),
            "forced_equal": False,
            "note": (
                "theoretical R_c0/R_s0, the event-frame contact/shock labels and "
                "the data-side RH diagnostic are reported side by side; "
                "differences are not removed and no trajectory point is invented "
                "to hide them"
            ),
        }

    def shock_tensors(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Torch PCHIP ``R_s(t)`` and exact ``dR_s/dt`` for the given times."""
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(np.asarray(t), dtype=torch.float64)
        radius, speed = self.shock.position_and_derivative_torch(t)
        shape = tuple(t.shape)
        return radius.reshape(shape), speed.reshape(shape)

    def shock_at(self, t) -> tuple[Any, Any]:
        values = np.asarray(t, dtype=np.float64)
        return self.shock.position(values), self.shock.derivative(values)

    def prediction_view(self) -> "PredictionCaseData":
        return PredictionCaseData.from_case(self)

    def subset_frames(self, frames: Sequence[int], *, role: str | None = None) -> "CaseData":
        indices = np.asarray(sorted({int(f) for f in frames}), dtype=np.int64)
        if indices.size == 0:
            raise CaseDataError(f"case {self.case_id!r}: empty frame subset")
        n = self.contact.times.size
        if np.any(indices < 0) or np.any(indices >= n):
            raise CaseDataError(f"case {self.case_id!r}: frame subset is out of range")
        obs_mask = np.isin(self.observations.frame_index, indices)
        observations = self.observations.subset(obs_mask)
        contact_mask = np.zeros(n, dtype=bool)
        contact_mask[indices] = True
        contact_mask &= self.contact.valid
        contact = ContactIntervalTable(
            case_id=self.contact.case_id,
            times=self.contact.times[contact_mask],
            low=self.contact.low[contact_mask], up=self.contact.up[contact_mask],
            valid=self.contact.valid[contact_mask],
            reasons=[reason for reason, keep in zip(self.contact.reasons, contact_mask) if keep]
            if self.contact.reasons else [],
        )
        selected_times = self.contact.times[indices]
        t_end = max(
            float(selected_times.max()),
            float(observations.time.max()) if observations.size else -np.inf,
        )
        if not (self.t0 < t_end):
            raise CaseDataError(
                f"case {self.case_id!r}: selected frames do not extend after t0"
            )
        try:
            self.shock.require_interval(self.t0, t_end)
        except GeometryError as exc:
            raise CaseDataError(
                f"case {self.case_id!r}: selected frames span an R_s gap: {exc}"
            ) from exc
        return CaseData(
            case_id=self.case_id, role=str(role or self.role),
            charge_radius_m=self.charge_radius_m, precursor=dict(self.precursor),
            contact=contact, shock=self.shock, initial=self.initial,
            observations=observations, t_end=t_end,
            tolerances=dict(self.tolerances),
            event_alignment=dict(self.event_alignment),
            contact_config=dict(self.contact_config),
            shock_config=dict(self.shock_config),
            source=dict(self.source),
            notes=list(self.notes) + [f"frame subset {indices.tolist()}"],
            separation=dict(self.separation),
            raw_times_s=None if self.raw_times_s is None else self.raw_times_s[indices],
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
                "Rs_data_at_t0": self.initial.Rs_data_at_t0,
                "zeta_s0": self.initial.zeta_s0,
            },
            "contact": self.contact.as_dict(),
            "shock": self.shock.as_dict(),
            "event_alignment": dict(self.event_alignment),
            "observations": self.observations.size,
            "scales": self.scale_dict(),
            "precursor": dict(self.precursor),
            "initial_compatibility": self.initial_compatibility(),
            "p_c_at_t0_pa": self.p_c_at_t0,
            "p_a_sep_pa": self.p_scale,
            "p_c_vs_p_a_sep_relative": abs(self.p_c_at_t0 - self.p_scale) / self.p_scale,
            "tolerances": dict(self.tolerances),
            "contact_config": dict(self.contact_config),
            "shock_config": dict(self.shock_config),
            "source": dict(self.source),
            "notes": list(self.notes),
            "separation": dict(self.separation),
        }


@dataclass
class PredictionCaseData:
    """Boundary-only view used by ``predict``.

    Only the shock trajectory, physics and initial geometry are exposed; no
    internal air observation or contact interval is read.
    """

    case_id: str
    charge_radius_m: float
    precursor: dict[str, Any]
    shock: ShockTrajectory
    initial: InitialGeometry
    t_end: float
    role: str = "test"
    tolerances: dict[str, Any] = field(default_factory=dict)
    event_alignment: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (self.initial.t0 < self.t_end):
            raise CaseDataError(
                f"case {self.case_id!r}: prediction t_end={self.t_end!r} must be "
                f"after t0={self.initial.t0!r}"
            )
        try:
            self.shock.require_interval(self.initial.t0, self.t_end)
        except GeometryError as exc:
            raise CaseDataError(
                f"case {self.case_id!r}: prediction interval is not one continuous "
                f"R_s segment: {exc}"
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

    def shock_tensors(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(np.asarray(t), dtype=torch.float64)
        radius, speed = self.shock.position_and_derivative_torch(t)
        shape = tuple(t.shape)
        return radius.reshape(shape), speed.reshape(shape)

    @classmethod
    def from_case(cls, case: CaseData) -> "PredictionCaseData":
        span = case.shock.continuous_interval_containing(case.t0)
        return cls(
            case_id=case.case_id, charge_radius_m=case.charge_radius_m,
            precursor=dict(case.precursor), shock=case.shock,
            initial=case.initial, t_end=float(span[1]), role=case.role,
            tolerances=dict(case.tolerances),
            event_alignment=dict(case.event_alignment),
            source=dict(case.source),
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
                "Rs_data_at_t0": self.initial.Rs_data_at_t0,
                "zeta_s0": self.initial.zeta_s0,
            },
            "shock": self.shock.as_dict(),
            "event_alignment": dict(self.event_alignment),
            "source": dict(self.source),
            "notes": list(self.notes),
            "prediction_fields": ["R_s(t)", "dR_s/dt", "precursor", "initial", "time range"],
        }


def _jwl_pressure_numpy(V: float, jwl: Mapping[str, Any]) -> float:
    return (
        float(jwl["A"]) * math.exp(-float(jwl["R1"]) * V)
        + float(jwl["B"]) * math.exp(-float(jwl["R2"]) * V)
        + float(jwl["C"]) / V ** (1.0 + float(jwl["omega"]))
    )


def _contact_tables(
    frames: Sequence[RadialFrame], config: ContactLabelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    n = len(frames)
    low = np.full(n, np.nan)
    up = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    reasons: list[str] = []
    for index, frame in enumerate(frames):
        lo, hi, ok, reason = contact_interval_from_fractions(
            frame, pure_product_threshold=config.pure_product_threshold,
            pure_air_threshold=config.pure_air_threshold,
            geometry_tolerance_m=config.geometry_tolerance_m,
        )
        low[index], up[index] = lo, hi
        valid[index] = ok
        reasons.append(reason)
    return low, up, valid, reasons


def build_air_observations(
    frames: Sequence[RadialFrame],
    *,
    times_model: np.ndarray,
    contact: ContactIntervalTable,
    shock: ShockTrajectory,
    contact_config: ContactLabelConfig,
) -> AirObservationStore:
    """Collect air cells on ``contact_up <= r <= R_s`` with independent masks.

    Position eligibility uses the air volume fraction only.  It does not
    require all three fields to be finite and it does not consult any shock
    supervision flag: a missing pressure does not throw away a valid density
    and velocity.  Observations store the **physical** radius; the input
    coordinate is recomputed at training time from the predicted ``R_c`` and
    the shock trajectory, never from a contact-interval midpoint.
    """
    obs_t, obs_r = [], []
    obs_rho, obs_u, obs_p = [], [], []
    obs_vd, obs_vu, obs_vp, obs_frame = [], [], [], []
    totals = {
        "air_cells": 0,
        "rejected_no_contact": 0,
        "rejected_no_shock": 0,
        "rejected_shock_not_outside_contact": 0,
        "rejected_outside_domain": 0,
        "selected": 0,
        "valid_density": 0,
        "valid_velocity": 0,
        "valid_pressure": 0,
    }
    frame_reports: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        report: dict[str, Any] = {
            "frame": int(index),
            "time_raw_s": float(frame.time_raw_s),
            "time_model_s": float(times_model[index]),
            "selected": 0,
            "rejected": {},
        }
        air = frame.air_mask(contact_config.pure_air_threshold)
        totals["air_cells"] += int(np.count_nonzero(air))
        if not bool(contact.valid[index]):
            rejected = int(np.count_nonzero(air))
            report["rejected"]["no_contact"] = rejected
            totals["rejected_no_contact"] += rejected
            frame_reports.append(report)
            continue
        try:
            rs = float(shock.position(float(times_model[index])))
        except GeometryError:
            rejected = int(np.count_nonzero(air))
            report["rejected"]["no_shock"] = rejected
            totals["rejected_no_shock"] += rejected
            frame_reports.append(report)
            continue
        up = float(contact.up[index])
        if not (rs > up):
            rejected = int(np.count_nonzero(air))
            report["rejected"]["shock_not_outside_contact"] = rejected
            totals["rejected_shock_not_outside_contact"] += rejected
            frame_reports.append(report)
            continue
        located = air & (frame.radius >= up) & (frame.radius <= rs)
        outside = int(np.count_nonzero(air & ~located))
        if outside:
            report["rejected"]["outside_domain"] = outside
            totals["rejected_outside_domain"] += outside
        indices = np.flatnonzero(located)
        if indices.size == 0:
            frame_reports.append(report)
            continue
        report["selected"] = int(indices.size)
        totals["selected"] += int(indices.size)
        totals["valid_density"] += int(np.count_nonzero(frame.valid_density[indices]))
        totals["valid_velocity"] += int(np.count_nonzero(frame.valid_velocity[indices]))
        totals["valid_pressure"] += int(np.count_nonzero(frame.valid_pressure[indices]))
        obs_t.append(np.full(indices.size, float(times_model[index])))
        obs_r.append(frame.radius[indices])
        obs_rho.append(frame.density[indices])
        obs_u.append(frame.velocity[indices])
        obs_p.append(frame.pressure[indices])
        obs_vd.append(frame.valid_density[indices])
        obs_vu.append(frame.valid_velocity[indices])
        obs_vp.append(frame.valid_pressure[indices])
        obs_frame.append(np.full(indices.size, index, dtype=np.int64))
        frame_reports.append(report)
    diagnostics = {
        "basis": (
            "air cells from volume fractions on contact_up <= r <= R_s; "
            "per-field validity is independent; position eligibility does not "
            "depend on shock supervision flags"
        ),
        "radial_support": (
            "raw one-dimensional cell representative positions; no shell "
            "averaging or direction sampling"
        ),
        "zeta_mapping": (
            "computed at training time as (r - R_c(pred))/(R_s - R_c(pred)); the "
            "contact interval midpoint is never substituted for R_c"
        ),
        "totals": totals,
        "frames": frame_reports,
    }
    if not obs_t:
        empty = np.zeros(0, dtype=np.float64)
        return AirObservationStore(
            empty, empty, empty, empty, empty,
            np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, bool),
            np.zeros(0, np.int64), diagnostics=diagnostics,
        )
    return AirObservationStore(
        time=np.concatenate(obs_t), radius=np.concatenate(obs_r),
        density=np.concatenate(obs_rho), velocity=np.concatenate(obs_u),
        pressure=np.concatenate(obs_p),
        valid_density=np.concatenate(obs_vd),
        valid_velocity=np.concatenate(obs_vu),
        valid_pressure=np.concatenate(obs_vp),
        frame_index=np.concatenate(obs_frame),
        diagnostics=diagnostics,
    )


def _separation_report(
    frames: Sequence[RadialFrame],
    *,
    times_model: np.ndarray,
    contact: ContactIntervalTable,
    shock: ShockTrajectory,
    contact_config: ContactLabelConfig,
    detections: Sequence[ShockDetection],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        contact_ok = bool(contact.valid[index])
        shock_ok = bool(shock.has_slope(float(times_model[index])))
        pure_air = False
        n_pure = 0
        reasons: list[str] = []
        if contact_ok and shock_ok:
            rs = float(shock.position(float(times_model[index])))
            up = float(contact.up[index])
            from .one_d import pure_air_run_between

            pure_air, n_pure = pure_air_run_between(
                frame, low=up, high=rs,
                pure_air_threshold=contact_config.pure_air_threshold,
                minimum=contact_config.min_pure_air_cells,
                geometry_tolerance_m=contact_config.geometry_tolerance_m,
            )
        if not contact_ok:
            reasons.append("contact interval is not locatable")
        if not shock_ok:
            reasons.append("outer shock is not identified")
        if contact_ok and shock_ok and not pure_air:
            reasons.append("no valid pure-air interval between contact and shock")
        rows.append({
            "time_raw_s": float(frame.time_raw_s),
            "time_model_s": float(times_model[index]),
            "contact_locatable": contact_ok,
            "shock_identified": shock_ok,
            "pure_air_between": bool(pure_air),
            "n_pure_air_cells": n_pure,
            "status": "separated" if (contact_ok and shock_ok and pure_air) else "unresolvable",
            "detection": (
                detections[index].as_dict() if index < len(detections) else None
            ),
            "reasons": reasons,
        })
    return {
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in ("separated", "unresolvable")
        },
        "frames": rows,
        "basis": (
            "contact locatable + outer shock identified + at least "
            f"{contact_config.min_pure_air_cells} valid pure-air cell(s) between "
            "them, each recorded separately"
        ),
    }


def build_case(
    source: RadialCase,
    *,
    precursor: PrecursorResult,
    role: str,
    tolerances: Mapping[str, Any],
    contact_config: ContactLabelConfig | None = None,
    shock_config: ShockDetectionConfig | None = None,
    t_end: float | None = None,
    source_meta: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
) -> CaseData:
    contact_config = contact_config or ContactLabelConfig()
    shock_config = shock_config or ShockDetectionConfig()
    frames = list(source.frames)
    raw_times = np.asarray([frame.time_raw_s for frame in frames], dtype=np.float64)
    if raw_times.size < 2 or np.any(np.diff(raw_times) <= 0.0):
        raise CaseDataError("raw frame times must be finite and strictly increasing")
    low, up, contact_valid, reasons = _contact_tables(frames, contact_config)
    shock_radii, shock_valid, detections = detect_shock_trajectory(
        frames, contact_low=low, contact_up=up, contact_valid=contact_valid,
        ambient_pressure=float(precursor.air_pressure),
        pure_air_threshold=contact_config.pure_air_threshold,
        config=shock_config,
    )
    try:
        alignment = align_event(
            frames, contact_up=up, contact_valid=contact_valid,
            shock_radius=shock_radii, shock_valid=shock_valid,
            t_separation=float(precursor.t_sep), t0=float(precursor.t0),
            pure_air_threshold=contact_config.pure_air_threshold,
            window_s=contact_config.event_search_window_s,
            min_pure_air_cells=contact_config.min_pure_air_cells,
            geometry_tolerance_m=contact_config.geometry_tolerance_m,
        )
    except OneDDataError as exc:
        raise CaseDataError(str(exc)) from exc
    offset = float(alignment.offset_s)
    times_model = raw_times + offset
    contact = ContactIntervalTable(
        case_id=source.case_id, times=times_model,
        low=low, up=up, valid=contact_valid, reasons=reasons,
    )
    shock = ShockTrajectory.from_arrays(
        case_id=source.case_id, times=times_model,
        radii=shock_radii, valid=shock_valid,
    )
    initial = build_initial_geometry(
        t0=float(precursor.t0), Rc0=float(precursor.Rc0), Rs0=float(precursor.Rs0),
        R_sep=float(precursor.R_sep), u_c_sep=float(precursor.u_c_sep),
        D_s_sep=float(precursor.D_s_sep), shock=shock,
    )
    observations = build_air_observations(
        frames, times_model=times_model, contact=contact, shock=shock,
        contact_config=contact_config,
    )
    separation = _separation_report(
        frames, times_model=times_model, contact=contact, shock=shock,
        contact_config=contact_config, detections=detections,
    )
    separation["event_alignment"] = alignment.as_dict()
    segment = shock.continuous_interval_containing(float(precursor.t0))
    end = float(segment[1] if t_end is None else t_end)
    if t_end is not None:
        shock.require_interval(float(precursor.t0), end)
    if not (precursor.t0 < end):
        raise CaseDataError(
            f"case {source.case_id!r}: no training interval after t0; "
            f"t0={precursor.t0!r}, t_end={end!r}"
        )
    # The right boundary is a finite-Mach RH state.  If no time in the single
    # t0 segment has D_s > c_inf, no RH sample can ever be drawn; report that
    # now instead of failing later inside sampling.
    c_inf_check = math.sqrt(
        float(precursor.gamma_air)
        * float(precursor.air_pressure)
        / float(precursor.air_density)
    )
    check_times = np.linspace(float(precursor.t0), end, 513)
    check_speed = np.asarray(shock.derivative(check_times), dtype=np.float64)
    supersonic = np.isfinite(check_speed) & (check_speed > c_inf_check)
    if not bool(np.any(supersonic)):
        raise CaseDataError(
            f"case {source.case_id!r}: no time in the t0 segment "
            f"[{precursor.t0!r}, {end!r}] s has D_s > c_inf={c_inf_check:.6g} m/s; "
            "the shock cannot be used as a finite-Mach RH boundary, so the smoke "
            "run cannot be assembled. Plateau/zero-speed positions are recorded "
            "but are not silently promoted to a usable shock speed."
        )
    source_payload = {
        "data_entry": source.source,
        "raw_position_type": source.position_type,
        "contact_thresholds": contact_config.as_dict(),
        "shock_detection": shock_config.as_dict(),
        "event_alignment": alignment.as_dict(),
        "shock_speed_availability": {
            "c_inf_m_s": float(c_inf_check),
            "n_supersonic_check_times": int(np.count_nonzero(supersonic)),
            "n_check_times": int(check_times.size),
            "basis": "R_s derivative sampled uniformly in the single t0 segment",
        },
        "radial_coordinate_convention": (
            "representative position as declared by the source "
            "(cell centre or node); no shell averaging"
        ),
        "pressure_convention": (
            "absolute pressure; overpressure is only formed at output time and "
            "is never fed to EOS/PDE/RH"
        ),
    }
    if bool(source.source.get("synthetic", False)):
        source_payload["synthetic"] = True
    if source_meta:
        source_payload.update(dict(source_meta))
    return CaseData(
        case_id=str(source.case_id), role=str(role),
        charge_radius_m=float(source.charge_radius_m),
        precursor=precursor.as_dict(), contact=contact, shock=shock,
        initial=initial, observations=observations, t_end=end,
        tolerances=dict(tolerances),
        event_alignment=alignment.as_dict(),
        contact_config=contact_config.as_dict(),
        shock_config=shock_config.as_dict(),
        source=source_payload,
        notes=list(notes or []) + [
            "contact intervals use actual cell boundaries, not alpha=0.5 or midpoints",
            "shock positions come from the -dp/dr peak on valid contiguous air cells",
            "event alignment t_model = t0 + (t_raw - t_event_raw)",
        ],
        separation=separation,
        raw_times_s=raw_times,
    )


def save_case(case: CaseData, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "spherical_blast_pinn.contact_air_case",
        "version": 7,
        "units": {
            "time": "s", "radius": "m", "density": "kg/m^3",
            "velocity": "m/s", "pressure": "Pa", "specific_energy": "J/kg",
            "energy_density": "J/m^3 (Pa)",
        },
        "case": case.as_dict(),
        "precursor": case.precursor,
        "prediction_requires_only": [
            "metadata", "shock_times", "shock_radius", "shock_valid",
        ],
        "cache_rebuild_reason": (
            "version 7: the geometric right boundary is the identified shock R_s, "
            "observations store physical radius (zeta is recomputed from predicted "
            "R_c at training time), and the raw/aligned clocks are stored. A "
            "version-6 cache is an R_b/shell-averaged artefact and cannot be "
            "converted silently; rebuild it with build-cache."
        ),
    }
    np.savez(
        path,
        metadata=np.asarray(json.dumps(metadata)),
        obs_time=case.observations.time,
        obs_radius=case.observations.radius,
        obs_density=case.observations.density,
        obs_velocity=case.observations.velocity,
        obs_pressure=case.observations.pressure,
        valid_density=case.observations.valid_density,
        valid_velocity=case.observations.valid_velocity,
        valid_pressure=case.observations.valid_pressure,
        obs_frame=case.observations.frame_index,
        contact_times=case.contact.times,
        contact_low=case.contact.low,
        contact_up=case.contact.up,
        contact_valid=case.contact.valid,
        shock_times=case.shock.times,
        shock_radius=case.shock.radii,
        shock_valid=case.shock.valid,
        raw_times=case.raw_times_s if case.raw_times_s is not None else np.zeros(0),
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
        if metadata.get("version") != 7:
            raise CaseDataError(
                f"{path}: cache version {metadata.get('version')!r} is not supported. "
                "Rebuild it with build-cache: version 7 uses the identified shock "
                "R_s and stores physical observation radii; older R_b/shell "
                "caches cannot be converted silently"
            )
        case_meta = metadata["case"]
        precursor = metadata["precursor"]
        contact = ContactIntervalTable(
            case_id=case_meta["case_id"], times=np.asarray(archive["contact_times"]),
            low=np.asarray(archive["contact_low"]), up=np.asarray(archive["contact_up"]),
            valid=np.asarray(archive["contact_valid"]),
        )
        shock = ShockTrajectory.from_arrays(
            case_id=case_meta["case_id"], times=np.asarray(archive["shock_times"]),
            radii=np.asarray(archive["shock_radius"]), valid=np.asarray(archive["shock_valid"]),
        )
        observations = AirObservationStore(
            time=np.asarray(archive["obs_time"]), radius=np.asarray(archive["obs_radius"]),
            density=np.asarray(archive["obs_density"]),
            velocity=np.asarray(archive["obs_velocity"]),
            pressure=np.asarray(archive["obs_pressure"]),
            valid_density=np.asarray(archive["valid_density"]),
            valid_velocity=np.asarray(archive["valid_velocity"]),
            valid_pressure=np.asarray(archive["valid_pressure"]),
            frame_index=np.asarray(archive["obs_frame"]),
            diagnostics=dict(
                (case_meta.get("separation", {}) or {}).get("air_supervision", {})
            ),
        )
        raw_times = np.asarray(archive["raw_times"]) if "raw_times" in archive.files else None
    initial = InitialGeometry(
        t0=float(case_meta["t0"]),
        Rc0=float(case_meta["initial"]["Rc0"]),
        Rs0=float(case_meta["initial"]["Rs0"]),
        zeta_s0=float(case_meta["initial"]["zeta_s0"]),
        R_sep=float(precursor["R_sep"]),
        u_c_sep=float(precursor["u_c_sep"]),
        D_s_sep=float(precursor["D_s_sep"]),
        Rs_data_at_t0=float(case_meta["initial"]["Rs_data_at_t0"]),
    )
    return CaseData(
        case_id=str(case_meta["case_id"]),
        role=str(role if role is not None else case_meta.get("role", "train")),
        charge_radius_m=float(case_meta["charge_radius_m"]),
        precursor=dict(precursor), contact=contact, shock=shock,
        initial=initial, observations=observations,
        t_end=float(case_meta["t_end"]), tolerances=dict(case_meta["tolerances"]),
        event_alignment=dict(case_meta.get("event_alignment", {})),
        contact_config=dict(case_meta.get("contact_config", {})),
        shock_config=dict(case_meta.get("shock_config", {})),
        source=dict(case_meta.get("source", {})),
        notes=list(case_meta.get("notes", [])),
        separation=dict(case_meta.get("separation", {})),
        raw_times_s=None if raw_times is None or raw_times.size == 0 else raw_times,
    )


def load_prediction_case(path: Path | str, *, role: str | None = None) -> PredictionCaseData:
    """Load only the metadata and shock trajectory needed for prediction."""
    path = Path(path)
    if not path.is_file():
        raise CaseDataError(f"case cache {path} does not exist")
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise CaseDataError(f"{path}: prediction cache has no metadata")
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("format") != "spherical_blast_pinn.contact_air_case":
            raise CaseDataError(f"{path}: not a contact-air case cache")
        if metadata.get("version") != 7:
            raise CaseDataError(
                f"{path}: cache version {metadata.get('version')!r} is not supported "
                "for boundary-only prediction; rebuild it with build-cache"
            )
        for key in ("shock_times", "shock_radius", "shock_valid"):
            if key not in archive.files:
                raise CaseDataError(f"{path}: prediction cache has no {key}")
        case_meta = metadata["case"]
        precursor = dict(metadata["precursor"])
        shock = ShockTrajectory.from_arrays(
            case_id=case_meta["case_id"],
            times=np.asarray(archive["shock_times"]),
            radii=np.asarray(archive["shock_radius"]),
            valid=np.asarray(archive["shock_valid"]),
        )
    t0 = float(case_meta.get("t0", precursor["t0"]))
    if not math.isclose(t0, float(precursor["t0"]), rel_tol=0.0, abs_tol=1.0e-12):
        raise CaseDataError(
            f"{path}: cached t0={t0!r} disagrees with precursor t0={precursor['t0']!r}"
        )
    initial = build_initial_geometry(
        t0=t0, Rc0=float(precursor["Rc0"]), Rs0=float(precursor["Rs0"]),
        R_sep=float(precursor["R_sep"]), u_c_sep=float(precursor["u_c_sep"]),
        D_s_sep=float(precursor["D_s_sep"]), shock=shock,
    )
    span = shock.continuous_interval_containing(t0)
    source = dict(case_meta.get("source", {}))
    source["prediction_cache"] = str(path)
    source["prediction_fields_read"] = ["metadata", "shock_times", "shock_radius", "shock_valid"]
    return PredictionCaseData(
        case_id=str(case_meta["case_id"]),
        charge_radius_m=float(case_meta["charge_radius_m"]),
        precursor=precursor, shock=shock, initial=initial,
        t_end=float(span[1]), role=str(role or case_meta.get("role", "test")),
        tolerances=dict(case_meta.get("tolerances", {})),
        event_alignment=dict(case_meta.get("event_alignment", {})),
        source=source,
        notes=list(case_meta.get("notes", [])) + [
            "boundary-only prediction load: no internal observations or contact "
            "interval was read"
        ],
    )


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def synthetic_radial_case(
    *,
    case_id: str,
    charge_radius_m: float,
    precursor: PrecursorResult,
    n_frames: int = 21,
    n_radial: int = 2400,
    duration_s: float | None = None,
) -> RadialCase:
    """Construct a 1-D radial fixture for interface and numerical tests.

    The fields are analytic: a central product region, a narrow mixing band,
    a post-shock air plateau and a finite-width shock.  This is an engineering
    fixture, clearly labelled ``synthetic``; it is not TNT validation data.
    """
    t0 = float(precursor.t0)
    duration = float(duration_s if duration_s is not None else 2.0e-5)
    raw_times = t0 + np.arange(int(n_frames), dtype=np.float64) * (
        duration / max(int(n_frames) - 1, 1)
    )
    rc0 = float(precursor.Rc0)
    rs0 = float(precursor.Rs0)
    r_max = max(
        (float(precursor.Rs0) + float(precursor.D_s_sep) * duration) * 1.12,
        float(charge_radius_m) * 2.2,
    )
    r_min = max(1.0e-5, 0.05 * rc0)
    boundaries = np.linspace(r_min, r_max, int(n_radial) + 1)
    centers = 0.5 * (boundaries[:-1] + boundaries[1:])
    dr = float(boundaries[1] - boundaries[0])
    frames: list[RadialFrame] = []
    air_state = (float(precursor.rho_a_sep), float(precursor.u_a_sep), float(precursor.p_sep))
    ambient = (float(precursor.air_density), 0.0, float(precursor.air_pressure))
    for index, time_raw in enumerate(raw_times):
        dt = float(time_raw - t0)
        rc = rc0 + float(precursor.u_c_sep) * dt
        rs = rs0 + float(precursor.D_s_sep) * dt
        band = max(2.0 * dr, 0.002 * rs0)
        alpha_product = np.clip((rc + 0.5 * band - centers) / band, 0.0, 1.0)
        rho = np.full(centers.size, ambient[0])
        u = np.zeros(centers.size)
        p = np.full(centers.size, ambient[2])
        # Post-shock plateau over the layer interior, finite-width shock at R_s.
        inner = centers <= rs - 3.0 * dr
        transition = (centers > rs - 3.0 * dr) & (centers <= rs)
        rho[inner], u[inner], p[inner] = air_state
        weight = _smoothstep((rs - centers[transition]) / (3.0 * dr))
        for field, value, target in (
            (rho, air_state[0], ambient[0]),
            (u, air_state[1], ambient[1]),
            (p, air_state[2], ambient[2]),
        ):
            field[transition] = target + (value - target) * weight
        product = centers < rc
        p[product] = float(precursor.p_c_t0)
        rho[product] = float(precursor.rho_a_sep)
        u[product] = float(precursor.u_c_sep)
        material = np.where(alpha_product >= 0.95, 0, np.where(alpha_product <= 0.05, 1, 2))
        valid = np.ones(centers.size, dtype=bool)
        frames.append(RadialFrame(
            case_id=case_id, charge_radius_m=float(charge_radius_m),
            frame_index=index, time_raw_s=float(time_raw),
            cell_id=np.arange(centers.size, dtype=np.int64),
            r_left=boundaries[:-1].copy(), r_right=boundaries[1:].copy(),
            radius=centers.copy(), alpha_product=alpha_product,
            alpha_air=1.0 - alpha_product, material=material,
            density=rho, velocity=u, pressure=p,
            valid_density=valid.copy(), valid_velocity=valid.copy(),
            valid_pressure=valid.copy(), position_type="cell_center",
            units={"length": "m", "time": "s", "density": "kg/m^3",
                   "velocity": "m/s", "pressure": "Pa"},
            source={"synthetic": True, "note": "analytic 1-D engineering fixture"},
        ))
    return RadialCase(
        case_id=case_id, charge_radius_m=float(charge_radius_m), frames=frames,
        units={"length": "m", "time": "s", "density": "kg/m^3",
               "velocity": "m/s", "pressure": "Pa"},
        position_type="cell_center",
        source={
            "synthetic": True,
            "note": "analytic 1-D engineering fixture, not TNT data",
            "reader_version": "synthetic/1",
        },
    )


def synthetic_case(
    *, case_id: str, charge_radius_m: float, precursor: PrecursorResult,
    role: str, n_frames: int = 21, n_radial: int = 2400,
    duration_s: float | None = None, tolerances: Mapping[str, Any] | None = None,
    contact_config: ContactLabelConfig | None = None,
    shock_config: ShockDetectionConfig | None = None,
) -> CaseData:
    """Synthetic case used only for engineering tests and the smoke run."""
    config = contact_config or ContactLabelConfig()
    shock_cfg = shock_config or ShockDetectionConfig(
        relative_prominence=4.0, min_cells_from_contact=1,
        min_pressure_jump_fraction=1.0e-2,
    )
    tol = dict(tolerances or {
        "ambient_pressure_pa": precursor.air_pressure,
        "ambient_density_kg_m3": precursor.air_density,
    })
    source = synthetic_radial_case(
        case_id=case_id, charge_radius_m=charge_radius_m, precursor=precursor,
        n_frames=n_frames, n_radial=n_radial, duration_s=duration_s,
    )
    return build_case(
        source, precursor=precursor, role=role, tolerances=tol,
        contact_config=config, shock_config=shock_cfg,
        source_meta={
            "synthetic": True,
            "note": "analytic 1-D engineering fixture, not TNT data",
        },
        notes=["synthetic engineering fixture; do not report as real TNT validation"],
    )


__all__ = [
    "AirObservationStore",
    "CaseData",
    "CaseDataError",
    "ContactLabelConfig",
    "PredictionCaseData",
    "build_air_observations",
    "build_case",
    "load_case",
    "load_prediction_case",
    "save_case",
    "synthetic_case",
    "synthetic_radial_case",
]
