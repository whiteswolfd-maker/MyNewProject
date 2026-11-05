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

    def sample(self, count: int, generator) -> dict[str, Any]:
        if self.size == 0 or count <= 0:
            empty = _tensor(np.zeros(0))
            return {
                "t": empty, "zeta": empty, "rho": empty, "u": empty, "p": empty,
                "valid_rho": empty.bool(), "valid_u": empty.bool(),
                "valid_p": empty.bool(), "count": 0,
            }
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        indices = torch.randint(0, self.size, (int(count),), generator=generator)
        idx = indices.cpu().numpy()
        return {
            "t": _tensor(self.time[idx]),
            "zeta": _tensor(self.zeta[idx]),
            "rho": _tensor(self.density[idx]),
            "u": _tensor(self.velocity[idx]),
            "p": _tensor(self.pressure[idx]),
            "valid_rho": _tensor(self.valid_density[idx]).bool(),
            "valid_u": _tensor(self.valid_velocity[idx]).bool(),
            "valid_p": _tensor(self.valid_pressure[idx]).bool(),
            "count": int(indices.numel()),
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
        if not (self.boundary.domain_min <= self.initial.t0 and self.t_end <= self.boundary.domain_max):
            raise CaseDataError(
                f"case {self.case_id!r}: [t0,t_end] is not inside the R_b domain "
                f"{self.boundary.domain}"
            )

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

    def scale_dict(self) -> dict[str, float]:
        return {
            "rho": self.rho_scale,
            "u": self.u_scale,
            "p": self.p_scale,
            "e": self.e_scale,
            "length": self.length_scale,
            "gamma": self.gamma_air,
        }

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
            "p_c_at_t0_pa": self.p_c_at_t0,
            "p_a_sep_pa": self.p_scale,
            "p_c_vs_p_a_sep_relative": abs(self.p_c_at_t0 - self.p_scale) / self.p_scale,
            "tolerances": dict(self.tolerances),
            "source": dict(self.source),
            "notes": list(self.notes),
            "separation": dict(self.separation),
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
) -> tuple[float, bool, str]:
    """Find the first outward cell that is ambient and has an ambient neighbour."""
    r = np.asarray(radius, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    u = np.asarray(velocity, dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)
    mat = np.asarray(material)
    good = (
        np.isfinite(r) & np.isfinite(p) & np.isfinite(u) & np.isfinite(rho)
        & (mat == 1)
        & (np.abs(p - ambient_pressure) <= tol_pressure)
        & (np.abs(rho - ambient_density) <= tol_density)
        & (np.abs(u) <= tol_velocity)
    )
    indices = np.flatnonzero(good)
    if indices.size == 0:
        return float("nan"), False, "no air cell satisfies the ambient tolerances"
    for index in indices:
        # Require the requested number of consecutive ambient cells starting at
        # this radius and continuing outward.  An isolated far-field spike is
        # not a boundary.
        run = 0
        j = index
        while j < r.size and good[j]:
            run += 1
            j += 1
        if run >= int(min_ambient_cells):
            return float(r[index]), True, "first outward ambient cell with a consecutive run"
    return float("nan"), False, "ambient cells are isolated; no consecutive run"


def contact_interval_from_material(
    radius: np.ndarray, material: np.ndarray
) -> tuple[float, float, float, bool, str]:
    """Derive ``(low, up, raw, valid, reason)`` from one frame's shell labels."""
    r = np.asarray(radius, dtype=np.float64)
    mat = np.asarray(material)
    mixed = mat == 2
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

    if arrays.rb_times is None:
        rb_t, rb_r, rb_v, reasons = [], [], [], []
        for frame, t in enumerate(times):
            value, ok, reason = detect_right_boundary(
                arrays.radius, arrays.pressure[frame], arrays.velocity[frame],
                arrays.density[frame], arrays.material[frame],
                ambient_pressure=float(tolerances["ambient_pressure_pa"]),
                ambient_density=float(tolerances["ambient_density_kg_m3"]),
                tol_pressure=float(tolerances["pressure_pa"]),
                tol_density=float(tolerances["density_kg_m3"]),
                tol_velocity=float(tolerances["velocity_m_s"]),
                min_ambient_cells=int(tolerances.get("min_ambient_cells", 2)),
            )
            rb_t.append(t); rb_r.append(value); rb_v.append(ok); reasons.append(reason)
        boundary = RightBoundaryTrajectory.from_arrays(
            case_id=arrays.case_id, times=rb_t, radii=rb_r, valid=rb_v
        )
        notes.append(f"right boundary derived from ambient tolerances: {sorted(set(reasons))}")
    else:
        boundary = RightBoundaryTrajectory.from_arrays(
            case_id=arrays.case_id, times=arrays.rb_times,
            radii=arrays.rb_radius, valid=arrays.rb_valid,
        )
    initial = build_initial_geometry(
        t0=precursor.t0, Rc0=precursor.Rc0, Rs0=precursor.Rs0,
        R_sep=precursor.R_sep, u_c_sep=precursor.u_c_sep, D_s_sep=precursor.D_s_sep,
        boundary=boundary,
    )
    obs = build_air_observations(arrays, contact, boundary)
    resolved_times = times[contact.valid]
    if resolved_times.size == 0:
        separation = {
            "status": "no_resolved_contact_frame",
            "first_resolved_time_s": None,
            "t0_s": float(precursor.t0),
            "note": "no frame has a single bracketed mixed segment; contact data "
                    "supervision is unavailable until the extraction is improved",
        }
    else:
        first = float(resolved_times.min())
        separation = {
            "status": (
                "contact_resolved_at_or_before_t0" if first <= precursor.t0
                else "first_resolved_after_t0"
            ),
            "first_resolved_time_s": first,
            "t0_s": float(precursor.t0),
            "n_resolved_frames": int(resolved_times.size),
            "note": (
                "separation evidence is the first frame with a single mixed segment "
                "bracketed by pure product and pure air; no 0.5/PLIC single-point label"
            ),
        }
    end = float(boundary.domain_max if t_end is None else t_end)
    if obs.size:
        end = min(end, float(obs.time.max())) if t_end is None else end
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
        "version": 1,
        "units": {
            "time": "s", "radius": "m", "density": "kg/m^3",
            "velocity": "m/s", "pressure": "Pa", "specific_energy": "J/kg",
        },
        "case": case.as_dict(),
        "precursor": case.precursor,
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
        eligible = np.asarray(archive["contact_has_supervision"], dtype=bool) \
            if "contact_has_supervision" in archive.files else None
    if eligible is None:
        eligible = np.asarray(case.contact_valid, dtype=bool)
    if low is None or up is None:
        low = np.full(case.times.size, np.nan, dtype=np.float64)
        up = np.full(case.times.size, np.nan, dtype=np.float64)
    # Prefer the cells adjacent to the single mixed shell segment: those are
    # the auditable R_c,low/R_c,up representatives.  The band edges are kept
    # as the fallback when shell labels do not expose a single bracketed mixed
    # segment (for example, a very coarse shell grid).
    derived_low = np.full(case.times.size, np.nan)
    derived_up = np.full(case.times.size, np.nan)
    derived_raw = np.full(case.times.size, np.nan)
    derived_ok = np.zeros(case.times.size, dtype=bool)
    for frame in range(case.times.size):
        lo, hi, rr, ok, _ = contact_interval_from_material(
            np.asarray(case.radius), np.asarray(case.material)[frame]
        )
        if ok:
            derived_low[frame], derived_up[frame], derived_raw[frame] = lo, hi, rr
            derived_ok[frame] = True
    low = np.where(derived_ok, derived_low, low)
    up = np.where(derived_ok, derived_up, up)
    raw = np.where(derived_ok, derived_raw, 0.5 * (low + up))
    valid = eligible & np.isfinite(low) & np.isfinite(up) & (low < up)
    field_validity = case.field_validity or {}
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
        source=source or {"profile_dir": str(directory)},
        notes=["built by build_case_from_profile; no alpha=0.5 label was used"],
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
    "build_air_observations",
    "build_case",
    "build_case_from_profile",
    "contact_interval_from_material",
    "detect_right_boundary",
    "load_case",
    "precursor_from_case_mapping",
    "save_case",
    "synthetic_case",
]
