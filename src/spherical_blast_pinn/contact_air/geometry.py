"""Shock trajectory, contact intervals and initial geometry.

The moving air domain is ``R_c(t) <= r <= R_s(t)``.  ``R_s`` is the **outer
transmitted shock** identified from the pressure gradient in each raw radial
profile; it is not an "environmental boundary" and it is not the simulation
outer boundary ``R_outer``.  Every query must lie inside a continuous,
strictly increasing segment of detected shock positions: no extrapolation, no
gap filling, and no copying of the previous frame.

``ShockTrajectory`` stores SciPy PCHIP coefficients as fixed external data and
also evaluates both the position and its analytic derivative in PyTorch.  The
torch path is required because ``R_s(t)`` enters ``R_c`` and ``L_dot``; wrapping
a NumPy lookup of a detached ``t`` would silently drop ``dR_s/dt``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator


class GeometryError(RuntimeError):
    """A geometry trajectory is missing, discontinuous or out of order."""


def _pchip_coefficients(
    times: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(breakpoints, c)`` with ``c`` in SciPy PPoly convention.

    ``S(dx) = c[0] dx^3 + c[1] dx^2 + c[2] dx + c[3]`` on each interval,
    where ``dx = t - x[i]``.  The coefficients are frozen data; they are not
    trained and they are not a denoiser.
    """
    interpolator = PchipInterpolator(times, radii)
    return np.asarray(interpolator.x, dtype=np.float64), np.asarray(
        interpolator.c, dtype=np.float64
    )


@dataclass(frozen=True)
class PchipSegment:
    start: float
    end: float
    x: np.ndarray
    c: np.ndarray

    @classmethod
    def build(cls, times: np.ndarray, radii: np.ndarray) -> "PchipSegment":
        x, c = _pchip_coefficients(times, radii)
        return cls(start=float(x[0]), end=float(x[-1]), x=x, c=c)

    def position_numpy(self, values: np.ndarray) -> np.ndarray:
        out = np.empty(values.shape, dtype=np.float64)
        for index, value in enumerate(values):
            if not (self.start <= float(value) <= self.end):
                raise GeometryError(f"t={float(value)!r} is outside the shock segment")
            i = int(np.searchsorted(self.x, float(value), side="right")) - 1
            i = min(max(i, 0), self.x.size - 2)
            dx = float(value) - float(self.x[i])
            cc = self.c[:, i]
            out[index] = ((cc[0] * dx + cc[1]) * dx + cc[2]) * dx + cc[3]
        return out

    def derivative_numpy(self, values: np.ndarray) -> np.ndarray:
        out = np.empty(values.shape, dtype=np.float64)
        for index, value in enumerate(values):
            if not (self.start <= float(value) <= self.end):
                raise GeometryError(f"t={float(value)!r} is outside the shock segment")
            i = int(np.searchsorted(self.x, float(value), side="right")) - 1
            i = min(max(i, 0), self.x.size - 2)
            dx = float(value) - float(self.x[i])
            cc = self.c[:, i]
            out[index] = (3.0 * cc[0] * dx + 2.0 * cc[1]) * dx + cc[2]
        return out

    def evaluate_torch(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Position and analytic derivative as functions of ``t``.

        ``t`` may require grad: the polynomial and its derivative are expressed
        directly in ``t`` (not by detaching and querying NumPy).  The interval
        index is selected on detached values only, which is the standard
        piecewise-polynomial convention.
        """
        flat = t.reshape(-1)
        inside = (flat >= self.start) & (flat <= self.end)
        if not bool(inside.all()):
            bad = flat.detach()[~inside]
            raise GeometryError(
                "shock trajectory is not defined at "
                f"t={bad[:3].tolist()}; extrapolation and gap filling are forbidden"
            )
        x = torch.as_tensor(self.x, dtype=flat.dtype, device=flat.device)
        c = torch.as_tensor(self.c, dtype=flat.dtype, device=flat.device)
        index = torch.searchsorted(x, flat.detach(), right=True) - 1
        index = torch.clamp(index, 0, x.numel() - 2)
        dx = flat - x[index]
        cc = c[:, index]
        position = ((cc[0] * dx + cc[1]) * dx + cc[2]) * dx + cc[3]
        derivative = (3.0 * cc[0] * dx + 2.0 * cc[1]) * dx + cc[2]
        return position, derivative

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_s": self.start,
            "end_s": self.end,
            "n_breakpoints": int(self.x.size),
            "interpolation": "PCHIP (shape preserving); coefficients fixed",
        }


@dataclass
class ShockTrajectory:
    case_id: str
    times: np.ndarray
    radii: np.ndarray
    valid: np.ndarray
    segments: list[PchipSegment] = field(default_factory=list)
    isolated: list[tuple[float, float]] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    invalid_frames: list[dict[str, Any]] = field(default_factory=list)
    plateaus: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        radii = np.asarray(self.radii, dtype=np.float64).reshape(-1)
        valid = np.asarray(self.valid, dtype=bool).reshape(-1)
        if times.size != radii.size or times.size != valid.size:
            raise GeometryError("times, radii and valid masks must have the same length")
        if not np.all(np.isfinite(times)):
            raise GeometryError("shock times must all be finite")
        if times.size and np.any(np.diff(times) <= 0.0):
            raise GeometryError(
                "shock times must be strictly increasing; duplicate or retrograde "
                "timestamps are not repaired"
            )
        requested_valid = valid.copy()
        usable = requested_valid & np.isfinite(radii) & (radii > 0.0)
        self.times = times
        self.radii = radii
        self.valid = usable
        self.segments = []
        self.isolated = []
        self.rejected = []
        self.invalid_frames = []
        self.plateaus = []
        for index in np.flatnonzero(requested_valid & ~usable):
            reason = "non-finite radius" if not np.isfinite(radii[index]) else "non-positive radius"
            self.invalid_frames.append({
                "index": int(index),
                "time_s": float(times[index]),
                "reason": reason,
            })
        start = None
        for index, flag in enumerate(np.append(usable, False)):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                run = slice(start, index)
                _t = times[run]
                _r = radii[run]
                if _t.size >= 2:
                    dr = np.diff(_r)
                    if np.any(dr < 0.0):
                        self.rejected.append(
                            f"segment {float(_t[0]):.6g}..{float(_t[-1]):.6g}: R_s "
                            "decreases at one or more frames; a retrograde shock "
                            "trajectory is not repaired and no PCHIP is built"
                        )
                    else:
                        plateau_count = int(np.count_nonzero(dr == 0.0))
                        if plateau_count:
                            self.plateaus.append({
                                "start_s": float(_t[0]),
                                "end_s": float(_t[-1]),
                                "n_equal_intervals": plateau_count,
                            })
                        self.segments.append(PchipSegment.build(_t, _r))
                elif _t.size == 1:
                    self.isolated.append((float(_t[0]), float(_r[0])))
                start = None
        self.segments.sort(key=lambda item: item.start)
        self.isolated.sort(key=lambda item: item[0])

    @classmethod
    def from_arrays(
        cls, *, case_id: str, times: Sequence[float], radii: Sequence[float],
        valid: Sequence[bool] | None = None,
    ) -> "ShockTrajectory":
        return cls(
            case_id=str(case_id),
            times=np.asarray(times, dtype=np.float64),
            radii=np.asarray(radii, dtype=np.float64),
            valid=np.ones(len(times), dtype=bool) if valid is None else np.asarray(valid, bool),
        )

    @property
    def available(self) -> bool:
        return bool(self.segments or self.isolated)

    @property
    def domain(self) -> tuple[float, float] | None:
        """Outer bounds of the valid samples, **not** a coverage interval."""
        if not self.segments:
            return None
        return self.segments[0].start, self.segments[-1].end

    @property
    def domain_min(self) -> float:
        domain = self.domain
        if domain is None:
            raise GeometryError(f"case {self.case_id!r}: no continuous R_s segment")
        return domain[0]

    @property
    def domain_max(self) -> float:
        domain = self.domain
        if domain is None:
            raise GeometryError(f"case {self.case_id!r}: no continuous R_s segment")
        return domain[1]

    def _segment_at(self, t: float) -> PchipSegment | None:
        for segment in self.segments:
            if segment.start <= t <= segment.end:
                return segment
        return None

    def _segment_span_at(self, t: float) -> tuple[float, float] | None:
        for segment in self.segments:
            if segment.start <= t <= segment.end:
                return float(segment.start), float(segment.end)
        return None

    @property
    def continuous_intervals(self) -> list[tuple[float, float]]:
        """Measured ``R_s`` runs.  There is no interval across a gap."""
        return [(float(segment.start), float(segment.end)) for segment in self.segments]

    def has_slope(self, t: float) -> bool:
        return self._segment_at(float(t)) is not None

    def is_covered(self, t: float) -> bool:
        return self.has_slope(t)

    def continuous_interval_containing(self, t: float) -> tuple[float, float]:
        span = self._segment_span_at(float(t))
        if span is None:
            if any(math.isclose(time, float(t), rel_tol=0.0, abs_tol=0.0)
                   for time, _ in self.isolated):
                raise GeometryError(
                    f"case {self.case_id!r}: t={t!r} s is an isolated R_s frame; "
                    "it has no slope and cannot define a solve interval"
                )
            raise GeometryError(
                f"case {self.case_id!r}: t={t!r} s is outside every continuous "
                "R_s segment; gap filling and extrapolation are forbidden"
            )
        return span

    def require_interval(self, start: float, end: float) -> tuple[float, float]:
        start = float(start)
        end = float(end)
        if not (math.isfinite(start) and math.isfinite(end)):
            raise GeometryError(f"case {self.case_id!r}: solve interval is not finite")
        if end < start:
            raise GeometryError(
                f"case {self.case_id!r}: solve interval end={end!r} is before start={start!r}"
            )
        span = self.continuous_interval_containing(start)
        if end < span[0] or end > span[1]:
            raise GeometryError(
                f"case {self.case_id!r}: solve interval [{start!r}, {end!r}] crosses "
                f"the continuous R_s segment {span!r}; a single connected segment is required"
            )
        return span

    def position(self, t: float | np.ndarray) -> np.ndarray | float:
        values = np.asarray(t, dtype=np.float64)
        flat = values.reshape(-1)
        out = np.full(flat.shape, np.nan, dtype=np.float64)
        for index, value in enumerate(flat):
            segment = self._segment_at(float(value))
            if segment is not None:
                out[index] = float(segment.position_numpy(np.asarray([float(value)]))[0])
                continue
            exact = [radius for time, radius in self.isolated
                     if math.isclose(time, float(value), rel_tol=0.0, abs_tol=0.0)]
            if exact:
                out[index] = float(exact[0])
        if np.any(~np.isfinite(out)):
            missing = flat[~np.isfinite(out)]
            raise GeometryError(
                f"case {self.case_id!r}: R_s is not defined at t={missing[:3].tolist()} s; "
                "extrapolation and gap filling are forbidden"
            )
        return out[0] if values.ndim == 0 else out.reshape(values.shape)

    def derivative(self, t: float | np.ndarray) -> np.ndarray | float:
        values = np.asarray(t, dtype=np.float64)
        flat = values.reshape(-1)
        out = np.full(flat.shape, np.nan, dtype=np.float64)
        for index, value in enumerate(flat):
            segment = self._segment_at(float(value))
            if segment is None:
                raise GeometryError(
                    f"case {self.case_id!r}: dR_s/dt is not defined at t={float(value)!r} s; "
                    "an isolated point cannot provide a slope"
                )
            out[index] = float(segment.derivative_numpy(np.asarray([float(value)]))[0])
        return out[0] if values.ndim == 0 else out.reshape(values.shape)

    def position_and_derivative(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.position(t), self.derivative(t)

    def position_and_derivative_torch(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Torch PCHIP value and analytic derivative for every valid ``t``.

        No SciPy/NumPy lookup of a detached tensor is used: the polynomial is
        evaluated in torch, so a ``t`` that requires grad still has the correct
        derivative.  The returned derivative is the derivative of the PCHIP
        interpolant, not a second application of autograd.
        """
        flat = t.reshape(-1)
        position = torch.full_like(flat, float("nan"))
        derivative = torch.full_like(flat, float("nan"))
        covered = torch.zeros(flat.shape, dtype=torch.bool, device=flat.device)
        for segment in self.segments:
            inside = (flat.detach() >= segment.start) & (flat.detach() <= segment.end)
            if not bool(inside.any()):
                continue
            selected = flat[inside]
            values, slopes = segment.evaluate_torch(selected)
            position = position.masked_scatter(inside, values)
            derivative = derivative.masked_scatter(inside, slopes)
            covered = covered | inside
        if not bool(covered.all()):
            missing = flat.detach()[~covered]
            raise GeometryError(
                f"case {self.case_id!r}: R_s(t) is not defined at "
                f"t={missing[:3].tolist()} s; extrapolation and gap filling are forbidden"
            )
        return position, derivative

    def values_torch(self, t: torch.Tensor) -> torch.Tensor:
        return self.position_and_derivative_torch(t)[0]

    def require_covering(self, t: float) -> None:
        if self._segment_at(float(t)) is not None:
            return
        if any(math.isclose(time, float(t), rel_tol=0.0, abs_tol=0.0)
               for time, _ in self.isolated):
            raise GeometryError(
                f"case {self.case_id!r}: R_s({t!r}) is an isolated frame; position is "
                "known but dR_s/dt is not, so the initial geometry is incomplete"
            )
        raise GeometryError(
            f"case {self.case_id!r}: t0={t!r} s is outside every continuous R_s segment; "
            "the initial condition cannot be built without gap filling or extrapolation"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "quantity": "R_s (outer transmitted shock), not R_outer and not R_b",
            "n_samples": int(self.times.size),
            "n_valid": int(np.count_nonzero(self.valid)),
            "segments": [segment.as_dict() for segment in self.segments],
            "continuous_intervals": [
                [float(a), float(b)] for a, b in self.continuous_intervals
            ],
            "isolated": [[float(t), float(r)] for t, r in self.isolated],
            "outer_bounds": (
                None if self.domain is None
                else [float(self.domain[0]), float(self.domain[1])]
            ),
            "invalid_frames": list(self.invalid_frames),
            "plateaus": list(self.plateaus),
            "rejected": list(self.rejected),
        }


@dataclass
class ContactIntervalTable:
    case_id: str
    times: np.ndarray
    low: np.ndarray
    up: np.ndarray
    raw: np.ndarray
    valid: np.ndarray
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        self.low = np.asarray(self.low, dtype=np.float64).reshape(-1)
        self.up = np.asarray(self.up, dtype=np.float64).reshape(-1)
        self.raw = np.asarray(self.raw, dtype=np.float64).reshape(-1)
        self.valid = np.asarray(self.valid, dtype=bool).reshape(-1)
        n = self.times.size
        if any(array.size != n for array in (self.low, self.up, self.raw, self.valid)):
            raise GeometryError("contact interval arrays must have the same length")
        if n and (not np.all(np.isfinite(self.times)) or np.any(np.diff(self.times) <= 0.0)):
            raise GeometryError(
                "contact interval times must be finite and strictly increasing"
            )
        finite = np.isfinite(self.low) & np.isfinite(self.up) & np.isfinite(self.raw)
        bad = self.valid & (~finite | (self.low > self.up) | (self.low <= 0.0))
        if np.any(bad):
            indices = np.flatnonzero(bad)[:3].tolist()
            raise GeometryError(
                f"case {self.case_id!r}: invalid contact intervals at frames {indices}; "
                "R_c,low must be positive and <= R_c,up"
            )
        outside = self.valid & ((self.raw < self.low) | (self.raw > self.up))
        if np.any(outside):
            raise GeometryError(
                f"case {self.case_id!r}: R_raw lies outside [R_c,low, R_c,up] at "
                f"{np.flatnonzero(outside)[:3].tolist()}"
            )
        if self.reasons and len(self.reasons) != n:
            raise GeometryError("contact interval reasons must have one entry per frame")

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))

    def indices(self) -> np.ndarray:
        return np.flatnonzero(self.valid)

    def at(self, index: int) -> tuple[float, float, float, float]:
        return (
            float(self.times[index]),
            float(self.low[index]),
            float(self.up[index]),
            float(self.raw[index]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "n_frames": int(self.times.size),
            "n_valid": self.n_valid,
            "time_range": (
                None if not self.times.size
                else [float(self.times.min()), float(self.times.max())]
            ),
            "definition": (
                "[R_c,low, R_c,up] is the interval between the actual cell "
                "boundaries bracketing the contact mixing segment. The midpoint "
                "R_raw is an observation-mapping convention only and is never a "
                "single-point label."
            ),
        }


@dataclass(frozen=True)
class InitialGeometry:
    t0: float
    Rc0: float
    Rs0: float
    zeta_s0: float
    R_sep: float
    u_c_sep: float
    D_s_sep: float
    Rs_data_at_t0: float


def build_initial_geometry(
    *,
    t0: float,
    Rc0: float,
    Rs0: float,
    R_sep: float,
    u_c_sep: float,
    D_s_sep: float,
    shock: ShockTrajectory,
) -> InitialGeometry:
    """Read ``R_s(t0)`` and require ``0 < R_c0 < R_s(t0)``.

    There is no ``R_b`` here: the geometric right boundary of the PINN domain is
    the identified shock.  The theoretical ``Rs0`` is recorded against the
    measured ``R_s(t0)``; a difference is reported, not removed.
    """
    shock.require_covering(float(t0))
    Rs_data = float(shock.position(float(t0)))
    for name, value in (
        ("t0", float(t0)), ("Rc0", float(Rc0)), ("Rs0", float(Rs0)),
        ("R_s_data(t0)", Rs_data),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise GeometryError(f"initial geometry {name}={value!r} is not finite and positive")
    if Rc0 >= Rs_data:
        raise GeometryError(
            f"case {shock.case_id!r}: theoretical R_c(t0)={Rc0!r} is not below the "
            f"measured shock R_s(t0)={Rs_data!r}; R_c = R_s sigmoid(raw) cannot "
            "contain the theoretical initial contact position"
        )
    zeta_s0 = (Rs0 - Rc0) / (Rs_data - Rc0)
    if not math.isfinite(zeta_s0):
        raise GeometryError("initial shock coordinate zeta_s0 is not finite")
    return InitialGeometry(
        t0=float(t0), Rc0=float(Rc0), Rs0=float(Rs0),
        zeta_s0=float(zeta_s0), R_sep=float(R_sep),
        u_c_sep=float(u_c_sep), D_s_sep=float(D_s_sep),
        Rs_data_at_t0=float(Rs_data),
    )


def split_valid_segments(times: Iterable[float], valid: Iterable[bool]) -> list[tuple[int, int]]:
    """Half-open index ranges of consecutive true entries."""
    times = list(times)
    valid = list(valid)
    segments: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(valid + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            segments.append((start, index))
            start = None
    return segments


__all__ = [
    "ContactIntervalTable",
    "GeometryError",
    "InitialGeometry",
    "PchipSegment",
    "ShockTrajectory",
    "build_initial_geometry",
    "split_valid_segments",
]
