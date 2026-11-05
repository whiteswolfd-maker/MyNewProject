"""Right-boundary trajectories, contact intervals and initial geometry.

The right boundary ``R_b`` is a measured environmental-side trajectory.  It is
not the shock ``R_s`` and it is not extrapolated: every query must fall inside
a continuous, strictly increasing valid segment.  A single valid frame gives a
position at that frame only and **no** derivative.

The contact interval ``[R_c,low, R_c,up]`` is the auditable bracket produced by
the extractor: the inner pure-product neighbour and the outer pure-air
neighbour around a mixed segment.  Its midpoint ``R_raw`` is an engineering
convention used only for observation mapping, never as a single-point contact
label.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator


class GeometryError(RuntimeError):
    """A geometry trajectory is missing, discontinuous or out of order."""


@dataclass
class RightBoundaryTrajectory:
    case_id: str
    times: np.ndarray
    radii: np.ndarray
    valid: np.ndarray
    segments: list[tuple[float, float, PchipInterpolator]] = field(default_factory=list)
    isolated: list[tuple[float, float]] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    #: Frames that were supplied with ``valid=True`` but whose radius is not a
    #: usable positive finite number.  They remain in the time axis as break
    #: markers; nothing is deleted and the two sides are never joined.
    invalid_frames: list[dict[str, Any]] = field(default_factory=list)
    #: Continuous runs in which consecutive radii are equal.  A plateau is a
    #: measured shape, not an error: it is retained and reported.  A strictly
    #: decreasing radius is a different thing (a retrograde trajectory) and the
    #: whole segment is rejected instead.
    plateaus: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        radii = np.asarray(self.radii, dtype=np.float64).reshape(-1)
        valid = np.asarray(self.valid, dtype=bool).reshape(-1)
        if times.size != radii.size or times.size != valid.size:
            raise GeometryError("times, radii and valid masks must have the same length")
        # Do not sort or drop rows.  The frame index is part of the data
        # contract, and silently reordering it would make a gap invisible.
        if not np.all(np.isfinite(times)):
            raise GeometryError("right-boundary times must all be finite")
        if times.size and np.any(np.diff(times) <= 0.0):
            raise GeometryError(
                "right-boundary times must be strictly increasing; duplicate or "
                "retrograde timestamps are not repaired"
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
                            f"segment {float(_t[0]):.6g}..{float(_t[-1]):.6g}: R_b "
                            "decreases at one or more frames; a retrograde trajectory "
                            "is not repaired and no PCHIP is built"
                        )
                    else:
                        plateau_count = int(np.count_nonzero(dr == 0.0))
                        if plateau_count:
                            self.plateaus.append({
                                "start_s": float(_t[0]),
                                "end_s": float(_t[-1]),
                                "n_equal_intervals": plateau_count,
                            })
                        self.segments.append(
                            (float(_t[0]), float(_t[-1]), PchipInterpolator(_t, _r))
                        )
                elif _t.size == 1:
                    self.isolated.append((float(_t[0]), float(_r[0])))
                start = None
        self.segments.sort(key=lambda item: item[0])
        self.isolated.sort(key=lambda item: item[0])

    @classmethod
    def from_arrays(
        cls, *, case_id: str, times: Sequence[float], radii: Sequence[float],
        valid: Sequence[bool] | None = None,
    ) -> "RightBoundaryTrajectory":
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
        return self.segments[0][0], self.segments[-1][1]

    @property
    def domain_min(self) -> float:
        """First valid sample time; do not use as a solve-interval start."""
        domain = self.domain
        if domain is None:
            raise GeometryError(f"case {self.case_id!r}: no continuous R_b segment")
        return domain[0]

    @property
    def domain_max(self) -> float:
        """Last valid sample time; do not use as a solve-interval end."""
        domain = self.domain
        if domain is None:
            raise GeometryError(f"case {self.case_id!r}: no continuous R_b segment")
        return domain[1]

    def _segment_at(self, t: float):
        for start, end, interpolator in self.segments:
            if start <= t <= end:
                return interpolator
        return None

    def _segment_span_at(self, t: float) -> tuple[float, float] | None:
        for start, end, _ in self.segments:
            if start <= t <= end:
                return float(start), float(end)
        return None

    @property
    def continuous_intervals(self) -> list[tuple[float, float]]:
        """Measured ``R_b`` runs.  There is no interval across a gap."""
        return [(float(start), float(end)) for start, end, _ in self.segments]

    def has_slope(self, t: float) -> bool:
        return self._segment_at(float(t)) is not None

    def is_covered(self, t: float) -> bool:
        return self.has_slope(t)

    def continuous_interval_containing(self, t: float) -> tuple[float, float]:
        """Return the one continuous segment that contains ``t``.

        The outer ``domain`` is deliberately *not* used for this: the earliest
        and latest valid frames can enclose a gap, and a solve interval must
        not be chosen by those two endpoints alone.
        """
        span = self._segment_span_at(float(t))
        if span is None:
            if any(math.isclose(time, float(t), rel_tol=0.0, abs_tol=0.0)
                   for time, _ in self.isolated):
                raise GeometryError(
                    f"case {self.case_id!r}: t={t!r} s is an isolated R_b frame; "
                    "it has no slope and cannot define a solve interval"
                )
            raise GeometryError(
                f"case {self.case_id!r}: t={t!r} s is outside every continuous "
                "R_b segment; gap filling and extrapolation are forbidden"
            )
        return span

    def require_interval(self, start: float, end: float) -> tuple[float, float]:
        """Require ``[start, end]`` to lie inside one continuous segment."""
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
                f"the continuous R_b segment {span!r}; a single connected segment is "
                "required"
            )
        # Every t between the two endpoints is inside the same PCHIP segment.
        return span

    def position(self, t: float | np.ndarray) -> np.ndarray | float:
        values = np.asarray(t, dtype=np.float64)
        flat = values.reshape(-1)
        out = np.full(flat.shape, np.nan, dtype=np.float64)
        for i, value in enumerate(flat):
            interpolator = self._segment_at(float(value))
            if interpolator is not None:
                out[i] = float(interpolator(float(value)))
                continue
            exact = [radius for time, radius in self.isolated
                     if math.isclose(time, float(value), rel_tol=0.0, abs_tol=0.0)]
            if exact:
                out[i] = float(exact[0])
        if np.any(~np.isfinite(out)):
            missing = flat[~np.isfinite(out)]
            raise GeometryError(
                f"case {self.case_id!r}: R_b is not defined at t={missing[:3].tolist()} s; "
                "extrapolation and gap filling are forbidden"
            )
        return out[0] if values.ndim == 0 else out.reshape(values.shape)

    def derivative(self, t: float | np.ndarray) -> np.ndarray | float:
        values = np.asarray(t, dtype=np.float64)
        flat = values.reshape(-1)
        out = np.full(flat.shape, np.nan, dtype=np.float64)
        for i, value in enumerate(flat):
            interpolator = self._segment_at(float(value))
            if interpolator is None:
                raise GeometryError(
                    f"case {self.case_id!r}: dR_b/dt is not defined at t={float(value)!r} s; "
                    "an isolated point cannot provide a slope"
                )
            out[i] = float(interpolator.derivative()(float(value)))
        return out[0] if values.ndim == 0 else out.reshape(values.shape)

    def position_and_derivative(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.position(t), self.derivative(t)

    def require_covering(self, t: float) -> None:
        if self._segment_at(float(t)) is not None:
            return
        if any(math.isclose(time, float(t), rel_tol=0.0, abs_tol=0.0)
               for time, _ in self.isolated):
            raise GeometryError(
                f"case {self.case_id!r}: R_b({t!r}) is an isolated frame; position is "
                "known but dR_b/dt is not, so the initial geometry is incomplete"
            )
        raise GeometryError(
            f"case {self.case_id!r}: t0={t!r} s is outside every continuous R_b segment; "
            "the initial condition cannot be built without gap filling or extrapolation"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "n_samples": int(self.times.size),
            "n_valid": int(np.count_nonzero(self.valid)),
            "segments": [[float(a), float(b)] for a, b, _ in self.segments],
            "continuous_intervals": [
                [float(a), float(b)] for a, b in self.continuous_intervals
            ],
            "isolated": [[float(t), float(r)] for t, r in self.isolated],
            # Outer bounds only; NOT a statement that every t in between is
            # covered.  Use continuous_intervals / require_interval.
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
        # R_raw is an observation-mapping convention, not a label.  Keep it
        # inside the bracket when possible, but do not move it silently.
        outside = self.valid & ((self.raw < self.low) | (self.raw > self.up))
        if np.any(outside):
            raise GeometryError(
                f"case {self.case_id!r}: R_raw lies outside [R_c,low, R_c,up] at "
                f"{np.flatnonzero(outside)[:3].tolist()}"
            )

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
        }


@dataclass(frozen=True)
class InitialGeometry:
    t0: float
    Rc0: float
    Rs0: float
    Rb0: float
    zeta_s0: float
    R_sep: float
    u_c_sep: float
    D_s_sep: float


def build_initial_geometry(
    *,
    t0: float,
    Rc0: float,
    Rs0: float,
    R_sep: float,
    u_c_sep: float,
    D_s_sep: float,
    boundary: RightBoundaryTrajectory,
) -> InitialGeometry:
    """Read ``R_b(t0)`` and enforce ``R_c(t0) < R_s(t0) < R_b(t0)``."""
    boundary.require_covering(float(t0))
    Rb0 = float(boundary.position(float(t0)))
    values = {
        "t0": float(t0),
        "Rc0": float(Rc0),
        "Rs0": float(Rs0),
        "Rb0": float(Rb0),
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise GeometryError(f"initial geometry {name}={value!r} is not finite and positive")
    if not (Rc0 < Rs0 < Rb0):
        raise GeometryError(
            f"case {boundary.case_id!r}: initial radii must satisfy R_c < R_s < R_b; got "
            f"{Rc0!r}, {Rs0!r}, {Rb0!r}"
        )
    zeta_s0 = (Rs0 - Rc0) / (Rb0 - Rc0)
    if not (0.0 < zeta_s0 < 1.0):
        raise GeometryError(f"initial shock coordinate zeta_s0={zeta_s0!r} is outside (0,1)")
    return InitialGeometry(
        t0=float(t0), Rc0=float(Rc0), Rs0=float(Rs0), Rb0=float(Rb0),
        zeta_s0=float(zeta_s0), R_sep=float(R_sep),
        u_c_sep=float(u_c_sep), D_s_sep=float(D_s_sep),
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
    "RightBoundaryTrajectory",
    "build_initial_geometry",
    "split_valid_segments",
]
