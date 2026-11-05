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

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        radii = np.asarray(self.radii, dtype=np.float64).reshape(-1)
        valid = np.asarray(self.valid, dtype=bool).reshape(-1)
        if times.size != radii.size or times.size != valid.size:
            raise GeometryError("times, radii and valid masks must have the same length")
        finite = np.isfinite(times) & np.isfinite(radii)
        order = np.argsort(times[finite], kind="stable")
        times = times[finite][order]
        radii = radii[finite][order]
        valid = valid[finite][order]
        if times.size and np.any(np.diff(times) <= 0.0):
            raise GeometryError("right-boundary times must be strictly increasing")
        if np.any(valid & ~(radii > 0.0)):
            raise GeometryError("valid right-boundary radii must be positive")
        self.times = times
        self.radii = radii
        self.valid = valid
        self.segments = []
        self.isolated = []
        self.rejected = []
        start = None
        for index, flag in enumerate(np.append(valid, False)):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                run = slice(start, index)
                _t = times[run]
                _r = radii[run]
                if _t.size >= 2:
                    if np.any(np.diff(_r) <= 0.0):
                        self.rejected.append(
                            f"segment {int(_t[0]):.6g}..{int(_t[-1]):.6g}: R_b is not "
                            "strictly increasing; no PCHIP built"
                        )
                    else:
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
        if not self.segments:
            return None
        return self.segments[0][0], self.segments[-1][1]

    @property
    def domain_min(self) -> float:
        domain = self.domain
        if domain is None:
            raise GeometryError(f"case {self.case_id!r}: no continuous R_b segment")
        return domain[0]

    @property
    def domain_max(self) -> float:
        domain = self.domain
        if domain is None:
            raise GeometryError(f"case {self.case_id!r}: no continuous R_b segment")
        return domain[1]

    def _segment_at(self, t: float):
        for start, end, interpolator in self.segments:
            if start <= t <= end:
                return interpolator
        return None

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
            "isolated": [[float(t), float(r)] for t, r in self.isolated],
            "domain": None if self.domain is None else [float(self.domain[0]), float(self.domain[1])],
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
