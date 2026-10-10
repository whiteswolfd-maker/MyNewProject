"""Array-in / array-out metric primitives shared by every evaluator.

Two conventions are enforced here, and they are the reason this module exists
rather than each evaluator rolling its own arithmetic:

**Every metric divides by a fixed physical scale, never by a local mean.**
A blast waveform crosses zero between its positive and negative phases.  A
denominator built from a local (or whole-trace) mean of the reference vanishes
at that crossing — and at the crossing is exactly where the prediction is most
interesting, so a relative error defined that way reports a spike to infinity
at the one place the model is being tested.  A *fixed* physical scale
(``scales.pressure``, a peak overpressure, the charge radius) is constant, has
the units of the quantity it normalises, and stays finite everywhere.  This is
the same argument that :meth:`Scales.tikhonov_epsilon` records for the loss
side; the evaluator must not contradict the loss.

**Every metric masks ``nan`` instead of propagating it or silencing it.**
The preprocessed reference fields carry ``nan`` where a shell held no pure cell
of that medium, and the model returns ``nan`` outside its own domain.  A metric
therefore counts what it actually measured and reports ``nan`` (not zero, and
not a silently reduced number) when it measured nothing.  A metric that returns
``0`` for an empty comparison is worse than useless: it reads as a perfect fit.

``delta`` and any other tunable are explicit arguments; nothing in this module
reads a configuration file or a global.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # ``np.trapezoid`` replaced ``np.trapz`` in NumPy 2.  Support both.
    _TRAPEZOID = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - NumPy < 2
    _TRAPEZOID = np.trapz  # type: ignore[attr-defined]


def trapezoid(values: np.ndarray, times: np.ndarray) -> float:
    """Composite trapezoidal integral of ``values`` over ``times``.

    Used for impulses and for mass/energy accounting.  Non-finite samples are
    dropped pairwise rather than treated as zero, so a gap in the history does
    not silently contribute a rectangle of zero height.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if values.size != times.size:
        raise ValueError(
            f"trapezoid needs matching lengths, got {values.size} values and {times.size} times"
        )
    if values.size < 2:
        return float("nan")
    finite = np.isfinite(values) & np.isfinite(times)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    return float(_TRAPEZOID(values[finite], times[finite]))


def _pair(pred, ref) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten ``pred``/``ref`` and return them with the jointly finite mask."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    if pred.size != ref.size:
        raise ValueError(
            f"metric needs matching shapes, got {pred.size} predictions and {ref.size} references"
        )
    return pred, ref, np.isfinite(pred) & np.isfinite(ref)


def _scale(scale: float) -> float:
    value = float(scale)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"metric scale must be finite and positive, got {scale!r}")
    return value


# --------------------------------------------------------------------------
# Norms
# --------------------------------------------------------------------------


def relative_l1(pred, ref, scale: float) -> float:
    """Mean absolute error divided by a fixed physical ``scale``.

    ``sum|pred - ref| / (n * scale)`` over the jointly finite samples.  The L1
    norm is the honest choice for a field that contains a discontinuity: a
    squared norm is dominated by the shock and tells you nothing about the tail,
    while a mean absolute error weights every shell that the reference actually
    resolved.
    """
    pred, ref, finite = _pair(pred, ref)
    scale = _scale(scale)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.abs(pred[finite] - ref[finite])) / scale)


def relative_l2(pred, ref, scale: float) -> float:
    """Root-mean-square error divided by a fixed physical ``scale``.

    ``sqrt(mean((pred - ref)^2)) / scale``.  By construction this is the same
    number as :func:`normalised_rmse` with the same scale; the two names exist
    because the field tables and the acceptance targets in
    ``evaluation.targets`` use different vocabulary for the same quantity, and
    having one function per name would invite them to drift apart.
    """
    return normalised_rmse(pred, ref, scale)


def normalised_rmse(pred, ref, scale: float | None = None) -> float:
    """RMS error normalised by a fixed physical ``scale``.

    When ``scale`` is ``None`` the normalisation falls back to the global RMS of
    the reference.  That is a *global* norm over the whole trace, not a local
    mean, so it does not suffer the vanishing-denominator problem at the
    pressure zero crossing — but it does make the metric dimensionless in a way
    that changes meaning between cases, so the fixed-scale form is the default
    everywhere a scale is available.
    """
    pred, ref, finite = _pair(pred, ref)
    if not np.any(finite):
        return float("nan")
    difference = pred[finite] - ref[finite]
    rms = float(np.sqrt(np.mean(difference * difference)))
    if scale is None:
        reference_rms = float(np.sqrt(np.mean(ref[finite] * ref[finite])))
        if reference_rms <= 0.0 or not np.isfinite(reference_rms):
            return float("nan")
        return rms / reference_rms
    return rms / _scale(scale)


def huber(a, b, delta: float) -> np.ndarray:
    """Elementwise Huber loss of ``a - b`` with transition point ``delta``.

    ``0.5 d^2`` for ``|d| <= delta`` and ``delta (|d| - 0.5 delta)`` beyond it.
    Kept here in NumPy so the evaluation tables report exactly the quantity the
    training loss optimises rather than a second, subtly different definition.
    ``nan`` in either input stays ``nan``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = _scale(delta)
    difference = np.abs(a - b)
    quadratic = 0.5 * difference * difference
    linear = delta * (difference - 0.5 * delta)
    result = np.where(difference <= delta, quadratic, linear)
    return np.where(np.isfinite(difference), result, np.nan)


def peak_error(pred, ref, scale: float) -> float:
    """Signed peak error ``(max pred - max ref) / scale``.

    Signed rather than absolute: a model that consistently over-predicts the
    peak and one that under-predicts are different failures, and the sign is
    the cheapest way to see which one is happening.
    """
    pred, ref, finite = _pair(pred, ref)
    scale = _scale(scale)
    if not np.any(finite):
        return float("nan")
    return float((np.max(pred[finite]) - np.max(ref[finite])) / scale)


def extremum_error(pred, ref, scale: float, *, mode: str = "min") -> float:
    """Signed extremum error for ``mode`` in ``{"min", "max"}``.

    The minimum form is how the negative phase is scored: its "peak" is a
    trough, and ``max`` would report the (usually much larger) positive peak
    again.
    """
    pred, ref, finite = _pair(pred, ref)
    scale = _scale(scale)
    if not np.any(finite):
        return float("nan")
    reducer = np.min if mode == "min" else np.max
    return float((reducer(pred[finite]) - reducer(ref[finite])) / scale)


def arrival_time_error(pred_times, ref_times, scale_time: float) -> float:
    """Mean signed arrival-time error divided by a fixed time scale.

    ``nan`` entries in either sequence drop the pair (an arrival that was never
    detected is a missing measurement, not a zero error) and ``nan`` is
    returned when no pair survives.
    """
    pred_times = np.asarray(pred_times, dtype=np.float64).reshape(-1)
    ref_times = np.asarray(ref_times, dtype=np.float64).reshape(-1)
    scale_time = _scale(scale_time)
    if pred_times.size != ref_times.size:
        raise ValueError(
            f"arrival_time_error needs matching lengths, got {pred_times.size} and {ref_times.size}"
        )
    finite = np.isfinite(pred_times) & np.isfinite(ref_times)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(pred_times[finite] - ref_times[finite]) / scale_time)


# --------------------------------------------------------------------------
# Integrated waveform quantities
# --------------------------------------------------------------------------


def _first_sign_change(values: np.ndarray, zero_tolerance: float = 0.0) -> int | None:
    """Index ``i`` of the first genuine sign change between ``i`` and ``i + 1``.

    Three conventions, each of which fixes a real failure:

    * **Zeros are carried forward** rather than treated as a phase boundary.  A
      sampled waveform may cross zero *between* two samples and land exactly on
      zero at one of them; comparing adjacent signs directly would see
      ``+, 0, -`` as "no change" and report a trace with an obvious negative
      phase as never changing sign.  Leading zeros are filled from the first
      non-zero sign, so a trace that starts at exactly ambient pressure is still
      read from the phase that follows it.
    * **Non-finite samples carry no sign**, so the previous sign propagates
      through a gap instead of a crossing being fabricated there.
    * **``zero_tolerance``** (default ``0.0``) treats ``|value| <= tolerance`` as
      zero.  This matters at scale: a reference trace built by interpolating a
      1e10 Pa product pressure against ambient air can sit 1e-11 Pa below
      ambient in a region that is physically undisturbed, and without the
      tolerance that residue would be read as the start of a negative phase and
      would attribute the entire impulse to the wrong half of the wave.
    """
    raw = np.asarray(values, dtype=np.float64)
    tolerance = abs(float(zero_tolerance))
    cleaned = np.where(np.isfinite(raw) & (np.abs(raw) > tolerance), raw, 0.0)
    sign = np.sign(cleaned)
    if sign.size < 2:
        return None
    filled = np.zeros_like(sign)
    last = 0.0
    for index in range(sign.size):
        if sign[index] != 0.0:
            last = sign[index]
        filled[index] = last
    first_nonzero = 0.0
    for value in sign:
        if value != 0.0:
            first_nonzero = value
            break
    filled = np.where(filled == 0.0, first_nonzero, filled)
    changes = np.flatnonzero(np.diff(filled) != 0.0)
    return int(changes[0]) if changes.size else None


def _crossing_time(times: np.ndarray, values: np.ndarray, index: int) -> float:
    """Linearly interpolated time at which a sign change between ``index`` and
    ``index + 1`` occurs."""
    v0, v1 = float(values[index]), float(values[index + 1])
    t0, t1 = float(times[index]), float(times[index + 1])
    if v1 == v0:
        return t0
    return t0 + (t1 - t0) * (-v0) / (v1 - v0)


def _sign_before(values: np.ndarray, index: int) -> float:
    """Sign of the phase that ends at the crossing at ``index``.

    A sampled waveform can land exactly on zero at the crossing sample, in which
    case the sign *at* ``index`` is no help.  Scanning back for the last non-zero
    sample identifies the phase that was actually in progress; ``0.0`` is
    returned when there is no earlier non-zero sample at all.
    """
    for position in range(index, -1, -1):
        if values[position] != 0.0:
            return float(np.sign(values[position]))
    return 0.0


def impulse(values, times, *, zero_tolerance: float = 0.0) -> dict[str, float]:
    """Positive- and negative-phase impulse, split at the first sign change.

    The split is taken at the **first** change of sign of ``overpressure``
    (positive to negative for a classical blast wave).  The two trapezoidal
    integrals are therefore

    ``I+ = int_{t_first}^{t_split} dp dt`` and ``I- = int_{t_split}^{t_end} dp dt``,

    and ``total`` is their sum.  When the trace never changes sign the whole
    impulse is attributed to a single phase and the other is reported as
    ``0.0`` with the split time at the relevant end of the window — recorded
    explicitly in ``split_time_s`` so a caller can tell "no negative phase in
    this window" from "the negative phase has zero impulse".

    ``zero_tolerance`` is passed through to the sign detection; see
    :func:`_first_sign_change` for why a trace can carry a numerically
    insignificant baseline that must not be mistaken for a phase.

    This is a **post-processing** quantity.  Nothing in the model reads it, and
    it must never be used to switch a network on or off: the phase boundary is a
    property of the recovered signal, not a modelling assumption.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if values.size != times.size:
        raise ValueError(
            f"impulse needs matching lengths, got {values.size} values and {times.size} times"
        )
    if values.size < 2:
        return {
            "positive": float("nan"),
            "negative": float("nan"),
            "total": float("nan"),
            "split_time_s": float("nan"),
        }

    finite = np.isfinite(values) & np.isfinite(times)
    if np.count_nonzero(finite) < 2:
        return {
            "positive": float("nan"),
            "negative": float("nan"),
            "total": float("nan"),
            "split_time_s": float("nan"),
        }

    index = _first_sign_change(values, zero_tolerance)
    if index is None:
        # No phase reversal inside the window: everything belongs to the sign
        # the trace actually has.  A trace that is identically zero integrates
        # to zero in either attribution.
        total = trapezoid(values, times)
        if np.all(values[finite] <= abs(float(zero_tolerance))):
            return {"positive": 0.0, "negative": total, "total": total,
                    "split_time_s": float(times[finite][0])}
        return {"positive": total, "negative": 0.0, "total": total,
                "split_time_s": float(times[finite][-1])}

    split = _crossing_time(times, values, index)
    positive = trapezoid(values[: index + 1], times[: index + 1])
    negative = trapezoid(values[index:], times[index:])
    # The crossing sample itself contributes to both halves; splitting it at the
    # interpolated zero removes the double count without changing the total.
    positive, negative = _split_at_crossing(
        values, times, index, split, positive, negative
    )
    return {
        "positive": positive,
        "negative": negative,
        "total": positive + negative,
        "split_time_s": float(split),
    }


def _split_at_crossing(
    values: np.ndarray,
    times: np.ndarray,
    index: int,
    split: float,
    positive: float,
    negative: float,
) -> tuple[float, float]:
    """Re-attribute the segment that straddles the zero crossing.

    The two half-integrals above meet at the sample ``index``, not at the
    interpolated zero.  The whole segment ``[t_i, t_{i+1}]`` therefore sits in
    the *negative* integral, although only its right-hand part is genuinely
    negative.  The left-hand triangle is moved across, so each phase carries
    exactly the area it owns and the total is unchanged.
    """
    t0, t1 = float(times[index]), float(times[index + 1])
    v0, v1 = float(values[index]), float(values[index + 1])
    if not (t0 < split < t1) or not np.isfinite([v0, v1]).all():
        return positive, negative
    left = 0.5 * v0 * (split - t0)
    return positive + left, negative - left


def phase_durations(
    times, overpressure, *, zero_tolerance: float = 0.0
) -> tuple[float, float]:
    """``(positive duration, negative duration)`` of an overpressure trace.

    The positive duration is measured from the moment the trace first exceeds
    ambient pressure to the first return to ambient pressure.  The negative
    duration runs from that crossing to the last sample at or below ambient
    pressure inside the window; a history that has not fully recovered by the end
    of the window therefore reports the duration *observed*, never an
    extrapolated one.

    ``(nan, nan)`` is returned when the trace never becomes positive or never
    changes sign: the quantity is then undefined for this window, and saying so
    is the only honest answer.
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    overpressure = np.asarray(overpressure, dtype=np.float64).reshape(-1)
    if times.size != overpressure.size:
        raise ValueError(
            "phase_durations needs matching lengths, got "
            f"{overpressure.size} samples and {times.size} times"
        )
    finite = np.isfinite(times) & np.isfinite(overpressure)
    if np.count_nonzero(finite) < 2:
        return float("nan"), float("nan")

    tolerance = abs(float(zero_tolerance))
    index = _first_sign_change(overpressure, tolerance)
    if index is None:
        return float("nan"), float("nan")

    split = _crossing_time(times, overpressure, index)
    sign_before = _sign_before(
        np.where(np.abs(overpressure) > tolerance, overpressure, 0.0), index
    )
    positive_first = sign_before > 0.0

    if positive_first:
        start_index = int(np.argmax(finite & (overpressure > tolerance)))
        positive = split - float(times[start_index])
    else:
        # The trace starts negative and turns positive: the window caught the
        # tail of the negative phase, and its start is not observable here.
        positive = float("nan")

    after = finite & (times > split)
    if not np.any(after):
        return positive, float("nan")
    if positive_first:
        negative_indices = np.flatnonzero(after & (overpressure <= tolerance))
        if negative_indices.size == 0:
            return positive, float("nan")
        negative = float(times[negative_indices[-1]]) - split
    else:
        negative = float(times[np.flatnonzero(after)[-1]]) - split
    return positive, negative


# --------------------------------------------------------------------------
# Simple relative errors
# --------------------------------------------------------------------------


def relative_error_with_reference(pred, ref) -> np.ndarray:
    """Elementwise ``|pred - ref| / |ref|`` for radii.

    The *reference-relative* form, without an external scale.  It is used for
    the front trajectories only, where the reference is strictly positive by
    construction (``R_c, R_s > 0`` for every ``tau``) and a percentage of the
    reference radius is the number the acceptance targets are written in.  A
    zero or non-finite reference yields ``nan`` rather than an infinity: an
    undefined ratio must not be reported as a large error.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: {pred.shape} predictions vs {ref.shape} references")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.abs(pred - ref) / np.abs(ref)
    return np.where(np.isfinite(ratio), ratio, np.nan)


def percentile_summary(values) -> dict[str, float]:
    """``min / p05 / median / p95 / max`` of the finite entries.

    Every field is ``nan`` when nothing finite was supplied, so an empty
    comparison is visibly empty rather than a row of zeros.
    """
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "min": float("nan"),
            "p05": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    return {
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5.0)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
        "count": int(finite.size),
    }


def summarise_masked(values) -> dict[str, float]:
    """Alias of :func:`percentile_summary` kept for call sites that read better.

    Both names appear in the reporting code; they are one implementation so a
    change to the percentile set can never apply to only half the tables.
    """
    return percentile_summary(values)


@dataclass
class MetricRecord:
    """One named number with its provenance, for the CSV/JSON tables.

    ``units`` and ``target`` are recorded next to the value because a bare
    number in a report is unverifiable: the reader cannot tell a relative error
    from an absolute one, and cannot check it against the acceptance target
    without the target being present.
    """

    name: str
    value: float
    units: str = ""
    target: float | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "units": self.units,
            "target": self.target,
            "note": self.note,
        }


__all__ = [
    "MetricRecord",
    "arrival_time_error",
    "extremum_error",
    "huber",
    "impulse",
    "normalised_rmse",
    "peak_error",
    "percentile_summary",
    "phase_durations",
    "relative_error_with_reference",
    "relative_l1",
    "relative_l2",
    "summarise_masked",
    "trapezoid",
]
