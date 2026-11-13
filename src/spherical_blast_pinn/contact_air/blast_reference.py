"""Empirical blast reference relations, with their provenance and limits.

Only relations whose coefficients are stated here are computed.  A relation
that cannot be traced and checked is reported as unavailable; it is never
replaced by a different correlation under the same name.

Henrych
-------
Spherical free-air burst, incident (side-on) overpressure, three-branch
polynomial in the scaled distance ``Z = r / W**(1/3)`` (r in m, W in kg of
TNT equivalent), result in MPa:

.. math::
    0.05 <= Z <= 0.3:  dp = 1.40717/Z + 0.55397/Z^2
                            - 0.03572/Z^3 + 0.000625/Z^4
    0.3  <= Z <= 1.0:  dp = 0.61938/Z - 0.03262/Z^2 + 0.21324/Z^3
    1.0  <= Z <= 10:   dp = 0.0662/Z + 0.405/Z^2 + 0.3288/Z^3

Status: ``provisional``.  The coefficients above are the commonly reproduced
three-branch Henrych set, but this repository has not compared them against a
printed copy of Henrych (1979), *The Dynamics of Explosion and Its Use*.
They must not be presented as verified values, and a different transcription
must not be spliced onto this one.

Kingery-Bulmash
---------------
Not implemented.  The KB polynomials (incident pressure, arrival time,
positive-phase duration, impulse) need the full coefficient tables from
UFC 3-340-02 / the original report.  No verified copy is present here, so the
KB functions raise :class:`ReferenceUnavailable` instead of guessing.  The
pressure range of a KB pressure fit must never be assumed to apply to its
arrival-time or impulse fits, so a partial implementation is not provided.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class ReferenceRangeError(ValueError):
    """The scaled distance lies outside the relation's stated range."""


class ReferenceUnavailable(RuntimeError):
    """The relation is not available because its source was not verified."""


HENRYCH_RANGE = (0.05, 10.0)
HENRYCH_UNITS = {"input": "Z in m/kg^(1/3)", "output": "MPa"}
HENRYCH_STATUS = "provisional"
HENRYCH_SOURCE = (
    "Henrych (1979), The Dynamics of Explosion and Its Use; spherical "
    "free-air burst, incident overpressure, three-branch polynomial as "
    "commonly reproduced. Coefficients not checked against a printed copy "
    "in this repository."
)

KB_STATUS = "unavailable"
KB_REASON = (
    "Kingery-Bulmash coefficient tables (incident pressure, arrival time, "
    "positive-phase duration, impulse) are not present in a verified form; "
    "the relation is not guessed or replaced by another correlation."
)


@dataclass(frozen=True)
class ReferenceValue:
    value: float | None
    unit: str
    status: str
    source: str
    range_note: str
    reason: str | None = None


def charge_mass_kg(radius_m: float, density_kg_m3: float) -> float:
    """Uniform-sphere charge mass ``4/3 pi rho R^3``."""
    radius = float(radius_m)
    density = float(density_kg_m3)
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError("radius_m must be positive and finite")
    if not (math.isfinite(density) and density > 0.0):
        raise ValueError("density_kg_m3 must be positive and finite")
    return 4.0 * math.pi / 3.0 * density * radius ** 3


def scaled_distance(r_m: float, charge_mass_kg_value: float) -> float:
    """``Z = r / W**(1/3)`` in m/kg^(1/3)."""
    r = float(r_m)
    mass = float(charge_mass_kg_value)
    if not (math.isfinite(r) and r >= 0.0):
        raise ValueError("r_m must be finite and non-negative")
    if not (math.isfinite(mass) and mass > 0.0):
        raise ValueError("charge_mass_kg must be positive and finite")
    return r / mass ** (1.0 / 3.0)


def henrych_incident_overpressure_mpa(scaled_distance_m_kg13: float) -> float:
    """Henrych spherical free-air incident overpressure in MPa.

    Raises :class:`ReferenceRangeError` outside ``0.05 <= Z <= 10``.
    """
    z = float(scaled_distance_m_kg13)
    if not math.isfinite(z):
        raise ReferenceRangeError("scaled distance must be finite")
    low, high = HENRYCH_RANGE
    if not (low <= z <= high):
        raise ReferenceRangeError(
            f"scaled distance Z={z!r} is outside Henrych range "
            f"[{low!r}, {high!r}] m/kg^(1/3)"
        )
    if z <= 0.3:
        return (
            1.40717 / z + 0.55397 / z ** 2
            - 0.03572 / z ** 3 + 0.000625 / z ** 4
        )
    if z <= 1.0:
        return 0.61938 / z - 0.03262 / z ** 2 + 0.21324 / z ** 3
    return 0.0662 / z + 0.405 / z ** 2 + 0.3288 / z ** 3


def henrych_incident_overpressure_pa(scaled_distance_m_kg13: float) -> float:
    return 1.0e6 * henrych_incident_overpressure_mpa(scaled_distance_m_kg13)


def henrych_reference(scaled_distance_m_kg13: float) -> ReferenceValue:
    """Henrych value plus provenance; out-of-range returns ``value=None``."""
    z = float(scaled_distance_m_kg13)
    low, high = HENRYCH_RANGE
    if not (low <= z <= high):
        return ReferenceValue(
            value=None, unit="MPa", status="not_comparable",
            source=HENRYCH_SOURCE,
            range_note=f"Z outside [{low}, {high}] m/kg^(1/3)",
            reason="scaled distance outside the relation's stated range",
        )
    return ReferenceValue(
        value=henrych_incident_overpressure_mpa(z), unit="MPa",
        status=HENRYCH_STATUS, source=HENRYCH_SOURCE,
        range_note=f"spherical free-air burst, incident pressure, {low} <= Z <= {high}",
    )


def kb_incident_overpressure(scaled_distance_m_kg13: float) -> None:
    raise ReferenceUnavailable(KB_REASON)


def kb_arrival_time(scaled_distance_m_kg13: float) -> None:
    raise ReferenceUnavailable(KB_REASON)


def kb_positive_duration(scaled_distance_m_kg13: float) -> None:
    raise ReferenceUnavailable(KB_REASON)


def kb_positive_impulse(scaled_distance_m_kg13: float) -> None:
    raise ReferenceUnavailable(KB_REASON)


__all__ = [
    "HENRYCH_RANGE", "HENRYCH_SOURCE", "HENRYCH_STATUS",
    "HENRYCH_UNITS", "KB_REASON", "KB_STATUS",
    "ReferenceRangeError", "ReferenceUnavailable", "ReferenceValue",
    "charge_mass_kg", "henrych_incident_overpressure_mpa",
    "henrych_incident_overpressure_pa", "henrych_reference",
    "kb_arrival_time", "kb_incident_overpressure", "kb_positive_duration",
    "kb_positive_impulse", "scaled_distance",
]
