"""Unit-system resolution and conversion to SI.

An LS-DYNA keyword deck carries **no** unit information.  The unit system is a
property of the case metadata, and this module never guesses silently:

* an explicit system in the configuration is honoured and cross-checked;
* ``auto`` detection shortlists the systems that make the observed TNT density,
  detonation velocity and JWL pressure physically plausible, and **raises** if
  the shortlist is not a single system.

Every physical quantity that enters the networks passes through here first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from ..constants import UNIT_SYSTEMS, UnitSystem, resolve_unit_system

#: Plausibility windows used only by the ``auto`` detector.  They are broad on
#: purpose: the job is to eliminate unit systems, not to validate the material.
PLAUSIBLE_TNT_DENSITY_KG_M3 = (900.0, 2200.0)
PLAUSIBLE_DETONATION_VELOCITY_M_S = (3000.0, 12000.0)
PLAUSIBLE_JWL_A_PA = (1.0e10, 1.0e12)


class UnitError(Exception):
    """Raised when the unit system is missing, ambiguous or inconsistent."""


@dataclass
class UnitResolution:
    """The resolved unit system plus how it was obtained."""

    system: UnitSystem
    source: str
    candidates: tuple[str, ...] = ()
    evidence: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.system.name

    def to_si(self, quantity: str, value):
        """Convert ``value`` of ``quantity`` to SI."""
        factor = self.factor(quantity)
        return np.asarray(value, dtype=np.float64) * factor if np.ndim(value) else float(value) * factor

    def factor(self, quantity: str) -> float:
        table = {
            "length": self.system.length_to_si,
            "time": self.system.time_to_si,
            "mass": self.system.mass_to_si,
            "density": self.system.density_to_si,
            "velocity": self.system.velocity_to_si,
            "acceleration": self.system.velocity_to_si / self.system.time_to_si,
            "pressure": self.system.pressure_to_si,
            "specific_energy": self.system.specific_energy_to_si,
            "energy_per_volume": self.system.energy_per_volume_to_si,
            "volume": self.system.volume_to_si,
            "force": self.system.mass_to_si
            * self.system.length_to_si
            / self.system.time_to_si**2,
            "energy": self.system.mass_to_si
            * self.system.length_to_si**2
            / self.system.time_to_si**2,
        }
        if quantity not in table:
            raise UnitError(f"unknown quantity {quantity!r}; known: {sorted(table)}")
        return table[quantity]

    def describe(self) -> dict[str, object]:
        return {
            "system": self.system.name,
            "source": self.source,
            "candidates": list(self.candidates),
            "factors": self.system.describe(),
            "evidence": dict(self.evidence),
        }


def resolve_units(
    *,
    system: str,
    overrides: Mapping[str, float] | None = None,
    evidence: Mapping[str, float] | None = None,
    allow_auto: bool = True,
) -> UnitResolution:
    """Resolve a unit system from configuration, optionally via ``auto``.

    Parameters
    ----------
    system:
        ``"auto"`` or a unit-system name/alias.
    overrides:
        Explicit per-quantity overrides, e.g. ``{"density": 1e12}``.  When an
        override is given it is applied *after* the base system and the result
        is re-derived as a synthetic system, so that all derived quantities
        stay consistent with the overridden ones.
    evidence:
        Observed values (in deck units) used by the ``auto`` detector, e.g.
        ``{"tnt_density": 1.63e-9, "detonation_velocity": 6930000.0,
        "jwl_A": 371200.0}``.
    """
    if system == "auto" or system is None:
        if not allow_auto:
            raise UnitError("unit system 'auto' is not allowed here")
        if not evidence:
            raise UnitError(
                "unit system is 'auto' but no evidence values were supplied; "
                "provide tnt_density / detonation_velocity / jwl_A, or set "
                "units.system explicitly"
            )
        candidates = detect_unit_systems(evidence)
        if len(candidates) != 1:
            raise UnitError(
                "unit-system auto-detection is ambiguous: "
                f"{candidates} all fit the supplied evidence {dict(evidence)}. "
                "Set units.system explicitly rather than letting the code guess."
            )
        resolution = UnitResolution(UNIT_SYSTEMS[candidates[0]], "auto", tuple(candidates), dict(evidence))
    else:
        try:
            resolution = UnitResolution(resolve_unit_system(system), "explicit", (), dict(evidence or {}))
        except KeyError as exc:
            raise UnitError(str(exc)) from exc

    if overrides:
        resolution = _apply_overrides(resolution, overrides)
    return resolution


def _apply_overrides(resolution: UnitResolution, overrides: Mapping[str, float]) -> UnitResolution:
    base = resolution.system
    length = float(overrides.get("length", base.length_to_si))
    time = float(overrides.get("time", base.time_to_si))
    mass = float(overrides.get("mass", base.mass_to_si))
    density = overrides.get("density")
    pressure = overrides.get("pressure")

    # Recover mass/length/time from whichever overrides were supplied, keeping
    # the rest consistent.
    if density is not None and "mass" not in overrides:
        mass = float(density) * length**3
    if pressure is not None and "mass" not in overrides:
        mass = float(pressure) * length * time**2

    derived = UnitSystem(f"{base.name}+overrides", length, time, mass)
    for name, target, actual in (
        ("density", density, derived.density_to_si),
        ("pressure", pressure, derived.pressure_to_si),
    ):
        if target is not None and abs(float(target) - actual) > 1.0e-6 * max(abs(float(target)), 1.0):
            raise UnitError(
                f"unit overrides are inconsistent: requested {name} scale {target!r} "
                f"but length/time/mass imply {actual!r}"
            )
    return UnitResolution(derived, resolution.source + "+overrides", resolution.candidates, resolution.evidence)


def detect_unit_systems(evidence: Mapping[str, float]) -> list[str]:
    """Return the unit systems consistent with the observed deck values.

    The evidence is given in *deck units*; the detector converts each candidate
    and keeps the systems for which every supplied value lands in a physically
    plausible window.  At least one of ``tnt_density``, ``detonation_velocity``
    or ``jwl_A`` must be supplied.
    """
    checks: list[tuple[str, float, tuple[float, float]]] = []
    if evidence.get("tnt_density"):
        checks.append(
            ("tnt_density", float(evidence["tnt_density"]), PLAUSIBLE_TNT_DENSITY_KG_M3)
        )
    if evidence.get("detonation_velocity"):
        checks.append(
            (
                "detonation_velocity",
                float(evidence["detonation_velocity"]),
                PLAUSIBLE_DETONATION_VELOCITY_M_S,
            )
        )
    if evidence.get("jwl_A"):
        checks.append(("jwl_A", float(evidence["jwl_A"]), PLAUSIBLE_JWL_A_PA))
    if not checks:
        raise UnitError(
            "auto unit detection needs at least one of tnt_density, "
            "detonation_velocity or jwl_A"
        )

    survivors: list[str] = []
    for name, system in UNIT_SYSTEMS.items():
        ok = True
        for quantity, value, (low, high) in checks:
            factor = {
                "tnt_density": system.density_to_si,
                "detonation_velocity": system.velocity_to_si,
                "jwl_A": system.pressure_to_si,
            }[quantity]
            si = value * factor
            if not (low <= si <= high):
                ok = False
                break
        if ok:
            survivors.append(name)
    return survivors


def cross_check_metadata(
    resolution: UnitResolution,
    deck_values: Mapping[str, float],
    expected_si: Mapping[str, float],
    *,
    rtol: float = 1.0e-3,
) -> dict[str, float]:
    """Compare the unit system against independent metadata.

    ``deck_values`` maps a quantity to the number *as written in the deck*;
    ``expected_si`` maps the same quantity to the value the case metadata
    claims in SI.  Every quantity present in both is converted and compared.

    Returns the relative deviations and raises :class:`UnitError` on any
    mismatch beyond ``rtol`` — a deck that was re-run in different units, or a
    metadata block copied from another charge, is caught here rather than
    showing up as an inexplicable training failure.
    """
    deviations: dict[str, float] = {}
    for quantity, deck_value in deck_values.items():
        if quantity not in expected_si:
            continue
        target = expected_si[quantity]
        if deck_value is None or target is None:
            continue
        converted = float(deck_value) * resolution.factor(quantity)
        denominator = max(abs(float(target)), np.finfo(np.float64).tiny)
        deviation = abs(converted - float(target)) / denominator
        deviations[quantity] = deviation
        if deviation > rtol:
            raise UnitError(
                f"unit cross-check failed for {quantity!r}: deck value "
                f"{deck_value!r} converts to {converted:.6g} SI but the metadata "
                f"declares {float(target):.6g} (relative deviation {deviation:.3e} "
                f"> rtol {rtol:.1e}) for unit system {resolution.name!r}"
            )
    return deviations


def summarise(quantities: Iterable[str], resolution: UnitResolution) -> str:
    return ", ".join(f"{q}={resolution.factor(q):.6g}" for q in quantities)


def density_kg_m3(value: float, resolution: UnitResolution) -> float:
    return float(value) * resolution.factor("density")


def pressure_pa(value: float, resolution: UnitResolution) -> float:
    return float(value) * resolution.factor("pressure")


def length_m(value: float, resolution: UnitResolution) -> float:
    return float(value) * resolution.factor("length")


def velocity_m_s(value: float, resolution: UnitResolution) -> float:
    return float(value) * resolution.factor("velocity")


def specific_energy_j_kg(value: float, resolution: UnitResolution) -> float:
    return float(value) * resolution.factor("specific_energy")


def choose_pressure_convention(
    *,
    declared: str,
    observed_min: float,
    observed_max: float,
    ambient_pressure_si: float,
) -> tuple[str, float]:
    """Return ``(convention, offset_to_add)`` after checking the declaration.

    The decks in this study store absolute pressure.  If a dataset instead
    stores overpressure, metadata must say so, and the offset is applied
    explicitly — this function refuses to infer the convention from the data
    alone when the evidence is contradictory.
    """
    declared = (declared or "absolute").strip().lower()
    if declared not in {"absolute", "overpressure", "gauge"}:
        raise UnitError(f"unknown pressure convention {declared!r}")
    positive_evidence = observed_min >= -1.0e-6 * max(abs(observed_max), 1.0)
    if declared == "absolute" and not positive_evidence:
        raise UnitError(
            "metadata declares absolute pressure but the sampled pressures go "
            f"negative (min = {observed_min:.6g}); this looks like an "
            "overpressure or a gauge-pressure field. Correct the metadata."
        )
    if declared in {"overpressure", "gauge"}:
        if observed_max < 0.0:
            raise UnitError(
                "metadata declares overpressure but all sampled pressures are "
                f"negative (max = {observed_max:.6g}); the sign convention is wrong"
            )
        return declared, float(ambient_pressure_si)
    return declared, 0.0


__all__ = [
    "PLAUSIBLE_DETONATION_VELOCITY_M_S",
    "PLAUSIBLE_JWL_A_PA",
    "PLAUSIBLE_TNT_DENSITY_KG_M3",
    "UnitError",
    "UnitResolution",
    "choose_pressure_convention",
    "cross_check_metadata",
    "density_kg_m3",
    "detect_unit_systems",
    "length_m",
    "pressure_pa",
    "resolve_units",
    "specific_energy_j_kg",
    "summarise",
    "velocity_m_s",
]
