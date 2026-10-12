"""Physical and unit constants for the spherical blast PINN.

Nothing in this module is a *scientific* choice for a specific LS-DYNA study.
The JWL coefficients, the explosive density and the ambient air state must all
come from the case metadata / keyword deck.  The values here are either exact
mathematical constants or clearly-labelled fallbacks that a run is required to
confirm before training (see :data:`FALLBACK_VALUES_REQUIRE_CONFIRMATION`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

# --------------------------------------------------------------------------
# Exact mathematical constants
# --------------------------------------------------------------------------

FOUR_PI: Final[float] = 4.0 * math.pi
"""Surface area of the unit sphere; appears in every global conservation law."""

# --------------------------------------------------------------------------
# Ambient air fallbacks.
#
# These are *not* used silently.  Any run that falls back to them records the
# fact in the audit metadata and the EOS validator reports it.  The real values
# must be read from the LS-DYNA air material / EOS cards and the case metadata.
# --------------------------------------------------------------------------

DEFAULT_AIR_GAMMA: Final[float] = 1.4
"""Ratio of specific heats for a calorically perfect diatomic gas (dry air)."""

DEFAULT_AIR_DENSITY: Final[float] = 1.225
"""kg/m^3, ISA sea-level density.  Fallback only."""

DEFAULT_AIR_PRESSURE: Final[float] = 101325.0
"""Pa, ISA sea-level absolute pressure.  Fallback only."""

FALLBACK_VALUES_REQUIRE_CONFIRMATION: Final[tuple[str, ...]] = (
    "air.gamma",
    "air.rho",
    "air.pressure",
    "jwl.A",
    "jwl.B",
    "jwl.R1",
    "jwl.R2",
    "jwl.omega",
    "jwl.rho0",
)
"""Dotted config paths whose value must be confirmed against the keyword deck."""

# --------------------------------------------------------------------------
# Physical admissibility floors
# --------------------------------------------------------------------------

TINY_DENSITY: Final[float] = 1.0e-12
"""kg/m^3 floor used only to keep divisions finite; never a physical model."""

TINY_PRESSURE: Final[float] = 1.0e-6
"""Pa floor used only to keep logarithms finite."""

SOUND_SPEED_FLOOR: Final[float] = 1.0
"""m/s floor for sound-speed comparisons."""


# --------------------------------------------------------------------------
# Unit systems
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitSystem:
    """A coherent unit system defined by its length, time and mass units.

    Every other unit in the system is derived, which removes a whole class of
    hand-transcription errors:

    ==================  ==============================
    quantity            derived unit
    ==================  ==============================
    length              ``L``
    time                ``T``
    mass                ``M``
    density             ``M / L^3``
    velocity            ``L / T``
    acceleration        ``L / T^2``
    pressure            ``M / (L T^2)``
    specific energy     ``L^2 / T^2``
    energy per volume   ``M / (L T^2)``  (same as pressure)
    volume              ``L^3``
    force               ``M L / T^2``
    ==================  ==============================
    """

    name: str
    length_to_si: float
    time_to_si: float
    mass_to_si: float

    @property
    def velocity_to_si(self) -> float:
        return self.length_to_si / self.time_to_si

    @property
    def density_to_si(self) -> float:
        return self.mass_to_si / self.length_to_si**3

    @property
    def pressure_to_si(self) -> float:
        return self.mass_to_si / (self.length_to_si * self.time_to_si**2)

    @property
    def specific_energy_to_si(self) -> float:
        return self.length_to_si**2 / self.time_to_si**2

    @property
    def energy_per_volume_to_si(self) -> float:
        return self.pressure_to_si

    @property
    def volume_to_si(self) -> float:
        return self.length_to_si**3

    # -- conversions ------------------------------------------------------
    def length_to_m(self, value: float) -> float:
        return value * self.length_to_si

    def time_to_s(self, value: float) -> float:
        return value * self.time_to_si

    def density_to_si_units(self, value: float) -> float:
        return value * self.density_to_si

    def pressure_to_pa(self, value: float) -> float:
        return value * self.pressure_to_si

    def velocity_to_m_per_s(self, value: float) -> float:
        return value * self.velocity_to_si

    def specific_energy_to_j_per_kg(self, value: float) -> float:
        return value * self.specific_energy_to_si

    def energy_per_volume_to_j_per_m3(self, value: float) -> float:
        return value * self.energy_per_volume_to_si

    def describe(self) -> dict[str, float]:
        return {
            "length_to_si": self.length_to_si,
            "time_to_si": self.time_to_si,
            "mass_to_si": self.mass_to_si,
            "velocity_to_si": self.velocity_to_si,
            "density_to_si": self.density_to_si,
            "pressure_to_si": self.pressure_to_si,
            "specific_energy_to_si": self.specific_energy_to_si,
            "energy_per_volume_to_si": self.energy_per_volume_to_si,
            "volume_to_si": self.volume_to_si,
        }


UNIT_SYSTEMS: Final[dict[str, UnitSystem]] = {
    # Coherent SI.
    "si": UnitSystem("si", 1.0, 1.0, 1.0),
    # LS-DYNA: mm, ms, kg  ->  pressure in GPa.  Very common for blast work.
    "mm_ms_kg": UnitSystem("mm_ms_kg", 1.0e-3, 1.0e-3, 1.0),
    # LS-DYNA: mm, ms, g   ->  pressure in MPa.
    "mm_ms_g": UnitSystem("mm_ms_g", 1.0e-3, 1.0e-3, 1.0e-3),
    # LS-DYNA: mm, ms, mg  ->  pressure in kPa.
    "mm_ms_mg": UnitSystem("mm_ms_mg", 1.0e-3, 1.0e-3, 1.0e-6),
    # LS-DYNA: cm, us, g   ->  pressure in Mbar (1e11 Pa).
    "cm_us_g": UnitSystem("cm_us_g", 1.0e-2, 1.0e-6, 1.0e-3),
    # LS-DYNA: m, ms, kg   ->  pressure in MPa.
    "m_ms_kg": UnitSystem("m_ms_kg", 1.0, 1.0e-3, 1.0),
    # LS-DYNA: m, s, kg with pressure in MPa.
    "m_s_kg_mpa": UnitSystem("m_s_kg_mpa", 1.0, 1.0, 1.0),
    # LS-DYNA: mm, s, kg with pressure in kPa (rare, but appears in tutorials).
    "mm_s_kg": UnitSystem("mm_s_kg", 1.0e-3, 1.0, 1.0),
    # Workbench Explicit default: mm, s, tonne (Mg).  This is the system the
    # four TNT blast decks in this study actually use: TNT density reads
    # 1.63e-09, D_CJ reads 6930000 and JWL A reads 371200, which are the
    # familiar 1630 kg/m^3, 6930 m/s and 3.712e11 Pa in these units.
    "mm_s_t": UnitSystem("mm_s_t", 1.0e-3, 1.0, 1.0e3),
}

UNIT_SYSTEM_ALIASES: Final[dict[str, str]] = {
    "auto": "auto",
    "": "auto",
    "SI": "si",
    "si_m_kg_s": "si",
    "kg_m_s": "si",
    "mm_ms_kg_gpa": "mm_ms_kg",
    "mm-ms-kg": "mm_ms_kg",
    "mm-ms-g": "mm_ms_g",
    "mm_ms_g_mpa": "mm_ms_g",
    "cm_us_g_mbar": "cm_us_g",
    "cm-us-g": "cm_us_g",
    "m_ms_kg_mpa": "m_ms_kg",
    "m-ms-kg": "m_ms_kg",
    "mm_s_t": "mm_s_t",
    "mm-s-t": "mm_s_t",
    "mm_s_tonne": "mm_s_t",
    "mm_s_Mg": "mm_s_t",
    "mm-Mg-s": "mm_s_t",
    "mm_tonne_s": "mm_s_t",
}

def resolve_unit_system(name: str) -> UnitSystem:
    """Return the :class:`UnitSystem` for ``name`` (aliases accepted)."""
    key = UNIT_SYSTEM_ALIASES.get(name.strip(), name.strip())
    if key == "auto":
        raise ValueError(
            "cannot resolve the 'auto' unit system here; run "
            "spherical_blast_pinn.data.units.detect_unit_system first"
        )
    if key not in UNIT_SYSTEMS:
        raise KeyError(
            f"unknown unit system {name!r}; known systems: "
            f"{sorted(UNIT_SYSTEMS)}"
        )
    return UNIT_SYSTEMS[key]
