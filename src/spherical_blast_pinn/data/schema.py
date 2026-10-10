"""Logical data schema and per-case metadata.

The reader never sees a hard-coded column name.  Raw files are described by a
:class:`FieldMapping` that maps *logical* field names (``density``, ``time``,
``pressure_absolute`` …) onto whatever the file actually calls them.  This is
what makes the pipeline indifferent to whether a case arrives as CSV, Parquet,
HDF5 or a d3plot family.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from ..constants import FOUR_PI
from ..physics.eos import JWLParameters

#: Every logical field the pipeline understands.  ``REQUIRED_FIELDS`` must be
#: mapped for a case to be usable; the rest are optional and their absence is
#: recorded rather than fatal.
LOGICAL_FIELDS: tuple[str, ...] = (
    "case_id",
    "time",
    "x",
    "y",
    "z",
    "cell_volume",
    "density",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "specific_internal_energy",
    "pressure_absolute",
    "tnt_volume_fraction",
    "burn_fraction",
    "material_id",
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "time",
    "x",
    "y",
    "z",
    "density",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "pressure_absolute",
    "tnt_volume_fraction",
)

#: Fields the reader can derive when the raw source does not store them.
DERIVABLE_FIELDS: tuple[str, ...] = (
    "cell_volume",
    "specific_internal_energy",
)


class DataSchemaError(Exception):
    """Raised when raw data does not match the declared schema."""


# --------------------------------------------------------------------------
# Field mapping
# --------------------------------------------------------------------------


@dataclass
class FieldMapping:
    """Logical name -> source column name.  ``None`` means "not available"."""

    mapping: dict[str, str | None]

    @classmethod
    def from_config(cls, config_map: Mapping[str, Any] | None) -> "FieldMapping":
        supplied = dict(config_map or {})
        unknown = sorted(set(supplied) - set(LOGICAL_FIELDS))
        if unknown:
            raise DataSchemaError(
                f"data.field_map contains unknown logical fields {unknown}; "
                f"known fields are {list(LOGICAL_FIELDS)}"
            )
        resolved = {name: supplied.get(name) for name in LOGICAL_FIELDS}
        return cls(resolved)

    def source(self, logical: str) -> str | None:
        if logical not in LOGICAL_FIELDS:
            raise DataSchemaError(f"unknown logical field {logical!r}")
        return self.mapping.get(logical)

    def require(self, logical: str) -> str:
        source = self.source(logical)
        if not source:
            raise DataSchemaError(
                f"logical field {logical!r} is required but is not mapped in "
                "data.field_map"
            )
        return source

    def has(self, logical: str) -> bool:
        return bool(self.mapping.get(logical))

    def missing_required(self) -> list[str]:
        return [name for name in REQUIRED_FIELDS if not self.has(name)]

    def derivable(self) -> list[str]:
        return [name for name in DERIVABLE_FIELDS if not self.has(name)]

    def as_dict(self) -> dict[str, str | None]:
        return dict(self.mapping)


# --------------------------------------------------------------------------
# Case metadata
# --------------------------------------------------------------------------


@dataclass
class CaseMetadata:
    """Everything needed to interpret one charge size."""

    case_id: str
    charge_radius_m: float
    charge_mass_kg: float
    unit_system: str
    tnt_initial_density_kg_m3: float
    jwl: JWLParameters
    ambient_density_kg_m3: float
    ambient_pressure_pa: float
    gamma_air: float
    mesh_size_m: float | None
    output_interval_s: float | None
    outer_boundary_radius_m: float | None
    valid_end_time_s: float | None
    split: str
    #: ``*EOS_JWL`` ``E0``: internal energy per unit **initial** volume, J/m^3.
    jwl_volume_energy_j_m3: float | None = None
    #: ``*MAT_HIGH_EXPLOSIVE_BURN`` ``D``, m/s.  Used for the ``R_0 / D``
    #: cross-check only — never as the reported burn time.
    detonation_velocity_m_s: float | None = None
    # -- provenance for the quantities above -----------------------------
    time_alignment_method: str | None = None
    time_alignment_confidence: str | None = None
    time_alignment_uncertainty_s: float | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise DataSchemaError("case_id must be non-empty")
        for name in (
            "charge_radius_m",
            "charge_mass_kg",
            "tnt_initial_density_kg_m3",
            "ambient_density_kg_m3",
            "ambient_pressure_pa",
            "gamma_air",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise DataSchemaError(f"{name} must be finite and positive, got {value!r}")
        if self.gamma_air <= 1.0:
            raise DataSchemaError(f"gamma_air must exceed 1, got {self.gamma_air}")
        if self.split not in {"train", "validation", "test", "extrapolation_test"}:
            raise DataSchemaError(
                f"split must be one of train/validation/test/extrapolation_test, got {self.split!r}"
            )

    # -- derived ----------------------------------------------------------
    @property
    def ambient_specific_energy_j_kg(self) -> float:
        return self.ambient_pressure_pa / ((self.gamma_air - 1.0) * self.ambient_density_kg_m3)

    @property
    def product_initial_specific_energy_j_kg(self) -> float | None:
        """``e_p0 = E0 / rho_p0`` where ``E0`` is the per-initial-volume energy.

        ``None`` when the deck did not declare ``E0``; the EOS validator then
        reports the missing field rather than silently assuming a value.
        """
        if self.jwl_volume_energy_j_m3 is None:
            return None
        return self.jwl.initial_specific_energy(
            self.jwl_volume_energy_j_m3
        )

    @property
    def ambient_mass_in_domain_kg(self) -> float | None:
        if self.outer_boundary_radius_m is None:
            return None
        return (
            FOUR_PI
            / 3.0
            * self.ambient_density_kg_m3
            * (self.outer_boundary_radius_m**3 - self.charge_radius_m**3)
        )

    @property
    def sound_speed_ambient_m_s(self) -> float:
        return math.sqrt(self.gamma_air * self.ambient_pressure_pa / self.ambient_density_kg_m3)

    @property
    def reflection_arrival_time_s(self) -> float | None:
        """Earliest time a reflection from the outer free surface can return.

        A free outer surface reflects the shock as a rarefaction.  For a
        spherical shell it returns after roughly ``2 * (R_out - r) / c``; the
        conservative bound below uses the ambient sound speed, which is the
        fastest the rarefaction can travel back.
        """
        if self.outer_boundary_radius_m is None or self.outer_boundary_radius_m <= self.charge_radius_m:
            return None
        return 2.0 * (self.outer_boundary_radius_m - self.charge_radius_m) / self.sound_speed_ambient_m_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "charge_radius_m": self.charge_radius_m,
            "charge_mass_kg": self.charge_mass_kg,
            "unit_system": self.unit_system,
            "tnt_initial_density_kg_m3": self.tnt_initial_density_kg_m3,
            "jwl": self.jwl.as_dict(),
            "jwl_volume_energy_j_m3": self.jwl_volume_energy_j_m3,
            "detonation_velocity_m_s": self.detonation_velocity_m_s,
            "product_initial_specific_energy_j_kg": self.product_initial_specific_energy_j_kg,
            "ambient_density_kg_m3": self.ambient_density_kg_m3,
            "ambient_pressure_pa": self.ambient_pressure_pa,
            "gamma_air": self.gamma_air,
            "mesh_size_m": self.mesh_size_m,
            "output_interval_s": self.output_interval_s,
            "outer_boundary_radius_m": self.outer_boundary_radius_m,
            "valid_end_time_s": self.valid_end_time_s,
            "split": self.split,
            "ambient_specific_energy_j_kg": self.ambient_specific_energy_j_kg,
            "ambient_sound_speed_m_s": self.sound_speed_ambient_m_s,
            "reflection_arrival_time_s": self.reflection_arrival_time_s,
            "time_alignment_method": self.time_alignment_method,
            "time_alignment_confidence": self.time_alignment_confidence,
            "time_alignment_uncertainty_s": self.time_alignment_uncertainty_s,
            "notes": list(self.notes),
        }


def charge_mass_from_radius(radius_m: float, density_kg_m3: float) -> float:
    """Mass of a uniform spherical charge."""
    if radius_m <= 0.0 or density_kg_m3 <= 0.0:
        raise DataSchemaError("radius and density must be positive")
    return FOUR_PI / 3.0 * density_kg_m3 * radius_m**3


def charge_radius_from_mass(mass_kg: float, density_kg_m3: float) -> float:
    if mass_kg <= 0.0 or density_kg_m3 <= 0.0:
        raise DataSchemaError("mass and density must be positive")
    return (3.0 * mass_kg / (FOUR_PI * density_kg_m3)) ** (1.0 / 3.0)


# --------------------------------------------------------------------------
# Frame validation
# --------------------------------------------------------------------------


@dataclass
class FrameValidation:
    """Outcome of validating one raw frame."""

    ok: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, float] = field(default_factory=dict)

    def raise_if_failed(self, context: str = "frame") -> None:
        if not self.ok:
            raise DataSchemaError(f"{context} failed schema validation: " + "; ".join(self.problems))


def validate_frame(
    frame: Mapping[str, Any],
    mapping: FieldMapping,
    *,
    require_finite: bool = True,
) -> FrameValidation:
    """Check that a raw frame contains the declared columns and sane values.

    ``frame`` is a mapping of *source column name* -> array (a pandas
    ``DataFrame`` works directly).  Values are expected in **deck units** at
    this stage; unit conversion happens afterwards.
    """
    problems: list[str] = []
    warnings: list[str] = []
    statistics: dict[str, float] = {}

    for logical in REQUIRED_FIELDS:
        source = mapping.source(logical)
        if source is None:
            if logical in DERIVABLE_FIELDS:
                warnings.append(f"{logical} is not mapped and will be derived")
                continue
            problems.append(f"required logical field {logical!r} is not mapped")
            continue
        if source not in frame:
            problems.append(f"source column {source!r} for logical field {logical!r} is absent")

    if problems:
        return FrameValidation(False, problems, warnings, statistics)

    def values(logical: str) -> np.ndarray:
        return np.asarray(frame[mapping.require(logical)], dtype=np.float64)

    if require_finite:
        for logical in REQUIRED_FIELDS:
            array = values(logical)
            if not np.all(np.isfinite(array)):
                problems.append(f"logical field {logical!r} contains non-finite values")

    density = values("density")
    if np.any(density <= 0.0):
        problems.append(f"density has {int(np.count_nonzero(density <= 0))} non-positive entries")

    alpha = values("tnt_volume_fraction")
    if np.any((alpha < -1.0e-6) | (alpha > 1.0 + 1.0e-6)):
        problems.append("tnt_volume_fraction leaves [0, 1]")

    radius = np.sqrt(values("x") ** 2 + values("y") ** 2 + values("z") ** 2)
    statistics["radius_min"] = float(radius.min())
    statistics["radius_max"] = float(radius.max())
    statistics["density_min"] = float(density.min())
    statistics["density_max"] = float(density.max())
    statistics["alpha_min"] = float(alpha.min())
    statistics["alpha_max"] = float(alpha.max())
    statistics["n_cells"] = float(radius.size)
    statistics["nonfinite_count"] = float(
        sum(int(np.count_nonzero(~np.isfinite(values(f)))) for f in REQUIRED_FIELDS)
    )
    if statistics["nonfinite_count"]:
        problems.append(f"{int(statistics['nonfinite_count'])} non-finite entries across required fields")

    return FrameValidation(not problems, problems, warnings, statistics)


def validate_case_metadata(
    metadata: CaseMetadata,
    *,
    declared_mass_kg: float | None = None,
    mass_rtol: float = 5.0e-3,
) -> FrameValidation:
    """Consistency checks that catch a mis-transcribed case block."""
    problems: list[str] = []
    warnings: list[str] = []
    statistics: dict[str, float] = {}

    derived_mass = charge_mass_from_radius(
        metadata.charge_radius_m, metadata.tnt_initial_density_kg_m3
    )
    statistics["derived_charge_mass_kg"] = derived_mass
    if declared_mass_kg is not None:
        statistics["declared_charge_mass_kg"] = declared_mass_kg
        deviation = abs(derived_mass - declared_mass_kg) / max(declared_mass_kg, 1.0e-30)
        statistics["charge_mass_relative_deviation"] = deviation
        if deviation > mass_rtol:
            problems.append(
                f"charge mass mismatch: sphere of radius {metadata.charge_radius_m} m at "
                f"{metadata.tnt_initial_density_kg_m3} kg/m^3 gives {derived_mass:.6g} kg, "
                f"but the metadata declares {declared_mass_kg:.6g} kg "
                f"(relative deviation {deviation:.3e})"
            )

    if metadata.reflection_arrival_time_s is not None:
        statistics["reflection_arrival_time_s"] = metadata.reflection_arrival_time_s
        if (
            metadata.valid_end_time_s is not None
            and metadata.valid_end_time_s > metadata.reflection_arrival_time_s
        ):
            warnings.append(
                "the case's valid end time exceeds the earliest outer-boundary "
                "reflection arrival; free-field data beyond "
                f"{metadata.reflection_arrival_time_s:.6g} s may be contaminated"
            )

    if metadata.mesh_size_m is None:
        warnings.append("mesh_size_m is not declared; the t_eps gap criterion cannot use it")
    if metadata.outer_boundary_radius_m is None:
        warnings.append("outer_boundary_radius_m is not declared; reflections cannot be bounded")

    return FrameValidation(not problems, problems, warnings, statistics)


def describe_missing_fields(mapping: FieldMapping, metadata: CaseMetadata) -> dict[str, list[str]]:
    """Report which logical fields and metadata entries are unavailable.

    This is the honest answer to "what is still missing from the real data",
    produced from the data itself rather than from a hand-maintained list.
    """
    return {
        "unmapped_logical_fields": mapping.missing_required(),
        "derived_logical_fields": mapping.derivable(),
        "missing_metadata": [
            name
            for name in ("mesh_size_m", "output_interval_s", "outer_boundary_radius_m",
                         "valid_end_time_s")
            if getattr(metadata, name) is None
        ],
        "alignment_fields": [
            name
            for name in ("time_alignment_method", "time_alignment_confidence",
                         "time_alignment_uncertainty_s")
            if getattr(metadata, name) is None
        ],
    }


def build_field_mapping(pairs: Iterable[tuple[str, str]]) -> FieldMapping:
    return FieldMapping({logical: source for logical, source in pairs})


__all__ = [
    "DERIVABLE_FIELDS",
    "LOGICAL_FIELDS",
    "REQUIRED_FIELDS",
    "CaseMetadata",
    "DataSchemaError",
    "FieldMapping",
    "FrameValidation",
    "build_field_mapping",
    "charge_mass_from_radius",
    "charge_radius_from_mass",
    "describe_missing_fields",
    "validate_case_metadata",
    "validate_frame",
]
