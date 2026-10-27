"""Convert extracted DYNA profiles into the new supervision contract.

The extractor already writes the material-partitioned shell profiles, fronts,
field validity and provenance.  This module is the narrow bridge from that
existing output to :class:`~spherical_blast_pinn.data.startup_supervision.SupervisionDataset`.
It does not read d3plot again and does not re-detect fronts.

Time is explicit and is applied in one place::

    T_origin = LT + R0 / D
    t_phys   = T_DYNA - T_origin
    tau      = t_phys - epsilon_case

``LT`` is the deck's ``*INITIAL_DETONATION`` lighting time and ``D`` is the
deck detonation velocity.  The formula is valid only for the confirmed centre-
detonation cases in this study; a non-zero detonation point or a missing LT/D
is refused rather than approximated.

Values in the extracted profile are SI.  The converter does not rescale density,
velocity, pressure or specific energy: those units were converted exactly once
by the extractor.  It preserves per-field validity, keeps ambient/mixed points
unlabelled, and never turns a missing value into zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..constants import UNIT_SYSTEMS
from .extracted_case import ExtractedCase
from .startup_supervision import (
    AIR_CODE,
    NO_MEDIUM,
    PRODUCT_CODE,
    SOURCE_REAL,
    SOURCE_SYNTHETIC,
    SupervisionDataset,
    SupervisionError,
    TimeOrigin,
)


class DynaSupervisionError(ValueError):
    """The extracted DYNA profiles cannot be converted under the stated mapping."""


@dataclass(frozen=True)
class DynaOriginParameters:
    """The explicitly recorded pieces of the DYNA-to-model time mapping."""

    charge_radius_m: float
    detonation_velocity_m_s: float
    lighting_time_s: float
    epsilon_s: float

    def __post_init__(self) -> None:
        values = {
            "charge_radius_m": self.charge_radius_m,
            "detonation_velocity_m_s": self.detonation_velocity_m_s,
            "lighting_time_s": self.lighting_time_s,
            "epsilon_s": self.epsilon_s,
        }
        for name, value in values.items():
            number = float(value)
            if not math.isfinite(number):
                raise DynaSupervisionError(f"{name} must be finite, got {value!r}")
            if name == "lighting_time_s":
                if number < 0.0:
                    raise DynaSupervisionError(
                        f"lighting_time_s must be non-negative, got {value!r}"
                    )
            elif number <= 0.0:
                raise DynaSupervisionError(
                    f"{name} must be positive, got {value!r}"
                )
        object.__setattr__(self, "charge_radius_m", float(self.charge_radius_m))
        object.__setattr__(
            self, "detonation_velocity_m_s", float(self.detonation_velocity_m_s)
        )
        object.__setattr__(self, "lighting_time_s", float(self.lighting_time_s))
        object.__setattr__(self, "epsilon_s", float(self.epsilon_s))

    @property
    def origin_time_s(self) -> float:
        return float(self.lighting_time_s + self.charge_radius_m / self.detonation_velocity_m_s)

    def as_dict(self) -> dict[str, Any]:
        return {
            "formula": "T_origin = LT + R0 / D",
            "charge_radius_m": self.charge_radius_m,
            "detonation_velocity_m_s": self.detonation_velocity_m_s,
            "lighting_time_s": self.lighting_time_s,
            "origin_time_s": self.origin_time_s,
            "epsilon_case_s": self.epsilon_s,
            "mapping": "t_phys = T_DYNA - T_origin; tau = t_phys - epsilon_case",
            "basis": (
                "centre detonation; T_origin is the estimated arrival time of the "
                "detonation wave at the initial charge surface, not a measured burn-"
                "fraction completion time"
            ),
        }

    def as_time_origin(self) -> TimeOrigin:
        return TimeOrigin(
            status="confirmed",
            origin_time_s=self.origin_time_s,
            method="LT + R0 / D",
            evidence=(
                f"LT={self.lighting_time_s!r} s, R0={self.charge_radius_m!r} m, "
                f"D={self.detonation_velocity_m_s!r} m/s; "
                f"epsilon_case={self.epsilon_s!r} s"
            ),
            source="dyna_extraction",
        )


def origin_parameters_from_metadata(
    case: ExtractedCase,
    *,
    epsilon_s: float,
    detonation_velocity_m_s: float | None = None,
    lighting_time_s: float | None = None,
) -> DynaOriginParameters:
    """Read LT and D from the extraction metadata, with explicit overrides.

    The parameters are SI.  A metadata LT is in deck time units and is converted
    exactly once here.  A non-centre ``*INITIAL_DETONATION`` or a missing value
    is a hard failure: the ``LT + R0/D`` origin assumes a centre detonation.
    """
    metadata = case.source.get("metadata_raw") or {}
    deck = metadata.get("deck") or {}
    radius = float(case.charge_radius)

    if detonation_velocity_m_s is None:
        raw_d = case.source.get("detonation_velocity_m_s")
        if raw_d is None:
            raw_d = metadata.get("detonation_velocity_m_s")
        if raw_d is None:
            raise DynaSupervisionError(
                "the extraction metadata carries no detonation velocity D; pass "
                "--detonation-velocity explicitly"
            )
        velocity = float(raw_d)
    else:
        velocity = float(detonation_velocity_m_s)

    point = deck.get("initial_detonation")
    if point is not None:
        coordinates = np.asarray(point, dtype=np.float64).reshape(-1)
        if coordinates.size != 3 or np.any(coordinates != 0.0):
            raise DynaSupervisionError(
                "the deck's *INITIAL_DETONATION point is not the origin "
                f"({point!r}); T_origin = LT + R0/D is only valid for the confirmed "
                "centre-detonation cases"
            )

    if lighting_time_s is None:
        raw_lt = deck.get("initial_detonation_time")
        if raw_lt is None:
            raise DynaSupervisionError(
                "the extraction metadata carries no *INITIAL_DETONATION lighting "
                "time LT; pass --lighting-time-s explicitly"
            )
        unit_name = ((metadata.get("units") or {}).get("source") or deck.get("unit_system"))
        if unit_name not in UNIT_SYSTEMS:
            raise DynaSupervisionError(
                f"cannot convert LT from unknown deck unit system {unit_name!r}; "
                "pass --lighting-time-s in seconds explicitly"
            )
        lighting = float(raw_lt) * UNIT_SYSTEMS[str(unit_name)].time_to_si
    else:
        lighting = float(lighting_time_s)

    return DynaOriginParameters(
        charge_radius_m=radius,
        detonation_velocity_m_s=velocity,
        lighting_time_s=lighting,
        epsilon_s=float(epsilon_s),
    )


def _companion_arrays(case: ExtractedCase) -> dict[str, np.ndarray]:
    path_value = case.source.get("material_companion")
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _field_masks(
    case: ExtractedCase,
    medium: np.ndarray,
    field_validity: Mapping[str, np.ndarray] | None,
    *,
    synthetic: bool,
) -> tuple[dict[str, np.ndarray], str]:
    if field_validity is None:
        field_validity = case.field_validity
    if field_validity is None:
        if not synthetic:
            raise DynaSupervisionError(
                "this extraction has no per-field validity arrays "
                "(field_valid_density/velocity/energy/pressure). Re-extract it with "
                "the current extractor; a joint fallback would silently drop a valid "
                "field when another field is missing"
            )
        masks = {
            name: np.asarray(case.valid, dtype=bool)
            for name in ("density", "velocity", "specific_energy", "pressure")
        }
        source = "joint valid mask (synthetic interface only)"
    else:
        missing = sorted(
            set(("density", "velocity", "specific_energy", "pressure"))
            - set(field_validity)
        )
        if missing:
            raise DynaSupervisionError(
                f"per-field validity is missing {missing}"
            )
        masks = {
            name: np.asarray(field_validity[name], dtype=bool)
            for name in ("density", "velocity", "specific_energy", "pressure")
        }
        source = "extracted per-field masks"
    for name, mask in masks.items():
        if mask.shape != medium.shape:
            raise DynaSupervisionError(
                f"valid_{name} has shape {mask.shape}, expected {medium.shape}"
            )
        masks[name] = mask & (medium != NO_MEDIUM)
    return masks, source


def _positive_median(values: Sequence[float] | np.ndarray, name: str) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array) & (array > 0.0)]
    if finite.size == 0:
        raise DynaSupervisionError(
            f"{name} has no finite positive values; cannot define L_geometry"
        )
    return float(np.median(finite))


def supervision_from_extracted_case(
    case: ExtractedCase,
    *,
    time_origin: TimeOrigin,
    epsilon_s: float,
    synthetic: bool = False,
    contact_valid: np.ndarray | None = None,
    shock_valid: np.ndarray | None = None,
    contact_delta_m: float | None = None,
    shock_delta_m: float | None = None,
    field_validity: Mapping[str, np.ndarray] | None = None,
    notes: Sequence[str] | None = None,
    source_extra: Mapping[str, Any] | None = None,
) -> SupervisionDataset:
    """Convert one :class:`ExtractedCase` without changing its physical values."""
    if not isinstance(case, ExtractedCase):
        raise DynaSupervisionError("case must be an ExtractedCase")
    if not isinstance(time_origin, TimeOrigin):
        raise DynaSupervisionError("time_origin must be a TimeOrigin")
    epsilon = float(epsilon_s)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise DynaSupervisionError(f"epsilon_s must be finite and positive, got {epsilon_s!r}")
    if not isinstance(synthetic, bool):
        raise DynaSupervisionError("synthetic must be an explicit boolean")

    times = np.asarray(case.times, dtype=np.float64).reshape(-1)
    radius = np.asarray(case.radius, dtype=np.float64).reshape(-1)
    if times.size == 0 or radius.size < 2:
        raise DynaSupervisionError("the extracted case has no usable time/radius grid")
    fields = {
        "density": np.asarray(case.density, dtype=np.float64),
        "velocity": np.asarray(case.velocity, dtype=np.float64),
        "specific_energy": np.asarray(case.specific_energy, dtype=np.float64),
        "pressure": np.asarray(case.pressure, dtype=np.float64),
    }
    for name, values in fields.items():
        if values.shape != (times.size, radius.size):
            raise DynaSupervisionError(
                f"{name} has shape {values.shape}, expected {(times.size, radius.size)}"
            )

    material = np.asarray(case.material)
    if material.shape != (times.size, radius.size):
        raise DynaSupervisionError(
            f"material has shape {material.shape}, expected {(times.size, radius.size)}"
        )
    medium = np.full(material.shape, NO_MEDIUM, dtype=np.int8)
    medium[material == 0] = PRODUCT_CODE
    medium[material == 1] = AIR_CODE
    if not np.all(np.isin(material, (0, 1, 2))):
        raise DynaSupervisionError("material must contain only the codes 0/1/2")

    masks, validity_source = _field_masks(
        case, medium, field_validity, synthetic=synthetic
    )
    any_field_valid = (
        masks["density"]
        | masks["velocity"]
        | masks["specific_energy"]
        | masks["pressure"]
    )
    # "Labelled product/air but no usable field" is a contradiction in the
    # supervision contract.  The extractor can leave a contaminated or
    # no-EOS-state shell labelled while all four fields are missing; do not
    # invent a value or keep the label.  Demote only those points to unlabelled.
    no_state = (medium != NO_MEDIUM) & ~any_field_valid
    unlabelled_count = int(np.count_nonzero(no_state))
    medium[no_state] = NO_MEDIUM
    for name in masks:
        masks[name] = masks[name] & (medium != NO_MEDIUM)

    companion = _companion_arrays(case)
    if contact_valid is None:
        contact_valid = np.asarray(case.contact_valid, dtype=bool)
    else:
        contact_valid = np.asarray(contact_valid, dtype=bool)
    if shock_valid is None:
        shock_valid = np.asarray(case.shock_valid, dtype=bool)
    else:
        shock_valid = np.asarray(shock_valid, dtype=bool)
    # The extractor's eligibility arrays are evidence; the ExtractedCase may also
    # have applied a front-order/inside-charge filter.  Intersect both when the
    # companion exists so neither rule silently widens the other.
    for name, key, mask in (
        ("contact_valid", "contact_has_supervision", contact_valid),
        ("shock_valid", "shock_has_supervision", shock_valid),
    ):
        if key in companion:
            mask &= np.asarray(companion[key], dtype=bool)
        if mask.shape != (times.size,):
            raise DynaSupervisionError(
                f"{name} has shape {mask.shape}, expected {(times.size,)}"
            )
        if name == "contact_valid":
            contact_valid = mask
        else:
            shock_valid = mask

    contact_radius = np.asarray(case.contact_radius, dtype=np.float64).reshape(-1)
    shock_radius = np.asarray(case.shock_radius, dtype=np.float64).reshape(-1)
    if contact_radius.size != times.size or shock_radius.size != times.size:
        raise DynaSupervisionError("front arrays must have one value per DYNA state")

    if contact_delta_m is None:
        uncertainty = companion.get("contact_position_uncertainty_m")
        if uncertainty is not None:
            # The uncertainty is a property of the locator, not of the label's
            # validity: one unusable contact frame must not make the shock/
            # field supervision impossible.  It is used only where a contact
            # label is actually valid.
            contact_delta = _positive_median(
                np.asarray(uncertainty),
                "contact_position_uncertainty_m",
            )
            contact_delta_source = "material.npz contact_position_uncertainty_m (median)"
        else:
            contact_delta = float(case.contact_position_uncertainty())
            contact_delta_source = "ExtractedCase.detector resolution fallback"
    else:
        contact_delta = float(contact_delta_m)
        contact_delta_source = "caller override"
    if shock_delta_m is None:
        thickness = companion.get("shock_cell_thickness_m")
        if thickness is not None:
            shock_delta = _positive_median(
                thickness, "shock_cell_thickness_m"
            )
            shock_delta_source = "material.npz shock_cell_thickness_m (median)"
        else:
            shock_delta = float(case.shock_position_uncertainty())
            shock_delta_source = "ExtractedCase.grid spacing fallback"
    else:
        shock_delta = float(shock_delta_m)
        shock_delta_source = "caller override"

    metadata = case.source.get("metadata_raw") or {}
    source = {
        "profile_directory": str(case.source.get("directory", "")),
        "conversion": "extracted DYNA shell profile -> supervision contract",
        "field_validity_source": validity_source,
        "front_position_source": {
            "contact": "material.npz contact_has_supervision / contact_radius; "
                       "ExtractedCase.contact_valid is intersected when available",
            "shock": "material.npz shock_has_supervision / shock_radius; "
                     "ExtractedCase.shock_valid is intersected when available",
            "contact_delta": contact_delta_source,
            "shock_delta": shock_delta_source,
        },
        "time_origin_parameters": {
            "epsilon_case_s": epsilon,
            "time_origin": time_origin.as_dict(),
        },
        "units": "SI as written by cli.extract_profiles; no second conversion",
        "density_definition": (
            "per-material ALE mass / (alpha_m * cell_volume), aggregated on pure-"
            "medium shells as sum(m_m)/sum(alpha_m V); the mixture density column is "
            "not substituted and mixed cells are unlabelled, never copied to both sides"
        ),
        "velocity_definition": (
            "radial projection of the eight-node mean cell velocity, "
            "u_r = u . (x-x0)/|x-x0|; sign is preserved and r=0 is zero"
        ),
        "pressure_definition": (
            "negative trace of the solid-element stress, p = -(sxx+syy+szz)/3; "
            "no unconfirmed artificial-viscosity correction was applied"
        ),
        "energy_definition": (
            "EOS-derived from (rho,p): e=(p-p_cold)/(omega rho) for products and "
            "e=p/((gamma-1)rho) for air; non-positive EOS states are unlabelled"
        ),
        "extraction": {
            key: metadata.get(key)
            for key in (
                "case_id", "R_0", "t_0", "t_end", "n_x", "n_t", "deck",
                "d3plot_directory", "field_provenance", "material_partition_source",
                "material_partition_is_volume_fraction_based", "profile_layout",
                "units", "symmetry_factor", "pressure_kind", "pressure_source",
            )
            if key in metadata
        },
    }
    if source_extra:
        source.update(dict(source_extra))
    source["unlabelled_no_state_points"] = unlabelled_count
    notes_out = list(notes or [])
    notes_out.append(
        "density/velocity/pressure/specific_energy are copied from the extracted "
        "profile in SI; invalid points remain invalid and are never zero-filled"
    )
    if unlabelled_count:
        notes_out.append(
            f"{unlabelled_count} labelled point(s) had no valid field at all and were "
            "demoted to unclassified; no value was invented or copied to either side"
        )
    if not synthetic and validity_source != "extracted per-field masks":
        notes_out.append(
            "per-field validity came from the joint mask; this is only permitted for "
            "the synthetic interface test"
        )

    dataset = SupervisionDataset(
        case_id=str(case.case_id),
        charge_radius_m=float(case.charge_radius),
        synthetic=bool(synthetic),
        source_kind=SOURCE_SYNTHETIC if synthetic else SOURCE_REAL,
        time_origin=time_origin,
        times_dyna=times,
        radius=radius,
        density=fields["density"],
        velocity=fields["velocity"],
        specific_energy=fields["specific_energy"],
        pressure=fields["pressure"],
        valid_density=masks["density"],
        valid_velocity=masks["velocity"],
        valid_specific_energy=masks["specific_energy"],
        valid_pressure=masks["pressure"],
        medium=medium,
        contact_radius=contact_radius,
        contact_valid=contact_valid,
        shock_radius=shock_radius,
        shock_valid=shock_valid,
        contact_delta_m=contact_delta,
        shock_delta_m=shock_delta,
        source=source,
        notes=notes_out,
    )
    return dataset


def pressure_gauge_series(
    dataset: SupervisionDataset,
    *,
    radius_m: float,
    ambient_pressure_pa: float,
    epsilon_s: float,
) -> dict[str, Any]:
    """Pressure history at a fixed physical radius from a supervision dataset.

    The extracted radial grid is an Eulerian, fixed spatial support grid.  A
    point is available only when the requested radius lies between two **air**
    samples with valid pressure in the same frame; no extrapolation and no
    material-interface crossing is performed.  The result does not claim to
    distinguish incident from reflected pressure because no gauge orientation
    is present in the d3plot output.
    """
    if not isinstance(dataset, SupervisionDataset):
        raise DynaSupervisionError("dataset must be a SupervisionDataset")
    radius = float(radius_m)
    ambient = float(ambient_pressure_pa)
    epsilon = float(epsilon_s)
    if not math.isfinite(radius) or radius < 0.0:
        raise DynaSupervisionError(f"gauge radius must be finite and non-negative, got {radius_m!r}")
    if not math.isfinite(ambient):
        raise DynaSupervisionError(f"ambient pressure must be finite, got {ambient_pressure_pa!r}")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise DynaSupervisionError(f"epsilon_s must be finite and non-negative, got {epsilon_s!r}")

    times_dyna = np.asarray(dataset.times_dyna, dtype=np.float64)
    pressure = np.full(times_dyna.size, np.nan)
    valid = np.zeros(times_dyna.size, dtype=bool)
    reasons: list[str] = []

    for frame in range(times_dyna.size):
        radial = np.asarray(dataset.radius[frame], dtype=np.float64)
        if radius < radial[0] or radius > radial[-1]:
            reasons.append("outside_radial_grid")
            continue
        exact = np.flatnonzero(np.isclose(radial, radius, rtol=0.0, atol=1.0e-15))
        if exact.size:
            index = int(exact[0])
            if dataset.medium[frame, index] != AIR_CODE:
                reasons.append("exact_point_not_air")
                continue
            if not dataset.valid_pressure[frame, index]:
                reasons.append("invalid_pressure")
                continue
            pressure[frame] = dataset.pressure[frame, index]
            valid[frame] = True
            reasons.append("exact_air_sample")
            continue

        right = int(np.searchsorted(radial, radius, side="right"))
        left = right - 1
        if left < 0 or right >= radial.size:
            reasons.append("outside_radial_grid")
            continue
        if (
            dataset.medium[frame, left] != AIR_CODE
            or dataset.medium[frame, right] != AIR_CODE
        ):
            reasons.append("bracket_not_all_air")
            continue
        if (
            not dataset.valid_pressure[frame, left]
            or not dataset.valid_pressure[frame, right]
        ):
            reasons.append("invalid_pressure_support")
            continue
        r0 = float(radial[left])
        r1 = float(radial[right])
        p0 = float(dataset.pressure[frame, left])
        p1 = float(dataset.pressure[frame, right])
        pressure[frame] = p0 + (p1 - p0) * (radius - r0) / (r1 - r0)
        valid[frame] = True
        reasons.append("linear_within_air_bracket")

    if dataset.time_origin.ready:
        physical = dataset.absolute_physical_time()
        tau = dataset.tau_time(epsilon)
    else:
        physical = None
        tau = None
    overpressure = pressure - ambient
    return {
        "radius_m": radius,
        "tracking": "fixed_spatial_eulerian_shell",
        "origin_status": dataset.time_origin.status,
        "times_dyna_s": [float(v) for v in times_dyna],
        "times_phys_s": None if physical is None else [float(v) for v in physical],
        "tau_s": None if tau is None else [float(v) for v in tau],
        "pressure_pa": [
            float(v) if math.isfinite(float(v)) else None for v in pressure
        ],
        "overpressure_pa": [
            float(v) if math.isfinite(float(v)) else None for v in overpressure
        ],
        "valid": [bool(v) for v in valid],
        "reason": reasons,
        "n_valid": int(np.count_nonzero(valid)),
        "n_frames": int(times_dyna.size),
        "interpolation": (
            "linear inside one air-only radial bracket; no extrapolation; "
            "no material-interface crossing"
        ),
        "incident_or_reflected": "undetermined (no gauge orientation in the d3plot profile)",
        "missing_policy": "unavailable points are null/False, never zero",
    }


__all__ = [
    "DynaOriginParameters",
    "DynaSupervisionError",
    "origin_parameters_from_metadata",
    "pressure_gauge_series",
    "supervision_from_extracted_case",
]
