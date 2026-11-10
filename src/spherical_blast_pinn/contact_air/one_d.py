"""One-dimensional radial ALE data entry, contact labels and shock detection.

This module is the data-side boundary of the ContactNet/AirNet scheme.  It
never samples a 3-D field, never assigns spherical shells and never averages
across directions.  A *standard 1-D frame* carries the radial cells of one raw
output state:

* ``cell_id``, ``r_left``, ``r_right`` and the representative radius;
* product/air volume fractions and an optional material identifier;
* density, radial velocity and **absolute** pressure;
* independent per-field validity masks;
* the position type (cell representative or node) and explicit units.

The binary LS-DYNA ``ALE1D`` layout is **not** guessed here.  A real reader must
first establish the file sample, field mapping and software version.  Until
then the supported entry is the documented intermediate exchange format below,
read by :func:`read_standard_1d`.  Synthetic fixtures produced by tests are
marked ``synthetic`` and are never TNT validation data.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks

STANDARD_FORMAT = "spherical_blast_pinn.standard_1d_source"
STANDARD_VERSION = 1
READER_VERSION = "one_d.standard/1"

#: Multipliers to SI.  Unit names are normalised (lower-case, ``µ`` -> ``u``).
_LENGTH = {"m": 1.0, "meter": 1.0, "metre": 1.0, "mm": 1.0e-3, "cm": 1.0e-2}
_TIME = {"s": 1.0, "sec": 1.0, "second": 1.0, "ms": 1.0e-3, "us": 1.0e-6}
_DENSITY = {
    "kg/m^3": 1.0, "kg/m3": 1.0, "kg_m3": 1.0,
    "g/cm^3": 1.0e3, "g/cm3": 1.0e3, "g_cm3": 1.0e3,
    "t/mm^3": 1.0e12, "t/mm3": 1.0e12,
}
_VELOCITY = {
    "m/s": 1.0, "m_s": 1.0,
    "mm/us": 1.0e3, "mm_us": 1.0e3, "km/s": 1.0e3, "km_s": 1.0e3,
    "cm/us": 1.0e4, "cm_us": 1.0e4,
}
_PRESSURE = {
    "pa": 1.0, "kpa": 1.0e3, "mpa": 1.0e6, "gpa": 1.0e9,
    "bar": 1.0e5, "kbar": 1.0e8,
}


class OneDDataError(RuntimeError):
    """A 1-D source is missing, inconsistent or not in SI after conversion."""


#: Absolute/relative tolerance for deciding whether two radial cells share a
#: boundary.  A positive gap larger than this is a real spatial gap and must
#: break the run; a negative gap smaller than ``-tolerance`` is an overlap.
GEOMETRY_TOLERANCE_M = 1.0e-9


def cell_gap_m(frame: "RadialFrame", left: int, right: int) -> float:
    """Signed gap from cell ``left`` right edge to cell ``right`` left edge."""
    if int(right) != int(left) + 1:
        return float("inf")
    return float(frame.r_left[int(right)]) - float(frame.r_right[int(left)])


def cells_geometrically_adjacent(
    frame: "RadialFrame", left: int, right: int, *, tolerance_m: float,
) -> bool:
    """True only when consecutive cells share a boundary within tolerance."""
    if int(right) != int(left) + 1:
        return False
    width = 0.5 * (
        float(frame.r_right[int(left)] - frame.r_left[int(left)])
        + float(frame.r_right[int(right)] - frame.r_left[int(right)])
    )
    tolerance = max(float(tolerance_m), 0.0) + 1.0e-9 * max(width, 1.0e-30)
    return abs(cell_gap_m(frame, left, right)) <= tolerance


def _geometric_runs(
    frame: "RadialFrame", indices: np.ndarray, *, tolerance_m: float,
) -> list[np.ndarray]:
    """Split sorted cell indices at both array gaps and real spatial gaps."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return []
    runs: list[np.ndarray] = []
    start = 0
    for position in range(1, indices.size + 1):
        if position == indices.size or not cells_geometrically_adjacent(
            frame, int(indices[position - 1]), int(indices[position]),
            tolerance_m=tolerance_m,
        ):
            runs.append(indices[start:position])
            start = position
    return runs


def _unit(value: Any, table: Mapping[str, float], *, what: str) -> float:
    if value is None:
        raise OneDDataError(
            f"{what} unit is not declared; units are never inferred from file "
            "names or numeric magnitudes"
        )
    key = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
    key = key.replace("per ", "/")
    if key not in table:
        raise OneDDataError(
            f"{what} unit {value!r} is not supported; supported units are "
            f"{sorted(table)}"
        )
    return float(table[key])


@dataclass
class RadialFrame:
    """One raw output state expressed on its own radial cells."""

    case_id: str
    charge_radius_m: float
    frame_index: int
    time_raw_s: float
    cell_id: np.ndarray
    r_left: np.ndarray
    r_right: np.ndarray
    radius: np.ndarray
    alpha_product: np.ndarray
    alpha_air: np.ndarray | None
    material: np.ndarray | None
    density: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    valid_density: np.ndarray
    valid_velocity: np.ndarray
    valid_pressure: np.ndarray
    position_type: str
    units: dict[str, str]
    source: dict[str, Any] = field(default_factory=dict)

    def air_mask(self, pure_air_threshold: float) -> np.ndarray:
        """Air occupancy from volume fractions only (no field finiteness)."""
        if self.alpha_air is not None:
            return np.asarray(self.alpha_air, dtype=np.float64) >= float(pure_air_threshold)
        return np.asarray(self.alpha_product, dtype=np.float64) <= (
            1.0 - float(pure_air_threshold)
        )

    def require_monotone(self) -> None:
        if self.radius.size < 2:
            raise OneDDataError(
                f"case {self.case_id!r} frame {self.frame_index}: fewer than two "
                "radial cells cannot define a profile"
            )
        if not np.all(np.isfinite(self.radius)):
            raise OneDDataError(
                f"case {self.case_id!r} frame {self.frame_index}: non-finite radius"
            )
        if np.any(np.diff(self.radius) <= 0.0):
            raise OneDDataError(
                f"case {self.case_id!r} frame {self.frame_index}: radial positions "
                "must be strictly increasing"
            )
        n = int(self.radius.size)
        for name, value in (
            ("r_left", self.r_left), ("r_right", self.r_right),
            ("alpha_product", self.alpha_product), ("density", self.density),
            ("velocity", self.velocity), ("pressure", self.pressure),
            ("valid_density", self.valid_density),
            ("valid_velocity", self.valid_velocity),
            ("valid_pressure", self.valid_pressure),
        ):
            if np.asarray(value).reshape(-1).size != n:
                raise OneDDataError(
                    f"case {self.case_id!r} frame {self.frame_index}: {name} has "
                    f"{np.asarray(value).size} entries, expected {n}"
                )
        if self.alpha_air is not None and self.alpha_air.size != n:
            raise OneDDataError(
                f"case {self.case_id!r} frame {self.frame_index}: alpha_air has "
                f"{self.alpha_air.size} entries, expected {n}"
            )
        if self.material is not None and self.material.size != n:
            raise OneDDataError(
                f"case {self.case_id!r} frame {self.frame_index}: material has "
                f"{self.material.size} entries, expected {n}"
            )
        if np.any(self.r_right <= self.r_left):
            raise OneDDataError(
                f"case {self.case_id!r} frame {self.frame_index}: cell r_right must "
                "exceed r_left"
            )
        if np.any(self.radius < self.r_left) or np.any(self.radius > self.r_right):
            raise OneDDataError(
                f"case {self.case_id!r} frame {self.frame_index}: representative "
                "radius lies outside its cell bounds"
            )
        width = np.maximum(self.r_right - self.r_left, 1.0e-30)
        tolerance = GEOMETRY_TOLERANCE_M + 1.0e-9 * float(np.max(width))
        overlap = self.r_left[1:] < (self.r_right[:-1] - tolerance)
        if np.any(overlap):
            raise OneDDataError(
                f"case {self.case_id!r} frame {self.frame_index}: radial cells overlap; "
                "an overlapping mesh is rejected (a positive gap is allowed and "
                "handled as a spatial break)"
            )


@dataclass
class RadialCase:
    case_id: str
    charge_radius_m: float
    frames: list[RadialFrame]
    units: dict[str, str]
    position_type: str
    source: dict[str, Any] = field(default_factory=dict)


def _frame_arrays(archive, metadata: Mapping[str, Any]) -> list[dict[str, np.ndarray]]:
    offsets = np.asarray(archive["frame_offsets"], dtype=np.int64)
    n_frames = int(offsets.size - 1)
    if n_frames < 1:
        raise OneDDataError("frame_offsets must contain at least two entries")
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(np.asarray(archive["radius"])):
        raise OneDDataError("frame_offsets do not span the concatenated arrays")
    if tuple(np.diff(offsets).tolist()) != tuple(
        int(v) for v in np.asarray(metadata.get("cells_per_frame", []))
    ):
        raise OneDDataError(
            "metadata.cells_per_frame does not match frame_offsets"
        )
    frames = []
    for index in range(n_frames):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        frames.append({name: np.asarray(archive[name])[start:stop] for name in archive.files
                       if name not in ("metadata", "frame_offsets")})
    return frames


def read_standard_1d(path: Path | str) -> RadialCase:
    """Read the documented intermediate exchange format into SI.

    Supported layouts:

    * ``<path>.npz`` containing ``metadata`` and the concatenated arrays;
    * a directory with ``metadata.json`` and ``data.npz``.

    Required metadata keys: ``format``, ``version``, ``case_id``,
    ``charge_radius_m``, ``units`` (length/time/density/velocity/pressure),
    ``position_type`` and ``absolute_pressure``.  Missing or unknown units are
    hard errors.  A per-field mask that is true over a non-finite value is an
    error, not a silently dropped sample.
    """
    path = Path(path)
    if path.is_dir():
        metadata_path = path / "metadata.json"
        data_path = path / "data.npz"
        if not metadata_path.is_file() or not data_path.is_file():
            raise OneDDataError(
                f"{path}: a directory source needs metadata.json and data.npz"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        archive_context = np.load(data_path, allow_pickle=False)
    else:
        if not path.is_file():
            raise OneDDataError(f"1-D source {path} does not exist")
        archive_context = np.load(path, allow_pickle=False)
        if "metadata" not in archive_context.files:
            raise OneDDataError(f"{path}: standard 1-D source has no metadata")
        metadata = json.loads(str(archive_context["metadata"]))
    with archive_context as archive:
        if metadata.get("format") != STANDARD_FORMAT:
            raise OneDDataError(
                f"{path}: not a {STANDARD_FORMAT} source (got {metadata.get('format')!r})"
            )
        if int(metadata.get("version", -1)) != STANDARD_VERSION:
            raise OneDDataError(
                f"{path}: standard 1-D version {metadata.get('version')!r} is not "
                f"supported (expected {STANDARD_VERSION})"
            )
        units = metadata.get("units")
        if not isinstance(units, Mapping):
            raise OneDDataError(f"{path}: units must be an explicit mapping")
        length_factor = _unit(units.get("length"), _LENGTH, what="length")
        time_factor = _unit(units.get("time"), _TIME, what="time")
        density_factor = _unit(units.get("density"), _DENSITY, what="density")
        velocity_factor = _unit(units.get("velocity"), _VELOCITY, what="velocity")
        pressure_factor = _unit(units.get("pressure"), _PRESSURE, what="pressure")
        if metadata.get("absolute_pressure") is not True:
            raise OneDDataError(
                f"{path}: absolute_pressure must be explicitly true; gauge/overpressure "
                "values must be converted before import"
            )
        if "coordinate" not in metadata or not str(metadata.get("coordinate")).strip():
            raise OneDDataError(
                f"{path}: coordinate is required (e.g. radial_r); a missing key is "
                "not silently treated as the radial coordinate"
            )
        coordinate = str(metadata["coordinate"]).strip().lower()
        if coordinate not in ("radial_r", "radial", "r"):
            raise OneDDataError(
                f"{path}: coordinate={coordinate!r} is not the radial coordinate; "
                "an x-axis model must be mapped to radius explicitly before import"
            )
        if (
            "velocity_component" not in metadata
            or not str(metadata.get("velocity_component")).strip()
        ):
            raise OneDDataError(
                f"{path}: velocity_component is required (e.g. radial_u); a missing "
                "key is not silently treated as the radial velocity, and a full "
                "speed magnitude is never accepted"
            )
        velocity_component = str(metadata["velocity_component"]).strip().lower()
        if velocity_component not in ("radial_u", "radial", "u_r"):
            raise OneDDataError(
                f"{path}: velocity_component={velocity_component!r} is not the "
                "radial velocity; a full speed magnitude is not accepted"
            )
        position_type = str(metadata.get("position_type", "")).strip().lower()
        if position_type not in ("cell", "cell_center", "node"):
            raise OneDDataError(
                f"{path}: position_type must be cell/cell_center/node, got "
                f"{position_type!r}"
            )
        node_convention = str(metadata.get("node_to_cell_convention", "")).strip()
        if position_type == "node" and node_convention not in (
            "preconverted_cell_representatives",
        ):
            raise OneDDataError(
                f"{path}: node sources must be explicitly preconverted to cell "
                "representatives and boundaries; the only accepted "
                "node_to_cell_convention is 'preconverted_cell_representatives'. "
                "Recording an arbitrary convention string is not a conversion"
            )
        case_id = str(metadata.get("case_id", "")).strip()
        if not case_id:
            raise OneDDataError(f"{path}: metadata.case_id is empty")
        charge_radius = float(metadata.get("charge_radius_m", float("nan")))
        if not (math.isfinite(charge_radius) and charge_radius > 0.0):
            raise OneDDataError(f"{path}: charge_radius_m must be finite and positive")
        frames_meta = _frame_arrays(archive, metadata)
        times = np.asarray(metadata.get("frame_times", []), dtype=np.float64)
        if times.size != len(frames_meta):
            raise OneDDataError(
                f"{path}: frame_times has {times.size} entries but there are "
                f"{len(frames_meta)} frames"
            )
    radial_frames: list[RadialFrame] = []
    any_alpha_air = True
    for index, (arrays, time) in enumerate(zip(frames_meta, times)):
        for required in ("radius", "r_left", "r_right", "alpha_product",
                         "density", "velocity", "pressure"):
            if required not in arrays:
                raise OneDDataError(
                    f"{path}: frame {index} has no {required!r} array"
                )
        radius = np.asarray(arrays["radius"], dtype=np.float64) * length_factor
        r_left = np.asarray(arrays["r_left"], dtype=np.float64) * length_factor
        r_right = np.asarray(arrays["r_right"], dtype=np.float64) * length_factor
        alpha_product = np.asarray(arrays["alpha_product"], dtype=np.float64)
        alpha_air = (
            np.asarray(arrays["alpha_air"], dtype=np.float64)
            if "alpha_air" in arrays else None
        )
        if alpha_air is None:
            any_alpha_air = False
        material = (
            np.asarray(arrays["material"]).astype(np.int64)
            if "material" in arrays else None
        )
        density = np.asarray(arrays["density"], dtype=np.float64) * density_factor
        velocity = np.asarray(arrays["velocity"], dtype=np.float64) * velocity_factor
        pressure = np.asarray(arrays["pressure"], dtype=np.float64) * pressure_factor
        cell_id = (
            np.asarray(arrays["cell_id"]).astype(np.int64)
            if "cell_id" in arrays
            else np.arange(radius.size, dtype=np.int64)
        )

        def mask(name: str, values: np.ndarray, label: str) -> np.ndarray:
            if name in arrays:
                provided = np.asarray(arrays[name], dtype=bool)
            else:
                provided = np.isfinite(values)
            if provided.shape != values.shape:
                raise OneDDataError(
                    f"{path}: frame {index} {name} does not match the radial length"
                )
            bad = provided & ~np.isfinite(values)
            if np.any(bad):
                raise OneDDataError(
                    f"{path}: frame {index} marks {int(np.count_nonzero(bad))} "
                    f"non-finite {label} value(s) as valid"
                )
            return provided

        valid_density = mask("valid_density", density, "density")
        valid_velocity = mask("valid_velocity", velocity, "velocity")
        valid_pressure = mask("valid_pressure", pressure, "pressure")
        bad_alpha = ~np.isfinite(alpha_product) | (alpha_product < -1.0e-9) | (
            alpha_product > 1.0 + 1.0e-9
        )
        if np.any(bad_alpha):
            raise OneDDataError(
                f"{path}: frame {index} has product volume fractions outside [0,1]"
            )
        if alpha_air is not None and np.any(
            ~np.isfinite(alpha_air) | (alpha_air < -1.0e-9) | (alpha_air > 1.0 + 1.0e-9)
        ):
            raise OneDDataError(
                f"{path}: frame {index} has air volume fractions outside [0,1]"
            )
        frame = RadialFrame(
            case_id=case_id, charge_radius_m=charge_radius, frame_index=index,
            time_raw_s=float(time) * time_factor, cell_id=cell_id,
            r_left=r_left, r_right=r_right, radius=radius,
            alpha_product=alpha_product, alpha_air=alpha_air,
            material=material, density=density, velocity=velocity, pressure=pressure,
            valid_density=valid_density, valid_velocity=valid_velocity,
            valid_pressure=valid_pressure, position_type=position_type,
            units=dict(units),
            source={
                "kind": "standard_1d_exchange",
                "reader_version": READER_VERSION,
                "path": str(path),
                "raw_source": metadata.get("source"),
            },
        )
        frame.require_monotone()
        radial_frames.append(frame)
    if len(radial_frames) < 2:
        raise OneDDataError(
            f"{path}: at least two frames are required to build a shock trajectory"
        )
    raw_times = np.asarray([frame.time_raw_s for frame in radial_frames])
    if np.any(np.diff(raw_times) <= 0.0):
        raise OneDDataError(
            f"{path}: raw frame times must be strictly increasing"
        )
    raw_source = metadata.get("source") or {}
    return RadialCase(
        case_id=case_id, charge_radius_m=charge_radius, frames=radial_frames,
        units=dict(units), position_type=position_type,
        source={
            "format": STANDARD_FORMAT,
            "version": STANDARD_VERSION,
            "reader_version": READER_VERSION,
            "alpha_air_available": any_alpha_air,
            "coordinate": coordinate,
            "velocity_component": velocity_component,
            "node_to_cell_convention": (
                node_convention if position_type == "node" else None
            ),
            "synthetic": bool(
                isinstance(raw_source, Mapping) and raw_source.get("synthetic", False)
            ),
            "metadata": raw_source,
        },
    )


def write_standard_1d(
    path: Path | str,
    *,
    case_id: str,
    charge_radius_m: float,
    frames: Sequence[Mapping[str, np.ndarray]],
    frame_times: Sequence[float],
    position_type: str = "cell_center",
    units: Mapping[str, str] | None = None,
    source: Mapping[str, Any] | None = None,
    node_to_cell_convention: str | None = None,
) -> Path:
    """Write the intermediate exchange format (used by tests and fixtures).

    The caller supplies SI arrays; ``units`` defaults to SI and every declared
    unit is still read back and checked by :func:`read_standard_1d`.
    """
    units = dict(units or {
        "length": "m", "time": "s", "density": "kg/m^3",
        "velocity": "m/s", "pressure": "Pa",
    })
    times = np.asarray(frame_times, dtype=np.float64)
    if times.size != len(frames):
        raise OneDDataError("frame_times and frames have different lengths")
    arrays: dict[str, list[np.ndarray]] = {}
    offsets = [0]
    for frame in frames:
        for key, value in frame.items():
            arrays.setdefault(key, []).append(np.asarray(value))
        offsets.append(offsets[-1] + int(np.asarray(frame["radius"]).size))
    payload = {key: np.concatenate(values) for key, values in arrays.items()}
    metadata = {
        "format": STANDARD_FORMAT,
        "version": STANDARD_VERSION,
        "case_id": str(case_id),
        "charge_radius_m": float(charge_radius_m),
        "units": units,
        "position_type": str(position_type),
        "absolute_pressure": True,
        "coordinate": "radial_r",
        "velocity_component": "radial_u",
        "node_to_cell_convention": node_to_cell_convention,
        "frame_times": times.tolist(),
        "cells_per_frame": [int(np.asarray(frame["radius"]).size) for frame in frames],
        "source": dict(source or {"kind": "synthetic_or_test_fixture"}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, metadata=np.asarray(json.dumps(metadata)), frame_offsets=np.asarray(offsets),
        **payload,
    )
    return path


def contact_interval_from_fractions(
    frame: RadialFrame,
    *,
    pure_product_threshold: float = 0.95,
    pure_air_threshold: float = 0.95,
    geometry_tolerance_m: float = GEOMETRY_TOLERANCE_M,
) -> tuple[float, float, bool, str]:
    """Locate the contact mixing segment from the volume fractions.

    Only the mixing segment connected to the **outer edge of the central
    product region** is considered.  ``low`` is the actual cell boundary
    between the last pure-product cell and the first mixed cell; ``up`` is the
    actual boundary between the last mixed cell and the first pure-air cell.
    A direct pure-product/pure-air transition uses the shared cell boundary.
    No midpoint is returned or stored: the training object never sees one.
    """
    alpha = np.asarray(frame.alpha_product, dtype=np.float64)
    if not (0.0 < float(pure_air_threshold) <= 1.0) or not (
        0.0 < float(pure_product_threshold) <= 1.0
    ):
        raise OneDDataError("volume-fraction thresholds must lie in (0, 1]")
    product_threshold = float(pure_product_threshold)
    air_product_cut = 1.0 - float(pure_air_threshold)
    if air_product_cut >= product_threshold:
        raise OneDDataError(
            "pure_air_threshold and pure_product_threshold leave no mixing band"
        )
    if not np.all(np.isfinite(alpha)):
        return float("nan"), float("nan"), False, (
            "product volume fraction has non-finite entries; the mixing band is "
            "not identifiable"
        )
    if np.any(alpha < -1.0e-9) or np.any(alpha > 1.0 + 1.0e-9):
        return float("nan"), float("nan"), False, (
            "product volume fraction lies outside [0,1]; the material convention "
            "is invalid"
        )
    pure_product = alpha >= product_threshold
    pure_air = alpha <= air_product_cut
    if not bool(pure_product[0]):
        return float("nan"), float("nan"), False, (
            "the innermost cell is not pure product; the central product region "
            "cannot be established"
        )
    index = 0
    while index < alpha.size and bool(pure_product[index]):
        index += 1
    last_product = index - 1
    mixed_start = index
    while index < alpha.size and not bool(pure_product[index]) and not bool(
        pure_air[index]
    ):
        index += 1
    mixed_stop = index  # exclusive
    # Every step across the central product / mixing / first-air transition must
    # be a real shared cell boundary, not merely a consecutive array index.
    for left in range(0, mixed_stop):
        right = left + 1
        if right >= alpha.size:
            break
        if not cells_geometrically_adjacent(
            frame, left, right, tolerance_m=geometry_tolerance_m
        ):
            return float("nan"), float("nan"), False, (
                "a spatial gap or overlap interrupts the central product / "
                "contact-mixing / pure-air path"
            )
    if mixed_stop > mixed_start:
        if mixed_stop >= alpha.size or not bool(pure_air[mixed_stop]):
            return float("nan"), float("nan"), False, (
                "the contact mixing segment is not bracketed by a pure-air cell "
                "on its outer side"
            )
        low = float(frame.r_left[mixed_start])
        up = float(frame.r_right[mixed_stop - 1])
        if not (low > 0.0 and up >= low):
            return float("nan"), float("nan"), False, (
                "the mixing-segment boundaries are not positive and ordered"
            )
        return low, up, True, (
            "mixing segment connected to the central product region; interval is "
            "bounded by the actual cell boundaries"
        )
    # No mixing cell: a direct pure-product/pure-air interface.
    if mixed_start < alpha.size and bool(pure_air[mixed_start]):
        boundary = float(frame.r_left[mixed_start])
        if boundary <= 0.0:
            return float("nan"), float("nan"), False, (
                "the direct material-transition boundary is non-positive"
            )
        return boundary, boundary, True, (
            "direct pure-product/pure-air transition; both interval bounds are "
            "the shared actual cell boundary"
        )
    return float("nan"), float("nan"), False, (
        "no pure-air cell is adjacent to the central product region"
    )


@dataclass(frozen=True)
class ShockDetectionConfig:
    relative_prominence: float = 8.0
    absolute_prominence_pa_per_m: float = 0.0
    transition_cells: int = 1
    min_cells_from_contact: int = 1
    min_pressure_jump_fraction: float = 1.0e-3
    require_ambient_outside: bool = False
    ambient_pressure_tolerance_fraction: float = 0.05
    #: Absolute tolerance for a shared cell boundary; a larger positive gap is a
    #: real spatial break and a larger negative gap is an overlap.
    geometry_tolerance_m: float = GEOMETRY_TOLERANCE_M
    #: Position difference below this is a repeated/plateau position, not an
    #: inward motion.  It is not a speed clamp: the speed remains unresolved.
    position_tolerance_m: float = 1.0e-9
    smoothing: str | None = None

    def __post_init__(self) -> None:
        if self.relative_prominence <= 0.0:
            raise OneDDataError("relative_prominence must be positive")
        if self.absolute_prominence_pa_per_m < 0.0:
            raise OneDDataError("absolute_prominence_pa_per_m must be non-negative")
        if int(self.transition_cells) < 1:
            raise OneDDataError("transition_cells must be at least one")
        if int(self.min_cells_from_contact) < 0:
            raise OneDDataError("min_cells_from_contact must be non-negative")
        if self.geometry_tolerance_m < 0.0:
            raise OneDDataError("geometry_tolerance_m must be non-negative")
        if self.position_tolerance_m < 0.0:
            raise OneDDataError("position_tolerance_m must be non-negative")
        if self.smoothing not in (None, "none"):
            raise OneDDataError(
                "no smoothing implementation is enabled; recording a smoothing "
                "name without applying it is forbidden"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ShockDetectionConfig":
        payload = dict(payload or {})
        return cls(
            relative_prominence=float(payload.get("relative_prominence", 8.0)),
            absolute_prominence_pa_per_m=float(
                payload.get("absolute_prominence_pa_per_m", 0.0)
            ),
            transition_cells=int(payload.get("transition_cells", 1)),
            min_cells_from_contact=int(payload.get("min_cells_from_contact", 1)),
            min_pressure_jump_fraction=float(
                payload.get("min_pressure_jump_fraction", 1.0e-3)
            ),
            require_ambient_outside=bool(payload.get("require_ambient_outside", False)),
            ambient_pressure_tolerance_fraction=float(
                payload.get("ambient_pressure_tolerance_fraction", 0.05)
            ),
            geometry_tolerance_m=float(
                payload.get("geometry_tolerance_m", GEOMETRY_TOLERANCE_M)
            ),
            position_tolerance_m=float(payload.get("position_tolerance_m", 1.0e-9)),
            smoothing=payload.get("smoothing"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_prominence": self.relative_prominence,
            "absolute_prominence_pa_per_m": self.absolute_prominence_pa_per_m,
            "transition_cells": self.transition_cells,
            "min_cells_from_contact": self.min_cells_from_contact,
            "min_pressure_jump_fraction": self.min_pressure_jump_fraction,
            "require_ambient_outside": self.require_ambient_outside,
            "ambient_pressure_tolerance_fraction": self.ambient_pressure_tolerance_fraction,
            "geometry_tolerance_m": self.geometry_tolerance_m,
            "position_tolerance_m": self.position_tolerance_m,
            "smoothing": self.smoothing,
            "gradient_definition": "g(r) = -dp/dr on valid contiguous air cells",
            "position_definition": "radius of the -dp/dr peak",
            "extrapolation": False,
        }


@dataclass
class ShockDetection:
    radius: float
    valid: bool
    status: str
    reasons: tuple[str, ...]
    candidate_radius: float
    peak_gradient: float
    prominence: float
    states: dict[str, Any] = field(default_factory=dict)
    rh: dict[str, Any] = field(default_factory=dict)
    run_bounds: tuple[int, int] | None = None
    tracked: bool = False
    outward_speed: float | None = None
    #: ``outward`` / ``plateau`` / ``inward`` / ``unresolved``.  A plateau keeps
    #: the position valid but does not provide a usable shock speed.
    speed_status: str = "unresolved"

    def as_dict(self) -> dict[str, Any]:
        return {
            "radius_m": self.radius if math.isfinite(self.radius) else None,
            "valid": self.valid,
            "status": self.status,
            "reasons": list(self.reasons),
            "candidate_radius_m": (
                self.candidate_radius if math.isfinite(self.candidate_radius) else None
            ),
            "peak_gradient_pa_per_m": self.peak_gradient,
            "prominence_pa_per_m": self.prominence,
            "states": self.states,
            "rh_diagnostic": self.rh,
            "run_bounds": None if self.run_bounds is None else list(self.run_bounds),
            "tracked": self.tracked,
            "outward_speed_m_s": self.outward_speed,
            "speed_status": self.speed_status,
        }


def _rh_diagnostic(
    rho_in: float, u_in: float, p_in: float,
    rho_out: float, u_out: float, p_out: float,
) -> dict[str, Any]:
    """RH consistency from measured states only (diagnostic, not a label)."""
    if not all(math.isfinite(v) for v in (rho_in, u_in, p_in, rho_out, u_out, p_out)):
        return {"status": "insufficient", "reason": "non-finite side state"}
    delta = rho_in - rho_out
    scale = max(abs(rho_in), abs(rho_out), 1.0e-12)
    if abs(delta) <= 1.0e-6 * scale:
        return {
            "status": "insufficient",
            "reason": "density jump is inside the state precision; D_s is ill-conditioned",
        }
    ds = (rho_in * u_in - rho_out * u_out) / delta
    if not math.isfinite(ds) or ds <= 0.0:
        return {
            "status": "inconsistent",
            "reason": f"mass-derived shock speed {ds!r} is not positive",
            "Ds_from_mass_m_s": ds,
        }
    p_ref = max(abs(p_in), abs(p_out), 1.0)
    momentum = (
        p_in + rho_in * (u_in - ds) ** 2 - p_out - rho_out * (u_out - ds) ** 2
    ) / p_ref
    g = 1.4
    energy = (
        (u_in - ds) ** 2 / 2.0 + p_in / rho_in / (g - 1.0)
        - (u_out - ds) ** 2 / 2.0 - p_out / rho_out / (g - 1.0)
    ) / max(abs((u_in - ds) ** 2), 1.0)
    return {
        "status": "computed_by_mass_from_measured_states",
        "Ds_from_mass_m_s": float(ds),
        "momentum_residual_normalised": float(momentum),
        "energy_residual_normalised": float(energy),
        "note": (
            "mass conservation was used to derive D_s, so it is not independent "
            "evidence; the side states are measured, not synthesized from RH"
        ),
    }


def detect_frame_shock(
    frame: RadialFrame,
    *,
    contact_up: float,
    ambient_pressure: float,
    pure_air_threshold: float,
    config: ShockDetectionConfig,
) -> ShockDetection:
    """Find the outer transmitted shock from ``-dp/dr`` only.

    Gradient evaluation is confined to each contiguous run of air cells with a
    valid pressure.  A run is never bridged across a missing value, a material
    mixing band or a disjoint radial interval.  No smoothing is applied; the
    configuration records ``smoothing=None``.
    """
    air = frame.air_mask(pure_air_threshold)
    pressure_ok = np.asarray(frame.valid_pressure, dtype=bool) & np.isfinite(
        frame.pressure
    )
    candidates = np.flatnonzero(air & pressure_ok & (frame.radius > contact_up))
    if candidates.size < 3:
        return ShockDetection(
            float("nan"), False, "not_detected",
            ("fewer than three valid air pressure cells outside the contact band",),
            float("nan"), float("nan"), float("nan"),
        )
    # Split at both array gaps and real spatial gaps.  Consecutive indices alone
    # do not mean the cells touch; the mesh may contain a void.
    runs = _geometric_runs(
        frame, candidates, tolerance_m=float(config.geometry_tolerance_m)
    )
    best: tuple[
        float, float, float, int, np.ndarray, np.ndarray, int, int
    ] | None = None
    rejected: list[str] = []
    for run in runs:
        if run.size < 3:
            continue
        r = frame.radius[run]
        p = frame.pressure[run]
        g = -np.gradient(p, r)
        noise = float(np.median(np.abs(g))) if g.size else 0.0
        prominence = max(
            float(config.relative_prominence) * noise,
            float(config.absolute_prominence_pa_per_m),
        )
        peaks, properties = find_peaks(g, prominence=prominence)
        k = max(1, int(config.transition_cells))
        for local in peaks:
            if local < int(config.min_cells_from_contact):
                rejected.append(
                    f"peak at r={r[local]:.6g} m is too close to the contact band"
                )
                continue
            inner_index = max(0, local - k)
            outer_index = min(run.size - 1, local + k)
            if outer_index <= inner_index:
                rejected.append("peak has no resolvable inner/outer pair")
                continue
            p_in = float(p[inner_index])
            p_out = float(p[outer_index])
            if not (p_in > p_out * (1.0 + float(config.min_pressure_jump_fraction))):
                rejected.append(
                    f"peak at r={r[local]:.6g} m does not separate a compressed "
                    "inner state from a lower outer state"
                )
                continue
            if config.require_ambient_outside:
                tolerance = float(config.ambient_pressure_tolerance_fraction) * abs(
                    float(ambient_pressure)
                )
                if abs(p_out - float(ambient_pressure)) > tolerance:
                    rejected.append(
                        f"outer state p={p_out:.6g} Pa is not within "
                        f"{tolerance:.6g} Pa of the ambient pressure"
                    )
                    continue
            candidate_radius = float(r[local])
            if best is None or candidate_radius > best[0]:
                # Keep this run's prominence with the selected candidate; using
                # the last loop iteration's value would misreport it.
                best = (
                    candidate_radius, float(g[local]), float(prominence),
                    int(local), run, g, inner_index, outer_index,
                )
    if best is None:
        reason = (
            "no significant positive -dp/dr peak satisfies the compression and "
            "distance checks"
        )
        if rejected:
            reason = reason + "; " + "; ".join(rejected[:3])
        return ShockDetection(
            float("nan"), False, "not_detected", (reason,), float("nan"),
            float("nan"), float("nan"),
        )
    (
        candidate_radius, peak_gradient, prominence, local, run, g,
        inner_index, outer_index,
    ) = best
    states: dict[str, Any] = {}
    rh: dict[str, Any] = {"status": "skipped"}
    density_ok = frame.valid_density[run[inner_index]] and frame.valid_density[run[outer_index]]
    velocity_ok = frame.valid_velocity[run[inner_index]] and frame.valid_velocity[run[outer_index]]
    states = {
        "inner_radius_m": float(frame.radius[run[inner_index]]),
        "outer_radius_m": float(frame.radius[run[outer_index]]),
        "p_inner_pa": float(frame.pressure[run[inner_index]]),
        "p_outer_pa": float(frame.pressure[run[outer_index]]),
        "rho_inner_available": bool(density_ok),
        "velocity_available": bool(velocity_ok),
    }
    if density_ok and velocity_ok:
        states.update({
            "rho_inner_kg_m3": float(frame.density[run[inner_index]]),
            "u_inner_m_s": float(frame.velocity[run[inner_index]]),
            "rho_outer_kg_m3": float(frame.density[run[outer_index]]),
            "u_outer_m_s": float(frame.velocity[run[outer_index]]),
        })
        rh = _rh_diagnostic(
            float(frame.density[run[inner_index]]),
            float(frame.velocity[run[inner_index]]),
            float(frame.pressure[run[inner_index]]),
            float(frame.density[run[outer_index]]),
            float(frame.velocity[run[outer_index]]),
            float(frame.pressure[run[outer_index]]),
        )
    return ShockDetection(
        radius=candidate_radius, valid=True, status="detected", reasons=(),
        candidate_radius=candidate_radius, peak_gradient=peak_gradient,
        prominence=float(prominence), states=states, rh=rh,
        run_bounds=(int(run[0]), int(run[-1])),
    )


def detect_shock_trajectory(
    frames: Sequence[RadialFrame],
    *,
    contact_low: np.ndarray,
    contact_up: np.ndarray,
    contact_valid: np.ndarray,
    ambient_pressure: float,
    pure_air_threshold: float,
    config: ShockDetectionConfig,
) -> tuple[np.ndarray, np.ndarray, list[ShockDetection]]:
    """Detect per-frame shocks and require an outward-moving front."""
    n = len(frames)
    radii = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    detections: list[ShockDetection] = []
    for index, frame in enumerate(frames):
        if not bool(contact_valid[index]):
            detections.append(ShockDetection(
                float("nan"), False, "not_detected",
                ("contact interval is not locatable, so the air side has no inner bound",),
                float("nan"), float("nan"), float("nan"),
            ))
            continue
        detection = detect_frame_shock(
            frame, contact_up=float(contact_up[index]),
            ambient_pressure=float(ambient_pressure),
            pure_air_threshold=float(pure_air_threshold), config=config,
        )
        if detection.valid and detection.radius <= float(contact_up[index]):
            detection = ShockDetection(
                float("nan"), False, "not_detected",
                ("shock candidate is not outside the contact interval",),
                detection.candidate_radius, detection.peak_gradient,
                detection.prominence, detection.states, detection.rh,
            )
        detections.append(detection)
        if detection.valid:
            radii[index] = detection.radius
            valid[index] = True
    # Tracking pass.  A repeated discrete peak position is a plateau, not an
    # inward motion: the position stays valid and the speed is unresolved.  A
    # strictly negative displacement beyond tolerance is a true retrograde
    # detection and is rejected.  No speed is invented or clamped.
    for index in range(1, n):
        if not valid[index] or not valid[index - 1]:
            continue
        dt = float(frames[index].time_raw_s - frames[index - 1].time_raw_s)
        displacement = float(radii[index] - radii[index - 1])
        tolerance = float(config.position_tolerance_m) + 1.0e-9 * max(
            abs(float(radii[index])), abs(float(radii[index - 1])), 1.0e-30
        )
        if abs(displacement) <= tolerance:
            detections[index].tracked = True
            detections[index].outward_speed = 0.0
            detections[index].speed_status = "plateau"
        elif displacement < 0.0:
            detections[index] = ShockDetection(
                float("nan"), False, "not_detected",
                (f"the candidate moved inward by {displacement!r} m relative to "
                 "the previous detected frame; an outward shock is not confirmed",),
                detections[index].candidate_radius, detections[index].peak_gradient,
                detections[index].prominence, detections[index].states,
                detections[index].rh, detections[index].run_bounds,
                tracked=False, outward_speed=displacement / dt,
                speed_status="inward",
            )
            radii[index] = np.nan
            valid[index] = False
        else:
            detections[index].tracked = True
            detections[index].outward_speed = displacement / dt
            detections[index].speed_status = "outward"
    return radii, valid, detections


@dataclass(frozen=True)
class EventAlignment:
    event_frame: int
    event_time_raw_s: float
    offset_s: float
    t0_model_s: float
    t_separation_theoretical_s: float
    searched_window_s: tuple[float, float] | None
    preceding_frame_confirmed: bool
    event_in_first_frame: bool
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_frame": int(self.event_frame),
            "event_time_raw_s": float(self.event_time_raw_s),
            "offset_s": float(self.offset_s),
            "t0_model_s": float(self.t0_model_s),
            "t_separation_theoretical_s": float(self.t_separation_theoretical_s),
            "searched_window_s": (
                None if self.searched_window_s is None
                else [float(self.searched_window_s[0]), float(self.searched_window_s[1])]
            ),
            "preceding_frame_confirmed": bool(self.preceding_frame_confirmed),
            "event_in_first_frame": bool(self.event_in_first_frame),
            "reasons": list(self.reasons),
            "time_convention": "t_model = t0 + (t_raw - t_event_raw)",
            "note": (
                "the data event is aligned to the model t0 by this offset; it is "
                "not claimed to be the same absolute physical time as t0"
            ),
        }


def pure_air_run_between(
    frame: RadialFrame, *, low: float, high: float, pure_air_threshold: float,
    minimum: int, geometry_tolerance_m: float = GEOMETRY_TOLERANCE_M,
) -> tuple[bool, int]:
    """Largest geometrically continuous, valid pure-air run in ``(low, high)``.

    The run must be contiguous in the array *and* share real cell boundaries.
    Counting disconnected valid cells as one interval would treat a mesh void as
    pure air.
    """
    air = frame.air_mask(pure_air_threshold)
    usable = (
        air
        & (frame.radius > float(low))
        & (frame.radius < float(high))
        & frame.valid_pressure
        & np.isfinite(frame.pressure)
    )
    indices = np.flatnonzero(usable)
    runs = _geometric_runs(
        frame, indices, tolerance_m=float(geometry_tolerance_m)
    )
    largest = max((int(run.size) for run in runs), default=0)
    return largest >= int(minimum), largest


def align_event(
    frames: Sequence[RadialFrame],
    *,
    contact_up: np.ndarray,
    contact_valid: np.ndarray,
    shock_radius: np.ndarray,
    shock_valid: np.ndarray,
    t_separation: float,
    t0: float,
    pure_air_threshold: float,
    window_s: float | None = None,
    min_pure_air_cells: int = 1,
    geometry_tolerance_m: float = GEOMETRY_TOLERANCE_M,
) -> EventAlignment:
    """Pick the first data-identifiable separation frame and time-align it."""
    n = len(frames)
    candidates: list[int] = []
    skipped_window = 0
    for index, frame in enumerate(frames):
        if not bool(contact_valid[index]) or not bool(shock_valid[index]):
            continue
        if window_s is not None and abs(frame.time_raw_s - float(t_separation)) > float(window_s):
            skipped_window += 1
            continue
        ok, _ = pure_air_run_between(
            frame, low=float(contact_up[index]), high=float(shock_radius[index]),
            pure_air_threshold=pure_air_threshold, minimum=min_pure_air_cells,
            geometry_tolerance_m=geometry_tolerance_m,
        )
        if ok:
            candidates.append(index)
    if not candidates:
        raise OneDDataError(
            "no raw frame provides a locatable contact, an identified outer shock "
            "and a valid pure-air interval between them; the separation event "
            "cannot be time-aligned"
            + (
                f" (absolute raw time window around t_sep={t_separation!r} s "
                f"contains no candidate; {skipped_window} frame(s) outside it)"
                if window_s is not None else ""
            )
        )
    event = int(min(candidates))
    frame = frames[event]
    preceding_confirmed = False
    if event > 0 and bool(contact_valid[event - 1]) and bool(shock_valid[event - 1]):
        ok, _ = pure_air_run_between(
            frames[event - 1], low=float(contact_up[event - 1]),
            high=float(shock_radius[event - 1]),
            pure_air_threshold=pure_air_threshold, minimum=min_pure_air_cells,
            geometry_tolerance_m=geometry_tolerance_m,
        )
        preceding_confirmed = not ok
    reasons = []
    if event == 0:
        reasons.append(
            "the event frame is the first available frame; whether separation "
            "occurred earlier cannot be checked from this source"
        )
    elif not preceding_confirmed:
        reasons.append(
            "the immediately preceding frame is missing or already separated; the "
            "event frame is the first usable one, but its onset is not bracketed"
        )
    offset = float(t0) - float(frame.time_raw_s)
    if window_s is None:
        searched = None
    else:
        searched = (float(t_separation) - float(window_s), float(t_separation) + float(window_s))
    return EventAlignment(
        event_frame=event,
        event_time_raw_s=float(frame.time_raw_s),
        offset_s=offset,
        t0_model_s=float(t0),
        t_separation_theoretical_s=float(t_separation),
        searched_window_s=searched,
        preceding_frame_confirmed=bool(preceding_confirmed),
        event_in_first_frame=bool(event == 0),
        reasons=tuple(reasons),
    )


__all__ = [
    "EventAlignment",
    "OneDDataError",
    "RadialCase",
    "RadialFrame",
    "READER_VERSION",
    "STANDARD_FORMAT",
    "STANDARD_VERSION",
    "ShockDetection",
    "ShockDetectionConfig",
    "align_event",
    "contact_interval_from_fractions",
    "detect_frame_shock",
    "detect_shock_trajectory",
    "GEOMETRY_TOLERANCE_M",
    "cell_gap_m",
    "cells_geometrically_adjacent",
    "pure_air_run_between",
    "read_standard_1d",
    "write_standard_1d",
]
