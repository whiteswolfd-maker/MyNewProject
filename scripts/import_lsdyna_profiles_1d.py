"""Import LS-DYNA ASCII ``*DATABASE_PROFILE`` output as a standard 1-D source.

This is a data importer, not a physics model.  It reads the ``.xy`` profile
files written by ``*DATABASE_PROFILE`` for an ALE1D spherical run and writes
the documented ``spherical_blast_pinn.standard_1d_source`` intermediate format
consumed by ``build-cache``.

What it does
------------
* pressure, density and radial velocity from the per-cell profiles;
* per-material volume fractions from two separate volume-fraction profiles
  (``MMG=1`` for TNT/product and ``MMG=2`` for air);
* node -> cell-centre averaging for the nodal velocity profile;
* cell boundaries taken from the *actual* node coordinates in the velocity
  profile, not from an assumed uniform grid;
* SI conversion (the LS-DYNA deck uses mm, tonne, s, MPa, mm/s).

What it does **not** do
-----------------------
* no renormalisation, filling or clamping of volume fractions;
* no derivation of material fractions from density or pressure;
* no conversion of LS-DYNA binary database slot layouts.

Time labels
-----------
``*DATABASE_PROFILE`` writes the output time only in the file name, with three
significant figures (for example ``time3.75E-04.xy``).  The exact simulation
time of that profile write is not present in the file or in ``d3hsp``.  The
importer therefore uses the parsed file-name time and records
``time_source="profile_filename_3sigfig"`` in the source metadata.  Do not
present this as exact solver time; it is the actual output time rounded by the
writer to three significant figures.

Usage (from the repository root)::

    python scripts/import_lsdyna_profiles_1d.py \
        --source-dir G:/R50_R100_reference_v2/R50_full_alpha_profiles \
        --case tnt_50mm --radius-m 0.05 \
        --alpha-product-pattern 'profile4_volume_fraction_x_time*.xy' \
        --alpha-air-pattern 'profile5_volume_fraction_x_time*.xy' \
        --dense-until-s 2.5e-5 --frame-stride 10 \
        --out outputs/contact_air_r50/standard_1d.npz
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spherical_blast_pinn.contact_air.one_d import (  # noqa: E402
    OneDDataError,
    write_standard_1d,
)

_TIME_RE = re.compile(r"_time(?P<time>[0-9]+\.[0-9]+E[+-][0-9]+)\.xy$")

#: LS-DYNA deck units (mm, tonne, s, MPa, mm/s) -> SI.
_MM_TO_M = 1.0e-3
_MM_PER_S_TO_M_PER_S = 1.0e-3
_MPA_TO_PA = 1.0e6
_TONNE_PER_MM3_TO_KG_PER_M3 = 1.0e12


def _read_xy(path: Path) -> np.ndarray:
    """Read one complete Curveplot text block (two numeric columns).

    ``endcurve`` is the only completion marker in the LS-DYNA ASCII profile
    format.  A file that stops before it may still be being written by the
    solver, so it is rejected rather than silently truncated.  The caller also
    checks that every frame has the same cell count and radial coverage.
    """
    rows: list[tuple[float, float]] = []
    started = False
    saw_endcurve = False
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not started:
                if text.startswith("#pts"):
                    started = True
                continue
            if not text:
                continue
            if text.lower() == "endcurve":
                saw_endcurve = True
                break
            parts = text.split()
            if len(parts) != 2:
                raise OneDDataError(
                    f"{path}: line {line_number} has {len(parts)} columns; "
                    "the profile reader expects two numeric columns"
                )
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError as exc:
                raise OneDDataError(
                    f"{path}: line {line_number} is not two numbers: {text!r}"
                ) from exc
    if not started:
        raise OneDDataError(f"{path}: no '#pts' header found")
    if not saw_endcurve:
        raise OneDDataError(
            f"{path}: incomplete profile: no 'endcurve' marker.  The solver may "
            "still be writing this frame; wait for the run to finish and re-import"
        )
    if len(rows) < 2:
        raise OneDDataError(f"{path}: fewer than two data rows")
    return np.asarray(rows, dtype=np.float64)


def _profile_frames(source_dir: Path, pattern: str, label: str) -> dict[str, Path]:
    frames: dict[str, Path] = {}
    for path in sorted(source_dir.glob(pattern)):
        match = _TIME_RE.search(path.name)
        if match is None:
            continue
        key = match.group("time")
        if key in frames:
            raise OneDDataError(
                f"{source_dir}: {label} profile time {key} appears twice "
                f"({frames[key].name} and {path.name})"
            )
        frames[key] = path
    if not frames:
        raise OneDDataError(
            f"{source_dir}: no {label} profiles match {pattern!r}"
        )
    return frames


def _strictly_increasing(values: np.ndarray, *, what: str) -> None:
    if not np.all(np.isfinite(values)):
        index = int(np.flatnonzero(~np.isfinite(values))[0])
        raise OneDDataError(f"{what}[{index}] is not finite")
    if np.any(np.diff(values) <= 0.0):
        index = int(np.flatnonzero(np.diff(values) <= 0.0)[0]) + 1
        raise OneDDataError(
            f"{what} must be strictly increasing; entry {index}="
            f"{values[index]!r} does not exceed the previous entry"
        )


def _select_keys(
    keys: Sequence[str], *, dense_until_s: float, frame_stride: int,
) -> list[str]:
    if frame_stride < 1:
        raise OneDDataError("frame_stride must be >= 1")
    ordered = sorted(keys, key=float)
    dense = [key for key in ordered if float(key) <= float(dense_until_s)]
    sparse = [key for key in ordered if float(key) > float(dense_until_s)]
    selected = list(dense) + list(sparse[:: int(frame_stride)])
    # The sparse slice can leave the final frame out; keep it only when the
    # requested stride lands on it.  Selection is recorded, not guessed.
    return selected


def _frame_arrays(
    paths: Mapping[str, Any], key: str, *, include_alpha_air: bool,
) -> dict[str, np.ndarray]:
    velocity = _read_xy(paths["velocity"][key])
    pressure = _read_xy(paths["pressure"][key])
    density = _read_xy(paths["density"][key])
    alpha_p = _read_xy(paths["alpha_product"][key])
    alpha_a = _read_xy(paths["alpha_air"][key]) if include_alpha_air else None

    if velocity.shape[0] != pressure.shape[0] + 1:
        raise OneDDataError(
            f"frame time {key}: velocity has {velocity.shape[0]} node rows but "
            f"pressure has {pressure.shape[0]} cell rows; expected one more node"
        )
    n_cells = int(pressure.shape[0])
    cell_fields = [("density", density), ("alpha_product", alpha_p)]
    if alpha_a is not None:
        cell_fields.append(("alpha_air", alpha_a))
    for name, array in cell_fields:
        if array.shape[0] != n_cells:
            raise OneDDataError(
                f"frame time {key}: {name} has {array.shape[0]} rows, expected "
                f"{n_cells} (same as pressure)"
            )

    node_r = velocity[:, 0] * _MM_TO_M
    cell_r = pressure[:, 0] * _MM_TO_M
    for name, array in cell_fields:
        if not np.allclose(array[:, 0] * _MM_TO_M, cell_r, rtol=0.0, atol=1.0e-12):
            index = int(np.flatnonzero(np.abs(array[:, 0] * _MM_TO_M - cell_r) > 1.0e-12)[0])
            raise OneDDataError(
                f"frame time {key}: {name} cell {index} has x="
                f"{array[index, 0]!r} mm but pressure has {pressure[index, 0]!r} mm"
            )

    _strictly_increasing(node_r, what=f"frame time {key}: node radius")
    _strictly_increasing(cell_r, what=f"frame time {key}: cell radius")
    if np.any(node_r < 0.0):
        index = int(np.flatnonzero(node_r < 0.0)[0])
        raise OneDDataError(
            f"frame time {key}: node radius[{index}]={node_r[index]!r} m is negative"
        )
    expected_center = 0.5 * (node_r[:-1] + node_r[1:])
    if not np.allclose(cell_r, expected_center, rtol=0.0, atol=1.0e-9):
        index = int(np.flatnonzero(np.abs(cell_r - expected_center) > 1.0e-9)[0])
        raise OneDDataError(
            f"frame time {key}: cell {index} representative radius "
            f"{cell_r[index]!r} m is not the midpoint of its node boundaries "
            f"({expected_center[index]!r} m)"
        )

    r_left = node_r[:-1]
    r_right = node_r[1:]
    if np.any(r_right <= r_left):
        index = int(np.flatnonzero(r_right <= r_left)[0])
        raise OneDDataError(
            f"frame time {key}: cell {index} has non-positive width "
            f"({r_left[index]!r}, {r_right[index]!r}) m"
        )

    cell_velocity = 0.5 * (velocity[:-1, 1] + velocity[1:, 1]) * _MM_PER_S_TO_M_PER_S
    pressure_pa = pressure[:, 1] * _MPA_TO_PA
    density_si = density[:, 1] * _TONNE_PER_MM3_TO_KG_PER_M3

    frame = {
        # No cell_id here: the profile files do not carry LS-DYNA element IDs.
        # The reader generates IDs and records them as generated; inventing
        # 1..N in the importer would falsely claim they came from the deck.
        "r_left": r_left,
        "r_right": r_right,
        "radius": cell_r,
        "alpha_product": alpha_p[:, 1].copy(),
        "density": density_si,
        "velocity": cell_velocity,
        "pressure": pressure_pa,
        "valid_density": np.isfinite(density_si),
        "valid_velocity": np.isfinite(cell_velocity),
        "valid_pressure": np.isfinite(pressure_pa),
    }
    if alpha_a is not None:
        frame["alpha_air"] = alpha_a[:, 1].copy()
    return frame


def _check_fractions(
    alpha_p: np.ndarray, alpha_a: np.ndarray | None, *, key: str,
) -> None:
    if not np.all(np.isfinite(alpha_p)):
        raise OneDDataError(f"frame time {key}: alpha_product is not finite")
    if alpha_a is not None and not np.all(np.isfinite(alpha_a)):
        raise OneDDataError(f"frame time {key}: alpha_air is not finite")
    bad_p = (alpha_p < -1.0e-9) | (alpha_p > 1.0 + 1.0e-9)
    if np.any(bad_p):
        index = int(np.flatnonzero(bad_p)[0])
        raise OneDDataError(
            f"frame time {key}: alpha_product[{index}]={alpha_p[index]!r} is "
            "outside [0,1]"
        )
    if alpha_a is None:
        return
    bad_a = (alpha_a < -1.0e-9) | (alpha_a > 1.0 + 1.0e-9)
    if np.any(bad_a):
        index = int(np.flatnonzero(bad_a)[0])
        raise OneDDataError(
            f"frame time {key}: alpha_air[{index}]={alpha_a[index]!r} is "
            "outside [0,1]"
        )
    total = alpha_p + alpha_a
    bad_sum = np.abs(total - 1.0) > 1.0e-4
    if np.any(bad_sum):
        index = int(np.flatnonzero(bad_sum)[0])
        raise OneDDataError(
            f"frame time {key}: cell {index} has alpha_product="
            f"{alpha_p[index]!r} and alpha_air={alpha_a[index]!r}; their sum "
            f"{total[index]!r} is not 1 within 1e-4.  The importer does not "
            "renormalise or fill material fractions"
        )


def convert(
    *, source_dir: Path, case_id: str, radius_m: float, out: Path,
    velocity_pattern: str, pressure_pattern: str, density_pattern: str,
    alpha_product_pattern: str, alpha_air_pattern: str | None,
    dense_until_s: float, frame_stride: int, allow_moving_mesh: bool = False,
    expected_cells: int | None = None,
    expected_outer_radius_m: float | None = None,
) -> dict[str, Any]:
    source_dir = Path(source_dir)
    include_alpha_air = alpha_air_pattern is not None
    paths = {
        "velocity": _profile_frames(source_dir, velocity_pattern, "velocity"),
        "pressure": _profile_frames(source_dir, pressure_pattern, "pressure"),
        "density": _profile_frames(source_dir, density_pattern, "density"),
        "alpha_product": _profile_frames(source_dir, alpha_product_pattern, "alpha_product"),
    }
    if include_alpha_air:
        paths["alpha_air"] = _profile_frames(
            source_dir, str(alpha_air_pattern), "alpha_air"
        )
    key_sets = {name: set(frames) for name, frames in paths.items()}
    reference = key_sets["pressure"]
    for name, keys in key_sets.items():
        if keys != reference:
            missing = sorted(reference - keys)
            extra = sorted(keys - reference)
            raise OneDDataError(
                f"{source_dir}: {name} profile times do not match pressure; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    selected = _select_keys(
        sorted(reference, key=float),
        dense_until_s=dense_until_s, frame_stride=frame_stride,
    )
    if len(selected) < 2:
        raise OneDDataError("fewer than two frames selected")

    frames: list[dict[str, np.ndarray]] = []
    times: list[float] = []
    cell_counts: set[int] = set()
    node_counts: set[int] = set()
    inner_radii: list[float] = []
    outer_radii: list[float] = []
    alpha_sum_deviation = 0.0
    alpha_min = math.inf
    alpha_max = -math.inf
    for key in selected:
        frame = _frame_arrays(paths, key, include_alpha_air=include_alpha_air)
        _check_fractions(
            frame["alpha_product"], frame.get("alpha_air"), key=key
        )
        cell_counts.add(int(frame["radius"].size))
        node_counts.add(int(frame["r_left"].size) + 1)
        inner_radii.append(float(frame["r_left"][0]))
        outer_radii.append(float(frame["r_right"][-1]))
        if "alpha_air" in frame:
            total = frame["alpha_product"] + frame["alpha_air"]
            alpha_sum_deviation = max(
                alpha_sum_deviation, float(np.max(np.abs(total - 1.0)))
            )
            alpha_min = min(
                alpha_min,
                float(np.min(frame["alpha_product"])),
                float(np.min(frame["alpha_air"])),
            )
            alpha_max = max(
                alpha_max,
                float(np.max(frame["alpha_product"])),
                float(np.max(frame["alpha_air"])),
            )
        else:
            alpha_min = min(alpha_min, float(np.min(frame["alpha_product"])))
            alpha_max = max(alpha_max, float(np.max(frame["alpha_product"])))
        frames.append(frame)
        times.append(float(key))

    if len(cell_counts) != 1:
        raise OneDDataError(
            f"{source_dir}: selected frames have differing cell counts "
            f"{sorted(cell_counts)}; a complete fixed-mesh profile set must have "
            "one count.  A truncated frame looks like this"
        )
    if len(node_counts) != 1:
        raise OneDDataError(
            f"{source_dir}: selected frames have differing node counts "
            f"{sorted(node_counts)}"
        )
    if not allow_moving_mesh:
        inner_span = max(inner_radii) - min(inner_radii)
        outer_span = max(outer_radii) - min(outer_radii)
        if inner_span > 1.0e-9 or outer_span > 1.0e-9:
            raise OneDDataError(
                f"{source_dir}: the radial coverage changes between frames "
                f"(inner span {inner_span!r} m, outer span {outer_span!r} m). "
                "This importer is for the fixed ALE1D mesh; pass "
                "--allow-moving-mesh only if that is intentional"
            )

    if expected_cells is not None:
        actual = int(frames[0]["radius"].size)
        if actual != int(expected_cells):
            raise OneDDataError(
                f"{source_dir}: expected {int(expected_cells)} cells but the "
                f"profile frames contain {actual}; this is a different mesh or a "
                "globally truncated profile set"
            )
    if expected_outer_radius_m is not None:
        actual_outer = float(frames[0]["r_right"][-1])
        if abs(actual_outer - float(expected_outer_radius_m)) > 1.0e-9:
            raise OneDDataError(
                f"{source_dir}: expected the outer radial boundary at "
                f"{float(expected_outer_radius_m)!r} m but the profile reaches "
                f"{actual_outer!r} m; check the deck's domain or the truncation"
            )

    _strictly_increasing(np.asarray(times, dtype=np.float64), what="frame_times")

    raw_keys = sorted(reference, key=float)
    source = {
        "kind": "lsdyna_database_profile_ascii",
        "source_dir": str(source_dir.resolve()),
        "deck_units": {
            "length": "mm", "time": "s", "density": "tonne/mm^3",
            "velocity": "mm/s", "pressure": "MPa",
        },
        "profiles": {
            "velocity": velocity_pattern,
            "pressure": pressure_pattern,
            "density": density_pattern,
            "alpha_product": alpha_product_pattern,
            "alpha_air": alpha_air_pattern,
        },
        "velocity_source": (
            "nodal x-velocity averaged to the two adjacent cell centres"
        ),
        "cell_geometry_source": (
            "actual node coordinates from the velocity profile; cell boundaries "
            "are node[i] and node[i+1]"
        ),
        "alpha_source": (
            "*DATABASE_PROFILE volume fraction: MMG=1 is TNT/product; MMG=2 is "
            "air when an air profile is supplied.  Without an air profile the "
            "reader uses implicit alpha_air = 1 - alpha_product.  No "
            "renormalisation or filling"
        ),
        "alpha_air_available": include_alpha_air,
        "cell_id_note": (
            "profile files contain values in element-set order, not element IDs; "
            "the importer does not invent IDs and the reader records generated IDs"
        ),
        "time_source": "profile_filename_3sigfig",
        "time_note": (
            "the profile writer records the output time only in the file name "
            "with three significant figures; the exact solver time is not "
            "present in the profile file or d3hsp"
        ),
        "raw_frame_count": len(raw_keys),
        "selected_raw_frame_keys": selected,
        "dense_until_s": float(dense_until_s),
        "frame_stride_after_dense": int(frame_stride),
        "cell_count": int(frames[0]["radius"].size),
        "expected_cells": expected_cells,
        "expected_outer_radius_m": expected_outer_radius_m,
        "mesh_consistency": (
            "all selected frames checked for equal cell/node counts and "
            "constant inner/outer radial coverage"
            if not allow_moving_mesh else
            "equal cell/node counts checked; radial coverage intentionally "
            "allowed to move"
        ),
        "alpha_sum_max_abs_deviation": alpha_sum_deviation,
        "alpha_min_seen": alpha_min,
        "alpha_max_seen": alpha_max,
        "no_normalization_or_filling": True,
    }
    write_standard_1d(
        out, case_id=case_id, charge_radius_m=float(radius_m),
        frames=frames, frame_times=times, position_type="cell_center",
        source=source,
    )
    return {
        "out": str(Path(out).resolve()),
        "raw_frames": len(raw_keys),
        "selected_frames": len(selected),
        "cells": int(frames[0]["radius"].size),
        "first_time_s": times[0],
        "last_time_s": times[-1],
        "alpha_sum_max_abs_deviation": alpha_sum_deviation,
        "alpha_min_seen": alpha_min,
        "alpha_max_seen": alpha_max,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--velocity-pattern", default="profile1_x-velocity_x_time*.xy"
    )
    parser.add_argument(
        "--pressure-pattern", default="profile2_pressure_x_time*.xy"
    )
    parser.add_argument(
        "--density-pattern", default="profile3_density_x_time*.xy"
    )
    parser.add_argument(
        "--alpha-product-pattern",
        default="profile4_volume_fraction_x_time*.xy",
        help="volume-fraction profile for ALE group 1 (TNT/product)",
    )
    parser.add_argument(
        "--alpha-air-pattern",
        default=None,
        help=(
            "optional volume-fraction profile for ALE group 2 (air).  Omit it "
            "when the deck only writes MMG=1; the standard source then carries "
            "alpha_product only and the reader uses implicit 1-alpha_product"
        ),
    )
    parser.add_argument(
        "--dense-until-s", type=float, default=2.5e-5,
        help="keep every profile frame up to this raw time (event alignment)",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=10,
        help="after --dense-until-s keep every Nth frame",
    )
    parser.add_argument(
        "--allow-moving-mesh", action="store_true",
        help=(
            "skip the constant inner/outer radial-coverage check.  The current "
            "ALE1D decks have a fixed mesh; use this only for a moving mesh"
        ),
    )
    parser.add_argument(
        "--expected-cells", type=int, default=None,
        help=(
            "optional expected cell count from the deck; catches a profile set "
            "that is globally truncated to the same wrong length"
        ),
    )
    parser.add_argument(
        "--expected-outer-radius-m", type=float, default=None,
        help="optional expected outer radial boundary in metres",
    )
    args = parser.parse_args(argv)
    try:
        summary = convert(
            source_dir=Path(args.source_dir), case_id=args.case,
            radius_m=float(args.radius_m), out=Path(args.out),
            velocity_pattern=args.velocity_pattern,
            pressure_pattern=args.pressure_pattern,
            density_pattern=args.density_pattern,
            alpha_product_pattern=args.alpha_product_pattern,
            alpha_air_pattern=args.alpha_air_pattern,
            dense_until_s=float(args.dense_until_s),
            frame_stride=int(args.frame_stride),
            allow_moving_mesh=bool(args.allow_moving_mesh),
            expected_cells=args.expected_cells,
            expected_outer_radius_m=args.expected_outer_radius_m,
        )
    except OneDDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
