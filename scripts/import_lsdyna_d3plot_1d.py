#!/usr/bin/env python3
"""Import LS-DYNA d3plot ALE1D states as a standard 1-D source.

The old ``*DATABASE_PROFILE`` ASCII path stores the output time only in the
file name with three significant figures.  At fine late-time steps several
profile files can share the same rounded name and overwrite each other.  This
importer reads the d3plot state database instead and uses the full-precision
state time from the d3plot header.

ALE1D d3plot layout used here (LS-DYNA R12, ``*SECTION_ALE1D`` with
``BEAMIP`` in ``*DATABASE_EXTENT_BINARY``):

* first beam "integration point" (layer 0) is the volume-averaged data:
  pressure, density, compression ratio, internal energy/volume, ALE group ID;
* additional layers are per-ALE-group data and are not needed for the
  volume-averaged fields consumed by this project.

The averaged group ID is the volume-fraction-weighted ALE group ID.  With
group 1 = product and group 2 = air, ``alpha_air = gid - 1`` and
``alpha_product = 2 - gid``.  This reproduces the two volume fractions that
the old profile importer read from separate profile files.

Velocity is taken from the nodal x-velocity and averaged to the two nodes of
each ALE1D beam element, matching the old profile importer's convention.
All arrays are converted to SI before calling ``write_standard_1d``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spherical_blast_pinn.contact_air.one_d import (  # noqa: E402
    OneDDataError,
    write_standard_1d,
)

_MM_TO_M = 1.0e-3
_MM_PER_S_TO_M_PER_S = 1.0e-3
_MPA_TO_PA = 1.0e6
_TONNE_PER_MM3_TO_KG_PER_M3 = 1.0e12

_STATE_ARRAYS = [
    "node_displacement",
    "node_velocity",
    "element_beam_axial_stress",
    "element_beam_shear_stress",
    "element_beam_plastic_strain",
    "element_beam_axial_strain",
    "timesteps",
]


def _load_d3plot(path: Path, state_filter=None, state_array_filter=None):
    try:
        from lasso.dyna import D3plot
    except ImportError as exc:  # pragma: no cover - dependency error
        raise OneDDataError(
            "lasso-python is required to read d3plot state databases"
        ) from exc
    return D3plot(
        str(path), state_filter=state_filter,
        state_array_filter=state_array_filter,
        buffered_reading=True,
    )


def _total_states(path: Path) -> int:
    # Geometry-only read gives the total state count without loading state
    # arrays into memory.
    d3plot = _load_d3plot(path, state_filter=set(), state_array_filter=[])
    count = int(getattr(d3plot._state_info, "n_timesteps", 0))
    if count < 1:
        raise OneDDataError(f"{path}: d3plot reports no states")
    return count


def _chunks(total: int, size: int) -> Iterator[Sequence[int]]:
    for start in range(0, total, size):
        yield list(range(start, min(start + size, total)))


def _as_state_array(value, n_states: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise OneDDataError(f"d3plot array {name!r} is scalar")
    if array.shape[0] != n_states:
        # Static geometry arrays may be returned without the state axis.
        if array.ndim >= 1 and n_states == 1:
            array = array[None, ...]
        else:
            raise OneDDataError(
                f"d3plot array {name!r} has leading shape {array.shape[0]} "
                f"but {n_states} states were requested"
            )
    return array


def _frame_from_state(
    *, axial_stress, shear_stress, plastic_strain, axial_strain,
    node_velocity, node_coordinates, node_displacement,
    node_indexes, time_s: float,
) -> dict[str, np.ndarray]:
    node0 = np.asarray(node_indexes[:, 0], dtype=np.int64).copy()
    node1 = np.asarray(node_indexes[:, 1], dtype=np.int64).copy()
    if node0.size == 0:
        raise OneDDataError("d3plot has no beam elements")
    if node0.min() < 0 or node1.max() >= node_coordinates.shape[0]:
        raise OneDDataError("d3plot beam connectivity is out of range")

    # lasso reports the per-state d3plot coordinate block as
    # ``node_displacement``.  In these ALE1D files it already contains the
    # current coordinates (at t=0 it equals node_coordinates), so it must not
    # be added to the initial coordinates.
    if node_displacement is not None:
        coords = np.asarray(node_displacement, dtype=np.float64)
    else:
        coords = np.asarray(node_coordinates, dtype=np.float64)
    node_r = coords[:, 0]
    r_left = node_r[node0] * _MM_TO_M
    r_right = node_r[node1] * _MM_TO_M
    swap = r_right <= r_left
    if np.any(swap):
        # LS-DYNA writes the two beam nodes in radial order, but a reversed
        # pair is still usable after sorting.
        r_left[swap], r_right[swap] = r_right[swap].copy(), r_left[swap].copy()
        node0[swap], node1[swap] = node1[swap].copy(), node0[swap].copy()
    radius = 0.5 * (r_left + r_right)

    velocity = np.asarray(node_velocity, dtype=np.float64)[:, 0]
    cell_velocity = 0.5 * (velocity[node0] + velocity[node1]) * _MM_PER_S_TO_M_PER_S

    axial = np.asarray(axial_stress, dtype=np.float64)
    shear = np.asarray(shear_stress, dtype=np.float64)
    plastic = np.asarray(plastic_strain, dtype=np.float64)
    axial_layer = np.asarray(axial_strain, dtype=np.float64)
    if axial.ndim != 2 or shear.ndim != 3 or plastic.ndim != 2 or axial_layer.ndim != 2:
        raise OneDDataError(
            "unexpected d3plot beam array shapes: "
            f"axial={axial.shape}, shear={shear.shape}, "
            f"plastic={plastic.shape}, axial_strain={axial_layer.shape}"
        )

    density = shear[:, 0, 0] * _TONNE_PER_MM3_TO_KG_PER_M3
    pressure = axial[:, 0] * _MPA_TO_PA
    group_id = axial_layer[:, 0]
    # Volume-fraction-weighted group ID: 1 product, 2 air.
    alpha_air = np.clip(group_id - 1.0, 0.0, 1.0)
    alpha_product = np.clip(2.0 - group_id, 0.0, 1.0)
    # A tiny numerical mismatch is reported by the writer; renormalising would
    # hide a genuine output problem.
    return {
        "r_left": r_left,
        "r_right": r_right,
        "radius": radius,
        "alpha_product": alpha_product,
        "alpha_air": alpha_air,
        "density": density,
        "velocity": cell_velocity,
        "pressure": pressure,
        "valid_density": np.isfinite(density),
        "valid_velocity": np.isfinite(cell_velocity),
        "valid_pressure": np.isfinite(pressure),
    }


def convert(
    *, d3plot_path: Path, case_id: str, radius_m: float, out: Path,
    expected_cells: int | None = 50000,
    expected_outer_radius_m: float | None = None,
    allow_moving_mesh: bool = False,
    chunk_size: int = 50,
    limit_states: int | None = None,
) -> dict:
    d3plot_path = Path(d3plot_path)
    if not d3plot_path.is_file():
        raise OneDDataError(f"d3plot file does not exist: {d3plot_path}")
    total = _total_states(d3plot_path)
    if limit_states is not None:
        total = min(total, int(limit_states))
    if total < 2:
        raise OneDDataError("fewer than two d3plot states are present")

    frames: list[dict[str, np.ndarray]] = []
    times: list[float] = []
    cell_counts: set[int] = set()
    node_counts: set[int] = set()
    inner_radii: list[float] = []
    outer_radii: list[float] = []
    alpha_sum_deviation = 0.0
    alpha_min = math.inf
    alpha_max = -math.inf
    geometry = None

    for state_chunk in _chunks(total, max(1, int(chunk_size))):
        d3plot = _load_d3plot(
            d3plot_path,
            state_filter=set(state_chunk),
            state_array_filter=_STATE_ARRAYS,
        )
        arrays = d3plot.arrays
        n_states = len(state_chunk)
        timesteps = np.asarray(arrays["timesteps"], dtype=np.float64).reshape(-1)
        if timesteps.size != n_states:
            raise OneDDataError(
                f"{d3plot_path}: requested {n_states} states but d3plot "
                f"returned {timesteps.size} state times"
            )
        for name in (
            "node_coordinates", "node_displacement", "node_velocity",
            "element_beam_node_indexes", "element_beam_axial_stress",
            "element_beam_shear_stress", "element_beam_plastic_strain",
            "element_beam_axial_strain",
        ):
            if name not in arrays:
                raise OneDDataError(f"{d3plot_path}: d3plot has no array {name!r}")

        node_coordinates = np.asarray(arrays["node_coordinates"], dtype=np.float64)
        if node_coordinates.ndim == 3:
            node_coordinates = node_coordinates[0]
        node_displacement = _as_state_array(
            arrays["node_displacement"], n_states, "node_displacement"
        )
        node_velocity = _as_state_array(
            arrays["node_velocity"], n_states, "node_velocity"
        )
        node_indexes = np.asarray(
            arrays["element_beam_node_indexes"], dtype=np.int64
        )
        axial = _as_state_array(
            arrays["element_beam_axial_stress"], n_states,
            "element_beam_axial_stress",
        )
        shear = _as_state_array(
            arrays["element_beam_shear_stress"], n_states,
            "element_beam_shear_stress",
        )
        plastic = _as_state_array(
            arrays["element_beam_plastic_strain"], n_states,
            "element_beam_plastic_strain",
        )
        axial_layer = _as_state_array(
            arrays["element_beam_axial_strain"], n_states,
            "element_beam_axial_strain",
        )
        if geometry is None:
            geometry = {
                "node_coordinates": node_coordinates.copy(),
                "node_indexes": node_indexes.copy(),
            }
        else:
            if not np.array_equal(geometry["node_indexes"], node_indexes):
                raise OneDDataError("beam connectivity changes between d3plot files")
            if not np.allclose(
                geometry["node_coordinates"], node_coordinates,
                rtol=0.0, atol=1.0e-12,
            ):
                raise OneDDataError("initial d3plot geometry changes between files")

        for offset, time_s in enumerate(timesteps):
            frame = _frame_from_state(
                axial_stress=axial[offset],
                shear_stress=shear[offset],
                plastic_strain=plastic[offset],
                axial_strain=axial_layer[offset],
                node_velocity=node_velocity[offset],
                node_coordinates=node_coordinates,
                node_displacement=node_displacement[offset],
                node_indexes=node_indexes,
                time_s=float(time_s),
            )
            cell_counts.add(int(frame["radius"].size))
            node_counts.add(int(frame["r_left"].size) + 1)
            inner_radii.append(float(frame["r_left"][0]))
            outer_radii.append(float(frame["r_right"][-1]))
            total_alpha = frame["alpha_product"] + frame["alpha_air"]
            alpha_sum_deviation = max(
                alpha_sum_deviation, float(np.max(np.abs(total_alpha - 1.0)))
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
            frames.append(frame)
            times.append(float(time_s))

    if len(cell_counts) != 1:
        raise OneDDataError(
            f"{d3plot_path}: states have differing cell counts {sorted(cell_counts)}"
        )
    if len(node_counts) != 1:
        raise OneDDataError(
            f"{d3plot_path}: states have differing node counts {sorted(node_counts)}"
        )
    if not allow_moving_mesh:
        inner_span = max(inner_radii) - min(inner_radii)
        outer_span = max(outer_radii) - min(outer_radii)
        if inner_span > 1.0e-9 or outer_span > 1.0e-9:
            raise OneDDataError(
                f"{d3plot_path}: radial coverage changes between states "
                f"(inner span {inner_span!r} m, outer span {outer_span!r} m); "
                "pass --allow-moving-mesh only if that is intentional"
            )
    if expected_cells is not None and int(frames[0]["radius"].size) != int(expected_cells):
        raise OneDDataError(
            f"{d3plot_path}: expected {int(expected_cells)} cells but d3plot "
            f"contains {int(frames[0]['radius'].size)}"
        )
    if expected_outer_radius_m is not None:
        actual = float(frames[0]["r_right"][-1])
        if abs(actual - float(expected_outer_radius_m)) > 1.0e-9:
            raise OneDDataError(
                f"{d3plot_path}: expected outer radius "
                f"{float(expected_outer_radius_m)!r} m but found {actual!r} m"
            )
    times_array = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(times_array) <= 0.0):
        index = int(np.flatnonzero(np.diff(times_array) <= 0.0)[0])
        raise OneDDataError(
            f"{d3plot_path}: state times are not strictly increasing at "
            f"index {index}: {times_array[index]!r} -> {times_array[index + 1]!r}"
        )

    source = {
        "kind": "lsdyna_d3plot_ale1d_state",
        "source_dir": str(d3plot_path.parent.resolve()),
        "d3plot_file": str(d3plot_path.resolve()),
        "deck_units": {
            "length": "mm", "time": "s", "density": "tonne/mm^3",
            "velocity": "mm/s", "pressure": "MPa",
        },
        "velocity_source": (
            "nodal x-velocity averaged to the two nodes of each ALE1D beam element"
        ),
        "cell_geometry_source": (
            "per-state node coordinates from the d3plot state block (lasso "
            "calls this array node_displacement); cell boundaries are each "
            "beam element's two nodes"
        ),
        "alpha_source": (
            "volume-fraction-weighted ALE group ID from the first d3plot beam "
            "layer: alpha_air = gid - 1, alpha_product = 2 - gid, clipped to "
            "[0, 1]. Group 1 = product, group 2 = air."
        ),
        "field_source": (
            "first ALE1D beam integration point: axial stress = volume-averaged "
            "pressure, rs-shear = volume-averaged density, tr-shear = compression "
            "ratio, plastic strain = internal energy/volume. LS-PrePost labels "
            "do not match these ALE meanings."
        ),
        "time_source": "d3plot_state_header",
        "time_note": (
            "the state time is read from the d3plot state header, not from a "
            "rounded *_time*.xy file name"
        ),
        "frame_selection_policy": "all_states",
        "frame_selection_note": (
            "every state in the d3plot family is imported; no stride or frame cap"
        ),
        "raw_frame_count": total,
        "imported_frame_count": len(frames),
        "dropped_frame_count": int(total - len(frames)),
        "selected_raw_frame_keys": [f"state_{index:06d}" for index in range(total)],
        "raw_frame_keys": [f"state_{index:06d}" for index in range(total)],
        "first_raw_time_s": float(times[0]),
        "last_raw_time_s": float(times[-1]),
        "cell_count": int(frames[0]["radius"].size),
        "expected_cells": expected_cells,
        "expected_outer_radius_m": expected_outer_radius_m,
        "mesh_consistency": (
            "all states checked for equal cell/node counts and constant radial "
            "coverage"
            if not allow_moving_mesh else
            "equal cell/node counts checked; moving radial coverage allowed"
        ),
        "alpha_sum_max_abs_deviation": alpha_sum_deviation,
        "alpha_min_seen": alpha_min,
        "alpha_max_seen": alpha_max,
        "no_normalization_or_filling": True,
        "d3plot_reader": "lasso-python D3plot",
        "d3plot_state_arrays": _STATE_ARRAYS,
    }
    write_standard_1d(
        out, case_id=case_id, charge_radius_m=float(radius_m),
        frames=frames, frame_times=times, position_type="cell_center",
        source=source,
    )
    return {
        "out": str(Path(out).resolve()),
        "d3plot": str(d3plot_path.resolve()),
        "frame_selection_policy": "all_states",
        "raw_states": total,
        "imported_states": len(frames),
        "cells": int(frames[0]["radius"].size),
        "first_time_s": times[0],
        "last_time_s": times[-1],
        "alpha_sum_max_abs_deviation": alpha_sum_deviation,
        "alpha_min_seen": alpha_min,
        "alpha_max_seen": alpha_max,
        "inner_radius_span_m": max(inner_radii) - min(inner_radii),
        "outer_radius_span_m": max(outer_radii) - min(outer_radii),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--d3plot", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-cells", type=int, default=50000)
    parser.add_argument("--expected-outer-radius-m", type=float, default=None)
    parser.add_argument("--allow-moving-mesh", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--limit-states", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        summary = convert(
            d3plot_path=Path(args.d3plot), case_id=args.case,
            radius_m=args.radius_m, out=Path(args.out),
            expected_cells=args.expected_cells,
            expected_outer_radius_m=args.expected_outer_radius_m,
            allow_moving_mesh=args.allow_moving_mesh,
            chunk_size=args.chunk_size,
            limit_states=args.limit_states,
        )
    except OneDDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
