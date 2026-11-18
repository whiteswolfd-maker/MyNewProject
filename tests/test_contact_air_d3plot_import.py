"""Unit tests for the d3plot ALE1D state importer's field mapping."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location(
    "import_lsdyna_d3plot_1d",
    ROOT / "scripts" / "import_lsdyna_d3plot_1d.py",
)
assert _SPEC is not None and _SPEC.loader is not None
imp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(imp)


def _state_arrays(group_ids, densities, pressures, node_x):
    node_x = np.asarray(node_x, dtype=np.float64)
    n_beams = len(group_ids)
    node_indexes = np.stack(
        [np.arange(n_beams), np.arange(1, n_beams + 1),
         np.arange(1, n_beams + 1),
         np.full(n_beams, -2), np.full(n_beams, -2)], axis=1,
    ).astype(np.int64)
    node_coordinates = np.zeros((len(node_x), 3), dtype=np.float64)
    node_coordinates[:, 0] = node_x
    node_velocity = np.zeros_like(node_coordinates)
    node_velocity[:, 0] = node_x * 2.0
    return dict(
        node_indexes=node_indexes,
        node_coordinates=node_coordinates,
        node_displacement=node_coordinates.copy(),
        node_velocity=node_velocity,
        axial_stress=np.asarray(pressures, dtype=np.float64).reshape(n_beams, 1),
        shear_stress=np.stack(
            [
                np.asarray(densities, dtype=np.float64),
                np.ones(n_beams, dtype=np.float64),
            ], axis=1,
        ).reshape(n_beams, 1, 2),
        plastic_strain=np.zeros((n_beams, 1), dtype=np.float64),
        axial_strain=np.asarray(group_ids, dtype=np.float64).reshape(n_beams, 1),
    )


def test_averaged_ale_slots_map_to_pressure_density_alpha_and_velocity():
    arrays = _state_arrays(
        group_ids=[1.0, 1.5, 2.0],
        densities=[1.63e-9, 8.15e-10, 1.225e-12],
        pressures=[0.0, 0.05, 0.10132],
        node_x=[0.0, 1.0, 2.0, 3.0],
    )
    frame = imp._frame_from_state(
        axial_stress=arrays["axial_stress"],
        shear_stress=arrays["shear_stress"],
        plastic_strain=arrays["plastic_strain"],
        axial_strain=arrays["axial_strain"],
        node_velocity=arrays["node_velocity"],
        node_coordinates=arrays["node_coordinates"],
        node_displacement=arrays["node_displacement"],
        node_indexes=arrays["node_indexes"],
        time_s=0.0,
    )
    assert math.isclose(frame["r_left"][0], 0.0, abs_tol=1e-15)
    assert math.isclose(frame["r_right"][0], 1.0e-3, abs_tol=1e-15)
    assert math.isclose(frame["radius"][0], 0.5e-3, abs_tol=1e-15)
    assert math.isclose(frame["density"][0], 1.63e3, rel_tol=1e-15)
    assert math.isclose(frame["density"][2], 1.225, rel_tol=1e-12)
    assert math.isclose(frame["pressure"][2], 0.10132e6, rel_tol=1e-12)
    assert np.allclose(frame["alpha_air"], [0.0, 0.5, 1.0])
    assert np.allclose(frame["alpha_product"], [1.0, 0.5, 0.0])
    assert np.allclose(
        frame["velocity"], [1.0e-3, 3.0e-3, 5.0e-3], rtol=0.0, atol=1e-15
    )


def test_node_displacement_is_current_coordinates_not_additive():
    arrays = _state_arrays(
        group_ids=[1.0, 2.0], densities=[1.63e-9, 1.225e-12],
        pressures=[0.0, 0.10132], node_x=[0.0, 1.0, 2.0],
    )
    current = arrays["node_displacement"].copy()
    current[:, 0] = [10.0, 11.0, 12.0]
    frame = imp._frame_from_state(
        axial_stress=arrays["axial_stress"],
        shear_stress=arrays["shear_stress"],
        plastic_strain=arrays["plastic_strain"],
        axial_strain=arrays["axial_strain"],
        node_velocity=arrays["node_velocity"],
        node_coordinates=arrays["node_coordinates"],
        node_displacement=current,
        node_indexes=arrays["node_indexes"],
        time_s=0.0,
    )
    assert math.isclose(frame["r_left"][0], 10.0e-3, abs_tol=1e-15)
    assert math.isclose(frame["r_right"][-1], 12.0e-3, abs_tol=1e-15)


def test_reversed_beam_node_order_is_sorted():
    arrays = _state_arrays(
        group_ids=[1.0, 2.0], densities=[1.63e-9, 1.225e-12],
        pressures=[0.0, 0.10132], node_x=[0.0, 1.0, 2.0],
    )
    arrays["node_indexes"][0] = [1, 0, 0, -2, -2]
    frame = imp._frame_from_state(
        axial_stress=arrays["axial_stress"],
        shear_stress=arrays["shear_stress"],
        plastic_strain=arrays["plastic_strain"],
        axial_strain=arrays["axial_strain"],
        node_velocity=arrays["node_velocity"],
        node_coordinates=arrays["node_coordinates"],
        node_displacement=arrays["node_displacement"],
        node_indexes=arrays["node_indexes"],
        time_s=0.0,
    )
    assert math.isclose(frame["r_left"][0], 0.0, abs_tol=1e-15)
    assert math.isclose(frame["r_right"][0], 1.0e-3, abs_tol=1e-15)
