"""Bridge from the standard 1-D exchange format to the contact-air cache."""
from __future__ import annotations

import numpy as np
import pytest

from spherical_blast_pinn.contact_air.data import (
    ContactLabelConfig,
    build_case,
    load_case,
    save_case,
    synthetic_radial_case,
)
from spherical_blast_pinn.contact_air.one_d import (
    ShockDetectionConfig,
    read_standard_1d,
    write_standard_1d,
)
from spherical_blast_pinn.contact_air.testing import (
    DEFAULT_TOLERANCES,
    default_precursor,
)


def _arrays(frame):
    return {
        "cell_id": frame.cell_id,
        "r_left": frame.r_left,
        "r_right": frame.r_right,
        "radius": frame.radius,
        "alpha_product": frame.alpha_product,
        "alpha_air": frame.alpha_air,
        "material": frame.material,
        "density": frame.density,
        "velocity": frame.velocity,
        "pressure": frame.pressure,
        "valid_density": frame.valid_density,
        "valid_velocity": frame.valid_velocity,
        "valid_pressure": frame.valid_pressure,
    }


def test_standard_source_builds_contact_labels_shock_and_cache(tmp_path):
    precursor = default_precursor(0.05)
    radial = synthetic_radial_case(
        case_id="bridge", charge_radius_m=0.05, precursor=precursor,
        n_frames=7, n_radial=1200,
    )
    path = write_standard_1d(
        tmp_path / "source.npz", case_id="bridge", charge_radius_m=0.05,
        frames=[_arrays(frame) for frame in radial.frames],
        frame_times=[frame.time_raw_s for frame in radial.frames],
        source={"kind": "test_fixture"},
    )
    source = read_standard_1d(path)
    assert source.position_type == "cell_center"
    case = build_case(
        source, precursor=precursor, role="train", tolerances=DEFAULT_TOLERANCES,
        contact_config=ContactLabelConfig(),
        shock_config=ShockDetectionConfig(relative_prominence=4.0,
                                          min_pressure_jump_fraction=1e-2),
    )
    assert case.case_id == "bridge"
    assert case.contact.n_valid > 0
    assert case.shock.valid.sum() >= 2
    assert case.observations.size > 0
    assert case.event_alignment["time_convention"] == "t_model = t0 + (t_raw - t_event_raw)"
    cached = load_case(save_case(case, tmp_path / "case.npz"))
    assert cached.observations.size == case.observations.size
    assert np.allclose(cached.shock.radii, case.shock.radii, equal_nan=True)


def test_a_source_without_explicit_units_is_refused(tmp_path):
    path = tmp_path / "no_units.npz"
    np.savez(
        path,
        metadata=np.asarray(
            '{"format":"spherical_blast_pinn.standard_1d_source","version":1,'
            '"case_id":"x","charge_radius_m":0.05,"position_type":"cell",'
            '"absolute_pressure":true,"frame_times":[0.0,1.0],'
            '"cells_per_frame":[1,1]}'
        ),
        frame_offsets=np.asarray([0, 1, 2]),
        radius=np.asarray([1.0, 1.0]), r_left=np.asarray([0.5, 0.5]),
        r_right=np.asarray([1.5, 1.5]), alpha_product=np.asarray([1.0, 1.0]),
        density=np.asarray([1.0, 1.0]), velocity=np.asarray([0.0, 0.0]),
        pressure=np.asarray([1.0, 1.0]),
    )
    with pytest.raises(Exception, match="unit"):
        read_standard_1d(path)
