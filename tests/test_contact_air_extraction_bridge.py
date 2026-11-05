"""Bridge from an extracted profile directory to the contact-air cache."""
from __future__ import annotations

import json

import numpy as np

from spherical_blast_pinn.contact_air.data import build_case_from_profile
from spherical_blast_pinn.contact_air.testing import (
    DEFAULT_TOLERANCES,
    default_precursor,
)


def _write_profile(directory, *, times, precursor):
    directory.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.02, 0.20, 100)
    n_t = len(times)
    material = np.full((n_t, grid.size), 1, dtype=np.int8)
    material[:, grid < 0.105] = 0
    material[:, (grid >= 0.105) & (grid < 0.11)] = 2
    pressure = np.full((n_t, grid.size), 101320.0)
    velocity = np.zeros((n_t, grid.size))
    density = np.full((n_t, grid.size), 1.225)
    rb_values = 0.16 + 2000.0 * (np.asarray(times) - times[0])
    for frame, rb in enumerate(rb_values):
        air = material[frame] == 1
        shocked = air & (grid < rb)
        pressure[frame, shocked] = precursor.p_sep
        velocity[frame, shocked] = precursor.u_a_sep
        density[frame, shocked] = precursor.rho_a_sep
    np.savez(
        directory / "raw_line.npz",
        t=np.asarray(times, dtype=float), x=grid,
        rho=density, u=velocity, P=pressure,
    )
    (directory / "metadata.json").write_text(json.dumps({
        "R_0": precursor.charge_radius,
        "detonation_velocity_m_s": 6930.0,
        "material_partition_is_volume_fraction_based": True,
        "material_partition_source": {
            "method": "ale_volume_fraction",
            "producer": "cli.extract_profiles",
            "producer_version": 1,
            "fields": ["tnt_volume_fraction"],
            "input_identifier": "sha256:test",
            "detail": {"pure_product_threshold": 0.95},
        },
        "symmetry_factor": 8.0,
        "symmetry_share": 0.125,
    }), encoding="utf-8")
    valid = np.ones((n_t, grid.size), dtype=bool)
    np.savez(
        directory / "material.npz",
        material=material,
        valid=valid,
        contact_radius=np.full(n_t, 0.1075),
        shock_radius=np.full(n_t, 0.16),
        contact_state=np.full(n_t, 2, dtype=np.int64),
        shock_state=np.full(n_t, 2, dtype=np.int64),
        contact_has_supervision=np.ones(n_t, dtype=bool),
        shock_has_supervision=np.ones(n_t, dtype=bool),
        contact_mixing_band_inner_m=np.full(n_t, 0.105),
        contact_mixing_band_outer_m=np.full(n_t, 0.11),
        field_valid_density=valid,
        field_valid_velocity=valid,
        field_valid_energy=valid,
        field_valid_pressure=valid,
    )
    return directory


def test_extracted_profile_builds_contact_interval_air_observations_and_rb(tmp_path):
    precursor = default_precursor(0.05)
    times = precursor.t0 + np.arange(6) * 2.0e-6
    directory = _write_profile(tmp_path / "profile", times=times, precursor=precursor)
    case = build_case_from_profile(
        directory, precursor=precursor, role="train",
        tolerances=DEFAULT_TOLERANCES,
    )
    assert case.case_id
    assert case.contact.n_valid == len(times)
    assert np.all(case.contact.low < case.contact.raw)
    assert np.all(case.contact.raw < case.contact.up)
    # The low/up values are the shell centres adjacent to the mixed segment,
    # not the band edges 0.105/0.11.
    assert np.all((case.contact.low > 0.10) & (case.contact.low < 0.105))
    assert np.all((case.contact.up > 0.107) & (case.contact.up < 0.112))
    assert case.observations.size > 0
    assert case.initial.Rb0 > case.initial.Rs0 > case.initial.Rc0
    assert case.t_end > case.t0
    assert case.source["profile_dir"] == str(directory)
