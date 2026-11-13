"""Tests for the empirical reference relations and the audit helpers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from spherical_blast_pinn.contact_air.blast_reference import (
    ReferenceRangeError,
    ReferenceUnavailable,
    charge_mass_kg,
    henrych_incident_overpressure_mpa,
    henrych_reference,
    kb_arrival_time,
    kb_incident_overpressure,
    kb_positive_duration,
    kb_positive_impulse,
    scaled_distance,
)
from spherical_blast_pinn.contact_air.one_d import _rh_diagnostic, write_standard_1d
from spherical_blast_pinn.contact_air.reference_audit import (
    AuditError,
    fixed_radius_histories,
    frame_keys,
    peak_preservation,
    pulse_metrics,
    read_xy,
    rh_audit,
    shock_jump_residuals,
)

UNITS = {
    "to_si": {"length": 1.0e-3, "density": 1.0e12,
              "pressure": 1.0e6, "velocity": 1.0e-3},
}
AMBIENT = {"pressure_mpa": 0.10132, "density_kg_m3": 1.225, "gamma": 1.4}
CHARGE = {"density_kg_m3": 1630.0}


def test_uniform_sphere_charge_mass_and_scaling():
    mass = charge_mass_kg(0.05, 1630.0)
    assert mass == pytest.approx(4.0 / 3.0 * math.pi * 1630.0 * 0.05 ** 3)
    assert charge_mass_kg(0.10, 1630.0) / mass == pytest.approx(8.0)
    assert scaled_distance(1.0, mass) == pytest.approx(1.0 / mass ** (1.0 / 3.0))


def test_henrych_is_incident_mpa_and_rejects_out_of_range():
    assert henrych_incident_overpressure_mpa(1.0) == pytest.approx(
        0.61938 - 0.03262 + 0.21324
    )
    value = henrych_reference(0.01)
    assert value.value is None
    assert value.status == "not_comparable"
    with pytest.raises(ReferenceRangeError):
        henrych_incident_overpressure_mpa(0.01)
    with pytest.raises(ReferenceRangeError):
        henrych_incident_overpressure_mpa(20.0)


def test_kingery_bulmash_is_not_guessed():
    for function in (kb_incident_overpressure, kb_arrival_time,
                     kb_positive_duration, kb_positive_impulse):
        with pytest.raises(ReferenceUnavailable):
            function(1.0)


def _first_pulse(end: float = 0.004, amplitude: float = 0.4,
                 samples: int = 2001):
    """A smooth pulse that arrives at 2 ms and returns to zero at ``end``."""
    t = np.linspace(0.0, 0.010, samples)
    dp = np.zeros_like(t)
    mask = (t >= 0.002) & (t <= end)
    dp[mask] = amplitude * np.sin(
        np.pi * (t[mask] - 0.002) / (end - 0.002)
    )
    return t, dp


def test_pulse_metrics_report_peak_duration_and_impulse():
    t, dp = _first_pulse()
    metrics = pulse_metrics(t, dp, record_ends=float(t[-1]))
    assert metrics["status"] == "complete"
    assert metrics["peak_mpa"] == pytest.approx(0.4, rel=0.02)
    assert metrics["arrival_s"] == pytest.approx(0.002, abs=2.0e-4)
    assert metrics["phase_end_s"] == pytest.approx(0.004, abs=1.0e-4)
    assert metrics["duration_s"] == pytest.approx(0.002, abs=2.0e-4)
    assert metrics["impulse_mpa_s"] > 0.0


def test_incomplete_positive_phase_is_not_a_full_impulse():
    t, dp = _first_pulse()
    keep = t <= 0.0035
    metrics = pulse_metrics(t[keep], dp[keep])
    assert metrics["status"] == "partial"
    assert metrics["duration_s"] is None
    assert "positive_phase_not_finished" in metrics["flags"]


def test_peak_is_taken_from_the_first_pulse_not_the_record_maximum():
    t, dp = _first_pulse()
    # A later, larger excursion must not become the incident peak.
    later = (t >= 0.006) & (t <= 0.008)
    dp[later] += 0.6 * np.sin(np.pi * (t[later] - 0.006) / 0.002)
    metrics = pulse_metrics(t, dp)
    assert metrics["peak_mpa"] == pytest.approx(0.4, rel=0.02)
    assert metrics["peak_time_s"] == pytest.approx(0.003, abs=2.0e-4)
    assert metrics["status"] == "complete"
    assert metrics["later_pulse_candidates_s"]
    assert max(metrics["later_pulse_candidate_peaks_mpa"]) == pytest.approx(
        0.6, rel=0.05
    )
    assert "later_pulse_candidates_present" in metrics["flags"]


def test_oscillation_inside_the_first_pulse_is_a_candidate_not_contamination():
    t, dp = _first_pulse()
    dp += 0.10 * np.exp(-((t - 0.0035) / 3.0e-5) ** 2)
    metrics = pulse_metrics(t, dp)
    assert metrics["status"] == "complete"
    assert "intra_pulse_maxima_candidates" in metrics["flags"]
    assert metrics["intra_pulse_maxima_s"]
    assert "secondary_pulse_inside_positive_phase" not in metrics["flags"]


def _mach3_rankine_hugoniot(gamma: float = 1.4, mach: float = 3.0,
                            rho1: float = 1.225, p1: float = 101325.0):
    c1 = math.sqrt(gamma * p1 / rho1)
    ds = mach * c1
    ratio = (gamma + 1.0) * mach ** 2 / (
        (gamma - 1.0) * mach ** 2 + 2.0
    )
    p2 = p1 * (1.0 + 2.0 * gamma / (gamma + 1.0) * (mach ** 2 - 1.0))
    rho2 = ratio * rho1
    u2 = ds * (1.0 - 1.0 / ratio)
    pre = {"rho": rho1, "u": 0.0, "p": p1}
    post = {"rho": rho2, "u": u2, "p": p2}
    return pre, post, ds


def test_moving_shock_energy_jump_vanishes_for_exact_rankine_hugoniot():
    gamma = 1.4
    pre, post, ds = _mach3_rankine_hugoniot(gamma=gamma)
    residuals = shock_jump_residuals(pre, post, ds, gamma)
    assert residuals["energy_flux_form"] == "u(E+p) - D_s E"
    assert residuals["mass_residual_normalised"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert residuals["momentum_residual_normalised"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert residuals["energy_residual_normalised"] == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_shock_frame_enthalpy_diagnostic_vanishes_for_exact_rankine_hugoniot():
    pre, post, _ = _mach3_rankine_hugoniot()
    diagnostic = _rh_diagnostic(
        post["rho"], post["u"], post["p"],
        pre["rho"], pre["u"], pre["p"],
    )
    assert diagnostic["status"] == "computed_by_mass_from_measured_states"
    assert diagnostic["momentum_residual_normalised"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert diagnostic["energy_residual_normalised"] == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_peak_preservation_uses_the_raw_first_pulse_window(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    times = [f"{i * 1.0e-6:.2E}" for i in range(8)]
    x_mm = np.array([400.0, 500.0, 600.0])
    # At r=500 mm: ambient, first pulse at 2-4 us, then a larger later pulse.
    pressure = [0.10132, 0.10132, 0.50, 0.30, 0.10132, 0.10132, 1.00, 0.10132]
    for key, value in zip(times, pressure):
        _write_xy(raw / f"profile2_pressure_x_time{key}.xy", x_mm,
                  np.array([0.10132, value, 0.10132]))
    # Imported frames deliberately omit the first-pulse peak (frame 2) but
    # retain the later, larger pulse (frame 6).  A record-maximum method would
    # wrongly report the later pulse; the common raw window must not.
    keep = [0, 1, 3, 4, 5, 6, 7]
    frames = []
    for index in keep:
        frames.append({
            "r_left": np.array([0.3995, 0.4995, 0.5995]),
            "r_right": np.array([0.4005, 0.5005, 0.6005]),
            "radius": x_mm * 1.0e-3,
            "alpha_product": np.zeros(3),
            "density": np.full(3, 1.225),
            "velocity": np.zeros(3),
            "pressure": np.array([0.10132, pressure[index], 0.10132]) * 1.0e6,
            "valid_density": np.ones(3, bool),
            "valid_velocity": np.ones(3, bool),
            "valid_pressure": np.ones(3, bool),
        })
    standard = tmp_path / "standard.npz"
    write_standard_1d(
        standard, case_id="fixture", charge_radius_m=0.05,
        frames=frames, frame_times=[float(times[index]) for index in keep],
        source={"selected_raw_frame_keys": [times[index] for index in keep],
                "frame_selection_policy": "all_frames"},
    )
    rows = peak_preservation(
        {"case_id": "fixture", "radius_m": 0.05, "raw_dir": str(raw),
         "standard": "standard.npz"},
        tmp_path, UNITS, AMBIENT, CHARGE, [0.5],
    )
    row = rows[0]
    assert row["raw_peak_mpa"] == pytest.approx(0.39868, abs=1.0e-6)
    assert row["imported_peak_mpa"] == pytest.approx(0.19868, abs=1.0e-6)
    assert row["common_window_end_s"] == pytest.approx(4.0e-6, abs=1.0e-12)
    assert row["raw_peak_frame_selected"] is False
    assert row["peak_difference_mpa"] < 0.0


def test_rh_audit_does_not_differentiate_across_an_invalid_shock_frame(tmp_path):
    n_cells = 30
    n_frames = 5
    r_left = np.linspace(0.0, 0.2, n_cells, endpoint=False)
    r_right = np.linspace(0.0, 0.2, n_cells + 1)[1:]
    radius = 0.5 * (r_left + r_right)
    frames = []
    for _ in range(n_frames):
        frames.append({
            "r_left": r_left.copy(),
            "r_right": r_right.copy(),
            "radius": radius.copy(),
            "alpha_product": np.zeros(n_cells),
            "density": np.full(n_cells, 1.225),
            "velocity": np.zeros(n_cells),
            "pressure": np.full(n_cells, 101325.0),
            "valid_density": np.ones(n_cells, bool),
            "valid_velocity": np.ones(n_cells, bool),
            "valid_pressure": np.ones(n_cells, bool),
        })
    standard = tmp_path / "standard.npz"
    write_standard_1d(
        standard, case_id="fixture", charge_radius_m=0.01,
        frames=frames, frame_times=[0.0, 1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5],
    )
    cache = tmp_path / "cache.npz"
    np.savez(
        cache,
        shock_times=np.array([0.0, 1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5]),
        shock_radius=np.array([0.05, 0.06, 0.07, 0.08, 0.09]),
        shock_valid=np.array([True, True, False, True, True]),
        raw_times=np.array([0.0, 1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5]),
    )
    audit = rh_audit(
        {"case_id": "fixture", "standard": str(standard), "cache": str(cache)},
        tmp_path, {"gamma": 1.4},
    )
    assert audit["samples"] == []
    assert len(audit["not_evaluable"]) == 3
    assert all(
        any(token in entry["reason"] for token in ("valid", "segment", "break"))
        for entry in audit["not_evaluable"]
    )


def _write_xy(path, x, y):
    lines = ["Curveplot", "fixture", "x", "y", " ", " #pts=1"]
    lines += [f"  {a:.12E}  {b:.12E}" for a, b in zip(x, y)]
    lines.append("endcurve")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_xy_reader_requires_the_endcurve_marker(tmp_path):
    path = tmp_path / "profile2_pressure_x_time0.00E+00.xy"
    path.write_text("Curveplot\nfixture\nx\ny\n \n #pts=1\n 1.0 2.0\n",
                    encoding="ascii")
    with pytest.raises(AuditError, match="endcurve"):
        read_xy(path)


def test_fixed_radius_history_uses_the_fixed_physical_radius(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    x_mm = np.array([1.0, 3.0, 5.0])
    for step, time in enumerate(("0.00E+00", "1.00E-06", "2.00E-06")):
        _write_xy(raw / f"profile2_pressure_x_time{time}.xy", x_mm,
                  np.array([0.10132, 0.2 + step, 0.10132]))
    assert frame_keys(raw, "profile2_pressure_x_time*.xy") == [
        "0.00E+00", "1.00E-06", "2.00E-06"
    ]
    z = 0.003 / charge_mass_kg(0.05, 1630.0) ** (1.0 / 3.0)
    histories = fixed_radius_histories(
        {"case_id": "fixture", "radius_m": 0.05, "raw_dir": str(raw)},
        UNITS, AMBIENT, CHARGE, [z],
    )
    entry = next(iter(histories.values()))
    assert np.allclose(entry["t"], [0.0, 1.0e-6, 2.0e-6])
    assert np.allclose(entry["dp_mpa"], [0.09868, 1.09868, 2.09868])
    assert abs(entry["offset_m"][0]) < 2.0e-3
