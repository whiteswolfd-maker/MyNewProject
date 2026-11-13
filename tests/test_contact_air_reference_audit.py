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
from spherical_blast_pinn.contact_air.reference_audit import (
    AuditError,
    fixed_radius_histories,
    frame_keys,
    pulse_metrics,
    read_xy,
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


def _pulse():
    t = np.linspace(0.0, 0.010, 2001)
    dp = np.where(t < 0.002, 0.0, 0.4 * np.exp(-(t - 0.002) / 0.0015))
    return t, dp


def test_pulse_metrics_report_peak_duration_and_impulse():
    t, dp = _pulse()
    metrics = pulse_metrics(t, dp, record_ends=float(t[-1]))
    assert metrics["status"] == "complete"
    assert metrics["peak_mpa"] == pytest.approx(0.4, rel=0.02)
    assert metrics["arrival_s"] == pytest.approx(0.002, abs=2.0e-4)
    assert metrics["duration_s"] is not None
    assert metrics["impulse_mpa_s"] > 0.0


def test_incomplete_positive_phase_is_not_a_full_impulse():
    t, dp = _pulse()
    keep = t <= 0.0035
    metrics = pulse_metrics(t[keep], dp[keep])
    assert metrics["status"] == "partial"
    assert metrics["duration_s"] is None
    assert "positive_phase_not_finished" in metrics["flags"]


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


def test_secondary_pulse_inside_the_positive_phase_is_flagged():
    t = np.linspace(0.0, 0.010, 4001)
    dp = np.where(t < 0.002, 0.0, 0.4 * np.exp(-(t - 0.002) / 0.0015))
    dp += 0.25 * np.exp(-((t - 0.004) / 3.0e-4) ** 2)
    metrics = pulse_metrics(t, dp)
    assert "secondary_pulse_inside_positive_phase" in metrics["flags"]
    assert metrics["status"] == "partial_contaminated"
