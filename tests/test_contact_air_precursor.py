"""Precursor physics checks for the ContactNet/AirNet scheme."""
from __future__ import annotations

import math

import pytest

from spherical_blast_pinn.contact_air.precursor import (
    JWLParameters,
    PrecursorConfig,
    PrecursorError,
    air_hugoniot,
    compare_shock_models,
    compute_precursor,
)
from spherical_blast_pinn.contact_air.testing import DEFAULT_PRECURSOR, default_precursor


def _config(**overrides):
    payload = dict(DEFAULT_PRECURSOR)
    payload.update(overrides)
    payload["charge_radius_m"] = 0.05
    return PrecursorConfig.from_mapping(payload)


def test_default_precursor_is_physical_and_all_checks_pass():
    result = default_precursor(0.05)
    assert result.V_cj < result.V_sep
    assert result.p_sep > result.config.air.pressure
    assert result.D_s_sep > 0.0 and result.u_a_sep > 0.0
    assert result.rho_a_sep > result.config.air.density
    assert result.R_sep > 0.05
    assert result.t0 > result.t_sep
    assert result.Rc0 < result.Rs0
    assert result.C > 0.0
    assert all(result.checks.values()), result.checks


def test_units_are_converted_once():
    result = default_precursor(0.05)
    assert result.charge_radius == 0.05
    assert math.isclose(8.8e-6, 8.8e-6)  # the command-line example stays in seconds
    assert result.epsilon == 1.0e-6
    assert result.t0 == pytest.approx(result.t_sep + result.epsilon)


def test_explicit_C_must_reproduce_the_CJ_pressure():
    result = default_precursor(0.05)
    payload = dict(DEFAULT_PRECURSOR)
    payload["charge_radius_m"] = 0.05
    payload["jwl"] = dict(payload["jwl"], C=result.C)
    explicit = compute_precursor(PrecursorConfig.from_mapping(payload))
    assert explicit.C == pytest.approx(result.C)
    assert explicit.V_sep == pytest.approx(result.V_sep, rel=1e-10)
    assert explicit.c_source == "supplied by material card"


def test_non_positive_C_is_refused():
    payload = dict(DEFAULT_PRECURSOR)
    payload["charge_radius_m"] = 0.05
    payload["jwl"] = dict(payload["jwl"], C=-1.0)
    with pytest.raises(PrecursorError):
        JWLParameters.from_mapping(payload["jwl"])


def test_strong_and_exact_air_shocks_are_close_but_reported_separately():
    config = _config()
    comparison = compare_shock_models(config)
    assert comparison["strong"]["shock_model"] == "strong"
    assert comparison["exact"]["shock_model"] == "exact"
    for name, value in comparison["relative_difference"].items():
        assert math.isfinite(value)
        assert value < 1.0e-2, (name, value)


def test_air_hugoniot_rejects_pressure_below_ambient():
    config = _config()
    with pytest.raises(PrecursorError):
        air_hugoniot(config.air.pressure * 0.5, config.air, "strong")


def test_root_failure_is_reported_not_invented():
    payload = dict(DEFAULT_PRECURSOR)
    payload["charge_radius_m"] = 0.05
    payload["v_max"] = 1.5  # the release root is outside this bracket
    with pytest.raises(PrecursorError, match="does not cross"):
        compute_precursor(PrecursorConfig.from_mapping(payload))
