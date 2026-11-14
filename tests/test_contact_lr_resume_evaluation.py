"""Corrected air/physics aggregation and physical-coordinate mapping tests."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

from spherical_blast_pinn.contact_air.mapping import (
    assemble_residuals,
    build_geometry,
    physical_residuals,
)
from spherical_blast_pinn.contact_air.models import AirFields

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_contact_lr_resume",
    ROOT / "scripts" / "evaluate_contact_lr_resume.py",
)
assert _SPEC is not None and _SPEC.loader is not None
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


def _metric(mse, n=10):
    return {"status": "ok", "mse": mse, "rmse": math.sqrt(mse), "n": n}


def test_mean_case_rmse_is_not_macro_rmse():
    aggregate = evaluation._aggregate_metric_rows([_metric(1.0), _metric(9.0)])
    assert aggregate["mean_case_rmse"] == 2.0
    assert aggregate["macro_mse"] == 5.0
    assert aggregate["macro_rmse"] == math.sqrt(5.0)
    assert aggregate["pooled_mse"] == 5.0
    assert aggregate["pooled_rmse"] == math.sqrt(5.0)


def test_macro_and_pooled_use_the_right_point_denominators():
    aggregate = evaluation._aggregate_metric_rows([
        _metric(1.0, n=10), _metric(9.0, n=30),
    ])
    assert aggregate["mean_case_rmse"] == 2.0
    assert aggregate["macro_mse"] == 5.0
    assert aggregate["pooled_mse"] == (1.0 * 10 + 9.0 * 30) / 40
    assert aggregate["pooled_rmse"] == math.sqrt(7.0)
    assert aggregate["n_points"] == 40


def test_missing_air_case_is_explicit_and_not_complete():
    aggregate = evaluation._aggregate_metric_rows([
        _metric(1.0),
        {"status": "not_evaluable", "mse": None, "rmse": None, "n": 0},
    ])
    assert aggregate["n_cases"] == 2
    assert aggregate["n_cases_evaluable"] == 1
    assert aggregate["n_cases_missing"] == 1
    assert aggregate["complete"] is False
    assert aggregate["status"] == "ok"
    empty = evaluation._aggregate_metric_rows([
        {"status": "not_evaluable", "mse": None, "rmse": None, "n": 0},
    ])
    assert empty["status"] == "not_evaluable"
    assert empty["macro_mse"] is None


def test_common_mask_counts_exclusions_and_empty_sets():
    models = {
        "a": {
            "inside": np.array([True, True, False, True]),
            "valid": {"rho": np.array([True, False, True, True])},
            "pred": {"rho": np.array([1.0, 2.0, 3.0, np.nan])},
        },
        "b": {
            "inside": np.array([True, True, True, False]),
            "valid": {"rho": np.array([True, True, True, True])},
            "pred": {"rho": np.array([1.0, np.nan, 3.0, 4.0])},
        },
    }
    mask = evaluation._common_mask(models, ("a", "b"), "rho")
    assert mask.tolist() == [True, False, False, False]
    metrics = evaluation._field_metrics(
        models["a"]["pred"]["rho"], np.array([1.0, 2.0, 3.0, 5.0]),
        mask, 1.0,
    )
    assert metrics["status"] == "ok"
    assert metrics["n"] == 1
    empty = evaluation._field_metrics(
        np.array([1.0]), np.array([1.0]), np.array([False]), 1.0
    )
    assert empty["status"] == "not_evaluable"
    assert empty["mse"] is None


def _analytic():
    dtype = torch.float64
    t = torch.tensor([1.0e-5, 2.0e-5, 3.0e-5], dtype=dtype).reshape(-1, 1)
    r = torch.tensor([0.13, 0.15, 0.18], dtype=dtype).reshape(-1, 1)
    rs = 0.20 + 1500.0 * (t - 1.0e-5) + 2.0e7 * (t - 1.0e-5) ** 2
    ds = 1500.0 + 4.0e7 * (t - 1.0e-5)
    return t, r, rs, ds


def test_different_rc_same_zeta_gives_different_physical_radius():
    t, _r, rs, ds = _analytic()
    zeta = torch.full_like(t, 0.4)
    g1 = build_geometry(
        torch.full_like(t, 0.10), torch.full_like(t, 1000.0),
        rs, ds, zeta,
    )
    g2 = build_geometry(
        torch.full_like(t, 0.12), torch.full_like(t, 1000.0),
        rs, ds, zeta,
    )
    assert not torch.allclose(g1.radius, g2.radius)


def test_same_physical_point_maps_back_to_same_radius_for_both_models():
    t, r, rs, ds = _analytic()
    for rc_value in (0.10, 0.12):
        rc = torch.full_like(t, rc_value)
        zeta = (r - rc) / (rs - rc)
        geometry = build_geometry(
            rc, torch.full_like(t, 1000.0), rs, ds, zeta
        )
        assert torch.allclose(geometry.radius, r, rtol=1e-14, atol=1e-14)


def test_mapped_residual_from_fixed_physical_point_has_no_double_chain():
    t, r, rs, ds = _analytic()
    rc_values = (0.10, 0.12)
    physical = None
    for rc_value in rc_values:
        rc = torch.full_like(t, rc_value)
        rc_dot = torch.full_like(t, 1000.0)
        zeta = (r - rc) / (rs - rc)
        geometry = build_geometry(rc, rc_dot, rs, ds, zeta)
        rho = 1.2 + 0.3 * r + 0.2 * t
        u = 1000.0 + 2.0 * r + 3.0 * t
        p = 1.0e5 + 10.0 * r + 20.0 * t
        fields = AirFields(
            density=rho, velocity=u, pressure=p,
            specific_energy=p / (0.4 * rho),
        )
        derivatives = {
            "rho_t": 0.2 + 0.3 * geometry.grid_speed,
            "rho_z": 0.3 * geometry.length,
            "u_t": 3.0 + 2.0 * geometry.grid_speed,
            "u_z": 2.0 * geometry.length,
            "p_t": 20.0 + 10.0 * geometry.grid_speed,
            "p_z": 10.0 * geometry.length,
        }
        gamma = torch.full_like(rho, 1.4)
        mapped = assemble_residuals(
            fields, derivatives, geometry, gamma=gamma
        )
        expected = physical_residuals(
            rho, u, p,
            torch.full_like(rho, 0.2), torch.full_like(rho, 0.3),
            torch.full_like(u, 3.0), torch.full_like(u, 2.0),
            torch.full_like(p, 20.0), torch.full_like(p, 10.0),
            r, gamma=gamma,
        )
        if physical is None:
            physical = expected
        for name in ("mass", "momentum", "energy"):
            assert torch.allclose(
                getattr(mapped, name), getattr(expected, name),
                rtol=1e-12, atol=1e-9,
            )
            assert torch.allclose(
                getattr(expected, name), getattr(physical, name),
                rtol=1e-12, atol=1e-9,
            )
