"""Contact-gradient diagnostic helpers: zero directions and angles."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "diagnose_contact_gradients",
    ROOT / "scripts" / "diagnose_contact_gradients.py",
)
assert _SPEC is not None and _SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)


def test_zero_gradient_has_no_direction():
    zero = torch.zeros(8, dtype=torch.float64)
    other = torch.ones(8, dtype=torch.float64)
    result = diagnostic._angle_deg(zero, other, 1.0e-12)
    assert result["status"] == "direction_undefined"
    # A tiny but nonzero vector must not be used to manufacture an angle.
    tiny = torch.full((8,), 1.0e-14, dtype=torch.float64)
    assert diagnostic._angle_deg(tiny, other, 1.0e-12)["status"] == "direction_undefined"


def test_gradient_angles_are_reported_in_degrees():
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    b = torch.tensor([0.0, 1.0], dtype=torch.float64)
    opposite = -a
    assert diagnostic._angle_deg(a, b, 1.0e-12)["angle_deg"] == 90.0
    assert diagnostic._angle_deg(a, opposite, 1.0e-12)["angle_deg"] == 180.0
    assert diagnostic._angle_deg(a, a, 1.0e-12)["angle_deg"] == 0.0
