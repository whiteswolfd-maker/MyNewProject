"""pytest configuration: put ``src`` on the path and fix float64.

PyTorch is a hard dependency of this project.  A missing dependency is an
error, not a reason to skip the suite: an import-time skip would turn an
uninstalled environment into a misleading "all tests passed" report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for entry in (str(SRC), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)
