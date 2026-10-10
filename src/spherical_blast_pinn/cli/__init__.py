"""命令行入口。每个命令都是 ``python -m spherical_blast_pinn.cli.<name>``。"""
from __future__ import annotations

COMMANDS = (
    "audit_dyna",
    "audit_data",
    "reference_field",
    "train",
    "predict",
    "compare",
    "validate_eos",
)

__all__ = ["COMMANDS"]
