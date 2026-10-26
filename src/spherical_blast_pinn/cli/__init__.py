"""命令行入口。每个命令都是 ``python -m spherical_blast_pinn.cli.<name>``。"""
from __future__ import annotations

COMMANDS = (
    "audit_dyna",
    "audit_data",
    "startup",
    "validate_startup_planar",
    "check_startup_chain",
    "train",
    "predict",
    "compare",
    "validate_eos",
)

__all__ = ["COMMANDS"]
