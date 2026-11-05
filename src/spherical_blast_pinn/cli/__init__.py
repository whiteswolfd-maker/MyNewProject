"""命令行入口。每个命令都是 ``python -m spherical_blast_pinn.cli.<name>``。"""
from __future__ import annotations

#: Training / prediction path.
TRAINING_COMMANDS = (
    "startup",
    "build_supervision",
    "validate_startup_planar",
    "check_startup_chain",
    "train",
    "predict",
    "compare",
)

#: Legacy extracted-profile audit.  ``audit_data`` and ``validate_eos``'s data
#: closure require an explicit ``data.cases`` block; neither is a training gate.
LEGACY_AUDIT_COMMANDS = (
    "audit_dyna",
    "audit_data",
    "validate_eos",
)

COMMANDS = TRAINING_COMMANDS + LEGACY_AUDIT_COMMANDS

__all__ = ["COMMANDS", "LEGACY_AUDIT_COMMANDS", "TRAINING_COMMANDS"]
