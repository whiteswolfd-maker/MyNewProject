"""Command-line entry points.

Only the ContactNet/AirNet scheme remains.  Use
``python -m spherical_blast_pinn.cli.contact_air`` or the equivalent
``python -m spherical_blast_pinn.contact_air.cli``.
"""
from __future__ import annotations

COMMANDS = ("contact_air",)

__all__ = ["COMMANDS"]
