"""``python -m spherical_blast_pinn.cli.contact_air`` wrapper."""
from __future__ import annotations

from ..contact_air.cli import build_parser, main

__all__ = ["build_parser", "main"]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
