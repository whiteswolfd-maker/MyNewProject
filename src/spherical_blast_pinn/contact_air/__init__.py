"""ContactNet + AirNet spherical TNT air-blast model.

This subpackage implements the 2026-11-04 scheme: one shared ``ContactNet``
``(t, R0) -> R_c``, one shared ``AirNet`` ``(zeta, t, R0) -> (rho, u, p)``,
two independent optimizers, and an alternating update loop.

The legacy product/air/front network in :mod:`spherical_blast_pinn.models` is
not imported here.  Only the material constants, JWL/ideal-gas EOS, generic
config loading and checkpoint conventions are reused.
"""
from __future__ import annotations

__all__: list[str] = []
