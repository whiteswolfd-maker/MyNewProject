"""ContactNet + AirNet spherical TNT air-blast model.

This subpackage implements the 2026-11-10 one-dimensional scheme: one shared
``ContactNet`` ``(t, R0) -> R_c``, one shared ``AirNet``
``(zeta, t, R0) -> (rho, u, p)``, the moving domain ``R_c <= r <= R_s`` where
``R_s`` is the identified outer shock, two persistent optimizers, and an
alternating update loop.  The standard 1-D data entry, shock detection and
event alignment live in :mod:`spherical_blast_pinn.contact_air.one_d`.

The legacy product/air/front network in :mod:`spherical_blast_pinn.models` is
not imported here.  Only the material constants, JWL/ideal-gas EOS, generic
config loading and checkpoint conventions are reused.
"""
from __future__ import annotations

__all__: list[str] = []
