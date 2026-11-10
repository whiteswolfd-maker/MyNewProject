"""One-dimensional spherically symmetric TNT air-blast PINN.

The only current scheme lives in :mod:`spherical_blast_pinn.contact_air`:
one shared ``ContactNet(t, R0) -> R_c``, one shared
``AirNet(zeta, t, R0) -> (rho, u, p)``, and the moving domain
``R_c(t) <= r <= R_s(t)`` where ``R_s`` is the outer shock identified from the
raw radial pressure gradient.  Two persistent optimizers alternate between the
air loss and the contact loss.

The one-dimensional data entry, contact labels, shock detection and event
alignment are in :mod:`spherical_blast_pinn.contact_air.one_d`.  There is no
product-field network, no front network, no shell averaging and no ``R_b``
environmental boundary in this package.
"""
from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
