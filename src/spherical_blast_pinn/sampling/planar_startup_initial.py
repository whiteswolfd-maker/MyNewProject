"""Adapt a validated planar-startup field to the existing IC sample contract.

The existing :class:`~spherical_blast_pinn.sampling.batch.SampleBatch` initial
condition block has eight tensors, all shaped ``(N, 1)``:

* product: ``xi_ic_product``, ``rho_ic_product``, ``u_ic_product``,
  ``e_ic_product``;
* air: ``eta_ic_air``, ``rho_ic_air``, ``u_ic_air``, ``e_ic_air``;

plus the scalar ``contact_ic``/``shock_ic`` positions.  The existing initial
loss evaluates both groups at ``s = 0``, so both sides of this adapter are at
the startup field's own ``tau = 0`` (``t_phys = epsilon``).  ``diagnostics``
records the physical time, ``tau`` and the dimensionless ``s`` obtained from the
scales themselves, so the batch and its scales cannot disagree silently.

Only the radial coordinates are non-dimensionalised, and only once:
``xi = r/R_c`` for ``0 <= r <= R_c`` and ``eta = (r-R_c)/(R_s-R_c)`` for
``R_c <= r <= R_s``.  Density, velocity and specific energy stay in SI; no
second unit conversion is applied.  The inverse is therefore exactly
``r = xi R_c`` and ``r = R_c + eta (R_s-R_c)``.

The batch must carry the **planar-startup** time convention,
:class:`PlanarStartupScales`, whose origin is the field's own ``epsilon``:
``s = 0`` -> ``tau = 0`` -> ``t_phys = epsilon``.  A scheme-B
:class:`SimilitudeScales` object (origin ``T_h = R_0/D + epsilon(R_0)``) is
rejected explicitly instead of being silently reinterpreted: both origins have
seconds as their unit, so a wrong one shifts the whole trajectory without
raising anything.  The batch therefore also carries no detonation velocity
(``None`` = not applicable); the new origin does not involve ``D``.

The adapter samples the field with its one-sided methods:

* product group via :meth:`PlanarStartupField.sample_product`;
* air group via :meth:`PlanarStartupField.sample_air`, so the contact air side
  and the post-shock side at ``R_s`` are used explicitly.

No interpolation across the contact or shock is performed, and no old
``ReferenceField`` or precursor is imported.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..physics.similitude import PlanarStartupScales, TIME_CONVENTION_PLANAR_STARTUP
from ..physics.two_medium_initial import AIR, PRODUCT
from ..physics.two_medium_startup_planar import PlanarStartupField
from .batch import SampleBatch

PLANAR_STARTUP_IC_ADAPTER_VERSION = "planar_startup_initial_batch_v1"

#: The existing sampler contract uses one column per sample.
_COLUMN_SHAPE_DESCRIPTION = "(N, 1)"

#: Only real floating-point targets are meaningful for the IC loss.  float64
#: is the existing sampler default; float32 is the only optional precision.
_SUPPORTED_DTYPES = (torch.float64, torch.float32) if torch is not None else ()


class PlanarStartupBatchError(ValueError):
    """The planar-startup field cannot be adapted to the IC batch contract."""


def _resolve_dtype(dtype: Any) -> Any:
    """Return a supported torch floating dtype or fail before tensor creation."""
    if dtype is None:
        return torch.float64
    if not isinstance(dtype, torch.dtype) or dtype not in _SUPPORTED_DTYPES:
        raise PlanarStartupBatchError(
            "unsupported dtype: "
            f"{dtype!r}; supported dtypes are torch.float64 (default) and torch.float32"
        )
    return dtype


def _validate_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise PlanarStartupBatchError(f"{name} must be an integer >= 1, got {value!r}")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise PlanarStartupBatchError(
            f"{name} must be an integer >= 1, got {value!r}"
        ) from exc
    if integer != value or integer < 1:
        raise PlanarStartupBatchError(f"{name} must be an integer >= 1, got {value!r}")
    return integer


def _validate_scales(field: PlanarStartupField, scales: PlanarStartupScales) -> None:
    """require the planar-startup convention, the same radius and the same epsilon."""
    if not isinstance(scales, PlanarStartupScales):
        raise PlanarStartupBatchError(
            "the initial-condition batch needs the planar-startup time convention "
            f"({TIME_CONVENTION_PLANAR_STARTUP!r}); got "
            f"{type(scales).__name__} with convention "
            f"{getattr(scales, 'time_convention', None)!r}. Its s=0 would be the "
            "scheme-B T_h = R_0/D + epsilon(R_0), not the startup field's epsilon. "
            "Build one with PlanarStartupScales.for_startup_field(field, "
            "time_scale=...)."
        )
    epsilon = float(field.epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise PlanarStartupBatchError(
            f"the startup field epsilon must be finite and positive, got {epsilon!r}"
        )
    if not math.isclose(
        float(scales.charge_radius),
        float(field.charge_radius),
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise PlanarStartupBatchError(
            "the supplied scales belong to R_0 = "
            f"{scales.charge_radius!r} m, but the startup field has R_0 = "
            f"{field.charge_radius!r} m"
        )
    if not math.isclose(
        float(scales.startup_epsilon),
        epsilon,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise PlanarStartupBatchError(
            "the supplied scales start at epsilon = "
            f"{scales.startup_epsilon!r} s, but the startup field is at epsilon = "
            f"{epsilon!r} s; the time origin must come from this field"
        )


def _radii_from_argument(
    values: Sequence[float] | np.ndarray | None,
    *,
    count: int,
    lower: float,
    upper: float,
    name: str,
) -> np.ndarray:
    if values is None:
        radii = np.linspace(float(lower), float(upper), int(count), dtype=np.float64)
    else:
        try:
            radii = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise PlanarStartupBatchError(
                f"{name} must be a one-dimensional sequence of real radii"
            ) from exc
        if radii.ndim != 1:
            raise PlanarStartupBatchError(
                f"{name} must be a one-dimensional sequence of physical radii; "
                f"got shape {radii.shape}. Refusing to flatten an ambiguous input."
            )
        if radii.size == 0:
            raise PlanarStartupBatchError(f"{name} must contain at least one radius")
    if not np.all(np.isfinite(radii)):
        raise PlanarStartupBatchError(f"{name} contains non-finite radii")
    if np.any(radii < float(lower)) or np.any(radii > float(upper)):
        raise PlanarStartupBatchError(
            f"{name} must lie in the closed interval [{lower!r}, {upper!r}] m"
        )
    return radii


def _with_anchor_radii(radii: np.ndarray, anchors: Sequence[float]) -> np.ndarray:
    """Add required one-sided anchors and return sorted unique radii."""
    combined = np.concatenate(
        [np.asarray(radii, dtype=np.float64), np.asarray(anchors, dtype=np.float64)]
    )
    return np.unique(combined)


def _column(values: np.ndarray, *, dtype: Any, device: Any) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise PlanarStartupBatchError(
            f"internal column values must be one-dimensional, got shape {array.shape}"
        )
    if array.size == 0:
        raise PlanarStartupBatchError("refusing to build an empty initial-condition group")
    return torch.as_tensor(array.reshape(-1, 1), dtype=dtype, device=device)


def build_planar_startup_initial_batch(
    field: PlanarStartupField,
    *,
    scales: PlanarStartupScales,
    product_radii: Sequence[float] | np.ndarray | None = None,
    air_radii: Sequence[float] | np.ndarray | None = None,
    n_ic_product: int = 256,
    n_ic_air: int = 256,
    case_id: str | None = None,
    dtype: Any = None,
    device: Any = None,
) -> SampleBatch:
    """Build the initial-condition block of a ``SampleBatch`` from ``field``.

    Parameters
    ----------
    field:
        A validated :class:`PlanarStartupField`.
    scales:
        A :class:`PlanarStartupScales` built from **this** field (typically via
        :meth:`PlanarStartupScales.for_startup_field`).  It supplies the material
        scales and the time origin ``t_phys(s=0) = field.epsilon``.  A scheme-B
        :class:`SimilitudeScales`, a different charge radius or a different
        startup epsilon is rejected.
    product_radii, air_radii:
        Optional explicit physical radii in metres.  A two-dimensional input is
        rejected rather than flattened.  When omitted, ``n_ic_product`` and
        ``n_ic_air`` uniformly spaced radii are used.
    n_ic_product, n_ic_air:
        Counts for the default uniform radii; ignored when explicit radii are
        supplied.
    case_id:
        Optional case label.  Defaults to a startup cache-key label.
    dtype, device:
        Torch tensor placement.  Only ``torch.float64`` (default) and
        ``torch.float32`` are supported.  Integer, boolean, complex, float16,
        bfloat16 and non-torch dtypes are rejected before any tensor is built;
        all eight converted tensors must also remain finite.

    Returns
    -------
    SampleBatch
        The initial-condition fields are populated.  PDE, data and time-window
        fields remain unset.  Product points carry ``xi = r/R_c``; air points
        carry ``eta = (r-R_c)/(R_s-R_c)``.  ``rho/u/e`` are physical SI values.

    The product group always contains ``r=0`` and the product side of ``R_c``.
    The air group always contains the air side of ``R_c`` and the post-shock
    side of ``R_s``.  These anchors are added even for explicit inputs.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the initial-condition batch")
    if not isinstance(field, PlanarStartupField):
        raise PlanarStartupBatchError(
            "field must be a validated PlanarStartupField"
        )
    if field.checks.get("passed") is not True:
        raise PlanarStartupBatchError(
            "the startup field has no passed consistency report; refusing to sample it"
        )
    _validate_scales(field, scales)
    dtype = _resolve_dtype(dtype)
    if case_id is None:
        case_id = f"planar_startup:{field.cache_key()[:16]}"
    if not isinstance(case_id, str) or not case_id:
        raise PlanarStartupBatchError("case_id must be a non-empty string")

    contact = float(field.contact_radius)
    shock = float(field.shock_radius)
    charge = float(field.charge_radius)
    outer = float(field.radii[-1])
    if not (0.0 < contact < shock <= outer):
        raise PlanarStartupBatchError(
            "the startup field has no usable product/air interval: "
            f"R_c={contact!r}, R_s={shock!r}, outer={outer!r}"
        )

    n_product = (
        _validate_positive_int("n_ic_product", n_ic_product)
        if product_radii is None
        else 1
    )
    n_air = (
        _validate_positive_int("n_ic_air", n_ic_air)
        if air_radii is None
        else 1
    )
    r_product = _radii_from_argument(
        product_radii,
        count=n_product,
        lower=0.0,
        upper=contact,
        name="product_radii",
    )
    r_air = _radii_from_argument(
        air_radii,
        count=n_air,
        lower=contact,
        upper=shock,
        name="air_radii",
    )
    r_product = _with_anchor_radii(r_product, (0.0, contact))
    r_air = _with_anchor_radii(r_air, (contact, shock))

    product_states = field.sample_product(r_product)
    air_states = field.sample_air(r_air)
    if not np.all(np.asarray(product_states["material"]) == PRODUCT):
        raise PlanarStartupBatchError(
            "the product samples contain cells not tagged PRODUCT"
        )
    if not np.all(np.asarray(air_states["material"]) == AIR):
        raise PlanarStartupBatchError(
            "the air samples contain cells not tagged AIR"
        )

    xi = r_product / contact
    gap = shock - contact
    eta = (r_air - contact) / gap
    for name, values in (
        ("xi_ic_product", xi),
        ("eta_ic_air", eta),
    ):
        if not np.all(np.isfinite(values)):
            raise PlanarStartupBatchError(f"{name} contains non-finite values")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise PlanarStartupBatchError(
                f"{name} lies outside the closed unit interval"
            )

    xi_column = _column(xi, dtype=dtype, device=device)
    eta_column = _column(eta, dtype=dtype, device=device)
    rho_p = _column(product_states["density"], dtype=dtype, device=device)
    u_p = _column(product_states["velocity"], dtype=dtype, device=device)
    e_p = _column(product_states["specific_energy"], dtype=dtype, device=device)
    rho_a = _column(air_states["density"], dtype=dtype, device=device)
    u_a = _column(air_states["velocity"], dtype=dtype, device=device)
    e_a = _column(air_states["specific_energy"], dtype=dtype, device=device)
    converted = (
        ("xi_ic_product", xi_column),
        ("rho_ic_product", rho_p),
        ("u_ic_product", u_p),
        ("e_ic_product", e_p),
        ("eta_ic_air", eta_column),
        ("rho_ic_air", rho_a),
        ("u_ic_air", u_a),
        ("e_ic_air", e_a),
    )
    for name, tensor in converted:
        if not bool(torch.isfinite(tensor).all()):
            raise PlanarStartupBatchError(
                f"{name} contains non-finite values after conversion to {dtype}; "
                "refusing to clip or replace them"
            )
    xi_shape = _COLUMN_SHAPE_DESCRIPTION.replace("N", str(xi.size))
    eta_shape = _COLUMN_SHAPE_DESCRIPTION.replace("N", str(eta.size))

    # Both groups sit at s=0.  Report the *actual* conversions of the scales the
    # batch carries, not literals: if the origin were wrong this block would
    # disagree with ``scales`` instead of hard-coding the expected answer.
    initial_physical_time = float(scales.physical_time_from_network(0.0))
    initial_tau = float(scales.pinn_time_from_physical(initial_physical_time))
    initial_network_time = float(scales.network_time_from_physical(initial_physical_time))
    diagnostics = {
        "adapter": PLANAR_STARTUP_IC_ADAPTER_VERSION,
        "planar_startup_model_id": field.model_id,
        "planar_startup_cache_key": field.cache_key(),
        "planar_startup_epsilon_s": float(field.epsilon),
        "time_convention": scales.time_convention,
        "similitude_start_time_s": float(scales.start_time()),
        "similitude_startup_epsilon_s": float(scales.startup_epsilon),
        "initial_physical_time_s": initial_physical_time,
        "initial_physical_time_product_s": initial_physical_time,
        "initial_physical_time_air_s": initial_physical_time,
        "initial_tau_s": initial_tau,
        "initial_tau_product_s": initial_tau,
        "initial_tau_air_s": initial_tau,
        # s 是无量纲的网络时间，不能带秒的后缀。
        "initial_network_time": initial_network_time,
        "initial_network_time_product": initial_network_time,
        "initial_network_time_air": initial_network_time,
        "initial_network_time_is_dimensionless": True,
        "initial_time_convention": (
            "s=0 -> tau=0 -> t_phys=epsilon_field for both IC groups; the batch "
            "scales carry the planar-startup origin, not T_h=R_0/D+epsilon"
        ),
        "similitude_time_origin_note": (
            "PlanarStartupScales.start_time() is the field's own epsilon; the "
            "scheme-B T_h=R_0/D+epsilon(R_0) is a different object and is rejected"
        ),
        "n_ic_product": int(xi.size),
        "n_ic_air": int(eta.size),
        "product_column_shape": xi_shape,
        "air_column_shape": eta_shape,
        "contact_ic_m": contact,
        "shock_ic_m": shock,
        "contact_product_side": "sample_product (left)",
        "contact_air_side": "sample_air (right)",
        "shock_air_side": "sample_air (post-shock left side)",
        "material_labels": {"product": int(PRODUCT), "air": int(AIR)},
        "one_sided_sampling": True,
        "nondimensionalization": "xi=r/Rc, eta=(r-Rc)/(Rs-Rc); rho/u/e remain SI",
    }

    return SampleBatch(
        case_id=str(case_id),
        charge_radius=charge,
        # 新原点不含 R_0/D：这里显式记“不适用”，不填任何伪造爆速。
        detonation_velocity=None,
        scales=scales,
        pinn_time_min=0.0,
        pinn_time_max=0.0,
        xi_ic_product=xi_column,
        rho_ic_product=rho_p,
        u_ic_product=u_p,
        e_ic_product=e_p,
        eta_ic_air=eta_column,
        rho_ic_air=rho_a,
        u_ic_air=u_a,
        e_ic_air=e_a,
        contact_ic=contact,
        shock_ic=shock,
        diagnostics=diagnostics,
    )


def initial_radii_from_batch(batch: SampleBatch) -> tuple[np.ndarray, np.ndarray]:
    """Invert the adapter's dimensionless coordinates back to physical radii.

    This is the documented inverse of the one conversion performed by
    :func:`build_planar_startup_initial_batch`; it does not touch densities,
    velocities or specific energies.
    """
    if not isinstance(batch, SampleBatch):
        raise PlanarStartupBatchError("batch must be a SampleBatch")
    required = (
        "xi_ic_product",
        "eta_ic_air",
    )
    missing = [name for name in required if getattr(batch, name, None) is None]
    if missing:
        raise PlanarStartupBatchError(
            f"batch is missing initial-condition field(s): {missing}"
        )
    contact = float(batch.contact_ic)
    shock = float(batch.shock_ic)
    if not (math.isfinite(contact) and math.isfinite(shock) and shock > contact > 0.0):
        raise PlanarStartupBatchError(
            f"batch has invalid initial fronts: contact={contact!r}, shock={shock!r}"
        )
    xi = batch.xi_ic_product.detach().reshape(-1).cpu().numpy()
    eta = batch.eta_ic_air.detach().reshape(-1).cpu().numpy()
    return contact * xi, contact + (shock - contact) * eta


__all__ = [
    "PLANAR_STARTUP_IC_ADAPTER_VERSION",
    "PlanarStartupBatchError",
    "build_planar_startup_initial_batch",
    "initial_radii_from_batch",
]
