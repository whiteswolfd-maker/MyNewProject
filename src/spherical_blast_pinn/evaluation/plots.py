"""Figures for the evaluation report.

Every function here takes the data it draws and returns the
:class:`matplotlib.figure.Figure`; saving is controlled by an optional ``path``.
Nothing in this module reads a configuration file, invents a file name, or
knows the layout of an output directory — the caller owns paths, so the same
figure can go into a run directory, a test's ``tmp_path``, or a notebook.

The palette is Okabe–Ito, which stays distinguishable under the common forms of
colour-vision deficiency and, just as importantly for a report that mixes line
plots with field maps, is also readable in greyscale print.  The same eight
colours are used everywhere, and each physical meaning keeps its own colour
across figures: truth is always the neutral grey, the model is always the blue,
and a baseline is always the orange.  A reader should not have to re-learn the
legend between two panels of the same report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

#: Okabe–Ito colour-blind-safe palette.  Names describe the *role* they play in
#: this report, so a call site reads ``MODEL`` rather than an index.
PALETTE: dict[str, str] = {
    "truth": "#000000",       # black
    "model": "#0072B2",       # blue
    "baseline": "#D55E00",    # vermillion
    "product": "#CC79A7",     # reddish purple
    "air": "#009E73",         # bluish green
    "accent": "#E69F00",      # orange
    "secondary": "#56B4E9",   # sky blue
    "muted": "#999999",       # grey
    "warning": "#F0E442",     # yellow
}

#: Marker/linestyle cycle so a figure that is printed in greyscale still
#: separates its series by shape as well as by colour.
LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")


def _finish(figure: Figure, path: Path | str | None, *, dpi: int = 140) -> Figure:
    """Apply the house defaults, save when asked, and return the figure.

    ``bbox_inches="tight"`` is not cosmetic here: field maps carry colourbar
    labels and long case names, and a clipped axis label in the report is worse
    than no label at all.
    """
    figure.tight_layout()
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=dpi, bbox_inches="tight")
    return figure


def _legend(axis, **kwargs: Any) -> None:
    """Add a legend, but only when the axis actually has labelled artists.

    ``legend()`` on an axis with nothing to label emits a warning and draws an
    empty box.  A panel can legitimately be empty — a caller that did not supply
    a field leaves one of the initial-profile panels blank — and an empty legend
    next to an empty panel is noise, not information.
    """
    handles, _labels = axis.get_legend_handles_labels()
    if not handles:
        return
    kwargs.setdefault("fontsize", 8)
    axis.legend(**kwargs)


def _as_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


# --------------------------------------------------------------------------
# 1. Sphericity audit
# --------------------------------------------------------------------------


def plot_sphericity_audit(
    times,
    dispersion: Mapping[str, Any],
    *,
    path: Path | str | None = None,
    threshold: float | None = 0.02,
    title: str = "Sphericity audit: angular dispersion of the shell-averaged fields",
) -> Figure:
    """Angular dispersion of each field against time.

    The audit is the evidence for the assumption the whole pipeline rests on: a
    3D ALE result may only be reduced to 1D radial profiles if the fields vary
    with radius alone.  Plotted against *time* because the assumption is most
    likely to fail early, while the interface band still straddles several
    shells and before the flow has spherised.
    """
    times = _as_1d(times)
    figure, axes = plt.subplots(figsize=(8.0, 4.6))
    colours = (PALETTE["model"], PALETTE["product"], PALETTE["air"], PALETTE["accent"])
    for index, (name, values) in enumerate(dispersion.items()):
        values = _as_1d(values)
        if values.size != times.size:
            raise ValueError(
                f"dispersion series {name!r} has {values.size} samples for {times.size} times"
            )
        axes.plot(
            times,
            values,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            color=colours[index % len(colours)],
            label=name,
        )
    if threshold is not None:
        axes.axhline(
            threshold,
            color=PALETTE["baseline"],
            linewidth=1.2,
            linestyle="--",
            label=f"acceptance threshold {threshold:g}",
        )
    axes.set_xlabel("tau since initial frame [s]")
    axes.set_ylabel("p95 relative dispersion [-]")
    axes.set_title(title)
    axes.grid(alpha=0.25)
    _legend(axes)
    return _finish(figure, path)


# --------------------------------------------------------------------------
# 2. Initial profiles
# --------------------------------------------------------------------------


def plot_initial_profiles(
    radius,
    product: Mapping[str, Any],
    air: Mapping[str, Any],
    *,
    path: Path | str | None = None,
    tau: float | None = None,
    fields: Sequence[str] = ("density", "pressure", "radial_velocity", "specific_internal_energy"),
    title: str = "Initial (t_eps) profiles: product and air",
) -> Figure:
    """The initial condition the PINN starts from, one panel per field.

    Drawn as the reference *and* as whatever the model produces at the same
    time when a model series is supplied, because the initial condition is
    imposed by a loss and is therefore the first thing that can silently be
    wrong.
    """
    radius = _as_1d(radius)
    fields = tuple(fields)
    figure, axes_grid = plt.subplots(1, len(fields), figsize=(3.4 * len(fields), 3.6))
    axes_list = np.atleast_1d(axes_grid).reshape(-1)
    for axis, name in zip(axes_list, fields):
        for medium, values, colour in (
            ("product", product.get(name), PALETTE["product"]),
            ("air", air.get(name), PALETTE["air"]),
        ):
            if values is None:
                continue
            values = _as_1d(values)
            if values.size != radius.size:
                raise ValueError(
                    f"{medium}/{name} has {values.size} samples for {radius.size} radii"
                )
            axis.plot(radius, values, color=colour, linewidth=1.6, label=medium)
        axis.set_xlabel("radius [m]")
        axis.set_ylabel(name)
        axis.set_title(name)
        axis.grid(alpha=0.25)
        _legend(axis)
    suffix = f" (tau = {tau:g} s)" if tau is not None else ""
    figure.suptitle(title + suffix, fontsize=11)
    return _finish(figure, path)


# --------------------------------------------------------------------------
# 3. Front trajectories
# --------------------------------------------------------------------------


def plot_front_trajectories(
    tau,
    truth: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    path: Path | str | None = None,
    title: str = "Contact and shock trajectories",
) -> Figure:
    """``R_c(tau)`` and ``R_s(tau)``: reference against prediction.

    The lower panel shows the *relative* error on a log axis — a relative error
    is the quantity the acceptance targets are written in, and a shared linear
    axis would make a 1 % error invisible next to a 0.1 m radius.
    """
    tau = _as_1d(tau)
    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(8.0, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    for name, colour in (("contact", PALETTE["model"]), ("shock", PALETTE["baseline"])):
        if name not in truth or name not in prediction:
            continue
        reference_values = _as_1d(truth[name])
        predicted_values = _as_1d(prediction[name])
        if reference_values.size != tau.size or predicted_values.size != tau.size:
            raise ValueError(
                f"{name} trajectories must have {tau.size} samples, got "
                f"{reference_values.size} reference / {predicted_values.size} prediction"
            )
        upper.plot(
            tau, reference_values, color=colour, linewidth=2.0, label=f"{name} reference"
        )
        upper.plot(
            tau,
            predicted_values,
            color=colour,
            linewidth=1.4,
            linestyle="--",
            label=f"{name} prediction",
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            relative = np.abs(predicted_values - reference_values) / reference_values
        lower.semilogy(tau, np.where(relative > 0.0, relative, np.nan), color=colour, linewidth=1.4,
                       label=f"{name} relative error")
    lower.set_xlabel("tau since initial frame [s]")
    lower.set_ylabel("relative error [-]")
    upper.set_ylabel("front radius [m]")
    upper.set_title(title)
    for axis in (upper, lower):
        axis.grid(alpha=0.25, which="both")
        _legend(axis)
    return _finish(figure, path)


# --------------------------------------------------------------------------
# 4-6. Space-time fields
# --------------------------------------------------------------------------


def plot_space_time_field(
    tau,
    radius,
    values,
    *,
    field_name: str,
    case_label: str = "",
    split_label: str = "",
    path: Path | str | None = None,
    reference=None,
    log_scale: bool = True,
    contact_radius=None,
    shock_radius=None,
    title: str | None = None,
) -> Figure:
    """One field on the space-time ``(tau, r)`` plane, with a difference panel.

    Parameterised rather than duplicated: the training, interpolation and
    extrapolation figures are the same picture of a different case, and the only
    thing that must differ between them is the label.  Keeping one function
    guarantees that a change to the colour limits cannot be applied to one size
    and forgotten for another, which is exactly how a favourable-looking figure
    gets into a report.

    ``reference``, when given, adds a model-minus-reference panel.  The
    difference is drawn with a diverging colour map centred at zero so an
    over-prediction and an under-prediction are never confused for each other.
    """
    tau = _as_1d(tau)
    radius = _as_1d(radius)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (tau.size, radius.size):
        raise ValueError(
            f"field {field_name!r} has shape {values.shape}, expected "
            f"({tau.size}, {radius.size}) from tau x radius"
        )

    panels = 1 if reference is None else 2
    figure, axes_grid = plt.subplots(
        1, panels, figsize=(6.2 * panels, 4.8), squeeze=False
    )
    axes_list = axes_grid.reshape(-1)
    axis = axes_list[0]
    mesh = values.T
    if log_scale and np.all(mesh[np.isfinite(mesh)] > 0.0):
        image = axis.pcolormesh(tau, radius, mesh, shading="auto", norm=LogNorm(),
                                cmap="viridis")
    else:
        image = axis.pcolormesh(tau, radius, mesh, shading="auto", cmap="viridis")
    figure.colorbar(image, ax=axis, label=field_name)
    if contact_radius is not None:
        axis.plot(tau, _as_1d(contact_radius), color=PALETTE["warning"], linewidth=1.6,
                  label="R_c")
    if shock_radius is not None:
        axis.plot(tau, _as_1d(shock_radius), color=PALETTE["baseline"], linewidth=1.6,
                  label="R_s")
    if contact_radius is not None or shock_radius is not None:
        _legend(axis)
    axis.set_xlabel("tau since initial frame [s]")
    axis.set_ylabel("radius [m]")
    axis.set_title(title or f"{field_name} — {case_label or 'case'} {split_label}".strip())

    if reference is not None:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.shape != values.shape:
            raise ValueError(
                f"reference field shape {reference.shape} does not match the model {values.shape}"
            )
        difference = values - reference
        finite = difference[np.isfinite(difference)]
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
        if limit <= 0.0:
            limit = 1.0
        second = axes_list[1]
        image = second.pcolormesh(
            tau,
            radius,
            difference.T,
            shading="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        figure.colorbar(image, ax=second, label=f"model - reference [{field_name}]")
        second.set_xlabel("tau since initial frame [s]")
        second.set_ylabel("radius [m]")
        second.set_title("difference")
    return _finish(figure, path)


def plot_space_time_training(tau, radius, values, *, path=None, **kwargs) -> Figure:
    """Space-time field for a **training** charge size."""
    return _split_field(tau, radius, values, "training", path, kwargs)


def plot_space_time_interpolation(tau, radius, values, *, path=None, **kwargs) -> Figure:
    """Space-time field for an **interpolation** size (inside the trained range)."""
    return _split_field(tau, radius, values, "interpolation", path, kwargs)


def plot_space_time_extrapolation(tau, radius, values, *, path=None, **kwargs) -> Figure:
    """Space-time field for an **extrapolation** size (outside the trained range).

    The figure that matters most: a model can interpolate perfectly between the
    sizes it saw while having learned nothing that transfers to a new one.
    """
    return _split_field(tau, radius, values, "extrapolation", path, kwargs)


def _split_field(tau, radius, values, split_label: str, path, kwargs) -> Figure:
    kwargs = dict(kwargs)
    kwargs.setdefault("split_label", f"[{split_label}]")
    return plot_space_time_field(tau, radius, values, path=path, **kwargs)


# --------------------------------------------------------------------------
# 7-8. Gauge waveforms
# --------------------------------------------------------------------------


def plot_gauge_waveforms(
    times,
    waveforms: Mapping[str, Mapping[str, Any]],
    *,
    path: Path | str | None = None,
    ambient_pressure_pa: float | None = None,
    title: str = "Gauge pressure histories (positive and negative phase)",
) -> Figure:
    """Complete pressure histories, one panel per gauge.

    Each panel draws the reference and the model over the *whole* window, from
    the arrival through the negative phase to full recovery.  Plotting only the
    positive phase is the standard way a waveform comparison is made to look
    better than it is: the negative phase is where a single-medium or
    phase-switched model falls apart.
    """
    times = _as_1d(times)
    figure, axes_grid = plt.subplots(
        len(waveforms), 1, figsize=(8.0, 2.5 * max(1, len(waveforms))), sharex=True, squeeze=False
    )
    axes_list = axes_grid.reshape(-1)
    for axis, (label, series) in zip(axes_list, waveforms.items()):
        for name, colour, style in (
            ("reference", PALETTE["truth"], "-"),
            ("model", PALETTE["model"], "--"),
        ):
            values = series.get(name)
            if values is None:
                continue
            values = _as_1d(values)
            if values.size != times.size:
                raise ValueError(
                    f"{label}/{name} has {values.size} samples for {times.size} times"
                )
            axis.plot(times, values, color=colour, linestyle=style, linewidth=1.5, label=name)
        if ambient_pressure_pa is not None:
            axis.axhline(
                ambient_pressure_pa, color=PALETTE["muted"], linewidth=1.0, linestyle=":", label="ambient"
            )
        axis.axhline(0.0, color=PALETTE["muted"], linewidth=0.8)
        axis.set_ylabel("pressure [Pa]")
        axis.set_title(label, fontsize=9)
        axis.grid(alpha=0.25)
        _legend(axis, loc="upper right")
    axes_list[-1].set_xlabel("tau since initial frame [s]")
    figure.suptitle(title, fontsize=11)
    return _finish(figure, path)


def plot_overpressure_phases(
    times,
    overpressure,
    *,
    path: Path | str | None = None,
    reference=None,
    label: str = "gauge",
    title: str = "Overpressure: positive and negative phase",
) -> Figure:
    """Overpressure with the two phases shaded and their areas annotated.

    The shaded areas *are* the impulses, so the figure and the reported impulse
    numbers cannot disagree — which is the failure mode a reader would have no
    way to detect from the numbers alone.
    """
    times = _as_1d(times)
    overpressure = _as_1d(overpressure)
    if overpressure.size != times.size:
        raise ValueError(
            f"overpressure has {overpressure.size} samples for {times.size} times"
        )
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    axis.axhline(0.0, color=PALETTE["muted"], linewidth=0.9)
    axis.plot(times, overpressure, color=PALETTE["model"], linewidth=1.7, label=f"{label} (model)")
    axis.fill_between(
        times, overpressure, 0.0, where=overpressure >= 0.0,
        color=PALETTE["baseline"], alpha=0.25, label="positive phase (area = I+)",
        interpolate=True,
    )
    axis.fill_between(
        times, overpressure, 0.0, where=overpressure < 0.0,
        color=PALETTE["secondary"], alpha=0.35, label="negative phase (area = I-)",
        interpolate=True,
    )
    if reference is not None:
        reference = _as_1d(reference)
        if reference.size != times.size:
            raise ValueError("reference overpressure length does not match the time grid")
        axis.plot(times, reference, color=PALETTE["truth"], linewidth=1.3, linestyle="--",
                  label=f"{label} (reference)")
    axis.set_xlabel("tau since initial frame [s]")
    axis.set_ylabel("overpressure above ambient [Pa]")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    _legend(axis)
    return _finish(figure, path)


# --------------------------------------------------------------------------
# 9. Drift
# --------------------------------------------------------------------------


def plot_drift(
    times,
    mass_drift,
    energy_drift,
    *,
    path: Path | str | None = None,
    mass_target: float | None = None,
    energy_target: float | None = None,
    title: str = "Conservation drift",
) -> Figure:
    """Relative mass and excess-energy drift, with the acceptance targets drawn.

    Drift is a *cumulative* diagnostic: a small per-step error that is biased in
    one direction grows without bound, which a residual plot sampled at a few
    times would happily hide.
    """
    times = _as_1d(times)
    mass_drift = _as_1d(mass_drift)
    energy_drift = _as_1d(energy_drift)
    if mass_drift.size != times.size or energy_drift.size != times.size:
        raise ValueError("drift series must share the time grid")
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    axis.plot(times, np.abs(mass_drift), color=PALETTE["model"], linewidth=1.6, label="|mass drift|")
    axis.plot(
        times, np.abs(energy_drift), color=PALETTE["baseline"], linewidth=1.6,
        linestyle="--", label="|excess energy drift|",
    )
    if mass_target is not None:
        axis.axhline(mass_target, color=PALETTE["model"], linewidth=1.0, linestyle=":",
                     label=f"mass target {mass_target:g}")
    if energy_target is not None:
        axis.axhline(energy_target, color=PALETTE["baseline"], linewidth=1.0, linestyle=":",
                     label=f"energy target {energy_target:g}")
    if np.any(np.abs(mass_drift) > 0.0) or np.any(np.abs(energy_drift) > 0.0):
        axis.set_yscale("log")
    axis.set_xlabel("tau since initial frame [s]")
    axis.set_ylabel("relative drift [-]")
    axis.set_title(title)
    axis.grid(alpha=0.25, which="both")
    _legend(axis)
    return _finish(figure, path)


# --------------------------------------------------------------------------
# 10-11. Interface conditions
# --------------------------------------------------------------------------


def plot_contact_condition_errors(
    tau,
    errors: Mapping[str, Any],
    *,
    path: Path | str | None = None,
    target: float | None = None,
    title: str = "Contact condition errors",
) -> Figure:
    """Velocity mismatch on both sides, pressure mismatch and flux mismatch.

    The three terms are kept separate.  A single combined contact error cannot
    distinguish "the interface is moving at the wrong speed" from "the two media
    disagree about the pressure there", and those are different bugs.
    """
    tau = _as_1d(tau)
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    colours = (PALETTE["model"], PALETTE["product"], PALETTE["air"], PALETTE["accent"])
    for index, (name, values) in enumerate(errors.items()):
        values = _as_1d(values)
        if values.size != tau.size:
            raise ValueError(f"contact term {name!r} has {values.size} samples for {tau.size} times")
        axis.semilogy(
            tau,
            np.where(np.abs(values) > 0.0, np.abs(values), np.nan),
            color=colours[index % len(colours)],
            linewidth=1.5,
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            label=name,
        )
    if target is not None:
        axis.axhline(target, color=PALETTE["baseline"], linewidth=1.1, linestyle="--",
                     label=f"target {target:g}")
    axis.set_xlabel("tau since initial frame [s]")
    axis.set_ylabel("scaled mismatch [-]")
    axis.set_title(title)
    axis.grid(alpha=0.25, which="both")
    _legend(axis)
    return _finish(figure, path)


def plot_rankine_hugoniot_errors(
    tau,
    mass,
    momentum,
    energy,
    *,
    path: Path | str | None = None,
    target: float | None = None,
    title: str = "Rankine-Hugoniot component errors",
) -> Figure:
    """The three RH residuals, normalised by their own flux scales.

    Mass, momentum and energy are on one axis because each is already divided by
    the flux scale of its own equation: the comparison is therefore of
    *dimensionless* residuals, and a single large component is immediately
    visible as the equation the shock is violating.
    """
    tau = _as_1d(tau)
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    for values, name, colour in (
        (mass, "mass", PALETTE["model"]),
        (momentum, "momentum", PALETTE["baseline"]),
        (energy, "energy", PALETTE["air"]),
    ):
        values = _as_1d(values)
        if values.size != tau.size:
            raise ValueError(f"RH {name} residual has {values.size} samples for {tau.size} times")
        axis.semilogy(
            tau,
            np.where(np.abs(values) > 0.0, np.abs(values), np.nan),
            color=colour,
            linewidth=1.6,
            label=name,
        )
    if target is not None:
        axis.axhline(target, color=PALETTE["muted"], linewidth=1.1, linestyle="--",
                     label=f"target {target:g}")
    axis.set_xlabel("tau since initial frame [s]")
    axis.set_ylabel("normalised residual [-]")
    axis.set_title(title)
    axis.grid(alpha=0.25, which="both")
    _legend(axis)
    return _finish(figure, path)


# --------------------------------------------------------------------------
# 12-13. Comparisons across runs
# --------------------------------------------------------------------------


def plot_ablation_comparison(
    labels: Sequence[str],
    values: Mapping[str, Sequence[float]],
    *,
    path: Path | str | None = None,
    ylabel: str = "metric value [-]",
    title: str = "Baseline and ablation comparison",
    target: float | None = None,
    log_scale: bool = False,
    higher_is_better: bool = False,
) -> Figure:
    """Grouped bar chart of one metric across variants.

    ``values`` is ``{series_name: [value per label]}`` so several metrics (for
    example the shock radius error and the contact radius error) can share one
    figure without a second legend.  Bars, not lines: the x axis is categorical
    and there is no ordering to interpolate along.
    """
    labels = [str(label) for label in labels]
    series = list(values.items())
    if not series:
        raise ValueError("plot_ablation_comparison needs at least one series")
    for name, entries in series:
        if len(entries) != len(labels):
            raise ValueError(
                f"series {name!r} has {len(entries)} values for {len(labels)} labels"
            )

    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.8 / len(series)
    colours = (PALETTE["model"], PALETTE["baseline"], PALETTE["air"], PALETTE["product"],
               PALETTE["accent"])
    figure, axis = plt.subplots(figsize=(max(7.0, 1.3 * len(labels) + 3.0), 4.6))
    for index, (name, entries) in enumerate(series):
        offset = (index - (len(series) - 1) / 2.0) * width
        axis.bar(
            positions + offset,
            [float(value) if np.isfinite(value) else np.nan for value in entries],
            width=width,
            color=colours[index % len(colours)],
            label=name,
            edgecolor="white",
            linewidth=0.6,
        )
    if target is not None:
        axis.axhline(target, color=PALETTE["truth"], linewidth=1.2, linestyle="--",
                     label=f"target {target:g}")
    if log_scale:
        axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel(ylabel + (" (lower is better)" if not higher_is_better else " (higher is better)"))
    axis.set_title(title)
    axis.grid(alpha=0.25, axis="y")
    _legend(axis)
    return _finish(figure, path)


def plot_extrapolation_curve(
    factors: Sequence[float],
    errors: Mapping[str, Sequence[float]],
    *,
    path: Path | str | None = None,
    target: float | None = None,
    xlabel: str = "charge-size factor relative to the largest training size [-]",
    ylabel: str = "relative error [-]",
    title: str = "Extrapolation: error against charge-size factor",
) -> Figure:
    """Error against how far outside the training range a case lies.

    The only curve that separates *learning the physics* from *interpolating the
    training set*: a model that has learned the physics degrades gradually, and
    one that has learned the similarity law or memorised the sizes shows a knee
    exactly at the edge of the trained range.
    """
    factors = _as_1d(factors)
    if factors.size == 0:
        raise ValueError("plot_extrapolation_curve needs at least one factor")
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    colours = (PALETTE["model"], PALETTE["baseline"], PALETTE["air"], PALETTE["product"])
    for index, (name, entries) in enumerate(errors.items()):
        entries = _as_1d(entries)
        if entries.size != factors.size:
            raise ValueError(
                f"series {name!r} has {entries.size} values for {factors.size} factors"
            )
        axis.plot(
            factors,
            entries,
            marker="o",
            markersize=5,
            linewidth=1.6,
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            color=colours[index % len(colours)],
            label=name,
        )
    axis.axvline(1.0, color=PALETTE["muted"], linewidth=1.1, linestyle=":", label="edge of training range")
    if target is not None:
        axis.axhline(target, color=PALETTE["truth"], linewidth=1.1, linestyle="--",
                     label=f"target {target:g}")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    _legend(axis)
    return _finish(figure, path)


__all__ = [
    "LINESTYLES",
    "PALETTE",
    "plot_ablation_comparison",
    "plot_contact_condition_errors",
    "plot_drift",
    "plot_extrapolation_curve",
    "plot_front_trajectories",
    "plot_gauge_waveforms",
    "plot_initial_profiles",
    "plot_overpressure_phases",
    "plot_rankine_hugoniot_errors",
    "plot_space_time_extrapolation",
    "plot_space_time_field",
    "plot_space_time_interpolation",
    "plot_space_time_training",
    "plot_sphericity_audit",
]
