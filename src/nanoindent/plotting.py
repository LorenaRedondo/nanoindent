"""Publication-quality figure generation for nanoindentation results.

This module produces figures suitable for a scientific journal: vector output
(PDF / SVG / EPS) plus high-resolution raster, a colour-blind-safe palette
(Okabe-Ito), correct single-/double-column physical widths, minimal chartjunk,
and consistent typography.

Design choices (all overridable):

* **Palette**: Okabe-Ito, safe for the common colour-vision deficiencies and
  legible in greyscale.
* **Widths**: single column = 3.4 in (~86 mm), double column = 7.0 in
  (~178 mm) — the standard for most physics/soft-matter journals.
* **Format**: vector **PDF** is the default deliverable (journals want scalable
  line art); a 600-dpi **PNG** is written alongside for quick preview.

Typical use::

    from nanoindent import plotting
    plotting.set_publication_style()
    fig = plotting.plot_force_curves({7.0: curve7, 14.0: curve14},
                                     compare=["hertz", "dimitriadis_bonded"])
    plotting.save_figure(fig, "figure2", formats=("pdf", "png"))
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple
import numpy as np

from . import analytics


# Okabe-Ito colour-blind-safe palette.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
_CYCLE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# Physical column widths in inches.
SINGLE_COLUMN = 3.4
DOUBLE_COLUMN = 7.0


def set_publication_style(font_size: int = 8, family: str = "sans-serif") -> None:
    """Apply a journal-friendly matplotlib style (idempotent).

    Small base font (8 pt) matches a figure printed at column width; line
    widths and tick sizes are scaled to look correct at that size.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,        # embed TrueType (editable text, not paths)
        "ps.fonttype": 42,
        "font.family": family,
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size - 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=_CYCLE),
    })


def _new_fig(width: float = SINGLE_COLUMN, height: Optional[float] = None,
             ncols: int = 1, nrows: int = 1):
    import matplotlib.pyplot as plt
    if height is None:
        height = width * 0.78 * nrows / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height),
                             squeeze=False)
    return fig, axes


def _panel_label(ax, label: str) -> None:
    ax.text(-0.02, 1.02, label, transform=ax.transAxes,
            fontweight="bold", va="bottom", ha="right")


def plot_force_curves(
    curves: Mapping[float, "ForceCurve"],
    compare: Optional[Sequence[str]] = ("hertz",),
    width: float = SINGLE_COLUMN,
    ax=None,
    colors: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
):
    """Overlay FEM force-indentation curves (keyed by film thickness) with
    optional analytic models.

    ``curves`` maps thickness (nm) -> :class:`ForceCurve`. Each FEM curve is
    drawn with markers; the requested analytic models are drawn as smooth
    dashed lines in the matching colour.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, axes = _new_fig(width)
        ax = axes[0, 0]
    else:
        fig = ax.figure

    colors = list(colors or _CYCLE)
    reg = analytics.model_registry()
    thicknesses = sorted(curves.keys())

    for i, h in enumerate(thicknesses):
        curve = curves[h]
        c = colors[i % len(colors)]
        ax.plot(curve.delta, curve.force, "o", color=c, ms=3.5,
                label=f"FEM $h={h:g}$ nm", zorder=5)
        if compare and curve.cfg is not None:
            dd = np.linspace(0, float(curve.delta.max()), 200)
            for key in compare:
                if key not in reg:
                    continue
                is_hertz = key == "hertz"
                # Hertz is a half-space reference: draw it once, in black.
                if is_hertz:
                    line_color, ls, alpha = OKABE_ITO["black"], ":", 0.6
                    lab = reg[key].label if i == len(thicknesses) - 1 else None
                else:
                    line_color, ls, alpha = c, "--", 0.9
                    lab = f"{reg[key].label} ($h={h:g}$)"
                ax.plot(dd, reg[key].func(dd, curve.cfg), ls,
                        color=line_color, lw=1.0, alpha=alpha, label=lab)

    ax.set_xlabel(r"indentation $\delta$ (nm)")
    ax.set_ylabel(r"force $F$ (pN)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    if title:
        ax.set_title(title)
    return fig


def plot_stiffening_ratio(
    curves: Mapping[float, "ForceCurve"],
    compare: Optional[Sequence[str]] = ("dimitriadis_bonded",),
    width: float = SINGLE_COLUMN,
    ax=None,
    colors: Optional[Sequence[str]] = None,
):
    """Plot the finite-thickness stiffening ratio ``F/F_Hertz`` vs. delta for
    each thickness, with an optional analytic-model ratio overlay."""
    if ax is None:
        fig, axes = _new_fig(width)
        ax = axes[0, 0]
    else:
        fig = ax.figure

    colors = list(colors or _CYCLE)
    reg = analytics.model_registry()
    for i, h in enumerate(sorted(curves.keys())):
        curve = curves[h]
        c = colors[i % len(colors)]
        ratio = curve.stiffening_ratio()
        m = np.isfinite(ratio)
        ax.plot(curve.delta[m], ratio[m], "o-", color=c, ms=3.5,
                label=f"FEM $h={h:g}$ nm")
        if compare and curve.cfg is not None:
            dd = np.linspace(max(curve.delta[m].min(), 1e-3),
                             float(curve.delta.max()), 200)
            fh = analytics.hertz_force(dd, curve.cfg.R, curve.cfg.E, curve.cfg.nu)
            for key in compare:
                if key not in reg:
                    continue
                ax.plot(dd, reg[key].func(dd, curve.cfg) / fh, "--",
                        color=c, lw=1.0, alpha=0.8)

    ax.axhline(1.0, color=OKABE_ITO["black"], lw=0.6, ls=":")
    ax.set_xlabel(r"indentation $\delta$ (nm)")
    ax.set_ylabel(r"$F_\mathrm{FEM}/F_\mathrm{Hertz}$")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    return fig


def plot_modulus_recovery(
    results: Sequence[dict],
    width: float = SINGLE_COLUMN,
    ax=None,
    true_E: Optional[float] = None,
):
    """Bar chart of recovered modulus vs. film thickness for each fit model.

    ``results`` is a list of dicts with keys ``h``, ``model``, ``E_fit``
    (as produced by the synthetic-analysis workflow).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, axes = _new_fig(width)
        ax = axes[0, 0]
    else:
        fig = ax.figure

    thicknesses = sorted({r["h"] for r in results})
    models = list(dict.fromkeys(r["model"] for r in results))
    x = np.arange(len(thicknesses))
    n = len(models)
    w = 0.8 / max(n, 1)
    colors = {"hertz": OKABE_ITO["vermillion"],
              "dimitriadis": OKABE_ITO["blue"],
              "dimitriadis_bonded": OKABE_ITO["blue"]}
    for j, model in enumerate(models):
        vals = [next((r["E_fit"] for r in results
                      if r["h"] == h and r["model"] == model), np.nan)
                for h in thicknesses]
        ax.bar(x + (j - (n - 1) / 2) * w, vals, w, label=model,
               color=colors.get(model, _CYCLE[j % len(_CYCLE)]),
               edgecolor="black", linewidth=0.4)

    if true_E is not None:
        ax.axhline(true_E, color=OKABE_ITO["black"], lw=0.8, ls="--",
                   label=f"true $E={true_E:g}$ MPa")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:g}" for h in thicknesses])
    ax.set_xlabel(r"film thickness $h$ (nm)")
    ax.set_ylabel(r"recovered $E$ (MPa)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    return fig


def plot_convergence(records: Sequence["ConvergenceRecord"],
                     width: float = SINGLE_COLUMN, ax=None):
    """Plot a convergence study: force at delta_max vs. the swept parameter."""
    if ax is None:
        fig, axes = _new_fig(width)
        ax = axes[0, 0]
    else:
        fig = ax.figure

    x = [r.param_value for r in records]
    y = [r.force_max for r in records]
    name = records[0].param_name if records else "parameter"
    ax.plot(x, y, "o-", color=OKABE_ITO["blue"])
    if name in ("mesh_size_contact", "penalty_alpha"):
        ax.set_xscale("log")
    ax.set_xlabel(name.replace("_", " "))
    ax.set_ylabel(r"$F(\delta_\mathrm{max})$ (pN)")
    ax.grid(True, alpha=0.25)
    return fig


def save_figure(fig, path_stem: str,
                formats: Sequence[str] = ("pdf", "png")) -> list[str]:
    """Save ``fig`` to ``path_stem`` in each requested format; return the paths.

    ``path_stem`` should have no extension (e.g. ``"figures/fig2"``); one file
    per format is written. Vector formats (pdf/svg/eps) are preferred for
    submission, png for previews.
    """
    fig.tight_layout(pad=0.3)
    written = []
    for fmt in formats:
        p = f"{path_stem}.{fmt}"
        fig.savefig(p, format=fmt)
        written.append(p)
    return written
