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

# Physical column widths in inches (generic defaults).
SINGLE_COLUMN = 3.4
DOUBLE_COLUMN = 7.0

# Per-journal figure specifications. Widths are in inches (1 mm = 1/25.4 in).
# ``panel_case`` controls how panel tags are rendered: "upper" -> "A", "B"
# (Cell Press style), "lower_paren" -> "(a)", "(b)".
JOURNAL_PRESETS = {
    # Biophysical Journal (Cell Press / Elsevier): Arial, uppercase bold panel
    # letters, columns 85 / 114 / 174 mm, min line ~0.5 pt, >=300 dpi raster.
    "biophysical_journal": {
        # Arial is the BJ requirement; Liberation Sans is metric-identical to
        # Arial and is the automatic fallback on Linux systems without Arial.
        "sans_serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "family": "sans-serif",
        "font_size": 7,
        "single_column": 85 / 25.4,     # ~3.35 in
        "onehalf_column": 114 / 25.4,   # ~4.49 in
        "double_column": 174 / 25.4,    # ~6.85 in
        "panel_case": "upper",
        "savefig_dpi": 500,
    },
    # A neutral generic preset (the previous default behaviour).
    "generic": {
        "sans_serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "family": "sans-serif",
        "font_size": 8,
        "single_column": SINGLE_COLUMN,
        "onehalf_column": 5.0,
        "double_column": DOUBLE_COLUMN,
        "panel_case": "lower_paren",
        "savefig_dpi": 600,
    },
}

# Active preset state, updated by set_publication_style().
_ACTIVE = dict(JOURNAL_PRESETS["generic"])


def journal_widths(journal: str = "biophysical_journal") -> dict:
    """Return the ``single/onehalf/double`` column widths (inches) for a
    journal preset, e.g. for sizing a figure explicitly."""
    p = JOURNAL_PRESETS[journal]
    return {
        "single": p["single_column"],
        "onehalf": p["onehalf_column"],
        "double": p["double_column"],
    }


def set_publication_style(font_size: Optional[int] = None,
                          family: Optional[str] = None,
                          journal: str = "biophysical_journal") -> None:
    """Apply a journal-specific matplotlib style (idempotent).

    Parameters
    ----------
    journal:
        Preset key from :data:`JOURNAL_PRESETS` (default
        ``"biophysical_journal"``: Arial, 7 pt, uppercase panel letters).
    font_size, family:
        Optional overrides of the preset's base font size and family.

    The chosen preset also updates the module-level column-width constants and
    the panel-label convention used by :func:`panel_label`.
    """
    import matplotlib as mpl

    global _ACTIVE, SINGLE_COLUMN, DOUBLE_COLUMN
    preset = dict(JOURNAL_PRESETS.get(journal, JOURNAL_PRESETS["generic"]))
    _ACTIVE = preset
    if font_size is None:
        font_size = preset["font_size"]
    if family is None:
        family = preset["family"]

    # Expose the preset widths through the module-level constants.
    SINGLE_COLUMN = preset["single_column"]
    DOUBLE_COLUMN = preset["double_column"]

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": preset["savefig_dpi"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,        # embed TrueType (editable text, not paths)
        "ps.fonttype": 42,
        "font.family": family,
        "font.sans-serif": preset["sans_serif"],
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


def panel_label(ax, index_or_letter, x: float = -0.02, y: float = 1.03) -> None:
    """Add a bold panel label to ``ax`` in the active journal's convention.

    ``index_or_letter`` may be an integer (0 -> first panel) or an explicit
    letter. Cell Press journals (e.g. Biophysical Journal) use uppercase
    letters ``A, B, ...``; the generic preset uses ``(a), (b), ...``.
    """
    if isinstance(index_or_letter, int):
        letter = chr(ord("a") + index_or_letter)
    else:
        letter = str(index_or_letter)
    if _ACTIVE.get("panel_case") == "upper":
        text = letter.upper()
    else:
        text = f"({letter.lower()})"
    ax.text(x, y, text, transform=ax.transAxes, fontweight="bold",
            va="bottom", ha="right")


def _new_fig(width: float = SINGLE_COLUMN, height: Optional[float] = None,
             ncols: int = 1, nrows: int = 1):
    import matplotlib.pyplot as plt
    if height is None:
        height = width * 0.78 * nrows / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height),
                             squeeze=False)
    return fig, axes


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


def plot_model_comparison(cfg, models=None, fem_curve=None,
                          width: float = SINGLE_COLUMN, ax=None,
                          delta=None):
    """Overlay several analytic contact models (and optionally an FEM curve).

    ``models`` defaults to :data:`nanoindent.analytics.BEC_MODELS`
    (Hertz, Garcia & Garcia, Dimitriadis 2002, Chadwick 2002). ``cfg`` supplies
    R / E / nu / h. Pass ``fem_curve`` (a ForceCurve) to show the simulation
    alongside the closed-form models.
    """
    if ax is None:
        fig, axes = _new_fig(width)
        ax = axes[0, 0]
    else:
        fig = ax.figure

    reg = analytics.model_registry()
    if models is None:
        models = analytics.BEC_MODELS
    if delta is None:
        dmax = getattr(cfg, "delta_max", 1.0)
        delta = np.linspace(0.0, dmax, 200)

    for i, key in enumerate(models):
        if key not in reg:
            continue
        style = ":" if key == "hertz" else "-"
        color = OKABE_ITO["black"] if key == "hertz" else _CYCLE[i % len(_CYCLE)]
        ax.plot(delta, reg[key].func(delta, cfg), style, color=color,
                lw=1.3, label=reg[key].label)

    if fem_curve is not None:
        ax.plot(fem_curve.delta, fem_curve.force, "o", color=OKABE_ITO["vermillion"],
                ms=3.5, label="FEM", zorder=6)

    ax.set_xlabel(r"indentation $\delta$ (nm)")
    ax.set_ylabel(r"force $F$ (pN)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    return fig


_FIELD_LABELS = {
    "von_mises": "von Mises stress (MPa)",
    "sigma_zz": r"$\sigma_{zz}$ (MPa)",
    "pressure": "pressure (MPa)",
    "displacement": "|displacement| (nm)",
    "uz": r"$u_z$ (nm)",
    "ur": r"$u_r$ (nm)",
}


def plot_field(solver, field: str = "von_mises", step_delta=None,
               mirror: bool = True, deformed: bool = False,
               disp_scale: float = 1.0, cmap: str = "viridis",
               levels: int = 24, width: float = SINGLE_COLUMN, ax=None,
               show_mesh: bool = False, show_tip: bool = True):
    """Heatmap of an FEM field over the axisymmetric (r, z) cross-section.

    Parameters
    ----------
    solver:
        An :class:`~nanoindent.IndentationSolver` that has been ``run()``.
    field:
        Field name understood by ``solver.nodal_field`` (``"von_mises"``,
        ``"sigma_zz"``, ``"pressure"``, ``"displacement"``, ``"uz"``, ``"ur"``).
    mirror:
        Mirror the half-domain across the axis ``r = 0`` to show the full
        cross-section (axisymmetric).
    deformed:
        Warp the mesh by the displacement (times ``disp_scale``).
    show_tip:
        Overlay the rigid indenter outline at the final indentation.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    if ax is None:
        fig, axes = _new_fig(width, width * 0.7)
        ax = axes[0, 0]
    else:
        fig = ax.figure

    mesh = solver.domain.mesh
    vals = solver.nodal_field(field)
    r = mesh.p[0].copy()
    z = mesh.p[1].copy()
    if deformed:
        ur, uz = solver.nodal_displacement()
        r = r + disp_scale * ur
        z = z + disp_scale * uz
    tris = mesh.t.T

    def _draw(rr, vv):
        triang = mtri.Triangulation(rr, z, tris)
        tcf = ax.tricontourf(triang, vv, levels=levels, cmap=cmap)
        if show_mesh:
            ax.triplot(triang, color="k", lw=0.15, alpha=0.3)
        return tcf

    tcf = _draw(r, vals)
    if mirror:
        _draw(-r, vals)

    # Rigid tip outline at the final indentation depth.
    if show_tip and getattr(solver, "_last_u", None) is not None:
        cfg = solver.cfg
        delta = cfg.delta_max if step_delta is None else step_delta
        rr = np.linspace(-cfg.L * 0.6, cfg.L * 0.6, 400)
        ztip = solver.tip.surface_z(np.abs(rr), delta)
        finite = np.isfinite(ztip)
        ax.plot(rr[finite], ztip[finite], color="crimson", lw=1.2, zorder=5)

    cbar = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(_FIELD_LABELS.get(field, field), fontsize=8)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_aspect("equal")
    ax.set_title(f"{field.replace('_', ' ')}"
                 + (f"  ($\\delta$={solver.cfg.delta_max:g} nm, "
                    f"h={solver.cfg.h:g} nm)"), fontsize=8)
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
