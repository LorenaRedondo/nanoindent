"""Supported-lipid-bilayer (SLB) force-map analysis.

Given a grid of AFM approach curves (e.g. from :func:`nanoindent.read_jpk_map`),
this module extracts, per curve and as distributions over the map:

* the **breakthrough (yield) force** — where the tip punctures the bilayer,
* the **bilayer height** — the penetration step at breakthrough,
* the **Young's modulus** fitted to the pre-breakthrough elastic region with the
  Hertz, Dimitriadis 2002, Garcia & Garcia 2018 and Chadwick 2002 models.

A :func:`fem_reference` helper runs the FEM for the fitted geometry so the
experimental modulus distributions can be checked against a full simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import warnings
import numpy as np

from .dataset import _shape_function

# models used for the modulus distributions
MODULUS_MODELS = ["hertz", "dimitriadis", "garcia_bec", "chadwick"]
MODEL_LABELS = {
    "hertz": "Hertz", "dimitriadis": "Dimitriadis 2002",
    "garcia_bec": "Garcia & Garcia 2018", "chadwick": "Chadwick 2002",
}


@dataclass
class CurveResult:
    pixel: int
    breakthrough_force: Optional[float]      # pN
    height: Optional[float]                  # nm — breakthrough step (definition B)
    contact_z: Optional[float]               # nm
    moduli: dict = field(default_factory=dict)   # model -> E (MPa)
    delta: Optional[np.ndarray] = None       # elastic-region indentation (nm)
    force: Optional[np.ndarray] = None       # elastic-region force (pN)
    height_contact_substrate: Optional[float] = None  # nm — contact->substrate (A)
    kappa_A: Optional[float] = None          # mN/m — Connell 2010 area compressibility

    @property
    def height_step(self) -> Optional[float]:
        """Alias for the breakthrough-step height (definition B)."""
        return self.height


@dataclass
class SLBAnalysis:
    results: list                            # list[CurveResult]
    R: float
    nu: float
    tip_shape: str

    # -- distribution accessors -------------------------------------------
    def breakthrough_forces(self) -> np.ndarray:
        return np.array([r.breakthrough_force for r in self.results
                         if r.breakthrough_force is not None]) / 1e3  # nN

    def heights(self) -> np.ndarray:
        """Breakthrough-step heights, definition B (nm)."""
        return np.array([r.height for r in self.results if r.height is not None])

    def heights_contact_substrate(self) -> np.ndarray:
        """Contact-point-to-substrate heights, definition A (nm), from the
        measured-height (capacitive) channel."""
        return np.array([r.height_contact_substrate for r in self.results
                         if r.height_contact_substrate is not None])

    def moduli(self, model: str) -> np.ndarray:
        return np.array([r.moduli.get(model) for r in self.results
                         if r.moduli.get(model) is not None])

    def kappa_A(self) -> np.ndarray:
        """Area compressibility modulus κ_A (mN/m), Connell 2010."""
        return np.array([r.kappa_A for r in self.results if r.kappa_A is not None])

    def summary(self) -> dict:
        def stat(a):
            a = np.asarray(a, float)
            return dict(n=int(a.size), mean=float(np.mean(a)) if a.size else float("nan"),
                        median=float(np.median(a)) if a.size else float("nan"),
                        std=float(np.std(a)) if a.size else float("nan"))
        out = {"breakthrough_force_nN": stat(self.breakthrough_forces()),
               "height_step_nm": stat(self.heights()),
               "height_contact_substrate_nm": stat(self.heights_contact_substrate())}
        for m in MODULUS_MODELS:
            out[f"E_{m}_MPa"] = stat(self.moduli(m))
        out["kappa_A_mN_per_m"] = stat(self.kappa_A())
        return out


# --------------------------------------------------------------------------
# per-curve processing
# --------------------------------------------------------------------------
def process_curve(curve, R: float, nu: float, pixel: int = 0,
                  contact_sigma: float = 8.0, min_drop_pN: float = 400.0,
                  models: Sequence[str] = MODULUS_MODELS,
                  fixed_height: Optional[float] = None,
                  fit_max_delta: Optional[float] = None,
                  film_thickness: Optional[float] = None) -> CurveResult:
    """Analyse a single approach curve: breakthrough, height, and moduli.

    The breakthrough is the largest sudden force drop after contact (the tip
    snapping through the bilayer); the pre-breakthrough elastic branch is fit
    for the modulus with each model.

    Parameters
    ----------
    fit_max_delta:
        Cap the indentation used for the modulus fit (nm). The finite-thickness
        models (Dimitriadis, Garcia, Chadwick) are only valid while the contact
        radius is comparable to the film thickness (delta <~ h); by default the
        fit is capped at the film thickness so all four models are compared on
        the same, physically valid small-indentation region.
    film_thickness:
        Thickness ``h`` (nm) used by the finite-thickness models; defaults to
        the measured breakthrough height of the curve.
    """
    k = curve.spring_constant
    if k is None:
        raise ValueError("curve needs a spring_constant")
    z = curve.z.astype(float)
    d = curve.deflection_nm().astype(float)

    order = np.argsort(-z)                    # far -> near (approach)
    z, d = z[order], d[order]
    nb = max(int(0.3 * len(z)), 5)
    d0 = float(np.median(d[:nb]))
    noise = float(np.std(d[:nb])) or 1e-6
    d_rel = d - d0
    force = k * d_rel * 1e3                   # pN

    # contact point: first sustained rise above the baseline noise
    thr = max(contact_sigma * noise, 0.3)
    above = d_rel > thr
    ci = next((i for i in range(len(above) - 2)
               if above[i] and above[i + 1] and above[i + 2]), None)
    empty = CurveResult(pixel, None, None, None)
    if ci is None or ci >= len(z) - 4:
        return empty

    zc = float(z[ci])
    delta = (zc - z) - d_rel                  # indentation into the sample (nm)

    # breakthrough: most negative force step after contact
    df = np.diff(force)
    seg = df[ci:]
    if seg.size < 2:
        return empty
    bi = ci + int(np.argmin(seg))
    drop = float(-seg[bi - ci])
    if drop < min_drop_pN:
        return empty
    Fb = float(force[bi])
    # (B) penetration step ~ jump in indentation across the snap-through
    height = float(delta[min(bi + 1, len(delta) - 1)] - delta[bi])
    if not (0.3 < height < 15.0):
        height = float(d_rel[bi] - d_rel[min(bi + 1, len(d_rel) - 1)])

    # (A) contact-point -> substrate height, on the measured-height (capacitive)
    # channel: distance from the bilayer-top contact to where the post-
    # breakthrough hard wall extrapolates back to zero force.
    height_cs = None
    mz_all = curve.metadata.get("measured_z_nm") if hasattr(curve, "metadata") else None
    mz = np.asarray(mz_all, float)[order] if mz_all is not None else z
    post = slice(bi + 1, len(mz))
    if len(mz[post]) >= 4 and np.ptp(mz[post]) > 1e-6:
        # linear hard wall: force = a*mz + b  (a<0 as mz decreases on approach)
        a, b = np.polyfit(mz[post], force[post], 1)
        if a < 0:
            mz_sub = -b / a                  # measured height at zero force = substrate
            height_cs = float(mz[ci] - mz_sub)
            if not (0.0 < height_cs < 60.0):
                height_cs = None

    # elastic (pre-breakthrough) branch, sorted by delta
    sl = slice(ci, bi + 1)
    de, fe = delta[sl], force[sl]
    m = de > 0
    de, fe = de[m], fe[m]
    o = np.argsort(de)
    de, fe = de[o], fe[o]

    moduli = {}
    h_fit = film_thickness or fixed_height or max(height, 0.5)
    # Cap the fit region so the finite-thickness models stay valid (delta <~ h).
    cap = fit_max_delta if fit_max_delta is not None else max(h_fit, 1.0)
    fit_mask = de <= cap
    de_f, fe_f = de[fit_mask], fe[fit_mask]
    if de_f.size >= 4:
        for model in models:
            try:
                s = _shape_function(model, R, nu, h_fit)(de_f)
                denom = float(np.dot(s, s))
                if denom > 0:
                    E = float(np.dot(fe_f, s) / denom)
                    if np.isfinite(E) and E > 0:
                        moduli[model] = E
            except Exception:
                pass

    # Connell 2010 area compressibility modulus κ_A (mN/m): fit the fluid-
    # bilayer thinning law F = π κ_A R z0^2/(2d - z0)^2 over z0 < 2d. Uses the
    # contact->substrate thickness (definition A) as 2d when available.
    kappa_A = None
    two_d = height_cs or film_thickness or (2.0 * max(height, 0.5))
    z_mask = de < 0.85 * two_d
    de_c, fe_c = de[z_mask], fe[z_mask]
    if de_c.size >= 4 and two_d > 0:
        s = np.pi * R * de_c ** 2 / (two_d - de_c) ** 2      # force per unit κ_A
        denom = float(np.dot(s, s))
        if denom > 0:
            kA = float(np.dot(fe_c, s) / denom)
            if np.isfinite(kA) and kA > 0:
                kappa_A = kA

    return CurveResult(pixel, Fb, height, zc, moduli, de, fe,
                       height_contact_substrate=height_cs, kappa_A=kappa_A)


def analyze_map(fmap, R: float = 20.0, nu: float = 0.5,
                tip_shape: str = "paraboloid",
                models: Sequence[str] = MODULUS_MODELS,
                film_thickness: Optional[float] = None,
                fit_max_delta: Optional[float] = None,
                max_curves: Optional[int] = None) -> SLBAnalysis:
    """Run :func:`process_curve` over every curve in a force map.

    ``fmap`` is a :class:`~nanoindent.experiment.JPKForceMap` (or any object with
    a ``curves`` list of ForceCurveData carrying a spring constant).

    ``film_thickness`` sets ``h`` for the finite-thickness models (defaults to
    each curve's own breakthrough height); ``fit_max_delta`` caps the fitted
    indentation (defaults to ``h``) so the models are compared on a common,
    physically valid small-indentation window.
    """
    curves = getattr(fmap, "curves", fmap)
    results = []
    for i, curve in enumerate(curves):
        if max_curves and i >= max_curves:
            break
        pix = curve.metadata.get("pixel", i) if hasattr(curve, "metadata") else i
        try:
            results.append(process_curve(
                curve, R, nu, pixel=pix, models=models,
                film_thickness=film_thickness, fit_max_delta=fit_max_delta))
        except Exception:
            results.append(CurveResult(pix, None, None, None))
    return SLBAnalysis(results=results, R=R, nu=nu, tip_shape=tip_shape)


# --------------------------------------------------------------------------
# FEM cross-check
# --------------------------------------------------------------------------
def fem_reference(E: float, R: float, nu: float, h: float,
                  delta_max: float, substrate: str = "bonded",
                  n_steps: int = 12):
    """Run the FEM for the fitted SLB geometry and return (delta, force).

    Use with the median fitted modulus and height to overlay a full simulation
    on the experimental force curves. Incompressible samples use the mixed
    formulation automatically.
    """
    import warnings
    from .config import SimulationConfig
    from .solver import IndentationSolver

    cfg = SimulationConfig(
        tip_shape="paraboloid", R=R, E=E, nu=nu, h=h, substrate=substrate,
        L=max(20.0, 10 * np.sqrt(R * delta_max)), delta_max=delta_max,
        n_steps=n_steps, mesh_size_contact=max(0.02, delta_max / 50),
        mesh_size_edge=4.0, penalty_alpha=600.0,
        formulation="mixed" if nu >= 0.45 else "primal",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curve = IndentationSolver(cfg).run()
    return curve.delta, curve.force


def median_elastic_curve(analysis: "SLBAnalysis", delta_max: float,
                         n: int = 40):
    """Median experimental force–indentation over the elastic branches (nm, pN)."""
    grid = np.linspace(0, delta_max, n)
    stacks = []
    for r in analysis.results:
        if r.delta is not None and r.delta.size > 3 and r.delta.max() > 0.7 * delta_max:
            stacks.append(np.interp(grid, r.delta, r.force,
                                    left=np.nan, right=np.nan))
    if not stacks:
        return grid, np.full_like(grid, np.nan)
    arr = np.vstack(stacks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(arr, axis=0)
    return grid, med


def plot_analysis(analysis: "SLBAnalysis", film_thickness: float = 5.0,
                  fit_max_delta: float = 3.0, run_fem: bool = True,
                  fem_model: str = "dimitriadis"):
    """Four-panel SLB summary: breakthrough force, height, moduli, FEM check.

    Returns the matplotlib figure. ``run_fem`` overlays a FEM curve computed
    with the median modulus from ``fem_model`` and the given film thickness.
    """
    import matplotlib.pyplot as plt

    Fb = analysis.breakthrough_forces()
    ht = analysis.heights()
    ht_cs = analysis.heights_contact_substrate()
    mods = {m: analysis.moduli(m) for m in MODULUS_MODELS}
    colors = {"hertz": "#D55E00", "dimitriadis": "#0072B2",
              "garcia_bec": "#009E73", "chadwick": "#CC79A7"}

    fig, axs = plt.subplots(2, 2, figsize=(9.5, 7))
    if Fb.size:
        axs[0, 0].hist(Fb, bins=30, color="#009E73", edgecolor="k", lw=0.3)
        axs[0, 0].axvline(np.median(Fb), color="crimson", ls="--")
        axs[0, 0].set_title(f"Breakthrough force "
                            f"(median {np.median(Fb):.1f} nN, n={Fb.size})", fontsize=9)
    axs[0, 0].set_xlabel("breakthrough force (nN)"); axs[0, 0].set_ylabel("count")

    if ht.size:
        axs[0, 1].hist(ht, bins=25, color="#56B4E9", edgecolor="k", lw=0.3,
                       alpha=0.7, label=f"(B) step, med {np.median(ht):.1f} nm")
    if ht_cs.size:
        axs[0, 1].hist(ht_cs, bins=25, color="#E69F00", edgecolor="k", lw=0.3,
                       alpha=0.6,
                       label=f"(A) contact→substrate, med {np.median(ht_cs):.1f} nm")
    axs[0, 1].set_title("SLB height", fontsize=9)
    axs[0, 1].legend(fontsize=6)
    axs[0, 1].set_xlabel("height (nm)"); axs[0, 1].set_ylabel("count")

    data = [mods[m] for m in MODULUS_MODELS if mods[m].size]
    labs = [MODEL_LABELS[m] for m in MODULUS_MODELS if mods[m].size]
    if data:
        bp = axs[1, 0].boxplot(data, tick_labels=labs, showfliers=False,
                               patch_artist=True)
        for patch, m in zip(bp["boxes"], [k for k in MODULUS_MODELS if mods[k].size]):
            patch.set_facecolor(colors[m]); patch.set_alpha(0.6)
        axs[1, 0].set_yscale("log")
    axs[1, 0].set_ylabel("Young's modulus E (MPa)")
    axs[1, 0].set_title(f"E by model (h={film_thickness} nm, δ≤{fit_max_delta} nm)",
                        fontsize=9)
    axs[1, 0].tick_params(axis="x", rotation=20, labelsize=7)
    kA = analysis.kappa_A()
    if kA.size:
        axs[1, 0].text(0.5, 0.02,
                       f"Connell 2010 κ$_A$ = {np.median(kA):.0f} mN/m",
                       transform=axs[1, 0].transAxes, ha="center", va="bottom",
                       fontsize=7, bbox=dict(boxstyle="round", fc="#F0E442", alpha=0.6))

    grid, med = median_elastic_curve(analysis, fit_max_delta)
    axs[1, 1].plot(grid, med, "o-", color="k", ms=3, label="experiment (median)")
    if run_fem and mods.get(fem_model) is not None and mods[fem_model].size:
        E = float(np.median(mods[fem_model]))
        try:
            df, ff = fem_reference(E, analysis.R, analysis.nu, film_thickness,
                                   fit_max_delta, n_steps=10)
            axs[1, 1].plot(df, ff, "--", color="#0072B2", lw=1.5,
                           label=f"FEM (E={E:.1f} MPa, h={film_thickness} nm)")
        except Exception:
            pass
    axs[1, 1].set_xlabel("indentation δ (nm)"); axs[1, 1].set_ylabel("force (pN)")
    axs[1, 1].set_title("FEM cross-check", fontsize=9)
    axs[1, 1].legend(fontsize=7)

    for a in axs.flat:
        a.grid(alpha=0.25)
    fig.tight_layout()
    return fig
