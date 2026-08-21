"""Elastic contact plus a DLVO electrostatic term.

Bottom-effect models explain why a *bonded thin film* reads stiffer than a
half-space, but on a supported lipid bilayer in electrolyte they still tend to
under-predict the force as the tip approaches the substrate. A candidate
explanation is that the elastic response is not the whole force: as the film
thins under the tip, the electrical double layers of the tip and the underlying
mica overlap, and their repulsion adds to the measured force.

The model
---------
Total force is the elastic (bottom-effect-corrected) response plus a screened
electrostatic term that grows as the *remaining* film thins::

    F(delta) = F_elastic(delta; E) + F0 * exp(-(h - delta) / lambda_D)

``h - delta`` is the tip-to-substrate gap still filled with bilayer, ``F0`` the
double-layer force amplitude extrapolated to zero gap (pN) and ``lambda_D`` the
Debye screening length (nm). This is the Derjaguin/weak-overlap exponential
that DLVO gives for a sphere near a plane, written in the indentation
coordinate the experiment actually controls.

What this is and is not
-----------------------
The exponential form and its ``lambda_D`` are standard DLVO; mapping the
separation onto ``h - delta`` is a **modelling choice**, and a strong one: it
assumes the bilayer under the tip thins affinely and contributes no dielectric
screening of its own. ``lambda_D`` recovered from a fit should therefore be
sanity-checked against the Debye length of the actual buffer
(:func:`debye_length`); agreement is evidence the term is capturing double-layer
physics rather than absorbing whatever the elastic model got wrong. Three free
parameters on a monotonic curve will fit a great deal, so
:func:`compare_with_and_without_dlvo` reports whether the extra two are earned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .dataset import _shape_function

__all__ = [
    "debye_length", "dlvo_force", "dlvo_total_force", "DLVOFit",
    "fit_dlvo", "fit_elastic_only", "compare_with_and_without_dlvo",
    "BOLTZMANN_NM", "DEBYE_1MM_MONOVALENT_NM",
]

#: Debye length of a 1 mM monovalent aqueous electrolyte at 25 C (nm).
#: ``lambda_D = 0.304 / sqrt(I[M])`` nm is the standard shortcut.
DEBYE_1MM_MONOVALENT_NM = 9.61

#: kT at 25 C in the library's pN*nm energy unit (1 pN*nm = 1e-21 J).
BOLTZMANN_NM = 4.11


def debye_length(ionic_strength_mM: float, valence: int = 1,
                 temperature_c: float = 25.0) -> float:
    """Debye screening length (nm) for a symmetric electrolyte.

    ``lambda_D = 0.304 / (z * sqrt(I[M]))`` nm at 25 C, scaled by
    ``sqrt(T/298)``. Use it to check a fitted ``lambda_D`` against the buffer
    the measurement was actually made in.
    """
    if ionic_strength_mM <= 0:
        raise ValueError("ionic strength must be > 0")
    lam25 = 0.304 / (valence * np.sqrt(ionic_strength_mM * 1e-3))
    return float(lam25 * np.sqrt((temperature_c + 273.15) / 298.15))


def dlvo_force(delta, h: float, F0: float, lambda_D: float) -> np.ndarray:
    """Screened double-layer repulsion in the indentation coordinate (pN).

    ``F = F0 * exp(-(h - delta) / lambda_D)``, clipped so the gap never goes
    negative: past ``delta = h`` the tip has reached the substrate and the term
    saturates at ``F0`` rather than diverging.
    """
    if lambda_D <= 0:
        raise ValueError("lambda_D must be > 0")
    delta = np.asarray(delta, dtype=float)
    gap = np.clip(h - delta, 0.0, None)
    return F0 * np.exp(-gap / lambda_D)


def dlvo_total_force(delta, R: float, E: float, nu: float, h: float,
                     F0: float, lambda_D: float,
                     model: str = "dimitriadis_bonded") -> np.ndarray:
    """Elastic (bottom-effect-corrected) force plus the DLVO term (pN)."""
    shape = _shape_function(model, R, nu, h)
    d = np.clip(np.asarray(delta, dtype=float), 0.0, None)
    return E * shape(d) + dlvo_force(d, h, F0, lambda_D)


@dataclass
class DLVOFit:
    """Parameters recovered by :func:`fit_dlvo`.

    ``E`` is the elastic modulus (MPa) the contact model sees once the
    electrostatic term is accounted for, ``F0`` the double-layer amplitude at
    zero gap (pN) and ``lambda_D`` the screening length (nm).
    """

    model: str
    E: float
    F0: float
    lambda_D: float
    E_std: float = np.nan
    F0_std: float = np.nan
    lambda_D_std: float = np.nan
    rmse_pN: float = np.nan
    r_squared: float = np.nan
    n_points: int = 0
    converged: bool = False
    message: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def relative_uncertainty(self) -> dict:
        """1-sigma uncertainty as a fraction of each fitted value."""
        out = {}
        for name, value, std in (("E", self.E, self.E_std),
                                 ("F0", self.F0, self.F0_std),
                                 ("lambda_D", self.lambda_D, self.lambda_D_std)):
            out[name] = (abs(std / value) if np.isfinite(std) and value
                         else np.inf)
        return out

    def well_constrained(self, max_rel: float = 0.5) -> bool:
        """Whether every parameter is pinned to better than ``max_rel``.

        Converging is not the same as being determined. A fit whose
        ``lambda_D`` carries an uncertainty several times its own value has
        found *a* minimum, not *the* screening length — the usual sign that the
        data does not span enough of the exponential to constrain it.
        """
        return (self.converged
                and all(v <= max_rel for v in self.relative_uncertainty.values()))

    def predict(self, delta, R: float, nu: float, h: float) -> np.ndarray:
        return dlvo_total_force(delta, R, self.E, nu, h, self.F0,
                                self.lambda_D, self.model)

    def summary(self) -> str:
        return (f"{self.model} + DLVO: E = {self.E:.3g} +/- {self.E_std:.2g} MPa, "
                f"F0 = {self.F0:.3g} +/- {self.F0_std:.2g} pN, "
                f"lambda_D = {self.lambda_D:.3g} +/- {self.lambda_D_std:.2g} nm "
                f"(R^2 = {self.r_squared:.4f}, RMSE = {self.rmse_pN:.1f} pN)")


def _goodness(force, predicted, n_params: int) -> tuple:
    resid = np.asarray(predicted) - np.asarray(force)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((force - np.mean(force)) ** 2))
    rmse = float(np.sqrt(ss_res / max(len(resid), 1)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, r2, ss_res


def fit_elastic_only(delta, force, R: float, nu: float, h: float,
                     model: str = "dimitriadis_bonded") -> DLVOFit:
    """One-parameter linear fit of ``E`` with no electrostatic term.

    The null hypothesis :func:`compare_with_and_without_dlvo` scores against.
    """
    delta = np.asarray(delta, dtype=float)
    force = np.asarray(force, dtype=float)
    m = delta > 0
    d, f = delta[m], force[m]
    s = _shape_function(model, R, nu, h)(d)
    denom = float(np.dot(s, s))
    if denom <= 0:
        raise ValueError("degenerate shape function; check R, h and delta")
    E = float(np.dot(f, s) / denom)
    rmse, r2, _ = _goodness(f, E * s, 1)
    return DLVOFit(model=model, E=E, F0=0.0, lambda_D=np.inf, rmse_pN=rmse,
                   r_squared=r2, n_points=int(d.size), converged=True,
                   message="linear least squares (no DLVO term)")


def fit_dlvo(delta, force, R: float, nu: float, h: float,
             model: str = "dimitriadis_bonded",
             lambda_bounds: tuple = (0.1, 50.0),
             F0_bounds: tuple = (0.0, np.inf),
             E_bounds: tuple = (0.0, np.inf),
             p0: Optional[tuple] = None) -> DLVOFit:
    """Fit ``E``, ``F0`` and ``lambda_D`` to a force-indentation curve.

    Parameters
    ----------
    lambda_bounds:
        Debye length bounds (nm). The default spans roughly 100 M down to
        0.04 mM monovalent, i.e. anything physically reachable in a buffer;
        narrow it to the measured ionic strength when that is known.
    p0:
        Optional ``(E, F0, lambda_D)`` start. Defaults to the elastic-only ``E``,
        a tenth of the peak force, and 1 nm.
    """
    from scipy.optimize import curve_fit

    delta = np.asarray(delta, dtype=float)
    force = np.asarray(force, dtype=float)
    m = np.isfinite(delta) & np.isfinite(force) & (delta > 0)
    d, f = delta[m], force[m]
    if d.size < 5:
        raise ValueError(f"need at least 5 in-contact points, got {d.size}")

    shape = _shape_function(model, R, nu, h)

    def total(dd, E, F0, lam):
        return E * shape(dd) + dlvo_force(dd, h, F0, lam)

    if p0 is None:
        E0 = fit_elastic_only(d, f, R, nu, h, model).E
        p0 = (max(E0, 1e-6), max(float(np.max(f)) * 0.1, 1e-6), 1.0)

    bounds = ([E_bounds[0], F0_bounds[0], lambda_bounds[0]],
              [E_bounds[1], F0_bounds[1], lambda_bounds[1]])
    p0 = tuple(np.clip(p0, bounds[0], bounds[1]))

    try:
        popt, pcov = curve_fit(total, d, f, p0=p0, bounds=bounds, maxfev=40000)
        perr = np.sqrt(np.abs(np.diag(pcov)))
        converged, message = True, "converged"
    except Exception as exc:                        # noqa: BLE001 — reported
        return DLVOFit(model=model, E=np.nan, F0=np.nan, lambda_D=np.nan,
                       n_points=int(d.size), converged=False,
                       message=f"{type(exc).__name__}: {exc}")

    rmse, r2, _ = _goodness(f, total(d, *popt), 3)
    fit = DLVOFit(model=model, E=float(popt[0]), F0=float(popt[1]),
                  lambda_D=float(popt[2]), E_std=float(perr[0]),
                  F0_std=float(perr[1]), lambda_D_std=float(perr[2]),
                  rmse_pN=rmse, r_squared=r2, n_points=int(d.size),
                  converged=converged, message=message)
    for name, value, (lo, hi) in (("E", fit.E, E_bounds),
                                  ("F0", fit.F0, F0_bounds),
                                  ("lambda_D", fit.lambda_D, lambda_bounds)):
        for bound in (lo, hi):
            if np.isfinite(bound) and np.isclose(value, bound, rtol=1e-6):
                fit.converged = False
                fit.message = f"{name} railed against bound {bound:g}"
    return fit


def compare_with_and_without_dlvo(delta, force, R: float, nu: float, h: float,
                                  model: str = "dimitriadis_bonded") -> dict:
    """Is the electrostatic term earning its two extra parameters?

    Fits the same data with and without the DLVO term and scores both by
    corrected Akaike information criterion, which penalises the extra
    parameters. Returns both fits, ``delta_aicc`` (negative favours DLVO) and a
    plain-language ``verdict``.

    A better R^2 alone means nothing here — three free parameters on a smooth
    monotonic curve will almost always beat one.
    """
    delta = np.asarray(delta, dtype=float)
    force = np.asarray(force, dtype=float)
    m = np.isfinite(delta) & np.isfinite(force) & (delta > 0)
    d, f = delta[m], force[m]

    elastic = fit_elastic_only(d, f, R, nu, h, model)
    with_dlvo = fit_dlvo(d, f, R, nu, h, model)

    def aicc(fit, k):
        pred = (fit.E * _shape_function(model, R, nu, h)(d) if k == 1
                else fit.predict(d, R, nu, h))
        _, _, ss_res = _goodness(f, pred, k)
        n = len(d)
        if ss_res <= 0 or n <= k + 1:
            return np.nan
        value = n * np.log(ss_res / n) + 2 * k
        return value + (2 * k * (k + 1)) / (n - k - 1)

    aicc_elastic = aicc(elastic, 1)
    aicc_dlvo = aicc(with_dlvo, 3) if with_dlvo.converged else np.nan
    delta_aicc = aicc_dlvo - aicc_elastic

    if not with_dlvo.converged:
        verdict = f"DLVO fit did not converge ({with_dlvo.message})"
    elif not with_dlvo.well_constrained():
        worst = max(with_dlvo.relative_uncertainty,
                    key=lambda k: with_dlvo.relative_uncertainty[k])
        rel = with_dlvo.relative_uncertainty[worst]
        verdict = (f"converged but NOT constrained — {worst} carries "
                   f"{rel:.0%} uncertainty; the data does not pin it down")
    elif not np.isfinite(delta_aicc):
        verdict = "inconclusive (degenerate residuals)"
    elif delta_aicc < -10:
        verdict = "DLVO strongly favoured"
    elif delta_aicc < -2:
        verdict = "DLVO favoured"
    elif delta_aicc <= 2:
        verdict = "indistinguishable — prefer the simpler elastic-only model"
    else:
        verdict = "elastic-only favoured; the DLVO term is not earning its parameters"

    return {"elastic": elastic, "dlvo": with_dlvo,
            "well_constrained": with_dlvo.well_constrained(),
            "relative_uncertainty": with_dlvo.relative_uncertainty,
            "aicc_elastic": aicc_elastic, "aicc_dlvo": aicc_dlvo,
            "delta_aicc": delta_aicc, "verdict": verdict,
            "n_points": int(d.size)}
