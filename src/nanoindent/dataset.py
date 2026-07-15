"""Synthetic AFM datasets and inverse analysis (modulus recovery).

This module closes the loop on the forward solver: it generates *realistic*
force-indentation data — the kind an AFM would record, with measurement noise,
a force baseline offset, and an unknown contact point — and then fits an
elastic modulus back out of that data with a chosen contact model.

The point is a genuine double-check of the analysis: if you take a thin bonded
film, synthesise data from the FEM (the ground truth), and then fit it with the
naive **Hertz** model, you recover an *inflated* modulus — that is the
bottom-effect artefact. Fitting the same data with the **Dimitriadis** bonded
correction recovers the true modulus. The utilities here make that quantitative.

Because force is linear in E for every model here
(``F = E * shape(delta; R, nu, h)``), the modulus fit is a one-parameter linear
least squares and needs no iterative optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import csv

import numpy as np

from .config import SimulationConfig
from . import analytics


# -- model shape functions (force per unit E) --------------------------------
def _shape_function(model: str, R: float, nu: float, h: float) -> Callable:
    """Return ``f(delta) -> force / E`` for the named model.

    Force is proportional to E for all supported models, so dividing by E
    leaves an E-independent "shape" used both to synthesise data at a known E
    and to fit E by linear least squares.
    """
    if model == "hertz":
        return lambda d: analytics.hertz_force(d, R, 1.0, nu)
    if model in ("dimitriadis", "dimitriadis_bonded"):
        return lambda d: analytics.dimitriadis_force(d, R, 1.0, nu, h, bonded=True)
    if model == "garcia_bec":
        return lambda d: analytics.garcia_bec_force(d, R, 1.0, nu, h)
    if model in ("chadwick", "chadwick_2002"):
        return lambda d: analytics.chadwick_force(d, R, 1.0, h)
    raise ValueError(f"unknown model {model!r}")


@dataclass
class SyntheticDataset:
    """A synthetic AFM force curve plus the ground truth it was built from."""

    delta: np.ndarray            # measured indentation (nm), may include <0 region
    force: np.ndarray            # measured force (pN), noisy
    true_E: float                # ground-truth modulus (MPa)
    R: float
    nu: float
    h: float
    source: str                  # how the ground truth was generated
    noise_pN: float = 0.0
    contact_point: float = 0.0   # true contact point offset applied (nm)
    baseline_pN: float = 0.0     # true baseline offset applied (pN)
    meta: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["# nanoindent synthetic AFM dataset"])
            w.writerow([f"# true_E_MPa={self.true_E}", f"R_nm={self.R}",
                        f"nu={self.nu}", f"h_nm={self.h}", f"source={self.source}",
                        f"noise_pN={self.noise_pN}"])
            w.writerow(["delta_nm", "force_pN"])
            for d, f in zip(self.delta, self.force):
                w.writerow([f"{d:.10g}", f"{f:.10g}"])


def synthetic_force_curve(
    cfg: SimulationConfig,
    source: str = "dimitriadis",
    true_E: Optional[float] = None,
    n_points: int = 120,
    noise_pN: float = 8.0,
    baseline_pN: float = 0.0,
    contact_point: float = 0.0,
    include_precontact: bool = True,
    fem_force: Optional[Callable] = None,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate a realistic noisy force-indentation dataset.

    Parameters
    ----------
    cfg:
        Geometry / material context (uses R, nu, h, delta_max, E).
    source:
        Ground-truth generator: ``"hertz"``, ``"dimitriadis"``, ``"garcia_bec"``
        or ``"fem"`` (requires ``fem_force``).
    true_E:
        Ground-truth modulus (defaults to ``cfg.E``).
    n_points:
        Number of sampled points over the approach.
    noise_pN:
        Std. dev. of additive Gaussian force noise (thermal/detection noise).
        A few-to-tens of pN is typical for soft-matter AFM.
    baseline_pN, contact_point:
        A constant force offset and a contact-point shift, mimicking the
        baseline drift and unknown tip-sample contact of a real experiment.
    include_precontact:
        If True, prepend a flat pre-contact region (delta < contact_point) so
        the dataset looks like a raw approach curve.
    fem_force:
        Callable ``delta -> force`` used when ``source == "fem"`` (e.g. an
        interpolator over a ForceCurve).
    seed:
        RNG seed for reproducibility.
    """
    E = cfg.E if true_E is None else true_E
    rng = np.random.default_rng(seed)

    if source == "fem":
        if fem_force is None:
            raise ValueError("source='fem' requires fem_force")
        force_of_delta = lambda d: np.asarray(fem_force(d), dtype=float)
    else:
        shape = _shape_function(source, cfg.R, cfg.nu, cfg.h)
        force_of_delta = lambda d: E * shape(np.clip(d, 0.0, None))

    # Sample the contact region on [0, delta_max].
    d_contact = np.linspace(0.0, cfg.delta_max, n_points)
    f_contact = force_of_delta(d_contact)

    if include_precontact:
        pre = np.linspace(-0.3 * cfg.delta_max, 0.0, max(n_points // 5, 5),
                          endpoint=False)
        d_all = np.concatenate([pre, d_contact])
        f_all = np.concatenate([np.zeros_like(pre), f_contact])
    else:
        d_all, f_all = d_contact, f_contact

    # Apply the experimental offsets and noise.
    measured_delta = d_all + contact_point
    measured_force = f_all + baseline_pN + rng.normal(0.0, noise_pN, size=f_all.shape)

    return SyntheticDataset(
        delta=measured_delta,
        force=measured_force,
        true_E=E,
        R=cfg.R,
        nu=cfg.nu,
        h=cfg.h,
        source=source,
        noise_pN=noise_pN,
        contact_point=contact_point,
        baseline_pN=baseline_pN,
    )


# -- inverse analysis --------------------------------------------------------
@dataclass
class ModulusFit:
    model: str
    E_fit: float                 # recovered modulus (MPa)
    E_std: float                 # 1-sigma uncertainty from the fit (MPa)
    contact_point: float         # contact point used (nm)
    rmse_pN: float               # RMS residual (pN)
    n_points: int


def _fit_E_at(delta, force, shape, contact_point, baseline):
    """Closed-form linear fit of E at a fixed contact point and baseline.

    Returns ``(E_fit, rmse, n_points, denom)`` or ``None`` if degenerate.
    """
    f = force - baseline
    d = delta - contact_point
    mask = d > 0.0
    d_c, f_c = d[mask], f[mask]
    if d_c.size < 3:
        return None
    s = shape(d_c)
    denom = float(np.dot(s, s))
    if denom <= 0:
        return None
    E_fit = float(np.dot(f_c, s) / denom)
    resid = f_c - E_fit * s
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return E_fit, rmse, int(d_c.size), denom, resid


def fit_modulus(
    delta: np.ndarray,
    force: np.ndarray,
    model: str,
    R: float,
    nu: float,
    h: float,
    contact_point: Optional[float] = None,
    baseline: Optional[float] = None,
    fit_contact_point: bool = False,
) -> ModulusFit:
    """Recover the elastic modulus from a force curve with the given model.

    Only points in contact (``delta > contact_point``) are used. If
    ``baseline`` is None it is estimated from the pre-contact region
    (``delta <= contact_point``, or ``delta <= 0`` when the contact point is
    also being fit). Force is linear in E, so E is obtained in closed form by
    linear least squares ``E = <F, s> / <s, s>`` with ``s(delta) = shape``.

    Parameters
    ----------
    fit_contact_point:
        If True (and ``contact_point`` is None), scan candidate contact points
        and pick the one minimising the fit RMSE — mimicking the contact-point
        determination step of a real AFM analysis. If a ``contact_point`` is
        given it is held fixed.
    """
    delta = np.asarray(delta, dtype=float)
    force = np.asarray(force, dtype=float)
    shape = _shape_function(model, R, nu, h)

    def baseline_at(cp: float) -> float:
        if baseline is not None:
            return baseline
        pre = force[delta <= cp]
        return float(np.mean(pre)) if pre.size else 0.0

    if contact_point is None and fit_contact_point:
        # Scan candidate contact points across the low-delta region.
        lo = float(np.min(delta))
        hi = min(float(np.max(delta)) * 0.5, lo + 0.6 * (delta.max() - lo))
        candidates = np.linspace(lo, hi, 40)
        best = None
        for cp in candidates:
            res = _fit_E_at(delta, force, shape, cp, baseline_at(cp))
            if res is None:
                continue
            E_fit, rmse, n, denom, resid = res
            if E_fit <= 0:
                continue
            if best is None or rmse < best[1]:
                best = (E_fit, rmse, n, denom, resid, cp)
        if best is None:
            raise ValueError("contact-point scan failed to find a valid fit")
        E_fit, rmse, n, denom, resid, cp = best
        contact_point = float(cp)
    else:
        cp = 0.0 if contact_point is None else contact_point
        res = _fit_E_at(delta, force, shape, cp, baseline_at(cp))
        if res is None:
            raise ValueError("not enough in-contact points to fit")
        E_fit, rmse, n, denom, resid = res
        contact_point = float(cp)

    dof = max(n - 1, 1)
    sigma2 = float(np.dot(resid, resid) / dof)
    E_std = float(np.sqrt(sigma2 / denom))

    return ModulusFit(model=model, E_fit=E_fit, E_std=E_std,
                      contact_point=contact_point, rmse_pN=rmse, n_points=n)
