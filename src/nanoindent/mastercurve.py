"""Average many force curves into one master curve.

Per-curve AFM data on a bilayer is noisy and each curve has its own thickness,
so a comparison against contact models is easier to read on an average. How you
average matters, and the two natural choices are not equivalent:

**Bin in force** (:func:`bin_by_force`)
    Force is the controlled variable — the ramp is driven to a force setpoint,
    so every curve spans the same force range and contributes to every bin.
    Bins are equally populated by construction and no curve is extrapolated.
    The cost is that the *abscissa* (indentation, or chi) becomes the averaged
    quantity, and its spread within a bin is the thing you are trying to plot.

**Bin in chi** (:func:`bin_by_chi`)
    Puts the averaging on the axis the models are expressed in, so the mean is
    taken at a fixed value of the reduced variable. But chi depends on each
    curve's own ``h``, so curves with thicker films stop contributing at high
    chi and the tail is averaged over a shrinking, biased subset — the effect
    :func:`coverage` reports.

Neither is right in the abstract. :func:`compare_binnings` runs both and reports
where they diverge, which is the honest way to choose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["CurveSet", "MasterCurve", "bin_by_force", "bin_by_chi",
           "compare_binnings", "chi_of"]


def chi_of(delta, R: float, h: float) -> np.ndarray:
    """Reduced bottom-effect variable ``chi = sqrt(R*delta)/h``."""
    d = np.clip(np.asarray(delta, dtype=float), 0.0, None)
    return np.sqrt(R * d) / h


@dataclass
class CurveSet:
    """Several force curves, each with its own film thickness.

    ``deltas`` / ``forces`` are per-curve arrays (nm, pN); ``heights`` is the
    per-curve film thickness ``h`` (nm) that sets its chi axis.
    """

    deltas: list
    forces: list
    heights: list
    R: float
    labels: list = field(default_factory=list)

    def __post_init__(self):
        n = len(self.deltas)
        if not (len(self.forces) == len(self.heights) == n):
            raise ValueError("deltas, forces and heights must be the same length")
        if n == 0:
            raise ValueError("need at least one curve")
        self.deltas = [np.asarray(d, dtype=float) for d in self.deltas]
        self.forces = [np.asarray(f, dtype=float) for f in self.forces]
        self.heights = [float(h) for h in self.heights]
        if not self.labels:
            self.labels = [f"curve {i}" for i in range(n)]

    def __len__(self) -> int:
        return len(self.deltas)

    @property
    def chis(self) -> list:
        return [chi_of(d, self.R, h) for d, h in zip(self.deltas, self.heights)]

    def flat(self) -> tuple:
        """All points pooled as ``(delta, force, chi, curve_index)``."""
        idx = np.concatenate([np.full(d.size, i)
                              for i, d in enumerate(self.deltas)])
        return (np.concatenate(self.deltas), np.concatenate(self.forces),
                np.concatenate(self.chis), idx)


@dataclass
class MasterCurve:
    """A binned average over a :class:`CurveSet`.

    ``n_curves`` counts *distinct curves* contributing to each bin, not points —
    twenty points from one curve is not twenty independent measurements, and the
    error bars are scaled accordingly.
    """

    binned_on: str                       # "force" or "chi"
    edges: np.ndarray
    chi: np.ndarray                      # bin mean chi
    chi_std: np.ndarray
    force: np.ndarray                    # bin mean force (pN)
    force_std: np.ndarray
    delta: np.ndarray                    # bin mean indentation (nm)
    delta_std: np.ndarray
    n_points: np.ndarray
    n_curves: np.ndarray
    R: float = np.nan
    h: float = np.nan                    # representative thickness (median)

    def __len__(self) -> int:
        return int(np.sum(np.isfinite(self.force)))

    @property
    def chi_sem(self) -> np.ndarray:
        """Standard error on chi, using the count of independent curves."""
        return self.chi_std / np.sqrt(np.maximum(self.n_curves, 1))

    @property
    def force_sem(self) -> np.ndarray:
        return self.force_std / np.sqrt(np.maximum(self.n_curves, 1))

    def coverage(self) -> np.ndarray:
        """Fraction of the set contributing to each bin.

        Well below 1 in the tail means the average there is over a biased
        subset — the failure mode of binning in chi.
        """
        total = np.max(self.n_curves) if self.n_curves.size else 1
        return self.n_curves / max(total, 1)

    def valid(self, min_curves: int = 3) -> np.ndarray:
        """Mask of bins backed by at least ``min_curves`` distinct curves."""
        return np.isfinite(self.force) & (self.n_curves >= min_curves)

    def summary(self) -> str:
        v = self.valid()
        return (f"MasterCurve(binned_on={self.binned_on!r}, {int(v.sum())}/"
                f"{len(self.edges) - 1} usable bins, "
                f"chi {np.nanmin(self.chi[v]):.2f}-{np.nanmax(self.chi[v]):.2f}, "
                f"force {np.nanmin(self.force[v]):.0f}-"
                f"{np.nanmax(self.force[v]):.0f} pN)")


def _aggregate(values, curve_idx, bin_idx, n_bins) -> tuple:
    """Per-bin mean, std, point count and distinct-curve count."""
    mean = np.full(n_bins, np.nan)
    std = np.full(n_bins, np.nan)
    n_pts = np.zeros(n_bins, dtype=int)
    n_curves = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = bin_idx == b
        if not np.any(sel):
            continue
        v = values[sel]
        mean[b] = np.nanmean(v)
        std[b] = np.nanstd(v)
        n_pts[b] = int(v.size)
        n_curves[b] = int(np.unique(curve_idx[sel]).size)
    return mean, std, n_pts, n_curves


def _build(curves: CurveSet, key, edges, binned_on: str) -> MasterCurve:
    delta, force, chi, idx = curves.flat()
    ok = np.isfinite(key) & np.isfinite(force) & np.isfinite(chi) & (delta > 0)
    delta, force, chi, idx, key = (delta[ok], force[ok], chi[ok], idx[ok],
                                   key[ok])
    n_bins = len(edges) - 1
    bin_idx = np.digitize(key, edges) - 1
    inside = (bin_idx >= 0) & (bin_idx < n_bins)
    delta, force, chi, idx, bin_idx = (delta[inside], force[inside],
                                       chi[inside], idx[inside],
                                       bin_idx[inside])

    chi_m, chi_s, n_pts, n_cur = _aggregate(chi, idx, bin_idx, n_bins)
    f_m, f_s, _, _ = _aggregate(force, idx, bin_idx, n_bins)
    d_m, d_s, _, _ = _aggregate(delta, idx, bin_idx, n_bins)
    return MasterCurve(binned_on=binned_on, edges=np.asarray(edges),
                       chi=chi_m, chi_std=chi_s, force=f_m, force_std=f_s,
                       delta=d_m, delta_std=d_s, n_points=n_pts,
                       n_curves=n_cur, R=curves.R,
                       h=float(np.median(curves.heights)))


def bin_by_force(curves: CurveSet, n_bins: int = 30,
                 force_range: Optional[tuple] = None,
                 log_spaced: bool = False) -> MasterCurve:
    """Average with bins on **force**, the controlled variable.

    Every curve spans the force range the ramp was driven over, so bins stay
    populated across the whole curve and no curve is extrapolated.
    ``log_spaced`` helps when the force spans decades, since linear bins put
    most points in the stiff tail.
    """
    _, force, _, _ = curves.flat()
    lo, hi = force_range or (np.nanpercentile(force[force > 0], 1),
                             np.nanmax(force))
    if log_spaced:
        edges = np.geomspace(max(lo, 1e-6), hi, n_bins + 1)
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
    _, f, _, _ = curves.flat()
    return _build(curves, f, edges, "force")


def bin_by_chi(curves: CurveSet, n_bins: int = 30,
               chi_range: Optional[tuple] = None) -> MasterCurve:
    """Average with bins on **chi**, the axis the models live on.

    Cleaner to compare against a model, but each curve's chi range depends on
    its own ``h``, so check :meth:`MasterCurve.coverage` before trusting the
    high-chi tail.
    """
    _, _, chi, _ = curves.flat()
    lo, hi = chi_range or (0.0, np.nanmax(chi))
    edges = np.linspace(lo, hi, n_bins + 1)
    _, _, c, _ = curves.flat()
    return _build(curves, c, edges, "chi")


def compare_binnings(curves: CurveSet, n_bins: int = 30,
                     min_curves: int = 3) -> dict:
    """Bin both ways and report where the two disagree.

    Interpolates the force-binned master curve onto the chi-binned one's chi
    grid and measures the relative difference. Small differences mean the
    choice does not matter for this dataset; large ones mean it does, and the
    coverage numbers say which to distrust.
    """
    by_force = bin_by_force(curves, n_bins=n_bins)
    by_chi = bin_by_chi(curves, n_bins=n_bins)

    vf, vc = by_force.valid(min_curves), by_chi.valid(min_curves)
    if vf.sum() < 2 or vc.sum() < 2:
        return {"by_force": by_force, "by_chi": by_chi,
                "max_rel_diff": np.nan, "median_rel_diff": np.nan,
                "note": "too few usable bins to compare"}

    order = np.argsort(by_force.chi[vf])
    chi_f = by_force.chi[vf][order]
    force_f = by_force.force[vf][order]

    chi_c, force_c = by_chi.chi[vc], by_chi.force[vc]
    inside = (chi_c >= chi_f.min()) & (chi_c <= chi_f.max())
    interp = np.interp(chi_c[inside], chi_f, force_f)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(interp - force_c[inside]) / np.abs(force_c[inside])

    tail = by_chi.coverage()[vc][inside]
    return {
        "by_force": by_force, "by_chi": by_chi,
        "chi": chi_c[inside], "rel_diff": rel,
        "max_rel_diff": float(np.nanmax(rel)) if rel.size else np.nan,
        "median_rel_diff": float(np.nanmedian(rel)) if rel.size else np.nan,
        "min_coverage_chi_binned": float(np.nanmin(tail)) if tail.size else np.nan,
        "note": "rel_diff is |force-binned - chi-binned| / chi-binned",
    }
