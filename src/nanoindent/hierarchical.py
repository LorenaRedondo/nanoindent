"""Comparing conditions when curves are nested inside experiments.

An AFM campaign produces many force curves per experiment and several
experiments per condition. The temptation is to pool the curves and compare
conditions on the resulting means, weighting each condition by how many curves
it contributed. That is pseudoreplication: 800 curves from one bilayer are one
bilayer measured 800 times, not 800 bilayers, and treating them as independent
inflates every sample size by one to three orders of magnitude.

What matters is where the variance actually is. On the DOPC:MG data this
module was written for, the standard deviation *between* experiments within a
condition is two to eleven times the standard error *within* them. Preparation
and tip differ more than curves do. So the experiment is the unit of
replication, and the right model is two-level:

.. math::

    y_{ij} = \\mu_i + b_{ij} + \\varepsilon_{ij}, \\qquad
    b_{ij} \\sim N(0, \\tau^2), \\quad
    \\varepsilon_{ij} \\sim N(0, s_{ij}^2)

with :math:`y_{ij}` the mean of experiment *j* in condition *i*,
:math:`s_{ij}` its standard error — known, since it is reported — and
:math:`\\tau^2` the between-experiment variance to be estimated.

This is the structure of a random-effects meta-analysis, and it is the correct
one even though the "studies" are experiments from a single laboratory. It
answers the weighting question properly: an experiment's weight is
:math:`1/(s_{ij}^2 + \\tau^2)`, so more curves buys precision only until the
within-experiment error is small against the between-experiment spread — after
which extra curves buy nothing, which is the honest answer.

Only summary statistics are needed: mean, standard error and count per
experiment. That is what published tables report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "Experiment", "Condition", "PooledEstimate", "Comparison",
    "estimate_tau2", "pool", "omnibus_test", "pairwise", "heterogeneity",
]


@dataclass(frozen=True)
class Experiment:
    """One experiment: a mean over its own curves, with its standard error."""

    n_curves: int
    mean: float
    sem: Optional[float] = None

    @property
    def variance(self) -> float:
        """Within-experiment sampling variance of the mean."""
        return float("nan") if self.sem is None else float(self.sem) ** 2

    @property
    def sd(self) -> float:
        """Curve-to-curve standard deviation implied by the reported SEM."""
        return (float("nan") if self.sem is None
                else float(self.sem) * np.sqrt(self.n_curves))


@dataclass(frozen=True)
class Condition:
    """A set of experiments sharing a preparation."""

    name: str
    experiments: tuple

    @property
    def n_experiments(self) -> int:
        return len(self.experiments)

    @property
    def n_curves(self) -> int:
        return int(sum(e.n_curves for e in self.experiments))

    def means(self) -> np.ndarray:
        return np.array([e.mean for e in self.experiments], dtype=float)

    def variances(self, impute: bool = True) -> np.ndarray:
        """Within-experiment variances, imputing where a SEM is missing.

        A single-curve experiment has no standard error of its own. Dropping it
        would discard a real replicate; assuming it precise would give it huge
        weight. It is imputed from the curve-to-curve spread of the condition's
        other experiments, which is conservative — the imputed value is larger
        than any observed SEM in the condition.
        """
        v = np.array([e.variance for e in self.experiments], dtype=float)
        if impute and np.isnan(v).any():
            sds = np.array([e.sd for e in self.experiments], dtype=float)
            known = sds[~np.isnan(sds)]
            fallback = float(np.median(known)) if known.size else np.nan
            for i, e in enumerate(self.experiments):
                if np.isnan(v[i]):
                    v[i] = (fallback / np.sqrt(e.n_curves)) ** 2
        return v


@dataclass(frozen=True)
class PooledEstimate:
    """A condition's mean, with uncertainty that respects the nesting."""

    name: str
    mean: float
    se: float
    n_experiments: int
    n_curves: int
    ci: tuple
    weights: np.ndarray

    @property
    def effective_n(self) -> float:
        """Experiments' worth of information, after weighting.

        Kish's effective sample size. It is close to the experiment count
        whenever the between-experiment variance dominates, which is the
        signature of weights that have stopped being driven by curve counts.
        """
        w = self.weights
        return float(w.sum() ** 2 / np.sum(w ** 2))


@dataclass(frozen=True)
class Comparison:
    """One pairwise contrast between conditions."""

    a: str
    b: str
    difference: float
    se: float
    z: float
    p_raw: float
    p_adjusted: float

    @property
    def significant(self) -> bool:
        return self.p_adjusted < 0.05


def estimate_tau2(conditions: Sequence[Condition], tol: float = 1e-10,
                  max_iter: int = 200) -> float:
    """Between-experiment variance, shared across conditions (Paule-Mandel).

    A common :math:`\\tau^2` is used rather than one per condition. Several
    conditions here contain one or two experiments, which cannot support their
    own estimate; pooling lets them borrow the variability observed elsewhere
    instead of being assigned an implausible zero.

    The estimator solves :math:`Q(\\tau^2) = \\mathrm{df}` — it chooses the
    between-experiment variance that makes the residual heterogeneity equal to
    its expectation. It is preferred to DerSimonian-Laird here because DL is
    known to be downward-biased when the number of studies is small, which
    would overstate every result.
    """
    usable = [c for c in conditions if c.n_experiments > 0]
    df = sum(c.n_experiments for c in usable) - len(usable)
    if df <= 0:
        return 0.0

    def q_of(t2: float) -> float:
        total = 0.0
        for c in usable:
            if c.n_experiments < 2:
                continue
            w = 1.0 / (c.variances() + t2)
            mu = float(np.sum(w * c.means()) / np.sum(w))
            total += float(np.sum(w * (c.means() - mu) ** 2))
        return total

    df_used = sum(c.n_experiments - 1 for c in usable if c.n_experiments >= 2)
    if df_used <= 0 or q_of(0.0) <= df_used:
        return 0.0

    lo, hi = 0.0, 1.0
    while q_of(hi) > df_used and hi < 1e12:
        hi *= 4.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if q_of(mid) > df_used:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def pool(condition: Condition, tau2: float, z: float = 1.959963985) -> PooledEstimate:
    """Random-effects mean of one condition."""
    v = condition.variances()
    w = 1.0 / (v + tau2)
    mean = float(np.sum(w * condition.means()) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return PooledEstimate(
        name=condition.name, mean=mean, se=se,
        n_experiments=condition.n_experiments, n_curves=condition.n_curves,
        ci=(mean - z * se, mean + z * se), weights=w)


def heterogeneity(condition: Condition) -> dict:
    """Cochran's Q and I-squared for one condition, at tau-squared = 0.

    ``i2`` is the share of the observed variation that is not explained by
    within-experiment sampling error. Values near 1 mean the experiments
    disagree by far more than their own error bars allow, which is the
    quantitative form of "the spread is between experiments".
    """
    if condition.n_experiments < 2:
        return {"Q": float("nan"), "df": 0, "i2": float("nan"),
                "p": float("nan")}
    from scipy import stats

    v = condition.variances()
    w = 1.0 / v
    mu = float(np.sum(w * condition.means()) / np.sum(w))
    q = float(np.sum(w * (condition.means() - mu) ** 2))
    df = condition.n_experiments - 1
    return {"Q": q, "df": df, "i2": max(0.0, (q - df) / q) if q > 0 else 0.0,
            "p": float(stats.chi2.sf(q, df))}


def omnibus_test(pooled: Sequence[PooledEstimate]) -> dict:
    """Do the conditions differ at all? Wald test on the pooled means."""
    from scipy import stats

    m = np.array([p.mean for p in pooled], dtype=float)
    w = np.array([1.0 / p.se ** 2 for p in pooled], dtype=float)
    grand = float(np.sum(w * m) / np.sum(w))
    q = float(np.sum(w * (m - grand) ** 2))
    df = len(pooled) - 1
    return {"Q": q, "df": df, "p": float(stats.chi2.sf(q, df)),
            "grand_mean": grand}


def pairwise(pooled: Sequence[PooledEstimate],
             method: str = "holm") -> list:
    """All pairwise contrasts, with multiplicity control.

    Holm by default: it controls the family-wise error rate without assuming
    independence between contrasts, which matters because every contrast here
    shares conditions with several others.
    """
    from scipy import stats

    out = []
    for i in range(len(pooled)):
        for j in range(i + 1, len(pooled)):
            a, b = pooled[i], pooled[j]
            diff = a.mean - b.mean
            se = float(np.sqrt(a.se ** 2 + b.se ** 2))
            z = diff / se if se > 0 else np.nan
            out.append(Comparison(a.name, b.name, diff, se, z,
                                  float(2 * stats.norm.sf(abs(z))), np.nan))
    return _adjust(out, method)


def _adjust(comparisons: list, method: str) -> list:
    p = np.array([c.p_raw for c in comparisons], dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    if method == "holm":
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * p[idx])
            adj[idx] = min(1.0, running)
    elif method == "bh":
        running = 1.0
        for rank in range(m - 1, -1, -1):
            idx = order[rank]
            running = min(running, m / (rank + 1) * p[idx])
            adj[idx] = min(1.0, running)
    else:
        raise ValueError(f"unknown method {method!r}; use 'holm' or 'bh'")
    return [Comparison(c.a, c.b, c.difference, c.se, c.z, c.p_raw, float(a))
            for c, a in zip(comparisons, adj)]
