"""Comparing conditions when curves are nested inside experiments.

The failure these guard against is pseudoreplication: treating 800 curves from
one experiment as 800 independent observations of a condition. The tests below
check that the model actually resists it — that weights stop tracking curve
counts once the between-experiment variance dominates, and that intervals widen
accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoindent.hierarchical import (
    Condition,
    Experiment,
    estimate_tau2,
    heterogeneity,
    omnibus_test,
    pairwise,
    pool,
)


def _cond(name, rows):
    return Condition(name, tuple(Experiment(*r) for r in rows))


# -- the primitives -----------------------------------------------------------
def test_sd_is_recovered_from_the_reported_sem():
    e = Experiment(n_curves=100, mean=10.0, sem=0.5)
    assert e.sd == pytest.approx(5.0)
    assert e.variance == pytest.approx(0.25)


def test_a_missing_sem_is_imputed_from_the_other_experiments():
    """A single-curve experiment has no error of its own.

    Dropping it discards a real replicate; treating it as precise would hand it
    enormous weight. It inherits the curve-to-curve spread of its condition.
    """
    c = _cond("x", [(1, 5.0, None), (100, 6.0, 0.5), (100, 7.0, 0.5)])
    v = c.variances()
    assert np.isfinite(v).all()
    assert v[0] > v[1], "the single-curve experiment must be the least precise"
    assert v[0] == pytest.approx(25.0)      # sd 5 over one curve


def test_imputation_can_be_switched_off():
    c = _cond("x", [(1, 5.0, None), (100, 6.0, 0.5)])
    assert np.isnan(c.variances(impute=False)).any()


# -- tau squared --------------------------------------------------------------
def test_tau2_is_zero_when_experiments_agree_within_their_error_bars():
    c = _cond("x", [(100, 10.0, 1.0), (100, 10.1, 1.0), (100, 9.9, 1.0)])
    assert estimate_tau2([c]) == pytest.approx(0.0, abs=1e-9)


def test_tau2_grows_with_disagreement_between_experiments():
    tight = _cond("t", [(100, 10.0, 0.1), (100, 10.1, 0.1), (100, 9.9, 0.1)])
    loose = _cond("l", [(100, 5.0, 0.1), (100, 15.0, 0.1), (100, 25.0, 0.1)])
    assert estimate_tau2([loose]) > estimate_tau2([tight])


def test_tau2_is_shared_so_a_single_experiment_condition_borrows_it():
    """A condition with one experiment cannot estimate its own spread.

    Left to itself it would be assigned zero and look impossibly precise.
    """
    many = _cond("many", [(50, 5.0, 0.5), (50, 15.0, 0.5), (50, 25.0, 0.5)])
    alone = _cond("alone", [(50, 10.0, 0.5)])
    tau2 = estimate_tau2([many, alone])
    assert tau2 > 0
    solo = pool(alone, tau2)
    assert solo.se > 0.5, "its interval must reflect the shared variability"


# -- the point of the exercise ------------------------------------------------
def test_effective_sample_size_tracks_experiments_not_curves():
    """The arithmetic form of "curves are not replicates".

    One experiment contributes 800 curves and the others 30; with a real
    between-experiment variance the effective N stays near the experiment
    count rather than being dragged towards the big one.
    """
    c = _cond("x", [(800, 12.0, 0.3), (30, 24.0, 1.0), (30, 4.0, 1.0)])
    tau2 = estimate_tau2([c])
    p = pool(c, tau2)
    assert p.n_curves == 860
    assert p.effective_n == pytest.approx(3.0, abs=0.35)


def test_random_effects_intervals_are_wider_than_curve_weighted_ones():
    c = _cond("x", [(600, 15.0, 0.2), (100, 30.0, 0.5), (100, 9.0, 0.5)])
    naive = pool(c, 0.0)
    honest = pool(c, estimate_tau2([c]))
    assert honest.se > 5 * naive.se


def test_curve_weighting_would_put_the_mean_on_the_largest_experiment():
    """Why it matters, not just that it differs.

    With curve weighting the 600-curve experiment dominates and the pooled
    mean sits almost on top of it; the random-effects mean sits among the
    three experiments instead.
    """
    c = _cond("x", [(600, 15.0, 0.2), (100, 30.0, 0.5), (100, 9.0, 0.5)])
    naive = pool(c, 0.0)
    honest = pool(c, estimate_tau2([c]))
    unweighted = float(c.means().mean())

    # the naive mean is dragged towards the 600-curve experiment; the
    # random-effects mean sits at the centre of the three experiments
    assert abs(naive.mean - 15.0) < abs(honest.mean - 15.0)
    assert honest.mean == pytest.approx(unweighted, abs=0.3)


def test_more_curves_stop_buying_precision_once_between_variance_dominates():
    few = _cond("few", [(20, 5.0, 1.0), (20, 15.0, 1.0), (20, 25.0, 1.0)])
    many = _cond("many", [(20000, 5.0, 0.03), (20000, 15.0, 0.03),
                          (20000, 25.0, 0.03)])
    se_few = pool(few, estimate_tau2([few])).se
    se_many = pool(many, estimate_tau2([many])).se
    assert se_many == pytest.approx(se_few, rel=0.25), (
        "a thousandfold more curves must not shrink the interval much")


# -- heterogeneity ------------------------------------------------------------
def test_i2_is_near_one_when_experiments_disagree_wildly():
    c = _cond("x", [(100, 5.0, 0.1), (100, 25.0, 0.1), (100, 45.0, 0.1)])
    assert heterogeneity(c)["i2"] > 0.95


def test_i2_is_zero_when_they_agree():
    c = _cond("x", [(100, 10.0, 1.0), (100, 10.1, 1.0), (100, 9.9, 1.0)])
    assert heterogeneity(c)["i2"] == pytest.approx(0.0, abs=1e-6)


def test_heterogeneity_is_undefined_for_a_single_experiment():
    assert np.isnan(heterogeneity(_cond("x", [(10, 1.0, 0.1)]))["i2"])


# -- inference ----------------------------------------------------------------
def test_omnibus_detects_a_real_difference_and_not_a_fake_one():
    tau2 = 0.0
    same = [pool(_cond("a", [(50, 10.0, 0.5), (50, 10.2, 0.5)]), tau2),
            pool(_cond("b", [(50, 10.1, 0.5), (50, 9.9, 0.5)]), tau2)]
    diff = [pool(_cond("a", [(50, 10.0, 0.5), (50, 10.2, 0.5)]), tau2),
            pool(_cond("b", [(50, 40.0, 0.5), (50, 40.2, 0.5)]), tau2)]
    assert omnibus_test(same)["p"] > 0.05
    assert omnibus_test(diff)["p"] < 1e-6


def test_holm_adjustment_never_lowers_a_p_value():
    ps = [pool(_cond(n, [(50, m, 0.5), (50, m + 0.2, 0.5)]), 0.0)
          for n, m in (("a", 10.0), ("b", 12.0), ("c", 30.0), ("d", 31.0))]
    for c in pairwise(ps):
        assert c.p_adjusted >= c.p_raw


def test_correcting_for_multiplicity_actually_removes_findings():
    """Marginal effects, which is where multiplicity control earns its keep."""
    ps = [pool(_cond(f"c{i}", [(50, 10.0 + 0.42 * i, 0.5),
                               (50, 10.2 + 0.42 * i, 0.5)]), 0.0)
          for i in range(6)]
    raw = sum(c.p_raw < 0.05 for c in pairwise(ps))
    adj = sum(c.significant for c in pairwise(ps))
    assert 0 < adj < raw, (raw, adj)


def test_bh_is_less_conservative_than_holm():
    ps = [pool(_cond(f"c{i}", [(50, 10.0 + 0.8 * i, 0.5),
                               (50, 10.2 + 0.8 * i, 0.5)]), 0.0)
          for i in range(6)]
    assert (sum(c.significant for c in pairwise(ps, "bh"))
            >= sum(c.significant for c in pairwise(ps, "holm")))


def test_an_unknown_correction_is_refused():
    ps = [pool(_cond("a", [(50, 10.0, 0.5)]), 1.0),
          pool(_cond("b", [(50, 20.0, 0.5)]), 1.0)]
    with pytest.raises(ValueError, match="unknown method"):
        pairwise(ps, "bonferroni-ish")


# -- the published dataset ----------------------------------------------------
