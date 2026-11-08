"""Master-curve averaging and the DLVO electrostatic term.

All synthetic — these pin the behaviour the DOPC notebook relies on without
needing the 2.3 GB of raw curves.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoindent import analytics as A
from nanoindent import dlvo
from nanoindent import mastercurve as mc

R, NU, H = 20.0, 0.5, 6.5


def synthetic_set(n_curves=25, E=10.0, h_spread=0.5, noise=20.0, seed=0,
                  n_points=40, delta_max=3.4, depth_spread=0.0):
    """Curves from the bonded Dimitriadis model with scattered thickness.

    ``depth_spread`` varies how deep each curve goes, which is what real data
    does — every curve stops at its own ``0.8 * F_breakthrough`` — and is the
    reason the two binnings differ in coverage. Leave it 0 for a clean set.
    """
    rng = np.random.default_rng(seed)
    deltas, forces, heights = [], [], []
    for _ in range(n_curves):
        h = H + rng.uniform(-h_spread, h_spread)
        dmax = delta_max * (1.0 + rng.uniform(-depth_spread, depth_spread))
        d = np.linspace(0.15, dmax, n_points)
        f = A.dimitriadis_force(d, R, E, NU, h) + rng.normal(0, noise, d.shape)
        deltas.append(d); forces.append(f); heights.append(h)
    return mc.CurveSet(deltas=deltas, forces=forces, heights=heights, R=R)


# ------------------------------------------------------------------ chi ----
def test_chi_definition():
    d = np.array([1.0, 4.0])
    np.testing.assert_allclose(mc.chi_of(d, R, H), np.sqrt(R * d) / H)
    assert mc.chi_of(-1.0, R, H) == 0.0          # clipped, not imaginary


def test_curveset_validates_shapes():
    with pytest.raises(ValueError, match="same length"):
        mc.CurveSet(deltas=[np.zeros(3)], forces=[np.zeros(3), np.zeros(3)],
                    heights=[1.0], R=R)
    with pytest.raises(ValueError, match="at least one curve"):
        mc.CurveSet(deltas=[], forces=[], heights=[], R=R)


# -------------------------------------------------------------- binning ----
@pytest.mark.parametrize("binner", [mc.bin_by_force, mc.bin_by_chi])
def test_binning_recovers_the_generating_model(binner):
    """An average over noisy curves must sit on the curve they came from."""
    cs = synthetic_set(n_curves=40, E=10.0, noise=15.0)
    m = binner(cs, n_bins=20)
    v = m.valid(3)
    assert v.sum() >= 8
    expected = A.dimitriadis_force(m.delta[v], R, 10.0, NU, float(np.median(cs.heights)))
    rel = np.abs(m.force[v] - expected) / np.maximum(expected, 1.0)
    assert np.median(rel) < 0.12


def test_coverage_counts_distinct_curves_per_bin():
    """`coverage` is the fraction of the set contributing, not of the points."""
    cs = synthetic_set(n_curves=20, depth_spread=0.0, noise=0.0)
    m = mc.bin_by_force(cs, n_bins=10)
    v = m.valid(3)
    assert np.all(m.n_curves[v] <= len(cs))
    np.testing.assert_allclose(m.coverage()[v],
                               m.n_curves[v] / m.n_curves.max())
    # identical delta grids -> every curve reaches every bin
    assert np.median(m.coverage()[v]) == pytest.approx(1.0)


def test_staggered_start_depths_thin_out_the_low_chi_bins():
    """The mechanism behind the notebook's binning choice.

    chi ~ sqrt(delta), so a bin at small chi is a narrow slice of indentation.
    Once curves start at different depths — as real ones do, since each has its
    own contact point and force floor — the low-chi bins are the ones that lose
    curves, not the tail.
    """
    rng = np.random.default_rng(1)
    deltas, forces, heights = [], [], []
    for _ in range(40):
        h = H + rng.uniform(-0.4, 0.4)
        start = rng.uniform(0.15, 1.6)           # staggered contact regions
        d = np.linspace(start, 3.4, 40)
        deltas.append(d)
        forces.append(A.dimitriadis_force(d, R, 10.0, NU, h))
        heights.append(h)
    cs = mc.CurveSet(deltas=deltas, forces=forces, heights=heights, R=R)

    by_chi = mc.bin_by_chi(cs, n_bins=20)
    v = by_chi.valid(3)
    chi, cov = by_chi.chi[v], by_chi.coverage()[v]
    low, high = chi < np.median(chi), chi >= np.median(chi)
    assert cov[low].mean() < cov[high].mean(), (
        f"low-chi coverage {cov[low].mean():.2f} should be worse than "
        f"high-chi {cov[high].mean():.2f}")


def test_error_bars_use_curve_count_not_point_count():
    """20 points from one curve are not 20 independent measurements."""
    cs = synthetic_set(n_curves=12, noise=50.0)
    m = mc.bin_by_force(cs, n_bins=10)
    v = m.valid(3)
    assert np.all(m.n_points[v] >= m.n_curves[v])
    np.testing.assert_allclose(m.force_sem[v],
                               m.force_std[v] / np.sqrt(m.n_curves[v]))


def test_compare_binnings_reports_a_difference_and_coverage():
    cs = synthetic_set(n_curves=30)
    out = mc.compare_binnings(cs, n_bins=20)
    assert 0 <= out["median_rel_diff"] < 0.5
    assert 0 <= out["min_coverage_chi_binned"] <= 1
    assert out["by_force"].binned_on == "force"
    assert out["by_chi"].binned_on == "chi"


def test_log_spaced_force_bins():
    cs = synthetic_set(n_curves=15)
    m = mc.bin_by_force(cs, n_bins=15, log_spaced=True)
    widths = np.diff(m.edges)
    assert widths[-1] > widths[0]                # geometric spacing


# ----------------------------------------------------------------- DLVO ----
def test_debye_length_matches_the_standard_shortcut():
    assert dlvo.debye_length(1.0) == pytest.approx(9.61, rel=2e-3)
    assert dlvo.debye_length(100.0) == pytest.approx(0.961, rel=2e-3)
    # divalent screens twice as hard at the same molarity
    assert dlvo.debye_length(10, valence=2) == pytest.approx(
        dlvo.debye_length(10) / 2, rel=1e-9)
    with pytest.raises(ValueError):
        dlvo.debye_length(0)


def test_dlvo_force_decays_with_the_gap_and_saturates():
    d = np.array([0.0, H / 2, H, H + 1.0])
    f = dlvo.dlvo_force(d, H, F0=1000.0, lambda_D=1.0)
    assert f[0] < f[1] < f[2]                    # grows as the gap closes
    assert f[2] == pytest.approx(1000.0)         # gap 0 -> F0
    assert f[3] == pytest.approx(1000.0)         # past the substrate: saturates
    with pytest.raises(ValueError):
        dlvo.dlvo_force(d, H, 1.0, lambda_D=0.0)


def test_recovers_planted_dlvo_parameters():
    """Ground truth in, ground truth out — over a window that spans the decay."""
    E_true, F0_true, lam_true = 8.0, 900.0, 1.2
    d = np.linspace(H - 5 * lam_true, H - 0.2, 90)
    f = dlvo.dlvo_total_force(d, R, E_true, NU, H, F0_true, lam_true)
    fit = dlvo.fit_dlvo(d, f, R, NU, H)
    assert fit.converged
    assert fit.E == pytest.approx(E_true, rel=1e-3)
    assert fit.F0 == pytest.approx(F0_true, rel=1e-3)
    assert fit.lambda_D == pytest.approx(lam_true, rel=1e-3)
    assert fit.r_squared > 0.999


def test_pure_elastic_data_does_not_earn_the_dlvo_term():
    """The null case: no electrostatics planted, so AICc must not prefer it."""
    d = np.linspace(0.2, 3.4, 80)
    f = A.dimitriadis_force(d, R, 10.0, NU, H)
    out = dlvo.compare_with_and_without_dlvo(d, f, R, NU, H)
    assert out["elastic"].E == pytest.approx(10.0, rel=1e-6)
    assert not (out["dlvo"].well_constrained() and out["delta_aicc"] < -10)


def test_far_from_the_surface_the_term_is_unidentifiable():
    """The notebook's headline: a flat exponential cannot be determined.

    With the gap several Debye lengths wide the term is effectively a constant
    offset, so with realistic noise the fit may converge but its parameters are
    not pinned down. (Noiseless data recovers it exactly — the degeneracy is
    between the *signal* and the noise, not a defect of the optimiser.)
    """
    lam = 0.5
    d = np.linspace(0.2, H - 4.0, 60)            # gap stays >= 4 nm = 8 lambda
    clean = A.dimitriadis_force(d, R, 10.0, NU, H) + dlvo.dlvo_force(
        d, H, 500.0, lam)
    noisy = clean + np.random.default_rng(0).normal(0, 20.0, d.shape)

    exact = dlvo.fit_dlvo(d, clean, R, NU, H)
    assert exact.lambda_D == pytest.approx(lam, rel=1e-3)   # noiseless: fine

    # Over this window the planted term is worth well under a pN — far below
    # the noise — so it is simply not there to be found.
    planted = float(dlvo.dlvo_force(d.max(), H, 500.0, lam))
    assert planted < 1.0

    out = dlvo.compare_with_and_without_dlvo(d, noisy, R, NU, H)
    assert out["dlvo"].lambda_D != pytest.approx(lam, rel=0.5), \
        "the planted screening length should not be recoverable here"
    assert out["delta_aicc"] > -2, \
        f"AICc should not favour an invisible term: {out['verdict']}"


def test_relative_uncertainty_and_constraint_flag():
    fit = dlvo.DLVOFit(model="m", E=10.0, F0=100.0, lambda_D=1.0,
                       E_std=0.1, F0_std=5.0, lambda_D_std=0.02,
                       converged=True)
    rel = fit.relative_uncertainty
    assert rel["E"] == pytest.approx(0.01)
    assert rel["lambda_D"] == pytest.approx(0.02)
    assert fit.well_constrained()
    fit.lambda_D_std = 8.0                        # 800% uncertainty
    assert not fit.well_constrained()


def test_fit_dlvo_reports_a_railed_bound_rather_than_hiding_it():
    d = np.linspace(0.2, 3.0, 40)
    f = A.dimitriadis_force(d, R, 10.0, NU, H)
    fit = dlvo.fit_dlvo(d, f, R, NU, H, lambda_bounds=(9.99, 10.0))
    if np.isfinite(fit.lambda_D):
        assert not fit.converged
        assert "railed" in fit.message


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 5"):
        dlvo.fit_dlvo(np.array([1.0, 2.0]), np.array([1.0, 2.0]), R, NU, H)


def test_dlvo_works_on_a_master_curve():
    """The path the notebook actually takes: bin, then fit."""
    cs = synthetic_set(n_curves=30, noise=25.0)
    m = mc.bin_by_force(cs, n_bins=20)
    v = m.valid(3)
    out = dlvo.compare_with_and_without_dlvo(m.delta[v], m.force[v], R, NU,
                                             float(np.median(cs.heights)))
    assert out["elastic"].E == pytest.approx(10.0, rel=0.15)
    assert isinstance(out["verdict"], str) and out["verdict"]


# --------------------------------------------------- six-model registry ----
def test_six_models_are_all_registered_and_ordered_physically():
    reg = A.model_registry()
    for key in A.SIX_MODELS:
        assert key in reg, key
    cfg = type("C", (), dict(R=R, E=10.0, nu=NU, h=H))()
    d = np.array([2.0])
    forces = {k: float(reg[k].func(d, cfg)[0]) for k in A.SIX_MODELS}
    # at fixed E, more confinement -> more force
    assert forces["hertz"] < forces["dimitriadis_free"]
    assert forces["dimitriadis_free"] < forces["dimitriadis_bonded"]


def test_the_two_chadwick_asymptotes_cross():
    """They scale differently, so neither is uniformly the stiffer one.

    Bonded goes as delta^3/h^3, unbonded as delta^2/h, so they swap order at
    ``delta = (3/4) h^2 / (R (1 - nu^2))``. Asserting a fixed ordering would be
    wrong; this pins the crossover instead.
    """
    d_cross = 0.75 * H ** 2 / (R * (1 - NU ** 2))
    for d, expect_bonded_larger in ((d_cross * 0.5, False),
                                    (d_cross * 2.0, True)):
        bonded = float(A.chadwick_force(d, R, 10.0, H))
        unbonded = float(A.chadwick_unbonded_force(d, R, 10.0, NU, H))
        assert (bonded > unbonded) is expect_bonded_larger, f"at delta={d:.2f}"
    at_cross_b = float(A.chadwick_force(d_cross, R, 10.0, H))
    at_cross_u = float(A.chadwick_unbonded_force(d_cross, R, 10.0, NU, H))
    assert at_cross_b == pytest.approx(at_cross_u, rel=1e-9)

