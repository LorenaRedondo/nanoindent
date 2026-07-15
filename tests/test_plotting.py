"""Smoke tests for the publication plotting module (Agg backend, no display)."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from nanoindent import SimulationConfig, plotting
from nanoindent.postproc import ForceCurve


def _fake_curve(h, E=130.0):
    cfg = SimulationConfig(h=h, E=E, nu=0.3, delta_max=1.0, n_steps=5)
    d = np.array([0.0] + cfg.delta_schedule())
    # Monotone synthetic force ~ delta^1.5 so plotting has real data.
    f = 200.0 * d ** 1.5
    return ForceCurve(delta=d, force=f,
                      force_bottom=f, contact_radius=np.sqrt(cfg.R * d),
                      max_penetration=0.001 * d, cfg=cfg, n_dofs=100)


def test_set_style_idempotent():
    plotting.set_publication_style()
    plotting.set_publication_style(font_size=9)
    import matplotlib as mpl
    assert mpl.rcParams["pdf.fonttype"] == 42


def test_biophysical_journal_preset():
    plotting.set_publication_style(journal="biophysical_journal")
    import matplotlib as mpl
    # Arial-first font stack, 7 pt base, ~85/174 mm columns.
    assert mpl.rcParams["font.sans-serif"][0] == "Arial"
    assert mpl.rcParams["font.size"] == 7
    w = plotting.journal_widths("biophysical_journal")
    assert w["single"] == pytest.approx(85 / 25.4, rel=1e-6)
    assert w["double"] == pytest.approx(174 / 25.4, rel=1e-6)


def test_panel_label_case_by_journal():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plotting.set_publication_style(journal="biophysical_journal")
    plotting.panel_label(ax, 0)
    assert any(t.get_text() == "A" for t in ax.texts)      # Cell Press uppercase
    fig2, ax2 = plt.subplots()
    plotting.set_publication_style(journal="generic")
    plotting.panel_label(ax2, 0)
    assert any(t.get_text() == "(a)" for t in ax2.texts)   # generic lowercase


def test_plot_force_curves_runs():
    curves = {7.0: _fake_curve(7.0), 14.0: _fake_curve(14.0)}
    fig = plotting.plot_force_curves(curves, compare=["hertz", "dimitriadis_bonded"])
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel()
    assert len(ax.lines) > 0


def test_plot_stiffening_ratio_runs():
    curves = {7.0: _fake_curve(7.0), 14.0: _fake_curve(14.0)}
    fig = plotting.plot_stiffening_ratio(curves, compare=["dimitriadis_bonded"])
    assert len(fig.axes[0].lines) > 0


def test_plot_model_comparison_runs():
    cfg = SimulationConfig(h=7.0, R=17.0, E=130.0, nu=0.3, delta_max=1.0)
    fig = plotting.plot_model_comparison(cfg, fem_curve=_fake_curve(7.0))
    ax = fig.axes[0]
    labels = [ln.get_label() for ln in ax.lines]
    assert any("Chadwick" in str(l) for l in labels)
    assert any("Hertz" in str(l) for l in labels)


def test_plot_modulus_recovery_runs():
    results = [
        {"h": 7.0, "model": "hertz", "E_fit": 267.0},
        {"h": 7.0, "model": "dimitriadis", "E_fit": 134.0},
        {"h": 14.0, "model": "hertz", "E_fit": 182.0},
        {"h": 14.0, "model": "dimitriadis", "E_fit": 130.0},
    ]
    fig = plotting.plot_modulus_recovery(results, true_E=130.0)
    assert len(fig.axes[0].patches) == 4  # 2 thicknesses x 2 models


def test_save_figure_writes_files(tmp_path):
    curves = {7.0: _fake_curve(7.0)}
    fig = plotting.plot_force_curves(curves, compare=["hertz"])
    stem = str(tmp_path / "fig")
    paths = plotting.save_figure(fig, stem, formats=("pdf", "png", "svg"))
    assert len(paths) == 3
    for p in paths:
        import os
        assert os.path.exists(p) and os.path.getsize(p) > 0
