import warnings
import pytest

from nanoindent import SimulationConfig


def test_defaults_and_derived():
    cfg = SimulationConfig()
    assert cfg.R == 17.0 and cfg.E == 130.0
    assert cfg.E_star == pytest.approx(130.0 / (1 - 0.3 ** 2))
    assert cfg.lame_mu > 0 and cfg.lame_lambda > 0
    assert cfg.expected_contact_radius == pytest.approx((17.0 * 1.0) ** 0.5)


def test_delta_schedule():
    cfg = SimulationConfig(delta_max=1.0, n_steps=4)
    sched = cfg.delta_schedule()
    assert sched == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_invalid_parameters():
    with pytest.raises(ValueError):
        SimulationConfig(h=-1.0)
    with pytest.raises(ValueError):
        SimulationConfig(nu=0.6)
    with pytest.raises(ValueError):
        SimulationConfig(delta_max=20.0, R=17.0)  # delta_max >= R
    with pytest.raises(ValueError):
        SimulationConfig(mesh_size_edge=0.1, mesh_size_contact=1.0)


def test_incompressible_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SimulationConfig(nu=0.49)
    assert any("incompressible" in str(x.message) for x in w)
