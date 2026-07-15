import numpy as np
import pytest

from nanoindent import SimulationConfig, build_mesh


@pytest.fixture
def domain():
    cfg = SimulationConfig(h=7.0, L=100.0, mesh_size_contact=0.5,
                           mesh_size_edge=10.0)
    return build_mesh(cfg)


def test_boundary_tags_present(domain):
    for name in ("axis", "edge", "bottom", "top"):
        assert name in domain.mesh.boundaries
        assert domain.mesh.boundaries[name].size > 0


def test_boundary_locations(domain):
    mesh = domain.mesh
    p = mesh.p

    def facet_midpoints(name):
        facets = mesh.boundaries[name]
        v = mesh.facets[:, facets]
        return 0.5 * (p[:, v[0]] + p[:, v[1]])

    assert np.allclose(facet_midpoints("axis")[0], 0.0)
    assert np.allclose(facet_midpoints("edge")[0], domain.L)
    assert np.allclose(facet_midpoints("bottom")[1], 0.0)
    assert np.allclose(facet_midpoints("top")[1], domain.h)


def test_grading_finer_near_contact(domain):
    # Near the axis/top corner elements should be smaller than near r=L.
    mesh = domain.mesh
    p = mesh.p
    r = p[0]
    # coordinate spacing along r near 0 vs near L
    ru = np.unique(np.round(r, 8))
    d = np.diff(ru)
    assert d[0] < d[-1]  # graded coarser outward


def test_thin_film_has_multiple_layers():
    cfg = SimulationConfig(h=7.0, mesh_size_contact=0.5, mesh_size_edge=10.0)
    dom = build_mesh(cfg)
    z = np.unique(np.round(dom.mesh.p[1], 8))
    assert len(z) >= 4  # at least a few layers across the film
