"""Axisymmetric mesh generation for the film-on-substrate problem.

The design doc targets gmsh + dolfinx.  In environments where dolfinx cannot
be installed we fall back to scikit-fem (pure pip), and gmsh is not required
either: the domain here is a simple graded rectangle in (r, z), which is built
directly as a tensor-product triangle mesh with a size field clustered at the
contact point ``(r=0, z=h)``.

Boundaries are tagged consistently with the physics spec:

* ``axis``   : r = 0        (axisymmetry, u_r = 0)
* ``edge``   : r = L        (traction-free lateral edge)
* ``bottom`` : z = 0        (bonded to rigid substrate, u = 0)
* ``top``    : z = h        (contact surface)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from skfem import MeshTri

from .config import SimulationConfig


def _graded_coords(x0: float, x1: float, focus: float,
                   size_min: float, size_max: float,
                   growth: float = 1.3) -> np.ndarray:
    """1-D node coordinates on ``[x0, x1]`` with element size growing from
    ``size_min`` at ``focus`` to at most ``size_max`` far away.

    The spacing at distance ``d`` from ``focus`` is
    ``min(size_max, size_min * growth**(number of elements so far))`` so that
    elements coarsen geometrically away from the focus.  ``focus`` must be one
    of the endpoints (the contact problem only needs one-sided grading).
    """
    if focus not in (x0, x1):
        raise ValueError("focus must be an endpoint of the interval")

    reverse = focus == x1
    lo, hi = (x0, x1)
    span = hi - lo

    # Build from the focus outward, then map back onto [lo, hi].
    dist = 0.0
    dists = [0.0]
    size = size_min
    while dist < span:
        dist = min(span, dist + size)
        dists.append(dist)
        size = min(size_max, size * growth)
    dists = np.array(dists)
    # Guard against a final sliver element.
    if len(dists) > 2 and (dists[-1] - dists[-2]) < 0.4 * (dists[-2] - dists[-3]):
        dists = np.delete(dists, -2)

    if reverse:
        coords = hi - dists[::-1]
    else:
        coords = lo + dists
    coords[0], coords[-1] = lo, hi
    return np.unique(coords)


@dataclass
class Domain:
    """A tagged axisymmetric mesh plus the geometry it was built from."""

    mesh: MeshTri
    L: float
    h: float

    @property
    def n_vertices(self) -> int:
        return self.mesh.p.shape[1]

    @property
    def n_elements(self) -> int:
        return self.mesh.t.shape[1]


def build_mesh(cfg: SimulationConfig) -> Domain:
    """Build a graded, boundary-tagged axisymmetric mesh from ``cfg``.

    Radial grading is clustered at the axis (the contact point sits at r=0)
    and coarsens toward ``r = L``.  Vertical grading is clustered at the top
    surface ``z = h`` and coarsens toward the bonded bottom, guaranteeing at
    least a few elements across even a 7 nm film.
    """
    L, h = cfg.L, cfg.h

    # Radial: fine within roughly the expected contact radius, coarse outward.
    r = _graded_coords(
        0.0, L, focus=0.0,
        size_min=cfg.mesh_size_contact,
        size_max=cfg.mesh_size_edge,
    )
    # Vertical: fine at the top (contact) surface, coarser (but still capped so
    # a thin film keeps several layers) toward the bottom.
    size_max_z = min(cfg.mesh_size_edge, max(cfg.mesh_size_contact, h / 4.0))
    z = _graded_coords(
        0.0, h, focus=h,
        size_min=cfg.mesh_size_contact,
        size_max=size_max_z,
    )

    mesh = MeshTri.init_tensor(r, z)

    eps_r = 1e-9 * L
    eps_z = 1e-9 * h
    mesh = mesh.with_boundaries(
        {
            "axis": lambda p: np.abs(p[0]) < eps_r,
            "edge": lambda p: np.abs(p[0] - L) < eps_r,
            "bottom": lambda p: np.abs(p[1]) < eps_z,
            "top": lambda p: np.abs(p[1] - h) < eps_z,
        }
    )
    return Domain(mesh=mesh, L=L, h=h)
