"""Rigid spherical indenter: gap function and penalty contact terms.

Geometry (all lengths nm).  The sphere of radius ``R`` starts tangent to the
undeformed top surface ``z = h`` at ``r = 0``.  After indenting by ``delta``
its centre sits at::

    z_c = h + R - delta

The exact (non-parabolic) lower surface of the sphere at radius ``r`` is
``z_sphere(r) = z_c - sqrt(R**2 - r**2)`` for ``r < R`` and is treated as
``+inf`` for ``r >= R`` (no contact possible).

The signed gap used throughout follows the design doc::

    g(r) = z_sphere(r) - (h + u_z(r))

so material *penetration* corresponds to ``g < 0`` and the penetration depth
is the Macaulay bracket ``[-g]_+ = max(-g, 0)``.
"""

from __future__ import annotations

import numpy as np


class RigidSphere:
    """Rigid spherical indenter represented purely by its gap function."""

    def __init__(self, R: float, h: float) -> None:
        if not (R > 0):
            raise ValueError("R must be > 0")
        self.R = float(R)
        self.h = float(h)

    def center_z(self, delta: float) -> float:
        """Vertical coordinate of the sphere centre at indentation ``delta``."""
        return self.h + self.R - delta

    def surface_z(self, r: np.ndarray, delta: float) -> np.ndarray:
        """Lower surface height ``z_sphere(r)`` of the sphere.

        Returns ``+inf`` where ``r >= R`` (the sphere does not extend there).
        """
        r = np.asarray(r, dtype=float)
        inside = r < self.R
        out = np.full_like(r, np.inf)
        rr = np.clip(r[inside], 0.0, self.R)
        out[inside] = self.center_z(delta) - np.sqrt(self.R ** 2 - rr ** 2)
        return out

    def gap(self, r: np.ndarray, uz: np.ndarray, delta: float) -> np.ndarray:
        """Signed gap ``g(r) = z_sphere(r) - (h + u_z)``.

        Negative values mean the deformed surface has penetrated the sphere.
        """
        return self.surface_z(r, delta) - (self.h + np.asarray(uz, dtype=float))

    def penetration(self, r: np.ndarray, uz: np.ndarray, delta: float) -> np.ndarray:
        """Penetration depth ``[-g]_+ = max(-g, 0)`` (>= 0)."""
        g = self.gap(r, uz, delta)
        pen = np.where(np.isfinite(g), -g, -np.inf)
        return np.maximum(pen, 0.0)

    def is_active(self, r: np.ndarray, uz: np.ndarray, delta: float) -> np.ndarray:
        """Boolean mask of quadrature points currently in contact."""
        return self.penetration(r, uz, delta) > 0.0


def penalty_stiffness(alpha: float, E: float, h_local: np.ndarray) -> np.ndarray:
    """Local penalty stiffness ``k_pen = alpha * E / h_local`` (MPa/nm).

    ``h_local`` is the local facet size; using a per-facet value keeps the
    penalty consistent under mesh refinement.
    """
    return alpha * E / np.asarray(h_local, dtype=float)
