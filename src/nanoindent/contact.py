"""Rigid indenters: gap functions and penalty contact terms.

The model is 2D **axisymmetric**, so the indenter must be a surface of
revolution.  Four exact axisymmetric shapes are provided:

* :class:`RigidSphere`      -- exact sphere (no parabolic approximation)
* :class:`Paraboloid`       -- ``z = r**2 / (2R)`` (the small-strain Hertz tip)
* :class:`Cone`             -- sharp cone of a given half-angle
* :class:`BluntedCone`      -- spherical cap of radius ``R`` blended C1 into a
  cone (a realistic AFM tip)

A **pyramid** (Berkovich, Vickers, cube-corner) is *not* axisymmetric and
cannot be meshed in a 2D axisymmetric model.  The accepted workaround is an
**equivalent cone** whose half-angle gives the same projected-area-vs-depth
law; :func:`equivalent_cone_half_angle` returns those angles, e.g.::

    tip = Cone(equivalent_cone_half_angle("berkovich"), h=h)

Geometry convention (all lengths nm).  Every tip starts tangent to / apex on
the undeformed top surface ``z = h`` at ``r = 0``.  After indenting by
``delta`` the apex sits at ``z = h - delta``.  Each tip implements

    surface_z(r, delta) -> lower surface height of the indenter at radius r

(returning ``+inf`` where the tip cannot make contact), and the signed gap and
penetration follow the common convention

    g(r) = surface_z(r) - (h + u_z),   penetration = [-g]_+ = max(-g, 0).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import numpy as np


# Equivalent-cone half-angles (measured from the indentation axis, degrees)
# that reproduce the projected-area-vs-depth function of common pyramids.
EQUIVALENT_CONE_HALF_ANGLE_DEG = {
    "berkovich": 70.3,
    "vickers": 70.3,
    "cube_corner": 42.3,
    "cube-corner": 42.3,
}


def equivalent_cone_half_angle(pyramid: str) -> float:
    """Half-angle (deg, from the axis) of the cone equivalent to ``pyramid``.

    Pyramidal tips are non-axisymmetric; the returned cone reproduces the same
    area-depth relation and is the standard axisymmetric stand-in.
    """
    key = pyramid.strip().lower()
    if key not in EQUIVALENT_CONE_HALF_ANGLE_DEG:
        raise ValueError(
            f"unknown pyramid {pyramid!r}; known: "
            f"{sorted(set(EQUIVALENT_CONE_HALF_ANGLE_DEG))}"
        )
    return EQUIVALENT_CONE_HALF_ANGLE_DEG[key]


class RigidIndenter(ABC):
    """Base class for rigid axisymmetric indenters."""

    def __init__(self, h: float) -> None:
        self.h = float(h)

    def apex_z(self, delta: float) -> float:
        """Height of the indenter apex at indentation ``delta``."""
        return self.h - delta

    @abstractmethod
    def surface_z(self, r: np.ndarray, delta: float) -> np.ndarray:
        """Lower surface height of the indenter at radius ``r`` (``+inf`` where
        contact is impossible)."""

    # -- shared gap / penetration machinery --------------------------------
    def gap(self, r: np.ndarray, uz: np.ndarray, delta: float) -> np.ndarray:
        """Signed gap ``g = surface_z(r) - (h + u_z)`` (negative = penetration)."""
        return self.surface_z(r, delta) - (self.h + np.asarray(uz, dtype=float))

    def penetration(self, r: np.ndarray, uz: np.ndarray, delta: float) -> np.ndarray:
        """Penetration depth ``[-g]_+ = max(-g, 0)`` (>= 0)."""
        g = self.gap(r, uz, delta)
        pen = np.where(np.isfinite(g), -g, -np.inf)
        return np.maximum(pen, 0.0)

    def is_active(self, r: np.ndarray, uz: np.ndarray, delta: float) -> np.ndarray:
        """Boolean mask of quadrature points currently in contact."""
        return self.penetration(r, uz, delta) > 0.0


class RigidSphere(RigidIndenter):
    """Rigid sphere of radius ``R`` (exact gap, not parabolic)."""

    def __init__(self, R: float, h: float) -> None:
        super().__init__(h)
        if not (R > 0):
            raise ValueError("R must be > 0")
        self.R = float(R)

    def center_z(self, delta: float) -> float:
        """Vertical coordinate of the sphere centre."""
        return self.h + self.R - delta

    def surface_z(self, r: np.ndarray, delta: float) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        out = np.full_like(r, np.inf)
        inside = r < self.R
        rr = r[inside]
        out[inside] = self.center_z(delta) - np.sqrt(self.R ** 2 - rr ** 2)
        return out


class Paraboloid(RigidIndenter):
    """Paraboloidal tip ``z = r**2 / (2R)`` (radius of curvature ``R`` at apex).

    This is the parabolic approximation of a sphere; for small ``r`` it matches
    :class:`RigidSphere` and it is the tip underlying the classic Hertz law.
    """

    def __init__(self, R: float, h: float) -> None:
        super().__init__(h)
        if not (R > 0):
            raise ValueError("R must be > 0")
        self.R = float(R)

    def surface_z(self, r: np.ndarray, delta: float) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return self.apex_z(delta) + r ** 2 / (2.0 * self.R)


class Cone(RigidIndenter):
    """Sharp cone of half-angle ``half_angle_deg`` measured from the axis.

    The cone surface is ``z = apex + r / tan(half_angle)``; a larger half-angle
    is a blunter cone.  Use :func:`equivalent_cone_half_angle` to stand in for
    a pyramidal tip.
    """

    def __init__(self, half_angle_deg: float, h: float) -> None:
        super().__init__(h)
        if not (0.0 < half_angle_deg < 90.0):
            raise ValueError("half_angle_deg must be in (0, 90)")
        self.half_angle_deg = float(half_angle_deg)
        self.beta = math.radians(self.half_angle_deg)

    def surface_z(self, r: np.ndarray, delta: float) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return self.apex_z(delta) + r / math.tan(self.beta)


class BluntedCone(RigidIndenter):
    """Spherical cap of radius ``R`` blended C1 into a cone of half-angle
    ``half_angle_deg`` (from the axis) -- a realistic AFM tip.

    The sphere covers ``r <= R*cos(beta)``; beyond the tangent radius the
    surface is conical.  Reduces to a sphere as ``half_angle -> 90`` and to a
    sharp cone as ``R -> 0``.
    """

    def __init__(self, R: float, half_angle_deg: float, h: float) -> None:
        super().__init__(h)
        if not (R > 0):
            raise ValueError("R must be > 0")
        if not (0.0 < half_angle_deg < 90.0):
            raise ValueError("half_angle_deg must be in (0, 90)")
        self.R = float(R)
        self.half_angle_deg = float(half_angle_deg)
        self.beta = math.radians(self.half_angle_deg)
        self.r_transition = self.R * math.cos(self.beta)

    def center_z(self, delta: float) -> float:
        return self.h + self.R - delta

    def surface_z(self, r: np.ndarray, delta: float) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        z_c = self.center_z(delta)
        r_t = self.r_transition
        # Spherical cap for r <= r_t, cone (C1-matched) beyond.
        sphere = z_c - np.sqrt(np.maximum(self.R ** 2 - np.minimum(r, self.R) ** 2, 0.0))
        cone = z_c - self.R / math.sin(self.beta) + r / math.tan(self.beta)
        return np.where(r <= r_t, sphere, cone)


def make_tip(cfg) -> RigidIndenter:
    """Construct the indenter described by a :class:`SimulationConfig`."""
    shape = getattr(cfg, "tip_shape", "sphere")
    key = str(shape).strip().lower()
    if key == "sphere":
        return RigidSphere(cfg.R, cfg.h)
    if key == "paraboloid":
        return Paraboloid(cfg.R, cfg.h)
    if key == "cone":
        angle = _resolve_half_angle(cfg)
        return Cone(angle, cfg.h)
    if key in ("blunted_cone", "blunted-cone", "bluntedcone"):
        angle = _resolve_half_angle(cfg)
        return BluntedCone(cfg.R, angle, cfg.h)
    raise ValueError(
        f"unknown tip_shape {shape!r}; supported: sphere, paraboloid, cone, "
        "blunted_cone"
    )


def _resolve_half_angle(cfg) -> float:
    """Return the cone half-angle for ``cfg``: an explicit angle, or the
    equivalent-cone angle of ``cfg.tip_pyramid`` if that is set."""
    pyramid = getattr(cfg, "tip_pyramid", None)
    if pyramid:
        return equivalent_cone_half_angle(pyramid)
    angle = getattr(cfg, "tip_half_angle_deg", None)
    if angle is None:
        raise ValueError(
            "cone/blunted_cone tips require tip_half_angle_deg or tip_pyramid"
        )
    return float(angle)


def penalty_stiffness(alpha: float, E: float, h_local: np.ndarray) -> np.ndarray:
    """Local penalty stiffness ``k_pen = alpha * E / h_local`` (MPa/nm)."""
    return alpha * E / np.asarray(h_local, dtype=float)
