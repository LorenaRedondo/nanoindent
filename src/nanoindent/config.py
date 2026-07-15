"""Simulation configuration for AFM nanoindentation.

Unit system (consistent throughout the whole library)
------------------------------------------------------
* length              : nanometre           (nm)
* modulus / stress    : megapascal           (MPa)
* force               : piconewton           (pN)

The force unit follows from the other two::

    1 MPa * nm^2 = 1e6 Pa * (1e-9 m)^2 = 1e6 * 1e-18 N = 1e-12 N = 1 pN

so any force obtained by integrating a stress (MPa) over an area (nm^2)
comes out directly in pN.  As a sanity check, the Hertz estimate for the
reference problem (E* ~= 143 MPa, R = 17 nm, delta = 1 nm) gives

    F ~= (4/3) * 143 * sqrt(17) * 1 ~= 786 MPa*nm^2 ~= 786 pN

which is a sensible AFM force scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import warnings


# Conversion factor documented for downstream users: 1 MPa*nm^2 == 1 pN.
MPA_NM2_TO_PN = 1.0


@dataclass
class SimulationConfig:
    """All physical and numerical parameters for a single indentation run.

    Every quantity a solver needs lives here; there are no magic numbers in
    the solver code.  Values are validated in :meth:`__post_init__`.
    """

    # -- indenter ----------------------------------------------------------
    tip_shape: str = "sphere"
    """Indenter geometry (axisymmetric): ``"sphere"``, ``"paraboloid"``,
    ``"cone"`` or ``"blunted_cone"``.  Pyramids (Berkovich, Vickers,
    cube-corner) are non-axisymmetric — use a ``cone`` with ``tip_pyramid``
    set to the equivalent cone (see ``contact.equivalent_cone_half_angle``)."""

    tip_half_angle_deg: Optional[float] = None
    """Cone half-angle from the axis (deg), required for ``cone`` /
    ``blunted_cone`` unless ``tip_pyramid`` is given."""

    tip_pyramid: Optional[str] = None
    """If set (``"berkovich"``, ``"vickers"``, ``"cube_corner"``), a
    ``cone`` / ``blunted_cone`` tip uses the matching equivalent-cone
    half-angle instead of ``tip_half_angle_deg``."""

    # -- geometry (nm) -----------------------------------------------------
    R: float = 17.0
    """Tip radius (nm).  Used by sphere / paraboloid / blunted_cone; ignored by
    a sharp cone.  The rigid tip is never meshed."""

    h: float = 7.0
    """Film thickness (nm)."""

    L: float = 100.0
    """Lateral size (nm).  Interpreted as the domain *radius* by default
    (see the design doc, section 10)."""

    substrate: str = "bonded"
    """Bottom boundary condition. ``"bonded"`` fixes the film to the rigid
    substrate (``u_r = u_z = 0``); ``"unbonded"`` lets the film slide freely on
    a rigid, frictionless substrate (``u_z = 0``, ``u_r`` free). A bonded
    incompressible film is markedly stiffer than an unbonded one, which can
    shear material out along the substrate."""

    # -- material ----------------------------------------------------------
    E: float = 130.0
    """Young's modulus of the film (MPa)."""

    nu: float = 0.3
    """Poisson's ratio.  The reference paper does not specify a value; 0.3 is
    a documented default.  Values >= 0.45 trigger a locking warning because
    the P2 displacement formulation is not designed for near-incompressible
    behaviour (see design doc section 5.3)."""

    # -- loading -----------------------------------------------------------
    delta_max: float = 1.0
    """Maximum indentation depth (nm)."""

    n_steps: int = 20
    """Number of displacement-controlled load increments."""

    # -- meshing (nm) ------------------------------------------------------
    mesh_size_contact: float = 0.25
    """Target element size near the contact point (r=0, z=h)."""

    mesh_size_edge: float = 5.0
    """Target element size at the far lateral edge.  The paper quotes 0.1 nm
    which is unreasonably fine for a 100 nm domain; a saner default is used
    and the convergence study is the arbiter (design doc section 5.1)."""

    element_order: int = 2
    """Displacement element order.  2 (P2) is strongly recommended for
    contact accuracy and bending across a thin film."""

    formulation: str = "primal"
    """Elasticity formulation: ``"primal"`` (single-field P2 displacement,
    default) or ``"mixed"`` (Taylor-Hood P2 displacement / P1 pressure).  The
    mixed formulation is locking-free and supports fully incompressible
    ``nu = 0.5`` (needed to verify the Chadwick thin-incompressible-layer
    limit); use it whenever ``nu >= 0.45``."""

    # -- contact -----------------------------------------------------------
    penalty_alpha: float = 100.0
    """Dimensionless penalty factor.  Local penalty stiffness is
    ``k_pen = penalty_alpha * E / h_local`` where ``h_local`` is the local
    facet size."""

    # -- solver ------------------------------------------------------------
    newton_rtol: float = 1e-8
    """Relative residual tolerance for the Newton contact solve."""

    newton_max_it: int = 50
    """Maximum Newton iterations per load step."""

    equilibrium_tol: float = 0.02
    """Allowed relative mismatch between the contact-integrated force and the
    bottom-reaction force before a warning is emitted."""

    # -- bookkeeping -------------------------------------------------------
    label: Optional[str] = None
    """Optional human-readable label carried into results."""

    def __post_init__(self) -> None:
        self._validate()

    # -- validation --------------------------------------------------------
    def _validate(self) -> None:
        positive = {
            "R": self.R,
            "h": self.h,
            "L": self.L,
            "E": self.E,
            "delta_max": self.delta_max,
            "mesh_size_contact": self.mesh_size_contact,
            "mesh_size_edge": self.mesh_size_edge,
            "penalty_alpha": self.penalty_alpha,
        }
        for name, value in positive.items():
            if not (value > 0):
                raise ValueError(f"{name} must be > 0, got {value!r}")

        mixed = str(self.formulation).strip().lower() == "mixed"
        if mixed:
            # Taylor-Hood supports full incompressibility.
            if not (-1.0 < self.nu <= 0.5):
                raise ValueError(
                    f"nu must be in (-1, 0.5] for the mixed formulation, "
                    f"got {self.nu!r}"
                )
        else:
            if not (-1.0 < self.nu < 0.5):
                raise ValueError(
                    f"nu must be in (-1, 0.5), got {self.nu!r}. "
                    "nu = 0.5 (fully incompressible) needs formulation='mixed'."
                )
            if self.nu >= 0.45:
                warnings.warn(
                    f"nu = {self.nu} is near-incompressible; the primal P2 "
                    "formulation may lock and over-stiffen. Use "
                    "formulation='mixed' (design doc section 5.3).",
                    stacklevel=2,
                )
        if self.formulation not in ("primal", "mixed"):
            raise ValueError("formulation must be 'primal' or 'mixed'")
        if self.substrate not in ("bonded", "unbonded"):
            raise ValueError("substrate must be 'bonded' or 'unbonded'")

        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if self.element_order not in (1, 2):
            raise ValueError("element_order must be 1 or 2")

        # -- indenter --------------------------------------------------------
        shape = str(self.tip_shape).strip().lower()
        valid_shapes = {"sphere", "paraboloid", "cone", "blunted_cone"}
        if shape not in valid_shapes:
            raise ValueError(
                f"tip_shape must be one of {sorted(valid_shapes)}, "
                f"got {self.tip_shape!r}"
            )
        # Sphere/blunted_cone use an exact spherical cap: delta_max must stay
        # within the sphere (contact radius < R).
        if shape in ("sphere", "blunted_cone") and self.delta_max >= self.R:
            raise ValueError(
                "delta_max must be < R for the spherical gap function to be "
                f"well defined, got delta_max={self.delta_max}, R={self.R}"
            )
        if shape in ("cone", "blunted_cone"):
            if self.tip_pyramid is None and self.tip_half_angle_deg is None:
                raise ValueError(
                    f"tip_shape={shape!r} requires tip_half_angle_deg or "
                    "tip_pyramid"
                )
            if (self.tip_half_angle_deg is not None
                    and not (0.0 < self.tip_half_angle_deg < 90.0)):
                raise ValueError("tip_half_angle_deg must be in (0, 90)")
        if self.mesh_size_edge < self.mesh_size_contact:
            raise ValueError(
                "mesh_size_edge must be >= mesh_size_contact "
                f"({self.mesh_size_edge} < {self.mesh_size_contact})"
            )

    # -- derived quantities ------------------------------------------------
    @property
    def E_star(self) -> float:
        """Reduced modulus ``E / (1 - nu**2)`` (MPa).  Tip is rigid, so only
        the sample contributes."""
        return self.E / (1.0 - self.nu ** 2)

    @property
    def lame_lambda(self) -> float:
        """First Lame parameter (MPa)."""
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def lame_mu(self) -> float:
        """Second Lame parameter / shear modulus (MPa)."""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def expected_contact_radius(self) -> float:
        """Hertz estimate of the contact radius at ``delta_max`` (nm)."""
        return math.sqrt(self.R * self.delta_max)

    def delta_schedule(self) -> list[float]:
        """Return the list of indentation depths, excluding delta=0."""
        step = self.delta_max / self.n_steps
        return [step * (i + 1) for i in range(self.n_steps)]

    def summary(self) -> str:
        return (
            f"SimulationConfig(R={self.R} nm, h={self.h} nm, L={self.L} nm, "
            f"E={self.E} MPa, nu={self.nu}, E*={self.E_star:.1f} MPa, "
            f"delta_max={self.delta_max} nm, n_steps={self.n_steps})"
        )
