"""nanoindent — FEM simulation of AFM nanoindentation of thin elastic films.

Open-source (scikit-fem / dolfinx) replacement for the COMSOL setup of a rigid
spherical tip indenting a linear-elastic film bonded to a rigid substrate.

Quick start
-----------
>>> from nanoindent import SimulationConfig, IndentationSolver
>>> cfg = SimulationConfig(h=7.0, nu=0.3)
>>> result = IndentationSolver(cfg).run()
>>> result.plot(compare=["hertz", "dimitriadis_bonded", "garcia_bec"])
"""

from .config import SimulationConfig, MPA_NM2_TO_PN
from .material import LinearElastic
from .contact import (
    RigidIndenter, RigidSphere, Paraboloid, Cone, BluntedCone,
    make_tip, equivalent_cone_half_angle,
)
from .geometry import build_mesh, Domain
from .solver import IndentationSolver
from .postproc import ForceCurve
from . import analytics
from . import convergence
from . import dataset
from . import plotting
from .dataset import synthetic_force_curve, fit_modulus, SyntheticDataset, ModulusFit

__all__ = [
    "SimulationConfig",
    "MPA_NM2_TO_PN",
    "LinearElastic",
    "RigidIndenter",
    "RigidSphere",
    "Paraboloid",
    "Cone",
    "BluntedCone",
    "make_tip",
    "equivalent_cone_half_angle",
    "build_mesh",
    "Domain",
    "IndentationSolver",
    "ForceCurve",
    "analytics",
    "convergence",
    "dataset",
    "plotting",
    "synthetic_force_curve",
    "fit_modulus",
    "SyntheticDataset",
    "ModulusFit",
]

__version__ = "0.1.0"
