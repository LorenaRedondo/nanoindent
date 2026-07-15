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
from .contact import RigidSphere
from .geometry import build_mesh, Domain
from .solver import IndentationSolver
from .postproc import ForceCurve
from . import analytics
from . import convergence

__all__ = [
    "SimulationConfig",
    "MPA_NM2_TO_PN",
    "LinearElastic",
    "RigidSphere",
    "build_mesh",
    "Domain",
    "IndentationSolver",
    "ForceCurve",
    "analytics",
    "convergence",
]

__version__ = "0.1.0"
