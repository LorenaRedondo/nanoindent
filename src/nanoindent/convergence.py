"""Convergence and verification utilities (mirrors the paper's procedure).

Three studies, each returning a list of records that can be inspected or
plotted:

1. mesh study    -- refine ``mesh_size_contact`` and watch F(delta_max);
2. penalty study -- vary ``penalty_alpha`` and watch F plateau / interpenetration;
3. load-step study -- vary ``n_steps`` (elastic problem is path-independent).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
import copy

import numpy as np

from .config import SimulationConfig
from .solver import IndentationSolver


@dataclass
class ConvergenceRecord:
    param_name: str
    param_value: float
    force_max: float
    n_dofs: int
    max_penetration: float
    equilibrium_residual: float


def _run(cfg: SimulationConfig) -> ConvergenceRecord:
    solver = IndentationSolver(cfg)
    curve = solver.run()
    res = curve.equilibrium_residual
    return ConvergenceRecord(
        param_name="",
        param_value=0.0,
        force_max=float(curve.force[-1]),
        n_dofs=int(curve.n_dofs or 0),
        max_penetration=float(curve.max_penetration[-1])
        if curve.max_penetration.size else 0.0,
        equilibrium_residual=float(np.nanmax(res)) if res.size else 0.0,
    )


def mesh_study(cfg: SimulationConfig,
               sizes=(0.5, 0.25, 0.125, 0.0625)) -> list[ConvergenceRecord]:
    """Refine ``mesh_size_contact`` (edge size scaled proportionally).

    Converged when successive change in F(delta_max) < 0.5%.
    """
    records = []
    ratio = cfg.mesh_size_edge / cfg.mesh_size_contact
    for s in sizes:
        c = replace(cfg, mesh_size_contact=s, mesh_size_edge=s * ratio)
        rec = _run(c)
        rec.param_name = "mesh_size_contact"
        rec.param_value = s
        records.append(rec)
    return records


def penalty_study(cfg: SimulationConfig,
                  alphas=(10.0, 100.0, 1000.0, 10000.0)) -> list[ConvergenceRecord]:
    """Vary ``penalty_alpha``; force should plateau and interpenetration drop
    below 1% of delta."""
    records = []
    for a in alphas:
        c = replace(cfg, penalty_alpha=a)
        rec = _run(c)
        rec.param_name = "penalty_alpha"
        rec.param_value = a
        records.append(rec)
    return records


def load_step_study(cfg: SimulationConfig,
                    n_steps_list=(5, 20, 50)) -> list[ConvergenceRecord]:
    """Vary ``n_steps``; the final force must be step-independent for an
    elastic (path-independent) problem."""
    records = []
    for n in n_steps_list:
        c = replace(cfg, n_steps=n)
        rec = _run(c)
        rec.param_name = "n_steps"
        rec.param_value = n
        records.append(rec)
    return records


def relative_changes(records: list[ConvergenceRecord]) -> np.ndarray:
    """Successive relative changes in ``force_max`` across a record list."""
    f = np.array([r.force_max for r in records])
    if len(f) < 2:
        return np.array([])
    return np.abs(np.diff(f)) / np.maximum(np.abs(f[:-1]), 1e-30)


def is_converged(records: list[ConvergenceRecord], tol: float = 0.005) -> bool:
    ch = relative_changes(records)
    return bool(ch.size and ch[-1] < tol)
