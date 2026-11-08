"""Post-processing: the ForceCurve container and helpers.

A :class:`ForceCurve` bundles the force-indentation data produced by a run
together with the config that generated it, and knows how to save itself to
CSV and plot itself against the analytic benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import csv

import numpy as np

from .config import SimulationConfig
from . import analytics


@dataclass
class ForceCurve:
    """Force-indentation result of an indentation run.

    Attributes
    ----------
    delta:
        Indentation depths (nm), including the leading 0.
    force:
        Reaction force on the indenter at each depth (pN), integrated from the
        contact pressure.
    force_bottom:
        Independent reaction force integrated over the bonded bottom boundary
        (pN), used as an equilibrium cross-check.
    contact_radius:
        Contact radius a(delta) (nm) at each depth.
    max_penetration:
        Maximum interpenetration [-g]_+ (nm) at each depth (penalty quality).
    cfg:
        The configuration that produced the curve.
    """

    delta: np.ndarray
    force: np.ndarray
    force_bottom: np.ndarray = field(default_factory=lambda: np.array([]))
    contact_radius: np.ndarray = field(default_factory=lambda: np.array([]))
    max_penetration: np.ndarray = field(default_factory=lambda: np.array([]))
    cfg: Optional[SimulationConfig] = None
    n_dofs: Optional[int] = None
    newton_log: list = field(default_factory=list)
    """Per-load-step Newton diagnostics (iterations, convergence, residual)."""

    # -- applicability and numerical quality -------------------------------
    def applicability(self):
        """Model assumptions, dimensionless ratios and applicability warnings.

        See :func:`nanoindent.quality.applicability`.
        """
        if self.cfg is None:
            raise ValueError("cfg is required to assess applicability")
        from .quality import applicability
        return applicability(self.cfg, self)

    def quality_report(self):
        """Numerical-quality report for this run (see :mod:`nanoindent.quality`)."""
        from .quality import quality_report
        return quality_report(self)

    # -- diagnostics -------------------------------------------------------
    @property
    def equilibrium_residual(self) -> np.ndarray:
        """Relative |F_contact - F_bottom| / max(F_contact) per step."""
        if self.force_bottom.size == 0:
            return np.array([])
        scale = np.maximum(np.abs(self.force), 1e-30)
        return np.abs(self.force - self.force_bottom) / scale

    def stiffening_ratio(self) -> np.ndarray:
        """F_FEM / F_Hertz at each depth (finite-thickness stiffening)."""
        if self.cfg is None:
            raise ValueError("cfg is required to compute the Hertz reference")
        fh = analytics.hertz_force(self.delta, self.cfg.R, self.cfg.E, self.cfg.nu)
        ratio = np.full_like(self.force, np.nan)
        nz = fh > 0
        ratio[nz] = self.force[nz] / fh[nz]
        return ratio

    # -- IO ----------------------------------------------------------------
    def save(self, path: str) -> None:
        """Write the curve to CSV (delta, force, and available diagnostics)."""
        cols = {"delta_nm": self.delta, "force_pN": self.force}
        if self.force_bottom.size:
            cols["force_bottom_pN"] = self.force_bottom
        if self.contact_radius.size:
            cols["contact_radius_nm"] = self.contact_radius
        if self.max_penetration.size:
            cols["max_penetration_nm"] = self.max_penetration
        keys = list(cols)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(keys)
            for i in range(len(self.delta)):
                writer.writerow([f"{cols[k][i]:.10g}" for k in keys])

    @classmethod
    def load(cls, path: str) -> "ForceCurve":
        import numpy as np

        data = {}
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            keys = reader.fieldnames or []
            for k in keys:
                data[k] = []
            for row in reader:
                for k in keys:
                    data[k].append(float(row[k]))
        arr = {k: np.array(v) for k, v in data.items()}
        return cls(
            delta=arr.get("delta_nm", np.array([])),
            force=arr.get("force_pN", np.array([])),
            force_bottom=arr.get("force_bottom_pN", np.array([])),
            contact_radius=arr.get("contact_radius_nm", np.array([])),
            max_penetration=arr.get("max_penetration_nm", np.array([])),
        )

    # -- plotting ----------------------------------------------------------
    def plot(self, compare: Optional[Sequence[str]] = None, ax=None,
             save: Optional[str] = None, show: bool = False):
        """Plot F(delta) with optional analytic overlays.

        ``compare`` is a list of model keys from
        :func:`nanoindent.analytics.model_registry` (e.g. ``["hertz",
        "dimitriadis_bonded", "garcia_bec"]``).
        """
        import matplotlib
        if save is not None and not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4.5))

        ax.plot(self.delta, self.force, "o-", color="k", lw=1.8, ms=4,
                label="FEM", zorder=5)

        if compare and self.cfg is not None:
            reg = analytics.model_registry()
            dd = np.linspace(0, float(self.delta.max()), 200)
            for key in compare:
                if key not in reg:
                    continue
                model = reg[key]
                ax.plot(dd, model.func(dd, self.cfg), "--", lw=1.4,
                        label=model.label)

        ax.set_xlabel("indentation $\\delta$ (nm)")
        ax.set_ylabel("force $F$ (pN)")
        title = "Force-indentation"
        if self.cfg is not None:
            title += (f"  (h={self.cfg.h} nm, R={self.cfg.R} nm, "
                      f"E={self.cfg.E} MPa, $\\nu$={self.cfg.nu})")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if save is not None:
            ax.figure.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax
