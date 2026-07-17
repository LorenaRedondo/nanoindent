"""Recovery between consecutive AFM indentations, and its limits.

When an AFM does force-volume or fast scanning it indents, retracts, steps a few
nm laterally, and indents again. On an elastic sample that is fine: the surface
springs back the instant the tip lifts. On a *viscoelastic* sample it is not --
the material recovers over a relaxation time, so if the next poke comes too soon
(fast scan) or too close (small lateral step), it lands on material still dented
by the previous one. Consecutive measurements then overlap and are not
independent. This module quantifies where that happens.

It combines two ingredients, each honest:

  * the **spatial footprint** `s(r)` -- how the surface depression spreads
    laterally, taken from a single real FEM solve (so it carries the
    finite-thickness/bottom effect: a bonded thin film localises the dent far
    more than a half-space would);

  * the **temporal recovery** -- a first-order (single retardation time `tau`)
    viscoelastic surface response, `tau dA/dt + A = D(t)`, driven by the tip
    while in contact and relaxing freely toward zero once the tip separates.

The axisymmetric solver cannot place two laterally-separated indents in one
solve, so the lateral raster is built by **linear superposition** of the FEM
footprint at each site -- valid in the same small-strain linear regime the rest
of the library uses. The result is a reduced-order model, not a 3-D solve, and
is labelled as such.

Conventions: lengths nm, time s, `tau` s. Depression `A` is positive-downward
(nm); surface height is `-A`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .config import SimulationConfig
from .solver import IndentationSolver


# --------------------------------------------------------------------------
# spatial footprint from one FEM solve
# --------------------------------------------------------------------------
@dataclass
class SurfaceFootprint:
    """Normalised surface-depression profile ``s(r) = u_z(r) / u_z(0)``.

    ``s(0) = 1`` at the indentation axis and decays with lateral distance ``r``;
    ``contact_radius`` is the Hertz estimate ``√(Rδ)`` for reference.
    """

    r: np.ndarray                 # nm, monotone increasing from 0
    s: np.ndarray                 # dimensionless, s(0)=1
    contact_radius: float
    delta_max: float
    R: float
    h: float

    def shape(self, dx) -> np.ndarray:
        """Footprint at lateral offset ``dx`` (nm); 0 beyond the solved extent."""
        d = np.abs(np.asarray(dx, dtype=float))
        out = np.interp(d, self.r, self.s, left=self.s[0], right=0.0)
        return out


def surface_footprint(cfg: SimulationConfig, n_steps: int = 6) -> SurfaceFootprint:
    """Extract the surface-depression footprint from one FEM solve.

    Solves the indentation to ``cfg.delta_max`` and reads the vertical
    displacement of the top (contact) surface, normalised to the axis value.
    """
    from dataclasses import replace

    solve_cfg = replace(cfg, n_steps=n_steps)
    solver = IndentationSolver(solve_cfg)
    solver.run(store_history=True)
    mesh = solver.domain.mesh
    r, z = mesh.p[0], mesh.p[1]
    uz = solver.nodal_field("uz", u=solver._last_u)
    top = np.abs(z - solve_cfg.h) < 1e-6
    rt, uzt = r[top], uz[top]
    o = np.argsort(rt)
    rt, uzt = rt[o], uzt[o]
    # Collapse duplicate radial nodes (P2 edge nodes share r) by averaging.
    rt_u, inv = np.unique(np.round(rt, 9), return_inverse=True)
    uz_u = np.zeros_like(rt_u)
    np.add.at(uz_u, inv, uzt)
    counts = np.bincount(inv)
    uz_u /= counts
    s = uz_u / uz_u[0]
    a = float(np.sqrt(solve_cfg.R * solve_cfg.delta_max))
    return SurfaceFootprint(r=rt_u, s=s, contact_radius=a,
                            delta_max=solve_cfg.delta_max, R=solve_cfg.R,
                            h=solve_cfg.h)


# --------------------------------------------------------------------------
# lateral raster of down-up insertions, with viscoelastic recovery
# --------------------------------------------------------------------------
@dataclass
class RasterRecovery:
    """Time-resolved surface of a lateral raster of down-up insertions."""

    x: np.ndarray                 # lateral grid (nm)
    t: np.ndarray                 # time samples (s)
    A: np.ndarray                 # depression A[t, x] (nm, positive down)
    tip_x: np.ndarray             # tip lateral position at each time (nm)
    tip_delta: np.ndarray         # tip indentation depth at each time (nm)
    in_contact: np.ndarray        # bool, tip touching at each time
    site_x: np.ndarray            # lateral position of each insertion (nm)
    start_idx: np.ndarray         # time index at which each insertion starts
    tau: float
    pitch: float
    ramp_freq_hz: float
    delta_max: float
    footprint: SurfaceFootprint

    def residual_at_starts(self) -> np.ndarray:
        """Residual depression at each new site *when its insertion begins*,
        as a fraction of the fresh indentation depth (excludes that site's own,
        not-yet-started poke)."""
        out = np.zeros(len(self.site_x))
        for i, (xi, k) in enumerate(zip(self.site_x, self.start_idx)):
            j = int(np.argmin(np.abs(self.x - xi)))
            out[i] = self.A[max(k - 1, 0), j] / self.delta_max
        return out


def raster_surface(
    footprint: SurfaceFootprint,
    tau: float,
    pitch_nm: float,
    ramp_freq_hz: float,
    n_insertions: int = 8,
    contact_fraction: float = 0.5,
    viscous_fraction: float = 1.0,
    steps_per_cycle: int = 60,
    x_pad: float = 3.0,
    dx: float = 0.5,
    delta_max: Optional[float] = None,
) -> RasterRecovery:
    """Integrate the viscoelastic surface response for a line of down-up pokes.

    The tip visits ``x_i = i·pitch`` in turn, one full approach-retract per
    ``1/ramp_freq_hz`` seconds (``contact_fraction`` of that in contact). The
    surface follows an **imprint-then-recover** model, appropriate when contact
    is fast (the material deforms with its glassy modulus) and recovery is the
    slow process: on approach the surface follows the tip elastically
    (``A = δ(t)·s``); once the tip peaks and lifts, a fraction ``1 −
    viscous_fraction`` springs back instantly (elastic) and the remaining
    ``viscous_fraction·δ_max·s`` relaxes freely, ``∝ exp(−(t−t_peak)/τ)``.

    ``viscous_fraction`` β sets *how viscoelastic* the material is: β = 0 is
    purely elastic (springs back completely, no residual ever); β = 1 is fully
    viscoelastic (nothing springs back instantly). It maps to a Standard Linear
    Solid as β = 1 − E_inf/E0 (the relaxation strength). The full surface is the
    linear superposition over all pokes, so a new poke lands on whatever residual
    earlier ones have left. Time-domain companion to :func:`residual_fraction`.

    Returns a :class:`RasterRecovery`.
    """
    beta = float(viscous_fraction)
    dmax = footprint.delta_max if delta_max is None else delta_max
    period = 1.0 / ramp_freq_hz
    contact_time = contact_fraction * period
    half = 0.5 * contact_time                       # approach (and retract) half

    site_x = np.arange(n_insertions) * pitch_nm
    x_lo = site_x[0] - x_pad * footprint.contact_radius
    x_hi = site_x[-1] + x_pad * footprint.contact_radius
    x = np.arange(x_lo, x_hi + dx, dx)

    dt = period / steps_per_cycle
    n_t = int(np.ceil(n_insertions * period / dt)) + 1
    t = np.arange(n_t) * dt

    site_start = np.arange(n_insertions) * period
    site_peak = site_start + half
    # Precompute each site's footprint column, s(x - x_i).
    foot = np.array([footprint.shape(x - xi) for xi in site_x])   # (n_ins, nx)

    A = np.zeros((n_t, x.size))
    tip_x = np.zeros(n_t)
    tip_delta = np.zeros(n_t)
    in_contact = np.zeros(n_t, dtype=bool)
    start_idx = np.round(site_start / dt).astype(int)

    for it in range(n_t):
        tnow = t[it]
        acc = np.zeros(x.size)
        for j in range(n_insertions):
            if tnow < site_start[j]:
                continue
            if tnow <= site_peak[j]:
                # approach: surface follows the tip down, elastically
                fj = dmax * (tnow - site_start[j]) / half
            else:
                # peak reached: (1-beta) springs back instantly, beta recovers
                fj = dmax * beta * np.exp(-(tnow - site_peak[j]) / tau)
            acc += fj * foot[j]
        A[it] = acc
        # active tip (the one currently in its contact window), for drawing
        i = min(int(tnow / period), n_insertions - 1)
        tau_in = tnow - site_start[i]
        if tau_in <= contact_time:
            frac = tau_in / contact_time
            tip_delta[it] = dmax * (1.0 - abs(2.0 * frac - 1.0))
            in_contact[it] = tip_delta[it] > 1e-9
        tip_x[it] = site_x[i]

    return RasterRecovery(
        x=x, t=t, A=A, tip_x=tip_x, tip_delta=tip_delta, in_contact=in_contact,
        site_x=site_x, start_idx=start_idx, tau=tau, pitch=pitch_nm,
        ramp_freq_hz=ramp_freq_hz, delta_max=dmax, footprint=footprint,
    )


# --------------------------------------------------------------------------
# the recovery limit: residual overlap as a function of pitch and speed
# --------------------------------------------------------------------------
def residual_fraction(footprint: SurfaceFootprint, tau: float,
                      pitch_nm: float, ramp_freq_hz: float,
                      contact_fraction: float = 0.5,
                      viscous_fraction: float = 1.0,
                      n_history: int = 60) -> float:
    """Steady-state residual at a new site when its poke starts, as a fraction
    of the fresh indentation depth.

    In a steady raster the poke ``k`` sites back peaked ``k/f − half`` seconds ago
    (``half`` the contact half-time) at lateral distance ``k·pitch``; only the
    viscous part ``β`` of its imprint persists, and it has recovered by
    ``exp(−(k/f − half)/τ)``, so its leftover is
    ``β·s(k·pitch)·exp(−(k/f − half)/τ)``. Summing over history gives the overlap
    the next measurement sees -- the closed-form companion to
    :func:`raster_surface` (they agree). ``viscous_fraction`` β = 1 − E_inf/E0
    sets how viscoelastic the material is (β = 0 → purely elastic → no overlap).
    The boundary ``residual ≈ 0.1`` is a reasonable "not recovered" threshold.
    """
    period = 1.0 / ramp_freq_hz
    half = 0.5 * contact_fraction * period
    k = np.arange(1, n_history + 1)
    elapsed = np.maximum(k * period - half, 0.0)
    contrib = footprint.shape(k * pitch_nm) * np.exp(-elapsed / tau)
    return float(viscous_fraction * np.sum(contrib))


def recovery_limit_map(footprint: SurfaceFootprint, tau: float,
                       pitches: Sequence[float],
                       freqs: Sequence[float],
                       viscous_fraction: float = 1.0) -> np.ndarray:
    """Residual-fraction map over a grid of lateral pitch × ramp frequency.

    Rows index ``pitches`` (nm), columns ``freqs`` (Hz). High values = the next
    poke lands on unrecovered material; low = independent measurements.
    ``viscous_fraction`` scales the whole map (a more elastic material overlaps
    less everywhere).
    """
    pitches = np.asarray(pitches, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    out = np.zeros((pitches.size, freqs.size))
    for ip, p in enumerate(pitches):
        for jf, f in enumerate(freqs):
            out[ip, jf] = residual_fraction(footprint, tau, p, f,
                                            viscous_fraction=viscous_fraction)
    return out
