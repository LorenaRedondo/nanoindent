"""Linear viscoelastic AFM indentation via the correspondence principle.

Where the elastic solver answers "what force at this depth", this module answers
"what force at this depth *and this rate*" -- the frequency dependence that makes
a soft-matter sample read stiffer when probed faster. It is what turns a single
force curve into dynamic mechanical spectroscopy: storage and loss moduli
``E'(omega)``, ``E''(omega)``.

The bridge is the same linearity the force-volume module rests on. Elastically,

    F(delta) = E * phi(delta),

with ``phi`` the modulus-independent geometric kernel from one FEM solve (see
:mod:`nanoindent.forcemap`). The elastic-viscoelastic **correspondence
principle** (Lee & Radok 1960) then replaces the constant modulus by a
hereditary convolution with the material's relaxation function ``E(t)``:

    F(t) = integral_0^t  E(t - t') d/dt'[ phi(delta(t')) ] dt'.

This is exact for a linear viscoelastic solid during monotonic loading (growing
contact), and it inherits the FEM kernel's geometry -- finite thickness, bottom
effect, exact sphere -- which the textbook Hertz-based viscoelastic formulae do
not carry. Drive ``delta(t)`` sinusoidally and the complex modulus falls out;
this module validates that extraction against the closed-form ``E*(omega)`` of
the relaxation model, keeping to the library's "validate against closed forms"
standard.

Conventions: ``E(t)`` and moduli are Young's moduli (MPa), Poisson's ratio held
constant; time in seconds, angular frequency ``omega`` in rad/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# relaxation models: E(t) in the time domain, E*(omega) in closed form
# --------------------------------------------------------------------------
@dataclass
class RelaxationModel:
    """A linear viscoelastic relaxation modulus as a Prony series.

        E(t) = E_inf + sum_k  E_k exp(-t / tau_k)

    ``E_inf`` is the equilibrium (long-time, "rubbery") modulus; each
    ``(E_k, tau_k)`` is a relaxing branch. The instantaneous ("glassy") modulus
    is ``E_0 = E_inf + sum_k E_k``. All moduli in MPa, times in s.
    """

    E_inf: float
    weights: np.ndarray          # E_k (MPa), shape (n_branches,)
    taus: np.ndarray             # tau_k (s), shape (n_branches,)

    def __post_init__(self):
        self.weights = np.atleast_1d(np.asarray(self.weights, dtype=float))
        self.taus = np.atleast_1d(np.asarray(self.taus, dtype=float))
        if self.weights.shape != self.taus.shape:
            raise ValueError("weights and taus must have the same length")
        if np.any(self.taus <= 0):
            raise ValueError("relaxation times must be > 0")
        if self.E_inf < 0 or np.any(self.weights < 0):
            raise ValueError("moduli must be >= 0")

    @property
    def E0(self) -> float:
        """Instantaneous (glassy) modulus (MPa)."""
        return float(self.E_inf + self.weights.sum())

    def E_relax(self, t: np.ndarray) -> np.ndarray:
        """Relaxation modulus ``E(t)`` (MPa) for ``t >= 0``."""
        t = np.asarray(t, dtype=float)
        out = np.full(t.shape, self.E_inf, dtype=float)
        for Ek, tk in zip(self.weights, self.taus):
            out = out + Ek * np.exp(-np.clip(t, 0.0, None) / tk)
        return out

    def E_complex(self, omega: np.ndarray) -> np.ndarray:
        """Closed-form complex modulus ``E*(omega) = E'(omega) + i E''(omega)`` (MPa).

        From the Fourier transform of the Prony relaxation function:

            E*(omega) = E_inf + sum_k E_k (i omega tau_k) / (1 + i omega tau_k).
        """
        omega = np.asarray(omega, dtype=float)
        out = np.full(omega.shape, self.E_inf, dtype=complex)
        for Ek, tk in zip(self.weights, self.taus):
            x = omega * tk
            out = out + Ek * (1j * x) / (1.0 + 1j * x)
        return out

    def storage(self, omega: np.ndarray) -> np.ndarray:
        """Storage modulus ``E'(omega)`` (MPa)."""
        return np.real(self.E_complex(omega))

    def loss(self, omega: np.ndarray) -> np.ndarray:
        """Loss modulus ``E''(omega)`` (MPa)."""
        return np.imag(self.E_complex(omega))

    def loss_tangent(self, omega: np.ndarray) -> np.ndarray:
        """``tan(delta) = E'' / E'``."""
        c = self.E_complex(omega)
        return np.imag(c) / np.real(c)


def standard_linear_solid(E0: float, E_inf: float, tau: float) -> RelaxationModel:
    """Standard Linear Solid (Zener): one relaxing branch.

    ``E0`` instantaneous, ``E_inf`` equilibrium, ``tau`` relaxation time (s).
    """
    if E0 <= E_inf:
        raise ValueError("E0 (instantaneous) must exceed E_inf (equilibrium)")
    return RelaxationModel(E_inf=E_inf, weights=np.array([E0 - E_inf]),
                           taus=np.array([tau]))


def generalized_maxwell(E_inf: float, weights: Sequence[float],
                        taus: Sequence[float]) -> RelaxationModel:
    """Generalized Maxwell / Prony series with several relaxation branches."""
    return RelaxationModel(E_inf=E_inf, weights=np.asarray(weights, float),
                           taus=np.asarray(taus, float))


# --------------------------------------------------------------------------
# hereditary (Lee-Radok) force from a geometric kernel and an indentation history
# --------------------------------------------------------------------------
def hereditary_force(
    t: np.ndarray,
    delta_of_t: np.ndarray,
    relax: RelaxationModel,
    phi: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Viscoelastic indentation force ``F(t)`` by the correspondence principle.

    Evaluates ``F(t) = integral_0^t E(t - t') dg/dt' dt'`` with ``g(t) =
    phi(delta(t))`` the geometric response. Uses the exact recurrence for a Prony
    relaxation kernel: each branch carries an internal variable updated in O(1)
    per step assuming ``g`` is piecewise linear over a step, so the whole history
    costs O(N) and is exact for that assumption (no growing convolution sum).

    Parameters
    ----------
    t:
        Monotone increasing time samples (s), starting at 0.
    delta_of_t:
        Indentation history ``delta(t)`` (nm), same shape as ``t``.
    relax:
        Relaxation model supplying ``E_inf`` and the Prony branches.
    phi:
        Geometric kernel ``delta -> force/E`` (pN/MPa), e.g.
        ``GeometricKernel.phi_of`` from a single FEM solve.

    Returns
    -------
    force:
        ``F(t)`` (pN), same shape as ``t``. Valid while contact grows
        (``delta`` non-decreasing); see Ting (1966) for the receding-contact
        correction on unloading, which is out of scope here.
    """
    t = np.asarray(t, dtype=float)
    g = np.asarray(phi(np.asarray(delta_of_t, dtype=float)), dtype=float)
    n = t.size
    force = np.zeros(n)
    # Internal variable per Prony branch: h_k(t) = int_0^t E_k e^{-(t-t')/tau_k} g'(t') dt'.
    h = np.zeros(relax.weights.size)
    force[0] = relax.E_inf * g[0]        # g(0) step contributes the equilibrium part
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        dg = g[i] - g[i - 1]
        for k, (Ek, tk) in enumerate(zip(relax.weights, relax.taus)):
            r = dt / tk
            decay = np.exp(-r)
            # exact for g linear over the step: analytic integral of the ramp
            coeff = (1.0 - decay) / r if r > 1e-8 else 1.0 - 0.5 * r
            h[k] = decay * h[k] + Ek * coeff * dg
        force[i] = relax.E_inf * g[i] + h.sum()
    return force


# --------------------------------------------------------------------------
# time-domain experiments
# --------------------------------------------------------------------------
def ramp_hold_curve(
    relax: RelaxationModel,
    phi: Callable[[np.ndarray], np.ndarray],
    delta_max: float,
    approach_velocity_nm_s: float,
    hold_s: float,
    n_ramp: int = 200,
    n_hold: int = 400,
):
    """Force-relaxation curve: ramp to ``delta_max`` then hold.

    Returns ``(t, delta, force)``. During the hold the indentation is constant
    and the force relaxes from the glassy toward the equilibrium response --
    the AFM stress-relaxation measurement, and a direct read on ``tau``.
    """
    if approach_velocity_nm_s <= 0:
        raise ValueError("approach_velocity must be > 0")
    t_ramp_end = delta_max / approach_velocity_nm_s
    t_r = np.linspace(0.0, t_ramp_end, n_ramp)
    d_r = approach_velocity_nm_s * t_r
    t_h = np.linspace(t_ramp_end, t_ramp_end + hold_s, n_hold + 1)[1:]
    d_h = np.full(n_hold, float(delta_max))
    t = np.concatenate([t_r, t_h])
    d = np.concatenate([d_r, d_h])
    f = hereditary_force(t, d, relax, phi)
    return t, d, f


# --------------------------------------------------------------------------
# frequency-domain: extract E'(omega), E''(omega) from oscillatory indentation
# --------------------------------------------------------------------------
@dataclass
class DynamicSpectrum:
    """Result of a frequency sweep."""

    omega: np.ndarray            # rad/s
    storage: np.ndarray          # E'(omega), recovered (MPa)
    loss: np.ndarray             # E''(omega), recovered (MPa)
    storage_exact: np.ndarray    # analytic E'(omega) (MPa)
    loss_exact: np.ndarray       # analytic E''(omega) (MPa)

    @property
    def freq_hz(self) -> np.ndarray:
        return self.omega / (2.0 * np.pi)

    def max_rel_error(self) -> float:
        """Largest relative error of recovered vs analytic complex modulus."""
        c = self.storage + 1j * self.loss
        ce = self.storage_exact + 1j * self.loss_exact
        return float(np.max(np.abs(c - ce) / np.abs(ce)))


def _project_sinusoid(t: np.ndarray, y: np.ndarray, omega: float):
    """In-phase (sin) and quadrature (cos) amplitudes of ``y`` at ``omega``.

    Least-squares projection onto {1, sin(omega t), cos(omega t)} over the given
    (whole-period) window. Returns ``(a_sin, a_cos)``.
    """
    A = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1]), float(coef[2])


def frequency_sweep(
    relax: RelaxationModel,
    phi: Callable[[np.ndarray], np.ndarray],
    dphi_ddelta: Callable[[float], float],
    omegas: Sequence[float],
    delta0: float,
    delta_amp: float,
    n_periods: int = 12,
    steps_per_period: int = 200,
) -> DynamicSpectrum:
    """Recover ``E'(omega)``, ``E''(omega)`` from small-amplitude oscillation.

    Drives ``delta(t) = delta0 + delta_amp sin(omega t)`` (with
    ``delta_amp << delta0`` so the geometric response is locally linear),
    computes ``F(t)`` by the hereditary integral, and projects the steady-state
    force (final period) onto sine and cosine. Dividing by the local geometric
    slope ``dphi/ddelta(delta0)`` -- the dynamic contact stiffness per unit
    modulus -- converts the measured force amplitudes into the material's
    storage and loss moduli, which are compared to the analytic ``E*(omega)``.

    Parameters
    ----------
    dphi_ddelta:
        Local slope ``dphi/ddelta`` at ``delta0`` (pN/MPa/nm). For a
        :class:`~nanoindent.forcemap.GeometricKernel` use a finite difference of
        ``phi_of`` (see :func:`kernel_slope`).
    """
    slope = float(dphi_ddelta(delta0))
    if slope <= 0:
        raise ValueError("dphi/ddelta at delta0 must be > 0")

    Es, Ls = [], []
    for omega in omegas:
        period = 2.0 * np.pi / omega
        n = int(n_periods * steps_per_period)
        t = np.linspace(0.0, n_periods * period, n + 1)
        d = delta0 + delta_amp * np.sin(omega * t)
        f = hereditary_force(t, d, relax, phi)
        # Project the last whole period (steady state) onto sin/cos.
        last = t >= (n_periods - 1) * period
        a_sin, a_cos = _project_sinusoid(t[last], f[last], omega)
        # F_osc = slope * delta_amp * [E' sin(wt) + E'' cos(wt)]
        Es.append(a_sin / (slope * delta_amp))
        Ls.append(a_cos / (slope * delta_amp))

    omega_arr = np.asarray(list(omegas), dtype=float)
    return DynamicSpectrum(
        omega=omega_arr,
        storage=np.asarray(Es), loss=np.asarray(Ls),
        storage_exact=relax.storage(omega_arr),
        loss_exact=relax.loss(omega_arr),
    )


def kernel_slope(phi: Callable[[np.ndarray], np.ndarray], delta0: float,
                 eps: float = 1e-3) -> float:
    """Central-difference ``dphi/ddelta`` at ``delta0`` (pN/MPa/nm)."""
    d = np.array([delta0 - eps, delta0 + eps])
    p = np.asarray(phi(d), dtype=float)
    return float((p[1] - p[0]) / (2.0 * eps))
