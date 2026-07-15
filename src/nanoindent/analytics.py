"""Closed-form benchmarks for spherical AFM indentation.

All functions take indentation ``delta`` (nm) and return force in pN using the
library's unit convention (1 MPa*nm^2 == 1 pN), so they can be overlaid on FEM
results directly.

References
---------
* Hertz (1882) elastic half-space contact of a sphere.
* Dimitriadis, Horkay, Maresca, Kachar, Chadwick, *Biophys. J.* **82**, 2798
  (2002): finite-thickness (bottom-effect) corrections for a spherical tip on
  a soft film bonded to / free from a rigid substrate.
* Garcia & Garcia, *Biophys. J.* **114**, 2923 (2018): bottom-effect
  correction (BEC) for paraboloidal tips, expressed as a polynomial in the
  reduced variable ``sqrt(R*delta)/h``.

The bottom-effect models are only valid in the small-indentation / small
contact-radius regime (``a <~ h`` is comfortable; the polynomials degrade as
``sqrt(R*delta)/h`` grows past ~1).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def hertz_force(delta, R: float, E: float, nu: float) -> np.ndarray:
    """Hertz force for a rigid sphere on an elastic half-space.

    ``F = (4/3) * E_star * sqrt(R) * delta**1.5`` with ``E_star = E/(1-nu^2)``.
    Returns pN for delta in nm, R in nm, E in MPa.
    """
    delta = np.asarray(delta, dtype=float)
    E_star = E / (1.0 - nu ** 2)
    F = (4.0 / 3.0) * E_star * np.sqrt(R) * np.clip(delta, 0.0, None) ** 1.5
    return F


# -- Dimitriadis 2002 --------------------------------------------------------
# General-nu polynomial coefficients (Dimitriadis et al. 2002).  The force is
#   F = (16/9) * E * sqrt(R) * delta^1.5 * P(chi),   chi = sqrt(R*delta)/h
# where the (16/9) E prefactor equals (4/3) E_star for the incompressible
# limit nu -> 1/2.  We keep the E_star prefactor and fold the nu-dependence
# into alpha0/beta0 so the models reduce to Hertz as h -> inf for any nu.

def _dimitriadis_alpha_beta(nu: float, bonded: bool) -> tuple[float, float]:
    """Return the ``(alpha0, beta0)`` coefficients of Dimitriadis 2002."""
    if bonded:
        alpha0 = -(1.2876 - 1.4678 * nu + 1.3442 * nu ** 2) / (1.0 - nu)
        beta0 = (0.6387 - 1.0277 * nu + 1.5164 * nu ** 2) / (1.0 - nu)
    else:  # free (not bonded) top film on a frictionless rigid substrate
        alpha0 = -(0.347 - 0.056 * nu + 0.0 * nu ** 2)
        beta0 = (0.1636 - 0.0 * nu)
    return alpha0, beta0


def dimitriadis_force(delta, R: float, E: float, nu: float, h: float,
                      bonded: bool = True) -> np.ndarray:
    """Finite-thickness force from the Dimitriadis 2002 correction.

    Parameters
    ----------
    bonded:
        ``True`` for a film bonded to the substrate (default; matches the
        reference problem), ``False`` for a frictionless / free film.

    The correction multiplies the Hertz force by a quartic polynomial in
    ``chi = sqrt(R*delta)/h``.
    """
    delta = np.asarray(delta, dtype=float)
    chi = np.sqrt(R * np.clip(delta, 0.0, None)) / h
    alpha0, beta0 = _dimitriadis_alpha_beta(nu, bonded)

    poly = (
        1.0
        - (2.0 * alpha0 / np.pi) * chi
        + (4.0 * alpha0 ** 2 / np.pi ** 2) * chi ** 2
        - (8.0 / np.pi ** 3)
        * (alpha0 ** 3 + (4.0 * np.pi ** 2 / 15.0) * beta0) * chi ** 3
        + (16.0 * alpha0 / np.pi ** 4)
        * (alpha0 ** 3 + (3.0 * np.pi ** 2 / 5.0) * beta0) * chi ** 4
    )
    return hertz_force(delta, R, E, nu) * poly


# Widely tabulated incompressible (nu -> 1/2) bonded coefficients.  These are
# the numbers most AFM codes hard-code; exposed for reference / cross-check.
DIMITRIADIS_BONDED_INCOMPRESSIBLE = (1.0, 1.133, 1.497, 1.469, 0.755)


def dimitriadis_bonded_incompressible(delta, R: float, E: float,
                                      h: float) -> np.ndarray:
    """Bonded, incompressible (nu = 1/2) Dimitriadis force using the classic
    tabulated coefficients ``[1, 1.133, 1.497, 1.469, 0.755]``.

    ``F = (16/9) E sqrt(R) delta^1.5 * sum(c_k chi^k)``, ``chi=sqrt(R delta)/h``.
    """
    delta = np.asarray(delta, dtype=float)
    chi = np.sqrt(R * np.clip(delta, 0.0, None)) / h
    c = DIMITRIADIS_BONDED_INCOMPRESSIBLE
    poly = c[0] + c[1] * chi + c[2] * chi ** 2 + c[3] * chi ** 3 + c[4] * chi ** 4
    prefactor = (16.0 / 9.0) * E * np.sqrt(R) * np.clip(delta, 0.0, None) ** 1.5
    return prefactor * poly


# -- Garcia & Garcia 2018 ----------------------------------------------------
# BEC polynomial for a paraboloidal (spherical, small-indentation) tip on a
# bonded sample.  Same tabulated coefficients as the incompressible bonded
# Dimitriadis result; the Garcia formulation keeps the E_star prefactor so it
# applies for arbitrary nu.
GARCIA_BONDED_COEFFS = (1.0, 1.133, 1.497, 1.469, 0.755)


def garcia_bec_force(delta, R: float, E: float, nu: float, h: float) -> np.ndarray:
    """Garcia & Garcia 2018 bottom-effect-corrected force (bonded, paraboloid).

    ``F = F_Hertz(delta) * sum_k c_k * (sqrt(R delta)/h)^k`` with the bonded
    coefficients ``[1, 1.133, 1.497, 1.469, 0.755]``.
    """
    delta = np.asarray(delta, dtype=float)
    chi = np.sqrt(R * np.clip(delta, 0.0, None)) / h
    c = GARCIA_BONDED_COEFFS
    poly = c[0] + c[1] * chi + c[2] * chi ** 2 + c[3] * chi ** 3 + c[4] * chi ** 4
    return hertz_force(delta, R, E, nu) * poly


def contact_radius_hertz(delta, R: float) -> np.ndarray:
    """Hertz contact radius ``a = sqrt(R*delta)`` (nm)."""
    delta = np.asarray(delta, dtype=float)
    return np.sqrt(R * np.clip(delta, 0.0, None))


# -- registry used by ForceCurve.plot(compare=[...]) -------------------------
@dataclass(frozen=True)
class AnalyticModel:
    key: str
    label: str
    func: object  # callable(delta, cfg) -> force array


def model_registry() -> dict:
    """Map model keys to callables of ``(delta, cfg)``."""

    def _hertz(delta, cfg):
        return hertz_force(delta, cfg.R, cfg.E, cfg.nu)

    def _dimitriadis_bonded(delta, cfg):
        return dimitriadis_force(delta, cfg.R, cfg.E, cfg.nu, cfg.h, bonded=True)

    def _dimitriadis_incompressible(delta, cfg):
        return dimitriadis_bonded_incompressible(delta, cfg.R, cfg.E, cfg.h)

    def _garcia(delta, cfg):
        return garcia_bec_force(delta, cfg.R, cfg.E, cfg.nu, cfg.h)

    return {
        "hertz": AnalyticModel("hertz", "Hertz (half-space)", _hertz),
        "dimitriadis_bonded": AnalyticModel(
            "dimitriadis_bonded", "Dimitriadis 2002 (bonded)", _dimitriadis_bonded
        ),
        "dimitriadis_incompressible": AnalyticModel(
            "dimitriadis_incompressible",
            "Dimitriadis (bonded, nu=0.5)",
            _dimitriadis_incompressible,
        ),
        "garcia_bec": AnalyticModel(
            "garcia_bec", "Garcia & Garcia 2018 BEC", _garcia
        ),
    }
