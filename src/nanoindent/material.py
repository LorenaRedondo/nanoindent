"""Linear isotropic elasticity for axisymmetric (r, z) small-strain FEM.

The axisymmetric strain vector is ordered

    eps = [eps_rr, eps_zz, eps_tt, gamma_rz]

with

    eps_rr   = du_r/dr
    eps_zz   = du_z/dz
    eps_tt   = u_r / r            (hoop strain)
    gamma_rz = du_r/dz + du_z/dr  (engineering shear)

and the constitutive matrix below maps that vector to the stress vector
``[sig_rr, sig_zz, sig_tt, sig_rz]``.  Keeping the hoop component explicit is
what distinguishes the axisymmetric operator from plane strain.
"""

from __future__ import annotations

import numpy as np


class LinearElastic:
    """Isotropic linear-elastic material.

    Parameters
    ----------
    E:
        Young's modulus (MPa).
    nu:
        Poisson's ratio (dimensionless).
    """

    def __init__(self, E: float, nu: float) -> None:
        if not (E > 0):
            raise ValueError("E must be > 0")
        if not (-1.0 < nu < 0.5):
            raise ValueError("nu must be in (-1, 0.5)")
        self.E = float(E)
        self.nu = float(nu)

    @property
    def lam(self) -> float:
        """First Lame parameter (MPa)."""
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def mu(self) -> float:
        """Shear modulus (MPa)."""
        return self.E / (2.0 * (1.0 + self.nu))

    def constitutive_matrix(self) -> np.ndarray:
        """4x4 constitutive matrix ``C`` for the axisymmetric strain ordering
        ``[eps_rr, eps_zz, eps_tt, gamma_rz]``.

        Note that the shear row uses the *engineering* shear strain, so the
        (3, 3) entry is ``mu`` (not ``2 mu``).
        """
        lam, mu = self.lam, self.mu
        C = np.array(
            [
                [lam + 2 * mu, lam, lam, 0.0],
                [lam, lam + 2 * mu, lam, 0.0],
                [lam, lam, lam + 2 * mu, 0.0],
                [0.0, 0.0, 0.0, mu],
            ]
        )
        return C

    # -- UFL-like strain/stress helpers used by the weak form --------------
    @staticmethod
    def strain_components(u, w):
        """Return the four axisymmetric strain components at quadrature points.

        ``u`` is a scikit-fem vector trial/test function proxy providing
        ``u[i]`` (value of component ``i``) and ``u.grad[i][j]`` (d u_i / d x_j
        with x_0 = r, x_1 = z).  ``w`` carries the global coordinates in
        ``w.x`` so the hoop strain ``u_r / r`` can be formed.
        """
        r = w.x[0]
        eps_rr = u.grad[0][0]
        eps_zz = u.grad[1][1]
        eps_tt = u[0] / r
        gamma_rz = u.grad[0][1] + u.grad[1][0]
        return eps_rr, eps_zz, eps_tt, gamma_rz

    def stress_from_strain(self, eps_rr, eps_zz, eps_tt, gamma_rz):
        """Return the stress components for the given strains.

        ``gamma_rz`` is the engineering shear strain, hence ``sig_rz = mu *
        gamma_rz``.
        """
        lam, mu = self.lam, self.mu
        tr = eps_rr + eps_zz + eps_tt
        sig_rr = lam * tr + 2 * mu * eps_rr
        sig_zz = lam * tr + 2 * mu * eps_zz
        sig_tt = lam * tr + 2 * mu * eps_tt
        sig_rz = mu * gamma_rz
        return sig_rr, sig_zz, sig_tt, sig_rz
