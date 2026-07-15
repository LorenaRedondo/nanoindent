import numpy as np
import pytest

from nanoindent.material import LinearElastic


def test_lame_parameters():
    E, nu = 130.0, 0.3
    m = LinearElastic(E, nu)
    assert m.mu == pytest.approx(E / (2 * (1 + nu)))
    assert m.lam == pytest.approx(E * nu / ((1 + nu) * (1 - 2 * nu)))


def test_constitutive_matrix_symmetry():
    C = LinearElastic(130.0, 0.3).constitutive_matrix()
    assert C.shape == (4, 4)
    # Upper-left 3x3 block is symmetric; shear decoupled.
    assert np.allclose(C[:3, :3], C[:3, :3].T)
    assert C[3, 0] == 0 and C[0, 3] == 0
    assert C[3, 3] == pytest.approx(LinearElastic(130.0, 0.3).mu)


def test_uniaxial_stress_recovers_youngs_modulus():
    # For a uniaxial stress state sig_zz != 0, others zero, the axial strain
    # equals sig_zz / E. Build compliance and check.
    E, nu = 130.0, 0.3
    C = LinearElastic(E, nu).constitutive_matrix()
    S = np.linalg.inv(C)
    # Apply unit sig_zz (index 1), zero others.
    stress = np.array([0.0, 1.0, 0.0, 0.0])
    strain = S @ stress
    assert strain[1] == pytest.approx(1.0 / E, rel=1e-10)
    # Lateral strains = -nu/E
    assert strain[0] == pytest.approx(-nu / E, rel=1e-10)
    assert strain[2] == pytest.approx(-nu / E, rel=1e-10)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        LinearElastic(-1.0, 0.3)
    with pytest.raises(ValueError):
        LinearElastic(130.0, 0.5)
