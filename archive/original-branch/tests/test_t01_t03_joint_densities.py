"""T1.2, T3: joint vector-weight densities, congruence updates, reciprocal influence, geometry."""

import numpy as np
import pytest

from vam.density import check_density, from_factor
from vam.geometry import density_metric
from vam.hamiltonian import heisenberg_derivative, mean_field_vector_hamiltonian, reciprocal_hamiltonian
from vam.linalg import (
    commutator,
    dag,
    is_psd,
    kron,
    min_eigenvalue,
    random_density,
    random_hermitian,
    random_pure_state,
)
from vam.space import Space


def test_joint_state_marginals(rng):
    """T3.1: R_V = sum p_w |v_w><v_w|, (R_W)_ww' = sqrt(p_w p_w') <v_w', v_w>."""
    dV, nW = 3, 4
    p = rng.random(nW)
    p /= p.sum()
    vs = [random_pure_state(dV, rng) for _ in range(nW)]
    psi = sum(np.sqrt(p[w]) * np.kron(vs[w], np.eye(nW)[w]) for w in range(nW))
    R = np.outer(psi, psi.conj())
    sp = Space.of(V=dV, W=nW)
    RV = sp.reduce(R, "V")
    RW = sp.reduce(R, "W")
    assert np.allclose(RV, sum(p[w] * np.outer(vs[w], vs[w].conj()) for w in range(nW)))
    expected_W = np.array([[np.sqrt(p[w] * p[u]) * np.vdot(vs[u], vs[w]) for u in range(nW)] for w in range(nW)])
    assert np.allclose(RW, expected_W)


def test_factor_form_is_positive(rng):
    A = rng.standard_normal((6, 2)) + 1j * rng.standard_normal((6, 2))
    check_density(from_factor(A))


def test_signed_weight_observable_with_positive_density(rng):
    W_hat = np.diag([-1.5, 0.0, 2.0]).astype(complex)
    R = random_density(3, rng)
    assert min_eigenvalue(W_hat) < 0 and is_psd(R)


def test_congruence_preserves_positivity_and_density_products_do_not(rng):
    """T3.2."""
    R = random_density(3, rng)
    Xu = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    out = Xu @ R @ dag(Xu)
    check_density(out / np.trace(out))
    R1 = np.diag([1, 0]).astype(complex)
    R2 = np.full((2, 2), 0.5, dtype=complex)
    prod = R1 @ R2
    assert not np.allclose(prod, dag(prod))


@pytest.mark.parametrize("hbar", [1.0, 0.37])
def test_reciprocal_heisenberg_equations(rng, hbar):
    """T3.3: both directions of influence come from the same coupling terms."""
    dV, dW = 3, 2
    sp = Space.of(V=dV, W=dW)
    H_D0, H_W = random_hermitian(dV, rng), random_hermitian(dW, rng)
    cpl = [(random_hermitian(dV, rng), random_hermitian(dW, rng)) for _ in range(2)]
    H = reciprocal_hamiltonian(sp, "V", "W", H_D0, H_W, cpl).matrix()
    R = random_density(dV * dW, rng)  # correlated joint state
    A, B = random_hermitian(dV, rng), random_hermitian(dW, rng)
    lhs_A = heisenberg_derivative(H, sp.embed(A, "V"), R, hbar)
    rhs_A = (1j / hbar) * np.trace(R @ sp.embed(commutator(H_D0, A), "V"))
    rhs_A += sum((1j / hbar) * np.trace(R @ kron(commutator(Da, A), Wa)) for Da, Wa in cpl)
    lhs_B = heisenberg_derivative(H, sp.embed(B, "W"), R, hbar)
    rhs_B = (1j / hbar) * np.trace(R @ sp.embed(commutator(H_W, B), "W"))
    rhs_B += sum((1j / hbar) * np.trace(R @ kron(Da, commutator(Wa, B))) for Da, Wa in cpl)
    assert np.isclose(lhs_A, rhs_A) and np.isclose(lhs_B, rhs_B)
    assert abs(lhs_A.imag) < 1e-12 and abs(lhs_B.imag) < 1e-12


def test_mean_field_exact_only_for_product_states(rng):
    dV, dW = 2, 2
    sp = Space.of(V=dV, W=dW)
    H_D0, H_W = random_hermitian(dV, rng), random_hermitian(dW, rng)
    cpl = [(random_hermitian(dV, rng), random_hermitian(dW, rng))]
    H = reciprocal_hamiltonian(sp, "V", "W", H_D0, H_W, cpl).matrix()
    A = random_hermitian(dV, rng)
    RV, RW = random_density(dV, rng), random_density(dW, rng)
    prod = kron(RV, RW)
    Heff = mean_field_vector_hamiltonian(H_D0, cpl, RW)
    assert np.isclose(heisenberg_derivative(H, sp.embed(A, "V"), prod), heisenberg_derivative(Heff, A, RV))
    psi = random_pure_state(4, rng)
    corr = np.outer(psi, psi.conj())
    Heff_c = mean_field_vector_hamiltonian(H_D0, cpl, sp.reduce(corr, "W"))
    assert not np.isclose(heisenberg_derivative(H, sp.embed(A, "V"), corr),
                          heisenberg_derivative(Heff_c, A, sp.reduce(corr, "V")))


def test_density_metric_positive_and_state_dependent(rng):
    """T3.4: v^T g v >= g0 |v|^2, and g changes with R."""
    obs = [random_hermitian(3, rng) for _ in range(4)]
    g0 = 0.2
    g1 = density_metric(random_density(3, rng), obs, g0)
    g2 = density_metric(random_density(3, rng), obs, g0)
    for g in (g1, g2):
        assert np.allclose(g, g.T)
        assert np.linalg.eigvalsh(g)[0] >= g0 - 1e-12
    assert not np.allclose(g1, g2)
