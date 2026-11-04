"""T5: an exact whirlpool-like system with a Markov blanket."""

import numpy as np
import sympy as sp

from vam import gaussian_blanket as gb


def test_exact_numbers_symbolically():
    J = sp.Matrix(gb.J.astype(int))
    Q = sp.Matrix(gb.Q.astype(int))
    assert [J[:k, :k].det() for k in range(1, 5)] == [2, 4, 10, 16]
    A = (Q - sp.eye(4)) * J
    assert A == sp.Matrix([[-2, -2, 1, 0], [2, -2, -1, 0], [1, 0, -3, 1], [0, 0, 1, -2]])
    Sigma = J.inv()
    assert sp.simplify(A * Sigma + Sigma * A.T + 2 * sp.eye(4)) == sp.zeros(4)
    Mi = A[:2, :2]
    lam = sp.Symbol("lam")
    assert set(sp.solve((Mi - lam * sp.eye(2)).det(), lam)) == {-2 - 2 * sp.I, -2 + 2 * sp.I}
    assert Mi[1, 0] - Mi[0, 1] == 4
    # first two rows have no exterior term
    assert A[0, 3] == 0 and A[1, 3] == 0


def test_numeric_module_matches():
    assert np.allclose(gb.leading_minors(gb.J), [2, 4, 10, 16])
    assert np.allclose(gb.lyapunov_residual(), 0, atol=1e-14)
    assert np.allclose(np.sort_complex(np.linalg.eigvals(gb.interior_drift())), [-2 - 2j, -2 + 2j])
    assert gb.planar_vorticity(gb.interior_drift()) == 4
    assert np.isclose(gb.radius_multiplier(np.log(2)), 0.25, atol=1e-14)
    # stable drift: symmetric part of (Q - I) negative definite after similarity
    assert np.all(np.linalg.eigvals(gb.A).real < 0)


def test_conditional_law_independent_of_exterior():
    for b, e in ((0.3, -2.0), (0.3, 5.0), (-1.0, 0.0)):
        mean, cov = gb.conditional_interior(b, e)
        assert np.allclose(mean, [b / 2, 0]) and np.allclose(cov, 0.5 * np.eye(2))


def test_internal_fixed_point_and_spiral():
    for b in (0.0, 1.3):
        z = np.array([b / 2, 0.0, b, 7.0])
        assert np.allclose((gb.A @ z)[:2], 0)


def test_blanket_on_simulated_data(rng):
    """Empirical check: regressing (x, y) on (b, e) gives coefficient b/2 and no exterior dependence."""
    X = gb.sample_stationary_path(200_000, 0.5, rng)
    design = np.column_stack([X[:, 2], X[:, 3]])
    coef, *_ = np.linalg.lstsq(design, X[:, :2], rcond=None)
    assert np.allclose(coef[:, 0], [0.5, 0.0], atol=0.01)
    assert np.allclose(coef[1], [0.0, 0.0], atol=0.01)
    assert np.allclose(np.cov(X.T), gb.SIGMA, atol=0.01)
