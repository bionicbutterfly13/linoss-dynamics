from __future__ import annotations

import numpy as np
import pytest

from linoss_dynamics import (
    InvalidDampingError,
    convergence_window,
    damped_linoss_step,
    linoss_step,
)
from linoss_dynamics.solver import InvalidShapeError, linoss_scan


def test_linoss_im_forced_closed_form_step():
    y = np.array([0.2, -0.1])
    z = np.array([0.3, 0.05])
    A = np.array([2.0, 0.5])
    dt = 0.1
    B = np.array([[1.2, 0.4], [0.1, 0.9]])
    u = np.array([0.5, -0.2])

    y_next, z_next, metrics = linoss_step(y, z, A, dt, mode="IM", B=B, u=u)

    forcing = B @ u
    schur = 1.0 / (1.0 + (dt**2) * A)
    z_expected = schur * z - dt * A * schur * y + dt * schur * forcing
    y_expected = schur * y + dt * schur * z + (dt**2) * schur * forcing

    np.testing.assert_allclose(y_next, y_expected)
    np.testing.assert_allclose(z_next, z_expected)
    assert metrics["mode"] == "implicit"


def test_linoss_imex_alias_and_forcing():
    y = np.array([1.0])
    z = np.array([0.0])
    A = np.array([1.0])
    dt = 0.05
    B = np.array([[1.0]])
    u = np.array([0.2])

    y_next, z_next, metrics = linoss_step(y, z, A, dt, mode="IMEX", B=B, u=u)

    forcing = (B @ u)[0]
    z_expected = z[0] - dt * A[0] * y[0] + dt * forcing
    y_expected = y[0] + dt * z_expected

    assert z_next[0] == pytest.approx(z_expected)
    assert y_next[0] == pytest.approx(y_expected)
    assert metrics["mode"] == "implicit_explicit"


def test_damped_step_uses_g_without_mutating_a():
    y = np.array([0.2])
    z = np.array([1.0])
    A = np.array([1.0])
    A_before = A.copy()
    dt = 0.1

    _, _, low_metrics = damped_linoss_step(y, z, A, np.array([0.0]), dt)
    _, _, high_metrics = damped_linoss_step(y, z, A, np.array([1.0]), dt)

    np.testing.assert_array_equal(A, A_before)
    assert high_metrics["energy_after"] < low_metrics["energy_after"]
    assert high_metrics["damping_mode"] == "explicit_g"


@pytest.mark.parametrize("mode", ["implicit", "implicit_explicit"])
def test_zero_g_matches_classic_step_for_selected_mode(mode):
    y = np.array([0.2, -0.1])
    z = np.array([0.3, 0.05])
    A = np.array([2.0, 0.5])
    dt = 0.1

    y_classic, z_classic, _ = linoss_step(y, z, A, dt, mode=mode)
    y_damped, z_damped, metrics = damped_linoss_step(y, z, A, 0.0, dt, mode=mode)

    np.testing.assert_allclose(y_damped, y_classic)
    np.testing.assert_allclose(z_damped, z_classic)
    assert metrics["damping_mode"] == "explicit_g"


def test_scalar_vector_and_diagonal_damping_are_equivalent():
    y = np.array([0.1, -0.2])
    z = np.array([0.4, 0.3])
    A = np.array([1.0, 2.0])
    dt = 0.05

    y_scalar, z_scalar, _ = damped_linoss_step(y, z, A, 0.5, dt)
    y_vector, z_vector, _ = damped_linoss_step(y, z, A, np.array([0.5, 0.5]), dt)
    y_diag, z_diag, _ = damped_linoss_step(y, z, A, np.diag([0.5, 0.5]), dt)

    np.testing.assert_allclose(y_scalar, y_vector)
    np.testing.assert_allclose(z_scalar, z_vector)
    np.testing.assert_allclose(y_scalar, y_diag)
    np.testing.assert_allclose(z_scalar, z_diag)


def test_negative_damping_rejected():
    with pytest.raises(InvalidDampingError):
        damped_linoss_step(
            y=np.array([1.0]),
            z=np.array([0.0]),
            A=np.array([1.0]),
            G=np.array([-0.1]),
            dt=0.1,
        )


def test_convergence_window_uses_recent_absolute_deltas():
    assert convergence_window([0.10, 0.01, 0.02, 0.03], threshold=0.05, window=3)
    assert not convergence_window([0.01, 0.20, 0.02], threshold=0.05, window=3)


# ---------------------------------------------------------------------------
# linoss_scan tests
# ---------------------------------------------------------------------------


def test_linoss_scan_zero_input_remains_zero():
    """Null input with zero initial state keeps the state at zero throughout."""
    T = 5
    n = 3
    U = np.zeros((T, n))
    A = np.ones(n)
    dt = 0.1

    Y, Z, metrics_seq = linoss_scan(U, A, dt, y0=np.zeros(n), z0=np.zeros(n))

    np.testing.assert_allclose(Y, np.zeros((T + 1, n)))
    np.testing.assert_allclose(Z, np.zeros((T + 1, n)))
    assert len(metrics_seq) == T


def test_linoss_scan_matches_step_loop_implicit():
    """Scan output for T=10 must match a manual linoss_step loop (implicit)."""
    rng = np.random.default_rng(0)
    T, n = 10, 2
    A = np.array([1.5, 0.8])
    dt = 0.05
    U = rng.standard_normal((T, n))
    y0 = rng.standard_normal(n)
    z0 = rng.standard_normal(n)

    Y_scan, Z_scan, _ = linoss_scan(U, A, dt, mode="implicit", y0=y0, z0=z0)

    # Manual reference loop.
    Y_ref = np.empty((T + 1, n))
    Z_ref = np.empty((T + 1, n))
    Y_ref[0] = y0
    Z_ref[0] = z0
    y_curr, z_curr = y0.copy(), z0.copy()
    for t in range(T):
        y_curr, z_curr, _ = linoss_step(y_curr, z_curr, A, dt, mode="implicit", u=U[t])
        Y_ref[t + 1] = y_curr
        Z_ref[t + 1] = z_curr

    np.testing.assert_allclose(Y_scan, Y_ref)
    np.testing.assert_allclose(Z_scan, Z_ref)


def test_linoss_scan_matches_step_loop_imex():
    """Scan output for T=10 must match a manual linoss_step loop (IMEX)."""
    rng = np.random.default_rng(1)
    T, n = 10, 2
    A = np.array([1.5, 0.8])
    dt = 0.05
    U = rng.standard_normal((T, n))
    y0 = rng.standard_normal(n)
    z0 = rng.standard_normal(n)

    Y_scan, Z_scan, _ = linoss_scan(U, A, dt, mode="implicit_explicit", y0=y0, z0=z0)

    Y_ref = np.empty((T + 1, n))
    Z_ref = np.empty((T + 1, n))
    Y_ref[0] = y0
    Z_ref[0] = z0
    y_curr, z_curr = y0.copy(), z0.copy()
    for t in range(T):
        y_curr, z_curr, _ = linoss_step(
            y_curr, z_curr, A, dt, mode="implicit_explicit", u=U[t]
        )
        Y_ref[t + 1] = y_curr
        Z_ref[t + 1] = z_curr

    np.testing.assert_allclose(Y_scan, Y_ref)
    np.testing.assert_allclose(Z_scan, Z_ref)


def test_linoss_scan_with_damping_matches_damped_step_loop():
    """Scan with G > 0 must match a manual damped_linoss_step loop."""
    rng = np.random.default_rng(2)
    T, n = 8, 3
    A = np.array([2.0, 1.0, 0.5])
    G = np.array([0.3, 0.1, 0.5])
    dt = 0.05
    U = rng.standard_normal((T, n))
    y0 = rng.standard_normal(n)
    z0 = rng.standard_normal(n)

    Y_scan, Z_scan, _ = linoss_scan(U, A, dt, G=G, y0=y0, z0=z0)

    Y_ref = np.empty((T + 1, n))
    Z_ref = np.empty((T + 1, n))
    Y_ref[0] = y0
    Z_ref[0] = z0
    y_curr, z_curr = y0.copy(), z0.copy()
    for t in range(T):
        y_curr, z_curr, _ = damped_linoss_step(y_curr, z_curr, A, G, dt, u=U[t])
        Y_ref[t + 1] = y_curr
        Z_ref[t + 1] = z_curr

    np.testing.assert_allclose(Y_scan, Y_ref)
    np.testing.assert_allclose(Z_scan, Z_ref)


def test_linoss_scan_initial_state_respected():
    """Non-zero y0/z0 must appear exactly at index 0 of the trajectory."""
    n, T = 4, 6
    y0 = np.arange(1, n + 1, dtype=float)
    z0 = np.arange(n + 1, 2 * n + 1, dtype=float)
    U = np.zeros((T, n))
    A = np.ones(n)

    Y, Z, _ = linoss_scan(U, A, dt=0.01, y0=y0, z0=z0)

    np.testing.assert_array_equal(Y[0], y0)
    np.testing.assert_array_equal(Z[0], z0)


def test_linoss_scan_metrics_seq_length():
    """metrics_seq must have exactly T entries for a length-T input sequence."""
    T = 7
    U = np.ones((T, 2))
    A = np.array([1.0, 1.0])

    _, _, metrics_seq = linoss_scan(U, A, dt=0.1)

    assert len(metrics_seq) == T


def test_linoss_scan_invalid_U_shape():
    """A 3-D U array must raise InvalidShapeError."""
    U_bad = np.ones((4, 2, 2))
    A = np.array([1.0, 2.0])

    with pytest.raises(InvalidShapeError):
        linoss_scan(U_bad, A, dt=0.1)


def test_linoss_scan_2d_input_with_B_matrix():
    """Full (T, m) input with an (n, m) B matrix must produce the correct trajectory."""
    rng = np.random.default_rng(3)
    T, n, m = 6, 3, 2
    A = np.array([1.0, 2.0, 0.5])
    B = rng.standard_normal((n, m))
    dt = 0.05
    U = rng.standard_normal((T, m))
    y0 = rng.standard_normal(n)
    z0 = rng.standard_normal(n)

    Y_scan, Z_scan, _ = linoss_scan(U, A, dt, B=B, y0=y0, z0=z0)

    # Reference: manual loop using B @ u_t forcing.
    Y_ref = np.empty((T + 1, n))
    Z_ref = np.empty((T + 1, n))
    Y_ref[0] = y0
    Z_ref[0] = z0
    y_curr, z_curr = y0.copy(), z0.copy()
    for t in range(T):
        y_curr, z_curr, _ = linoss_step(y_curr, z_curr, A, dt, B=B, u=U[t])
        Y_ref[t + 1] = y_curr
        Z_ref[t + 1] = z_curr

    np.testing.assert_allclose(Y_scan, Y_ref)
    np.testing.assert_allclose(Z_scan, Z_ref)
