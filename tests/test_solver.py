from __future__ import annotations

import numpy as np
import pytest

from linoss_dynamics import (
    InvalidDampingError,
    convergence_window,
    damped_linoss_step,
    linoss_step,
)


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
