"""Tests for linoss_dynamics.filters: Kalman filter and RTS smoother."""

from __future__ import annotations

import numpy as np
import pytest

from linoss_dynamics.filters import kalman_filter, rts_smoother
from linoss_dynamics.solver import InvalidShapeError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_random_walk_system(n: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Ad, Qd, H, R) for a scalar random-walk model."""
    Ad = np.eye(n)
    Qd = 0.1 * np.eye(n)
    H = np.eye(1, n)        # first state component observed
    R = np.array([[1.0]])
    return Ad, Qd, H, R


# ---------------------------------------------------------------------------
# Shape / basic correctness
# ---------------------------------------------------------------------------


def test_kalman_filter_shapes() -> None:
    """Filtered/predicted arrays have the correct shapes; loglik is finite."""
    rng = np.random.default_rng(0)
    T, n = 30, 2
    y = rng.standard_normal(T)
    Ad = np.eye(n)
    Qd = 0.1 * np.eye(n)
    H = np.ones((1, n)) / n
    R = np.array([[0.5]])

    filt = kalman_filter(y, Ad, Qd, H, R)

    assert filt["m_f"].shape == (T, n)
    assert filt["P_f"].shape == (T, n, n)
    assert filt["m_p"].shape == (T, n)
    assert filt["P_p"].shape == (T, n, n)
    assert np.isfinite(filt["loglik"])


def test_kalman_filter_converges_random_walk() -> None:
    """For identity dynamics, filtered means should track observations well at long t."""
    rng = np.random.default_rng(42)
    T = 200
    # True state: scalar random walk
    x_true = np.cumsum(rng.standard_normal(T) * np.sqrt(0.1))
    y = x_true + rng.standard_normal(T)  # R=1

    Ad, Qd, H, R = _make_random_walk_system(n=1)
    filt = kalman_filter(y, Ad, Qd, H, R)

    # After burn-in, RMSE should be well below raw observation noise (sigma=1)
    rmse_filter = np.sqrt(np.mean((filt["m_f"][50:, 0] - x_true[50:]) ** 2))
    assert rmse_filter < 0.8


def test_rts_smoother_reduces_variance() -> None:
    """Summed trace of smoothed covariances <= summed trace of filtered covariances."""
    rng = np.random.default_rng(7)
    T = 50
    y = rng.standard_normal(T)
    Ad, Qd, H, R = _make_random_walk_system()

    filt = kalman_filter(y, Ad, Qd, H, R)
    smth = rts_smoother(Ad, filt)

    trace_filtered = np.sum([np.trace(P) for P in filt["P_f"]])
    trace_smoothed = np.sum([np.trace(P) for P in smth["P_s"]])

    # Smoother can never inflate variance relative to filter
    assert trace_smoothed <= trace_filtered + 1e-10


def test_kalman_with_control_input() -> None:
    """Filter runs without error and produces finite outputs when Bd and u are provided."""
    rng = np.random.default_rng(3)
    T, n, p = 20, 2, 1
    y = rng.standard_normal(T)
    Ad = 0.9 * np.eye(n)
    Qd = 0.05 * np.eye(n)
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.5]])
    Bd = rng.standard_normal((n, p))
    u = rng.standard_normal((T, p))

    filt = kalman_filter(y, Ad, Qd, H, R, Bd=Bd, u=u)

    assert filt["m_f"].shape == (T, n)
    assert np.all(np.isfinite(filt["m_f"]))
    assert np.isfinite(filt["loglik"])


def test_kalman_perfect_observation() -> None:
    """With R near zero, the filter mean should track the observations exactly."""
    rng = np.random.default_rng(11)
    T = 30
    y = rng.standard_normal(T)
    Ad = np.eye(1)
    Qd = np.eye(1)
    H = np.eye(1)
    R = np.array([[1e-12]])

    filt = kalman_filter(y, Ad, Qd, H, R)

    np.testing.assert_allclose(filt["m_f"][:, 0], y, atol=1e-6)


def test_kalman_no_observation_information() -> None:
    """With R very large, the filter barely updates; means stay near prior mean."""
    T = 50
    y = 10.0 * np.ones(T)   # large observations

    Ad = np.eye(1)
    Qd = 1e-10 * np.eye(1)  # nearly frozen state
    H = np.eye(1)
    R = np.array([[1e12]])   # enormous noise — ignore observations
    m0 = np.array([0.0])
    P0 = 1e-10 * np.eye(1)

    filt = kalman_filter(y, Ad, Qd, H, R, m0=m0, P0=P0)

    # Filter means should stay close to zero (the prior)
    assert np.all(np.abs(filt["m_f"][:, 0]) < 1.0)


def test_kalman_loglik_finite_and_negative() -> None:
    """Log-likelihood is a finite float; negative for typical noisy data."""
    rng = np.random.default_rng(99)
    T = 100
    y = rng.standard_normal(T)
    Ad, Qd, H, R = _make_random_walk_system()

    filt = kalman_filter(y, Ad, Qd, H, R)

    assert isinstance(filt["loglik"], float)
    assert np.isfinite(filt["loglik"])
    assert filt["loglik"] < 0.0


def test_rts_endpoints() -> None:
    """At t = T-1, smoothed distribution equals filtered distribution."""
    rng = np.random.default_rng(55)
    T = 25
    y = rng.standard_normal(T)
    Ad, Qd, H, R = _make_random_walk_system()

    filt = kalman_filter(y, Ad, Qd, H, R)
    smth = rts_smoother(Ad, filt)

    np.testing.assert_allclose(smth["m_s"][-1], filt["m_f"][-1], atol=1e-12)
    np.testing.assert_allclose(smth["P_s"][-1], filt["P_f"][-1], atol=1e-12)


# ---------------------------------------------------------------------------
# Damped oscillator: smoother beats filter
# ---------------------------------------------------------------------------


def _simulate_damped_oscillator(
    T: int, dt: float, omega: float, gamma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a noisy damped oscillator; return (true_x, observations).

    State x = [position, velocity], observation = position + noise.
    """
    # Continuous: d^2x/dt^2 + 2*gamma*dx/dt + omega^2*x = noise
    # Discretised with Euler for simplicity (small dt).
    n = 2
    x = np.zeros((T + 1, n))
    x[0] = np.array([1.0, 0.0])
    q_std = 0.05

    for t in range(T):
        pos, vel = x[t]
        acc = -omega**2 * pos - 2.0 * gamma * vel + rng.standard_normal() * q_std
        x[t + 1, 0] = pos + dt * vel
        x[t + 1, 1] = vel + dt * acc

    obs_noise = 0.1
    y = x[1:, 0] + rng.standard_normal(T) * obs_noise
    return x[1:, 0], y   # true positions, noisy observations


def test_smoother_beats_filter_damped_oscillator() -> None:
    """RTS-smoothed RMSE <= Kalman-filtered RMSE on damped oscillator trajectory."""
    rng = np.random.default_rng(123)
    T = 100
    dt = 0.05
    omega = 2.0
    gamma = 0.3

    x_true, y = _simulate_damped_oscillator(T, dt, omega, gamma, rng)

    # State-space model: x = [pos, vel], transition by Euler
    Ad = np.array([[1.0, dt], [-omega**2 * dt, 1.0 - 2.0 * gamma * dt]])
    Qd = (0.05**2 * dt) * np.eye(2)
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.01]])

    filt = kalman_filter(y, Ad, Qd, H, R)
    smth = rts_smoother(Ad, filt)

    # Compare position estimates only (what is observed)
    rmse_filter = np.sqrt(np.mean((filt["m_f"][:, 0] - x_true) ** 2))
    rmse_smoother = np.sqrt(np.mean((smth["m_s"][:, 0] - x_true) ** 2))

    assert rmse_smoother <= rmse_filter + 1e-6


# ---------------------------------------------------------------------------
# Error / shape mismatch
# ---------------------------------------------------------------------------


def test_kalman_error_y_H_mismatch() -> None:
    """InvalidShapeError when H column count does not match Ad state dimension."""
    y = np.zeros(10)
    Ad = np.eye(3)
    Qd = np.eye(3)
    H_bad = np.ones((1, 4))   # 4 cols but n=3
    R = np.array([[1.0]])

    with pytest.raises(InvalidShapeError):
        kalman_filter(y, Ad, Qd, H_bad, R)


def test_kalman_error_Ad_not_square() -> None:
    """InvalidShapeError when Ad is not square."""
    y = np.zeros(10)
    Ad_bad = np.ones((2, 3))
    Qd = np.eye(2)
    H = np.ones((1, 2))
    R = np.array([[1.0]])

    with pytest.raises(InvalidShapeError):
        kalman_filter(y, Ad_bad, Qd, H, R)


def test_kalman_error_Qd_shape_mismatch() -> None:
    """InvalidShapeError when Qd shape is inconsistent with Ad."""
    y = np.zeros(10)
    Ad = np.eye(2)
    Qd_bad = np.eye(3)   # wrong size
    H = np.ones((1, 2))
    R = np.array([[1.0]])

    with pytest.raises(InvalidShapeError):
        kalman_filter(y, Ad, Qd_bad, H, R)


def test_kalman_error_control_only_Bd() -> None:
    """InvalidShapeError when only Bd is given without u."""
    y = np.zeros(10)
    Ad, Qd, H, R = _make_random_walk_system()
    Bd = np.ones((1, 1))

    with pytest.raises(InvalidShapeError):
        kalman_filter(y, Ad, Qd, H, R, Bd=Bd)


def test_kalman_error_control_only_u() -> None:
    """InvalidShapeError when only u is given without Bd."""
    y = np.zeros(10)
    Ad, Qd, H, R = _make_random_walk_system()
    u = np.ones((10, 1))

    with pytest.raises(InvalidShapeError):
        kalman_filter(y, Ad, Qd, H, R, u=u)


def test_rts_error_Ad_shape_mismatch() -> None:
    """InvalidShapeError when Ad dimension does not match filter state dimension."""
    y = np.zeros(10)
    Ad_filt = np.eye(1)
    Ad_bad = np.eye(2)   # wrong for n=1 filter
    Ad, Qd, H, R = _make_random_walk_system()

    filt = kalman_filter(y, Ad_filt, Qd, H, R)

    with pytest.raises(InvalidShapeError):
        rts_smoother(Ad_bad, filt)
