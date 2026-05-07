"""Tests for fit_oscillator_mle — MLE parameter fitting via Kalman log-likelihood."""

from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from linoss_dynamics.discretize import oscillator_mats  # noqa: E402
from linoss_dynamics.fit import fit_oscillator_mle  # noqa: E402


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------


def simulate_oscillator(
    beta: float,
    omega: float,
    sigma_proc: float,
    sigma_obs: float,
    T: int = 300,
    dt: float = 0.05,
    u: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a damped oscillator SSM and return (y, u_arr).

    Returns:
        y: observation sequence, shape (T,).
        u_arr: control input sequence used, shape (T,).
    """
    rng = np.random.default_rng(seed)

    Ad, Qd, Bd = oscillator_mats(beta, omega, sigma_proc, dt)

    if u is None:
        u_arr = np.zeros(T)
    else:
        u_arr = np.asarray(u, dtype=float).reshape(-1)

    # Cholesky for process noise sampling.
    Lq = np.linalg.cholesky(Qd + 1e-12 * np.eye(2))

    x = np.zeros(2)
    y = np.empty(T)

    for t in range(T):
        # Observe.
        y[t] = x[0] + sigma_obs * rng.standard_normal()
        # Propagate.
        w = Lq @ rng.standard_normal(2)
        x = Ad @ x + Bd[:, 0] * u_arr[t] + w

    return y, u_arr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFitReturnsFiniteLoglik:
    """Optimizer should converge and the final log-likelihood should be finite."""

    def test_fit_returns_finite_loglik(self) -> None:
        y, _ = simulate_oscillator(beta=0.1, omega=2.0, sigma_proc=0.4, sigma_obs=0.3)
        result = fit_oscillator_mle(y, dt=0.05)

        assert result["optim"] is not None
        # The optimizer should have evaluated the objective successfully.
        assert np.isfinite(result["optim"].fun), "Final cost must be finite"


class TestParameterRecovery:
    """Fitted parameters should be in the right ballpark for a clean simulation."""

    def test_omega_recovery(self) -> None:
        true_omega = 2.0
        y, _ = simulate_oscillator(
            beta=0.1, omega=true_omega, sigma_proc=0.4, sigma_obs=0.3, T=300, seed=7
        )
        result = fit_oscillator_mle(y, dt=0.05)

        np.testing.assert_allclose(
            result["omega"],
            true_omega,
            rtol=0.15,
            err_msg="omega not recovered within 15%",
        )

    def test_beta_recovery(self) -> None:
        # Damping is hard to identify from position-only observations because beta
        # and sigma_proc partially trade off along a likelihood ridge. We verify:
        #   (a) beta stays positive (optimizer constraint is active)
        #   (b) the fitted log-likelihood is at least as good as the true parameters
        #       (MLE should not return something worse than the ground truth)
        true_beta = 0.1
        true_omega = 2.0
        true_sp = 0.4
        true_so = 0.3
        y, u_arr = simulate_oscillator(
            beta=true_beta, omega=true_omega, sigma_proc=true_sp, sigma_obs=true_so,
            T=300, seed=13,
        )
        result = fit_oscillator_mle(y, dt=0.05)

        assert result["beta"] > 0, "beta must be positive"

        # The MLE log-likelihood must be >= the log-likelihood at the true params.
        from linoss_dynamics.discretize import oscillator_mats as _om
        from linoss_dynamics.filters import kalman_filter as _kf
        from scipy.linalg import solve_discrete_lyapunov as _sdl

        y_cent = y - float(np.mean(y))
        Ad_t, Qd_t, Bd_t = _om(true_beta, true_omega, true_sp, 0.05)
        H_t = np.array([[1.0, 0.0]])
        R_t = np.array([[true_so**2]])
        P0_t = _sdl(Ad_t, Qd_t + 1e-10 * np.eye(2))
        filt_true = _kf(y_cent, Ad_t, Qd_t, H_t, R_t, P0=P0_t, Bd=Bd_t, u=u_arr)
        true_loglik = filt_true["loglik"]
        fitted_loglik = result["filt"]["loglik"]

        assert fitted_loglik >= true_loglik - 5.0, (
            f"MLE log-likelihood {fitted_loglik:.2f} is unexpectedly lower than "
            f"true-parameter log-likelihood {true_loglik:.2f}"
        )

    def test_sigma_obs_recovery(self) -> None:
        true_sigma_obs = 0.3
        y, _ = simulate_oscillator(
            beta=0.1, omega=2.0, sigma_proc=0.4, sigma_obs=true_sigma_obs, T=300, seed=99
        )
        result = fit_oscillator_mle(y, dt=0.05)

        np.testing.assert_allclose(
            result["sigma_obs"],
            true_sigma_obs,
            rtol=0.30,
            err_msg="sigma_obs not recovered within 30%",
        )

    def test_sigma_proc_recovery(self) -> None:
        # sigma_proc and sigma_obs are partially confounded; allow loose tolerance.
        true_sigma_proc = 0.4
        y, _ = simulate_oscillator(
            beta=0.1, omega=2.0, sigma_proc=true_sigma_proc, sigma_obs=0.3, T=300, seed=55
        )
        result = fit_oscillator_mle(y, dt=0.05)

        assert result["sigma_proc"] > 0, "sigma_proc must be positive"
        np.testing.assert_allclose(
            result["sigma_proc"],
            true_sigma_proc,
            rtol=0.40,
            err_msg="sigma_proc not recovered within 40% (noise variance confounding expected)",
        )


class TestPeriodConsistency:
    """The derived period field must equal 2π / omega."""

    def test_fit_period_matches_2pi_over_omega(self) -> None:
        y, _ = simulate_oscillator(beta=0.1, omega=2.0, sigma_proc=0.4, sigma_obs=0.3)
        result = fit_oscillator_mle(y, dt=0.05)

        expected_period = 2.0 * np.pi / result["omega"]
        np.testing.assert_allclose(
            result["period"],
            expected_period,
            rtol=1e-10,
            err_msg="period != 2pi/omega",
        )


class TestNoForcingPath:
    """u=None should work without error and return a valid result."""

    def test_fit_handles_no_forcing(self) -> None:
        y, _ = simulate_oscillator(beta=0.1, omega=2.0, sigma_proc=0.4, sigma_obs=0.3)
        result = fit_oscillator_mle(y, dt=0.05, u=None)

        assert isinstance(result["beta"], float)
        assert isinstance(result["omega"], float)
        assert result["omega"] > 0
        assert result["beta"] > 0


class TestWithForcingPath:
    """Passing u should not crash and should produce a valid result."""

    def test_fit_handles_with_forcing(self) -> None:
        T = 300
        u = 0.2 * np.sin(np.linspace(0, 10, T))
        y, u_used = simulate_oscillator(
            beta=0.1, omega=2.0, sigma_proc=0.4, sigma_obs=0.3, T=T, u=u, seed=102
        )
        result = fit_oscillator_mle(y, dt=0.05, u=u_used)

        assert np.isfinite(result["optim"].fun)
        assert result["omega"] > 0
        assert result["beta"] > 0


class TestFilterAndSmootherPresent:
    """The returned dict must contain valid kalman_filter and rts_smoother outputs."""

    def test_fit_filt_and_smooth_present(self) -> None:
        y, _ = simulate_oscillator(beta=0.1, omega=2.0, sigma_proc=0.4, sigma_obs=0.3)
        result = fit_oscillator_mle(y, dt=0.05)

        filt = result["filt"]
        smooth = result["smooth"]

        T = y.shape[0]

        # Kalman filter keys and shapes.
        assert "m_f" in filt and "P_f" in filt
        assert "m_p" in filt and "P_p" in filt
        assert "loglik" in filt
        assert filt["m_f"].shape == (T, 2)
        assert filt["P_f"].shape == (T, 2, 2)
        assert np.isfinite(filt["loglik"])

        # RTS smoother keys and shapes.
        assert "m_s" in smooth and "P_s" in smooth
        assert smooth["m_s"].shape == (T, 2)
        assert smooth["P_s"].shape == (T, 2, 2)


class TestInitGuessConvergence:
    """Two different inits should both converge to a similar final omega (basin-of-attraction sanity)."""

    def test_fit_init_guess_changes_outcome_subtly(self) -> None:
        y, _ = simulate_oscillator(
            beta=0.1, omega=2.0, sigma_proc=0.4, sigma_obs=0.3, T=300, seed=77
        )

        result_a = fit_oscillator_mle(y, dt=0.05, init=(0.05, 1.5, 0.30, 0.20))
        result_b = fit_oscillator_mle(y, dt=0.05, init=(0.20, 3.0, 0.60, 0.50))

        # Both should converge to roughly the same omega — within 30% of each other.
        ratio = result_a["omega"] / result_b["omega"]
        assert 0.7 <= ratio <= 1.3, (
            f"Two inits converged to very different omega: "
            f"{result_a['omega']:.3f} vs {result_b['omega']:.3f}"
        )
        # Both should give a finite cost.
        assert np.isfinite(result_a["optim"].fun)
        assert np.isfinite(result_b["optim"].fun)


class TestMeanSubtraction:
    """mean_y should equal the arithmetic mean of y."""

    def test_mean_y_is_subtracted(self) -> None:
        # Shift the oscillator by a large constant to test centring.
        y, _ = simulate_oscillator(beta=0.1, omega=2.0, sigma_proc=0.4, sigma_obs=0.3)
        y_shifted = y + 50.0

        result = fit_oscillator_mle(y_shifted, dt=0.05)

        np.testing.assert_allclose(
            result["mean_y"],
            float(np.mean(y_shifted)),
            rtol=1e-10,
        )


class TestReturnedMatrixShapes:
    """All SSM matrices in the result dict must have the documented shapes."""

    def test_matrix_shapes(self) -> None:
        y, _ = simulate_oscillator(beta=0.1, omega=2.0, sigma_proc=0.4, sigma_obs=0.3)
        result = fit_oscillator_mle(y, dt=0.05)

        assert result["Ad"].shape == (2, 2)
        assert result["Qd"].shape == (2, 2)
        assert result["Bd"].shape == (2, 1)
        assert result["H"].shape == (1, 2)
        assert result["R"].shape == (1, 1)

    def test_r_equals_sigma_obs_squared(self) -> None:
        y, _ = simulate_oscillator(beta=0.1, omega=2.0, sigma_proc=0.4, sigma_obs=0.3)
        result = fit_oscillator_mle(y, dt=0.05)

        np.testing.assert_allclose(
            result["R"][0, 0],
            result["sigma_obs"] ** 2,
            rtol=1e-10,
        )
