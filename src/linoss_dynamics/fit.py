"""Maximum-likelihood estimation for damped oscillator state-space models.

Fits the four-parameter model

    x_{t+1} = Ad @ x_t + Bd @ u_t + w_t,   w_t ~ N(0, Qd)
    y_t      =  H @ x_t + v_t,              v_t ~ N(0, R)

where the latent state is ``x = [position, velocity]^T``, by maximising the
Kalman filter log-likelihood using SciPy's ``minimize``.

Public API:
    fit_oscillator_mle -- MLE fitting returning parameters, matrices, and filter outputs
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.linalg import solve_discrete_lyapunov
    from scipy.optimize import minimize as _scipy_minimize

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False
    _scipy_minimize = None  # type: ignore[assignment]
    solve_discrete_lyapunov = None  # type: ignore[assignment]

from .discretize import oscillator_mats
from .filters import kalman_filter, rts_smoother
from .solver import LinOSSError

__all__ = ["fit_oscillator_mle"]

_SCIPY_INSTALL_MSG = (
    "SciPy is required for MLE fitting. "
    "Install it with: pip install linoss-dynamics[probabilistic]"
)


def _require_scipy() -> None:
    """Raise LinOSSError with install hint when SciPy is unavailable."""
    if not _HAS_SCIPY:
        raise LinOSSError(_SCIPY_INSTALL_MSG)


def fit_oscillator_mle(
    y: np.ndarray,
    dt: float,
    *,
    u: np.ndarray | None = None,
    init: tuple[float, float, float, float] | None = None,
    method: str = "L-BFGS-B",
) -> dict[str, Any]:
    """Fit ``(beta, omega, sigma_proc, sigma_obs)`` of a single damped oscillator SSM.

    Maximises the Kalman filter log-likelihood with respect to the four scalar
    parameters of a forced damped harmonic oscillator:

    - ``beta``       — damping coefficient ``β ≥ 0``
    - ``omega``      — natural frequency ``ω > 0``
    - ``sigma_proc`` — process-noise standard deviation
    - ``sigma_obs``  — observation-noise standard deviation

    Optimisation is performed in log-parameter space so all parameters remain
    strictly positive throughout.  The observation mean is subtracted before
    fitting and recorded in the returned dict so predictions can be un-centred.

    Args:
        y: Observation sequence, shape ``(T,)``.
        dt: Discretization time-step (seconds), must be positive.
        u: Optional scalar control-input sequence, shape ``(T,)``.  If ``None``
            a zero sequence is used.
        init: Optional initial parameter guess ``(beta, omega, sigma_proc, sigma_obs)``.
            Defaults to ``(0.10, 2.0, 0.40, 0.30)``.
        method: SciPy ``minimize`` method string.  Default ``"L-BFGS-B"``.

    Returns:
        A dict containing:

        * ``"beta"``       -- ``float`` fitted damping coefficient.
        * ``"omega"``      -- ``float`` fitted natural frequency.
        * ``"sigma_proc"`` -- ``float`` fitted process-noise standard deviation.
        * ``"sigma_obs"``  -- ``float`` fitted observation-noise standard deviation.
        * ``"period"``     -- ``float`` derived oscillation period ``2π/omega``.
        * ``"mean_y"``     -- ``float`` mean subtracted from ``y`` before fitting.
        * ``"Ad"``         -- ``(2, 2)`` discrete state-transition matrix.
        * ``"Qd"``         -- ``(2, 2)`` discrete process-noise covariance.
        * ``"Bd"``         -- ``(2, 1)`` discrete control input matrix.
        * ``"H"``          -- ``(1, 2)`` observation matrix ``[[1, 0]]``.
        * ``"R"``          -- ``(1, 1)`` observation-noise covariance ``[[sigma_obs²]]``.
        * ``"filt"``       -- dict returned by :func:`~linoss_dynamics.filters.kalman_filter`.
        * ``"smooth"``     -- dict returned by :func:`~linoss_dynamics.filters.rts_smoother`.
        * ``"optim"``      -- ``scipy.optimize.OptimizeResult`` from the minimisation.

    Raises:
        LinOSSError: If SciPy is not installed.
        ValueError: If ``dt`` is not positive.

    Example:
        >>> import numpy as np
        >>> from linoss_dynamics.fit import fit_oscillator_mle
        >>> rng = np.random.default_rng(0)
        >>> y = np.sin(np.linspace(0, 10, 200)) + 0.1 * rng.standard_normal(200)
        >>> result = fit_oscillator_mle(y, dt=0.05)
        >>> 0 < result["omega"] < 20
        True
    """
    _require_scipy()

    if dt <= 0:
        raise ValueError(f"'dt' must be positive, got {dt!r}")

    y_arr = np.asarray(y, dtype=float).reshape(-1)
    T = y_arr.size

    # Centre observations — the SSM assumes zero mean.
    y_mean = float(np.mean(y_arr))
    y_cent = y_arr - y_mean

    # Build control input array (always present; zeros if no forcing).
    if u is None:
        u_arr = np.zeros(T)
    else:
        u_arr = np.asarray(u, dtype=float).reshape(-1)

    # Initial guess in natural space, then log-transform.
    if init is None:
        init = (0.10, 2.0, 0.40, 0.30)
    theta0 = np.log(np.asarray(init, dtype=float))

    # ------------------------------------------------------------------ #
    # Objective: negative log-likelihood in log-parameter space           #
    # ------------------------------------------------------------------ #

    def objective(theta: np.ndarray) -> float:
        beta, omega, sigma_proc, sigma_obs = np.exp(theta)
        try:
            Ad, Qd, Bd = oscillator_mats(beta, omega, sigma_proc, dt)
            H = np.array([[1.0, 0.0]])
            R = np.array([[sigma_obs**2]])
            # Regularise Qd for numerical stability before Lyapunov solve.
            P0 = solve_discrete_lyapunov(Ad, Qd + 1e-10 * np.eye(2))  # type: ignore[misc]
            filt = kalman_filter(y_cent, Ad, Qd, H, R, P0=P0, Bd=Bd, u=u_arr)
            loglik = filt["loglik"]
            if not np.isfinite(loglik):
                return np.inf
            return -loglik
        except Exception:  # noqa: BLE001
            # Numerical failures (singular matrices, etc.) → skip this region.
            return np.inf

    # ------------------------------------------------------------------ #
    # Optimise                                                            #
    # ------------------------------------------------------------------ #

    res = _scipy_minimize(objective, theta0, method=method)  # type: ignore[misc]

    # Recover parameters from best theta found.
    beta, omega, sigma_proc, sigma_obs = np.exp(res.x)

    # ------------------------------------------------------------------ #
    # Final run with fitted parameters                                    #
    # ------------------------------------------------------------------ #

    Ad, Qd, Bd = oscillator_mats(beta, omega, sigma_proc, dt)
    H = np.array([[1.0, 0.0]])
    R = np.array([[sigma_obs**2]])
    P0 = solve_discrete_lyapunov(Ad, Qd + 1e-10 * np.eye(2))  # type: ignore[misc]

    filt = kalman_filter(y_cent, Ad, Qd, H, R, P0=P0, Bd=Bd, u=u_arr)
    smooth = rts_smoother(Ad, filt)

    return {
        "beta": float(beta),
        "omega": float(omega),
        "sigma_proc": float(sigma_proc),
        "sigma_obs": float(sigma_obs),
        "period": float(2.0 * np.pi / omega),
        "mean_y": y_mean,
        "Ad": Ad,
        "Qd": Qd,
        "Bd": Bd,
        "H": H,
        "R": R,
        "filt": filt,
        "smooth": smooth,
        "optim": res,
    }
