"""Linear-Gaussian Kalman filter and Rauch-Tung-Striebel smoother.

This module provides batch (offline) state-estimation utilities for discrete-time
linear-Gaussian state-space models of the form:

    x_{t+1} = Ad @ x_t + Bd @ u_t + w_t,   w_t ~ N(0, Qd)
    y_t      = H @ x_t + v_t,               v_t ~ N(0, R)

The forward Kalman filter computes filtered (causal) posterior means and covariances;
the RTS smoother refines these using the full sequence in a single backward pass.

Public API:
    kalman_filter  -- forward Kalman filter returning filtered/predicted moments + log-likelihood
    rts_smoother   -- Rauch-Tung-Striebel backward smoother
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .solver import InvalidShapeError, LinOSSError  # noqa: F401  (re-exported for callers)

__all__ = ["kalman_filter", "rts_smoother"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_shape(name: str, arr: np.ndarray, *expected: tuple[int, ...]) -> None:
    """Raise InvalidShapeError when *arr* does not match any *expected* shape tuple."""
    if arr.shape not in expected:
        exp_str = " or ".join(repr(s) for s in expected)
        raise InvalidShapeError(
            f"{name} must have shape {exp_str}, got {arr.shape!r}"
        )


def _symmetrize(P: np.ndarray) -> np.ndarray:
    """Return (P + P.T) / 2 to suppress numerical asymmetry."""
    return 0.5 * (P + P.T)


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------


def kalman_filter(
    y: np.ndarray,
    Ad: np.ndarray,
    Qd: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    *,
    m0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    Bd: np.ndarray | None = None,
    u: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run a linear-Gaussian Kalman filter forward over a 1-D observation sequence.

    Implements the standard predict-update cycle for the model:

        x_{t+1} = Ad @ x_t [+ Bd @ u_t] + w_t,   w_t ~ N(0, Qd)
        y_t      = H @ x_t + v_t,                  v_t ~ N(0, R)

    Covariances are symmetrized after each update to suppress numerical drift.
    Log-likelihood is accumulated using the numerically stable slogdet form.

    Args:
        y: Observation sequence, shape ``(T,)``.
        Ad: State transition matrix, shape ``(n, n)``.
        Qd: Process-noise covariance, shape ``(n, n)``.
        H: Observation matrix, shape ``(1, n)``.
        R: Observation-noise covariance, shape ``(1, 1)``.
        m0: Initial state mean, shape ``(n,)``.  Defaults to zeros.
        P0: Initial state covariance, shape ``(n, n)``.  Defaults to identity.
        Bd: Control input matrix, shape ``(n, p)``.  Optional.
        u: Control input sequence, shape ``(T,)`` or ``(T, p)``.  Optional.

    Returns:
        A dict with keys:

        * ``"m_f"`` -- ``(T, n)`` filtered posterior means.
        * ``"P_f"`` -- ``(T, n, n)`` filtered posterior covariances.
        * ``"m_p"`` -- ``(T, n)`` one-step-ahead predicted means.
        * ``"P_p"`` -- ``(T, n, n)`` one-step-ahead predicted covariances.
        * ``"loglik"`` -- ``float`` total log-likelihood of the observation sequence.

    Raises:
        InvalidShapeError: When ``Ad``, ``Qd``, ``H``, ``R``, ``m0``, ``P0``,
            ``Bd``, or ``u`` have shapes inconsistent with each other or with ``y``.

    Examples:
        >>> import numpy as np
        >>> from linoss_dynamics.filters import kalman_filter, rts_smoother
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(50)
        >>> filt = kalman_filter(y, np.eye(1), 0.1 * np.eye(1), np.eye(1), np.eye(1))
        >>> filt["m_f"].shape
        (50, 1)
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    T = y_arr.size

    Ad = np.asarray(Ad, dtype=float)
    Qd = np.asarray(Qd, dtype=float)
    H = np.atleast_2d(np.asarray(H, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))

    # --- shape validation ---
    if Ad.ndim != 2 or Ad.shape[0] != Ad.shape[1]:
        raise InvalidShapeError(f"Ad must be a square 2-D matrix, got shape {Ad.shape!r}")
    n = Ad.shape[0]

    _require_shape("Qd", Qd, (n, n))
    if H.ndim != 2 or H.shape[0] != 1 or H.shape[1] != n:
        raise InvalidShapeError(
            f"H must have shape (1, {n}), got {H.shape!r}"
        )
    _require_shape("R", R, (1, 1))

    # --- defaults ---
    if m0 is None:
        m0 = np.zeros(n)
    else:
        m0 = np.asarray(m0, dtype=float).reshape(-1)
        _require_shape("m0", m0, (n,))

    if P0 is None:
        P0 = np.eye(n)
    else:
        P0 = np.asarray(P0, dtype=float)
        _require_shape("P0", P0, (n, n))

    # --- control input handling ---
    has_control = Bd is not None or u is not None
    if has_control:
        if Bd is None or u is None:
            raise InvalidShapeError("Both Bd and u must be provided together, or neither.")
        Bd_arr = np.asarray(Bd, dtype=float)
        if Bd_arr.ndim != 2 or Bd_arr.shape[0] != n:
            raise InvalidShapeError(
                f"Bd must have shape (n={n}, p), got {Bd_arr.shape!r}"
            )
        p = Bd_arr.shape[1]
        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 1:
            # shape (T,) interpreted as (T, 1) when p == 1, else error
            if p == 1:
                u_arr = u_arr.reshape(T, 1)
            else:
                raise InvalidShapeError(
                    f"u has shape {u_arr.shape!r} but Bd has {p} columns; "
                    f"u must have shape (T, {p}) = ({T}, {p})."
                )
        if u_arr.shape != (T, p):
            raise InvalidShapeError(
                f"u must have shape ({T}, {p}), got {u_arr.shape!r}"
            )
    else:
        Bd_arr = None
        u_arr = None

    # --- pre-allocate ---
    m_f = np.empty((T, n))
    P_f = np.empty((T, n, n))
    m_p = np.empty((T, n))
    P_p = np.empty((T, n, n))
    loglik = 0.0

    m = m0.copy()
    P = P0.copy()

    for t in range(T):
        # Predict
        m_pred = Ad @ m
        if Bd_arr is not None:
            m_pred = m_pred + Bd_arr @ u_arr[t]
        P_pred = Ad @ P @ Ad.T + Qd
        P_pred = _symmetrize(P_pred)

        m_p[t] = m_pred
        P_p[t] = P_pred

        # Innovation
        innov = np.array([y_arr[t]]) - H @ m_pred  # shape (1,)
        S = H @ P_pred @ H.T + R                   # shape (1, 1)
        S_inv = np.linalg.inv(S)
        K = P_pred @ H.T @ S_inv                   # shape (n, 1)

        # Update
        m = m_pred + (K @ innov).reshape(-1)
        P = P_pred - K @ H @ P_pred
        P = _symmetrize(P)

        m_f[t] = m
        P_f[t] = P

        # Log-likelihood contribution: -0.5 * (log|S| + v^T S^{-1} v + k log 2pi)
        sign, logdet = np.linalg.slogdet(S)
        loglik += -0.5 * (logdet + innov @ S_inv @ innov + H.shape[0] * np.log(2.0 * np.pi))

    return {
        "m_f": m_f,
        "P_f": P_f,
        "m_p": m_p,
        "P_p": P_p,
        "loglik": float(loglik),
    }


# ---------------------------------------------------------------------------
# RTS smoother
# ---------------------------------------------------------------------------


def rts_smoother(
    Ad: np.ndarray,
    filt: dict[str, Any],
) -> dict[str, Any]:
    """Run a Rauch-Tung-Striebel smoother backward through Kalman filter outputs.

    Starting from the filtered distribution at ``t = T-1`` (which equals the
    smoothed distribution at the last timestep), the smoother propagates backward
    to refine the mean and covariance at each earlier timestep.

    The smoother gain is:

        J_t = P_f[t] @ Ad.T @ inv(P_p[t+1])

    and the update equations are:

        m_s[t] = m_f[t] + J_t @ (m_s[t+1] - m_p[t+1])
        P_s[t] = P_f[t] + J_t @ (P_s[t+1] - P_p[t+1]) @ J_t.T

    Args:
        Ad: State transition matrix, shape ``(n, n)``.  Must match the matrix
            used during filtering.
        filt: Dict returned by :func:`kalman_filter`, containing ``"m_f"``,
            ``"P_f"``, ``"m_p"``, and ``"P_p"``.

    Returns:
        A dict with keys:

        * ``"m_s"`` -- ``(T, n)`` smoothed means.
        * ``"P_s"`` -- ``(T, n, n)`` smoothed covariances.

    Note:
        At ``t = T-1`` the smoothed distribution equals the filtered distribution
        (RTS endpoint property).

    Raises:
        InvalidShapeError: When ``Ad`` is not square or does not match state
            dimension inferred from ``filt``.
    """
    Ad = np.asarray(Ad, dtype=float)
    if Ad.ndim != 2 or Ad.shape[0] != Ad.shape[1]:
        raise InvalidShapeError(f"Ad must be a square 2-D matrix, got shape {Ad.shape!r}")

    m_f: np.ndarray = filt["m_f"]
    P_f: np.ndarray = filt["P_f"]
    m_p: np.ndarray = filt["m_p"]
    P_p: np.ndarray = filt["P_p"]

    T, n = m_f.shape
    if Ad.shape != (n, n):
        raise InvalidShapeError(
            f"Ad shape {Ad.shape!r} does not match state dimension n={n} from filt"
        )

    m_s = m_f.copy()
    P_s = P_f.copy()

    for t in range(T - 2, -1, -1):
        J = P_f[t] @ Ad.T @ np.linalg.inv(P_p[t + 1])  # (n, n)
        m_s[t] = m_f[t] + J @ (m_s[t + 1] - m_p[t + 1])
        P_s[t] = P_f[t] + J @ (P_s[t + 1] - P_p[t + 1]) @ J.T
        P_s[t] = _symmetrize(P_s[t])

    return {"m_s": m_s, "P_s": P_s}
