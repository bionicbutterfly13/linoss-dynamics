"""Stability analysis helpers for LinOSS oscillator systems."""

from __future__ import annotations

from typing import Any

import numpy as np

from .solver import (
    InvalidShapeError,
    _normalize_mode,
)

__all__ = [
    "is_stable",
    "eigvals_to_freq_damping",
    "freq_damping_to_oscillator_block",
    "period_from_omega",
    "harmonic_stack",
]


def is_stable(
    A: Any,
    G: Any | None = None,
    dt: float | None = None,
    mode: str = "implicit",
) -> tuple[bool, str]:
    """Return (stable, reason) for the given LinOSS oscillator parameters.

    Args:
        A: Diagonal stiffness parameter(s). Scalar or array.
        G: Diagonal damping parameter(s). Scalar or array. Optional.
        dt: Time-step size. Required when mode is ``"implicit_explicit"``.
        mode: Discretization mode — ``"implicit"`` / ``"im"`` or
            ``"implicit_explicit"`` / ``"imex"``.

    Returns:
        A ``(stable, reason)`` tuple where *reason* is a human-readable string
        explaining which check passed or failed.

    Raises:
        UnsupportedModeError: If *mode* is not recognized.
    """
    mode_name = _normalize_mode(mode)
    A_arr = np.asarray(A, dtype=float).reshape(-1)

    # Vacuously stable when A has no entries — nothing to violate any condition.
    if A_arr.size == 0:
        return True, "vacuously stable: empty A"

    if mode_name == "implicit":
        if np.any(A_arr < 0.0):
            neg_vals = A_arr[A_arr < 0.0]
            return False, f"A has negative entries {neg_vals!r}; implicit mode requires A >= 0"
        if G is not None:
            G_arr = np.asarray(G, dtype=float).reshape(-1)
            if np.any(G_arr < 0.0):
                neg_vals = G_arr[G_arr < 0.0]
                return False, f"G has negative entries {neg_vals!r}; implicit mode requires G >= 0"
        return True, "implicit mode: A >= 0" + (" and G >= 0" if G is not None else "")

    # mode_name == "implicit_explicit"
    if dt is None:
        return False, "dt required for IMEX stability check"

    dt_f = float(dt)

    if np.any(A_arr < 0.0):
        neg_vals = A_arr[A_arr < 0.0]
        return False, f"A has negative entries {neg_vals!r}; IMEX mode requires A >= 0"

    if G is not None:
        G_arr = np.asarray(G, dtype=float).reshape(-1)
        if np.any(G_arr < 0.0):
            neg_vals = G_arr[G_arr < 0.0]
            return False, f"G has negative entries {neg_vals!r}; IMEX mode requires G >= 0"

    max_A = float(np.max(A_arr))
    cfl_value = dt_f**2 * max_A
    if cfl_value >= 4.0:
        return (
            False,
            f"IMEX CFL condition violated: dt^2 * max(A) = {cfl_value!r} >= 4"
            f" (dt={dt_f!r}, max(A)={max_A!r})",
        )

    return (
        True,
        f"IMEX mode: A >= 0, G >= 0 (if provided), and dt^2 * max(A) = {cfl_value!r} < 4",
    )


def eigvals_to_freq_damping(eigenvalues: Any) -> tuple[np.ndarray, np.ndarray]:
    """Map complex eigenvalues r·e^(±iθ) to (frequencies, damping_ratios).

    Frequency is computed as θ / dt (where dt = 1 by convention here, so
    frequency = angle in radians per step). Damping ratio = -ln(r) per step.

    Args:
        eigenvalues: Array of complex eigenvalues. Can be 1-D or 0-D.

    Returns:
        Tuple ``(frequencies, damping_ratios)`` as 1-D float arrays,
        both of length ``N`` where ``N = len(eigenvalues)``.
    """
    eig = np.asarray(eigenvalues, dtype=complex).reshape(-1)
    r = np.abs(eig)
    theta = np.angle(eig)
    frequencies = np.abs(theta)
    # Avoid log(0) — zero eigenvalue has infinite damping; clamp r to a tiny floor
    r_safe = np.where(r > 0.0, r, np.finfo(float).tiny)
    damping_ratios = -np.log(r_safe)
    return frequencies, damping_ratios


def freq_damping_to_oscillator_block(
    omega: float,
    gamma: float,
    dt: float = 1.0,
) -> np.ndarray:
    """Return a 2×2 real-form oscillator block for given frequency and damping.

    The block represents the discrete-time state transition for a single
    harmonic oscillator with angular frequency *omega* and damping *gamma*,
    sampled at time-step *dt*::

        [[exp(-gamma*dt)*cos(omega*dt), -exp(-gamma*dt)*sin(omega*dt)],
         [exp(-gamma*dt)*sin(omega*dt),  exp(-gamma*dt)*cos(omega*dt)]]

    Args:
        omega: Angular frequency (rad / unit time). Must be non-negative.
        gamma: Damping coefficient. May be zero (undamped).
        dt: Time-step size. Defaults to 1.0.

    Returns:
        Shape ``(2, 2)`` NumPy float array.

    Raises:
        InvalidShapeError: If *omega* is negative.
    """
    omega_f = float(omega)
    gamma_f = float(gamma)
    dt_f = float(dt)

    if omega_f < 0.0:
        raise InvalidShapeError(f"omega must be non-negative, got {omega!r}")

    decay = np.exp(-gamma_f * dt_f)
    c = decay * np.cos(omega_f * dt_f)
    s = decay * np.sin(omega_f * dt_f)
    return np.array([[c, -s], [s, c]], dtype=float)


def period_from_omega(omega: Any) -> np.ndarray:
    """Return 2π / omega, vectorized over array input.

    Args:
        omega: Positive angular frequency or array of frequencies.

    Returns:
        Period array of the same shape as *omega*.

    Raises:
        InvalidShapeError: If any element of *omega* is non-positive.
    """
    omega_arr = np.asarray(omega, dtype=float)
    if np.any(omega_arr <= 0.0):
        bad = omega_arr[omega_arr <= 0.0]
        raise InvalidShapeError(f"omega must be positive, got {bad!r}")
    return 2.0 * np.pi / omega_arr


def harmonic_stack(
    omegas: Any,
    dampings: Any | None = None,
    dt: float = 1.0,
) -> np.ndarray:
    """Build a block-diagonal state matrix from (omega, damping) pairs.

    Each pair produces a 2×2 oscillator block via
    :func:`freq_damping_to_oscillator_block`. The blocks are assembled into a
    ``(2N, 2N)`` block-diagonal matrix where ``N = len(omegas)``.

    Args:
        omegas: Sequence of angular frequencies. Length N.
        dampings: Sequence of damping coefficients. Length N. If ``None``,
            all dampings default to 0.0 (undamped).
        dt: Time-step size. Defaults to 1.0.

    Returns:
        Shape ``(2N, 2N)`` NumPy float array.

    Raises:
        InvalidShapeError: If *omegas* and *dampings* have different lengths,
            or if any omega is negative.
    """
    omegas_arr = np.asarray(omegas, dtype=float).reshape(-1)
    n = omegas_arr.size

    if dampings is None:
        dampings_arr = np.zeros(n, dtype=float)
    else:
        dampings_arr = np.asarray(dampings, dtype=float).reshape(-1)
        if dampings_arr.size != n:
            raise InvalidShapeError(
                f"omegas and dampings must have the same length, "
                f"got {n} and {dampings_arr.size}"
            )

    size = 2 * n
    result = np.zeros((size, size), dtype=float)
    for i in range(n):
        block = freq_damping_to_oscillator_block(omegas_arr[i], dampings_arr[i], dt)
        result[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block

    return result
