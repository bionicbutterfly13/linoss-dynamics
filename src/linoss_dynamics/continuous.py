"""Closed-form analytic step for the continuous-time damped harmonic oscillator.

Solves the second-order ODE:

    ÿ + 2γ·ẏ + ω²·y = u(t)

for a constant forcing ``u`` over an arbitrary interval ``dt`` using the exact
matrix exponential of the system's companion matrix.  Because the solution is
algebraically exact, there is no numerical integration error — only the usual
floating-point rounding.  This makes the function particularly useful for
**irregular sampling**: each call may use a different ``dt`` without any
accumulated discretisation bias.

Typical use
-----------
>>> import numpy as np
>>> from linoss_dynamics.continuous import damped_oscillator_closed_form
>>> y, z = np.array([1.0]), np.array([0.0])
>>> y1, z1 = damped_oscillator_closed_form(y, z, omega=2.0, gamma=0.1, dt=0.5)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .solver import (
    InvalidDampingError,
    InvalidShapeError,
    LinOSSError,  # noqa: F401  re-exported for convenience
)

__all__ = ["damped_oscillator_closed_form"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CRITICAL_DAMP_TOL: float = 1e-10


def _broadcast_params(
    y: Any,
    z: Any,
    omega: Any,
    gamma: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coerce and broadcast all four oscillator-state parameters.

    Args:
        y: Position(s).
        z: Velocity(ies).
        omega: Angular frequency(ies).
        gamma: Damping coefficient(s).

    Returns:
        Tuple ``(y, z, omega, gamma)`` as 1-D float64 arrays of equal length.

    Raises:
        InvalidShapeError: If the shapes cannot be broadcast to a common 1-D
            vector.
        ValueError: If ``omega`` contains non-positive values.
        InvalidDampingError: If ``gamma`` contains negative values.
    """
    y_a = np.asarray(y, dtype=float).reshape(-1)
    z_a = np.asarray(z, dtype=float).reshape(-1)
    omega_a = np.asarray(omega, dtype=float).reshape(-1)
    gamma_a = np.asarray(gamma, dtype=float).reshape(-1)

    # Broadcast to a common size.
    try:
        y_a, z_a, omega_a, gamma_a = np.broadcast_arrays(y_a, z_a, omega_a, gamma_a)
    except ValueError as exc:
        raise InvalidShapeError(
            f"y, z, omega, gamma must be broadcastable; got shapes "
            f"{np.asarray(y).shape}, {np.asarray(z).shape}, "
            f"{np.asarray(omega).shape}, {np.asarray(gamma).shape}"
        ) from exc

    y_a = y_a.copy().astype(float)
    z_a = z_a.copy().astype(float)
    omega_a = omega_a.copy().astype(float)
    gamma_a = gamma_a.copy().astype(float)

    if np.any(omega_a <= 0.0):
        raise ValueError(f"omega must be positive, got {omega_a!r}")
    if np.any(gamma_a < 0.0):
        raise InvalidDampingError(f"gamma (damping) must be non-negative, got {gamma_a!r}")

    return y_a, z_a, omega_a, gamma_a


def _matrix_exp_row(
    omega: np.ndarray,
    gamma: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the four entries of ``exp(A_c · dt)`` per oscillator.

    The companion matrix is::

        A_c = [[0, 1], [-ω², -2γ]]

    and the matrix exponential is partitioned as::

        [[m00, m01],
         [m10, m11]]

    so that ``[y_next, z_next] = exp(A_c·dt) @ [y, z]``.

    Args:
        omega: Angular frequencies, 1-D float64 array.
        gamma: Damping coefficients, matching shape.
        dt: Time interval (positive).

    Returns:
        Tuple ``(m00, m01, m10, m11)`` as 1-D float64 arrays.
    """
    n = omega.size
    m00 = np.empty(n)
    m01 = np.empty(n)
    m10 = np.empty(n)
    m11 = np.empty(n)

    discriminant = gamma**2 - omega**2

    # --- underdamped mask (ω > γ, Δ < 0) ---
    ud = discriminant < -_CRITICAL_DAMP_TOL
    if np.any(ud):
        g = gamma[ud]
        w = omega[ud]
        wd = np.sqrt(omega[ud] ** 2 - gamma[ud] ** 2)
        edt = np.exp(-g * dt)
        c = np.cos(wd * dt)
        s = np.sin(wd * dt)
        sinc = s / wd          # sin(ω_d·dt)/ω_d — well conditioned since ω_d > 0

        m00[ud] = edt * (c + g * sinc)
        m01[ud] = edt * sinc
        m10[ud] = edt * (-w**2 * sinc)
        m11[ud] = edt * (c - g * sinc)

    # --- critically damped mask (ω ≈ γ) ---
    cd = np.abs(discriminant) <= _CRITICAL_DAMP_TOL
    if np.any(cd):
        g = gamma[cd]
        w = omega[cd]
        edt = np.exp(-g * dt)

        m00[cd] = edt * (1.0 + g * dt)
        m01[cd] = edt * dt
        m10[cd] = edt * (-w**2 * dt)
        m11[cd] = edt * (1.0 - g * dt)

    # --- overdamped mask (β > ω, Δ > 0) ---
    od = discriminant > _CRITICAL_DAMP_TOL
    if np.any(od):
        g = gamma[od]
        w = omega[od]
        wh = np.sqrt(discriminant[od])
        edt = np.exp(-g * dt)
        ch = np.cosh(wh * dt)
        sh = np.sinh(wh * dt)
        sinch = sh / wh        # sinh(ω_h·dt)/ω_h

        m00[od] = edt * (ch + g * sinch)
        m01[od] = edt * sinch
        m10[od] = edt * (-w**2 * sinch)
        m11[od] = edt * (ch - g * sinch)

    return m00, m01, m10, m11


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def damped_oscillator_closed_form(
    y: Any,
    z: Any,
    omega: Any,
    gamma: Any,
    dt: float,
    *,
    forcing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance a damped harmonic oscillator state by ``dt`` using the analytic
    closed-form solution.

    Solves the second-order ODE::

        ÿ + 2γ·ẏ + ω²·y = u

    where ``u = forcing`` is treated as a **constant** across the interval.
    When ``forcing != 0`` the trajectory is decomposed into homogeneous and
    particular parts:

    * Particular solution: ``y_p = u / ω²``, ``z_p = 0``.
    * Homogeneous solution started from ``(y - y_p, z)``.
    * Final state: ``y_p + homogeneous_y_next``, ``homogeneous_z_next``.

    The matrix exponential ``exp(A_c · dt)`` is evaluated analytically for
    each damping regime:

    * **Underdamped** (``ω > γ``): oscillatory decay via ``cos``/``sin``.
    * **Critically damped** (``ω ≈ γ``): polynomial-exponential decay.
    * **Overdamped** (``γ > ω``): two-exponential decay via ``cosh``/``sinh``.

    Args:
        y: Position(s), scalar or 1-D array.
        z: Velocity(ies), same broadcastable shape as ``y``.
        omega: Angular frequency(ies), scalar or 1-D array.  Must be strictly
            positive.
        gamma: Damping coefficient(s), scalar or 1-D array.  Must be
            non-negative.  Equivalent to ``β`` in the physics literature.
        dt: Time interval to advance (must be strictly positive).  May vary
            between calls, enabling exact stepping on irregular time grids.
        forcing: Constant scalar forcing applied across the interval.  Defaults
            to ``0.0`` (unforced / homogeneous).

    Returns:
        Tuple ``(y_next, z_next)`` at time ``t + dt``, both 1-D float64
        arrays.

    Raises:
        ValueError: If ``dt <= 0`` or any entry of ``omega`` is non-positive.
        InvalidDampingError: If any entry of ``gamma`` is negative.
        InvalidShapeError: If ``y``, ``z``, ``omega``, ``gamma`` cannot be
            broadcast to a common 1-D shape.

    Example:
        >>> import numpy as np
        >>> from linoss_dynamics.continuous import damped_oscillator_closed_form
        >>> y, z = np.array([1.0]), np.array([0.0])
        >>> y1, z1 = damped_oscillator_closed_form(y, z, omega=1.0, gamma=0.0, dt=np.pi)
        >>> np.allclose(y1, [[-1.0]], atol=1e-12)  # half-period of pure cosine
        True
    """
    dt_f = float(dt)
    if dt_f <= 0.0:
        raise ValueError(f"dt must be strictly positive, got {dt!r}")

    y_a, z_a, omega_a, gamma_a = _broadcast_params(y, z, omega, gamma)

    # Particular solution for constant forcing: y_p = u/ω², z_p = 0.
    forcing_f = float(forcing)
    if forcing_f != 0.0:
        y_p = forcing_f / omega_a**2
        y_shifted = y_a - y_p
    else:
        y_p = np.zeros_like(y_a)
        y_shifted = y_a

    # Compute matrix exponential entries for each oscillator.
    m00, m01, m10, m11 = _matrix_exp_row(omega_a, gamma_a, dt_f)

    # Apply the matrix exponential to the (shifted) state vector.
    y_hom_next = m00 * y_shifted + m01 * z_a
    z_hom_next = m10 * y_shifted + m11 * z_a

    y_next = y_p + y_hom_next
    z_next = z_hom_next

    return y_next, z_next
