"""Exact matrix-exponential discretization helpers for continuous-time LTI systems.

This module implements van Loan's method for jointly discretizing the state-transition
matrix and the process-noise covariance, as well as zero-order-hold (ZOH) discretization
for control inputs.  All functions require SciPy (``pip install linoss-dynamics[probabilistic]``).
"""

from __future__ import annotations

import numpy as np

# SciPy is an optional dependency — imported at module level with a guard so the
# rest of the package remains functional without it.
try:
    from scipy.linalg import expm

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False
    expm = None  # type: ignore[assignment]

from .solver import InvalidShapeError, LinOSSError

__all__ = [
    "discretize_lti_with_noise",
    "discretize_control",
    "oscillator_mats",
]

_SCIPY_INSTALL_MSG = (
    "SciPy is required for matrix-exponential discretization. "
    "Install it with: pip install linoss-dynamics[probabilistic]"
)


def _require_scipy() -> None:
    """Raise a helpful LinOSSError if SciPy is not available."""
    if not _HAS_SCIPY:
        raise LinOSSError(_SCIPY_INSTALL_MSG)


def _check_square(name: str, arr: np.ndarray) -> int:
    """Return the side length of a square matrix, or raise InvalidShapeError."""
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise InvalidShapeError(
            f"{name!r} must be a square 2-D matrix, got shape {arr.shape!r}"
        )
    return arr.shape[0]


def discretize_lti_with_noise(
    A_c: np.ndarray,
    L: np.ndarray,
    Qc: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact discretization of ``dx = A_c·x·dt + L·dW`` with continuous spectral density Qc.

    Uses van Loan's matrix-exponential method::

        M = [[A_c, L·Qc·L.T], [0, -A_c.T]] * dt
        Phi = expm(M)
        Ad  = Phi[:n, :n]
        Qd  = Phi[:n, n:] @ Ad.T

    The result ``Qd`` is symmetrised numerically to counteract floating-point drift.

    Args:
        A_c: Continuous-time state matrix, shape ``(n, n)``.
        L:   Noise input matrix, shape ``(n, q)``.
        Qc:  Continuous-time process-noise spectral density, shape ``(q, q)``.
             Must be symmetric positive semi-definite.
        dt:  Discretization time-step (seconds), must be positive.

    Returns:
        A 2-tuple ``(Ad, Qd)`` where:

        - ``Ad`` — discrete state-transition matrix, shape ``(n, n)``.
        - ``Qd`` — discrete process-noise covariance matrix, shape ``(n, n)``,
          symmetric positive semi-definite.

    Raises:
        LinOSSError: If SciPy is not installed.
        InvalidShapeError: If ``A_c`` is not square, or if the shapes of ``L``
            and ``Qc`` are incompatible.
        ValueError: If ``dt`` is not positive.

    Example:
        >>> import numpy as np
        >>> A_c = np.array([[-1.0]])
        >>> L   = np.array([[1.0]])
        >>> Qc  = np.array([[0.0]])
        >>> Ad, Qd = discretize_lti_with_noise(A_c, L, Qc, dt=0.1)
        >>> float(Ad[0, 0])  # doctest: +ELLIPSIS
        0.904...
    """
    _require_scipy()

    A_c = np.asarray(A_c, dtype=float)
    L = np.asarray(L, dtype=float)
    Qc = np.asarray(Qc, dtype=float)

    n = _check_square("A_c", A_c)

    if L.ndim != 2 or L.shape[0] != n:
        raise InvalidShapeError(
            f"'L' must have shape (n, q) with n={n}, got {L.shape!r}"
        )
    q = L.shape[1]

    if Qc.shape != (q, q):
        raise InvalidShapeError(
            f"'Qc' must have shape (q, q)=({q}, {q}), got {Qc.shape!r}"
        )

    if dt <= 0:
        raise ValueError(f"'dt' must be positive, got {dt!r}")

    zeros_nn = np.zeros((n, n))
    M = np.block(
        [
            [A_c, L @ Qc @ L.T],
            [zeros_nn, -A_c.T],
        ]
    ) * dt

    Phi = expm(M)  # type: ignore[misc]
    Ad = Phi[:n, :n]
    Qd_raw = Phi[:n, n:] @ Ad.T
    Qd = 0.5 * (Qd_raw + Qd_raw.T)  # enforce numerical symmetry

    return Ad, Qd


def discretize_control(
    A_c: np.ndarray,
    B: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold (ZOH) discretization of ``dx/dt = A_c·x + B·u``.

    Constructs the augmented system::

        M = [[A_c, B], [0, 0]] * dt
        Phi = expm(M)
        Ad  = Phi[:n, :n]
        Bd  = Phi[:n, n:n+m]

    Args:
        A_c: Continuous-time state matrix, shape ``(n, n)``.
        B:   Input matrix, shape ``(n, m)``.
        dt:  Discretization time-step (seconds), must be positive.

    Returns:
        A 2-tuple ``(Ad, Bd)`` where:

        - ``Ad`` — discrete state-transition matrix, shape ``(n, n)``.
        - ``Bd`` — discrete input matrix, shape ``(n, m)``.

    Raises:
        LinOSSError: If SciPy is not installed.
        InvalidShapeError: If ``A_c`` is not square, or if ``B`` has an
            incompatible number of rows.
        ValueError: If ``dt`` is not positive.

    Example:
        >>> import numpy as np
        >>> A_c = np.zeros((1, 1))
        >>> B   = np.ones((1, 1))
        >>> Ad, Bd = discretize_control(A_c, B, dt=0.5)
        >>> float(Bd[0, 0])  # doctest: +ELLIPSIS
        0.5...
    """
    _require_scipy()

    A_c = np.asarray(A_c, dtype=float)
    B = np.asarray(B, dtype=float)

    n = _check_square("A_c", A_c)

    if B.ndim != 2 or B.shape[0] != n:
        raise InvalidShapeError(
            f"'B' must have shape (n, m) with n={n}, got {B.shape!r}"
        )
    m = B.shape[1]

    if dt <= 0:
        raise ValueError(f"'dt' must be positive, got {dt!r}")

    M = np.block(
        [
            [A_c, B],
            [np.zeros((m, n + m))],
        ]
    ) * dt

    Phi = expm(M)  # type: ignore[misc]
    Ad = Phi[:n, :n]
    Bd = Phi[:n, n : n + m]

    return Ad, Bd


def oscillator_mats(
    beta: float,
    omega: float,
    sigma_proc: float,
    dt: float = 1.0,
    kappa: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(Ad, Qd, Bd)`` for a single forced damped oscillator.

    The continuous-time state equation is::

        ẏ =  z
        ż = -ω²·y - 2β·z + κ·u + σ_proc·w

    where ``y`` is position, ``z`` is velocity, ``u`` is a scalar control input,
    and ``w`` is scalar white noise.

    Args:
        beta:       Damping coefficient ``β ≥ 0``.  Set to ``0`` for undamped.
        omega:      Natural frequency ``ω > 0``.
        sigma_proc: Standard deviation of the continuous-time process noise.
                    The spectral density is ``Qc = sigma_proc²``.
        dt:         Discretization time-step (seconds), default ``1.0``.
        kappa:      Control gain ``κ``, default ``1.0``.

    Returns:
        A 3-tuple ``(Ad, Qd, Bd)`` where:

        - ``Ad`` — 2×2 discrete state-transition matrix.
        - ``Qd`` — 2×2 discrete process-noise covariance (symmetric PSD).
        - ``Bd`` — 2×1 discrete input matrix.

    Raises:
        LinOSSError: If SciPy is not installed.
        ValueError: If ``dt`` is not positive.

    Example:
        >>> import numpy as np
        >>> Ad, Qd, Bd = oscillator_mats(beta=0.1, omega=1.0, sigma_proc=0.5, dt=0.1)
        >>> Ad.shape, Qd.shape, Bd.shape
        ((2, 2), (2, 2), (2, 1))
    """
    A_c = np.array(
        [
            [0.0, 1.0],
            [-(omega**2), -2.0 * beta],
        ]
    )
    L = np.array([[0.0], [1.0]])
    Qc = np.array([[sigma_proc**2]])
    B = np.array([[0.0], [kappa]])

    Ad, Qd = discretize_lti_with_noise(A_c, L, Qc, dt)
    _, Bd = discretize_control(A_c, B, dt)

    return Ad, Qd, Bd
