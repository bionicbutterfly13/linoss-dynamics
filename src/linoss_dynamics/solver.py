"""NumPy LinOSS stepping helpers with explicit damping support."""

from __future__ import annotations

from typing import Any

import numpy as np


class LinOSSError(ValueError):
    """Base error for LinOSS dynamics failures."""


class InvalidShapeError(LinOSSError):
    """Raised when LinOSS inputs cannot be broadcast to the oscillator state."""


class InvalidDampingError(LinOSSError):
    """Raised when damping `G` is outside the supported stable path."""


class UnsupportedModeError(LinOSSError):
    """Raised when a discretization mode is not supported."""


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "implicit").lower().replace("-", "_")
    if normalized in {"im", "implicit"}:
        return "implicit"
    if normalized in {"imex", "implicit_explicit"}:
        return "implicit_explicit"
    raise UnsupportedModeError(f"Unsupported LinOSS mode: {mode!r}")


def _state_vectors(y: Any, z: Any) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    if y_arr.shape != z_arr.shape:
        raise InvalidShapeError(
            f"y and z must have the same shape, got {y_arr.shape} and {z_arr.shape}"
        )
    return y_arr, z_arr


def _diagonal_vector(name: str, value: Any, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(size, float(arr))
    if arr.ndim == 1:
        if arr.size == 1:
            return np.full(size, float(arr[0]))
        if arr.size == size:
            return arr.astype(float, copy=True)
    if arr.ndim == 2 and arr.shape == (size, size):
        off_diag = arr - np.diag(np.diag(arr))
        if np.allclose(off_diag, 0.0):
            return np.diag(arr).astype(float, copy=True)
    raise InvalidShapeError(
        f"{name} must be scalar, length-{size} vector, or {size}x{size} diagonal matrix"
    )


def _forcing_vector(B: Any, u: Any, size: int) -> np.ndarray:
    if B is None and u is None:
        return np.zeros(size)

    if B is None:
        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 0:
            return np.full(size, float(u_arr))
        u_flat = u_arr.reshape(-1)
        if u_flat.size == size:
            return u_flat
        raise InvalidShapeError(f"u must be scalar or length-{size} when B is omitted")

    if u is None:
        return np.zeros(size)

    B_arr = np.asarray(B, dtype=float)
    u_arr = np.asarray(u, dtype=float)

    if B_arr.ndim == 1:
        if B_arr.size != size:
            raise InvalidShapeError(f"B vector must have length {size}")
        if u_arr.ndim == 0:
            return B_arr * float(u_arr)
        u_flat = u_arr.reshape(-1)
        if u_flat.size == size:
            return B_arr * u_flat
        raise InvalidShapeError(f"u must be scalar or length-{size} for vector B")

    if B_arr.ndim == 2 and B_arr.shape[0] == size:
        try:
            forcing = B_arr @ u_arr.reshape(-1)
        except ValueError as exc:
            raise InvalidShapeError(f"B and u shapes are incompatible: {exc}") from exc
        return np.asarray(forcing, dtype=float).reshape(-1)

    raise InvalidShapeError(f"B must be length-{size} vector or matrix with {size} rows")


def energy(y: Any, z: Any, A: Any, G: Any | None = None) -> float:
    """Return oscillator energy for diagonal LinOSS state parameters."""

    del G
    y_arr, z_arr = _state_vectors(y, z)
    A_vec = _diagonal_vector("A", A, y_arr.size)
    return float(0.5 * np.sum(A_vec * y_arr**2 + z_arr**2))


def delta_energy(previous_energy: float, next_energy: float) -> float:
    """Return signed energy delta."""

    return float(next_energy) - float(previous_energy)


def convergence_window(deltas: list[float] | tuple[float, ...], threshold: float, window: int) -> bool:
    """Return true when the most recent absolute deltas are all below threshold."""

    if window <= 0:
        raise ValueError("window must be positive")
    if len(deltas) < window:
        return False
    return all(abs(float(delta)) < float(threshold) for delta in deltas[-window:])


def linoss_step(
    y: Any,
    z: Any,
    A: Any,
    dt: float,
    mode: str = "implicit",
    B: Any | None = None,
    u: Any | None = None,
    damping: Any | None = None,
    G: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Advance one LinOSS step with optional explicit damping."""

    if damping is not None or G is not None:
        return damped_linoss_step(
            y=y,
            z=z,
            A=A,
            G=damping if damping is not None else G,
            dt=dt,
            mode=mode,
            B=B,
            u=u,
        )

    y_arr, z_arr = _state_vectors(y, z)
    A_vec = _diagonal_vector("A", A, y_arr.size)
    forcing = _forcing_vector(B, u, y_arr.size)
    dt_f = float(dt)
    mode_name = _normalize_mode(mode)
    before = energy(y_arr, z_arr, A_vec)

    if mode_name == "implicit":
        schur = 1.0 / (1.0 + (dt_f**2) * A_vec)
        z_next = schur * z_arr - dt_f * A_vec * schur * y_arr + dt_f * schur * forcing
        y_next = schur * y_arr + dt_f * schur * z_arr + (dt_f**2) * schur * forcing
    elif mode_name == "implicit_explicit":
        z_next = z_arr - dt_f * A_vec * y_arr + dt_f * forcing
        y_next = y_arr + dt_f * z_next
    else:  # pragma: no cover - _normalize_mode guards this
        raise UnsupportedModeError(f"Unsupported LinOSS mode: {mode!r}")

    after = energy(y_next, z_next, A_vec)
    return y_next, z_next, {
        "mode": mode_name,
        "energy_before": before,
        "energy_after": after,
        "delta_energy": delta_energy(before, after),
        "damping_mode": "none",
    }


def damped_linoss_step(
    y: Any,
    z: Any,
    A: Any,
    G: Any,
    dt: float,
    mode: str = "implicit",
    B: Any | None = None,
    u: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Advance one LinOSS step with explicit non-negative damping `G`."""

    y_arr, z_arr = _state_vectors(y, z)
    A_vec = _diagonal_vector("A", A, y_arr.size)
    G_vec = _diagonal_vector("G", G, y_arr.size)
    if np.any(G_vec < 0.0):
        raise InvalidDampingError("G damping must be non-negative")

    forcing = _forcing_vector(B, u, y_arr.size)
    dt_f = float(dt)
    mode_name = _normalize_mode(mode)
    before = energy(y_arr, z_arr, A_vec)

    if mode_name == "implicit":
        denom = 1.0 + dt_f * G_vec + (dt_f**2) * A_vec
        z_next = (z_arr - dt_f * A_vec * y_arr + dt_f * forcing) / denom
        y_next = y_arr + dt_f * z_next
    elif mode_name == "implicit_explicit":
        denom = 1.0 + dt_f * G_vec
        z_next = (z_arr - dt_f * A_vec * y_arr + dt_f * forcing) / denom
        y_next = y_arr + dt_f * z_next
    else:  # pragma: no cover - _normalize_mode guards this
        raise UnsupportedModeError(f"Unsupported LinOSS mode: {mode!r}")

    after = energy(y_next, z_next, A_vec)
    return y_next, z_next, {
        "mode": mode_name,
        "energy_before": before,
        "energy_after": after,
        "delta_energy": delta_energy(before, after),
        "damping_mode": "explicit_g",
        "damping": G_vec,
    }


def linoss_step_impl(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Backward-compatible implementation alias for callers using this name."""

    return linoss_step(*args, **kwargs)


def linoss_scan(
    U: Any,
    A: Any,
    dt: float,
    *,
    mode: str = "implicit",
    B: Any | None = None,
    y0: Any | None = None,
    z0: Any | None = None,
    damping: Any | None = None,
    G: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run linoss_step over a time-varying input, returning the full trajectory.

    Iterates over the rows of ``U``, calling ``linoss_step`` (or
    ``damped_linoss_step`` when damping is requested) at each timestep and
    collecting the per-step state and metrics into pre-allocated arrays.

    Args:
        U: Input sequence of shape ``(T, m)`` or ``(T,)`` for scalar/vector
            forcing.  Each row ``U[t]`` is passed as ``u`` to the step
            function.
        A: Stiffness/frequency parameter — scalar or length-``n`` vector.
        dt: Timestep.
        mode: Discretization scheme.  ``"implicit"``/``"im"`` or
            ``"implicit_explicit"``/``"imex"``.  Defaults to
            ``"implicit"``.
        B: Optional forcing matrix of shape ``(n, m)``.  When ``None`` and
            ``U`` is 2-D, the number of columns must equal ``n``.
        y0: Optional initial displacement vector of length ``n``.  Defaults
            to zeros.
        z0: Optional initial velocity vector of length ``n``.  Defaults to
            zeros.
        damping: Alias for ``G`` (non-negative scalar or vector).  When
            either ``damping`` or ``G`` is provided, ``damped_linoss_step``
            is called at each timestep.
        G: Non-negative damping scalar or length-``n`` vector.  Mutually
            interchangeable with ``damping``; ``damping`` takes precedence
            when both are supplied.

    Returns:
        A three-tuple ``(Y, Z, metrics_seq)`` where:

        * ``Y`` — shape ``(T+1, n)`` displacement trajectory including the
          initial state at ``Y[0]``.
        * ``Z`` — shape ``(T+1, n)`` velocity trajectory including the
          initial state at ``Z[0]``.
        * ``metrics_seq`` — list of length ``T`` of per-step metrics dicts
          as returned by the underlying step function.

    Raises:
        InvalidShapeError: When ``U``, ``y0``, ``z0``, or ``B`` have
            incompatible shapes.
        InvalidDampingError: When ``G``/``damping`` contains negative
            values.
        UnsupportedModeError: When ``mode`` is not a recognised
            discretization scheme.
    """
    U_arr = np.asarray(U, dtype=float)
    if U_arr.ndim == 1:
        T = U_arr.shape[0]
        u_seq: list[Any] = [U_arr[t] for t in range(T)]
    elif U_arr.ndim == 2:
        T = U_arr.shape[0]
        u_seq = [U_arr[t] for t in range(T)]
    else:
        raise InvalidShapeError(
            f"U must be 1-D (T,) or 2-D (T, m), got shape {U_arr.shape!r}"
        )

    # Resolve initial state — infer n from y0/z0 if provided, else from A.
    if y0 is not None:
        y_curr = np.asarray(y0, dtype=float).reshape(-1)
        n = y_curr.size
        z_curr = np.zeros(n) if z0 is None else np.asarray(z0, dtype=float).reshape(-1)
    elif z0 is not None:
        z_curr = np.asarray(z0, dtype=float).reshape(-1)
        n = z_curr.size
        y_curr = np.zeros(n)
    else:
        # Infer n from A; _diagonal_vector will validate later per step.
        A_arr = np.asarray(A, dtype=float)
        if A_arr.ndim == 0:
            # Scalar A — try to infer n from U column count or B.
            if B is not None:
                B_arr = np.asarray(B, dtype=float)
                if B_arr.ndim == 2:
                    n = B_arr.shape[0]
                elif B_arr.ndim == 1:
                    n = B_arr.size
                else:
                    raise InvalidShapeError(
                        f"Cannot infer state dimension from scalar A and B shape {B_arr.shape!r}"
                    )
            elif U_arr.ndim == 2:
                n = U_arr.shape[1]
            else:
                n = 1
        else:
            n = A_arr.reshape(-1).size
        y_curr = np.zeros(n)
        z_curr = np.zeros(n)

    if y_curr.shape != z_curr.shape:
        raise InvalidShapeError(
            f"y0 and z0 must have the same length, got {y_curr.shape} and {z_curr.shape}"
        )

    # Pre-allocate trajectory arrays (T+1 rows).
    Y = np.empty((T + 1, y_curr.size), dtype=float)
    Z = np.empty((T + 1, z_curr.size), dtype=float)
    Y[0] = y_curr
    Z[0] = z_curr

    use_damping = damping is not None or G is not None
    G_eff = damping if damping is not None else G

    metrics_seq: list[dict[str, Any]] = []

    for t in range(T):
        u_t = u_seq[t]
        if use_damping:
            y_next, z_next, step_metrics = damped_linoss_step(
                y=y_curr,
                z=z_curr,
                A=A,
                G=G_eff,
                dt=dt,
                mode=mode,
                B=B,
                u=u_t,
            )
        else:
            y_next, z_next, step_metrics = linoss_step(
                y=y_curr,
                z=z_curr,
                A=A,
                dt=dt,
                mode=mode,
                B=B,
                u=u_t,
            )
        Y[t + 1] = y_next
        Z[t + 1] = z_next
        metrics_seq.append(step_metrics)
        y_curr = y_next
        z_curr = z_next

    return Y, Z, metrics_seq
