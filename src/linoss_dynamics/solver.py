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
