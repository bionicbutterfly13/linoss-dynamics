"""Tests for linoss_dynamics.discretize module.

Requires SciPy.  If SciPy is absent the entire module is skipped so CI on a
bare-NumPy install remains green.
"""

from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")  # noqa: F841 — skips module if scipy absent

from linoss_dynamics.discretize import (  # noqa: E402 — must come after importorskip
    discretize_control,
    discretize_lti_with_noise,
    oscillator_mats,
)


# ---------------------------------------------------------------------------
# discretize_lti_with_noise
# ---------------------------------------------------------------------------


def test_discretize_lti_recovers_zoh_for_zero_noise():
    """With Qc=0 the discrete noise covariance must be zero and Ad = e^{A_c·dt}."""
    # 1-D first-order system: dx = -x dt  =>  Ad = e^{-dt}
    dt = 0.1
    A_c = np.array([[-1.0]])
    L = np.array([[1.0]])
    Qc = np.array([[0.0]])

    Ad, Qd = discretize_lti_with_noise(A_c, L, Qc, dt)

    expected_Ad = np.exp(-dt)
    assert Ad.shape == (1, 1)
    assert Qd.shape == (1, 1)
    assert float(Ad[0, 0]) == pytest.approx(expected_Ad, rel=1e-10)
    assert float(Qd[0, 0]) == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# discretize_control
# ---------------------------------------------------------------------------


def test_discretize_control_zoh_simple_integrator():
    """A_c=0, B=I => Ad=I, Bd=dt·I (pure integrator, scalar case)."""
    dt = 0.25
    A_c = np.zeros((1, 1))
    B = np.ones((1, 1))

    Ad, Bd = discretize_control(A_c, B, dt)

    assert Ad.shape == (1, 1)
    assert Bd.shape == (1, 1)
    assert float(Ad[0, 0]) == pytest.approx(1.0, rel=1e-12)
    assert float(Bd[0, 0]) == pytest.approx(dt, rel=1e-12)


# ---------------------------------------------------------------------------
# oscillator_mats — shape checks
# ---------------------------------------------------------------------------


def test_oscillator_mats_correct_shapes():
    """Return values must be 2×2, 2×2, 2×1."""
    Ad, Qd, Bd = oscillator_mats(beta=0.1, omega=1.0, sigma_proc=0.5, dt=0.05)

    assert Ad.shape == (2, 2)
    assert Qd.shape == (2, 2)
    assert Bd.shape == (2, 1)


# ---------------------------------------------------------------------------
# oscillator_mats — eigenvalue tests
# ---------------------------------------------------------------------------


def test_oscillator_mats_undamped():
    """For β=0 the discrete eigenvalues of Ad lie on the unit circle: |λ| = 1."""
    omega = 2.0
    dt = 0.05
    Ad, _, _ = oscillator_mats(beta=0.0, omega=omega, sigma_proc=0.0, dt=dt)

    eigvals = np.linalg.eigvals(Ad)
    magnitudes = np.abs(eigvals)

    # Undamped system: eigenvalues of A_c are ±iω  =>  discrete are e^{±iω·dt}
    assert magnitudes == pytest.approx(np.ones(2), rel=1e-8)

    # Also check the angle matches ω·dt
    expected_angle = omega * dt
    angles = np.sort(np.abs(np.angle(eigvals)))
    assert angles[0] == pytest.approx(0.0, abs=1e-10) or angles[0] == pytest.approx(
        expected_angle, rel=1e-8
    )
    # The non-trivial angle must equal ω·dt
    assert np.max(angles) == pytest.approx(expected_angle, rel=1e-8)


def test_oscillator_mats_damped():
    """For β>0 all discrete eigenvalues must be strictly inside the unit circle."""
    Ad, _, _ = oscillator_mats(beta=0.3, omega=2.0, sigma_proc=0.5, dt=0.1)

    eigvals = np.linalg.eigvals(Ad)
    magnitudes = np.abs(eigvals)

    assert np.all(magnitudes < 1.0), (
        f"Expected all |λ| < 1 for damped oscillator, got magnitudes={magnitudes}"
    )


# ---------------------------------------------------------------------------
# oscillator_mats — positive semi-definite noise covariance
# ---------------------------------------------------------------------------


def test_qd_is_psd():
    """Qd must be symmetric positive semi-definite (all eigenvalues >= -1e-10)."""
    _, Qd, _ = oscillator_mats(beta=0.1, omega=1.5, sigma_proc=0.8, dt=0.05)

    # Symmetry
    assert Qd == pytest.approx(Qd.T, abs=1e-12)

    # PSD: smallest eigenvalue must not be significantly negative
    min_eigval = float(np.linalg.eigvalsh(Qd).min())
    assert min_eigval >= -1e-10, (
        f"Qd has a negative eigenvalue {min_eigval!r}, not positive semi-definite"
    )
