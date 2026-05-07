"""Tests for damped_oscillator_closed_form in linoss_dynamics.continuous.

Covers:
- Underdamped: oscillatory decay, energy envelope ∝ exp(-2γt)
- Critically damped: no oscillation, monotonic approach to zero
- Overdamped: two-exponential decay
- Zero damping (pure cosine/sine): energy conservation
- Irregular dt: two half-steps == one full step
- Constant forcing: equilibrium at y_p = forcing/ω²
- Vectorized arrays of (y, z, omega, gamma)
- Error cases: dt <= 0, negative gamma, non-positive omega, shape mismatch
"""

from __future__ import annotations

import numpy as np
import pytest

from linoss_dynamics.continuous import damped_oscillator_closed_form
from linoss_dynamics.solver import InvalidDampingError, InvalidShapeError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oscillator_energy(y: np.ndarray, z: np.ndarray, omega: float) -> float:
    """Return total mechanical energy (½ ω² y² + ½ z²)."""
    return float(0.5 * omega**2 * np.sum(y**2) + 0.5 * np.sum(z**2))


# ---------------------------------------------------------------------------
# Underdamped
# ---------------------------------------------------------------------------


def test_underdamped_returns_arrays():
    """Smoke test: shapes and types are correct."""
    y, z = np.array([1.0]), np.array([0.0])
    y1, z1 = damped_oscillator_closed_form(y, z, omega=2.0, gamma=0.3, dt=0.1)
    assert y1.shape == (1,)
    assert z1.shape == (1,)
    assert y1.dtype == float


def test_underdamped_energy_decays_at_correct_rate():
    """Energy envelope decays as exp(-2γ·t) for the underdamped case.

    We start at a velocity-only initial condition (y=0, z=z0) so that energy
    equals ½z0² at t=0 and the amplitude envelope is exactly A·exp(-γt).
    Measuring energy at integer multiples of the damped period (where the
    oscillator returns to z=0) isolates the amplitude decay.
    """
    omega = 3.0
    gamma = 0.5
    omega_d = np.sqrt(omega**2 - gamma**2)
    # Start from rest at maximum velocity so the motion is a pure sine.
    # At t = n * T_d the state returns to (0, z_n) with |z_n| = z0·exp(-γ·n·T_d).
    period_d = 2.0 * np.pi / omega_d
    y0, z0 = np.array([0.0]), np.array([1.0])

    # Advance two full damped periods and measure energy at each.
    state1 = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=period_d)
    state2 = damped_oscillator_closed_form(*state1, omega=omega, gamma=gamma, dt=period_d)

    e0 = _oscillator_energy(y0, z0, omega)
    e1 = _oscillator_energy(state1[0], state1[1], omega)
    e2 = _oscillator_energy(state2[0], state2[1], omega)

    # After each full period energy must have decayed by exp(-2γ·T_d).
    expected_decay = np.exp(-2.0 * gamma * period_d)
    assert np.isclose(e1 / e0, expected_decay, rtol=1e-9), (
        f"After 1 period: expected energy ratio {expected_decay}, got {e1 / e0}"
    )
    assert np.isclose(e2 / e1, expected_decay, rtol=1e-9), (
        f"After 2 periods: expected energy ratio {expected_decay}, got {e2 / e1}"
    )


def test_underdamped_oscillates():
    """Position must change sign within one full oscillation period."""
    omega = 2 * np.pi  # period = 1 s
    gamma = 0.05
    y0, z0 = np.array([1.0]), np.array([0.0])
    # After half-period position should be negative (damping barely shifts it)
    y_half, _ = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=0.5)
    assert y_half[0] < 0.0, "Underdamped oscillator must cross zero within a half-period"


def test_underdamped_amplitude_monotone_per_period():
    """Peak amplitude at each full period is strictly smaller than the previous."""
    omega = 2.0
    gamma = 0.3
    period = 2.0 * np.pi / np.sqrt(omega**2 - gamma**2)
    y0, z0 = np.array([1.0]), np.array([0.0])

    peaks = [1.0]
    state = (y0, z0)
    for _ in range(3):
        state = damped_oscillator_closed_form(*state, omega=omega, gamma=gamma, dt=period)
        peaks.append(float(np.abs(state[0][0])))

    for i in range(1, len(peaks)):
        assert peaks[i] < peaks[i - 1], f"Peak {i} not smaller than peak {i-1}"


# ---------------------------------------------------------------------------
# Critically damped
# ---------------------------------------------------------------------------


def test_critically_damped_no_sign_change():
    """Critically damped oscillator released from rest should not cross zero."""
    omega = 1.5
    gamma = 1.5  # critically damped: γ == ω
    y0, z0 = np.array([1.0]), np.array([0.0])

    signs = []
    state = (y0, z0)
    for _ in range(20):
        state = damped_oscillator_closed_form(*state, omega=omega, gamma=gamma, dt=0.3)
        signs.append(np.sign(state[0][0]))

    # All signs must be the same (no crossing zero)
    assert all(s == signs[0] for s in signs), "Critically damped must not oscillate"


def test_critically_damped_decays_to_zero():
    """Critically damped oscillator must approach zero asymptotically."""
    omega = 2.0
    gamma = 2.0
    y0, z0 = np.array([1.0]), np.array([0.0])
    y_late, z_late = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=20.0)
    assert np.abs(y_late[0]) < 1e-6, "Critically damped y should be near zero after long time"
    assert np.abs(z_late[0]) < 1e-6, "Critically damped z should be near zero after long time"


def test_critically_damped_monotone_decay_from_rest():
    """Position decreases monotonically when released from rest with γ = ω."""
    omega = 1.0
    gamma = 1.0
    y0, z0 = np.array([1.0]), np.array([0.0])

    positions = [1.0]
    state = (y0, z0)
    for _ in range(15):
        state = damped_oscillator_closed_form(*state, omega=omega, gamma=gamma, dt=0.5)
        positions.append(float(state[0][0]))

    for i in range(1, len(positions)):
        assert positions[i] < positions[i - 1], "Critical damping must produce monotone decay"


# ---------------------------------------------------------------------------
# Overdamped
# ---------------------------------------------------------------------------


def test_overdamped_no_oscillation():
    """Overdamped oscillator must not change sign."""
    omega = 1.0
    gamma = 5.0  # strongly overdamped
    y0, z0 = np.array([1.0]), np.array([0.0])

    state = (y0, z0)
    for _ in range(20):
        state = damped_oscillator_closed_form(*state, omega=omega, gamma=gamma, dt=0.2)
        assert state[0][0] > 0.0, "Overdamped oscillator must not cross zero"


def test_overdamped_decays_to_zero():
    """Overdamped oscillator state must vanish at long time.

    The slowest eigenvalue is -γ + sqrt(γ²-ω²).  For ω=1, γ=3:
    λ_slow = -3 + sqrt(8) ≈ -0.172, so τ_slow ≈ 5.8 s.
    After 200 s (>> 34 time-constants) the state must be negligible.
    """
    omega = 1.0
    gamma = 3.0
    y0, z0 = np.array([1.0]), np.array([0.0])
    y_late, z_late = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=200.0)
    assert np.abs(y_late[0]) < 1e-8
    assert np.abs(z_late[0]) < 1e-8


def test_overdamped_slower_than_critically_damped():
    """Overdamped oscillator returns to zero more slowly than critically damped."""
    omega = 1.0
    gamma_crit = 1.0
    gamma_over = 5.0
    y0, z0 = np.array([1.0]), np.array([0.0])
    dt = 2.0

    y_crit, _ = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma_crit, dt=dt)
    y_over, _ = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma_over, dt=dt)

    # Overdamped should still be larger in magnitude at intermediate t
    assert float(np.abs(y_over[0])) > float(np.abs(y_crit[0])), (
        "Overdamped decay is slower than critically damped"
    )


# ---------------------------------------------------------------------------
# Zero damping (pure oscillation)
# ---------------------------------------------------------------------------


def test_zero_damping_energy_conserved():
    """With γ=0 the oscillator is conservative; energy must be preserved."""
    omega = 2.5
    gamma = 0.0
    y0, z0 = np.array([1.0]), np.array([0.5])
    e0 = _oscillator_energy(y0, z0, omega)

    state = (y0, z0)
    for _ in range(50):
        state = damped_oscillator_closed_form(*state, omega=omega, gamma=gamma, dt=0.13)

    e_final = _oscillator_energy(state[0], state[1], omega)
    assert np.isclose(e_final, e0, rtol=1e-9, atol=0.0), (
        f"Energy must be conserved; got {e_final}, expected {e0}"
    )


def test_zero_damping_full_period_returns_to_start():
    """After exactly one period the state must match the initial state."""
    omega = 3.0
    gamma = 0.0
    y0, z0 = np.array([1.0]), np.array([0.0])
    period = 2.0 * np.pi / omega

    y1, z1 = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=period)
    assert np.allclose(y1, y0, rtol=1e-9, atol=1e-12)
    assert np.allclose(z1, z0, rtol=1e-9, atol=1e-12)


def test_zero_damping_quarter_period():
    """At ¼ period, y=0 and z=-ω·y₀ for pure cosine initial condition."""
    omega = 4.0
    gamma = 0.0
    y0, z0 = np.array([1.0]), np.array([0.0])
    quarter = np.pi / (2.0 * omega)

    y1, z1 = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=quarter)
    assert np.allclose(y1, [0.0], atol=1e-12)
    assert np.allclose(z1, [-omega], rtol=1e-9)


# ---------------------------------------------------------------------------
# Irregular dt: two half-steps == one full step
# ---------------------------------------------------------------------------


def test_two_half_steps_equal_one_full_step_underdamped():
    """Stepping twice with dt/2 must give the same result as one step with dt."""
    omega, gamma = 2.0, 0.3
    y0, z0 = np.array([1.5]), np.array([-0.3])
    dt = 0.7

    # One full step
    y_full, z_full = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=dt)

    # Two half-steps
    y_half1, z_half1 = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=dt / 2)
    y_two, z_two = damped_oscillator_closed_form(
        y_half1, z_half1, omega=omega, gamma=gamma, dt=dt / 2
    )

    assert np.allclose(y_two, y_full, rtol=1e-12, atol=1e-14)
    assert np.allclose(z_two, z_full, rtol=1e-12, atol=1e-14)


def test_two_half_steps_equal_one_full_step_overdamped():
    """Same composition property must hold in the overdamped regime."""
    omega, gamma = 1.0, 4.0
    y0, z0 = np.array([2.0]), np.array([0.5])
    dt = 0.4

    y_full, z_full = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=dt)
    y_h1, z_h1 = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=dt / 2)
    y_two, z_two = damped_oscillator_closed_form(y_h1, z_h1, omega=omega, gamma=gamma, dt=dt / 2)

    assert np.allclose(y_two, y_full, rtol=1e-12, atol=1e-14)
    assert np.allclose(z_two, z_full, rtol=1e-12, atol=1e-14)


def test_many_irregular_steps_match_single_step():
    """A sequence of variable-length steps must compose exactly."""
    omega, gamma = 2.5, 0.8
    y0, z0 = np.array([1.0]), np.array([0.0])
    dt_total = 1.3
    # Split into unequal sub-intervals
    dts = [0.1, 0.3, 0.05, 0.4, 0.45]
    assert np.isclose(sum(dts), dt_total, rtol=1e-12)

    y_full, z_full = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=dt_total)

    state = (y0, z0)
    for sub_dt in dts:
        state = damped_oscillator_closed_form(*state, omega=omega, gamma=gamma, dt=sub_dt)

    assert np.allclose(state[0], y_full, rtol=1e-11, atol=1e-13)
    assert np.allclose(state[1], z_full, rtol=1e-11, atol=1e-13)


# ---------------------------------------------------------------------------
# Constant forcing
# ---------------------------------------------------------------------------


def test_forcing_equilibrium():
    """With constant forcing u, equilibrium must be at y_p = u/ω²."""
    omega = 2.0
    gamma = 0.5
    forcing = 4.0
    y_eq = forcing / omega**2  # = 1.0

    # Start from rest at equilibrium; state must not change.
    y0, z0 = np.array([y_eq]), np.array([0.0])
    y1, z1 = damped_oscillator_closed_form(
        y0, z0, omega=omega, gamma=gamma, dt=1.0, forcing=forcing
    )
    assert np.allclose(y1, [y_eq], rtol=1e-12)
    assert np.allclose(z1, [0.0], atol=1e-12)


def test_forcing_convergence_to_equilibrium():
    """Starting away from equilibrium must converge to y_p = u/ω²."""
    omega = 1.5
    gamma = 0.8
    forcing = 3.0
    y_eq = forcing / omega**2

    y0, z0 = np.array([0.0]), np.array([0.0])
    state = (y0, z0)
    for _ in range(200):
        state = damped_oscillator_closed_form(*state, omega=omega, gamma=gamma, dt=0.1, forcing=forcing)

    assert np.allclose(state[0], [y_eq], rtol=1e-6, atol=1e-8), (
        f"Should converge to equilibrium {y_eq}; got {state[0]}"
    )


def test_forcing_zero_same_as_unforced():
    """Explicitly passing forcing=0 must give identical results to the default."""
    omega, gamma = 2.0, 0.4
    y0, z0 = np.array([1.0]), np.array([-0.5])
    dt = 0.5

    y_default, z_default = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=dt)
    y_zero, z_zero = damped_oscillator_closed_form(
        y0, z0, omega=omega, gamma=gamma, dt=dt, forcing=0.0
    )
    assert np.allclose(y_default, y_zero)
    assert np.allclose(z_default, z_zero)


# ---------------------------------------------------------------------------
# Vectorized arrays
# ---------------------------------------------------------------------------


def test_vectorized_array_of_oscillators():
    """Arrays of (y, z, omega, gamma) must produce per-oscillator results."""
    y0 = np.array([1.0, 0.5, 2.0])
    z0 = np.array([0.0, 0.2, -0.1])
    omega = np.array([1.0, 2.0, 3.0])
    gamma = np.array([0.1, 0.5, 1.0])
    dt = 0.4

    y1, z1 = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=dt)
    assert y1.shape == (3,)
    assert z1.shape == (3,)

    # Each entry must match a single-oscillator call.
    for i in range(3):
        y_i, z_i = damped_oscillator_closed_form(
            y0[i : i + 1], z0[i : i + 1], omega=omega[i], gamma=gamma[i], dt=dt
        )
        assert np.allclose(y1[i], y_i[0], rtol=1e-14)
        assert np.allclose(z1[i], z_i[0], rtol=1e-14)


def test_vectorized_scalar_omega_gamma():
    """Scalar omega/gamma broadcast over array y/z."""
    y0 = np.array([1.0, -1.0, 0.5])
    z0 = np.array([0.0, 0.0, 0.0])
    y1, z1 = damped_oscillator_closed_form(y0, z0, omega=2.0, gamma=0.3, dt=0.5)
    assert y1.shape == (3,)

    # Anti-symmetry: y1 for ±y0 must be ±y1 with same z1.
    y_pos, z_pos = damped_oscillator_closed_form(
        np.array([1.0]), np.array([0.0]), omega=2.0, gamma=0.3, dt=0.5
    )
    y_neg, z_neg = damped_oscillator_closed_form(
        np.array([-1.0]), np.array([0.0]), omega=2.0, gamma=0.3, dt=0.5
    )
    assert np.allclose(y_pos, -y_neg, rtol=1e-14)
    assert np.allclose(z_pos, -z_neg, rtol=1e-14)


def test_vectorized_mixed_regimes():
    """Array of oscillators with different damping regimes gives correct output."""
    # Three oscillators: underdamped, critically damped, overdamped
    omega = np.array([3.0, 2.0, 1.0])
    gamma = np.array([0.5, 2.0, 5.0])   # underdamped, critical, overdamped
    y0 = np.array([1.0, 1.0, 1.0])
    z0 = np.array([0.0, 0.0, 0.0])
    dt = 1.0

    y1, z1 = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma, dt=dt)
    assert y1.shape == (3,)

    # Each element must match a single-oscillator call for that regime.
    for i in range(3):
        y_i, z_i = damped_oscillator_closed_form(
            y0[i : i + 1],
            z0[i : i + 1],
            omega=omega[i],
            gamma=gamma[i],
            dt=dt,
        )
        np.testing.assert_allclose(y1[i], y_i[0], rtol=1e-14)
        np.testing.assert_allclose(z1[i], z_i[0], rtol=1e-14)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_error_on_dt_zero():
    """dt == 0 must raise ValueError."""
    with pytest.raises(ValueError, match="dt must be strictly positive"):
        damped_oscillator_closed_form(np.array([1.0]), np.array([0.0]), omega=1.0, gamma=0.1, dt=0.0)


def test_error_on_dt_negative():
    """Negative dt must raise ValueError."""
    with pytest.raises(ValueError, match="dt must be strictly positive"):
        damped_oscillator_closed_form(
            np.array([1.0]), np.array([0.0]), omega=1.0, gamma=0.1, dt=-0.5
        )


def test_error_on_negative_gamma():
    """Negative damping must raise InvalidDampingError."""
    with pytest.raises(InvalidDampingError):
        damped_oscillator_closed_form(
            np.array([1.0]), np.array([0.0]), omega=2.0, gamma=-0.1, dt=0.1
        )


def test_error_on_zero_omega():
    """omega == 0 must raise ValueError (non-positive)."""
    with pytest.raises(ValueError, match="omega must be positive"):
        damped_oscillator_closed_form(
            np.array([1.0]), np.array([0.0]), omega=0.0, gamma=0.1, dt=0.1
        )


def test_error_on_negative_omega():
    """Negative omega must raise ValueError."""
    with pytest.raises(ValueError, match="omega must be positive"):
        damped_oscillator_closed_form(
            np.array([1.0]), np.array([0.0]), omega=-1.0, gamma=0.1, dt=0.1
        )


def test_error_on_shape_mismatch():
    """y and z with incompatible shapes must raise InvalidShapeError."""
    with pytest.raises(InvalidShapeError):
        damped_oscillator_closed_form(
            np.array([1.0, 2.0]),
            np.array([0.0, 0.1, 0.2]),  # length mismatch
            omega=1.0,
            gamma=0.1,
            dt=0.1,
        )


def test_error_on_omega_shape_mismatch():
    """omega array length mismatch must raise InvalidShapeError."""
    with pytest.raises(InvalidShapeError):
        damped_oscillator_closed_form(
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
            omega=np.array([1.0, 2.0, 3.0]),  # wrong length
            gamma=0.1,
            dt=0.1,
        )


# ---------------------------------------------------------------------------
# Regression / edge cases
# ---------------------------------------------------------------------------


def test_scalar_inputs_return_1d_arrays():
    """Scalar y and z must return 1-D arrays of length 1."""
    y1, z1 = damped_oscillator_closed_form(1.0, 0.0, omega=1.0, gamma=0.5, dt=0.2)
    assert y1.ndim == 1
    assert y1.size == 1


def test_near_critically_damped_continuous():
    """Discriminant near zero must not produce NaN (regime boundary stability)."""
    epsilon = 1e-11  # very close to critical
    omega = 1.0
    gamma_under = omega - epsilon
    gamma_over = omega + epsilon

    y0, z0 = np.array([1.0]), np.array([0.0])

    y_u, z_u = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma_under, dt=0.5)
    y_o, z_o = damped_oscillator_closed_form(y0, z0, omega=omega, gamma=gamma_over, dt=0.5)

    assert not np.any(np.isnan(y_u)), "Near-underdamped boundary produced NaN"
    assert not np.any(np.isnan(y_o)), "Near-overdamped boundary produced NaN"
