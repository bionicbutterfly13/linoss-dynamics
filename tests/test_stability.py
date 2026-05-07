"""Tests for linoss_dynamics.stability module."""

from __future__ import annotations

import numpy as np
import pytest

from linoss_dynamics.stability import (
    eigvals_to_freq_damping,
    freq_damping_to_oscillator_block,
    harmonic_stack,
    is_stable,
    period_from_omega,
)
from linoss_dynamics.solver import InvalidShapeError, UnsupportedModeError


# ---------------------------------------------------------------------------
# is_stable — implicit mode
# ---------------------------------------------------------------------------


def test_is_stable_implicit_positive_A_is_stable():
    stable, reason = is_stable(np.array([1.0, 2.0]), mode="implicit")
    assert stable is True
    assert "implicit" in reason


def test_is_stable_implicit_zero_A_is_stable():
    stable, reason = is_stable(0.0, mode="implicit")
    assert stable is True


def test_is_stable_implicit_negative_A_is_unstable():
    stable, reason = is_stable(np.array([1.0, -0.5]), mode="implicit")
    assert stable is False
    assert "A" in reason and "negative" in reason


def test_is_stable_implicit_positive_G_is_stable():
    stable, reason = is_stable(np.array([1.0]), G=np.array([0.5]), mode="implicit")
    assert stable is True
    assert "G" in reason


def test_is_stable_implicit_negative_G_is_unstable():
    stable, reason = is_stable(np.array([1.0]), G=np.array([-0.1]), mode="implicit")
    assert stable is False
    assert "G" in reason and "negative" in reason


def test_is_stable_im_alias():
    stable, _ = is_stable(np.array([1.0]), mode="im")
    assert stable is True


# ---------------------------------------------------------------------------
# is_stable — IMEX mode
# ---------------------------------------------------------------------------


def test_is_stable_imex_no_dt_returns_false():
    stable, reason = is_stable(np.array([1.0]), mode="implicit_explicit", dt=None)
    assert stable is False
    assert "dt required" in reason


def test_is_stable_imex_cfl_satisfied_is_stable():
    # dt^2 * max(A) = 0.1^2 * 2.0 = 0.02 < 4  → stable
    stable, reason = is_stable(np.array([1.0, 2.0]), dt=0.1, mode="implicit_explicit")
    assert stable is True


def test_is_stable_imex_cfl_violated_is_unstable():
    # dt^2 * max(A) = 2.0^2 * 2.0 = 8.0 >= 4  → unstable
    stable, reason = is_stable(np.array([2.0]), dt=2.0, mode="implicit_explicit")
    assert stable is False
    assert "CFL" in reason


def test_is_stable_imex_negative_A_is_unstable():
    stable, reason = is_stable(np.array([-1.0]), dt=0.1, mode="implicit_explicit")
    assert stable is False
    assert "A" in reason


def test_is_stable_imex_negative_G_is_unstable():
    stable, reason = is_stable(
        np.array([1.0]), G=np.array([-0.1]), dt=0.1, mode="implicit_explicit"
    )
    assert stable is False
    assert "G" in reason


def test_is_stable_imex_alias():
    stable, _ = is_stable(np.array([1.0]), dt=0.1, mode="imex")
    assert stable is True


@pytest.mark.parametrize("mode", ["implicit", "implicit_explicit"])
def test_is_stable_mode_parametrized_positive_A(mode):
    kwargs: dict = {"A": np.array([0.5, 1.0]), "mode": mode}
    if mode == "implicit_explicit":
        kwargs["dt"] = 0.1
    stable, _ = is_stable(**kwargs)
    assert stable is True


def test_is_stable_unsupported_mode_raises():
    with pytest.raises(UnsupportedModeError):
        is_stable(np.array([1.0]), mode="explicit_runge_kutta")


def test_is_stable_empty_A_implicit():
    """Empty A is vacuously stable in implicit mode."""
    stable, reason = is_stable(np.array([]), mode="implicit")
    assert stable is True
    assert "vacuously" in reason


def test_is_stable_empty_A_imex():
    """Empty A is vacuously stable in IMEX mode (no np.max on empty array)."""
    stable, reason = is_stable(np.array([]), mode="implicit_explicit", dt=0.1)
    assert stable is True
    assert "vacuously" in reason


# ---------------------------------------------------------------------------
# eigvals_to_freq_damping
# ---------------------------------------------------------------------------


def test_eigvals_to_freq_damping_unit_circle_zero_damping():
    # r=1, theta=pi/4  → freq=pi/4, damping=0
    eig = np.array([np.exp(1j * np.pi / 4)])
    freqs, damps = eigvals_to_freq_damping(eig)
    np.testing.assert_allclose(freqs, [np.pi / 4])
    np.testing.assert_allclose(damps, [0.0], atol=1e-12)


def test_eigvals_to_freq_damping_decaying_eigenvalue():
    # r=0.5, theta=pi/3  → freq=pi/3, damping=-ln(0.5)=ln(2)
    r = 0.5
    theta = np.pi / 3
    eig = np.array([r * np.exp(1j * theta)])
    freqs, damps = eigvals_to_freq_damping(eig)
    np.testing.assert_allclose(freqs, [theta])
    np.testing.assert_allclose(damps, [np.log(2)], rtol=1e-7)


def test_eigvals_to_freq_damping_real_positive_eigenvalue():
    # r=0.8, theta=0  → freq=0, damping=-ln(0.8)
    eig = np.array([0.8 + 0j])
    freqs, damps = eigvals_to_freq_damping(eig)
    np.testing.assert_allclose(freqs, [0.0], atol=1e-12)
    np.testing.assert_allclose(damps, [-np.log(0.8)], rtol=1e-7)


def test_eigvals_to_freq_damping_multiple_eigenvalues():
    r1, theta1 = 1.0, np.pi / 6
    r2, theta2 = 0.9, np.pi / 3
    eigs = np.array([r1 * np.exp(1j * theta1), r2 * np.exp(1j * theta2)])
    freqs, damps = eigvals_to_freq_damping(eigs)
    assert freqs.shape == (2,)
    assert damps.shape == (2,)
    np.testing.assert_allclose(freqs[0], theta1)
    np.testing.assert_allclose(freqs[1], theta2)
    np.testing.assert_allclose(damps[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(damps[1], -np.log(0.9), rtol=1e-7)


def test_eigvals_roundtrip_via_oscillator_block():
    """Eigenvalues of a freq_damping_to_oscillator_block round-trip through eigvals_to_freq_damping."""
    omega = 1.2
    gamma = 0.05
    dt = 0.1
    block = freq_damping_to_oscillator_block(omega, gamma, dt)
    eigs = np.linalg.eigvals(block)
    freqs, damps = eigvals_to_freq_damping(eigs)
    # Frequencies: both eigenvalues should have |theta| = omega * dt
    np.testing.assert_allclose(np.sort(freqs), [omega * dt, omega * dt], rtol=1e-7)
    # Damping: both should equal gamma * dt
    np.testing.assert_allclose(damps, [gamma * dt, gamma * dt], rtol=1e-6)


# ---------------------------------------------------------------------------
# freq_damping_to_oscillator_block
# ---------------------------------------------------------------------------


def test_freq_damping_to_oscillator_block_shape():
    block = freq_damping_to_oscillator_block(1.0, 0.1)
    assert block.shape == (2, 2)


def test_freq_damping_to_oscillator_block_zero_gamma_is_rotation():
    # With zero damping, block should be a pure rotation matrix (orthogonal, det=1)
    omega = 0.5
    block = freq_damping_to_oscillator_block(omega, gamma=0.0, dt=1.0)
    np.testing.assert_allclose(block @ block.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(block), 1.0, atol=1e-12)


def test_freq_damping_to_oscillator_block_zero_omega():
    # omega=0, gamma=g → pure damping, no oscillation
    gamma = 0.3
    dt = 1.0
    block = freq_damping_to_oscillator_block(0.0, gamma, dt)
    decay = np.exp(-gamma * dt)
    np.testing.assert_allclose(block, np.array([[decay, 0.0], [0.0, decay]]), atol=1e-12)


def test_freq_damping_to_oscillator_block_negative_omega_raises():
    with pytest.raises(InvalidShapeError, match="non-negative"):
        freq_damping_to_oscillator_block(-1.0, 0.0)


def test_freq_damping_to_oscillator_block_known_values():
    # omega=pi/2, gamma=0, dt=1 → 90-degree rotation
    block = freq_damping_to_oscillator_block(np.pi / 2, 0.0, 1.0)
    expected = np.array([[0.0, -1.0], [1.0, 0.0]])
    np.testing.assert_allclose(block, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# period_from_omega
# ---------------------------------------------------------------------------


def test_period_from_omega_scalar():
    result = period_from_omega(np.pi)
    np.testing.assert_allclose(result, 2.0)


def test_period_from_omega_array():
    omegas = np.array([np.pi, 2.0 * np.pi])
    result = period_from_omega(omegas)
    np.testing.assert_allclose(result, [2.0, 1.0])


def test_period_from_omega_vectorized_shape():
    omegas = np.array([1.0, 2.0, 4.0])
    result = period_from_omega(omegas)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, 2.0 * np.pi / omegas)


def test_period_from_omega_zero_raises():
    with pytest.raises(InvalidShapeError, match="positive"):
        period_from_omega(0.0)


def test_period_from_omega_negative_raises():
    with pytest.raises(InvalidShapeError, match="positive"):
        period_from_omega(-1.0)


def test_period_from_omega_array_with_nonpositive_raises():
    with pytest.raises(InvalidShapeError):
        period_from_omega(np.array([1.0, 0.0, 2.0]))


# ---------------------------------------------------------------------------
# harmonic_stack
# ---------------------------------------------------------------------------


def test_harmonic_stack_single_oscillator_2x2():
    omega = 1.0
    gamma = 0.0
    stack = harmonic_stack([omega], [gamma], dt=1.0)
    assert stack.shape == (2, 2)
    expected = freq_damping_to_oscillator_block(omega, gamma, dt=1.0)
    np.testing.assert_allclose(stack, expected)


def test_harmonic_stack_two_oscillators_4x4():
    omegas = [1.0, 2.0]
    dampings = [0.0, 0.1]
    dt = 0.5
    stack = harmonic_stack(omegas, dampings, dt=dt)
    assert stack.shape == (4, 4)

    block0 = freq_damping_to_oscillator_block(omegas[0], dampings[0], dt)
    block1 = freq_damping_to_oscillator_block(omegas[1], dampings[1], dt)

    np.testing.assert_allclose(stack[:2, :2], block0)
    np.testing.assert_allclose(stack[2:, 2:], block1)
    # Off-diagonal blocks should be zero
    np.testing.assert_allclose(stack[:2, 2:], np.zeros((2, 2)), atol=1e-15)
    np.testing.assert_allclose(stack[2:, :2], np.zeros((2, 2)), atol=1e-15)


def test_harmonic_stack_no_dampings_defaults_to_zero():
    omegas = [0.5, 1.5]
    stack = harmonic_stack(omegas, dt=1.0)
    assert stack.shape == (4, 4)
    stack_with_zeros = harmonic_stack(omegas, [0.0, 0.0], dt=1.0)
    np.testing.assert_allclose(stack, stack_with_zeros)


def test_harmonic_stack_mismatched_lengths_raises():
    with pytest.raises(InvalidShapeError, match="same length"):
        harmonic_stack([1.0, 2.0], [0.1], dt=1.0)


def test_harmonic_stack_three_oscillators_6x6():
    omegas = [0.5, 1.0, 2.0]
    stack = harmonic_stack(omegas)
    assert stack.shape == (6, 6)
