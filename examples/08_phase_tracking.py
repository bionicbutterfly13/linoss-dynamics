"""08_phase_tracking.py — Real-time phase estimation via Kalman filter.

Simulates a single damped oscillator with additive observation noise, then
uses kalman_filter() to obtain filtered position/velocity estimates.  From
those estimates we extract:

  - Instantaneous amplitude: sqrt(y_filt^2 + (z_filt / omega)^2)
  - Instantaneous phase:    arctan2(z_filt / omega, y_filt)

and compare against ground truth.

Requires the [probabilistic] extra:
    pip install linoss-dynamics[probabilistic]
"""

from __future__ import annotations

try:
    from scipy import optimize  # noqa: F401
except ImportError:
    print(
        "This example requires scipy.  "
        "Install with: pip install linoss-dynamics[probabilistic]"
    )
    raise SystemExit(0)

import math

import numpy as np

from linoss_dynamics.discretize import oscillator_mats
from linoss_dynamics.filters import kalman_filter


def simulate_oscillator(
    omega: float,
    beta: float,
    sigma_proc: float,
    sigma_obs: float,
    T: int,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a noisy damped oscillator, return (y_obs, x_true, times).

    Returns:
        y_obs:  (T,) noisy scalar observations.
        x_true: (T, 2) ground-truth [position, velocity] states.
        times:  (T,) time vector in seconds.
    """
    Ad, Qd, _ = oscillator_mats(beta, omega, sigma_proc, dt)
    # Cholesky for process noise sampling
    L_q = np.linalg.cholesky(Qd + 1e-12 * np.eye(2))

    x = np.array([1.0, 0.0])  # start at amplitude=1, phase=0
    x_true = np.empty((T, 2))
    y_obs = np.empty(T)

    for t in range(T):
        x_true[t] = x
        # Observation: position + Gaussian noise
        y_obs[t] = x[0] + sigma_obs * rng.standard_normal()
        # Propagate state
        w = L_q @ rng.standard_normal(2)
        x = Ad @ x + w

    times = np.arange(T) * dt
    return y_obs, x_true, times


def main() -> None:
    rng = np.random.default_rng(0)

    omega = 2.0 * math.pi / 11.0  # ~11-year period (sunspot-like)
    beta = 0.02                    # gentle damping
    sigma_proc = 0.05              # process noise std
    sigma_obs = 0.15               # observation noise std
    T = 100
    dt = 1.0                       # annual steps

    print("=== 08 Phase Tracking ===")
    print(f"omega={omega:.4f} rad/step, period={2*math.pi/omega:.1f} steps")
    print(f"beta={beta}, sigma_proc={sigma_proc}, sigma_obs={sigma_obs}")
    print(f"T={T} steps, dt={dt}")
    print()

    y_obs, x_true, times = simulate_oscillator(
        omega, beta, sigma_proc, sigma_obs, T, dt, rng
    )

    # Build system matrices for the Kalman filter.
    Ad, Qd, _ = oscillator_mats(beta, omega, sigma_proc, dt)
    H = np.array([[1.0, 0.0]])          # observe position only
    R = np.array([[sigma_obs**2]])

    # Run the Kalman filter.
    filt = kalman_filter(y_obs, Ad, Qd, H, R)
    m_f = filt["m_f"]  # shape (T, 2): [position, velocity] posterior means

    # Extract filtered position and velocity.
    y_filt = m_f[:, 0]  # filtered position estimate
    z_filt = m_f[:, 1]  # filtered velocity estimate

    # Convert to instantaneous amplitude and phase.
    # For a simple harmonic oscillator: x(t) = A·cos(omega·t + phi)
    # velocity z = dx/dt = -A·omega·sin(omega·t + phi)
    # => z / omega ≈ -A·sin(...)
    # => amplitude = sqrt(y^2 + (z/omega)^2)
    # => phase = arctan2(-z/omega, y)   [consistent with cos convention]
    amp_filt = np.sqrt(y_filt**2 + (z_filt / omega) ** 2)
    phase_filt = np.arctan2(-z_filt / omega, y_filt)

    # Ground-truth amplitude and phase from the true state.
    y_true = x_true[:, 0]
    z_true = x_true[:, 1]
    amp_true = np.sqrt(y_true**2 + (z_true / omega) ** 2)
    phase_true = np.arctan2(-z_true / omega, y_true)

    print("Phase estimates vs ground truth (first 10 steps):")
    print(f"{'t':>4}  {'y_obs':>8}  {'y_filt':>8}  {'phi_true':>10}  {'phi_filt':>10}  {'amp_filt':>10}")
    for t in range(10):
        print(
            f"{t:4d}  {y_obs[t]:8.3f}  {y_filt[t]:8.3f}  "
            f"{phase_true[t]:10.4f}  {phase_filt[t]:10.4f}  {amp_filt[t]:10.4f}"
        )

    print()
    print("Phase estimates vs ground truth (last 5 steps):")
    print(f"{'t':>4}  {'phi_true':>10}  {'phi_filt':>10}  {'amp_true':>10}  {'amp_filt':>10}")
    for t in range(T - 5, T):
        print(
            f"{t:4d}  {phase_true[t]:10.4f}  {phase_filt[t]:10.4f}  "
            f"{amp_true[t]:10.4f}  {amp_filt[t]:10.4f}"
        )

    # Summary statistics
    phase_rmse = float(np.sqrt(np.mean((phase_filt - phase_true) ** 2)))
    amp_rmse = float(np.sqrt(np.mean((amp_filt - amp_true) ** 2)))
    print()
    print(f"Phase RMSE: {phase_rmse:.4f} rad")
    print(f"Amplitude RMSE: {amp_rmse:.4f}")
    print(f"Filter log-likelihood: {filt['loglik']:.2f}")


if __name__ == "__main__":
    main()
