"""03_damped.py — Damped vs undamped oscillator comparison.

Uses linoss_scan with a damping parameter G to show energy decay, then
validates against the analytic closed-form solution from continuous.py.
"""

from __future__ import annotations

import numpy as np

from linoss_dynamics import energy
from linoss_dynamics.continuous import damped_oscillator_closed_form
from linoss_dynamics.solver import linoss_scan


def run_scan(
    A: np.ndarray,
    T: int,
    dt: float,
    *,
    damping: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run T forced steps; return Y, Z, per-step energies."""
    # Scalar forcing — impulse at t=0 only
    U = np.zeros((T, A.size))
    U[0] = 1.0

    Y, Z, _ = linoss_scan(U, A, dt=dt, damping=damping)
    energies = [energy(Y[t], Z[t], A) for t in range(T + 1)]
    return Y, Z, energies


def main() -> None:
    # Single oscillator for easy comparison with the analytic solution.
    # omega = sqrt(A) = 2 rad/s, so period ≈ 3.14 s.
    omega = 2.0
    A = np.array([omega**2])   # LinOSS stiffness = omega^2
    gamma = 0.15               # damping coefficient
    G = np.array([gamma])      # LinOSS explicit damping vector
    dt = 0.05
    T = 50

    print("=== 03 Damped vs Undamped ===")
    print(f"omega={omega} rad/s, gamma={gamma}, dt={dt}, T={T} steps")
    print()

    Y_un, Z_un, energies_un = run_scan(A, T, dt)
    Y_dm, Z_dm, energies_dm = run_scan(A, T, dt, damping=G)

    print("Energy at key timesteps:")
    print(f"{'t':>5}  {'undamped':>14}  {'damped':>14}  {'ratio':>10}")
    for t in [0, 10, 25, 50]:
        e_un = energies_un[t]
        e_dm = energies_dm[t]
        ratio = e_dm / (e_un or 1e-30)
        print(f"{t:5d}  {e_un:14.6f}  {e_dm:14.6f}  {ratio:10.4f}")
    print()

    # Analytic comparison — closed-form exact solution for the damped oscillator.
    # After the impulse step at t=0, Y_dm[1] and Z_dm[1] hold the first true state.
    # We advance the analytic solver from that state and compare with the scan.
    y_a = Y_dm[1].copy()
    z_a = Z_dm[1].copy()

    print("Analytic closed-form comparison (selected checkpoints):")
    print(f"{'step':>5}  {'LinOSS y':>12}  {'analytic y':>12}  {'abs diff':>12}")

    for t in range(2, T + 1):
        y_a, z_a = damped_oscillator_closed_form(
            y_a, z_a, omega=omega, gamma=gamma, dt=dt
        )
        if t in {5, 10, 20, 35, 50}:
            linoss_y = float(Y_dm[t, 0])
            diff = abs(linoss_y - float(y_a[0]))
            print(f"{t:5d}  {linoss_y:12.6f}  {float(y_a[0]):12.6f}  {diff:12.2e}")

    print()
    print("Note: small discrepancies are expected — LinOSS uses a discrete")
    print("approximation while the closed-form is exact for the continuous ODE.")


if __name__ == "__main__":
    main()
