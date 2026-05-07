"""02_forced.py — Forced oscillator trajectory with linoss_scan.

Builds a sinusoidal forcing sequence and runs it through a small LinOSS system
to produce the full (T+1) trajectory.  Energy is tracked at every step.
"""

from __future__ import annotations

import numpy as np

from linoss_dynamics import energy
from linoss_dynamics.solver import linoss_scan


def ascii_plot(values: list[float], width: int = 50, label: str = "") -> None:
    """Minimal ASCII bar chart — positive values only, normalised to `width`."""
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    print(f"  {label} (min={lo:.3f}, max={hi:.3f})")
    for i, v in enumerate(values):
        bar_len = int((v - lo) / span * width)
        print(f"  t={i:3d} |{'#' * bar_len}")


def main() -> None:
    # 2-oscillator system
    n = 2
    A = np.array([1.0, 4.0])   # ω² for each oscillator
    dt = 0.05
    T = 80                      # number of timesteps

    # Sinusoidal forcing sequence — shape (T, n)
    t_vec = np.arange(T) * dt
    U = np.column_stack([
        np.sin(2.0 * np.pi * 0.5 * t_vec),   # 0.5 Hz drive for oscillator 0
        np.cos(2.0 * np.pi * 1.0 * t_vec),   # 1.0 Hz drive for oscillator 1
    ])

    print("=== 02 Forced Oscillator ===")
    print(f"n={n} oscillators, T={T} steps, dt={dt}")
    print(f"U shape: {U.shape}")

    # Run the scan — returns full trajectory including initial state at index 0
    Y, Z, metrics = linoss_scan(U, A, dt=dt, mode="implicit")

    print(f"Y shape: {Y.shape}  (T+1 rows, one per timestep + initial state)")
    print(f"Z shape: {Z.shape}")
    print(f"metrics list length: {len(metrics)}  (one dict per step)")
    print()

    # Compute energy at every step using the energy() helper
    energies = [energy(Y[t], Z[t], A) for t in range(T + 1)]
    print(f"Energy at t=0:   {energies[0]:.6f}")
    print(f"Energy at t={T}: {energies[-1]:.6f}")
    print()

    # Print every 10th step so the output is readable
    print("Step-by-step excerpt (every 10th timestep):")
    print(f"{'t':>4}  {'y[0]':>10}  {'y[1]':>10}  {'energy':>12}")
    for t in range(0, T + 1, 10):
        print(f"{t:4d}  {Y[t, 0]:10.4f}  {Y[t, 1]:10.4f}  {energies[t]:12.6f}")
    print()

    # ASCII energy plot (every 5th step to keep output manageable)
    ascii_plot(energies[::5], width=40, label="Energy profile (every 5th step)")


if __name__ == "__main__":
    main()
