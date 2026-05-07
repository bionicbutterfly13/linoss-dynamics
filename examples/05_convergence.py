"""05_convergence.py — Detecting convergence with convergence_window.

Runs a heavily damped oscillator and uses convergence_window() to detect
when the absolute energy change has been below a threshold for 5 consecutive
steps, signalling that the system has effectively settled to rest.
"""

from __future__ import annotations

import numpy as np

from linoss_dynamics import (
    convergence_window,
    damped_linoss_step,
    delta_energy,
    energy,
)


def main() -> None:
    rng = np.random.default_rng(0)

    # Heavy damping drives the oscillator to rest quickly.
    A = np.array([1.0, 2.25])
    G = np.array([0.8, 1.0])
    dt = 0.1
    max_steps = 500
    threshold = 1e-6
    window = 5   # convergence requires 5 consecutive |Δenergy| < threshold

    y = rng.standard_normal(2)
    z = rng.standard_normal(2)

    print("=== 05 Convergence Window ===")
    print(f"A={A}, G={G}, dt={dt}")
    print(f"Convergence: |Δenergy| < {threshold} for {window} consecutive steps")
    print(f"Initial y={y}, z={z}")
    print(f"Initial energy: {energy(y, z, A):.6f}")
    print()

    # Accumulate absolute energy deltas.
    prev_e = energy(y, z, A)
    abs_deltas: list[float] = []
    converged_at: int | None = None

    for step in range(1, max_steps + 1):
        y, z, _ = damped_linoss_step(y, z, A, G, dt=dt)
        curr_e = energy(y, z, A)
        abs_deltas.append(abs(delta_energy(prev_e, curr_e)))
        prev_e = curr_e

        # convergence_window only checks the last `window` entries.
        if convergence_window(abs_deltas, threshold=threshold, window=window):
            converged_at = step
            break

    if converged_at is not None:
        print(f"Convergence detected at step {converged_at}")
        print(f"Final energy: {energy(y, z, A):.2e}")
        print(f"Last {window} |Δenergy| values: "
              f"{[f'{v:.2e}' for v in abs_deltas[-window:]]}")
    else:
        print(f"Did not converge within {max_steps} steps")
        print(f"Final energy: {energy(y, z, A):.6f}")

    # Also show the early high-energy region so the reader can see the decay.
    print()
    print("Early absolute delta_energy values:")
    for i, d in enumerate(abs_deltas[:10], start=1):
        print(f"  step {i:3d}: |Δenergy| = {d:.4e}")


if __name__ == "__main__":
    main()
