"""04_energy.py — Energy tracking over 100 steps.

Shows how to use energy() and delta_energy() to monitor whether a damped
oscillator is losing energy monotonically.
"""

from __future__ import annotations

import numpy as np

from linoss_dynamics import (
    damped_linoss_step,
    delta_energy,
    energy,
)


def main() -> None:
    rng = np.random.default_rng(0)

    # 2-oscillator system with mild damping — no external forcing.
    A = np.array([1.0, 4.0])
    G = np.array([0.2, 0.3])   # per-oscillator damping
    dt = 0.1
    N = 100

    # Start from a non-zero state.
    y = rng.standard_normal(2) * 2.0
    z = rng.standard_normal(2) * 1.0

    print("=== 04 Energy Tracking ===")
    print(f"A={A}, G={G}, dt={dt}, N={N} steps")
    print(f"Initial y={y}, z={z}")
    print()

    # Step forward, recording energy at each step.
    energies: list[float] = [energy(y, z, A)]

    for _ in range(N):
        y, z, _ = damped_linoss_step(y, z, A, G, dt=dt)
        energies.append(energy(y, z, A))

    # Compute delta_energy between consecutive steps.
    deltas = [delta_energy(energies[t], energies[t + 1]) for t in range(N)]

    print(f"Energy at step 0:   {energies[0]:.6f}")
    print(f"Energy at step {N}: {energies[N]:.6f}")
    print(f"Total energy lost:  {energies[0] - energies[N]:.6f}")
    print()
    print(f"delta_energy at step 0→1: {deltas[0]:.6f}  (should be negative)")
    print(f"delta_energy at step 1→2: {deltas[1]:.6f}")
    print()

    # Verify monotonic energy decay — every delta should be <= 0.
    all_negative = all(d <= 1e-12 for d in deltas)
    print(f"All delta_energy values <= 0: {all_negative}")
    if not all_negative:
        bad = [(i, d) for i, d in enumerate(deltas) if d > 1e-12]
        print(f"  Violations at steps: {bad}")

    # Print every 10th energy value so the decay profile is visible.
    print()
    print("Energy profile (every 10th step):")
    print(f"{'step':>5}  {'energy':>14}  {'cumulative delta':>18}")
    for t in range(0, N + 1, 10):
        cum_delta = energies[t] - energies[0]
        print(f"{t:5d}  {energies[t]:14.6f}  {cum_delta:18.6f}")


if __name__ == "__main__":
    main()
