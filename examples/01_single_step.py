"""01_single_step.py — LinOSS single-step basics.

Demonstrates how to advance a small oscillator state by one timestep using
both supported discretization modes and the backward-compatible alias.
"""

from __future__ import annotations

import numpy as np

from linoss_dynamics import linoss_step, linoss_step_impl


def main() -> None:
    # A 3-oscillator system: stiffness A, displacement y, velocity z.
    # A controls the squared natural frequency for each oscillator.
    rng = np.random.default_rng(0)
    A = np.array([1.0, 4.0, 9.0])        # ω² values: ω = 1, 2, 3 rad/s
    y = rng.standard_normal(3)            # random initial displacement
    z = rng.standard_normal(3)            # random initial velocity
    dt = 0.1

    print("=== 01 Single Step ===")
    print(f"A (stiffness): {A}")
    print(f"y (displacement before): {y}")
    print(f"z (velocity before):     {z}")
    print()

    # --- implicit mode (default, unconditionally stable) ---
    y_im, z_im, metrics_im = linoss_step(y, z, A, dt=dt, mode="implicit")

    print("--- implicit mode ---")
    print(f"y after:  {y_im}")
    print(f"z after:  {z_im}")
    print(f"energy before: {metrics_im['energy_before']:.6f}")
    print(f"energy after:  {metrics_im['energy_after']:.6f}")
    print(f"delta_energy:  {metrics_im['delta_energy']:.6f}")
    print(f"mode:          {metrics_im['mode']}")
    print()

    # --- implicit-explicit (IMEX) mode ---
    # Stability condition: dt^2 * max(A) < 4
    # Here: 0.01 * 9 = 0.09 << 4, so we're well inside the CFL limit.
    y_imex, z_imex, metrics_imex = linoss_step(y, z, A, dt=dt, mode="implicit_explicit")

    print("--- implicit_explicit (IMEX) mode ---")
    print(f"y after:  {y_imex}")
    print(f"z after:  {z_imex}")
    print(f"energy before: {metrics_imex['energy_before']:.6f}")
    print(f"energy after:  {metrics_imex['energy_after']:.6f}")
    print(f"delta_energy:  {metrics_imex['delta_energy']:.6f}")
    print()

    # --- linoss_step_impl is a backward-compatible alias ---
    y_alias, z_alias, _ = linoss_step_impl(y, z, A, dt=dt)
    print("--- linoss_step_impl (alias check) ---")
    print(f"y matches linoss_step result: {np.allclose(y_alias, y_im)}")
    print(f"z matches linoss_step result: {np.allclose(z_alias, z_im)}")


if __name__ == "__main__":
    main()
