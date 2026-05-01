"""NumPy runtime helpers for LinOSS-style oscillator dynamics.

The package keeps the core dependency-light and separates stiffness/frequency
(`A`) from explicit damping/forgetting (`G`).
"""

from .solver import (
    InvalidDampingError,
    InvalidShapeError,
    LinOSSError,
    UnsupportedModeError,
    convergence_window,
    damped_linoss_step,
    delta_energy,
    energy,
    linoss_step,
    linoss_step_impl,
)

__all__ = [
    "InvalidDampingError",
    "InvalidShapeError",
    "LinOSSError",
    "UnsupportedModeError",
    "convergence_window",
    "damped_linoss_step",
    "delta_energy",
    "energy",
    "linoss_step",
    "linoss_step_impl",
]
