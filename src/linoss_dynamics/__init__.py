"""NumPy runtime helpers for LinOSS-style oscillator dynamics.

The package keeps the core dependency-light and separates stiffness/frequency
(`A`) from explicit damping/forgetting (`G`).
"""

from .continuous import damped_oscillator_closed_form
from .filters import kalman_filter, rts_smoother
from .solver import (
    InvalidDampingError,
    InvalidShapeError,
    LinOSSError,
    UnsupportedModeError,
    convergence_window,
    damped_linoss_step,
    delta_energy,
    energy,
    linoss_scan,
    linoss_step,
    linoss_step_impl,
)
from .stability import (
    eigvals_to_freq_damping,
    freq_damping_to_oscillator_block,
    harmonic_stack,
    is_stable,
    period_from_omega,
)

# Optional probabilistic API (scipy-gated).  Symbols are resolved lazily via
# __getattr__ so that importing the package never fails on a scipy-absent
# install, but callers get a clear install hint if they try to use them.

_PROBABILISTIC_SYMBOLS: dict[str, tuple[str, str]] = {
    "discretize_control": ("linoss_dynamics.discretize", "discretize_control"),
    "discretize_lti_with_noise": ("linoss_dynamics.discretize", "discretize_lti_with_noise"),
    "oscillator_mats": ("linoss_dynamics.discretize", "oscillator_mats"),
    "fit_oscillator_mle": ("linoss_dynamics.fit", "fit_oscillator_mle"),
}

try:
    import scipy  # noqa: F401

    _HAS_PROBABILISTIC = True
except ImportError:
    _HAS_PROBABILISTIC = False


def __getattr__(name: str):  # noqa: ANN201
    """Lazy-load scipy-gated symbols with a helpful error on missing scipy."""
    if name in _PROBABILISTIC_SYMBOLS:
        if not _HAS_PROBABILISTIC:
            raise LinOSSError(
                f"{name!r} requires the optional scipy dependency. "
                f"Install with: pip install linoss-dynamics[probabilistic]"
            )
        import importlib

        module_path, symbol_name = _PROBABILISTIC_SYMBOLS[name]
        module = importlib.import_module(module_path)
        return getattr(module, symbol_name)
    raise AttributeError(f"module 'linoss_dynamics' has no attribute {name!r}")

__all__ = [
    # Core errors
    "InvalidDampingError",
    "InvalidShapeError",
    "LinOSSError",
    "UnsupportedModeError",
    # Continuous closed-form
    "damped_oscillator_closed_form",
    # Solver
    "convergence_window",
    "damped_linoss_step",
    "delta_energy",
    "energy",
    "linoss_scan",
    "linoss_step",
    "linoss_step_impl",
    # Stability
    "eigvals_to_freq_damping",
    "freq_damping_to_oscillator_block",
    "harmonic_stack",
    "is_stable",
    "period_from_omega",
    # Filters
    "kalman_filter",
    "rts_smoother",
    # Optional probabilistic (scipy-gated)
    "discretize_control",
    "discretize_lti_with_noise",
    "fit_oscillator_mle",
    "oscillator_mats",
]
