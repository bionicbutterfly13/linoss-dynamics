# Changelog

## [Unreleased]

## [0.2.0] — 2026-05-07

### Added
- `linoss_scan(U, A, dt, ...)` — sequence-level helper that runs `linoss_step` over a `(T, n)` input and returns the full trajectory.
- `damped_oscillator_closed_form(y, z, omega, gamma, dt)` — analytic closed-form damped harmonic oscillator step supporting variable `dt` (irregular sampling). Handles underdamped, critically damped, and overdamped regimes via exact matrix exponential.
- `stability` module: `is_stable`, `eigvals_to_freq_damping`, `freq_damping_to_oscillator_block`, `period_from_omega`, `harmonic_stack`.
- Optional `[probabilistic]` extra (requires scipy):
  - `discretize_lti_with_noise`, `discretize_control`, `oscillator_mats` — exact matrix-exponential discretization helpers (van Loan's method).
  - `kalman_filter`, `rts_smoother` — Bayesian state estimation over oscillator state-space models.
  - `fit_oscillator_mle` — MLE parameter recovery from observed time series via log-space optimization.
- `examples/` directory with 8 runnable tutorials covering single-step, forced, damped, energy, convergence, validation errors, sunspots, and phase tracking.
- `docs/api.md` — per-function API reference.
- `docs/roadmap.md` — public vision document.

### Changed
- `__init__.py` — `__all__` extended with all new public symbols (23 total). Optional scipy-gated symbols included unconditionally; modules raise `LinOSSError` at call time if scipy is missing.
- `README.md` — Public API table extended; new Vision section linking to roadmap; new Quickstart subsections for scan, Kalman filtering, and MLE fitting.
- `PROVENANCE.md` — added framing-only attribution clarification and code-provenance section.
- `docs/architecture.md` — added Lineage section with mermaid timeline diagram.

### Notes
- Core remains NumPy-only. SciPy is gated behind the `[probabilistic]` extra.
- v0.1.0 was never released to PyPI; v0.2.0 is the first PyPI release.

## 0.1.0.dev0 — Unreleased (pre-release work)

- Initial public-alpha split of `linoss-dynamics` as a standalone package.
- NumPy-only LinOSS-style stepping helpers.
- Explicit non-negative damping `G`.
- Energy, delta-energy, and convergence-window helpers.
- Public GitHub polish: expanded README, contribution guidance, issue templates,
  pull-request template, and security policy.
- Release-readiness polish: citation metadata, package author metadata, release
  checklist, and expanded project URLs.
