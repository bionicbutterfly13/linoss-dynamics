# Heredoc Context Packet

This document is a copy-ready context packet for agents, reviews, and prompts that need
one bounded description of what `linoss-dynamics` is, what problem it solves, which
symbols exist in the code, and how the package relates to LinOSS and adjacent
oscillatory state-space technologies.

````bash
cat <<'LINOSS_DYNAMICS_CONTEXT'
# linoss-dynamics Context

## One-line Scope

`linoss-dynamics` is a NumPy runtime package for LinOSS-style oscillator dynamics
with explicit damping support, inspired by LinOSS and D-LinOSS research.

It is not the official LinOSS research repository, does not claim invention of
LinOSS or D-LinOSS, and does not replace JAX, Discretax, or the upstream
training-time LinOSS stack.

## Problem This Code Solves

Many sequence and control problems contain oscillatory latent state:

- a signal has position-like and velocity-like components;
- the useful state cycles, rings, resonates, decays, or tracks phase;
- the model needs stable stepping over time instead of an unconstrained recurrent
  update;
- a runtime system needs deterministic replay, simple arrays, and minimal
  dependencies.

`linoss-dynamics` solves the runtime side of that problem. It gives callers a
small set of deterministic primitives for stepping, damping, scanning,
discretizing, filtering, smoothing, and fitting oscillator state-space models.

The package keeps two concepts separate:

- `A` is stiffness/frequency. It controls how strongly displacement pulls the
  system back toward equilibrium.
- `G` is damping/forgetting. It controls energy dissipation and must be
  non-negative.

That separation is the central package contract. Damping is not hidden inside
`A`, and negative damping is rejected on the explicit damping path.

## Oscillatory State Spaces

An oscillatory state-space model represents state evolution with latent variables
that naturally rotate or resonate. A common second-order form is:

```text
y_dot = z
z_dot = -omega^2 * y - 2 * gamma * z + forcing
```

where:

- `y` is position, displacement, or observed signal level;
- `z` is velocity, momentum, or phase derivative;
- `omega` is angular frequency;
- `gamma` or `G` is damping;
- forcing injects external input.

The state-space framing matters because it separates:

- transition dynamics: how latent state moves through time;
- observation dynamics: how latent state emits data;
- inference: how observations update beliefs about latent state;
- control: how inputs drive future state.

This package covers all four at a small runtime scale:

- deterministic stepping: `linoss_step`, `damped_linoss_step`, `linoss_scan`;
- continuous-time exact stepping: `damped_oscillator_closed_form`;
- stability and frequency utilities: `is_stable`, `harmonic_stack`, and related
  helpers;
- probabilistic inference: `kalman_filter`, `rts_smoother`, `fit_oscillator_mle`
  when SciPy is installed through the `probabilistic` extra.

## Relationship To LinOSS

LinOSS, from the oscillatory state-space model research line, uses second-order
oscillator structure as a sequence-modeling primitive. The upstream
`tk-rusch/linoss` repository is a JAX/Equinox training-time implementation with
learnable layers and scan-based sequence processing.

`linoss-dynamics` is different in purpose:

- it is runtime-oriented, not training-framework-oriented;
- it uses NumPy as the core dependency;
- it exposes functions, not neural-network layer classes;
- it provides single-step and sequence-step helpers for deterministic systems;
- it makes explicit non-negative damping available through `G`;
- it adds energy, convergence, stability, Kalman, RTS, discretization, and MLE
  utilities around the oscillator runtime seam.

Use the upstream LinOSS implementation when training LinOSS neural sequence
models. Use `linoss-dynamics` when a host application needs a small deterministic
oscillator physics primitive.

## Related Technology Categories

`linoss-dynamics` sits near several technology families:

| Category | Similarity | Difference |
| --- | --- | --- |
| Classical state-space models and Kalman filters | Transition, observation, process noise, and posterior state estimation | This package centers the transition on oscillator dynamics and also exposes deterministic step helpers. |
| Structural time series and stochastic resonators | Oscillatory latent components explain cycles and phase | This package is a lightweight runtime primitive, not a full forecasting platform. |
| LinOSS and D-LinOSS | Second-order oscillatory state-space dynamics and damping motivation | This package is an independent NumPy runtime implementation, not the official research code. |
| S4, S5, Mamba, and modern sequence SSMs | State-space structure for long sequences | Those systems are training-time deep-learning architectures; this package is a small numerical runtime. |
| coRNN, UnICORNN, and oscillator RNNs | Recurrent dynamics constrained by oscillator physics | This package exposes physics stepping and inference helpers rather than neural recurrent cells. |
| DMD and Koopman methods | Spectral modes, eigenvalues, frequency, and damping | This package currently provides oscillator blocks and eigenvalue conversion, not full DMD identification. |
| Signal processing filters | Stable time-series smoothing and phase/frequency tracking | This package keeps the state-space model explicit and programmable. |
| Control systems | State transition, forcing input, damping, and stability checks | This package is a primitive layer, not a full control-design toolbox. |
| Neural ODE/CDE and continuous-time models | Continuous-time dynamics and irregular-time stepping | This package uses closed-form damped oscillator steps, not learned continuous vector fields. |

## Package Properties

- Package name: `linoss-dynamics`
- Import name: `linoss_dynamics`
- Core runtime dependency: `numpy>=1.24`
- Optional probabilistic dependency: `scipy>=1.10`
- Python version: `>=3.10`
- Core style: pure functions over explicit arrays
- Stateful model classes: none
- Public class hierarchy: error classes only
- Default public status: public alpha
- Main state variables: `y`, `z`
- Main dynamics parameters: `A`, `G`, `dt`, optional `B`, optional `u`
- Supported step modes: `implicit` / `im`, `implicit_explicit` / `imex`
- Damping contract: explicit `G` must be non-negative
- Dependency boundary: no web framework, graph database, event bus, agent
  framework, host service getter, JAX, or Discretax dependency in the package
  core.

## Public Symbols

### Error Classes

Source: `src/linoss_dynamics/solver.py`

- `LinOSSError(ValueError)`: base package error for LinOSS dynamics failures.
- `InvalidShapeError(LinOSSError)`: raised when arrays cannot be made compatible
  with the oscillator state shape.
- `InvalidDampingError(LinOSSError)`: raised when damping is outside the
  supported stable path, especially negative explicit `G`.
- `UnsupportedModeError(LinOSSError)`: raised when a discretization mode is not
  recognized.

### Solver Functions

Source: `src/linoss_dynamics/solver.py`

- `linoss_step(y, z, A, dt, mode="implicit", B=None, u=None, damping=None, G=None)`
  advances one oscillator step. If `damping` or `G` is provided, it dispatches to
  `damped_linoss_step`.
- `damped_linoss_step(y, z, A, G, dt, mode="implicit", B=None, u=None)` advances
  one step with explicit non-negative damping.
- `linoss_scan(U, A, dt, mode="implicit", B=None, y0=None, z0=None, damping=None,
  G=None)` runs the step function over a 1-D or 2-D input sequence and returns
  the full `(T + 1, n)` trajectories.
- `linoss_step_impl(*args, **kwargs)` is a backward-compatible alias for
  `linoss_step`.
- `energy(y, z, A, G=None)` returns diagonal oscillator energy
  `0.5 * sum(A * y**2 + z**2)`. `G` is accepted but ignored because energy is
  defined by stiffness and velocity, not damping.
- `delta_energy(previous_energy, next_energy)` returns the signed energy change.
- `convergence_window(deltas, threshold, window)` returns true when the latest
  `window` absolute energy deltas are all below `threshold`.

`linoss_step` and `damped_linoss_step` return:

```text
(y_next, z_next, metrics)
```

where `metrics` contains:

- `mode`
- `energy_before`
- `energy_after`
- `delta_energy`
- `damping_mode`
- `damping` when explicit `G` is used

### Stability Functions

Source: `src/linoss_dynamics/stability.py`

- `is_stable(A, G=None, dt=None, mode="implicit")` returns `(stable, reason)`.
  Implicit mode requires non-negative `A` and, when provided, non-negative `G`.
  IMEX mode also requires `dt**2 * max(A) < 4`.
- `eigvals_to_freq_damping(eigenvalues)` maps complex eigenvalues to per-step
  frequency and damping arrays.
- `freq_damping_to_oscillator_block(omega, gamma, dt=1.0)` builds a 2 by 2 real
  discrete oscillator block from frequency and damping.
- `period_from_omega(omega)` returns `2*pi / omega` and rejects non-positive
  frequencies.
- `harmonic_stack(omegas, dampings=None, dt=1.0)` builds a block-diagonal matrix
  from multiple oscillator blocks.

### Continuous-Time Function

Source: `src/linoss_dynamics/continuous.py`

- `damped_oscillator_closed_form(y, z, omega, gamma, dt, forcing=0.0)` advances a
  damped harmonic oscillator with an analytic matrix-exponential solution.

It handles:

- underdamped systems, where `omega > gamma`;
- critically damped systems, where `omega` is approximately `gamma`;
- overdamped systems, where `gamma > omega`;
- constant forcing through a shifted particular solution;
- irregular sampling because each call can use a different positive `dt`.

### Discretization Functions

Source: `src/linoss_dynamics/discretize.py`

These require SciPy through `pip install linoss-dynamics[probabilistic]`.

- `discretize_lti_with_noise(A_c, L, Qc, dt)` uses van Loan's
  matrix-exponential method to return `(Ad, Qd)`.
- `discretize_control(A_c, B, dt)` uses zero-order-hold discretization to return
  `(Ad, Bd)`.
- `oscillator_mats(beta, omega, sigma_proc, dt=1.0, kappa=1.0)` builds `(Ad, Qd,
  Bd)` for a single forced damped oscillator.

### Filtering Functions

Source: `src/linoss_dynamics/filters.py`

- `kalman_filter(y, Ad, Qd, H, R, m0=None, P0=None, Bd=None, u=None)` runs a
  linear-Gaussian Kalman filter over a 1-D observation sequence.
- `rts_smoother(Ad, filt)` runs a Rauch-Tung-Striebel backward smoother over the
  filter output.

`kalman_filter` returns:

- `m_f`: filtered means, shape `(T, n)`;
- `P_f`: filtered covariances, shape `(T, n, n)`;
- `m_p`: predicted means, shape `(T, n)`;
- `P_p`: predicted covariances, shape `(T, n, n)`;
- `loglik`: scalar log-likelihood.

`rts_smoother` returns:

- `m_s`: smoothed means, shape `(T, n)`;
- `P_s`: smoothed covariances, shape `(T, n, n)`.

### Fitting Function

Source: `src/linoss_dynamics/fit.py`

- `fit_oscillator_mle(y, dt, u=None, init=None, method="L-BFGS-B")` fits
  `(beta, omega, sigma_proc, sigma_obs)` by maximizing Kalman filter
  log-likelihood in log-parameter space.

It returns fitted parameters, derived period, centered-data mean, state-space
matrices, filter output, smoother output, and the SciPy optimizer result.

## Internal Symbols

These are implementation helpers, not stable public API.

Source: `src/linoss_dynamics/solver.py`

- `_normalize_mode(mode)` maps aliases to canonical mode names.
- `_state_vectors(y, z)` coerces state vectors and enforces equal shapes.
- `_diagonal_vector(name, value, size)` accepts scalar, length-`n` vector, or
  `n` by `n` diagonal matrix and returns a length-`n` vector.
- `_forcing_vector(B, u, size)` resolves forcing from `B` and `u`.

Source: `src/linoss_dynamics/continuous.py`

- `_CRITICAL_DAMP_TOL` sets the tolerance for the critical damping branch.
- `_broadcast_params(y, z, omega, gamma)` broadcasts continuous oscillator
  parameters and validates positive frequency plus non-negative damping.
- `_matrix_exp_row(omega, gamma, dt)` returns the four entries of the analytic
  matrix exponential for each oscillator.

Source: `src/linoss_dynamics/discretize.py`

- `_SCIPY_INSTALL_MSG` stores the optional dependency hint.
- `_require_scipy()` raises `LinOSSError` when SciPy is unavailable.
- `_check_square(name, arr)` validates a square 2-D matrix and returns its size.

Source: `src/linoss_dynamics/filters.py`

- `_require_shape(name, arr, *expected)` validates exact shape membership.
- `_symmetrize(P)` returns `(P + P.T) / 2` to suppress covariance asymmetry.

Source: `src/linoss_dynamics/fit.py`

- `_SCIPY_INSTALL_MSG` stores the optional dependency hint.
- `_require_scipy()` raises `LinOSSError` when SciPy is unavailable.
- `objective(theta)` is nested inside `fit_oscillator_mle` and evaluates
  negative log-likelihood in log-parameter space.

Source: `src/linoss_dynamics/__init__.py`

- `_PROBABILISTIC_SYMBOLS` maps SciPy-gated public symbols to modules.
- `_HAS_PROBABILISTIC` records whether SciPy imports successfully.
- `__getattr__(name)` lazy-loads optional probabilistic symbols and raises a
  helpful `LinOSSError` when SciPy is missing.
- `__all__` declares the package public import surface.

## Shape And Validation Contracts

- `y` and `z` are flattened to 1-D arrays and must have the same shape.
- `A` and `G` may be scalars, length-`n` vectors, or `n` by `n` diagonal
  matrices.
- `G` must be non-negative on explicit damping paths.
- `u` may be scalar or length-`n` when `B` is omitted.
- vector `B` must have length `n`.
- matrix `B` must have `n` rows.
- `linoss_scan` accepts `U` with shape `(T,)` or `(T, m)` and returns `Y` and
  `Z` with shape `(T + 1, n)`.
- continuous closed-form stepping requires positive `omega`, non-negative
  `gamma`, and positive `dt`.
- probabilistic discretization requires positive `dt`.
- Kalman filtering currently handles a 1-D observation sequence with `H` shape
  `(1, n)` and `R` shape `(1, 1)`.

## When To Use This Package

Use `linoss-dynamics` when the host needs:

- deterministic oscillator stepping in NumPy;
- replay-safe energy and convergence metrics;
- explicit damping separate from stiffness;
- a small baseline for comparing JAX or PyTorch ports;
- exact irregular-time damped oscillator stepping;
- Kalman or RTS inference over a linear-Gaussian oscillator model;
- a clean runtime primitive that host systems can wrap without pulling service
  dependencies into the package core.

Do not use this package as:

- a full neural sequence-model training framework;
- a replacement for the official JAX LinOSS code;
- a full GP, DMD, Koopman, or control-design platform;
- a web service, agent runtime, event bus, graph database, or host application.

## Minimal Usage

```python
import numpy as np
from linoss_dynamics import damped_linoss_step

y = np.array([0.2])
z = np.array([1.0])
A = np.array([1.0])
G = np.array([0.5])

y_next, z_next, metrics = damped_linoss_step(y, z, A, G, dt=0.1)
```

## Verification Commands

Run these from the repository root after changing package behavior:

```bash
python -m pytest tests -v --tb=short
python -m ruff check src tests
python -m compileall -q src tests
```
LINOSS_DYNAMICS_CONTEXT
````
