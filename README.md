# linoss-dynamics

[![CI](https://github.com/bionicbutterfly13/linoss-dynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/bionicbutterfly13/linoss-dynamics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-public%20alpha-orange)

`linoss-dynamics` is a small NumPy runtime package for LinOSS-style oscillator
dynamics with explicit non-negative damping support.

Status: **public alpha**.

## Vision

`linoss-dynamics` is the runtime physics layer for oscillatory state-space dynamics. The current scope (v0.2) covers the deterministic oscillator step, irregular-time stepping, Bayesian filtering, and stability/frequency introspection. The longer roadmap — including unified oscillator + attractor primitives and spatio-temporal extensions — is documented in [docs/roadmap.md](docs/roadmap.md).

## How this differs from the upstream LinOSS reference

The original [tk-rusch/linoss](https://github.com/tk-rusch/linoss) is a
**training-time JAX/Equinox neural-network library** that exposes LinOSS as
trainable layers (`LinOSSLayer`, `LinOSSBlock`, `LinOSS`) inside a deep-learning
model. It is the right choice when you are building or training a sequence model
that uses LinOSS as a learnable encoder.

`linoss-dynamics` is a **runtime physics package**. Same mathematics, different
audience and abstraction:

| Dimension | Upstream `tk-rusch/linoss` | `linoss-dynamics` |
|---|---|---|
| Use case | Training neural networks with LinOSS layers | Runtime simulation, control, replay-safe systems |
| Stack | JAX + Equinox | NumPy only — zero deep-learning dependencies |
| Granularity | Sequence-level via JAX `scan` (`apply_linoss_im`, `apply_linoss_imex`) | Single-step (`linoss_step`, `damped_linoss_step`) |
| Damping | Implicit (rolled into `A`) | **Explicit, non-negativity validated** (`G` parameter) |
| Energy diagnostics | Not exposed | `energy`, `delta_energy`, `convergence_window` |
| Errors | Generic shape errors | Typed hierarchy (`LinOSSError`, `InvalidShapeError`, `InvalidDampingError`, `UnsupportedModeError`) |
| Determinism contract | Best-effort (training-oriented) | Replay-safe — pure NumPy, deterministic, no hidden state |
| Install footprint | Heavy (JAX, Equinox, dataset deps) | Tiny (NumPy 1.24+) |

### Who needs `linoss-dynamics` specifically

- **Replay-safe agentic systems** — needed inside deterministic kernels where
  every step must be byte-reproducible (e.g. consumed by
  [`elume`](https://github.com/bionicbutterfly13/elume) at runtime).
- **Simulation and control** — robotics, signal processing, vibration analysis,
  and any context that uses oscillatory dynamics as a runtime model rather than
  a learnable layer.
- **Reference implementation for porting** — a clean NumPy baseline to validate
  custom JAX/PyTorch ports of LinOSS against.
- **Embedded or minimal-dependency contexts** — anywhere a JAX install is too
  heavy or unavailable.
- **Teaching and pedagogy** — a self-contained, well-tested, single-file solver
  that students can read end-to-end in an afternoon.
- **Energy-based analysis** — applications that need to monitor energy drift
  and detect convergence; these diagnostics are not exposed by the upstream
  reference.

If you are training a neural network, use the upstream package. If you are
running a system that needs LinOSS as a deterministic physics step, use this one.

## What It Provides

- Classic implicit and implicit-explicit LinOSS step helpers.
- Explicit non-negative damping `G` for D-LinOSS-style runtime damping.
- Energy, energy-delta, and convergence-window helpers.
- A dependency-light package core with NumPy as the only runtime dependency.

## Install

From PyPI (recommended):

```bash
pip install linoss-dynamics
```

From GitHub (latest main):

```bash
python -m pip install "linoss-dynamics @ git+https://github.com/bionicbutterfly13/linoss-dynamics.git@main"
```

For local development:

```bash
git clone https://github.com/bionicbutterfly13/linoss-dynamics.git
cd linoss-dynamics
python -m pip install -e ".[test]"
```

## Quickstart

Classic LinOSS-style step:

```python
import numpy as np
from linoss_dynamics import linoss_step

y = np.array([0.2])
z = np.array([1.0])
A = np.array([1.0])

y_next, z_next, metrics = linoss_step(y, z, A, dt=0.1, mode="implicit")

print(y_next, z_next, metrics["energy_after"])
```

Damped step with explicit `G`:

```python
import numpy as np
from linoss_dynamics import damped_linoss_step

y = np.array([0.2])
z = np.array([1.0])
A = np.array([1.0])
G = np.array([0.5])

y_next, z_next, metrics = damped_linoss_step(y, z, A, G, dt=0.1)

assert metrics["damping_mode"] == "explicit_g"
```

Sequence-level scan over a time series:

```python
import numpy as np
from linoss_dynamics import linoss_scan

U = np.random.default_rng(0).standard_normal((100, 1))  # (T, n)
A = np.array([1.0])

Y, Z, metrics = linoss_scan(U, A, dt=0.05)
print(Y.shape)  # (101, 1) — includes initial state at Y[0]
```

Bayesian filtering over an observed time series (requires `[probabilistic]`):

```python
# pip install linoss-dynamics[probabilistic]
import numpy as np
from linoss_dynamics import kalman_filter, rts_smoother, oscillator_mats

Ad, Qd, Bd = oscillator_mats(beta=0.1, omega=2.0, sigma_proc=0.3, dt=0.05)
H = np.array([[1.0, 0.0]])
R = np.array([[0.1]])

y_obs = np.sin(np.linspace(0, 10, 200)) + 0.1 * np.random.default_rng(42).standard_normal(200)
filt = kalman_filter(y_obs, Ad, Qd, H, R)
smooth = rts_smoother(Ad, filt)

print(filt["m_f"].shape)   # (200, 2)
print(smooth["m_s"].shape) # (200, 2)
```

Parameter fit on observed data (requires `[probabilistic]`):

```python
# pip install linoss-dynamics[probabilistic]
import numpy as np
from linoss_dynamics import fit_oscillator_mle

y = np.sin(np.linspace(0, 20, 400)) + 0.15 * np.random.default_rng(0).standard_normal(400)
result = fit_oscillator_mle(y, dt=0.05)

print(f"fitted omega = {result['omega']:.3f} rad/s")
print(f"fitted period = {result['period']:.3f} s")
```

Runnable tutorials for all use cases are in [examples/](examples/README.md).

For a single bounded prompt/context packet covering the package purpose, symbol
surface, LinOSS relationship, and adjacent technology categories, see
[docs/heredoc-context.md](docs/heredoc-context.md).

## Numerical Contract

`A` controls oscillator stiffness/frequency.

`G` controls damping/forgetting.

The package deliberately keeps those controls separate. Damping is not modeled
as hidden `A` scaling, and negative `G` values are rejected.

Supported damping shapes:

- scalar `G`
- vector `G` with the same length as `y` and `z`
- diagonal matrix `G`

Supported step modes:

- `implicit` / `IM`
- `implicit_explicit` / `IMEX`

## Public API

### Solver

| Name | Role |
| --- | --- |
| `linoss_step(y, z, A, dt, mode="implicit", B=None, u=None, damping=None, G=None)` | Advance one step, optionally dispatching to explicit damping when `damping` or `G` is supplied. |
| `damped_linoss_step(y, z, A, G, dt, mode="implicit", B=None, u=None)` | Advance one step with explicit non-negative damping. |
| `linoss_scan(U, A, dt, mode="implicit", B=None, y0=None, z0=None, damping=None, G=None)` | Run `linoss_step` over a `(T, n)` input sequence; returns `(Y, Z, metrics)`. |
| `energy(y, z, A)` | Return diagonal oscillator energy. |
| `delta_energy(previous_energy, next_energy)` | Return signed energy delta. |
| `convergence_window(deltas, threshold, window)` | Return true when recent absolute deltas are below a threshold. |
| `linoss_step_impl(...)` | Backward-compatible alias for callers using the implementation name. |

### Stability

| Name | Role |
| --- | --- |
| `is_stable(A, G=None, dt=None, mode="implicit")` | Return `(stable, reason)` for given oscillator parameters. |
| `eigvals_to_freq_damping(eigenvalues)` | Map complex eigenvalues to `(frequencies, damping_ratios)`. |
| `freq_damping_to_oscillator_block(omega, gamma, dt=1.0)` | Return a 2×2 real-form discrete oscillator block. |
| `period_from_omega(omega)` | Return `2π / omega`, vectorized. |
| `harmonic_stack(omegas, dampings=None, dt=1.0)` | Build a block-diagonal state matrix from `(omega, damping)` pairs. |

### Continuous (irregular-time)

| Name | Role |
| --- | --- |
| `damped_oscillator_closed_form(y, z, omega, gamma, dt, forcing=0.0)` | Analytic closed-form step using matrix exponential. Supports variable `dt` for irregular sampling. |

### Probabilistic (requires `pip install linoss-dynamics[probabilistic]`)

| Name | Role |
| --- | --- |
| `discretize_lti_with_noise(A_c, L, Qc, dt)` | Exact van Loan discretization — returns `(Ad, Qd)`. |
| `discretize_control(A_c, B, dt)` | ZOH discretization for control input — returns `(Ad, Bd)`. |
| `oscillator_mats(beta, omega, sigma_proc, dt=1.0, kappa=1.0)` | Build `(Ad, Qd, Bd)` for a single forced damped oscillator. |
| `kalman_filter(y, Ad, Qd, H, R, ...)` | Linear-Gaussian Kalman filter; returns filtered moments and log-likelihood. |
| `rts_smoother(Ad, filt)` | Rauch-Tung-Striebel backward smoother over Kalman filter outputs. |
| `fit_oscillator_mle(y, dt, ...)` | MLE parameter recovery for a damped oscillator SSM; returns fitted parameters and filter outputs. |

Public errors:

| Error | Raised when |
| --- | --- |
| `LinOSSError` | Base error for all LinOSS dynamics failures. |
| `InvalidShapeError` | Inputs cannot be broadcast to the oscillator state. |
| `InvalidDampingError` | Damping is outside the supported stable path, including negative `G`. |
| `UnsupportedModeError` | The discretization mode is unsupported. |

## Scope

The package does not implement JAX training loops, Discretax integrations,
active-inference runtimes, metacognitive policy, event buses, graph databases,
or web APIs.

Host applications should keep adapters outside this package and depend on
`linoss-dynamics` through the public API above.

## Development

Run the full local check set:

```bash
python -m ruff check src tests
python -m pytest tests -v --tb=short
python -m compileall -q src tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [PROVENANCE.md](PROVENANCE.md), and
[CLAIMS.md](CLAIMS.md) before changing package behavior or public wording.

## Citation

If this package is useful in research or technical writing, cite the package
metadata in [CITATION.cff](CITATION.cff) and cite the upstream LinOSS and
D-LinOSS papers listed below.

Release and publishing steps are tracked in
[docs/release-checklist.md](docs/release-checklist.md).

## Attribution

This package is not the original LinOSS or D-LinOSS research implementation.

- LinOSS / Oscillatory State-Space Models: T. Konstantin Rusch and Daniela Rus,
  `Oscillatory State-Space Models`, arXiv:2410.03943.
- D-LinOSS / learned damping: Jared Boyer, T. Konstantin Rusch, and Daniela Rus,
  `Learning to Dissipate Energy in Oscillatory State-Space Models`,
  arXiv:2505.12171.
- Official LinOSS ecosystem: <https://github.com/tk-rusch/linoss>.
- Discretax / former Linax ecosystem: <https://github.com/camail-official/discretax>.

See [PROVENANCE.md](PROVENANCE.md) and [CLAIMS.md](CLAIMS.md) before making
public claims.
