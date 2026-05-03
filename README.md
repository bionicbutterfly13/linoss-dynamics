# linoss-dynamics

[![CI](https://github.com/bionicbutterfly13/linoss-dynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/bionicbutterfly13/linoss-dynamics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-public%20alpha-orange)

`linoss-dynamics` is a small NumPy runtime package for LinOSS-style oscillator
dynamics with explicit non-negative damping support.

Status: **public alpha**. The package is not published to PyPI yet.

## What It Provides

- Classic implicit and implicit-explicit LinOSS step helpers.
- Explicit non-negative damping `G` for D-LinOSS-style runtime damping.
- Energy, energy-delta, and convergence-window helpers.
- A dependency-light package core with NumPy as the only runtime dependency.

## Install

Install from GitHub:

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

| Name | Role |
| --- | --- |
| `linoss_step(y, z, A, dt, mode="implicit", B=None, u=None, damping=None, G=None)` | Advance one step, optionally dispatching to explicit damping when `damping` or `G` is supplied. |
| `damped_linoss_step(y, z, A, G, dt, mode="implicit", B=None, u=None)` | Advance one step with explicit non-negative damping. |
| `energy(y, z, A)` | Return diagonal oscillator energy. |
| `delta_energy(previous_energy, next_energy)` | Return signed energy delta. |
| `convergence_window(deltas, threshold, window)` | Return true when recent absolute deltas are below a threshold. |
| `linoss_step_impl(...)` | Backward-compatible alias for callers using the implementation name. |

Public errors:

| Error | Raised when |
| --- | --- |
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
