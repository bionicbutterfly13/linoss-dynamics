# linoss-dynamics

`linoss-dynamics` is a small NumPy runtime package for LinOSS-style oscillator dynamics.

Status: private pre-release package. Not published to PyPI.

## What It Provides

- Classic implicit and implicit-explicit LinOSS step helpers.
- Explicit non-negative damping `G` for D-LinOSS-style runtime damping.
- Energy, energy-delta, and convergence-window helpers.
- A dependency-light package core with NumPy as the only runtime dependency.

## Boundary

`A` controls oscillator stiffness/frequency.

`G` controls damping/forgetting.

The package does not implement JAX training loops, Discretax integrations, active-inference runtimes, metacognitive policy, event buses, graph databases, or web APIs.

## Usage

```python
import numpy as np
from linoss_dynamics import damped_linoss_step

y = np.array([0.2])
z = np.array([1.0])
A = np.array([1.0])
G = np.array([0.5])

y_next, z_next, metrics = damped_linoss_step(y, z, A, G, dt=0.1)
```

## Attribution

This package is not the original LinOSS or D-LinOSS research implementation.

- LinOSS / Oscillatory State-Space Models: T. Konstantin Rusch and Daniela Rus, `Oscillatory State-Space Models`, arXiv:2410.03943.
- D-LinOSS / learned damping: Jared Boyer, T. Konstantin Rusch, and Daniela Rus, `Learning to Dissipate Energy in Oscillatory State-Space Models`, arXiv:2505.12171.
- Official LinOSS ecosystem: <https://github.com/tk-rusch/linoss>.
- Discretax / former Linax ecosystem: <https://github.com/camail-official/discretax>.

See [PROVENANCE.md](PROVENANCE.md) and [CLAIMS.md](CLAIMS.md) before making public claims.

## Local Checks

From this repository root:

```bash
python -m pytest tests -v --tb=short
python -m ruff check src tests
```
