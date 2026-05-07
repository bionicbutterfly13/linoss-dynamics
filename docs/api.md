# API Reference

Per-function reference for all public symbols in `linoss-dynamics`. Grouped by submodule.

---

## Errors

### `LinOSSError`

```python
class LinOSSError(ValueError)
```

Base error for all LinOSS dynamics failures. Catch this to handle any library-level error.

---

### `InvalidShapeError`

```python
class InvalidShapeError(LinOSSError)
```

Raised when inputs (`y`, `z`, `A`, `G`, `B`, `u`) cannot be broadcast to the oscillator state shape.

---

### `InvalidDampingError`

```python
class InvalidDampingError(LinOSSError)
```

Raised when damping `G` or `gamma` contains negative values.

---

### `UnsupportedModeError`

```python
class UnsupportedModeError(LinOSSError)
```

Raised when `mode` is not a recognized discretization scheme. Valid values: `"implicit"` / `"im"`, `"implicit_explicit"` / `"imex"`.

---

## Solver (`linoss_dynamics.solver`)

### `linoss_step`

```python
def linoss_step(
    y: Any,
    z: Any,
    A: Any,
    dt: float,
    mode: str = "implicit",
    B: Any | None = None,
    u: Any | None = None,
    damping: Any | None = None,
    G: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]
```

Advance one LinOSS step. When `damping` or `G` is provided, dispatches to `damped_linoss_step`.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `y` | array-like | Displacement vector, length `n`. |
| `z` | array-like | Velocity vector, length `n`. |
| `A` | array-like | Stiffness/frequency — scalar, length-`n` vector, or `n×n` diagonal matrix. |
| `dt` | float | Time step. |
| `mode` | str | `"implicit"` (default) or `"implicit_explicit"`. |
| `B` | array-like or None | Forcing matrix, shape `(n, m)`. Optional. |
| `u` | array-like or None | Forcing input, shape `(m,)`. Optional. |
| `damping` / `G` | array-like or None | Non-negative damping. When either is set, uses `damped_linoss_step`. |

**Returns** `(y_next, z_next, metrics)` — next state and a dict with `mode`, `energy_before`, `energy_after`, `delta_energy`, `damping_mode`.

**Raises** `InvalidShapeError`, `InvalidDampingError`, `UnsupportedModeError`.

**Example**

```python
import numpy as np
from linoss_dynamics import linoss_step

y_next, z_next, m = linoss_step(np.array([0.2]), np.array([1.0]), np.array([1.0]), dt=0.1)
```

---

### `damped_linoss_step`

```python
def damped_linoss_step(
    y: Any,
    z: Any,
    A: Any,
    G: Any,
    dt: float,
    mode: str = "implicit",
    B: Any | None = None,
    u: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]
```

Advance one LinOSS step with explicit non-negative damping `G`. Negative values in `G` raise `InvalidDampingError`.

**Parameters** Same as `linoss_step` plus `G` (required, non-negative scalar or vector).

**Returns** `(y_next, z_next, metrics)`. Metrics include `damping_mode: "explicit_g"` and the resolved `damping` vector.

**Raises** `InvalidShapeError`, `InvalidDampingError`, `UnsupportedModeError`.

---

### `linoss_scan`

```python
def linoss_scan(
    U: Any,
    A: Any,
    dt: float,
    *,
    mode: str = "implicit",
    B: Any | None = None,
    y0: Any | None = None,
    z0: Any | None = None,
    damping: Any | None = None,
    G: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]
```

Run `linoss_step` over a time-varying input sequence, returning the full trajectory.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `U` | array-like `(T,)` or `(T, m)` | Input sequence. Each row `U[t]` is the forcing at step `t`. |
| `A` | array-like | Stiffness parameter. |
| `dt` | float | Time step. |
| `y0` / `z0` | array-like or None | Initial state. Defaults to zeros; `n` is inferred from `A`. |
| `damping` / `G` | array-like or None | Non-negative damping for all steps. |

**Returns** `(Y, Z, metrics_seq)` where `Y` and `Z` have shape `(T+1, n)` including the initial state, and `metrics_seq` is a list of length `T`.

**Raises** `InvalidShapeError`, `InvalidDampingError`, `UnsupportedModeError`.

**Example**

```python
import numpy as np
from linoss_dynamics import linoss_scan

U = np.ones((50, 1))
Y, Z, metrics = linoss_scan(U, A=np.array([1.0]), dt=0.05)
assert Y.shape == (51, 1)
```

---

### `linoss_step_impl`

```python
def linoss_step_impl(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]
```

Backward-compatible alias for `linoss_step`. Accepts identical arguments and returns identical values.

---

### `energy`

```python
def energy(y: Any, z: Any, A: Any, G: Any | None = None) -> float
```

Return oscillator energy `0.5 * sum(A * y^2 + z^2)` for diagonal parameters. `G` is accepted but ignored (energy is defined by stiffness, not damping).

**Parameters** `y`, `z` — state vectors; `A` — diagonal stiffness.

**Returns** Scalar float.

**Raises** `InvalidShapeError` if `y` and `z` have different shapes.

---

### `delta_energy`

```python
def delta_energy(previous_energy: float, next_energy: float) -> float
```

Return signed energy change `next_energy - previous_energy`.

---

### `convergence_window`

```python
def convergence_window(
    deltas: list[float] | tuple[float, ...],
    threshold: float,
    window: int,
) -> bool
```

Return `True` when the `window` most recent absolute energy deltas are all below `threshold`. Useful for detecting steady-state convergence.

**Parameters**

| Name | Description |
| --- | --- |
| `deltas` | Sequence of signed energy deltas accumulated over time. |
| `threshold` | Convergence tolerance (positive). |
| `window` | Number of most-recent deltas to check (must be positive). |

**Returns** `bool`.

**Raises** `ValueError` if `window <= 0`.

---

## Stability (`linoss_dynamics.stability`)

### `is_stable`

```python
def is_stable(
    A: Any,
    G: Any | None = None,
    dt: float | None = None,
    mode: str = "implicit",
) -> tuple[bool, str]
```

Return `(stable, reason)` for the given oscillator parameters.

For `"implicit"` mode: stable when `A >= 0` and `G >= 0` (if provided).

For `"implicit_explicit"` mode: stable when `A >= 0`, `G >= 0`, and `dt^2 * max(A) < 4` (CFL condition). Requires `dt`.

**Returns** `(bool, str)` — stability flag and human-readable explanation.

**Raises** `UnsupportedModeError`.

**Example**

```python
from linoss_dynamics import is_stable
stable, reason = is_stable(A=1.0, G=0.5, dt=0.1, mode="implicit_explicit")
assert stable
```

---

### `eigvals_to_freq_damping`

```python
def eigvals_to_freq_damping(eigenvalues: Any) -> tuple[np.ndarray, np.ndarray]
```

Map complex eigenvalues `r·e^(±iθ)` to `(frequencies, damping_ratios)`.

Frequency = `|θ|` (radians per step). Damping ratio = `-ln(r)` per step.

**Returns** `(frequencies, damping_ratios)` — 1-D float arrays of length `N`.

---

### `freq_damping_to_oscillator_block`

```python
def freq_damping_to_oscillator_block(
    omega: float,
    gamma: float,
    dt: float = 1.0,
) -> np.ndarray
```

Return a 2×2 real-form discrete state-transition block for a single harmonic oscillator with angular frequency `omega` and damping `gamma`.

**Returns** Shape `(2, 2)` NumPy float array.

**Raises** `InvalidShapeError` if `omega < 0`.

---

### `period_from_omega`

```python
def period_from_omega(omega: Any) -> np.ndarray
```

Return `2π / omega`, vectorized over array input.

**Raises** `InvalidShapeError` if any element of `omega` is non-positive.

---

### `harmonic_stack`

```python
def harmonic_stack(
    omegas: Any,
    dampings: Any | None = None,
    dt: float = 1.0,
) -> np.ndarray
```

Build a `(2N, 2N)` block-diagonal state matrix from `N` `(omega, damping)` pairs. Each pair yields a 2×2 oscillator block via `freq_damping_to_oscillator_block`.

**Parameters** `omegas` — length-`N` frequencies; `dampings` — length-`N` damping coefficients (default: all zeros).

**Returns** Shape `(2N, 2N)` float array.

**Raises** `InvalidShapeError` if `omegas` and `dampings` have different lengths.

---

## Continuous (`linoss_dynamics.continuous`)

### `damped_oscillator_closed_form`

```python
def damped_oscillator_closed_form(
    y: Any,
    z: Any,
    omega: Any,
    gamma: Any,
    dt: float,
    *,
    forcing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]
```

Advance a damped harmonic oscillator by `dt` using the analytic closed-form solution of `ÿ + 2γẏ + ω²y = u`.

Supports **variable `dt` between calls** — each call may use a different time interval, enabling exact stepping on irregular time grids. Handles all damping regimes:

- **Underdamped** (`ω > γ`): oscillatory decay via `cos`/`sin`.
- **Critically damped** (`ω ≈ γ`): polynomial-exponential decay.
- **Overdamped** (`γ > ω`): two-exponential decay via `cosh`/`sinh`.

**Parameters**

| Name | Description |
| --- | --- |
| `y` | Position(s), scalar or 1-D array. |
| `z` | Velocity(ies), same broadcastable shape as `y`. |
| `omega` | Angular frequency(ies). Must be strictly positive. |
| `gamma` | Damping coefficient(s). Must be non-negative. |
| `dt` | Time interval to advance. Must be strictly positive. |
| `forcing` | Constant scalar forcing over the interval. Default `0.0`. |

**Returns** `(y_next, z_next)` at time `t + dt`, both 1-D float64 arrays.

**Raises** `ValueError` if `dt <= 0` or any `omega <= 0`. `InvalidDampingError` if any `gamma < 0`. `InvalidShapeError` if shapes are incompatible.

**Example**

```python
import numpy as np
from linoss_dynamics import damped_oscillator_closed_form

y, z = np.array([1.0]), np.array([0.0])
y1, z1 = damped_oscillator_closed_form(y, z, omega=1.0, gamma=0.0, dt=np.pi)
# half-period of pure cosine: y1 ≈ -1.0
```

---

## Probabilistic (`linoss_dynamics.discretize`, `linoss_dynamics.filters`, `linoss_dynamics.fit`)

These symbols require `pip install linoss-dynamics[probabilistic]` (SciPy). Importing without SciPy will succeed, but calling any function raises `LinOSSError` with an install hint.

---

### `discretize_lti_with_noise`

```python
def discretize_lti_with_noise(
    A_c: np.ndarray,
    L: np.ndarray,
    Qc: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]
```

Exact discretization of `dx = A_c·x·dt + L·dW` using van Loan's matrix-exponential method.

**Parameters** `A_c` — `(n, n)` continuous state matrix; `L` — `(n, q)` noise input; `Qc` — `(q, q)` spectral density; `dt` — time step.

**Returns** `(Ad, Qd)` — discrete state-transition and process-noise covariance, both `(n, n)`.

**Raises** `LinOSSError` if SciPy is missing; `InvalidShapeError` for shape violations; `ValueError` if `dt <= 0`.

---

### `discretize_control`

```python
def discretize_control(
    A_c: np.ndarray,
    B: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]
```

Zero-order-hold (ZOH) discretization of `dx/dt = A_c·x + B·u`.

**Returns** `(Ad, Bd)` — discrete state-transition `(n, n)` and input `(n, m)`.

**Raises** `LinOSSError` if SciPy is missing; `InvalidShapeError`; `ValueError` if `dt <= 0`.

---

### `oscillator_mats`

```python
def oscillator_mats(
    beta: float,
    omega: float,
    sigma_proc: float,
    dt: float = 1.0,
    kappa: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Build `(Ad, Qd, Bd)` for a single forced damped oscillator with continuous-time dynamics `ẏ = z`, `ż = -ω²y - 2βz + κu + σ_proc·w`.

**Parameters** `beta` — damping coefficient; `omega` — natural frequency; `sigma_proc` — process noise std dev; `dt` — time step; `kappa` — control gain.

**Returns** `(Ad, Qd, Bd)` where `Ad` and `Qd` are `(2, 2)` and `Bd` is `(2, 1)`.

**Raises** `LinOSSError` if SciPy is missing; `ValueError` if `dt <= 0`.

**Example**

```python
import numpy as np
from linoss_dynamics import oscillator_mats
Ad, Qd, Bd = oscillator_mats(beta=0.1, omega=2.0, sigma_proc=0.3, dt=0.05)
assert Ad.shape == (2, 2)
```

---

### `kalman_filter`

```python
def kalman_filter(
    y: np.ndarray,
    Ad: np.ndarray,
    Qd: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    *,
    m0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    Bd: np.ndarray | None = None,
    u: np.ndarray | None = None,
) -> dict[str, Any]
```

Run a linear-Gaussian Kalman filter over a 1-D observation sequence `y` of length `T`.

**Parameters**

| Name | Shape | Description |
| --- | --- | --- |
| `y` | `(T,)` | Observation sequence. |
| `Ad` | `(n, n)` | Discrete state-transition matrix. |
| `Qd` | `(n, n)` | Process-noise covariance. |
| `H` | `(1, n)` | Observation matrix. |
| `R` | `(1, 1)` | Observation-noise covariance. |
| `m0` | `(n,)` | Initial state mean. Default: zeros. |
| `P0` | `(n, n)` | Initial state covariance. Default: identity. |
| `Bd` | `(n, p)` | Control input matrix. Optional. |
| `u` | `(T,)` or `(T, p)` | Control sequence. Required when `Bd` is provided. |

**Returns** Dict with keys `"m_f"` `(T, n)`, `"P_f"` `(T, n, n)`, `"m_p"` `(T, n)`, `"P_p"` `(T, n, n)`, `"loglik"` float.

**Raises** `InvalidShapeError` for shape violations.

---

### `rts_smoother`

```python
def rts_smoother(
    Ad: np.ndarray,
    filt: dict[str, Any],
) -> dict[str, Any]
```

Run a Rauch-Tung-Striebel smoother backward through Kalman filter outputs.

**Parameters** `Ad` — state-transition matrix `(n, n)`; `filt` — dict returned by `kalman_filter`.

**Returns** Dict with keys `"m_s"` `(T, n)` and `"P_s"` `(T, n, n)`.

**Raises** `InvalidShapeError` if `Ad` is not square or does not match state dimension in `filt`.

**Note** At `t = T-1` the smoothed distribution equals the filtered distribution (RTS endpoint property).

---

### `fit_oscillator_mle`

```python
def fit_oscillator_mle(
    y: np.ndarray,
    dt: float,
    *,
    u: np.ndarray | None = None,
    init: tuple[float, float, float, float] | None = None,
    method: str = "L-BFGS-B",
) -> dict[str, Any]
```

Fit `(beta, omega, sigma_proc, sigma_obs)` of a single damped oscillator SSM by maximising the Kalman filter log-likelihood. Optimization is performed in log-parameter space so all parameters remain strictly positive.

**Parameters**

| Name | Description |
| --- | --- |
| `y` | Observation sequence `(T,)`. |
| `dt` | Time step (positive). |
| `u` | Optional scalar control sequence `(T,)`. Default: zeros. |
| `init` | Initial guess `(beta, omega, sigma_proc, sigma_obs)`. Default: `(0.10, 2.0, 0.40, 0.30)`. |
| `method` | SciPy `minimize` method. Default: `"L-BFGS-B"`. |

**Returns** Dict with fitted parameters (`beta`, `omega`, `sigma_proc`, `sigma_obs`, `period`, `mean_y`), matrices (`Ad`, `Qd`, `Bd`, `H`, `R`), and filter outputs (`filt`, `smooth`, `optim`).

**Raises** `LinOSSError` if SciPy is missing; `ValueError` if `dt <= 0`.

**Example**

```python
import numpy as np
from linoss_dynamics import fit_oscillator_mle

y = np.sin(np.linspace(0, 20, 400)) + 0.15 * np.random.default_rng(0).standard_normal(400)
result = fit_oscillator_mle(y, dt=0.05)
assert 0 < result["omega"] < 20
```
