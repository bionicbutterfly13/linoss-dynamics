# linoss-dynamics Examples

Runnable tutorials for `linoss-dynamics`.  Each script is self-contained and
prints labeled output so you can follow along without reading source code.

Run any example from the repository root:

```bash
python examples/<script>.py
```

---

## Tutorial Index

| # | Script | Description | Extras needed |
|---|--------|-------------|---------------|
| 01 | `01_single_step.py` | Advance one LinOSS step in both `implicit` and `implicit_explicit` modes; verify the backward-compatible `linoss_step_impl` alias | — |
| 02 | `02_forced.py` | Build a sinusoidal forcing sequence and scan the full trajectory with `linoss_scan`; compute energy at every step | — |
| 03 | `03_damped.py` | Compare undamped vs damped trajectories; validate the discrete solver against the exact `damped_oscillator_closed_form` | — |
| 04 | `04_energy.py` | Track energy with `energy()` and `delta_energy()` over 100 steps; verify monotonic decay under damping | — |
| 05 | `05_convergence.py` | Run a heavily damped oscillator until `convergence_window()` reports the absolute energy change has settled below a threshold | — |
| 06 | `06_validation_errors.py` | Intentionally trigger `InvalidShapeError`, `InvalidDampingError`, and `UnsupportedModeError`; catch each with typed `except` clauses | — |
| 07 | `07_sunspots.py` | Fit a damped oscillator SSM to the Wolfer sunspot series via `fit_oscillator_mle`; print fitted period and first 10 smoothed values | `[probabilistic]` + statsmodels (fallback to synthetic data if statsmodels absent) |
| 08 | `08_phase_tracking.py` | Simulate a noisy oscillator, run `kalman_filter`, and extract instantaneous phase and amplitude; compare against ground truth | `[probabilistic]` |

---

## Installing extras

The core examples (01–06) run with the base install:

```bash
pip install linoss-dynamics
```

Examples 07 and 08 need SciPy (and optionally statsmodels for real sunspot data):

```bash
pip install "linoss-dynamics[probabilistic]"
pip install statsmodels   # optional, for example 07
```

---

## Running all examples

```bash
for f in examples/0*.py; do
    echo "=== $f ===" && python "$f" || echo "FAILED: $f"
done
```
