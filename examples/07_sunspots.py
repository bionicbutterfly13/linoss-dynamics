"""07_sunspots.py — Bayesian oscillator fit on the sunspot time series.

Fits a single damped oscillator SSM to the Wolfer sunspot number series
(annual data, ~11-year cycle) using maximum-likelihood via the Kalman filter.

Requires the [probabilistic] extra:
    pip install linoss-dynamics[probabilistic]

If statsmodels is unavailable a synthetic dataset with the same structure
(period ~11 years, 200 observations) is used as a fallback.
"""

from __future__ import annotations

try:
    from scipy import optimize  # noqa: F401
except ImportError:
    print(
        "This example requires scipy.  "
        "Install with: pip install linoss-dynamics[probabilistic]"
    )
    raise SystemExit(0)

import math

import numpy as np

from linoss_dynamics.fit import fit_oscillator_mle


def load_sunspots() -> np.ndarray:
    """Return the Wolfer annual sunspot series (normalised), or a synthetic fallback."""
    try:
        from statsmodels.datasets import sunspots

        data = sunspots.load_pandas().data
        y_raw = data["SUNACTIVITY"].to_numpy(dtype=float)
        print(f"Loaded statsmodels sunspot dataset: {len(y_raw)} annual observations")
        return y_raw
    except Exception:  # noqa: BLE001
        print("statsmodels unavailable — using synthetic sunspot-like data (fallback)")
        rng = np.random.default_rng(0)
        t = np.arange(200, dtype=float)
        # ~11-year cycle + noise, mean ~50, std ~30 (mimics Wolfer data statistics)
        y = 50.0 + 30.0 * np.sin(2.0 * math.pi / 11.0 * t) + 10.0 * rng.standard_normal(200)
        y = np.clip(y, 0.0, None)  # sunspot counts are non-negative
        return y


def main() -> None:
    print("=== 07 Sunspots — Oscillator MLE Fit ===")
    print()

    y_raw = load_sunspots()

    # Use the first 200 observations for speed; the rest can be used for evaluation.
    y = y_raw[:200]
    dt = 1.0  # annual data, dt = 1 year

    # Initial parameter guess informed by domain knowledge:
    #   beta   ~ 0.05  (mild decay over decades)
    #   omega  ~ 2π/11 (approx 11-year cycle)
    #   sigma_proc ~ 0.5, sigma_obs ~ 0.5
    omega_guess = 2.0 * math.pi / 11.0
    init = (0.05, omega_guess, 0.5, 0.5)

    print(f"Fitting {len(y)} annual observations, dt={dt} year")
    print(f"Initial guess: beta={init[0]}, omega={init[1]:.4f} "
          f"(period={2*math.pi/init[1]:.1f} yr), "
          f"sigma_proc={init[2]}, sigma_obs={init[3]}")
    print()

    result = fit_oscillator_mle(y, dt=dt, init=init)

    print("=== Fitted parameters ===")
    print(f"  beta       = {result['beta']:.4f}  (damping coefficient)")
    print(f"  omega      = {result['omega']:.4f} rad/yr")
    print(f"  period     = {result['period']:.2f} years  (2π/omega)")
    print(f"  sigma_proc = {result['sigma_proc']:.4f}")
    print(f"  sigma_obs  = {result['sigma_obs']:.4f}")
    print(f"  mean_y     = {result['mean_y']:.2f}  (subtracted before fitting)")
    print()

    # The Kalman filter and RTS smoother outputs are in result["filt"] / result["smooth"].
    # m_f has shape (T, n) where n=2 (position, velocity state).
    filt = result["filt"]
    smooth = result["smooth"]

    print("First 10 filtered state means (position | velocity):")
    print(f"{'t':>4}  {'pos_filt':>12}  {'vel_filt':>12}  {'pos_smooth':>12}")
    for t in range(10):
        pos_f = filt["m_f"][t, 0] + result["mean_y"]   # un-centre
        vel_f = filt["m_f"][t, 1]
        pos_s = smooth["m_s"][t, 0] + result["mean_y"]
        print(f"{t:4d}  {pos_f:12.3f}  {vel_f:12.4f}  {pos_s:12.3f}")

    print()
    print(f"Log-likelihood of fitted model: {filt['loglik']:.2f}")


if __name__ == "__main__":
    main()
