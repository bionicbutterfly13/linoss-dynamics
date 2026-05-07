# Roadmap

`linoss-dynamics` is a runtime physics package for oscillatory state-space dynamics. This roadmap documents the trajectory of the package — what's shipped, what's next, and what's explicitly out of scope.

---

## Where we are: v0.2.0 (current)

v0.2.0 is the first PyPI release. The core is NumPy-only; optional probabilistic tools require SciPy.

**Shipped in v0.2.0:**

- **Deterministic oscillator step** — `linoss_step` and `damped_linoss_step` implement the implicit (LinOSS-IM) and implicit-explicit (LinOSS-IMEX) discretizations of the forced harmonic oscillator with explicit non-negative damping `G`.
- **Sequence-level scan** — `linoss_scan` runs the step function over a `(T, n)` input sequence and returns the full trajectory.
- **Closed-form irregular-time stepping** — `damped_oscillator_closed_form` solves the damped second-order ODE analytically using the matrix exponential. Because each call can use a different `dt`, this enables exact stepping on irregular time grids without accumulated discretization bias.
- **Stability and frequency tools** — `is_stable`, `eigvals_to_freq_damping`, `freq_damping_to_oscillator_block`, `period_from_omega`, `harmonic_stack`.
- **Kalman filter and RTS smoother** — `kalman_filter` and `rts_smoother` provide Bayesian state estimation over discrete-time linear-Gaussian oscillator SSMs.
- **Exact discretization** — `discretize_lti_with_noise` (van Loan's method) and `discretize_control` (ZOH) convert continuous-time oscillator parameters to discrete-time matrices. `oscillator_mats` is a convenience wrapper for single forced damped oscillators.
- **MLE parameter fit** — `fit_oscillator_mle` recovers `(beta, omega, sigma_proc, sigma_obs)` from an observed time series by maximising the Kalman filter log-likelihood.
- **8 runnable tutorials** covering single-step, forced oscillators, damping, energy, convergence, validation errors, sunspot modeling, and phase tracking.

---

## v0.3.0 — Candidate next-version work

Items below are research-grounded extensions identified in the oscillatory SSM literature. Each is a candidate for v0.3.0 inclusion, with scope notes.

### PLSO (piecewise locally stationary oscillatory) regimes

PLSO (Wodeyar et al. 2021) models time series as piecewise-stationary oscillatory mixtures, using Kalman-based inference with proximal-gradient regularization over piecewise windows. The runtime-physics analog is a `switch_oscillator` primitive that can change frequency and damping parameters at detected change points while maintaining exact matrix-exponential transitions at each segment boundary. This builds naturally on `damped_oscillator_closed_form` and `oscillator_mats`. Effort: medium. No new external dependencies if the change-point detection is left to the caller.

### Switching state-space models

Switching SSMs extend the PLSO idea to a latent discrete mode variable, where the oscillator transitions between a small set of modes (e.g., spindle vs. non-spindle in neuroscience, high-frequency vs. low-frequency in economics). The approximate inference stack (Kim's algorithm or variational) is non-trivial but well-studied. A v0.3.0 candidate would provide the oscillator-mode building blocks and leave the switching inference to a thin API the caller can plug into. This is a natural extension of `kalman_filter` and `rts_smoother`. Effort: high.

### DMD / Koopman-DMD spectral identification

Dynamic Mode Decomposition (Schmid 2010) identifies oscillatory spatial modes and their temporal evolution from snapshot data. Kalman-filter DMD and Koopman-autoencoder approaches extend this to data assimilation and online tracking. A v0.3.0 candidate would provide a pure-NumPy DMD routine that returns eigenvalues compatible with `eigvals_to_freq_damping` and blocks compatible with `harmonic_stack`, closing the loop between data-driven identification and physics-stepping. Effort: medium.

### Koopman-Kalman hybrids for online tracking

Recent 2024–2025 work combines Koopman latent linearizations with Kalman or ensemble-Kalman updates to track slowly changing oscillatory systems online. The linoss-dynamics angle would be a `tracking_filter` that accepts a time-varying `Ad` sequence (updated by a Koopman estimator) and runs a standard Kalman recursion — the oscillator stepping provides the physics prior, the Kalman filter handles the estimation. Effort: low (once DMD identification is in place).

---

## v1.0 — Path B: unified oscillators + attractor basins

The largest open question in this research lineage is whether oscillator dynamics (LinOSS-family) and attractor memory (Hopfield-family) should converge into one substrate or remain composed primitives.

The oscillatory SSM literature identifies this as the synthesis direction: "uncertainty-calibrated, continuous-time or irregular-time, second-order oscillatory SSMs that remain competitive at the scale of modern sequence benchmarks" (deep-research dossier, "Recent research trends"). The runtime-physics version of that synthesis is a unified oscillator + attractor library.

`linoss-dynamics` v1.0 is committed to providing the unified primitives. Until then, downstream consumers compose `linoss-dynamics` with their own attractor layer (Hopfield/CAN-style). The API boundary is designed so that attractor state can be stored alongside oscillator state without structural changes — both are plain NumPy arrays passed explicitly.

---

## v1.x — Spatio-temporal extensions (BioOSS-class)

Spatio-temporal oscillator models (BioOSS, 2025) extend the state to a 2D field with lateral coupling between oscillator sites, generating wave-like propagation across a structured substrate. This breaks the 1D state contract of the current API — `y` and `z` are flat vectors, not field arrays — and is therefore an explicit non-goal for v0.2–1.0.

v1.x will introduce a separate `spatial.*` namespace with field-shaped state and a 2D stepping API. The core scalar stepping functions will remain unchanged.

---

## Explicit non-goals

The following are intentionally outside the scope of this package — use the cited alternatives:

| Goal | Alternative |
| --- | --- |
| Training-time JAX neural networks with LinOSS layers | Use the upstream `tk-rusch/linoss` JAX/Equinox reference. |
| Full probabilistic GP inference with arbitrary kernels | Use GPflow, GPyTorch, or Stheno. |
| Web or service runtimes | Host applications integrate via normal Python imports. |
| Agent frameworks | `linoss-dynamics` provides primitives; agentic composition belongs in the host application. |
| Symbolic or autodiff-based parameter learning | Use JAX or PyTorch; this package is NumPy-only. |

---

## Citations

This roadmap draws from the LinOSS and D-LinOSS papers (Rusch & Rus 2024; Boyer, Rusch & Rus 2025), the classical oscillator-SSM literature (Hartikainen & Särkkä 2010; Solin & Särkkä 2014; Beck et al. 2018; Wodeyar et al. 2021), and the 2025–2026 research line on continuous-time and uncertainty-calibrated oscillatory SSMs. See `PROVENANCE.md` for full attribution and `CITATION.cff` for machine-readable citation metadata.
