# Provenance

Status: active provenance record for public alpha.

## Upstream Foundations

| Source | Role | Link |
| --- | --- | --- |
| T. Konstantin Rusch and Daniela Rus, `Oscillatory State-Space Models` | LinOSS mathematical foundation and discretization context | <https://arxiv.org/abs/2410.03943> |
| `tk-rusch/linoss` | Official LinOSS research implementation ecosystem | <https://github.com/tk-rusch/linoss> |
| Jared Boyer, T. Konstantin Rusch, and Daniela Rus, `Learning to Dissipate Energy in Oscillatory State-Space Models` | D-LinOSS damping motivation and attribution source | <https://arxiv.org/abs/2505.12171> |
| D-LinOSS OpenReview page | Current public framing for learned dissipation | <https://openreview.net/forum?id=dw2vxWVrA9> |
| Discretax | JAX SSM ecosystem reference, formerly Linax | <https://github.com/camail-official/discretax> |

## Package Delta

`linoss-dynamics` packages a small NumPy runtime seam for agent and control-loop use:

- classic implicit and implicit-explicit one-step helpers
- explicit non-negative damping `G`
- energy and convergence helper functions
- dependency-light tests and documentation

## License Posture

The package is planned as MIT because it is a pure math/runtime primitive with no novelty claim over LinOSS or D-LinOSS.

## Code Provenance (Framing-only)

`linoss-dynamics` is an original NumPy implementation of the LinOSS algorithm described in Rusch & Rus (2024, arXiv:2410.03943) and the D-LinOSS damping extension (Boyer, Rusch & Rus, 2025, arXiv:2505.12171). No code is shared with the upstream JAX/Equinox reference implementation at `tk-rusch/linoss`.

**Upstream vs. this package:**

| Aspect | `tk-rusch/linoss` (upstream) | `linoss-dynamics` (this package) |
| --- | --- | --- |
| Runtime | JAX + Equinox | NumPy only |
| API style | `LinOSSLayer`, `LinOSSBlock`, `LinOSS` (Equinox modules) | Functions and error classes |
| Entry points | `apply_linoss_im`, `apply_linoss_imex` (JAX scan ops) | `linoss_step`, `damped_linoss_step`, `linoss_scan` |
| Class name collision | None — namespaces do not overlap | — |

**License obligation:** None. The LinOSS algorithm is described in the public papers above; this package is an independently written implementation. The upstream repository uses the MIT license; this package is also MIT-licensed, but there is no dependency or code-sharing relationship that creates an attribution obligation beyond the paper citations already present in `CITATION.cff`.

**Continuous.py provenance:** `damped_oscillator_closed_form` implements the analytic matrix-exponential solution for the forced damped harmonic oscillator. This technique is standard in the classical oscillator state-space literature (Hartikainen & Särkkä 2010; Solin & Särkkä 2014) and was motivated by 2026 work on closed-form damped oscillators for irregular time series (see the deep-research dossier section "Recent research trends" for the underlying line of work; no arXiv id is cited here to avoid fabricating an identifier for a paper whose exact metadata has not been independently verified).

## Evidence

- Package core: `src/linoss_dynamics/`
- Package tests: `tests/`
- Initial development source: `bionicbutterfly13/dionysus3-core`, before the
  standalone package split and public repository publication.
