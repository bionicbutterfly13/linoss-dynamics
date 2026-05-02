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

## Evidence

- Package core: `src/linoss_dynamics/`
- Package tests: `tests/`
- Initial development source: `bionicbutterfly13/dionysus3-core`, before the
  standalone package split and public repository publication.
