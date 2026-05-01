# Architecture

`linoss-dynamics` is a pure runtime primitive package.

## Layers

| Layer | Files | Responsibility |
| --- | --- | --- |
| Public API | `src/linoss_dynamics/__init__.py` | Re-export stable functions and errors. |
| Solver core | `src/linoss_dynamics/solver.py` | NumPy stepping, damping, energy, validation, and convergence helpers. |
| Package tests | `tests/test_solver.py` | Verify package behavior without host-application imports. |

## Boundary

Inputs:

- `y`, `z`: oscillator position and velocity arrays
- `A`: stiffness/frequency parameter
- `G` or `damping`: explicit non-negative damping
- `dt`: timestep
- optional `B`, `u`: forcing matrix/vector and input

Outputs:

- `y_next`, `z_next`
- metrics dict with mode, energy, signed energy delta, and damping mode

## Non-Dependencies

The package core must not depend on:

- host service getters
- FastAPI
- Graphiti
- Neo4j
- EventBus
- metacognitive-runtime or agent-framework objects
- JAX or Discretax

## Host Relationship

Host applications should depend on this package through normal Python package
installation and keep host-specific adapters outside the package core.
