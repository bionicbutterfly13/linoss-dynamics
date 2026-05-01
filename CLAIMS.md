# Claims

## Safe Claims

- `linoss-dynamics` provides a small NumPy runtime implementation of LinOSS-style oscillator stepping.
- The package supports explicit non-negative damping `G` for D-LinOSS-style runtime damping.
- The package is dependency-light and intended for agent runtimes, control loops, examples, and tests.
- The package separates stiffness/frequency (`A`) from damping/forgetting (`G`).

## Forbidden Claims

- Do not claim invention of LinOSS.
- Do not claim invention of D-LinOSS.
- Do not claim state-of-the-art sequence-model training performance.
- Do not claim replacement of the official JAX LinOSS repo or Discretax.
- Do not claim complete active-inference, metacognitive-runtime, memory, or basin-runtime support.
- Do not describe this package as production-stable until release work is complete.

## Required Public Wording

Use wording like:

> A NumPy runtime package for LinOSS-style oscillator dynamics with explicit damping support, inspired by LinOSS and D-LinOSS research.

Avoid wording like:

> The LinOSS implementation.

or:

> A new state-of-the-art sequence model.
