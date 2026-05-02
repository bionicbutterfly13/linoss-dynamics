# Contributing

`linoss-dynamics` is currently a public alpha package.

## Setup

```bash
git clone https://github.com/bionicbutterfly13/linoss-dynamics.git
cd linoss-dynamics
python -m pip install -e ".[test]"
```

## Local Checks

Run these before opening a PR:

```bash
python -m pytest tests -v --tb=short
python -m ruff check src tests
python -m compileall -q src tests
```

## Pull Requests

- Keep PRs small and focused.
- Include tests for behavior changes.
- Update [README.md](README.md), [PROVENANCE.md](PROVENANCE.md), or
  [CLAIMS.md](CLAIMS.md) when public-facing behavior or wording changes.
- Mention the verification commands you ran.

## Boundaries

- Keep the runtime dependency surface NumPy-only by default.
- Preserve the `A` versus `G` distinction: `A` controls stiffness/frequency;
  `G` controls damping/forgetting.
- Do not claim invention of LinOSS or D-LinOSS.
- Update [PROVENANCE.md](PROVENANCE.md) and [CLAIMS.md](CLAIMS.md) when public
  wording changes.
- Do not add host-application dependencies such as web frameworks, graph
  databases, event buses, metacognitive runtimes, or global service getters.

## Compatibility

The public API is still alpha. Breaking changes are possible before a stable
release, but changes should preserve the core numerical contract unless a PR
explicitly documents and tests the reason for changing it.
