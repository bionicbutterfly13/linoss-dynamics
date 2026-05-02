# Contributing

`linoss-dynamics` is currently a public alpha package.

## Local Checks

Run these before opening a PR:

```bash
python -m pytest tests -v --tb=short
python -m ruff check src tests
python -m compileall -q src tests
```

## Boundaries

- Keep the runtime dependency surface NumPy-only by default.
- Preserve the `A` versus `G` distinction: `A` controls stiffness/frequency;
  `G` controls damping/forgetting.
- Do not claim invention of LinOSS or D-LinOSS.
- Update [PROVENANCE.md](PROVENANCE.md) and [CLAIMS.md](CLAIMS.md) when public
  wording changes.
