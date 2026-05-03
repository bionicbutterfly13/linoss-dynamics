# Release Checklist

`linoss-dynamics` is currently a public alpha package and is not published to
PyPI. Use this checklist before creating a GitHub release, TestPyPI upload, or
PyPI upload.

## Pre-Release Gate

- [ ] Confirm the release version in `pyproject.toml`.
- [ ] Confirm the release version in `CITATION.cff`.
- [ ] Move the relevant `CHANGELOG.md` entries from `Unreleased` to a dated
  release section.
- [ ] Run the local verification suite:

```bash
python -m ruff check src tests
python -m pytest tests -v --tb=short
python -m compileall -q src tests
python -m pip install --dry-run .
```

- [ ] Confirm GitHub Actions CI is green on the release commit.
- [ ] Re-read `CLAIMS.md` and `PROVENANCE.md` for public wording accuracy.
- [ ] Confirm no host-application dependencies are present in the package core.

## Optional Build Check

Install build tooling in a disposable environment:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Do not commit `dist/` artifacts.

## GitHub Release

- [ ] Create an annotated tag matching the package version.
- [ ] Create a GitHub release from that tag.
- [ ] Include:
  - package status
  - supported Python versions
  - verification commands
  - attribution note pointing to `PROVENANCE.md`

## TestPyPI / PyPI

- [ ] Publish to TestPyPI first.
- [ ] Install from TestPyPI into a clean virtual environment.
- [ ] Run a minimal import and one-step smoke:

```python
import numpy as np
from linoss_dynamics import damped_linoss_step

y_next, z_next, metrics = damped_linoss_step(
    np.array([0.2]),
    np.array([1.0]),
    np.array([1.0]),
    np.array([0.5]),
    dt=0.1,
)

assert metrics["damping_mode"] == "explicit_g"
```

- [ ] Publish to PyPI only after the TestPyPI install works.

## Downstream Pin

After any code-bearing release commit, update downstream Git pins such as
Dionysus. Docs-only releases do not require downstream pin updates unless the
downstream repository wants to point at the latest public surface.
