## Summary

<!-- What changed? -->

## Type

- [ ] Bug fix
- [ ] Feature
- [ ] Documentation
- [ ] Tests / CI
- [ ] Maintenance

## Boundary Checklist

- [ ] Preserves the `A` versus `G` distinction.
- [ ] Keeps NumPy as the only runtime dependency.
- [ ] Does not add host runtime, web framework, graph database, or event-bus dependencies.
- [ ] Does not claim invention of LinOSS or D-LinOSS.
- [ ] Updates README / PROVENANCE / CLAIMS if public wording changed.

## Verification

```bash
python -m ruff check src tests
python -m pytest tests -v --tb=short
python -m compileall -q src tests
```

## Notes

<!-- Remaining risks, follow-ups, or intentionally out-of-scope items. -->
