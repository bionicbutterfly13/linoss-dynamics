---
tracker:
  kind: linear
  project_slug: cognitive-substrate
  active_states:
    - Ready for Agent
    - In Progress
  terminal_states:
    - Done
    - Canceled
    - Cancelled
    - Duplicate
polling:
  interval_ms: 60000
workspace:
  root: .worktrees/symphony
hooks:
  before_run: |
    test -f AGENTS.md
    git status --short --branch
  after_run: |
    git status --short --branch || true
  timeout_ms: 60000
agent:
  max_concurrent_agents: 2
  max_concurrent_agents_by_state:
    Ready for Agent: 2
    In Progress: 1
  max_turns: 8
codex:
  command: codex app-server
  turn_timeout_ms: 3600000
  read_timeout_ms: 5000
  stall_timeout_ms: 300000
---

# linoss-dynamics Agent Workflow

You are working on the `linoss-dynamics` package.

## Issue Context

- Identifier: `{{ issue.identifier }}`
- Title: `{{ issue.title }}`
- URL: `{{ issue.url }}`
- Attempt: `{{ attempt }}`

## Operating Rules

1. Work only inside the assigned Symphony workspace or linked worktree.
2. Read [AGENTS.md](AGENTS.md), [PROVENANCE.md](PROVENANCE.md), and [CLAIMS.md](CLAIMS.md) before changing package code.
3. Keep package core independent from host runtime services.
4. Keep runtime dependencies minimal; NumPy is the only default dependency.
5. Preserve attribution and claim boundaries in public docs.
6. Write or update focused tests before claiming implementation is complete.
7. Open or update a PR with verification commands and results.

## Required Checks

Run the package checks from the repository root:

```bash
python -m pytest tests -v --tb=short
python -m ruff check src tests
```

## Handoff

End at review when code, tests, docs, and provenance updates are committed and the PR body records verification plus any remaining risk.
