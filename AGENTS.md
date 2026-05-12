# linoss-dynamics Agent Guide

Start here, then read the task or issue you were assigned.

You are helping me write a technical blog post about this project.
The article explains how I built a class to solve [your problem].
Tone: conversational but precise. Audience: developers and other professionals interested in cognition, machine consciousness, active inference, and memory; avoid academic prose.
Always refer to actual files in this repo when writing examples.
Do not invent code — only reference what exists here.

## Map

- [README.md](README.md) — package purpose and usage.
- [PROVENANCE.md](PROVENANCE.md) — upstream sources and attribution boundaries.
- [CLAIMS.md](CLAIMS.md) — safe and forbidden public claims.
- [docs/architecture.md](docs/architecture.md) — package boundary and module map.
- [WORKFLOW.md](WORKFLOW.md) — Symphony-style agent workflow and required checks.

## Rules

- Keep the core NumPy-only unless a future issue explicitly adds optional extras.
- Preserve the `A` versus `G` boundary: `A` is stiffness/frequency; `G` is damping/forgetting.
- Do not claim invention of LinOSS or D-LinOSS.
- Do not add runtime-service, web-framework, graph-database, event-bus, or agent-framework dependencies to package core.
- Update tests and public docs when changing APIs.
- keep track of how you relate to LinOSS- what you do specifically

