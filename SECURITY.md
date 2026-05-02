# Security Policy

`linoss-dynamics` is a small numerical runtime package with no network, web,
database, credential, or filesystem integration in the package core.

## Supported Versions

The project is currently in public alpha. Security fixes apply to the `main`
branch until the first stable release policy is published.

## Reporting a Vulnerability

Please do not open a public issue for a vulnerability report.

Use GitHub's private vulnerability reporting flow when available, or email the
maintainer listed on the GitHub repository profile with:

- affected version or commit
- a minimal reproduction
- expected impact
- any known mitigation

Reports that involve host applications should identify which behavior is in
`linoss-dynamics` itself and which behavior belongs to the host adapter.

## Scope

In scope:

- crashes, denial-of-service behavior, or unsafe numerical validation in the
  package core
- packaging or dependency issues that affect users installing the package

Out of scope:

- vulnerabilities in downstream host applications
- claims about model quality, training performance, or research novelty
- issues requiring optional dependencies not shipped by this package
