# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/) once it leaves alpha.

## [Unreleased]

### Added
- `nanoindent.quality`: machine-readable applicability metadata (assumptions,
  depth/thickness and contact-radius/thickness ratios, warnings) and a
  numerical-quality report (equilibrium residual, interpenetration, per-step
  Newton convergence, software versions), available as
  `ForceCurve.applicability()` and `ForceCurve.quality_report()`.
- `docs/validation.md`: validation matrix linking each benchmark to its
  reference, configuration, asserted tolerance and test.
- Public open-source release under the EUPL-1.2.
- `uv`-managed project: `uv.lock`, `dev` dependency group, `uv sync` workflow.
- Community files (contributing guide, code of conduct, security policy,
  citation metadata), issue and pull-request templates.
- Project website published with GitHub Pages.

### Changed
- Minimum Python is now 3.11.

## [0.1.0]

- Axisymmetric FEM of AFM nanoindentation of thin elastic films, analytic
  benchmarks (Hertz, Dimitriadis, Garcia, Chadwick), force-volume and
  viscoelastic analysis, supported-lipid-bilayer analysis, HS-AFM movie tools,
  and streaming/closed-loop control scaffolds.
