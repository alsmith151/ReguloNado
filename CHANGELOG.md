# Changelog

All notable changes to ReguloNado are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note:** After the first release, this file is managed automatically by
> [release-please](https://github.com/googleapis/release-please) based on
> [conventional commits](https://www.conventionalcommits.org/).

---

## [Unreleased]

### Changed

- Split monolithic dependency list into focused extras: `data`, `train`, `gpu`,
  `viz`, `dev`, `jupyter`, `test`, `all`.
- Widened Python version range to `>=3.12,<3.14` for compatibility with the
  wider nadoverse ecosystem (nadoverse, quantnado).
- Bumped `numpy` to `>=2` (aligns with bamnado + quantnado).
- Bumped `pysam` to `>=0.23` (aligns with quantnado).
- Removed `torch<2.7` upper bound (allows co-installation with packages that
  pin later torch versions).
- Removed `lightning<3` upper bound.
- Removed `flash_attn<2.8` upper bound.
- Moved `ruff` and `ninja` from core deps to `[dev]`/`[gpu]` extras.
- Made top-level model imports in `__init__.py` lazy — core install no longer
  requires torch, borzoi-pytorch, or enformer-pytorch.
- Added `__version__` to the top-level namespace.
- Added `[project.urls]` with repository link.

### Added

- GitHub Actions CI workflow (`ci.yml`): Python 3.12/3.13 matrix, Rust
  toolchain, pytest over data/chrom-pass tests (no GPU required).
- GitHub Actions release workflow (`release.yml`): maturin wheel matrix
  (Linux x86_64/aarch64, macOS x86_64/aarch64) + PyPI OIDC Trusted Publisher.
- Automated versioning via release-please (`release-please.yml`,
  `release-please-config.json`, `.release-please-manifest.json`).

---

## [0.1.0] — 2026-06-22

Initial versioned release.

### Added

- Rust/PyO3 chromosome-pass Arrow writer (`chrom_pass`).
- `regulonado build` CLI for Arrow DatasetDict construction from BED/FASTA/BigWig.
- `regulonado train` CLI delegating to Hydra + HF Trainer.
- `regulonado scale` and dataset utility subcommands.
- Borzoi and Enformer backbone adapters.
- FiLM, ResidualFiLM, and TransferMLP prediction heads.
- Poisson-multinomial and related loss functions.
- Per-run provenance outputs (`provenance.json`, `resolved_config.json`,
  `git_diff.patch`).
