# Changelog

All notable changes to ReguloNado are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note:** After the first release, this file is managed automatically by
> [release-please](https://github.com/googleapis/release-please) based on
> [conventional commits](https://www.conventionalcommits.org/).

---

## [0.2.0](https://github.com/alsmith151/ReguloNado/compare/regulonado-v0.1.0...regulonado-v0.2.0) (2026-08-14)


### Features

* **adapter:** adapt Borzoi for transformers v5 weight loading and improve architecture building ([725e400](https://github.com/alsmith151/ReguloNado/commit/725e400c0cb6c698460a53d85cb820146f4bbc2d))
* **config:** enhance backbone architecture config handling for model weight loading ([ed2e8d3](https://github.com/alsmith151/ReguloNado/commit/ed2e8d3589f6e3c57b4591eed70047a546380aae))
* **model:** add pretrained_name to backbone architecture initialization ([7fac5b0](https://github.com/alsmith151/ReguloNado/commit/7fac5b064d546d0501cf2cfc97f161adcac746cc))
* **model:** upgrade to HF PreTrainedModel and add BigWig prediction ([69004a6](https://github.com/alsmith151/ReguloNado/commit/69004a61bcb6ce73f999fa7a3cce1ec54c9d4810))
* **predict:** add option to include one-hot input in RegionPrediction ([7815dfb](https://github.com/alsmith151/ReguloNado/commit/7815dfb8f8d6625427414f249679920695f779fe))
* **scripts:** add installation script for flash-attn with support for prebuilt wheels ([d629c07](https://github.com/alsmith151/ReguloNado/commit/d629c07634ccfb941ced9d447530fa4c214d2af7))
* **scripts:** update dataset tag handling in training script output directory ([e4efc36](https://github.com/alsmith151/ReguloNado/commit/e4efc3610bf27b8a6b412af505fde5f0f7c379e0))
* **sharding:** add shard size options to control Arrow shard file count and size ([8c622b2](https://github.com/alsmith151/ReguloNado/commit/8c622b220f579c3a612dadfd84f6ba07da612b68))
* **training:** enhance logging and add rank handling for distributed training ([02b4be5](https://github.com/alsmith151/ReguloNado/commit/02b4be5ae0e0b2ca0c8f3084214077b92cb4e769))


### Bug Fixes

* **backbone:** fail on unset pretrained_name instead of silent random init ([078893d](https://github.com/alsmith151/ReguloNado/commit/078893da9482d1758a9587955c8c7e10dce32397))
* **callbacks:** handle NaN/Inf values in axis limit calculations for plots ([ef49516](https://github.com/alsmith151/ReguloNado/commit/ef495165595bf890b4aa22e47571356705249a06))
* **slurm:** require explicit REPO_DIR instead of unreliable BASH_SOURCE ([1283ec1](https://github.com/alsmith151/ReguloNado/commit/1283ec1fa884bf188ce0a601e0e7ec3691b08f11))


### Documentation

* document debug-writers feature build command ([a5edb20](https://github.com/alsmith151/ReguloNado/commit/a5edb20def11d84a81ca45c3b9d243ff97ec0e5d))

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
