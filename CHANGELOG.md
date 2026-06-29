# Changelog

All notable changes to ReguloNado are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note:** After the first release, this file is managed automatically by
> [release-please](https://github.com/googleapis/release-please) based on
> [conventional commits](https://www.conventionalcommits.org/).

---

## [0.2.0](https://github.com/alsmith151/ReguloNado/compare/regulonado-v0.1.0...regulonado-v0.2.0) (2026-06-29)


### Features

* **adapter:** adapt Borzoi for transformers v5 weight loading and improve architecture building ([4a9fff2](https://github.com/alsmith151/ReguloNado/commit/4a9fff2d498c2811f3cad3ab99c33a498b89ec62))
* **config:** enhance backbone architecture config handling for model weight loading ([300b423](https://github.com/alsmith151/ReguloNado/commit/300b423603c88c00fe68ebaea9e4d3a2d2331e21))
* **model:** add pretrained_name to backbone architecture initialization ([d9289f0](https://github.com/alsmith151/ReguloNado/commit/d9289f05cd8338179ba87d7d46c782a5adf21ec6))
* **model:** upgrade to HF PreTrainedModel and add BigWig prediction ([67c637c](https://github.com/alsmith151/ReguloNado/commit/67c637c8dec7324fde8c924c37846af811b86a79))
* **predict:** add option to include one-hot input in RegionPrediction ([254ff3d](https://github.com/alsmith151/ReguloNado/commit/254ff3d925f0ae4249034970c316c5a64edc1eee))
* **scripts:** add installation script for flash-attn with support for prebuilt wheels ([e430e38](https://github.com/alsmith151/ReguloNado/commit/e430e3852e3f368de67049be172e19078deab650))
* **scripts:** update dataset tag handling in training script output directory ([525d670](https://github.com/alsmith151/ReguloNado/commit/525d670749cfbaee089562a0246583ee96f6fdd6))


### Bug Fixes

* **backbone:** fail on unset pretrained_name instead of silent random init ([d2b83ae](https://github.com/alsmith151/ReguloNado/commit/d2b83ae7cabfd2b48c4254cfe3b126ee2de9ff86))
* **callbacks:** handle NaN/Inf values in axis limit calculations for plots ([e8972f4](https://github.com/alsmith151/ReguloNado/commit/e8972f4fbbc4715e7db04f768be3c63c1df450ab))
* **slurm:** require explicit REPO_DIR instead of unreliable BASH_SOURCE ([26dad37](https://github.com/alsmith151/ReguloNado/commit/26dad37d837d84d90c578e718cfbb9a009e32fed))


### Documentation

* document debug-writers feature build command ([f553c04](https://github.com/alsmith151/ReguloNado/commit/f553c04552696e4c9615f032e4177afc675c0050))

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
