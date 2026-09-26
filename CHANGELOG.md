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

- **Workflow config (breaking):** every model is a run under `train.runs`, choosing its
  `backbone` (`{type, pretrained}`), `trunk` (`live` | `cached`), `target` (`profile` |
  `region_counts`) and `recipe`, a named phase chain under `train.recipes`. What a run
  predicts is configured under `targets`, and each dataset is built only if a run uses
  it:
  - `train.phases` → `train.recipes.<name>`; `runs[].pretrained_model` →
    `runs[].recipe` + `runs[].backbone`;
  - `inputs.intervals` + `dataset:` → `targets.profile`;
    `dataset.drop_missing`/`dedupe_tracks` → `inputs`;
  - tracks are discovered from bigWigs when `targets.profile` is set, otherwise from BAMs,
    into one `results/tracks/tracks.parquet`;
  - prediction, attribution and design only accept `target: profile` runs, and default to
    all of them;
  - examples are migrated. `hl60_anchor_folds`, `enhancer_candidates_anchor_folds` and
    `hl60_design` also drop their obsolete dotted `data.apply_squash` key.
- Renamed the `regulonado build` CLI command to `regulonado dataset` — the
  name now matches what it produces.
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
- `EnformerBackboneAdapter.feature_dim` now reports `2 * dim` (3072), which matches what
  `return_only_embeddings=True` actually returns (it was previously `dim`).

### Added

- Region-count modelling on cached trunk embeddings (`docs/region-counts.md`): a
  `trunk: cached`, `target: region_counts` run trains an NB count head on embeddings
  its pretrained trunk computed once:
  - `regulonado counts regions/bam/gather`: BAM region counting, ported from UEF, with
    anchor-based size factors, stored as parquet. `counts regions` writes the canonical
    region table that counting and embedding share, so embedding caches build while BAMs
    are still being counted. A region set's own `target_start`/`target_end` are kept;
  - `regulonado embed regions`: a backbone-agnostic per-chromosome parquet embedding
    cache. In the workflow, one cache per distinct trunk setup, shared by every run
    that uses it;
  - `regulonado train --trunk cached --embeddings DIR`, with `pretrain`/`specific`/`target`
    presets (`python/configs/cached_experiment/`, over `train_cached.yaml`);
  - `data.exclude_regions` / `targets.region_counts.exclude_regions`: held-out
    candidates leave training at every phase (UEF `--exclude_bed`).
- `regulonado tracks discover --format bam`: tracks discovered from BAMs share the
  bigWig tracks' `discover`/`assemble` steps and `tracks.parquet` schema. `--bam-dir`
  finds BAMs by `sample_id` (a whole-token match, so run accessions work) and, for bigWig
  discovery, records each track's `bam`. Track sheets and SeqNado projects now carry
  `bam` into `tracks.parquet`. bigWig-only commands refuse a BAM track table.
- `regulonado pipeline` composes every run × recipe phase before scheduling, and the
  cached-trunk config rejects keys `train_cached.yaml` does not declare.
- `AlphaGenomeBackboneAdapter` (`alphagenome` extra), and adapter output geometry
  (`output_bin_size`, `fixed_input_length`, `input_multiple`, `output_span`).
- `regulonado.sequence`: pid-safe pyfastx genome access and zero-padded window fetches.
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
