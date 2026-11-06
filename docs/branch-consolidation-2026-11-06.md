# Alpha main consolidation — 6 November 2026

The consolidated main uses `feat/enhancer-generators` at `285db1c` as its
baseline. Remote references were refreshed before comparing branches. This
retains the latest profile training, region-count training, enhancer design,
attribution, and diffusion generation together in one version.

## Branch accounting

| Branch | Disposition |
| --- | --- |
| `feat/enhancer-generators` | Baseline, including numeric diffusion training, pipeline checkpoint resolution, partial-mask editing, and recoverable sequence provenance. |
| `feat/region-counts-cached-embeddings`, `feat/clear-specificity-vocabulary` | Both point to `0166c48`, an ancestor of the baseline. Includes cached/live trunks, AlphaGenome embeddings, region-count heads, parameter sweeps, design, and attribution. |
| `feat/peft-lora-locon` | Ancestor of the baseline; adapter training and checkpoint validation are retained. |
| `feat/parquet-dataset` | Local and remote tips are ancestors of the baseline; includes the local extra ignore-file commit. |
| `feat/enhancer-design` | Ancestor of the baseline; design and streaming epoch sizing are retained. |
| `audit-fixes-2026-09-10` | Its complete tree is identical to `a2e519f`, already in the baseline. Audit fixes and regression tests are included through that squash merge. |
| `chore/pypi-release-prep` | Superseded by the released CLI/configuration structure already in the baseline. The former remote tip `9dbe9cd` adds Slurm executor settings, which are retained in `cli/pipeline.py`. Remote branch has been deleted; local branch is preserved. |
| Local `main` | Its tree at `80fec8a` is identical to remote main at `15cf576`; the different commit IDs reflect duplicated history, not missing features. Preserved as `archive/main-before-consolidation-2026-11-06`. |
| `origin/feat/training-cli-discovery` | Preserved as historical work. Its June experiment discovery, interactive launcher, and condition-agnostic presets target an obsolete CLI/configuration model. August commit `99aecfb` deliberately replaced that model with `train --preset` / `--set`, composed configuration, and curated phase presets. Restoring its old entrypoint would break the newer workflow. |
| `origin/release-please--branches--main--components--regulonado` | Pending automated version/changelog proposal, not a feature branch. Version 0.2.0 is not declared released by this consolidation. |
| `origin/copilot/remove-asmith151-collaborator` | Same tip as remote main; no additional code. |

No feature branches were deleted and no remote main history was rewritten.
The consolidation also fixes existing Python lint and Rust formatting failures.

## Validation

- Full Python suite: 1,121 passed and 15 skipped across the initial run and
  successful retry of all 29 cache-permission failures. The 27 workflow DAG
  tests also passed in a separate run with cache access.
- `ruff check python/ tests/`.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and `cargo test --all-targets`.
- `uv lock --check --offline` with a writable temporary cache.
- Regenerated the workflow schema and checked it against the tracked version.

These are local CPU/data/workflow checks; full GPU training and cluster execution
were not run as part of branch consolidation.
