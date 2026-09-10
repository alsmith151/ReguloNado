# ReguloNado

Build Arrow datasets from BigWig/FASTA and fine-tune sequence-to-function genomics models
(Borzoi/flashzoi backbones) for cell-type-specific enhancer prediction and design.

## Layout

- Rust (`src/*.rs` → `regulonado._rs`): dataset build only, **not** on the inference path.
- Python (`python/regulonado/`), packaged via maturin with `python-source = "python"`.
- Hydra configs in `python/configs/`.
- Snakemake workflow packaged inside the wheel at `python/regulonado/workflow/`.

## CLI

Single Typer app, entry point `regulonado = "regulonado.cli.app:main"`. Commands live in
[cli/app.py](python/regulonado/cli/app.py); `pipeline` is registered from
[cli/pipeline.py](python/regulonado/cli/pipeline.py); the `tracks` sub-app (discover/qc/assemble/
show/targets/interval-means) lives in [cli/tracks.py](python/regulonado/cli/tracks.py).

## Model

`RegulonadoModel(PreTrainedModel)` = backbone + head, `forward(input_ids, **head_kwargs)`
([model/wrapper.py:56](python/regulonado/model/wrapper.py#L56)). Geometry lives entirely in
`RegulonadoConfig` ([model/config.py](python/regulonado/model/config.py)):
`context_length=524288`, `n_pred_bins=6144`, `bin_size=32` → **the model predicts only the
central 196 608 bp**; flanks are cropped inside the backbone. Input `(B, 4, L)`, output
`(B, n_tracks, n_bins)`.

Loading: `load_model_for_inference()` ([inference.py:187](python/regulonado/inference.py#L187));
metadata-conditioned heads need the kwargs from `_model_track_metadata()`
([inference.py:260](python/regulonado/inference.py#L260)).

"Fold" means two different things: BED column 4 (data split) and `train.runs[]` in the workflow
config (independent training runs, e.g. `examples/bamnado_csaw_folds.yaml`). They are never
ensembled by the training code — `regulonado.design.predictor.FoldEnsemble` is the one place that
does.

## Tracks

`TrackSheet`/`TrackRecord` ([tracks.py](python/regulonado/tracks.py)) is the sheet format; its
`CATEGORICAL_FIELDS` label→id mapping is mirrored in
[tracks_table.py](python/regulonado/tracks_table.py) as the canonical version. Track state is
persisted as `tracks.parquet`, not `regulonado_metadata.json` — see **Track table** below.
Categorical ids (`condition_id`, `source_id`, `assay_type_id`, `target_id`) are never stored;
they are *derived* at load time by sorted factorisation over the label columns
(`to_track_records()`), so an id always means the same thing regardless of which stage populated
the label. Nothing hardcodes the track count.

## Track table

`tracks.parquet` is the single artifact naming which tracks exist, in what order, and why —
produced by `regulonado tracks assemble` and read by everything downstream (build, train,
inference, design, attribution). It is a **strict superset of every stage output**: discovery,
scaling and QC each write to `tracks/_stages/*.parquet` (`discovered.parquet`,
`interval_means.parquet`, `scale_factors.parquet`, `qc_report.parquet`), but those are rule
plumbing only — kept so re-running assembly doesn't re-scan every BigWig — and should never be
read directly. The pandera schema in
[tracks_table.py](python/regulonado/tracks_table.py) (`TrackTableSchema`) is the contract:
`status` ∈ `included/dropped_duplicate/missing/qc_failed/excluded`, `track_index` contiguous
`0..n-1` over `included` rows and null elsewhere, `track_name` unique and non-null. Column prefixes
namespace a wide table: `fp_*` fingerprint (verified against disk, never joined on), `scale_*`
scaling diagnostics, `qc_*` QC metrics and verdict; identity/annotation/dedupe columns keep bare
names. `track_name` is the join key everywhere; `track_index` is positional and renumbers under
filtering, so never persist it across a re-assemble.

## Flashzoi caveat

`BorzoiBackboneAdapter` casts to bf16 and runs with `torch.autocast(enabled=False)` because
flash_attn's rotary cache goes NaN under autocast
([model/adapters.py:138](python/regulonado/model/adapters.py#L138)).

## Dependencies

Deps are split by extra: `core` has no torch — but **does** have pandas/polars/pyarrow/tqdm/pandera,
since [tracks_table.py](python/regulonado/tracks_table.py) and
[normalization.py](python/regulonado/normalization.py) need them at module scope; `train` has
torch; `data` adds the genuinely heavy set (datasets/pybigtools/pysam/scipy/tangermeme/psutil/
bioframe); `test` has pyfaidx. **Torch imports must be lazy or inside torch-only modules** so `pip
install regulonado` (core) still imports the package and runs the CLI.

## Workflow

`regulonado pipeline` drives Snakemake through its **Python API** (not the CLI), so profile keys
like `use-conda`/`use-apptainer` must be translated explicitly
([cli/pipeline.py](python/regulonado/cli/pipeline.py)). Scaling and QC now run **before** the
Arrow build, directly from BigWigs, not from Arrow shards afterward — this inverts the pipeline's
old shape and is the single most likely thing to assume backwards. Adding a stage means three
things (down from four — `config.schema.yaml` is generated, not hand-mirrored, see below):

1. a pydantic model in [config/models.py](python/regulonado/config/models.py)
2. a `rules/*.smk`
3. an `include:` plus a `rule all` entry in the [Snakefile](python/regulonado/workflow/Snakefile)

Settings dicts are rendered to CLI flags (`_override_flags` in
[rules/train.smk](python/regulonado/workflow/rules/train.smk)); checkpoints are resolved with
[scripts/resolve_checkpoint.py](python/regulonado/workflow/scripts/resolve_checkpoint.py).

`workflow/schemas/config.schema.yaml` is **generated** from `RegulonadoConfig.model_json_schema()`
by [config/schema_gen.py](python/regulonado/config/schema_gen.py) — run
`python -m regulonado.config.schema_gen` after changing `config/models.py`; CI fails if
regenerating produces a diff. Hand-editing the schema file is wrong.

## Normalisation

`scaling.method: anchor` uses bioframe-read BED/parquet biological reference windows;
`data.apply_squash` is *forced* to `false` by the `_anchor_disables_squash` validator when anchor
scaling is selected — it is enforced, not merely documented.

## Attribution module

`regulonado attribute` ([design/attribution.py](python/regulonado/design/attribution.py)) is the
pre-selection front end for `design`: one ISM sweep against **one** named track (resolved with
`inference._resolve_tracks`), then smooth/threshold/merge core calling via `scipy.ndimage`. Its
`core_regions.bed` is fed back in as `design --candidates`. Attribution is base-resolution — the
profile is indexed by mutated position, not by output bin. See [docs/attribution.md](docs/attribution.md).

Not `tangermeme.saturation_mutagenesis`: it materialises every mutant up front, which is 3.8 GB
per 600 bp candidate at a 524 288 bp context.

## Design module

`python/regulonado/design/` (`regulonado design` CLI + optional `design:` workflow stage) mutates
user-supplied endogenous enhancer candidates to sharpen cell-type specificity, rather than
generating sequence from scratch — the folds have only ever seen genomic sequence, so de-novo
sampling puts predictions off-manifold. See [docs/design.md](docs/design.md).

## Test convention

No conftest; hand-rolled `TinyBackbone`/`DummyAdapter` exposing
`feature_dim`/`forward_features`/`iter_named_blocks`, tiny geometry constants
(`tests/test_predict_bigwig.py`, `tests/test_model_transfer.py`). CI runs a named subset only —
see [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Lint

ruff, line-length 100, `select = ["E","F","I"]`.
