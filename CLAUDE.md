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
[cli/pipeline.py](python/regulonado/cli/pipeline.py).

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

`TrackSheet`/`TrackRecord` ([tracks.py](python/regulonado/tracks.py)), `CATEGORICAL_FIELDS` maps
sheet columns → id fields; persisted as `final_track_records` in `regulonado_metadata.json`.
Nothing hardcodes the track count.

## Flashzoi caveat

`BorzoiBackboneAdapter` casts to bf16 and runs with `torch.autocast(enabled=False)` because
flash_attn's rotary cache goes NaN under autocast
([model/adapters.py:138](python/regulonado/model/adapters.py#L138)).

## Dependencies

Deps are split by extra: `core` has no torch; `train` has torch; `data` has polars/tqdm/tangermeme;
`test` has pyfaidx. **Torch imports must be lazy or inside torch-only modules** so `pip install
regulonado` (core) still imports the package and runs the CLI.

## Workflow

`regulonado pipeline` drives Snakemake through its **Python API** (not the CLI), so profile keys
like `use-conda`/`use-apptainer` must be translated explicitly
([cli/pipeline.py](python/regulonado/cli/pipeline.py)). Adding a stage means four things:

1. a pydantic model in [config/models.py](python/regulonado/config/models.py)
2. a mirrored block in
   [workflow/schemas/config.schema.yaml](python/regulonado/workflow/schemas/config.schema.yaml)
3. a `rules/*.smk`
4. an `include:` plus a `rule all` entry in the [Snakefile](python/regulonado/workflow/Snakefile)

Settings dicts are rendered to CLI flags (`_override_flags` in
[rules/train.smk](python/regulonado/workflow/rules/train.smk)); checkpoints are resolved with
[scripts/resolve_checkpoint.py](python/regulonado/workflow/scripts/resolve_checkpoint.py).

## Normalisation

`scaling.method: anchor` uses PyRanges BED/parquet biological reference windows; set
`data.apply_squash: false` when training anchor-scaled targets.

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
