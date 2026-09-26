# Region counts on cached trunk embeddings

A run with `trunk: cached` and `target: region_counts` trains a small region-level count
head on a **frozen** pretrained trunk (AlphaGenome, Borzoi, Flashzoi, Enformer, …). The
trunk runs once, at long context, over every region. The embeddings covering each
region's scored 1 kb target are cached as a Hugging Face `datasets` Arrow file, and the
head trains on the cache in minutes. It is a run like any other (see [training.md](training.md)); only its trunk,
target and recipe differ from a fine-tuned profile model.

```
targets.region_counts.regions ── counts regions ──► region_set.parquet ─┬─ counts bam/gather ──► region_counts/dataset
                                                    (canonical rows)     │   (tracks.parquet)
                                                                         └─ embed regions ──────► embeddings/<cache>/
FASTA ────────────────────────────────────────────────────────────────────────┘                  │
                                                  train --trunk cached ◄───────────────────────────┘ (recipe phases)
```

In the workflow (`examples/2026-09-26-hl60-region-counts-alphagenome.yaml`):

```yaml
targets:
  region_counts:
    regions: regions.parquet
    anchor_regions: anchors.bed          # default: scaling.anchor_regions
    background_regions: background.bed   # default: scaling.background_regions
    exclude_regions: candidates.parquet  # dropped from train, every phase

train:
  recipes:
    curriculum:
      - {name: pretrain, preset: pretrain}
      - {name: specific, preset: specific}
      - {name: target, preset: target}
  runs:
    - name: hl60_alphagenome
      seed: 0
      recipe: curriculum
      backbone: {type: alphagenome, pretrained: all_folds}
      trunk: cached
      target: region_counts
      target_group: HL-60
      cache: {context: 1048576, stride: 524288}
```

```bash
regulonado pipeline config.yaml train -n   # dry run
regulonado pipeline config.yaml train
```

- **Embedding caches are derived from runs.** Each distinct trunk setup (backbone type,
  checkpoint and `cache` settings) gets one cache under `results/embeddings/`, named by
  that setup. Runs that share a setup share the cache, so extra seeds cost no GPU time.
- **Swapping a model is an edit to one run:** its `backbone`, `trunk`, `target` or
  `recipe`.
- With no runs, the workflow still builds `targets.region_counts`' dataset.

## One track table, either file type

Tracks are discovered once, with `tracks discover` then `tracks assemble`, into
`results/tracks/tracks.parquet`:

- **Region counts only:** tracks are discovered from BAMs (`--format bam`), from
  `inputs.track_sheet`, `inputs.bam_dir` or SeqNado projects. Only `inputs.fasta` is
  otherwise required, and no bigWig rule runs.
- **With a profile target:** tracks are discovered from bigWigs, and each row keeps its
  BAM: from the sheet's `bam` column, `inputs.bam_dir`, or the SeqNado project. Profile
  and region-count runs then train on exactly the same tracks, with the same QC,
  exclusions and groups.
- A BAM track table is stamped `track_format: bam`. The bigWig-only commands
  (`tracks interval-means`/`qc`, `normalization`, `dataset`) refuse it.
- **Same scored windows:** a region set that already has `target_start`/`target_end`
  (such as UEF's `regions.parquet`) keeps them. Count targets then line up row-for-row
  with that set's signal targets and splits.

## 1. Region counts from BAMs

```bash
regulonado tracks discover discovered.parquet --format bam --track-sheet tracks.csv [--bam-dir bams/]
regulonado tracks assemble discovered.parquet -o tracks.parquet [--annotations extra.csv]
regulonado counts regions regions.parquet -o region_set.parquet [--val-chroms chr8 --test-chroms chr9]
regulonado counts bam region_set.parquet --tracks tracks.parquet \
    --anchor-regions anchors.bed --background-regions background.bed \
    --out-dir counts_cache [--track <name>]
regulonado counts gather region_set.parquet --tracks tracks.parquet \
    --out-dir counts_cache --dataset-dir region_dataset
```

- The track sheet is the usual one (`sample_id`, `track_name`, `group`, `assay`, …):
  - `group` is the cell type replicates pool into;
  - `assay` is `atac`, `chip`, `cutandrun` or a common variant spelling
    (`counts.bam.ASSAY_CLASS_ALIASES`); `assay_class` can instead be merged in with
    `--annotations`;
  - `bam` is resolved relative to the sheet. Without it, `--bam-dir` is searched for
    `<sample_id>.bam`, then for the one BAM anywhere below it whose name contains
    `sample_id` as a whole token. So a sheet naming samples by run accession finds
    `…/GSE74912/SRR2920511.sorted.bam`, as UEF's `count-bams` did.
- `--bam-dir` alone, with no sheet, makes every `*.bam` in it a track; add `group` and
  `assay` with `--annotations`.
- The target is the 1 kb window centred on each input region, unless the regions already
  carry `target_start`/`target_end`. Regions that fall off a contig, and duplicate
  regions, are dropped.
- `counts regions` writes those rows without reading any BAM. `counts bam`, `counts
  gather` and `embed regions` all read its output, so the embedding cache can be built
  while counting is still running.
- ATAC counts Tn5-shifted insertions; ChIP and CUT&RUN count fragments. Duplicates and
  low-MAPQ reads are filtered (`counts.bam.DEFAULT_COUNT_SPECS`).
- Size factors are anchor-based, in count units:
  - `count_scale_high` is the median best-subwindow count over the anchor windows;
  - `count_scale_low` is the same median over the background windows;
  - `log_size_factor = log(high − low)`, centred across tracks.
- `counts bam` is resumable: it writes one parquet per track.
- Splits (set by `counts regions`) come from the regions' own `split` column when present, otherwise from
  `--val-chroms/--test-chroms`.

The region dataset is a directory of three row-aligned parquet files:
- `regions.parquet`: chrom, start, end, target_start, target_end, split;
- `counts.parquet`: one float32 column per track;
- `tracks.parquet`: track metadata, `group`, `count_unit`, scales and `log_size_factor`.

## 2. Embedding cache

```bash
regulonado embed regions region_set.parquet genome.fa --backbone alphagenome \
    --pretrained all_folds --out embeddings/alphagenome [--chroms chr21] [--rc] [--pool-to 128] \
    [--context 1048576] [--stride 524288] [--device auto]
```

- **Tiling** comes from the adapter's geometry
  (`output_bin_size`, `fixed_input_length`, `input_multiple`, `output_span`):
  - fixed-input backbones (Borzoi/Flashzoi 524,288 bp, Enformer 196,608 bp) tile by their
    own cropped output span;
  - flexible backbones (AlphaGenome) run at `--context` and keep the central `--stride`,
    so every kept bin has at least 256 kb of flanking context by default.
- **Per region** the cache stores `K = ceil(1000 / bin) + 1` bins, starting at
  `floor(target_start / bin)`. A region that straddles a tile boundary is stitched from
  the two adjacent tiles; no extra forward passes are run.
- **Pooling:** `--pool-to` averages adjacent bins after stitching.
- **Reverse complement:** `--rc` stores a reverse-complement pass flipped back into forward
  bin order (`features_rc`).
- **Storage:** one `<chrom>.arrow` per chromosome, a Hugging Face `datasets` Arrow file
  with `region_row` (int64) and `features` as `Array2D((K, D), "float16")` (plus
  `features_rc` with `--rc`).
  - Open one with `datasets.Dataset.from_file("chr1.arrow")`, or the whole cache with
    `load_dataset("arrow", data_files="<cache>/*.arrow")`.
  - Training memory-maps the files, so each region costs one read of its own `K x D`
    values: cheap on network filesystems such as Ceph, and shared by forked `DataLoader`
    workers. `data.in_memory: true` reads the whole cache into RAM once instead.
  - `manifest.parquet` records the backbone, checkpoint, bin size, K, D, context, stride,
    pool, rc and a hash of the region table.
  - A rerun with different settings, or against a different region table, fails rather
    than mixing.
- **Resumability:** one chromosome per job. Finished chromosome files are skipped. Tiles
  and writes are streamed, so memory stays at a few tiles.
- **Size:** Borzoi at native 32 bp is about 140 GB for 1.16M regions; `--pool-to 128`
  brings it to about 35 GB.

### AlphaGenome notes

- Install with the `alphagenome` extra (`uv sync --extra alphagenome`). It uses
  [`alphagenome-pytorch`](https://pypi.org/project/alphagenome-pytorch/).
- Weights come from the Hugging Face repo `gtca/alphagenome_pytorch`: `all_folds`
  (default) or `fold_0`…`fold_3`. A local `.safetensors` path also works as
  `--pretrained`.
- **Licence:** the pretrained AlphaGenome weights are released under **non-commercial**
  terms. Check the model card before using them or anything derived from them.
- The adapter calls
  `encode(x_NLC, organism_index=zeros(B), resolutions=(128,), channels_last=False)`
  and uses the `[B, 3072, L/128]` 128 bp embedding. It uses bf16 autocast on CUDA and
  float32 elsewhere.
- Input must be a multiple of 128 bp and at least 2,048 bp.
- **MPS** needs `PYTORCH_ENABLE_MPS_FALLBACK=1` in the environment *before* Python starts,
  because the rotary embedding uses `aten::logspace`. At 131 kb, one pass takes about
  1 s on an M-series laptop.

## 3. Training the head

```bash
regulonado train region_dataset --trunk cached --embeddings embeddings/alphagenome --preset pretrain
regulonado train region_dataset --trunk cached --embeddings embeddings/alphagenome --preset specific \
    --init-weights-from-checkpoint <pretrain_dir>
regulonado train region_dataset --trunk cached --embeddings embeddings/alphagenome --preset target \
    --target-group HL-60 --init-weights-from-checkpoint <specific_dir>
```

**Model** (`training/cached/model.py`):
- attention pooling over the K bins → LayerNorm → MLP (D → 512);
- per-group log rates, with an optional soft cap (`model.eta_max`);
- a `CountHead` with fixed log size factors, centred per-group replicate offsets and a
  per-track NB dispersion.

`CountHead` parameters are excluded from weight decay.

**Presets** (`python/configs/cached_experiment/`, composed over
`python/configs/train_cached.yaml`):

| preset     | what changes                                                                 |
|------------|------------------------------------------------------------------------------|
| `pretrain` | all regions                                                                  |
| `specific` | `data.specific_only` (group-level Gini filter, `data.gini_std_threshold`)   |
| `target`   | `data.contrast_weighting` on `data.target_group`, contrast loss, frozen noise |

`data.exclude_regions` (workflow: `targets.region_counts.exclude_regions`, applied to every
phase) drops train regions that overlap held-out sequences such as benchmark candidates.
Val and test are not touched. This is UEF's `--exclude_bed`.

Settings go through the recipe and run `settings:` layers, or `--set`. Keys that
`train_cached.yaml` does not declare are rejected, and the workflow composes every run ×
phase before scheduling anything.

The metric names match UEF (`eval_contrast_pearson_mean`, `eval_contrast_pearson_<group>`,
`eval_contrast_pearson_top_decile_<target>`), so runs can be compared directly with UEF v2.
