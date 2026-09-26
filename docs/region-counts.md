# Region-count modelling on cached backbone embeddings

This path trains a small region-level count head on top of a **frozen** pretrained
backbone (AlphaGenome, Borzoi, Flashzoi, Enformer, …). The backbone runs once, at long
context, over every region. The embeddings covering each region's scored 1 kb target are
cached as parquet, and the head then trains on the cache in minutes.

```
regions.parquet ─┬─ regulonado counts bam/gather ──► region dataset (regions/counts/tracks.parquet)
tracks.parquet  ─┘                                        │
FASTA ───────────── regulonado embed regions ──────────► embeddings/<name>/<chrom>.parquet + manifest.parquet
                                                          │
                    regulonado train-regions ◄────────────┘   (pretrain → specific → target)
```

Snakemake is the primary entry point (a `regions:` section in the config; see
`examples/2026-09-26-hl60-region-counts-alphagenome.yaml`). Each step is also a CLI command.

## 1. Region counts from BAMs

```bash
regulonado counts bam regions.parquet --tracks tracks.parquet \
    --anchor-regions anchors.bed --background-regions background.bed \
    --out-dir counts_cache [--track <name>]
regulonado counts gather regions.parquet --tracks tracks.parquet \
    --out-dir counts_cache --dataset-dir region_dataset [--val-chroms chr8 --test-chroms chr9]
```

- The target is the 1 kb window centred on each input region. Regions that fall off a
  contig, and duplicate regions, are dropped.
- ATAC counts Tn5-shifted insertions; ChIP and CUT&RUN count fragments. Duplicates and
  low-MAPQ reads are filtered (`counts.bam.DEFAULT_COUNT_SPECS`).
- Size factors are anchor-based, in count units:
  - `count_scale_high` is the median best-subwindow count over the anchor windows;
  - `count_scale_low` is the same median over the background windows;
  - `log_size_factor = log(high − low)`, centred across tracks.
- `counts bam` is resumable: it writes one parquet per track.
- Splits come from the regions' own `split` column when present, otherwise from
  `--val-chroms/--test-chroms`.

The region dataset is a directory of three row-aligned parquet files:
- `regions.parquet`: chrom, start, end, target_start, target_end, split;
- `counts.parquet`: one float32 column per track;
- `tracks.parquet`: track metadata, `group`, `count_unit`, scales and `log_size_factor`.

## 2. Embedding cache

```bash
regulonado embed regions region_dataset genome.fa --backbone alphagenome \
    --out embeddings/alphagenome [--chroms chr21] [--rc] [--pool-to 128] \
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
- **Storage:** one uncompressed parquet per chromosome, with `region_row` (int64) and
  `features` as `fixed_size_list<halffloat>[K*D]`.
  - `manifest.parquet` records the backbone, checkpoint, bin size, K, D, context, stride,
    pool, rc and a hash of the region table.
  - A rerun with different settings, or against a different region table, fails rather
    than mixing.
- **Resumability:** one chromosome per job. Finished chromosome files are skipped. Tiles
  and row groups are streamed, so memory stays at a few tiles.
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
regulonado train-regions region_dataset --embeddings embeddings/alphagenome --preset pretrain
regulonado train-regions region_dataset --embeddings embeddings/alphagenome --preset specific \
    --init-weights-from-checkpoint <pretrain_dir>
regulonado train-regions region_dataset --embeddings embeddings/alphagenome --preset target \
    --target-group HL-60 --init-weights-from-checkpoint <specific_dir>
```

**Model** (`regions/model.py`):
- attention pooling over the K bins → LayerNorm → MLP (D → 512);
- per-group log rates, with an optional soft cap (`model.eta_max`);
- a `CountHead` with fixed log size factors, centred per-group replicate offsets and a
  per-track NB dispersion.

`CountHead` parameters are excluded from weight decay.

**Stages** (presets in `python/configs/regions_experiment/`):

| preset     | what changes                                                                 |
|------------|------------------------------------------------------------------------------|
| `pretrain` | all regions                                                                  |
| `specific` | `data.specific_only` (group-level Gini filter, `data.gini_std_threshold`)   |
| `target`   | `data.contrast_weighting` on `data.target_group`, contrast loss, frozen noise |

The metric names match UEF (`eval_contrast_pearson_mean`, `eval_contrast_pearson_<group>`,
`eval_contrast_pearson_top_decile_<target>`), so runs can be compared directly with UEF
v2. Other settings go through `--set key=value`; see `python/configs/regions.yaml`.
