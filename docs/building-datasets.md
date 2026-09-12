# Build a dataset

`regulonado dataset` combines genomic intervals, reference sequence, and a pre-assembled track table
into a Hugging Face `DatasetDict`. Track discovery, dedupe, scaling, and QC all happen *before*
this command, via `regulonado tracks` — see [track-table.md](track-table.md) and
[normalization.md](normalization.md).

## Prepare the inputs

- BED column 4 contains the fold label used for splitting.
- The FASTA has a neighbouring `.fai` index and uses the same contig names as
  the BED and BigWigs.
- BigWigs share the same genome assembly.

## Assemble the track table

```bash
regulonado tracks discover results/tracks/_stages/discovered.parquet --bigwig-dir bigwigs/
regulonado normalization original results/tracks/_stages/discovered.parquet -o results/tracks/_stages/scale_factors.parquet
regulonado tracks assemble results/tracks/_stages/discovered.parquet \
  --scale-factors results/tracks/_stages/scale_factors.parquet \
  --output results/tracks/tracks.parquet
```

`discover` fills `track_name` from a track sheet when one is given, else the BigWig's file stem;
colliding stems among included tracks raise here, not later. `assemble` is the step that always
runs regardless of whether QC did, and is the only place a track's final `status` (and therefore
`track_index`) is decided — see [track-table.md](track-table.md) for the full column reference and
the five `status` values.

## Build

```bash
regulonado dataset intervals.bed genome.fa dataset/ \
  --track-table results/tracks/tracks.parquet \
  --split train:fold0,fold1,fold2 \
  --split validation:fold4 \
  --split test:fold3 \
  --shift-max-bp 128 \
  --n-extract-threads 16 \
  --stage
```

The default `in_memory` strategy is normally the best choice. `--stage`
copies source files to node-local scratch before reading them, which is useful
on network storage but requires enough scratch space for the FASTA and tracks.

Before writing any Arrow shard, `build` re-verifies each included track's fingerprint
(`fp_size_bytes`/`fp_mtime_ns`/the BigWig header summary) against disk and fails naming the track
and both values if a BigWig changed since assembly.

The output includes train, validation, and test Arrow shards plus `tracks.parquet` — the same
table passed in via `--track-table`, with build-time scalars (`context_length`, `bin_size`,
`splits`, ...) merged in. Keep it beside the dataset; there is no separate metadata-enrichment
step to run afterward.

## Recompress an existing dataset

Smaller Arrow record batches can improve random reads during training:

```bash
regulonado recompress-dataset dataset/ dataset_rechunked/ \
  --max-batch-size 4 --workers 8
```

The destination must not already exist. `tracks.parquet` is copied alongside the Arrow shards. The
workflow can perform this step automatically when `recompress.enabled` is true — in which case the
raw `dataset/` build output is a pure intermediate and Snakemake removes it once recompression has
consumed it.

Use `regulonado dataset --help` for geometry, compression, shard sizing, and chromosome filtering
options; use `regulonado tracks --help` for discovery, dedupe, and QC options.
