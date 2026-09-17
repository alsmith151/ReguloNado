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
the five `status` values. `--annotations` merges in a CSV/parquet the same way `--scale-factors`/
`--qc-report` do, for attaching a `group` label (or any other freeform metadata) after `discover`
without redoing it from a hand-written `--track-sheet` — see
[track-table.md](track-table.md#adding-metadata-after-discovery).

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

Each bin stores the mean BigWig signal over the bin. Two options control edge cases:

- `--bin-denominator bin_width` (default) divides a bin's summed signal by its in-contig
  width, so bases with no BigWig record count as zero coverage. `covered_bases` divides
  by recorded bases only. The two agree for BigWigs that write explicit zeros. For sparse
  BigWigs that omit zero stretches, `covered_bases` inflates low-coverage bins (a single
  read covering 5 of 32 bases reads as full-bin coverage).
- `--missing-bins nan` (default) stores NaN for bins with no data: bins past the contig
  end (window padding) and bins whose every base is NaN in the BigWig. Training masks
  them (`data.mask_missing`). `zero` stores 0.0, which trains them as observed zero
  signal. A base with no record is zero coverage, not missing.

Both values are recorded in the output `tracks.parquet` metadata.

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

## Dataset layout and sharing

The build output is a Hugging Face dataset in Parquet format:

```
dataset/
  README.md              HF dataset card (configs, features, split row counts)
  tracks.parquet         track table + geometry metadata
  data/
    train-00000-of-00092.parquet
    train-00001-of-00092.parquet
    ...
    validation-*.parquet
    test-*.parquet
```

Each `.parquet` file contains one or more row groups (default: one row per row group). Compression
is zstd level 3 by default; adjust with `--zstd-level`. Data columns are:

- `sequence_tokens`: uint8 fixed-size list (A=0, C=1, G=2, T=3; 4=N/padding)
- `signal`: float32 nested list `[n_tracks][n_bins]`
- `interval`, `index`, `local_index`: unchanged

The `README.md` is written last as a sentinel. It contains the dataset card YAML (configs and
features) with per-split row counts, making `load_dataset` report accurate split sizes.

### Load and inspect the dataset

`regulonado train` does not use `datasets` to read the data. It reads single Parquet row groups at
random, so it knows the exact row count, shuffles the whole dataset each epoch and needs no
streaming setup.

Other tools can open the same folder with `datasets`. Streaming is the practical way to inspect it:

```python
from datasets import load_dataset
dataset = load_dataset("path/to/dataset", streaming=True)
dataset["train"].info.splits["train"].num_examples  # row count, read from README.md
```

Avoid map-style `load_dataset` (without `streaming=True`) on full datasets. It writes an uncompressed
Arrow cache of about 3.5 MB per row. The `datasets` numpy and torch formatters widen `uint8`
columns to `int64`; use `with_format("arrow")` to keep the original type:

```python
dataset = load_dataset("path/to/dataset", streaming=True).with_format("arrow")
```

### Share on the Hugging Face Hub

Upload the whole `dataset/` directory to a Hub dataset repository:

```bash
hf upload <repo> dataset/ --repo-type dataset
```

Training and other tools load it by name or path:

```bash
regulonado train johahi/my-dataset
```

Use `regulonado dataset --help` for geometry, compression, shard sizing, and chromosome filtering
options; use `regulonado tracks --help` for discovery, dedupe, and QC options.
