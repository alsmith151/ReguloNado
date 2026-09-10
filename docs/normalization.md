# Normalize track signal

## Anchor scaling

Anchor scaling puts every track into literal biological units: `1.0` is the
track's housekeeping-promoter reference. It subtracts the median of a supplied
GC-matched, regulatory-sequence-depleted background set and divides by the
median common-essential-promoter anchor signal. Values above 1 remain
unclipped. Region inputs may be BED files (read with PyRanges) or parquet files
with `chrom`, `start`, and `end` columns.

```yaml
scaling:
  method: anchor
  anchor_regions: resources/common_essential_windows.bed
  background_regions: resources/null-windows.parquet
  heldout_regions: resources/common_essential_windows_heldout.bed
  window_stat_bp: 1000
  background_sample: 5000

train:
  common:
    data.apply_squash: false
```

The command emits `background`, `scale_factor`, anchor/background quantiles,
held-out recovery, and a diagnostic `quality` band. Degenerate tracks are a
hard error; use `--allow-degenerate` only for exploratory runs. The source
project's provenance notebooks should be used to construct the two region
sets. The enriched metadata must include the `background` field so training
applies subtraction and scaling together.

Normalization is optional. Use it when BigWigs contain normalized values such
as RPKM and training should operate on approximate raw counts.

## Calculate scale factors

Infer the original RPKM-to-count factors:

```bash
regulonado normalization original dataset/regulonado_metadata.json \
  --output results/scaling/scale_factors.parquet
```

This command reads library information through the external `bamnado` binary,
which must be on `PATH` (or supplied through `BAMNADO`).

Optionally apply edgeR-style TMM correction using the training split:

```bash
regulonado normalization tmm dataset/regulonado_metadata.json \
  --scale-factors results/scaling/scale_factors.parquet \
  --output results/scaling/scale_factors.tmm.parquet
```

Or correct using bamnado's own BAM-based normalisation instead of the
dataset-restricted TMM above — `tmm`, `csaw-background` (bamnado's default),
`cpm`, `median-of-ratios`, or `spike-in`. This calls `bamnado bam-normalize`
directly, so it needs the original BAM files, matched to tracks by filename
stem (track `sample1.bw` needs a `sample1.bam` in `--bam-dir`):

```bash
regulonado normalization bamnado dataset/regulonado_metadata.json \
  --bam-dir path/to/bams \
  --method csaw-background \
  --scale-factors results/scaling/scale_factors.parquet \
  --output results/scaling/scale_factors.bamnado.parquet
```

In the Snakemake pipeline, set `scaling.method: bamnado`, `inputs.bam_dir`,
and optionally `scaling.bamnado_method` (default `csaw-background`) and
`scaling.bamnado_exogenous_prefix` (for `spike-in`).

## Reuse SeqNado's spike-in factors

When the tracks came from a SeqNado project that already computed spike-in
normalisation factors, reuse them instead of estimating a correction here. The
factors are read from `resources/<method>/normalisation_factors.tsv` inside the
project and applied to the library-size factors exactly as the TMM correction
is — `scale_factor = old_scale_factor / factor` — after being normalised to a
geometric mean of 1 so the overall magnitude is unchanged:

```bash
regulonado normalization seqnado dataset/regulonado_metadata.json \
  --project /data/expA/seqnado_output \
  --method orlando \
  --scale-factors results/scaling/scale_factors.parquet \
  --output results/scaling/scale_factors.seqnado.parquet
```

`--method` names which spike-in method's table to read. Omit it and SeqNado
returns every table it finds concatenated, with the first row for a sample
winning — so name the method explicitly whenever a project ran more than one.

Tracks are matched to rows by sample name (`sample_id`, falling back to
`track_name` and then the BigWig stem), not by position. A track with no factor
is an error rather than a silently unscaled track, and the message lists the
samples that do have one.

This method needs the optional `seqnado` extra; see
[Work alongside SeqNado](seqnado-interop.md).

In the Snakemake pipeline, set `scaling.method: seqnado` and optionally
`scaling.seqnado_spikein_method`. The project is taken from
`inputs.seqnado_projects` when exactly one is configured, or named explicitly in
`scaling.seqnado_project`.

This method is single-project only. SeqNado's factors are computed within one
project and are not comparable between projects, so applying one project's
factors across an aggregated dataset would put tracks on different scales
without saying so. Aggregated datasets should use `tmm`, which is derived from
the merged dataset itself. Configuring several projects without naming one in
`scaling.seqnado_project` is rejected — by the config model and again by the
workflow, so a hand-written config cannot slip past.

## Create enriched metadata

Write a new metadata file rather than changing the dataset's original record:

```bash
regulonado enrich-metadata \
  dataset/regulonado_metadata.json \
  results/scaling/scale_factors.tmm.parquet \
  --output results/scaling/regulonado_metadata.enriched.json
```

Rows are matched to tracks by `track_index`, not by sample name or file path
— every row in the scale-factors file must carry the `track_index` from
`final_track_records` in the metadata JSON, matching whichever track it
corresponds to. This is what lets you supply a hand-built table of custom
scale factors: any parquet or CSV with a `track_index` column plus whichever
of `scale_factor`, `clip_soft`, `clip_hard` you want to override (pick which
with `--field`, repeatable) works as input, it doesn't need to come from a
`normalization` command.

Pass that file to training explicitly:

```bash
regulonado train dataset/ --preset head_only \
  --metadata results/scaling/regulonado_metadata.enriched.json
```

The pipeline connects these paths automatically. It passes the enriched copy
to every phase while leaving the source metadata untouched.
