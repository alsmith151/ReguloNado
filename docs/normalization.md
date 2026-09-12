# Normalize track signal

Scaling runs **before** the Arrow build, directly against BigWigs named in a track table
(`tracks discover`'s `discovered.parquet`, or any `tracks.parquet`-shaped file) — not against a
built dataset. `regulonado tracks assemble` merges the resulting `scale_factors.parquet` into
`tracks.parquet`, which the build then reads. See [building-datasets.md](building-datasets.md) and
[track-table.md](track-table.md).

## Anchor scaling

Anchor scaling puts every track into literal biological units: `1.0` is the
track's housekeeping-promoter reference. It subtracts the median of a supplied
GC-matched, regulatory-sequence-depleted background set and divides by the
median common-essential-promoter anchor signal. Values above 1 remain
unclipped. Region inputs may be BED files (read with bioframe) or parquet files
with `chrom`, `start`, and `end` columns.

```yaml
scaling:
  method: anchor
  anchor_regions: resources/common_essential_windows.bed
  background_regions: resources/null-windows.parquet
  heldout_regions: resources/common_essential_windows_heldout.bed
  window_stat_bp: 1000
  # Per-assay override, e.g. wider windows for broader marks:
  window_stat_bp_by_assay: {H3K27ac: 2000}
  background_sample: 5000

qc:
  checks: [anchor]     # optional: drop degenerate tracks at assembly instead of training on them
  drop_degenerate: false

train:
  common:
    data.apply_squash: false
```

The command emits `background`, `scale_factor`, anchor/background quantiles,
held-out recovery, and a diagnostic `quality` band (`informative`/`weak`/`failed`). Degenerate
tracks (`quality == "failed"`) are written as-is — `anchor_scale_factors` already falls back to
`scale_factor=1.0` for them — and are **not** dropped here. Add `anchor` to `qc.checks` (with
`qc.drop_degenerate: true`) to drop them at `tracks assemble` instead, since that decision belongs
to the opt-in QC gate. The source project's provenance notebooks should be used to construct the
two region sets.

Normalization is optional. Use it when BigWigs contain normalized values such
as RPKM and training should operate on approximate raw counts.

## Calculate scale factors

Infer the original RPKM-to-count factors from every included track in a track table:

```bash
regulonado normalization original results/tracks/_stages/discovered.parquet \
  --bin-size 32 \
  --output results/tracks/_stages/scale_factors.parquet
```

This command reads library information through the external `bamnado` binary,
which must be on `PATH` (or supplied through `BAMNADO`).

Optionally apply edgeR-style TMM correction, scanning `inputs.intervals` directly (the same scan
`tracks qc --check interval_signal/replicate_concordance` shares via `tracks interval-means`, so
compute that once and pass it to both with `--interval-means` if you're running QC too):

```bash
regulonado normalization tmm results/tracks/_stages/discovered.parquet \
  --intervals intervals.bed \
  --scale-factors results/tracks/_stages/scale_factors.parquet \
  --bin-size 32 --n-pred-bins 6144 --shift-max-bp 64 \
  --output results/tracks/_stages/scale_factors.parquet
```

Or correct using bamnado's own BAM-based normalisation instead of the
dataset-restricted TMM above — `tmm`, `csaw-background` (bamnado's default),
`cpm`, `median-of-ratios`, or `spike-in`. This calls `bamnado bam-normalize`
directly, so it needs the original BAM files, matched to tracks by filename
stem (track `sample1.bw` needs a `sample1.bam` in `--bam-dir`):

```bash
regulonado normalization bamnado results/tracks/_stages/discovered.parquet \
  --bam-dir path/to/bams \
  --method csaw-background \
  --scale-factors results/tracks/_stages/scale_factors.parquet \
  --output results/tracks/_stages/scale_factors.parquet
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
regulonado normalization seqnado results/tracks/_stages/discovered.parquet \
  --project /data/expA/seqnado_output \
  --method orlando \
  --scale-factors results/tracks/_stages/scale_factors.parquet \
  --output results/tracks/_stages/scale_factors.parquet
```

`--method` names which spike-in method's table to read. Omit it and SeqNado
returns every table it finds concatenated, with the first row for a sample
winning — so name the method explicitly whenever a project ran more than one.

Tracks are matched to rows by sample name (`sample_id`, falling back to
`track_name`), not by position. A track with no factor is an error rather than
a silently unscaled track, and the message lists the samples that do have one.

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

## Assemble into tracks.parquet

Every scaling command writes bare column names (`scale_factor`, `clip_soft`, `background`, ...);
`tracks assemble` namespaces them (`scale_factor`, `scale_clip_soft`, `scale_background`, ...) and
merges them into `tracks.parquet` by `track_name`:

```bash
regulonado tracks assemble results/tracks/_stages/discovered.parquet \
  --scale-factors results/tracks/_stages/scale_factors.parquet \
  --output results/tracks/tracks.parquet
```

`regulonado dataset --track-table results/tracks/tracks.parquet ...` then reads scale factors
straight from the table it already needs for track identity — there is no separate metadata-
enrichment step. See [track-table.md](track-table.md) for the full column reference.
