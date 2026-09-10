# Track QC

QC is opt-in and user-selected: `qc.checks` is an explicit list, and an absent or empty list means
no QC runs at all — `regulonado tracks qc` is simply never invoked by the workflow. See
[track-table.md](track-table.md) for where the resulting columns land.

```yaml
qc:
  checks: [sparsity, interval_signal]
  rules:
    qc_nonzero_bin_fraction: {min: 0.01}
    qc_dynamic_range: {min: 1.5}
  drop_degenerate: true
  sample_windows: 20000     # optional: sample this many BED rows for interval-based checks
  max_workers: 16
```

`tracks qc` only **computes** metrics and a verdict — it never mutates a track's `status`.
Dropping happens in `tracks assemble` (`--drop-degenerate`, driven by `qc.drop_degenerate` +
`qc.rules`), which is the step that always runs, QC or not.

## Checks

| Check | Metrics | Cost |
|---|---|---|
| *(always, free)* | `qc_bases_covered`, `qc_genome_sum/min/max`, `qc_n_chroms_with_signal`, `qc_is_constant` | BigWig header only |
| `sparsity` | Gates on the free header metrics (`qc_bases_covered >= 1`, `qc_n_chroms_with_signal >= 1`, not constant) | Free |
| `interval_signal` | `qc_nonzero_bin_fraction`, `qc_interval_mean/sd`, `qc_dynamic_range` (q99/q50), `qc_top1pct_signal_fraction` | One BigWig scan over `inputs.intervals` |
| `replicate_concordance` | `qc_replicate_r` (median Pearson to tracks sharing `scaling_group`/`condition`), `qc_nearest_other_r`, `qc_nearest_is_replicate` | Same scan as `interval_signal` |
| `anchor` | Reuses `scale_quality`, `scale_heldout_recovery`, `scale_anchor_reference`, `scale_background_q50/q99` from `--scale-factors` | Free — no second scan |

`interval_signal` and `replicate_concordance` share one BigWig scan
(`regulonado tracks interval-means`, `qc.py:track_interval_means`) — the same scan `tmm` scaling
uses. In the Snakemake workflow this is a dedicated rule (`track_interval_means`), built once and
consumed by both `scale_factors` and `track_qc` so BigWigs are never scanned twice; from the CLI
directly, pass `--interval-means` to skip a fresh scan when you already have one.

`replicate_concordance` needs a group label (`scaling_group` or `condition`); with none it degrades
to `qc_nearest_other_r` alone (`qc_replicate_r`/`qc_nearest_is_replicate` stay null) — the same
comparison `deeptools multiBigwigSummary` + `plotCorrelation` makes, without the replicate
labelling. deeptools itself wasn't adopted here: it's a heavy CLI dependency for exactly the
computation the interval-means matrix already gives for free.

## `anchor` requires anchor scaling

`qc.checks: [anchor]` is rejected unless `scaling.method: anchor` — the diagnostics it reuses only
exist under that scaling method. This is enforced by a config cross-validator, not just documented.

## When *not* to use the `anchor` check

`quality` is `anchor_reference > background_q50` against housekeeping promoters — a claim about
ATAC/H3K27ac-like enhancer activity. On assays where that comparison isn't meaningful (most
non-accessibility, non-active-enhancer marks), don't enable it; use `sparsity`/`interval_signal`/
`replicate_concordance` instead, which make no assay-specific assumption.

## Rules and dropping

`qc.rules` is `{column: {min: x, max: y}}` — any configured bound a row violates adds that column
name to `qc_failed_rules` and sets `qc_verdict` to `"failed"`. `qc.drop_degenerate: true` then
turns a `"failed"` verdict into `status: qc_failed` at assembly (renumbering `track_index`
contiguously over the survivors); with it `false` (the default), failed tracks stay `included` and
visibly marked, e.g. for a first look at how many tracks a threshold would actually drop:

```bash
regulonado pipeline config.yaml --dry-run   # always dry-run before changing qc.rules —
                                             # tracks.parquet is an ordinary Snakemake input to
                                             # build_dataset, so a threshold tweak re-triggers it
```

`inputs.exclude_tracks` (or `tracks assemble --exclude-track`) is a manual name list honoured
**always**, independent of whether QC ran at all.
