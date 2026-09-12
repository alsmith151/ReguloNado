# The track table

`tracks.parquet` is the single artifact naming which tracks exist, in what order, and why. It
replaces the old `regulonado_metadata.json` / `final_track_records` / `track_metadata_vocab`
combination entirely — that file no longer exists.

## Why one table

Before this design, the workflow built the Arrow dataset first and only then learned anything
about track quality: `scale_factors` read the *built* dataset's metadata, so a bad track was only
discoverable after the most expensive stage had already committed a fixed-width label tensor to
disk. Track identity was also decided twice — once implicitly by whatever `--bigwig`/`--bigwig-dir`
list was passed to `build`, and again by whatever survived dedupe inside the builder.

Now: `regulonado tracks discover` resolves raw BigWig sources into one row per track exactly once,
`regulonado tracks assemble` is the only place a track's final status is decided, and
`regulonado dataset --track-table` is the only way to name a track list.

## The five statuses

| `status` | Meaning |
|---|---|
| `included` | Survives discovery, dedupe, and any QC/exclude filtering; has a `track_index`. |
| `dropped_duplicate` | Identical (identity or content, per `--dedupe-tracks`) to another track. |
| `missing` | The BigWig path didn't exist at discovery time. |
| `qc_failed` | `qc_verdict == "failed"` and `qc.drop_degenerate` was set at assembly. |
| `excluded` | Named in `inputs.exclude_tracks` / `--exclude-track`, honoured whether or not QC ran. |

Only `included` rows have a non-null `track_index`, contiguous `0..n-1` in that order — this is the
schema's load-bearing invariant (`TrackTableSchema.track_index_is_contiguous` in
[tracks_table.py](../python/regulonado/tracks_table.py)): get it wrong and a label-tensor join
silently corrupts.

## Column groups

| Prefix | Group | Notes |
|---|---|---|
| *(none)* | identity | `track_name` (unique, non-null — the join key everywhere), `track_index`, `path`, `resolved_path` |
| *(none)* | annotation | `sample_id`, `condition`, `source`, `assay`, `ip`, `scaling_group`, ... — **labels, not ids** |
| *(none)* | dedupe | `dedupe_key`, `dedupe_method`, `duplicate_of` |
| `fp_` | fingerprint | `size_bytes`, `mtime_ns`, `bases_covered`, `genome_sum`, `content_hash` (nullable) — verified against disk at `tracks qc`/`build`/(warn-only) `train`, never joined on |
| `scale_` | scaling | `factor`, `background`, `clip_soft`, `clip_hard`, plus method diagnostics (`anchor_reference`, `background_q50/q99`, `heldout_recovery`, `quality`, `tmm_factor`, `library_size`, ...) |
| `qc_` | QC | `verdict`, `failed_rules` (list), plus one column per computed metric — see [qc.md](qc.md) |

**Categorical ids are derived, never stored.** `condition_id`/`source_id`/`assay_type_id`/
`target_id` don't exist as columns — `to_track_records()` derives them at load time by sorted
factorisation over the label columns, so an id always means the same thing regardless of which
stage populated the label, and there is nothing to keep `track_metadata_vocab`-style in sync.

Every column group is present in the final table whether or not its stage ran — an unrun stage
leaves its columns null rather than absent, so a reader never has to branch on which stages were
enabled ("stable schema").

## Adding metadata after discovery

A `group` label (or any other freeform annotation — a project name, an attribution note) doesn't
have to be known before `regulonado tracks discover` runs. Discover from bigwigs first, then
layer labels on with `tracks assemble --annotations`:

```bash
regulonado tracks discover results/tracks/_stages/discovered.parquet --bigwig-dir bigwigs/

# groups.csv:
#   track_name,group
#   atac_hl60_rep1,hl60
#   atac_hl60_rep2,hl60
#   atac_k562_rep1,k562

regulonado tracks assemble results/tracks/_stages/discovered.parquet \
  --annotations groups.csv \
  --output results/tracks/tracks.parquet
```

`--annotations` merges any CSV/parquet with a `track_name` column onto the table by that column,
the same way `--scale-factors`/`--qc-report` already do; every one of its other columns becomes a
bare, additive column on `tracks.parquet` (an unknown `track_name` in the file is rejected rather
than silently ignored). `group_by: group` in a `design`/`attribution` target then reads it back —
see [design.md](design.md#the-objective) and [attribution.md](attribution.md#attributing-against-a-track-group).

In the full pipeline (`regulonado.yaml`, not the bare CLI), point `inputs.track_annotations` at
the file instead of hand-invoking `tracks assemble`:

```yaml
inputs:
  bigwig_dir: bigwigs/
  track_annotations: groups.csv
```

`track_assemble` is a separate Snakemake rule from `track_discovery`
(`workflow/rules/tracks.smk`), and `track_annotations` is wired as *its* input, not
`track_discovery`'s — editing `groups.csv` (to add a group, fix a label, ...) only invalidates
`track_assemble` (and everything downstream of `tracks.parquet`). It never forces
`track_discovery` to re-scan every BigWig, the same way editing `inputs.exclude_tracks` already
doesn't.

## `tracks.parquet` is a strict superset

`tracks/_stages/discovered.parquet`, `interval_means.parquet`, `scale_factors.parquet`, and
`qc_report.parquet` are rule plumbing — kept only so re-running `tracks assemble` doesn't re-scan
every BigWig. **Nothing should read them directly.** Every column any stage produces appears in
`tracks.parquet`, for every track any stage saw.

## Inspecting a table

```bash
regulonado tracks show results/tracks/tracks.parquet
regulonado tracks targets results/tracks/tracks.parquet -o targets.txt  # Borzoi-format TSV
```

## Run-level metadata

Scalars that describe the whole run — `schema_version`, `created_at`, `regulonado_version`, the
originating command, `bin_size`/`context_length`/`n_pred_bins`/`shift_max_bp`, `splits`,
`build_strategy`, the `dedupe_tracks` summary, the scaling method, and the resolved QC config —
live in the Arrow schema metadata (`df.attrs` after `read_track_table`), not as columns. They
round-trip through `pq.write_table`/`read_table`.
