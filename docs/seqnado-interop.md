# Work alongside SeqNado

[SeqNado](https://github.com/Milne-Group/SeqNado) is the Milne Group's Snakemake
NGS pipeline. It takes FASTQs to aligned BAMs, peaks and BigWig tracks.
ReguloNado starts where SeqNado stops: it takes those BigWigs and builds the
Arrow dataset a sequence-to-function model trains on.

Three things are shared rather than duplicated:

- **Execution presets** — one set of Snakemake profiles in
  `~/.config/snakemake/`, used by both pipelines.
- **The genome registry** — one `genome_config.json` recording where each
  genome's FASTA and companion files live.
- **The sample-sheet vocabulary** — ReguloNado's track sheet reuses SeqNado's
  design column names, so a design sheet is close to a drop-in.

On top of that, ReguloNado can read a SeqNado output directory directly: track
paths, BAMs, per-sample annotation, and spike-in normalisation factors.

## Install

SeqNado is an optional extra:

```bash
pip install "regulonado[seqnado]"
```

It is optional deliberately. SeqNado pins `pydantic==2.13.4` exactly and pulls
in a full NGS stack — aligners, peak callers, QC tooling — that model training
never touches. Making it a hard dependency would constrain every ReguloNado
install for the sake of features many users do not need.

**The project API is not in a published SeqNado release yet.** The extra floors
at `seqnado>=1.1`; `seqnado.open_project` landed after 1.0.7. Until 1.1 ships,
install from the development branch:

```bash
pip install "seqnado @ git+https://github.com/Milne-Group/SeqNado@develop"
```

ReguloNado checks for the API rather than only for the import, so a SeqNado that
imports cleanly but predates `open_project` produces one actionable message
instead of an `AttributeError` somewhere further down.

Without SeqNado installed, ReguloNado still works completely: hand-written track
sheets, `--bigwig-dir` globbing, and every scaling method except `seqnado` are
unaffected. Only the project-derived features need it.

## Shared execution presets

Both tools read Snakemake profiles from
`~/.config/snakemake/profile_<words>/config.yaml`, and both address them by a
shortcode built from the initials of the words after `profile_` —
`profile_slurm_gpu` is `sg`.

`regulonado init` installs the three ReguloNado ships with:

| Shortcode | Profile directory | Use |
| --- | --- | --- |
| `le` | `profile_local_environment` | Local execution; dry runs and smoke tests |
| `sg` | `profile_slurm_gpu` | Slurm, GPU training in the ambient environment |
| `ssg` | `profile_slurm_singularity_gpu` | As `sg`, but rules run under Apptainer |

Presets that already exist on disk are left alone, including SeqNado's, so
`regulonado init` never overwrites a profile you have edited. Use `--force` if
you want the packaged copy back, or `--dry-run` to see what would be written.

If SeqNado is installed, its presets resolve here too — `lc`, `ls`, `ld`, `ss`,
`a` and any others your version ships. `regulonado init` prints the full
resolved list, which is the authoritative answer for your machine. Where a
shortcode is claimed by both tools, ReguloNado's wins, because its resource
requests are the ones sized for this workflow. In practice only `le` collides,
and both tools name that directory `profile_local_environment`.

Select one when running the pipeline:

```bash
regulonado pipeline config.yaml --preset sg
```

Choosing a SeqNado preset that carries no `set-resources` for `build_dataset` or
`train_phase` prints a warning. Its defaults (`mem: 3G`, `runtime: 1h`) are
correct for alignment jobs and hopeless for dataset building or training. See
[Run on Slurm](slurm.md) for the resource layout in detail.

## Shared genome registry

SeqNado records where each genome's FASTA, chromosome sizes, GTF and blacklist
live in `~/.config/seqnado/genome_config.json`. Setting `SEQNADO_CONFIG`
overrides the home directory that path is built from. ReguloNado reads the same
file, so a genome configured once is available to both tools:

```bash
regulonado config --genome hg38
```

That resolves the FASTA from the registry and uses it as `inputs.fasta`. If the
name is not in the registry the command fails and lists what is available; if
there is no registry at all it warns and asks for a FASTA path directly.

Entries whose `fasta`, `bt2_index` or `chromosome_sizes` contain `PATH_TO` are
skipped. Those are the template stubs SeqNado writes on a fresh install, and
resolving one would only produce a confusing failure much later.

ReguloNado reads the JSON directly instead of calling SeqNado's loader — see
[Reuse and reimplementation](#reuse-and-reimplementation) for why.

## The track sheet

ReguloNado's training code reads per-track categorical ids — `condition_id`,
`source_id`, `assay_type_id`, `target_id` — and `timepoint_minutes` from the
dataset metadata. Before the track sheet existed, nothing populated them: tracks
were discovered by globbing a directory, so every track trained with the "value
absent" sentinel. The track sheet is what fills that gap. Everything else on
this page follows from it.

A sheet is a CSV. Its column names deliberately match SeqNado's design sheet, so
a SeqNado user recognises the file and a design is close to a drop-in:

| Column | Shared with SeqNado | Meaning |
| --- | --- | --- |
| `sample_id` | yes | Sample name; the key SeqNado resolves paths from |
| `condition` | yes | Biological condition or experimental group; becomes `condition_id` |
| `scaling_group` | yes | Samples scaled together (a batch, or one antibody) |
| `consensus_group` | yes | Samples merged into a consensus track |
| `group` | yes | Experimental group label |
| `ip` | yes | IP target / antibody; becomes `target_id` |
| `control` | yes | Control sample name for an IP assay |
| `assay` | yes | Assay type; becomes `assay_type_id` |
| `track_name` | no | Track's name in the dataset; defaults to `sample_id` |
| `bigwig` | no | Path to the BigWig; relative paths resolve against the sheet |
| `bam` | no | Path to the matching BAM |
| `source` | no | Biological source; becomes `source_id` |
| `timepoint_minutes` | no | Numeric time point |
| `project` | no | Which SeqNado project this row belongs to |
| `method` | no | Pileup method the BigWig came from |
| `scale` | no | Scaling variant the BigWig came from |

Unrecognised columns are warned about and ignored.

Every label column — `track_name`, `sample_id`, `condition`, `scaling_group`,
`consensus_group`, `group`, `ip`, `control`, `assay`, `source`, `project` — must
match `^[a-zA-Z0-9_-]+$`. This is SeqNado's rule, and it exists for the same
reason in both tools: these labels are interpolated into output paths, and
whitespace or shell metacharacters there produce invalid or ambiguous file
names. `bigwig`, `bam` and `timepoint_minutes` are exempt — they are paths and a
number, not labels.

### `source`

`source` is the biological source of the material: a cell line, primary cells, a
tissue, an organoid. It is deliberately broader than "cell line" — a model
trained across primary material and organoids should not have to pretend those
are cell lines.

SeqNado has no equivalent column, so `source` can only come from the sheet.
Neither can `timepoint_minutes`. These two are the reason a sheet is worth
writing even when a SeqNado project supplies everything else.

## The sheet is an overlay

A sheet does not have to repeat anything SeqNado can already derive. Rows that
give a `project` and a `sample_id` but no `bigwig` are resolved against that
project: the BigWig path, the matching BAM, and the per-sample `condition`, `ip`
and `scaling_group` are all taken from there.

That makes the minimal useful sheet four columns wide:

```text
project,sample_id,source,timepoint_minutes
expA,CTCF_DMSO,K562,0
expA,CTCF_IAA,K562,120
```

Values written in the sheet always win. SeqNado only fills blanks, so the sheet
can correct or extend what the project recorded — override a `condition` that
was mislabelled in the design, or set an `ip` the design left empty, without
touching the SeqNado run.

Resolution rules worth knowing:

- A row needs at least one of `sample_id`, `track_name` or `bigwig`. Missing
  `sample_id` is filled from `track_name`, or from the BigWig's stem.
- With exactly one project configured, `project` may be omitted. With several,
  a row that has no `bigwig` must name its project.
- A sample that the named project does not contain is an error, and the message
  lists the samples that are present.
- Rows that still have no `bigwig` after resolution are an error — nothing
  downstream can proceed without a path.
- Track names must be unique across the whole sheet.

## Reading tracks out of a project

Two commands take SeqNado projects, with different flag names:

```bash
# Generate a workflow config; also writes the derived track sheet.
regulonado config --from-seqnado expA=/data/expA/seqnado_output

# Build a dataset directly.
regulonado dataset intervals.bed genome.fa dataset/ \
  --seqnado-project expA=/data/expA/seqnado_output
```

Both accept `PATH` or `NAME=PATH` and both are repeatable. Without an explicit
name, the project is labelled by the directory containing the output directory —
`2026-08-10_myproj/seqnado_output` becomes `2026-08-10_myproj`.

`regulonado config --from-seqnado` writes `track_sheet.csv` next to the config
(override with `--track-sheet-out`) and points `inputs.track_sheet` at it. That
sheet is a normal CSV: edit it to add `source` and `timepoint_minutes`, then run
the pipeline. A sheet you wrote by hand and named in the config yourself is
never overwritten.

`regulonado dataset` accepts a sheet and projects together: `--track-sheet` plus
`--seqnado-project` resolves the sheet's `sample_id`-only rows against the named
projects. With projects and no sheet, the sheet is built from the projects
directly.

### What comes from where

By default ReguloNado takes unmerged, unstranded `deeptools` / `unscaled`
tracks. Projects can select others through `method` and `scale` in
`inputs.seqnado_projects`; `regulonado config` offers the methods and scales
common to every configured project.

| SeqNado output | Read through | Track sheet |
| --- | --- | --- |
| `bigwigs/{method}/{scale}/{sample}.bigWig` | `bigwig_dataframe()` | `bigwig`, `sample_id`, `method`, `scale` |
| `bigwigs/{method}/merged/{scale}/{sample}.bigWig` | same, `merged=True` | excluded by default |
| `bigwigs/.../{sample}_plus.bigWig`, `_minus` | same, `strand` set | dropped, with a warning |
| `aligned/{sample}.bam` | `bams()`, matched on filename stem | `bam` |
| design `condition` | `metadata_for(sample)` | `condition` |
| design `ip` (SeqNado's internal `antibody`) | `metadata_for(sample)` | `ip` |
| design `group` | `metadata_for(sample)` | `scaling_group` |
| the project's assay | `project.assay` / multiomics keys | `assay` |
| `resources/{spikein}/normalisation_factors.tsv` | `load_normalisation_factors()` | not a column — see [Scaling](#scaling-with-seqnados-factors) |
| the project config's `genome.name` | `project.config` | not a column — used for the consistency check |

Stranded pairs are dropped rather than silently doubling the track count. Pass
`keep_stranded=True` to `TrackSheet.from_seqnado_project` to include them as two
separate tracks.

Multiomics output directories are handled: each assay subdirectory contributes
its own tracks, tagged with its own `assay`.

Nothing here reimplements SeqNado's path conventions. The layout column above
describes what SeqNado produces; ReguloNado asks the project API for it.

## Aggregating several projects

Repeat the flag:

```bash
regulonado config \
  --from-seqnado expA=/data/expA/seqnado_output \
  --from-seqnado expB=/data/expB/seqnado_output
```

Aggregation is not a plain concatenation. Five things happen:

**Track names are namespaced.** They become `<project>__<sample>`, so `input_1`
in `expA` and `input_1` in `expB` stay distinct. A single project keeps bare
sample names — namespacing only earns its keep when names can actually collide.
The separator stays inside the label alphabet, so composed names remain valid.

**Every project must use the same reference genome.** Signal bins only
correspond across projects built on one reference; merging tracks from different
assemblies produces a dataset that is quietly meaningless. A genuine mismatch
raises. A project whose config cannot be read is reported as unknown rather than
as a mismatch, and that case — and only that case — is covered by
`--assume-same-genome`. The flag never overrides a real mismatch.

**Categorical ids are factorised over the union of all projects.** Labels are
collected across the whole sheet and numbered once, so `condition_id=0` means
the same condition in every project. Factorising per project and concatenating
would produce ids that collide on value and disagree on meaning. Tracks with no
label get `-1`, which the training code already treats as absent. The label
ordering is kept in the dataset metadata so ids stay decodable later.

**`scaling_group` defaults to the project name.** Tracks are scaled within their
source batch unless the design gave a `group`, or the sheet says otherwise.

**Duplicate tracks can be collapsed.** Shared inputs and controls are frequently
copied between projects. Setting `build.dedupe_tracks: content` hashes track
contents and keeps one copy of each. `regulonado config` chooses `content`
automatically when more than one project is configured, and `none` for a single
project. On the command line, `regulonado dataset --dedupe-tracks` defaults to
`none`.

## Scaling with SeqNado's factors

`scaling.method: seqnado` reuses the spike-in normalisation factors a SeqNado
run already computed, rather than estimating new ones. They are read from
`resources/<method>/normalisation_factors.tsv` and applied as a correction to
the library-size factors, in exactly the way the `tmm` method applies TMM.

This is single-project only: SeqNado's factors are computed within a project and
are not comparable between projects, so applying one project's factors across an
aggregated dataset would put tracks on different scales without saying so. Use
`tmm` for aggregated datasets — it is derived from the merged dataset itself.

The restriction is enforced twice, in the pydantic config model and again in the
Snakefile, so it holds whether a config is generated by `regulonado config` or
written by hand. With several projects configured, naming exactly one of them in
`scaling.seqnado_project` is the supported escape hatch; leaving it unset is an
error.

See [Normalize track signal](normalization.md#reuse-seqnados-spike-in-factors)
for the config keys and the standalone command.

## Reuse and reimplementation

When SeqNado is installed, ReguloNado calls it. Three things are reimplemented,
each for a specific reason.

| Concern | Approach | Why |
| --- | --- | --- |
| Project contents: tracks, BAMs, per-sample metadata, normalisation factors | **Reused** — `seqnado.open_project` and the project API | The layout is SeqNado's to define; reimplementing its path conventions would rot on the first change |
| Snakemake preset resolution | **Reused** — `seqnado.utils` profile lookup | Same directory, same shortcode scheme, so one implementation is enough |
| Interactive prompts | **Reused** — `seqnado.config.user_input.get_user_input` | `regulonado config` should feel exactly like `seqnado config` |
| Preset installer (`regulonado init`) | **Reimplemented** | SeqNado's is a Typer command body, not a callable function; there is nothing to import |
| Config generator (`regulonado config`) | **Reimplemented** | Same reason, and the generated config is a different shape |
| Genome registry reader | **Reimplemented** | SeqNado's loader calls `sys.exit(1)` when the file is missing, which is unusable from library code. When SeqNado is present its `GenomeConfig` model is still used to validate entries, so the error messages are SeqNado's |
| `build` / `recompress` / `train` config sections | **ReguloNado-specific** | No SeqNado analogue — dataset geometry, Arrow compression and training phases have no counterpart in an NGS pipeline |

The reimplemented pieces are behavioural copies, not forks. Preset resolution
and the prompt primitive both have a fallback that reproduces SeqNado's
behaviour exactly, so a machine without SeqNado gets the same shortcodes and the
same prompt rendering. Neither is a place where the two tools may drift: preset
resolution is a filesystem convention, and the prompt is a string format.

## Not shared

Dataset geometry, Arrow compression and rechunking, training phases, presets and
checkpoints are ReguloNado's alone. The track sheet is the seam: everything
upstream of it is SeqNado's vocabulary, everything downstream is ReguloNado's.
