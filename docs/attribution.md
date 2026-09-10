# Attribution: finding the core to design

`regulonado design` optimises the span you hand it, and it optimises *every* base in that span.
For synthesis that is usually the wrong span: a called enhancer is often 500–1000 bp of which only
a couple of hundred actually drive anything. `regulonado attribute` is the pre-selection step that
finds that part.

It runs one saturation-mutagenesis sweep against **one** output track, scores every base by how far
mutating it moves that track, and calls the contiguous sub-span carrying the signal — in practice
the nucleosome-free core. The `core_regions.bed` it writes is designed to be handed straight back
to `regulonado design --candidates`.

```
candidates.bed ──▶ regulonado attribute ──▶ core_regions.bed ──▶ regulonado design ──▶ synthesis
```

## Why the 32 bp bin is not the resolution limit

The obvious worry is that the model predicts in 32 bp bins, so a ~200 bp NFR is only ~6 bins and
there is nothing to localise. That is not how the profile is built.

The attribution profile is indexed by the **mutated position**, not by the output bin. Every point
on it comes from a separate forward pass in which exactly one base was changed, so the profile is
base-resolution whatever `bin_size` is. `--track` and the bin window only decide *which scalar*
gets read back out of the model after each mutation.

What the bin window does control is **contrast**. A single-base effect lands in a couple of bins;
averaging over a whole 600 bp candidate's worth of bins divides that effect by ~19 while dividing
the noise by rather less. That is what `--bin-reduction` is for:

| `--bin-reduction` | Use when |
|---|---|
| `mean` (default) | The safe choice. Stable, unbiased about where in the candidate the signal sits. |
| `topk` | Signal is confined to a small part of the candidate; `--topk-bins` sets how small. |
| `max` | Sharpest contrast, noisiest. Sensitive to a single bin. |

## The score

For each position, every alternative base is substituted in turn and the track is re-read:

```
importance[i] = mean over the 3 alternatives of ( ref_score - alt_score )
```

So **positive importance means mutating here lowers the track** — the base matters. The full
matrix is kept too, as the `delta_A/C/G/T` columns of `attributions.tsv`, using the conventional
ISM sign (`alt - ref`), with the reference base's own column at 0.

Positions whose column is not a clean one-hot (an `N`, or a multi-hot column) have no identity
substitution, so all four bases are scored there rather than three.

## Calling the core

```
smooth ──▶ threshold at a quantile ──▶ bridge small dips ──▶ drop short runs ──▶ rank
```

| Flag | Default | Meaning |
|---|---|---|
| `--smooth-bp` | 25 | Width of the centred rolling mean. |
| `--quantile` | 0.90 | Threshold, as a quantile of this candidate's own smoothed profile. |
| `--merge-gap-bp` | 20 | Dips up to this wide are bridged rather than splitting a core. |
| `--min-width-bp` | 50 | Minimum width of an emitted core. |
| `--min-zscore` | 1.5 | How far above the candidate's own mean a core must sit. |
| `--max-cores-per-candidate` | 1 | One row per candidate by default. |

`--min-zscore` is the flag that matters most for pre-selection. A quantile threshold *always*
fires — on a completely flat profile the top 10% of positions still exist — so without an
effect-size guard every candidate would yield a core and you would synthesise noise. A candidate
that fails it produces **no BED row**, and `summary.tsv` records why:

| `reason` | Meaning |
|---|---|
| `ok` | Core called. |
| `low_zscore` | Profile too flat; nothing stands out. Drop the candidate. |
| `no_segment_meets_min_width` | Signal present but too fragmented. |
| `candidate_shorter_than_min_width` | Candidate emitted whole; nothing to choose. |
| `no_finite_positions` | Nothing was scanned. |

Two behaviours worth knowing:

- A centred box filter ramps a step edge over `smooth_bp`, so thresholding a smoothed profile cuts
  roughly `smooth_bp/2` off each end of a real segment. That erosion is corrected for, so a called
  core matches the real feature width rather than coming back ~`smooth_bp` narrow.
- A feature *narrower* than `--smooth-bp` is smeared rather than eroded, and comes back about the
  width of the smoothing window. Lower `--smooth-bp` to resolve features that fine.

## Fixed-width cores for synthesis

Synthesis usually wants a defined length. `--fix-width 250` re-centres each core and sizes it to
exactly 250 bp, clamped so it can never leave the model's predicted region.

The default anchor is `--anchor centroid`: the importance-weighted centre of mass of the
thresholded segment, which uses the whole segment's shape. `--anchor peak` re-centres on the single
highest base instead, which is sharper but hostage to one noisy position.

## Cost

One candidate costs `3 × span × folds` forward passes at full context. A 600 bp candidate across
3 folds is ~5400 passes. The command prints its projection before it starts.

To cut it down:
- `--stride N` scans every Nth position. Keep `--stride` ≤ `--smooth-bp`, or smoothing windows end
  up containing a single scanned point (the command warns if you cross that line).
- `--positions motifs.bed` restricts the sweep to positions you already care about.

## Run it

```bash
regulonado attribute \
  --candidates enhancer_shortlist.bed \
  --fasta genome.fa \
  --dataset-dir results/dataset \
  --checkpoint results/train/fold_0/peak_finetune \
  --checkpoint results/train/fold_1/peak_finetune \
  --checkpoint results/train/fold_2/peak_finetune \
  --track atac_hl60 \
  --fix-width 250 \
  --out results/attribution/hl60
```

Then design against the cores:

```bash
regulonado design \
  --candidates results/attribution/hl60/core_regions.bed \
  ... --out results/design/hl60
```

## Outputs

| File | Contents |
|---|---|
| `core_regions.bed` | BED6 of called cores, named `{candidate}_core{rank}`. The file you feed to `design`. |
| `cores.tsv` | Per core: width, rank, score, z-score, peak, anchor, source candidate, whether clamped. |
| `summary.tsv` | One row per candidate, including the ones that produced nothing and why. |
| `attributions.tsv` | Per-base profile: `importance`, smoothed importance, `in_core`, and `delta_A/C/G/T`. |
| `attributions.bw` | Per-base BigWig. Load it next to your real ATAC/ChIP tracks — the profile lining up with actual signal is the check that this is working. |
| `run.json` | Resolved parameters for the run. |

Outputs are rewritten after every candidate, so a long run stays inspectable while it is going.

## In the workflow

```yaml
attribution:
  candidates: enhancer_shortlist.bed
  shards: 8
  runs: [fold_0, fold_1, fold_2]
  common:
    stride: 1
    fix_width: 250
    min_zscore: 1.5
  targets:
    - name: hl60
      track: atac_hl60
      settings: {quantile: 0.92}

design:
  # Chained automatically: Snakemake sees this is the attribution stage's output.
  candidates: results/attribution/hl60/core_regions.bed
  ...
```

Candidates are sharded round-robin across jobs and merged back, as the design stage does. Because
shards hold disjoint candidates that can still overlap genomically, the merged BigWig is rebuilt
from the merged per-base table rather than spliced from the per-shard BigWigs.

## A note on `tangermeme`

`tangermeme.saturation_mutagenesis` does exactly this job and is a declared dependency, but it is
not used here: it materialises every mutant up front (`X[i].repeat(n_edits, 1, 1)`), which at
Borzoi's 524 288 bp context is 3.8 GB for a 600 bp candidate and 12.6 GB for a 2 kb one, against
~17 MB for the chunked sweep. It also takes an `nn.Module` and has no notion of a fold ensemble.
Its identity-edit handling is better than the obvious `argmax`, though, and is reproduced here.
