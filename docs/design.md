# Design synthetic enhancers

`regulonado design` mutates a user-supplied enhancer to sharpen its cell-type
specificity, rather than generating sequence from scratch. The trained folds
are fine-tuned Borzoi/flashzoi models that have only ever seen genomic
sequence — sampling truly de-novo sequence pushes the search far off that
training manifold and predictions stop being reliable. Starting from an
endogenous enhancer and mutating it keeps the search on-manifold.

## Why the dataset window, not the candidate, sets the context

Each candidate in `--candidates` must fall inside the *predicted* region of a
window from `--intervals` — the same build-time interval BED used to train
the folds. The module resolves that window, uses its genomic sequence as the
model context, and makes only the candidate's own span editable. Re-centring
on the candidate instead would put it in a context the folds never trained or
were evaluated on, so predictions would not be comparable to the measured
signal for that region. If a candidate matches no window's predicted region,
the default is to raise (`--on-missing error`) rather than silently changing
the context under it; `center` and `skip` are available when that is
acceptable.

## The objective

`SpecificityEnergy` scores a design by contrasting its target cell-type group
against every other group, aggregated over replicate tracks within each
group and reduced across the design folds:

```
energy = offtarget_reduction(other_group_scores) - target_alpha * target_score
```

Lower is better. `--offtarget-reduction logsumexp` (the default) is a
temperature-scaled soft-max: a hard `max` gives a flat, whack-a-mole
landscape where the search fixes whichever group is currently worst and
immediately swaps to another. Cell-type groups come from `--group-by`
(default `source`), resolved from an explicit `--track-sheet`, then
`--dataset-dir`'s `regulonado_metadata.json`, then the id vectors already
baked into a checkpoint's config.

### Seed-relative selective activation

Absolute specificity can look favourable when every cell type is nearly silent, and it can also
improve when a mutation raises every cell type but raises the target slightly more.  The opt-in
`--objective selective-activation` mode instead measures every group against the unedited
endogenous seed:

```
target_gain = target(design) - target(seed)
positive_offtarget_gain = max(offtarget(design) - offtarget(seed) - tolerance, 0)
energy = -target_alpha * target_gain
         + offtarget_boost_weight * smooth_max(positive_offtarget_gain)
```

Off-target decreases are clipped to zero, so silencing other cell types alone is not rewarded.
The normalized smooth maximum is also zero when no other group is boosted.  Consequently the seed
has energy zero and useful improvements have negative energy.  Use
`--offtarget-boost-weight`, `--offtarget-boost-tolerance`, and `--offtarget-temperature` to tune
the penalty.  Absolute group scores and specificity remain in the output as diagnostics.  Energy
values from `specificity` and `selective-activation` runs are not directly comparable.

## Fold hold-out

Optimise against 3 folds and score the result on a 4th, held-out one
(`--holdout-checkpoint`). A design whose energy only improves on the folds it
was optimised against — and not on the held-out fold — is an adversarial
example for that specific set of folds, not a real gain in specificity. The
held-out score is the number that decides whether a design is real.

## Search methods

Both start from the candidate's endogenous sequence — never a random one —
and only ever touch its editable span.

- **`--method ism`** (default): greedy in-silico mutagenesis. Each round
  scores every alternative base at every editable position and accepts the
  best non-conflicting substitutions (`--top-k`), stopping once no
  substitution helps. This is exhaustive, so it is also the expensive one: a
  600 bp candidate is 1800 forward passes per round, times the folds
  optimised against, times the number of candidates. `--ism-stride` and
  `--ism-positions` (restrict to a motif BED) cut that down; the CLI prints
  the projected forward-pass count before starting.
- **`--method adalead`**: an evolutionary search seeded with the endogenous
  sequence plus mutants of it, with real recombination
  (`--population-size` > 1, `--recomb-rate`).

## Run it

```bash
regulonado design \
  --candidates my_enhancers.bed \
  --intervals data/atlas/intervals.bed \
  --checkpoint runs/fold_0 --checkpoint runs/fold_1 --checkpoint runs/fold_2 \
  --holdout-checkpoint runs/fold_3 \
  --fasta genome.fa --dataset-dir data/atlas \
  --target K562 \
  --objective selective-activation \
  --out designs/
```

Candidates are processed independently, so the command is restartable and
safe to shard over a BED split. Outputs land in `--out`:
`designs.fa`/`.tsv`/`.bed`, `trajectory.tsv` (per-round energy),
`edits.tsv` (accepted substitutions), and `run.json` (resolved config,
checkpoints, group assignment, and how each candidate matched a dataset
window).

## Run it at scale via the workflow

Design is an optional `design:` section in the same config
[training](training.md) uses (see `examples/enhancer_design.yaml`). Snakemake
shards the candidate BED, wires up the checkpoints the training runs already
produced, and merges the shards back together:

```bash
regulonado pipeline examples/enhancer_design.yaml --profile <slurm_gpu>
```

Absent `design:`, the workflow behaves exactly as it does without this
module.
