# Full HL-60 design experiment

Run from the repository root, with MPS access:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/regulonado-mpl \
  .venv/bin/python -u experiments/hl60-full/run.py
PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/regulonado-mpl \
  .venv/bin/python experiments/hl60-full/analyse.py
```

Outputs: `local/hl60_full_2026-11-02/`. Use a fresh `--out` for a new experiment.
The full runner supports `--stage prepare|train|design|evaluate` to recover a
stage with the same settings. Do not reuse completion markers after changing
inputs/settings. Paths default to the supplied local oracle, count dataset,
curated BED, and hg38 reference. Network access is unnecessary when backbone
assets are cached.

The experiment uses 8,192 sampled training regions (all 260 curated training
loci plus broader enhancer-like cCREs), 1,024 validation regions (all 24 curated
validation loci), and all 20 curated test loci. Original chromosome splits and
target windows are preserved. Ambiguous, duplicate, reverse-complement duplicate,
and cross-split identical sequences are removed; the manifest records final
counts. Per-assay specificity values are averaged per locus. Unknown specificity
remains missing. Measured replicate counts are pooled with checkpoint size
factors, then transformed to log1p(rate), for the other 26 numeric conditions.

The 200 bp test cores are selected inside each original region and the oracle's
scored span using positive gradient-times-input attribution for HL-60 contrast.
Training cores remain centred on source-region target windows. This difference
and the broader context of the measurements are modelling limitations.

The denoiser has four width-128 Transformer layers (about 0.9 million parameters).
Training uses batch size 64, up to 200 epochs, at least 40 epochs, and early stopping after 40 epochs
without a 0.0001 validation-loss improvement. Rows with measured specificity get
sampling weight 16 to expose that sparse condition. Validation remains unweighted;
check specificity-labelled validation separately. Best checkpoint selection and
normalisation use only training/validation data, never test scores.

All four methods use seed-relative selective activation, penalising off-target
increases with weight 2 and temperature 0.25. Their budgets differ and are recorded:

- ISM: three exhaustive rounds, stride one, one accepted edit per round.
- AdaLead: six rounds, population 32, 256 queries per proposal batch.
- Gradient: 50 steps, learning rate 0.1.
- Diffusion: 32 completed whole-core samples, 64 reverse steps, guidance 1.5.

The design stage reuses one resident oracle and records requested prediction
counts, differentiable evaluations, and wall time in `performance.json`.
Forward calls and backward calls are not equivalent units of compute.

Evaluation rescoring uses actual final FASTA records, including the winning
rather than last diffusion proposal. Each design gets three random controls
matched for edit count and genomic context. `evaluation.tsv` reports target gain,
mean and largest off-target gain, edits, GC, and specificity. A diverse panel
reserves four slots per method when possible and filters reverse-complement and
near-duplicate sequences. These are candidates for review, not validated enhancers.

The graded-generation test uses 32 paired random seeds at each of three numeric
specificity requests, holding the 25 off-target activity requests fixed. The HL-60 activity request
is changed coherently with specificity using a linear calibration fitted only
on the 260 curated training loci. The extra specificity dimension is partly
redundant with the full activity vector; its separate ablation tests incremental
information, not all forms of conditional control. Samples are evaluated
before predictor selection in all 20 test contexts. Paired diagnostics aggregate
by generated sample before bootstrapping to avoid counting repeated contexts as
independent generated sequences. Conditional denoising is also compared with
shuffled/missing conditions, with a separate paired analysis of the 24
specificity-labelled validation loci.

Inspect `REPORT.md`, `results.png`, `paired_diagnostics.json`, and
`conditioning_ablation.json` to determine whether this denoiser learned graded
control. The supplied specificity scale differs from the oracle contrast scale.
A single oracle, even on held-out DNA, does not provide independent evidence of
reporter activity. No additional oracle or experimental readout is available here.

## Weights & Biases

Aggregate metrics and settings can be tracked in a resumable W&B run while the
experiment runs:

```bash
.venv/bin/python experiments/hl60-full/track_wandb.py --follow
```

This requires permission to send the aggregate experiment metrics and training
settings to the configured W&B account. The tracker prints the run URL and saves
it to `wandb_state.json`. It imports completed training history and watches for
incrementally written design outputs, so it can attach to an already running
experiment. It uploads aggregate scalar metrics only: sequences, coordinates,
tables, images, code, machine metadata, and model files remain local. The account
used for this run is `asmith151-university-of-oxford`, project `regulonado-design`.

The expanded tracker logs means/medians for every named cell-group score and
gain, plus mean/median/min/max/standard deviation for objective energy, target
activity, target gain, off-target penalty, edit counts, edit penalty, flank
scores, and derived specificity gaps. It also reports largest/mean off-target
activity gains and the number of boosted off-target groups. Missing independent
holdout-oracle values are not fabricated.

A curated dashboard makes those series visible explicitly:

https://wandb.ai/asmith151-university-of-oxford/regulonado-design?nw=sui6oqq009q

`dashboard_wandb.py` recreates the chart layout using the optional official
`wandb-workspaces` package. It sends chart configuration only. The live tracker
requires only the existing `wandb` dependency.

## Loss definitions and training patience

The longer-patience run uses `epochs: 200`, `early_stopping_patience: 40`,
`early_stopping_min_epochs: 40`, and `early_stopping_min_delta: 0.0001`.
The initial 16-epoch run is preserved in `mdlm_short_patience/`; its validation
minimum is not treated as evidence of convergence.

- **MDLM objective:** sum masked-base negative log probabilities, weight each
  sequence by `1/t` for the linear mask schedule, divide by its valid base count,
  then average regions. Units: nats per valid DNA base. Training uses weighted
  region sampling; validation is uniform, so their curves do not directly give
  a generalisation gap. Neither curve measures enhancer activity or oracle loss.
- **Fixed-mask token NLL:** ordinary cross-entropy averaged over masked bases
  only, without `1/t` weighting, at 15%, 50%, and 90% masks. Units: nats per
  masked base; lower is better. Uniform random base predictions have NLL ln(4).
- **Fixed-mask accuracy:** fraction of masked bases correctly reconstructed at
  each mask rate; higher is better. Uniform random guessing has expected 25%
  accuracy. Fixed train/validation subsets and mask draws make epochs comparable.

Fixed-mask diagnostics use 128 uniformly selected training and 128 validation
examples. W&B uses explicit `training/mdlm_objective_nats_per_base`,
`validation/mdlm_objective_nats_per_base`, and `*/masked_token_nll_*` /
`*/masked_token_accuracy_*` series, with units in dashboard titles. Legacy
`training/loss` aliases remain for compatibility.
