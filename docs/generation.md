# Enhancer generation and gradient design

ReguloNado supports four complementary design methods: greedy ISM, evolutionary
AdaLead, gradient optimisation, and numeric-conditioned masked diffusion.
The diffusion model is a separate learned sequence generator. It generates an
entire enhancer from masks; it does not perform a greedy mutation walk.
Existing activity predictors score completed designs and provide the objective
for the three optimisation methods.

## Train a small conditional generator

Install the training extra (`regulonado[train]`). Prepare a TSV or parquet table
with `chrom`, `sequence`, and one numeric column per activity dimension:

```text
chrom  sequence       HL60_activity  K562_activity
chr1   ACGTACGT...    2.4            0.3
chr8   TGCATGCA...    0.2            1.8
```

The actual text file must be tab separated. Supply real complete sequences,
not the ellipses shown above. Pool regulatory regions across activity profiles;
a final shortlist of a few candidate enhancers is unlikely to train a useful
sequence prior. Values can be measured normalised signals or predictor-derived
scores, but keep one consistent assay/transform per column. Predictor-derived
conditioning is a surrogate, not independent validation. Blank measurements
are missing, not zero. A numeric zero is an observed measurement.

ReguloNado fits a mean and standard deviation per column on training chromosomes
only. Requests use the **original input units**; the checkpoint applies the same
standardisation. It does not infer cross-assay calibration. No cell-type class
labels or arbitrary high/low thresholds are required.

```yaml
# diffusion.yaml
activity_columns: [HL60_activity, K562_activity]
max_length: 512
width: 128
layers: 4
heads: 4
batch_size: 16
epochs: 30
learning_rate: 0.0003
condition_dropout: 0.1
validation_chromosomes: [chr8, chr9]
device: auto
seed: 17
```

```bash
regulonado generate train --table enhancers.tsv --params diffusion.yaml --out runs/mdlm
regulonado generate sample --checkpoint runs/mdlm/best.pt \
  --activities '{"HL60_activity": 2.4, "K562_activity": 0.3}' \
  --length 200 --count 64 --steps 64 --guidance 1.5 --seed 17 --out generated/
```

The default network has four small bidirectional Transformer layers. Training
uses variable masking rates with the MDLM loss and a linear survival schedule;
reverse sampling only unmasks bases and never changes a revealed base. Random
conditioning dropout enables classifier-free guidance. `guidance: 1` uses the
conditional model directly; larger values strengthen conditioning. Choose
requests within the observed activity range initially. Omit a dimension or use
JSON `null` to leave it unspecified. `{}` samples unconditionally.

Training accepts A/C/G/T/N, excludes unknown bases and padding from the loss,
augments training sequences by reverse complementation, and rejects sequences
longer than `max_length`. Both chromosome splits must be nonempty; identical
and reverse-complement sequences crossing splits are rejected. Training writes
`best.pt`, `last.pt`, and `history.json`; checkpoints include architecture,
normalisation, split settings, and input provenance. This version starts a new
training run rather than resuming an optimiser.

`auto` chooses CUDA, then Apple MPS, then CPU. Reduce width, layers, length, or
batch size for constrained laptops. The activity predictors have their own
hardware requirements, independently of the small denoiser.

Standalone sampling writes `sequences.fa`, `sequences.tsv` (including GC
fraction), and `run.json`. Samples are deduplicated; `--min-distance` optionally
filters by Hamming distance. Fewer than the requested count may survive.

## Score whole generated enhancers in existing genomic contexts

Use `regulonado design` with the usual candidate BED, intervals, reference,
checkpoints, and target. Each candidate supplies the enhancer length and a
fixed genomic context for scoring. Diffusion replaces the **entire editable
span**, not just selected seed bases. The best generated enhancer is retained
even when the endogenous baseline scores better; the baseline remains in the
trajectory for comparison.

```yaml
# diffusion-design.yaml; supply usual I/O through CLI or this file
candidates: enhancer_cores.bed
diffusion_checkpoint: runs/mdlm/best.pt
diffusion_activities: {HL60_activity: 2.4, K562_activity: 0.3}
diffusion_samples: 32
diffusion_steps: 64
diffusion_guidance: 1.5
targets:
  - {name: hl60_diffusion, target: HL-60, method: diffusion}
```

`diffusion_activities` names must match the denoiser's training columns. These
names need not match predictor group names: the numeric generation request and
the predictor scoring target are explicitly separate. Requested activity is
not a guarantee. Existing specificity/selective-activation objectives, group
scores, held-out scoring, and output files remain available. Only the selected
final design is evaluated by the held-out checkpoint, which never guides search.
All completed diffusion proposals and their optimisation scores appear in
`trajectory.tsv`.

## Gradient optimisation

```yaml
candidates: enhancer_cores.bed
rounds: 100
fold_mode: resident
gradient_learning_rate: 0.1
gradient_temperature: 1.0
targets:
  - {name: hl60_gradient, target: HL-60, method: gradient}
```

This baseline optimises categorical sequence logits using straight-through
one-hot samples and Adam. It is not a Ledidi reimplementation. Only editable
bases change; predictor weights remain unchanged. The best discretely scored
sequence, including the original seed, is retained. It requires resident folds
and a predictor with gradients through its input; incompatible predictors fail
explicitly. Shared objectives, including edit penalties, still apply; the
Hamming edit penalty is evaluated on the forward discrete sample.

For multi-method workflow configurations, place method-specific settings in
each target's `settings` mapping. Standalone single-target runs resolve those
settings too. Diffusion training is a separate command; the design workflow
consumes an existing denoiser checkpoint.

## Select a diverse experimental panel

All methods expose sequences and optimisation scores in `trajectory.tsv`.
Pool comparable runs and retain unique, separated sequences:

```bash
regulonado generate select \
  --table results/ism/trajectory.tsv --table results/adalead/trajectory.tsv \
  --table results/gradient/trajectory.tsv --table results/diffusion/trajectory.tsv \
  --score energy --count 24 --min-distance 10 --out panel/
```

Lower scores are preferred. Use identical scoring objectives, predictor folds,
and comparable contexts when pooling scores. Selection is greedy diversity
filtering, not a new validation model; panel exports include original table
provenance. Different lengths are treated as separate diversity classes. An
unmodified seed can appear in trajectories; choose controls and method quotas
explicitly when assembling experiments. The selector does not automatically
reserve slots per method. Compare optimisation query costs and wall time
separately from diffusion training cost. Experimental activity is not established
by prediction scores.

## Training inside the existing pipeline

Use `examples/diffusion_training.yaml` with your reference, count dataset and encoder
paths, then run `regulonado pipeline examples/diffusion_training.yaml --preset sg`.
Select just this stage with `regulonado pipeline CONFIG.yaml diffusion --preset sg`.
Use the usual installed GPU profile. The pipeline CLI reuses its `train_phase`
partition/account when there is no explicit `diffusion_train` entry. This uses one GPU; batch size is adjustable.
CUDA throughput and the example batch size have not been benchmarked locally.

To attach this to a pipeline that already builds counts and trains an AlphaGenome
encoder, add the `diffusion:` section, omit `dataset_dir`, and set
`encoder_run: YOUR_RUN_NAME` instead of `training.encoder_checkpoint`. Snakemake
waits for the count dataset and the final encoder phase before diffusion training.
The encoder remains frozen; the conditional denoising decoder learns from corrupted
DNA and numeric activity measurements.

`train_regions: null` uses every eligible training region; an integer caps it while
retaining all matched curated loci. `ccre_classes: []` includes all region classes;
`[dELS, pELS]` restricts the background to enhancers while retaining curated loci.
`validation_regions: null` uses the entire validation split. Recorded chromosome
splits are preserved and rejected if a chromosome occurs in multiple splits.
Without recorded splits, the configured validation/test chromosomes define them.
Test regions are excluded from fitting and normalization. Exact and reverse-complement
sequence matches are removed from training in favour of held-out splits.

The curated BED uses column 8 as numeric specificity; per-assay repeats are averaged
once per original locus. Missing specificity remains unobserved. Change these controls:

- `priority_sampling_weight`: relative sampling weight per curated training view.
- `priority_views`: maximum distinct views per curated training locus.
- `priority_jitter_bp`: bounded positional shifts within the source target interval.
- `priority_reverse_complement`: reverse-complement alternate prepared views.
- `training.reverse_complement_probability`: online augmentation probability for all training DNA.

Views and sampling weights combine: four views at weight four give approximately
16 times the mass of one background view. Duplicate views are removed, so zero
jitter does not multiply identical DNA. Jitter retains the source-region measurement;
use zero when that label cannot reasonably apply to shifted cores. No sequence edits
are made under an unchanged activity label. Validation is centered and unaugmented.

Outputs are under `results_dir/diffusion/`: prepared table and split manifest, resolved
settings, `model/best.pt`, resumable `model/last.pt`, epoch history and training summary.
Retries resume the optimizer and random generators after verifying dataset and model
settings. Increase `epochs` to extend a run; keep its prepared data unchanged.

`train_loss` and `validation_loss` are timestep-weighted masked nucleotide cross-entropy
in nats per valid base (the MDLM objective), rather than whole-sequence activity error.
Training uses stochastic corruption, augmentation and conditioning dropout; validation
uses held-out chromosomes, fixed corruption seeds and full conditioning. Fixed-mask
15/50/90% NLL is nats per masked base; accuracy is a fraction. Long patience and a
plateau learning-rate schedule accommodate small improvements in flat diffusion losses.
Optional W&B logging sends scalar losses, diagnostics, aggregate counts and safe
training settings only. It does not upload sequence tables, coordinates, model files,
code or artifacts. Its run URL is saved in `model/wandb.json`.
