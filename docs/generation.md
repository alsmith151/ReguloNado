# Enhancer generation and gradient design

ReguloNado supports four complementary design methods: greedy ISM, evolutionary
AdaLead, gradient optimisation, and numeric-conditioned masked diffusion.
The diffusion model is a separate learned sequence generator. By default it edits
an endogenous enhancer through partial masking. Explicit de novo mode generates
an entire core from masks. Existing activity predictors score the completed
proposals; their gradients do not guide the diffusion steps.

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
  --mode de_novo --length 200 --count 64 --steps 64 --guidance 1.5 --seed 17 --out generated/
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

Standalone sampling now defaults to **edit**, requiring a single-record A/C/G/T
template FASTA; length is inferred. Existing whole-core scripts must explicitly
request `--mode de_novo` (the examples and experiment runners are updated).
The same trained checkpoint supports both modes; no retraining is required.

```bash
regulonado generate sample --checkpoint runs/mdlm/best.pt \
  --activities '{"HL60_activity": 2.4, "K562_activity": 0.3}' \
  --template-fasta endogenous.fa --mask-fraction 0.25 --out edited/ \
  --wandb-project regulonado-design
```

Sampling writes `sequences.fa`, `sequences.tsv` and `run.json`. The selected pool
is deduplicated as before; `--min-distance` optionally filters it. Every generation
event, including repeats and the unchanged baseline, remains in the provenance
records. Standalone sampling has no oracle: selection there only applies the
existing uniqueness/distance filter. Use `regulonado design` for oracle selection.

## Edit enhancers or generate new cores in genomic contexts

Use `regulonado design` with the usual candidate BED, intervals, reference,
checkpoints, and target. Each candidate supplies the enhancer length and a
fixed genomic context for scoring. The default `diffusion_mode: edit` masks 25%
of core positions per proposal and leaves all other bases fixed. Each proposal
gets a reproducible random mask; only masked positions can change. The number
masked is `max(1, floor(fraction * core_length))`, so short cores can have a
larger actual fraction. Actual substitutions may be fewer than masked bases.
The reverse process starts at the actual masked fraction rather than full noise.

Edit mode retains the original enhancer unless a proposal improves the objective.
Set `diffusion_mode: de_novo` to generate an entire core; this mode retains the
best generated sequence even when the endogenous baseline scores better. Both
modes preserve genomic flanks and use the same oracle objective.

```yaml
# diffusion-design.yaml; supply usual I/O through CLI or this file
candidates: enhancer_cores.bed
diffusion_checkpoint: runs/mdlm/best.pt
diffusion_mode: edit
diffusion_mask_fraction: 0.25
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

For a completed encoder from a previous pipeline, set `diffusion.encoder_run_dir`
to its phase directory (for example `results/previous/train/encoder/target`)
and omit `training.encoder_checkpoint`. The workflow depends on that phase's
`trainer_state.json` and resolves the generated checkpoint at runtime, preferring
the final exported model, then the recorded best checkpoint, then the highest
numbered checkpoint directory. This does not schedule another encoder training run.

## Recoverable sequence provenance

Each diffusion invocation writes a unique directory under `provenance/`, containing
`proposals.fa`, `proposals.tsv`, append-only `proposals.jsonl`, `provenance.json`,
and final `selection.fa`/`selection.json`. JSONL snapshots are authoritative after
an interruption: the last snapshot for each proposal ID is its latest state. All
completed generations are written before the first oracle evaluation.

Every generation event has a unique proposal ID; SHA-256 of the uppercase core
provides a stable sequence ID. Repeated DNA shares a sequence ID while preserving
separate events. Records include the full parent and generated sequences, parent
identity, requested conditions, checkpoint SHA-256, random seed and sample index,
mask positions (zero-based within the core), substitutions, and available oracle
scores. The baseline and rejected proposals remain available. Repeated DNA is
scored once within the same context/objective; selection records report requested
proposals, completed proposals and search oracle evaluations. These budgets do
not imply equal compute across diffusion, ISM and AdaLead.

Design W&B runs upload full proposal tables incrementally and a versioned
`sequence-design` artifact with all journal files, including the selected FASTA.
Standalone sampling enables this with `--wandb-project`. Upload errors are logged
and saved in `upload_errors.jsonl`; local results are preserved. Shard merging
copies every provenance directory without changing proposal or sequence IDs. The
workflow tracks a separate completion marker, keeping recovery journals outside
the outputs Snakemake deletes after a failed job.
