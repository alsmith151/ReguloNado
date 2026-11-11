# Cell-type-conditioned enhancer activity: implementation handoff

Status: proposed implementation, not implemented or GPU-benchmarked. Prepared 2026-11-11.

Updated scope: the user requires two model products, one for track profiles and one for
cell-type-specific enhancer features/activity, with sequence perturbation and saliency for
both. Existing enhancer supervision uses counts over approximately 2-kb regions.

## Objective and agreed scope

Train on the user's own ATAC-seq and histone-mark corpus across different and closely
related cell types, using approximately 2–4 NVIDIA L40 GPUs. Learn how the same DNA
produces different regulatory activity across those conditions, then identify sequence
features associated with those differences. Use learned categorical cell-type embeddings;
an expression encoder and expression conditioning are explicitly outside this project.

Implement an opt-in, parameter-efficient conditioning path inside a pretrained sequence
backbone. First freeze pretrained weights and train FiLM, condition embeddings, and task
heads. Compare against existing unconditioned and head-conditioned baselines. Add LoRA
only as a measured ablation. This is an AG2-inspired adaptation, not a reproduction of
AG2 training or a promise of its accuracy.

Outputs are predictions of accessibility and histone signal and model-derived hypotheses
about enhancer activity. These assays alone do not establish enhancer–gene links or causal
necessity. Do not add expression prediction, enhancer–gene linking, or sequence generation
to the initial implementation. Both profile and enhancer-region models are required;
neither is merely an optional fallback for the other.

## Two model products, shared infrastructure

Maintain separate training presets, checkpoints, metrics, and interpretation targets.
Share metadata, genome splits, conditioning primitives, sequence loading, and perturbation
code. Joint training or transferring a conditioned profile checkpoint into the enhancer
model is a later ablation, not a dependency for delivering either product.

| Product | Profile model | Enhancer activity/features model |
| --- | --- | --- |
| Primary task | Predict position-resolved ATAC and histone signal accurately | Predict region activity and cell differences; expose features underlying specificity |
| Initial backbone | Pretrained full trunk, live conditioned forward | Live AG encoder baseline with early FiLM; compare short-context full trunk if local features underfit |
| Input context | Initially 32–64 kb, increase after benchmarking | Initially 4,096 bp around the existing approximately 2-kb target |
| Supervision | Per-assay genomic profiles and masks | Per-assay, per-cell region counts with replicate-aware noise/exposure |
| Output | Shared assay channels over aligned bins | Shared assay rates for the selected cell; optional auxiliary local profile |
| Core evaluation | Profile shape, amplitude, peaks/domains, cell contrasts | Region activity, differential activity, cell rankings, attribution stability |
| Interpretation target | Explicit assay and fixed genomic output region | Fixed enhancer target's assay rate or matched-cell contrast |

The encoder-only model is a local-sequence hypothesis, not automatically the best enhancer
model. Existing code documents an approximately 1.5-kb receptive field per encoder bin;
pooling bins can combine features but does not reproduce transformer context integration.
Benchmark both accuracy and perturbation throughput.

### Keep target span, input context, and perturbation span independent

Persist all three coordinate ranges in dataset/checkpoint/attribution outputs:

1. **Target span:** the interval over which observations are counted or predictions scored.
2. **Input context:** the longer DNA window supplied to the model.
3. **Perturbation span:** bases changed while the input length and target stay fixed.

Preserve the user's existing 2-kb count targets as the first reproducible baseline. Confirm
whether “2 kb” means 2,000 or 2,048 bp and preserve actual coordinates; do not silently
replace them with the repository's default 1-kb target. Dataset geometry may require
partial-bin overlap weights or boundary handling rather than expanding the scored target.

Proposed ablations after that baseline:

- Hold the 2-kb target fixed and compare supported 4-kb versus 8-kb input contexts. A larger
  context is not equivalent to summing more observed counts.
- Recount nested central 512-bp/1,024-bp/2,048-bp targets only if raw tracks or BAMs exist.
  Do not derive smaller-window labels by dividing existing 2-kb totals. Narrow ATAC targets
  may reduce neighboring-element contamination but increase noise; select by validation
  and replicate reliability.
- Give histone marks assay-specific scored spans where justified. Flanking nucleosomes
  and broad domains need not peak at the ATAC summit; one narrow central window is not
  an automatic improvement for every mark.
- Optionally retain binned enhancer signal as an auxiliary profile target, keeping region
  counts primary. Test whether this improves specificity and localization.

Choose centers once from a cross-cell consensus or fixed reference annotation, then use
identical coordinates across conditions. Per-cell recentering would conflate sequence and
activity differences. Record candidate-selection rules and inspect neighboring peaks or
promoters inside the 2-kb regions.

Scalar count supervision can still support nucleotide-level perturbation scores. It does
not uniquely identify causal bases, and attention-pooling weights are not perturbation
saliency. No unsupervised embedding objective is required initially: learn features through
activity/specificity supervision, then expose representations and motif-associated
attributions for analysis.

## Start here: repository and user inputs

At handoff the workspace has extensive pre-existing modified and untracked files, including
training, label-space, dataset, count, and workflow code. Record `git status --short` and
inspect current diffs before editing. Preserve all unrelated work. No commit or training
job was created while preparing this plan. Re-read applicable repository instructions.

Resolve these inputs from existing configurations and metadata first; ask the user only
for information that cannot be discovered:

- Reference genome/build and FASTA; corpus paths; BigWig versus raw-count availability.
- Histone marks, cell-type vocabulary, number of biological replicates, donor/study/batch
  metadata, controls, and missing assays. Do not assume every cell has every assay.
- Whether existing `condition` or `group` labels actually identify the intended biological
  cell states. Do not silently use study/sample IDs as cell identities.
- Which normalization has already been applied, whether spike-ins exist, and whether
  cross-cell global changes should be preserved.
- Available pretrained checkpoint, provenance, upstream package version, and access terms.
- GPU memory, cluster launch method, interconnect, and an initial pilot runtime budget.

These inputs need not block CPU implementation and synthetic tests. They do block a
meaningful biological experiment. Use explicit placeholders rather than invented data.

## Existing implementation to reuse

Paths below were inspected during planning; recheck against the working tree.

| Area | Existing location | Implication |
| --- | --- | --- |
| Backbone interface and AG adapter | `python/regulonado/model/adapters.py` | `forward_features` currently accepts sequence only. AG supports full trunk and encoder-only features; encoder-only is a local baseline, not a long-range substitute. |
| Model forward/freeze policy | `python/regulonado/model/wrapper.py` | Forward computes unconditional features before passing metadata to the head. Existing frozen BatchNorm handling must remain correct. |
| Head conditioning | `python/regulonado/model/heads.py` | Metadata/FiLM and condition-shared channel support already exist. They do not implement early backbone conditioning. |
| Model serialization | `python/regulonado/model/config.py` | Extend versioned configuration and reconstruction, not just runtime objects. |
| PEFT | `python/regulonado/model/peft_adapters.py` | Existing profile target resolution is Borzoi/Flashzoi-oriented; do not assume it supports AG attention modules. |
| Profile training | `python/regulonado/training/runner.py`, `config.py`, `label_space.py` | Reuse label transformations, masks, optimizer and checkpoint infrastructure. Audit optimizer grouping for new FiLM parameters. |
| Region training | `python/regulonado/training/regions/live.py` and neighboring modules | Required enhancer integration; reuse the same conditioning primitive as the profile path. |
| Metadata | `python/regulonado/tracks_table.py`, `docs/track-table.md` | `tracks.parquet` owns track identity; IDs are derived from labels. Preserve contiguous included-track ordering. |
| Configs/workflow | `python/configs/`, `python/regulonado/workflow/` | Add opt-in presets through existing strict config composition. No second workflow tree. |
| Inference/interpretation | `python/regulonado/inference.py`, `docs/attribution.md` | Thread condition selection through prediction, attribution, and checkpoint loading. |

Read `CONTRIBUTING.md`, `docs/training.md`, `docs/checkpoints.md`,
`docs/normalization.md`, and `docs/building-datasets.md` before implementation.
Use live/profile for profiles and live/region_counts for enhancers. Implement the common
primitive first, then integrate both. Inspect count-head grouping: it currently predicts
one latent rate per group with per-track offsets. Biological rates must retain assay/mark
identity; ATAC and histone tracks must not become interchangeable replicates of one cell
rate. Extend the mapping to condition × assay, with replicate effects centered within each
such group and masks for missing combinations. Preserve existing single-assay models.

## Architecture contract

### AG2-derived metadata policy and local audit (2026-11-11)

AG2 separates **biological condition**, **output channel**, and **replicate observation**.
The paper's Ontology organization section (p. 15) defines a composite tuple of CLO,
UBERON, CL, EFO, and NTR identifiers. Methods pp. 21–22 assign a learned embedding to
each distinct tuple, plus a species embedding. This is not evidence that AG2 independently
sums embeddings for tissue, cell type, disease, and treatment, or uses ontology distances.
Its output heads (p. 25) distinguish modality, target, and selected protocol attributes:
ATAC paired-/single-end channels; histone mark × endedness; single-cell chemistry and
processing attributes. Replicates sharing a condition and output channel are aggregated
in AG2; this project's enhancer model can instead retain replicate-aware likelihoods.

Checked public resources:

- [Choosing a condition](https://www.alphagenomedocs.com/colabs/v2/choosing_a_condition.html)
  documents tissue+cell composites, observed track metadata, and one condition per prediction.
- [Condition construction API](https://www.alphagenomedocs.com/api/generated/alphagenome.models.v2.model_condition.from_curies.html)
  accepts ontology CURIEs and rejects duplicate ontology types.
- Its linked `src/alphagenome/models/v2/model_condition.py` source could not be retrieved
  during this audit. The public research repository inspected still describes AG1; the
  attached preprint defers AG2 model source/weights to final publication. Do not label this
  implementation as copied or verified against released AG2 neural-network source.

Use an explicit field allowlist for each role, shared by both model products:

| Role | Proposed fields | Policy |
| --- | --- | --- |
| Biological condition | cell type/subtype or cell line, tissue/origin, biologically meaningful state | Stable composite label; learned embedding per observed combination |
| Biological state when actually sampled | disease, stimulation/treatment, differentiation stage, dose/time | Include only documented distinctions needed for the scientific task; unknown is not healthy/untreated |
| Species/reference | species, genome build | Species conditioning if multiple species; genome build is coordinate/provenance metadata, not a learned biology shortcut |
| Output channel | assay family, measured target/mark, applicable strand | ATAC differs from H3K27ac, H3K4me1, etc.; ChIP and CUT&RUN protocols remain distinguishable unless harmonization is demonstrated |
| Protocol/channel refinement | endedness, chemistry, bulk/single-cell, processing/count definition | Preserve metadata; split channels or model a technical effect only where observed coverage supports estimation |
| Replicate/nuisance | donor, replicate, batch, study, depth/exposure, control/antibody lot | Retain for noise models, QC and held-out-study evaluation; exclude from biological FiLM by default |
| Validation annotation | enhancer-set membership, set's target cell, confidence, source | Evaluation only; never an input feature or biological condition |
| Provenance | track name, accession, file path, display name | Join/audit fields, not learned condition embeddings |

For a small corpus, begin with one embedding per curated composite biological condition.
Ontology IDs standardize names but do not give a lookup embedding zero-shot biology.
Factorized embeddings (cell + tissue + state) are a later ablation only when there are
enough crossed combinations to distinguish components; do not promise unseen-combination
generalization. Do not embed arbitrary free text or every available metadata column.
If each cell occurs in a unique study, batch and biology cannot be resolved by architecture
alone. Preserve this limitation in conclusions even when replicate offsets are fitted.

Concrete initial keys, to finalize after curation:

```text
condition_key = (species, cell_identity, tissue_origin, biological_state)
channel_key   = (assay_family, target_mark, required_protocol_class, strand_if_applicable)
observation   = (condition_key, channel_key, donor, biological_replicate, track_name)
```

Use distinct explicit unknown/not-applicable values. Do not force unresolved samples into
a specific tissue or fine subtype. Local stable labels are acceptable where ontology terms
are missing; record mappings and confidence rather than inventing CURIEs. Save the mapping
in checkpoints and validate it on reload. Specify exact channel-defining fields in config
rather than inferring channels from whichever columns happen to be present.

Local read-only findings (not necessarily the final corpus): root `tracks.parquet` and
`track_annotations.parquet` each have 132 rows joined one-to-one by track_name. The former
has 57 distinct group labels, the latter 27; 115 matched tracks have differing group text.
Examples include bone-marrow versus cord-blood CD34-positive HSPC collapsed into HSPC,
and bone-marrow naive B cells collapsed into B_cell. Some other changes may be aliases.
Do not infer that every difference is a conflict, or overwrite either file automatically.
Both tables lack explicit tissue/cell-type/mark columns in the inspected schemas.
`assay_class` contains ChIP (76), ATAC (47), CUT&RUN (9); it does not identify histone mark.

Required metadata deliverable before biological training:

1. A row-level reconciliation table retaining both existing group labels, proposed coarse
   evaluation group, fine biological condition, tissue, state, assay target, protocol,
   provenance/evidence, and unresolved fields.
2. Recover mark/target and biological annotations from authoritative source manifests or
   sample sheets. Filename parsing may propose values but needs validation; do not assume
   all ChIP tracks are histone marks rather than TF assays.
3. Cross-tabs of conditions × channels × studies × replicates, missingness, and coverage
   supporting biological versus technical contrasts. No unbounded Cartesian channel expansion.
4. Tests that adding a QC/provenance column cannot change condition or channel IDs; replicate
   IDs cannot split biological conditions; different marks cannot merge; tissue/subtype
   distinctions are preserved unless explicitly collapsed; shuffled metadata rows preserve IDs.

Code audit: `model/heads.py` currently considers source, cell_line, timepoint, replicate,
and experiment_series in automatic condition-shared channel grouping. These mixed roles
are unsuitable as an implicit policy for the new path. Introduce explicit role mappings
without silently changing legacy checkpoint behavior. `group` may remain a coarse reporting
label while a separate curated condition key selects the finer biological embedding.

Implement one backbone first: prefer the existing AlphaGenome PyTorch full-trunk adapter
if a usable checkpoint is available. Confirm installed module names, tensor layouts,
supported input lengths, checkpointing behavior, and which stages actually execute in
`encode(..., resolutions=(128,))`. Do not infer those details from another package version.
Do not install FiLM on decoder branches that never execute and report them as trained.

The proposed interfaces and shapes are design requirements, not existing APIs:

```text
sequence:       [B, 4, L]
condition_ids:  [B]                 # one biological condition per sequence
condition:     Embedding(N, 256) -> [B, 256]
backbone:      forward_features(sequence, condition=condition)
predictions:   [B, A, P]            # A assay/mark outputs; P aligned genomic bins
targets/mask:  [B, A, P]            # mask false for unobserved/invalid entries
```

For enhancer counts, pool over the explicit target bins, respecting boundary overlap,
and return `[B, A]` biological rates. Map those rates to observed replicate tracks with
exposure/offsets and noise parameters. Preserve replicate observations for this likelihood;
aggregation below is an optional profile-pilot strategy, not a counts-model requirement.

Keep assay identity separate from cell identity: ATAC, H3K27ac, etc. use shared output
channels across cells. Establish an explicit mapping from existing track indices to
condition, assay/mark, and replicate. Do not indiscriminately collapse all non-condition
metadata into output identities; donor or sample fields can accidentally defeat sharing.
Preserve replicate provenance even if labels are aggregated for the pilot.

Add residual modulation at selected normalization outputs:

```text
h_conditioned = (1 + delta_gamma(condition)) * h + delta_beta(condition)
```

Each insertion point has independent projections to its channel dimension. Initialize
projection weights and biases to zero; initialize condition embeddings nonzero. Preserve
existing normalization and affine parameters. Do not replace AG1 normalization with AG2
RMSNorm or copy AG2's non-identity initialization into a pretrained checkpoint.

Provide explicit placements: disabled, encoder, transformer, encoder+transformer;
add decoder only after verifying that the chosen output path uses it. Validate all target
modules and report selected names, channel sizes, and trainable counts. Unsupported
backbones or placements must fail clearly, never silently behave as unconditioned models.

Prefer explicit condition propagation through wrappers over mutable global condition
state or hooks. Checkpoint recomputation must receive the same condition as the forward
pass; avoid closures that reuse a later batch's condition. Keep unconditional calls and
old checkpoints backward-compatible when conditioning is disabled.

Freeze pretrained weights, then explicitly enable FiLM, embeddings, and head parameters.
Audit the resulting trainable set and optimizer groups. Do not put the frozen computation
after FiLM under `no_grad()` or detach its activations. Gradients must pass through it.
Keep frozen BatchNorm running statistics fixed. Preserve input gradients for attribution.
With zero-initialized modulation, embedding gradients may be zero on the first update;
test learning over multiple updates rather than incorrectly requiring a nonzero embedding
gradient on the very first backward pass.

Cached unconditional backbone embeddings are valid for baselines only. Reject cached-trunk
training when early conditioning is enabled: representations depend on trainable FiLM and
the selected cell. Likewise, reusing a single backbone forward for many conditions is not
equivalent to independently conditioning the backbone for each cell.

## Data, normalization, and splits

1. Build a corpus audit table: cell, assay/mark, replicate, donor/study, source path, genome,
   normalization units, QC, and observation status. Check for cell/study confounding.
2. Reuse track-table discovery, QC and assembly. Verify label-to-track alignment with a
   small hand-inspected fixture before training. A missing track/bin is not zero signal.
3. Choose a documented replicate strategy. For a pilot, aggregate compatible biological
   replicates after justified normalization, retaining replicate-level data for reliability
   estimates and evaluation. Do not pool incompatible studies silently.
4. Inspect signal units and choose compatible losses. Raw counts can use exposure-aware
   count likelihoods. Normalized or signed enrichment tracks need an appropriate transformed
   regression objective; do not call arbitrary normalized signal raw counts or apply a
   count likelihood to negative values. Reuse current label-space machinery where valid.
5. Validate normalization separately per assay/mark. Promoter anchors are not automatically
   suitable for every histone mark. Avoid transformations that erase genuine global cell
   differences. Fit learned scaling/clipping statistics using training data only.
6. Split genomic blocks/chromosomes before selecting windows, fitting preprocessing, or
   mining differential training examples. Keep overlapping contexts and augmentations within
   their split. Record checkpoint pretraining overlap; fine-tuning-held-out is not necessarily
   unseen during foundation-model pretraining.
7. Sample a mix of shared active regions, differential regions, and genomic background.
   Candidate regions should span all training cell types rather than one favored cell.
   Differential oversampling uses training evidence only; validation/test sampling remains
   fixed and reports both representative and differential subsets.
8. For matched-cell batches, use identical genomic coordinates and augmentations for both
   conditions. Shift/RC labels and masks consistently. Never independently shift paired cells.

Use compatible 128-bp output bins for the initial AG full-trunk pilot unless inspection
justifies another resolution. Score central valid bins and document the input-to-output
coordinate mapping. Short windows can truncate broad histone domains; evaluate window
length separately by mark, not only pooled across assays.

## Losses and evaluation

Start with a masked, assay-balanced signal loss. Normalize each assay's contribution by
its valid observations and expose assay weights. No-observation batches must not create
NaNs, unintended optimizer updates, or inconsistent distributed synchronization.

Add an optional matched-cell contrast loss only after the base model works:

```text
delta_pred = transform(pred_cell_a) - transform(pred_cell_b)
delta_true = transform(label_cell_a) - transform(label_cell_b)
L = L_signal + lambda_contrast * masked_huber(delta_pred, delta_true)
```

Use the same documented assay-specific transform on predictions and targets. Contrast
masks require both observations. Consider replicate reliability when weighting pairs.
Keep `lambda_contrast=0` as the baseline and select any nonzero value on validation only.
Inspect existing group-contrast code/tests before adding overlapping functionality; verify
that it preserves assay identity and paired condition semantics before reusing it.

Report per assay and cell pair, with macro averages rather than only pooled correlations:

- Signal Pearson/Spearman correlation and a scale-sensitive error or deviance metric.
- Correlation/error of between-cell differences at the same genomic loci.
- Differential-direction agreement above a training/validation-defined noise threshold.
- AUPRC for reproducible differential activity where binary labels are defensible, with
  class prevalence and replicate agreement reported.
- Cell ranking at a fixed locus; results stratified by related versus distant cell types.
- Peak/local and broad-domain performance where appropriate; background versus active loci.
- Block-bootstrap uncertainty, avoiding treating adjacent bins as independent samples.

For the profile product, evaluate shape separately from total amplitude. For compatible
nonnegative count profiles, compare a profile-shape objective plus a total-count objective
against the existing profile loss; mask zero-total windows in shape-only metrics. Do not
normalize away amplitude from the overall evaluation. Check peak location/width and broad
domain recovery by assay. A 128-bp head predicts binned tracks, not base-resolution footprints.
If base-resolution ATAC profiles are required, implement and validate an appropriate
higher-resolution head and assay-bias treatment as an explicit additional milestone.

Tn5 sequence preference can contaminate ATAC shape and motif interpretation. Audit count
construction and bias handling; consider a ChromBPNet-style bias-factorized ATAC baseline
or bias component if insertion-level profiles are available. Train any bias estimator on
training regions only and distinguish measured-profile predictions from regulatory-only
attributions. Do not apply an ATAC bias model indiscriminately to histone assays.

Hold out donors/studies as an additional robustness test where replication permits.
A learned lookup table does not provide zero-shot prediction for an unseen cell type.
Reject unknown cell labels at inference; do not map them silently to another embedding.

### Required validation using the user's 2-kb enhancer sets

The user has multiple cell-type-specific enhancer sets intended to validate what the model
learned. Preserve their original approximately 2-kb scored intervals and identities as a
primary benchmark, not just a training-window choice. Obtain a set manifest with genome,
coordinates, assigned cell(s), evidence type, source study, confidence, and how specificity
was defined. Determine whether sets derive from training ATAC/histone tracks, independent
assays, or perturbation experiments; these establish different levels of evidence.

- Keep entire overlapping loci and input contexts together across splits and sets. Define
  separate validation and locked test subsets; never tune metadata choices or attribution
  parameters on the locked test. A locus belonging to several cell sets is multilabel,
  not automatically a false positive for the other cells.
- Predict every held-out set under all relevant observed cell conditions. Produce a matrix
  of enhancer-set identity × predicted cell activity/specificity, with per-assay views and
  related-cell comparisons. Include shared enhancers and matched other-cell enhancers;
  easy inactive background alone does not test specificity.
- Measure within-locus target-versus-other-cell margins, cell ranking/top-k recovery,
  and set discrimination AUPRC with explicit negative definitions and prevalence. Compare
  activity/GC/length/mappability-matched controls where feasible and assay noise limits.
- If coarse enhancer labels map to several fine conditions, predefine the aggregation and
  report fine-condition results as well. Never select the best-matching condition using test
  labels. Do not supply enhancer-set identity or measured specificity as a model input.
- Apply both products to the same intervals: integrate profile predictions over the fixed
  target and compare region-model scores in compatible assay units. Report separately so
  excellent generic peak prediction cannot conceal weak cell-specific enhancer ranking.
- Use differential perturbation scores to identify motifs/regions supporting target-cell
  preference. Evaluate known motif disruptions or independent functional measurements when
  available. Recovery of an enhancer set validates activity discrimination, not automatically
  the causal correctness of a saliency map.

If sets were called from the same training experiments, held-out loci test generalization
to those assay-defined labels, not independent functional validation. Keep a separate result
for independent sets. Preserve the 2-kb benchmark even when training alternative target
widths; those runs must still expose a comparable 2-kb output or be reported as a different
task rather than having their smaller-window scores treated as 2-kb count predictions.

## Staged work packages and acceptance gates

### 0. Inventory and experimental contract

Deliver a corpus manifest, checkpoint/package pin, split manifest, architecture inspection,
and resolved pilot config. Identify unresolved data limitations. Inventory work is complete
when the input units, target mapping, biological conditions, and executable backbone path
are unambiguous. Record deviations from this plan.

### 1. Conditioning primitive and backbone integration

Implement the condition embedding, FiLM modules, explicit forward plumbing, and configuration
serialization for one backbone. Focused CPU tests must establish:

- Identity initialization reproduces the same unconditioned model outputs within tolerance.
- Multiple updates learn a synthetic condition-dependent response to identical sequence.
- FiLM/head weights update; frozen backbone weights and normalization statistics do not.
- Gradients reach early modulation with checkpointing enabled and disabled.
- Conditions do not leak across batches or checkpoint recomputation.
- Channel layouts, broadcasting, sequence lengths, and invalid IDs are checked.

### 2. Dataset and training integration

Implement condition/assay gathering, masks, balanced sampling, optimizer groups, and an
opt-in preset. Add a tiny deterministic synthetic corpus including missing assays and
replicates. Verify correct target alignment, masked losses, same-locus pair augmentation,
and a small overfit run. Existing unconditioned/head-conditioned tests must remain valid.
Use the current strict config schema: the final preset must resolve through the real CLI.

Deliver two runnable presets and synthetic fixtures: profile and enhancer counts. Test
fixed 2-kb targets inside larger inputs, partial-bin boundaries, condition × assay ×
replicate mapping, and missing assays. Do not collapse ATAC and histone rates through
replicate offsets. Cached/head-only training remains a baseline; the required enhancer
product must have a reconstructable live trunk for mutation and gradient analyses.

### 3. Checkpoint and inference integration

Persist vocabulary labels/order, assay ordering, insertion placements, backbone identity
and revision, normalization, coordinate geometry, freeze policy, and format version.
Checkpoint round-trip must preserve predictions, condition selection, and training resume.
Use saved vocabulary mapping at inference even if metadata rows are reordered or subset.
Condition-dependent FiLM cannot generally be merged into one static backbone; do not let
existing LoRA merge-on-save code discard or pretend to merge it. Record base-checkpoint
dependencies if exporting small adapter-only artifacts.

### 4. GPU feasibility pilot

Start with 32,768 bp, microbatch 1, BF16 autocast, checkpointing, one GPU, and only needed
outputs. Try 65,536 bp only after the first configuration works. These are proposed starting
points, not measured fits. Preserve existing FP32 master weights initially; do not casually
cast the entire training model to BF16. Verify precision behavior and finite gradients.

Benchmark warmup plus at least 20 complete optimizer steps, including evaluation and a
checkpoint save. Record allocated/reserved peak GPU memory, step time, examples/s, trainable
and frozen parameters, effective batch size, hardware and package versions. Paired-cell
contrast doubles the sequence-condition work; measure it separately. Keep approximately
10–15% memory headroom and estimate runtime from measured throughput and sample budget.

Only enable DDP once one complete example fits on one GPU. DDP replicates weights and does
not pool device memory. Gradient accumulation changes effective batch size, not the memory
needed for one example. Account for per-rank assay masks in distributed loss normalization.
If memory fails: verify unused branches, shorten input, improve checkpointing, or test the
encoder-only baseline with its limited receptive field explicitly recorded. Defer sequence
parallelism and full 1-Mb training unless the biological results justify that engineering.
Do not launch an open-ended sweep before the hardware pilot and data QC succeed.

Benchmark the enhancer product separately, starting at 4,096-bp input with the original
approximately 2-kb target and live encoder. Measure training and batched mutant inference.
The profile benchmark is not its default resource requirement. Preserve target coordinates
when changing input context and report each product's geometry.

### 5. Controlled biological experiments

Run sequentially, stopping to diagnose failures before expanding the matrix:

Run this comparison separately for the two products. For the enhancer product, the main
local run uses encoder FiLM, followed by a full-trunk encoder+transformer comparison;
transformer conditioning is unavailable in encoder-only mode. Existing region-count
shared/per-group pooling supplies baseline readouts, not evidence of early conditioning.
Select checkpoints separately for profile fidelity and enhancer specificity.

| Run | Purpose |
| --- | --- |
| A: frozen backbone, cell-specific output tracks | Standard multitask baseline |
| B: existing head-only FiLM | Tests whether existing late conditioning is enough |
| C: encoder+transformer FiLM, frozen backbone | Main proposed method |
| D: C plus matched-cell contrast loss | Optional objective ablation |
| E: C plus rank-4/8 LoRA | Optional capacity ablation if C underfits |

Use the same checkpoint, data, output geometry, normalization, split and evaluation regions.
Report both sequence-condition exposure and GPU hours: a multitask baseline predicts many
conditions per sequence, whereas C recomputes the backbone for each condition. Do not claim
fairness from equal optimizer steps alone. Use at least three seeds for finalists if feasible.
Choose one supported LoRA implementation per backbone; inspect actual projection names and
ensure its freeze logic does not disable FiLM. Avoid stacking two adapter frameworks.

Engineering success means reproducible execution, correct conditioning/gradients, and a
measured fit within hardware limits. Scientific success requires better held-out cell-specific
metrics than A/B with uncertainty assessed and acceptable signal accuracy. Set the acceptable
tradeoff on validation before opening the test set. A negative result is a valid deliverable;
do not call a software smoke test evidence of biological improvement.

### 6. Enhancer activity interpretation

For held-out candidate distal regulatory regions, export measured and predicted assay signals
and condition contrasts. Choose enhancer annotations/distance rules with the genome build
recorded. Interpret marks according to their biology; not every histone mark is an enhancer
activation mark. Require ATAC/activity-mark concordance only where those assays are observed.

Add condition-aware attribution targets, including the difference between the same assay's
score in two cells. Keep the reference sequence, output span and reduction identical. Score
motif disruptions/ISM with fixed genomic flanks and compare ref-to-mutant changes across cells.
For paired contrast attribution, compute both conditioned forwards; cached unconditional
features cannot substitute. Use motif-matched controls and multiple perturbations rather
than treating a single deletion/shuffle as causal evidence. Report stability across seeds.
Reuse the existing attribution machinery; implement differential targets only where absent.

Deliver a table with genomic interval, condition pair, assay, observed contrast, predicted
contrast, perturbation effect, replicate reliability where available, and model provenance.
Keep observational activity, model predictions, and perturbation hypotheses distinguishable.

### Required perturbation and saliency interface for both products

Expose one scorer contract with checkpoint, condition, assay, input interval, target
interval, output reduction/transform, and optional comparison condition. Reuse this scorer
for prediction, gradients, and perturbation methods; do not let each invent a target.
For profile predictions, fix the scored bins before mutation rather than reselecting the
largest mutant peak. For counts, score biological rates rather than technical replicate
offsets. Persist the complete scoring definition with every output.

Implement staged analysis to keep the cost manageable:

1. Gradient-based attribution on one-hot sequence for rapid prioritization. If integrated
   gradients are used, record reference sequences, integration steps, and convergence.
   Dinucleotide-preserving references are a sensitivity analysis, not guaranteed biological
   nulls. Verify finite, nonzero input gradients through the frozen backbone.
2. Saturation single-nucleotide mutagenesis on selected cores, with three alternatives per
   reference base and batched forwards. A 2-kb core entails about 6,000 mutant evaluations
   per condition; benchmark and chunk rather than materializing all activations on GPU.
3. Fixed-length sliding-window motif disruption or replacement to test region importance.
   Use several matched/dinucleotide-preserving replacements and controls; report their
   variation. Avoid an all-N replacement as the sole analysis because it is out of distribution.

Define mutation effects consistently as `score(mutant) - score(reference)`. For specificity,
use the difference of those effects between cells:

```text
effect_specificity = [s(mutant, cell_a) - s(reference, cell_a)]
                   - [s(mutant, cell_b) - s(reference, cell_b)]
```

Keep genomic context and all other bases fixed. True deletions change spacing and coordinate
geometry; defer them or implement explicit coordinate remapping rather than calling a
same-length shuffle a deletion. Distinguish saliency at a mutated base from the genomic
location of the output being affected. Per-base ISM of a scalar or binned score is possible
without a base-resolution output head, but spatial output detail remains limited.

Acceptance tests: reference-to-reference effects are zero; perturbations affect only requested
bases; target coordinates stay fixed; RC transforms restore consistent coordinates; both
conditions use the same mutated sequence; checkpoint reload preserves effects; cached-only
checkpoints fail clearly; gradients and ISM are finite and expose a known synthetic motif's
condition-specific effect. Report agreement and disagreement of gradient and ISM rankings,
seed stability, and appropriate sequence/model randomization controls. Attention weights
alone do not satisfy this milestone. Motif hits nominate sequence patterns and often TF
families; they do not establish the identity or occupancy of a specific TF.

## Suggested change boundaries and verification

Prefer small reviewable changes: (1) primitive/backbone tests, (2) data and training,
(3) serialization/inference, (4) experiment config and benchmark, (5) interpretation.
Suggested new module: `python/regulonado/model/conditioning.py`; finalize naming after
inspecting current abstractions. Extend existing config files and runners rather than
creating a standalone training stack. Initial support for one backbone is sufficient.

Relevant existing tests include `test_backbone_adapters.py`, `test_model_transfer.py`,
`test_training_configs.py`, `test_training_runner.py`, `test_losses.py`,
`test_group_contrast_loss.py`, `test_group_contrast_metrics.py`, and `test_attribution.py`.
Add targeted tests for the new contracts above. Run relevant tests and Ruff while iterating;
run the checks required by `CONTRIBUTING.md` before integration. If workflow configuration
changes, include a dry run. Rebuild Rust only if Rust code actually changes.

The next agent's final handoff should contain:

- Implemented features and explicit remaining limitations.
- Exact runnable pilot and inference commands, validated against the actual CLI.
- Config, package/checkpoint pins, split and metadata provenance.
- Tests run and failures or unavailable GPU checks honestly recorded.
- GPU benchmark results or an executable benchmark awaiting cluster access.
- Two usable model presets/checkpoint formats, with profile and region-count evaluation
  reports and condition-aware perturbation examples for each.
- Biological results separated from engineering checks, including unsuccessful ablations.

## Evidence and boundaries of inference

Primary source: user-supplied `/Users/asmith/Desktop/alphagenome-2.pdf`,
*AlphaGenome 2 predicts regulatory activity and variant effects across cell states*.
Relevant sections: early-conditioning ablation (p. 5), condition embeddings and FiLM
(pp. 21–22), condition refinement (pp. 28–29), expression post-training (pp. 29–30),
and the dataset fine-tuning baseline (search for “Fine-tuning baseline”). The paper's
Code availability section says model source and weights will be provided upon final
publication; recheck availability when implementing rather than assuming AG2 is installed.

Reference implementations/documentation checked during the preceding discussion:

- [Official AlphaGenome research repository](https://github.com/google-deepmind/alphagenome_research)
- [Community AlphaGenome PyTorch fine-tuning documentation](https://alphagenome-pytorch.readthedocs.io/en/latest/finetuning/python_api.html)
- [CREsted primary paper](https://www.nature.com/articles/s41592-026-03057-2):
  supports local sequence-to-accessibility modeling and nucleotide-level interpretation
  as a relevant enhancer baseline, not a requirement to replace this repository.
- [ChromBPNet primary preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/):
  motivates separating assay enzyme bias from regulatory sequence effects when modeling
  and interpreting high-resolution chromatin accessibility.

Identity-initialized retrofit FiLM, the short-window hardware pilot, and the optional
matched-cell contrast objective are project proposals. The paper does not demonstrate that
this retrofit into a frozen AG1 model matches AG2. Its early conditioning was learned during
large-scale pretraining. No local GPU memory, throughput, or biological accuracy measurement
has yet been made for this proposal.
