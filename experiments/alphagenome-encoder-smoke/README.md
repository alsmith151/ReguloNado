# HL-60 enhancer design smoke test

This compares ISM, AdaLead, straight-through gradient optimisation, and whole-core
MDLM generation with `models_from_ccb/alphagenome-encoder` as the oracle.
The checkpoint has 26 group outputs, including `HL-60`, and takes 4096 bp windows.
It scores nine 128 bp bins (1152 bp), starting 1408 bp into each window.

Run from the repository root:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/regulonado-mpl \
  .venv/bin/python experiments/alphagenome-encoder-smoke/run.py \
  --fasta /path/to/hg38.fa
```

Outputs are under `local/alphagenome_encoder_hl60_specificity_smoke/`, which is ignored
by Git. Use `--prepare-only` to create regions and configs without model execution;
`--out` selects a different output directory. A rerun overwrites the same outputs.
Device selection defaults to `auto`: CUDA, then Apple MPS, then CPU. On this Mac,
use MPS outside the sandbox; the sandbox can hide the GPU from PyTorch.
`--device mps` requires MPS explicitly; `--device cpu` opts into CPU.
The oracle's pretrained backbone assets must be cached or downloadable.

The default input is
`models_from_ccb/region_counts/dataset/regions.parquet`. The script:

1. Uses `models_from_ccb/regions_more_specific_to_HL60.bed` as the curated
   HL-60 set: 514 per-assay rows collapse to 304 loci, with arithmetic mean
   specificity per locus. Larger scores mean stronger HL-60 preference. Raw
   per-assay observations remain in `priority_regions.tsv` for audit.
   Includes broader enhancer-like cCREs (`dELS`/`pELS`), retains curated peakset
   regions even though they have no cCRE class annotation, and preserves original
   `target_start`/`target_end` coordinates and train/val/test splits.
2. Samples 16 training regions (eight curated and eight broader enhancers),
   four validation regions (two of each), and two curated test regions.
   Only the two test regions are used for the design comparison. The original
   table contains 260 curated training loci, 24 validation loci, and 20 test loci.
3. Places a 200 bp editable core at each target interval's midpoint. This is a
   convenient software test, not an attribution-based identification of its driver.
4. Extracts hg38 cores and scores their full genomic contexts with the oracle.
5. Reads matching rows of `counts.parquet`, pools replicates with checkpoint
   size factors, and applies `log1p` for numeric conditioning across all 26 groups.
   Raw counts are not used directly as activity requests. Measured labels and
   oracle predictions are saved separately. Adds `HL60_specificity` as a 27th
   numeric condition, using the supplied mean score for curated regions and
   missing values for unannotated regions. Missing specificity is never zero.
6. Trains a two-layer, width-32 MDLM for three epochs. The request is the measured
   activity profile of the sampled training region with the highest supplied
   mean HL-60 specificity. Lower, intermediate, and higher requests use observed
   training profiles at specificity quantiles 25/50/75, saved in
   `specificity_profiles.json` and ready-to-run `diffusion-q25.yaml`,
   `diffusion-q50.yaml`, and `diffusion-q75.yaml`. Their conditions remain numeric;
   the quantile names are only convenient run identifiers.
7. Runs very small design budgets: one ISM round (stride 50), one AdaLead round
   (population four, 12 queries), eight gradient steps, and four diffusion samples
   with eight denoising steps. Uses the seed-relative selective-activation
   objective: reward HL-60 gain, penalise positive off-target gains, and do not
   reward off-target suppression alone. These budgets are not intended as a fair performance
   benchmark. Full genomic flanks remain fixed for scoring.

Outputs include `priority_regions.tsv`, `selected_regions.tsv`, `anchors.bed`, `candidates.bed`,
`measured_labels.tsv`, `oracle_labels.tsv`, `diffusion-training.tsv`, denoiser
checkpoints/history, method configs, and each method's usual FASTA/TSV/trajectory
outputs. `summary.tsv` compares seed and final scores; `panel/` contains a small
pooled diversity-filtered panel, with no automatic method quotas.

## Which regions to supply

The region-count dataset supplies measured activity and context windows;
`--candidate-regions` supplies the curated scored HL-60 BED. Both default to your
provided files. The curated BED is deduplicated by original coordinates and
joined to the dataset via `orig_chrom`/`orig_start`/`orig_end`. The script matches
all 304 loci exactly. It retains score variation rather than turning membership
into a categorical target. Supplied specificity and activity contrast computed
by the oracle have different scales and are not silently substituted for one another.
For a larger experiment, use a diverse union of accessible enhancer candidates
across the target and off-target cells, rather than only the strongest HL-60 peaks.
The generator needs variation in the activity profiles it is supposed to control.
Prefer non-promoter enhancer annotations and, for short inserts, attribution cores
or summit-centred intervals. Use hg38 coordinates with the matching reference.

A replacement parquet can use `chrom`, `target_start`, `target_end`, and `split`
(`train`, `val`/`validation`, `test`); `start`/`end` are also required for source
provenance. A sibling `counts.parquet` must be row-aligned and contain the oracle's
track columns to use measured labels. The dataset's numeric scale and track identity
must match this checkpoint. Without sibling counts, the script uses oracle scores
as explicitly recorded surrogate labels. BED input uses its first three columns,
deduplicates intervals, and creates new centred 1000 bp anchors; those are not the
original training windows. BED fallback assigns chr8/chr9 to validation and chr10
to design; parquet input preserves its own splits instead.

The production design command can use a candidate BED of short editable cores
and the original region table separately as `--intervals`. Each core must lie
inside its corresponding 1152 bp scored span; supply the short core, not the full
4096 bp model input. ReguloNado extracts that context from the reference.

This run checks execution, not biological success. Twenty training examples and
three epochs cannot establish a useful generative model. The oracle also scores a
broader genomic span than the 200 bp insert: its activity is not an isolated-core
measurement. Observed source-region specificity also is not a measured score for every newly
generated short core. Preserve independent sequences and evaluate properly trained designs
before experimental selection. Original dataset split labels alone do not prove
that a particular checkpoint never saw a sequence.
