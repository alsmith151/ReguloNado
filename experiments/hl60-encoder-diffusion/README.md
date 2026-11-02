# Denoising with the trained AlphaGenome encoder

This variant initialises a frozen encoder from
`models_from_ccb/alphagenome-encoder`, retaining the user's trained genomic
features. It adds a conditional feature-pyramid reconstruction decoder with
659,140 trainable parameters; the encoder has 89,957,632 frozen parameters.
The model uses all seven feature resolutions, including single-base features,
instead of attempting to reconstruct bases from the 128 bp pooled output alone.

The decoder receives nucleotide/mask embeddings, positions, time, and numeric
conditioning through per-block feature modulation. Training predicts A/C/G/T at
masked positions using MDLM. Frozen encoder features use float16 on MPS; the
trainable decoder remains float32. Frozen weight standardisation is precomputed
once and tested for numerical equivalence. The activity head remains in the
separate full-context scoring oracle; it is not a reconstruction head.

Run from the repository root:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/regulonado-mpl \
  .venv/bin/python experiments/hl60-encoder-diffusion/run.py --stage prepare
PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/regulonado-mpl \
  .venv/bin/python -u experiments/hl60-encoder-diffusion/run.py --stage train
PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/regulonado-mpl \
  .venv/bin/python experiments/hl60-encoder-diffusion/run.py --stage evaluate
```

`--stage all` runs those stages sequentially. `continue_after_training.py` can
wait for training and run evaluation automatically. Outputs are in
`local/hl60_encoder_diffusion_2026-11-02/`.

## Data and comparison

This first comparison uses exactly the same 8,191 training sequences, 1,024
validation sequences, and 20 attributed test cores as the scratch baseline.
The new encoder-backed model also uses summary conditioning and stratified
random diffusion times, so this is a comparison of complete approaches rather
than an isolated causal test of pretraining alone.

Numeric conditions are HL-60 activity, mean off-target activity, maximum
off-target activity, HL-60 minus mean-off-target contrast (all measured on the
log1p-rate scale), and the supplied per-locus mean specificity score. The first
four are observed for every region; the supplied score remains missing where
unmeasured. No zero labels are fabricated. Curated examples receive sampling
weight four, reducing the strong oversampling used in the original run.

Graded generation varies HL-60 activity, its contrast, and supplied specificity
coherently while holding mean/maximum off-target requests fixed. The original
26-group oracle still evaluates every off-target cell group individually.
The model is trained for up to 120 epochs, with patience 40, a minimum of 40
epochs, and a 0.0001 validation improvement threshold. Fixed-mask cross-entropy
and accuracy use the same held-out diagnostic sequences and masks as baseline.

A frozen functional encoder was not pretrained for masked reconstruction; the
decoder must learn that task. Its short masked inputs differ from the oracle's
original 4096 bp context. The full 4096 bp context is preserved for final design
scoring. The encoder-backed generator is larger in total parameters, even though
its trainable part is small. Stronger denoising does not itself establish stronger
functional specificity.

## Checkpoints and W&B

Format-v2 generator checkpoints contain decoder weights and a SHA256-checked
reference to the original encoder checkpoint. They do not duplicate or overwrite
the original encoder. Keep that checkpoint and its pretrained assets available
when loading the denoiser. Original transformer generator checkpoints remain
loadable.

With explicit approval for aggregate tracking:

```bash
.venv/bin/python experiments/hl60-full/track_wandb.py \
  --out local/hl60_encoder_diffusion_2026-11-02 \
  --name hl60-encoder-diffusion-v2 --follow
```

Only aggregate metrics/settings are uploaded. Sequences, coordinates, data tables,
images, and weights remain local. The two models are pinned in the existing
full-metrics dashboard. Because they may share MPS during execution, elapsed times
are not controlled throughput benchmarks.
