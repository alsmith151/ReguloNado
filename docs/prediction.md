# Generate predictions

Prediction is deliberately separate from the training pipeline. Choose a
checkpoint after reviewing training results, then run `regulonado predict`.

## Inputs and outputs

The command requires:

- **Checkpoint directory**: HF model directory with `config.json` and
  `model.safetensors`, or legacy run root with `resolved_config.json`
  (requires `--dataset` in the latter case)
- **FASTA file**: Reference genome, with an `.fai` index (created by `samtools
  faidx` if missing)
- **Output directory**: Where BigWig files will be written (one per track)

The prediction window size is fixed by the model's training geometry (bin_size,
n_pred_bins). Only the central prediction window is written to ensure windows
never overlap. Run `regulonado predict --help` to see all options.

## Targeted regions

To predict over specific genomic intervals:

```bash
regulonado predict path/to/checkpoint-N genome.fa predictions/ \
  --bed regions.bed
```

The BED file specifies one region per line (chrom, start, end columns). Each
region's centre is predicted with a context window large enough to cover the
full predicted span. Prediction windows must not overlap.

## Whole-genome tiling

To scan the whole genome at regular intervals (e.g., for multi-track visibility):

```bash
regulonado predict path/to/checkpoint-N genome.fa predictions/ \
  --whole-genome --chromsizes hg38.chrom.sizes
```

The `--chromsizes` file (chrom<TAB>size, one per line) selects which
chromosomes to tile and their lengths; it also becomes the BigWig header.
Without `--chromsizes`, the FASTA `.fai` index defines the contigs.

Whole-genome output can be large. Reduce it by selecting specific tracks:

```bash
regulonado predict path/to/checkpoint-N genome.fa predictions/ \
  --whole-genome --chromsizes hg38.chrom.sizes \
  --tracks CTCF_plus,CTCF_minus,H3K27ac
```

Tracks can be specified by name (as above) or by 0-based index. Run the command
without `--tracks` to see the model's full track list.

## Useful options

- `--batch-size` (default 4): Windows per forward pass. Increase for faster
  GPU inference if VRAM is available.
- `--device`: Torch device (default: cuda if available else cpu).
- `--inverse-squash`: Undo the (x+1)^0.75 training squash so values approximate
  raw counts.
- `--rtol` (default 0.01): Relative tolerance for collapsing adjacent equal bins
  in the BigWig; lower values produce more bins.

## Legacy checkpoints

Older checkpoints without `config.json` fall back to rebuilding from
`resolved_config.json` and dataset metadata. Provide the source dataset:

```bash
regulonado predict legacy-run/ genome.fa predictions/ \
  --dataset dataset/ --bed regions.bed
```
