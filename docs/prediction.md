# Generate predictions

Prediction is deliberately separate from the training pipeline. Choose a
checkpoint after reviewing training results, then run `regulonado predict`.

## Selected regions

```bash
regulonado predict path/to/checkpoint-N genome.fa predictions/ \
  --bed regions.bed
```

The command writes one BigWig per selected track for the model's central output
window around each BED row. Prediction windows must not overlap.

## Whole genome

```bash
regulonado predict path/to/checkpoint-N genome.fa predictions/ \
  --whole-genome --chromsizes hg38.chrom.sizes \
  --tracks CTCF_plus,CTCF_minus
```

Without `--chromsizes`, the FASTA index defines the contigs. Whole-genome output
can be large, so select tracks with `--tracks` when possible.

Useful options include `--batch-size`, `--device`, `--tracks`, and
`--inverse-squash`. Run `regulonado predict --help` for the full reference.

Normal checkpoints contain their model geometry and track metadata. Older
checkpoints without `config.json` may also require the source dataset:

```bash
regulonado predict legacy-run/ genome.fa predictions/ \
  --dataset dataset/ --bed regions.bed
```
