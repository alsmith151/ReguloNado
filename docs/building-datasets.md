# Build a dataset

`regulonado build` combines genomic intervals, reference sequence, and signal
tracks into a Hugging Face `DatasetDict`.

## Prepare the inputs

- BED column 4 contains the fold label used for splitting.
- The FASTA has a neighbouring `.fai` index and uses the same contig names as
  the BED and BigWigs.
- BigWigs share the same genome assembly. With `--bigwig-dir`, filenames are
  sorted to define track order; repeated `--bigwig` options preserve the order
  given.

## Build

```bash
regulonado build intervals.bed genome.fa dataset/ \
  --bigwig-dir bigwigs/ \
  --split train:fold0,fold1,fold2 \
  --split validation:fold4 \
  --split test:fold3 \
  --shift-max-bp 128 \
  --n-extract-threads 16 \
  --stage
```

The default chromosome-pass strategy is normally the best choice. `--stage`
copies source files to node-local scratch before reading them, which is useful
on network storage but requires enough scratch space for the FASTA and tracks.

The output includes train, validation, and test Arrow shards plus
`regulonado_metadata.json`. Keep the metadata beside the dataset unless you
intentionally create an enriched copy for training.

## Recompress an existing dataset

Smaller Arrow record batches can improve random reads during training:

```bash
regulonado recompress-dataset dataset/ dataset_rechunked/ \
  --max-batch-size 4 --workers 8
```

The destination must not already exist. The workflow can perform this step
automatically when `recompress.enabled` is true.

Use `regulonado build --help` for geometry, compression, shard sizing,
deduplication, and chromosome filtering options.
