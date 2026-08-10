# Normalize track signal

Normalization is optional. Use it when BigWigs contain normalized values such
as RPKM and training should operate on approximate raw counts.

## Calculate scale factors

Infer the original RPKM-to-count factors:

```bash
regulonado normalization original dataset/regulonado_metadata.json \
  --output results/scaling/scale_factors.parquet
```

This command reads library information through the external `bamnado` binary,
which must be on `PATH` (or supplied through `BAMNADO`).

Optionally apply edgeR-style TMM correction using the training split:

```bash
regulonado normalization tmm dataset/regulonado_metadata.json \
  --scale-factors results/scaling/scale_factors.parquet \
  --output results/scaling/scale_factors.tmm.parquet
```

## Create enriched metadata

Write a new metadata file rather than changing the dataset's original record:

```bash
regulonado enrich-metadata \
  dataset/regulonado_metadata.json \
  results/scaling/scale_factors.tmm.parquet \
  --output results/scaling/regulonado_metadata.enriched.json
```

Pass that file to training explicitly:

```bash
regulonado train dataset/ --preset head_only \
  --metadata results/scaling/regulonado_metadata.enriched.json
```

The pipeline connects these paths automatically. It passes the enriched copy
to every phase while leaving the source metadata untouched.
