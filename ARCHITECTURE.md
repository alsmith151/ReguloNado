# Architecture

ReguloNado is a mixed Rust and Python package. Rust handles the expensive,
dense I/O path; Python owns configuration, orchestration, model training, and
inference.

## Data flow

```text
BED + FASTA + BigWigs
        |
        v
Rust chromosome-pass writer
        |
        v
Hugging Face Arrow DatasetDict + regulonado_metadata.json
        |
        +---- optional scale factors and enriched metadata
        |
        v
Python training runner
        |
        v
self-contained Hugging Face checkpoints
```

The production writer scans a chromosome's tracks once, then slices samples
from the in-memory signal matrix into compressed Arrow shards. This avoids a
random BigWig seek for every sample. A sample-batched writer remains available
for parity tests.

## Responsibilities

- `src/` contains FASTA and BigWig reading, binning, Arrow schema construction,
  and the chromosome-pass writer. PyO3 exposes the compiled extension as
  `regulonado._rs`.
- `python/regulonado/dataset/` assembles inputs, metadata, splits, and staging
  around the Rust writer.
- `python/regulonado/training/` contains model training, losses, data transforms,
  metrics, checkpoints, and provenance.
- `python/regulonado/cli/` is the public Typer interface. It translates friendly
  options into validated training and workflow configuration.
- `python/regulonado/workflow/` is the packaged Snakemake workflow. It expands
  independent runs, orders phases within each run, and delegates computation
  back to the public CLI.
- `python/configs/` holds Hydra defaults and reusable training presets.

Typer, Hydra, and Snakemake have distinct jobs: Typer is the user interface,
Hydra composes one training job, and Snakemake schedules many jobs and their
dependencies.

## Stable boundaries

The saved Arrow schema contains one-hot sequence in `input_ids` and binned
track signal in `labels`. Dataset metadata records geometry and ordered track
information. A training checkpoint includes model configuration and weights,
so normal prediction does not require the original dataset.

Training phases only exchange model checkpoints. A later phase warm-starts
weights with a new optimizer and scheduler. A full resume, by contrast,
restores the optimizer, scheduler, and trainer state for the same phase.

## Known constraints

- Rust errors currently reach Python as `PyRuntimeError` rather than a hierarchy
  of domain-specific exceptions.
- Rayon's global thread pool can only be configured once per Python process.
- The two writer strategies retain some duplicate record-batch assembly code;
  parity tests protect changes to that area.

See [CONTRIBUTING.md](CONTRIBUTING.md) for build commands and where to make
changes.
