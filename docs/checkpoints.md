# Resume and reuse checkpoints

There are two different checkpoint operations.

## Resume an interrupted phase

Full resume restores model weights, optimizer, scheduler, random number state,
and trainer progress. Use it only to continue the same phase with compatible
settings:

```bash
regulonado train dataset/ --preset deep_finetune \
  --output-dir results/train/flashzoi_0/deep_finetune \
  --resume-from-checkpoint results/train/flashzoi_0/deep_finetune/checkpoint-N
```

## Start a new phase from learned weights

A warm start loads weights but creates a fresh optimizer and scheduler. Use it
when changing the phase, learning rate, unfreezing policy, or objective:

```bash
regulonado train dataset/ --preset unfreeze_output \
  --init-weights-from-checkpoint results/train/flashzoi_0/head_only/checkpoint-N \
  --output-dir results/train/flashzoi_0/unfreeze_output
```

The pipeline resolves this relationship automatically from each run's trainer
state. It never warm-starts one named run from another.

## Checkpoint structure

Saved checkpoints are Hugging Face model directories containing:

- **config.json**: Model architecture, backbone settings, and per-track metadata
  (e.g., cell line or timepoint labels for metadata-conditioned heads)
- **model.safetensors**: Model weights with clean keys (no "model." prefix),
  loadable via `from_pretrained`
- **training_args.json**: Training configuration and hyperparameters for
  traceability

Notably, checkpoints do **not** contain optimizer state, scheduler state, or
full trainer progress; these are saved separately only during training for
resuming interrupted runs. Use the checkpoint for inference and warm-starts;
use the trainer's full checkpoint directory for resuming the same training phase.

## Loading from Python

Saved checkpoints can be loaded for prediction or fine-tuning:

```python
from regulonado.model import RegulonadoModel

model = RegulonadoModel.from_pretrained("path/to/checkpoint-N")
model.eval()

# Predict on input_ids (one-hot encoded sequences)
# Input shape: (batch_size, 4, context_length)
with torch.no_grad():
    logits = model(input_ids)  # shape: (batch_size, n_tracks, n_pred_bins)
```

For legacy checkpoints without `config.json`, use `load_model_for_inference` with
the dataset directory:

```python
from regulonado.inference import load_model_for_inference

model = load_model_for_inference(
    "path/to/legacy-run",
    dataset_dir="path/to/dataset"
)
```
