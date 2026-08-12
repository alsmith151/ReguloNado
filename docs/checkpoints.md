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

Saved checkpoints are Hugging Face model directories and can be loaded in
Python:

```python
from regulonado.model import RegulonadoModel

model = RegulonadoModel.from_pretrained("path/to/checkpoint-N")
```

Keep `config.json` and `model.safetensors` together. Trainer state files are
also required for full resume and for automatic phase chaining.
