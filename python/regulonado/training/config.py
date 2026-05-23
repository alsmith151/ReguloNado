from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainerConfig:
    """Runtime training knobs shared by the Hydra entrypoint and Trainer wiring.

    These fields intentionally stay close to Hugging Face Trainer concepts so the
    training config reads the same way the underlying framework behaves.
    """

    # Per-device train batch size used by Trainer.
    batch_size: int = 1
    # Per-device eval batch size; defaults to the train batch size when omitted.
    eval_batch_size: int | None = None
    # DataLoader worker processes per Trainer process.
    num_workers: int = 4
    # Learning rate for the trainable head parameters.
    learning_rate: float = 1e-3
    # Optional lower learning rate for backbone parameters.
    backbone_learning_rate: float | None = None
    # AdamW weight decay applied to all optimizer parameter groups.
    weight_decay: float = 1e-2
    # Scheduler name passed through to transformers.get_scheduler.
    scheduler: str = "linear"
    # Scheduler warmup steps counted in optimizer-update steps.
    warmup_steps: int = 0
    # Number of full passes over the training split when max_steps is unset.
    max_epochs: int = 1
    # Explicit optimizer-update budget; overrides max_epochs when set.
    max_steps: int | None = None
    # Number of forward passes to accumulate before each optimizer step.
    gradient_accumulation_steps: int = 1
    # Mixed precision mode understood by TrainingArguments: no, fp16, or bf16.
    mixed_precision: str = "bf16"
    # Global gradient clipping norm; set to None to disable clipping.
    gradient_clip_norm: float | None = 1.0
    # Frequency of Trainer logging events in optimizer-update steps.
    log_every_n_steps: int = 50
    # Frequency of validation evaluation in optimizer-update steps; falls back to
    # checkpoint_every_n_steps, then log_every_n_steps when None.
    eval_every_n_steps: int | None = None
    # Save checkpoints every N optimizer-update steps; disabled when None.
    checkpoint_every_n_steps: int | None = None
    # Freeze all backbone parameters before optional selective unfreezing.
    freeze_backbone: bool = True
    # Re-enable gradients on the final N adapter-reported backbone stages nearest the output.
    unfreeze_backbone_stages_from_output_end: int = 0
    # Explicit backbone module names to unfreeze in addition to the block rule.
    unfreeze_module_names: tuple[str, ...] = field(default_factory=tuple)
    # Keep DataLoader workers alive between iterations instead of respawning each epoch.
    persistent_workers: bool = True
    # Number of batches each worker prefetches ahead; None uses PyTorch's default (2).
    prefetch_factor: int | None = 2
    # Logging backends passed to TrainingArguments report_to. Use ["wandb"] to enable W&B.
    report_to: list[str] = field(default_factory=list)

    @property
    def unfreeze_backbone_blocks_from_end(self) -> int:
        return self.unfreeze_backbone_stages_from_output_end

    @property
    def unfreeze_last_n_blocks(self) -> int:
        return self.unfreeze_backbone_stages_from_output_end

    def resolved_eval_batch_size(self) -> int:
        return self.eval_batch_size or self.batch_size
