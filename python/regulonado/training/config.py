from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProvenanceConfig:
    enabled: bool = True
    save_git_diff: bool = False
    save_git_diff_max_bytes: int = 100_000


@dataclass(slots=True)
class DataConfig:
    """Dataset settings accepted by the training entrypoint."""

    path: str = ""
    metadata_path: str | None = None
    apply_scale: bool = True
    apply_squash: bool = True
    apply_clip: bool = True
    enable_rc_aug: bool = False
    context_length: int = 524288
    n_pred_bins: int = 6144


@dataclass(slots=True)
class ModelConfig:
    """Settings for the Regulonado wrapper around a backbone and head."""

    use_track_metadata: bool = False
    # Metadata field used as the categorical condition identity for FiLM heads.
    # Accepted values: "condition_id", "group" (an opaque freeform tracks.parquet label
    # encoded at training time). Validated at runtime, not by this annotation — omegaconf
    # 2.3.0 cannot structure typing.Literal.
    condition_source: str = "condition_id"
    share_condition_base_channels: bool = False
    metadata_hidden: int = 32
    activation_type: str = "softplus"


@dataclass(slots=True)
class BackboneConfig:
    """Settings shared by the supported backbone adapters."""

    name: str = "borzoi"
    pretrained_name: str | None = None
    target_length: int | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    allow_random_init: bool = False


@dataclass(slots=True)
class HeadConfig:
    """Union of settings used by the supported prediction heads."""

    type: str = "transfer_mlp"
    hidden: int = 512
    dropout: float = 0.0
    refinement_kernel: int | None = None
    mlp_hidden: int | None = None
    output_init: str = "default"
    output_init_samples: int = 256


@dataclass(slots=True)
class LossConfig:
    """Union of settings used by the supported training losses."""

    name: str = "poisson_multinomial"
    delta: float | None = None
    poisson_weight: float | None = None
    weight_range: float | None = None
    weight_exp: float | None = None
    topk_fraction: float | None = None
    topk_weight: float | None = None
    profile_weight: float | None = None
    total_weight: float | None = None
    bin_weight: float | None = None
    bin_mode: str | None = None
    bin_signal_power: float | None = None
    bin_threshold: float | None = None
    topk_bin_weight: float | None = None
    topk_bin_count: int | None = None
    topk_huber_delta: float | None = None
    # Weight of the cross-track specificity correlation term added to any base loss;
    # unset/0 disables it. Region geometry (bins, pseudocount, active fraction) comes from
    # ``trainer.contrast_*`` — shared with the metric that defines the same quantity.
    contrast_weight: float | None = None
    # Kendall et al. homoscedastic uncertainty weighting across tracks for the base loss.
    learn_track_weights: bool | None = None


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
    # Number of full passes over the training split when max_steps is unset. Streaming
    # runs size an epoch from the local Arrow shard headers.
    max_epochs: int = 1
    # Explicit optimizer-update budget; overrides max_epochs when set. Required only when
    # streaming from a source whose split size cannot be counted.
    max_steps: int | None = None
    # Number of forward passes to accumulate before each optimizer step.
    gradient_accumulation_steps: int = 1
    # Mixed precision mode understood by TrainingArguments: no, fp16, or bf16.
    mixed_precision: str = "bf16"
    # Global gradient clipping norm; set to None to disable clipping.
    gradient_clip_norm: float | None = 1.0
    # Frequency of Trainer logging events in optimizer-update steps.
    log_every_n_steps: int = 50
    # Frequency of validation evaluation in optimizer-update steps. When None, falls back
    # to evals_per_epoch, then once per epoch for epoch-driven runs, then
    # checkpoint_every_n_steps, then log_every_n_steps.
    eval_every_n_steps: int | None = None
    # Evaluations per training epoch (4 = every ~25% of an epoch). Checkpoints follow
    # the evaluation cadence unless checkpoint_every_n_steps is set.
    evals_per_epoch: int | None = None
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
    prefetch_factor: int | None = 1
    # Logging backends passed to TrainingArguments report_to. Use ["wandb"] to enable W&B.
    report_to: list[str] = field(default_factory=list)
    wandb_project: str = "regulonado-training"
    wandb_group: str | None = None
    wandb_job_type: str = "training"
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    # Full Trainer resume: restores model, optimizer, scheduler, RNG, and Trainer state.
    resume_from_checkpoint: str | bool | None = None
    # Warm start: loads model weights only and creates a fresh optimizer/scheduler.
    init_weights_from_checkpoint: str | None = None
    # Best-model selection metric passed to TrainingArguments.
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    # Weight of profile-shape Pearson in calibration_shape_objective and contrast_objective.
    calibration_shape_pearson_weight: float = 0.1
    # Number of strongest target bins used by the top-k Pearson metric.
    topk_bins: int = 256
    # Cross-track specificity region geometry: bins per region, per-bin log pseudocount, and
    # the most active fraction of regions (by observed family mean signal) that are scored.
    # Drives both the contrast_* metrics and, when loss.contrast_weight > 0, the loss term —
    # single source of truth so the two compute the same function.
    contrast_region_bins: int = 16
    contrast_pseudocount: float = 0.1
    contrast_active_fraction: float = 0.1
    # Stop training when eval metric has not improved for this many eval calls.
    # None disables early stopping.
    early_stopping_patience: int | None = None
    # Minimum absolute change in metric to count as an improvement.
    early_stopping_threshold: float = 0.0
    # Flush accumulated eval predictions/labels to CPU every N eval batches.
    # None means accumulate entirely on GPU (safe when labels are pre-reduced to [batch, n_tracks]).
    eval_accumulation_steps: int | None = None
    # Subsample the validation split to at most this many examples per eval call.
    # None = use the full validation set. 500–1000 is enough for stable pearson_median.
    max_eval_samples: int | None = None
    # Run a full evaluation pass before the first training step.
    eval_on_start: bool = True
    # Number of eval examples to save prediction plots for. 0 disables plotting.
    num_plot_examples: int = 4
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)

    @property
    def unfreeze_backbone_blocks_from_end(self) -> int:
        return self.unfreeze_backbone_stages_from_output_end

    @property
    def unfreeze_last_n_blocks(self) -> int:
        return self.unfreeze_backbone_stages_from_output_end

    def resolved_eval_batch_size(self) -> int:
        return self.eval_batch_size or self.batch_size
