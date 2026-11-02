"""Diffusion schemas; safe to import without the optional Torch training stack."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiffusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activity_columns: list[str] = Field(min_length=1)
    architecture: Literal["transformer", "alphagenome_encoder"] = "transformer"
    encoder_checkpoint: str | None = None
    encoder_sha256: str | None = None
    encoder_compute_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    stratified_timesteps: bool = False
    max_length: int = Field(default=512, ge=1)
    width: int = Field(default=128, ge=8)
    layers: int = Field(default=4, ge=1)
    heads: int = Field(default=4, ge=1)
    batch_size: int = Field(default=16, ge=1)
    epochs: int = Field(default=30, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    condition_dropout: float = Field(default=0.1, ge=0, le=1)
    condition_sampling_weights: dict[str, float] = Field(default_factory=dict)
    early_stopping_patience: int = Field(default=0, ge=0)
    early_stopping_min_delta: float = Field(default=0.001, ge=0)
    early_stopping_min_epochs: int = Field(default=0, ge=0)
    diagnostic_examples: int = Field(default=0, ge=0)
    num_workers: int = Field(default=0, ge=0)
    sampling_weight_column: str | None = None
    reverse_complement_probability: float = Field(default=0.5, ge=0, le=1)
    heldout_chromosomes: list[str] = Field(default_factory=list)
    lr_schedule: Literal["constant", "plateau"] = "constant"
    warmup_epochs: int = Field(default=0, ge=0)
    scheduler_patience: int = Field(default=8, ge=1)
    min_learning_rate: float = Field(default=1e-6, gt=0)
    validation_chromosomes: list[str] = Field(default_factory=lambda: ["chr8", "chr9"])
    seed: int = Field(default=0, ge=0)
    device: str = "auto"

    @model_validator(mode="after")
    def consistent(self):
        if self.early_stopping_min_epochs > self.epochs:
            raise ValueError("early_stopping_min_epochs cannot exceed epochs")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if len(set(self.activity_columns)) != len(self.activity_columns):
            raise ValueError("activity_columns must be unique")
        if set(self.condition_sampling_weights) - set(self.activity_columns):
            raise ValueError("Sampling weights must name activity columns")
        if any(not np.isfinite(v) or v < 1 for v in self.condition_sampling_weights.values()):
            raise ValueError("Condition sampling weights must be finite and at least one")
        return self
