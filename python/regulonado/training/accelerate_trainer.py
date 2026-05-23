from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from regulonado.model import FreezePolicy, HeadedSequenceModel
from regulonado.training.config import TrainerConfig
from regulonado.training.data import stack_batch_tensors

Batch = Mapping[str, Any]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class AccelerateTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        config: TrainerConfig,
        loss_fn: LossFn,
        optimizer: torch.optim.Optimizer | None = None,
        accelerator: Accelerator | None = None,
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        mixed_precision = (
            config.mixed_precision if config.mixed_precision in {"no", "fp16", "bf16"} else "no"
        )
        self.accelerator = accelerator or Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        self.optimizer = optimizer or self._build_optimizer(model)
        self.history: dict[str, list[float]] = {"train/loss": [], "eval/loss": []}

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        lr = self.config.learning_rate
        backbone_lr = self.config.backbone_learning_rate or lr

        if isinstance(model, HeadedSequenceModel):
            head_params = [
                parameter for parameter in model.head_parameters() if parameter.requires_grad
            ]
            trunk_params = [
                parameter for parameter in model.trunk_parameters() if parameter.requires_grad
            ]
            param_groups = []
            if trunk_params:
                param_groups.append({"params": trunk_params, "lr": backbone_lr})
            if head_params:
                param_groups.append({"params": head_params, "lr": lr})
            if param_groups:
                return AdamW(param_groups, weight_decay=self.config.weight_decay)

        return AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

    def _maybe_apply_freeze_policy(self) -> None:
        if isinstance(self.model, HeadedSequenceModel):
            self.model.apply_freeze_policy(
                FreezePolicy(
                    freeze_backbone=self.config.freeze_backbone,
                    unfreeze_backbone_stages_from_output_end=
                    self.config.unfreeze_backbone_stages_from_output_end,
                    unfreeze_module_names=self.config.unfreeze_module_names,
                )
            )

    def _make_dataloader(
        self,
        dataset: Sequence[Batch] | torch.utils.data.Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        collate_fn: Callable[[list[Batch]], Batch] | None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

    def _default_collate(self, batch: list[Batch]) -> dict[str, torch.Tensor]:
        return stack_batch_tensors(batch, tensor_keys=tuple(batch[0].keys()))

    def _prepare_batch(
        self,
        batch: Batch,
        *,
        input_key: str,
        target_key: str,
        metadata_keys: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        model_inputs = batch[input_key]
        targets = batch[target_key]
        if not isinstance(model_inputs, torch.Tensor):
            model_inputs = torch.as_tensor(model_inputs)
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)
        metadata = {
            key: value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            for key, value in batch.items()
            if key in metadata_keys
        }
        return model_inputs, targets, metadata

    def fit(
        self,
        train_dataset: Sequence[Batch] | torch.utils.data.Dataset,
        *,
        eval_dataset: Sequence[Batch] | torch.utils.data.Dataset | None = None,
        input_key: str = "input_ids",
        target_key: str = "targets",
        metadata_keys: Sequence[str] = (),
        collate_fn: Callable[[list[Batch]], Batch] | None = None,
        scheduler_name: str = "linear",
        output_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        self._maybe_apply_freeze_policy()

        collate = collate_fn or self._default_collate
        train_loader = self._make_dataloader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        eval_loader = None
        if eval_dataset is not None:
            eval_loader = self._make_dataloader(
                eval_dataset,
                batch_size=self.config.resolved_eval_batch_size(),
                shuffle=False,
                collate_fn=collate,
            )

        total_train_steps = self.config.max_steps or max(
            len(train_loader) * self.config.max_epochs,
            1,
        )
        scheduler = get_scheduler(
            scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_train_steps,
        )

        if eval_loader is None:
            self.model, self.optimizer, train_loader, scheduler = self.accelerator.prepare(
                self.model,
                self.optimizer,
                train_loader,
                scheduler,
            )
        else:
            self.model, self.optimizer, train_loader, eval_loader, scheduler = (
                self.accelerator.prepare(
                    self.model,
                    self.optimizer,
                    train_loader,
                    eval_loader,
                    scheduler,
                )
            )

        completed_steps = 0
        for _ in range(self.config.max_epochs):
            self.model.train()
            for batch in train_loader:
                with self.accelerator.accumulate(self.model):
                    inputs, targets, metadata = self._prepare_batch(
                        batch,
                        input_key=input_key,
                        target_key=target_key,
                        metadata_keys=metadata_keys,
                    )
                    predictions = self.model(inputs, **metadata)
                    loss = self.loss_fn(predictions, targets)
                    self.accelerator.backward(loss)
                    if (
                        self.config.gradient_clip_norm is not None
                        and self.accelerator.sync_gradients
                    ):
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_norm
                        )
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.history["train/loss"].append(float(loss.detach().item()))
                    completed_steps += 1
                    if (
                        self.config.max_steps is not None
                        and completed_steps >= self.config.max_steps
                    ):
                        break

            if eval_loader is not None:
                self.history["eval/loss"].append(
                    self.evaluate(
                        eval_loader,
                        input_key=input_key,
                        target_key=target_key,
                        metadata_keys=metadata_keys,
                    )
                )

            if self.config.max_steps is not None and completed_steps >= self.config.max_steps:
                break

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self.accelerator.save_state(output_path)
        return self.history

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        *,
        input_key: str,
        target_key: str,
        metadata_keys: Sequence[str],
    ) -> float:
        self.model.eval()
        losses: list[float] = []
        for batch in eval_loader:
            inputs, targets, metadata = self._prepare_batch(
                batch,
                input_key=input_key,
                target_key=target_key,
                metadata_keys=metadata_keys,
            )
            predictions = self.model(inputs, **metadata)
            loss = self.loss_fn(predictions, targets)
            gathered = self.accelerator.gather_for_metrics(loss.detach().reshape(1))
            losses.extend(gathered.cpu().tolist())
        self.model.train()
        return float(sum(losses) / len(losses)) if losses else float("nan")
