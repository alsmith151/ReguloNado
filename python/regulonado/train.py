from __future__ import annotations

import dataclasses
import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from datasets import DatasetDict, IterableDataset as HFIterableDataset, load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments, get_scheduler

from regulonado.dataset import build_rc_permutation, make_transform
from regulonado.model import (
    BackboneSpec,
    FreezePolicy,
    HeadedSequenceModel,
    build_backbone_adapter,
    build_condition_shared_track_index,
    build_perturb_head,
)
from regulonado.training.config import TrainerConfig
from regulonado.training.losses import (
    log1p_huber_loss,
    poisson_multinomial_loss,
    scaled_poisson_multinomial_loss,
)


class _LRLogCallback(TrainerCallback):
    """Log per-param-group learning rates so head and backbone LRs are both visible."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        optimizer = kwargs.get("optimizer")
        if optimizer is None or not state.is_world_process_zero:
            return
        logs: dict[str, float] = {}
        names = ["backbone", "head"]
        for i, group in enumerate(optimizer.param_groups):
            label = names[i] if i < len(names) else f"group{i}"
            logs[f"learning_rate/{label}"] = group["lr"]
        if state.log_history:
            state.log_history[-1].update(logs)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dataset_metadata(data_path: Path) -> dict[str, Any]:
    candidates = [data_path / "regulonado_metadata.json", data_path / "track_metadata.json"]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(
        "No dataset metadata JSON found under "
        f"{data_path}; expected one of {[path.name for path in candidates]}"
    )


def _load_dataset_streaming(data_path: Path) -> dict[str, Any]:
    from datasets import load_dataset

    return load_dataset(
        data_path.as_posix(),
        streaming=True,
    )


def _track_records(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = metadata.get("final_track_records") or metadata.get("track_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Dataset metadata does not contain any track records")
    return [dict(record) for record in records]


def _track_array(
    records: Sequence[Mapping[str, Any]],
    *keys: str,
    dtype: np.dtype[Any],
    fill_value: float | int,
) -> np.ndarray:
    values: list[float | int] = []
    for record in records:
        value: Any = None
        for key in keys:
            if key in record and record[key] is not None:
                value = record[key]
                break
        if value is None:
            value = fill_value
        values.append(value)
    return np.asarray(values, dtype=dtype)


def _infer_cardinality(records: Sequence[Mapping[str, Any]], key: str) -> int:
    values = {
        int(record[key])
        for record in records
        if record.get(key) is not None and int(record[key]) >= 0
    }
    return max(values) + 1 if values else 0


def _constant_track_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    field_map = {
        "track_condition_ids": ("condition_id",),
        "track_timepoint_minutes": ("timepoint_minutes",),
        "track_cell_line_ids": ("cell_line_id",),
        "track_assay_type_ids": ("assay_type_id",),
        "track_target_ids": ("target_id",),
    }
    tensors: dict[str, torch.Tensor] = {}
    for out_key, keys in field_map.items():
        if out_key == "track_timepoint_minutes":
            array = _track_array(records, *keys, dtype=np.float32, fill_value=float("nan"))
            if np.all(np.isnan(array)):
                continue
            tensors[out_key] = torch.as_tensor(array, dtype=torch.float32)
        else:
            array = _track_array(records, *keys, dtype=np.int64, fill_value=-1)
            if np.all(array < 0):
                continue
            tensors[out_key] = torch.as_tensor(array, dtype=torch.long)
    return tensors


def _resolve_scale_and_clip(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale_factors = _track_array(records, "scale_factor", dtype=np.float32, fill_value=1.0)
    clip_soft = _track_array(records, "clip_soft", dtype=np.float32, fill_value=348.0)
    clip_hard = _track_array(records, "clip_hard", dtype=np.float32, fill_value=796.0)
    return scale_factors, clip_soft, clip_hard


def _build_collate_fn(
    track_metadata_tensors: Mapping[str, torch.Tensor],
) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    constant_tensors = {key: value.clone() for key, value in track_metadata_tensors.items()}

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        collated = {
            "input_ids": torch.stack(
                [torch.as_tensor(example["input_ids"]) for example in batch]
            ).float(),
            "labels": torch.stack(
                [torch.as_tensor(example["labels"]) for example in batch]
            ).float(),
        }
        for key, value in constant_tensors.items():
            collated[key] = value
        return collated

    return collate


def _build_loss_fn(
    loss_cfg: Mapping[str, Any],
    *,
    scale_factors: np.ndarray,
    clip_hard: np.ndarray,
    labels_already_scaled: bool,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    loss_name = str(loss_cfg.get("name", "poisson_multinomial"))
    poisson_weight = float(loss_cfg.get("poisson_weight", 0.2))
    huber_delta = float(loss_cfg.get("delta", 1.0))

    if loss_name == "scaled_poisson_multinomial":
        if labels_already_scaled:
            return lambda pred, target: scaled_poisson_multinomial_loss(
                pred,
                target,
                poisson_weight=poisson_weight,
            )
        scale_tensor = torch.as_tensor(scale_factors, dtype=torch.float32)
        clip_tensor = torch.as_tensor(clip_hard, dtype=torch.float32)
        return lambda pred, target: scaled_poisson_multinomial_loss(
            pred,
            target,
            scale_factors=scale_tensor,
            clip_hard=clip_tensor,
            poisson_weight=poisson_weight,
        )
    if loss_name == "poisson_multinomial":
        return lambda pred, target: poisson_multinomial_loss(
            pred,
            target,
            poisson_weight=poisson_weight,
        )
    if loss_name == "mse":
        return lambda pred, target: torch.nn.functional.mse_loss(pred, target)
    if loss_name == "log1p_huber":
        return lambda pred, target: log1p_huber_loss(pred, target, delta=huber_delta)
    raise ValueError(f"Unsupported loss name {loss_name!r}")


class TrainerCompatibleModel(torch.nn.Module):
    def __init__(self, model: HeadedSequenceModel, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **metadata: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        model_metadata = {
            key: value
            for key, value in metadata.items()
            if key.startswith("track_")
        }
        logits = self.model(input_ids, **model_metadata)
        outputs: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            outputs["loss"] = self.loss_fn(logits, labels)
        return outputs


def _apply_dataset_transforms(
    dataset_dict: DatasetDict | dict[str, Any],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    data_cfg: Mapping[str, Any],
) -> DatasetDict | dict[str, Any]:
    scale_factors, clip_soft, clip_hard = _resolve_scale_and_clip(records)
    bin_size = int(metadata.get("bin_size", 32))
    shift_max_bp = int(metadata.get("shift_max_bp", 0))
    context_length = int(metadata.get("context_length", data_cfg.get("context_length", 524_288)))
    n_pred_bins = int(metadata.get("n_pred_bins", data_cfg.get("n_pred_bins", 6_144)))
    shift_max_bins = shift_max_bp // bin_size
    rc_perm = (
        build_rc_permutation(list(records)) if bool(data_cfg.get("enable_rc_aug", False)) else None
    )

    train_transform = make_transform(
        scale_factors,
        clip_soft,
        clip_hard,
        apply_scale=bool(data_cfg.get("apply_scale", True)),
        apply_squash=bool(data_cfg.get("apply_squash", True)),
        apply_clip=bool(data_cfg.get("apply_clip", True)),
        enable_rc_aug=bool(data_cfg.get("enable_rc_aug", False)),
        rc_permutation=rc_perm,
        shift_max_bins=shift_max_bins,
        context_length=context_length,
        n_pred_bins=n_pred_bins,
        bin_size=bin_size,
    )
    eval_transform = make_transform(
        scale_factors,
        clip_soft,
        clip_hard,
        apply_scale=bool(data_cfg.get("apply_scale", True)),
        apply_squash=bool(data_cfg.get("apply_squash", True)),
        apply_clip=bool(data_cfg.get("apply_clip", True)),
        enable_rc_aug=False,
        rc_permutation=None,
        shift_max_bins=shift_max_bins,
        context_length=context_length,
        n_pred_bins=n_pred_bins,
        bin_size=bin_size,
        center_crop=True,
    )
    is_streaming = isinstance(dataset_dict.get("train"), HFIterableDataset)
    if is_streaming:
        # IterableDataset.map is lazy — the transform is applied on-the-fly during iteration.
        dataset_dict["train"] = dataset_dict["train"].map(train_transform)
        if "validation" in dataset_dict:
            dataset_dict["validation"] = dataset_dict["validation"].map(eval_transform)
        if "test" in dataset_dict:
            dataset_dict["test"] = dataset_dict["test"].map(eval_transform)
    else:
        dataset_dict["train"].set_transform(train_transform)
        if "validation" in dataset_dict:
            dataset_dict["validation"].set_transform(eval_transform)
        if "test" in dataset_dict:
            dataset_dict["test"].set_transform(eval_transform)
    return dataset_dict


def _make_backbone_spec(
    backbone_cfg: Mapping[str, Any], metadata: Mapping[str, Any]
) -> BackboneSpec:
    target_length = int(metadata.get("n_pred_bins", 0)) or None
    return BackboneSpec(
        backbone_type=str(backbone_cfg.get("name", "borzoi")),
        pretrained_name=backbone_cfg.get("pretrained_name"),
        target_length=backbone_cfg.get("target_length") or target_length,
        config_overrides=dict(backbone_cfg.get("config_overrides") or {}),
    )


def _build_model(
    cfg: Mapping[str, Any],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    adapter_builder: Callable[[BackboneSpec], torch.nn.Module],
) -> HeadedSequenceModel:
    backbone_spec = _make_backbone_spec(cfg["backbone"], metadata)
    backbone = adapter_builder(backbone_spec)
    model_cfg = cfg["model"]
    head_cfg = cfg["head"]
    use_track_metadata = bool(model_cfg.get("use_track_metadata", False))
    shared_track_index = None
    if bool(model_cfg.get("share_condition_base_channels", True)) and use_track_metadata:
        shared_track_index = build_condition_shared_track_index(records)

    head_type = str(head_cfg.get("type", "residual_film"))
    head_kwargs: dict[str, Any] = {
        "in_ch": int(getattr(backbone, "feature_dim")),
        "hidden": int(head_cfg.get("hidden", 512)),
        "n_tracks": len(records),
        "use_track_metadata": use_track_metadata,
        "num_conditions": _infer_cardinality(records, "condition_id") if use_track_metadata else 0,
        "num_cell_lines": _infer_cardinality(records, "cell_line_id") if use_track_metadata else 0,
        "num_assay_types": _infer_cardinality(records, "assay_type_id")
        if use_track_metadata
        else 0,
        "num_targets": _infer_cardinality(records, "target_id") if use_track_metadata else 0,
        "metadata_hidden": int(model_cfg.get("metadata_hidden", 32)),
        "condition_shared_track_index": shared_track_index,
        "dropout": float(head_cfg.get("dropout", 0.0)),
    }
    if head_type == "residual_film":
        head_kwargs["refinement_kernel"] = int(head_cfg.get("refinement_kernel", 9))
    if head_type == "transfer_mlp" and head_cfg.get("mlp_hidden") is not None:
        head_kwargs["mlp_hidden"] = int(head_cfg.get("mlp_hidden"))

    head = build_perturb_head(
        head_type=head_type,
        activation_type=str(model_cfg.get("activation_type", "softplus")),
        **head_kwargs,
    )
    return HeadedSequenceModel(backbone=backbone, head=head)


def _apply_freeze_policy(model: HeadedSequenceModel, trainer_cfg: TrainerConfig) -> None:
    model.apply_freeze_policy(
        FreezePolicy(
            freeze_backbone=trainer_cfg.freeze_backbone,
            unfreeze_backbone_stages_from_output_end=
            trainer_cfg.unfreeze_backbone_stages_from_output_end,
            unfreeze_module_names=trainer_cfg.unfreeze_module_names,
        )
    )


def _build_optimizer(model: HeadedSequenceModel, trainer_cfg: TrainerConfig) -> torch.optim.Optimizer:
    lr = trainer_cfg.learning_rate
    backbone_lr = trainer_cfg.backbone_learning_rate or lr
    head_params = [parameter for parameter in model.head_parameters() if parameter.requires_grad]
    trunk_params = [parameter for parameter in model.trunk_parameters() if parameter.requires_grad]

    param_groups: list[dict[str, Any]] = []
    if trunk_params:
        param_groups.append({"params": trunk_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    if not param_groups:
        param_groups.append(
            {
                "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
                "lr": lr,
            }
        )
    return AdamW(param_groups, weight_decay=trainer_cfg.weight_decay)


def _build_training_arguments(
    output_dir: Path,
    trainer_cfg: TrainerConfig,
    *,
    has_eval: bool,
) -> TrainingArguments:
    save_steps = trainer_cfg.checkpoint_every_n_steps or trainer_cfg.log_every_n_steps
    eval_strategy = "epoch" if has_eval and trainer_cfg.max_steps is None else "steps"
    if not has_eval:
        eval_strategy = "no"

    save_strategy = "steps" if trainer_cfg.checkpoint_every_n_steps else "no"
    logging_strategy = "steps"
    metric_for_best_model = "eval_loss" if has_eval else None
    load_best_model_at_end = has_eval and save_strategy != "no"

    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=os.environ.get("WANDB_NAME") or output_dir.name,
        per_device_train_batch_size=trainer_cfg.batch_size,
        per_device_eval_batch_size=trainer_cfg.resolved_eval_batch_size(),
        dataloader_num_workers=trainer_cfg.num_workers,
        dataloader_persistent_workers=trainer_cfg.num_workers > 0 and trainer_cfg.persistent_workers,
        dataloader_prefetch_factor=trainer_cfg.prefetch_factor if trainer_cfg.num_workers > 0 else None,
        learning_rate=trainer_cfg.learning_rate,
        weight_decay=trainer_cfg.weight_decay,
        num_train_epochs=float(trainer_cfg.max_epochs),
        max_steps=trainer_cfg.max_steps or -1,
        warmup_steps=trainer_cfg.warmup_steps,
        gradient_accumulation_steps=trainer_cfg.gradient_accumulation_steps,
        bf16=trainer_cfg.mixed_precision == "bf16",
        fp16=trainer_cfg.mixed_precision == "fp16",
        logging_strategy=logging_strategy,
        logging_steps=trainer_cfg.log_every_n_steps,
        eval_strategy=eval_strategy,
        eval_steps=(
            trainer_cfg.eval_every_n_steps
            or trainer_cfg.checkpoint_every_n_steps
            or trainer_cfg.log_every_n_steps
        ) if has_eval else None,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_total_limit=2 if save_strategy == "steps" else None,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=trainer_cfg.report_to,
        disable_tqdm=True,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=False if metric_for_best_model is not None else None,
        max_grad_norm=trainer_cfg.gradient_clip_norm or 0.0,
    )


def _build_scheduler_for_trainer(
    optimizer: torch.optim.Optimizer,
    trainer_cfg: TrainerConfig,
    *,
    train_dataset_size: int | None,
) -> torch.optim.lr_scheduler.LRScheduler:
    if train_dataset_size is None:
        if trainer_cfg.max_steps is None:
            raise ValueError(
                "trainer.max_steps must be set when data.streaming=true "
                "(dataset size is not known ahead of time)"
            )
        total_train_steps = trainer_cfg.max_steps
    else:
        steps_per_epoch = max(
            train_dataset_size // max(trainer_cfg.batch_size, 1),
            1,
        )
        total_train_steps = trainer_cfg.max_steps or max(
            steps_per_epoch * trainer_cfg.max_epochs // max(trainer_cfg.gradient_accumulation_steps, 1),
            1,
        )
    return get_scheduler(
        trainer_cfg.scheduler,
        optimizer=optimizer,
        num_warmup_steps=trainer_cfg.warmup_steps,
        num_training_steps=total_train_steps,
    )


def run_training(
    cfg: Mapping[str, Any],
    *,
    adapter_builder: Callable[[BackboneSpec], torch.nn.Module] = build_backbone_adapter,
) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)

    data_path = Path(str(cfg["data"]["path"]))
    streaming = bool(cfg["data"].get("streaming", False))
    dataset_dict = (
        _load_dataset_streaming(data_path) if streaming else load_from_disk(str(data_path))
    )
    metadata = _load_dataset_metadata(data_path)
    records = _track_records(metadata)

    # Only shuffle the training split — eval splits don't need shuffling and the buffer
    # (n_samples × sample_size × num_workers) consumes significant memory on large sequences.
    # At ~45 MB/sample (1507 tracks × 6148 bins) a buffer of 30 uses ~1.4 GB per worker.
    if streaming and "train" in dataset_dict:
        dataset_dict["train"] = dataset_dict["train"].shuffle(buffer_size=30, seed=seed)

    dataset_dict = _apply_dataset_transforms(dataset_dict, metadata, records, cfg["data"])

    model = _build_model(cfg, metadata, records, adapter_builder)
    track_metadata_tensors = (
        _constant_track_metadata(records)
        if bool(cfg["model"].get("use_track_metadata", False))
        else {}
    )
    collate_fn = _build_collate_fn(track_metadata_tensors)

    scale_factors, _, clip_hard = _resolve_scale_and_clip(records)
    labels_already_scaled = bool(
        cfg["data"].get("apply_scale", True)
        or cfg["data"].get("apply_squash", True)
        or cfg["data"].get("apply_clip", True)
    )
    loss_fn = _build_loss_fn(
        cfg["loss"],
        scale_factors=scale_factors,
        clip_hard=clip_hard,
        labels_already_scaled=labels_already_scaled,
    )

    trainer_cfg = TrainerConfig(
        batch_size=int(cfg["trainer"].get("batch_size", 1)),
        eval_batch_size=int(
            cfg["trainer"].get("eval_batch_size") or cfg["trainer"].get("batch_size", 1)
        ),
        num_workers=int(cfg["trainer"].get("num_workers", 4)),
        learning_rate=float(cfg["trainer"].get("learning_rate", 1e-3)),
        backbone_learning_rate=float(
            cfg["trainer"].get(
                "backbone_learning_rate",
                cfg["trainer"].get("learning_rate", 1e-3),
            )
        ),
        weight_decay=float(cfg["trainer"].get("weight_decay", 1e-2)),
        scheduler=str(cfg["trainer"].get("scheduler", "linear")),
        warmup_steps=int(cfg["trainer"].get("warmup_steps", 0)),
        max_epochs=int(cfg["trainer"].get("max_epochs", 1)),
        max_steps=(
            int(cfg["trainer"].get("max_steps"))
            if cfg["trainer"].get("max_steps") is not None
            else None
        ),
        gradient_accumulation_steps=int(cfg["trainer"].get("gradient_accumulation_steps", 1)),
        mixed_precision=str(cfg["trainer"].get("mixed_precision", "bf16")),
        gradient_clip_norm=(
            float(cfg["trainer"].get("gradient_clip_norm", 1.0))
            if cfg["trainer"].get("gradient_clip_norm") is not None
            else None
        ),
        eval_every_n_steps=(
            int(cfg["trainer"].get("eval_every_n_steps"))
            if cfg["trainer"].get("eval_every_n_steps") is not None
            else None
        ),
        checkpoint_every_n_steps=(
            int(cfg["trainer"].get("checkpoint_every_n_steps"))
            if cfg["trainer"].get("checkpoint_every_n_steps") is not None
            else None
        ),
        persistent_workers=bool(cfg["trainer"].get("persistent_workers", True)),
        prefetch_factor=(
            int(cfg["trainer"]["prefetch_factor"])
            if cfg["trainer"].get("prefetch_factor") is not None
            else 2
        ),
        freeze_backbone=bool(cfg["trainer"].get("freeze_backbone", True)),
        unfreeze_backbone_stages_from_output_end=int(
            cfg["trainer"].get(
                "unfreeze_backbone_stages_from_output_end",
                cfg["trainer"].get(
                    "unfreeze_backbone_blocks_from_end",
                    cfg["trainer"].get("unfreeze_last_n_blocks", 0),
                ),
            )
        ),
        unfreeze_module_names=tuple(cfg["trainer"].get("unfreeze_module_names", [])),
        report_to=list(cfg["trainer"].get("report_to", [])),
    )

    # Persistent workers with HF IterableDataset accumulate Arrow file handles and
    # shuffle-buffer state between iterator cycles — workers never restart to clear them.
    # Force non-persistent workers for streaming datasets to prevent this memory leak.
    if streaming and trainer_cfg.persistent_workers:
        import warnings
        warnings.warn(
            "persistent_workers=True is unsafe with streaming datasets (memory leak). "
            "Overriding to persistent_workers=False.",
            stacklevel=2,
        )
        trainer_cfg = dataclasses.replace(trainer_cfg, persistent_workers=False)

    output_dir = Path(str(cfg.get("output_dir") or Path.cwd() / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    _apply_freeze_policy(model, trainer_cfg)
    optimizer = _build_optimizer(model, trainer_cfg)
    train_size = None if streaming else len(dataset_dict["train"])
    scheduler = _build_scheduler_for_trainer(
        optimizer,
        trainer_cfg,
        train_dataset_size=train_size,
    )
    training_args = _build_training_arguments(
        output_dir,
        trainer_cfg,
        has_eval="validation" in dataset_dict,
    )
    trainer_model = TrainerCompatibleModel(model, loss_fn)
    trainer = Trainer(
        model=trainer_model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("validation"),
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler),
        callbacks=[_LRLogCallback()],
    )
    trainer.train()
    train_losses = [
        float(entry["loss"])
        for entry in trainer.state.log_history
        if "loss" in entry and "eval_loss" not in entry
    ]
    eval_losses = [float(entry["eval_loss"]) for entry in trainer.state.log_history if "eval_loss" in entry]
    if not train_losses:
        train_losses = [
            float(entry["train_loss"])
            for entry in trainer.state.log_history
            if "train_loss" in entry
        ]
    trainer.save_model(output_dir)
    trainer.save_state()
    history = {"train/loss": train_losses, "eval/loss": eval_losses}

    summary = {
        "output_dir": str(output_dir),
        "n_tracks": len(records),
        "backbone": cfg["backbone"]["name"],
        "head": cfg["head"]["type"],
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def hydra_entrypoint(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Hydra config did not resolve to a dictionary")
    run_training(config)


def main() -> None:
    hydra_entrypoint()


if __name__ == "__main__":
    main()
