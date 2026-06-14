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
from datasets import DatasetDict, load_from_disk
from datasets import IterableDataset as HFIterableDataset
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    get_scheduler,
)

from regulonado.dataset import build_rc_permutation, make_transform
from regulonado.model import (
    BackboneSpec,
    FreezePolicy,
    HeadedSequenceModel,
    build_backbone_adapter,
    build_condition_shared_track_index,
    build_perturb_head,
)
from regulonado.training.callbacks import (
    _EvalPlotCallback,
    _LRLogCallback,
    _WandbConfigCallback,
)
from regulonado.training.config import (
    ProvenanceConfig,
    TrainerConfig,
    nested_config,
)
from regulonado.training.losses import (
    log1p_huber_loss,
    poisson_multinomial_loss,
    poisson_nll_loss,
    scaled_poisson_multinomial_loss,
    topk_additive_loss,
    topk_reweight_loss,
    transfer_calibration_loss,
)
from regulonado.training.metrics import (
    _make_compute_metrics,
    _make_preprocess_logits_for_metrics,
)
from regulonado.training.provenance import _write_provenance


def _normalise_checkpoint_mode(value: Any) -> str | bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "yes", "1"}:
        return True
    if isinstance(value, str) and value.lower() in {"false", "no", "0", "none", "null"}:
        return None
    return str(value)


def load_model_weights_only(model: torch.nn.Module, checkpoint: str | Path) -> None:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file():
        weight_path = checkpoint_path
    else:
        safetensors_path = checkpoint_path / "model.safetensors"
        bin_path = checkpoint_path / "pytorch_model.bin"
        if safetensors_path.exists():
            weight_path = safetensors_path
        elif bin_path.exists():
            weight_path = bin_path
        else:
            raise FileNotFoundError(
                f"No model weights found in {checkpoint_path}; expected model.safetensors "
                "or pytorch_model.bin"
            )

    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(weight_path), device="cpu")
    else:
        state_dict = torch.load(weight_path, map_location="cpu")
    # HF Trainer saves the TrainerCompatibleModel wrapper, so keys are prefixed with "model.".
    # Strip that prefix if present so the state dict loads into a bare HeadedSequenceModel.
    first_keys = list(state_dict)[:5]
    if all(k.startswith("model.") for k in first_keys):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # num_batches_tracked are non-trainable BatchNorm counters; safe to ignore.
    unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys when warm-starting: {unexpected[:10]}")
    if missing:
        raise RuntimeError(f"Missing checkpoint keys when warm-starting: {missing[:10]}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_metadata(data_path: Path) -> dict[str, Any]:
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


def track_records(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
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


def infer_cardinality(records: Sequence[Mapping[str, Any]], key: str) -> int:
    values = {
        int(record[key])
        for record in records
        if record.get(key) is not None and int(record[key]) >= 0
    }
    return max(values) + 1 if values else 0


def constant_track_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
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


def resolve_scale_and_clip(
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
    if loss_name == "poisson_nll":
        return lambda pred, target: poisson_nll_loss(pred, target)
    if loss_name == "transfer_calibration":
        profile_weight = float(loss_cfg.get("profile_weight", 1.0))
        total_weight = float(loss_cfg.get("total_weight", 0.5))
        bin_weight = float(loss_cfg.get("bin_weight", 0.1))
        topk_bin_weight = float(loss_cfg.get("topk_bin_weight", 0.0))
        topk_bin_count = int(loss_cfg.get("topk_bin_count", 0))
        topk_huber_delta = float(loss_cfg.get("topk_huber_delta", 1.0))
        return lambda pred, target: transfer_calibration_loss(
            pred,
            target,
            profile_weight=profile_weight,
            total_weight=total_weight,
            bin_weight=bin_weight,
            topk_bin_weight=topk_bin_weight,
            topk_bin_count=topk_bin_count,
            topk_huber_delta=topk_huber_delta,
        )
    topk_fraction = float(loss_cfg.get("topk_fraction", 0.04))
    topk_weight = float(loss_cfg.get("topk_weight", 1.0))
    if loss_name == "topk_additive":
        return lambda pred, target: topk_additive_loss(
            pred,
            target,
            topk_fraction=topk_fraction,
            topk_weight=topk_weight,
            poisson_weight=poisson_weight,
        )
    if loss_name == "topk_reweight":
        return lambda pred, target: topk_reweight_loss(
            pred,
            target,
            topk_fraction=topk_fraction,
            topk_weight=topk_weight,
            poisson_weight=poisson_weight,
        )
    raise ValueError(f"Unsupported loss name {loss_name!r}")


class TrainerCompatibleModel(torch.nn.Module):
    def __init__(
        self,
        model: HeadedSequenceModel,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
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
            # Labels may be loaded as [B, L, T] by the HF datasets library; align to [B, T, L].
            aligned = labels if labels.shape[-2:] == logits.shape[-2:] else labels.transpose(-2, -1)
            outputs["loss"] = self.loss_fn(logits, aligned)
        return outputs


def _apply_dataset_transforms(
    dataset_dict: DatasetDict | dict[str, Any],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    data_cfg: Mapping[str, Any],
) -> DatasetDict | dict[str, Any]:
    scale_factors, clip_soft, clip_hard = resolve_scale_and_clip(records)
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


def build_model(
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
        "num_conditions": infer_cardinality(records, "condition_id") if use_track_metadata else 0,
        "num_cell_lines": infer_cardinality(records, "cell_line_id") if use_track_metadata else 0,
        "num_assay_types": infer_cardinality(records, "assay_type_id")
        if use_track_metadata
        else 0,
        "num_targets": infer_cardinality(records, "target_id") if use_track_metadata else 0,
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


def _build_optimizer(
    model: HeadedSequenceModel, trainer_cfg: TrainerConfig
) -> torch.optim.Optimizer:
    lr = trainer_cfg.learning_rate
    backbone_lr = trainer_cfg.backbone_learning_rate or lr

    head_ids = {id(parameter) for parameter in model.head_parameters()}
    trunk_ids = {id(parameter) for parameter in model.trunk_parameters()}
    no_decay_terms = ("bias", "norm", "ln", "embedding", "scale", "shift")
    grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {
        ("backbone", True): [],
        ("backbone", False): [],
        ("head", True): [],
        ("head", False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in trunk_ids:
            family = "backbone"
        elif id(parameter) in head_ids:
            family = "head"
        else:
            family = "head"
        use_decay = parameter.ndim > 1 and not any(term in name.lower() for term in no_decay_terms)
        grouped[(family, use_decay)].append(parameter)

    param_groups: list[dict[str, Any]] = []
    for family, group_lr in (("backbone", backbone_lr), ("head", lr)):
        decay_params = grouped[(family, True)]
        no_decay_params = grouped[(family, False)]
        if decay_params:
            param_groups.append(
                {"params": decay_params, "lr": group_lr, "weight_decay": trainer_cfg.weight_decay}
            )
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "lr": group_lr, "weight_decay": 0.0})
    if not param_groups:
        param_groups.append(
            {
                "params": [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ],
                "lr": lr,
                "weight_decay": trainer_cfg.weight_decay,
            }
        )
    return AdamW(param_groups)


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
    metric_for_best_model = trainer_cfg.metric_for_best_model if has_eval else None
    load_best_model_at_end = has_eval and save_strategy != "no"

    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=os.environ.get("WANDB_NAME") or output_dir.name,
        per_device_train_batch_size=trainer_cfg.batch_size,
        per_device_eval_batch_size=trainer_cfg.resolved_eval_batch_size(),
        dataloader_num_workers=trainer_cfg.num_workers,
        dataloader_persistent_workers=(
            trainer_cfg.num_workers > 0 and trainer_cfg.persistent_workers
        ),
        dataloader_prefetch_factor=(
            trainer_cfg.prefetch_factor if trainer_cfg.num_workers > 0 else None
        ),
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        eval_accumulation_steps=trainer_cfg.eval_accumulation_steps,
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
        greater_is_better=(
            trainer_cfg.greater_is_better if metric_for_best_model is not None else None
        ),
        max_grad_norm=trainer_cfg.gradient_clip_norm or 0.0,
        eval_on_start=has_eval and trainer_cfg.eval_on_start,
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
            steps_per_epoch
            * trainer_cfg.max_epochs
            // max(trainer_cfg.gradient_accumulation_steps, 1),
            1,
        )
    return get_scheduler(
        trainer_cfg.scheduler,
        optimizer=optimizer,
        num_warmup_steps=trainer_cfg.warmup_steps,
        num_training_steps=total_train_steps,
    )


class RegulonadoTrainer(Trainer):
    """Trainer subclass that applies preprocess_logits_for_metrics inside prediction_step
    (with raw labels) and then reduces labels to [B, T] before accumulation to avoid OOM.

    HF's evaluation_loop calls preprocess *after* prediction_step with whatever labels
    prediction_step returned.  By handling preprocess here and passing None to the base
    class, we ensure preprocess always sees full [B, T, L] labels while only [B, T]
    reduced labels are stored across the eval set.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._metrics_preprocess: Callable | None = kwargs.pop("preprocess_logits_for_metrics", None)
        super().__init__(*args, **kwargs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        if self._metrics_preprocess is not None and logits is not None and labels is not None:
            logits = self._metrics_preprocess(logits, labels)
            # Reduce labels to [B, T] to avoid storing full [B, T, L] across the eval set.
            if labels.ndim == 3:
                bin_dim = -1 if labels.shape[-1] > labels.shape[-2] else -2
                labels = labels.sum(dim=bin_dim)
        return loss, logits, labels


def _estimate_shuffle_buffer(
    data_cfg: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> int:
    ram_gb = float(data_cfg.get("shuffle_buffer_ram_gb", 4.0))
    context_length = int(metadata.get("context_length", data_cfg.get("context_length", 524_288)))
    n_pred_bins = int(metadata.get("n_pred_bins", data_cfg.get("n_pred_bins", 6_144)))
    n_tracks = int(metadata.get("n_final_tracks") or metadata.get("n_tracks") or 1)
    bytes_per_sample = (context_length * 4 + n_tracks * n_pred_bins) * 4  # float32
    return max(10, int(ram_gb * 1e9 / bytes_per_sample))


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
    metadata = load_dataset_metadata(data_path)
    records = track_records(metadata)

    if streaming and "train" in dataset_dict:
        shuffle_buffer = _estimate_shuffle_buffer(cfg["data"], metadata)
        dataset_dict["train"] = dataset_dict["train"].shuffle(buffer_size=shuffle_buffer, seed=seed)

    max_eval_samples = (
        int(cfg["trainer"]["max_eval_samples"])
        if cfg["trainer"].get("max_eval_samples") is not None
        else None
    )
    if max_eval_samples is not None and "validation" in dataset_dict:
        val = dataset_dict["validation"]
        if isinstance(val, HFIterableDataset):
            # Stride-filter is memory-free and gives uniform coverage across all chromosomes,
            # which is statistically equivalent for an unbiased Pearson estimate.
            n_val = (
                val.info.splits["validation"].num_examples
                if val.info and val.info.splits and "validation" in val.info.splits
                else None
            )
            if n_val and n_val > max_eval_samples:
                stride = n_val // max_eval_samples
                dataset_dict["validation"] = val.filter(
                    lambda _, idx: idx % stride == 0, with_indices=True
                ).take(max_eval_samples)
            else:
                dataset_dict["validation"] = val.take(max_eval_samples)
        else:
            n_val = len(val)
            if n_val > max_eval_samples:
                rng = np.random.default_rng(seed)
                indices = sorted(rng.choice(n_val, size=max_eval_samples, replace=False).tolist())
                dataset_dict["validation"] = val.select(indices)
            else:
                dataset_dict["validation"] = val

    dataset_dict = _apply_dataset_transforms(dataset_dict, metadata, records, cfg["data"])

    model = build_model(cfg, metadata, records, adapter_builder)
    track_metadata_tensors = (
        constant_track_metadata(records)
        if bool(cfg["model"].get("use_track_metadata", False))
        else {}
    )
    collate_fn = _build_collate_fn(track_metadata_tensors)

    scale_factors, _, clip_hard = resolve_scale_and_clip(records)
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
        resume_from_checkpoint=_normalise_checkpoint_mode(
            cfg["trainer"].get("resume_from_checkpoint")
        ),
        init_weights_from_checkpoint=(
            str(cfg["trainer"].get("init_weights_from_checkpoint"))
            if cfg["trainer"].get("init_weights_from_checkpoint") not in (None, "")
            else None
        ),
        metric_for_best_model=str(cfg["trainer"].get("metric_for_best_model", "eval_loss")),
        greater_is_better=bool(cfg["trainer"].get("greater_is_better", False)),
        early_stopping_patience=(
            int(cfg["trainer"]["early_stopping_patience"])
            if cfg["trainer"].get("early_stopping_patience") is not None
            else None
        ),
        early_stopping_threshold=float(cfg["trainer"].get("early_stopping_threshold", 0.0)),
        eval_accumulation_steps=(
            int(cfg["trainer"]["eval_accumulation_steps"])
            if cfg["trainer"].get("eval_accumulation_steps") is not None
            else None
        ),
        max_eval_samples=(
            int(cfg["trainer"]["max_eval_samples"])
            if cfg["trainer"].get("max_eval_samples") is not None
            else None
        ),
        eval_on_start=bool(cfg["trainer"].get("eval_on_start", True)),
        num_plot_examples=int(cfg["trainer"].get("num_plot_examples", 4)),
        provenance=nested_config(cfg["trainer"].get("provenance"), ProvenanceConfig),
    )
    if trainer_cfg.resume_from_checkpoint and trainer_cfg.init_weights_from_checkpoint:
        raise ValueError(
            "Set only one of trainer.resume_from_checkpoint or "
            "trainer.init_weights_from_checkpoint"
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
    trainer_model = TrainerCompatibleModel(model, loss_fn)
    if trainer_cfg.init_weights_from_checkpoint:
        load_model_weights_only(trainer_model.model, trainer_cfg.init_weights_from_checkpoint)

    _write_provenance(
        output_dir=output_dir,
        cfg=cfg,
        data_path=data_path,
        dataset_dict=dataset_dict,
        metadata=metadata,
        records=records,
        trainer_cfg=trainer_cfg,
    )

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
    topk_bins = int(cfg["trainer"].get("topk_bins", 256))
    callbacks: list[TrainerCallback] = [_WandbConfigCallback(cfg), _LRLogCallback()]
    if trainer_cfg.early_stopping_patience is not None and "validation" in dataset_dict:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=trainer_cfg.early_stopping_patience,
                early_stopping_threshold=trainer_cfg.early_stopping_threshold,
            )
        )
    track_names = [Path(r["bigwig_path"]).stem for r in records if r.get("bigwig_path")]
    val_dataset = dataset_dict.get("validation")
    if val_dataset is not None and trainer_cfg.num_plot_examples > 0:
        callbacks.append(
            _EvalPlotCallback(
                dataset=val_dataset,
                collate_fn=collate_fn,
                num_examples=trainer_cfg.num_plot_examples,
                output_dir=output_dir,
                track_names=track_names or None,
                scale_factors=scale_factors,
                apply_squash=bool(cfg["data"].get("apply_squash", True)),
                apply_scale=bool(cfg["data"].get("apply_scale", True)),
            )
        )
    trainer = RegulonadoTrainer(
        model=trainer_model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,
        compute_metrics=_make_compute_metrics(len(records)),
        preprocess_logits_for_metrics=_make_preprocess_logits_for_metrics(topk_bins),
    )
    trainer.train(resume_from_checkpoint=trainer_cfg.resume_from_checkpoint)
    train_losses = [
        float(entry["loss"])
        for entry in trainer.state.log_history
        if "loss" in entry and "eval_loss" not in entry
    ]
    eval_losses = [
        float(entry["eval_loss"]) for entry in trainer.state.log_history if "eval_loss" in entry
    ]
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
        "resume_from_checkpoint": trainer_cfg.resume_from_checkpoint,
        "init_weights_from_checkpoint": trainer_cfg.init_weights_from_checkpoint,
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
