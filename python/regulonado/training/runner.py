from __future__ import annotations

import dataclasses
import json
import logging
import random
from collections.abc import Callable, Mapping, Sequence
from os import environ
from pathlib import Path
from time import perf_counter
from typing import Any

import hydra
import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from datasets import IterableDataset as HFIterableDataset
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from torch.optim import AdamW
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    get_scheduler,
)

from regulonado.dataset.build import build_rc_permutation, make_transform
from regulonado.model import (
    BackboneSpec,
    FreezePolicy,
    RegulonadoConfig,
    RegulonadoModel,
    build_backbone_adapter,
    build_condition_shared_track_index,
)
from regulonado.training.callbacks import (
    EvalPlotCallback,
    LRLogCallback,
    WandbConfigCallback,
)
from regulonado.training.config import TrainerConfig
from regulonado.training.losses import (
    log1p_huber_loss,
    poisson_multinomial_binwise_loss,
    poisson_multinomial_loss,
    poisson_nll_loss,
    scaled_poisson_multinomial_loss,
    topk_additive_loss,
    topk_reweight_loss,
    transfer_calibration_loss,
)
from regulonado.training.metrics import (
    make_compute_metrics,
    make_preprocess_logits_for_metrics,
)
from regulonado.training.provenance import write_provenance

logger = logging.getLogger(__name__)


def _rank() -> int:
    """Process rank under torchrun/DDP (0 when launched single-process)."""
    return int(environ.get("RANK") or environ.get("LOCAL_RANK") or 0)


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
    """Load model weights from a checkpoint without re-initializing missing keys.

    Accepts either a single weight file (safetensors or .bin) or a directory
    containing ``model.safetensors`` or ``pytorch_model.bin``. Strips the
    ``model.`` prefix from keys if present (set by HF Trainer). Raises on
    unexpected keys or missing keys (beyond non-trainable BatchNorm counters).

    Parameters
    ----------
    model : torch.nn.Module
        Model to load weights into.
    checkpoint : str | Path
        Path to a single weight file or directory containing model weights.

    Raises
    ------
    FileNotFoundError
        If no model weights file is found.
    RuntimeError
        If unexpected or missing keys are encountered (beyond num_batches_tracked).
    """
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
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    # HF Trainer saves the TrainerCompatibleModel wrapper, so keys are prefixed with "model.".
    # Strip that prefix if present so the state dict loads into a bare RegulonadoModel.
    first_keys = list(state_dict)[:5]
    if all(k.startswith("model.") for k in first_keys):
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

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


def load_dataset_metadata(
    data_path: Path,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Load dataset metadata from ``tracks.parquet``, preferring an explicit path.

    Returns a plain dict merging the table's run-level scalars (``df.attrs`` —
    ``context_length``, ``bin_size``, ...) with ``final_track_records``, so
    every existing caller of :func:`track_records` keeps working unchanged.
    """
    import pandas as pd

    from regulonado.tracks_table import read_track_table, to_track_records, verify_fingerprint

    table_path = metadata_path if metadata_path is not None else data_path / "tracks.parquet"
    if not table_path.is_file():
        raise FileNotFoundError(f"Track table not found: {table_path}")
    table = read_track_table(table_path)

    # Unlike track_qc/build, a fingerprint mismatch here only warns: the Arrow
    # data is already committed, so there is nothing left to abort.
    for _, row in table[table["status"] == "included"].iterrows():
        expected = {k: row[k] for k in row.index if k.startswith("fp_") and pd.notna(row[k])}
        if not expected:
            continue
        try:
            problems = verify_fingerprint(row["resolved_path"], expected)
        except OSError as exc:
            logger.warning(f"Could not verify fingerprint for {row['track_name']!r}: {exc}")
            continue
        if problems:
            logger.warning(
                f"Track {row['track_name']!r} fingerprint drift since assembly: "
                f"{'; '.join(problems)}"
            )

    return {**table.attrs, "final_track_records": to_track_records(table)}


def _load_dataset_streaming(data_path: Path) -> dict[str, Any]:
    from datasets import load_dataset

    return load_dataset(
        data_path.as_posix(),
        streaming=True,
    )


def track_records(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract track metadata records (``final_track_records``) from dataset metadata.

    Parameters
    ----------
    metadata : Mapping[str, Any]
        Dataset metadata dictionary, as returned by :func:`load_dataset_metadata`.

    Returns
    -------
    list[dict[str, Any]]
        List of track record dictionaries.

    Raises
    ------
    ValueError
        If no track records are found or the field is not a non-empty list.
    """
    records = metadata.get("final_track_records")
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
    """Infer the vocabulary size for a categorical metadata field.

    Finds the maximum integer value in the specified field across all records,
    then adds 1. Treats missing or negative values as absent. Returns 0 if no
    valid values are found.

    Parameters
    ----------
    records : Sequence[Mapping[str, Any]]
        Track record list.
    key : str
        Metadata field name to inspect (e.g., "condition_id").

    Returns
    -------
    int
        Vocabulary size: max_value + 1, or 0 if no valid values found.

    Examples
    --------
    >>> records = [{"condition_id": 0}, {"condition_id": 1}, {"condition_id": 2}]
    >>> infer_cardinality(records, "condition_id")
    3
    """
    values = {
        int(record[key])
        for record in records
        if record.get(key) is not None and int(record[key]) >= 0
    }
    return max(values) + 1 if values else 0


def infer_cardinality_any(records: Sequence[Mapping[str, Any]], *keys: str) -> int:
    """Cardinality for the first key that any record actually carries.

    Used where a field has been renamed and both spellings must be accepted
    (``source_id`` superseding ``cell_line_id``).
    """
    for key in keys:
        cardinality = infer_cardinality(records, key)
        if cardinality:
            return cardinality
    return 0


# Metadata key(s) per model input, most preferred first. `source_id` is the
# current name for the biological source of a track (cell line, primary cells,
# tissue, organoid); `cell_line_id` is its narrower predecessor, kept so datasets
# built before the rename still load. The model-side tensor name stays
# `track_cell_line_ids` because it is baked into saved checkpoints.
_TRACK_METADATA_FIELD_MAP = {
    "track_condition_ids": ("condition_id",),
    "track_timepoint_minutes": ("timepoint_minutes",),
    "track_cell_line_ids": ("source_id", "cell_line_id"),
    "track_assay_type_ids": ("assay_type_id",),
    "track_target_ids": ("target_id",),
}


def constant_track_metadata_values(records: Sequence[Mapping[str, Any]]) -> dict[str, list]:
    values: dict[str, list] = {}
    for out_key, keys in _TRACK_METADATA_FIELD_MAP.items():
        if out_key == "track_timepoint_minutes":
            array = _track_array(records, *keys, dtype=np.float32, fill_value=float("nan"))
            if np.all(np.isnan(array)):
                continue
            values[out_key] = [None if np.isnan(value) else float(value) for value in array]
        else:
            array = _track_array(records, *keys, dtype=np.int64, fill_value=-1)
            if np.all(array < 0):
                continue
            values[out_key] = [int(value) for value in array]
    return values


def constant_track_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for out_key, values in constant_track_metadata_values(records).items():
        if out_key == "track_timepoint_minutes":
            tensors[out_key] = torch.as_tensor(
                [float("nan") if value is None else value for value in values],
                dtype=torch.float32,
            )
        else:
            tensors[out_key] = torch.as_tensor(values, dtype=torch.long)
    return tensors


def resolve_scale_and_clip(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract scale factors and clipping thresholds from track records.

    Retrieves per-track normalization and clipping parameters from dataset
    metadata. Uses default values if fields are missing: 1.0 for scale,
    348.0 for soft clip, 796.0 for hard clip.

    Parameters
    ----------
    records : Sequence[Mapping[str, Any]]
        Track record list from dataset metadata.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Three float32 arrays of shape [n_tracks]:
        - scale_factors: per-track normalization factor (default 1.0)
        - clip_soft: soft clipping threshold (default 348.0)
        - clip_hard: hard clipping threshold (default 796.0)
    """
    scale_factors = _track_array(records, "scale_factor", dtype=np.float32, fill_value=1.0)
    clip_soft = _track_array(records, "clip_soft", dtype=np.float32, fill_value=348.0)
    clip_hard = _track_array(records, "clip_hard", dtype=np.float32, fill_value=796.0)
    background = _track_array(records, "background", dtype=np.float32, fill_value=0.0)
    return scale_factors, clip_soft, clip_hard, background


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
    if loss_name == "poisson_multinomial_binwise":
        return lambda pred, target: poisson_multinomial_binwise_loss(
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


def _apply_dataset_transforms(
    dataset_dict: DatasetDict | dict[str, Any],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    data_cfg: Mapping[str, Any],
) -> DatasetDict | dict[str, Any]:
    scale_factors, clip_soft, clip_hard, background = resolve_scale_and_clip(records)
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
        background,
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
        background,
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
        allow_random_init=bool(backbone_cfg.get("allow_random_init", False)),
    )


def _build_regulonado_config(
    cfg: Mapping[str, Any],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    backbone: Any,
) -> RegulonadoConfig:
    model_cfg = cfg["model"]
    head_cfg = cfg["head"]
    backbone_cfg = cfg["backbone"]
    data_cfg = cfg.get("data", {})
    use_track_metadata = bool(model_cfg.get("use_track_metadata", False))
    target_length = int(metadata.get("n_pred_bins", 0)) or None

    shared_track_index: list[int] = []
    if bool(model_cfg.get("share_condition_base_channels", True)) and use_track_metadata:
        shared_track_index = build_condition_shared_track_index(records)

    head_type = str(head_cfg.get("type", "residual_film"))

    # Derive filesystem-safe track names from the resolved BigWig paths.
    track_names: list[str] = []
    seen: dict[str, int] = {}
    for idx, record in enumerate(records):
        path = (
            record.get("resolved_path")
            or record.get("path")
            or record.get("bigwig_path")
            or f"track{idx}"
        )
        stem = Path(str(path)).name
        for suffix in (".bigWig", ".bigwig", ".bw", ".bedGraph", ".bedgraph"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        stem = "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem) or f"track{idx}"
        if stem in seen:
            seen[stem] += 1
            stem = f"{stem}_{seen[stem]}"
        else:
            seen[stem] = 0
        track_names.append(stem)

    # Store the effective backbone architecture config used at build time so
    # HF from_pretrained(checkpoint-*) can reconstruct the exact module layout
    # before loading weights (e.g. flashed attention variants for flashzoi).
    backbone_model = getattr(backbone, "model", None)
    backbone_cfg_obj = getattr(backbone_model, "config", None)
    if backbone_cfg_obj is not None and hasattr(backbone_cfg_obj, "to_dict"):
        config_overrides = dict(backbone_cfg_obj.to_dict())
    else:
        config_overrides = dict(backbone_cfg.get("config_overrides") or {})

    return RegulonadoConfig(
        backbone_type=str(backbone_cfg.get("name", "borzoi")),
        pretrained_name=backbone_cfg.get("pretrained_name"),
        config_overrides=config_overrides,
        target_length=backbone_cfg.get("target_length") or target_length,
        head_type=head_type,
        head_hidden=int(head_cfg.get("hidden", 512)),
        head_dropout=float(head_cfg.get("dropout", 0.0)),
        refinement_kernel=int(head_cfg.get("refinement_kernel", 9)),
        mlp_hidden=int(head_cfg["mlp_hidden"]) if head_cfg.get("mlp_hidden") is not None else None,
        n_tracks=len(records),
        feature_dim=int(getattr(backbone, "feature_dim", 1920)),
        use_track_metadata=use_track_metadata,
        activation_type=str(model_cfg.get("activation_type", "softplus")),
        num_conditions=infer_cardinality(records, "condition_id") if use_track_metadata else 0,
        num_cell_lines=(
            infer_cardinality_any(records, "source_id", "cell_line_id")
            if use_track_metadata
            else 0
        ),
        num_assay_types=infer_cardinality(records, "assay_type_id") if use_track_metadata else 0,
        num_targets=infer_cardinality(records, "target_id") if use_track_metadata else 0,
        metadata_hidden=int(model_cfg.get("metadata_hidden", 32)),
        condition_shared_track_index=shared_track_index,
        context_length=int(metadata.get("context_length", data_cfg.get("context_length", 524_288))),
        n_pred_bins=int(metadata.get("n_pred_bins", data_cfg.get("n_pred_bins", 6_144))),
        bin_size=int(metadata.get("bin_size", 32)),
        track_names=track_names,
        track_metadata=constant_track_metadata_values(records) if use_track_metadata else {},
        data_path=str(cfg["data"]["path"]),
    )


def build_model(
    cfg: Mapping[str, Any],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    adapter_builder: Callable[[BackboneSpec], torch.nn.Module],
) -> RegulonadoModel:
    """Build a RegulonadoModel from configuration and data metadata.

    Constructs a backbone adapter and prediction head, then wraps them in a
    RegulonadoModel. Derives head configuration (number of tracks, metadata
    parameters, etc.) from dataset records and metadata. The returned model
    is ready for training or inference.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        Training configuration with keys: backbone, head, model, data.
    metadata : Mapping[str, Any]
        Dataset metadata (e.g., n_pred_bins, bin_size, context_length).
    records : Sequence[Mapping[str, Any]]
        Track metadata records used to build the head configuration.
    adapter_builder : Callable[[BackboneSpec], torch.nn.Module]
        Function to build backbone adapter from spec (default:
        build_backbone_adapter).

    Returns
    -------
    RegulonadoModel
        Initialized model with backbone and prediction head.
    """
    backbone_spec = _make_backbone_spec(cfg["backbone"], metadata)
    backbone = adapter_builder(backbone_spec)
    regulonado_config = _build_regulonado_config(cfg, metadata, records, backbone)
    model = RegulonadoModel(regulonado_config, backbone=backbone)
    return model


def _apply_freeze_policy(model: RegulonadoModel, trainer_cfg: TrainerConfig) -> None:
    model.apply_freeze_policy(
        FreezePolicy(
            freeze_backbone=trainer_cfg.freeze_backbone,
            unfreeze_backbone_stages_from_output_end=trainer_cfg.unfreeze_backbone_stages_from_output_end,
            unfreeze_module_names=trainer_cfg.unfreeze_module_names,
        )
    )


def _build_optimizer(model: RegulonadoModel, trainer_cfg: TrainerConfig) -> torch.optim.Optimizer:
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
        run_name=environ.get("WANDB_NAME") or output_dir.name,
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
        )
        if has_eval
        else None,
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
    """Trainer subclass with custom loss, checkpoint saving, and metrics preprocessing.

    Saves checkpoints as self-contained HF ``PreTrainedModel`` directories (``config.json`` +
    ``model.safetensors``) so any checkpoint can be loaded with
    ``RegulonadoModel.from_pretrained(checkpoint_dir)`` without external metadata files.

    Applies ``preprocess_logits_for_metrics`` inside ``prediction_step`` (with raw labels) and
    then reduces labels to ``[B, T]`` before accumulation to avoid OOM.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._loss_fn: Callable = kwargs.pop("loss_fn", torch.nn.functional.mse_loss)
        self._metrics_preprocess: Callable | None = kwargs.pop(
            "preprocess_logits_for_metrics", None
        )
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss and optionally return model outputs.

        Extracts track metadata from inputs, passes them to the model alongside
        input_ids, applies the loss function to logits and labels, and
        optionally returns model outputs for metric computation.

        Parameters
        ----------
        model : torch.nn.Module
            Model to compute loss for.
        inputs : dict[str, Any]
            Batch with keys input_ids, labels, and optional track_* metadata.
        return_outputs : bool, optional
            Whether to return model outputs, by default False.
        num_items_in_batch : int | None, optional
            Not used; for compatibility with parent Trainer.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]
            Loss tensor, or (loss, outputs) if return_outputs=True.
        """
        input_ids = inputs["input_ids"]
        labels: torch.Tensor | None = inputs.get("labels")
        track_metadata = {k: v for k, v in inputs.items() if k.startswith("track_")}

        logits = model(input_ids, **track_metadata)
        outputs: dict[str, torch.Tensor] = {"logits": logits}

        loss: torch.Tensor | None = None
        if labels is not None:
            # Labels may arrive as [B, L, T]; align to [B, T, L] expected by the loss.
            aligned = labels if labels.shape[-2:] == logits.shape[-2:] else labels.transpose(-2, -1)
            loss = self._loss_fn(logits, aligned)
            outputs["loss"] = loss

        if return_outputs:
            return loss, outputs  # type: ignore[return-value]
        return loss  # type: ignore[return-value]

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """Save model and training arguments to output directory.

        Saves the bare RegulonadoModel with config.json and model.safetensors
        (clean keys, no "model." prefix), plus training_args.json for
        traceability. Checkpoint can be loaded with
        ``RegulonadoModel.from_pretrained(output_dir)``.

        Parameters
        ----------
        output_dir : str | None, optional
            Output directory; defaults to trainer's output_dir if None.
        _internal_call : bool, optional
            For compatibility with parent Trainer (ignored).
        """
        output_path = Path(output_dir or self.args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Save the RegulonadoModel directly — config.json + model.safetensors with clean keys.
        self.model.save_pretrained(output_path, safe_serialization=True)
        # Preserve training arguments alongside the model for traceability.
        args_path = output_path / "training_args.json"
        if hasattr(self.args, "to_json_file"):
            self.args.to_json_file(args_path)
        else:
            Path(args_path).write_text(self.args.to_json_string())

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Compute predictions and optionally preprocess for metrics.

        Calls parent prediction_step then applies optional metrics preprocessing
        (e.g., log-squashing, bin-wise averaging) to logits. Reduces label
        dimension to [B, T] to conserve eval-set memory.

        Parameters
        ----------
        model : torch.nn.Module
            Model to generate predictions.
        inputs : dict[str, Any]
            Batch to predict on.
        prediction_loss_only : bool
            If True, only return loss (logits and labels are None).
        ignore_keys : list[str] | None, optional
            Keys to exclude from model output (unused).

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
            (loss, logits, labels) where logits and labels may be preprocessed
            or reduced.
        """
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


def _resolve_trainer_config(cfg: Mapping[str, Any]) -> TrainerConfig:
    """Validate ``cfg`` and merge ``cfg["trainer"]`` into a :class:`TrainerConfig`.

    Raises
    ------
    ValueError
        If a required top-level section is missing, the trainer section fails
        to validate, or both ``resume_from_checkpoint`` and
        ``init_weights_from_checkpoint`` are set.
    """
    required_sections = ("data", "backbone", "head", "model", "loss", "trainer")
    missing_sections = [section for section in required_sections if section not in cfg]
    if missing_sections:
        raise ValueError(
            "Training configuration is missing required section(s): "
            + ", ".join(missing_sections)
        )
    try:
        trainer_cfg = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(TrainerConfig), cfg["trainer"])
        )
    except OmegaConfBaseException as exc:
        raise ValueError(f"Invalid trainer configuration: {exc}") from exc
    if not isinstance(trainer_cfg, TrainerConfig):
        raise TypeError("Trainer configuration did not resolve to TrainerConfig")
    trainer_cfg = dataclasses.replace(
        trainer_cfg,
        resume_from_checkpoint=_normalise_checkpoint_mode(trainer_cfg.resume_from_checkpoint),
        init_weights_from_checkpoint=(trainer_cfg.init_weights_from_checkpoint or None),
    )
    if trainer_cfg.resume_from_checkpoint and trainer_cfg.init_weights_from_checkpoint:
        raise ValueError(
            "Set only one of trainer.resume_from_checkpoint or trainer.init_weights_from_checkpoint"
        )
    return trainer_cfg


def _load_training_dataset(
    data_path: Path,
    *,
    streaming: bool,
    rank: int,
) -> DatasetDict | dict[str, Any]:
    """Load the dataset from disk or as an HF streaming source, with timing logs."""
    logger.info(f"[rank {rank}] loading dataset from {data_path} (streaming={streaming}) ...")
    t0 = perf_counter()
    dataset_dict = (
        _load_dataset_streaming(data_path) if streaming else load_from_disk(str(data_path))
    )
    if streaming:
        logger.info(f"[rank {rank}] dataset opened in {perf_counter() - t0:.1f}s (streaming)")
    else:
        sizes = {split: len(dataset_dict[split]) for split in dataset_dict}
        logger.info(f"[rank {rank}] dataset loaded in {perf_counter() - t0:.1f}s | splits={sizes}")
    return dataset_dict


def _load_metadata_and_records(
    data_path: Path,
    metadata_path: Path | None,
    dataset_dict: DatasetDict | dict[str, Any],
    *,
    streaming: bool,
    rank: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load track metadata/records and cross-check them against the Arrow dataset.

    Raises
    ------
    ValueError
        If the Arrow dataset's label width disagrees with the included track count.
    """
    metadata = load_dataset_metadata(data_path, metadata_path)
    records = track_records(metadata)
    logger.info(f"[rank {rank}] metadata loaded | {len(records)} tracks")

    if not streaming:
        labels_feature = dataset_dict[next(iter(dataset_dict))].features.get("labels")
        # Array2D (the real, Rust-built dataset) exposes .shape; other feature
        # types (e.g. a plain Sequence, as in small hand-built test datasets)
        # don't carry a fixed track count to check against.
        arrow_n_tracks = getattr(labels_feature, "shape", (None,))[0]
        if arrow_n_tracks is not None and arrow_n_tracks != len(records):
            raise ValueError(
                f"Arrow label width ({arrow_n_tracks}) != included track count "
                f"({len(records)}); tracks.parquet and the Arrow dataset are out of sync — "
                "rebuild the dataset."
            )
    return metadata, records


def _prepare_dataset_splits(
    dataset_dict: DatasetDict | dict[str, Any],
    data_cfg: Mapping[str, Any],
    trainer_cfg: TrainerConfig,
    metadata: Mapping[str, Any],
    *,
    streaming: bool,
    seed: int,
) -> DatasetDict | dict[str, Any]:
    """Apply streaming shuffle and eval-sample capping, ahead of transform application."""
    if streaming and "train" in dataset_dict:
        shuffle_buffer = _estimate_shuffle_buffer(data_cfg, metadata)
        dataset_dict["train"] = dataset_dict["train"].shuffle(buffer_size=shuffle_buffer, seed=seed)

    max_eval_samples = (
        int(trainer_cfg.max_eval_samples) if trainer_cfg.max_eval_samples is not None else None
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
    return dataset_dict


def _build_model_with_logging(
    cfg: Mapping[str, Any],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    adapter_builder: Callable[[BackboneSpec], torch.nn.Module],
    *,
    rank: int,
) -> RegulonadoModel:
    """Build the model via :func:`build_model`, logging timing and parameter count."""
    logger.info(f"[rank {rank}] building model ...")
    t0 = perf_counter()
    model = build_model(cfg, metadata, records, adapter_builder)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[rank {rank}] model built in {perf_counter() - t0:.1f}s | {n_params / 1e6:.1f}M params"
    )
    return model


def _build_collate_and_loss(
    cfg: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    np.ndarray,
    np.ndarray,
]:
    """Build the batch collate function and loss function, plus scale/background arrays.

    The scale factors and background arrays are also needed later by the eval-plot
    callback, so they are returned alongside the loss function rather than recomputed.
    """
    track_metadata_tensors = (
        constant_track_metadata(records)
        if bool(cfg["model"].get("use_track_metadata", False))
        else {}
    )
    collate_fn = _build_collate_fn(track_metadata_tensors)

    scale_factors, _, clip_hard, background = resolve_scale_and_clip(records)
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
    return collate_fn, loss_fn, scale_factors, background


def _guard_streaming_persistent_workers(
    trainer_cfg: TrainerConfig,
    *,
    streaming: bool,
) -> TrainerConfig:
    """Disable persistent dataloader workers for streaming datasets.

    Persistent workers with HF IterableDataset accumulate Arrow file handles and
    shuffle-buffer state between iterator cycles — workers never restart to clear them.
    """
    if streaming and trainer_cfg.persistent_workers:
        import warnings

        warnings.warn(
            "persistent_workers=True is unsafe with streaming datasets (memory leak). "
            "Overriding to persistent_workers=False.",
            stacklevel=2,
        )
        trainer_cfg = dataclasses.replace(trainer_cfg, persistent_workers=False)
    return trainer_cfg


def _prepare_model_for_training(
    model: RegulonadoModel,
    trainer_cfg: TrainerConfig,
    *,
    rank: int,
) -> None:
    """Apply the freeze policy and, if configured, warm-start weights from a checkpoint."""
    _apply_freeze_policy(model, trainer_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[rank {rank}] freeze policy applied (freeze_backbone={trainer_cfg.freeze_backbone}) | "
        f"{n_trainable / 1e6:.1f}M trainable params"
    )
    if trainer_cfg.init_weights_from_checkpoint:
        logger.info(
            f"[rank {rank}] warm-starting weights from "
            f"{trainer_cfg.init_weights_from_checkpoint} ..."
        )
        load_model_weights_only(model, trainer_cfg.init_weights_from_checkpoint)
        logger.info(f"[rank {rank}] warm-start weights loaded")


def _setup_optimization(
    model: RegulonadoModel,
    trainer_cfg: TrainerConfig,
    dataset_dict: DatasetDict | dict[str, Any],
    *,
    streaming: bool,
    output_dir: Path,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, TrainingArguments]:
    """Build the optimizer, LR scheduler, and HF ``TrainingArguments`` together."""
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
    return optimizer, scheduler, training_args


def _build_training_callbacks(
    cfg: Mapping[str, Any],
    trainer_cfg: TrainerConfig,
    val_dataset: Any,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]],
    scale_factors: np.ndarray,
    background: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[TrainerCallback]:
    """Assemble the trainer callback list (wandb config, LR logging, early stopping, eval plots)."""
    callbacks: list[TrainerCallback] = [WandbConfigCallback(cfg), LRLogCallback()]
    if trainer_cfg.early_stopping_patience is not None and val_dataset is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=trainer_cfg.early_stopping_patience,
                early_stopping_threshold=trainer_cfg.early_stopping_threshold,
            )
        )
    track_names = [Path(r["bigwig_path"]).stem for r in records if r.get("bigwig_path")]
    if val_dataset is not None and trainer_cfg.num_plot_examples > 0:
        callbacks.append(
            EvalPlotCallback(
                dataset=val_dataset,
                collate_fn=collate_fn,
                num_examples=trainer_cfg.num_plot_examples,
                output_dir=output_dir,
                track_names=track_names or None,
                scale_factors=scale_factors,
                background=background,
                apply_squash=bool(cfg["data"].get("apply_squash", True)),
                apply_scale=bool(cfg["data"].get("apply_scale", True)),
            )
        )
    return callbacks


def _run_training_loop(
    trainer: "RegulonadoTrainer",
    trainer_cfg: TrainerConfig,
    *,
    rank: int,
) -> None:
    """Log start-of-training context and run ``trainer.train()``."""
    logger.info(
        f"[rank {rank}] starting trainer.train() | "
        f"max_steps={trainer_cfg.max_steps} max_epochs={trainer_cfg.max_epochs} "
        f"batch_size={trainer_cfg.batch_size} grad_accum={trainer_cfg.gradient_accumulation_steps} "
        f"resume={trainer_cfg.resume_from_checkpoint} eval_on_start={trainer_cfg.eval_on_start}"
    )
    t0 = perf_counter()
    trainer.train(resume_from_checkpoint=trainer_cfg.resume_from_checkpoint)
    logger.info(f"[rank {rank}] trainer.train() returned after {perf_counter() - t0:.1f}s")


def _finalize_trainer_outputs(
    trainer: "RegulonadoTrainer",
    output_dir: Path,
) -> dict[str, list[float]]:
    """Save the final model/trainer state and extract the loss history."""
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
    return {"train/loss": train_losses, "eval/loss": eval_losses}


def _build_training_summary(
    cfg: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    metadata_path: Path | None,
    records: Sequence[Mapping[str, Any]],
    trainer_cfg: TrainerConfig,
    history: dict[str, list[float]],
) -> dict[str, Any]:
    """Assemble the training summary dict and write it to ``output_dir/training_summary.json``."""
    summary = {
        "output_dir": str(output_dir),
        "seed": seed,
        "metadata_path": str(metadata_path) if metadata_path else None,
        "n_tracks": len(records),
        "backbone": cfg["backbone"]["name"],
        "pretrained_model": cfg["backbone"].get("pretrained_name"),
        "head": cfg["head"]["type"],
        "resume_from_checkpoint": trainer_cfg.resume_from_checkpoint,
        "init_weights_from_checkpoint": trainer_cfg.init_weights_from_checkpoint,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_training(
    cfg: Mapping[str, Any],
    *,
    adapter_builder: Callable[[BackboneSpec], torch.nn.Module] = build_backbone_adapter,
) -> dict[str, Any]:
    """Run end-to-end model training with dataset loading, model building, and
    evaluation.

    Loads a dataset from disk or streaming source, applies transforms,
    builds a model, constructs optimizer and scheduler, and trains using
    HuggingFace Trainer. Logs progress, saves checkpoints, and returns
    training history.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        Complete training configuration with keys:
        - data.path: dataset directory or HF streaming path
        - data.streaming: whether to stream the dataset (default False)
        - backbone: backbone configuration
        - head: head configuration
        - model: model configuration (e.g. use_track_metadata)
        - loss: loss function configuration
        - trainer: training hyperparameters
        - output_dir: checkpoint and output directory
        - seed: random seed (default 42)
    adapter_builder : Callable, optional
        Function to build backbone adapter (default: build_backbone_adapter).

    Returns
    -------
    dict[str, Any]
        Training summary with keys:
        - output_dir: where checkpoints were saved
        - n_tracks: number of tracks
        - backbone: backbone type
        - head: head type
        - resume_from_checkpoint: checkpoint path if resumed
        - init_weights_from_checkpoint: path if warm-started
        - history: dict with "train/loss" and "eval/loss" lists
    """
    trainer_cfg = _resolve_trainer_config(cfg)

    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)

    rank = _rank()
    data_path = Path(str(cfg["data"]["path"]))
    streaming = bool(cfg["data"].get("streaming", False))
    logger.info(
        f"[rank {rank}] run_training start | backbone={cfg['backbone'].get('name')} "
        f"head={cfg['head'].get('type')} loss={cfg['loss'].get('name')} "
        f"streaming={streaming} output_dir={cfg.get('output_dir')}"
    )

    dataset_dict = _load_training_dataset(data_path, streaming=streaming, rank=rank)

    metadata_path_value = cfg["data"].get("metadata_path")
    metadata_path = Path(str(metadata_path_value)) if metadata_path_value else None
    metadata, records = _load_metadata_and_records(
        data_path, metadata_path, dataset_dict, streaming=streaming, rank=rank
    )

    dataset_dict = _prepare_dataset_splits(
        dataset_dict, cfg["data"], trainer_cfg, metadata, streaming=streaming, seed=seed
    )
    dataset_dict = _apply_dataset_transforms(dataset_dict, metadata, records, cfg["data"])
    logger.info(f"[rank {rank}] dataset transforms applied")

    model = _build_model_with_logging(cfg, metadata, records, adapter_builder, rank=rank)
    collate_fn, loss_fn, scale_factors, background = _build_collate_and_loss(cfg, records)
    trainer_cfg = _guard_streaming_persistent_workers(trainer_cfg, streaming=streaming)

    output_dir = Path(str(cfg.get("output_dir") or Path.cwd() / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    _prepare_model_for_training(model, trainer_cfg, rank=rank)

    write_provenance(
        output_dir=output_dir,
        cfg=cfg,
        data_path=data_path,
        dataset_dict=dataset_dict,
        metadata=metadata,
        records=records,
        trainer_cfg=trainer_cfg,
    )

    optimizer, scheduler, training_args = _setup_optimization(
        model, trainer_cfg, dataset_dict, streaming=streaming, output_dir=output_dir
    )
    val_dataset = dataset_dict.get("validation")
    callbacks = _build_training_callbacks(
        cfg, trainer_cfg, val_dataset, collate_fn, scale_factors, background, records, output_dir
    )
    topk_bins = int(cfg["trainer"].get("topk_bins", 256))
    trainer = RegulonadoTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,
        loss_fn=loss_fn,
        compute_metrics=make_compute_metrics(len(records)),
        preprocess_logits_for_metrics=make_preprocess_logits_for_metrics(topk_bins),
    )

    _run_training_loop(trainer, trainer_cfg, rank=rank)
    history = _finalize_trainer_outputs(trainer, output_dir)

    return _build_training_summary(
        cfg, output_dir, seed, metadata_path, records, trainer_cfg, history
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def hydra_entrypoint(cfg: DictConfig) -> None:
    """Hydra entrypoint for training from YAML configuration.

    Loads training config via Hydra, resolves interpolations, converts to
    plain dict, and calls run_training.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object from configs/train.yaml.
    """
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Hydra config did not resolve to a dictionary")
    run_training(config)


def main() -> None:
    """CLI entry point for training.

    Calls hydra_entrypoint to load configuration and run training.
    """
    hydra_entrypoint()


if __name__ == "__main__":
    main()
